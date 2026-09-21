"""Train source-r2 directly on Task4C and conditionally confirm one candidate.

Only the episode distribution changes: classic games against three independently
seeded full-strength rule agents.  Three replicas and dense immutable snapshots
feed a cheap Task4C-only signal gate.  Prior-course and outer evaluation runs
only when at least two replicas independently contain a supportive snapshot.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v4_task4c.config import (  # noqa: E402
    OUTER_STRATA, REPLICAS, SNAPSHOT_ROUNDS, confirmed_checkpoint_path,
    final_checkpoint_path, load_protocol, selected_checkpoint_path,
    sha256_file, snapshot_path,
)
from tools.v4_task4_duel_training import (  # noqa: E402
    aggregate, atomic_json, load_completed, metrics_by_agent, opponent_summary,
    relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task4c-direct-s127000.json"
TRAINING_KIND = "model-a-v4-task4c-direct-training"
EVALUATION_KIND = "model-a-v4-task4c-direct-evaluation"
SELECTION_KIND = "model-a-v4-task4c-direct-inner-selection"
REPORT_KIND = "model-a-v4-task4c-direct-course-report"
FREEZE_KIND = "model-a-v4-task4c-direct-confirmed-selection"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = set()
    for case in protocol["training"]["seeds"].values():
        registered.update(int(value) for value in case.values())
    for cases in protocol["inner_signal"]["cases_by_replica"].values():
        for case in cases:
            registered.update(int(value) for value in case.values())
    for spec in protocol["outer_confirmation"]["strata"].values():
        for case in spec["cases"]:
            registered.update(int(value) for value in case.values())
    excluded_roots = {
        (ROOT / protocol["checkpoint_directory"]).resolve(),
        (ROOT / protocol["training_manifest_directory"]).resolve(),
        (ROOT / protocol["evaluation_manifest_directory"]).resolve(),
    }
    excluded_files = {
        protocol_path.resolve(),
        (ROOT / protocol["selection_manifest_path"]).resolve(),
        (ROOT / protocol["confirmed_selection_manifest_path"]).resolve(),
        (ROOT / protocol["report_path"]).resolve(),
    }
    conflicts = []
    for path in (ROOT / "experiments").rglob("*.json"):
        resolved = path.resolve()
        if resolved in excluded_files or any(root == resolved or root in resolved.parents for root in excluded_roots):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overlap = sorted(registered & seed_values(payload))
        if overlap:
            conflicts.append({"path": relative(path), "seeds": overlap})
    if conflicts:
        raise ValueError(f"registered s127000 seeds were already used: {conflicts}")


def training_manifest_path(protocol: dict, replica: str) -> Path:
    return ROOT / protocol["training_manifest_directory"] / f"{replica}.json"


def evaluation_manifest_path(protocol: dict, suite: str, stratum: str, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / suite / stratum / f"{label}-s{world_seed}.json"


def selection_manifest_path(protocol: dict) -> Path:
    return ROOT / protocol["selection_manifest_path"]


def freeze_manifest_path(protocol: dict) -> Path:
    return ROOT / protocol["confirmed_selection_manifest_path"]


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def validate_candidate_checkpoint(protocol: dict, protocol_hash: str, path: Path, replica: str, stage_round: int) -> dict:
    from agent_code.model_a_v4_task4c.callbacks import _torch_load

    if not path.is_file():
        raise RuntimeError(f"Task4C checkpoint missing: {path}")
    payload = _torch_load(path)
    expected = {
        "architecture": "model-a-mlp-v4-global7",
        "protocol_sha256": protocol_hash,
        "replica": replica,
        "stage_id": "task4c_three_rule",
        "stage_completed_rounds": stage_round,
        "parent_sha256": protocol["source_parent"]["sha256"],
        "training_variable": "task4c_episode_distribution_only",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Task4C checkpoint {key} mismatch: {path}")
    if abs(float(payload.get("epsilon", -1.0)) - 0.05) > 1e-12:
        raise RuntimeError(f"Task4C epsilon changed: {path}")
    return payload


def validate_training_artifacts(protocol: dict, protocol_hash: str, replica: str) -> list[dict]:
    snapshots = []
    for stage_round in SNAPSHOT_ROUNDS:
        path = snapshot_path(protocol, replica, stage_round)
        payload = validate_candidate_checkpoint(protocol, protocol_hash, path, replica, stage_round)
        snapshots.append({
            "stage_round": stage_round,
            "path": relative(path),
            "sha256": sha256_file(path),
            "completed_rounds": int(payload["completed_rounds"]),
            "training_steps": int(payload["training_steps"]),
            "gradient_steps": int(payload["gradient_steps"]),
            "epsilon": float(payload["epsilon"]),
        })
    final = final_checkpoint_path(protocol, replica)
    validate_candidate_checkpoint(protocol, protocol_hash, final, replica, SNAPSHOT_ROUNDS[-1])
    if sha256_file(final) != snapshots[-1]["sha256"]:
        raise RuntimeError(f"Task4C final is not the immutable round-200 snapshot: {replica}")
    return snapshots


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, replica: str) -> dict:
    manifest = training_manifest_path(protocol, replica)
    output = final_checkpoint_path(protocol, replica)
    existing = load_completed(manifest, TRAINING_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed Task4C training protocol drift: {replica}")
        validate_training_artifacts(protocol, protocol_hash, replica)
        return existing
    stats_path = manifest.with_suffix(".stats.json")
    if output.parent.exists() or stats_path.exists():
        raise RuntimeError(f"refusing orphaned Task4C training artifacts: {replica}")
    parent = (ROOT / protocol["source_parent"]["path"]).resolve()
    spec = protocol["training"]
    case = spec["seeds"][replica]
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_task4c", *spec["opponents"],
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides = {
        "MODEL_A_V4T4C_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_V4T4C_CHECKPOINT_PATH": str(output),
        "MODEL_A_V4T4C_PARENT_PATH": str(parent),
        "MODEL_A_V4T4C_REPLICA": replica,
        "MODEL_A_V4T4C_SEED": str(case["agent_seed"]),
        "TASK4_RULE_SEED": str(case["opponent_seed"]),
    }
    record = {
        "schema_version": 1,
        "kind": TRAINING_KIND,
        "status": "running",
        "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "replica": replica,
        "active_subcourse": "task4c_three_rule",
        "training": {**case, **{key: spec[key] for key in ("scenario", "opponents", "rounds")}},
        "parent_checkpoint": {"path": relative(parent), "sha256": sha256_file(parent)},
        "checkpoint": {"path": relative(output), "sha256": None},
        "command": command,
        "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats_path.is_file():
            raise RuntimeError("Task4C training process or stats failed")
        record["snapshots"] = validate_training_artifacts(protocol, protocol_hash, replica)
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats_path.read_text(encoding="utf-8")))
        record["checkpoint"]["sha256"] = sha256_file(output)
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task4C training failed: {replica}: {record.get('error')}")
    return record


def checkpoint_environment(protocol_path: Path, protocol: dict, identity: dict, case: dict) -> tuple[str, Path, dict[str, str]]:
    if identity["kind"] == "v4":
        checkpoint = (ROOT / protocol["frozen_v4_baseline"]["path"]).resolve()
        return "model_a_dqn", checkpoint, {
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_SEED": str(case["agent_seed"]),
        }
    if identity["kind"] == "source":
        checkpoint = (ROOT / protocol["source_parent"]["path"]).resolve()
        lineage = protocol["source_parent"]["lineage"]
        return "model_a_v4_curriculum", checkpoint, {
            "MODEL_A_V4C_PROTOCOL_PATH": str(ROOT / lineage["protocol_path"]),
            "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V4C_ARM": "curriculum",
            "MODEL_A_V4C_REPLICA": "r2",
            "MODEL_A_V4C_STAGE": "task2",
            "MODEL_A_V4C_SEED": str(case["agent_seed"]),
        }
    replica = identity["replica"]
    checkpoint = Path(identity["path"]).resolve()
    return "model_a_v4_task4c", checkpoint, {
        "MODEL_A_V4T4C_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_V4T4C_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_V4T4C_REPLICA": replica,
        "MODEL_A_V4T4C_SEED": str(case["agent_seed"]),
    }


def evaluate_one(
    protocol_path: Path, protocol: dict, protocol_hash: str, suite: str,
    stratum: str, spec: dict, label: str, identity: dict, case: dict,
) -> dict:
    target, checkpoint, overrides = checkpoint_environment(protocol_path, protocol, identity, case)
    checkpoint_hash = sha256_file(checkpoint)
    output = evaluation_manifest_path(protocol, suite, stratum, label, int(case["world_seed"]))
    expected = {
        **case,
        "scenario": spec["scenario"],
        "opponents": spec["opponents"],
        "rounds_per_case": int(spec["rounds_per_case"]),
        "target_agent": target,
    }
    existing = load_completed(output, EVALUATION_KIND)
    if existing is not None:
        if (
            existing.get("protocol_sha256") != protocol_hash
            or existing.get("evaluation") != expected
            or existing.get("checkpoint", {}).get("sha256") != checkpoint_hash
        ):
            raise RuntimeError(f"completed Task4C evaluation drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned Task4C stats: {stats_path}")
    command = [
        sys.executable, "main.py", "play", "--agents", target, *spec["opponents"],
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    if any(name == "seeded_rule_based_agent" for name in spec["opponents"]):
        overrides["TASK4_RULE_SEED"] = str(case["opponent_seed"])
    elif spec["opponents"]:
        overrides["TASK3_OPPONENT_SEED"] = str(case["opponent_seed"])
    record = {
        "schema_version": 1,
        "kind": EVALUATION_KIND,
        "status": "running",
        "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "suite": suite,
        "stratum": stratum,
        "label": label,
        "evaluation": expected,
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
        "command": command,
        "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(output, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    if completed.returncode == 0 and stats_path.is_file() and sha256_file(checkpoint) == checkpoint_hash:
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats_path.read_text(encoding="utf-8")))
        record["target_metrics"] = record["metrics_by_agent"].get(target)
        record["status"] = "completed" if record["target_metrics"] else "failed"
    else:
        record["status"] = "failed"
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task4C evaluation failed: {suite}/{stratum}/{label}")
    return record


def evaluate_cases(
    protocol_path: Path, protocol: dict, protocol_hash: str, suite: str,
    stratum: str, spec: dict, cases: list[dict], label: str, identity: dict,
) -> tuple[dict, list[str], list[dict]]:
    runs = [
        evaluate_one(protocol_path, protocol, protocol_hash, suite, stratum, spec, label, identity, case)
        for case in cases
    ]
    target, _, _ = checkpoint_environment(protocol_path, protocol, identity, cases[0])
    row = {"target": aggregate(runs), "opponents": opponent_summary(runs, target)}
    manifests = [
        relative(evaluation_manifest_path(protocol, suite, stratum, label, int(case["world_seed"])))
        for case in cases
    ]
    return row, manifests, runs


def inner_score(protocol: dict, candidate: dict, source: dict, v4: dict) -> dict:
    rule = protocol["inner_signal"]["supportive_rule"]
    c, s, v = candidate["target"], source["target"], v4["target"]
    gates = {
        "score_gain_over_source": c["score_per_round"] >= s["score_per_round"] + float(rule["minimum_score_gain_over_source"]) - 1e-12,
        "score_not_below_v4": c["score_per_round"] >= v["score_per_round"] + float(rule["minimum_score_delta_to_v4"]) - 1e-12,
        "kills_not_below_source": c["kills_per_round"] >= s["kills_per_round"] + float(rule["minimum_kills_delta_to_source"]) - 1e-12,
        "suicide_safe_vs_v4": c["suicides_per_round"] <= v["suicides_per_round"] + float(rule["maximum_suicide_increase_over_v4"]) + 1e-12,
        "beats_rule_mean": c["score_per_round"] >= candidate["opponents"]["mean_score_per_agent_round"] + float(rule["minimum_score_margin_over_rule_mean"]) - 1e-12,
    }
    return {
        "gates": gates,
        "supportive": all(gates.values()),
        "score_gain_over_source": c["score_per_round"] - s["score_per_round"],
        "score_gain_over_v4": c["score_per_round"] - v["score_per_round"],
        "score_margin_over_rule": c["score_per_round"] - candidate["opponents"]["mean_score_per_agent_round"],
    }


def selection_rank(item: dict) -> tuple:
    target = item["row"]["target"]
    return (
        min(item["score"]["score_gain_over_source"], item["score"]["score_gain_over_v4"]),
        item["score"]["score_margin_over_rule"],
        -target["suicides_per_round"],
        target["kills_per_round"],
        -int(item["stage_round"]),
    )


def run_inner(protocol_path: Path, protocol: dict, protocol_hash: str) -> dict:
    path = selection_manifest_path(protocol)
    existing = load_completed(path, SELECTION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed Task4C selection protocol drift")
        if existing["result"]["selection_passed"]:
            chosen = existing["result"]["selected_candidate"]
            selected = selected_checkpoint_path(protocol)
            validate_candidate_checkpoint(protocol, protocol_hash, selected, chosen["replica"], chosen["stage_round"])
            if sha256_file(selected) != chosen["sha256"]:
                raise RuntimeError("selected Task4C checkpoint drift")
        return existing
    spec = {
        "scenario": protocol["inner_signal"]["scenario"],
        "opponents": protocol["inner_signal"]["opponents"],
        "rounds_per_case": protocol["inner_signal"]["rounds_per_case"],
    }
    baselines, candidates, choices, supportive_choices = {}, {}, {}, []
    baseline_manifests, candidate_manifests = {}, {}
    for replica in REPLICAS:
        cases = protocol["inner_signal"]["cases_by_replica"][replica]
        baselines[replica], baseline_manifests[replica] = {}, {}
        for label, identity in (("source-r2", {"kind": "source"}), ("v4", {"kind": "v4"})):
            row, manifests, _ = evaluate_cases(
                protocol_path, protocol, protocol_hash, "inner_signal", "task4c_three_rule",
                spec, cases, f"{label}-{replica}", identity,
            )
            baselines[replica][label] = row
            baseline_manifests[replica][label] = manifests
        items = []
        for stage_round in SNAPSHOT_ROUNDS:
            checkpoint = snapshot_path(protocol, replica, stage_round)
            label = f"{replica}-round-{stage_round:04d}"
            identity = {"kind": "candidate", "replica": replica, "path": str(checkpoint)}
            row, manifests, _ = evaluate_cases(
                protocol_path, protocol, protocol_hash, "inner_signal", "task4c_three_rule",
                spec, cases, label, identity,
            )
            item = {
                "replica": replica,
                "stage_round": stage_round,
                "path": relative(checkpoint),
                "sha256": sha256_file(checkpoint),
                "row": row,
            }
            item["score"] = inner_score(protocol, row, baselines[replica]["source-r2"], baselines[replica]["v4"])
            candidates[label] = item
            candidate_manifests[label] = manifests
            items.append(item)
        supportive = [item for item in items if item["score"]["supportive"]]
        chosen = max(supportive or items, key=selection_rank)
        choices[replica] = {
            "supportive_snapshot_exists": bool(supportive),
            "chosen_for_replica": {key: chosen[key] for key in ("replica", "stage_round", "path", "sha256", "score")},
        }
        if supportive:
            supportive_choices.append(max(supportive, key=selection_rank))
    required = int(protocol["inner_signal"]["supportive_replicas_required"])
    selected_item = max(supportive_choices, key=selection_rank) if len(supportive_choices) >= required else None
    if selected_item is not None:
        selected = selected_checkpoint_path(protocol)
        if selected.exists():
            raise RuntimeError("refusing orphaned selected Task4C checkpoint")
        atomic_copy(ROOT / selected_item["path"], selected)
        if sha256_file(selected) != selected_item["sha256"]:
            raise RuntimeError("selected Task4C checkpoint copy mismatch")
    result = {
        "supportive_replicas_required": required,
        "supportive_replica_count": len(supportive_choices),
        "replica_choices": choices,
        "selection_passed": selected_item is not None,
        "selected_candidate": None if selected_item is None else {
            "replica": selected_item["replica"],
            "stage_round": selected_item["stage_round"],
            "source_path": selected_item["path"],
            "path": relative(selected_checkpoint_path(protocol)),
            "sha256": selected_item["sha256"],
            "score": selected_item["score"],
        },
        "outer_confirmation_allowed": selected_item is not None,
    }
    record = {
        "schema_version": 1,
        "kind": SELECTION_KIND,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "baselines_by_replica": baselines,
        "baseline_evaluation_manifests": baseline_manifests,
        "candidates": candidates,
        "candidate_evaluation_manifests": candidate_manifests,
        "result": result,
    }
    atomic_json(path, record)
    return record


def evaluate_outer_identity(protocol_path: Path, protocol: dict, protocol_hash: str, label: str, identity: dict) -> tuple[dict, dict, dict]:
    rows, manifests, case_runs = {}, {}, {}
    for stratum, spec in protocol["outer_confirmation"]["strata"].items():
        row, paths, runs = evaluate_cases(
            protocol_path, protocol, protocol_hash, "outer_confirmation", stratum,
            spec, spec["cases"], label, identity,
        )
        rows[stratum], manifests[stratum], case_runs[stratum] = row, paths, runs
    return rows, manifests, case_runs


def outer_decision(protocol: dict, rows: dict, case_runs: dict) -> dict:
    rule = protocol["outer_confirmation"]["decision_rule"]
    candidate, source, v4 = rows["candidate"], rows["source-r2"], rows["v4"]
    t = lambda group, name: group[name]["target"]
    prior = {
        "task1": t(candidate, "task1")["score_per_round"] >= float(rule["minimum_task1_fraction_of_v4"]) * t(v4, "task1")["score_per_round"] - 1e-12,
        "task2": t(candidate, "task2")["score_per_round"] >= t(source, "task2")["score_per_round"] - float(rule["maximum_task2_score_drop_from_source"]) - 1e-12,
        "peaceful_score": t(candidate, "task3_peaceful")["score_per_round"] >= t(source, "task3_peaceful")["score_per_round"] - float(rule["maximum_task3_score_drop_from_source"]) - 1e-12,
        "peaceful_kills": t(candidate, "task3_peaceful")["kills_per_round"] >= t(source, "task3_peaceful")["kills_per_round"] - float(rule["maximum_task3_kills_drop_from_source"]) - 1e-12,
        "peaceful_suicide": t(candidate, "task3_peaceful")["suicides_per_round"] <= t(source, "task3_peaceful")["suicides_per_round"] + float(rule["maximum_task3_suicide_increase_over_source"]) + 1e-12,
        "coin_score": t(candidate, "task3_coin")["score_per_round"] >= t(source, "task3_coin")["score_per_round"] - float(rule["maximum_task3_score_drop_from_source"]) - 1e-12,
        "coin_kills": t(candidate, "task3_coin")["kills_per_round"] >= t(source, "task3_coin")["kills_per_round"] - float(rule["maximum_task3_kills_drop_from_source"]) - 1e-12,
        "coin_suicide": t(candidate, "task3_coin")["suicides_per_round"] <= t(source, "task3_coin")["suicides_per_round"] + float(rule["maximum_task3_suicide_increase_over_source"]) + 1e-12,
        "classic_duel_score": t(candidate, "task4b_classic_duel")["score_per_round"] >= t(source, "task4b_classic_duel")["score_per_round"] - float(rule["maximum_duel_score_drop_from_source"]) - 1e-12,
        "classic_duel_suicide": t(candidate, "task4b_classic_duel")["suicides_per_round"] <= t(source, "task4b_classic_duel")["suicides_per_round"] + float(rule["maximum_duel_suicide_increase_over_source"]) + 1e-12,
    }
    c = t(candidate, "task4c_three_rule")
    s = t(source, "task4c_three_rule")
    v = t(v4, "task4c_three_rule")
    support = {}
    for index, case in enumerate(protocol["outer_confirmation"]["strata"]["task4c_three_rule"]["cases"]):
        support[str(case["world_seed"])] = (
            case_runs["candidate"]["task4c_three_rule"][index]["target_metrics"]["score_per_round"] + 1e-12
            >= max(
                case_runs["source-r2"]["task4c_three_rule"][index]["target_metrics"]["score_per_round"],
                case_runs["v4"]["task4c_three_rule"][index]["target_metrics"]["score_per_round"],
            )
        )
    gain = float(rule["minimum_three_rule_score_gain_over_each_baseline"])
    current = {
        "score_gain_over_source": c["score_per_round"] >= s["score_per_round"] + gain - 1e-12,
        "score_gain_over_v4": c["score_per_round"] >= v["score_per_round"] + gain - 1e-12,
        "kills_not_below_source": c["kills_per_round"] >= s["kills_per_round"] + float(rule["minimum_three_rule_kills_delta_to_source"]) - 1e-12,
        "suicide_safe_vs_v4": c["suicides_per_round"] <= v["suicides_per_round"] + float(rule["maximum_three_rule_suicide_increase_over_v4"]) + 1e-12,
        "beats_rule_mean": c["score_per_round"] >= candidate["task4c_three_rule"]["opponents"]["mean_score_per_agent_round"] + float(rule["minimum_three_rule_score_margin_over_rule_mean"]) - 1e-12,
        "case_block_reproducibility": sum(support.values()) >= int(rule["minimum_three_rule_case_blocks_not_below_both_baselines"]),
    }
    passed = all(prior.values()) and all(current.values())
    return {
        "decision": rule["pass_decision"] if passed else rule["fail_decision"],
        "passed": passed,
        "prior_course_gates": prior,
        "task4c_gates": current,
        "task4a_open_rule_report_only": candidate["task4a_open_rule"],
        "three_rule_case_block_support": support,
        "three_rule_case_blocks_not_below_both_baselines": sum(support.values()),
        "task4c_complete": passed,
        "task4_complete": passed,
    }


def run_outer(protocol_path: Path, protocol: dict, protocol_hash: str, selection: dict) -> tuple[dict, dict, dict]:
    chosen = selection["result"]["selected_candidate"]
    identities = {
        "v4": {"kind": "v4"},
        "source-r2": {"kind": "source"},
        "candidate": {"kind": "candidate", "replica": chosen["replica"], "path": str(selected_checkpoint_path(protocol))},
    }
    rows, manifests, case_runs = {}, {}, {}
    for label, identity in identities.items():
        rows[label], manifests[label], case_runs[label] = evaluate_outer_identity(
            protocol_path, protocol, protocol_hash, label, identity,
        )
    return rows, manifests, outer_decision(protocol, rows, case_runs)


def freeze_task4c(protocol_path: Path, protocol: dict, protocol_hash: str, selection: dict, decision: dict) -> dict | None:
    if not decision["passed"]:
        return None
    path = freeze_manifest_path(protocol)
    destination = confirmed_checkpoint_path(protocol)
    existing = load_completed(path, FREEZE_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or sha256_file(destination) != existing["checkpoint"]["sha256"]:
            raise RuntimeError("confirmed Task4C artifact drift")
        return existing
    if destination.exists():
        raise RuntimeError("refusing orphaned confirmed Task4C checkpoint")
    source = selected_checkpoint_path(protocol)
    atomic_copy(source, destination)
    chosen = selection["result"]["selected_candidate"]
    if sha256_file(destination) != chosen["sha256"]:
        raise RuntimeError("confirmed Task4C checkpoint copy mismatch")
    record = {
        "schema_version": 1,
        "kind": FREEZE_KIND,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "decision": decision["decision"],
        "selected_candidate": chosen,
        "checkpoint": {"path": relative(destination), "sha256": sha256_file(destination)},
        "task4c_complete": True,
        "task4_complete": True,
        "default_checkpoint_modified": False,
    }
    atomic_json(path, record)
    return record


def write_report(
    protocol_path: Path, protocol: dict, protocol_hash: str, training: dict,
    selection: dict, outer_rows: dict | None, outer_manifests: dict | None,
    decision: dict, freeze: dict | None,
) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed Task4C report drift")
        return existing, sha256_file(path)
    record = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "active_subcourse": "task4c_three_rule",
        "training_manifests": {
            replica: {"path": relative(training_manifest_path(protocol, replica)), "sha256": sha256_file(training_manifest_path(protocol, replica))}
            for replica in REPLICAS
        },
        "selection_manifest": {"path": relative(selection_manifest_path(protocol)), "sha256": sha256_file(selection_manifest_path(protocol))},
        "outer_rows": outer_rows,
        "outer_evaluation_manifests": outer_manifests,
        "result": {
            **decision,
            "inner_selection_passed": selection["result"]["selection_passed"],
            "supportive_replica_count": selection["result"]["supportive_replica_count"],
            "selected_candidate": selection["result"]["selected_candidate"],
            "outer_confirmation_started": outer_rows is not None,
            "training_completed": True,
            "automatic_followup_started": False,
        },
        "confirmed_selection": None if freeze is None else {
            "path": relative(freeze_manifest_path(protocol)),
            "sha256": sha256_file(freeze_manifest_path(protocol)),
            "checkpoint": freeze["checkpoint"],
        },
        "awaiting_user_instruction": True,
        "default_checkpoint_modified": False,
        "source_artifacts_modified": False,
    }
    atomic_json(path, record)
    return record, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    inner_cases = sum(len(cases) for cases in protocol["inner_signal"]["cases_by_replica"].values())
    candidate_inner = len(REPLICAS) * len(SNAPSHOT_ROUNDS) * 2 * int(protocol["inner_signal"]["rounds_per_case"])
    baseline_inner = 2 * inner_cases * int(protocol["inner_signal"]["rounds_per_case"])
    outer_rounds = 3 * sum(
        len(spec["cases"]) * int(spec["rounds_per_case"])
        for spec in protocol["outer_confirmation"]["strata"].values()
    )
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "parent": "source-r2",
        "active_subcourse": "task4c_three_rule",
        "replicas": list(REPLICAS),
        "rounds_per_replica": int(protocol["training"]["rounds"]),
        "total_training_rounds": len(REPLICAS) * int(protocol["training"]["rounds"]),
        "snapshots": len(REPLICAS) * len(SNAPSHOT_ROUNDS),
        "inner_signal_evaluation_rounds": candidate_inner + baseline_inner,
        "maximum_outer_evaluation_rounds": outer_rounds,
        "maximum_total_evaluation_rounds": candidate_inner + baseline_inner + outer_rounds,
        "formal_training_started": False,
        "formal_evaluation_started": False,
        "automatic_followup_started": False,
        "writes_terminal_report_and_stops": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol, protocol_hash = load_protocol(protocol_path)
    assert_registered_seeds_untouched(protocol_path, protocol)
    if not args.execute:
        print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
        return 0
    existing = load_completed(report_path(protocol), REPORT_KIND)
    if existing is not None:
        print(json.dumps({
            "status": existing["status"], "report": relative(report_path(protocol)),
            "report_sha256": sha256_file(report_path(protocol)), **existing["result"],
        }, indent=2, sort_keys=True))
        return 0
    training = {replica: train_one(protocol_path.resolve(), protocol, protocol_hash, replica) for replica in REPLICAS}
    selection = run_inner(protocol_path.resolve(), protocol, protocol_hash)
    if not selection["result"]["selection_passed"]:
        decision = {
            "decision": "task4c_training_signal_not_reproducible_stop",
            "passed": False,
            "reason": "fewer than two replicas produced a stronger and safe three-rule snapshot",
            "task4c_complete": False,
            "task4_complete": False,
        }
        outer_rows = outer_manifests = freeze = None
    else:
        outer_rows, outer_manifests, decision = run_outer(protocol_path.resolve(), protocol, protocol_hash, selection)
        freeze = freeze_task4c(protocol_path.resolve(), protocol, protocol_hash, selection, decision)
    report, report_hash = write_report(
        protocol_path.resolve(), protocol, protocol_hash, training, selection,
        outer_rows, outer_manifests, decision, freeze,
    )
    print(json.dumps({
        "status": report["status"], "report": relative(report_path(protocol)),
        "report_sha256": report_hash, **report["result"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
