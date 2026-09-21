"""Run Task4A of the split exact-v4 curriculum and stop before Task4B.

Task4A trains source-r2 on an official coin-heaven board against one seeded
full-strength rule agent.  This isolates open-board score competition before
classic bomb/crate duels and three-rule play.  Dry-run is the default.
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

from agent_code.model_a_v4_task4_split.config import (  # noqa: E402
    REPLICAS, SNAPSHOT_ROUNDS, STRATA, confirmed_checkpoint_path,
    final_checkpoint_path, load_protocol, selected_checkpoint_path,
    sha256_file, snapshot_path,
)
from tools.v4_task4_duel_training import (  # noqa: E402
    aggregate, atomic_json, load_completed, metrics_by_agent, opponent_summary,
    relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task4-split-a-s125000.json"
TRAINING_KIND = "model-a-v4-task4-split-a-training"
EVALUATION_KIND = "model-a-v4-task4-split-a-evaluation"
SELECTION_KIND = "model-a-v4-task4-split-a-inner-selection"
REPORT_KIND = "model-a-v4-task4-split-a-report"
FREEZE_KIND = "model-a-v4-task4-split-a-confirmed-selection"


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
    for suite_name in ("inner_selection", "outer_confirmation"):
        for spec in protocol[suite_name]["strata"].values():
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
        raise ValueError(f"registered s125000 seeds were already used: {conflicts}")


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
    from agent_code.model_a_v4_task4_split.callbacks import _torch_load

    if not path.is_file():
        raise RuntimeError(f"Task4A checkpoint missing: {path}")
    payload = _torch_load(path)
    expected = {
        "architecture": "model-a-mlp-v4-global7",
        "protocol_sha256": protocol_hash,
        "replica": replica,
        "stage_id": "task4a_open_rule",
        "stage_completed_rounds": stage_round,
        "parent_sha256": protocol["source_parent"]["sha256"],
        "training_variable": "task4_subcourse_episode_distribution_only",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Task4A checkpoint {key} mismatch: {path}")
    if abs(float(payload.get("epsilon", -1.0)) - 0.05) > 1e-12:
        raise RuntimeError(f"Task4A epsilon changed: {path}")
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
        raise RuntimeError(f"Task4A final is not the immutable round-200 snapshot: {replica}")
    return snapshots


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, replica: str) -> dict:
    manifest_path = training_manifest_path(protocol, replica)
    output = final_checkpoint_path(protocol, replica)
    existing = load_completed(manifest_path, TRAINING_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed Task4A training protocol drift: {replica}")
        validate_training_artifacts(protocol, protocol_hash, replica)
        return existing
    stats_path = manifest_path.with_suffix(".stats.json")
    if output.parent.exists() or stats_path.exists():
        raise RuntimeError(f"refusing orphaned Task4A artifacts: {replica}")
    parent = (ROOT / protocol["source_parent"]["path"]).resolve()
    spec = protocol["training"]
    case = spec["seeds"][replica]
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_task4_split", *spec["opponents"],
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides = {
        "MODEL_A_V4T4S_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_V4T4S_CHECKPOINT_PATH": str(output),
        "MODEL_A_V4T4S_PARENT_PATH": str(parent),
        "MODEL_A_V4T4S_REPLICA": replica,
        "MODEL_A_V4T4S_SEED": str(case["agent_seed"]),
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
        "active_subcourse": protocol["active_subcourse"],
        "training": {**case, **{key: spec[key] for key in ("scenario", "opponents", "rounds")}},
        "parent_checkpoint": {"path": relative(parent), "sha256": sha256_file(parent)},
        "checkpoint": {"path": relative(output), "sha256": None},
        "command": command,
        "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest_path, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats_path.is_file():
            raise RuntimeError("Task4A training process or stats failed")
        snapshots = validate_training_artifacts(protocol, protocol_hash, replica)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        record["metrics_by_agent"] = metrics_by_agent(stats)
        record["snapshots"] = snapshots
        record["checkpoint"]["sha256"] = sha256_file(output)
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest_path, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task4A training failed: {replica}: {record.get('error')}")
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
    return "model_a_v4_task4_split", checkpoint, {
        "MODEL_A_V4T4S_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_V4T4S_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_V4T4S_REPLICA": replica,
        "MODEL_A_V4T4S_SEED": str(case["agent_seed"]),
    }


def evaluate_one(
    protocol_path: Path, protocol: dict, protocol_hash: str, suite: str,
    stratum: str, spec: dict, label: str, identity: dict, case: dict,
) -> dict:
    target, checkpoint, overrides = checkpoint_environment(protocol_path, protocol, identity, case)
    checkpoint_hash = sha256_file(checkpoint)
    output = evaluation_manifest_path(protocol, suite, stratum, label, int(case["world_seed"]))
    expected = {**case, **spec, "target_agent": target}
    existing = load_completed(output, EVALUATION_KIND)
    if existing is not None:
        if (
            existing.get("protocol_sha256") != protocol_hash
            or existing.get("evaluation") != expected
            or existing.get("checkpoint", {}).get("sha256") != checkpoint_hash
        ):
            raise RuntimeError(f"completed Task4A evaluation drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned Task4A stats: {stats_path}")
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
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        record["metrics_by_agent"] = metrics_by_agent(stats)
        record["target_metrics"] = record["metrics_by_agent"].get(target)
        record["status"] = "completed" if record["target_metrics"] else "failed"
    else:
        record["status"] = "failed"
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task4A evaluation failed: {suite}/{stratum}/{label}")
    return record


def evaluate_identity(
    protocol_path: Path, protocol: dict, protocol_hash: str, suite: str,
    label: str, identity: dict,
) -> tuple[dict, dict]:
    rows, manifests = {}, {}
    for stratum, spec in protocol[suite]["strata"].items():
        runs = [
            evaluate_one(protocol_path, protocol, protocol_hash, suite, stratum, spec, label, identity, case)
            for case in spec["cases"]
        ]
        target, _, _ = checkpoint_environment(protocol_path, protocol, identity, spec["cases"][0])
        rows[stratum] = {"target": aggregate(runs), "opponents": opponent_summary(runs, target)}
        manifests[stratum] = [
            relative(evaluation_manifest_path(protocol, suite, stratum, label, int(case["world_seed"])))
            for case in spec["cases"]
        ]
    return rows, manifests


def inner_score(protocol: dict, candidate: dict, source: dict) -> dict:
    rule = protocol["inner_selection"]["supportive_rule"]
    target = lambda name: candidate[name]["target"]
    parent = lambda name: source[name]["target"]
    retention = {
        "task1": target("task1")["score_per_round"] >= float(rule["minimum_task1_fraction_of_source"]) * parent("task1")["score_per_round"] - 1e-12,
        "task2": target("task2")["score_per_round"] >= parent("task2")["score_per_round"] - float(rule["maximum_task2_score_drop_from_source"]) - 1e-12,
        "peaceful_score": target("task3_peaceful")["score_per_round"] >= parent("task3_peaceful")["score_per_round"] - float(rule["maximum_task3_score_drop_from_source"]) - 1e-12,
        "peaceful_kills": target("task3_peaceful")["kills_per_round"] >= parent("task3_peaceful")["kills_per_round"] - float(rule["maximum_task3_kills_drop_from_source"]) - 1e-12,
        "coin_score": target("task3_coin")["score_per_round"] >= parent("task3_coin")["score_per_round"] - float(rule["maximum_task3_score_drop_from_source"]) - 1e-12,
        "coin_kills": target("task3_coin")["kills_per_round"] >= parent("task3_coin")["kills_per_round"] - float(rule["maximum_task3_kills_drop_from_source"]) - 1e-12,
        "classic_duel_score": target("task4b_classic_duel")["score_per_round"] >= parent("task4b_classic_duel")["score_per_round"] - float(rule["maximum_classic_duel_score_drop_from_source"]) - 1e-12,
        "classic_duel_suicide": target("task4b_classic_duel")["suicides_per_round"] <= parent("task4b_classic_duel")["suicides_per_round"] + float(rule["maximum_classic_duel_suicide_increase_over_source"]) + 1e-12,
        "three_rule_score": target("task4c_three_rule")["score_per_round"] >= parent("task4c_three_rule")["score_per_round"] - float(rule["maximum_three_rule_score_drop_from_source"]) - 1e-12,
        "three_rule_kills": target("task4c_three_rule")["kills_per_round"] >= parent("task4c_three_rule")["kills_per_round"] - float(rule["maximum_three_rule_kills_drop_from_source"]) - 1e-12,
        "three_rule_suicide": target("task4c_three_rule")["suicides_per_round"] <= parent("task4c_three_rule")["suicides_per_round"] + float(rule["maximum_three_rule_suicide_increase_over_source"]) + 1e-12,
    }
    current = {
        "open_rule_score_gain": target("task4a_open_rule")["score_per_round"] >= parent("task4a_open_rule")["score_per_round"] + float(rule["minimum_open_rule_score_gain_over_source"]) - 1e-12,
        "open_rule_beats_rule": target("task4a_open_rule")["score_per_round"] >= candidate["task4a_open_rule"]["opponents"]["mean_score_per_agent_round"] - 1e-12,
        "open_rule_suicide": target("task4a_open_rule")["suicides_per_round"] <= parent("task4a_open_rule")["suicides_per_round"] + float(rule["maximum_open_rule_suicide_increase_over_source"]) + 1e-12,
    }
    return {
        "retention_gates": retention,
        "current_subcourse_gates": current,
        "retention_eligible": all(retention.values()),
        "supportive": all(retention.values()) and all(current.values()),
        "open_rule_score_gain_over_source": target("task4a_open_rule")["score_per_round"] - parent("task4a_open_rule")["score_per_round"],
        "open_rule_score_margin_over_rule": target("task4a_open_rule")["score_per_round"] - candidate["task4a_open_rule"]["opponents"]["mean_score_per_agent_round"],
    }


def selection_rank(item: dict) -> tuple:
    rows = item["rows"]
    return (
        item["score"]["open_rule_score_margin_over_rule"],
        rows["task4b_classic_duel"]["target"]["score_per_round"],
        -rows["task4b_classic_duel"]["target"]["suicides_per_round"],
        rows["task4c_three_rule"]["target"]["score_per_round"],
        -int(item["stage_round"]),
    )


def run_inner(protocol_path: Path, protocol: dict, protocol_hash: str) -> dict:
    path = selection_manifest_path(protocol)
    existing = load_completed(path, SELECTION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed Task4A selection protocol drift")
        if existing["result"]["selection_passed"]:
            chosen = existing["result"]["selected_candidate"]
            selected = selected_checkpoint_path(protocol)
            validate_candidate_checkpoint(
                protocol, protocol_hash, selected, chosen["replica"], chosen["stage_round"],
            )
            if sha256_file(selected) != chosen["sha256"]:
                raise RuntimeError("selected Task4A checkpoint drift")
        return existing
    baselines, baseline_manifests = {}, {}
    for label, identity in (("v4", {"kind": "v4"}), ("source-r2", {"kind": "source"})):
        baselines[label], baseline_manifests[label] = evaluate_identity(
            protocol_path, protocol, protocol_hash, "inner_selection", label, identity,
        )
    source = baselines["source-r2"]
    candidates, candidate_manifests, choices, supportive_choices = {}, {}, {}, []
    for replica in REPLICAS:
        items = []
        for stage_round in SNAPSHOT_ROUNDS:
            checkpoint = snapshot_path(protocol, replica, stage_round)
            label = f"{replica}-round-{stage_round:04d}"
            identity = {"kind": "candidate", "replica": replica, "path": str(checkpoint)}
            rows, manifests = evaluate_identity(
                protocol_path, protocol, protocol_hash, "inner_selection", label, identity,
            )
            item = {
                "replica": replica,
                "stage_round": stage_round,
                "path": relative(checkpoint),
                "sha256": sha256_file(checkpoint),
                "rows": rows,
            }
            item["score"] = inner_score(protocol, rows, source)
            candidates[label], candidate_manifests[label] = item, manifests
            items.append(item)
        supportive = [item for item in items if item["score"]["supportive"]]
        eligible = [item for item in items if item["score"]["retention_eligible"]]
        chosen = max(supportive or eligible or items, key=selection_rank)
        choices[replica] = {
            "supportive_snapshot_exists": bool(supportive),
            "chosen_for_replica": {
                key: chosen[key] for key in ("replica", "stage_round", "path", "sha256", "score")
            },
        }
        if supportive:
            supportive_choices.append(max(supportive, key=selection_rank))
    required = int(protocol["inner_selection"]["supportive_replicas_required"])
    selected_item = max(supportive_choices, key=selection_rank) if len(supportive_choices) >= required else None
    if selected_item is not None:
        selected = selected_checkpoint_path(protocol)
        if selected.exists():
            raise RuntimeError("refusing orphaned Task4A selected checkpoint")
        atomic_copy(ROOT / selected_item["path"], selected)
        if sha256_file(selected) != selected_item["sha256"]:
            raise RuntimeError("Task4A selected checkpoint copy mismatch")
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
        "baselines": baselines,
        "baseline_evaluation_manifests": baseline_manifests,
        "candidates": candidates,
        "candidate_evaluation_manifests": candidate_manifests,
        "result": result,
    }
    atomic_json(path, record)
    return record


def outer_decision(protocol: dict, rows: dict) -> dict:
    rule = protocol["outer_confirmation"]["decision_rule"]
    candidate, source, v4 = rows["candidate"], rows["source-r2"], rows["v4"]
    t = lambda group, name: group[name]["target"]
    prior = {
        "task1": t(candidate, "task1")["score_per_round"] >= float(rule["minimum_task1_fraction_of_v4"]) * t(v4, "task1")["score_per_round"] - 1e-12,
        "task2": t(candidate, "task2")["score_per_round"] >= t(v4, "task2")["score_per_round"] - float(rule["maximum_task2_score_gap_to_v4"]) - 1e-12,
        "peaceful_score": t(candidate, "task3_peaceful")["score_per_round"] >= t(v4, "task3_peaceful")["score_per_round"] + float(rule["minimum_peaceful_score_gain_over_v4"]) - 1e-12,
        "peaceful_kills": t(candidate, "task3_peaceful")["kills_per_round"] >= max(float(rule["minimum_peaceful_kills_per_round"]), t(v4, "task3_peaceful")["kills_per_round"]) - 1e-12,
        "peaceful_suicide": t(candidate, "task3_peaceful")["suicides_per_round"] <= t(v4, "task3_peaceful")["suicides_per_round"] + float(rule["maximum_suicide_increase_over_v4"]) + 1e-12,
        "coin_score": t(candidate, "task3_coin")["score_per_round"] >= t(v4, "task3_coin")["score_per_round"] - float(rule["maximum_coin_score_gap_to_v4"]) - 1e-12,
        "coin_kills": t(candidate, "task3_coin")["kills_per_round"] >= max(float(rule["minimum_coin_kills_per_round"]), t(v4, "task3_coin")["kills_per_round"]) - 1e-12,
        "coin_suicide": t(candidate, "task3_coin")["suicides_per_round"] <= t(v4, "task3_coin")["suicides_per_round"] + float(rule["maximum_suicide_increase_over_v4"]) + 1e-12,
    }
    current = {
        "open_rule_beats_rule": t(candidate, "task4a_open_rule")["score_per_round"] >= candidate["task4a_open_rule"]["opponents"]["mean_score_per_agent_round"] - 1e-12,
        "open_rule_not_below_source": t(candidate, "task4a_open_rule")["score_per_round"] >= t(source, "task4a_open_rule")["score_per_round"] - 1e-12,
        "open_rule_suicide": t(candidate, "task4a_open_rule")["suicides_per_round"] <= t(source, "task4a_open_rule")["suicides_per_round"] + float(rule["maximum_open_rule_suicide_increase_over_source"]) + 1e-12,
    }
    future = {
        "classic_duel_score": t(candidate, "task4b_classic_duel")["score_per_round"] >= t(source, "task4b_classic_duel")["score_per_round"] - float(rule["maximum_future_score_drop_from_source"]) - 1e-12,
        "classic_duel_suicide": t(candidate, "task4b_classic_duel")["suicides_per_round"] <= t(source, "task4b_classic_duel")["suicides_per_round"] + float(rule["maximum_future_suicide_increase_over_source"]) + 1e-12,
        "three_rule_score": t(candidate, "task4c_three_rule")["score_per_round"] >= t(source, "task4c_three_rule")["score_per_round"] - float(rule["maximum_future_score_drop_from_source"]) - 1e-12,
        "three_rule_kills": t(candidate, "task4c_three_rule")["kills_per_round"] >= t(source, "task4c_three_rule")["kills_per_round"] - float(rule["maximum_future_kills_drop_from_source"]) - 1e-12,
        "three_rule_suicide": t(candidate, "task4c_three_rule")["suicides_per_round"] <= t(source, "task4c_three_rule")["suicides_per_round"] + float(rule["maximum_future_suicide_increase_over_source"]) + 1e-12,
    }
    passed = all(prior.values()) and all(current.values()) and all(future.values())
    return {
        "decision": "task4a_open_rule_confirmed_stop" if passed else "task4a_candidate_rejected_stop",
        "passed": passed,
        "prior_course_gates": prior,
        "current_subcourse_gates": current,
        "future_task4_retention_gates": future,
        "task4a_complete": passed,
        "task4_complete": False,
        "next_subcourse": "task4b_classic_duel" if passed else None,
    }


def run_outer(protocol_path: Path, protocol: dict, protocol_hash: str, selection: dict) -> tuple[dict, dict, dict]:
    chosen = selection["result"]["selected_candidate"]
    identities = {
        "v4": {"kind": "v4"},
        "source-r2": {"kind": "source"},
        "candidate": {"kind": "candidate", "replica": chosen["replica"], "path": str(selected_checkpoint_path(protocol))},
    }
    rows, manifests = {}, {}
    for label, identity in identities.items():
        rows[label], manifests[label] = evaluate_identity(
            protocol_path, protocol, protocol_hash, "outer_confirmation", label, identity,
        )
    return rows, manifests, outer_decision(protocol, rows)


def freeze_task4a(protocol_path: Path, protocol: dict, protocol_hash: str, selection: dict, decision: dict) -> dict | None:
    if not decision["passed"]:
        return None
    path = freeze_manifest_path(protocol)
    existing = load_completed(path, FREEZE_KIND)
    destination = confirmed_checkpoint_path(protocol)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or sha256_file(destination) != existing["checkpoint"]["sha256"]:
            raise RuntimeError("confirmed Task4A artifact drift")
        return existing
    if destination.exists():
        raise RuntimeError("refusing orphaned confirmed Task4A checkpoint")
    source = selected_checkpoint_path(protocol)
    atomic_copy(source, destination)
    chosen = selection["result"]["selected_candidate"]
    if sha256_file(destination) != chosen["sha256"]:
        raise RuntimeError("confirmed Task4A checkpoint copy mismatch")
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
        "task4a_complete": True,
        "task4_complete": False,
        "next_subcourse": "task4b_classic_duel",
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
            raise RuntimeError("completed Task4A report drift")
        return existing, sha256_file(path)
    record = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "active_subcourse": "task4a_open_rule",
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
            "automatic_task4b_started": False,
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
    inner_labels = 2 + len(REPLICAS) * len(SNAPSHOT_ROUNDS)
    inner_rounds = inner_labels * sum(
        len(spec["cases"]) * int(spec["rounds_per_case"])
        for spec in protocol["inner_selection"]["strata"].values()
    )
    outer_rounds = 3 * sum(
        len(spec["cases"]) * int(spec["rounds_per_case"])
        for spec in protocol["outer_confirmation"]["strata"].values()
    )
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "task4_subcourse_order": protocol["task4_subcourse_order"],
        "active_subcourse": protocol["active_subcourse"],
        "replicas": list(REPLICAS),
        "rounds_per_replica": int(protocol["training"]["rounds"]),
        "total_training_rounds": len(REPLICAS) * int(protocol["training"]["rounds"]),
        "snapshots": len(REPLICAS) * len(SNAPSHOT_ROUNDS),
        "inner_evaluation_rounds": inner_rounds,
        "maximum_outer_evaluation_rounds": outer_rounds,
        "maximum_total_evaluation_rounds": inner_rounds + outer_rounds,
        "formal_training_started": False,
        "formal_evaluation_started": False,
        "automatic_task4b_started": False,
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
            "decision": "task4a_training_not_reproducible_stop",
            "passed": False,
            "reason": "fewer than two replicas mastered open-rule competition while retaining prior abilities",
            "task4a_complete": False,
            "task4_complete": False,
            "next_subcourse": None,
        }
        outer_rows = outer_manifests = freeze = None
    else:
        outer_rows, outer_manifests, decision = run_outer(protocol_path.resolve(), protocol, protocol_hash, selection)
        freeze = freeze_task4a(protocol_path.resolve(), protocol, protocol_hash, selection, decision)
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
