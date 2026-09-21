"""Run the single-variable exact-v4 Task4 duel course and stop.

Dry-run is the default. ``--execute`` trains three replications of one fixed
distribution, evaluates immutable snapshots on independent inner seeds, and
only when at least two replicas are supportive confirms one preregistered
candidate on fresh outer seeds.  It never launches a follow-up experiment.
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

from agent_code.model_a_v4_task4_duel.config import (  # noqa: E402
    REPLICAS, SNAPSHOT_ROUNDS, STRATA, confirmed_checkpoint_path,
    final_checkpoint_path, load_protocol, selected_checkpoint_path,
    sha256_file, snapshot_path,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task4-duel-training-s124000.json"
TRAINING_KIND = "model-a-v4-task4-duel-training-run"
EVALUATION_KIND = "model-a-v4-task4-duel-evaluation"
SELECTION_KIND = "model-a-v4-task4-duel-inner-selection"
REPORT_KIND = "model-a-v4-task4-duel-training-report"
FREEZE_KIND = "model-a-v4-task4-duel-confirmed-selection"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def load_completed(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("kind") != kind:
        raise RuntimeError(f"refusing incomplete or incompatible artifact: {path}")
    return payload


def seed_values(payload: object) -> set[int]:
    found: set[int] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "seed" or key.endswith("_seed") or key.endswith("_seeds"):
                values = value if isinstance(value, list) else [value]
                found.update(item for item in values if isinstance(item, int))
            found.update(seed_values(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(seed_values(value))
    return found


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = set()
    for seeds in protocol["training"]["seeds"].values():
        registered.update(int(value) for value in seeds.values())
    for suite_name in ("inner_selection", "outer_confirmation"):
        for spec in protocol[suite_name]["strata"].values():
            for case in spec["cases"]:
                registered.update(int(value) for value in case.values())
    excluded_roots = {
        (ROOT / protocol["training_manifest_directory"]).resolve(),
        (ROOT / protocol["evaluation_manifest_directory"]).resolve(),
        (ROOT / protocol["checkpoint_directory"]).resolve(),
    }
    excluded_files = {
        protocol_path.resolve(),
        (ROOT / protocol["selection_manifest_path"]).resolve(),
        (ROOT / protocol["report_path"]).resolve(),
        (ROOT / protocol["confirmed_selection_manifest_path"]).resolve(),
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
        raise ValueError(f"registered s124000 seeds were already used: {conflicts}")


def metrics_by_agent(stats: dict) -> dict[str, dict]:
    result = {}
    for name, raw in stats.get("by_agent", {}).items():
        rounds = int(raw.get("rounds", 0))
        steps = int(raw.get("steps", 0))
        moves = int(raw.get("moves", 0))
        bombs = int(raw.get("bombs", 0))
        invalid = int(raw.get("invalid", 0))
        waits = max(0, steps - moves - bombs - invalid)
        result[name] = {
            "rounds": rounds,
            "score": int(raw.get("score", 0)),
            "score_per_round": int(raw.get("score", 0)) / rounds if rounds else 0.0,
            "coins": int(raw.get("coins", 0)),
            "coins_per_round": int(raw.get("coins", 0)) / rounds if rounds else 0.0,
            "kills": int(raw.get("kills", 0)),
            "kills_per_round": int(raw.get("kills", 0)) / rounds if rounds else 0.0,
            "crates": int(raw.get("crates", 0)),
            "crates_per_round": int(raw.get("crates", 0)) / rounds if rounds else 0.0,
            "bombs": bombs,
            "bombs_per_round": bombs / rounds if rounds else 0.0,
            "moves": moves,
            "move_fraction": moves / steps if steps else 0.0,
            "waits": waits,
            "wait_fraction": waits / steps if steps else 0.0,
            "suicides": int(raw.get("suicides", 0)),
            "suicides_per_round": int(raw.get("suicides", 0)) / rounds if rounds else 0.0,
            "invalid_actions": invalid,
            "invalid_actions_per_round": invalid / rounds if rounds else 0.0,
            "steps": steps,
        }
    return result


def aggregate(runs: list[dict]) -> dict:
    additive = (
        "rounds", "score", "coins", "kills", "crates", "bombs", "moves",
        "waits", "suicides", "invalid_actions", "steps",
    )
    totals = {key: sum(int(run["target_metrics"][key]) for run in runs) for key in additive}
    rounds, steps = totals["rounds"], totals["steps"]
    return {
        **totals,
        "score_per_round": totals["score"] / rounds,
        "coins_per_round": totals["coins"] / rounds,
        "kills_per_round": totals["kills"] / rounds,
        "crates_per_round": totals["crates"] / rounds,
        "bombs_per_round": totals["bombs"] / rounds,
        "move_fraction": totals["moves"] / steps if steps else 0.0,
        "wait_fraction": totals["waits"] / steps if steps else 0.0,
        "suicides_per_round": totals["suicides"] / rounds,
        "invalid_actions_per_round": totals["invalid_actions"] / rounds,
    }


def opponent_summary(runs: list[dict], target: str) -> dict:
    by_name: dict[str, dict[str, int]] = {}
    for run in runs:
        for name, metrics in run["metrics_by_agent"].items():
            if name == target:
                continue
            totals = by_name.setdefault(name, {"rounds": 0, "score": 0, "kills": 0, "suicides": 0})
            for key in totals:
                totals[key] += int(metrics[key])
    summaries = {
        name: {
            **values,
            "score_per_round": values["score"] / values["rounds"],
            "kills_per_round": values["kills"] / values["rounds"],
            "suicides_per_round": values["suicides"] / values["rounds"],
        }
        for name, values in by_name.items()
    }
    agent_rounds = sum(item["rounds"] for item in summaries.values())
    return {
        "by_agent": summaries,
        "opponent_count": len(summaries),
        "mean_score_per_agent_round": (
            sum(item["score"] for item in summaries.values()) / agent_rounds if agent_rounds else 0.0
        ),
        "mean_kills_per_agent_round": (
            sum(item["kills"] for item in summaries.values()) / agent_rounds if agent_rounds else 0.0
        ),
    }


def training_manifest_path(protocol: dict, replica: str) -> Path:
    return ROOT / protocol["training_manifest_directory"] / f"{replica}.json"


def evaluation_manifest_path(protocol: dict, suite: str, stratum: str, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / suite / stratum / f"{label}-s{world_seed}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def selection_manifest_path(protocol: dict) -> Path:
    return ROOT / protocol["selection_manifest_path"]


def freeze_manifest_path(protocol: dict) -> Path:
    return ROOT / protocol["confirmed_selection_manifest_path"]


def validate_candidate_checkpoint(
    protocol: dict, protocol_hash: str, path: Path, replica: str, stage_round: int,
) -> dict:
    from agent_code.model_a_v4_task4_duel.callbacks import _torch_load

    if not path.is_file():
        raise RuntimeError(f"Task4 checkpoint missing: {path}")
    payload = _torch_load(path)
    expected = {
        "architecture": "model-a-mlp-v4-global7",
        "protocol_sha256": protocol_hash,
        "replica": replica,
        "stage_id": "task4_duel",
        "stage_completed_rounds": stage_round,
        "parent_sha256": protocol["source_parent"]["sha256"],
        "training_variable": "opponent_sampling_distribution_only",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Task4 checkpoint {key} mismatch: {path}")
    if abs(float(payload.get("epsilon", -1.0)) - 0.05) > 1e-12:
        raise RuntimeError(f"Task4 checkpoint changed the epsilon floor: {path}")
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
    validate_candidate_checkpoint(protocol, protocol_hash, final, replica, 600)
    if sha256_file(final) != snapshots[-1]["sha256"]:
        raise RuntimeError(f"Task4 final checkpoint is not the immutable round-600 snapshot: {replica}")
    return snapshots


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, replica: str) -> dict:
    manifest_path = training_manifest_path(protocol, replica)
    output = final_checkpoint_path(protocol, replica)
    existing = load_completed(manifest_path, TRAINING_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed Task4 training protocol drift: {replica}")
        validate_training_artifacts(protocol, protocol_hash, replica)
        return existing
    stats_path = manifest_path.with_suffix(".stats.json")
    if output.parent.exists() or stats_path.exists():
        raise RuntimeError(f"refusing orphaned Task4 training artifacts: {replica}")
    parent = (ROOT / protocol["source_parent"]["path"]).resolve()
    spec = protocol["training"]
    seeds = spec["seeds"][replica]
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_task4_duel", *spec["opponents"],
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds"]),
        "--seed", str(seeds["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides = {
        "MODEL_A_V4T4D_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_V4T4D_CHECKPOINT_PATH": str(output),
        "MODEL_A_V4T4D_PARENT_PATH": str(parent),
        "MODEL_A_V4T4D_REPLICA": replica,
        "MODEL_A_V4T4D_SEED": str(seeds["agent_seed"]),
        "TASK4_RULE_SEED": str(seeds["opponent_seed"]),
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
        "single_training_variable": protocol["single_training_variable"],
        "training": {**seeds, **{key: spec[key] for key in ("scenario", "opponents", "rounds")}},
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
            raise RuntimeError("Task4 training process or stats output failed")
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
        raise RuntimeError(f"Task4 training failed: {replica}: {record.get('error')}")
    return record


def checkpoint_environment(protocol_path: Path, protocol: dict, identity: dict, case: dict) -> tuple[str, Path, dict[str, str]]:
    kind = identity["kind"]
    if kind == "v4":
        checkpoint = (ROOT / protocol["frozen_v4_baseline"]["path"]).resolve()
        return "model_a_dqn", checkpoint, {
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_SEED": str(case["agent_seed"]),
        }
    if kind == "source":
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
    return "model_a_v4_task4_duel", checkpoint, {
        "MODEL_A_V4T4D_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_V4T4D_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_V4T4D_REPLICA": replica,
        "MODEL_A_V4T4D_SEED": str(case["agent_seed"]),
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
            raise RuntimeError(f"completed Task4 evaluation drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned Task4 evaluation stats: {stats_path}")
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
        raise RuntimeError(f"Task4 evaluation failed: {suite}/{stratum}/{label}")
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
        rows[stratum] = {
            "target": aggregate(runs),
            "opponents": opponent_summary(runs, target),
        }
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
        "task1_score": target("task1")["score_per_round"] >= float(rule["minimum_task1_fraction_of_source"]) * parent("task1")["score_per_round"] - 1e-12,
        "task2_score": target("task2")["score_per_round"] >= parent("task2")["score_per_round"] - float(rule["maximum_task2_score_drop_from_source"]) - 1e-12,
        "peaceful_score": target("task3_peaceful")["score_per_round"] >= parent("task3_peaceful")["score_per_round"] - float(rule["maximum_task3_score_drop_from_source"]) - 1e-12,
        "peaceful_kills": target("task3_peaceful")["kills_per_round"] >= parent("task3_peaceful")["kills_per_round"] - float(rule["maximum_task3_kills_drop_from_source"]) - 1e-12,
        "coin_score": target("task3_coin")["score_per_round"] >= parent("task3_coin")["score_per_round"] - float(rule["maximum_task3_score_drop_from_source"]) - 1e-12,
        "coin_kills": target("task3_coin")["kills_per_round"] >= parent("task3_coin")["kills_per_round"] - float(rule["maximum_task3_kills_drop_from_source"]) - 1e-12,
        "three_rule_score": target("task4_three_rule")["score_per_round"] >= parent("task4_three_rule")["score_per_round"] - float(rule["maximum_three_rule_score_drop_from_source"]) - 1e-12,
        "three_rule_kills": target("task4_three_rule")["kills_per_round"] >= parent("task4_three_rule")["kills_per_round"] - float(rule["maximum_three_rule_kills_drop_from_source"]) - 1e-12,
        "three_rule_suicide": target("task4_three_rule")["suicides_per_round"] <= parent("task4_three_rule")["suicides_per_round"] + float(rule["maximum_three_rule_suicide_increase_over_source"]) + 1e-12,
    }
    progress = {
        "duel_score": target("task4_rule_duel")["score_per_round"] >= parent("task4_rule_duel")["score_per_round"] + float(rule["minimum_duel_score_gain_over_source"]) - 1e-12,
        "duel_suicide": target("task4_rule_duel")["suicides_per_round"] <= parent("task4_rule_duel")["suicides_per_round"] - float(rule["minimum_duel_suicide_reduction_from_source"]) + 1e-12,
    }
    return {
        "retention_gates": retention,
        "duel_progress_gates": progress,
        "retention_eligible": all(retention.values()),
        "supportive": all(retention.values()) and all(progress.values()),
        "duel_score_gain_over_source": target("task4_rule_duel")["score_per_round"] - parent("task4_rule_duel")["score_per_round"],
        "duel_suicide_reduction_from_source": parent("task4_rule_duel")["suicides_per_round"] - target("task4_rule_duel")["suicides_per_round"],
        "duel_score_margin_over_rule": target("task4_rule_duel")["score_per_round"] - candidate["task4_rule_duel"]["opponents"]["mean_score_per_agent_round"],
    }


def selection_rank(item: dict) -> tuple:
    rows = item["rows"]
    return (
        rows["task4_rule_duel"]["target"]["score_per_round"],
        -rows["task4_rule_duel"]["target"]["suicides_per_round"],
        rows["task4_rule_duel"]["target"]["kills_per_round"],
        rows["task4_three_rule"]["target"]["score_per_round"],
        -int(item["stage_round"]),
    )


def run_inner_selection(protocol_path: Path, protocol: dict, protocol_hash: str) -> dict:
    path = selection_manifest_path(protocol)
    existing = load_completed(path, SELECTION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed Task4 selection protocol drift")
        if existing["result"]["supportive_replica_count"] >= 2:
            selected = selected_checkpoint_path(protocol)
            chosen = existing["result"]["selected_candidate"]
            validate_candidate_checkpoint(protocol, protocol_hash, selected, chosen["replica"], chosen["stage_round"])
            if sha256_file(selected) != chosen["sha256"]:
                raise RuntimeError("selected Task4 checkpoint drift")
        return existing

    baseline_rows, baseline_manifests = {}, {}
    for label, identity in (("v4", {"kind": "v4"}), ("source-r2", {"kind": "source"})):
        baseline_rows[label], baseline_manifests[label] = evaluate_identity(
            protocol_path, protocol, protocol_hash, "inner_selection", label, identity,
        )
    source = baseline_rows["source-r2"]
    candidates, candidate_manifests = {}, {}
    replica_choices = {}
    supportive_choices = []
    for replica in REPLICAS:
        items = []
        for stage_round in SNAPSHOT_ROUNDS:
            checkpoint = snapshot_path(protocol, replica, stage_round)
            label = f"{replica}-round-{stage_round:04d}"
            identity = {"kind": "candidate", "replica": replica, "path": str(checkpoint)}
            rows, manifests = evaluate_identity(
                protocol_path, protocol, protocol_hash, "inner_selection", label, identity,
            )
            score = inner_score(protocol, rows, source)
            item = {
                "replica": replica,
                "stage_round": stage_round,
                "path": relative(checkpoint),
                "sha256": sha256_file(checkpoint),
                "rows": rows,
                "score": score,
            }
            candidates[label] = item
            candidate_manifests[label] = manifests
            items.append(item)
        supportive = [item for item in items if item["score"]["supportive"]]
        eligible = [item for item in items if item["score"]["retention_eligible"]]
        choice_pool = supportive or eligible or items
        chosen = max(choice_pool, key=selection_rank)
        replica_choices[replica] = {
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
            raise RuntimeError("refusing orphaned Task4 selected checkpoint")
        atomic_copy(ROOT / selected_item["path"], selected)
        if sha256_file(selected) != selected_item["sha256"]:
            raise RuntimeError("Task4 selected checkpoint copy mismatch")
    result = {
        "supportive_replicas_required": required,
        "supportive_replica_count": len(supportive_choices),
        "replica_choices": replica_choices,
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
        "baselines": baseline_rows,
        "baseline_evaluation_manifests": baseline_manifests,
        "candidates": candidates,
        "candidate_evaluation_manifests": candidate_manifests,
        "result": result,
    }
    atomic_json(path, record)
    return record


def outer_decision(protocol: dict, rows: dict) -> dict:
    rule = protocol["outer_confirmation"]["decision_rule"]
    candidate = rows["candidate"]
    v4 = rows["v4"]
    gates = {
        "task1_retention": candidate["task1"]["target"]["score_per_round"] >= float(rule["minimum_task1_retained_fraction_vs_v4"]) * v4["task1"]["target"]["score_per_round"] - 1e-12,
        "task2_score": candidate["task2"]["target"]["score_per_round"] >= v4["task2"]["target"]["score_per_round"] - float(rule["maximum_task2_score_gap_to_v4"]) - 1e-12,
        "peaceful_score": candidate["task3_peaceful"]["target"]["score_per_round"] >= v4["task3_peaceful"]["target"]["score_per_round"] + float(rule["minimum_peaceful_score_gain_over_v4"]) - 1e-12,
        "peaceful_kills_absolute": candidate["task3_peaceful"]["target"]["kills_per_round"] >= float(rule["minimum_peaceful_kills_per_round"]) - 1e-12,
        "peaceful_kills_relative": candidate["task3_peaceful"]["target"]["kills_per_round"] >= v4["task3_peaceful"]["target"]["kills_per_round"] - 1e-12,
        "peaceful_suicide": candidate["task3_peaceful"]["target"]["suicides_per_round"] <= v4["task3_peaceful"]["target"]["suicides_per_round"] + float(rule["maximum_suicide_increase_over_v4"]) + 1e-12,
        "coin_score": candidate["task3_coin"]["target"]["score_per_round"] >= v4["task3_coin"]["target"]["score_per_round"] - float(rule["maximum_coin_score_gap_to_v4"]) - 1e-12,
        "coin_kills_absolute": candidate["task3_coin"]["target"]["kills_per_round"] >= float(rule["minimum_coin_kills_per_round"]) - 1e-12,
        "coin_kills_relative": candidate["task3_coin"]["target"]["kills_per_round"] >= v4["task3_coin"]["target"]["kills_per_round"] - 1e-12,
        "coin_suicide": candidate["task3_coin"]["target"]["suicides_per_round"] <= v4["task3_coin"]["target"]["suicides_per_round"] + float(rule["maximum_suicide_increase_over_v4"]) + 1e-12,
    }
    task4 = {}
    for stratum in ("task4_rule_duel", "task4_three_rule"):
        task4[stratum] = {
            "beats_rule": candidate[stratum]["target"]["score_per_round"] >= candidate[stratum]["opponents"]["mean_score_per_agent_round"] - 1e-12,
            "score_noninferior_to_v4": candidate[stratum]["target"]["score_per_round"] >= v4[stratum]["target"]["score_per_round"] - float(rule["maximum_task4_score_gap_to_v4"]) - 1e-12,
            "kills_not_below_v4": candidate[stratum]["target"]["kills_per_round"] >= v4[stratum]["target"]["kills_per_round"] - 1e-12,
            "suicide_safe_vs_v4": candidate[stratum]["target"]["suicides_per_round"] <= v4[stratum]["target"]["suicides_per_round"] + float(rule["maximum_suicide_increase_over_v4"]) + 1e-12,
        }
    passed = all(gates.values()) and all(all(item.values()) for item in task4.values())
    return {
        "decision": "task4_duel_training_confirmed" if passed else "task4_duel_candidate_rejected_stop",
        "passed": passed,
        "retention_and_task3_gates": gates,
        "task4_gates": task4,
        "invalid_actions_policy": "reported only; legal mask and policy code are unchanged",
    }


def run_outer(protocol_path: Path, protocol: dict, protocol_hash: str, selection: dict) -> tuple[dict, dict, dict]:
    chosen = selection["result"]["selected_candidate"]
    identities = {
        "v4": {"kind": "v4"},
        "source-r2": {"kind": "source"},
        "candidate": {
            "kind": "candidate",
            "replica": chosen["replica"],
            "path": str(selected_checkpoint_path(protocol)),
        },
    }
    rows, manifests = {}, {}
    for label, identity in identities.items():
        rows[label], manifests[label] = evaluate_identity(
            protocol_path, protocol, protocol_hash, "outer_confirmation", label, identity,
        )
    return rows, manifests, outer_decision(protocol, rows)


def freeze_confirmed(protocol_path: Path, protocol: dict, protocol_hash: str, selection: dict, decision: dict) -> dict | None:
    if not decision["passed"]:
        return None
    manifest_path = freeze_manifest_path(protocol)
    existing = load_completed(manifest_path, FREEZE_KIND)
    destination = confirmed_checkpoint_path(protocol)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or sha256_file(destination) != existing["checkpoint"]["sha256"]:
            raise RuntimeError("confirmed Task4 artifact drift")
        return existing
    if destination.exists():
        raise RuntimeError("refusing orphaned confirmed Task4 checkpoint")
    source = selected_checkpoint_path(protocol)
    atomic_copy(source, destination)
    chosen = selection["result"]["selected_candidate"]
    if sha256_file(destination) != chosen["sha256"]:
        raise RuntimeError("confirmed Task4 checkpoint is not byte-identical to selected candidate")
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
        "task4_complete": True,
        "default_checkpoint_modified": False,
    }
    atomic_json(manifest_path, record)
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
            raise RuntimeError("completed Task4 report protocol drift")
        return existing, sha256_file(path)
    record = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "single_training_variable": protocol["single_training_variable"],
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
            "task4_complete": bool(freeze),
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
        "single_training_variable": protocol["single_training_variable"],
        "replicas": list(REPLICAS),
        "rounds_per_replica": int(protocol["training"]["rounds"]),
        "total_training_rounds": len(REPLICAS) * int(protocol["training"]["rounds"]),
        "snapshots": len(REPLICAS) * len(SNAPSHOT_ROUNDS),
        "inner_evaluation_rounds": inner_rounds,
        "maximum_outer_evaluation_rounds": outer_rounds,
        "maximum_total_evaluation_rounds": inner_rounds + outer_rounds,
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
            "status": existing["status"],
            "report": relative(report_path(protocol)),
            "report_sha256": sha256_file(report_path(protocol)),
            **existing["result"],
        }, indent=2, sort_keys=True))
        return 0
    training = {replica: train_one(protocol_path.resolve(), protocol, protocol_hash, replica) for replica in REPLICAS}
    selection = run_inner_selection(protocol_path.resolve(), protocol, protocol_hash)
    if not selection["result"]["selection_passed"]:
        decision = {
            "decision": "task4_duel_training_not_reproducible_stop",
            "passed": False,
            "reason": "fewer than two replicas produced a retention-safe duel improvement",
        }
        outer_rows = outer_manifests = freeze = None
    else:
        outer_rows, outer_manifests, decision = run_outer(
            protocol_path.resolve(), protocol, protocol_hash, selection,
        )
        freeze = freeze_confirmed(protocol_path.resolve(), protocol, protocol_hash, selection, decision)
    report, report_hash = write_report(
        protocol_path.resolve(), protocol, protocol_hash, training, selection,
        outer_rows, outer_manifests, decision, freeze,
    )
    print(json.dumps({
        "status": report["status"],
        "report": relative(report_path(protocol)),
        "report_sha256": report_hash,
        **report["result"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
