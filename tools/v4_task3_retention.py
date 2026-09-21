"""Run retention-aware exact-v4 Task3 from the selected Task2 r2 parent.

Dry-run is the default. ``--execute`` trains three branches through separate
peaceful and coin stages, performs preregistered inner checkpoint selection,
runs outer evaluation, writes one report, and never starts Task4.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v4_task3_retention.config import (  # noqa: E402
    BRANCHES, STAGES, final_checkpoint_path, load_protocol,
    selected_checkpoint_path, sha256_file, snapshot_path,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task3-retention-s119000.json"


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


def load_completed(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("kind") != kind:
        raise RuntimeError(f"refusing incomplete/incompatible artifact: {path}")
    return payload


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


def aggregate(manifests: list[dict]) -> dict:
    additive = (
        "rounds", "score", "coins", "kills", "crates", "bombs", "moves",
        "waits", "suicides", "invalid_actions", "steps",
    )
    totals = {key: sum(int(item["target_metrics"][key]) for item in manifests) for key in additive}
    rounds = totals["rounds"]
    steps = totals["steps"]
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


def training_manifest_path(protocol: dict, branch: str, stage: str) -> Path:
    return ROOT / protocol["training_manifest_directory"] / f"{branch}-{stage}.json"


def selection_manifest_path(protocol: dict, branch: str, stage: str) -> Path:
    return ROOT / protocol["selection_manifest_directory"] / f"{branch}-{stage}.json"


def evaluation_manifest_path(
    protocol: dict, suite: str, stage: str, stratum: str, label: str, world_seed: int,
) -> Path:
    return (
        ROOT / protocol["evaluation_manifest_directory"] / suite / stage / stratum
        / f"{label}-s{world_seed}.json"
    )


def source_parent_path(protocol: dict) -> Path:
    return (ROOT / protocol["source_parent"]["checkpoint"]["path"]).resolve()


def parent_path(protocol: dict, branch: str, stage: str) -> Path:
    if stage == "task3_peaceful":
        return source_parent_path(protocol)
    return selected_checkpoint_path(protocol, branch, "task3_peaceful")


def validate_new_checkpoint(
    protocol: dict,
    protocol_hash: str,
    path: Path,
    branch: str,
    stage: str,
    stage_round: int | None = None,
    selected: bool | None = None,
) -> dict:
    from agent_code.model_a_v4_task3_retention.callbacks import _torch_load

    if not path.is_file():
        raise RuntimeError(f"Task3 checkpoint missing: {path}")
    payload = _torch_load(path)
    expected = {
        "architecture": "model-a-mlp-v4-global7",
        "protocol_sha256": protocol_hash,
        "branch": branch,
        "stage_id": stage,
        "parent_sha256": sha256_file(parent_path(protocol, branch, stage)),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Task3 checkpoint {key} mismatch: {path}")
    if stage_round is not None and int(payload.get("stage_completed_rounds", -1)) != stage_round:
        raise RuntimeError(f"Task3 snapshot round mismatch: {path}")
    if selected is not None and bool(payload.get("selected_by_inner_validation")) is not selected:
        raise RuntimeError(f"Task3 checkpoint selection flag mismatch: {path}")
    if abs(float(payload.get("epsilon", -1)) - 0.05) > 1e-12:
        raise RuntimeError(f"Task3 checkpoint epsilon mismatch: {path}")
    return payload


def validate_training_artifacts(
    protocol: dict, protocol_hash: str, branch: str, stage: str,
) -> tuple[dict, list[dict]]:
    final = final_checkpoint_path(protocol, branch, stage)
    payload = validate_new_checkpoint(protocol, protocol_hash, final, branch, stage, 600, False)
    snapshots = []
    for stage_round in range(50, 601, 50):
        path = snapshot_path(protocol, branch, stage, stage_round)
        item = validate_new_checkpoint(
            protocol, protocol_hash, path, branch, stage, stage_round, False,
        )
        snapshots.append({
            "stage_round": stage_round,
            "path": relative(path),
            "sha256": sha256_file(path),
            "completed_rounds": int(item["completed_rounds"]),
            "training_steps": int(item["training_steps"]),
            "gradient_steps": int(item["gradient_steps"]),
            "epsilon": float(item["epsilon"]),
        })
    return payload, snapshots


def train_one(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    branch: str,
    stage: str,
) -> dict:
    manifest_path = training_manifest_path(protocol, branch, stage)
    output = final_checkpoint_path(protocol, branch, stage)
    existing = load_completed(manifest_path, "model-a-v4-task3-retention-training")
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed Task3 training protocol drift: {manifest_path}")
        validate_training_artifacts(protocol, protocol_hash, branch, stage)
        if existing["checkpoint"]["sha256"] != sha256_file(output):
            raise RuntimeError(f"completed Task3 checkpoint drift: {branch}/{stage}")
        return existing
    stats_path = manifest_path.with_suffix(".stats.json")
    stage_root = output.parent
    if output.exists() or stats_path.exists() or stage_root.exists():
        raise RuntimeError(f"refusing orphaned Task3 training artifact: {branch}/{stage}")
    parent = parent_path(protocol, branch, stage)
    if not parent.is_file():
        raise RuntimeError(f"Task3 parent missing: {branch}/{stage}")
    spec = protocol["training"][stage]
    seeds = spec["branches"][branch]
    agents = ["model_a_v4_task3_retention", *spec["opponents"]]
    command = [
        sys.executable, "main.py", "play", "--agents", *agents,
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds"]),
        "--seed", str(seeds["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides = {
        "MODEL_A_V4T3R_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_V4T3R_CHECKPOINT_PATH": str(output),
        "MODEL_A_V4T3R_PARENT_PATH": str(parent),
        "MODEL_A_V4T3R_BRANCH": branch,
        "MODEL_A_V4T3R_STAGE": stage,
        "MODEL_A_V4T3R_SEED": str(seeds["agent_seed"]),
        "TASK3_OPPONENT_SEED": str(seeds["opponent_seed"]),
    }
    record = {
        "schema_version": 1,
        "kind": "model-a-v4-task3-retention-training",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "branch": branch,
        "stage": stage,
        "status": "running",
        "started_at_utc": utc_now(),
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
            raise RuntimeError("Task3 training process or stats output failed")
        payload, snapshots = validate_training_artifacts(protocol, protocol_hash, branch, stage)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        record["metrics_by_agent"] = metrics_by_agent(stats)
        record["snapshots"] = snapshots
        record["final_training_state"] = {
            "completed_rounds": int(payload["completed_rounds"]),
            "training_steps": int(payload["training_steps"]),
            "gradient_steps": int(payload["gradient_steps"]),
            "epsilon": float(payload["epsilon"]),
        }
        record["checkpoint"]["sha256"] = sha256_file(output)
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest_path, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task3 training failed: {branch}/{stage}: {record.get('error')}")
    return record


def checkpoint_environment(
    protocol_path: Path,
    protocol: dict,
    identity: dict,
    case: dict,
) -> tuple[str, Path, dict[str, str]]:
    kind = identity["kind"]
    if kind == "v4":
        checkpoint = ROOT / protocol["frozen_v4_baseline"]["checkpoint_path"]
        return "model_a_dqn", checkpoint, {
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_SEED": str(case["agent_seed"]),
        }
    if kind == "source":
        checkpoint = source_parent_path(protocol)
        return "model_a_v4_curriculum", checkpoint, {
            "MODEL_A_V4C_PROTOCOL_PATH": str(ROOT / protocol["source_parent"]["clean_protocol"]["path"]),
            "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V4C_ARM": "curriculum",
            "MODEL_A_V4C_REPLICA": "r2",
            "MODEL_A_V4C_STAGE": "task2",
            "MODEL_A_V4C_SEED": str(case["agent_seed"]),
        }
    branch = identity["branch"]
    stage = identity["stage"]
    checkpoint = Path(identity["path"]).resolve()
    return "model_a_v4_task3_retention", checkpoint, {
        "MODEL_A_V4T3R_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_V4T3R_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_V4T3R_BRANCH": branch,
        "MODEL_A_V4T3R_STAGE": stage,
        "MODEL_A_V4T3R_SEED": str(case["agent_seed"]),
    }


def evaluate_one(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    suite: str,
    stage: str,
    stratum: str,
    spec: dict,
    label: str,
    identity: dict,
    case: dict,
) -> dict:
    target, checkpoint, overrides = checkpoint_environment(protocol_path, protocol, identity, case)
    checkpoint_hash = sha256_file(checkpoint)
    output = evaluation_manifest_path(
        protocol, suite, stage, stratum, label, int(case["world_seed"]),
    )
    expected_evaluation = {**case, **spec, "target_agent": target}
    existing = load_completed(output, "model-a-v4-task3-retention-evaluation")
    if existing is not None:
        if (
            existing.get("protocol_sha256") != protocol_hash
            or existing.get("evaluation") != expected_evaluation
            or existing.get("checkpoint", {}).get("sha256") != checkpoint_hash
        ):
            raise RuntimeError(f"completed Task3 evaluation drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned Task3 evaluation stats: {stats_path}")
    agents = [target, *spec["opponents"]]
    command = [
        sys.executable, "main.py", "play", "--agents", *agents,
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    if spec["opponents"]:
        overrides["TASK3_OPPONENT_SEED"] = str(case["opponent_seed"])
    record = {
        "schema_version": 1,
        "kind": "model-a-v4-task3-retention-evaluation",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "suite": suite,
        "selection_stage": stage,
        "stratum": stratum,
        "label": label,
        "status": "running",
        "started_at_utc": utc_now(),
        "evaluation": expected_evaluation,
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
        "command": command,
        "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(output, record)
    child = os.environ.copy()
    child.update(overrides)
    before = sha256_file(checkpoint)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    if completed.returncode == 0 and stats_path.is_file() and sha256_file(checkpoint) == before:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        record["metrics_by_agent"] = metrics_by_agent(stats)
        record["target_metrics"] = record["metrics_by_agent"].get(target)
        record["status"] = "completed" if record["target_metrics"] else "failed"
    else:
        record["status"] = "failed"
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task3 evaluation failed: {suite}/{stage}/{stratum}/{label}")
    return record


def evaluate_identity_on_strata(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    suite: str,
    stage: str,
    label: str,
    identity: dict,
    strata_names: tuple[str, ...],
    strata: dict,
) -> tuple[dict, dict]:
    rows = {}
    manifests = {}
    for stratum in strata_names:
        spec = strata[stratum]
        runs = [
            evaluate_one(
                protocol_path, protocol, protocol_hash, suite, stage, stratum,
                spec, label, identity, case,
            )
            for case in spec["cases"]
        ]
        rows[stratum] = aggregate(runs)
        manifests[stratum] = [
            relative(evaluation_manifest_path(
                protocol, suite, stage, stratum, label, int(case["world_seed"]),
            ))
            for case in spec["cases"]
        ]
    return rows, manifests


def selection_score(
    current_metrics: dict[str, float],
    best_current_metrics: dict[str, float],
    prior_scores: dict[str, float],
    parent_scores: dict[str, float],
    current_fraction: float,
) -> dict:
    def threshold(best: float) -> float:
        # Preserve the natural "within 90% of best" interpretation even if a
        # future metric is negative. Current Bomberman score/kills are normally
        # non-negative, but fail predictably instead of excluding the best row.
        return current_fraction * best if best >= 0.0 else best / current_fraction

    metric_gates = {
        name: value >= threshold(best_current_metrics[name]) - 1e-12
        for name, value in current_metrics.items()
    }
    eligible = all(metric_gates.values())
    normalized = {
        name: (prior_scores[name] - parent_scores[name]) / max(abs(parent_scores[name]), 1.0)
        for name in prior_scores
    }
    return {
        "current_task_metrics": current_metrics,
        "best_current_task_metrics": best_current_metrics,
        "current_task_metric_gates": metric_gates,
        "current_task_eligible": eligible,
        "normalized_retention_deltas": normalized,
        "worst_normalized_retention_delta": min(normalized.values()),
        "mean_normalized_retention_delta": sum(normalized.values()) / len(normalized),
    }


def select_checkpoint(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    branch: str,
    stage: str,
) -> dict:
    manifest_path = selection_manifest_path(protocol, branch, stage)
    selected_path = selected_checkpoint_path(protocol, branch, stage)
    existing = load_completed(manifest_path, "model-a-v4-task3-retention-selection")
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed selection protocol drift: {manifest_path}")
        validate_new_checkpoint(protocol, protocol_hash, selected_path, branch, stage, selected=True)
        if existing["selected_checkpoint"]["sha256"] != sha256_file(selected_path):
            raise RuntimeError(f"selected checkpoint drift: {branch}/{stage}")
        return existing
    if selected_path.exists():
        raise RuntimeError(f"refusing orphaned selected checkpoint: {selected_path}")
    training = load_completed(
        training_manifest_path(protocol, branch, stage),
        "model-a-v4-task3-retention-training",
    )
    if training is None:
        raise RuntimeError(f"training missing before selection: {branch}/{stage}")
    config = protocol["inner_selection"]
    stage_config = config["stages"][stage]
    strata_names = tuple(stage_config["strata"])
    current_stratum = stage_config["current_stratum"]
    prior_strata = tuple(name for name in strata_names if name != current_stratum)
    strata = config["strata"]
    if stage == "task3_peaceful":
        parent_identity = {"kind": "source"}
    else:
        parent_identity = {
            "kind": "candidate", "branch": branch, "stage": "task3_peaceful",
            "path": str(selected_checkpoint_path(protocol, branch, "task3_peaceful")),
        }
    parent_rows, parent_manifests = evaluate_identity_on_strata(
        protocol_path, protocol, protocol_hash, "inner", stage,
        f"parent-{branch}", parent_identity, prior_strata, strata,
    )
    snapshot_rows = {}
    snapshot_manifests = {}
    for stage_round in range(50, 601, 50):
        label = f"{branch}-round-{stage_round:04d}"
        identity = {
            "kind": "candidate", "branch": branch, "stage": stage,
            "path": str(snapshot_path(protocol, branch, stage, stage_round)),
        }
        rows, manifests = evaluate_identity_on_strata(
            protocol_path, protocol, protocol_hash, "inner", stage,
            label, identity, strata_names, strata,
        )
        snapshot_rows[str(stage_round)] = rows
        snapshot_manifests[str(stage_round)] = manifests
    current_metrics = tuple(config["current_task_metrics"])
    best_current = {
        metric: max(rows[current_stratum][metric] for rows in snapshot_rows.values())
        for metric in current_metrics
    }
    scored = {}
    for stage_round, rows in snapshot_rows.items():
        prior_scores = {
            f"{stratum}:{metric}": rows[stratum][metric]
            for stratum, metrics in stage_config["prior_retention_metrics"].items()
            for metric in metrics
        }
        parent_scores = {
            f"{stratum}:{metric}": parent_rows[stratum][metric]
            for stratum, metrics in stage_config["prior_retention_metrics"].items()
            for metric in metrics
        }
        scored[stage_round] = selection_score(
            {metric: rows[current_stratum][metric] for metric in current_metrics},
            best_current,
            prior_scores,
            parent_scores,
            float(config["current_task_best_fraction"]),
        )
    eligible = [stage_round for stage_round, score in scored.items() if score["current_task_eligible"]]
    if not eligible:
        raise RuntimeError(f"no eligible Task3 checkpoint: {branch}/{stage}")
    chosen_round = max(
        eligible,
        key=lambda value: (
            scored[value]["worst_normalized_retention_delta"],
            scored[value]["mean_normalized_retention_delta"],
            scored[value]["current_task_metrics"]["kills_per_round"],
            scored[value]["current_task_metrics"]["score_per_round"],
            -int(value),
        ),
    )
    chosen_snapshot = snapshot_path(protocol, branch, stage, int(chosen_round))
    from agent_code.model_a_v4_task3_retention.callbacks import _torch_load
    from agent_code.model_a_dqn.network import torch

    payload = _torch_load(chosen_snapshot)
    payload["selected_by_inner_validation"] = True
    payload["selection"] = {
        "selected_stage_round": int(chosen_round),
        "source_snapshot_sha256": sha256_file(chosen_snapshot),
        "rule": protocol["inner_selection"]["tie_break_order"],
    }
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected_path.with_suffix(selected_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(selected_path)
    validate_new_checkpoint(protocol, protocol_hash, selected_path, branch, stage, int(chosen_round), True)
    record = {
        "schema_version": 1,
        "kind": "model-a-v4-task3-retention-selection",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "branch": branch,
        "stage": stage,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "selection_rule": {
            "current_task_best_fraction": float(config["current_task_best_fraction"]),
            "current_task_metrics": list(current_metrics),
            "prior_retention_metrics": stage_config["prior_retention_metrics"],
            "tie_break_order": config["tie_break_order"],
        },
        "parent_rows": parent_rows,
        "parent_evaluation_manifests": parent_manifests,
        "snapshot_rows": snapshot_rows,
        "snapshot_evaluation_manifests": snapshot_manifests,
        "snapshot_scores": scored,
        "eligible_stage_rounds": [int(value) for value in eligible],
        "selected_stage_round": int(chosen_round),
        "selected_checkpoint": {
            "path": relative(selected_path), "sha256": sha256_file(selected_path),
            "source_snapshot": relative(chosen_snapshot),
            "source_snapshot_sha256": sha256_file(chosen_snapshot),
        },
    }
    atomic_json(manifest_path, record)
    return record


def run_outer(
    protocol_path: Path, protocol: dict, protocol_hash: str,
) -> tuple[dict, dict]:
    identities = {
        "v4": {"kind": "v4"},
        "source-r2-task2": {"kind": "source"},
        **{
            f"selected-{branch}": {
                "kind": "candidate", "branch": branch, "stage": "task3_coin",
                "path": str(selected_checkpoint_path(protocol, branch, "task3_coin")),
            }
            for branch in BRANCHES
        },
    }
    strata = protocol["outer_evaluation"]["strata"]
    rows = {}
    manifests = {}
    for label, identity in identities.items():
        rows[label], manifests[label] = evaluate_identity_on_strata(
            protocol_path, protocol, protocol_hash, "outer", "task3_coin",
            label, identity, tuple(strata), strata,
        )
    return rows, manifests


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def write_report(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    training: dict,
    selections: dict,
    outer_rows: dict,
    outer_manifests: dict,
) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, "model-a-v4-task3-retention-report")
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed Task3 retention report protocol drift")
        return existing, sha256_file(path)
    report = {
        "schema_version": 1,
        "kind": "model-a-v4-task3-retention-report",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "training_manifests": {
            stage: {
                branch: {
                    "path": relative(training_manifest_path(protocol, branch, stage)),
                    "sha256": sha256_file(training_manifest_path(protocol, branch, stage)),
                }
                for branch in BRANCHES
            }
            for stage in STAGES
        },
        "selection_manifests": {
            stage: {
                branch: {
                    "path": relative(selection_manifest_path(protocol, branch, stage)),
                    "sha256": sha256_file(selection_manifest_path(protocol, branch, stage)),
                    "selected_stage_round": selections[stage][branch]["selected_stage_round"],
                    "selected_checkpoint": selections[stage][branch]["selected_checkpoint"],
                }
                for branch in BRANCHES
            }
            for stage in STAGES
        },
        "training_summary": {
            stage: {
                branch: training[stage][branch]["metrics_by_agent"]
                for branch in BRANCHES
            }
            for stage in STAGES
        },
        "outer_rows": outer_rows,
        "outer_evaluation_manifests": outer_manifests,
        "result": {
            "outer_is_selection_free": True,
            "automatic_branch_selected": False,
            "awaiting_user_instruction": True,
            "task4_started": False,
            "automatic_followup_started": False,
        },
        "default_checkpoint_modified": False,
    }
    atomic_json(path, report)
    return report, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    inner_rounds = 0
    for stage in STAGES:
        stage_config = protocol["inner_selection"]["stages"][stage]
        strata = protocol["inner_selection"]["strata"]
        current = stage_config["current_stratum"]
        snapshot_rounds = sum(
            strata[name]["rounds_per_case"] * len(strata[name]["cases"])
            for name in stage_config["strata"]
        ) * 12
        parent_rounds = sum(
            strata[name]["rounds_per_case"] * len(strata[name]["cases"])
            for name in stage_config["strata"] if name != current
        )
        inner_rounds += (snapshot_rounds + parent_rounds) * len(BRANCHES)
    outer = protocol["outer_evaluation"]["strata"]
    outer_rounds = sum(
        spec["rounds_per_case"] * len(spec["cases"]) for spec in outer.values()
    ) * 5
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "source_parent": protocol["source_parent"]["checkpoint"],
        "branches": list(BRANCHES),
        "stages": [
            {
                "stage": stage,
                "rounds_per_branch": protocol["training"][stage]["rounds"],
                "scenario": protocol["training"][stage]["scenario"],
                "opponents": protocol["training"][stage]["opponents"],
                "snapshots_per_branch": 12,
                "inner_strata": protocol["inner_selection"]["stages"][stage]["strata"],
                "selection_then_next_stage": stage == "task3_peaceful",
            }
            for stage in STAGES
        ],
        "training_rounds": sum(protocol["training"][stage]["rounds"] for stage in STAGES) * len(BRANCHES),
        "inner_evaluation_rounds": inner_rounds,
        "outer_evaluation_rounds": outer_rounds,
        "total_evaluation_rounds": inner_rounds + outer_rounds,
        "task1_or_task2_training_mixed_in": False,
        "formal_training_started": False,
        "formal_evaluation_started": False,
        "task4_started": False,
        "writes_terminal_report_and_stops": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    try:
        protocol, protocol_hash = load_protocol(protocol_path)
        if not args.execute:
            print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
            return 0
        existing = load_completed(report_path(protocol), "model-a-v4-task3-retention-report")
        if existing is not None:
            if existing.get("protocol_sha256") != protocol_hash:
                raise RuntimeError("completed Task3 report protocol drift")
            print(json.dumps({
                "status": "completed",
                "report": relative(report_path(protocol)),
                "report_sha256": sha256_file(report_path(protocol)),
                "awaiting_user_instruction": True,
                "task4_started": False,
            }, indent=2, sort_keys=True))
            return 0
        training = {stage: {} for stage in STAGES}
        selections = {stage: {} for stage in STAGES}
        for stage in STAGES:
            for branch in BRANCHES:
                training[stage][branch] = train_one(
                    protocol_path.resolve(), protocol, protocol_hash, branch, stage,
                )
            for branch in BRANCHES:
                selections[stage][branch] = select_checkpoint(
                    protocol_path.resolve(), protocol, protocol_hash, branch, stage,
                )
        outer_rows, outer_manifests = run_outer(protocol_path.resolve(), protocol, protocol_hash)
        report, report_hash = write_report(
            protocol_path.resolve(), protocol, protocol_hash,
            training, selections, outer_rows, outer_manifests,
        )
        print(json.dumps({
            "status": "completed",
            "report": relative(report_path(protocol)),
            "report_sha256": report_hash,
            "selected_stage_rounds": {
                stage: {
                    branch: selections[stage][branch]["selected_stage_round"]
                    for branch in BRANCHES
                }
                for stage in STAGES
            },
            "awaiting_user_instruction": True,
            "automatic_branch_selected": False,
            "task4_started": False,
        }, indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
