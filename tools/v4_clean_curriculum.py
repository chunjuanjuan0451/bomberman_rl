"""Run one explicitly selected course of the preregistered clean-v4 experiment.

Dry-run is the default.  ``--execute`` never advances beyond ``--course``.
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

from agent_code.model_a_v4_curriculum.config import (  # noqa: E402
    ARM_NAMES, STAGE_ORDER, checkpoint_path, load_protocol, parent_stage,
    sha256_file, stage_environment,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-clean-curriculum-s114000.json"
COURSE_STAGES = {
    "task1": ("task1",),
    "task2": ("task2",),
    "task3": ("task3_peaceful", "task3_coin"),
    "task4": ("task4",),
}
PREVIOUS_COURSE = {
    "task1": None,
    "task2": "task1",
    "task3": "task2",
    "task4": "task3",
}


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


def metrics_by_agent(stats: dict) -> dict[str, dict]:
    result = {}
    for name, raw in stats.get("by_agent", {}).items():
        rounds = int(raw.get("rounds", 0))
        steps = int(raw.get("steps", 0))
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
            "bombs": int(raw.get("bombs", 0)),
            "bombs_per_round": int(raw.get("bombs", 0)) / rounds if rounds else 0.0,
            "suicides": int(raw.get("suicides", 0)),
            "suicides_per_round": int(raw.get("suicides", 0)) / rounds if rounds else 0.0,
            "invalid_actions": int(raw.get("invalid", 0)),
            "invalid_actions_per_round": int(raw.get("invalid", 0)) / rounds if rounds else 0.0,
            "steps": steps,
            "mean_decision_time_ms": 1000.0 * float(raw.get("time", 0.0)) / steps if steps else 0.0,
        }
    return result


def manifest_path(protocol: dict, arm: str, replica: str, stage_id: str) -> Path:
    root = ROOT / protocol["training_manifest_directory"]
    return root / f"{arm}-{replica}-{stage_id}.json"


def course_report_path(protocol: dict, course: str) -> Path:
    return ROOT / protocol["course_report_directory"] / f"{course}.json"


def load_completed(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("kind") != kind:
        raise RuntimeError(f"refusing incomplete/incompatible artifact: {path}")
    return payload


def validate_previous_approval(protocol: dict, protocol_hash: str, course: str, supplied_hash: str | None) -> dict | None:
    previous = PREVIOUS_COURSE[course]
    if previous is None:
        if supplied_hash is not None:
            raise RuntimeError("Task1 does not accept a previous-course approval")
        return None
    report_path = course_report_path(protocol, previous)
    report = load_completed(report_path, "model-a-v4-clean-curriculum-course-report")
    if report is None:
        raise RuntimeError(f"previous course report is missing: {previous}")
    if report.get("protocol_sha256") != protocol_hash or report.get("course") != previous:
        raise RuntimeError(f"previous course report contract mismatch: {report_path}")
    actual_hash = sha256_file(report_path)
    if supplied_hash is None:
        raise RuntimeError(
            f"{course} requires explicit user approval: "
            f"--approve-previous-report-sha256 {actual_hash}"
        )
    if supplied_hash != actual_hash:
        raise RuntimeError(f"previous course approval SHA-256 mismatch: expected {actual_hash}")
    return {"course": previous, "path": relative(report_path), "sha256": actual_hash}


def cumulative_rounds(protocol: dict, replica: str, stage_id: str) -> int:
    total = 0
    for current in STAGE_ORDER:
        total += int(protocol["replicas"][replica]["stages"][current]["rounds"])
        if current == stage_id:
            return total
    raise AssertionError(stage_id)


def validate_checkpoint(protocol: dict, protocol_hash: str, arm: str, replica: str, stage_id: str) -> dict:
    from agent_code.model_a_v4_curriculum.callbacks import _torch_load

    path = checkpoint_path(protocol, arm, replica, stage_id)
    if not path.is_file():
        raise RuntimeError(f"checkpoint is missing: {path}")
    payload = _torch_load(path)
    expected = {
        "architecture": "model-a-mlp-v4-global7",
        "protocol_sha256": protocol_hash,
        "arm": arm,
        "replica": replica,
        "stage_id": stage_id,
        "completed_rounds": cumulative_rounds(protocol, replica, stage_id),
        "stage_completed_rounds": int(protocol["replicas"][replica]["stages"][stage_id]["rounds"]),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"checkpoint {key} mismatch for {arm}/{replica}/{stage_id}")
    previous = parent_stage(stage_id)
    expected_parent_hash = (
        None if previous is None else sha256_file(checkpoint_path(protocol, arm, replica, previous))
    )
    if payload.get("parent_sha256") != expected_parent_hash:
        raise RuntimeError(f"checkpoint parent hash mismatch for {arm}/{replica}/{stage_id}")
    return payload


def training_overrides(protocol_path: Path, protocol: dict, arm: str, replica: str, stage_id: str) -> dict[str, str]:
    stage = protocol["replicas"][replica]["stages"][stage_id]
    output = checkpoint_path(protocol, arm, replica, stage_id)
    overrides = {
        "MODEL_A_V4C_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_V4C_CHECKPOINT_PATH": str(output),
        "MODEL_A_V4C_ARM": arm,
        "MODEL_A_V4C_REPLICA": replica,
        "MODEL_A_V4C_STAGE": stage_id,
        "MODEL_A_V4C_SEED": str(stage["agent_seed"]),
    }
    previous = parent_stage(stage_id)
    if previous is not None:
        overrides["MODEL_A_V4C_PARENT_PATH"] = str(checkpoint_path(protocol, arm, replica, previous))
    environment = stage_environment(protocol, arm, stage_id)
    if "seeded_rule_based_agent" in environment["opponents"]:
        overrides["TASK4_RULE_SEED"] = str(stage["opponent_seed"])
    if any(name in environment["opponents"] for name in ("seeded_peaceful_agent", "seeded_coin_collector_agent")):
        overrides["TASK3_OPPONENT_SEED"] = str(stage["opponent_seed"])
    return overrides


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, arm: str, replica: str, stage_id: str) -> dict:
    output = checkpoint_path(protocol, arm, replica, stage_id)
    manifest = manifest_path(protocol, arm, replica, stage_id)
    existing = load_completed(manifest, "model-a-v4-clean-curriculum-training")
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"protocol drift for completed run: {manifest}")
        validate_checkpoint(protocol, protocol_hash, arm, replica, stage_id)
        if existing.get("checkpoint", {}).get("sha256") != sha256_file(output):
            raise RuntimeError(f"checkpoint hash drift for completed run: {manifest}")
        return existing
    stats_path = manifest.with_suffix(".stats.json")
    if output.exists() or stats_path.exists():
        raise RuntimeError(f"refusing orphaned artifact for {arm}/{replica}/{stage_id}")
    previous = parent_stage(stage_id)
    if previous is not None:
        previous_manifest = manifest_path(protocol, arm, replica, previous)
        if load_completed(previous_manifest, "model-a-v4-clean-curriculum-training") is None:
            raise RuntimeError(f"previous stage is not completed: {arm}/{replica}/{previous}")
        validate_checkpoint(protocol, protocol_hash, arm, replica, previous)
    stage = protocol["replicas"][replica]["stages"][stage_id]
    game = stage_environment(protocol, arm, stage_id)
    agents = ["model_a_v4_curriculum", *game["opponents"]]
    command = [
        sys.executable, "main.py", "play", "--agents", *agents,
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", game["scenario"], "--n-rounds", str(stage["rounds"]),
        "--seed", str(stage["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides = training_overrides(protocol_path, protocol, arm, replica, stage_id)
    record = {
        "schema_version": 1,
        "kind": "model-a-v4-clean-curriculum-training",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "arm": arm,
        "replica": replica,
        "stage_id": stage_id,
        "course": "task3" if stage_id.startswith("task3_") else stage_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "training": {**stage, **game, "training_agents": 1},
        "checkpoint": {"path": relative(output), "sha256": None},
        "parent_checkpoint": None if previous is None else {
            "path": relative(checkpoint_path(protocol, arm, replica, previous)),
            "sha256": sha256_file(checkpoint_path(protocol, arm, replica, previous)),
        },
        "command": command,
        "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest, record)
    output.parent.mkdir(parents=True, exist_ok=True)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats_path.is_file():
            raise RuntimeError("training process or stats output failed")
        validate_checkpoint(protocol, protocol_hash, arm, replica, stage_id)
        record["checkpoint"]["sha256"] = sha256_file(output)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        record["metrics_by_agent"] = metrics_by_agent(stats)
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"training failed for {arm}/{replica}/{stage_id}: {record.get('error')}")
    return record


def write_course_report(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    course: str,
    previous_course_approval: dict | None,
    validation: dict,
) -> tuple[dict, str]:
    report_path = course_report_path(protocol, course)
    existing = load_completed(report_path, "model-a-v4-clean-curriculum-course-report")
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or existing.get("course") != course:
            raise RuntimeError(f"course report drift: {report_path}")
        return existing, sha256_file(report_path)
    stages = COURSE_STAGES[course]
    results = {}
    manifest_refs = {}
    checkpoint_refs = {}
    for stage_id in stages:
        results[stage_id] = {}
        manifest_refs[stage_id] = {}
        checkpoint_refs[stage_id] = {}
        for arm in ARM_NAMES:
            results[stage_id][arm] = {}
            manifest_refs[stage_id][arm] = {}
            checkpoint_refs[stage_id][arm] = {}
            for replica in protocol["replicas"]:
                path = manifest_path(protocol, arm, replica, stage_id)
                manifest = load_completed(path, "model-a-v4-clean-curriculum-training")
                if manifest is None:
                    raise RuntimeError(f"training manifest missing before course report: {path}")
                validate_checkpoint(protocol, protocol_hash, arm, replica, stage_id)
                checkpoint = checkpoint_path(protocol, arm, replica, stage_id)
                results[stage_id][arm][replica] = manifest["metrics_by_agent"]
                manifest_refs[stage_id][arm][replica] = {
                    "path": relative(path), "sha256": sha256_file(path),
                }
                checkpoint_refs[stage_id][arm][replica] = {
                    "path": relative(checkpoint), "sha256": sha256_file(checkpoint),
                    "cumulative_rounds": cumulative_rounds(protocol, replica, stage_id),
                }
    next_course = {"task1": "task2", "task2": "task3", "task3": "task4", "task4": None}[course]
    report = {
        "schema_version": 1,
        "kind": "model-a-v4-clean-curriculum-course-report",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "course": course,
        "stages": list(stages),
        "status": "completed",
        "completed_at_utc": utc_now(),
        "results_by_stage_arm_replica": results,
        "training_manifests": manifest_refs,
        "checkpoints": checkpoint_refs,
        "previous_course_approval": previous_course_approval,
        "heldout_validation": validation,
        "next_course": next_course,
        "awaiting_user_instruction": True,
        "next_course_started": False,
        "automatic_followup_started": False,
    }
    atomic_json(report_path, report)
    return report, sha256_file(report_path)


def evaluation_manifest(protocol: dict, suite: str, stratum: str, label: str, world_seed: int) -> Path:
    root = ROOT / protocol["evaluation_manifest_directory"]
    return root / suite / f"{stratum}-{label}-s{world_seed}.json"


def evaluation_identity(protocol: dict, label: str, checkpoint_stage: str) -> tuple[str, Path, str | None, str | None]:
    if label == "v4":
        path = (ROOT / protocol["frozen_v4_baseline"]["checkpoint_path"]).resolve()
        return "model_a_dqn", path, None, None
    arm, replica = label.split("-", 1)
    return "model_a_v4_curriculum", checkpoint_path(protocol, arm, replica, checkpoint_stage), arm, replica


def evaluate_one(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    suite: str,
    checkpoint_stage: str,
    stratum: str,
    spec: dict,
    label: str,
    case: dict,
) -> dict:
    target, checkpoint, arm, replica = evaluation_identity(protocol, label, checkpoint_stage)
    expected_hash = sha256_file(checkpoint)
    manifest = evaluation_manifest(protocol, suite, stratum, label, int(case["world_seed"]))
    existing = load_completed(manifest, "model-a-v4-clean-curriculum-evaluation")
    if existing is not None:
        if (
            existing.get("protocol_sha256") != protocol_hash
            or existing.get("suite") != suite
            or existing.get("checkpoint_stage") != checkpoint_stage
            or existing["checkpoint"]["sha256"] != expected_hash
            or existing.get("evaluation") != {**case, **spec, "target_agent": target}
        ):
            raise RuntimeError(f"evaluation artifact drift: {manifest}")
        return existing
    stats_path = manifest.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned evaluation stats: {stats_path}")
    agents = [target, *spec["opponents"]]
    command = [
        sys.executable, "main.py", "play", "--agents", *agents,
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    if target == "model_a_dqn":
        overrides = {
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_SEED": str(case["agent_seed"]),
        }
    else:
        overrides = {
            "MODEL_A_V4C_PROTOCOL_PATH": str(protocol_path),
            "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V4C_ARM": str(arm),
            "MODEL_A_V4C_REPLICA": str(replica),
            "MODEL_A_V4C_STAGE": checkpoint_stage,
            "MODEL_A_V4C_SEED": str(case["agent_seed"]),
        }
    if "seeded_rule_based_agent" in spec["opponents"]:
        overrides["TASK4_RULE_SEED"] = str(case["opponent_seed"])
    if any(name in spec["opponents"] for name in ("seeded_peaceful_agent", "seeded_coin_collector_agent")):
        overrides["TASK3_OPPONENT_SEED"] = str(case["opponent_seed"])
    record = {
        "schema_version": 1,
        "kind": "model-a-v4-clean-curriculum-evaluation",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "suite": suite,
        "checkpoint_stage": checkpoint_stage,
        "stratum": stratum,
        "label": label,
        "status": "running",
        "started_at_utc": utc_now(),
        "evaluation": {**case, **spec, "target_agent": target},
        "checkpoint": {"path": relative(checkpoint), "sha256": expected_hash},
        "command": command,
        "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest, record)
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
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"evaluation failed for {stratum}/{label}/{case['world_seed']}")
    return record


def aggregate(manifests: list[dict]) -> dict:
    additive = ("rounds", "score", "coins", "kills", "crates", "bombs", "suicides", "invalid_actions", "steps")
    totals = {key: sum(float(item["target_metrics"][key]) for item in manifests) for key in additive}
    rounds = totals["rounds"]
    return {
        **{key: int(value) for key, value in totals.items()},
        "score_per_round": totals["score"] / rounds,
        "coins_per_round": totals["coins"] / rounds,
        "kills_per_round": totals["kills"] / rounds,
        "crates_per_round": totals["crates"] / rounds,
        "bombs_per_round": totals["bombs"] / rounds,
        "suicides_per_round": totals["suicides"] / rounds,
        "invalid_actions_per_round": totals["invalid_actions"] / rounds,
    }


def decide(protocol: dict, rows: dict[str, dict[str, dict]]) -> dict:
    rule = protocol["evaluation"]["decision_rule"]
    supportive = []
    per_replica = {}
    v4_task4 = rows["task4"]["v4"]
    for replica in protocol["replicas"]:
        candidate = rows["task4"][f"curriculum-{replica}"]
        direct = rows["task4"][f"direct-{replica}"]
        delta = candidate["score_per_round"] - direct["score_per_round"]
        passed = delta >= float(rule["curriculum_minus_paired_direct_task4_score_per_round"])
        per_replica[replica] = {"curriculum_minus_direct": delta, "supportive": passed}
        if passed:
            supportive.append(replica)
    candidate_family_task4 = sum(rows["task4"][f"curriculum-{r}"]["score_per_round"] for r in protocol["replicas"]) / 3
    family_delta = candidate_family_task4 - v4_task4["score_per_round"]
    family_gate = (
        len(supportive) >= int(rule["minimum_supportive_replicas"])
        and family_delta >= float(rule["curriculum_family_minus_v4_task4_score_per_round"])
    )
    eligible = []
    for replica in protocol["replicas"]:
        label = f"curriculum-{replica}"
        early_ok = all(
            rows[stratum][label]["score_per_round"]
            >= rows[stratum]["v4"]["score_per_round"] - float(rule["maximum_early_task_score_regression"])
            for stratum in ("task1", "task2", "task3_peaceful", "task3_coin")
        )
        safety_ok = (
            rows["task4"][label]["suicides_per_round"]
            <= v4_task4["suicides_per_round"] + float(rule["maximum_suicide_rate_regression"])
            and rows["task4"][label]["invalid_actions_per_round"]
            <= v4_task4["invalid_actions_per_round"] + float(rule["maximum_invalid_actions_per_round_regression"])
        )
        per_replica[replica].update({"early_regression_gate": early_ok, "task4_safety_gate": safety_ok})
        if family_gate and per_replica[replica]["supportive"] and early_ok and safety_ok:
            eligible.append(replica)
    selected = max(
        eligible,
        key=lambda replica: (rows["task4"][f"curriculum-{replica}"]["score_per_round"], -int(replica[1:])),
        default=None,
    )
    return {
        "supportive_replicas": supportive,
        "candidate_family_task4_score_per_round": candidate_family_task4,
        "candidate_family_minus_v4": family_delta,
        "family_gate": family_gate,
        "per_replica": per_replica,
        "eligible_replicas": eligible,
        "selected_candidate": None if selected is None else f"curriculum-{selected}",
        "decision": "clean_curriculum_confirmed" if selected else "retain_frozen_v4",
    }


def validation_family_summary(protocol: dict, rows: dict[str, dict[str, dict]]) -> dict:
    summaries = {}
    for stratum, labels in rows.items():
        metric_summaries = {}
        for metric in (
            "score_per_round", "coins_per_round", "kills_per_round",
            "suicides_per_round", "invalid_actions_per_round",
        ):
            v4_value = labels["v4"][metric]
            curriculum_mean = sum(
                labels[f"curriculum-{replica}"][metric]
                for replica in protocol["replicas"]
            ) / len(protocol["replicas"])
            direct_mean = sum(
                labels[f"direct-{replica}"][metric]
                for replica in protocol["replicas"]
            ) / len(protocol["replicas"])
            metric_summaries[metric] = {
                "v4": v4_value,
                "curriculum_family": curriculum_mean,
                "direct_family": direct_mean,
                "curriculum_minus_v4": curriculum_mean - v4_value,
                "curriculum_minus_direct": curriculum_mean - direct_mean,
            }
        summaries[stratum] = {
            "metrics": metric_summaries,
            "paired_score_per_round_curriculum_minus_direct": {
                replica: (
                    labels[f"curriculum-{replica}"]["score_per_round"]
                    - labels[f"direct-{replica}"]["score_per_round"]
                )
                for replica in protocol["replicas"]
            },
        }
    return summaries


def run_validation_suite(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    suite: str,
    checkpoint_stage: str,
    strata: dict,
) -> dict:
    labels = ["v4", *[f"{arm}-{replica}" for arm in ARM_NAMES for replica in protocol["replicas"]]]
    rows = {}
    manifest_refs = {}
    for stratum, spec in strata.items():
        rows[stratum] = {}
        manifest_refs[stratum] = {}
        for label in labels:
            runs = [
                evaluate_one(
                    protocol_path, protocol, protocol_hash, suite, checkpoint_stage,
                    stratum, spec, label, case,
                )
                for case in spec["cases"]
            ]
            rows[stratum][label] = aggregate(runs)
            manifest_refs[stratum][label] = [
                relative(evaluation_manifest(protocol, suite, stratum, label, int(case["world_seed"])))
                for case in spec["cases"]
            ]
    return {
        "suite": suite,
        "checkpoint_stage": checkpoint_stage,
        "status": "completed",
        "rows": rows,
        "family_summary": validation_family_summary(protocol, rows),
        "evaluation_manifests": manifest_refs,
        "automatic_progression_decision": False,
    }


def run_course_validation(protocol_path: Path, protocol: dict, protocol_hash: str, course: str) -> dict:
    checkpoint_stage = COURSE_STAGES[course][-1]
    return run_validation_suite(
        protocol_path,
        protocol,
        protocol_hash,
        f"after-{course}",
        checkpoint_stage,
        protocol["course_validation"][course]["strata"],
    )


def run_evaluation(protocol_path: Path, protocol: dict, protocol_hash: str) -> dict:
    for arm in ARM_NAMES:
        for replica in protocol["replicas"]:
            previous_manifest = manifest_path(protocol, arm, replica, "task4")
            if load_completed(previous_manifest, "model-a-v4-clean-curriculum-training") is None:
                raise RuntimeError(f"Task4 is not completed: {arm}/{replica}")
            validate_checkpoint(protocol, protocol_hash, arm, replica, "task4")
    validation = run_validation_suite(
        protocol_path, protocol, protocol_hash, "final-after-task4", "task4",
        protocol["evaluation"]["strata"],
    )
    rows = validation["rows"]
    manifest_refs = validation["evaluation_manifests"]
    result = decide(protocol, rows)
    report_path = ROOT / protocol["report_path"]
    if report_path.exists():
        existing = load_completed(report_path, "model-a-v4-clean-curriculum-report")
        if existing is None or existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"final report drift: {report_path}")
        return existing
    report = {
        "schema_version": 1,
        "kind": "model-a-v4-clean-curriculum-report",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "rows": rows,
        "evaluation_manifests": manifest_refs,
        "result": result,
        "default_checkpoint_modified": False,
        "automatic_followup_started": False,
    }
    atomic_json(report_path, report)
    return report


def dry_run(protocol: dict, course: str) -> dict:
    stages = COURSE_STAGES[course]
    rows = []
    for stage_id in stages:
        for arm in ARM_NAMES:
            game = stage_environment(protocol, arm, stage_id)
            for replica, replica_spec in protocol["replicas"].items():
                stage = replica_spec["stages"][stage_id]
                rows.append({
                    "arm": arm, "replica": replica, "stage": stage_id,
                    "rounds": stage["rounds"], "scenario": game["scenario"],
                    "opponents": game["opponents"], "world_seed": stage["world_seed"],
                    "agent_seed": stage["agent_seed"], "opponent_seed": stage["opponent_seed"],
                    "checkpoint": relative(checkpoint_path(protocol, arm, replica, stage_id)),
                })
    validation_strata = (
        protocol["evaluation"]["strata"] if course == "task4"
        else protocol["course_validation"][course]["strata"]
    )
    return {
        "mode": "dry-run", "course": course, "runs": rows,
        "course_training_rounds": sum(int(row["rounds"]) for row in rows),
        "automatic_heldout_validation": True,
        "heldout_validation_strata": list(validation_strata),
        "heldout_validation_rounds": sum(
            int(spec["rounds_per_case"]) * len(spec["cases"]) * 7
            for spec in validation_strata.values()
        ),
        "automatically_starts_next_course": False,
        "previous_course": PREVIOUS_COURSE[course],
        "requires_previous_report_approval": PREVIOUS_COURSE[course] is not None,
        "writes_course_report_and_stops": True,
        "formal_training_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--course", required=True, choices=list(COURSE_STAGES))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--approve-previous-report-sha256",
        help="Exact SHA-256 of the prior course report, supplied only after user approval.",
    )
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    try:
        protocol, protocol_hash = load_protocol(protocol_path)
        if not args.execute:
            print(json.dumps(dry_run(protocol, args.course), indent=2, sort_keys=True))
            return 0
        approval = validate_previous_approval(
            protocol, protocol_hash, args.course, args.approve_previous_report_sha256,
        )
        completed = []
        for stage_id in COURSE_STAGES[args.course]:
            for arm in ARM_NAMES:
                for replica in protocol["replicas"]:
                    run = train_one(protocol_path.resolve(), protocol, protocol_hash, arm, replica, stage_id)
                    completed.append(relative(manifest_path(protocol, arm, replica, stage_id)))
        if args.course == "task4":
            final_report = run_evaluation(protocol_path.resolve(), protocol, protocol_hash)
            validation = {
                "suite": "final-after-task4",
                "status": "completed",
                "rows": final_report["rows"],
                "result": final_report["result"],
                "evaluation_manifests": final_report["evaluation_manifests"],
                "final_report": {
                    "path": relative(ROOT / protocol["report_path"]),
                    "sha256": sha256_file(ROOT / protocol["report_path"]),
                },
                "automatic_progression_decision": False,
            }
            next_course = None
        else:
            validation = run_course_validation(
                protocol_path.resolve(), protocol, protocol_hash, args.course,
            )
            next_course = {"task1": "task2", "task2": "task3", "task3": "task4"}[args.course]
        report, report_hash = write_course_report(
            protocol_path.resolve(), protocol, protocol_hash, args.course, approval, validation,
        )
        output = {
            "course": args.course, "status": "completed", "manifests": completed,
            "stopped_before": next_course,
            "course_report": relative(course_report_path(protocol, args.course)),
            "course_report_sha256": report_hash,
            "awaiting_user_instruction": report["awaiting_user_instruction"],
            "previous_course_approval": approval,
            "heldout_validation": {"suite": validation["suite"], "status": validation["status"]},
            "next_course_started": False,
        }
        if next_course is not None:
            output["next_command_after_user_approval"] = (
                f"{sys.executable} tools/v4_clean_curriculum.py --course {next_course} --execute "
                f"--approve-previous-report-sha256 {report_hash}"
            )
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
