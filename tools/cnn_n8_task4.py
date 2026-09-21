"""Run the unified three-rule Task4 course, rank all course champions, then stop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_cnn_n8.network import ARCHITECTURE  # noqa: E402
from tools.cnn_n8_task1 import (  # noqa: E402
    _completed, _metrics, _torch_load, atomic_json, preflight, relative,
    select_common_milestone, sha256_file, utc_now,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-task4-s151000.json"
KIND_TRAIN = "model-a-cnn-n8-task4-training"
KIND_EVAL = "model-a-cnn-n8-task4-evaluation"
KIND_REPORT = "model-a-cnn-n8-task4-report"
INTEGER_METRICS = (
    "rounds", "score", "coins", "kills", "crates", "bombs", "moves",
    "waits", "invalid_actions", "suicides", "steps",
)


def load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if (protocol.get("kind") != "model-a-cnn-n8-clean-curriculum"
            or protocol.get("authorized_course") != "task4"
            or protocol.get("next_stage_started")):
        raise RuntimeError("wrong Task4 protocol")
    return protocol, hashlib.sha256(raw).hexdigest()


def validate_lineage(protocol: dict) -> None:
    reports = {}
    for course, spec in protocol["course_reports"].items():
        path = ROOT / spec["path"]
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"{course} report drift")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "completed":
            raise RuntimeError(f"{course} report incomplete")
        reports[course] = report
    if reports["task3_coin"]["selection"]["selected_common_milestone"] != 150:
        raise RuntimeError("Task3-C report did not select round 150")
    for replica, parent in protocol["parents"].items():
        if reports["task3_coin"]["selected_checkpoints"][replica] != parent:
            raise RuntimeError(f"Task4 parent mismatch for {replica}")
        path = ROOT / parent["path"]
        if sha256_file(path) != parent["sha256"]:
            raise RuntimeError(f"Task4 parent drift for {replica}")
        payload = _torch_load(path)
        if (payload.get("architecture") != ARCHITECTURE or payload.get("stage") != "task3_coin"
                or payload.get("replica") != replica):
            raise RuntimeError(f"Task4 parent identity mismatch for {replica}")
    for course, champions in protocol["course_champions"].items():
        report = reports[course]
        for replica, spec in champions.items():
            if report["selected_checkpoints"][replica] != spec or sha256_file(ROOT / spec["path"]) != spec["sha256"]:
                raise RuntimeError(f"course champion drift: {course}/{replica}")


def checkpoint_path(protocol: dict, replica: str, milestone: int) -> Path:
    return ROOT / protocol["checkpoint_directory"] / replica / f"round-{milestone:04d}.pt"


def validate_checkpoint(protocol: dict, protocol_hash: str, replica: str, milestone: int) -> dict:
    path = checkpoint_path(protocol, replica, milestone)
    payload = _torch_load(path)
    expected = {
        "architecture": ARCHITECTURE, "protocol_sha256": protocol_hash, "stage": "task4",
        "replica": replica, "stage_rounds": milestone, "completed_total_rounds": 1950 + milestone,
        "n_step": 8, "parent_sha256": protocol["parents"][replica]["sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Task4 checkpoint {key} mismatch for {replica}/round-{milestone}")
    diagnostics = payload["training_diagnostics"]
    if (diagnostics["raw_transitions"] != diagnostics["matured_targets"]
            or len(diagnostics["per_round"]) != milestone
            or sum(diagnostics["action_counts"].values()) != diagnostics["raw_transitions"]):
        raise RuntimeError("Task4 transition accounting mismatch")
    return payload


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, replica: str) -> dict:
    manifest = ROOT / protocol["training_manifest_directory"] / f"{replica}.json"
    existing = _completed(manifest, KIND_TRAIN, protocol_hash)
    if existing is not None:
        for milestone in protocol["task4"]["milestones"]:
            path = checkpoint_path(protocol, replica, int(milestone))
            validate_checkpoint(protocol, protocol_hash, replica, int(milestone))
            if existing["checkpoints"][str(milestone)]["sha256"] != sha256_file(path):
                raise RuntimeError("completed Task4 checkpoint drift")
        return existing
    stats = manifest.with_suffix(".stats.json")
    checkpoint_dir = ROOT / protocol["checkpoint_directory"] / replica
    if stats.exists() or any(checkpoint_dir.glob("round-*.pt")):
        raise RuntimeError(f"orphaned Task4 artifact for {replica}")
    seed = protocol["task4"]["replicas"][replica]
    parent = protocol["parents"][replica]
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_cnn_n8",
        "seeded_rule_based_agent", "seeded_rule_based_agent", "seeded_rule_based_agent",
        "--train", "1", "--continue-without-training", "--no-gui", "--scenario", "classic",
        "--n-rounds", str(protocol["task4"]["rounds_per_replica"]), "--seed", str(seed["world_seed"]),
        "--save-stats", str(stats),
    ]
    overrides = {
        "MODEL_A_CNN_PROTOCOL_PATH": str(protocol_path), "MODEL_A_CNN_MODE": "train",
        "MODEL_A_CNN_STAGE": "task4", "MODEL_A_CNN_REPLICA": replica,
        "MODEL_A_CNN_SEED": str(seed["agent_seed"]), "TASK4_RULE_SEED": str(seed["rule_seed"]),
        "MODEL_A_CNN_CHECKPOINT_PATH": str((ROOT / parent["path"]).resolve()),
        "MODEL_A_CNN_PARENT_SHA256": parent["sha256"], "MODEL_A_CNN_RESUME": "1",
        "MODEL_A_CNN_CHECKPOINT_DIR": str(checkpoint_dir.resolve()), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    record = {
        "schema_version": 1, "kind": KIND_TRAIN, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash, "replica": replica, "status": "running", "started_at_utc": utc_now(),
        "parent": parent, "command": command, "environment_overrides": overrides, "raw_stats": relative(stats),
    }
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides); child.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file():
            raise RuntimeError("Task4 training process failed")
        checkpoints = {}
        for milestone in protocol["task4"]["milestones"]:
            path = checkpoint_path(protocol, replica, int(milestone))
            validate_checkpoint(protocol, protocol_hash, replica, int(milestone))
            checkpoints[str(milestone)] = {"path": relative(path), "sha256": sha256_file(path)}
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"]["model_a_cnn_n8"]
        record.update({"status": "completed", "training_metrics": _metrics(raw), "checkpoints": checkpoints})
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task4 training failed for {replica}: {record.get('error')}")
    return record


def baseline_labels(protocol: dict) -> dict[str, tuple[str, Path, str, str]]:
    return {label: ("model_a_dqn", ROOT / spec["path"], "", "task4")
            for label, spec in protocol["baselines"].items()}


def task4_milestone_labels(protocol: dict) -> dict[str, tuple[str, Path, str, str]]:
    result = baseline_labels(protocol)
    for replica in protocol["task4"]["replicas"]:
        for milestone in protocol["task4"]["milestones"]:
            result[f"task4-{replica}-round{int(milestone):04d}"] = (
                "model_a_cnn_n8", checkpoint_path(protocol, replica, int(milestone)), replica, "task4")
    return result


def retention_labels(protocol: dict, selected: int) -> dict[str, tuple[str, Path, str, str]]:
    result = baseline_labels(protocol)
    for replica, parent in protocol["parents"].items():
        result[f"task3c-parent-{replica}"] = ("model_a_cnn_n8", ROOT / parent["path"], replica, "task3_coin")
        result[f"task4-selected-{replica}"] = (
            "model_a_cnn_n8", checkpoint_path(protocol, replica, selected), replica, "task4")
    return result


def candidate_pool_labels(protocol: dict, selected: int) -> dict[str, tuple[str, Path, str, str]]:
    result = baseline_labels(protocol)
    stages = {"task1": "task1", "task2": "task2", "task3_peaceful": "task3_peaceful",
              "task3_coin": "task3_coin"}
    for course, champions in protocol["course_champions"].items():
        for replica, spec in champions.items():
            result[f"{course}-{replica}-selected"] = (
                "model_a_cnn_n8", ROOT / spec["path"], replica, stages[course])
    for replica in protocol["task4"]["replicas"]:
        result[f"task4-{replica}-round{selected:04d}"] = (
            "model_a_cnn_n8", checkpoint_path(protocol, replica, selected), replica, "task4")
    return result


def environment_overrides(opponents: list[str], case: dict) -> dict[str, str]:
    result = {}
    if "seeded_rule_based_agent" in opponents:
        result["TASK4_RULE_SEED"] = str(case["rule_seed"])
    if "seeded_coin_collector_agent" in opponents or "seeded_peaceful_agent" in opponents:
        key = "coin_seed" if "seeded_coin_collector_agent" in opponents else "peaceful_seed"
        result["TASK3_OPPONENT_SEED"] = str(case[key])
    if "seeded_random_agent" in opponents:
        result["SEEDED_RANDOM_AGENT_SEED"] = str(case["random_seed"])
    return result


def evaluate_one(protocol_path: Path, protocol: dict, protocol_hash: str, suite: str, suite_spec: dict,
                 label: str, identity: tuple[str, Path, str, str], case: dict) -> dict:
    target, checkpoint, replica, stage = identity
    manifest = ROOT / protocol["evaluation_manifest_directory"] / suite / label / f"s{case['world_seed']}.json"
    existing = _completed(manifest, KIND_EVAL, protocol_hash)
    checkpoint_hash = sha256_file(checkpoint)
    if existing is not None:
        if existing["checkpoint"]["sha256"] != checkpoint_hash or existing["case"] != case:
            raise RuntimeError(f"Task4 evaluation drift: {relative(manifest)}")
        return existing
    stats = manifest.with_suffix(".stats.json")
    if stats.exists():
        raise RuntimeError(f"orphaned Task4 evaluation stats: {relative(stats)}")
    opponents = list(suite_spec["opponents"])
    command = [
        sys.executable, "main.py", "play", "--agents", target, *opponents, "--train", "0",
        "--continue-without-training", "--no-gui", "--scenario", suite_spec["scenario"],
        "--n-rounds", str(suite_spec["rounds_per_case"]), "--seed", str(case["world_seed"]),
        "--save-stats", str(stats),
    ]
    if target == "model_a_dqn":
        overrides = {"MODEL_A_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_SEED": str(case["agent_seed"])}
    else:
        overrides = {
            "MODEL_A_CNN_PROTOCOL_PATH": str(protocol_path), "MODEL_A_CNN_MODE": "evaluate",
            "MODEL_A_CNN_STAGE": stage, "MODEL_A_CNN_REPLICA": replica,
            "MODEL_A_CNN_SEED": str(case["agent_seed"]), "MODEL_A_CNN_CHECKPOINT_PATH": str(checkpoint),
        }
    overrides.update(environment_overrides(opponents, case))
    record = {
        "schema_version": 1, "kind": KIND_EVAL, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash, "suite": suite, "label": label, "case": case,
        "status": "running", "started_at_utc": utc_now(),
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
        "command": command, "environment_overrides": overrides, "raw_stats": relative(stats),
    }
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides); child.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file() or sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("Task4 evaluation process failed or mutated checkpoint")
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"][target]
        record.update({"status": "completed", "target_metrics": _metrics(raw)})
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task4 evaluation failed: {suite}/{label}/{case['world_seed']}")
    return record


def combine_metrics(rows: list[dict]) -> dict:
    result = {key: sum(int(row[key]) for row in rows) for key in INTEGER_METRICS}
    rounds, steps = result["rounds"], result["steps"]
    for key in INTEGER_METRICS[1:-1]:
        result[f"{key}_per_round"] = result[key] / rounds if rounds else 0.0
    result["wait_fraction"] = result["waits"] / steps if steps else 0.0
    result["mean_decision_time_ms"] = (
        sum(row["mean_decision_time_ms"] * row["steps"] for row in rows) / steps if steps else 0.0
    )
    return result


def evaluate_suite(protocol_path: Path, protocol: dict, protocol_hash: str, suite: str,
                   identities: dict[str, tuple[str, Path, str, str]]) -> dict[str, dict]:
    spec = protocol["validation"][suite]
    rows = {}
    for label, identity in identities.items():
        items = [evaluate_one(protocol_path, protocol, protocol_hash, suite, spec, label, identity, case)
                 for case in spec["cases"]]
        rows[label] = combine_metrics([item["target_metrics"] for item in items])
    return rows


def select_task4_milestone(rule_rows: dict, mixed_rows: dict, protocol: dict) -> tuple[dict, dict]:
    combined = {label: combine_metrics([rule_rows[label], mixed_rows[label]]) for label in rule_rows}
    adapted = {label.replace("task4-", "cnn-", 1): row for label, row in combined.items()
               if label.startswith("task4-")}
    selection = select_common_milestone(
        adapted, [int(x) for x in protocol["task4"]["milestones"]],
        float(protocol["selection"]["near_best_tolerance_per_round"]),
    )
    return selection, combined


def rank_course_champions(rows: dict) -> dict:
    candidates = {label: metrics for label, metrics in rows.items() if label not in ("frozen-v4", "source-r2")}
    order = {"task1": 0, "task2": 1, "task3_peaceful": 2, "task3_coin": 3, "task4": 4}

    def key(item):
        label, metrics = item
        course = next(name for name in order if label.startswith(name + "-"))
        replica = int(label.split("-r", 1)[1][0])
        return (-metrics["score_per_round"], metrics["suicides_per_round"],
                -metrics["coins_per_round"], metrics["wait_fraction"],
                metrics["mean_decision_time_ms"], order[course], replica)

    ranked = sorted(candidates.items(), key=key)
    return {"selected_label": ranked[0][0], "ranking": [
        {"rank": index, "label": label, "metrics": metrics}
        for index, (label, metrics) in enumerate(ranked, 1)
    ]}


def execute(protocol_path: Path, protocol: dict, protocol_hash: str, hardware: dict) -> dict:
    validate_lineage(protocol)
    for replica in protocol["task4"]["replicas"]:
        train_one(protocol_path, protocol, protocol_hash, replica)
    milestones = task4_milestone_labels(protocol)
    task4_rule = evaluate_suite(protocol_path, protocol, protocol_hash, "task4_rule_selection", milestones)
    task4_mixed = evaluate_suite(protocol_path, protocol, protocol_hash, "task4_mixed_selection", milestones)
    selection, combined_milestones = select_task4_milestone(task4_rule, task4_mixed, protocol)
    selected = int(selection["selected_common_milestone"])
    pool = candidate_pool_labels(protocol, selected)
    pool_rule = evaluate_suite(protocol_path, protocol, protocol_hash, "task4_rule_selection", pool)
    pool_mixed = evaluate_suite(protocol_path, protocol, protocol_hash, "task4_mixed_selection", pool)
    pool_combined = {label: combine_metrics([pool_rule[label], pool_mixed[label]]) for label in pool}
    champion_selection = rank_course_champions(pool_combined)
    retention_identities = retention_labels(protocol, selected)
    retention = {
        "task3_coin": evaluate_suite(protocol_path, protocol, protocol_hash, "task3c_retention", retention_identities),
        "task3_peaceful": evaluate_suite(protocol_path, protocol, protocol_hash, "task3p_retention", retention_identities),
        "task2": evaluate_suite(protocol_path, protocol, protocol_hash, "task2_retention", retention_identities),
        "task1": evaluate_suite(protocol_path, protocol, protocol_hash, "task1_retention", retention_identities),
    }
    report = {
        "schema_version": 1, "kind": KIND_REPORT, "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "course": "task4", "status": "completed", "completed_at_utc": utc_now(),
        "hardware_preflight": hardware, "task4_rule_by_label": task4_rule,
        "task4_mixed_by_label": task4_mixed, "task4_combined_milestones": combined_milestones,
        "selection": selection,
        "selected_checkpoints": {
            replica: {"path": relative(checkpoint_path(protocol, replica, selected)),
                      "sha256": sha256_file(checkpoint_path(protocol, replica, selected))}
            for replica in protocol["task4"]["replicas"]
        },
        "course_champion_task4_rule": pool_rule, "course_champion_task4_mixed": pool_mixed,
        "course_champion_task4_combined": pool_combined, "course_champion_selection": champion_selection,
        "retention_by_task": retention, "next_stage": "selection_free_confirmation",
        "awaiting_user_instruction": True, "next_stage_started": False,
        "default_model_replaced": False, "automatic_followup_started": False,
    }
    output = ROOT / protocol["report_path"]
    if output.exists():
        raise RuntimeError(f"refusing to overwrite Task4 report: {relative(output)}")
    atomic_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol, protocol_hash = load_protocol(protocol_path)
    validate_lineage(protocol)
    hardware = preflight(protocol_path, protocol)
    if args.execute:
        result = execute(protocol_path, protocol, protocol_hash, hardware)
    else:
        result = {
            "mode": "dry-run", "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
            "hardware_preflight": hardware, "parents": protocol["parents"],
            "training_rounds": protocol["task4"]["total_training_rounds"],
            "evaluation_rounds": protocol["validation"]["total_rounds"],
            "candidate_pool_size": 15,
            "will_stop_after": "Task4 selection, all-course ranking, four retention suites, and Task4 report",
            "will_not_do": ["selection-free confirmation", "checkpoint deployment", "Docker submission"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
