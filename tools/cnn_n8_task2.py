"""Continue the three selected CNN Task1 parents through Task2, then stop."""

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

from agent_code.model_a_cnn_n8.network import ARCHITECTURE, torch  # noqa: E402
from tools.cnn_n8_task1 import (  # noqa: E402
    _completed, _metrics, _torch_load, aggregate, atomic_json, preflight,
    relative, select_common_milestone, sha256_file, utc_now,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-task2-s148000.json"
KIND_TRAIN = "model-a-cnn-n8-task2-training"
KIND_EVAL = "model-a-cnn-n8-task2-evaluation"
KIND_REPORT = "model-a-cnn-n8-task2-report"


def load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes(); protocol = json.loads(raw)
    if protocol.get("kind") != "model-a-cnn-n8-clean-curriculum" or protocol.get("authorized_course") != "task2":
        raise RuntimeError("wrong Task2 protocol")
    if protocol.get("next_course_started"):
        raise RuntimeError("Task3 is outside this authorization")
    return protocol, hashlib.sha256(raw).hexdigest()


def validate_task1_lineage(protocol: dict) -> None:
    report_path = ROOT / protocol["task1_parent_report"]["path"]
    if sha256_file(report_path) != protocol["task1_parent_report"]["sha256"]:
        raise RuntimeError("Task1 parent report drift")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report["selection"]["selected_common_milestone"] != 400:
        raise RuntimeError("Task1 report does not authorize the selected parents")
    for replica, spec in protocol["parents"].items():
        path = ROOT / spec["path"]
        if sha256_file(path) != spec["sha256"] or report["selected_checkpoints"][replica]["sha256"] != spec["sha256"]:
            raise RuntimeError(f"Task1 parent drift for {replica}")
        payload = _torch_load(path)
        if payload.get("architecture") != ARCHITECTURE or payload.get("stage") != "task1" or payload.get("replica") != replica:
            raise RuntimeError(f"Task1 parent identity mismatch for {replica}")


def checkpoint_path(protocol: dict, replica: str, milestone: int) -> Path:
    return ROOT / protocol["checkpoint_directory"] / replica / f"round-{milestone:04d}.pt"


def validate_checkpoint(protocol: dict, protocol_hash: str, replica: str, milestone: int) -> dict:
    path = checkpoint_path(protocol, replica, milestone); payload = _torch_load(path)
    parent = protocol["parents"][replica]
    expected = {
        "architecture": ARCHITECTURE, "protocol_sha256": protocol_hash, "stage": "task2",
        "replica": replica, "stage_rounds": milestone, "completed_total_rounds": 400 + milestone,
        "n_step": 8, "parent_sha256": parent["sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Task2 checkpoint {key} mismatch for {replica}/round-{milestone}")
    diagnostics = payload["training_diagnostics"]
    if diagnostics["raw_transitions"] != diagnostics["matured_targets"]:
        raise RuntimeError("Task2 n-step accounting mismatch")
    if len(diagnostics["per_round"]) != milestone or sum(diagnostics["action_counts"].values()) != diagnostics["raw_transitions"]:
        raise RuntimeError("Task2 round/action accounting mismatch")
    return payload


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, replica: str) -> dict:
    manifest = ROOT / protocol["training_manifest_directory"] / f"{replica}.json"
    existing = _completed(manifest, KIND_TRAIN, protocol_hash)
    if existing is not None:
        for milestone in protocol["task2"]["milestones"]:
            path = checkpoint_path(protocol, replica, int(milestone))
            validate_checkpoint(protocol, protocol_hash, replica, int(milestone))
            if existing["checkpoints"][str(milestone)]["sha256"] != sha256_file(path):
                raise RuntimeError("completed Task2 checkpoint drift")
        return existing
    stats = manifest.with_suffix(".stats.json")
    checkpoint_dir = ROOT / protocol["checkpoint_directory"] / replica
    if stats.exists() or any(checkpoint_dir.glob("round-*.pt")):
        raise RuntimeError(f"orphaned Task2 artifact for {replica}")
    seed = protocol["task2"]["replicas"][replica]; parent = protocol["parents"][replica]
    command = [sys.executable, "main.py", "play", "--agents", "model_a_cnn_n8", "--train", "1",
               "--continue-without-training", "--no-gui", "--scenario", "classic",
               "--n-rounds", str(protocol["task2"]["rounds_per_replica"]), "--seed", str(seed["world_seed"]),
               "--save-stats", str(stats)]
    overrides = {
        "MODEL_A_CNN_PROTOCOL_PATH": str(protocol_path), "MODEL_A_CNN_MODE": "train", "MODEL_A_CNN_STAGE": "task2",
        "MODEL_A_CNN_REPLICA": replica, "MODEL_A_CNN_SEED": str(seed["agent_seed"]),
        "MODEL_A_CNN_CHECKPOINT_PATH": str((ROOT / parent["path"]).resolve()),
        "MODEL_A_CNN_PARENT_SHA256": parent["sha256"], "MODEL_A_CNN_RESUME": "1",
        "MODEL_A_CNN_CHECKPOINT_DIR": str((ROOT / protocol["checkpoint_directory"] / replica).resolve()),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    record = {"schema_version": 1, "kind": KIND_TRAIN, "protocol_id": protocol["protocol_id"],
              "protocol_sha256": protocol_hash, "replica": replica, "status": "running", "started_at_utc": utc_now(),
              "parent": parent, "command": command, "environment_overrides": overrides, "raw_stats": relative(stats)}
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides); child.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file():
            raise RuntimeError("Task2 training process failed")
        checkpoints = {}
        for milestone in protocol["task2"]["milestones"]:
            path = checkpoint_path(protocol, replica, int(milestone))
            validate_checkpoint(protocol, protocol_hash, replica, int(milestone))
            checkpoints[str(milestone)] = {"path": relative(path), "sha256": sha256_file(path)}
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"]["model_a_cnn_n8"]
        record.update({"status": "completed", "training_metrics": _metrics(raw), "checkpoints": checkpoints})
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task2 training failed for {replica}: {record.get('error')}")
    return record


def primary_labels(protocol: dict) -> dict[str, tuple[str, Path, str, str]]:
    result = {label: ("model_a_dqn", ROOT / spec["path"], "", "task2") for label, spec in protocol["baselines"].items()}
    for replica in protocol["task2"]["replicas"]:
        for milestone in protocol["task2"]["milestones"]:
            label = f"cnn-{replica}-round{int(milestone):04d}"
            result[label] = ("model_a_cnn_n8", checkpoint_path(protocol, replica, int(milestone)), replica, "task2")
    return result


def retention_labels(protocol: dict, selected: int) -> dict[str, tuple[str, Path, str, str]]:
    result = {label: ("model_a_dqn", ROOT / spec["path"], "", "task1") for label, spec in protocol["baselines"].items()}
    for replica, parent in protocol["parents"].items():
        result[f"task1-parent-{replica}"] = ("model_a_cnn_n8", ROOT / parent["path"], replica, "task1")
        result[f"task2-selected-{replica}"] = ("model_a_cnn_n8", checkpoint_path(protocol, replica, selected), replica, "task2")
    return result


def evaluate_one(protocol_path: Path, protocol: dict, protocol_hash: str, suite: str, suite_spec: dict,
                 label: str, identity: tuple[str, Path, str, str], case: dict) -> dict:
    target, checkpoint, replica, stage = identity
    manifest = ROOT / protocol["evaluation_manifest_directory"] / suite / label / f"s{case['world_seed']}.json"
    existing = _completed(manifest, KIND_EVAL, protocol_hash); checkpoint_hash = sha256_file(checkpoint)
    if existing is not None:
        if existing["checkpoint"]["sha256"] != checkpoint_hash or existing["case"] != case:
            raise RuntimeError(f"Task2 evaluation drift: {relative(manifest)}")
        return existing
    stats = manifest.with_suffix(".stats.json")
    if stats.exists():
        raise RuntimeError(f"orphaned Task2 evaluation stats: {relative(stats)}")
    command = [sys.executable, "main.py", "play", "--agents", target, "--train", "0",
               "--continue-without-training", "--no-gui", "--scenario", suite_spec["scenario"],
               "--n-rounds", str(suite_spec["rounds_per_case"]), "--seed", str(case["world_seed"]),
               "--save-stats", str(stats)]
    if target == "model_a_dqn":
        overrides = {"MODEL_A_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_SEED": str(case["agent_seed"])}
    else:
        overrides = {"MODEL_A_CNN_PROTOCOL_PATH": str(protocol_path), "MODEL_A_CNN_MODE": "evaluate",
                     "MODEL_A_CNN_STAGE": stage, "MODEL_A_CNN_REPLICA": replica,
                     "MODEL_A_CNN_SEED": str(case["agent_seed"]), "MODEL_A_CNN_CHECKPOINT_PATH": str(checkpoint)}
    record = {"schema_version": 1, "kind": KIND_EVAL, "protocol_id": protocol["protocol_id"],
              "protocol_sha256": protocol_hash, "suite": suite, "label": label, "case": case, "status": "running",
              "started_at_utc": utc_now(), "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
              "command": command, "environment_overrides": overrides, "raw_stats": relative(stats)}
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides); child.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file() or sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("Task2 evaluation process failed or mutated checkpoint")
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"][target]
        record.update({"status": "completed", "target_metrics": _metrics(raw)})
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task2 evaluation failed: {suite}/{label}/{case['world_seed']}")
    return record


def evaluate_suite(protocol_path: Path, protocol: dict, protocol_hash: str, suite: str,
                   identities: dict[str, tuple[str, Path, str, str]]) -> dict[str, dict]:
    spec = protocol["validation"][suite]
    return {label: aggregate([evaluate_one(protocol_path, protocol, protocol_hash, suite, spec, label, identity, case)
                              for case in spec["cases"]]) for label, identity in identities.items()}


def execute(protocol_path: Path, protocol: dict, protocol_hash: str, hardware: dict) -> dict:
    validate_task1_lineage(protocol)
    for replica in protocol["task2"]["replicas"]:
        train_one(protocol_path, protocol, protocol_hash, replica)
    primary = evaluate_suite(protocol_path, protocol, protocol_hash, "task2_selection", primary_labels(protocol))
    selection = select_common_milestone(primary, [int(x) for x in protocol["task2"]["milestones"]],
                                        float(protocol["selection"]["near_best_tolerance_per_round"]))
    selected = int(selection["selected_common_milestone"])
    retention = evaluate_suite(protocol_path, protocol, protocol_hash, "task1_retention", retention_labels(protocol, selected))
    report = {"schema_version": 1, "kind": KIND_REPORT, "protocol_id": protocol["protocol_id"],
              "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash, "course": "task2",
              "status": "completed", "completed_at_utc": utc_now(), "hardware_preflight": hardware,
              "task2_validation_by_label": primary, "selection": selection,
              "selected_checkpoints": {replica: {"path": relative(checkpoint_path(protocol, replica, selected)),
                                                    "sha256": sha256_file(checkpoint_path(protocol, replica, selected))}
                                       for replica in protocol["task2"]["replicas"]},
              "task1_retention_by_label": retention, "next_course": "task3_peaceful",
              "awaiting_user_instruction": True, "next_course_started": False,
              "default_model_replaced": False, "automatic_followup_started": False}
    output = ROOT / protocol["report_path"]
    if output.exists():
        raise RuntimeError(f"refusing to overwrite Task2 report: {relative(output)}")
    atomic_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv); protocol_path = args.protocol.resolve()
    protocol, protocol_hash = load_protocol(protocol_path)
    validate_task1_lineage(protocol)
    hardware = preflight(protocol_path, protocol)
    if args.execute:
        result = execute(protocol_path, protocol, protocol_hash, hardware)
    else:
        result = {"mode": "dry-run", "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
                  "hardware_preflight": hardware, "parents": protocol["parents"],
                  "training_rounds": protocol["task2"]["total_training_rounds"],
                  "evaluation_rounds": protocol["validation"]["total_rounds"],
                  "will_stop_after": "Task2 selection, Task1 retention evaluation, and Task2 report",
                  "will_not_do": ["Task3", "checkpoint deployment", "final confirmation"]}
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
