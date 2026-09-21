"""Run the preregistered Task1 course for the compact full-board CNN.

Dry-run is the default. ``--execute`` performs Task1 training, validation and
common-milestone selection, then stops.  It never starts Task2 or replaces the
tournament checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_cnn_n8.network import ARCHITECTURE, FullBoardDuelingCNN, torch  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-task1-s147000.json"
KIND_TRAIN = "model-a-cnn-n8-task1-training"
KIND_EVAL = "model-a-cnn-n8-task1-evaluation"
KIND_REPORT = "model-a-cnn-n8-task1-report"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("kind") != "model-a-cnn-n8-clean-curriculum":
        raise RuntimeError("wrong protocol kind")
    if protocol.get("authorized_course") != "task1" or protocol.get("next_course_started"):
        raise RuntimeError("this runner is authorized for Task1 only")
    return protocol, hashlib.sha256(raw).hexdigest()


def _all_seeds(protocol: dict) -> list[int]:
    values: list[int] = []

    def visit(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith("_seed") and isinstance(value, int):
                    values.append(value)
                else:
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(protocol)
    return values


def validate_source_bindings(protocol_path: Path, protocol: dict) -> None:
    bindings = protocol.get("source_bindings", {})
    if not bindings:
        raise RuntimeError("source_bindings are empty; protocol is not frozen")
    for name, expected in bindings.items():
        path = (ROOT / name).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"source binding mismatch: {name}")
    for label, spec in protocol["baselines"].items():
        path = ROOT / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"baseline drift: {label}")
    for other in (ROOT / "experiments/configs").glob("*.json"):
        if other.resolve() == protocol_path.resolve():
            continue
        text = other.read_text(encoding="utf-8")
        for seed in _all_seeds(protocol):
            if str(seed) in text:
                raise RuntimeError(f"seed {seed} is not fresh; found in {relative(other)}")


def preflight(protocol_path: Path, protocol: dict) -> dict:
    if torch is None:
        raise RuntimeError("PyTorch is missing; use the pinned prm_env interpreter from the Luna prompt")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; formal training will not fall back to CPU")
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") is not None:
        raise RuntimeError("remove PYTORCH_MPS_HIGH_WATERMARK_RATIO; memory-limit overrides are forbidden")
    validate_source_bindings(protocol_path, protocol)
    torch.set_num_threads(1)
    model = FullBoardDuelingCNN()
    parameters = sum(item.numel() for item in model.parameters())
    if parameters != int(protocol["architecture"]["parameter_count"]):
        raise RuntimeError(f"parameter-count drift: {parameters}")

    spatial = torch.zeros((1, 11, 33, 33), dtype=torch.float32)
    scalars = torch.zeros((1, 6), dtype=torch.float32)
    samples = []
    with torch.no_grad():
        for _ in range(20):
            output = model(spatial, scalars)
        for _ in range(1000):
            start = time.perf_counter(); output = model(spatial, scalars)
            samples.append(1000.0 * (time.perf_counter() - start))
    ordered = sorted(samples)
    latency = {
        "p50": ordered[499], "p95": ordered[949], "p99": ordered[989], "max": ordered[-1],
    }
    if (tuple(output.shape) != (1, 6) or latency["p50"] >= 20.0 or latency["p95"] >= 50.0
            or latency["p99"] >= 100.0 or latency["max"] >= 250.0):
        raise RuntimeError("CPU inference preflight failed the 0.5-second action budget")

    device = torch.device("mps")
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(protocol["learning"]["learning_rate"]))
    batch = int(protocol["learning"]["batch_size"])
    x = torch.zeros((batch, 11, 33, 33), dtype=torch.float32, device=device)
    s = torch.zeros((batch, 6), dtype=torch.float32, device=device)
    start = time.perf_counter()
    for _ in range(3):
        loss = model(x, s).square().mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    allocated = int(torch.mps.current_allocated_memory())
    del optimizer, model, x, s, loss
    torch.mps.empty_cache()
    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "mps_available": True,
        "parameters": parameters,
        "cpu_forward_latency_ms": latency,
        "mps_batch": batch,
        "mps_three_updates_seconds": elapsed,
        "mps_allocated_bytes_after_smoke": allocated,
    }


def _metrics(raw: dict) -> dict:
    rounds = int(raw.get("rounds", 0)); steps = int(raw.get("steps", 0))
    moves = int(raw.get("moves", 0)); bombs = int(raw.get("bombs", 0)); invalid = int(raw.get("invalid", 0))
    waits = max(0, steps - moves - bombs - invalid)
    result = {
        "rounds": rounds, "score": int(raw.get("score", 0)), "coins": int(raw.get("coins", 0)),
        "kills": int(raw.get("kills", 0)), "crates": int(raw.get("crates", 0)),
        "bombs": bombs, "moves": moves, "waits": waits, "invalid_actions": invalid,
        "suicides": int(raw.get("suicides", 0)), "steps": steps, "decision_time_seconds": float(raw.get("time", 0.0)),
    }
    for key in ("score", "coins", "kills", "crates", "bombs", "moves", "waits", "invalid_actions", "suicides"):
        result[f"{key}_per_round"] = result[key] / rounds if rounds else 0.0
    result["wait_fraction"] = waits / steps if steps else 0.0
    result["mean_decision_time_ms"] = 1000.0 * result["decision_time_seconds"] / steps if steps else 0.0
    return result


def _completed(path: Path, kind: str, protocol_hash: str) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != kind or value.get("status") != "completed" or value.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(f"incompatible existing artifact: {relative(path)}")
    return value


def checkpoint_path(protocol: dict, replica: str, milestone: int) -> Path:
    return ROOT / protocol["checkpoint_directory"] / replica / f"round-{milestone:04d}.pt"


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_checkpoint(path: Path, protocol_hash: str, replica: str, milestone: int) -> dict:
    payload = _torch_load(path)
    expected = {"architecture": ARCHITECTURE, "protocol_sha256": protocol_hash, "stage": "task1",
                "replica": replica, "stage_rounds": milestone, "completed_total_rounds": milestone, "n_step": 8}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"checkpoint {key} mismatch: {relative(path)}")
    diagnostics = payload["training_diagnostics"]
    if diagnostics["raw_transitions"] != diagnostics["matured_targets"]:
        raise RuntimeError("n-step accounting mismatch")
    if len(diagnostics["per_round"]) != milestone:
        raise RuntimeError("per-round diagnostic length mismatch")
    if sum(diagnostics["action_counts"].values()) != diagnostics["raw_transitions"]:
        raise RuntimeError("action accounting mismatch")
    return payload


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, replica: str) -> dict:
    directory = ROOT / protocol["training_manifest_directory"]
    manifest = directory / f"{replica}.json"
    existing = _completed(manifest, KIND_TRAIN, protocol_hash)
    if existing is not None:
        for milestone in protocol["task1"]["milestones"]:
            path = checkpoint_path(protocol, replica, int(milestone))
            validate_checkpoint(path, protocol_hash, replica, int(milestone))
            if existing["checkpoints"][str(milestone)]["sha256"] != sha256_file(path):
                raise RuntimeError("completed checkpoint drift")
        return existing
    stats = directory / f"{replica}.stats.json"
    if stats.exists() or any(checkpoint_path(protocol, replica, int(m)).exists() for m in protocol["task1"]["milestones"]):
        raise RuntimeError(f"orphaned Task1 artifact for {replica}")
    spec = protocol["task1"]["replicas"][replica]
    command = [sys.executable, "main.py", "play", "--agents", "model_a_cnn_n8", "--train", "1",
               "--continue-without-training", "--no-gui", "--scenario", protocol["task1"]["scenario"],
               "--n-rounds", str(protocol["task1"]["rounds_per_replica"]), "--seed", str(spec["world_seed"]),
               "--save-stats", str(stats)]
    overrides = {
        "MODEL_A_CNN_PROTOCOL_PATH": str(protocol_path), "MODEL_A_CNN_MODE": "train",
        "MODEL_A_CNN_STAGE": "task1", "MODEL_A_CNN_REPLICA": replica,
        "MODEL_A_CNN_SEED": str(spec["agent_seed"]),
        "MODEL_A_CNN_CHECKPOINT_DIR": str((ROOT / protocol["checkpoint_directory"] / replica).resolve()),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    record = {"schema_version": 1, "kind": KIND_TRAIN, "protocol_id": protocol["protocol_id"],
              "protocol_sha256": protocol_hash, "replica": replica, "status": "running",
              "started_at_utc": utc_now(), "command": command, "environment_overrides": overrides,
              "raw_stats": relative(stats)}
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides); child.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file():
            raise RuntimeError("training process failed")
        checkpoints = {}
        for milestone in protocol["task1"]["milestones"]:
            path = checkpoint_path(protocol, replica, int(milestone))
            validate_checkpoint(path, protocol_hash, replica, int(milestone))
            checkpoints[str(milestone)] = {"path": relative(path), "sha256": sha256_file(path)}
        raw_stats = json.loads(stats.read_text(encoding="utf-8"))["by_agent"]["model_a_cnn_n8"]
        record.update({"status": "completed", "training_metrics": _metrics(raw_stats), "checkpoints": checkpoints})
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task1 training failed for {replica}: {record.get('error')}")
    return record


def labels(protocol: dict) -> dict[str, tuple[str, Path, str]]:
    result = {}
    for label, spec in protocol["baselines"].items():
        result[label] = ("model_a_dqn", ROOT / spec["path"], "")
    for replica in protocol["task1"]["replicas"]:
        for milestone in protocol["task1"]["milestones"]:
            label = f"cnn-{replica}-round{int(milestone):04d}"
            result[label] = ("model_a_cnn_n8", checkpoint_path(protocol, replica, int(milestone)), replica)
    return result


def evaluate_one(protocol_path: Path, protocol: dict, protocol_hash: str, label: str,
                 target: str, checkpoint: Path, replica: str, case: dict) -> dict:
    root = ROOT / protocol["evaluation_manifest_directory"]
    manifest = root / label / f"s{case['world_seed']}.json"
    existing = _completed(manifest, KIND_EVAL, protocol_hash)
    checkpoint_hash = sha256_file(checkpoint)
    if existing is not None:
        if existing["checkpoint"]["sha256"] != checkpoint_hash or existing["case"] != case:
            raise RuntimeError(f"evaluation drift: {relative(manifest)}")
        return existing
    stats = manifest.with_suffix(".stats.json")
    if stats.exists():
        raise RuntimeError(f"orphaned evaluation stats: {relative(stats)}")
    command = [sys.executable, "main.py", "play", "--agents", target, "--train", "0",
               "--continue-without-training", "--no-gui", "--scenario", protocol["validation"]["scenario"],
               "--n-rounds", str(protocol["validation"]["rounds_per_case"]), "--seed", str(case["world_seed"]),
               "--save-stats", str(stats)]
    if target == "model_a_dqn":
        overrides = {"MODEL_A_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_SEED": str(case["agent_seed"])}
    else:
        overrides = {"MODEL_A_CNN_PROTOCOL_PATH": str(protocol_path), "MODEL_A_CNN_MODE": "evaluate",
                     "MODEL_A_CNN_STAGE": "task1", "MODEL_A_CNN_REPLICA": replica,
                     "MODEL_A_CNN_SEED": str(case["agent_seed"]), "MODEL_A_CNN_CHECKPOINT_PATH": str(checkpoint)}
    record = {"schema_version": 1, "kind": KIND_EVAL, "protocol_id": protocol["protocol_id"],
              "protocol_sha256": protocol_hash, "label": label, "case": case, "status": "running",
              "started_at_utc": utc_now(), "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
              "command": command, "environment_overrides": overrides, "raw_stats": relative(stats)}
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides); child.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file() or sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("evaluation process failed or mutated checkpoint")
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"][target]
        record.update({"status": "completed", "target_metrics": _metrics(raw)})
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"evaluation failed: {label}/{case['world_seed']}")
    return record


def aggregate(items: list[dict]) -> dict:
    integer = ("rounds", "score", "coins", "kills", "crates", "bombs", "moves", "waits", "invalid_actions", "suicides", "steps")
    totals = {key: sum(int(item["target_metrics"][key]) for item in items) for key in integer}
    rounds, steps = totals["rounds"], totals["steps"]
    for key in integer[1:-1]:
        totals[f"{key}_per_round"] = totals[key] / rounds if rounds else 0.0
    totals["wait_fraction"] = totals["waits"] / steps if steps else 0.0
    totals["mean_decision_time_ms"] = (
        sum(float(item["target_metrics"]["decision_time_seconds"]) for item in items) * 1000.0 / steps if steps else 0.0
    )
    return totals


def select_common_milestone(rows: dict[str, dict], milestones: list[int], tolerance: float) -> dict:
    pooled = {}
    for milestone in milestones:
        members = [rows[f"cnn-{replica}-round{int(milestone):04d}"] for replica in ("r1", "r2", "r3")]
        rounds = sum(item["rounds"] for item in members); steps = sum(item["steps"] for item in members)
        pooled[str(milestone)] = {
            "rounds": rounds, "score": sum(item["score"] for item in members),
            "score_per_round": sum(item["score"] for item in members) / rounds,
            "waits": sum(item["waits"] for item in members),
            "wait_fraction": sum(item["waits"] for item in members) / steps if steps else 0.0,
        }
    best = max(item["score_per_round"] for item in pooled.values())
    eligible = [int(key) for key, item in pooled.items() if item["score_per_round"] >= best - tolerance]
    selected = min(eligible, key=lambda m: (m, pooled[str(m)]["wait_fraction"]))
    return {"pooled_by_milestone": pooled, "best_score_per_round": best,
            "near_best_tolerance_per_round": tolerance, "eligible_milestones": sorted(eligible),
            "selected_common_milestone": selected}


def execute(protocol_path: Path, protocol: dict, protocol_hash: str, preflight_result: dict) -> dict:
    for replica in protocol["task1"]["replicas"]:
        train_one(protocol_path, protocol, protocol_hash, replica)
    manifests = {}
    rows = {}
    for label, (target, checkpoint, replica) in labels(protocol).items():
        items = [evaluate_one(protocol_path, protocol, protocol_hash, label, target, checkpoint, replica, case)
                 for case in protocol["validation"]["cases"]]
        rows[label] = aggregate(items)
        manifests[label] = [{"path": relative(ROOT / protocol["evaluation_manifest_directory"] / label / f"s{case['world_seed']}.json")}
                             for case in protocol["validation"]["cases"]]
    selection = select_common_milestone(rows, [int(x) for x in protocol["task1"]["milestones"]],
                                        float(protocol["selection"]["near_best_tolerance_per_round"]))
    selected = selection["selected_common_milestone"]
    report = {"schema_version": 1, "kind": KIND_REPORT, "protocol_id": protocol["protocol_id"],
              "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash, "course": "task1",
              "status": "completed", "completed_at_utc": utc_now(), "hardware_preflight": preflight_result,
              "validation_by_label": rows, "selection": selection,
              "selected_checkpoints": {replica: {"path": relative(checkpoint_path(protocol, replica, selected)),
                                                    "sha256": sha256_file(checkpoint_path(protocol, replica, selected))}
                                       for replica in protocol["task1"]["replicas"]},
              "next_course": "task2", "awaiting_user_instruction": True, "next_course_started": False,
              "default_model_replaced": False, "automatic_followup_started": False}
    report_path = ROOT / protocol["report_path"]
    if report_path.exists():
        raise RuntimeError(f"refusing to overwrite report: {relative(report_path)}")
    atomic_json(report_path, report)
    return report


def dry_run(protocol: dict, protocol_hash: str, preflight_result: dict) -> dict:
    return {"mode": "dry-run", "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
            "hardware_preflight": preflight_result, "course": "task1",
            "training_rounds": protocol["task1"]["total_training_rounds"],
            "evaluation_rounds": protocol["validation"]["total_rounds"],
            "replicas": list(protocol["task1"]["replicas"]), "milestones": protocol["task1"]["milestones"],
            "will_stop_after": "Task1 validation, one common milestone selection, and course report",
            "will_not_do": ["Task2", "checkpoint deployment", "final confirmation"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol, protocol_hash = load_protocol(protocol_path)
    preflight_result = preflight(protocol_path, protocol)
    result = execute(protocol_path, protocol, protocol_hash, preflight_result) if args.execute else dry_run(protocol, protocol_hash, preflight_result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
