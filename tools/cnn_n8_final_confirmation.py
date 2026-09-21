"""Selection-free final confirmation for the chosen CNN checkpoint.

The fixed CNN candidate, frozen-v4 and source-r2 are evaluated on six fresh
strata.  The pinned external agent is diagnostic-only: it is evaluated on the
two Task4 strata and in an additional seat-balanced shared four-agent match.
No code path trains, selects, copies or deploys a checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_cnn_n8.network import (  # noqa: E402
    ARCHITECTURE, FullBoardDuelingCNN, torch,
)
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE  # noqa: E402
from tools.v4_task4_duel_training import atomic_json, relative  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-final-confirmation-s152000.json"
INTERNAL_LABELS = ("candidate-cnn", "frozen-v4", "source-r2")
EXTERNAL_LABEL = "external-strong"
STANDARD_STRATA = ("task1", "task2", "task3_peaceful", "task3_coin", "task4_rule", "task4_mixed")
TASK4_STRATA = ("task4_rule", "task4_mixed")
AGENT_CODES = {
    "candidate-cnn": "model_a_cnn_n8",
    "frozen-v4": "model_a_dqn",
    "source-r2": "model_a_v4_curriculum",
    "external-strong": "feature_is_everything_confirm",
}
KIND_EVAL = "model-a-cnn-n8-final-confirmation-evaluation"
KIND_DIRECT = "model-a-cnn-n8-final-confirmation-direct-evaluation"
KIND_REPORT = "model-a-cnn-n8-final-confirmation-report"
ADDITIVE = (
    "rounds", "score", "coins", "kills", "crates", "bombs", "moves",
    "waits", "invalid_actions", "suicides", "steps",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "online_net" not in payload:
        raise ValueError(f"invalid confirmation checkpoint: {path}")
    return payload


def _completed(path: Path, kind: str, protocol_hash: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        return None
    if payload.get("kind") != kind or payload.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(f"completed confirmation artifact drift: {relative(path)}")
    return payload


def _metrics(raw: dict) -> dict:
    rounds = int(raw.get("rounds", 0)); steps = int(raw.get("steps", 0))
    moves = int(raw.get("moves", 0)); bombs = int(raw.get("bombs", 0)); invalid = int(raw.get("invalid", 0))
    waits = max(0, steps - moves - bombs - invalid)
    result = {
        "rounds": rounds, "score": int(raw.get("score", 0)), "coins": int(raw.get("coins", 0)),
        "kills": int(raw.get("kills", 0)), "crates": int(raw.get("crates", 0)),
        "bombs": bombs, "moves": moves, "waits": waits, "invalid_actions": invalid,
        "suicides": int(raw.get("suicides", 0)), "steps": steps,
        "decision_time_seconds": float(raw.get("time", 0.0)),
    }
    for key in ("score", "coins", "kills", "crates", "bombs", "invalid_actions", "suicides"):
        result[f"{key}_per_round"] = result[key] / rounds if rounds else 0.0
    result["wait_fraction"] = waits / steps if steps else 0.0
    result["mean_decision_time_ms"] = 1000.0 * result["decision_time_seconds"] / steps if steps else 0.0
    return result


def combine_metrics(rows: list[dict]) -> dict:
    result = {key: sum(int(row[key]) for row in rows) for key in ADDITIVE}
    result["decision_time_seconds"] = sum(float(row.get("decision_time_seconds", 0.0)) for row in rows)
    rounds, steps = result["rounds"], result["steps"]
    for key in ("score", "coins", "kills", "crates", "bombs", "invalid_actions", "suicides"):
        result[f"{key}_per_round"] = result[key] / rounds if rounds else 0.0
    result["wait_fraction"] = result["waits"] / steps if steps else 0.0
    result["mean_decision_time_ms"] = 1000.0 * result["decision_time_seconds"] / steps if steps else 0.0
    return result


def _integer_values(payload: object) -> set[int]:
    if isinstance(payload, int) and not isinstance(payload, bool):
        return {payload}
    if isinstance(payload, dict):
        result: set[int] = set()
        for value in payload.values():
            result.update(_integer_values(value))
        return result
    if isinstance(payload, list):
        result: set[int] = set()
        for value in payload:
            result.update(_integer_values(value))
        return result
    return set()


def _seed_values(payload: object) -> set[int]:
    result: set[int] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "seed" or key.endswith("_seed") or key.endswith("_seeds"):
                result.update(_integer_values(value))
            result.update(_seed_values(value))
    elif isinstance(payload, list):
        for value in payload:
            result.update(_seed_values(value))
    return result


def registered_seeds(protocol: dict) -> set[int]:
    return _seed_values(protocol["evaluation"])


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = registered_seeds(protocol)
    output_root = (ROOT / protocol["evaluation_manifest_directory"]).resolve()
    excluded = {protocol_path.resolve(), (ROOT / protocol["report_path"]).resolve()}
    conflicts = []
    for path in (ROOT / "experiments").rglob("*.json"):
        resolved = path.resolve()
        if resolved in excluded or output_root == resolved or output_root in resolved.parents:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overlap = sorted(registered & _seed_values(payload))
        if overlap:
            conflicts.append({"path": relative(path), "seeds": overlap})
    if conflicts:
        raise ValueError(f"registered s152000 seeds were already used: {conflicts}")


def _validate_checkpoint(protocol: dict, label: str) -> None:
    item = protocol["checkpoint_inventory"][label]
    path = ROOT / item["path"]
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise ValueError(f"confirmation checkpoint mismatch: {label}")
    payload = _torch_load(path)
    if label == "frozen-v4":
        if payload.get("architecture") not in (None, MODEL_ARCHITECTURE):
            raise ValueError("frozen-v4 architecture mismatch")
        return
    for key, value in item["required_metadata"].items():
        if payload.get(key) != value:
            raise ValueError(f"confirmation lineage mismatch: {label}/{key}")
    source_protocol = ROOT / item["protocol_path"]
    if not source_protocol.is_file() or sha256_file(source_protocol) != item["required_metadata"]["protocol_sha256"]:
        raise ValueError(f"confirmation source protocol mismatch: {label}")


def _validate_protocol(protocol_path: Path, protocol: dict) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-cnn-n8-final-confirmation":
        raise ValueError("wrong s152000 protocol kind")
    if protocol.get("protocol_id") != "model-a-cnn-n8-final-confirmation-s152000":
        raise ValueError("wrong s152000 protocol id")
    if tuple(protocol.get("internal_labels", ())) != INTERNAL_LABELS or protocol.get("fixed_candidate") != "candidate-cnn":
        raise ValueError("s152000 identity set changed")
    if not protocol.get("selection_free") or protocol.get("training_allowed") is not False:
        raise ValueError("s152000 must remain selection-free and evaluation-only")
    if protocol.get("checkpoint_copy_allowed") is not False or protocol.get("automatic_followup") is not False:
        raise ValueError("s152000 cannot copy checkpoints or start a follow-up")
    if protocol.get("external_reference_is_diagnostic_only") is not True:
        raise ValueError("external reference must remain diagnostic-only")
    if protocol.get("external_reference_used_for_gate") is not False:
        raise ValueError("external reference cannot affect the replacement gate")

    source = protocol["selection_source"]
    source_path = ROOT / source["path"]
    if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
        raise ValueError("Task4 selection report drift")
    source_report = json.loads(source_path.read_text(encoding="utf-8"))
    if (source_report.get("status") != "completed"
            or source_report["course_champion_selection"]["selected_label"] != "task4-r3-round0800"
            or source_report.get("next_stage_started") is not False):
        raise ValueError("Task4 selection terminal state changed")

    if set(protocol.get("checkpoint_inventory", {})) != set(INTERNAL_LABELS):
        raise ValueError("s152000 checkpoint inventory changed")
    for label in INTERNAL_LABELS:
        _validate_checkpoint(protocol, label)

    strata = protocol["evaluation"]["standard_strata"]
    if tuple(strata) != STANDARD_STRATA:
        raise ValueError("s152000 standard stratum order changed")
    expected_opponents = {
        "task1": ("coin-heaven", []),
        "task2": ("classic", []),
        "task3_peaceful": ("classic", ["seeded_peaceful_agent"]),
        "task3_coin": ("classic", ["seeded_coin_collector_agent"]),
        "task4_rule": ("classic", ["seeded_rule_based_agent"] * 3),
        "task4_mixed": ("classic", ["seeded_rule_based_agent", "seeded_coin_collector_agent", "seeded_random_agent"]),
    }
    for stratum, (scenario, opponents) in expected_opponents.items():
        spec = strata[stratum]
        if (spec.get("scenario") != scenario or spec.get("opponents") != opponents
                or int(spec.get("rounds_per_case", 0)) != 50 or len(spec.get("cases", ())) != 4):
            raise ValueError(f"s152000 standard contract changed: {stratum}")
        expected_targets = list(INTERNAL_LABELS) + ([EXTERNAL_LABEL] if stratum in TASK4_STRATA else [])
        if spec.get("target_labels") != expected_targets:
            raise ValueError(f"s152000 target labels changed: {stratum}")

    direct = protocol["evaluation"]["direct_shared"]
    if (direct.get("scenario") != "classic" or int(direct.get("rounds_per_case", 0)) != 25
            or len(direct.get("cases", ())) != 8):
        raise ValueError("s152000 direct-shared contract changed")
    expected_rosters = (
        ["candidate-cnn", "frozen-v4", "source-r2", "external-strong"],
        ["frozen-v4", "source-r2", "external-strong", "candidate-cnn"],
        ["source-r2", "external-strong", "candidate-cnn", "frozen-v4"],
        ["external-strong", "candidate-cnn", "frozen-v4", "source-r2"],
    ) * 2
    if tuple(case["roster"] for case in direct["cases"]) != expected_rosters:
        raise ValueError("s152000 direct roster rotation changed")

    budget = protocol["evaluation"]["budget"]
    if budget != {"standard_manifests": 80, "direct_manifests": 8, "total_manifests": 88,
                  "standard_identity_rounds": 4000, "direct_shared_rounds": 200,
                  "total_environment_rounds": 4200}:
        raise ValueError("s152000 evaluation budget changed")
    if len(registered_seeds(protocol)) != 168:
        raise ValueError("s152000 requires 168 globally unique seed values")

    gate = protocol["replacement_gate"]
    expected_gate = {
        "minimum_pooled_task4_score_gain_over_v4": 0.15,
        "minimum_task4_rule_score_delta_over_v4": 0.0,
        "minimum_task4_mixed_score_delta_over_v4": 0.0,
        "minimum_task4_block_wins_out_of_8": 5,
        "maximum_pooled_task4_suicide_increase_over_v4": 0.03,
        "maximum_pooled_task4_invalid_increase_over_v4": 0.05,
        "maximum_cpu_forward_p99_ms": 100.0,
        "maximum_cpu_forward_max_ms": 250.0,
    }
    if gate != expected_gate:
        raise ValueError("s152000 replacement gate changed")

    external = protocol["external_stress_opponent"]
    if (external.get("commit") != "a7fe5041b02548ce4438e502ca3adb11576bae75"
            or external.get("training_or_teacher_use_allowed") is not False
            or external.get("license_file_present") is not False):
        raise ValueError("s152000 external binding changed")
    for name, expected in protocol["source_bindings"].items():
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"s152000 source binding mismatch: {name}")
    assert_registered_seeds_untouched(protocol_path, protocol)


def load_protocol(path: Path) -> tuple[dict, Path, str]:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    raw = resolved.read_bytes()
    protocol = json.loads(raw)
    _validate_protocol(resolved, protocol)
    return protocol, resolved, hashlib.sha256(raw).hexdigest()


@contextmanager
def pinned_external_checkout(protocol: dict):
    external = protocol["external_stress_opponent"]
    with tempfile.TemporaryDirectory(prefix="bomberman-s152000-") as temporary:
        checkout = Path(temporary) / "repository"
        commands = (
            ["git", "init", "-q", str(checkout)],
            ["git", "-C", str(checkout), "remote", "add", "origin", external["repository"]],
            ["git", "-C", str(checkout), "fetch", "-q", "--depth", "1", "origin", external["commit"]],
            ["git", "-C", str(checkout), "checkout", "-q", "--detach", "FETCH_HEAD"],
        )
        for command in commands:
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            if completed.returncode:
                raise RuntimeError(f"external checkout failed: {' '.join(command)}\n{completed.stderr}")
        head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if head != external["commit"]:
            raise RuntimeError("external confirmation commit mismatch")
        agent_dir = checkout / external["agent_subdirectory"]
        for name, expected in external["file_sha256"].items():
            if sha256_file(agent_dir / name) != expected:
                raise RuntimeError(f"external confirmation hash mismatch: {name}")
        yield agent_dir


def _identity_environment(protocol_path: Path, protocol: dict, label: str, seed: int,
                          external_agent_dir: Path) -> tuple[str, dict[str, str]]:
    item = protocol["checkpoint_inventory"].get(label)
    if label == "candidate-cnn":
        checkpoint = (ROOT / item["path"]).resolve()
        return AGENT_CODES[label], {
            "MODEL_A_CNN_PROTOCOL_PATH": str((ROOT / item["protocol_path"]).resolve()),
            "MODEL_A_CNN_MODE": "evaluate", "MODEL_A_CNN_STAGE": "task4",
            "MODEL_A_CNN_REPLICA": "r3", "MODEL_A_CNN_SEED": str(seed),
            "MODEL_A_CNN_CHECKPOINT_PATH": str(checkpoint),
        }
    if label == "frozen-v4":
        checkpoint = (ROOT / item["path"]).resolve()
        return AGENT_CODES[label], {"MODEL_A_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_SEED": str(seed)}
    if label == "source-r2":
        checkpoint = (ROOT / item["path"]).resolve()
        return AGENT_CODES[label], {
            "MODEL_A_V4C_PROTOCOL_PATH": str((ROOT / item["protocol_path"]).resolve()),
            "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_V4C_ARM": "curriculum",
            "MODEL_A_V4C_REPLICA": "r2", "MODEL_A_V4C_STAGE": "task2", "MODEL_A_V4C_SEED": str(seed),
        }
    if label == EXTERNAL_LABEL:
        return AGENT_CODES[label], {
            "CNN_CONFIRM_PROTOCOL_PATH": str(protocol_path),
            "CNN_CONFIRM_EXTERNAL_AGENT_DIR": str(external_agent_dir),
            "CNN_CONFIRM_EXTERNAL_SEED": str(seed),
        }
    raise ValueError(f"unknown confirmation label: {label}")


def _opponent_environment(spec: dict, case: dict) -> dict[str, str]:
    opponents = spec["opponents"]
    result: dict[str, str] = {}
    if "seeded_rule_based_agent" in opponents:
        result["TASK4_RULE_SEED"] = str(case["opponent_seeds"]["rule"])
    if "seeded_coin_collector_agent" in opponents:
        result["TASK3_OPPONENT_SEED"] = str(case["opponent_seeds"]["coin"])
    if "seeded_peaceful_agent" in opponents:
        result["TASK3_OPPONENT_SEED"] = str(case["opponent_seeds"]["peaceful"])
    if "seeded_random_agent" in opponents:
        result["SEEDED_RANDOM_AGENT_SEED"] = str(case["opponent_seeds"]["random"])
    return result


def _checkpoint_hashes(protocol: dict) -> dict[str, str]:
    return {label: sha256_file(ROOT / protocol["checkpoint_inventory"][label]["path"]) for label in INTERNAL_LABELS}


def _standard_manifest(protocol: dict, stratum: str, label: str, case: dict) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / "standard" / stratum / label / f"s{case['world_seed']}.json"


def evaluate_standard(protocol_path: Path, protocol: dict, protocol_hash: str, stratum: str,
                      label: str, case: dict, external_agent_dir: Path) -> dict:
    spec = protocol["evaluation"]["standard_strata"][stratum]
    output = _standard_manifest(protocol, stratum, label, case)
    existing = _completed(output, KIND_EVAL, protocol_hash)
    expected = {"stratum": stratum, "label": label, "case": case, "scenario": spec["scenario"],
                "opponents": spec["opponents"], "rounds": spec["rounds_per_case"]}
    if existing is not None:
        if existing["evaluation"] != expected:
            raise RuntimeError(f"completed standard confirmation drift: {relative(output)}")
        return existing
    stats = output.with_suffix(".stats.json")
    if output.exists() or stats.exists():
        raise RuntimeError(f"orphaned standard confirmation output: {relative(output)}")
    target, overrides = _identity_environment(
        protocol_path, protocol, label, int(case["agent_seeds"][label]), external_agent_dir)
    overrides.update(_opponent_environment(spec, case))
    command = [
        sys.executable, "main.py", "play", "--agents", target, *spec["opponents"], "--train", "0",
        "--continue-without-training", "--no-gui", "--scenario", spec["scenario"],
        "--n-rounds", str(spec["rounds_per_case"]), "--seed", str(case["world_seed"]),
        "--save-stats", str(stats),
    ]
    before = _checkpoint_hashes(protocol)
    redacted = {key: ("<temporary-pinned-checkout>" if key == "CNN_CONFIRM_EXTERNAL_AGENT_DIR" else value)
                for key, value in overrides.items()}
    record = {
        "schema_version": 1, "kind": KIND_EVAL, "status": "running", "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
        "evaluation": expected, "command": command, "environment_overrides": redacted,
        "checkpoint_sha256_before": before, "raw_stats": relative(stats), "policy_updates": 0,
    }
    atomic_json(output, record)
    child = os.environ.copy(); child.update(overrides); child.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        after = _checkpoint_hashes(protocol)
        if completed.returncode or not stats.is_file() or after != before:
            raise RuntimeError("standard confirmation process, stats or checkpoint immutability failed")
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"]
        if target not in raw:
            raise RuntimeError("standard confirmation target missing from stats")
        record.update({"status": "completed", "checkpoint_sha256_after": after,
                       "target_metrics": _metrics(raw[target]),
                       "metrics_by_agent": {name: _metrics(value) for name, value in raw.items()}})
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"standard confirmation failed: {stratum}/{label}/{case['world_seed']}")
    return record


def _direct_manifest(protocol: dict, case: dict) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / "direct_shared" / f"{case['case_id']}.json"


def evaluate_direct(protocol_path: Path, protocol: dict, protocol_hash: str, case: dict,
                    external_agent_dir: Path) -> dict:
    spec = protocol["evaluation"]["direct_shared"]
    output = _direct_manifest(protocol, case)
    existing = _completed(output, KIND_DIRECT, protocol_hash)
    expected = {"case": case, "scenario": spec["scenario"], "rounds": spec["rounds_per_case"]}
    if existing is not None:
        if existing["evaluation"] != expected:
            raise RuntimeError(f"completed direct confirmation drift: {relative(output)}")
        return existing
    stats = output.with_suffix(".stats.json")
    if output.exists() or stats.exists():
        raise RuntimeError(f"orphaned direct confirmation output: {relative(output)}")
    codes = [AGENT_CODES[label] for label in case["roster"]]
    overrides: dict[str, str] = {}
    for label in case["roster"]:
        _, identity_overrides = _identity_environment(
            protocol_path, protocol, label, int(case["agent_seeds"][label]), external_agent_dir)
        overlap = set(overrides) & set(identity_overrides)
        if overlap:
            raise RuntimeError(f"direct confirmation environment collision: {sorted(overlap)}")
        overrides.update(identity_overrides)
    command = [
        sys.executable, "main.py", "play", "--agents", *codes, "--train", "0",
        "--continue-without-training", "--no-gui", "--scenario", spec["scenario"],
        "--n-rounds", str(spec["rounds_per_case"]), "--seed", str(case["world_seed"]),
        "--save-stats", str(stats),
    ]
    before = _checkpoint_hashes(protocol)
    redacted = {key: ("<temporary-pinned-checkout>" if key == "CNN_CONFIRM_EXTERNAL_AGENT_DIR" else value)
                for key, value in overrides.items()}
    record = {
        "schema_version": 1, "kind": KIND_DIRECT, "status": "running", "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
        "evaluation": expected, "command": command, "environment_overrides": redacted,
        "checkpoint_sha256_before": before, "external_commit": protocol["external_stress_opponent"]["commit"],
        "raw_stats": relative(stats), "policy_updates": 0,
    }
    atomic_json(output, record)
    child = os.environ.copy(); child.update(overrides); child.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        after = _checkpoint_hashes(protocol)
        if completed.returncode or not stats.is_file() or after != before:
            raise RuntimeError("direct confirmation process, stats or checkpoint immutability failed")
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"]
        reverse = {code: label for label, code in AGENT_CODES.items()}
        if set(raw) != set(reverse):
            raise RuntimeError(f"unexpected direct confirmation agents: {sorted(raw)}")
        record.update({"status": "completed", "checkpoint_sha256_after": after,
                       "metrics_by_label": {reverse[name]: _metrics(value) for name, value in raw.items()}})
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"direct confirmation failed: {case['case_id']}")
    return record


def candidate_latency(protocol: dict) -> dict:
    item = protocol["checkpoint_inventory"]["candidate-cnn"]
    payload = _torch_load(ROOT / item["path"])
    model = FullBoardDuelingCNN().cpu().eval()
    model.load_state_dict(payload["online_net"], strict=True)
    torch.set_num_threads(1)
    spatial = torch.zeros((1, 11, 33, 33), dtype=torch.float32)
    scalars = torch.zeros((1, 6), dtype=torch.float32)
    samples = []
    with torch.no_grad():
        for _ in range(20):
            output = model(spatial, scalars)
        for _ in range(1000):
            start = time.perf_counter(); output = model(spatial, scalars)
            samples.append(1000.0 * (time.perf_counter() - start))
    if tuple(output.shape) != (1, 6):
        raise RuntimeError("candidate latency benchmark output mismatch")
    ordered = sorted(samples)
    return {"p50": ordered[499], "p95": ordered[949], "p99": ordered[989], "max": ordered[-1]}


def summarize_standard(protocol: dict, runs: list[dict]) -> tuple[dict, dict]:
    rows: dict[str, dict] = {label: {} for label in (*INTERNAL_LABELS, EXTERNAL_LABEL)}
    cases: dict[str, dict] = {stratum: {} for stratum in STANDARD_STRATA}
    for stratum in STANDARD_STRATA:
        labels = list(INTERNAL_LABELS) + ([EXTERNAL_LABEL] if stratum in TASK4_STRATA else [])
        for label in labels:
            selected = [run for run in runs if run["evaluation"]["stratum"] == stratum
                        and run["evaluation"]["label"] == label]
            rows[label][stratum] = combine_metrics([run["target_metrics"] for run in selected])
        for case in protocol["evaluation"]["standard_strata"][stratum]["cases"]:
            world = int(case["world_seed"])
            cases[stratum][str(world)] = {
                label: next(run["target_metrics"] for run in runs
                            if run["evaluation"]["stratum"] == stratum
                            and run["evaluation"]["label"] == label
                            and int(run["evaluation"]["case"]["world_seed"]) == world)
                for label in labels
            }
    for label in (*INTERNAL_LABELS, EXTERNAL_LABEL):
        rows[label]["pooled_task4"] = combine_metrics([rows[label][name] for name in TASK4_STRATA])
    return rows, cases


def summarize_direct(runs: list[dict]) -> dict:
    pooled = {label: combine_metrics([run["metrics_by_label"][label] for run in runs])
              for label in (*INTERNAL_LABELS, EXTERNAL_LABEL)}
    blocks = []
    for run in runs:
        metrics = run["metrics_by_label"]
        ranking = sorted(metrics, key=lambda label: (
            metrics[label]["score_per_round"], metrics[label]["kills_per_round"],
            -metrics[label]["suicides_per_round"]), reverse=True)
        blocks.append({"case_id": run["evaluation"]["case"]["case_id"],
                       "world_seed": run["evaluation"]["case"]["world_seed"],
                       "roster": run["evaluation"]["case"]["roster"],
                       "metrics": metrics, "ranking": ranking})
    return {"pooled": pooled, "blocks": blocks}


def decide(protocol: dict, rows: dict, cases: dict, latency: dict) -> dict:
    gate = protocol["replacement_gate"]
    candidate = rows["candidate-cnn"]; v4 = rows["frozen-v4"]
    block_wins = sum(
        cases[stratum][str(case["world_seed"])]["candidate-cnn"]["score_per_round"]
        > cases[stratum][str(case["world_seed"])]["frozen-v4"]["score_per_round"] + 1e-12
        for stratum in TASK4_STRATA
        for case in protocol["evaluation"]["standard_strata"][stratum]["cases"]
    )
    checks = {
        "pooled_task4_score_gain": candidate["pooled_task4"]["score_per_round"]
        >= v4["pooled_task4"]["score_per_round"] + gate["minimum_pooled_task4_score_gain_over_v4"] - 1e-12,
        "task4_rule_not_below_v4": candidate["task4_rule"]["score_per_round"]
        >= v4["task4_rule"]["score_per_round"] + gate["minimum_task4_rule_score_delta_over_v4"] - 1e-12,
        "task4_mixed_not_below_v4": candidate["task4_mixed"]["score_per_round"]
        >= v4["task4_mixed"]["score_per_round"] + gate["minimum_task4_mixed_score_delta_over_v4"] - 1e-12,
        "task4_block_support": block_wins >= gate["minimum_task4_block_wins_out_of_8"],
        "pooled_task4_suicide_guard": candidate["pooled_task4"]["suicides_per_round"]
        <= v4["pooled_task4"]["suicides_per_round"] + gate["maximum_pooled_task4_suicide_increase_over_v4"] + 1e-12,
        "pooled_task4_invalid_guard": candidate["pooled_task4"]["invalid_actions_per_round"]
        <= v4["pooled_task4"]["invalid_actions_per_round"] + gate["maximum_pooled_task4_invalid_increase_over_v4"] + 1e-12,
        "latency_p99": latency["p99"] < gate["maximum_cpu_forward_p99_ms"],
        "latency_max": latency["max"] < gate["maximum_cpu_forward_max_ms"],
        "checkpoint_hashes_unchanged": True,
    }
    passed = all(checks.values())
    return {
        "decision": ("candidate_confirmed_replace_v4_pending_docker_stop" if passed
                     else "candidate_not_confirmed_retain_v4_stop"),
        "candidate_confirmed": passed, "checks": checks,
        "task4_block_wins_out_of_8": block_wins,
        "pooled_task4_score_gain_over_v4": candidate["pooled_task4"]["score_per_round"] - v4["pooled_task4"]["score_per_round"],
        "external_used_for_gate": False, "selection_made": False, "training_started": False,
        "checkpoint_copied": False, "default_model_replaced": False,
        "automatic_followup_started": False, "awaiting_user_instruction": True,
    }


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def write_report(protocol_path: Path, protocol: dict, protocol_hash: str, standard_runs: list[dict],
                 direct_runs: list[dict], latency: dict) -> tuple[dict, str]:
    output = report_path(protocol)
    existing = _completed(output, KIND_REPORT, protocol_hash)
    if existing is not None:
        return existing, sha256_file(output)
    rows, cases = summarize_standard(protocol, standard_runs)
    direct = summarize_direct(direct_runs)
    result = decide(protocol, rows, cases, latency)
    result["external_standard_minus_candidate_task4_score"] = (
        rows[EXTERNAL_LABEL]["pooled_task4"]["score_per_round"]
        - rows["candidate-cnn"]["pooled_task4"]["score_per_round"])
    result["external_direct_minus_candidate_score"] = (
        direct["pooled"][EXTERNAL_LABEL]["score_per_round"]
        - direct["pooled"]["candidate-cnn"]["score_per_round"])
    report = {
        "schema_version": 1, "kind": KIND_REPORT, "status": "completed", "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash, "standard_rows": rows, "standard_cases": cases,
        "direct_shared": direct, "candidate_cpu_forward_latency_ms": latency, "result": result,
        "source_artifacts_modified": False, "external_source_persisted_in_repository": False,
        "default_checkpoint_modified": False, "automatic_followup_started": False,
        "awaiting_user_instruction": True,
    }
    atomic_json(output, report)
    return report, sha256_file(output)


def dry_run(protocol: dict) -> dict:
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"],
        "fixed_candidate": protocol["fixed_candidate"], "internal_labels": list(INTERNAL_LABELS),
        "standard_strata": list(STANDARD_STRATA), "standard_manifests": 80,
        "direct_shared_manifests": 8, "total_manifests": 88,
        "total_environment_rounds": 4200, "external_diagnostic_rounds": 600,
        "external_used_for_gate": False, "external_checkout_started": False,
        "formal_evaluation_started": False, "training_started": False,
        "selection_allowed": False, "checkpoint_copy_allowed": False,
        "automatic_followup_started": False, "writes_terminal_report_and_stops": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol, protocol_path, protocol_hash = load_protocol(args.protocol)
    if not args.execute:
        print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
        return 0
    existing = _completed(report_path(protocol), KIND_REPORT, protocol_hash)
    if existing is not None:
        report, report_hash = existing, sha256_file(report_path(protocol))
    else:
        with pinned_external_checkout(protocol) as external_agent_dir:
            standard_runs = []
            for stratum, spec in protocol["evaluation"]["standard_strata"].items():
                for label in spec["target_labels"]:
                    for case in spec["cases"]:
                        standard_runs.append(evaluate_standard(
                            protocol_path, protocol, protocol_hash, stratum, label, case, external_agent_dir))
            direct_runs = [evaluate_direct(protocol_path, protocol, protocol_hash, case, external_agent_dir)
                           for case in protocol["evaluation"]["direct_shared"]["cases"]]
        latency = candidate_latency(protocol)
        report, report_hash = write_report(
            protocol_path, protocol, protocol_hash, standard_runs, direct_runs, latency)
    print(json.dumps({"status": report["status"], "report": relative(report_path(protocol)),
                      "report_sha256": report_hash, **report["result"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
