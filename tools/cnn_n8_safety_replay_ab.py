"""Paired replay-sampling-only safety fine-tune of the fixed CNN candidate."""

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
from tools.cnn_n8_task1 import _metrics, _torch_load, atomic_json, preflight, relative, sha256_file, utc_now  # noqa: E402
from tools.cnn_n8_task4 import combine_metrics  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-safety-replay-ab-s153000.json"
ARMS = ("uniform_n8", "balanced_safety_n8")
REPLICAS = ("r1", "r2", "r3")
MILESTONES = (100, 200)
TASK4_STRATA = ("task4_rule", "task4_mixed")
RETENTION_STRATA = ("task2", "task3_peaceful", "task3_coin")
KIND_TRAIN = "model-a-cnn-n8-safety-replay-ab-training"
KIND_EVAL = "model-a-cnn-n8-safety-replay-ab-evaluation"
KIND_REPORT = "model-a-cnn-n8-safety-replay-ab-report"


def _completed(path: Path, kind: str, protocol_hash: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        return None
    if payload.get("kind") != kind or payload.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(f"completed artifact drift: {relative(path)}")
    return payload


def _seed_values(node) -> list[int]:
    values: list[int] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("_seed") and isinstance(value, int):
                values.append(value)
            else:
                values.extend(_seed_values(value))
    elif isinstance(node, list):
        for value in node:
            values.extend(_seed_values(value))
    return values


def _validate_seed_freshness(protocol_path: Path, protocol: dict) -> None:
    registered = _seed_values({"training": protocol["training"], "evaluation": protocol["evaluation"]})
    current = set(registered)
    if len(registered) != len(current) or len(current) != 41:
        raise RuntimeError("safety A/B must register exactly 41 unique stored seed values")
    if not current or min(current) < 153000:
        raise RuntimeError("safety A/B seeds must use the untouched 153xxx+ namespace")
    old_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "experiments/configs").glob("*.json")
        if path.resolve() != protocol_path.resolve()
    )
    if any(str(seed) in old_text for seed in current):
        raise RuntimeError("safety A/B seed was already registered")


def load_protocol(path: Path) -> tuple[dict, Path, str]:
    path = path.resolve()
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("kind") != "model-a-cnn-n8-safety-replay-ab":
        raise RuntimeError("wrong CNN safety A/B protocol")
    if tuple(protocol.get("arms", ())) != ARMS or tuple(protocol.get("replicas", ())) != REPLICAS:
        raise RuntimeError("CNN safety A/B identities changed")
    if protocol.get("single_training_variable") != "replay_sampling_uniform_vs_reserved_8_kill_8_self":
        raise RuntimeError("CNN safety A/B must change replay sampling only")
    learning = protocol["learning"]
    if learning["arms"] != {
        "uniform_n8": {"uniform_slots": 256, "kill_chain_slots": 0, "self_kill_chain_slots": 0},
        "balanced_safety_n8": {"uniform_slots": 240, "kill_chain_slots": 8, "self_kill_chain_slots": 8},
    }:
        raise RuntimeError("CNN safety replay allocation changed")
    if (learning["batch_size"] != 256 or learning["n_step"] != 8
            or learning["loss"] != "smooth_l1" or learning["reward_contract_changed"]):
        raise RuntimeError("CNN safety learning contract changed")
    if protocol["training"]["rounds_per_arm_replica"] != 200:
        raise RuntimeError("CNN safety training budget changed")
    if tuple(protocol["training"]["milestones"]) != MILESTONES:
        raise RuntimeError("CNN safety milestones changed")
    if tuple(protocol["evaluation"]["task4_strata"]) != TASK4_STRATA:
        raise RuntimeError("CNN safety Task4 strata changed")
    if tuple(protocol["evaluation"]["retention_strata"]) != RETENTION_STRATA:
        raise RuntimeError("CNN safety retention strata changed")
    if protocol.get("automatic_followup") is not False or protocol.get("checkpoint_copy_allowed") is not False:
        raise RuntimeError("CNN safety A/B must stop without deployment")
    for name, expected in protocol["source_bindings"].items():
        source = ROOT / name
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"CNN safety source binding mismatch: {name}")
    for key in ("parent", "s152_report"):
        spec = protocol[key]
        if sha256_file(ROOT / spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"CNN safety bound artifact drift: {key}")
    for key, spec in protocol["baselines"].items():
        if sha256_file(ROOT / spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"CNN safety baseline drift: {key}")
    source_protocol = ROOT / protocol["parent"]["source_protocol"]
    if sha256_file(source_protocol) != protocol["parent"]["source_protocol_sha256"]:
        raise RuntimeError("CNN safety parent protocol drift")
    parent = _torch_load(ROOT / protocol["parent"]["path"])
    if (parent.get("architecture") != ARCHITECTURE or parent.get("stage") != "task4"
            or parent.get("replica") != "r3" or parent.get("stage_rounds") != 800):
        raise RuntimeError("CNN safety parent identity mismatch")
    report = json.loads((ROOT / protocol["s152_report"]["path"]).read_text(encoding="utf-8"))
    if (report.get("status") != "completed"
            or report.get("result", {}).get("decision") != "candidate_not_confirmed_retain_v4_stop"):
        raise RuntimeError("s152 prerequisite decision changed")
    _validate_seed_freshness(path, protocol)
    return protocol, path, hashlib.sha256(raw).hexdigest()


def checkpoint_path(protocol: dict, arm: str, replica: str, milestone: int) -> Path:
    return ROOT / protocol["checkpoint_directory"] / arm / replica / f"round-{milestone:04d}.pt"


def validate_checkpoint(protocol: dict, protocol_hash: str, arm: str, replica: str,
                        milestone: int) -> dict:
    path = checkpoint_path(protocol, arm, replica, milestone)
    payload = _torch_load(path)
    expected = {
        "architecture": ARCHITECTURE,
        "protocol_sha256": protocol_hash,
        "stage": "task4_safety_replay_ab",
        "arm": arm,
        "replica": replica,
        "stage_rounds": milestone,
        "completed_total_rounds": 2750 + milestone,
        "parent_sha256": protocol["parent"]["sha256"],
        "n_step": 8,
        "single_training_variable": "replay_sampling_uniform_vs_reserved_8_kill_8_self",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"safety checkpoint mismatch {arm}/{replica}/{milestone}: {key}")
    slots = protocol["learning"]["arms"][arm]
    if payload.get("reserved_slots") != {
        "kill_chain": slots["kill_chain_slots"], "self_kill_chain": slots["self_kill_chain_slots"],
    }:
        raise RuntimeError("safety checkpoint sampling allocation mismatch")
    diagnostics = payload["training_diagnostics"]
    if (diagnostics["raw_transitions"] != diagnostics["matured_targets"]
            or len(diagnostics["per_round"]) != milestone):
        raise RuntimeError("safety checkpoint transition accounting mismatch")
    return payload


def train_one(protocol: dict, protocol_path: Path, protocol_hash: str, arm: str, replica: str) -> dict:
    manifest = ROOT / protocol["training_manifest_directory"] / arm / f"{replica}.json"
    existing = _completed(manifest, KIND_TRAIN, protocol_hash)
    if existing is not None:
        for milestone in MILESTONES:
            path = checkpoint_path(protocol, arm, replica, milestone)
            validate_checkpoint(protocol, protocol_hash, arm, replica, milestone)
            if existing["checkpoints"][str(milestone)]["sha256"] != sha256_file(path):
                raise RuntimeError("completed safety checkpoint drift")
        return existing
    stats = manifest.with_suffix(".stats.json")
    directory = checkpoint_path(protocol, arm, replica, 100).parent
    if stats.exists() or any(directory.glob("round-*.pt")):
        raise RuntimeError(f"orphaned safety training artifact: {arm}/{replica}")
    seed = protocol["training"]["paired_seeds"][replica]
    parent = protocol["parent"]
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_cnn_n8_safety_ab",
        "seeded_rule_based_agent", "seeded_rule_based_agent", "seeded_rule_based_agent",
        "--train", "1", "--continue-without-training", "--no-gui", "--scenario", "classic",
        "--n-rounds", str(protocol["training"]["rounds_per_arm_replica"]),
        "--seed", str(seed["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = {
        "CNN_SAFETY_AB_PROTOCOL_PATH": str(protocol_path), "CNN_SAFETY_AB_MODE": "train",
        "CNN_SAFETY_AB_ARM": arm, "CNN_SAFETY_AB_REPLICA": replica,
        "CNN_SAFETY_AB_SEED": str(seed["agent_seed"]),
        "CNN_SAFETY_AB_CHECKPOINT_PATH": str((ROOT / parent["path"]).resolve()),
        "CNN_SAFETY_AB_CHECKPOINT_SHA256": parent["sha256"],
        "CNN_SAFETY_AB_CHECKPOINT_DIR": str(directory.resolve()),
        "TASK4_RULE_SEED": str(seed["rule_seed"]), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    record = {
        "schema_version": 1, "kind": KIND_TRAIN, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash, "arm": arm, "replica": replica, "paired_seed": seed,
        "status": "running", "started_at_utc": utc_now(), "command": command,
        "environment_overrides": overrides, "raw_stats": relative(stats),
    }
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides); child.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file():
            raise RuntimeError("safety training process failed")
        checkpoints = {}
        endpoint = None
        for milestone in MILESTONES:
            path = checkpoint_path(protocol, arm, replica, milestone)
            endpoint = validate_checkpoint(protocol, protocol_hash, arm, replica, milestone)
            checkpoints[str(milestone)] = {"path": relative(path), "sha256": sha256_file(path)}
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"]["model_a_cnn_n8_safety_ab"]
        diagnostics = endpoint["training_diagnostics"]
        sampling_names = (
            "sampled_batches", "sampled_transitions", "kill_chain_targets_created",
            "self_chain_targets_created", "requested_kill_slots", "fulfilled_kill_slots",
            "requested_self_slots", "fulfilled_self_slots", "fallback_uniform_slots",
            "realized_kill_chain_samples", "realized_self_chain_samples",
            "replay_size", "replay_kill_items", "replay_self_items",
        )
        record.update({
            "status": "completed", "training_metrics": _metrics(raw), "checkpoints": checkpoints,
            "sampling_diagnostics": {name: int(diagnostics[name]) for name in sampling_names},
        })
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"safety training failed: {arm}/{replica}: {record.get('error')}")
    return record


def _identities(protocol: dict) -> dict[str, tuple[str, Path, str, str]]:
    result = {
        "parent-cnn": ("model_a_cnn_n8_safety_ab", ROOT / protocol["parent"]["path"], "parent", "r3"),
        "frozen-v4": ("model_a_dqn", ROOT / protocol["baselines"]["frozen-v4"]["path"], "", ""),
        "source-r2": ("model_a_dqn", ROOT / protocol["baselines"]["source-r2"]["path"], "", ""),
    }
    for arm in ARMS:
        for replica in REPLICAS:
            for milestone in MILESTONES:
                label = f"{arm}-{replica}-round{milestone:04d}"
                result[label] = ("model_a_cnn_n8_safety_ab", checkpoint_path(protocol, arm, replica, milestone), arm, replica)
    return result


def _opponent_overrides(opponents: list[str], case: dict) -> dict[str, str]:
    result = {}
    if "seeded_rule_based_agent" in opponents:
        result["TASK4_RULE_SEED"] = str(case["rule_seed"])
    if "seeded_coin_collector_agent" in opponents:
        result["TASK3_OPPONENT_SEED"] = str(case["coin_seed"])
    if "seeded_peaceful_agent" in opponents:
        result["TASK3_OPPONENT_SEED"] = str(case["peaceful_seed"])
    if "seeded_random_agent" in opponents:
        result["SEEDED_RANDOM_AGENT_SEED"] = str(case["random_seed"])
    return result


def evaluate_one(protocol: dict, protocol_path: Path, protocol_hash: str, stratum: str,
                 label: str, identity: tuple[str, Path, str, str], case: dict) -> dict:
    target, checkpoint, arm, replica = identity
    manifest = ROOT / protocol["evaluation_manifest_directory"] / stratum / label / f"s{case['world_seed']}.json"
    existing = _completed(manifest, KIND_EVAL, protocol_hash)
    checkpoint_hash = sha256_file(checkpoint)
    if existing is not None:
        if existing["checkpoint"]["sha256"] != checkpoint_hash or existing["case"] != case:
            raise RuntimeError("completed safety evaluation drift")
        return existing
    stats = manifest.with_suffix(".stats.json")
    if stats.exists():
        raise RuntimeError(f"orphaned safety evaluation stats: {relative(stats)}")
    spec = protocol["evaluation"]["strata"][stratum]
    opponents = list(spec["opponents"])
    command = [
        sys.executable, "main.py", "play", "--agents", target, *opponents, "--train", "0",
        "--continue-without-training", "--no-gui", "--scenario", spec["scenario"],
        "--n-rounds", str(spec["rounds_per_case"]), "--seed", str(case["world_seed"]),
        "--save-stats", str(stats),
    ]
    if target == "model_a_dqn":
        overrides = {"MODEL_A_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_SEED": str(case["agent_seed"])}
    else:
        overrides = {
            "CNN_SAFETY_AB_PROTOCOL_PATH": str(protocol_path), "CNN_SAFETY_AB_MODE": "evaluate",
            "CNN_SAFETY_AB_ARM": arm, "CNN_SAFETY_AB_REPLICA": replica,
            "CNN_SAFETY_AB_SEED": str(case["agent_seed"]),
            "CNN_SAFETY_AB_CHECKPOINT_PATH": str(checkpoint),
            "CNN_SAFETY_AB_CHECKPOINT_SHA256": checkpoint_hash,
        }
    overrides.update(_opponent_overrides(opponents, case))
    record = {
        "schema_version": 1, "kind": KIND_EVAL, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash, "stratum": stratum, "label": label, "case": case,
        "status": "running", "started_at_utc": utc_now(), "command": command,
        "environment_overrides": overrides,
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
        "raw_stats": relative(stats),
    }
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides); child.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file() or sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("safety evaluation failed or mutated checkpoint")
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"][target]
        record.update({"status": "completed", "target_metrics": _metrics(raw)})
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"safety evaluation failed: {stratum}/{label}/{case['world_seed']}")
    return record


def evaluate_stratum(protocol: dict, protocol_path: Path, protocol_hash: str, stratum: str,
                     identities: dict[str, tuple[str, Path, str, str]]) -> dict[str, dict]:
    cases = protocol["evaluation"]["strata"][stratum]["cases"]
    return {
        label: combine_metrics([
            evaluate_one(protocol, protocol_path, protocol_hash, stratum, label, identity, case)["target_metrics"]
            for case in cases
        ])
        for label, identity in identities.items()
    }


def decide_task4(protocol: dict, rows: dict[str, dict[str, dict]]) -> dict:
    combined = {
        label: combine_metrics([rows["task4_rule"][label], rows["task4_mixed"][label]])
        for label in rows["task4_rule"]
    }
    gate = protocol["decision_rule"]
    milestone_results = {}
    passing = []
    for milestone in MILESTONES:
        pair_results = {}
        control_rows, treatment_rows = [], []
        for replica in REPLICAS:
            control = combined[f"uniform_n8-{replica}-round{milestone:04d}"]
            treatment = combined[f"balanced_safety_n8-{replica}-round{milestone:04d}"]
            control_rows.append(control); treatment_rows.append(treatment)
            pair_results[replica] = {
                "score_delta": treatment["score_per_round"] - control["score_per_round"],
                "suicide_delta": treatment["suicides_per_round"] - control["suicides_per_round"],
                "supportive": (
                    treatment["suicides_per_round"] <= control["suicides_per_round"]
                    - gate["pair_minimum_suicide_reduction"] + 1e-12
                    and treatment["score_per_round"] >= control["score_per_round"]
                    + gate["pair_minimum_score_delta"] - 1e-12
                ),
            }
        control_pool = combine_metrics(control_rows)
        treatment_pool = combine_metrics(treatment_rows)
        checks = {
            "supportive_pairs": sum(item["supportive"] for item in pair_results.values())
            >= gate["minimum_supportive_pairs"],
            "pooled_suicide_reduction": treatment_pool["suicides_per_round"]
            <= control_pool["suicides_per_round"] - gate["pooled_minimum_suicide_reduction"] + 1e-12,
            "pooled_score_preserved": treatment_pool["score_per_round"]
            >= control_pool["score_per_round"] + gate["pooled_minimum_score_delta"] - 1e-12,
            "pooled_kills_preserved": treatment_pool["kills_per_round"]
            >= control_pool["kills_per_round"] + gate["pooled_minimum_kills_delta"] - 1e-12,
        }
        passed = all(checks.values())
        if passed:
            passing.append(milestone)
        milestone_results[str(milestone)] = {
            "control": control_pool, "treatment": treatment_pool,
            "pair_results": pair_results, "checks": checks, "passed": passed,
        }
    if not passing:
        return {"supported": False, "decision": "safety_replay_not_supported_stop",
                "milestones": milestone_results, "selected_milestone": None,
                "selected_label": None, "combined_rows": combined}
    selected_milestone = min(
        passing,
        key=lambda value: (
            milestone_results[str(value)]["treatment"]["suicides_per_round"],
            -milestone_results[str(value)]["treatment"]["score_per_round"], value,
        ),
    )
    parent = combined["parent-cnn"]
    candidates = []
    for replica in REPLICAS:
        label = f"balanced_safety_n8-{replica}-round{selected_milestone:04d}"
        item = combined[label]
        if (item["suicides_per_round"] <= parent["suicides_per_round"]
                - gate["candidate_minimum_suicide_reduction_from_parent"] + 1e-12
                and item["score_per_round"] >= parent["score_per_round"]
                + gate["candidate_minimum_score_delta_from_parent"] - 1e-12):
            candidates.append(label)
    if not candidates:
        return {"supported": True, "decision": "pooled_signal_without_eligible_checkpoint_stop",
                "milestones": milestone_results, "selected_milestone": selected_milestone,
                "selected_label": None, "combined_rows": combined}
    best_score = max(combined[label]["score_per_round"] for label in candidates)
    near = [label for label in candidates
            if combined[label]["score_per_round"] >= best_score - gate["candidate_near_best_score_tolerance"]]
    selected = min(near, key=lambda label: (
        combined[label]["suicides_per_round"],
        combined[label]["invalid_actions_per_round"],
        -combined[label]["score_per_round"], label,
    ))
    return {"supported": True, "decision": "evaluate_selected_safety_candidate_retention",
            "milestones": milestone_results, "selected_milestone": selected_milestone,
            "selected_label": selected, "combined_rows": combined}


def decide_retention(protocol: dict, selected: str, rows: dict[str, dict[str, dict]]) -> dict:
    gate = protocol["decision_rule"]
    checks = {}
    for stratum in RETENTION_STRATA:
        candidate, parent = rows[stratum][selected], rows[stratum]["parent-cnn"]
        checks[f"{stratum}_score"] = candidate["score_per_round"] >= (
            parent["score_per_round"] + gate["retention_minimum_score_delta"] - 1e-12
        )
        checks[f"{stratum}_suicide"] = candidate["suicides_per_round"] <= (
            parent["suicides_per_round"] + gate["retention_maximum_suicide_increase"] + 1e-12
        )
    passed = all(checks.values())
    return {
        "checks": checks, "passed": passed,
        "decision": "safety_candidate_selected_stop_before_confirmation" if passed
        else "safety_candidate_failed_retention_stop",
    }


def execute(protocol: dict, protocol_path: Path, protocol_hash: str, hardware: dict) -> dict:
    training = {}
    for replica in REPLICAS:
        for arm in ARMS:
            training[f"{arm}-{replica}"] = train_one(protocol, protocol_path, protocol_hash, arm, replica)
    identities = _identities(protocol)
    task4_rows = {
        stratum: evaluate_stratum(protocol, protocol_path, protocol_hash, stratum, identities)
        for stratum in TASK4_STRATA
    }
    task4_result = decide_task4(protocol, task4_rows)
    retention_rows = {}
    retention_result = None
    selected = task4_result["selected_label"]
    if selected is not None:
        retention_identities = {label: identities[label] for label in ("parent-cnn", "frozen-v4", "source-r2", selected)}
        retention_rows = {
            stratum: evaluate_stratum(protocol, protocol_path, protocol_hash, stratum, retention_identities)
            for stratum in RETENTION_STRATA
        }
        retention_result = decide_retention(protocol, selected, retention_rows)
    decision = task4_result["decision"] if retention_result is None else retention_result["decision"]
    report = {
        "schema_version": 1, "kind": KIND_REPORT, "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "status": "completed", "completed_at_utc": utc_now(), "hardware_preflight": hardware,
        "single_training_variable": protocol["single_training_variable"],
        "training_manifests": {key: relative(ROOT / protocol["training_manifest_directory"]
                                             / key.rsplit("-", 1)[0] / f"{key.rsplit('-', 1)[1]}.json")
                               for key in training},
        "training_rows": {
            key: {"training_metrics": item["training_metrics"],
                  "sampling_diagnostics": item["sampling_diagnostics"]}
            for key, item in training.items()
        },
        "task4_rows": task4_rows, "task4_result": task4_result,
        "retention_rows": retention_rows, "retention_result": retention_result,
        "result": {
            "decision": decision, "selected_label": selected,
            "training_started": True, "checkpoint_copied": False,
            "default_model_replaced": False, "confirmation_started": False,
            "automatic_followup_started": False,
        },
        "awaiting_user_instruction": True, "default_checkpoint_modified": False,
        "automatic_followup_started": False,
    }
    output = ROOT / protocol["report_path"]
    if output.exists():
        raise RuntimeError(f"refusing to overwrite safety report: {relative(output)}")
    atomic_json(output, report)
    return report


def dry_run(protocol: dict, protocol_hash: str, hardware: dict) -> dict:
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash, "hardware_preflight": hardware,
        "parent": protocol["parent"], "arms": list(ARMS), "replicas": list(REPLICAS),
        "milestones": list(MILESTONES), "training_environment_rounds": 1200,
        "task4_evaluation_rounds": 1500, "conditional_retention_rounds": 600,
        "maximum_environment_rounds": 3300,
        "single_training_variable": protocol["single_training_variable"],
        "formal_training_started": False, "formal_evaluation_started": False,
        "selection_started": False, "checkpoint_copy_allowed": False,
        "automatic_followup_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol, protocol_path, protocol_hash = load_protocol(args.protocol)
    hardware = preflight(protocol_path, protocol)
    result = execute(protocol, protocol_path, protocol_hash, hardware) if args.execute else dry_run(
        protocol, protocol_hash, hardware,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
