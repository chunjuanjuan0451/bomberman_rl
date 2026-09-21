"""Run the preregistered local7/full33 resource-visibility stage-1 experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v4_global_resource.callbacks import _torch_load  # noqa: E402
from agent_code.model_a_v4_global_resource.config import (  # noqa: E402
    ARMS, ENDPOINT_ROUND, REPLICAS, architecture_name, checkpoint_path,
    load_protocol, milestone_path, sha256_file,
)
from agent_code.model_a_v4_global_resource.features import resource_features  # noqa: E402
from agent_code.model_a_v4_global_resource.network import GlobalResourceDQN, torch  # noqa: E402
from tools.v4_task4_duel_training import (  # noqa: E402
    aggregate, atomic_json, load_completed, metrics_by_agent, relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-global-resource-stage1-s140000.json"
TRAINING_KIND = "model-a-v4-global-resource-stage1-training"
EVALUATION_KIND = "model-a-v4-global-resource-stage1-evaluation"
REPORT_KIND = "model-a-v4-global-resource-stage1-report"
TARGET_AGENT = "model_a_v4_global_resource"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def training_manifest_path(protocol: dict, arm: str, replica: str) -> Path:
    return ROOT / protocol["training_manifest_directory"] / arm / f"{replica}.json"


def evaluation_manifest_path(protocol: dict, stratum: str, label: str, seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / stratum / f"{label}-s{seed}.json"


def diagnostic_path(protocol: dict, stratum: str, label: str, seed: int) -> Path:
    return ROOT / protocol["evaluation_diagnostic_directory"] / stratum / f"{label}-s{seed}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def registered_seeds(protocol: dict) -> set[int]:
    values: set[int] = set()
    for case in protocol["training"]["seeds_by_replica"].values():
        values.update(int(value) for value in case.values())
    for spec in protocol["evaluation"]["strata"].values():
        for case in spec["cases"]:
            values.update(int(value) for value in case.values())
    return values


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = registered_seeds(protocol)
    excluded_roots = {
        (ROOT / protocol[key]).resolve() for key in (
            "checkpoint_directory", "training_manifest_directory",
            "evaluation_manifest_directory", "evaluation_diagnostic_directory",
        )
    }
    excluded_files = {protocol_path.resolve(), report_path(protocol).resolve()}
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
        raise ValueError(f"registered s140000 seeds were already used: {conflicts}")


def _synthetic_state() -> dict:
    field = -np.ones((17, 17), dtype=int)
    field[1:16, 1:16] = 0
    field[6, 6] = 1
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("me", 0, True, (1, 1)),
        "others": [("other", 0, True, (14, 14))],
        "bombs": [((10, 10), 3)], "coins": [(15, 15)],
        "explosion_map": np.zeros_like(field), "user_input": None,
    }


def preflight(protocol: dict) -> dict:
    parent_path = ROOT / protocol["frozen_v4"]["path"]
    parent = _torch_load(parent_path)
    state = _synthetic_state()
    local_map = resource_features(state, "local7")
    full_map = resource_features(state, "full33")
    no_opponent = dict(state, others=[])
    no_bomb = dict(state, bombs=[])
    neutral_contract = bool(
        np.array_equal(full_map, resource_features(no_opponent, "full33"))
        and np.array_equal(full_map, resource_features(no_bomb, "full33"))
    )
    models = {}
    for arm in ARMS:
        torch.manual_seed(140000)
        model = GlobalResourceDQN(protocol["learning_contract"]["delta_cap"])
        model.base.load_state_dict(parent["online_net"], strict=True)
        model.freeze_base()
        models[arm] = model
    local = torch.randn(8, 4, 7, 7)
    glob = torch.randn(8, 7)
    resources = torch.randn(8, 2, 33, 33)
    with torch.no_grad():
        parent_model = models["local7"].base
        base_q = parent_model(local, glob)
        local_q, local_delta = models["local7"](local, glob, resources, True)
        full_q, full_delta = models["full33"](local, glob, resources, True)
    base_exact = all(
        torch.equal(models["local7"].base.state_dict()[key], value)
        for key, value in parent["online_net"].items()
    )
    result = {
        "resource_shape_equal": local_map.shape == full_map.shape == (2, 33, 33),
        "far_coin_hidden_local7": float(local_map[1].sum()) == 0.0,
        "far_coin_visible_full33": float(full_map[1].sum()) == 1.0,
        "opponents_and_bombs_absent_from_resource_input": neutral_contract,
        "zero_delta_local7": bool(torch.equal(local_delta, torch.zeros_like(local_delta))),
        "zero_delta_full33": bool(torch.equal(full_delta, torch.zeros_like(full_delta))),
        "initial_q_exactly_equals_frozen_v4": bool(
            torch.equal(base_q, local_q) and torch.equal(base_q, full_q)
        ),
        "base_tensors_exactly_loaded": base_exact,
        "base_parameters_frozen": all(not parameter.requires_grad for parameter in models["local7"].base.parameters()),
    }
    result["passed"] = all(result.values())
    return result


def validate_checkpoint(protocol: dict, protocol_hash: str, arm: str, replica: str, round_number: int = 400) -> dict:
    path = milestone_path(protocol, arm, replica, round_number)
    if not path.is_file():
        raise RuntimeError(f"global-resource checkpoint missing: {path}")
    payload = _torch_load(path)
    expected = {
        "architecture": architecture_name(), "protocol_sha256": protocol_hash,
        "parent_sha256": protocol["frozen_v4"]["sha256"], "arm": arm,
        "replica": replica, "completed_rounds": round_number,
        "single_training_variable": protocol["single_training_variable"],
        "epsilon": 0.10,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"global-resource checkpoint {key} mismatch: {path}")
    parent = _torch_load(ROOT / protocol["frozen_v4"]["path"])["online_net"]
    online = payload.get("online_net", {})
    if set(f"base.{key}" for key in parent) - set(online):
        raise RuntimeError(f"global-resource checkpoint base schema mismatch: {path}")
    if any(not torch.equal(online[f"base.{key}"], value) for key, value in parent.items()):
        raise RuntimeError(f"frozen-v4 tensors changed: {path}")
    diagnostics = payload.get("training_diagnostics", {})
    if (
        diagnostics.get("raw_transitions") != payload.get("training_steps")
        or diagnostics.get("gradient_updates") != payload.get("gradient_steps")
        or len(diagnostics.get("per_round", ())) != round_number
        or sum(item["transitions"] for item in diagnostics.get("per_round", ())) != diagnostics.get("raw_transitions")
        or sum(item["optimizer_updates"] for item in diagnostics.get("per_round", ())) != diagnostics.get("gradient_updates")
    ):
        raise RuntimeError(f"global-resource training accounting mismatch: {path}")
    return payload


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, arm: str, replica: str) -> dict:
    manifest = training_manifest_path(protocol, arm, replica)
    existing = load_completed(manifest, TRAINING_KIND)
    if existing is not None:
        validate_checkpoint(protocol, protocol_hash, arm, replica)
        if existing["checkpoint"]["sha256"] != sha256_file(checkpoint_path(protocol, arm, replica)):
            raise RuntimeError(f"completed global-resource checkpoint drift: {arm}/{replica}")
        return existing
    endpoint = checkpoint_path(protocol, arm, replica)
    stats = manifest.with_suffix(".stats.json")
    if endpoint.parent.exists() or manifest.exists() or stats.exists():
        raise RuntimeError(f"refusing orphaned global-resource artifacts: {arm}/{replica}")
    case = protocol["training"]["seeds_by_replica"][replica]
    command = [
        sys.executable, "main.py", "play", "--agents", TARGET_AGENT,
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", "classic", "--n-rounds", str(ENDPOINT_ROUND),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = {
        "MODEL_A_GLOBAL_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_GLOBAL_RUN_MODE": "train", "MODEL_A_GLOBAL_ARM": arm,
        "MODEL_A_GLOBAL_REPLICA": replica, "MODEL_A_GLOBAL_SEED": str(case["agent_seed"]),
        "MODEL_A_GLOBAL_CHECKPOINT_PATH": str(endpoint),
    }
    record = {
        "schema_version": 1, "kind": TRAINING_KIND, "status": "running",
        "started_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "arm": arm, "replica": replica,
        "training": {**case, "scenario": "classic", "opponents": [], "rounds": ENDPOINT_ROUND},
        "command": command, "environment_overrides": overrides,
        "checkpoint": {"path": relative(endpoint), "sha256": None},
        "raw_stats": relative(stats),
    }
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now(); record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats.is_file():
            raise RuntimeError("global-resource training process or stats failed")
        payload = validate_checkpoint(protocol, protocol_hash, arm, replica)
        for round_number in (100, 200, 300, 400):
            validate_checkpoint(protocol, protocol_hash, arm, replica, round_number)
        record["checkpoint"]["sha256"] = sha256_file(endpoint)
        record["training_diagnostics"] = payload["training_diagnostics"]
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats.read_text(encoding="utf-8")))
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"; record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"global-resource training failed: {arm}/{replica}: {record.get('error')}")
    return record


def label_parts(label: str) -> tuple[str, str]:
    if label == "frozen-v4":
        return label, "baseline"
    arm, replica = label.rsplit("-", 1)
    if arm not in ARMS or replica not in REPLICAS:
        raise ValueError(f"unknown global-resource label: {label}")
    return arm, replica


def validate_diagnostic(path: Path, protocol_hash: str, arm: str, replica: str, rounds: int) -> dict:
    if not path.is_file():
        raise RuntimeError(f"global-resource evaluation diagnostic missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "kind": "model-a-v4-global-resource-stage1-evaluation-diagnostic",
        "protocol_sha256": protocol_hash, "arm": arm, "replica": replica,
        "policy_updates": 0,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"global-resource evaluation diagnostic mismatch: {path}")
    entries = payload.get("rounds", [])
    if len(entries) != rounds or any(sum(item["actions"].values()) != item["transitions"] for item in entries):
        raise RuntimeError(f"global-resource evaluation diagnostic accounting mismatch: {path}")
    return payload


def evaluate_one(protocol_path: Path, protocol: dict, protocol_hash: str, stratum: str, label: str, case: dict) -> dict:
    arm, replica = label_parts(label)
    manifest = evaluation_manifest_path(protocol, stratum, label, int(case["world_seed"]))
    existing = load_completed(manifest, EVALUATION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed global-resource evaluation drift: {manifest}")
        return existing
    stats = manifest.with_suffix(".stats.json")
    diagnostic = diagnostic_path(protocol, stratum, label, int(case["world_seed"]))
    if manifest.exists() or stats.exists() or diagnostic.exists():
        raise RuntimeError(f"refusing orphaned global-resource evaluation: {manifest}")
    spec = protocol["evaluation"]["strata"][stratum]
    command = [
        sys.executable, "main.py", "play", "--agents", TARGET_AGENT,
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = {
        "MODEL_A_GLOBAL_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_GLOBAL_RUN_MODE": "evaluate", "MODEL_A_GLOBAL_ARM": arm,
        "MODEL_A_GLOBAL_REPLICA": replica, "MODEL_A_GLOBAL_SEED": str(case["agent_seed"]),
        "MODEL_A_GLOBAL_DIAGNOSTIC_PATH": str(diagnostic),
    }
    weight = None
    if arm in ARMS:
        weight = checkpoint_path(protocol, arm, replica)
        overrides["MODEL_A_GLOBAL_CHECKPOINT_PATH"] = str(weight)
    record = {
        "schema_version": 1, "kind": EVALUATION_KIND, "status": "running",
        "started_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "stratum": stratum, "label": label, "arm": arm, "replica": replica,
        "evaluation": {**case, "scenario": spec["scenario"], "opponents": [], "rounds": spec["rounds_per_case"]},
        "checkpoint": None if weight is None else {"path": relative(weight), "sha256": sha256_file(weight)},
        "command": command, "environment_overrides": overrides,
        "raw_stats": relative(stats), "diagnostic": {"path": relative(diagnostic), "sha256": None},
        "policy_updates": 0,
    }
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now(); record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats.is_file():
            raise RuntimeError("global-resource evaluation process or stats failed")
        diag = validate_diagnostic(diagnostic, protocol_hash, arm, replica, int(spec["rounds_per_case"]))
        by_agent = metrics_by_agent(json.loads(stats.read_text(encoding="utf-8")))
        if TARGET_AGENT not in by_agent:
            raise RuntimeError("global-resource target missing from evaluation stats")
        target = dict(by_agent[TARGET_AGENT])
        target["episode_length"] = target["steps"] / target["rounds"]
        target["crates_per_bomb"] = target["crates"] / target["bombs"] if target["bombs"] else 0.0
        record["metrics_by_agent"] = by_agent; record["target_metrics"] = target
        record["first_coin"] = diag["first_coin"]
        record["diagnostic"]["sha256"] = sha256_file(diagnostic)
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"; record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"global-resource evaluation failed: {label}/{stratum}: {record.get('error')}")
    return record


def _aggregate_runs(runs: list[dict]) -> dict:
    result = aggregate(runs)
    result["episode_length"] = result["steps"] / result["rounds"]
    result["crates_per_bomb"] = result["crates"] / result["bombs"] if result["bombs"] else 0.0
    latencies = []
    missing = 0
    for run in runs:
        payload = json.loads((ROOT / run["diagnostic"]["path"]).read_text(encoding="utf-8"))
        for item in payload["rounds"]:
            if item["first_coin_step"] is None:
                missing += 1
            else:
                latencies.append(int(item["first_coin_step"]))
    result["first_coin_observed_rounds"] = len(latencies)
    result["first_coin_missing_rounds"] = missing
    result["first_coin_mean_step_when_observed"] = float(np.mean(latencies)) if latencies else None
    return result


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else (float("inf") if numerator > 0 else 1.0)


def decide(protocol: dict, runs: list[dict], parameter_effect: dict) -> tuple[str, dict]:
    by_label_stratum = {}
    for stratum in ("task1", "task2"):
        for label in protocol["evaluation"]["labels"]:
            selected = [run for run in runs if run["stratum"] == stratum and run["label"] == label]
            by_label_stratum[(label, stratum)] = _aggregate_runs(selected)
    arm_pooled = {}
    for arm in ARMS:
        for stratum in ("task1", "task2"):
            labels = [f"{arm}-{replica}" for replica in REPLICAS]
            selected = [run for run in runs if run["stratum"] == stratum and run["label"] in labels]
            arm_pooled[(arm, stratum)] = _aggregate_runs(selected)
    local_t2, full_t2 = arm_pooled[("local7", "task2")], arm_pooled[("full33", "task2")]
    local_t1, full_t1 = arm_pooled[("local7", "task1")], arm_pooled[("full33", "task1")]
    frozen_t2 = by_label_stratum[("frozen-v4", "task2")]
    pairs = {}
    for replica in REPLICAS:
        local = by_label_stratum[(f"local7-{replica}", "task2")]
        full = by_label_stratum[(f"full33-{replica}", "task2")]
        pairs[replica] = {
            "full33_minus_local7_score_per_round": full["score_per_round"] - local["score_per_round"],
            "positive": full["score_per_round"] > local["score_per_round"],
        }
    positive_pairs = sum(item["positive"] for item in pairs.values())
    gate = protocol["signal_gate"]
    checks = {
        "parameter_effect": parameter_effect["evaluation_authorized"],
        "positive_replica_pairs": positive_pairs >= gate["minimum_positive_full33_vs_local7_task2_replica_pairs"],
        "task2_score_vs_local7": _ratio(full_t2["score_per_round"], local_t2["score_per_round"]) >= gate["minimum_pooled_task2_score_ratio_full33_vs_local7"],
        "task2_coins_vs_local7": _ratio(full_t2["coins_per_round"], local_t2["coins_per_round"]) >= gate["minimum_pooled_task2_coin_ratio_full33_vs_local7"],
        "task2_wait_lower": full_t2["wait_fraction"] < local_t2["wait_fraction"],
        "task2_score_vs_frozen_v4": _ratio(full_t2["score_per_round"], frozen_t2["score_per_round"]) >= gate["minimum_pooled_task2_score_ratio_full33_vs_frozen_v4"],
        "task1_retention": _ratio(full_t1["score_per_round"], local_t1["score_per_round"]) >= gate["minimum_task1_score_ratio_full33_vs_local7"],
        "task2_suicide_guard": full_t2["suicides_per_round"] - local_t2["suicides_per_round"] <= gate["maximum_task2_suicide_increase_per_round_vs_local7"],
        "task2_invalid_guard": full_t2["invalid_actions_per_round"] - local_t2["invalid_actions_per_round"] <= gate["maximum_task2_invalid_increase_per_round_vs_local7"],
    }
    decision = (
        "fullboard_resource_visibility_supported_stop_before_confirmation"
        if all(checks.values()) else "fullboard_resource_visibility_not_supported_stop"
    )
    details = {
        "checks": checks, "positive_replica_pairs": positive_pairs,
        "minimum_positive_replica_pairs": gate["minimum_positive_full33_vs_local7_task2_replica_pairs"],
        "task2_replica_pairs": pairs,
        "by_label_and_stratum": {f"{label}/{stratum}": value for (label, stratum), value in by_label_stratum.items()},
        "pooled_by_arm_and_stratum": {f"{arm}/{stratum}": value for (arm, stratum), value in arm_pooled.items()},
    }
    return decision, details


def parameter_effect(protocol: dict) -> dict:
    pairs = {}
    for replica in REPLICAS:
        local = _torch_load(checkpoint_path(protocol, "local7", replica))["online_net"]
        full = _torch_load(checkpoint_path(protocol, "full33", replica))["online_net"]
        keys = [key for key in local if key.startswith("resource_")]
        diverged = set(local) == set(full) and any(not torch.equal(local[key], full[key]) for key in keys)
        nonzero = {
            arm: any(bool((tensor != 0).any()) for key, tensor in values.items() if key.startswith("resource_head."))
            for arm, values in (("local7", local), ("full33", full))
        }
        pairs[replica] = {"endpoint_diverged": diverged, "resource_head_nonzero": nonzero}
    count = sum(item["endpoint_diverged"] and all(item["resource_head_nonzero"].values()) for item in pairs.values())
    return {"pairs": pairs, "effective_pairs": count, "minimum_effective_pairs": 2, "evaluation_authorized": count >= 2}


def execute(protocol_path: Path, protocol: dict, protocol_hash: str, preflight_result: dict) -> dict:
    if not preflight_result["passed"]:
        raise RuntimeError("global-resource preflight failed")
    training = []
    for replica in REPLICAS:
        for arm in ARMS:
            training.append(train_one(protocol_path, protocol, protocol_hash, arm, replica))
    effect = parameter_effect(protocol)
    evaluations = []
    decision_details = None
    if effect["evaluation_authorized"]:
        for stratum, spec in protocol["evaluation"]["strata"].items():
            for label in protocol["evaluation"]["labels"]:
                for case in spec["cases"]:
                    evaluations.append(evaluate_one(protocol_path, protocol, protocol_hash, stratum, label, case))
        decision, decision_details = decide(protocol, evaluations, effect)
    else:
        decision = "global_resource_no_parameter_effect_stop"
    report = {
        "schema_version": 1, "kind": REPORT_KIND, "status": "completed",
        "completed_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "preflight": preflight_result, "parameter_effect": effect,
        "training_manifests": [{"path": relative(training_manifest_path(protocol, run["arm"], run["replica"])), "sha256": sha256_file(training_manifest_path(protocol, run["arm"], run["replica"]))} for run in training],
        "evaluation_manifests": [{"path": relative(evaluation_manifest_path(protocol, run["stratum"], run["label"], run["evaluation"]["world_seed"])), "sha256": sha256_file(evaluation_manifest_path(protocol, run["stratum"], run["label"], run["evaluation"]["world_seed"]))} for run in evaluations],
        "decision": decision, "decision_details": decision_details,
        "training_started": True, "evaluation_started": bool(evaluations),
        "intermediate_checkpoint_selection_used": False,
        "checkpoint_selected": False, "checkpoint_copied": False,
        "automatic_stage2_started": False, "awaiting_user_instruction": True,
    }
    atomic_json(report_path(protocol), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol, protocol_path, protocol_hash = load_protocol(args.protocol)
    assert_registered_seeds_untouched(protocol_path, protocol)
    preflight_result = preflight(protocol)
    dry = {
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
        "single_training_variable": protocol["single_training_variable"],
        "arms": ["frozen-v4", *ARMS], "replicas": list(REPLICAS),
        "training_rounds": 2 * len(REPLICAS) * ENDPOINT_ROUND,
        "evaluation_rounds_if_authorized": sum(
            len(protocol["evaluation"]["labels"]) * len(spec["cases"]) * spec["rounds_per_case"]
            for spec in protocol["evaluation"]["strata"].values()
        ),
        "candidate_endpoint": 400, "intermediate_selection_allowed": False,
        "new_attack_reward_added": False, "kills_used_for_gate": False,
        "automatic_stage2_started": False, "formal_execution_started": False,
        "preflight": preflight_result,
    }
    if not args.execute:
        print(json.dumps(dry, indent=2, sort_keys=True))
        return 0
    terminal = report_path(protocol)
    if terminal.exists():
        raise SystemExit(f"terminal report already exists: {terminal}")
    report = execute(protocol_path, protocol, protocol_hash, preflight_result)
    print(json.dumps({"status": report["status"], "decision": report["decision"], "report": relative(terminal)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
