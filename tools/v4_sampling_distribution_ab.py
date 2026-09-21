"""Run the fixed-endpoint all-action n=8 replay-sampling-only matched A/B."""

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

from agent_code.model_a_v4_sampling_ab.callbacks import _torch_load  # noqa: E402
from agent_code.model_a_v4_sampling_ab.config import (  # noqa: E402
    ARMS, ENDPOINT_ROUND, EVALUATION_STRATA, REPLICAS, checkpoint_path,
    evaluation_diagnostic_path, load_protocol, sha256_file,
)
from tools.v4_task4_duel_training import (  # noqa: E402
    aggregate, atomic_json, load_completed, metrics_by_agent, opponent_summary,
    relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-sampling-distribution-ab-s136000.json"
TRAINING_KIND = "model-a-v4-sampling-distribution-ab-training"
EVALUATION_KIND = "model-a-v4-sampling-distribution-ab-evaluation"
REPORT_KIND = "model-a-v4-sampling-distribution-ab-report"
ATTACK_KEYS = (
    "rounds", "rows", "approach_steps", "bomb_legal", "bombs", "threat_bombs",
    "trap_bombs", "kill_events", "self_kill_events", "got_killed_events",
    "survived_rounds", "linked_kill_events", "linked_threat_kill_events",
    "unlinked_or_ambiguous_kill_events", "oracle_evaluated", "oracle_timeouts",
    "successful_threat_bombs",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def training_manifest_path(protocol: dict, arm: str, replica: str) -> Path:
    return ROOT / protocol["training_manifest_directory"] / arm / f"{replica}.json"


def evaluation_manifest_path(protocol: dict, stratum: str, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / stratum / f"{label}-s{world_seed}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def registered_seeds(protocol: dict) -> set[int]:
    values = set()
    for case in protocol["training"]["seeds_by_replica"].values():
        values.update(int(value) for value in case.values())
    for spec in protocol["evaluation"]["strata"].values():
        for case in spec["cases"]:
            values.update(int(value) for value in case.values())
    return values


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = registered_seeds(protocol)
    excluded_roots = {
        (ROOT / protocol["checkpoint_directory"]).resolve(),
        (ROOT / protocol["training_manifest_directory"]).resolve(),
        (ROOT / protocol["evaluation_manifest_directory"]).resolve(),
        (ROOT / protocol["evaluation_diagnostic_directory"]).resolve(),
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
        raise ValueError(f"registered s136000 seeds were already used: {conflicts}")


def validate_checkpoint(protocol: dict, protocol_hash: str, arm: str, replica: str) -> dict:
    path = checkpoint_path(protocol, arm, replica)
    if not path.is_file():
        raise RuntimeError(f"sampling-distribution endpoint missing: {path}")
    payload = _torch_load(path)
    expected = {
        "architecture": "model-a-mlp-v4-global7", "protocol_sha256": protocol_hash,
        "arm": arm, "replica": replica, "stage_id": "task4c_sampling_ab",
        "stage_completed_rounds": ENDPOINT_ROUND,
        "parent_sha256": protocol["source_parent"]["sha256"],
        "single_training_variable": protocol["single_training_variable"],
        "all_action_return_horizon": 8, "sampling_mode": arm,
        "reserved_slots": {
            "kill_chain": 0 if arm == "uniform_n8" else 2,
            "self_kill_chain": 0 if arm == "uniform_n8" else 2,
        },
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"sampling-distribution endpoint {key} mismatch: {path}")
    if abs(float(payload.get("epsilon", -1)) - 0.05) > 1e-12:
        raise RuntimeError(f"sampling-distribution epsilon changed: {path}")
    d = payload.get("training_diagnostics", {})
    required = {
        "raw_transitions", "matured_targets", "return_steps_histogram",
        "kill_chain_targets_created", "self_chain_targets_created", "observed_kill_events",
        "observed_self_events", "survivor_terminal_merges", "dead_terminal_appends",
        "gradient_updates", "sampled_batches", "requested_kill_slots", "fulfilled_kill_slots",
        "requested_self_slots", "fulfilled_self_slots", "fallback_uniform_slots",
        "realized_kill_chain_samples", "realized_self_chain_samples", "replay_size",
        "replay_kill_items", "replay_self_items",
    }
    if set(d) != required:
        raise RuntimeError(f"sampling-distribution diagnostics changed: {path}")
    if d["raw_transitions"] != d["matured_targets"] or sum(d["return_steps_histogram"].values()) != d["matured_targets"]:
        raise RuntimeError(f"sampling-distribution target accounting mismatch: {path}")
    if d["survivor_terminal_merges"] + d["dead_terminal_appends"] != ENDPOINT_ROUND:
        raise RuntimeError(f"sampling-distribution terminal accounting mismatch: {path}")
    if d["gradient_updates"] != d["sampled_batches"]:
        raise RuntimeError(f"sampling-distribution batch accounting mismatch: {path}")
    if arm == "uniform_n8":
        if any(d[key] for key in ("requested_kill_slots", "fulfilled_kill_slots", "requested_self_slots", "fulfilled_self_slots", "fallback_uniform_slots")):
            raise RuntimeError(f"uniform arm used reserved sampling: {path}")
    else:
        if d["requested_kill_slots"] != 2 * d["sampled_batches"] or d["requested_self_slots"] != 2 * d["sampled_batches"]:
            raise RuntimeError(f"stratified arm reserved-slot accounting mismatch: {path}")
        if d["fulfilled_kill_slots"] + d["fulfilled_self_slots"] + d["fallback_uniform_slots"] != 4 * d["sampled_batches"]:
            raise RuntimeError(f"stratified arm fallback accounting mismatch: {path}")
    return payload


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, arm: str, replica: str) -> dict:
    manifest = training_manifest_path(protocol, arm, replica)
    existing = load_completed(manifest, TRAINING_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed sampling-distribution training drift: {arm}/{replica}")
        payload = validate_checkpoint(protocol, protocol_hash, arm, replica)
        if existing["checkpoint"]["sha256"] != sha256_file(checkpoint_path(protocol, arm, replica)):
            raise RuntimeError(f"completed sampling-distribution checkpoint drift: {arm}/{replica}")
        return existing
    output = checkpoint_path(protocol, arm, replica)
    stats_path = manifest.with_suffix(".stats.json")
    if output.parent.exists() or stats_path.exists() or manifest.exists():
        raise RuntimeError(f"refusing orphaned sampling-distribution artifacts: {arm}/{replica}")
    spec = protocol["training"]
    seeds = spec["seeds_by_replica"][replica]
    parent = (ROOT / protocol["source_parent"]["path"]).resolve()
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_sampling_ab", *spec["opponents"],
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_arm"]),
        "--seed", str(seeds["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides = {
        "MODEL_A_SAMPLING_PROTOCOL_PATH": str(protocol_path), "MODEL_A_SAMPLING_RUN_MODE": "train",
        "MODEL_A_SAMPLING_ARM": arm, "MODEL_A_SAMPLING_REPLICA": replica,
        "MODEL_A_SAMPLING_SEED": str(seeds["agent_seed"]),
        "MODEL_A_SAMPLING_PARENT_PATH": str(parent), "MODEL_A_SAMPLING_CHECKPOINT_PATH": str(output),
        "TASK4_RULE_SEED": str(seeds["opponent_seed"]),
    }
    record = {
        "schema_version": 1, "kind": TRAINING_KIND, "status": "running",
        "started_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "arm": arm, "replica": replica,
        "training": {**seeds, "scenario": spec["scenario"], "opponents": spec["opponents"],
                     "rounds": spec["rounds_per_arm"]},
        "target_contract": {"all_action_return_horizon": 8, "gamma": 0.99},
        "sampling_contract": protocol["learning_contract"]["uniform_arm_batch" if arm == "uniform_n8" else "stratified_arm_batch"],
        "parent_checkpoint": {"path": relative(parent), "sha256": sha256_file(parent)},
        "checkpoint": {"path": relative(output), "sha256": None},
        "command": command, "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now(); record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats_path.is_file():
            raise RuntimeError("sampling-distribution training process or stats failed")
        payload = validate_checkpoint(protocol, protocol_hash, arm, replica)
        if sha256_file(parent) != protocol["source_parent"]["sha256"]:
            raise RuntimeError("source-r2 changed during sampling-distribution training")
        record["checkpoint"]["sha256"] = sha256_file(output)
        record["training_diagnostics"] = payload["training_diagnostics"]
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats_path.read_text(encoding="utf-8")))
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"; record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"sampling-distribution training failed: {arm}/{replica}: {record.get('error')}")
    return record


def parameter_effect(protocol: dict) -> dict:
    pairs = {}
    for replica in REPLICAS:
        left = _torch_load(checkpoint_path(protocol, "uniform_n8", replica))["online_net"]
        right = _torch_load(checkpoint_path(protocol, "stratified_n8", replica))["online_net"]
        equal = set(left) == set(right) and all(bool((left[key] == right[key]).all()) for key in left)
        pairs[replica] = {"endpoint_diverged": not equal}
    count = sum(item["endpoint_diverged"] for item in pairs.values())
    minimum = int(protocol["parameter_effect_gate"]["minimum_divergent_endpoint_pairs"])
    return {
        "pairs": pairs, "endpoint_divergent_pairs": count,
        "minimum_divergent_endpoint_pairs": minimum,
        "evaluation_authorized": count >= minimum,
    }


def checkpoint_for_label(protocol: dict, label: str) -> Path:
    if label == "source-r2":
        return (ROOT / protocol["source_parent"]["path"]).resolve()
    if label == "v4":
        return (ROOT / protocol["frozen_v4_baseline"]["path"]).resolve()
    arm, replica = label.rsplit("-", 1)
    return checkpoint_path(protocol, arm, replica)


def validate_attack_diagnostic(protocol: dict, protocol_hash: str, stratum: str, label: str, case: dict) -> dict:
    path = evaluation_diagnostic_path(protocol, stratum, label, int(case["world_seed"]))
    if not path.is_file():
        raise RuntimeError(f"sampling-distribution attack diagnostic missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1, "kind": "model-a-v4-sampling-ab-evaluation-diagnostic",
        "protocol_sha256": protocol_hash, "label": label, "stratum": stratum,
        "world_seed": int(case["world_seed"]), "agent_seed": int(case["agent_seed"]),
        "policy_updates": 0, "actions_overridden": 0,
    }
    if {key: payload.get(key) for key in expected} != expected:
        raise RuntimeError(f"sampling-distribution attack diagnostic metadata mismatch: {path}")
    d = payload.get("diagnostics", {})
    expected_keys = set(ATTACK_KEYS) | {"threat_to_kill_conversion", "linked_kills_per_threat"}
    if set(d) != expected_keys or d["rounds"] != int(protocol["evaluation"]["strata"][stratum]["rounds_per_case"]):
        raise RuntimeError(f"sampling-distribution attack diagnostic schema mismatch: {path}")
    if d["survived_rounds"] + d["got_killed_events"] != d["rounds"]:
        raise RuntimeError(f"sampling-distribution terminal coverage mismatch: {path}")
    if d["linked_kill_events"] + d["unlinked_or_ambiguous_kill_events"] != d["kill_events"]:
        raise RuntimeError(f"sampling-distribution kill linkage mismatch: {path}")
    expected_rate = d["successful_threat_bombs"] / d["threat_bombs"] if d["threat_bombs"] else 0.0
    if abs(d["threat_to_kill_conversion"] - expected_rate) > 1e-12:
        raise RuntimeError(f"sampling-distribution threat conversion mismatch: {path}")
    return {"path": relative(path), "sha256": sha256_file(path), "diagnostics": d}


def evaluate_one(protocol_path: Path, protocol: dict, protocol_hash: str, stratum: str, label: str, case: dict) -> dict:
    spec = protocol["evaluation"]["strata"][stratum]
    checkpoint = checkpoint_for_label(protocol, label)
    checkpoint_hash = sha256_file(checkpoint)
    output = evaluation_manifest_path(protocol, stratum, label, int(case["world_seed"]))
    diagnostic = evaluation_diagnostic_path(protocol, stratum, label, int(case["world_seed"]))
    expected = {**case, "scenario": spec["scenario"], "opponents": spec["opponents"],
                "rounds_per_case": spec["rounds_per_case"], "target_agent": "model_a_v4_sampling_ab"}
    existing = load_completed(output, EVALUATION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or existing.get("evaluation") != expected or existing["checkpoint"]["sha256"] != checkpoint_hash:
            raise RuntimeError(f"completed sampling-distribution evaluation drift: {output}")
        attack = validate_attack_diagnostic(protocol, protocol_hash, stratum, label, case)
        if existing["attack_diagnostic"]["sha256"] != attack["sha256"]:
            raise RuntimeError(f"completed sampling-distribution attack diagnostic drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if output.exists() or stats_path.exists() or diagnostic.exists():
        raise RuntimeError(f"refusing orphaned sampling-distribution evaluation: {output}")
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_sampling_ab", *spec["opponents"],
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides = {
        "MODEL_A_SAMPLING_PROTOCOL_PATH": str(protocol_path), "MODEL_A_SAMPLING_RUN_MODE": "evaluate",
        "MODEL_A_SAMPLING_EVAL_LABEL": label, "MODEL_A_SAMPLING_EVAL_STRATUM": stratum,
        "MODEL_A_SAMPLING_EVAL_WORLD_SEED": str(case["world_seed"]),
        "MODEL_A_SAMPLING_EVAL_DIAGNOSTIC_PATH": str(diagnostic),
        "MODEL_A_SAMPLING_SEED": str(case["agent_seed"]),
        "MODEL_A_SAMPLING_CHECKPOINT_PATH": str(checkpoint),
    }
    if spec["opponents"]:
        key = "TASK4_RULE_SEED" if "seeded_rule_based_agent" in spec["opponents"] else "TASK3_OPPONENT_SEED"
        overrides[key] = str(case["opponent_seed"])
    record = {
        "schema_version": 1, "kind": EVALUATION_KIND, "status": "running",
        "started_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "stratum": stratum, "label": label, "evaluation": expected,
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
        "command": command, "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(output, record)
    child = os.environ.copy(); child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now(); record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats_path.is_file() or sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("sampling-distribution evaluation process, stats, or checkpoint failed")
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats_path.read_text(encoding="utf-8")))
        record["target_metrics"] = record["metrics_by_agent"].get("model_a_v4_sampling_ab")
        if not record["target_metrics"]:
            raise RuntimeError("sampling-distribution target metrics missing")
        record["attack_diagnostic"] = validate_attack_diagnostic(protocol, protocol_hash, stratum, label, case)
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"; record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"sampling-distribution evaluation failed: {stratum}/{label}: {record.get('error')}")
    return record


def aggregate_attack(runs_or_values: list[dict]) -> dict:
    totals = {key: 0 for key in ATTACK_KEYS}
    for item in runs_or_values:
        d = item.get("attack_diagnostic", {}).get("diagnostics", item)
        for key in ATTACK_KEYS:
            totals[key] += int(d[key])
    totals["threat_to_kill_conversion"] = totals["successful_threat_bombs"] / totals["threat_bombs"] if totals["threat_bombs"] else 0.0
    totals["linked_kills_per_threat"] = totals["linked_threat_kill_events"] / totals["threat_bombs"] if totals["threat_bombs"] else 0.0
    return totals


def run_evaluations(protocol_path: Path, protocol: dict, protocol_hash: str) -> tuple[dict, dict]:
    labels = ["v4", "source-r2", *[f"{arm}-{replica}" for replica in REPLICAS for arm in ARMS]]
    rows, manifests = {}, {}
    for label in labels:
        rows[label], manifests[label] = {}, {}
        for stratum in EVALUATION_STRATA:
            spec = protocol["evaluation"]["strata"][stratum]
            runs = [evaluate_one(protocol_path, protocol, protocol_hash, stratum, label, case) for case in spec["cases"]]
            rows[label][stratum] = {
                "target": aggregate(runs), "opponents": opponent_summary(runs, "model_a_v4_sampling_ab"),
                "attack": aggregate_attack(runs),
            }
            manifests[label][stratum] = [relative(evaluation_manifest_path(protocol, stratum, label, int(case["world_seed"]))) for case in spec["cases"]]
    return rows, manifests


def combine_metrics(values: list[dict]) -> dict:
    additive = ("rounds", "score", "coins", "kills", "crates", "bombs", "moves", "waits", "suicides", "invalid_actions", "steps")
    totals = {key: sum(int(value[key]) for value in values) for key in additive}
    rounds, steps = totals["rounds"], totals["steps"]
    return {**totals, "score_per_round": totals["score"] / rounds, "kills_per_round": totals["kills"] / rounds,
            "suicides_per_round": totals["suicides"] / rounds, "bombs_per_round": totals["bombs"] / rounds,
            "invalid_actions_per_round": totals["invalid_actions"] / rounds,
            "move_fraction": totals["moves"] / steps if steps else 0.0, "wait_fraction": totals["waits"] / steps if steps else 0.0}


def combined_task4(rows: dict, label: str) -> tuple[dict, dict]:
    strata = ("task4b_duel", "task4c_three_rule")
    return (
        combine_metrics([rows[label][name]["target"] for name in strata]),
        aggregate_attack([rows[label][name]["attack"] for name in strata]),
    )


def pooled_task4(rows: dict, labels: list[str]) -> tuple[dict, dict]:
    metrics, attacks = zip(*(combined_task4(rows, label) for label in labels))
    return combine_metrics(list(metrics)), aggregate_attack(list(attacks))


def decide(protocol: dict, rows: dict, training: dict, effect: dict) -> dict:
    rule = protocol["evaluation"]["decision_rule"]
    if not effect["evaluation_authorized"]:
        return {"passed": False, "decision": rule["no_effect_decision"], "parameter_effect": effect,
                "evaluation_started": False, "automatic_followup_started": False, "task4_complete": False}
    source_metrics, source_attack = combined_task4(rows, "source-r2")
    pairs = {}
    for replica in REPLICAS:
        control_metrics, control_attack = combined_task4(rows, f"uniform_n8-{replica}")
        candidate_metrics, candidate_attack = combined_task4(rows, f"stratified_n8-{replica}")
        checks = {
            "linked_kills_gain": candidate_attack["linked_kill_events"] >= control_attack["linked_kill_events"] + rule["pair_minimum_combined_linked_kills_gain"],
            "threat_to_kill_gain": candidate_attack["threat_to_kill_conversion"] >= control_attack["threat_to_kill_conversion"] + rule["pair_minimum_combined_threat_to_kill_gain"] - 1e-12,
        }
        pairs[replica] = {
            "supportive": all(checks.values()), "checks": checks,
            "control": {"metrics": control_metrics, "attack": control_attack},
            "candidate": {"metrics": candidate_metrics, "attack": candidate_attack},
        }
    control_labels = [f"uniform_n8-{replica}" for replica in REPLICAS]
    candidate_labels = [f"stratified_n8-{replica}" for replica in REPLICAS]
    control_metrics, control_attack = pooled_task4(rows, control_labels)
    candidate_metrics, candidate_attack = pooled_task4(rows, candidate_labels)
    checks = {
        "kills_gain_over_uniform": candidate_metrics["kills_per_round"] >= control_metrics["kills_per_round"] + rule["pooled_minimum_combined_kills_per_round_gain_over_uniform"] - 1e-12,
        "conversion_gain_over_uniform": candidate_attack["threat_to_kill_conversion"] >= control_attack["threat_to_kill_conversion"] + rule["pooled_minimum_combined_threat_to_kill_gain_over_uniform"] - 1e-12,
        "kills_gain_over_source": candidate_metrics["kills_per_round"] >= source_metrics["kills_per_round"] + rule["pooled_minimum_combined_kills_per_round_gain_over_source"] - 1e-12,
        "conversion_gain_over_source": candidate_attack["threat_to_kill_conversion"] >= source_attack["threat_to_kill_conversion"] + rule["pooled_minimum_combined_threat_to_kill_gain_over_source"] - 1e-12,
        "score_safe_vs_uniform": candidate_metrics["score_per_round"] >= control_metrics["score_per_round"] + rule["pooled_minimum_combined_score_delta_to_uniform"] - 1e-12,
        "suicide_safe_vs_uniform": candidate_metrics["suicides_per_round"] <= control_metrics["suicides_per_round"] + rule["pooled_maximum_combined_suicide_delta_to_uniform"] + 1e-12,
        "score_safe_vs_source": candidate_metrics["score_per_round"] >= source_metrics["score_per_round"] + rule["pooled_minimum_combined_score_delta_to_source"] - 1e-12,
        "suicide_safe_vs_source": candidate_metrics["suicides_per_round"] <= source_metrics["suicides_per_round"] + rule["pooled_maximum_combined_suicide_delta_to_source"] + 1e-12,
    }
    retention = {}
    for stratum in ("task1", "task2", "task3_peaceful", "task3_coin"):
        candidate = combine_metrics([rows[label][stratum]["target"] for label in candidate_labels])
        source = rows["source-r2"][stratum]["target"]
        if stratum == "task1":
            retention[stratum] = candidate["score_per_round"] >= rule["minimum_task1_fraction_of_source"] * source["score_per_round"] - 1e-12
        elif stratum == "task2":
            retention[stratum] = candidate["score_per_round"] >= source["score_per_round"] - rule["maximum_task2_score_drop_from_source"] - 1e-12
        else:
            retention[f"{stratum}_score"] = candidate["score_per_round"] >= source["score_per_round"] - rule["maximum_task3_score_drop_from_source"] - 1e-12
            retention[f"{stratum}_kills"] = candidate["kills_per_round"] >= source["kills_per_round"] - rule["maximum_task3_kills_drop_from_source"] - 1e-12
            retention[f"{stratum}_suicide"] = candidate["suicides_per_round"] <= source["suicides_per_round"] + rule["maximum_task3_suicide_increase_over_source"] + 1e-12
    task4_stratum_safety = {}
    for stratum in ("task4b_duel", "task4c_three_rule"):
        candidate = combine_metrics([rows[label][stratum]["target"] for label in candidate_labels])
        source = rows["source-r2"][stratum]["target"]
        task4_stratum_safety[f"{stratum}_kills"] = candidate["kills_per_round"] >= source["kills_per_round"] - rule["maximum_task4_stratum_kills_drop_from_source"] - 1e-12
        task4_stratum_safety[f"{stratum}_score"] = candidate["score_per_round"] >= source["score_per_round"] + rule["minimum_task4_stratum_score_delta_to_source"] - 1e-12
        task4_stratum_safety[f"{stratum}_suicide"] = candidate["suicides_per_round"] <= source["suicides_per_round"] + rule["maximum_task4_stratum_suicide_delta_to_source"] + 1e-12
    integrity = {}
    for arm in ARMS:
        for replica in REPLICAS:
            d = training[arm][replica]["training_diagnostics"]
            base = d["raw_transitions"] == d["matured_targets"] and d["gradient_updates"] == d["sampled_batches"]
            if arm == "stratified_n8":
                base = base and d["fulfilled_kill_slots"] > 0 and d["fulfilled_self_slots"] > 0
            integrity[f"{arm}-{replica}"] = base
    supportive = sum(pair["supportive"] for pair in pairs.values())
    passed = supportive >= rule["minimum_supportive_pairs"] and all(checks.values()) and all(retention.values()) and all(task4_stratum_safety.values()) and all(integrity.values())
    return {
        "passed": passed, "decision": rule["pass_decision"] if passed else rule["fail_decision"],
        "parameter_effect": effect, "evaluation_started": True, "supportive_pairs": supportive,
        "pair_results": pairs, "pooled_task4": {
            "uniform": {"metrics": control_metrics, "attack": control_attack},
            "stratified": {"metrics": candidate_metrics, "attack": candidate_attack},
            "source-r2": {"metrics": source_metrics, "attack": source_attack},
        },
        "pooled_checks": checks, "retention_checks": retention,
        "task4_stratum_safety": task4_stratum_safety, "training_integrity": integrity,
        "automatic_followup_started": False, "task4_complete": False,
    }


def write_report(protocol_path: Path, protocol: dict, protocol_hash: str, training: dict, rows: dict | None, manifests: dict | None, result: dict) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed sampling-distribution report drift")
        return existing, sha256_file(path)
    record = {
        "schema_version": 1, "kind": REPORT_KIND, "status": "completed", "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash, "single_training_variable": protocol["single_training_variable"],
        "training_manifests": {arm: {replica: {"path": relative(training_manifest_path(protocol, arm, replica)), "sha256": sha256_file(training_manifest_path(protocol, arm, replica))} for replica in REPLICAS} for arm in ARMS},
        "evaluation_rows": rows, "evaluation_manifests": manifests, "result": result,
        "awaiting_user_instruction": True, "source_checkpoint_modified": False,
        "default_checkpoint_modified": False, "automatic_followup_started": False,
    }
    atomic_json(path, record)
    return record, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    training_rounds = len(ARMS) * len(REPLICAS) * int(protocol["training"]["rounds_per_arm"])
    identities = 2 + len(ARMS) * len(REPLICAS)
    evaluation_rounds = identities * sum(len(spec["cases"]) * int(spec["rounds_per_case"]) for spec in protocol["evaluation"]["strata"].values())
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"], "parent": "source-r2",
        "single_variable": protocol["single_training_variable"], "arms": list(ARMS),
        "replicas": list(REPLICAS), "all_action_return_horizon_both_arms": 8,
        "uniform_batch": protocol["learning_contract"]["uniform_arm_batch"],
        "stratified_batch": protocol["learning_contract"]["stratified_arm_batch"],
        "fixed_endpoint_round": ENDPOINT_ROUND, "snapshot_search": False,
        "total_training_rounds": training_rounds, "conditional_evaluation_rounds": evaluation_rounds,
        "maximum_total_game_rounds": training_rounds + evaluation_rounds,
        "formal_training_started": False, "formal_evaluation_started": False,
        "new_reward_added": False, "automatic_followup_started": False,
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
        print(json.dumps(dry_run(protocol), indent=2, sort_keys=True)); return 0
    existing = load_completed(report_path(protocol), REPORT_KIND)
    if existing is not None:
        print(json.dumps({"status": existing["status"], "report": relative(report_path(protocol)),
                          "report_sha256": sha256_file(report_path(protocol)), **existing["result"]}, indent=2, sort_keys=True)); return 0
    source = ROOT / protocol["source_parent"]["path"]
    source_before = sha256_file(source)
    training = {arm: {replica: train_one(protocol_path.resolve(), protocol, protocol_hash, arm, replica) for replica in REPLICAS} for arm in ARMS}
    if sha256_file(source) != source_before:
        raise RuntimeError("source-r2 changed during sampling-distribution training")
    effect = parameter_effect(protocol)
    if effect["evaluation_authorized"]:
        rows, manifests = run_evaluations(protocol_path.resolve(), protocol, protocol_hash)
    else:
        rows = manifests = None
    result = decide(protocol, rows, training, effect)
    report, digest = write_report(protocol_path.resolve(), protocol, protocol_hash, training, rows, manifests, result)
    print(json.dumps({"status": report["status"], "report": relative(report_path(protocol)), "report_sha256": digest, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
