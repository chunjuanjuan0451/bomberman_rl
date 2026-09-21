"""Run the fixed 25% source-task rehearsal pilot and conditional confirmation."""

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

from agent_code.model_a_v4_rehearsal.callbacks import _torch_load  # noqa: E402
from agent_code.model_a_v4_rehearsal.config import (  # noqa: E402
    ARMS, ENDPOINT_ROUND, EVAL_STRATA, OLD_STRATA, REPLICAS, checkpoint_path,
    dataset_path, evaluation_diagnostic_path, load_protocol, sha256_file,
)
from agent_code.model_a_v4_rehearsal.replay import load_dataset  # noqa: E402
from tools.v4_task4_duel_training import (  # noqa: E402
    aggregate, atomic_json, load_completed, metrics_by_agent, opponent_summary,
    relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-rehearsal-pilot-s137000.json"
COLLECTION_KIND = "model-a-v4-rehearsal-collection"
TRAINING_KIND = "model-a-v4-rehearsal-training"
EVALUATION_KIND = "model-a-v4-rehearsal-evaluation"
REPORT_KIND = "model-a-v4-rehearsal-report"
ATTACK_KEYS = (
    "rounds", "rows", "approach_steps", "bomb_legal", "bombs", "threat_bombs",
    "trap_bombs", "kill_events", "self_kill_events", "got_killed_events",
    "survived_rounds", "linked_kill_events", "linked_threat_kill_events",
    "unlinked_or_ambiguous_kill_events", "oracle_evaluated", "oracle_timeouts",
    "successful_threat_bombs",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collection_manifest_path(protocol: dict, replica: str, stratum: str) -> Path:
    return ROOT / protocol["collection_manifest_directory"] / replica / f"{stratum}.json"


def training_manifest_path(protocol: dict, arm: str, replica: str) -> Path:
    return ROOT / protocol["training_manifest_directory"] / arm / f"{replica}.json"


def evaluation_manifest_path(protocol: dict, suite: str, stratum: str, label: str, seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / suite / stratum / f"{label}-s{seed}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def registered_seeds(protocol: dict) -> set[int]:
    values = set()
    for spec in protocol["rehearsal_collection"]["strata"].values():
        for case in spec["seeds_by_replica"].values():
            values.update(int(value) for value in case.values())
    for case in protocol["training"]["seeds_by_replica"].values():
        values.update(int(value) for value in case.values())
    for suite in ("validation", "confirmation"):
        for spec in protocol["evaluation"][suite]["strata"].values():
            for case in spec["cases"]:
                values.update(int(value) for value in case.values())
    return values


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = registered_seeds(protocol)
    excluded_roots = {
        (ROOT / protocol[key]).resolve() for key in (
            "rehearsal_dataset_directory", "collection_manifest_directory",
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
        raise ValueError(f"registered s137000 seeds were already used: {conflicts}")


def validate_dataset(protocol: dict, protocol_hash: str, replica: str, stratum: str) -> dict:
    path = dataset_path(protocol, replica, stratum)
    expected = {
        "kind": "model-a-v4-rehearsal-dataset", "protocol_sha256": protocol_hash,
        "replica": replica, "stratum": stratum,
        "source_sha256": protocol["source_parent"]["sha256"], "all_action_return_horizon": 8,
    }
    transitions = load_dataset(path, expected)
    payload = _torch_load_dataset(path)
    d = payload["collection_diagnostics"]
    if payload.get("policy_updates") != 0 or payload.get("collection_rounds") != 25:
        raise RuntimeError(f"rehearsal collection mutated policy or changed budget: {path}")
    if (
        d["raw_transitions"] != d["matured_targets"] or len(d["per_round"]) != 25
        or d["survivor_terminal_merges"] + d["dead_terminal_appends"] != 25
    ):
        raise RuntimeError(f"rehearsal collection accounting mismatch: {path}")
    if len(transitions) != int(payload["count"]):
        raise RuntimeError(f"rehearsal collection count mismatch: {path}")
    return {"path": relative(path), "sha256": sha256_file(path), "count": len(transitions), "diagnostics": d}


def _torch_load_dataset(path: Path) -> dict:
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def collect_one(protocol_path: Path, protocol: dict, protocol_hash: str, replica: str, stratum: str) -> dict:
    manifest = collection_manifest_path(protocol, replica, stratum)
    existing = load_completed(manifest, COLLECTION_KIND)
    if existing is not None:
        dataset = validate_dataset(protocol, protocol_hash, replica, stratum)
        if existing.get("protocol_sha256") != protocol_hash or existing["dataset"]["sha256"] != dataset["sha256"]:
            raise RuntimeError(f"completed rehearsal collection drift: {replica}/{stratum}")
        return existing
    output = dataset_path(protocol, replica, stratum)
    stats = manifest.with_suffix(".stats.json")
    if output.exists() or manifest.exists() or stats.exists():
        raise RuntimeError(f"refusing orphaned rehearsal collection: {replica}/{stratum}")
    spec = protocol["rehearsal_collection"]["strata"][stratum]
    case = spec["seeds_by_replica"][replica]
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_rehearsal", *spec["opponents"],
        "--train", "1", "--continue-without-training", "--no-gui", "--scenario", spec["scenario"],
        "--n-rounds", "25", "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = {
        "MODEL_A_REHEARSAL_PROTOCOL_PATH": str(protocol_path), "MODEL_A_REHEARSAL_RUN_MODE": "collect",
        "MODEL_A_REHEARSAL_REPLICA": replica, "MODEL_A_REHEARSAL_STRATUM": stratum,
        "MODEL_A_REHEARSAL_SEED": str(case["agent_seed"]), "MODEL_A_REHEARSAL_DATASET_PATH": str(output),
    }
    if spec["opponents"]:
        overrides["TASK3_OPPONENT_SEED"] = str(case["opponent_seed"])
    record = {
        "schema_version": 1, "kind": COLLECTION_KIND, "status": "running",
        "started_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "replica": replica, "stratum": stratum, "collection": {**case, **{k: spec[k] for k in ("scenario", "opponents")}, "rounds": 25},
        "policy_updates": 0, "command": command, "environment_overrides": overrides,
        "dataset": {"path": relative(output), "sha256": None}, "raw_stats": relative(stats),
    }
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now(); record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats.is_file():
            raise RuntimeError("rehearsal collection process or stats failed")
        dataset = validate_dataset(protocol, protocol_hash, replica, stratum)
        record["dataset"] = dataset; record["metrics_by_agent"] = metrics_by_agent(json.loads(stats.read_text()))
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"; record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"rehearsal collection failed: {replica}/{stratum}: {record.get('error')}")
    return record


def validate_checkpoint(protocol: dict, protocol_hash: str, arm: str, replica: str) -> dict:
    path = checkpoint_path(protocol, arm, replica)
    payload = _torch_load(path)
    expected = {
        "architecture": "model-a-mlp-v4-global7", "protocol_sha256": protocol_hash,
        "arm": arm, "replica": replica, "stage_id": "task4c_rehearsal_pilot",
        "stage_completed_rounds": ENDPOINT_ROUND, "parent_sha256": protocol["source_parent"]["sha256"],
        "single_training_variable": protocol["single_training_variable"], "all_action_return_horizon": 8,
        "rehearsal_fraction": 0.0 if arm == "no_rehearsal" else 0.25,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"rehearsal checkpoint {key} mismatch: {path}")
    d = payload["training_diagnostics"]
    if (
        d["raw_transitions"] != d["matured_targets"]
        or d["gradient_updates"] != d["sampled_batches"]
        or len(d["per_round"]) != 100
        or d["survivor_terminal_merges"] + d["dead_terminal_appends"] != 100
    ):
        raise RuntimeError(f"rehearsal checkpoint accounting mismatch: {path}")
    requested = 16 * d["sampled_batches"] if arm == "rehearsal25" else 0
    if d["requested_rehearsal_slots"] != requested or d["fulfilled_rehearsal_slots"] != requested:
        raise RuntimeError(f"rehearsal slot accounting mismatch: {path}")
    if sum(item["transitions"] for item in d["per_round"]) != d["raw_transitions"]:
        raise RuntimeError(f"rehearsal per-round transition accounting mismatch: {path}")
    if sum(item["optimizer_updates"] for item in d["per_round"]) != d["gradient_updates"]:
        raise RuntimeError(f"rehearsal per-round update accounting mismatch: {path}")
    if any(
        item["sampled_transitions"] != 64 * item["optimizer_updates"]
        or sum(item["opponent_count_state_counts"].values()) != item["transitions"]
        for item in d["per_round"]
    ):
        raise RuntimeError(f"rehearsal per-round exposure accounting mismatch: {path}")
    return payload


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, arm: str, replica: str) -> dict:
    manifest = training_manifest_path(protocol, arm, replica)
    existing = load_completed(manifest, TRAINING_KIND)
    if existing is not None:
        payload = validate_checkpoint(protocol, protocol_hash, arm, replica)
        if existing.get("protocol_sha256") != protocol_hash or existing["checkpoint"]["sha256"] != sha256_file(checkpoint_path(protocol, arm, replica)):
            raise RuntimeError(f"completed rehearsal training drift: {arm}/{replica}")
        return existing
    output = checkpoint_path(protocol, arm, replica); stats = manifest.with_suffix(".stats.json")
    if output.parent.exists() or manifest.exists() or stats.exists():
        raise RuntimeError(f"refusing orphaned rehearsal training: {arm}/{replica}")
    spec = protocol["training"]; case = spec["seeds_by_replica"][replica]
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_rehearsal", *spec["opponents"],
        "--train", "1", "--continue-without-training", "--no-gui", "--scenario", spec["scenario"],
        "--n-rounds", "100", "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = {
        "MODEL_A_REHEARSAL_PROTOCOL_PATH": str(protocol_path), "MODEL_A_REHEARSAL_RUN_MODE": "train",
        "MODEL_A_REHEARSAL_ARM": arm, "MODEL_A_REHEARSAL_REPLICA": replica,
        "MODEL_A_REHEARSAL_SEED": str(case["agent_seed"]), "MODEL_A_REHEARSAL_CHECKPOINT_PATH": str(output),
        "TASK4_RULE_SEED": str(case["opponent_seed"]),
    }
    record = {
        "schema_version": 1, "kind": TRAINING_KIND, "status": "running", "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash, "arm": arm, "replica": replica,
        "training": {**case, "scenario": spec["scenario"], "opponents": spec["opponents"], "rounds": 100},
        "checkpoint": {"path": relative(output), "sha256": None}, "raw_stats": relative(stats),
        "command": command, "environment_overrides": overrides,
    }
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now(); record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats.is_file():
            raise RuntimeError("rehearsal training process or stats failed")
        payload = validate_checkpoint(protocol, protocol_hash, arm, replica)
        record["checkpoint"]["sha256"] = sha256_file(output)
        record["training_diagnostics"] = payload["training_diagnostics"]
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats.read_text()))
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"; record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"rehearsal training failed: {arm}/{replica}: {record.get('error')}")
    return record


def parameter_effect(protocol: dict) -> dict:
    pairs = {}
    for replica in REPLICAS:
        left = _torch_load(checkpoint_path(protocol, "no_rehearsal", replica))["online_net"]
        right = _torch_load(checkpoint_path(protocol, "rehearsal25", replica))["online_net"]
        equal = set(left) == set(right) and all(bool((left[key] == right[key]).all()) for key in left)
        pairs[replica] = {"endpoint_diverged": not equal}
    count = sum(item["endpoint_diverged"] for item in pairs.values())
    return {"pairs": pairs, "endpoint_divergent_pairs": count, "evaluation_authorized": count >= 2}


def checkpoint_for_label(protocol: dict, label: str) -> Path:
    if label == "source-r2": return (ROOT / protocol["source_parent"]["path"]).resolve()
    if label == "v4": return (ROOT / protocol["frozen_v4_baseline"]["path"]).resolve()
    arm, replica = label.rsplit("-", 1)
    return checkpoint_path(protocol, arm, replica)


def aggregate_attack(runs_or_values: list[dict]) -> dict:
    totals = {key: 0 for key in ATTACK_KEYS}
    for item in runs_or_values:
        d = item.get("attack_diagnostic", {}).get("diagnostics", item)
        for key in ATTACK_KEYS: totals[key] += int(d[key])
    totals["threat_to_kill_conversion"] = totals["successful_threat_bombs"] / totals["threat_bombs"] if totals["threat_bombs"] else 0.0
    totals["linked_kills_per_threat"] = totals["linked_threat_kill_events"] / totals["threat_bombs"] if totals["threat_bombs"] else 0.0
    return totals


def validate_attack(protocol: dict, protocol_hash: str, suite: str, stratum: str, label: str, case: dict) -> dict:
    path = evaluation_diagnostic_path(protocol, suite, stratum, label, int(case["world_seed"]))
    payload = json.loads(path.read_text())
    expected = {"kind": "model-a-v4-rehearsal-evaluation-diagnostic", "protocol_sha256": protocol_hash, "suite": suite, "label": label, "stratum": stratum, "world_seed": int(case["world_seed"]), "agent_seed": int(case["agent_seed"]), "policy_updates": 0, "actions_overridden": 0}
    if {key: payload.get(key) for key in expected} != expected:
        raise RuntimeError(f"rehearsal evaluation diagnostic metadata mismatch: {path}")
    d = payload["diagnostics"]
    if set(d) != set(ATTACK_KEYS) | {"threat_to_kill_conversion", "linked_kills_per_threat"} or d["rounds"] != 25:
        raise RuntimeError(f"rehearsal evaluation diagnostic schema mismatch: {path}")
    if d["survived_rounds"] + d["got_killed_events"] != d["rounds"] or d["linked_kill_events"] + d["unlinked_or_ambiguous_kill_events"] != d["kill_events"]:
        raise RuntimeError(f"rehearsal evaluation diagnostic accounting mismatch: {path}")
    expected_rate = d["successful_threat_bombs"] / d["threat_bombs"] if d["threat_bombs"] else 0.0
    if abs(d["threat_to_kill_conversion"] - expected_rate) > 1e-12:
        raise RuntimeError(f"rehearsal evaluation conversion mismatch: {path}")
    return {"path": relative(path), "sha256": sha256_file(path), "diagnostics": d}


def evaluate_one(protocol_path: Path, protocol: dict, protocol_hash: str, suite: str, stratum: str, label: str, case: dict) -> dict:
    spec = protocol["evaluation"][suite]["strata"][stratum]
    checkpoint = checkpoint_for_label(protocol, label); checkpoint_hash = sha256_file(checkpoint)
    output = evaluation_manifest_path(protocol, suite, stratum, label, int(case["world_seed"]))
    existing = load_completed(output, EVALUATION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or existing["checkpoint"]["sha256"] != checkpoint_hash:
            raise RuntimeError(f"completed rehearsal evaluation drift: {output}")
        attack = validate_attack(protocol, protocol_hash, suite, stratum, label, case)
        if existing.get("attack_diagnostic", {}).get("sha256") != attack["sha256"]:
            raise RuntimeError(f"completed rehearsal attack diagnostic drift: {output}")
        return existing
    stats = output.with_suffix(".stats.json")
    diagnostic = evaluation_diagnostic_path(protocol, suite, stratum, label, int(case["world_seed"]))
    if output.exists() or stats.exists() or diagnostic.exists():
        raise RuntimeError(f"refusing orphaned rehearsal evaluation: {output}")
    command = [sys.executable, "main.py", "play", "--agents", "model_a_v4_rehearsal", *spec["opponents"], "--train", "1", "--continue-without-training", "--no-gui", "--scenario", spec["scenario"], "--n-rounds", "25", "--seed", str(case["world_seed"]), "--save-stats", str(stats)]
    overrides = {
        "MODEL_A_REHEARSAL_PROTOCOL_PATH": str(protocol_path), "MODEL_A_REHEARSAL_RUN_MODE": "evaluate",
        "MODEL_A_REHEARSAL_EVAL_SUITE": suite, "MODEL_A_REHEARSAL_EVAL_STRATUM": stratum,
        "MODEL_A_REHEARSAL_EVAL_LABEL": label, "MODEL_A_REHEARSAL_EVAL_WORLD_SEED": str(case["world_seed"]),
        "MODEL_A_REHEARSAL_EVAL_DIAGNOSTIC_PATH": str(diagnostic), "MODEL_A_REHEARSAL_SEED": str(case["agent_seed"]),
        "MODEL_A_REHEARSAL_CHECKPOINT_PATH": str(checkpoint),
    }
    if spec["opponents"]:
        key = "TASK4_RULE_SEED" if "seeded_rule_based_agent" in spec["opponents"] else "TASK3_OPPONENT_SEED"
        overrides[key] = str(case["opponent_seed"])
    record = {"schema_version": 1, "kind": EVALUATION_KIND, "status": "running", "started_at_utc": utc_now(), "protocol_id": protocol["protocol_id"], "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash, "suite": suite, "stratum": stratum, "label": label, "evaluation": {**case, "scenario": spec["scenario"], "opponents": spec["opponents"], "rounds": 25}, "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash}, "command": command, "environment_overrides": overrides, "raw_stats": relative(stats)}
    atomic_json(output, record)
    child = os.environ.copy(); child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now(); record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats.is_file() or sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("rehearsal evaluation process, stats, or checkpoint failed")
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats.read_text()))
        record["target_metrics"] = record["metrics_by_agent"].get("model_a_v4_rehearsal")
        if not record["target_metrics"]: raise RuntimeError("rehearsal target metrics missing")
        record["attack_diagnostic"] = validate_attack(protocol, protocol_hash, suite, stratum, label, case)
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"; record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(output, record)
    if record["status"] != "completed": raise RuntimeError(f"rehearsal evaluation failed: {suite}/{stratum}/{label}: {record.get('error')}")
    return record


def run_suite(protocol_path: Path, protocol: dict, protocol_hash: str, suite: str, labels: list[str]) -> tuple[dict, dict]:
    rows, manifests = {}, {}
    for label in labels:
        rows[label], manifests[label] = {}, {}
        for stratum in EVAL_STRATA:
            spec = protocol["evaluation"][suite]["strata"][stratum]
            runs = [evaluate_one(protocol_path, protocol, protocol_hash, suite, stratum, label, case) for case in spec["cases"]]
            rows[label][stratum] = {"target": aggregate(runs), "opponents": opponent_summary(runs, "model_a_v4_rehearsal"), "attack": aggregate_attack(runs)}
            manifests[label][stratum] = [relative(evaluation_manifest_path(protocol, suite, stratum, label, int(case["world_seed"]))) for case in spec["cases"]]
    return rows, manifests


def combine_metrics(values: list[dict]) -> dict:
    additive = ("rounds", "score", "coins", "kills", "crates", "bombs", "moves", "waits", "suicides", "invalid_actions", "steps")
    totals = {key: sum(int(value[key]) for value in values) for key in additive}; rounds = totals["rounds"]; steps = totals["steps"]
    return {**totals, "score_per_round": totals["score"] / rounds, "kills_per_round": totals["kills"] / rounds, "suicides_per_round": totals["suicides"] / rounds, "bombs_per_round": totals["bombs"] / rounds, "invalid_actions_per_round": totals["invalid_actions"] / rounds, "move_fraction": totals["moves"] / steps if steps else 0.0, "wait_fraction": totals["waits"] / steps if steps else 0.0, "episode_length": steps / rounds}


def combined_task4(rows: dict, label: str) -> tuple[dict, dict]:
    names = ("task4b_duel", "task4c_three_rule")
    return combine_metrics([rows[label][name]["target"] for name in names]), aggregate_attack([rows[label][name]["attack"] for name in names])


def retention_index(rows: dict, label: str) -> float:
    return sum(rows[label][name]["target"]["score_per_round"] / max(rows["source-r2"][name]["target"]["score_per_round"], 1e-12) for name in ("task1", "task2", "task3_peaceful")) / 3


def validation_decision(protocol: dict, rows: dict, effect: dict) -> dict:
    rule = protocol["evaluation"]["validation"]["decision_rule"]
    if not effect["evaluation_authorized"]:
        return {"passed": False, "decision": "rehearsal_no_parameter_effect_stop", "evaluation_started": False, "parameter_effect": effect}
    pairs = {}; supportive = 0
    for replica in REPLICAS:
        control = f"no_rehearsal-{replica}"; candidate = f"rehearsal25-{replica}"
        cm, ca = combined_task4(rows, control); rm, ra = combined_task4(rows, candidate)
        checks = {
            "retention_index_gain": retention_index(rows, candidate) >= retention_index(rows, control) + rule["pair_minimum_retention_index_gain"] - 1e-12,
            "task4_kills_noninferior": rm["kills_per_round"] >= cm["kills_per_round"] - rule["maximum_task4_kills_drop"] - 1e-12,
            "task4_conversion_noninferior": ra["threat_to_kill_conversion"] >= ca["threat_to_kill_conversion"] - rule["maximum_task4_conversion_drop"] - 1e-12,
            "task4_score_noninferior": rm["score_per_round"] >= cm["score_per_round"] - rule["maximum_task4_score_drop"] - 1e-12,
            "task4_suicide_safe": rm["suicides_per_round"] <= cm["suicides_per_round"] + rule["maximum_task4_suicide_increase"] + 1e-12,
        }
        supportive += int(all(checks.values()))
        pairs[replica] = {"supportive": all(checks.values()), "checks": checks, "control_retention_index": retention_index(rows, control), "candidate_retention_index": retention_index(rows, candidate), "control_task4": {"metrics": cm, "attack": ca}, "candidate_task4": {"metrics": rm, "attack": ra}}
    controls = [f"no_rehearsal-{r}" for r in REPLICAS]; candidates = [f"rehearsal25-{r}" for r in REPLICAS]
    def pooled_stratum(labels, name): return combine_metrics([rows[label][name]["target"] for label in labels])
    control4 = combine_metrics([combined_task4(rows, label)[0] for label in controls]); candidate4 = combine_metrics([combined_task4(rows, label)[0] for label in candidates])
    control_attack = aggregate_attack([combined_task4(rows, label)[1] for label in controls]); candidate_attack = aggregate_attack([combined_task4(rows, label)[1] for label in candidates])
    pooled = {
        "task1_gain": pooled_stratum(candidates, "task1")["score_per_round"] >= pooled_stratum(controls, "task1")["score_per_round"] + rule["pooled_minimum_task1_score_gain"] - 1e-12,
        "task2_gain": pooled_stratum(candidates, "task2")["score_per_round"] >= pooled_stratum(controls, "task2")["score_per_round"] + rule["pooled_minimum_task2_score_gain"] - 1e-12,
        "task3_peaceful_gain": pooled_stratum(candidates, "task3_peaceful")["score_per_round"] >= pooled_stratum(controls, "task3_peaceful")["score_per_round"] + rule["pooled_minimum_task3_peaceful_score_gain"] - 1e-12,
        "task4_kills_noninferior": candidate4["kills_per_round"] >= control4["kills_per_round"] - rule["maximum_task4_kills_drop"] - 1e-12,
        "task4_conversion_noninferior": candidate_attack["threat_to_kill_conversion"] >= control_attack["threat_to_kill_conversion"] - rule["maximum_task4_conversion_drop"] - 1e-12,
        "task4_score_noninferior": candidate4["score_per_round"] >= control4["score_per_round"] - rule["maximum_task4_score_drop"] - 1e-12,
        "task4_suicide_safe": candidate4["suicides_per_round"] <= control4["suicides_per_round"] + rule["maximum_task4_suicide_increase"] + 1e-12,
    }
    passed = supportive >= rule["minimum_supportive_pairs"] and all(pooled.values())
    selected = None
    if passed:
        eligible = [replica for replica in REPLICAS if pairs[replica]["supportive"]]
        selected = max(eligible, key=lambda r: (retention_index(rows, f"rehearsal25-{r}"), combined_task4(rows, f"rehearsal25-{r}")[0]["kills_per_round"], combined_task4(rows, f"rehearsal25-{r}")[0]["score_per_round"], -combined_task4(rows, f"rehearsal25-{r}")[0]["suicides_per_round"], -REPLICAS.index(r)))
    return {"passed": passed, "decision": "rehearsal_validation_supported" if passed else "rehearsal_not_supported_stop", "evaluation_started": True, "parameter_effect": effect, "supportive_pairs": supportive, "pair_results": pairs, "pooled_checks": pooled, "pooled_task4": {"control": {"metrics": control4, "attack": control_attack}, "rehearsal25": {"metrics": candidate4, "attack": candidate_attack}}, "selected_replica": selected, "selection_rule": "max retention_index; then task4 kills, score, lower suicide, replica order"}


def confirmation_decision(protocol: dict, rows: dict, selected_replica: str) -> dict:
    rule = protocol["evaluation"]["confirmation"]["decision_rule"]; label = f"rehearsal25-{selected_replica}"
    source4, source_attack = combined_task4(rows, "source-r2"); candidate4, candidate_attack = combined_task4(rows, label)
    checks = {
        "task1": rows[label]["task1"]["target"]["score_per_round"] >= rule["minimum_task1_fraction_of_source"] * rows["source-r2"]["task1"]["target"]["score_per_round"] - 1e-12,
        "task2": rows[label]["task2"]["target"]["score_per_round"] >= rows["source-r2"]["task2"]["target"]["score_per_round"] - rule["maximum_task2_score_drop_from_source"] - 1e-12,
        "task4_kills": candidate4["kills_per_round"] >= source4["kills_per_round"] + rule["minimum_task4_kills_gain_over_source"] - 1e-12,
        "task4_conversion": candidate_attack["threat_to_kill_conversion"] >= source_attack["threat_to_kill_conversion"] + rule["minimum_task4_conversion_gain_over_source"] - 1e-12,
        "task4_score": candidate4["score_per_round"] >= source4["score_per_round"] - rule["maximum_task4_score_drop_from_source"] - 1e-12,
        "task4_suicide": candidate4["suicides_per_round"] <= source4["suicides_per_round"] + rule["maximum_task4_suicide_increase_over_source"] + 1e-12,
    }
    for name in ("task3_peaceful", "task3_coin"):
        c = rows[label][name]["target"]; s = rows["source-r2"][name]["target"]
        checks[f"{name}_score"] = c["score_per_round"] >= s["score_per_round"] - rule["maximum_task3_score_drop_from_source"] - 1e-12
        checks[f"{name}_kills"] = c["kills_per_round"] >= s["kills_per_round"] - rule["maximum_task3_kills_drop_from_source"] - 1e-12
        checks[f"{name}_suicide"] = c["suicides_per_round"] <= s["suicides_per_round"] + rule["maximum_task3_suicide_increase_over_source"] + 1e-12
    for name in ("task4b_duel", "task4c_three_rule"):
        c = rows[label][name]["target"]; s = rows["source-r2"][name]["target"]
        checks[f"{name}_kills"] = c["kills_per_round"] >= s["kills_per_round"] - rule["maximum_task4_stratum_kills_drop_from_source"] - 1e-12
        checks[f"{name}_score"] = c["score_per_round"] >= s["score_per_round"] - rule["maximum_task4_stratum_score_drop_from_source"] - 1e-12
        checks[f"{name}_suicide"] = c["suicides_per_round"] <= s["suicides_per_round"] + rule["maximum_task4_stratum_suicide_increase_over_source"] + 1e-12
    passed = all(checks.values())
    return {"passed": passed, "decision": "rehearsal_confirmation_passed_stop_before_promotion" if passed else "rehearsal_validation_supported_confirmation_failed_stop", "selected_replica": selected_replica, "selected_label": label, "checks": checks, "candidate_task4": {"metrics": candidate4, "attack": candidate_attack}, "source_task4": {"metrics": source4, "attack": source_attack}, "v4_reported_not_used_for_selection": True}


def dry_run(protocol: dict) -> dict:
    collection = len(REPLICAS) * len(OLD_STRATA) * 25
    training = len(REPLICAS) * len(ARMS) * 100
    cases = sum(len(spec["cases"]) * 25 for spec in protocol["evaluation"]["validation"]["strata"].values())
    validation = (1 + len(REPLICAS) * len(ARMS)) * cases
    confirmation = 3 * cases
    return {"mode": "dry-run", "protocol_id": protocol["protocol_id"], "stage0": "complete_no_comparable_intermediate_s136_checkpoints_except_round_100", "stage1": "satisfied_by_completed_s136000_no_repeat", "single_variable": protocol["single_training_variable"], "rehearsal_batch": protocol["learning_contract"]["rehearsal25_batch"], "collection_rounds": collection, "training_rounds": training, "validation_rounds": validation, "conditional_confirmation_rounds": confirmation, "maximum_total_rounds": collection + training + validation + confirmation, "formal_work_started": False, "automatic_followup_started": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL); parser.add_argument("--execute", action="store_true"); args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol, protocol_hash = load_protocol(protocol_path); assert_registered_seeds_untouched(protocol_path, protocol)
    if not args.execute:
        print(json.dumps(dry_run(protocol), indent=2, sort_keys=True)); return 0
    existing = load_completed(report_path(protocol), REPORT_KIND)
    if existing is not None:
        print(json.dumps(existing["result"], indent=2, sort_keys=True)); return 0
    source_before = sha256_file(ROOT / protocol["source_parent"]["path"])
    collections = {replica: {stratum: collect_one(protocol_path.resolve(), protocol, protocol_hash, replica, stratum) for stratum in OLD_STRATA} for replica in REPLICAS}
    training = {arm: {replica: train_one(protocol_path.resolve(), protocol, protocol_hash, arm, replica) for replica in REPLICAS} for arm in ARMS}
    if sha256_file(ROOT / protocol["source_parent"]["path"]) != source_before: raise RuntimeError("source-r2 changed during rehearsal experiment")
    effect = parameter_effect(protocol)
    if effect["evaluation_authorized"]:
        validation_labels = ["source-r2", *[f"{arm}-{replica}" for replica in REPLICAS for arm in ARMS]]
        validation_rows, validation_manifests = run_suite(protocol_path.resolve(), protocol, protocol_hash, "validation", validation_labels)
    else:
        validation_rows = validation_manifests = None
    validation = validation_decision(protocol, validation_rows, effect)
    confirmation_rows = confirmation_manifests = confirmation = None
    if validation["passed"]:
        selected = validation["selected_replica"]
        confirmation_rows, confirmation_manifests = run_suite(protocol_path.resolve(), protocol, protocol_hash, "confirmation", ["v4", "source-r2", f"rehearsal25-{selected}"])
        confirmation = confirmation_decision(protocol, confirmation_rows, selected)
        result = confirmation
    else:
        result = validation
    record = {"schema_version": 1, "kind": REPORT_KIND, "status": "completed", "completed_at_utc": utc_now(), "protocol_id": protocol["protocol_id"], "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash, "stage0": protocol["stage0_adjudication"], "stage1": protocol["stage1_adjudication"], "collection_manifests": {r: {s: relative(collection_manifest_path(protocol, r, s)) for s in OLD_STRATA} for r in REPLICAS}, "training_manifests": {a: {r: relative(training_manifest_path(protocol, a, r)) for r in REPLICAS} for a in ARMS}, "parameter_effect": effect, "validation": {"decision": validation, "rows": validation_rows, "manifests": validation_manifests}, "confirmation": {"decision": confirmation, "rows": confirmation_rows, "manifests": confirmation_manifests}, "result": result, "source_checkpoint_modified": False, "default_checkpoint_modified": False, "automatic_followup_started": False, "awaiting_user_instruction": True}
    atomic_json(report_path(protocol), record)
    print(json.dumps({"report": relative(report_path(protocol)), "report_sha256": sha256_file(report_path(protocol)), **result}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
