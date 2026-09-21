"""Immutable contract for the source-r2 counterfactual signal gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASES = (
    "duel_c1", "duel_c2", "duel_c3", "duel_c4",
    "three_c1", "three_c2", "three_c3", "three_c4",
)
STRATA = ("task4b_duel", "task4c_three_rule")
SOURCE_PATHS = {
    "environment.py",
    "settings.py",
    "agent_code/model_a_dqn/callbacks.py",
    "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py",
    "agent_code/model_a_v7a/planner.py",
    "agent_code/model_a_v7b/tactical.py",
    "agent_code/model_a_v4_causal_signal/config.py",
    "agent_code/model_a_v4_causal_signal/callbacks.py",
    "agent_code/model_a_v4_causal_signal/train.py",
    "agent_code/seeded_rule_based_agent/callbacks.py",
    "tools/v4_task4_counterfactual_signal.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _register(seen: set[int], case: dict, identity: str) -> None:
    if set(case) != {"world_seed", "agent_seed", "opponent_seed"}:
        raise ValueError(f"invalid seed tuple: {identity}")
    for raw in case.values():
        value = int(raw)
        if value <= 0 or value in seen:
            raise ValueError(f"seeds must be positive and globally unique: {identity}")
        seen.add(value)


def _expected_strata() -> dict:
    return {
        "task4b_duel": {
            "opponents": ["seeded_rule_based_agent"],
            "cases": ["duel_c1", "duel_c2", "duel_c3", "duel_c4"],
        },
        "task4c_three_rule": {
            "opponents": ["seeded_rule_based_agent"] * 3,
            "cases": ["three_c1", "three_c2", "three_c3", "three_c4"],
        },
    }


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-counterfactual-signal-audit":
        raise ValueError("wrong counterfactual-signal protocol")
    if protocol.get("scope") != "phase0_oracle_validation_and_phase1_passive_signal_only":
        raise ValueError("counterfactual signal gate may not train or modify a policy")

    collection = protocol.get("collection", {})
    if (
        collection.get("scenario") != "classic"
        or int(collection.get("rounds_per_case", 0)) != 50
        or float(collection.get("epsilon", -1)) != 0.05
        or float(collection.get("oracle_deadline_seconds", -1)) != 0.03
        or int(collection.get("policy_updates", -1)) != 0
        or collection.get("strata") != _expected_strata()
        or tuple(collection.get("cases", {})) != CASES
    ):
        raise ValueError("counterfactual signal collection contract changed")
    seen: set[int] = set()
    for label, case in collection["cases"].items():
        _register(seen, case, f"collection/{label}")
    if len(seen) != 24:
        raise ValueError("counterfactual signal gate must register exactly 24 unique seeds")

    expected_labeling = {
        "strict_opportunity": "BOMB is legal; counterfactual rollout finds at least one newly guaranteed trapped opponent; own bomb rollout survives with bottleneck >=2 and terminal positions >=3",
        "attempt": "strict_opportunity and the unchanged source-r2 behavior actually selects BOMB",
        "missed_opportunity": "strict_opportunity and the unchanged source-r2 behavior selects a non-BOMB action",
        "kill_horizon_steps": 6,
        "self_risk_horizon_steps": 8,
        "successful_causal_segment": "attempt followed by KILLED_OPPONENT within the same episode and six-transition window, with no KILLED_SELF within the same episode and eight-transition window",
        "terminal_transition_deduplication": "one row per observed (round, step, action)",
    }
    if protocol.get("labeling") != expected_labeling:
        raise ValueError("counterfactual signal labels changed")

    expected_gate = {
        "minimum_total_attempts": 20,
        "minimum_total_successful_causal_segments": 8,
        "minimum_kill_realization_rate": 0.40,
        "maximum_self_risk_rate": 0.15,
        "minimum_successful_causal_segments_per_stratum": 2,
        "maximum_oracle_timeout_fraction": 0.001,
        "pass_decision": "counterfactual_causal_signal_supported_stop_before_replay_training",
        "fail_decision": "counterfactual_causal_signal_not_supported_stop",
    }
    if protocol.get("decision_gate") != expected_gate:
        raise ValueError("counterfactual signal decision gate changed")
    if protocol.get("phase2_training_enabled") is not False or protocol.get("automatic_followup") is not False:
        raise ValueError("counterfactual signal audit must always stop before training")

    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("counterfactual signal source binding set mismatch")
    for relative_path, expected_hash in protocol["source_bindings"].items():
        source = ROOT / relative_path
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"counterfactual signal source binding mismatch: {relative_path}")

    for field in ("source_parent", "historical_v7b_result", "previous_kill_probe_report"):
        item = protocol.get(field, {})
        artifact = ROOT / item.get("path", "missing")
        if not artifact.is_file() or sha256_file(artifact) != item.get("sha256"):
            raise ValueError(f"counterfactual signal bound artifact mismatch: {field}")
    previous = json.loads((ROOT / protocol["previous_kill_probe_report"]["path"]).read_text(encoding="utf-8"))
    if previous.get("result", {}).get("decision") != "kill_head_signal_not_supported_stop":
        raise ValueError("counterfactual signal gate is not bound to the terminal s128000 failure")
    return protocol, sha256_file(path)


def case_stratum(protocol: dict, case: str) -> str:
    if case not in CASES:
        raise ValueError("unknown counterfactual signal case")
    memberships = [
        stratum for stratum, spec in protocol["collection"]["strata"].items()
        if case in spec["cases"]
    ]
    if len(memberships) != 1:
        raise ValueError(f"case must belong to exactly one stratum: {case}")
    return memberships[0]


def trace_path(protocol: dict, case: str) -> Path:
    if case not in CASES:
        raise ValueError("unknown counterfactual signal case")
    return (ROOT / protocol["trace_directory"] / f"{case}.npz").resolve()
