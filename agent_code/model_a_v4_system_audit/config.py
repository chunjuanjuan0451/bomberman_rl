"""Frozen protocol helpers for the passive reward/credit/attack-chain audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASES = ("duel_c1", "duel_c2", "three_c1", "three_c2")
STRATA = ("task4b_duel", "task4c_three_rule")
HORIZONS = (1, 4, 8, 16, 32)
EVENT_NAMES = (
    "MOVED_LEFT", "MOVED_RIGHT", "MOVED_UP", "MOVED_DOWN", "WAITED",
    "INVALID_ACTION", "BOMB_DROPPED", "BOMB_EXPLODED", "CRATE_DESTROYED",
    "COIN_FOUND", "COIN_COLLECTED", "KILLED_OPPONENT", "KILLED_SELF",
    "GOT_KILLED", "OPPONENT_ELIMINATED", "SURVIVED_ROUND",
)
CUSTOM_EVENTS_CONFIRMED_ABSENT = (
    "MOVE_TO_DEAD", "MOVE_TO_TARGET", "ATTACK_TARGET", "ATTACK_ENEMY", "KILL_ENEMY",
)
SOURCE_PATHS = {
    "agents.py", "environment.py", "events.py", "settings.py",
    "agent_code/model_a_dqn/callbacks.py", "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py", "agent_code/model_a_dqn/train.py",
    "agent_code/model_a_dqn/replay_buffer.py", "agent_code/model_a_v4_curriculum/config.py",
    "agent_code/model_a_v4_curriculum/callbacks.py", "agent_code/model_a_v7a/planner.py",
    "agent_code/model_a_v7b/tactical.py", "agent_code/model_a_v4_system_audit/config.py",
    "agent_code/model_a_v4_system_audit/callbacks.py", "agent_code/model_a_v4_system_audit/train.py",
    "agent_code/seeded_rule_based_agent/callbacks.py", "tools/v4_task4_duel_training.py",
    "tools/v4_training_system_audit.py",
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
            "cases": ["duel_c1", "duel_c2"],
        },
        "task4c_three_rule": {
            "opponents": ["seeded_rule_based_agent"] * 3,
            "cases": ["three_c1", "three_c2"],
        },
    }


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-training-system-audit":
        raise ValueError("wrong training-system audit protocol")
    if protocol.get("scope") != "phase_a_passive_ledger_and_phase_b_offline_horizon_only":
        raise ValueError("training-system audit scope changed")
    collection = protocol.get("collection", {})
    if (
        collection.get("scenario") != "classic"
        or int(collection.get("rounds_per_case", 0)) != 50
        or float(collection.get("epsilon", -1)) != 0.05
        or float(collection.get("oracle_deadline_seconds", -1)) != 0.03
        or int(collection.get("policy_updates", -1)) != 0
        or int(collection.get("actions_overridden", -1)) != 0
        or collection.get("strata") != _expected_strata()
        or tuple(collection.get("cases", {})) != CASES
    ):
        raise ValueError("training-system collection contract changed")
    seen: set[int] = set()
    for label, case in collection["cases"].items():
        _register(seen, case, f"collection/{label}")
    if len(seen) != 12:
        raise ValueError("training-system audit must register 12 unique seeds")
    audit = protocol.get("audit_contract", {})
    expected_audit = {
        "horizons": list(HORIZONS),
        "gamma": 0.99,
        "batch_size": 64,
        "loss": "smooth_l1_beta_1_mean",
        "event_names": list(EVENT_NAMES),
        "custom_events_confirmed_absent": list(CUSTOM_EVENTS_CONFIRMED_ABSENT),
        "attack_stages": ["approach", "positioning", "threat", "bomb", "trap", "escape", "kill"],
        "positioning_definition": "BOMB legal and counterfactual max_space_reduction >= 0.5",
        "threat_definition": "actual BOMB with affected_opponents >= 1",
        "trap_definition": "actual BOMB with max_space_reduction >= 0.5",
        "escape_definition": "actual BOMB followed by no KILLED_SELF in same-episode eight-step window",
        "kill_definition": "official KILLED_OPPONENT uniquely linked to previous BOMB at lag 4 or 5",
        "horizon_mode": "offline reconstruction on identical frozen-policy trajectories; no optimizer step",
    }
    if audit != expected_audit:
        raise ValueError("training-system audit definitions changed")
    if protocol.get("decision") != "diagnostic_complete_stop_before_any_training":
        raise ValueError("training-system audit must not select or train a horizon")
    if protocol.get("training_enabled") is not False or protocol.get("automatic_followup") is not False:
        raise ValueError("training-system audit must stop before training")
    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("training-system source binding set mismatch")
    for relative_path, expected_hash in protocol["source_bindings"].items():
        source = ROOT / relative_path
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"training-system source binding mismatch: {relative_path}")
    for field in ("source_parent", "prior_counterfactual_report", "prior_safe500_report", "failed_predecessor"):
        item = protocol.get(field, {})
        artifact = ROOT / item.get("path", "missing")
        if not artifact.is_file() or sha256_file(artifact) != item.get("sha256"):
            raise ValueError(f"training-system bound artifact mismatch: {field}")
    prior = json.loads((ROOT / protocol["prior_safe500_report"]["path"]).read_text(encoding="utf-8"))
    if prior.get("result", {}).get("decision") != "reward500_signal_not_supported_stop":
        raise ValueError("training-system audit is not bound to terminal s133000 evidence")
    failed = json.loads((ROOT / protocol["failed_predecessor"]["path"]).read_text(encoding="utf-8"))
    if (
        failed.get("status") != "failed"
        or failed.get("protocol_id") != "model-a-v4-training-system-audit-s134000"
        or failed.get("case") != "duel_c1"
        or int(failed.get("exit_code", 0)) == 0
    ):
        raise ValueError("training-system recovery is not bound to the s134000 callback failure")
    return protocol, sha256_file(path)


def case_stratum(protocol: dict, case: str) -> str:
    if case not in CASES:
        raise ValueError("unknown training-system audit case")
    matches = [name for name, spec in protocol["collection"]["strata"].items() if case in spec["cases"]]
    if len(matches) != 1:
        raise ValueError(f"case must belong to one stratum: {case}")
    return matches[0]


def trace_path(protocol: dict, case: str) -> Path:
    if case not in CASES:
        raise ValueError("unknown training-system audit case")
    return (ROOT / protocol["trace_directory"] / f"{case}.npz").resolve()
