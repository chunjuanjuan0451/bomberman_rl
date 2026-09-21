"""Frozen protocol helpers for the corrected BOMB-credit matched A/B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("control", "credit")
REPLICAS = ("r1", "r2", "r3")
SNAPSHOT_ROUNDS = (25, 50, 75, 100)
EVALUATION_STRATA = ("task1", "task2", "task3_peaceful", "task3_coin", "task4c_three_rule")
EXPECTED_DISTRIBUTIONS = {
    "task1": ("coin-heaven", []),
    "task2": ("classic", []),
    "task3_peaceful": ("classic", ["seeded_peaceful_agent"]),
    "task3_coin": ("classic", ["seeded_coin_collector_agent"]),
    "task4c_three_rule": ("classic", ["seeded_rule_based_agent"] * 3),
}
SOURCE_PATHS = {
    "agents.py",
    "environment.py",
    "settings.py",
    "agent_code/model_a_dqn/callbacks.py",
    "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py",
    "agent_code/model_a_dqn/train.py",
    "agent_code/model_a_dqn/replay_buffer.py",
    "agent_code/model_a_v4_curriculum/config.py",
    "agent_code/model_a_v4_curriculum/callbacks.py",
    "agent_code/model_a_v4_bomb_credit6/config.py",
    "agent_code/model_a_v4_bomb_credit6/callbacks.py",
    "agent_code/model_a_v4_bomb_credit6/replay.py",
    "agent_code/model_a_v4_bomb_credit6/train.py",
    "agent_code/seeded_peaceful_agent/callbacks.py",
    "agent_code/seeded_coin_collector_agent/callbacks.py",
    "agent_code/seeded_rule_based_agent/callbacks.py",
    "tools/v4_task4_duel_training.py",
    "tools/v4_bomb_credit_lifecycle_audit.py",
    "tools/v4_task4_bomb_credit6.py",
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


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-task4-bomb-credit6":
        raise ValueError("wrong BOMB-credit6 protocol")
    if protocol.get("active_subcourse") != "task4c_three_rule":
        raise ValueError("BOMB-credit6 pilot may run Task4C only")
    if tuple(protocol.get("arms", ())) != ARMS or tuple(protocol.get("replicas", ())) != REPLICAS:
        raise ValueError("BOMB-credit6 arms or replicas changed")
    if tuple(protocol.get("snapshot_rounds", ())) != SNAPSHOT_ROUNDS:
        raise ValueError("BOMB-credit6 snapshots changed")
    if protocol.get("single_training_variable") != "bomb_transition_return_horizon_1_vs_6":
        raise ValueError("BOMB-credit6 experiment must change one variable")
    expected_learning = {
        "architecture": "unchanged exact-v4 model-a-mlp-v4-global7",
        "features": "unchanged 96-dimensional exact-v4 latent",
        "action_mask": "unchanged exact-v4 legal mask",
        "reward_and_shaping": "unchanged exact-v4 values and events",
        "non_bomb_target": "one-step in both arms",
        "uniform_replay": "unchanged capacity 50000 and batch 64; reset at boundary",
        "optimizer": "resume source-r2 Adam state unchanged",
        "epsilon": "resume source-r2 at unchanged 0.05 floor",
        "teacher_or_oracle": "none",
    }
    if protocol.get("learning_contract") != expected_learning:
        raise ValueError("BOMB-credit6 learning contract changed")
    expected_target = {
        "gamma": 0.99,
        "bomb_maturation_steps_both_arms": 6,
        "control_bomb_return_steps": 1,
        "credit_bomb_return_steps": 6,
        "allowed_event_lag_steps": [4, 5],
        "terminal_truncation": "same-episode only",
        "terminal_event_delivery": "posthumous owned-bomb kills attach to the last terminal transition",
    }
    if protocol.get("target_contract") != expected_target:
        raise ValueError("BOMB-credit6 target contract changed")

    seen: set[int] = set()
    training = protocol.get("training", {})
    if (
        training.get("scenario") != "classic"
        or training.get("opponents") != ["seeded_rule_based_agent"] * 3
        or int(training.get("rounds_per_arm", 0)) != 100
        or tuple(training.get("seeds_by_replica", {})) != REPLICAS
    ):
        raise ValueError("BOMB-credit6 training distribution or budget changed")
    for replica, case in training["seeds_by_replica"].items():
        _register(seen, case, f"training/{replica}")

    evaluation = protocol.get("evaluation", {})
    if tuple(evaluation.get("strata", {})) != EVALUATION_STRATA:
        raise ValueError("BOMB-credit6 evaluation strata changed")
    for stratum, spec in evaluation["strata"].items():
        scenario, opponents = EXPECTED_DISTRIBUTIONS[stratum]
        expected_cases = 4 if stratum == "task4c_three_rule" else 1
        if spec.get("scenario") != scenario or spec.get("opponents") != opponents:
            raise ValueError(f"BOMB-credit6 evaluation distribution changed: {stratum}")
        if int(spec.get("rounds_per_case", 0)) != 25 or len(spec.get("cases", ())) != expected_cases:
            raise ValueError(f"BOMB-credit6 evaluation budget changed: {stratum}")
        for index, case in enumerate(spec["cases"]):
            _register(seen, case, f"evaluation/{stratum}/{index}")
    if len(seen) != 33:
        raise ValueError("BOMB-credit6 protocol must register exactly 33 unique seed values")

    expected_decision = {
        "minimum_supportive_pairs": 2,
        "minimum_observed_kill_events_per_credit_replica": 1,
        "pair_minimum_task4c_kills_gain": 0.000000001,
        "pair_minimum_task4c_score_delta": -0.25,
        "pair_maximum_task4c_suicide_delta": 0.05,
        "pooled_minimum_task4c_kills_gain_over_control": 0.03,
        "pooled_minimum_task4c_score_delta_to_control": -0.10,
        "pooled_maximum_task4c_suicide_delta_to_control": 0.05,
        "pooled_minimum_task4c_kills_gain_over_source": 0.02,
        "pooled_minimum_task4c_score_delta_to_source": -0.25,
        "pooled_maximum_task4c_suicide_delta_to_source": 0.05,
        "minimum_task1_fraction_of_source": 0.90,
        "maximum_task2_score_drop_from_source": 0.50,
        "maximum_task3_score_drop_from_source": 0.50,
        "maximum_task3_kills_drop_from_source": 0.05,
        "maximum_task3_suicide_increase_over_source": 0.05,
        "pass_decision": "bomb_six_step_credit_supported_stop_before_task4b",
        "fail_decision": "bomb_six_step_credit_not_supported_stop",
    }
    if evaluation.get("decision_rule") != expected_decision:
        raise ValueError("BOMB-credit6 decision rule changed")
    if protocol.get("automatic_followup") is not False:
        raise ValueError("BOMB-credit6 experiment must always stop")

    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("BOMB-credit6 source binding set mismatch")
    for relative_path, expected_hash in protocol["source_bindings"].items():
        source = ROOT / relative_path
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"BOMB-credit6 source binding mismatch: {relative_path}")
    for field in ("source_parent", "frozen_v4_baseline", "lifecycle_audit", "failed_predecessor"):
        item = protocol.get(field, {})
        artifact = ROOT / item.get("path", "missing")
        if not artifact.is_file() or sha256_file(artifact) != item.get("sha256"):
            raise ValueError(f"BOMB-credit6 bound artifact mismatch: {field}")
    audit = json.loads((ROOT / protocol["lifecycle_audit"]["path"]).read_text(encoding="utf-8"))
    if audit.get("decision") != "bomb_outcome_lags_four_and_five_confirmed":
        raise ValueError("BOMB-credit6 lifecycle audit did not pass")
    failure = json.loads((ROOT / protocol["failed_predecessor"]["path"]).read_text(encoding="utf-8"))
    if failure.get("status") != "failed" or failure.get("protocol_id") != "model-a-v4-task4-bomb-credit-s130000":
        raise ValueError("BOMB-credit6 predecessor failure changed")
    lineage = protocol["source_parent"].get("lineage", {})
    lineage_protocol = ROOT / lineage.get("protocol_path", "missing")
    if not lineage_protocol.is_file() or sha256_file(lineage_protocol) != lineage.get("protocol_sha256"):
        raise ValueError("BOMB-credit6 source lineage changed")
    return protocol, sha256_file(path)


def checkpoint_root(protocol: dict, arm: str, replica: str) -> Path:
    if arm not in ARMS or replica not in REPLICAS:
        raise ValueError("invalid BOMB-credit6 checkpoint identity")
    return (ROOT / protocol["checkpoint_directory"] / arm / replica).resolve()


def final_checkpoint_path(protocol: dict, arm: str, replica: str) -> Path:
    return checkpoint_root(protocol, arm, replica) / "final.pt"


def snapshot_path(protocol: dict, arm: str, replica: str, stage_round: int) -> Path:
    if stage_round not in SNAPSHOT_ROUNDS:
        raise ValueError("invalid BOMB-credit6 snapshot round")
    return checkpoint_root(protocol, arm, replica) / "snapshots" / f"round-{stage_round:04d}.pt"
