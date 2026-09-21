"""Frozen protocol helpers for the safe-kill BOMB reward-500 gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("safe12", "safe500")
REPLICAS = ("r1", "r2", "r3")
SNAPSHOT_ROUNDS = (5, 10, 15, 25)
EVALUATION_STRATA = ("task1", "task2", "task4c_three_rule")
EXPECTED_DISTRIBUTIONS = {
    "task1": ("coin-heaven", []),
    "task2": ("classic", []),
    "task4c_three_rule": ("classic", ["seeded_rule_based_agent"] * 3),
}
SOURCE_PATHS = {
    "agents.py", "environment.py", "settings.py",
    "agent_code/model_a_dqn/callbacks.py", "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py", "agent_code/model_a_dqn/train.py",
    "agent_code/model_a_dqn/replay_buffer.py",
    "agent_code/model_a_v4_curriculum/config.py", "agent_code/model_a_v4_curriculum/callbacks.py",
    "agent_code/model_a_v4_bomb_credit6/train.py", "agent_code/model_a_v4_bomb_credit6/replay.py",
    "agent_code/model_a_v4_bomb_safe500/config.py", "agent_code/model_a_v4_bomb_safe500/callbacks.py",
    "agent_code/model_a_v4_bomb_safe500/train.py", "agent_code/seeded_rule_based_agent/callbacks.py",
    "tools/v4_task4_duel_training.py", "tools/v4_task4_bomb_safe500.py",
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
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-task4-bomb-safe500":
        raise ValueError("wrong BOMB-safe500 protocol")
    if protocol.get("active_subcourse") != "task4c_three_rule":
        raise ValueError("BOMB-safe500 gate may run Task4C only")
    if tuple(protocol.get("arms", ())) != ARMS or tuple(protocol.get("replicas", ())) != REPLICAS:
        raise ValueError("BOMB-safe500 arms or replicas changed")
    if tuple(protocol.get("snapshot_rounds", ())) != SNAPSHOT_ROUNDS:
        raise ValueError("BOMB-safe500 snapshots changed")
    if protocol.get("single_training_variable") != "safe_bomb_kill_reward_12_vs_500":
        raise ValueError("BOMB-safe500 experiment must change one variable")
    expected_learning = {
        "architecture": "unchanged exact-v4 model-a-mlp-v4-global7",
        "features": "unchanged 96-dimensional exact-v4 latent",
        "action_mask": "unchanged exact-v4 legal mask",
        "reward_difference": "only safe BOMB-attributed KILLED_OPPONENT is 12 versus 500",
        "safe_kill_definition": "owned BOMB kill in six-transition window with zero KILLED_SELF in that window",
        "trade_definition": "any BOMB window containing both KILLED_OPPONENT and KILLED_SELF receives zero kill reward and unchanged self penalties",
        "non_bomb_rewards": "unchanged exact-v4 one-step rewards",
        "all_targets": "BOMB six-step; non-BOMB one-step in both arms",
        "uniform_replay": "unchanged capacity 50000 and batch 64; reset at boundary",
        "optimizer": "resume source-r2 Adam state unchanged",
        "epsilon": "resume source-r2 at unchanged 0.05 floor",
        "teacher_or_oracle": "none",
    }
    if protocol.get("learning_contract") != expected_learning:
        raise ValueError("BOMB-safe500 learning contract changed")
    expected_target = {
        "gamma": 0.99, "bomb_maturation_steps_both_arms": 6,
        "bomb_return_steps_both_arms": 6, "non_bomb_return_steps_both_arms": 1,
        "allowed_event_lag_steps": [4, 5], "terminal_truncation": "same-episode only",
        "terminal_event_delivery": "posthumous owned-bomb kills attach to the last terminal transition",
    }
    if protocol.get("target_contract") != expected_target:
        raise ValueError("BOMB-safe500 target contract changed")
    if protocol.get("reward_profiles") != {
        "safe12": {"SAFE_BOMB_KILL": 12.0, "TRADE_KILL": 0.0},
        "safe500": {"SAFE_BOMB_KILL": 500.0, "TRADE_KILL": 0.0},
    }:
        raise ValueError("BOMB-safe500 reward profiles changed")
    seen: set[int] = set()
    training = protocol.get("training", {})
    if (training.get("scenario") != "classic" or training.get("opponents") != ["seeded_rule_based_agent"] * 3
            or int(training.get("rounds_per_arm", 0)) != 25 or tuple(training.get("seeds_by_replica", {})) != REPLICAS):
        raise ValueError("BOMB-safe500 training distribution or budget changed")
    for replica, case in training["seeds_by_replica"].items():
        _register(seen, case, f"training/{replica}")
    evaluation = protocol.get("evaluation", {})
    if tuple(evaluation.get("strata", {})) != EVALUATION_STRATA:
        raise ValueError("BOMB-safe500 evaluation strata changed")
    for stratum, spec in evaluation["strata"].items():
        scenario, opponents = EXPECTED_DISTRIBUTIONS[stratum]
        expected_cases = 2 if stratum == "task4c_three_rule" else 1
        if spec.get("scenario") != scenario or spec.get("opponents") != opponents:
            raise ValueError(f"BOMB-safe500 evaluation distribution changed: {stratum}")
        if int(spec.get("rounds_per_case", 0)) != 25 or len(spec.get("cases", ())) != expected_cases:
            raise ValueError(f"BOMB-safe500 evaluation budget changed: {stratum}")
        for index, case in enumerate(spec["cases"]):
            _register(seen, case, f"evaluation/{stratum}/{index}")
    if len(seen) != 21:
        raise ValueError("BOMB-safe500 protocol must register exactly 21 unique seeds")
    expected_decision = {
        "minimum_parameter_divergent_pairs_before_evaluation": 2,
        "minimum_supportive_pairs": 2,
        "minimum_safe_kill_events_per_arm_replica": 1,
        "pair_minimum_task4c_kills_gain": 0.000000001,
        "pair_minimum_task4c_score_delta": -0.25,
        "pair_maximum_task4c_suicide_delta": 0.05,
        "pooled_minimum_task4c_kills_gain": 0.04,
        "pooled_minimum_task4c_score_delta": -0.10,
        "pooled_maximum_task4c_suicide_delta": 0.05,
        "pooled_minimum_task4c_bomb_fraction_of_control": 0.80,
        "minimum_task1_fraction_of_source": 0.80,
        "maximum_task2_score_drop_from_source": 0.75,
        "no_effect_decision": "reward500_no_replicated_parameter_effect_stop",
        "pass_decision": "reward500_signal_supported_stop_before_expansion",
        "fail_decision": "reward500_signal_not_supported_stop",
    }
    if evaluation.get("decision_rule") != expected_decision:
        raise ValueError("BOMB-safe500 decision rule changed")
    if protocol.get("automatic_followup") is not False:
        raise ValueError("BOMB-safe500 gate must always stop")
    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("BOMB-safe500 source binding set mismatch")
    for relative_path, expected_hash in protocol["source_bindings"].items():
        source = ROOT / relative_path
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"BOMB-safe500 source binding mismatch: {relative_path}")
    for field in ("source_parent", "frozen_v4_baseline", "prior_reward80_report"):
        item = protocol.get(field, {})
        artifact = ROOT / item.get("path", "missing")
        if not artifact.is_file() or sha256_file(artifact) != item.get("sha256"):
            raise ValueError(f"BOMB-safe500 bound artifact mismatch: {field}")
    prior = json.loads((ROOT / protocol["prior_reward80_report"]["path"]).read_text(encoding="utf-8"))
    if prior.get("result", {}).get("decision") != "reward80_signal_not_supported_stop":
        raise ValueError("prior reward80 evidence changed")
    lineage = protocol["source_parent"].get("lineage", {})
    lineage_protocol = ROOT / lineage.get("protocol_path", "missing")
    if not lineage_protocol.is_file() or sha256_file(lineage_protocol) != lineage.get("protocol_sha256"):
        raise ValueError("BOMB-safe500 source lineage changed")
    return protocol, sha256_file(path)


def checkpoint_root(protocol: dict, arm: str, replica: str) -> Path:
    if arm not in ARMS or replica not in REPLICAS:
        raise ValueError("invalid BOMB-safe500 checkpoint identity")
    return (ROOT / protocol["checkpoint_directory"] / arm / replica).resolve()


def final_checkpoint_path(protocol: dict, arm: str, replica: str) -> Path:
    return checkpoint_root(protocol, arm, replica) / "final.pt"


def snapshot_path(protocol: dict, arm: str, replica: str, stage_round: int) -> Path:
    if stage_round not in SNAPSHOT_ROUNDS:
        raise ValueError("invalid BOMB-safe500 snapshot round")
    return checkpoint_root(protocol, arm, replica) / "snapshots" / f"round-{stage_round:04d}.pt"
