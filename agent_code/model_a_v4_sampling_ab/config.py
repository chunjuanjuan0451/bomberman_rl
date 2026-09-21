"""Frozen protocol helpers for the n=8 replay-sampling-only A/B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("uniform_n8", "stratified_n8")
REPLICAS = ("r1", "r2", "r3")
ENDPOINT_ROUND = 100
EVALUATION_STRATA = (
    "task1", "task2", "task3_peaceful", "task3_coin", "task4b_duel", "task4c_three_rule",
)
EXPECTED_DISTRIBUTIONS = {
    "task1": ("coin-heaven", []),
    "task2": ("classic", []),
    "task3_peaceful": ("classic", ["seeded_peaceful_agent"]),
    "task3_coin": ("classic", ["seeded_coin_collector_agent"]),
    "task4b_duel": ("classic", ["seeded_rule_based_agent"]),
    "task4c_three_rule": ("classic", ["seeded_rule_based_agent"] * 3),
}
SOURCE_PATHS = {
    "agents.py", "environment.py", "events.py", "settings.py",
    "agent_code/model_a_dqn/callbacks.py", "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py", "agent_code/model_a_dqn/train.py",
    "agent_code/model_a_v4_curriculum/config.py", "agent_code/model_a_v4_curriculum/callbacks.py",
    "agent_code/model_a_v7a/planner.py", "agent_code/model_a_v7b/tactical.py",
    "agent_code/model_a_v4_sampling_ab/__init__.py", "agent_code/model_a_v4_sampling_ab/config.py",
    "agent_code/model_a_v4_sampling_ab/callbacks.py",
    "agent_code/model_a_v4_sampling_ab/replay.py", "agent_code/model_a_v4_sampling_ab/train.py",
    "agent_code/seeded_peaceful_agent/callbacks.py", "agent_code/seeded_coin_collector_agent/callbacks.py",
    "agent_code/seeded_rule_based_agent/callbacks.py", "tools/v4_task4_duel_training.py",
    "tools/v4_sampling_distribution_ab.py",
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
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-sampling-distribution-ab":
        raise ValueError("wrong sampling-distribution protocol")
    if tuple(protocol.get("arms", ())) != ARMS or tuple(protocol.get("replicas", ())) != REPLICAS:
        raise ValueError("sampling-distribution identities changed")
    if protocol.get("single_training_variable") != "replay_sampling_uniform_vs_reserved_2_kill_2_self":
        raise ValueError("sampling-distribution experiment must change only replay sampling")
    expected_learning = {
        "architecture": "unchanged exact-v4 model-a-mlp-v4-global7",
        "features": "unchanged 96-dimensional exact-v4 latent",
        "action_mask": "unchanged exact-v4 legal mask",
        "reward_and_shaping": "unchanged exact-v4 values and events",
        "all_action_return_horizon": 8,
        "gamma": 0.99,
        "replay_capacity": 50000,
        "batch_size": 64,
        "uniform_arm_batch": {"uniform": 64, "kill_chain": 0, "self_kill_chain": 0},
        "stratified_arm_batch": {"uniform": 60, "kill_chain": 2, "self_kill_chain": 2},
        "kill_chain_definition": "same-episode n=8 target includes official KILLED_OPPONENT and excludes KILLED_SELF",
        "self_kill_chain_definition": "same-episode n=8 target includes official KILLED_SELF",
        "bucket_shortage": "reserved slot falls back to uniform without replacement and is logged",
        "optimizer": "resume source-r2 Adam state unchanged",
        "epsilon": "resume source-r2 at unchanged 0.05 floor",
        "loss": "unchanged Smooth-L1 beta=1 mean",
        "teacher_or_oracle_in_training": "none",
    }
    if protocol.get("learning_contract") != expected_learning:
        raise ValueError("sampling-distribution learning contract changed")
    training = protocol.get("training", {})
    if (
        training.get("scenario") != "classic"
        or training.get("opponents") != ["seeded_rule_based_agent"] * 3
        or int(training.get("rounds_per_arm", 0)) != ENDPOINT_ROUND
        or training.get("selection") != "fixed_round_100_only_no_snapshot_search"
        or tuple(training.get("seeds_by_replica", {})) != REPLICAS
    ):
        raise ValueError("sampling-distribution training contract changed")
    seen: set[int] = set()
    for replica, case in training["seeds_by_replica"].items():
        _register(seen, case, f"training/{replica}")
    evaluation = protocol.get("evaluation", {})
    expected_labels = [
        "v4", "source-r2",
        *[f"{arm}-{replica}" for replica in REPLICAS for arm in ARMS],
    ]
    if evaluation.get("labels") != expected_labels:
        raise ValueError("sampling-distribution evaluation labels changed")
    if tuple(evaluation.get("strata", {})) != EVALUATION_STRATA:
        raise ValueError("sampling-distribution evaluation strata changed")
    for stratum, spec in evaluation["strata"].items():
        scenario, opponents = EXPECTED_DISTRIBUTIONS[stratum]
        expected_cases = 2 if stratum.startswith("task4") else 1
        if (
            spec.get("scenario") != scenario
            or spec.get("opponents") != opponents
            or int(spec.get("rounds_per_case", 0)) != 25
            or len(spec.get("cases", ())) != expected_cases
        ):
            raise ValueError(f"sampling-distribution evaluation contract changed: {stratum}")
        for index, case in enumerate(spec["cases"]):
            _register(seen, case, f"evaluation/{stratum}/{index}")
    if len(seen) != 33:
        raise ValueError("sampling-distribution protocol must register 33 unique seed values")
    expected_rule = {
        "minimum_supportive_pairs": 2,
        "pair_minimum_combined_linked_kills_gain": 1,
        "pair_minimum_combined_threat_to_kill_gain": 0.000001,
        "pooled_minimum_combined_kills_per_round_gain_over_uniform": 0.03,
        "pooled_minimum_combined_threat_to_kill_gain_over_uniform": 0.001,
        "pooled_minimum_combined_kills_per_round_gain_over_source": 0.02,
        "pooled_minimum_combined_threat_to_kill_gain_over_source": 0.0005,
        "pooled_minimum_combined_score_delta_to_uniform": -0.10,
        "pooled_maximum_combined_suicide_delta_to_uniform": 0.05,
        "pooled_minimum_combined_score_delta_to_source": -0.25,
        "pooled_maximum_combined_suicide_delta_to_source": 0.05,
        "minimum_task1_fraction_of_source": 0.90,
        "maximum_task2_score_drop_from_source": 0.50,
        "maximum_task3_score_drop_from_source": 0.50,
        "maximum_task3_kills_drop_from_source": 0.05,
        "maximum_task3_suicide_increase_over_source": 0.05,
        "maximum_task4_stratum_kills_drop_from_source": 0.02,
        "minimum_task4_stratum_score_delta_to_source": -0.50,
        "maximum_task4_stratum_suicide_delta_to_source": 0.05,
        "pass_decision": "stratified_replay_supported_stop_before_expansion",
        "fail_decision": "stratified_replay_not_supported_stop",
        "no_effect_decision": "sampling_intervention_no_parameter_effect_stop",
    }
    if evaluation.get("decision_rule") != expected_rule:
        raise ValueError("sampling-distribution decision rule changed")
    if protocol.get("parameter_effect_gate") != {
        "minimum_divergent_endpoint_pairs": 2,
        "on_failure": "skip all evaluation and stop as no demonstrated sampling intervention effect",
    }:
        raise ValueError("sampling-distribution parameter-effect gate changed")
    if protocol.get("automatic_followup") is not False:
        raise ValueError("sampling-distribution experiment must stop")
    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("sampling-distribution source binding set mismatch")
    for relative, expected in protocol["source_bindings"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"sampling-distribution source binding mismatch: {relative}")
    for field in ("source_parent", "frozen_v4_baseline", "diagnostic_report"):
        item = protocol.get(field, {})
        artifact = ROOT / item.get("path", "missing")
        if not artifact.is_file() or sha256_file(artifact) != item.get("sha256"):
            raise ValueError(f"sampling-distribution bound artifact mismatch: {field}")
    diagnostic = json.loads((ROOT / protocol["diagnostic_report"]["path"]).read_text(encoding="utf-8"))
    if (
        diagnostic.get("status") != "completed"
        or diagnostic.get("phase_b_offline_analysis", {}).get("conclusions", {}).get("decision")
        != "diagnostic_complete_stop_before_any_training"
        or diagnostic.get("training_started") is not False
    ):
        raise ValueError("sampling-distribution prerequisite diagnostic is not terminal and valid")
    conclusions = diagnostic.get("phase_b_offline_analysis", {}).get("conclusions", {})
    if conclusions.get("minimum_tested_horizon_covering_95pct_unique_linked_bomb_kills") != 8:
        raise ValueError("sampling-distribution experiment lacks the preregistered n=8 evidence")
    return protocol, sha256_file(path)


def checkpoint_path(protocol: dict, arm: str, replica: str) -> Path:
    if arm not in ARMS or replica not in REPLICAS:
        raise ValueError("invalid sampling-distribution checkpoint identity")
    return (ROOT / protocol["checkpoint_directory"] / arm / replica / "round-0100.pt").resolve()


def evaluation_diagnostic_path(protocol: dict, stratum: str, label: str, world_seed: int) -> Path:
    if stratum not in EVALUATION_STRATA:
        raise ValueError("invalid sampling-distribution evaluation stratum")
    return (ROOT / protocol["evaluation_diagnostic_directory"] / stratum / f"{label}-s{world_seed}.json").resolve()
