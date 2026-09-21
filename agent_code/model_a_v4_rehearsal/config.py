"""Frozen protocol helpers for the source-r2 rehearsal pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("no_rehearsal", "rehearsal25")
REPLICAS = ("r1", "r2", "r3")
OLD_STRATA = ("task1", "task2", "task3_peaceful", "task3_coin")
EVAL_STRATA = (*OLD_STRATA, "task4b_duel", "task4c_three_rule")
ENDPOINT_ROUND = 100
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
    "agent_code/model_a_v4_rehearsal/__init__.py", "agent_code/model_a_v4_rehearsal/config.py",
    "agent_code/model_a_v4_rehearsal/callbacks.py", "agent_code/model_a_v4_rehearsal/replay.py",
    "agent_code/model_a_v4_rehearsal/train.py", "agent_code/model_a_v4_sampling_ab/replay.py",
    "agent_code/model_a_v4_sampling_ab/train.py", "agent_code/model_a_v7b/tactical.py",
    "agent_code/seeded_peaceful_agent/callbacks.py", "agent_code/seeded_coin_collector_agent/callbacks.py",
    "agent_code/seeded_rule_based_agent/callbacks.py", "tools/v4_task4_duel_training.py",
    "tools/v4_rehearsal_pilot.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _register(seen: set[int], case: dict, identity: str) -> None:
    if set(case) != {"world_seed", "agent_seed", "opponent_seed"}:
        raise ValueError(f"invalid rehearsal seed tuple: {identity}")
    for raw in case.values():
        value = int(raw)
        if value <= 0 or value in seen:
            raise ValueError(f"rehearsal seeds must be positive and globally unique: {identity}")
        seen.add(value)


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-rehearsal-pilot":
        raise ValueError("wrong rehearsal protocol")
    if tuple(protocol.get("arms", ())) != ARMS or tuple(protocol.get("replicas", ())) != REPLICAS:
        raise ValueError("rehearsal identities changed")
    if protocol.get("single_training_variable") != "fixed_25pct_source_task1_to_task3_rehearsal":
        raise ValueError("rehearsal pilot must change only old-task replay share")
    expected_learning = {
        "architecture": "unchanged exact-v4 model-a-mlp-v4-global7",
        "features": "unchanged 96-dimensional exact-v4 latent",
        "action_mask": "unchanged exact-v4 legal mask",
        "reward_and_shaping": "unchanged exact-v4 values and events",
        "all_action_return_horizon": 8,
        "gamma": 0.99,
        "replay_capacity": 50000,
        "batch_size": 64,
        "no_rehearsal_batch": {"task4_uniform": 60, "task4_kill": 2, "task4_self": 2, "old_task": 0},
        "rehearsal25_batch": {
            "task4_uniform": 44, "task4_kill": 2, "task4_self": 2,
            "task1": 4, "task2": 4, "task3_peaceful": 4, "task3_coin": 4,
        },
        "kill_chain_definition": "same-episode n=8 target includes official KILLED_OPPONENT and excludes KILLED_SELF",
        "self_kill_chain_definition": "same-episode n=8 target includes official KILLED_SELF",
        "optimizer": "resume source-r2 Adam state unchanged",
        "epsilon": "resume source-r2 at unchanged 0.05 floor",
        "loss": "unchanged Smooth-L1 beta=1 mean",
        "teacher_or_oracle_in_training": "none",
    }
    if protocol.get("learning_contract") != expected_learning:
        raise ValueError("rehearsal learning contract changed")
    seen: set[int] = set()
    collection = protocol.get("rehearsal_collection", {})
    if int(collection.get("rounds_per_stratum", 0)) != 25 or tuple(collection.get("strata", {})) != OLD_STRATA:
        raise ValueError("rehearsal collection contract changed")
    for stratum, spec in collection["strata"].items():
        scenario, opponents = EXPECTED_DISTRIBUTIONS[stratum]
        if spec.get("scenario") != scenario or spec.get("opponents") != opponents or tuple(spec.get("seeds_by_replica", {})) != REPLICAS:
            raise ValueError(f"rehearsal collection distribution changed: {stratum}")
        for replica, case in spec["seeds_by_replica"].items():
            _register(seen, case, f"collection/{stratum}/{replica}")
    training = protocol.get("training", {})
    if (
        training.get("scenario") != "classic"
        or training.get("opponents") != ["seeded_rule_based_agent"] * 3
        or int(training.get("rounds_per_arm", 0)) != ENDPOINT_ROUND
        or training.get("selection") != "fixed_round_100_only"
        or tuple(training.get("seeds_by_replica", {})) != REPLICAS
    ):
        raise ValueError("rehearsal training contract changed")
    for replica, case in training["seeds_by_replica"].items():
        _register(seen, case, f"training/{replica}")
    evaluation = protocol.get("evaluation", {})
    for suite, expected_cases in (("validation", (1, 1, 1, 1, 2, 2)), ("confirmation", (1, 1, 1, 1, 2, 2))):
        strata = evaluation.get(suite, {}).get("strata", {})
        if tuple(strata) != EVAL_STRATA:
            raise ValueError(f"rehearsal {suite} strata changed")
        for stratum, count in zip(EVAL_STRATA, expected_cases):
            spec = strata[stratum]
            scenario, opponents = EXPECTED_DISTRIBUTIONS[stratum]
            if spec.get("scenario") != scenario or spec.get("opponents") != opponents or int(spec.get("rounds_per_case", 0)) != 25 or len(spec.get("cases", ())) != count:
                raise ValueError(f"rehearsal {suite} contract changed: {stratum}")
            for index, case in enumerate(spec["cases"]):
                _register(seen, case, f"{suite}/{stratum}/{index}")
    if len(seen) != 93:
        raise ValueError("rehearsal protocol must register 93 unique seed values")
    expected_validation_rule = {
        "minimum_supportive_pairs": 2, "pair_minimum_retention_index_gain": 0.05,
        "pooled_minimum_task1_score_gain": 3.0, "pooled_minimum_task2_score_gain": 0.25,
        "pooled_minimum_task3_peaceful_score_gain": 0.5, "maximum_task4_kills_drop": 0.02,
        "maximum_task4_conversion_drop": 0.001, "maximum_task4_score_drop": 0.25,
        "maximum_task4_suicide_increase": 0.03,
    }
    expected_confirmation_rule = {
        "minimum_task1_fraction_of_source": 0.9, "maximum_task2_score_drop_from_source": 0.5,
        "maximum_task3_score_drop_from_source": 0.5, "maximum_task3_kills_drop_from_source": 0.05,
        "maximum_task3_suicide_increase_over_source": 0.05,
        "minimum_task4_kills_gain_over_source": 0.02,
        "minimum_task4_conversion_gain_over_source": 0.0005,
        "maximum_task4_score_drop_from_source": 0.25,
        "maximum_task4_suicide_increase_over_source": 0.05,
        "maximum_task4_stratum_kills_drop_from_source": 0.02,
        "maximum_task4_stratum_score_drop_from_source": 0.5,
        "maximum_task4_stratum_suicide_increase_over_source": 0.05,
    }
    if evaluation["validation"].get("decision_rule") != expected_validation_rule or evaluation["confirmation"].get("decision_rule") != expected_confirmation_rule:
        raise ValueError("rehearsal decision rule changed")
    if protocol.get("automatic_followup") is not False:
        raise ValueError("rehearsal experiment must stop")
    for field in ("source_parent", "frozen_v4_baseline", "prior_sampling_report"):
        item = protocol.get(field, {})
        artifact = ROOT / item.get("path", "missing")
        if not artifact.is_file() or sha256_file(artifact) != item.get("sha256"):
            raise ValueError(f"rehearsal bound artifact mismatch: {field}")
    prior = json.loads((ROOT / protocol["prior_sampling_report"]["path"]).read_text(encoding="utf-8"))
    if prior.get("status") != "completed" or prior.get("result", {}).get("decision") != "stratified_replay_not_supported_stop" or prior.get("result", {}).get("supportive_pairs") != 2:
        raise ValueError("rehearsal prerequisite sampling result changed")
    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("rehearsal source binding set mismatch")
    for relative, expected in protocol["source_bindings"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"rehearsal source binding mismatch: {relative}")
    return protocol, sha256_file(path)


def dataset_path(protocol: dict, replica: str, stratum: str) -> Path:
    if replica not in REPLICAS or stratum not in OLD_STRATA:
        raise ValueError("invalid rehearsal dataset identity")
    return (ROOT / protocol["rehearsal_dataset_directory"] / replica / f"{stratum}.pt").resolve()


def checkpoint_path(protocol: dict, arm: str, replica: str) -> Path:
    if arm not in ARMS or replica not in REPLICAS:
        raise ValueError("invalid rehearsal checkpoint identity")
    return (ROOT / protocol["checkpoint_directory"] / arm / replica / "round-0100.pt").resolve()


def evaluation_diagnostic_path(protocol: dict, suite: str, stratum: str, label: str, seed: int) -> Path:
    if suite not in {"validation", "confirmation"} or stratum not in EVAL_STRATA:
        raise ValueError("invalid rehearsal evaluation identity")
    return (ROOT / protocol["evaluation_diagnostic_directory"] / suite / stratum / f"{label}-s{seed}.json").resolve()
