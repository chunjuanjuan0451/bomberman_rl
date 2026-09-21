"""Frozen protocol contract for direct Task4C training from source-r2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPLICAS = ("r1", "r2", "r3")
SNAPSHOT_ROUNDS = (25, 50, 75, 100, 125, 150, 175, 200)
OUTER_STRATA = (
    "task1", "task2", "task3_peaceful", "task3_coin",
    "task4a_open_rule", "task4b_classic_duel", "task4c_three_rule",
)
EXPECTED_DISTRIBUTIONS = {
    "task1": ("coin-heaven", []),
    "task2": ("classic", []),
    "task3_peaceful": ("classic", ["seeded_peaceful_agent"]),
    "task3_coin": ("classic", ["seeded_coin_collector_agent"]),
    "task4a_open_rule": ("coin-heaven", ["seeded_rule_based_agent"]),
    "task4b_classic_duel": ("classic", ["seeded_rule_based_agent"]),
    "task4c_three_rule": ("classic", ["seeded_rule_based_agent"] * 3),
}
SOURCE_PATHS = {
    "agent_code/model_a_dqn/callbacks.py",
    "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py",
    "agent_code/model_a_dqn/train.py",
    "agent_code/model_a_dqn/replay_buffer.py",
    "agent_code/model_a_v4_curriculum/config.py",
    "agent_code/model_a_v4_curriculum/callbacks.py",
    "agent_code/model_a_v4_task4c/config.py",
    "agent_code/model_a_v4_task4c/callbacks.py",
    "agent_code/model_a_v4_task4c/train.py",
    "agent_code/seeded_peaceful_agent/callbacks.py",
    "agent_code/seeded_coin_collector_agent/callbacks.py",
    "agent_code/seeded_rule_based_agent/callbacks.py",
    "tools/v4_task4_duel_training.py",
    "tools/v4_task4c_direct.py",
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
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-task4c-direct":
        raise ValueError("wrong direct Task4C protocol")
    if protocol.get("active_subcourse") != "task4c_three_rule":
        raise ValueError("this runner may execute Task4C only")
    if tuple(protocol.get("replicas", ())) != REPLICAS:
        raise ValueError("Task4C requires exactly r1/r2/r3")
    if tuple(protocol.get("snapshot_rounds", ())) != SNAPSHOT_ROUNDS:
        raise ValueError("Task4C snapshots must be every 25 rounds through 200")
    if protocol.get("single_training_variable") != "task4c_episode_distribution":
        raise ValueError("Task4C may change only the episode distribution")
    expected_learning = {
        "architecture": "unchanged exact-v4 model-a-mlp-v4-global7",
        "features": "unchanged",
        "action_mask": "unchanged",
        "reward_and_shaping": "unchanged",
        "uniform_replay": "unchanged and reset at the Task4C boundary",
        "optimizer": "resume source-r2 optimizer state unchanged",
        "epsilon": "resume source-r2 schedule at the unchanged 0.05 floor",
    }
    if protocol.get("learning_contract") != expected_learning:
        raise ValueError("exact-v4 learning contract changed")

    seen: set[int] = set()
    training = protocol.get("training", {})
    if (
        training.get("scenario") != "classic"
        or training.get("opponents") != ["seeded_rule_based_agent"] * 3
        or int(training.get("rounds", 0)) != 200
        or tuple(training.get("seeds", {})) != REPLICAS
    ):
        raise ValueError("Task4C training distribution or budget changed")
    for replica, case in training["seeds"].items():
        _register(seen, case, f"training/{replica}")

    inner = protocol.get("inner_signal", {})
    if tuple(inner.get("cases_by_replica", {})) != REPLICAS or inner.get("scenario") != "classic":
        raise ValueError("Task4C inner replica cases changed")
    if inner.get("opponents") != ["seeded_rule_based_agent"] * 3 or int(inner.get("rounds_per_case", 0)) != 10:
        raise ValueError("Task4C inner distribution or budget changed")
    for replica, cases in inner["cases_by_replica"].items():
        if len(cases) != 2:
            raise ValueError(f"Task4C inner requires two cases: {replica}")
        for index, case in enumerate(cases):
            _register(seen, case, f"inner/{replica}/{index}")
    if inner.get("supportive_replicas_required") != 2:
        raise ValueError("Task4C requires two supportive replicas")
    expected_inner = {
        "minimum_score_gain_over_source": 0.25,
        "minimum_score_delta_to_v4": 0.0,
        "minimum_kills_delta_to_source": 0.0,
        "maximum_suicide_increase_over_v4": 0.05,
        "minimum_score_margin_over_rule_mean": 0.0,
    }
    if inner.get("supportive_rule") != expected_inner:
        raise ValueError("Task4C inner signal rule changed")

    outer = protocol.get("outer_confirmation", {})
    if tuple(outer.get("strata", {})) != OUTER_STRATA:
        raise ValueError("Task4C outer strata changed")
    for stratum, spec in outer["strata"].items():
        scenario, opponents = EXPECTED_DISTRIBUTIONS[stratum]
        expected_cases = 4 if stratum == "task4c_three_rule" else 2
        if spec.get("scenario") != scenario or spec.get("opponents") != opponents:
            raise ValueError(f"Task4C outer distribution changed: {stratum}")
        if int(spec.get("rounds_per_case", 0)) != 25 or len(spec.get("cases", ())) != expected_cases:
            raise ValueError(f"Task4C outer budget changed: {stratum}")
        for index, case in enumerate(spec["cases"]):
            _register(seen, case, f"outer/{stratum}/{index}")
    if len(seen) != 75:
        raise ValueError("Task4C must register exactly 75 unique seeds")
    expected_outer = {
        "minimum_task1_fraction_of_v4": 0.9,
        "maximum_task2_score_drop_from_source": 0.5,
        "maximum_task3_score_drop_from_source": 0.5,
        "maximum_task3_kills_drop_from_source": 0.05,
        "maximum_task3_suicide_increase_over_source": 0.05,
        "maximum_duel_score_drop_from_source": 0.25,
        "maximum_duel_suicide_increase_over_source": 0.05,
        "minimum_three_rule_score_gain_over_each_baseline": 0.15,
        "minimum_three_rule_kills_delta_to_source": 0.0,
        "maximum_three_rule_suicide_increase_over_v4": 0.05,
        "minimum_three_rule_score_margin_over_rule_mean": 0.0,
        "minimum_three_rule_case_blocks_not_below_both_baselines": 2,
        "pass_decision": "task4c_confirmed_complete_stop",
        "fail_decision": "task4c_candidate_rejected_stop",
    }
    if outer.get("decision_rule") != expected_outer:
        raise ValueError("Task4C outer decision rule changed")
    if protocol.get("automatic_followup") is not False:
        raise ValueError("Task4C must always stop")

    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("Task4C source binding set mismatch")
    for relative_path, expected_hash in protocol["source_bindings"].items():
        source = ROOT / relative_path
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"Task4C source binding mismatch: {relative_path}")
    for field in ("source_parent", "frozen_v4_baseline", "previous_confirmation_report"):
        item = protocol.get(field, {})
        artifact = ROOT / item.get("path", "missing")
        if not artifact.is_file() or sha256_file(artifact) != item.get("sha256"):
            raise ValueError(f"Task4C bound artifact mismatch: {field}")
    previous = json.loads((ROOT / protocol["previous_confirmation_report"]["path"]).read_text(encoding="utf-8"))
    if previous.get("result", {}).get("decision") != "r3_round100_rejected_retain_source_r2_stop":
        raise ValueError("Task4C is not bound to the rejected r3 confirmation")
    lineage = protocol["source_parent"].get("lineage", {})
    clean_protocol = ROOT / lineage.get("protocol_path", "missing")
    if not clean_protocol.is_file() or sha256_file(clean_protocol) != lineage.get("protocol_sha256"):
        raise ValueError("Task4C source-parent lineage mismatch")
    return protocol, sha256_file(path)


def checkpoint_root(protocol: dict, replica: str) -> Path:
    if replica not in REPLICAS:
        raise ValueError("invalid Task4C replica")
    return (ROOT / protocol["checkpoint_directory"] / replica).resolve()


def final_checkpoint_path(protocol: dict, replica: str) -> Path:
    return checkpoint_root(protocol, replica) / "final.pt"


def snapshot_path(protocol: dict, replica: str, stage_round: int) -> Path:
    if stage_round not in SNAPSHOT_ROUNDS:
        raise ValueError("invalid Task4C snapshot round")
    return checkpoint_root(protocol, replica) / "snapshots" / f"round-{stage_round:04d}.pt"


def selected_checkpoint_path(protocol: dict) -> Path:
    return (ROOT / protocol["checkpoint_directory"] / "selected-task4c.pt").resolve()


def confirmed_checkpoint_path(protocol: dict) -> Path:
    return (ROOT / protocol["confirmed_checkpoint_path"]).resolve()
