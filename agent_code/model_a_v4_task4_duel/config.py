"""Contract helpers for the single-variable exact-v4 Task4 experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPLICAS = ("r1", "r2", "r3")
SNAPSHOT_ROUNDS = (100, 200, 300, 400, 500, 600)
STRATA = (
    "task1", "task2", "task3_peaceful", "task3_coin",
    "task4_rule_duel", "task4_three_rule",
)
EXPECTED_DISTRIBUTIONS = {
    "task1": ("coin-heaven", []),
    "task2": ("classic", []),
    "task3_peaceful": ("classic", ["seeded_peaceful_agent"]),
    "task3_coin": ("classic", ["seeded_coin_collector_agent"]),
    "task4_rule_duel": ("classic", ["seeded_rule_based_agent"]),
    "task4_three_rule": ("classic", ["seeded_rule_based_agent"] * 3),
}
SOURCE_PATHS = {
    "agent_code/model_a_dqn/callbacks.py",
    "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py",
    "agent_code/model_a_dqn/train.py",
    "agent_code/model_a_dqn/replay_buffer.py",
    "agent_code/model_a_v4_curriculum/config.py",
    "agent_code/model_a_v4_curriculum/callbacks.py",
    "agent_code/model_a_v4_task4_duel/config.py",
    "agent_code/model_a_v4_task4_duel/callbacks.py",
    "agent_code/model_a_v4_task4_duel/train.py",
    "agent_code/seeded_peaceful_agent/callbacks.py",
    "agent_code/seeded_coin_collector_agent/callbacks.py",
    "agent_code/seeded_rule_based_agent/callbacks.py",
    "tools/v4_task4_duel_training.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _register_seed_tuple(seen: set[int], seeds: dict, identity: str) -> None:
    if set(seeds) != {"world_seed", "agent_seed", "opponent_seed"}:
        raise ValueError(f"invalid seed tuple: {identity}")
    for raw_value in seeds.values():
        value = int(raw_value)
        if value <= 0 or value in seen:
            raise ValueError(f"seeds must be positive and globally unique: {identity}")
        seen.add(value)


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-task4-duel-training":
        raise ValueError("wrong Task4 duel-training protocol")
    if tuple(protocol.get("replicas", ())) != REPLICAS:
        raise ValueError("Task4 requires exactly the ordered replicas r1/r2/r3")
    if tuple(protocol.get("snapshot_rounds", ())) != SNAPSHOT_ROUNDS:
        raise ValueError("Task4 immutable snapshots must be rounds 100..600")
    if protocol.get("single_training_variable") != "opponent_sampling_distribution":
        raise ValueError("Task4 may change only opponent sampling distribution")
    if protocol.get("learning_contract") != {
        "architecture": "unchanged exact-v4 model-a-mlp-v4-global7",
        "features": "unchanged",
        "action_mask": "unchanged",
        "reward_and_shaping": "unchanged",
        "uniform_replay": "unchanged and reset at Task4 boundary",
        "optimizer": "resume source-r2 optimizer state unchanged",
        "epsilon": "resume source-r2 schedule at the unchanged 0.05 floor",
    }:
        raise ValueError("exact-v4 learning contract changed")

    seen: set[int] = set()
    training = protocol.get("training", {})
    if (
        training.get("scenario") != "classic"
        or training.get("opponents") != ["seeded_rule_based_agent"]
        or int(training.get("rounds", 0)) != 600
        or tuple(training.get("seeds", {})) != REPLICAS
    ):
        raise ValueError("Task4 training distribution or budget changed")
    for replica, seeds in training["seeds"].items():
        _register_seed_tuple(seen, seeds, f"training/{replica}")

    for suite_name, expected_rounds in (("inner_selection", 10), ("outer_confirmation", 25)):
        suite = protocol.get(suite_name, {})
        strata = suite.get("strata", {})
        if tuple(strata) != STRATA:
            raise ValueError(f"{suite_name} strata changed")
        for stratum, spec in strata.items():
            scenario, opponents = EXPECTED_DISTRIBUTIONS[stratum]
            if spec.get("scenario") != scenario or spec.get("opponents") != opponents:
                raise ValueError(f"{suite_name}/{stratum} distribution changed")
            if int(spec.get("rounds_per_case", 0)) != expected_rounds or len(spec.get("cases", ())) != 2:
                raise ValueError(f"{suite_name}/{stratum} budget changed")
            for index, case in enumerate(spec["cases"]):
                _register_seed_tuple(seen, case, f"{suite_name}/{stratum}/{index}")
    if len(seen) != 81:
        raise ValueError("Task4 protocol must register exactly 81 unique seeds")

    if protocol.get("inner_selection", {}).get("supportive_replicas_required") != 2:
        raise ValueError("Task4 requires at least two supportive replicas")
    if protocol.get("automatic_followup") is not False:
        raise ValueError("Task4 runner must stop after its terminal report")
    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("Task4 source binding set mismatch")
    for relative_path, expected_hash in protocol["source_bindings"].items():
        source = ROOT / relative_path
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"Task4 source binding mismatch: {relative_path}")

    for field in ("source_parent", "frozen_v4_baseline", "baseline_gate"):
        item = protocol.get(field, {})
        artifact = ROOT / item.get("path", "missing")
        if not artifact.is_file() or sha256_file(artifact) != item.get("sha256"):
            raise ValueError(f"Task4 bound artifact mismatch: {field}")
    lineage = protocol["source_parent"].get("lineage", {})
    clean_protocol = ROOT / lineage.get("protocol_path", "missing")
    if not clean_protocol.is_file() or sha256_file(clean_protocol) != lineage.get("protocol_sha256"):
        raise ValueError("Task4 source-parent clean protocol mismatch")
    baseline = json.loads((ROOT / protocol["baseline_gate"]["path"]).read_text(encoding="utf-8"))
    if (
        baseline.get("status") != "completed"
        or baseline.get("result", {}).get("decision") != "task4_training_required"
        or baseline.get("result", {}).get("training_started") is not False
    ):
        raise ValueError("Task4 baseline does not authorize the preregistered training step")
    return protocol, sha256_file(path)


def checkpoint_root(protocol: dict, replica: str) -> Path:
    if replica not in REPLICAS:
        raise ValueError("invalid Task4 replica")
    return (ROOT / protocol["checkpoint_directory"] / replica).resolve()


def final_checkpoint_path(protocol: dict, replica: str) -> Path:
    return checkpoint_root(protocol, replica) / "final.pt"


def snapshot_path(protocol: dict, replica: str, stage_round: int) -> Path:
    if stage_round not in SNAPSHOT_ROUNDS:
        raise ValueError("invalid Task4 snapshot round")
    return checkpoint_root(protocol, replica) / "snapshots" / f"round-{stage_round:04d}.pt"


def selected_checkpoint_path(protocol: dict) -> Path:
    return (ROOT / protocol["checkpoint_directory"] / "selected.pt").resolve()


def confirmed_checkpoint_path(protocol: dict) -> Path:
    return (ROOT / protocol["confirmed_checkpoint_path"]).resolve()
