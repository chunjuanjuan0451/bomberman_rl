"""Contract helpers for the exact-v4 retention-aware Task3 experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BRANCHES = ("b1", "b2", "b3")
STAGES = ("task3_peaceful", "task3_coin")
SOURCE_PATHS = {
    "agent_code/model_a_dqn/callbacks.py",
    "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py",
    "agent_code/model_a_dqn/train.py",
    "agent_code/model_a_dqn/replay_buffer.py",
    "agent_code/model_a_v4_curriculum/config.py",
    "agent_code/model_a_v4_curriculum/callbacks.py",
    "agent_code/model_a_v4_task3_retention/config.py",
    "agent_code/model_a_v4_task3_retention/callbacks.py",
    "agent_code/model_a_v4_task3_retention/train.py",
    "agent_code/seeded_peaceful_agent/callbacks.py",
    "agent_code/seeded_coin_collector_agent/callbacks.py",
    "tools/v4_task3_retention.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("Task3 retention protocol requires schema_version=1")
    if protocol.get("kind") != "model-a-v4-task3-retention":
        raise ValueError("wrong Task3 retention protocol kind")
    if tuple(protocol.get("branches", ())) != BRANCHES:
        raise ValueError("ordered Task3 branches must be b1/b2/b3")
    if tuple(protocol.get("stage_order", ())) != STAGES:
        raise ValueError("Task3 peaceful and coin stages must remain separate and ordered")
    if int(protocol.get("snapshot_every_rounds", 0)) != 50:
        raise ValueError("Task3 snapshots must be saved every 50 rounds")
    seen_seeds = set()

    def register_seed_tuple(seeds: dict, identity: str) -> None:
        if set(seeds) != {"world_seed", "agent_seed", "opponent_seed"}:
            raise ValueError(f"invalid seed tuple: {identity}")
        for raw_value in seeds.values():
            value = int(raw_value)
            if value <= 0 or value in seen_seeds:
                raise ValueError(f"all training/inner/outer seeds must be positive and globally unique: {identity}")
            seen_seeds.add(value)

    stages = protocol.get("training", {})
    if tuple(stages) != STAGES:
        raise ValueError("Task3 training stages changed")
    for stage in STAGES:
        spec = stages[stage]
        expected_opponent = {
            "task3_peaceful": "seeded_peaceful_agent",
            "task3_coin": "seeded_coin_collector_agent",
        }[stage]
        if (
            int(spec.get("rounds", 0)) != 600
            or spec.get("scenario") != "classic"
            or spec.get("opponents") != [expected_opponent]
        ):
            raise ValueError(f"Task3 training distribution changed: {stage}")
        if tuple(spec.get("branches", ())) != BRANCHES:
            raise ValueError(f"incomplete training seeds: {stage}")
        for branch, seeds in spec["branches"].items():
            register_seed_tuple(seeds, f"training/{stage}/{branch}")
    selection = protocol.get("inner_selection", {})
    if float(selection.get("current_task_best_fraction", -1)) != 0.9:
        raise ValueError("current-task eligibility fraction changed")
    if tuple(selection.get("current_task_metrics", ())) != (
        "score_per_round", "kills_per_round",
    ):
        raise ValueError("Task3 current-task score/kill eligibility metrics changed")
    expected_inner = {
        "task3_peaceful": ("task1", "task2", "task3_peaceful"),
        "task3_coin": ("task1", "task2", "task3_peaceful", "task3_coin"),
    }
    expected_prior_metrics = {
        "task3_peaceful": {
            "task1": ["score_per_round"],
            "task2": ["score_per_round"],
        },
        "task3_coin": {
            "task1": ["score_per_round"],
            "task2": ["score_per_round"],
            "task3_peaceful": ["score_per_round", "kills_per_round"],
        },
    }
    for stage, strata in expected_inner.items():
        stage_selection = selection.get("stages", {}).get(stage, {})
        if tuple(stage_selection.get("strata", ())) != strata:
            raise ValueError(f"inner retention strata changed: {stage}")
        if stage_selection.get("prior_retention_metrics") != expected_prior_metrics[stage]:
            raise ValueError(f"inner retention metrics changed: {stage}")
    expected_distributions = {
        "task1": ("coin-heaven", []),
        "task2": ("classic", []),
        "task3_peaceful": ("classic", ["seeded_peaceful_agent"]),
        "task3_coin": ("classic", ["seeded_coin_collector_agent"]),
    }
    suites = [selection["strata"], protocol["outer_evaluation"]["strata"]]
    for suite in suites:
        if tuple(suite) != tuple(expected_distributions):
            raise ValueError("Task3 retention evaluation strata changed")
        for name, spec in suite.items():
            scenario, opponents = expected_distributions[name]
            if spec.get("scenario") != scenario or spec.get("opponents") != opponents:
                raise ValueError(f"evaluation distribution changed: {name}")
            if int(spec.get("rounds_per_case", 0)) <= 0 or not spec.get("cases"):
                raise ValueError(f"invalid evaluation budget: {name}")
            for index, case in enumerate(spec["cases"]):
                register_seed_tuple(case, f"evaluation/{name}/{index}")
    bindings = protocol.get("source_bindings", {})
    if set(bindings) != SOURCE_PATHS:
        raise ValueError("Task3 retention source binding set mismatch")
    for relative_path, expected_hash in bindings.items():
        source = ROOT / relative_path
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"Task3 retention source binding mismatch: {relative_path}")
    source = protocol.get("source_parent", {})
    for key in ("clean_protocol", "task2_report", "retention_audit_report", "checkpoint"):
        item = source.get(key, {})
        artifact = ROOT / item.get("path", "missing")
        if not artifact.is_file() or sha256_file(artifact) != item.get("sha256"):
            raise ValueError(f"source parent binding mismatch: {key}")
    frozen = protocol.get("frozen_v4_baseline", {})
    checkpoint = ROOT / frozen.get("checkpoint_path", "missing")
    if not checkpoint.is_file() or sha256_file(checkpoint) != frozen.get("checkpoint_sha256"):
        raise ValueError("frozen-v4 checkpoint binding mismatch")
    return protocol, sha256_file(path)


def final_checkpoint_path(protocol: dict, branch: str, stage: str) -> Path:
    if branch not in BRANCHES or stage not in STAGES:
        raise ValueError("invalid Task3 final checkpoint identity")
    return (ROOT / protocol["checkpoint_directory"] / branch / stage / "final.pt").resolve()


def snapshot_path(protocol: dict, branch: str, stage: str, stage_round: int) -> Path:
    if (
        branch not in BRANCHES
        or stage not in STAGES
        or stage_round not in range(50, 601, 50)
    ):
        raise ValueError("invalid Task3 snapshot identity")
    return (
        ROOT / protocol["checkpoint_directory"] / branch / stage
        / "snapshots" / f"round-{stage_round:04d}.pt"
    ).resolve()


def selected_checkpoint_path(protocol: dict, branch: str, stage: str) -> Path:
    if branch not in BRANCHES or stage not in STAGES:
        raise ValueError("invalid Task3 selected checkpoint identity")
    return (ROOT / protocol["checkpoint_directory"] / branch / stage / "selected.pt").resolve()
