"""Configuration contract for the clean v4 curriculum experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGE_ORDER = ("task1", "task2", "task3_peaceful", "task3_coin", "task4")
ARM_NAMES = ("curriculum", "direct")
COURSE_BOUNDARIES = {
    "task1": ["task1"],
    "task2": ["task2"],
    "task3": ["task3_peaceful", "task3_coin"],
    "task4": ["task4"],
}
V4_SOURCE_PATHS = {
    "agent_code/model_a_dqn/callbacks.py",
    "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py",
    "agent_code/model_a_dqn/train.py",
    "agent_code/model_a_dqn/replay_buffer.py",
}
EXPERIMENT_SOURCE_PATHS = {
    "agent_code/model_a_v4_curriculum/config.py",
    "agent_code/model_a_v4_curriculum/callbacks.py",
    "agent_code/model_a_v4_curriculum/train.py",
    "agent_code/seeded_rule_based_agent/callbacks.py",
    "tools/v4_clean_curriculum.py",
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
        raise ValueError("clean-v4 protocol requires schema_version=1")
    if protocol.get("kind") != "model-a-v4-clean-curriculum":
        raise ValueError("wrong clean-v4 protocol kind")
    if tuple(protocol.get("stage_order", ())) != STAGE_ORDER:
        raise ValueError(f"stage_order must be {list(STAGE_ORDER)}")
    if tuple(protocol.get("arms", ())) != ARM_NAMES:
        raise ValueError(f"arms must be {list(ARM_NAMES)}")
    if protocol.get("course_boundaries") != COURSE_BOUNDARIES:
        raise ValueError("Task1, Task2, Task3 and Task4 course boundaries changed")
    if protocol.get("human_approval_policy") != (
        "Each course writes a terminal report and stops; every later course and final evaluation "
        "requires the exact SHA-256 of the previous report after explicit user instruction."
    ):
        raise ValueError("human approval policy changed")
    if not protocol.get("course_report_directory"):
        raise ValueError("course_report_directory is required")
    replicas = protocol.get("replicas", {})
    if tuple(replicas) != ("r1", "r2", "r3"):
        raise ValueError("exactly the ordered replicas r1, r2, r3 are required")
    for replica, spec in replicas.items():
        stages = spec.get("stages", {})
        if tuple(stages) != STAGE_ORDER:
            raise ValueError(f"{replica} has an incomplete or reordered stage schedule")
        for stage_id, stage in stages.items():
            required = {"rounds", "world_seed", "agent_seed", "opponent_seed"}
            if set(stage) != required or int(stage["rounds"]) <= 0:
                raise ValueError(f"invalid seed/round contract for {replica}/{stage_id}")
            if int(stage["rounds"]) % 50:
                raise ValueError(f"{replica}/{stage_id} rounds must be divisible by 50")
    totals = {
        replica: sum(int(stage["rounds"]) for stage in spec["stages"].values())
        for replica, spec in replicas.items()
    }
    if set(totals.values()) != {int(protocol.get("rounds_per_arm", -1))}:
        raise ValueError("rounds_per_arm does not match the stage schedule")
    if int(protocol.get("total_training_rounds", -1)) != sum(totals.values()) * len(ARM_NAMES):
        raise ValueError("total_training_rounds does not match arms and replicas")
    binding_groups = (
        ("v4_source_bindings", V4_SOURCE_PATHS, "frozen v4"),
        ("experiment_source_bindings", EXPERIMENT_SOURCE_PATHS, "experiment"),
    )
    for field, required_paths, label in binding_groups:
        bindings = protocol.get(field, {})
        if set(bindings) != required_paths:
            raise ValueError(f"{label} source binding set mismatch")
        for relative_path, expected_hash in bindings.items():
            source = (ROOT / relative_path).resolve()
            if not source.is_file() or sha256_file(source) != expected_hash:
                raise ValueError(f"{label} source binding mismatch: {relative_path}")
    checkpoint = (ROOT / protocol["frozen_v4_baseline"]["checkpoint_path"]).resolve()
    if not checkpoint.is_file():
        raise ValueError("frozen v4 baseline checkpoint is missing")
    if sha256_file(checkpoint) != protocol["frozen_v4_baseline"]["checkpoint_sha256"]:
        raise ValueError("frozen v4 baseline checkpoint hash mismatch")
    expected_strata = {
        "task1": ("coin-heaven", []),
        "task2": ("classic", []),
        "task3_peaceful": ("classic", ["seeded_peaceful_agent"]),
        "task3_coin": ("classic", ["seeded_coin_collector_agent"]),
        "task4": ("classic", ["seeded_rule_based_agent"] * 3),
    }
    final_strata = protocol.get("evaluation", {}).get("strata", {})
    if tuple(final_strata) != tuple(expected_strata):
        raise ValueError("evaluation strata changed")
    course_validation = protocol.get("course_validation", {})
    expected_course_strata = {
        "task1": ("task1",),
        "task2": ("task1", "task2"),
        "task3": ("task1", "task2", "task3_peaceful", "task3_coin"),
    }
    if tuple(course_validation) != tuple(expected_course_strata):
        raise ValueError("course validation suites changed")
    seen_seeds = {
        int(value)
        for spec in replicas.values()
        for stage in spec["stages"].values()
        for value in (stage["world_seed"], stage["agent_seed"], stage["opponent_seed"])
    }
    if len(seen_seeds) != len(replicas) * len(STAGE_ORDER) * 3:
        raise ValueError("training seeds must be globally unique")
    suites = [("final", final_strata)]
    for course, expected_names in expected_course_strata.items():
        expected_checkpoint = {
            "task1": "task1", "task2": "task2", "task3": "task3_coin",
        }[course]
        if course_validation[course].get("checkpoint_stage") != expected_checkpoint:
            raise ValueError(f"held-out checkpoint changed after {course}")
        course_strata = course_validation[course].get("strata", {})
        if tuple(course_strata) != expected_names:
            raise ValueError(f"held-out strata changed after {course}")
        suites.append((f"after-{course}", course_strata))
    for suite_name, strata in suites:
        for stratum, spec in strata.items():
            scenario, opponents = expected_strata[stratum]
            if spec.get("scenario") != scenario or spec.get("opponents") != opponents:
                raise ValueError(f"evaluation distribution changed: {suite_name}/{stratum}")
            if int(spec.get("rounds_per_case", 0)) <= 0 or not spec.get("cases"):
                raise ValueError(f"invalid evaluation budget: {suite_name}/{stratum}")
            for case in spec["cases"]:
                if set(case) != {"world_seed", "agent_seed", "opponent_seed"}:
                    raise ValueError(f"incomplete evaluation seed tuple: {suite_name}/{stratum}")
                for value in case.values():
                    seed = int(value)
                    if seed in seen_seeds:
                        raise ValueError("formal training/evaluation seeds overlap")
                    seen_seeds.add(seed)
    return protocol, sha256_file(path)


def checkpoint_path(protocol: dict, arm: str, replica: str, stage_id: str) -> Path:
    if arm not in ARM_NAMES or replica not in protocol["replicas"] or stage_id not in STAGE_ORDER:
        raise ValueError("invalid clean-v4 checkpoint identity")
    root = (ROOT / protocol["checkpoint_directory"]).resolve()
    return root / arm / replica / f"{stage_id}.pt"


def parent_stage(stage_id: str) -> str | None:
    index = STAGE_ORDER.index(stage_id)
    return None if index == 0 else STAGE_ORDER[index - 1]


def stage_environment(protocol: dict, arm: str, stage_id: str) -> dict:
    """Return the actual game distribution for one arm/stage."""
    if arm == "direct":
        return {"scenario": "classic", "opponents": ["seeded_rule_based_agent"] * 3}
    mapping = {
        "task1": {"scenario": "coin-heaven", "opponents": []},
        "task2": {"scenario": "classic", "opponents": []},
        "task3_peaceful": {"scenario": "classic", "opponents": ["seeded_peaceful_agent"]},
        "task3_coin": {"scenario": "classic", "opponents": ["seeded_coin_collector_agent"]},
        "task4": {"scenario": "classic", "opponents": ["seeded_rule_based_agent"] * 3},
    }
    return mapping[stage_id]
