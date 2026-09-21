"""Contract tests for the exact-v4 Task1-to-Task4 retraining protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.network import DuelingDQN as V4Network
from agent_code.model_a_v4_curriculum import train as curriculum_train
from agent_code.model_a_v4_curriculum.config import (
    ARM_NAMES, STAGE_ORDER, load_protocol, parent_stage, stage_environment,
)
from tools.v4_clean_curriculum import (
    COURSE_STAGES, decide, dry_run, evaluation_identity, evaluation_manifest,
    sha256_file, validate_previous_approval, validation_family_summary,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-clean-curriculum-s114000.json"


def test_protocol_has_separate_courses_and_fresh_fixed_seeds():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert tuple(protocol["course_boundaries"]) == ("task1", "task2", "task3", "task4")
    assert COURSE_STAGES == {
        "task1": ("task1",),
        "task2": ("task2",),
        "task3": ("task3_peaceful", "task3_coin"),
        "task4": ("task4",),
    }
    assert protocol["rounds_per_arm"] == 4000
    assert protocol["total_training_rounds"] == 24000
    all_seeds = []
    for replica in protocol["replicas"].values():
        assert tuple(replica["stages"]) == STAGE_ORDER
        for stage in replica["stages"].values():
            all_seeds.extend((stage["world_seed"], stage["agent_seed"], stage["opponent_seed"]))
    assert len(all_seeds) == len(set(all_seeds))


def test_only_distribution_differs_between_matched_arms():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    for replica, spec in protocol["replicas"].items():
        for stage_id, stage in spec["stages"].items():
            assert stage == protocol["replicas"][replica]["stages"][stage_id]
            assert stage_environment(protocol, "curriculum", stage_id) != stage_environment(protocol, "direct", stage_id) or stage_id == "task4"
    assert stage_environment(protocol, "curriculum", "task4") == stage_environment(protocol, "direct", "task4")


def test_curriculum_agent_reuses_v4_network_and_learner_primitives():
    from agent_code.model_a_v4_curriculum.callbacks import DuelingDQN as CurriculumNetwork
    from agent_code.model_a_dqn import train as v4_train

    assert CurriculumNetwork is V4Network
    assert MODEL_ARCHITECTURE == "model-a-mlp-v4-global7"
    assert curriculum_train.v4 is v4_train
    assert curriculum_train.setup_training.__code__.co_names == ("v4", "setup_training")


def test_stage_lineage_and_dry_run_never_cross_course():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert parent_stage("task1") is None
    assert parent_stage("task2") == "task1"
    assert parent_stage("task3_peaceful") == "task2"
    assert parent_stage("task3_coin") == "task3_peaceful"
    assert parent_stage("task4") == "task3_coin"
    for course, stages in COURSE_STAGES.items():
        summary = dry_run(protocol, course)
        assert summary["formal_training_started"] is False
        assert summary["automatically_starts_next_course"] is False
        assert summary["automatic_heldout_validation"] is True
        assert summary["writes_course_report_and_stops"] is True
        assert {run["stage"] for run in summary["runs"]} == set(stages)


def test_each_course_has_cumulative_independent_heldout_validation():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    expected = {
        "task1": (["task1"], 350),
        "task2": (["task1", "task2"], 525),
        "task3": (["task1", "task2", "task3_peaceful", "task3_coin"], 1050),
        "task4": (["task1", "task2", "task3_peaceful", "task3_coin", "task4"], 4200),
    }
    validation_seeds = set()
    for course, (strata, rounds) in expected.items():
        summary = dry_run(protocol, course)
        assert summary["heldout_validation_strata"] == strata
        assert summary["heldout_validation_rounds"] == rounds
        specs = (
            protocol["evaluation"]["strata"] if course == "task4"
            else protocol["course_validation"][course]["strata"]
        )
        for spec in specs.values():
            for case in spec["cases"]:
                for seed in case.values():
                    assert seed not in validation_seeds
                    validation_seeds.add(seed)


def test_heldout_artifacts_bind_the_current_course_checkpoint_and_suite():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    _, task1_path, arm, replica = evaluation_identity(protocol, "curriculum-r2", "task1")
    _, task2_path, _, _ = evaluation_identity(protocol, "curriculum-r2", "task2")
    assert arm == "curriculum" and replica == "r2"
    assert task1_path.name == "task1.pt"
    assert task2_path.name == "task2.pt"
    assert task1_path != task2_path
    manifest = evaluation_manifest(protocol, "after-task1", "task1", "curriculum-r2", 116101)
    assert "after-task1" in manifest.parts
    assert manifest.name == "task1-curriculum-r2-s116101.json"


def test_next_course_requires_exact_prior_report_hash():
    with tempfile.TemporaryDirectory() as directory:
        protocol_hash = "protocol-hash"
        protocol = {"course_report_directory": directory}
        report_path = Path(directory) / "task1.json"
        report_path.write_text(json.dumps({
            "kind": "model-a-v4-clean-curriculum-course-report",
            "status": "completed",
            "protocol_sha256": protocol_hash,
            "course": "task1",
        }), encoding="utf-8")
        try:
            validate_previous_approval(protocol, protocol_hash, "task2", None)
        except RuntimeError as exc:
            assert "explicit user approval" in str(exc)
        else:
            raise AssertionError("Task2 must not start without the prior report hash")
        approved = validate_previous_approval(
            protocol, protocol_hash, "task2", sha256_file(report_path),
        )
        assert approved["course"] == "task1"
        assert approved["sha256"] == sha256_file(report_path)


def test_seeded_rule_opponents_have_reproducible_independent_streams():
    from agent_code.seeded_rule_based_agent.callbacks import setup

    os.environ["TASK4_RULE_SEED"] = "909901"
    try:
        logger = SimpleNamespace(name="unused")
        first = SimpleNamespace(name="seeded_rule_based_agent_1", logger=logger)
        repeated = SimpleNamespace(name="seeded_rule_based_agent_1", logger=logger)
        second = SimpleNamespace(name="seeded_rule_based_agent_2", logger=logger)
        setup(first)
        setup(repeated)
        setup(second)
        first_values = [first.rng.random() for _ in range(8)]
        assert first_values == [repeated.rng.random() for _ in range(8)]
        assert first_values != [second.rng.random() for _ in range(8)]
    finally:
        os.environ.pop("TASK4_RULE_SEED", None)


def _row(score, suicide=0.01, invalid=0.0):
    return {
        "score_per_round": score,
        "suicides_per_round": suicide,
        "invalid_actions_per_round": invalid,
    }


def test_decision_requires_two_paired_wins_and_regression_gates():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    rows = {}
    for stratum in protocol["evaluation"]["strata"]:
        rows[stratum] = {"v4": _row(3.0)}
        for replica in protocol["replicas"]:
            rows[stratum][f"curriculum-{replica}"] = _row(3.0)
            rows[stratum][f"direct-{replica}"] = _row(3.0)
    rows["task4"]["curriculum-r1"] = _row(3.4)
    rows["task4"]["curriculum-r2"] = _row(3.3)
    rows["task4"]["curriculum-r3"] = _row(3.0)
    rows["task4"]["direct-r1"] = _row(3.1)
    rows["task4"]["direct-r2"] = _row(3.1)
    rows["task4"]["direct-r3"] = _row(3.0)
    result = decide(protocol, rows)
    assert result["family_gate"] is True
    assert result["supportive_replicas"] == ["r1", "r2"]
    assert result["selected_candidate"] == "curriculum-r1"

    rows["task2"]["curriculum-r1"] = _row(2.0)
    rows["task2"]["curriculum-r2"] = _row(2.0)
    result = decide(protocol, rows)
    assert result["selected_candidate"] is None
    assert result["decision"] == "retain_frozen_v4"


def test_validation_summary_exposes_performance_and_safety_deltas():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    metrics = {
        "score_per_round": 3.0,
        "coins_per_round": 2.0,
        "kills_per_round": 0.1,
        "suicides_per_round": 0.02,
        "invalid_actions_per_round": 0.01,
    }
    labels = {"v4": dict(metrics)}
    for replica in protocol["replicas"]:
        labels[f"curriculum-{replica}"] = {key: value + 0.1 for key, value in metrics.items()}
        labels[f"direct-{replica}"] = dict(metrics)
    summary = validation_family_summary(protocol, {"task1": labels})["task1"]
    assert abs(summary["metrics"]["score_per_round"]["curriculum_minus_v4"] - 0.1) < 1e-12
    assert set(summary["paired_score_per_round_curriculum_minus_direct"]) == {"r1", "r2", "r3"}
