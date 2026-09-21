"""Contract tests for the no-training clean-v4 Task1 retention audit."""

from __future__ import annotations

from pathlib import Path

from tools.v4_task1_retention_audit import (
    REPLICAS, checkpoint_identity, decide, dry_run, load_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task1-retention-audit-s118100.json"


def test_retention_protocol_is_frozen_evaluation_only():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    assert protocol["scope"].startswith("frozen-checkpoint evaluation only")
    assert tuple(protocol["replicas"]) == REPLICAS
    assert [case["world_seed"] for case in protocol["evaluation"]["cases"]] == [118101, 118102]
    assert protocol["evaluation"]["rounds_per_case"] == 25


def test_dry_run_has_seven_labels_and_no_training_or_task3():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["total_evaluation_rounds"] == 350
    assert len(summary["labels"]) == 7
    assert summary["paired_same_seeds"] is True
    assert summary["training_started"] is False
    assert summary["task3_started"] is False
    assert summary["formal_evaluation_started"] is False


def test_checkpoint_identity_pairs_task1_and_task2_within_replica():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    target1, path1, stage1, replica1 = checkpoint_identity(protocol, "task1-r2")
    target2, path2, stage2, replica2 = checkpoint_identity(protocol, "task2-r2")
    assert target1 == target2 == "model_a_v4_curriculum"
    assert stage1 == "task1" and stage2 == "task2"
    assert replica1 == replica2 == "r2"
    assert path1 != path2


def _row(score: float) -> dict:
    return {"score_per_round": score}


def test_decision_confirms_two_replica_forgetting_without_auto_selection():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    rows = {
        "v4": _row(30.0),
        "task1-r1": _row(25.0), "task2-r1": _row(10.0),
        "task1-r2": _row(20.0), "task2-r2": _row(30.0),
        "task1-r3": _row(24.0), "task2-r3": _row(8.0),
    }
    task2_report = {
        "heldout_validation": {"rows": {"task2": {
            "v4": _row(2.0),
            "curriculum-r1": _row(3.0),
            "curriculum-r2": _row(3.0),
            "curriculum-r3": _row(3.0),
        }}}
    }
    result = decide(protocol, rows, task2_report)
    assert result["forgetting_confirmed"] is True
    assert result["affected_replicas"] == ["r1", "r3"]
    assert result["joint_parent_candidates"] == ["r2"]
    assert result["decision"] == "retention_failure_confirmed_with_joint_parent"
    assert result["automatic_checkpoint_selected"] is False
    assert result["training_started"] is False
    assert result["task3_started"] is False
