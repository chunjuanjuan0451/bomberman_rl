"""Contract tests for the selection-free s142000 CNN confirmation."""

from __future__ import annotations

from pathlib import Path

from tools.v4_local7_n8_confirmation import (
    LABELS,
    LOCAL_LABELS,
    decide,
    dry_run,
    load_protocol,
    registered_seeds,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-v4-local7-n8-confirmation-s142000.json"


def _target(score: float, *, coins: float | None = None, suicides: float = 0.1) -> dict:
    return {
        "score_per_round": score,
        "coins_per_round": score if coins is None else coins,
        "suicides_per_round": suicides,
        "invalid_actions_per_round": 0.0,
        "kills_per_round": 0.0,
    }


def _entry(
    score: float,
    *,
    coins: float | None = None,
    suicides: float = 0.1,
    margin: float = 0.0,
) -> dict:
    return {
        "target": _target(score, coins=coins, suicides=suicides),
        "score_margin": margin,
    }


def _passing_rows() -> tuple[dict, dict]:
    rows = {}
    for label in LABELS:
        rows[label] = {
            "task1": _entry(30.0),
            "task2": _entry(2.0, coins=2.0),
            "task4b_duel": _entry(3.0, margin=-0.4),
            "task4c_three_rule": _entry(3.0, margin=-0.2),
            "pooled_task4": _entry(3.0, margin=-0.3),
        }
    for index, label in enumerate(LOCAL_LABELS):
        rows[label]["task2"] = _entry(2.2 + 0.1 * index, coins=2.2 + 0.1 * index)
    family = {
        "task1": _entry(29.0),
        "task2": _entry(2.3, coins=2.3),
        "task4b_duel": _entry(3.0, margin=-0.4),
        "task4c_three_rule": _entry(3.0, margin=-0.2),
        "pooled_task4": _entry(3.0, margin=-0.3),
    }
    return rows, family


def test_protocol_is_evaluation_only_and_uses_fresh_unique_seeds():
    protocol, _, digest = load_protocol(PROTOCOL)
    assert tuple(protocol["labels"]) == LABELS
    assert tuple(protocol["fixed_candidate_family"]) == LOCAL_LABELS
    assert protocol["training_allowed"] is False
    assert protocol["individual_replica_selection_allowed"] is False
    assert protocol["checkpoint_copy_allowed"] is False
    assert protocol["automatic_followup"] is False
    assert len(registered_seeds(protocol)) == 34
    assert len(digest) == 64


def test_dry_run_has_fixed_budget_without_training_or_selection():
    protocol, _, _ = load_protocol(PROTOCOL)
    summary = dry_run(protocol)
    assert summary["evaluation_manifests"] == 65
    assert summary["rounds_per_label"] == 525
    assert summary["total_observed_rounds"] == 2625
    assert summary["training_started"] is False
    assert summary["individual_replica_selection_allowed"] is False
    assert summary["checkpoint_copy_allowed"] is False
    assert summary["kills_used_for_gate"] is False
    assert summary["formal_evaluation_started"] is False


def test_family_can_pass_without_using_kills_or_selecting_a_replica():
    protocol, _, _ = load_protocol(PROTOCOL)
    rows, family = _passing_rows()
    result = decide(protocol, rows, family)
    assert result["passed"] is True
    assert result["task2_family_confirmed"] is True
    assert result["competition_safe"] is True
    assert result["positive_task2_replicas"] == 3
    assert result["kills_used_for_gate"] is False
    assert result["individual_replica_selected"] is False
    assert result["training_started"] is False
    assert result["checkpoint_copied"] is False


def test_task2_signal_without_task4_safety_is_rejected():
    protocol, _, _ = load_protocol(PROTOCOL)
    rows, family = _passing_rows()
    family["pooled_task4"] = _entry(2.7, margin=-0.6)
    result = decide(protocol, rows, family)
    assert result["task2_family_confirmed"] is True
    assert result["competition_safe"] is False
    assert result["passed"] is False
    assert result["decision"] == protocol["decision_branches"]["task2_only"]


def test_weak_family_task2_signal_fails_even_when_task4_is_safe():
    protocol, _, _ = load_protocol(PROTOCOL)
    rows, family = _passing_rows()
    family["task2"] = _entry(2.05, coins=2.05)
    result = decide(protocol, rows, family)
    assert result["task2_family_confirmed"] is False
    assert result["competition_safe"] is True
    assert result["passed"] is False
    assert result["decision"] == protocol["decision_branches"]["fail"]
