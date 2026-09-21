"""Contract tests for the s139000 opponent-shift evaluation."""

from __future__ import annotations

from pathlib import Path

from tools.v4_opponent_shift import (
    EXPECTED_CASES, INTERNAL_LABELS, LABELS, decide, dry_run, load_protocol,
    registered_seeds,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-v4-opponent-shift-s139000.json"


def _metric(score: float, kills: float = 0.1, suicides: float = 0.1) -> dict:
    return {
        "score_per_round": score, "kills_per_round": kills,
        "suicides_per_round": suicides,
    }


def _pooled(v4: float, source: float, r3: float, external: float) -> dict:
    return {
        "v4": _metric(v4), "source-r2": _metric(source),
        "candidate-r3": _metric(r3), "external-strong": _metric(external),
    }


def _blocks(v4_leads: int) -> list[dict]:
    return [{"v4_leads_internal_by_score": index < v4_leads} for index in range(8)]


def _historical() -> dict:
    return {label: _metric(3.0) for label in INTERNAL_LABELS}


def test_protocol_is_frozen_evaluation_only_and_has_unique_seeds():
    protocol, _ = load_protocol(PROTOCOL)
    assert tuple(protocol["labels"]) == LABELS
    assert tuple(protocol["evaluation"]["cases"]) == EXPECTED_CASES
    assert protocol["training_allowed"] is False
    assert protocol["selection_allowed"] is False
    assert protocol["automatic_followup"] is False
    assert len(registered_seeds(protocol)) == 40


def test_roster_order_is_balanced_twice_per_slot():
    protocol, _ = load_protocol(PROTOCOL)
    cases = protocol["evaluation"]["cases"]
    for label in LABELS:
        assert [case["roster"].index(label) for case in cases].count(0) == 2
        assert [case["roster"].index(label) for case in cases].count(1) == 2
        assert [case["roster"].index(label) for case in cases].count(2) == 2
        assert [case["roster"].index(label) for case in cases].count(3) == 2


def test_dry_run_is_400_shared_games_without_checkout_or_training():
    protocol, _ = load_protocol(PROTOCOL)
    summary = dry_run(protocol)
    assert summary["cases"] == 8 and summary["rounds_per_case"] == 50
    assert summary["shared_games_total"] == 400
    assert summary["rounds_observed_per_label"] == 400
    assert summary["external_checkout_started"] is False
    assert summary["formal_evaluation_started"] is False
    assert summary["training_started"] is False
    assert summary["selection_allowed"] is False
    assert summary["checkpoint_copy_allowed"] is False


def test_v4_stable_internal_lead_reports_external_gap_separately():
    protocol, _ = load_protocol(PROTOCOL)
    result = decide(protocol, _pooled(3.0, 2.8, 2.7, 4.0), _blocks(6), _historical())
    assert result["v4_internal_leader"] is True
    assert result["v4_internal_lead_stable"] is True
    assert result["large_external_gap"] is True
    assert result["decision"] == "v4_internal_lead_retained_external_gap_stop"
    assert result["training_started"] is False and result["selection_made"] is False


def test_v4_can_retain_internal_lead_without_large_external_gap():
    protocol, _ = load_protocol(PROTOCOL)
    result = decide(protocol, _pooled(3.0, 2.8, 2.7, 3.49), _blocks(5), _historical())
    assert result["large_external_gap"] is False
    assert result["decision"] == "v4_internal_lead_retained_no_large_external_gap_stop"


def test_unstable_or_lost_internal_lead_never_selects_or_trains():
    protocol, _ = load_protocol(PROTOCOL)
    unstable = decide(protocol, _pooled(3.0, 2.8, 2.7, 3.1), _blocks(4), _historical())
    lost = decide(protocol, _pooled(2.5, 2.8, 2.7, 3.1), _blocks(3), _historical())
    assert unstable["decision"] == "v4_internal_lead_unstable_stop"
    assert lost["decision"] == "v4_not_internal_leader_under_shift_stop"
    for result in (unstable, lost):
        assert result["selection_made"] is False
        assert result["training_started"] is False
        assert result["checkpoint_copied"] is False
        assert result["automatic_followup_started"] is False
