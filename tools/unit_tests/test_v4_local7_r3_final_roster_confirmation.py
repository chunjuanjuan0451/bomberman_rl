"""Contract tests for the s143000 local7-r3 final-roster confirmation."""

from __future__ import annotations

from pathlib import Path

from tools.v4_local7_r3_final_roster_confirmation import (
    LABELS,
    decide,
    dry_run,
    load_protocol,
    registered_seeds,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-v4-local7-r3-final-roster-confirmation-s143000.json"


def _metric(
    score: float,
    *,
    coins: float | None = None,
    kills: float = 0.1,
    suicides: float = 0.1,
    invalid: float = 0.2,
) -> dict:
    return {
        "score_per_round": score,
        "coins_per_round": score if coins is None else coins,
        "kills_per_round": kills,
        "suicides_per_round": suicides,
        "invalid_actions_per_round": invalid,
    }


def _pooled(candidate_score: float = 3.2, *, candidate_suicides: float = 0.1) -> dict:
    return {
        "v4": _metric(3.0),
        "source-r2": _metric(2.8, suicides=0.2),
        "local7-r3": _metric(candidate_score, suicides=candidate_suicides),
        "external-strong": _metric(5.0, suicides=0.05),
    }


def _blocks(candidate_leads: int) -> list[dict]:
    return [{"candidate_leads_v4_by_score": index < candidate_leads} for index in range(8)]


def test_protocol_is_fresh_balanced_evaluation_only_confirmation():
    protocol, _, digest = load_protocol(PROTOCOL)
    assert tuple(protocol["labels"]) == LABELS
    assert protocol["post_hoc_candidate_fixed_before_s143000"] == "local7-r3"
    assert protocol["training_allowed"] is False
    assert protocol["selection_allowed"] is False
    assert protocol["checkpoint_copy_allowed"] is False
    assert protocol["automatic_followup"] is False
    assert len(registered_seeds(protocol)) == 40
    assert len(digest) == 64


def test_each_identity_occupies_each_roster_slot_twice():
    protocol, _, _ = load_protocol(PROTOCOL)
    cases = protocol["evaluation"]["cases"]
    for label in LABELS:
        for slot in range(4):
            assert [case["roster"].index(label) for case in cases].count(slot) == 2


def test_dry_run_is_400_shared_games_and_starts_nothing():
    protocol, _, _ = load_protocol(PROTOCOL)
    summary = dry_run(protocol)
    assert summary["fixed_candidate"] == "local7-r3"
    assert summary["cases"] == 8 and summary["rounds_per_case"] == 50
    assert summary["shared_games_total"] == 400
    assert summary["rounds_observed_per_label"] == 400
    assert summary["external_checkout_started"] is False
    assert summary["formal_evaluation_started"] is False
    assert summary["training_started"] is False
    assert summary["selection_allowed"] is False
    assert summary["checkpoint_copy_allowed"] is False


def test_candidate_pass_requires_score_blocks_internal_lead_and_safety():
    protocol, _, _ = load_protocol(PROTOCOL)
    result = decide(protocol, _pooled(), _blocks(5))
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["decision"] == protocol["decision_branches"]["pass"]
    assert result["selection_made_within_s143000"] is False
    assert result["training_started"] is False
    assert result["checkpoint_copied"] is False


def test_candidate_fails_on_unstable_blocks_or_suicide_regression():
    protocol, _, _ = load_protocol(PROTOCOL)
    unstable = decide(protocol, _pooled(), _blocks(4))
    unsafe = decide(protocol, _pooled(candidate_suicides=0.14), _blocks(6))
    assert unstable["checks"]["block_support"] is False
    assert unsafe["checks"]["suicide_guard"] is False
    for result in (unstable, unsafe):
        assert result["passed"] is False
        assert result["decision"] == protocol["decision_branches"]["fail"]
        assert result["automatic_followup_started"] is False
