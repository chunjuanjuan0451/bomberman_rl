"""Contract tests for the read-only s144000 WAIT Q-gap audit."""

from pathlib import Path

from tools.v4_wait_qgap_audit import dry_run, load_protocol, summarize


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-v4-wait-qgap-audit-s144000.json"
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def _case(gaps):
    waits = [
        {
            "q_gap": gap,
            "repeat_wait_same_position": index > 0,
        }
        for index, gap in enumerate(gaps)
    ]
    return {
        "steps": 100,
        "action_counts": {action: (len(waits) if action == "WAIT" else 0) for action in ACTIONS},
        "wait_records": waits,
    }


def test_wait_audit_protocol_is_small_and_read_only():
    protocol, _, digest = load_protocol(PROTOCOL)
    summary = dry_run(protocol)
    assert summary["cases"] == 4
    assert summary["rounds_per_case"] == 25
    assert summary["total_rounds"] == 100
    assert summary["training_started"] is False
    assert summary["policy_override_allowed"] is False
    assert summary["formal_evaluation_started"] is False
    assert len(digest) == 64


def test_near_tie_waits_are_classified_as_action_selection_issue():
    protocol, _, _ = load_protocol(PROTOCOL)
    result = summarize(protocol, [_case([0.01, 0.02, 0.04, 0.20])] * 4)
    assert result["diagnosis"] == "near_tie_action_selection_issue"
    assert result["near_tie_share_of_discretionary_waits"] == 0.75
    assert result["training_started"] is False
    assert result["actions_changed_by_audit"] == 0


def test_large_gap_waits_are_classified_as_learned_preference():
    protocol, _, _ = load_protocol(PROTOCOL)
    result = summarize(protocol, [_case([0.21, 0.30, 0.40, None])] * 4)
    assert result["forced_wait_count"] == 4
    assert result["diagnosis"] == "learned_wait_q_preference"
    assert result["checkpoint_modified"] is False
