"""Contract tests for the evaluation-only Task4 rule-agent entry gate."""

from __future__ import annotations

from pathlib import Path

from tools.v4_task4_rule_baseline_gate import decide, dry_run, load_protocol


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task4-rule-baseline-gate-s123000.json"


def _item(score: float, kills: float, suicide: float, opponent_score: float) -> dict:
    return {
        "target": {
            "score_per_round": score,
            "kills_per_round": kills,
            "suicides_per_round": suicide,
            "invalid_actions_per_round": 0.0,
        },
        "opponents": {"mean_score_per_agent_round": opponent_score},
    }


def _passing_rows() -> dict:
    return {
        "task4_rule_duel": {
            "v4": _item(2.0, 0.1, 0.10, 2.5),
            "source-r2": _item(3.0, 0.2, 0.08, 2.8),
        },
        "task4_three_rule": {
            "v4": _item(2.2, 0.1, 0.12, 3.0),
            "source-r2": _item(3.1, 0.2, 0.10, 3.0),
        },
    }


def test_task4_entry_gate_is_400_rounds_and_never_trains_automatically():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["strata"] == ["task4_rule_duel", "task4_three_rule"]
    assert summary["total_evaluation_rounds"] == 400
    assert summary["training_started"] is False
    assert summary["automatic_followup_started"] is False
    assert summary["formal_evaluation_started"] is False


def test_source_r2_is_already_competitive_only_when_every_task4_gate_passes():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    result = decide(protocol, _passing_rows())
    assert result["already_competitive"] is True
    assert all(result["gates"].values())
    assert result["decision"] == "source_r2_already_task4_competitive"
    assert result["training_started"] is False


def test_losing_to_three_rule_mean_requires_a_separate_training_decision():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    rows = _passing_rows()
    rows["task4_three_rule"]["source-r2"]["target"]["score_per_round"] = 2.9
    result = decide(protocol, rows)
    assert result["gates"]["three_rule_beats_rule_mean"] is False
    assert result["already_competitive"] is False
    assert result["decision"] == "task4_training_required"
    assert result["automatic_followup_started"] is False
