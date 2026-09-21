"""Contracts for the Task4-only rehearsal25-r3 confirmation."""

from __future__ import annotations

from pathlib import Path

from tools.v4_rehearsal_r3_task4_confirmation import (
    EXPECTED_CASES, LABELS, STRATA, decide, dry_run, load_protocol,
    registered_seeds,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-v4-rehearsal-r3-task4-confirmation-s138000.json"


def _metrics(score: float, kills: float, suicide: float) -> dict:
    return {
        "rounds": 200, "score": int(score * 200), "coins": 0, "kills": int(kills * 200),
        "crates": 0, "bombs": 0, "moves": 0, "waits": 0,
        "suicides": int(suicide * 200), "invalid_actions": 0, "steps": 200,
        "score_per_round": score, "coins_per_round": 0.0, "kills_per_round": kills,
        "crates_per_round": 0.0, "bombs_per_round": 0.0,
        "suicides_per_round": suicide, "invalid_actions_per_round": 0.0,
        "move_fraction": 0.0, "wait_fraction": 0.0, "episode_length": 1.0,
    }


def _entry(score: float, kills: float, suicide: float, margin: float) -> dict:
    return {"target": _metrics(score, kills, suicide), "score_margin": margin}


def _passing_rows() -> dict:
    rows = {
        "source-r2": {
            "task4b_duel": _entry(3.10, 0.10, 0.20, 0.10),
            "task4c_three_rule": _entry(2.90, 0.15, 0.25, 0.00),
            "pooled_task4": _entry(3.00, 0.125, 0.225, 0.05),
        },
        "v4": {
            "task4b_duel": _entry(3.05, 0.12, 0.18, 0.05),
            "task4c_three_rule": _entry(2.95, 0.18, 0.22, 0.05),
            "pooled_task4": _entry(3.00, 0.15, 0.20, 0.05),
        },
        "candidate-r3": {
            "task4b_duel": _entry(3.20, 0.20, 0.15, 0.20),
            "task4c_three_rule": _entry(3.00, 0.25, 0.20, 0.10),
            "pooled_task4": _entry(3.10, 0.225, 0.175, 0.15),
        },
    }
    return rows


def _passing_cases() -> dict:
    result = {}
    for stratum in STRATA:
        result[stratum] = {}
        for case in EXPECTED_CASES[stratum]:
            result[stratum][str(case["world_seed"])] = {
                "source-r2": _entry(3.0, 0.10, 0.25, 0.00),
                "v4": _entry(3.0, 0.12, 0.20, 0.00),
                "candidate-r3": _entry(3.1, 0.16, 0.18, 0.10),
            }
    return result


def test_protocol_fixes_one_candidate_and_twenty_four_fresh_seeds():
    protocol, _ = load_protocol(PROTOCOL)
    assert tuple(protocol["labels"]) == LABELS
    assert protocol["fixed_candidate"] == "candidate-r3"
    assert protocol["selection_free"] is True
    assert protocol["training_allowed"] is False
    assert protocol["task1_to_task3_are_diagnostic_only"] is True
    assert registered_seeds(protocol) == {
        value for cases in EXPECTED_CASES.values() for case in cases for value in case.values()
    }
    assert len(registered_seeds(protocol)) == 24


def test_dry_run_is_task4_only_and_exactly_1200_rounds():
    protocol, _ = load_protocol(PROTOCOL)
    summary = dry_run(protocol)
    assert summary["strata"] == list(STRATA)
    assert summary["rounds_per_label"] == 400
    assert summary["total_evaluation_rounds"] == 1200
    assert summary["formal_evaluation_started"] is False
    assert summary["training_started"] is False
    assert summary["checkpoint_copy_allowed"] is False


def test_relative_confirmation_and_rule_competitiveness_are_separate():
    protocol, _ = load_protocol(PROTOCOL)
    result = decide(protocol, _passing_rows(), _passing_cases())
    assert result["relative_candidate_confirmed"] is True
    assert result["rule_competitive"] is True
    assert all(result["relative_gates"].values())
    assert result["decision"] == "candidate_confirmed_rule_competitive_stop_before_promotion"
    assert result["training_started"] is False
    assert result["checkpoint_copied"] is False


def test_rule_margin_can_reject_a_high_kill_checkpoint():
    protocol, _ = load_protocol(PROTOCOL)
    rows = _passing_rows()
    rows["candidate-r3"]["task4b_duel"]["score_margin"] = -0.30
    rows["candidate-r3"]["pooled_task4"]["score_margin"] = -0.20
    result = decide(protocol, rows, _passing_cases())
    assert result["relative_candidate_confirmed"] is False
    assert result["relative_gates"]["pooled_margin_vs_source-r2"] is False
    assert result["relative_gates"]["task4b_duel_vs_source-r2_margin"] is False
    assert result["decision"] == "candidate_not_confirmed_retain_incumbent_stop"


def test_relative_pass_without_beating_rules_is_reported_not_promoted():
    protocol, _ = load_protocol(PROTOCOL)
    rows = _passing_rows()
    rows["candidate-r3"]["task4b_duel"]["score_margin"] = -0.05
    result = decide(protocol, rows, _passing_cases())
    assert result["relative_candidate_confirmed"] is True
    assert result["rule_competitive"] is False
    assert result["decision"] == "candidate_confirmed_relative_only_stop_before_promotion"
