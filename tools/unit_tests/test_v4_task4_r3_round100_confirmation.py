"""Contract tests for the selection-free Task4 r3 round-100 confirmation."""

from __future__ import annotations

from pathlib import Path

from tools.v4_task4_r3_round100_confirmation import (
    EXPECTED_CASES,
    LABELS,
    decide,
    dry_run,
    load_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task4-r3-round100-confirmation-s126000.json"


def _row(score: float, kills: float, suicide: float, opponent_score: float = 3.0) -> dict:
    return {
        "target": {
            "score_per_round": score,
            "kills_per_round": kills,
            "suicides_per_round": suicide,
            "invalid_actions_per_round": 0.0,
            "coins_per_round": score,
        },
        "opponents": {"mean_score_per_agent_round": opponent_score},
    }


def _passing_rows() -> dict:
    return {
        "v4": _row(3.1, 0.20, 0.25),
        "source-r2": _row(3.2, 0.22, 0.30),
        "candidate-r3-round-0100": _row(3.3, 0.25, 0.10, 3.2),
    }


def _passing_cases() -> dict:
    rows = {}
    for index, case in enumerate(EXPECTED_CASES):
        candidate_score = 3.4 if index < 2 else 3.0
        rows[str(case["world_seed"])] = {
            "v4": _row(3.1, 0.2, 0.25),
            "source-r2": _row(3.2, 0.2, 0.25),
            "candidate-r3-round-0100": _row(candidate_score, 0.2, 0.1),
        }
    return rows


def test_protocol_fixes_one_posthoc_candidate_on_twelve_fresh_seeds():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    assert tuple(protocol["labels"]) == LABELS
    assert protocol["fixed_candidate"] == "candidate-r3-round-0100"
    assert protocol["selection_free"] is True
    seeds = [value for case in protocol["evaluation"]["cases"] for value in case.values()]
    assert len(seeds) == 12
    assert len(set(seeds)) == 12


def test_dry_run_is_exactly_300_rounds_and_never_trains_or_copies():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["total_evaluation_rounds"] == 300
    assert summary["formal_evaluation_started"] is False
    assert summary["training_started"] is False
    assert summary["checkpoint_copy_allowed"] is False
    assert summary["task4c_started"] is False


def test_confirmation_requires_score_kills_suicide_rule_and_case_gates():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    result = decide(protocol, _passing_rows(), _passing_cases())
    assert result["passed"] is True
    assert all(result["gates"].values())
    assert result["decision"] == "r3_round100_signal_replicated_for_task4c_consideration_stop"
    assert result["checkpoint_copied"] is False
    assert result["automatic_followup_started"] is False


def test_score_or_suicide_failure_rejects_candidate_and_retains_incumbent():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    rows = _passing_rows()
    rows["candidate-r3-round-0100"]["target"]["score_per_round"] = 3.05
    rows["candidate-r3-round-0100"]["target"]["suicides_per_round"] = 0.20
    result = decide(protocol, rows, _passing_cases())
    assert result["passed"] is False
    assert result["gates"]["score_not_below_source_r2"] is False
    assert result["gates"]["suicide_reduction_vs_v4"] is False
    assert result["decision"] == "r3_round100_rejected_retain_source_r2_stop"
    assert result["incumbent_replaced"] is False
