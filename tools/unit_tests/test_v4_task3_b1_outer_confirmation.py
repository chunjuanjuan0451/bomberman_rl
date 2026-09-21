"""Contract tests for the fixed-candidate Task3 b1 outer confirmation."""

from __future__ import annotations

from pathlib import Path

from tools.v4_task3_b1_outer_confirmation import (
    LABELS, STRATA, decide, dry_run, load_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task3-b1-outer-confirmation-s120000.json"


def test_protocol_uses_exact_untouched_source_outer_suite_and_one_candidate():
    protocol, _, source, _ = load_protocol(PROTOCOL_PATH)
    assert tuple(protocol["labels"]) == LABELS
    assert protocol["candidate_count"] == 1
    assert protocol["selection_free"] is True
    assert protocol["evaluation"]["strata"] == source["outer_evaluation"]["strata"]
    assert protocol["checkpoint_inventory"]["candidate_b1"]["source_snapshot_path"].endswith(
        "task3_peaceful/snapshots/round-0100.pt"
    )


def test_dry_run_is_exactly_600_rounds_and_never_trains_or_starts_task4():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["labels"] == list(LABELS)
    assert summary["strata"] == list(STRATA)
    assert summary["total_evaluation_rounds"] == 600
    assert summary["selection_free"] is True
    assert summary["uses_untouched_source_outer_seeds"] is True
    assert summary["training_started"] is False
    assert summary["checkpoint_copy_started"] is False
    assert summary["task4_started"] is False


def _row(score: float, kills: float = 0.0, suicide: float = 0.0, invalid: float = 0.0) -> dict:
    return {
        "score_per_round": score,
        "coins_per_round": score - 5.0 * kills,
        "kills_per_round": kills,
        "suicides_per_round": suicide,
        "invalid_actions_per_round": invalid,
    }


def _passing_rows() -> dict:
    candidate = "candidate-b1-peaceful-r0100"
    return {
        "task1": {"v4": _row(10.0), candidate: _row(9.0)},
        "task2": {"v4": _row(3.2), candidate: _row(3.0)},
        "task3_peaceful": {
            "v4": _row(2.0, 0.1, 0.01),
            candidate: _row(3.0, 0.2, 0.03),
        },
        "task3_coin": {
            "v4": _row(3.1, 0.05, 0.02),
            candidate: _row(3.0, 0.1, 0.04),
        },
    }


def test_decision_confirms_only_when_every_preregistered_gate_passes():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    result = decide(protocol, _passing_rows())
    assert result["passed"] is True
    assert all(result["gates"].values())
    assert result["decision"] == "b1_peaceful_confirmed_for_task3"
    assert result["candidate_count"] == 1
    assert result["selection_free"] is True
    assert result["automatic_checkpoint_copied"] is False
    assert result["task4_started"] is False


def test_absolute_coin_kill_floor_prevents_a_vacuous_zero_kill_pass():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    rows = _passing_rows()
    rows["task3_coin"]["v4"] = _row(3.1, 0.0)
    rows["task3_coin"]["candidate-b1-peaceful-r0100"] = _row(3.0, 0.0)
    result = decide(protocol, rows)
    assert result["gates"]["coin_kills_vs_v4"] is True
    assert result["gates"]["coin_kills_absolute"] is False
    assert result["passed"] is False
    assert result["decision"] == "b1_peaceful_outer_rejected"
