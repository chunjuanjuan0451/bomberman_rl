"""Contracts for the CNN+n=8 selection-free final confirmation."""

from __future__ import annotations

from pathlib import Path

from tools.cnn_n8_final_confirmation import (
    EXTERNAL_LABEL, INTERNAL_LABELS, STANDARD_STRATA, TASK4_STRATA,
    decide, dry_run, load_protocol, registered_seeds,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-final-confirmation-s152000.json"


def _metrics(score: float, suicide: float = 0.10, invalid: float = 0.10) -> dict:
    return {
        "rounds": 400, "score": int(score * 400), "score_per_round": score,
        "coins": 0, "coins_per_round": 0.0, "kills": 0, "kills_per_round": 0.0,
        "crates": 0, "crates_per_round": 0.0, "bombs": 0, "bombs_per_round": 0.0,
        "moves": 0, "waits": 0, "wait_fraction": 0.0,
        "invalid_actions": int(invalid * 400), "invalid_actions_per_round": invalid,
        "suicides": int(suicide * 400), "suicides_per_round": suicide,
        "steps": 400, "decision_time_seconds": 0.0, "mean_decision_time_ms": 0.0,
    }


def _passing_inputs(protocol: dict):
    rows = {
        "candidate-cnn": {
            "task4_rule": _metrics(3.30, 0.12, 0.12),
            "task4_mixed": _metrics(3.40, 0.12, 0.12),
            "pooled_task4": _metrics(3.35, 0.12, 0.12),
        },
        "frozen-v4": {
            "task4_rule": _metrics(3.10, 0.10, 0.10),
            "task4_mixed": _metrics(3.20, 0.10, 0.10),
            "pooled_task4": _metrics(3.15, 0.10, 0.10),
        },
        "source-r2": {
            "task4_rule": _metrics(2.80), "task4_mixed": _metrics(2.90),
            "pooled_task4": _metrics(2.85),
        },
        EXTERNAL_LABEL: {
            "task4_rule": _metrics(6.0), "task4_mixed": _metrics(6.0),
            "pooled_task4": _metrics(6.0),
        },
    }
    cases = {name: {} for name in STANDARD_STRATA}
    for stratum in TASK4_STRATA:
        for case in protocol["evaluation"]["standard_strata"][stratum]["cases"]:
            cases[stratum][str(case["world_seed"])] = {
                "candidate-cnn": _metrics(3.3), "frozen-v4": _metrics(3.1),
                "source-r2": _metrics(2.8), EXTERNAL_LABEL: _metrics(6.0),
            }
    return rows, cases, {"p99": 1.0, "max": 2.0}


def test_protocol_fixes_candidate_budget_and_external_boundary():
    protocol, _, _ = load_protocol(PROTOCOL)
    assert tuple(protocol["internal_labels"]) == INTERNAL_LABELS
    assert protocol["fixed_candidate"] == "candidate-cnn"
    assert protocol["selection_free"] is True
    assert protocol["training_allowed"] is False
    assert protocol["external_reference_is_diagnostic_only"] is True
    assert protocol["external_reference_used_for_gate"] is False
    assert protocol["evaluation"]["budget"]["total_environment_rounds"] == 4200
    assert protocol["evaluation"]["budget"]["total_manifests"] == 88
    assert len(registered_seeds(protocol)) == 168


def test_dry_run_starts_no_game_checkout_or_followup():
    protocol, _, _ = load_protocol(PROTOCOL)
    summary = dry_run(protocol)
    assert summary["total_environment_rounds"] == 4200
    assert summary["external_diagnostic_rounds"] == 600
    assert summary["external_checkout_started"] is False
    assert summary["formal_evaluation_started"] is False
    assert summary["training_started"] is False
    assert summary["checkpoint_copy_allowed"] is False
    assert summary["automatic_followup_started"] is False


def test_all_eight_internal_blocks_can_pass_frozen_gate():
    protocol, _, _ = load_protocol(PROTOCOL)
    rows, cases, latency = _passing_inputs(protocol)
    result = decide(protocol, rows, cases, latency)
    assert result["candidate_confirmed"] is True
    assert result["task4_block_wins_out_of_8"] == 8
    assert all(result["checks"].values())
    assert result["decision"] == "candidate_confirmed_replace_v4_pending_docker_stop"


def test_suicide_guard_rejects_score_winner():
    protocol, _, _ = load_protocol(PROTOCOL)
    rows, cases, latency = _passing_inputs(protocol)
    rows["candidate-cnn"]["pooled_task4"] = _metrics(3.50, 0.14, 0.10)
    result = decide(protocol, rows, cases, latency)
    assert result["checks"]["pooled_task4_suicide_guard"] is False
    assert result["candidate_confirmed"] is False
    assert result["decision"] == "candidate_not_confirmed_retain_v4_stop"


def test_external_score_cannot_change_internal_decision():
    protocol, _, _ = load_protocol(PROTOCOL)
    rows, cases, latency = _passing_inputs(protocol)
    first = decide(protocol, rows, cases, latency)
    rows[EXTERNAL_LABEL]["pooled_task4"] = _metrics(0.0)
    second = decide(protocol, rows, cases, latency)
    assert first == second
    assert first["external_used_for_gate"] is False
