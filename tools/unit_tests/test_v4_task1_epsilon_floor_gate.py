"""Contract tests for the exact-v4 Task1 epsilon-floor signal gate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent_code.model_a_dqn import train as v4
from agent_code.model_a_v4_epsilon_gate import train as epsilon_train
from agent_code.model_a_v4_epsilon_gate.config import REPLICAS, load_protocol
from tools.v4_task1_epsilon_floor_gate import curve_summary, decide, dry_run


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task1-epsilon-floor-e015-paired-s114000.json"


def test_protocol_changes_only_task1_action_selection_epsilon_floor():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert protocol["single_changed_variable"] == (
        "action-selection epsilon lower bound: 0.05 control versus 0.15 candidate"
    )
    assert protocol["training_rounds_per_replica"] == 400
    assert protocol["total_training_rounds"] == 1200
    assert protocol["epsilon"]["floor"] == 0.15
    assert protocol["epsilon"]["floor_first_applies_at_transition"] == 71579
    assert tuple(protocol["replicas"]) == REPLICAS
    assert [protocol["replicas"][replica]["world_seed"] for replica in REPLICAS] == [
        114001, 114002, 114003,
    ]


def test_epsilon_clamp_preserves_original_slope_above_floor():
    holder = SimpleNamespace(epsilon=0.42)
    epsilon_train._apply_floor(holder)
    assert holder.epsilon == 0.42
    holder.epsilon = 0.05
    epsilon_train._apply_floor(holder)
    assert holder.epsilon == 0.15
    assert v4.EPSILON_START == 1.0
    assert v4.EPSILON_FINAL == 0.05
    assert v4.EPSILON_DECAY_STEPS == 80_000


def test_dry_run_is_bounded_and_stops_before_followup():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["training_rounds"] == 1200
    assert summary["evaluation_rounds"] == 500
    assert summary["formal_experiment_started"] is False
    assert summary["task2_started"] is False
    assert summary["matched_full_restart_started"] is False
    assert len(summary["training"]) == 3


def test_curve_gate_uses_four_fixed_100_round_windows():
    stats = {"by_round": {}}
    values = [20] * 100 + [40] * 100 + [35] * 100 + [36] * 100
    for index, coins in enumerate(values, 1):
        stats["by_round"][f"Round {index:03d}"] = {"coins": coins}
    curve = curve_summary(stats)
    assert curve["nonoverlapping_100_round_coin_means"] == [20, 40, 35, 36]
    assert curve["final_to_best_preceding_ratio"] == 0.9


def _training(ratio: float) -> dict:
    return {
        "curve": {"final_to_best_preceding_ratio": ratio},
    }


def _row(score: float) -> dict:
    return {"score_per_round": score}


def test_decision_requires_stability_paired_support_and_fresh_quality():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    training = {replica: _training(0.95) for replica in REPLICAS}
    rows = {
        "paired_seen": {
            "candidate-r1": _row(30.0),
            "candidate-r2": _row(20.0),
            "candidate-r3": _row(31.0),
        },
        "fresh_confirmation": {
            "candidate-r1": _row(31.0),
            "candidate-r2": _row(30.0),
            "candidate-r3": _row(32.0),
        },
    }
    result = decide(protocol, training, rows)
    assert result["passed"] is True
    assert result["decision"] == "epsilon_floor_signal_supported"

    rows["fresh_confirmation"]["candidate-r2"] = _row(9.0)
    result = decide(protocol, training, rows)
    assert result["passed"] is False
    assert result["catastrophic_fresh_replicas"] == ["r2"]
    assert result["decision"] == "epsilon_floor_signal_rejected"
