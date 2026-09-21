"""Regression tests for the isolated Task-3 v6s2 checkpoint gate."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_code.seeded_coin_collector_agent import callbacks as seeded_coin
from agent_code.seeded_peaceful_agent import callbacks as seeded_peaceful
from tools.task3_v6s2_gate import evaluate_gate, load_protocol


ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "experiments/configs/task3-v6s2-s7501-gate-s107300.json"


def _holder(name: str):
    return SimpleNamespace(logger=logging.getLogger(name))


def _game_state():
    field = -np.ones((9, 9), dtype=int)
    field[1:-1, 1:-1] = 0
    field[4, 3] = 1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("opponent", 0, True, (3, 3)),
        "others": [("target", 0, True, (5, 5))],
        "bombs": [],
        "coins": [(6, 3)],
        "explosion_map": np.zeros_like(field),
    }


def test_seeded_task3_opponents_are_reproducible():
    previous = os.environ.get("TASK3_OPPONENT_SEED")
    try:
        os.environ["TASK3_OPPONENT_SEED"] = "307300"
        first = _holder("seeded_peaceful_agent_0_code")
        second = _holder("seeded_peaceful_agent_0_code")
        other = _holder("seeded_peaceful_agent_1_code")
        for holder in (first, second, other):
            seeded_peaceful.setup(holder)
        sequence_a = [seeded_peaceful.act(first, {}) for _ in range(24)]
        sequence_b = [seeded_peaceful.act(second, {}) for _ in range(24)]
        sequence_c = [seeded_peaceful.act(other, {}) for _ in range(24)]
        assert sequence_a == sequence_b
        assert sequence_a != sequence_c

        coin_a = _holder("seeded_coin_collector_agent_code")
        coin_b = _holder("seeded_coin_collector_agent_code")
        seeded_coin.setup(coin_a)
        seeded_coin.setup(coin_b)
        assert [seeded_coin.act(coin_a, _game_state()) for _ in range(12)] == [
            seeded_coin.act(coin_b, _game_state()) for _ in range(12)
        ]
    finally:
        if previous is None:
            os.environ.pop("TASK3_OPPONENT_SEED", None)
        else:
            os.environ["TASK3_OPPONENT_SEED"] = previous


def _metrics(score: int, coins: int, kills: int) -> dict:
    return {
        "rounds": 50, "score": score, "score_per_round": score / 50,
        "coins": coins, "kills": kills, "crates": 90, "bombs": 55,
        "suicides": 4, "invalid_actions": 2, "steps": 1_000,
        "mean_decision_time_ms": 0.4,
    }


def _row(seed: int) -> dict:
    return {
        "seed_tuple": {"world_seed": seed, "agent_seed": seed + 100_000,
                       "opponent_seed": seed + 200_000},
        "v4": {"target_metrics": _metrics(100, 70, 6)},
        "candidate": {"target_metrics": _metrics(110, 72, 8)},
    }


def test_task3_gate_requires_official_score_and_kill_gain_in_both_strata():
    gate = {
        "minimum_stratum_score_per_round_delta": 0.0,
        "minimum_stratum_kills_per_round_delta": 0.0,
        "minimum_overall_score_per_round_delta": 0.15,
        "minimum_overall_kills_per_round_delta": 0.03,
        "maximum_overall_coins_per_round_regression": 0.10,
        "minimum_paired_net_wins": 2,
        "maximum_single_cell_score_per_round_regression": 0.50,
        "maximum_suicides_per_round_increase": 0.02,
        "maximum_invalid_actions_per_round_increase": 0.05,
        "maximum_seed_mean_decision_time_ms": 100.0,
    }
    rows = {
        "peaceful": [_row(1), _row(2)],
        "coin_collector": [_row(3), _row(4)],
    }
    result = evaluate_gate(rows, gate)
    assert result["passed"]
    rows["coin_collector"][0]["candidate"]["target_metrics"] = _metrics(80, 72, 4)
    failed = evaluate_gate(rows, gate)
    assert not failed["strata"]["coin_collector"]["passed"]
    assert not failed["passed"]


def test_task3_protocol_is_evaluation_only_and_excludes_task4():
    protocol = load_protocol(GATE)
    assert protocol["task"] == 3 and protocol["includes_task4"] is False
    assert protocol["training"] is False and protocol["automatic_training"] is False
    assert protocol["arms"]["candidate"]["checkpoint_sha256"].startswith("bbc3e395")
    assert set(protocol["strata"]) == {"peaceful", "coin_collector"}
    peaceful = {case["world_seed"] for case in protocol["strata"]["peaceful"]["cases"]}
    coin = {case["world_seed"] for case in protocol["strata"]["coin_collector"]["cases"]}
    assert peaceful.isdisjoint(coin)
