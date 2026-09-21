"""Regression tests for the Task-3 training-distribution experiment."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import settings as s
from agent_code.model_a_task3_killrich.config import load_config
from tools.task3_distribution_world import (
    KILLRICH_CRATE_DENSITY, KILLRICH_START_POSITION_PAIRS,
    Task3TrainingDistributionWorld,
)
from tools.task3_training_distribution import LABELS, load_protocol


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/task3-training-distribution-matched-s112000.json"


def _algorithm_contract(config: dict) -> dict:
    ignored = {
        "scenario", "training_distribution", "run_id", "variant", "checkpoint_path",
    }
    return {key: value for key, value in config.items() if key not in ignored}


def test_killrich_scenario_changes_only_registered_state_distribution_fields():
    classic = s.SCENARIOS["classic"]
    assert (s.COLS, s.ROWS, s.MAX_STEPS, s.BOMB_POWER, s.BOMB_TIMER) == (17, 17, 400, 3, 4)
    assert classic["CRATE_DENSITY"] == 0.75 and classic["COIN_COUNT"] == 9
    assert KILLRICH_CRATE_DENSITY == 0.15
    assert len(KILLRICH_START_POSITION_PAIRS) == 12
    for pair in KILLRICH_START_POSITION_PAIRS:
        assert len(pair) == 2
        assert 2 <= sum(abs(a - b) for a, b in zip(*pair)) <= 4
        assert all(not ((x + 1) * (y + 1) % 2 == 1) for x, y in pair)


def test_arena_uses_only_one_registered_pair_and_rejects_a_third_agent():
    agents = [SimpleNamespace(x=None, y=None), SimpleNamespace(x=None, y=None)]
    world = Task3TrainingDistributionWorld.__new__(Task3TrainingDistributionWorld)
    world.rng = np.random.default_rng(112000)
    world.args = SimpleNamespace(scenario="classic")
    world.agents = agents
    os.environ["TASK3_TRAINING_DISTRIBUTION"] = "kill-rich"
    try:
        arena, coins, active = world.build_arena()
    finally:
        os.environ.pop("TASK3_TRAINING_DISTRIBUTION", None)
    starts = {(agent.x, agent.y) for agent in active}
    expected_pairs = [{tuple(position) for position in pair}
                      for pair in KILLRICH_START_POSITION_PAIRS]
    assert starts in expected_pairs
    assert arena.shape == (17, 17) and coins == [] and len(active) == 2
    assert all(arena[position] == 0 for position in starts)

    world.agents = agents + [SimpleNamespace(x=None, y=None)]
    os.environ["TASK3_TRAINING_DISTRIBUTION"] = "kill-rich"
    try:
        world.build_arena()
    except RuntimeError as exc:
        assert "exactly two agents" in str(exc)
    else:
        raise AssertionError("kill-rich scenario accepted a third agent")
    finally:
        os.environ.pop("TASK3_TRAINING_DISTRIBUTION", None)


def test_three_matched_pairs_differ_only_in_training_distribution_and_artifacts():
    for index, seed in enumerate((112100, 112101, 112102), start=1):
        control, _, _ = load_config(ROOT / f"experiments/configs/task3-distribution-control-r{index}-s{seed}.json")
        candidate, _, _ = load_config(ROOT / f"experiments/configs/task3-distribution-killrich-r{index}-s{seed}.json")
        assert _algorithm_contract(control) == _algorithm_contract(candidate)
        assert control["scenario"] == "classic"
        assert candidate["scenario"] == "classic"
        assert control["training_distribution"] == "classic-control"
        assert candidate["training_distribution"] == "kill-rich"
        assert control["sampling_profile"] == candidate["sampling_profile"] == "uniform"
        assert control["parent_sha256"] == candidate["parent_sha256"]


def test_protocol_has_signal_gate_before_conditional_training_and_no_task4():
    protocol = load_protocol(PROTOCOL)
    assert protocol["task"] == 3 and protocol["includes_task4"] is False
    assert protocol["automatic_followup"] is False
    assert protocol["formal_budget"] == {
        "signal_runs": 16, "signal_rounds": 400,
        "conditional_training_runs": 6, "conditional_training_rounds": 1800,
        "conditional_evaluation_runs": 56, "conditional_evaluation_rounds": 2800,
    }
    assert protocol["signal_gate"]["minimum_candidate_kills"] == 40
    assert protocol["signal_gate"]["minimum_positive_cases"] == 6
    assert protocol["signal_gate"]["minimum_kill_step_rate"] == "1/250"
    assert protocol["signal_gate"]["minimum_rate_multiplier"] == 10
    assert protocol["selection"]["order"] == ["K1", "K2", "K3"]
    assert LABELS == ("v4", "C1", "K1", "C2", "K2", "C3", "K3")
    assert all(stratum["rounds_per_seed"] == 50 for stratum in protocol["strata"].values())
    output = ROOT / protocol["output_path"]
    if output.exists():
        report = __import__("json").loads(output.read_text())
        assert report["status"] == "completed"
        assert report["decision"] in {
            "environment_signal_failed_stop", "killrich_distribution_confirmed",
            "killrich_distribution_not_confirmed_stop",
        }
        assert report["automatic_followup_started"] is False
        assert report["task4_started"] is False


def test_all_decision_branches_stop_without_automatic_followup():
    protocol = load_protocol(PROTOCOL)
    assert set(protocol["decision_branches"]) == {
        "environment_signal_failed_stop",
        "killrich_distribution_confirmed",
        "killrich_distribution_not_confirmed_stop",
    }
    assert all("Stop" in text for text in protocol["decision_branches"].values())
    assert all("Task 4" in text for text in protocol["decision_branches"].values())
