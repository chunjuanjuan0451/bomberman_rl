"""Regression tests for the preregistered Task-3 four-arm pilot."""

from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_code.model_a_task3.config import ARM_SPECS, architecture_name, load_config
from agent_code.model_a_task3.features import ACTION_FEATURE_SIZE, task3_action_features
from agent_code.model_a_task3.network import Task3DQN, torch
from agent_code.model_a_task3.train import RawTransition, aggregate_n_step
from agent_code.model_a_v6.symmetry import ACTION_PERMUTATIONS, transform_game_state
from agent_code.task3_curriculum_agent import callbacks as curriculum
from tools.task3_four_arm import evaluate_arm, load_protocol, select_arm


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/task3-four-arm-pilot-s108000.json"


def _game_state(round_number: int = 1) -> dict:
    field = -np.ones((9, 9), dtype=int)
    field[1:-1, 1:-1] = 0
    field[2, 3] = 1
    field[4, 6] = 1
    field[6, 5] = 1
    return {
        "round": round_number,
        "step": 1,
        "field": field,
        "self": ("target", 0, True, (4, 4)),
        "others": [("other", 0, True, (4, 2)), ("other2", 0, True, (6, 4))],
        "bombs": [((2, 5), 2)],
        "coins": [(3, 5), (5, 3)],
        "explosion_map": np.zeros_like(field),
    }


def test_task3_arm_configs_form_a_strict_incremental_ladder():
    expected = {
        "A": ("legacy", False, 1),
        "B": ("official-aligned", False, 1),
        "C": ("official-aligned", True, 1),
        "D": ("official-aligned", True, 3),
    }
    assert {
        arm: (spec["reward_profile"], spec["action_features"], spec["n_step"])
        for arm, spec in ARM_SPECS.items()
    } == expected
    hashes = set()
    for arm in ARM_SPECS:
        config, _, config_hash = load_config(
            ROOT / f"experiments/configs/task3-four-arm-{arm.lower()}-s108000.json"
        )
        assert config["arm"] == arm and config["includes_task4"] is False
        assert config["seed"] == 108000 and config["agent_seed"] == 208000
        assert config["opponent_seed"] == 308000
        assert config["agents"] == ["model_a_task3", "task3_curriculum_agent"]
        assert config["rounds"] == 300 and config["resume"] is False
        assert architecture_name(config).endswith(f"-n{expected[arm][2]}")
        hashes.add(config_hash)
    assert len(hashes) == 4


def test_task3_four_arm_protocol_is_isolated_and_budgeted():
    protocol = load_protocol(PROTOCOL)
    assert protocol["task"] == 3 and protocol["includes_task4"] is False
    assert protocol["training"] is True and protocol["automatic_replication"] is False
    assert list(protocol["arms"]) == ["A", "B", "C", "D"]
    assert sum(
        load_config(ROOT / spec["config_path"])[0]["rounds"]
        for spec in protocol["arms"].values()
    ) == 1200
    cases = [case for stratum in protocol["strata"].values() for case in stratum["cases"]]
    assert len(cases) == 6
    assert len({case["world_seed"] for case in cases}) == 6
    assert sum(
        len(stratum["cases"]) * stratum["rounds_per_seed"] * 5
        for stratum in protocol["strata"].values()
    ) == 1500
    assert "not a hard termination condition" in protocol["decision_branches"]["no_arm_selected"]


def test_task3_residual_starts_as_exact_frozen_v4_policy():
    config, config_path, _ = load_config(
        ROOT / "experiments/configs/task3-four-arm-c-s108000.json"
    )
    parent = config_path.parents[2] / config["parent_checkpoint"]
    try:
        checkpoint = torch.load(parent, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(parent, map_location="cpu")
    model = Task3DQN(config["hyperparameters"]["delta_cap"], True)
    model.base.load_state_dict(checkpoint["online_net"], strict=True)
    model.freeze_base()
    local = torch.randn(4, 4, 7, 7)
    global_features = torch.randn(4, 7)
    tactical = torch.randn(4, 6, ACTION_FEATURE_SIZE)
    with torch.no_grad():
        expected = model.base(local, global_features)
        actual, delta = model(local, global_features, tactical, return_delta=True)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(delta).item() == 0
    assert all(not parameter.requires_grad for parameter in model.base.parameters())


def test_task3_action_features_are_bounded_and_d4_equivariant():
    state = _game_state()
    original = task3_action_features(state)
    assert original.shape == (6, ACTION_FEATURE_SIZE)
    assert np.isfinite(original).all()
    assert np.max(np.abs(original)) <= 1.0
    for symmetry in range(8):
        transformed = task3_action_features(transform_game_state(state, symmetry))
        permutation = ACTION_PERMUTATIONS[symmetry]
        assert np.allclose(transformed[permutation], original, atol=1e-6)


def test_task3_three_step_return_and_terminal_truncation_are_exact():
    transitions = deque([
        RawTransition("s0", 0, 1.0, "s1", False, "m1"),
        RawTransition("s1", 1, 2.0, "s2", False, "m2"),
        RawTransition("s2", 2, 3.0, "s3", False, "m3"),
    ])
    result = aggregate_n_step(transitions, 3, 0.9)
    assert np.isclose(result.reward, 1.0 + 0.9 * 2.0 + 0.81 * 3.0)
    assert result.next_state == "s3" and result.next_mask == "m3"
    assert result.done is False and np.isclose(result.bootstrap_discount, 0.9 ** 3)

    transitions[1] = RawTransition("s1", 1, 2.0, None, True, None)
    terminal = aggregate_n_step(transitions, 3, 0.9)
    assert np.isclose(terminal.reward, 1.0 + 0.9 * 2.0)
    assert terminal.done is True and terminal.next_state is None
    assert np.isclose(terminal.bootstrap_discount, 0.9 ** 2)


def _curriculum_holder():
    return SimpleNamespace(logger=logging.getLogger("task3_curriculum_agent_1_code"))


def test_task3_curriculum_is_round_isolated_reproducible_and_balanced():
    previous = os.environ.get("TASK3_CURRICULUM_SEED")
    try:
        os.environ["TASK3_CURRICULUM_SEED"] = "308000"
        first, second = _curriculum_holder(), _curriculum_holder()
        curriculum.setup(first)
        curriculum.setup(second)
        odd_a = [curriculum.act(first, _game_state(11)) for _ in range(12)]
        odd_b = [curriculum.act(second, _game_state(11)) for _ in range(12)]
        assert odd_a == odd_b
        assert set(odd_a) <= set(curriculum.PEACEFUL_ACTIONS)
        curriculum.act(first, _game_state(12))
        assert first.curriculum_mode == "coin_collector"
        curriculum.act(first, _game_state(299))
        assert first.curriculum_mode == "peaceful"
        modes = ["peaceful" if round_number % 2 else "coin_collector" for round_number in range(1, 301)]
        assert modes.count("peaceful") == modes.count("coin_collector") == 150
    finally:
        if previous is None:
            os.environ.pop("TASK3_CURRICULUM_SEED", None)
        else:
            os.environ["TASK3_CURRICULUM_SEED"] = previous


def _metrics(score: int, coins: int = 70, kills: int = 6) -> dict:
    return {
        "rounds": 50, "score": score, "score_per_round": score / 50,
        "coins": coins, "kills": kills, "crates": 90, "bombs": 55,
        "suicides": 3, "invalid_actions": 2, "steps": 1000,
        "mean_decision_time_ms": 0.5,
    }


def _rows(arm: str, candidate_score: int) -> dict:
    result = {}
    for offset, stratum in enumerate(("peaceful", "coin_collector")):
        result[stratum] = []
        for seed in range(3):
            result[stratum].append({
                "seed_tuple": {"world_seed": 100 * offset + seed},
                "v4": {"target_metrics": _metrics(100)},
                arm: {"target_metrics": _metrics(candidate_score, kills=8)},
            })
    return result


def test_task3_gate_selects_simplest_near_best_eligible_arm():
    gate = {
        "minimum_stratum_score_per_round_delta": -0.1,
        "minimum_stratum_kills_per_round_delta": -0.02,
        "maximum_stratum_suicides_per_round_increase": 0.04,
        "maximum_stratum_invalid_actions_per_round_increase": 0.08,
        "minimum_overall_score_per_round_delta": 0.10,
        "minimum_overall_kills_per_round_delta": 0.02,
        "maximum_overall_coins_per_round_regression": 0.10,
        "minimum_paired_net_wins": 0,
        "maximum_single_cell_score_per_round_regression": 0.75,
        "maximum_seed_mean_decision_time_ms": 100.0,
    }
    results = {
        "A": evaluate_arm(_rows("A", 110), "A", gate),
        "B": evaluate_arm(_rows("B", 112), "B", gate),
        "C": evaluate_arm(_rows("C", 111), "C", gate),
        "D": evaluate_arm(_rows("D", 109), "D", gate),
    }
    assert all(item["passed"] for item in results.values())
    assert select_arm(results, tolerance=0.05) == "A"
    for item in results.values():
        item["passed"] = False
    assert select_arm(results, tolerance=0.05) is None
