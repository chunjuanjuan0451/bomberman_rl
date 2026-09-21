"""Tests for the opponent-aware s103011 collection and training route."""

import os
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_code.seeded_random_agent.callbacks import act, setup
from agent_code.model_a_v10.interfaces import BlendedExpertNetwork, LinearExpertNetwork
from agent_code.model_a_v10.runtime import opponent_aware_state_features, state_features
from tools.v10_multiplayer_policy_train import (
    balanced_indices, collect_episode, initial_multiplayer_state,
)


def _config():
    return {"opponent_count": 3, "search_horizon_plies": 2,
            "teacher_search_simulations": 2, "max_steps": 2}


def test_multiplayer_state_contains_three_visible_opponents():
    state = initial_multiplayer_state(77, 2)
    original = state_features(state)
    features = opponent_aware_state_features(state)
    assert len(state.agents) == 4
    assert np.count_nonzero(features[3] < -0.5) == 3
    assert np.count_nonzero(features[3] > 0.5) == 1
    assert np.count_nonzero(original[3] < -0.5) < 3


def test_seeded_random_agent_is_reproducible_per_index():
    old = os.environ.get("SEEDED_RANDOM_AGENT_SEED")
    try:
        os.environ["SEEDED_RANDOM_AGENT_SEED"] = "991"
        # Match the fake callback object constructed by official AgentRunner:
        # it has a logger, but no name attribute.
        first = SimpleNamespace(logger=logging.getLogger("seeded_random_agent_1_code"))
        second = SimpleNamespace(logger=logging.getLogger("seeded_random_agent_1_code"))
        other = SimpleNamespace(logger=logging.getLogger("seeded_random_agent_2_code"))
        setup(first); setup(second); setup(other)
        first_actions = [act(first, {}) for _ in range(20)]
        assert first_actions == [act(second, {}) for _ in range(20)]
        assert first_actions != [act(other, {}) for _ in range(20)]
    finally:
        if old is None:
            os.environ.pop("SEEDED_RANDOM_AGENT_SEED", None)
        else:
            os.environ["SEEDED_RANDOM_AGENT_SEED"] = old


def test_collection_preserves_opponents_and_soft_target_schema():
    state = initial_multiplayer_state(78, 2)
    model = LinearExpertNetwork.load(
        Path("experiments/checkpoints/model-a-v10-centered-blend-coin-heaven-s103004.npz")
    )
    teacher = BlendedExpertNetwork(model, policy_weight=0.0, value_weight=0.1,
                                   opponent_weight=0.0, risk_weight=0.0)
    arrays, terminal = collect_episode(state, teacher, _config(), 78)
    count = len(arrays["features"])
    assert 1 <= count <= 2 and terminal.step_count == count
    assert arrays["features"].shape == (count, 5, 17, 17)
    assert arrays["opponent_actions"].shape == (count, 3)
    assert np.allclose(arrays["visit_policy"].sum(axis=1), 1.0)
    assert not arrays["legal_mask"][:, 5].any()
    assert np.any(arrays["features"][:, 3] < -0.5)


def test_collection_stops_after_multiplayer_interaction_ends():
    state = initial_multiplayer_state(79, 400)
    for opponent in state.agents[1:]:
        opponent.alive = False
    model = LinearExpertNetwork.load(
        Path("experiments/checkpoints/model-a-v10-centered-blend-coin-heaven-s103004.npz")
    )
    teacher = BlendedExpertNetwork(model, policy_weight=0.0, value_weight=0.1,
                                   opponent_weight=0.0, risk_weight=0.0)
    try:
        collect_episode(state, teacher, _config(), 79)
    except ValueError as exc:
        assert "no root states" in str(exc)
    else:
        raise AssertionError("solo cleanup state was collected as multiplayer data")


def test_wait_balancing_is_reproducible_and_respects_cap():
    actions = np.asarray([4] * 20 + [0] * 10, dtype=np.int8)
    first = balanced_indices(actions, 0.4, np.random.default_rng(5))
    second = balanced_indices(actions, 0.4, np.random.default_rng(5))
    assert np.array_equal(first, second)
    assert np.mean(actions[first] == 4) <= 0.4
