import numpy as np

from agent_code.model_b_linear.features import (
    FEATURE_SIZE,
    _blast_positions,
    danger_time_map,
    legal_action_mask,
    shortest_safe_coin_distance,
    state_to_features,
)
from agent_code.model_b_linear.potential import shaping_reward
from agent_code.model_b_linear.train import reward_from_events
import events as e


def make_state():
    field = -np.ones((7, 7), dtype=int)
    field[1:6, 1:6] = 0
    field[3, 2] = 1
    return {
        "round": 1, "step": 1, "field": field, "self": ("me", 0, 1, (2, 2)),
        "others": [], "bombs": [((4, 2), 2)], "coins": [(2, 4)],
        "explosion_map": np.zeros_like(field), "user_input": None,
    }


def test_model_b_features_and_legal_actions():
    state = make_state()
    assert state_to_features(state).shape == (FEATURE_SIZE,)
    assert not legal_action_mask(state)[1]  # RIGHT is blocked by a crate.
    assert shortest_safe_coin_distance(state) == 2


def test_potential_shaping_is_bounded_for_same_state():
    assert abs(shaping_reward(make_state(), make_state())) < 0.01


def test_reward_avoids_waiting_and_penalizes_unneeded_bombs():
    assert reward_from_events([e.SURVIVED_ROUND]) == 0.0
    assert reward_from_events([e.BOMB_DROPPED]) == -0.2


def test_model_b_blast_and_danger_continue_through_crates():
    state = make_state()
    state["field"][3, 3] = 1
    state["bombs"] = [((3, 2), 2)]

    assert (3, 4) in _blast_positions(state["field"], (3, 2))
    assert danger_time_map(state)[3, 4] == 2
