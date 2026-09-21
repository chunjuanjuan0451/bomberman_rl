from time import perf_counter

import numpy as np

from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask
from agent_code.model_a_v7a.planner import (
    _bomb_schedule,
    _traversable,
    profile_action,
    select_action,
)


def _field():
    field = -np.ones((9, 9), dtype=int)
    field[1:8, 1:8] = 0
    return field


def _state(position=(4, 4), bombs=(), can_bomb=True):
    field = _field()
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("me", 0, can_bomb, position), "others": [],
        "bombs": list(bombs), "coins": [],
        "explosion_map": np.zeros_like(field), "user_input": None,
    }


def test_v7a_profiles_survive_through_the_complete_known_hazard_schedule():
    state = _state(bombs=[((4, 4), 1)], can_bomb=False)
    profile = profile_action(state, "RIGHT", perf_counter() + 1.0)
    assert profile is not None and profile.survived
    assert profile.horizon >= 4
    assert profile.terminal_positions > 1


def test_v7a_rejects_a_root_action_hit_on_the_current_step():
    state = _state()
    state["explosion_map"][5, 4] = 1
    profile = profile_action(state, "RIGHT", perf_counter() + 1.0)
    assert profile is not None and not profile.survived


def test_v7a_bomb_search_models_the_new_bomb_and_escape():
    state = _state(position=(1, 1))
    state["field"][1, 2] = 1
    profile = profile_action(state, "BOMB", perf_counter() + 1.0)
    assert profile is not None
    assert profile.horizon >= 7
    assert profile.survived


def test_v7a_deadline_returns_the_precomputed_v4_fallback():
    state = _state()
    mask = legal_action_mask(state)
    q_values = np.arange(len(ACTIONS), dtype=np.float32)
    action, diagnostics = select_action(
        state, q_values, mask, np.random.default_rng(1), perf_counter() - 1.0,
    )
    expected = int(np.flatnonzero(mask)[np.argmax(q_values[mask])])
    assert action == expected
    assert diagnostics["fallback"] and diagnostics["reason"] == "deadline"


def test_v7a_does_not_treat_a_distant_explosion_as_an_emergency():
    state = _state(bombs=[((4, 4), 4)], can_bomb=False)
    mask = legal_action_mask(state)
    q_values = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32)
    action, diagnostics = select_action(
        state, q_values, mask, np.random.default_rng(1), perf_counter() + 1.0,
    )
    assert action == int(np.flatnonzero(mask)[np.argmax(q_values[mask])])
    assert diagnostics["reason"] == "v4-q"


def test_v7a_exact_search_can_use_a_crate_after_a_known_blast_clears_it():
    state = _state(position=(3, 4), bombs=[((4, 4), 0)], can_bomb=False)
    state["field"][5, 4] = 1
    bombs, _, crate_clear, _ = _bomb_schedule(state, False)
    assert not _traversable(state["field"], (5, 4), (5, 3), 1, bombs, crate_clear)
    assert _traversable(state["field"], (5, 4), (5, 3), 2, bombs, crate_clear)
