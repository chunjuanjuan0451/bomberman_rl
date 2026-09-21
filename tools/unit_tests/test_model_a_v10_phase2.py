"""Phase-2 callback and budget-policy tests; no official environment edits."""

import numpy as np

from agent_code.model_a_v10.runtime import (NORMAL_DEADLINE_MS, NORMAL_MIN_SIMULATIONS,
                                             TACTICAL_DEADLINE_MS, TACTICAL_MIN_SIMULATIONS,
                                             Phase2Controller, is_tactical, search_budget_for_profile,
                                             state_features)
from agent_code.model_a_v10.simulator import SimAgent, SimBomb, SimState
from agent_code.model_a_v10.tactics import tactical_cases


def _open_state():
    field = -np.ones((9, 9), dtype=int)
    field[1:-1, 1:-1] = 0
    return SimState(field, [SimAgent("root", 3, 3), SimAgent("other", 7, 7)])


def test_v10_phase2_feature_schema_and_normal_budget():
    state = _open_state()
    features = state_features(state)
    assert features.shape == (5, 9, 9) and features.dtype == np.float32
    assert not is_tactical(state)
    result = Phase2Controller().select(state, seed=2)
    assert result.simulations >= NORMAL_MIN_SIMULATIONS
    assert result.reason == "deadline" and result.elapsed_ms >= NORMAL_DEADLINE_MS


def test_v10_phase2_tactical_budget_and_legal_action():
    state = _open_state()
    state.bombs.append(SimBomb(3, 2, 1, 1))
    assert is_tactical(state)
    result = Phase2Controller().select(state, seed=3)
    assert result.simulations >= TACTICAL_MIN_SIMULATIONS
    assert result.reason == "deadline" and result.elapsed_ms >= TACTICAL_DEADLINE_MS
    assert result.action in result.raw_prior


def test_v10_phase2_controller_observer_adapter_returns_action():
    case = next(case for case in tactical_cases() if case.name == "unique_escape")
    root = case.state.agents[0]
    game_state = {"round": 3, "step": 9, "field": case.state.field,
                  "self": (root.name, root.score, root.bombs_left, (root.x, root.y)),
                  "others": [], "bombs": [((bomb.x, bomb.y), bomb.timer) for bomb in case.state.bombs],
                  "coins": [], "explosion_map": case.state.explosion_map(), "user_input": "WAIT"}
    action = Phase2Controller().act(game_state)
    assert action in ("WAIT", "RIGHT", "BOMB")


def test_v10_phase2_ryzen_budget_is_named_and_leaves_safety_margin():
    default = search_budget_for_profile("research-m4-v1")
    target = search_budget_for_profile("ryzen-safe-v1")
    assert target.normal_deadline_ms > default.normal_deadline_ms
    assert target.tactical_deadline_ms > default.tactical_deadline_ms
    assert target.tactical_deadline_ms == 300.0
    assert target.tactical_deadline_ms < 500.0
    utilized = search_budget_for_profile("ryzen-utilized-v2")
    assert utilized.normal_deadline_ms == 240.0
    assert utilized.tactical_deadline_ms == 400.0
    assert utilized.normal_min_simulations == 50
    assert utilized.tactical_min_simulations == 100
    assert utilized.tactical_deadline_ms < 500.0
    try:
        search_budget_for_profile("typo")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown budget profile was accepted")
