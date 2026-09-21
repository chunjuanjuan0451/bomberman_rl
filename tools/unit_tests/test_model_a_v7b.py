from time import perf_counter

import numpy as np

from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask
from agent_code.model_a_v7b.tactical import evaluate_bomb, maybe_override_with_bomb


def _state(trapped: bool):
    field = -np.ones((9, 9), dtype=int)
    field[1:8, 1:8] = 0
    if trapped:
        # The opponent can retreat along the blast line but cannot turn out of
        # it before offset 5. The agent still has perpendicular escape space.
        for x in (5, 6, 7):
            field[x, 3] = -1
            field[x, 5] = -1
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("me", 0, True, (4, 4)),
        "others": [("other", 0, True, (7, 4))],
        "bombs": [], "coins": [], "explosion_map": np.zeros_like(field),
        "user_input": None,
    }


def test_v7b_rollout_detects_a_counterfactual_guaranteed_trap():
    tactics = evaluate_bomb(_state(True), perf_counter() + 1.0)
    assert tactics is not None
    assert tactics.guaranteed_traps == 1
    assert tactics.own_terminal_positions >= 3


def test_v7b_rollout_does_not_call_an_open_arena_a_trap():
    tactics = evaluate_bomb(_state(False), perf_counter() + 1.0)
    assert tactics is not None
    assert tactics.guaranteed_traps == 0


def test_v7b_only_overrides_for_a_safe_high_confidence_trap():
    state = _state(True)
    mask = legal_action_mask(state)
    baseline = ACTIONS.index("LEFT")
    selected, tactics, reason = maybe_override_with_bomb(
        state, baseline, mask, perf_counter() + 1.0,
    )
    assert mask[ACTIONS.index("BOMB")]
    assert selected == ACTIONS.index("BOMB")
    assert tactics is not None and reason == "guaranteed-trap"


def test_v7b_deadline_preserves_v4_action():
    state = _state(True)
    mask = legal_action_mask(state)
    baseline = ACTIONS.index("LEFT")
    selected, tactics, _ = maybe_override_with_bomb(
        state, baseline, mask, perf_counter() - 1.0,
    )
    assert selected == baseline and tactics is None


def test_v7b_open_space_does_not_trigger_a_pressure_override():
    state = _state(False)
    mask = legal_action_mask(state)
    baseline = ACTIONS.index("LEFT")
    selected, _, reason = maybe_override_with_bomb(
        state, baseline, mask, perf_counter() + 1.0,
    )
    assert selected == baseline
    assert reason == "no-high-confidence-trap"
