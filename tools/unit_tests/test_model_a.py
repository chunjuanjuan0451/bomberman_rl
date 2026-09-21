import numpy as np

from agent_code.model_a_dqn.features import (
    GLOBAL_SIZE,
    LOCAL_SHAPE,
    blast_positions,
    danger_time_map,
    legal_action_mask,
    state_to_features,
)


def _state(with_escape: bool = True):
    field = -np.ones((9, 9), dtype=int)
    field[1:8, 1:8] = 0
    if not with_escape:
        field[3, 4] = field[5, 4] = field[4, 3] = field[4, 5] = -1
    return {
        "round": 1, "step": 1, "field": field, "self": ("me", 0, True, (4, 4)),
        "others": [], "bombs": [], "coins": [(3, 4)], "explosion_map": np.zeros_like(field),
        "user_input": None,
    }


def test_model_a_feature_shapes_and_safe_bomb_mask():
    local, global_features = state_to_features(_state())
    assert local.shape == LOCAL_SHAPE
    assert global_features.shape == (GLOBAL_SIZE,)
    assert legal_action_mask(_state())[-1]
    assert not legal_action_mask(_state(with_escape=False))[-1]


def test_model_a_offensive_global_features_are_bounded_and_informative():
    state = _state()
    state["others"] = [("opponent", 0, True, (7, 4))]

    _, global_features = state_to_features(state)

    assert np.all((0.0 <= global_features) & (global_features <= 1.0))
    assert global_features[4] > 0.0  # Opponent proximity.
    assert global_features[5] > 0.0  # Boundary/walls constrain its exits.
    assert global_features[6] == 1.0  # A safe bomb here hits the opponent.


def test_model_a_masks_an_immediately_dangerous_move_when_an_exit_exists():
    state = _state()
    # The RIGHT tile is in a bomb blast due next turn, while LEFT stays safe.
    state["bombs"] = [((5, 6), 1)]
    mask = legal_action_mask(state)
    assert not mask[1]  # RIGHT
    assert mask[3]  # LEFT


def test_model_a_blast_and_danger_continue_through_crates():
    state = _state()
    state["field"][5, 4] = 1
    state["bombs"] = [((4, 4), 2)]

    assert (6, 4) in blast_positions(state["field"], (4, 4))
    assert danger_time_map(state)[6, 4] == 2


def test_model_a_v3_rejects_wait_that_consumes_escape_time():
    state = _state()
    # From the centre of this blast, waiting leaves too little time to move
    # three tiles sideways before the timer-zero bomb explodes.
    state["bombs"] = [((4, 4), 2)]

    mask = legal_action_mask(state)

    assert not mask[4]  # WAIT
    assert any(mask[:4])


def test_model_a_v3_keeps_only_survivable_first_step():
    state = _state()
    state["bombs"] = [((4, 4), 1)]
    # Walls make RIGHT the only route that turns out of the blast in time.
    state["field"][3, 4] = -1
    state["field"][4, 3] = -1
    state["field"][4, 5] = -1

    mask = legal_action_mask(state)

    assert mask[1]
    assert not mask[0]
    assert not mask[2]
    assert not mask[3]
    assert not mask[4]


def test_model_a_v3_accepts_survivable_single_route_with_timing_slack():
    state = _state()
    # A winding corridor needs three moves and exposes only one first step.
    # It still leaves the blast one action before detonation, so the balanced
    # v3 gate keeps this useful crate-clearing placement.
    state["field"][3, 4] = -1
    state["field"][4, 3] = -1
    state["field"][4, 5] = -1
    state["field"][5, 3] = -1
    state["field"][5, 5] = -1

    assert legal_action_mask(state)[5]


def test_model_a_v3_avoids_contested_exit_when_alternative_exists():
    state = _state()
    state["bombs"] = [((4, 4), 2)]
    state["others"] = [("opponent", 0, True, (6, 4))]

    mask = legal_action_mask(state)

    assert not mask[1]  # RIGHT could be claimed by the opponent this turn.
    assert mask[0] or mask[2] or mask[3]
