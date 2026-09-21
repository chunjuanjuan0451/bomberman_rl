import numpy as np

from agent_code.model_a_cnn_n8_postbomb_movement_audit.callbacks import (
    immediate_collision_action_details,
    robust_postbomb_action_details,
)
from tools.cnn_n8_postbomb_movement_audit import _fraction


def _state():
    field = np.zeros((7, 7), dtype=np.int16)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "round": 1,
        "step": 2,
        "field": field,
        "self": ("me", 0, False, (2, 2)),
        "others": [("other", 0, True, (5, 5))],
        "bombs": [((2, 2), 3)],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


def test_fraction_handles_empty_denominator():
    assert _fraction(0, 0) is None
    assert _fraction(3, 4) == 0.75


def test_postbomb_action_details_are_bounded_and_consistent():
    details = robust_postbomb_action_details(_state())
    assert 0 <= details["robust_action_count"] <= 5
    assert details["robust_action_count"] == len(details["robust_actions"])
    assert len(details["robust_action_mask"]) == 6
    assert details["robust_action_mask"][-1] is False


def test_immediate_collision_details_are_bounded_and_consistent():
    details = immediate_collision_action_details(_state())
    assert 0 <= details["robust_action_count"] <= 5
    assert details["robust_action_count"] == len(details["robust_actions"])
    assert len(details["robust_action_mask"]) == 6
    assert details["robust_action_mask"][-1] is False
