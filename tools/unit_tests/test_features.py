"""Basic feature extraction contract tests."""

import numpy as np

from agent_code.model_a_dqn.features import GLOBAL_SIZE, LOCAL_SHAPE, state_to_features


def test_empty_board_feature_layout():
    field = -np.ones((7, 7), dtype=int)
    field[1:6, 1:6] = 0
    state = {
        "round": 1, "step": 1, "field": field,
        "self": ("me", 0, True, (3, 3)), "others": [], "bombs": [],
        "coins": [], "explosion_map": np.zeros_like(field), "user_input": None,
    }
    local, global_features = state_to_features(state)
    assert local.shape == LOCAL_SHAPE
    assert global_features.shape == (GLOBAL_SIZE,)
    assert local.dtype == np.float32
    assert global_features.dtype == np.float32
