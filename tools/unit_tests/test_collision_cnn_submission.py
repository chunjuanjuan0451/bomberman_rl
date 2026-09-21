"""Submission-isolation and parity checks for collision_cnn_agent."""

from types import SimpleNamespace

import numpy as np
import torch

from agent_code.collision_cnn_agent import callbacks as submission_callbacks
from agent_code.collision_cnn_agent.features import (
    collision_filtered_mask as submission_collision_mask,
    legal_action_mask as submission_legal_mask,
    state_to_features as submission_features,
)
from agent_code.model_a_cnn_n8.features import state_to_features as source_features
from agent_code.model_a_cnn_n8_postbomb_movement_audit.callbacks import immediate_collision_action_details
from agent_code.model_a_dqn.features import legal_action_mask as source_legal_mask


def _state(*, can_bomb=True, position=(2, 2), others=(), bombs=()):
    field = np.zeros((17, 17), dtype=np.int8)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[4, 2] = 1
    return {
        "round": 1, "step": 3, "field": field,
        "self": ("me", 0, can_bomb, position),
        "others": list(others), "bombs": list(bombs), "coins": [(3, 3)],
        "explosion_map": np.zeros_like(field),
    }


def test_submission_features_and_base_mask_match_training_implementation():
    states = [
        _state(),
        _state(can_bomb=False, bombs=[((2, 2), 3)], others=[("o", 0, True, (3, 2))]),
        _state(position=(3, 3), bombs=[((3, 5), 1)], others=[("o", 0, True, (5, 3))]),
    ]
    for state in states:
        actual_spatial, actual_scalars = submission_features(state)
        expected_spatial, expected_scalars = source_features(state)
        assert np.array_equal(actual_spatial, expected_spatial)
        assert np.array_equal(actual_scalars, expected_scalars)
        assert np.array_equal(submission_legal_mask(state), source_legal_mask(state))


def test_submission_collision_filter_matches_s159_rule():
    state = _state(can_bomb=False, bombs=[((2, 2), 3)], others=[("o", 0, True, (3, 2))])
    base = source_legal_mask(state)
    robust = np.asarray(immediate_collision_action_details(state)["robust_action_mask"], dtype=bool)
    expected = base & robust
    if not expected.any():
        expected = base
    assert np.array_equal(submission_collision_mask(state, base), expected)


def test_submission_checkpoint_is_inference_only_and_callback_runs():
    payload = submission_callbacks._load_model(submission_callbacks.MODEL_PATH)
    assert set(payload) == {
        "architecture", "online_net", "source_checkpoint_sha256", "stage", "replica", "stage_rounds"
    }
    assert "optimizer" not in payload and "target_net" not in payload
    holder = SimpleNamespace(train=False)
    submission_callbacks.setup(holder)
    action = submission_callbacks.act(holder, _state())
    assert action in ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
    assert all(not parameter.requires_grad for parameter in holder.online_net.parameters())
    with torch.no_grad():
        assert sum(parameter.numel() for parameter in holder.online_net.parameters()) == 367_863
