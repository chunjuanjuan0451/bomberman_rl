"""Check that the standalone agent matches the evaluated D4 policy."""

from types import SimpleNamespace

import numpy as np
import torch

from agent_code.model_a_cnn_n8.features import state_to_features as original_features
from agent_code.model_a_cnn_n8_league_train.callbacks import effective_action_mask as original_mask
from agent_code.model_a_cnn_n8_symmetry_ab.callbacks import act as original_act, d4_ensemble_q_values
from agent_code.niulai import callbacks
from agent_code.niulai.features import state_to_features
from agent_code.niulai.mask import effective_action_mask


def _state(can_bomb=True, position=(2, 2), bombs=(), others=()):
    field = np.zeros((17, 17), dtype=np.int8)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[4, 2] = 1
    return {
        "round": 1,
        "step": 3,
        "field": field,
        "self": ("me", 0, can_bomb, position),
        "others": list(others),
        "bombs": list(bombs),
        "coins": [(3, 3)],
        "explosion_map": np.zeros_like(field),
    }


def test_standalone_features_and_masks_match_original():
    states = [
        _state(),
        _state(False, bombs=[((2, 2), 3)], others=[("opponent", 0, True, (3, 2))]),
        _state(True, position=(3, 3), bombs=[((3, 5), 1)]),
    ]
    for game_state in states:
        spatial, scalars = state_to_features(game_state)
        old_spatial, old_scalars = original_features(game_state)
        assert np.array_equal(spatial, old_spatial)
        assert np.array_equal(scalars, old_scalars)
        for active_bomb in (False, True):
            assert np.array_equal(
                effective_action_mask(game_state, active_bomb),
                original_mask(game_state, active_bomb),
            )


def test_masks_match_on_varied_boards():
    rng = np.random.default_rng(4819)
    for _ in range(50):
        state = _state(can_bomb=bool(rng.integers(2)), position=(8, 8))
        field = state["field"]
        field[4, 2] = 0
        for x, y in rng.integers(1, 16, size=(18, 2)):
            field[x, y] = 1
        field[8, 8] = 0
        state["others"] = [("opponent", 0, True, (int(rng.integers(1, 16)), 6))]
        state["bombs"] = [((int(rng.integers(1, 16)), int(rng.integers(1, 16))), int(rng.integers(0, 4)))]
        for active_bomb in (False, True):
            assert np.array_equal(
                effective_action_mask(state, active_bomb),
                original_mask(state, active_bomb),
            )


def test_standalone_q_values_match_original():
    holder = SimpleNamespace(train=False)
    callbacks.setup(holder)
    checkpoint = callbacks._load_model(callbacks.MODEL_PATH)
    assert set(checkpoint) == {
        "architecture", "online_net", "source_checkpoint_sha256", "stage", "replica", "stage_rounds"
    }
    assert checkpoint["source_checkpoint_sha256"] == (
        "3fe04112793ea8fd8fde7f72e7da40f4bb5e54f7ebe0a45f5f1b92f378ac3dee"
    )
    state = _state(False, bombs=[((2, 2), 3)], others=[("opponent", 0, True, (3, 2))])
    with torch.no_grad():
        original, _ = d4_ensemble_q_values(holder.online_net, state, holder.device)
        standalone = callbacks.d4_q_values(holder.online_net, state, holder.device)
    np.testing.assert_array_equal(standalone, original)
    assert callbacks.act(holder, state) in ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def test_standalone_actions_match_original():
    holder = SimpleNamespace(train=False)
    callbacks.setup(holder)
    reference = SimpleNamespace(
        online_net=holder.online_net,
        device=holder.device,
        rng=np.random.default_rng(20260917),
        audit_arm="treatment",
        _active_bomb=False,
        _active_bomb_position=None,
    )
    states = [
        _state(),
        _state(False, bombs=[((2, 2), 3)], others=[("opponent", 0, True, (3, 2))]),
        _state(True, position=(3, 3), bombs=[((3, 5), 1)]),
    ]
    for state in states:
        assert callbacks.act(holder, state) == original_act(reference, state)
