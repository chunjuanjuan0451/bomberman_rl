"""Function-style regression tests for the compact CNN curriculum line."""

from collections import deque

import numpy as np
import events as e

from agent_code.model_a_cnn_n8.features import CENTER, state_to_features
from agent_code.model_a_cnn_n8.network import FullBoardDuelingCNN, torch
from agent_code.model_a_cnn_n8.rewards import reward_components
from agent_code.model_a_cnn_n8.train import RawTransition, aggregate_n_step
from tools.cnn_n8_task1 import select_common_milestone


def _state(position=(1, 1), *, coins=((3, 1),), bombs=(), others=(), step=1):
    field = np.zeros((17, 17), dtype=np.int8)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    return {
        "round": 1, "step": step, "field": field,
        "self": ("model", 0, True, position),
        "others": list(others), "bombs": list(bombs), "coins": list(coins),
        "explosion_map": np.zeros((17, 17), dtype=np.int8),
    }


def test_cnn_features_are_agent_centered_and_uint8():
    spatial, scalars = state_to_features(_state())
    assert spatial.shape == (11, 33, 33)
    assert spatial.dtype == np.uint8 and scalars.shape == (6,)
    assert spatial[3, CENTER, CENTER] == 255
    assert spatial[2, CENTER, CENTER + 2] == 255
    assert spatial[0, CENTER - 1, CENTER - 1] == 255


def test_cnn_network_shape_and_parameter_budget():
    assert torch is not None
    model = FullBoardDuelingCNN()
    assert sum(item.numel() for item in model.parameters()) == 367_863
    output = model(torch.zeros(2, 11, 33, 33), torch.zeros(2, 6))
    assert tuple(output.shape) == (2, 6)


def test_cnn_reward_self_kill_is_exclusive_and_shaping_is_bounded():
    old = _state(); new = _state(position=(2, 1), step=2)
    self_kill = reward_components(old, None, "BOMB", [e.GOT_KILLED, e.KILLED_SELF], "task4")
    assert self_kill[e.KILLED_SELF] == -8.0
    assert e.GOT_KILLED not in self_kill
    shaped = reward_components(old, new, "RIGHT", [], "task1")
    shaping_total = sum(abs(value) for key, value in shaped.items() if key not in (e.COIN_COLLECTED,))
    assert shaping_total <= 0.2 + 1e-12


def test_cnn_n_step_return_stops_at_terminal_and_never_crosses_episode():
    state = (np.zeros((11, 33, 33), np.uint8), np.zeros(6, np.float32))
    queue = deque([
        RawTransition(state, 0, 1.0, {}, state, False, np.ones(6, bool), 1, 1, ()),
        RawTransition(state, 1, 2.0, {}, None, True, None, 1, 2, ()),
    ])
    item = aggregate_n_step(queue, 8, 0.5)
    assert item.reward == 2.0 and item.done and item.return_steps == 2
    assert item.bootstrap_discount == 0.25


def test_cnn_selection_uses_one_common_earliest_near_best_milestone():
    rows = {}
    scores = {100: (1.8, 1.7, 1.9), 200: (2.0, 2.0, 2.0),
              300: (2.04, 2.03, 2.05), 400: (1.7, 1.6, 1.8)}
    for milestone, values in scores.items():
        for index, replica in enumerate(("r1", "r2", "r3")):
            rows[f"cnn-{replica}-round{milestone:04d}"] = {
                "rounds": 100, "score": int(values[index] * 100), "steps": 1000,
                "waits": 100 + index,
            }
    selected = select_common_milestone(rows, [100, 200, 300, 400], 0.10)
    assert selected["selected_common_milestone"] == 200
    assert selected["eligible_milestones"] == [200, 300]
