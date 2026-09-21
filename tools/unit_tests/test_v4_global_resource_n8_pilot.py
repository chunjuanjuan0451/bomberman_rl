from collections import deque
from types import SimpleNamespace

import numpy as np

from agent_code.model_a_v4_global_resource_n8.config import (
    ARMS, ENDPOINT_ROUND, MILESTONES, REPLICAS, architecture_name, load_protocol,
)
from agent_code.model_a_v4_global_resource_n8.train import RawTransition, _raw as make_raw, aggregate_n_step
from tools.v4_global_resource_stage1 import registered_seeds


def _raw(reward, step, *, done=False):
    state = (
        np.zeros((4, 7, 7), dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        np.zeros((2, 33, 33), dtype=np.float32),
    )
    return RawTransition(
        state=state,
        action=0,
        reward=float(reward),
        state_shaping=0.0,
        next_state=None if done else state,
        done=done,
        next_mask=None if done else np.ones(6, dtype=bool),
        episode_id=1,
        start_step=step,
        events=(),
    )


def test_n8_return_uses_eight_rewards_and_gamma_to_the_eighth():
    queue = deque(_raw(index + 1, index + 1) for index in range(8))
    target = aggregate_n_step(queue, 8, 0.9)
    expected = sum((0.9 ** index) * (index + 1) for index in range(8))
    assert np.isclose(target.reward, expected)
    assert np.isclose(target.bootstrap_discount, 0.9 ** 8)
    assert target.return_steps == 8 and target.done is False


def test_n8_terminal_prefix_stops_without_crossing_episode():
    queue = deque([_raw(1, 1), _raw(2, 2, done=True), _raw(100, 3)])
    target = aggregate_n_step(queue, 8, 0.5)
    assert target.reward == 2.0
    assert target.return_steps == 2 and target.done is True
    assert target.next_state is None and target.next_mask is None


def test_n8_protocol_is_short_fresh_and_selection_free():
    protocol, _, digest = load_protocol(
        "experiments/configs/model-a-v4-global-resource-n8-pilot-s141000.json",
    )
    assert tuple(protocol["training"]["arms"]) == ARMS
    assert tuple(protocol["training"]["replicas"]) == REPLICAS
    assert ENDPOINT_ROUND == 200 and tuple(protocol["training"]["checkpoint_rounds"]) == MILESTONES
    assert protocol["learning_contract"]["n_step"] == 8
    assert protocol["reward_contract"]["replay_sampling"] == "uniform"
    assert protocol["training"]["intermediate_checkpoint_selection_allowed"] is False
    assert protocol["checkpoint_selection_allowed"] is False
    assert len(registered_seeds(protocol)) == 12
    assert len(digest) == 64 and architecture_name().endswith("n8-v1")


def test_game_callback_builds_arm_specific_raw_transition():
    field = -np.ones((17, 17), dtype=int)
    field[1:16, 1:16] = 0
    state = {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("me", 0, True, (1, 1)),
        "others": [],
        "bombs": [],
        "coins": [(15, 15)],
        "explosion_map": np.zeros_like(field),
        "user_input": None,
    }
    raw = make_raw(SimpleNamespace(global_arm="full33"), state, "RIGHT", state, [], False)
    assert raw is not None and raw.start_step == 1 and raw.done is False
    assert raw.state[2][1].sum() == 1
