"""Uniform replay tests; PER is not enabled in the current Model A."""

import numpy as np

from agent_code.model_a_dqn.replay_buffer import ReplayBuffer, Transition


def test_buffer_accepts_and_replaces_transition():
    buffer = ReplayBuffer(capacity=1)
    first = Transition(None, 0, 0.0, None, True, None)
    second = Transition(None, 1, 1.0, None, True, None)
    buffer.add(first)
    buffer.add(second)
    assert len(buffer) == 1
    assert buffer.sample(1, np.random.default_rng(1))[0] is second
