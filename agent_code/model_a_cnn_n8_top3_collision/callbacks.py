"""Collision-filtered top-three CNN callbacks."""

from agent_code.model_a_cnn_n8_top3_common.callbacks import act_agent, setup_agent


def setup(self):
    setup_agent(self, "collision-cnn", True)


def act(self, game_state):
    return act_agent(self, game_state)
