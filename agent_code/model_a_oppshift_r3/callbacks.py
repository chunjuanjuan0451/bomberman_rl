"""Frozen candidate-r3 callback identity for s139000."""

from agent_code.model_a_opponent_shift.common import act_frozen, setup_frozen


def setup(self):
    setup_frozen(self, "candidate-r3")


def act(self, game_state: dict) -> str:
    return act_frozen(self, game_state)
