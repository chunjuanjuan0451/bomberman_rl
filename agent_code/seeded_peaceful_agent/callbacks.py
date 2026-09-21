"""Supplied peaceful-agent policy with an explicit evaluation seed."""

from __future__ import annotations

import os
import re

import numpy as np


ACTIONS = np.asarray(["RIGHT", "LEFT", "UP", "DOWN"])


def _agent_index(name: str) -> int:
    match = re.search(r"_(\d+)(?:_code)?$", name)
    return int(match.group(1)) if match else 0


def setup(self):
    seed_text = os.environ.get("TASK3_OPPONENT_SEED")
    if seed_text is None:
        raise ValueError("seeded_peaceful_agent requires TASK3_OPPONENT_SEED")
    callback_name = getattr(self, "name", self.logger.name)
    self.rng = np.random.default_rng(
        np.random.SeedSequence([int(seed_text), _agent_index(callback_name), 0x5033])
    )


def act(self, game_state: dict) -> str:
    del game_state
    return str(self.rng.choice(ACTIONS))
