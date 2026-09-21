"""Random-agent distribution with an explicit reproducible seed."""

from __future__ import annotations

import os
import re

import numpy as np


ACTIONS = np.asarray(["RIGHT", "LEFT", "UP", "DOWN", "BOMB"])
PROBABILITIES = np.asarray([0.23, 0.23, 0.23, 0.23, 0.08])


def _agent_index(name: str) -> int:
    match = re.search(r"_(\d+)(?:_code)?$", name)
    return int(match.group(1)) if match else 0


def setup(self):
    seed_text = os.environ.get("SEEDED_RANDOM_AGENT_SEED")
    if seed_text is None:
        raise ValueError("seeded_random_agent requires SEEDED_RANDOM_AGENT_SEED")
    base_seed = int(seed_text)
    # Official AgentRunner exposes only ``train`` and ``logger`` on callback
    # self.  Repeated agents are named ``seeded_random_agent_0`` etc., and the
    # corresponding logger adds a ``_code`` suffix.
    callback_name = getattr(self, "name", self.logger.name)
    self.rng = np.random.default_rng(np.random.SeedSequence([base_seed, _agent_index(callback_name)]))


def act(self, game_state: dict) -> str:
    return str(self.rng.choice(ACTIONS, p=PROBABILITIES))
