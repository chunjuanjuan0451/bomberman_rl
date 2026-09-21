"""Alternate supplied peaceful and coin-collector behavior by round."""

from __future__ import annotations

import logging
import os
import random
import re
from types import SimpleNamespace

import numpy as np

from agent_code.seeded_coin_collector_agent.callbacks import act as coin_collector_act


PEACEFUL_ACTIONS = np.asarray(["RIGHT", "LEFT", "UP", "DOWN"])


def _agent_index(name: str) -> int:
    match = re.search(r"_(\d+)(?:_code)?$", name)
    return int(match.group(1)) if match else 0


def setup(self):
    seed_text = os.environ.get("TASK3_CURRICULUM_SEED")
    if seed_text is None:
        raise ValueError("task3_curriculum_agent requires TASK3_CURRICULUM_SEED")
    callback_name = getattr(self, "name", self.logger.name)
    self.curriculum_seed = int(seed_text)
    self.curriculum_index = _agent_index(callback_name)
    self.curriculum_round = None
    self.curriculum_mode = None
    self.curriculum_policy = None


def _start_round(self, round_number: int) -> None:
    self.curriculum_round = int(round_number)
    self.curriculum_mode = "peaceful" if round_number % 2 == 1 else "coin_collector"
    sequence = np.random.SeedSequence([
        self.curriculum_seed, self.curriculum_index, round_number, 0xC033,
    ])
    words = sequence.generate_state(4)
    if self.curriculum_mode == "peaceful":
        rng = np.random.default_rng(sequence)
    else:
        rng = random.Random(
            (int(words[0]) << 96) | (int(words[1]) << 64)
            | (int(words[2]) << 32) | int(words[3])
        )
    self.curriculum_policy = SimpleNamespace(
        rng=rng,
        logger=getattr(self, "logger", logging.getLogger("task3_curriculum_agent")),
    )


def act(self, game_state: dict) -> str:
    round_number = int(game_state["round"])
    if round_number != self.curriculum_round:
        _start_round(self, round_number)
    if self.curriculum_mode == "peaceful":
        return str(self.curriculum_policy.rng.choice(PEACEFUL_ACTIONS))
    return coin_collector_act(self.curriculum_policy, game_state)
