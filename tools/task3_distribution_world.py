"""Training-only world for the preregistered Task-3 state distribution."""

from __future__ import annotations

import os

import numpy as np

import settings as s
from environment import BombeRLeWorld


KILLRICH_CRATE_DENSITY = 0.15
KILLRICH_START_POSITION_PAIRS = (
    ((3, 3), (6, 3)),
    ((5, 3), (5, 6)),
    ((7, 7), (10, 7)),
    ((11, 3), (14, 3)),
    ((3, 11), (6, 11)),
    ((11, 13), (14, 13)),
    ((9, 9), (12, 9)),
    ((13, 7), (13, 10)),
    ((3, 3), (5, 5)),
    ((11, 3), (13, 5)),
    ((3, 11), (5, 13)),
    ((9, 11), (11, 13)),
)


class Task3TrainingDistributionWorld(BombeRLeWorld):
    """Keep official dynamics but replace candidate reset-state sampling."""

    def build_arena(self):
        distribution = os.environ.get("TASK3_TRAINING_DISTRIBUTION")
        if distribution == "classic-control":
            return super().build_arena()
        if distribution != "kill-rich":
            raise RuntimeError("TASK3_TRAINING_DISTRIBUTION must be classic-control or kill-rich")
        if self.args.scenario != "classic":
            raise RuntimeError("the training-only world must retain the official classic scenario flag")
        if len(self.agents) != 2:
            raise RuntimeError("kill-rich training requires exactly two agents")

        wall, free, crate = -1, 0, 1
        arena = np.zeros((s.COLS, s.ROWS), dtype=int)
        arena[self.rng.random((s.COLS, s.ROWS)) < KILLRICH_CRATE_DENSITY] = crate
        arena[:1, :] = wall
        arena[-1:, :] = wall
        arena[:, :1] = wall
        arena[:, -1:] = wall
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    arena[x, y] = wall

        pair_index = int(self.rng.integers(len(KILLRICH_START_POSITION_PAIRS)))
        start_positions = KILLRICH_START_POSITION_PAIRS[pair_index]
        for x, y in start_positions:
            for xx, yy in ((x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if arena[xx, yy] == crate:
                    arena[xx, yy] = free

        active_agents = []
        for agent, start_position in zip(self.agents, self.rng.permutation(start_positions)):
            active_agents.append(agent)
            agent.x, agent.y = (int(start_position[0]), int(start_position[1]))
        return arena, [], active_agents
