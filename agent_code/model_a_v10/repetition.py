"""Shared history-aware root-prior repetition context for v10 Phase 3.2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import exp

from .interfaces import ACTIONS
from .simulator import MOVES, SimState


@dataclass(slots=True)
class RepetitionContext:
    enabled: bool = True
    history_window: int = 16
    activation_steps: int = 8
    lambda_: float = 0.6
    minimum_multiplier: float = 0.05
    round_id: int | None = None
    last_score: int | None = None
    steps_since_score: int = 0
    positions: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=16))

    def __post_init__(self) -> None:
        self.positions = deque(self.positions, maxlen=int(self.history_window))

    def update(self, round_id: int, score: int, position: tuple[int, int]) -> None:
        """Observe one official state; round/score boundaries clear history."""
        position = (int(position[0]), int(position[1]))
        if self.round_id != int(round_id) or self.last_score is None or int(score) > self.last_score:
            self.round_id = int(round_id)
            self.last_score = int(score)
            self.steps_since_score = 0
            self.positions.clear()
        else:
            self.steps_since_score += 1
        self.positions.append(position)
        self.last_score = int(score)

    def reset(self) -> None:
        self.round_id = None
        self.last_score = None
        self.steps_since_score = 0
        self.positions.clear()

    def candidate_position(self, state: SimState, player: int, action: str) -> tuple[int, int]:
        agent = state.agents[player]
        if action in MOVES:
            dx, dy, _ = MOVES[action]
            candidate = (agent.x + dx, agent.y + dy)
            if 0 <= candidate[0] < state.field.shape[0] and 0 <= candidate[1] < state.field.shape[1]:
                return candidate
        return (agent.x, agent.y)

    def multipliers(self, state: SimState, player: int, legal: tuple[str, ...]) -> dict[str, float]:
        values = {action: 1.0 for action in legal}
        if not self.enabled or self.steps_since_score < int(self.activation_steps):
            return values
        for action in legal:
            position = self.candidate_position(state, player, action)
            count = sum(position == previous for previous in self.positions)
            values[action] = max(float(self.minimum_multiplier), exp(-float(self.lambda_) * count))
        return values

    def adjust_root_prior(self, state: SimState, player: int, prior: dict[str, float]) -> dict[str, float]:
        if not prior:
            return {}
        multipliers = self.multipliers(state, player, tuple(prior))
        adjusted = {action: float(value) * multipliers[action] for action, value in prior.items()}
        total = sum(adjusted.values())
        if total <= 0 or not all(value >= 0 for value in adjusted.values()):
            return dict(prior)
        return {action: value / total for action, value in adjusted.items()}
