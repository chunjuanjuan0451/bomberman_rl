"""Small, deterministic uniform replay buffer for the first Model A version."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Transition:
    state: object
    action: int
    reward: float
    next_state: object
    done: bool
    next_mask: object


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self._items: list[Transition] = []
        self._next_index = 0

    def add(self, transition: Transition) -> None:
        if len(self._items) < self.capacity:
            self._items.append(transition)
            return
        self._items[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        indices = rng.choice(len(self._items), size=batch_size, replace=False)
        return [self._items[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self._items)
