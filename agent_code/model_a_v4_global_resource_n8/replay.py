"""Uniform replay records carrying explicit n-step bootstrap discounts."""

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
    bootstrap_discount: float
    return_steps: int


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._items: list[Transition] = []
        self._next_index = 0

    def add(self, transition: Transition) -> None:
        if len(self._items) < self.capacity:
            self._items.append(transition)
            return
        self._items[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        indices = rng.choice(len(self._items), size=int(batch_size), replace=False)
        return [self._items[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self._items)
