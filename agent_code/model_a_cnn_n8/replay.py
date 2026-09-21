"""Memory-conscious uniform replay with explicit n-step discounts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: tuple[np.ndarray, np.ndarray]
    action: int
    reward: float
    next_state: tuple[np.ndarray, np.ndarray] | None
    done: bool
    next_mask: np.ndarray | None
    bootstrap_discount: float
    return_steps: int


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._items: list[Transition] = []
        self._next = 0

    def add(self, transition: Transition) -> None:
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._next] = transition
            self._next = (self._next + 1) % self.capacity

    def sample(self, size: int, rng: np.random.Generator) -> list[Transition]:
        indices = rng.choice(len(self._items), size=int(size), replace=False)
        return [self._items[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self._items)
