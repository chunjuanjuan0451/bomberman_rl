"""Memory-conscious replay with auditable kill/self-kill reserved slots."""

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
    episode_id: int
    start_step: int
    kill_chain: bool
    self_chain: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._items: list[Transition] = []
        self._next = 0
        self._kill: set[int] = set()
        self._self: set[int] = set()

    def add(self, transition: Transition) -> None:
        if len(self._items) < self.capacity:
            index = len(self._items)
            self._items.append(transition)
        else:
            index = self._next
            self._kill.discard(index)
            self._self.discard(index)
            self._items[index] = transition
            self._next = (self._next + 1) % self.capacity
        if transition.kill_chain:
            self._kill.add(index)
        if transition.self_chain:
            self._self.add(index)

    @staticmethod
    def _draw(pool, amount: int, rng: np.random.Generator) -> list[int]:
        values = np.asarray(sorted(pool), dtype=np.int64)
        if amount <= 0 or not values.size:
            return []
        return [int(value) for value in rng.choice(values, size=min(amount, len(values)), replace=False)]

    def sample(self, size: int, rng: np.random.Generator, kill_slots: int = 0,
               self_slots: int = 0) -> tuple[list[Transition], dict[str, int]]:
        size, kill_slots, self_slots = int(size), int(kill_slots), int(self_slots)
        if min(size, kill_slots, self_slots) < 0 or kill_slots + self_slots > size:
            raise ValueError("invalid reserved replay slots")
        selected = self._draw(self._kill, kill_slots, rng)
        selected_set = set(selected)
        self_selected = self._draw(self._self - selected_set, self_slots, rng)
        selected.extend(self_selected)
        selected_set.update(self_selected)
        selected.extend(self._draw(set(range(len(self._items))) - selected_set,
                                   size - len(selected), rng))
        if len(selected) != size or len(set(selected)) != size:
            raise RuntimeError("replay failed to produce a unique full batch")
        items = [self._items[index] for index in selected]
        return items, {
            "requested_kill_slots": kill_slots,
            "fulfilled_kill_slots": min(kill_slots, len(self._kill)),
            "requested_self_slots": self_slots,
            "fulfilled_self_slots": len(self_selected),
            "fallback_uniform_slots": (
                kill_slots - min(kill_slots, len(self._kill))
                + self_slots - len(self_selected)
            ),
            "realized_kill_chain_samples": sum(item.kill_chain for item in items),
            "realized_self_chain_samples": sum(item.self_chain for item in items),
        }

    def __len__(self) -> int:
        return len(self._items)

    @property
    def kill_items(self) -> int:
        return len(self._kill)

    @property
    def self_items(self) -> int:
        return len(self._self)
