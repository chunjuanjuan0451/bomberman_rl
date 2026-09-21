"""Replay buffer with auditable reserved kill/self-chain sampling slots."""

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
    episode_id: int
    start_step: int
    kill_chain: bool
    self_chain: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = int(capacity)
        self._items: list[Transition] = []
        self._next_index = 0
        self._kill_indices: set[int] = set()
        self._self_indices: set[int] = set()

    def add(self, transition: Transition) -> None:
        if len(self._items) < self.capacity:
            index = len(self._items)
            self._items.append(transition)
        else:
            index = self._next_index
            self._kill_indices.discard(index)
            self._self_indices.discard(index)
            self._items[index] = transition
            self._next_index = (self._next_index + 1) % self.capacity
        if transition.kill_chain:
            self._kill_indices.add(index)
        if transition.self_chain:
            self._self_indices.add(index)

    @staticmethod
    def _draw(pool, count: int, rng: np.random.Generator) -> list[int]:
        values = np.asarray(sorted(pool), dtype=np.int64)
        if count <= 0 or values.size == 0:
            return []
        amount = min(int(count), int(values.size))
        return [int(value) for value in rng.choice(values, size=amount, replace=False)]

    def sample_uniform(self, batch_size: int, rng: np.random.Generator) -> tuple[list[Transition], dict]:
        indices = self._draw(range(len(self._items)), int(batch_size), rng)
        items = [self._items[index] for index in indices]
        return items, {
            "requested_kill_slots": 0, "fulfilled_kill_slots": 0,
            "requested_self_slots": 0, "fulfilled_self_slots": 0,
            "fallback_uniform_slots": 0,
            "realized_kill_chain_samples": sum(item.kill_chain for item in items),
            "realized_self_chain_samples": sum(item.self_chain for item in items),
        }

    def sample_stratified(
        self,
        batch_size: int,
        kill_slots: int,
        self_slots: int,
        rng: np.random.Generator,
    ) -> tuple[list[Transition], dict]:
        batch_size = int(batch_size)
        kill_slots = int(kill_slots)
        self_slots = int(self_slots)
        if kill_slots < 0 or self_slots < 0 or kill_slots + self_slots > batch_size:
            raise ValueError("invalid reserved replay slots")
        selected: list[int] = []
        kill = self._draw(self._kill_indices, kill_slots, rng)
        selected.extend(kill)
        selected_set = set(selected)
        self_pool = self._self_indices - selected_set
        self_selected = self._draw(self_pool, self_slots, rng)
        selected.extend(self_selected)
        selected_set.update(self_selected)
        remaining = batch_size - len(selected)
        uniform_pool = set(range(len(self._items))) - selected_set
        uniform = self._draw(uniform_pool, remaining, rng)
        selected.extend(uniform)
        if len(selected) != batch_size or len(set(selected)) != batch_size:
            raise RuntimeError("stratified replay failed to produce a unique full batch")
        items = [self._items[index] for index in selected]
        return items, {
            "requested_kill_slots": kill_slots,
            "fulfilled_kill_slots": len(kill),
            "requested_self_slots": self_slots,
            "fulfilled_self_slots": len(self_selected),
            "fallback_uniform_slots": (kill_slots - len(kill)) + (self_slots - len(self_selected)),
            "realized_kill_chain_samples": sum(item.kill_chain for item in items),
            "realized_self_chain_samples": sum(item.self_chain for item in items),
        }

    def __len__(self) -> int:
        return len(self._items)

    @property
    def kill_items(self) -> int:
        return len(self._kill_indices)

    @property
    def self_items(self) -> int:
        return len(self._self_indices)
