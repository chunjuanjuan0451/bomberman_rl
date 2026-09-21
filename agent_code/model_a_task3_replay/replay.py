"""One circular replay buffer with optional kill-causal stratified sampling."""

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
    episode_id: int
    start_step: int


class ReplayBuffer:
    """Circular storage; tags alter probability, never retention or targets."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._items: list[Transition] = []
        self._next_index = 0
        self._tagged: set[int] = set()
        self._kill_windows: dict[int, list[tuple[int, int]]] = {}
        self.sample_calls = 0
        self.uniform_fallback_calls = 0
        self.requested_causal_draws = 0
        self.realized_causal_draws = 0

    def _is_tagged(self, transition: Transition) -> bool:
        return any(
            start <= transition.start_step <= end
            for start, end in self._kill_windows.get(transition.episode_id, ())
        )

    def add(self, transition: Transition) -> None:
        if len(self._items) < self.capacity:
            index = len(self._items)
            self._items.append(transition)
        else:
            index = self._next_index
            self._tagged.discard(index)
            self._items[index] = transition
            self._next_index = (self._next_index + 1) % self.capacity
        if self._is_tagged(transition):
            self._tagged.add(index)

    def mark_kill_window(self, episode_id: int, end_step: int, window: int) -> None:
        start = max(1, int(end_step) - int(window) + 1)
        interval = (start, int(end_step))
        windows = self._kill_windows.setdefault(int(episode_id), [])
        if interval not in windows:
            windows.append(interval)
        for index, transition in enumerate(self._items):
            if transition.episode_id == episode_id and start <= transition.start_step <= end_step:
                self._tagged.add(index)

    def sample(
        self, batch_size: int, rng: np.random.Generator, causal_fraction: float = 0.0,
    ) -> list[Transition]:
        batch_size = int(batch_size)
        self.sample_calls += 1
        # This is deliberately byte-for-byte the legacy uniform choice path.
        if causal_fraction <= 0 or not self._tagged:
            if causal_fraction > 0:
                self.uniform_fallback_calls += 1
            indices = rng.choice(len(self._items), size=batch_size, replace=False)
            return [self._items[int(index)] for index in indices]

        requested = int(round(batch_size * float(causal_fraction)))
        causal_count = min(requested, len(self._tagged))
        causal_pool = np.asarray(sorted(self._tagged), dtype=np.int64)
        causal = np.atleast_1d(rng.choice(causal_pool, size=causal_count, replace=False)).astype(np.int64)
        causal_set = {int(index) for index in causal}
        untagged = [index for index in range(len(self._items)) if index not in self._tagged]
        ordinary_count = batch_size - causal_count
        if len(untagged) >= ordinary_count:
            ordinary_pool = np.asarray(untagged, dtype=np.int64)
        else:
            # Only a mathematically necessary shortage may spill into other tagged slots.
            ordinary_pool = np.asarray(
                [index for index in range(len(self._items)) if index not in causal_set],
                dtype=np.int64,
            )
        ordinary = np.atleast_1d(
            rng.choice(ordinary_pool, size=ordinary_count, replace=False),
        ).astype(np.int64)
        indices = np.concatenate((causal, ordinary))
        rng.shuffle(indices)
        self.requested_causal_draws += requested
        self.realized_causal_draws += causal_count
        return [self._items[int(index)] for index in indices]

    @property
    def tagged_count(self) -> int:
        return len(self._tagged)

    def diagnostics(self) -> dict:
        return {
            "stored_transitions": len(self._items),
            "tagged_transitions": len(self._tagged),
            "sample_calls": self.sample_calls,
            "uniform_fallback_calls": self.uniform_fallback_calls,
            "requested_causal_draws": self.requested_causal_draws,
            "realized_causal_draws": self.realized_causal_draws,
        }

    def __len__(self) -> int:
        return len(self._items)
