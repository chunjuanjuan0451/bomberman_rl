"""Small proportional prioritized replay for v9."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Transition:
    state: object
    action: int
    reward: float
    discount: float
    next_state: object
    done: bool
    next_mask: object
    risk: object
    risk_mask: object


class PrioritizedReplay:
    def __init__(self, capacity, alpha):
        self.capacity, self.alpha = int(capacity), float(alpha)
        self.items, self.priorities, self.next_index = [], np.zeros(int(capacity), np.float32), 0

    def add(self, item):
        priority = float(self.priorities[:len(self.items)].max()) if self.items else 1.0
        if len(self.items) < self.capacity:
            self.items.append(item)
            self.priorities[len(self.items) - 1] = priority
        else:
            self.items[self.next_index] = item
            self.priorities[self.next_index] = priority
            self.next_index = (self.next_index + 1) % self.capacity

    def sample(self, size, beta, rng):
        priorities = np.maximum(self.priorities[:len(self.items)], 1e-6) ** self.alpha
        probabilities = priorities / priorities.sum()
        indices = rng.choice(len(self.items), int(size), replace=False, p=probabilities)
        weights = (len(self.items) * probabilities[indices]) ** (-float(beta))
        weights /= weights.max()
        return [self.items[int(i)] for i in indices], indices, weights.astype(np.float32)

    def update(self, indices, priorities):
        self.priorities[indices] = np.asarray(priorities, dtype=np.float32) + 1e-5

    def __len__(self):
        return len(self.items)
