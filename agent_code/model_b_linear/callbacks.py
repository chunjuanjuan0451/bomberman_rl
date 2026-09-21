"""Official Bomberman callback entry points for Model B."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .features import ACTIONS, FEATURE_SIZE, legal_action_mask, state_to_features

WEIGHTS_PATH = Path(os.environ.get("MODEL_B_WEIGHTS_PATH", Path(__file__).with_name("weights.npy")))


def _new_weights() -> np.ndarray:
    return np.zeros((FEATURE_SIZE, len(ACTIONS)), dtype=np.float32)


def _load_weights() -> np.ndarray:
    weights = np.load(WEIGHTS_PATH, allow_pickle=False)
    expected_shape = (FEATURE_SIZE, len(ACTIONS))
    if weights.shape != expected_shape:
        raise ValueError(f"weights.npy has shape {weights.shape}, expected {expected_shape}")
    return weights.astype(np.float32, copy=False)


def setup(self):
    """Create fresh training weights or load existing weights for evaluation."""
    seed_text = os.environ.get("MODEL_B_SEED")
    self.agent_seed = None if seed_text is None else int(seed_text)
    self.rng = np.random.default_rng(self.agent_seed)
    resume = os.environ.get("MODEL_B_RESUME", "") == "1"
    if self.train and not resume:
        self.logger.info("Starting a fresh Model B linear Q-learning run.")
        self.weights = _new_weights()
    elif WEIGHTS_PATH.exists():
        self.logger.info("Loading Model B weights from weights.npy.")
        self.weights = _load_weights()
    else:
        raise FileNotFoundError("weights.npy is required for evaluation. Train first, or set MODEL_B_RESUME=1 to resume.")


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)
    legal_indices = np.flatnonzero(legal_action_mask(game_state))
    if legal_indices.size == 0:
        return "WAIT"
    if self.train and self.rng.random() < self.epsilon:
        return ACTIONS[int(self.rng.choice(legal_indices))]
    q_values = features @ self.weights
    legal_values = q_values[legal_indices]
    best_indices = legal_indices[np.isclose(legal_values, legal_values.max())]
    return ACTIONS[int(self.rng.choice(best_indices))]
