"""Linear Q-learning updates for Model B."""

from __future__ import annotations

import numpy as np

import events as e
from .callbacks import ACTIONS, WEIGHTS_PATH
from .features import legal_action_mask, state_to_features
from .potential import GAMMA, shaping_reward

ALPHA = 0.02
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
REWARD_BY_EVENT = {
    e.COIN_COLLECTED: 5.0, e.CRATE_DESTROYED: 1.0, e.KILLED_OPPONENT: 10.0,
    e.KILLED_SELF: -15.0, e.GOT_KILLED: -15.0, e.INVALID_ACTION: -2.0,
    # In coin-heaven, bombs create risk but do not help collect coins. Keep the
    # penalty small enough that it is not a hand-coded policy, while still
    # discouraging needless early exploration of BOMB.
    e.BOMB_DROPPED: -0.2,
}


def setup_training(self):
    self.alpha, self.gamma, self.epsilon = ALPHA, GAMMA, EPSILON_START
    self.rng = np.random.default_rng(self.agent_seed)
    self.training_steps = 0
    self.logger.info("Model B: linear Q-learning with bounded potential shaping.")


def reward_from_events(events: list[str]) -> float:
    return sum(REWARD_BY_EVENT.get(event, 0.0) for event in events)


def _update(self, old_game_state, action: str, new_game_state, reward: float) -> None:
    features = state_to_features(old_game_state)
    if features is None or action not in ACTIONS:
        return
    action_index = ACTIONS.index(action)
    prediction = float(features @ self.weights[:, action_index])
    if new_game_state is None:
        target = reward
    else:
        next_features = state_to_features(new_game_state)
        legal_next = np.flatnonzero(legal_action_mask(new_game_state))
        bootstrap = 0.0 if legal_next.size == 0 else float(np.max((next_features @ self.weights)[legal_next]))
        target = reward + self.gamma * bootstrap
    self.weights[:, action_index] += self.alpha * (target - prediction) * features
    self.training_steps += 1
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    base_reward = reward_from_events(events)
    shaped_reward = shaping_reward(old_game_state, new_game_state)
    _update(self, old_game_state, self_action, new_game_state, base_reward + shaped_reward)
    self.logger.debug("Model B reward: base=%.3f shaping=%.3f epsilon=%.4f", base_reward, shaped_reward, self.epsilon)


def end_of_round(self, last_game_state, last_action, events):
    _update(self, last_game_state, last_action, None, reward_from_events(events) + shaping_reward(last_game_state, None))
    np.save(WEIGHTS_PATH, self.weights.astype(np.float32, copy=False), allow_pickle=False)
    self.logger.info("Saved Model B weights after %d updates.", self.training_steps)
