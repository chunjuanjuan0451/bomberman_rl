"""Double-DQN training callbacks for Model A."""

from __future__ import annotations

import copy

import numpy as np

import events as e
from .callbacks import ACTIONS, CHECKPOINT_PATH, MODEL_ARCHITECTURE, _load_network_state
from .features import GLOBAL_SIZE, danger_time_map, state_to_features
from .network import torch
from .replay_buffer import ReplayBuffer, Transition

GAMMA = 0.99
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
WARMUP_TRANSITIONS = 2_000
UPDATE_EVERY = 4
TARGET_UPDATE_EVERY = 1_000
EPSILON_START = 1.0
EPSILON_FINAL = 0.05
EPSILON_DECAY_STEPS = 80_000
CHECKPOINT_EVERY_ROUNDS = 50
# Compatibility value for checkpoints saved before training state was persisted.
# It preserves learned behaviour while leaving some room for opponent adaptation.
LEGACY_RESUME_EPSILON = 0.10

REWARD_BY_EVENT = {
    e.COIN_COLLECTED: 8.0,
    e.CRATE_DESTROYED: 1.5,
    e.KILLED_OPPONENT: 12.0,
    e.KILLED_SELF: -20.0,
    e.GOT_KILLED: -20.0,
    e.INVALID_ACTION: -2.0,
    e.BOMB_DROPPED: -0.05,
    e.SURVIVED_ROUND: 3.0,
}


def reward_from_events(events: list[str]) -> float:
    return sum(REWARD_BY_EVENT.get(event, 0.0) for event in events)


def _state_shaping(old_game_state, new_game_state) -> float:
    """Small danger-only shaping; rewards remain dominated by game events."""
    if old_game_state is None or new_game_state is None:
        return 0.0
    old_position = tuple(old_game_state["self"][3])
    new_position = tuple(new_game_state["self"][3])
    old_danger = danger_time_map(old_game_state)[old_position]
    new_danger = danger_time_map(new_game_state)[new_position]
    if old_danger <= 2 and new_danger > 2:
        return 0.4
    if old_danger > 2 and new_danger <= 1:
        return -0.8
    return 0.0


def setup_training(self):
    self.target_net = copy.deepcopy(self.online_net).to(self.device)
    self.target_net.eval()
    self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
    self.replay_buffer = ReplayBuffer()
    self.training_rng = np.random.default_rng(self.agent_seed)
    checkpoint = self.resume_checkpoint
    if checkpoint is None:
        self.training_steps = 0
        self.gradient_steps = 0
        self.epsilon = EPSILON_START
        self.logger.info("Model A training: fresh Double DQN run with uniform replay and safe BOMB mask.")
        return
    if "target_net" in checkpoint:
        _load_network_state(self.target_net, checkpoint["target_net"])
    if "optimizer" in checkpoint and not self.feature_schema_migrated:
        self.optimizer.load_state_dict(checkpoint["optimizer"])
    elif self.feature_schema_migrated:
        self.logger.info("Reset optimizer state after feature-schema migration.")
    if "training_steps" in checkpoint:
        self.training_steps = int(checkpoint["training_steps"])
    else:
        # Old checkpoints only have network weights.  Place them at the point
        # on the linear schedule corresponding to LEGACY_RESUME_EPSILON, so
        # the first callback cannot jump back to epsilon ~= 1.0.
        self.training_steps = round(
            EPSILON_DECAY_STEPS * (EPSILON_START - LEGACY_RESUME_EPSILON)
            / (EPSILON_START - EPSILON_FINAL)
        )
    self.gradient_steps = int(checkpoint.get("gradient_steps", 0))
    self.epsilon = float(checkpoint.get("epsilon", LEGACY_RESUME_EPSILON))
    self.logger.info(
        "Resuming Model A: env_steps=%d updates=%d epsilon=%.3f.",
        self.training_steps, self.gradient_steps, self.epsilon,
    )


def _batch_tensors(transitions: list[Transition], device):
    local = np.stack([item.state[0] for item in transitions])
    global_features = np.stack([item.state[1] for item in transitions])
    actions = np.asarray([item.action for item in transitions], dtype=np.int64)
    rewards = np.asarray([item.reward for item in transitions], dtype=np.float32)
    dones = np.asarray([item.done for item in transitions], dtype=np.float32)
    next_local = np.stack([item.next_state[0] if item.next_state is not None else np.zeros_like(item.state[0]) for item in transitions])
    next_global = np.stack([item.next_state[1] if item.next_state is not None else np.zeros_like(item.state[1]) for item in transitions])
    next_masks = np.stack([item.next_mask if item.next_state is not None else np.zeros(len(ACTIONS), dtype=bool) for item in transitions])
    return tuple(torch.as_tensor(value, device=device) for value in (
        local, global_features, actions, rewards, dones, next_local, next_global, next_masks,
    ))


def _learn(self):
    if len(self.replay_buffer) < WARMUP_TRANSITIONS or self.training_steps % UPDATE_EVERY:
        return
    transitions = self.replay_buffer.sample(BATCH_SIZE, self.training_rng)
    local, global_features, actions, rewards, dones, next_local, next_global, next_masks = _batch_tensors(transitions, self.device)
    q_values = self.online_net(local.float(), global_features.float()).gather(1, actions.long().unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        online_next = self.online_net(next_local.float(), next_global.float())
        online_next = online_next.masked_fill(~next_masks.bool(), -1e9)
        next_actions = online_next.argmax(dim=1)
        target_next = self.target_net(next_local.float(), next_global.float()).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards.float() + GAMMA * (1.0 - dones.float()) * target_next
    loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
    self.optimizer.step()
    self.gradient_steps += 1
    if self.gradient_steps % TARGET_UPDATE_EVERY == 0:
        self.target_net.load_state_dict(self.online_net.state_dict())
    if self.gradient_steps % 100 == 0:
        self.logger.info("Model A update=%d replay=%d loss=%.4f epsilon=%.3f", self.gradient_steps, len(self.replay_buffer), loss.item(), self.epsilon)


def _record(self, old_game_state, action: str, new_game_state, reward: float, done: bool):
    state = state_to_features(old_game_state)
    if state is None or action not in ACTIONS:
        return
    next_state = None if done else state_to_features(new_game_state)
    next_mask = None
    if next_state is not None:
        from .features import legal_action_mask
        next_mask = legal_action_mask(new_game_state)
    self.replay_buffer.add(Transition(state, ACTIONS.index(action), reward, next_state, done, next_mask))
    self.training_steps += 1
    progress = min(1.0, self.training_steps / EPSILON_DECAY_STEPS)
    self.epsilon = EPSILON_START + progress * (EPSILON_FINAL - EPSILON_START)
    _learn(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = reward_from_events(events) + _state_shaping(old_game_state, new_game_state)
    _record(self, old_game_state, self_action, new_game_state, reward, done=False)


def _save_checkpoint(self):
    torch.save(
        {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "completed_rounds": self.completed_rounds,
            "training_steps": self.training_steps,
            "gradient_steps": self.gradient_steps,
            "epsilon": self.epsilon,
            "global_feature_size": GLOBAL_SIZE,
            "architecture": MODEL_ARCHITECTURE,
            "agent_seed": self.agent_seed,
        },
        CHECKPOINT_PATH,
    )


def end_of_round(self, last_game_state, last_action, events):
    _record(self, last_game_state, last_action, None, reward_from_events(events), done=True)
    self.completed_rounds += 1
    if self.completed_rounds % CHECKPOINT_EVERY_ROUNDS == 0:
        _save_checkpoint(self)
        self.logger.info("Saved Model A checkpoint after round %d.", self.completed_rounds)
