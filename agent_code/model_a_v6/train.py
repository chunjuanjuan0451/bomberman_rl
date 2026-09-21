"""Fresh-run Double-DQN training callbacks for Model A v6 variants."""

from __future__ import annotations

import copy

import numpy as np

import events as e
from .callbacks import ACTIONS, CHECKPOINT_PATH
from .features import ACTION_FEATURE_SIZE, GLOBAL_SIZE, action_tactical_features, danger_time_map, state_to_features
from .network import torch
from .replay_buffer import ReplayBuffer, Transition
from .symmetry import augment_transition_arrays_random

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


def reward_from_events(events: list[str], reward_map: dict[str, float] | None = None) -> float:
    rewards = REWARD_BY_EVENT if reward_map is None else reward_map
    return sum(rewards.get(event, 0.0) for event in events)


def _state_shaping(
    old_game_state, new_game_state,
    danger_exit_reward: float = 0.4,
    danger_entry_penalty: float = -0.8,
) -> float:
    """Small danger-only shaping; rewards remain dominated by game events."""
    if old_game_state is None or new_game_state is None:
        return 0.0
    old_position = tuple(old_game_state["self"][3])
    new_position = tuple(new_game_state["self"][3])
    old_danger = danger_time_map(old_game_state)[old_position]
    new_danger = danger_time_map(new_game_state)[new_position]
    if old_danger <= 2 and new_danger > 2:
        return danger_exit_reward
    if old_danger > 2 and new_danger <= 1:
        return danger_entry_penalty
    return 0.0


def setup_training(self):
    if not self.v6_config.get("enabled_for_training", False):
        raise RuntimeError(f"Training is disabled by config for {self.v6_config['variant']}")
    hyperparameters = self.v6_config["hyperparameters"]
    self.reward_by_event = {
        getattr(e, name): float(value)
        for name, value in self.v6_config["rewards"].items()
    }
    self.shaping_config = self.v6_config["shaping"]
    self.gamma = float(hyperparameters["gamma"])
    self.batch_size = int(hyperparameters["batch_size"])
    self.warmup_transitions = int(hyperparameters["warmup_transitions"])
    self.update_every = int(hyperparameters["update_every"])
    self.target_update_every = int(hyperparameters["target_update_every"])
    self.epsilon_start = float(hyperparameters["epsilon_start"])
    self.epsilon_final = float(hyperparameters["epsilon_final"])
    self.epsilon_decay_steps = int(hyperparameters["epsilon_decay_steps"])
    self.checkpoint_every_rounds = int(hyperparameters["checkpoint_every_rounds"])
    self.target_net = copy.deepcopy(self.online_net).to(self.device)
    self.target_net.eval()
    self.optimizer = torch.optim.Adam(
        self.online_net.parameters(), lr=float(hyperparameters["learning_rate"]),
    )
    self.replay_buffer = ReplayBuffer(capacity=int(hyperparameters["replay_capacity"]))
    # Keep replay sampling identical between matched control and D4 runs.
    # Augmentation has a separate stream so drawing a symmetry cannot perturb
    # later replay indices.
    self.replay_rng = np.random.default_rng(self.agent_seed)
    self.augmentation_rng = np.random.default_rng(
        np.random.SeedSequence([0 if self.agent_seed is None else self.agent_seed, 0xD4]),
    )
    self.training_steps = 0
    self.gradient_steps = 0
    self.epsilon = self.epsilon_start
    self.logger.info(
        "Fresh %s: augmentation=%s tactical_residual=%s seed=%s.",
        self.v6_config["variant"], self.v6_config["augmentation"],
        self.v6_config["tactical_residual"], self.agent_seed,
    )


def _batch_tensors(transitions: list[Transition], device, rng, augmentation: str):
    local = np.stack([item.state[0] for item in transitions])
    global_features = np.stack([item.state[1] for item in transitions])
    tactical = np.stack([item.state[2] for item in transitions])
    actions = np.asarray([item.action for item in transitions], dtype=np.int64)
    rewards = np.asarray([item.reward for item in transitions], dtype=np.float32)
    dones = np.asarray([item.done for item in transitions], dtype=np.float32)
    next_local = np.stack([item.next_state[0] if item.next_state is not None else np.zeros_like(item.state[0]) for item in transitions])
    next_global = np.stack([item.next_state[1] if item.next_state is not None else np.zeros_like(item.state[1]) for item in transitions])
    next_tactical = np.stack([
        item.next_state[2] if item.next_state is not None
        else np.zeros((len(ACTIONS), ACTION_FEATURE_SIZE), dtype=np.float32)
        for item in transitions
    ])
    next_masks = np.stack([item.next_mask if item.next_state is not None else np.zeros(len(ACTIONS), dtype=bool) for item in transitions])
    if augmentation == "random_d4":
        (
            local, global_features, tactical, actions, rewards, dones,
            next_local, next_global, next_tactical, next_masks,
        ) = augment_transition_arrays_random(
            local, global_features, tactical, actions, rewards, dones,
            next_local, next_global, next_tactical, next_masks, rng,
        )
    return tuple(torch.as_tensor(value, device=device) for value in (
        local, global_features, tactical, actions, rewards, dones,
        next_local, next_global, next_tactical, next_masks,
    ))


def _learn(self):
    if len(self.replay_buffer) < self.warmup_transitions or self.training_steps % self.update_every:
        return
    transitions = self.replay_buffer.sample(self.batch_size, self.replay_rng)
    (
        local, global_features, tactical, actions, rewards, dones,
        next_local, next_global, next_tactical, next_masks,
    ) = _batch_tensors(
        transitions, self.device, self.augmentation_rng, self.v6_config["augmentation"],
    )
    q_values = self.online_net(local.float(), global_features.float(), tactical.float()).gather(1, actions.long().unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        online_next = self.online_net(next_local.float(), next_global.float(), next_tactical.float())
        online_next = online_next.masked_fill(~next_masks.bool(), -1e9)
        next_actions = online_next.argmax(dim=1)
        target_next = self.target_net(next_local.float(), next_global.float(), next_tactical.float()).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards.float() + self.gamma * (1.0 - dones.float()) * target_next
    loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
    self.optimizer.step()
    self.gradient_steps += 1
    if self.gradient_steps % self.target_update_every == 0:
        self.target_net.load_state_dict(self.online_net.state_dict())
    if self.gradient_steps % 100 == 0:
        self.logger.info("Model A update=%d replay=%d loss=%.4f epsilon=%.3f", self.gradient_steps, len(self.replay_buffer), loss.item(), self.epsilon)


def _record(self, old_game_state, action: str, new_game_state, reward: float, done: bool):
    base_state = state_to_features(old_game_state)
    if base_state is None or action not in ACTIONS:
        return
    tactical = (
        action_tactical_features(old_game_state) if self.v6_config["tactical_residual"]
        else np.zeros((len(ACTIONS), ACTION_FEATURE_SIZE), dtype=np.float32)
    )
    state = (*base_state, tactical)
    next_base = None if done else state_to_features(new_game_state)
    next_state = None
    if next_base is not None:
        next_tactical = (
            action_tactical_features(new_game_state) if self.v6_config["tactical_residual"]
            else np.zeros((len(ACTIONS), ACTION_FEATURE_SIZE), dtype=np.float32)
        )
        next_state = (*next_base, next_tactical)
    next_mask = None
    if next_state is not None:
        from .features import legal_action_mask
        next_mask = legal_action_mask(new_game_state)
    self.replay_buffer.add(Transition(state, ACTIONS.index(action), reward, next_state, done, next_mask))
    self.training_steps += 1
    progress = min(1.0, self.training_steps / self.epsilon_decay_steps)
    self.epsilon = self.epsilon_start + progress * (self.epsilon_final - self.epsilon_start)
    _learn(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = reward_from_events(events, self.reward_by_event) + _state_shaping(
        old_game_state, new_game_state,
        danger_exit_reward=float(self.shaping_config["danger_exit_reward"]),
        danger_entry_penalty=float(self.shaping_config["danger_entry_penalty"]),
    )
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
            "architecture": self.model_architecture,
            "agent_seed": self.agent_seed,
            "variant": self.v6_config["variant"],
            "config_sha256": self.v6_config_sha256,
        },
        CHECKPOINT_PATH,
    )


def end_of_round(self, last_game_state, last_action, events):
    _record(
        self, last_game_state, last_action, None,
        reward_from_events(events, self.reward_by_event), done=True,
    )
    self.completed_rounds += 1
    if self.completed_rounds % self.checkpoint_every_rounds == 0:
        _save_checkpoint(self)
        self.logger.info("Saved Model A checkpoint after round %d.", self.completed_rounds)
