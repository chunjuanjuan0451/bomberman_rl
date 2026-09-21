"""Residual-only Double-DQN training with optional weak D4 consistency."""

from __future__ import annotations

import copy

import numpy as np
import events as e

from agent_code.model_a_v6.features import ACTIONS, danger_time_map, legal_action_mask, state_to_features
from agent_code.model_a_v6.replay_buffer import ReplayBuffer, Transition
from agent_code.model_a_v6.symmetry import ACTION_PERMUTATIONS, transform_local
from .callbacks import CHECKPOINT_PATH
from .config import architecture_name
from .network import torch

REWARD_BY_EVENT = {
    e.COIN_COLLECTED: 8.0, e.CRATE_DESTROYED: 1.5,
    e.KILLED_OPPONENT: 12.0, e.KILLED_SELF: -20.0,
    e.GOT_KILLED: -20.0, e.INVALID_ACTION: -2.0,
    e.BOMB_DROPPED: -0.05, e.SURVIVED_ROUND: 3.0,
}


def _reward(events):
    return sum(REWARD_BY_EVENT.get(event, 0.0) for event in events)


def _shaping(old, new):
    if old is None or new is None:
        return 0.0
    old_pos, new_pos = tuple(old["self"][3]), tuple(new["self"][3])
    old_danger, new_danger = danger_time_map(old)[old_pos], danger_time_map(new)[new_pos]
    if old_danger <= 2 and new_danger > 2:
        return 0.4
    if old_danger > 2 and new_danger <= 1:
        return -0.8
    return 0.0


def setup_training(self):
    hyper = self.stable_config["hyperparameters"]
    self.gamma = float(hyper["gamma"])
    self.batch_size = int(hyper["batch_size"])
    self.warmup_transitions = int(hyper["warmup_transitions"])
    self.update_every = int(hyper["update_every"])
    self.target_update_every = int(hyper["target_update_every"])
    self.checkpoint_every_rounds = int(hyper["checkpoint_every_rounds"])
    self.gradient_clip = float(hyper["gradient_clip"])
    self.epsilon_start = float(hyper["epsilon_start"])
    self.epsilon_final = float(hyper["epsilon_final"])
    self.epsilon_decay_steps = int(hyper["epsilon_decay_steps"])
    self.d4_fraction = float(hyper["d4_batch_fraction"])
    self.d4_lambda_final = float(hyper["d4_lambda_final"])
    self.d4_lambda_warmup = int(hyper["d4_lambda_warmup_updates"])
    self.target_net = copy.deepcopy(self.online_net).to(self.device)
    self.target_net.freeze_base()
    residual_parameters = list(self.online_net.residual.parameters())
    self.optimizer = torch.optim.Adam(residual_parameters, lr=float(hyper["learning_rate"]))
    self.replay_buffer = ReplayBuffer(int(hyper["replay_capacity"]))
    self.replay_rng = np.random.default_rng(self.agent_seed)
    self.d4_rng = np.random.default_rng(np.random.SeedSequence([self.agent_seed or 0, 0xD4]))
    self.training_steps = self.gradient_steps = 0
    self.epsilon = self.epsilon_start


def _batch(transitions, device):
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


def _consistency_loss(self, local, global_features):
    count = int(round(self.batch_size * self.d4_fraction))
    if not self.stable_config["d4_consistency"] or count == 0:
        return local.new_zeros(())
    indices = self.d4_rng.choice(self.batch_size, size=count, replace=False)
    source_local = local[indices]
    source_global = global_features[indices]
    transformed = np.empty(tuple(source_local.shape), dtype=np.float32)
    permutations = []
    symmetries = self.d4_rng.integers(1, 8, size=count)
    source_numpy = source_local.detach().cpu().numpy()
    for row, symmetry in enumerate(symmetries):
        transformed[row] = transform_local(source_numpy[row], int(symmetry))
        permutations.append(ACTION_PERMUTATIONS[int(symmetry)])
    transformed_q = self.online_net(
        torch.as_tensor(transformed, device=self.device), source_global.float(),
    )
    permutation_tensor = torch.as_tensor(np.stack(permutations), device=self.device)
    aligned_q = transformed_q.gather(1, permutation_tensor.long())
    with torch.no_grad():
        source_q = self.online_net(source_local.float(), source_global.float())
    return torch.nn.functional.smooth_l1_loss(aligned_q, source_q)


def _learn(self):
    if len(self.replay_buffer) < self.warmup_transitions or self.training_steps % self.update_every:
        return
    transitions = self.replay_buffer.sample(self.batch_size, self.replay_rng)
    local, global_features, actions, rewards, dones, next_local, next_global, next_masks = _batch(transitions, self.device)
    q_values = self.online_net(local.float(), global_features.float()).gather(1, actions.long().unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        online_next = self.online_net(next_local.float(), next_global.float()).masked_fill(~next_masks.bool(), -1e9)
        next_actions = online_next.argmax(dim=1)
        target_next = self.target_net(next_local.float(), next_global.float()).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards.float() + self.gamma * (1.0 - dones.float()) * target_next
    td_loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
    consistency = _consistency_loss(self, local.float(), global_features.float())
    weight = self.d4_lambda_final * min(1.0, self.gradient_steps / max(1, self.d4_lambda_warmup))
    loss = td_loss + weight * consistency
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.online_net.residual.parameters(), self.gradient_clip)
    self.optimizer.step()
    self.gradient_steps += 1
    if self.gradient_steps % self.target_update_every == 0:
        self.target_net.residual.load_state_dict(self.online_net.residual.state_dict())


def _record(self, old, action, new, reward, done):
    state = state_to_features(old)
    if state is None or action not in ACTIONS:
        return
    next_state = None if done else state_to_features(new)
    next_mask = None if next_state is None else legal_action_mask(new)
    self.replay_buffer.add(Transition(state, ACTIONS.index(action), reward, next_state, done, next_mask))
    self.training_steps += 1
    progress = min(1.0, self.training_steps / self.epsilon_decay_steps)
    self.epsilon = self.epsilon_start + progress * (self.epsilon_final - self.epsilon_start)
    _learn(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _record(self, old_game_state, self_action, new_game_state, _reward(events) + _shaping(old_game_state, new_game_state), False)


def _save(self):
    torch.save({
        "online_net": self.online_net.state_dict(),
        "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "completed_rounds": self.completed_rounds,
        "training_steps": self.training_steps,
        "gradient_steps": self.gradient_steps,
        "epsilon": self.epsilon,
        "architecture": architecture_name(),
        "config_sha256": self.stable_config_sha256,
        "parent_sha256": self.stable_config["parent_sha256"],
        "variant": self.stable_config["variant"],
    }, CHECKPOINT_PATH)


def end_of_round(self, last_game_state, last_action, events):
    _record(self, last_game_state, last_action, None, _reward(events), True)
    self.completed_rounds += 1
    if self.completed_rounds % self.checkpoint_every_rounds == 0:
        _save(self)
