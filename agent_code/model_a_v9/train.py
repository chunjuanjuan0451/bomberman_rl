"""From-scratch n-step prioritized quantile-DQN training for v9."""

from __future__ import annotations

import copy
from collections import deque

import numpy as np
import events as e

from agent_code.model_a_v6.symmetry import ACTION_PERMUTATIONS, transform_local
from .callbacks import CHECKPOINT_PATH
from .config import architecture_name
from .features import ACTIONS, counterfactual_risk_labels, legal_action_mask, state_to_features
from .network import torch
from .replay import PrioritizedReplay, Transition

REWARD = {
    e.COIN_COLLECTED: 1.0, e.KILLED_OPPONENT: 5.0, e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0, e.INVALID_ACTION: -0.25, e.CRATE_DESTROYED: 0.1,
    e.SURVIVED_ROUND: 0.5,
}


def setup_training(self):
    h = self.v9_config["hyperparameters"]
    for key, value in h.items():
        setattr(self, key, value)
    self.target_net = copy.deepcopy(self.online_net).to(self.device).eval()
    self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=self.learning_rate, weight_decay=1e-5)
    self.replay_buffer = PrioritizedReplay(self.replay_capacity, self.per_alpha)
    self.replay_rng = np.random.default_rng(self.agent_seed)
    self.nstep_queue = deque()
    self.training_steps = self.gradient_steps = 0
    self.epsilon = self.epsilon_start


def _reward(events):
    return sum(REWARD.get(event, 0.0) for event in events)


def _cached_state(self, game_state):
    key = None if game_state is None else (game_state["round"], game_state["step"])
    if key == self.last_v9_state_key:
        return self.last_v9_features
    return state_to_features(game_state, self.frame_history)


def _emit_nstep(self, length):
    items = list(self.nstep_queue)[:length]
    first, last = items[0], items[-1]
    reward = sum((self.gamma ** index) * item.reward for index, item in enumerate(items))
    self.replay_buffer.add(Transition(
        first.state, first.action, reward, self.gamma ** length, last.next_state,
        last.done, last.next_mask, first.risk, first.risk_mask,
    ))
    self.nstep_queue.popleft()


def _record(self, old, action, new, events, done):
    state = _cached_state(self, old)
    if state is None or action not in ACTIONS:
        return
    next_state = None if done else state_to_features(new, self.frame_history)
    next_mask = None if done else legal_action_mask(new)
    risk, risk_mask = counterfactual_risk_labels(old)
    raw = Transition(state, ACTIONS.index(action), _reward(events), self.gamma, next_state, done, next_mask, risk, risk_mask)
    self.nstep_queue.append(raw)
    if len(self.nstep_queue) >= self.n_step:
        _emit_nstep(self, self.n_step)
    if done:
        while self.nstep_queue:
            _emit_nstep(self, len(self.nstep_queue))
    self.training_steps += 1
    progress = min(1.0, self.training_steps / self.epsilon_decay_steps)
    self.epsilon = self.epsilon_start + progress * (self.epsilon_final - self.epsilon_start)
    _learn(self)


def _augment(self, items):
    spatial = np.stack([item.state[0] for item in items])
    glob = np.stack([item.state[1] for item in items])
    actions = np.asarray([item.action for item in items], np.int64)
    rewards = np.asarray([item.reward for item in items], np.float32)
    discounts = np.asarray([item.discount for item in items], np.float32)
    dones = np.asarray([item.done for item in items], np.float32)
    next_spatial = np.stack([item.next_state[0] if item.next_state is not None else np.zeros_like(item.state[0]) for item in items])
    next_glob = np.stack([item.next_state[1] if item.next_state is not None else np.zeros_like(item.state[1]) for item in items])
    next_masks = np.stack([item.next_mask if item.next_state is not None else np.zeros(len(ACTIONS), bool) for item in items])
    risks = np.stack([item.risk for item in items])
    risk_masks = np.stack([item.risk_mask for item in items])
    for row, symmetry in enumerate(self.replay_rng.integers(0, 8, size=len(items))):
        if symmetry == 0:
            continue
        permutation = ACTION_PERMUTATIONS[int(symmetry)]
        spatial[row] = transform_local(spatial[row], int(symmetry))
        next_spatial[row] = transform_local(next_spatial[row], int(symmetry))
        actions[row] = permutation[actions[row]]
        transformed_mask = np.zeros(len(ACTIONS), bool)
        transformed_mask[permutation] = next_masks[row]
        next_masks[row] = transformed_mask
        transformed_risk = np.zeros(len(ACTIONS), np.float32)
        transformed_risk[permutation] = risks[row]
        risks[row] = transformed_risk
        transformed_risk_mask = np.zeros(len(ACTIONS), bool)
        transformed_risk_mask[permutation] = risk_masks[row]
        risk_masks[row] = transformed_risk_mask
    return spatial, glob, actions, rewards, discounts, dones, next_spatial, next_glob, next_masks, risks, risk_masks


def _learn(self):
    if len(self.replay_buffer) < self.warmup_transitions or self.training_steps % self.update_every:
        return
    beta = self.per_beta_start + (1.0 - self.per_beta_start) * min(1.0, self.training_steps / self.per_beta_steps)
    items, indices, weights = self.replay_buffer.sample(self.batch_size, beta, self.replay_rng)
    arrays = _augment(self, items)
    spatial, glob, actions, rewards, discounts, dones, next_spatial, next_glob, next_masks, risks, risk_masks = [
        torch.as_tensor(value, device=self.device) for value in arrays
    ]
    weights = torch.as_tensor(weights, device=self.device)
    quantiles, risk_logits = self.online_net(spatial.float(), glob.float(), return_risk=True)
    predicted = quantiles[torch.arange(self.batch_size, device=self.device), actions.long()]
    with torch.no_grad():
        next_online = self.online_net(next_spatial.float(), next_glob.float()).mean(-1).masked_fill(~next_masks.bool(), -1e9)
        next_actions = next_online.argmax(1)
        next_quantiles = self.target_net(next_spatial.float(), next_glob.float())[
            torch.arange(self.batch_size, device=self.device), next_actions,
        ]
        target = rewards[:, None] + discounts[:, None] * (1.0 - dones[:, None]) * next_quantiles
    td = target[:, None, :] - predicted[:, :, None]
    abs_td = td.abs()
    huber = torch.where(abs_td <= 1.0, 0.5 * td.square(), abs_td - 0.5)
    taus = ((torch.arange(self.n_quantiles, device=self.device) + 0.5) / self.n_quantiles)[None, :, None]
    quantile_loss = (torch.abs(taus - (td.detach() < 0).float()) * huber).mean((1, 2))
    risk_loss = torch.nn.functional.binary_cross_entropy_with_logits(risk_logits, risks.float(), reduction="none")
    risk_loss = (risk_loss * risk_masks.float()).sum(1) / risk_masks.float().sum(1).clamp_min(1.0)
    loss = (weights * (quantile_loss + self.risk_lambda * risk_loss)).mean()
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip)
    self.optimizer.step()
    priorities = (target.mean(1) - predicted.detach().mean(1)).abs().cpu().numpy()
    self.replay_buffer.update(indices, priorities)
    self.gradient_steps += 1
    if self.gradient_steps % self.target_update_every == 0:
        self.target_net.load_state_dict(self.online_net.state_dict())


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _record(self, old_game_state, self_action, new_game_state, events, False)


def _save(self):
    payload = {
        "online_net": self.online_net.state_dict(), "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(), "completed_rounds": self.completed_rounds,
        "training_steps": self.training_steps, "gradient_steps": self.gradient_steps,
        "epsilon": self.epsilon, "architecture": architecture_name(),
        "config_sha256": self.v9_config_sha256, "variant": self.v9_config["variant"],
    }
    torch.save(payload, CHECKPOINT_PATH)
    if self.completed_rounds % 500 == 0:
        milestone = CHECKPOINT_PATH.with_name(f"{CHECKPOINT_PATH.stem}-r{self.completed_rounds:04d}{CHECKPOINT_PATH.suffix}")
        torch.save(payload, milestone)


def end_of_round(self, last_game_state, last_action, events):
    _record(self, last_game_state, last_action, None, events, True)
    self.completed_rounds += 1
    if self.completed_rounds % self.checkpoint_every_rounds == 0:
        _save(self)
