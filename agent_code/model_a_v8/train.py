"""Residual Double-DQN training with soft rollout and behavior constraints."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import events as e

from agent_code.model_a_v6.features import ACTIONS, danger_time_map, legal_action_mask, state_to_features
from agent_code.model_a_v6.replay_buffer import ReplayBuffer
from agent_code.model_a_v6.symmetry import ACTION_PERMUTATIONS, transform_local
from agent_code.model_a_v7b.tactical import evaluate_bomb
from .callbacks import CHECKPOINT_PATH
from .config import architecture_name
from .network import torch

REWARD_BY_EVENT = {
    e.COIN_COLLECTED: 8.0, e.CRATE_DESTROYED: 1.5, e.KILLED_OPPONENT: 12.0,
    e.KILLED_SELF: -20.0, e.GOT_KILLED: -20.0, e.INVALID_ACTION: -2.0,
    e.BOMB_DROPPED: -0.05, e.SURVIVED_ROUND: 3.0,
}


@dataclass
class V8Transition:
    state: object
    action: int
    reward: float
    next_state: object
    done: bool
    next_mask: object
    rollout_target: np.ndarray
    rollout_mask: np.ndarray


def _reward(events):
    return sum(REWARD_BY_EVENT.get(event, 0.0) for event in events)


def _shaping(old, new):
    if old is None or new is None:
        return 0.0
    old_pos, new_pos = tuple(old["self"][3]), tuple(new["self"][3])
    old_danger, new_danger = danger_time_map(old)[old_pos], danger_time_map(new)[new_pos]
    return 0.4 if old_danger <= 2 and new_danger > 2 else (-0.8 if old_danger > 2 and new_danger <= 1 else 0.0)


def rollout_soft_target(game_state, delta_cap):
    """Return a bounded counterfactual label without making a hard decision."""
    target = np.zeros(len(ACTIONS), dtype=np.float32)
    mask = np.zeros(len(ACTIONS), dtype=bool)
    legal = legal_action_mask(game_state)
    bomb = ACTIONS.index("BOMB")
    if not legal[bomb]:
        return target, mask
    tactics = evaluate_bomb(game_state, perf_counter() + 0.03)
    if tactics is None:
        return target, mask
    raw = (
        0.70 * min(1, tactics.guaranteed_traps)
        + 0.25 * tactics.max_space_reduction
        + 0.05 * min(3, tactics.crates_destroyed)
        - 0.20
    )
    if tactics.own_bottleneck <= 1:
        raw -= 0.30
    target[bomb] = float(delta_cap * np.clip(raw, -1.0, 1.0))
    mask[bomb] = True
    return target, mask


def setup_training(self):
    h = self.v8_config["hyperparameters"]
    for key in (
        "gamma", "batch_size", "warmup_transitions", "update_every", "target_update_every",
        "checkpoint_every_rounds", "gradient_clip", "epsilon_start", "epsilon_final",
        "epsilon_decay_steps", "d4_batch_fraction", "d4_lambda_final",
        "d4_lambda_warmup_updates", "rollout_lambda", "behavior_kl_lambda",
        "behavior_temperature", "delta_cap",
    ):
        setattr(self, key, type(h[key])(h[key]))
    self.target_net = copy.deepcopy(self.online_net).to(self.device)
    self.target_net.freeze_base()
    self.optimizer = torch.optim.Adam(self.online_net.residual.parameters(), lr=float(h["learning_rate"]))
    self.replay_buffer = ReplayBuffer(int(h["replay_capacity"]))
    self.replay_rng = np.random.default_rng(self.agent_seed)
    self.d4_rng = np.random.default_rng(np.random.SeedSequence([self.agent_seed or 0, 0xD4, 8]))
    self.training_steps = self.gradient_steps = 0
    self.epsilon = self.epsilon_start


def _batch(items, device):
    local = np.stack([x.state[0] for x in items])
    glob = np.stack([x.state[1] for x in items])
    actions = np.asarray([x.action for x in items])
    rewards = np.asarray([x.reward for x in items], dtype=np.float32)
    dones = np.asarray([x.done for x in items], dtype=np.float32)
    next_local = np.stack([x.next_state[0] if x.next_state is not None else np.zeros_like(x.state[0]) for x in items])
    next_glob = np.stack([x.next_state[1] if x.next_state is not None else np.zeros_like(x.state[1]) for x in items])
    next_masks = np.stack([x.next_mask if x.next_state is not None else np.zeros(len(ACTIONS), bool) for x in items])
    rollout = np.stack([x.rollout_target for x in items])
    rollout_masks = np.stack([x.rollout_mask for x in items])
    return tuple(torch.as_tensor(x, device=device) for x in (
        local, glob, actions, rewards, dones, next_local, next_glob, next_masks, rollout, rollout_masks,
    ))


def _d4_loss(self, local, glob):
    count = int(round(self.batch_size * self.d4_batch_fraction))
    if count == 0:
        return local.new_zeros(())
    indices = self.d4_rng.choice(self.batch_size, size=count, replace=False)
    source = local[indices]
    transformed = np.empty(tuple(source.shape), dtype=np.float32)
    permutations = []
    symmetries = self.d4_rng.integers(1, 8, size=count)
    source_np = source.detach().cpu().numpy()
    for row, symmetry in enumerate(symmetries):
        transformed[row] = transform_local(source_np[row], int(symmetry))
        permutations.append(ACTION_PERMUTATIONS[int(symmetry)])
    out = self.online_net(torch.as_tensor(transformed, device=self.device), glob[indices].float())
    aligned = out.gather(1, torch.as_tensor(np.stack(permutations), device=self.device).long())
    with torch.no_grad():
        reference = self.online_net(source.float(), glob[indices].float())
    return torch.nn.functional.smooth_l1_loss(aligned, reference)


def _learn(self):
    if len(self.replay_buffer) < self.warmup_transitions or self.training_steps % self.update_every:
        return
    batch = _batch(self.replay_buffer.sample(self.batch_size, self.replay_rng), self.device)
    local, glob, actions, rewards, dones, next_local, next_glob, next_masks, rollout, rollout_masks = batch
    final, delta = self.online_net(local.float(), glob.float(), return_delta=True)
    chosen = final.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        online_next = self.online_net(next_local.float(), next_glob.float()).masked_fill(~next_masks.bool(), -1e9)
        next_actions = online_next.argmax(1)
        target_next = self.target_net(next_local.float(), next_glob.float()).gather(1, next_actions[:, None]).squeeze(1)
        targets = rewards.float() + self.gamma * (1.0 - dones.float()) * target_next
        base = final - delta
        base_policy = torch.softmax(base / self.behavior_temperature, dim=1)
    td = torch.nn.functional.smooth_l1_loss(chosen, targets)
    if rollout_masks.any():
        distill = torch.nn.functional.smooth_l1_loss(delta[rollout_masks.bool()], rollout.float()[rollout_masks.bool()])
    else:
        distill = final.new_zeros(())
    log_policy = torch.log_softmax(final / self.behavior_temperature, dim=1)
    behavior_kl = torch.nn.functional.kl_div(log_policy, base_policy, reduction="batchmean")
    d4_weight = self.d4_lambda_final * min(1.0, self.gradient_steps / max(1, self.d4_lambda_warmup_updates))
    loss = td + self.rollout_lambda * distill + self.behavior_kl_lambda * behavior_kl + d4_weight * _d4_loss(self, local.float(), glob.float())
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
    next_mask = None if done else legal_action_mask(new)
    rollout, rollout_mask = rollout_soft_target(old, self.delta_cap)
    self.replay_buffer.add(V8Transition(
        state, ACTIONS.index(action), reward, next_state, done, next_mask, rollout, rollout_mask,
    ))
    self.training_steps += 1
    progress = min(1.0, self.training_steps / self.epsilon_decay_steps)
    self.epsilon = self.epsilon_start + progress * (self.epsilon_final - self.epsilon_start)
    _learn(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _record(self, old_game_state, self_action, new_game_state, _reward(events) + _shaping(old_game_state, new_game_state), False)


def _save(self):
    torch.save({
        "online_net": self.online_net.state_dict(), "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(), "completed_rounds": self.completed_rounds,
        "training_steps": self.training_steps, "gradient_steps": self.gradient_steps,
        "epsilon": self.epsilon, "architecture": architecture_name(),
        "config_sha256": self.v8_config_sha256,
        "parent_sha256": self.v8_config["parent_sha256"], "variant": self.v8_config["variant"],
    }, CHECKPOINT_PATH)


def end_of_round(self, last_game_state, last_action, events):
    _record(self, last_game_state, last_action, None, _reward(events), True)
    self.completed_rounds += 1
    if self.completed_rounds % self.checkpoint_every_rounds == 0:
        _save(self)
