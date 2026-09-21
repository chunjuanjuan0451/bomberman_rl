"""Frozen-D Double DQN whose only experimental variable is replay sampling."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass

import events as e
import numpy as np

from agent_code.model_a_task3.features import task3_action_features
from agent_code.model_a_task3.network import torch
from agent_code.model_a_v6.features import ACTIONS, danger_time_map, legal_action_mask, state_to_features
from agent_code.model_a_v6.symmetry import ACTION_PERMUTATIONS, transform_local

from .callbacks import CHECKPOINT_PATH
from .config import architecture_name
from .replay import ReplayBuffer, Transition


COMMON_REWARD = {
    e.COIN_COLLECTED: 8.0,
    e.CRATE_DESTROYED: 1.5,
    e.KILLED_SELF: -20.0,
    e.GOT_KILLED: -20.0,
    e.INVALID_ACTION: -2.0,
    e.BOMB_DROPPED: -0.05,
    e.SURVIVED_ROUND: 3.0,
}
KILL_REWARD = 40.0


@dataclass
class RawTransition:
    state: object
    action: int
    reward: float
    next_state: object
    done: bool
    next_mask: object
    episode_id: int
    start_step: int


def aggregate_n_step(queue, n_step: int, gamma: float) -> Transition:
    window = list(queue)[:n_step]
    if not window:
        raise ValueError("cannot aggregate an empty n-step queue")
    reward, used, final = 0.0, 0, window[0]
    for index, raw in enumerate(window):
        reward += (gamma ** index) * float(raw.reward)
        final, used = raw, index + 1
        if raw.done:
            break
    first = window[0]
    return Transition(
        state=first.state, action=first.action, reward=reward,
        next_state=final.next_state, done=final.done, next_mask=final.next_mask,
        bootstrap_discount=gamma ** used, episode_id=first.episode_id,
        start_step=first.start_step,
    )


def _reward(events) -> float:
    return sum(COMMON_REWARD.get(event, 0.0) for event in events) + KILL_REWARD * events.count(e.KILLED_OPPONENT)


def _shaping(old, new) -> float:
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
    hyper = self.task3_config["hyperparameters"]
    self.gamma = float(hyper["gamma"])
    self.n_step = 3
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
    self.kill_causal_fraction = float(self.task3_config["kill_causal_fraction"])
    self.kill_causal_window = int(self.task3_config["kill_causal_window"])
    self.target_net = copy.deepcopy(self.online_net).to(self.device)
    self.target_net.freeze_base()
    self.optimizer = torch.optim.Adam(
        self.online_net.residual.parameters(), lr=float(hyper["learning_rate"]),
    )
    self.replay_buffer = ReplayBuffer(int(hyper["replay_capacity"]))
    self.n_step_queue = deque()
    self.replay_rng = np.random.default_rng(self.agent_seed)
    self.d4_rng = np.random.default_rng(np.random.SeedSequence([self.agent_seed, 0xD4]))
    self.training_steps = self.gradient_steps = self.kill_events_seen = 0
    self.epsilon = self.epsilon_start


def _batch(transitions, device):
    local = np.stack([item.state[0] for item in transitions])
    global_features = np.stack([item.state[1] for item in transitions])
    tactical = np.stack([item.state[2] for item in transitions])
    actions = np.asarray([item.action for item in transitions], dtype=np.int64)
    rewards = np.asarray([item.reward for item in transitions], dtype=np.float32)
    dones = np.asarray([item.done for item in transitions], dtype=np.float32)
    discounts = np.asarray([item.bootstrap_discount for item in transitions], dtype=np.float32)
    next_local = np.stack([
        item.next_state[0] if item.next_state is not None else np.zeros_like(item.state[0])
        for item in transitions
    ])
    next_global = np.stack([
        item.next_state[1] if item.next_state is not None else np.zeros_like(item.state[1])
        for item in transitions
    ])
    next_tactical = np.stack([
        item.next_state[2] if item.next_state is not None else np.zeros_like(item.state[2])
        for item in transitions
    ])
    next_masks = np.stack([
        item.next_mask if item.next_state is not None else np.zeros(len(ACTIONS), dtype=bool)
        for item in transitions
    ])
    arrays = (
        local, global_features, tactical, actions, rewards, dones, discounts,
        next_local, next_global, next_tactical, next_masks,
    )
    return tuple(torch.as_tensor(value, device=device) for value in arrays)


def _consistency_loss(self, local, global_features, tactical):
    count = int(round(self.batch_size * self.d4_fraction))
    if count == 0:
        return local.new_zeros(())
    indices = self.d4_rng.choice(self.batch_size, size=count, replace=False)
    source_local, source_global, source_tactical = local[indices], global_features[indices], tactical[indices]
    transformed_local = np.empty(tuple(source_local.shape), dtype=np.float32)
    transformed_tactical = np.empty(tuple(source_tactical.shape), dtype=np.float32)
    permutations = []
    symmetries = self.d4_rng.integers(1, 8, size=count)
    local_numpy, tactical_numpy = source_local.detach().cpu().numpy(), source_tactical.detach().cpu().numpy()
    for row, symmetry in enumerate(symmetries):
        permutation = ACTION_PERMUTATIONS[int(symmetry)]
        transformed_local[row] = transform_local(local_numpy[row], int(symmetry))
        transformed_tactical[row, permutation] = tactical_numpy[row]
        permutations.append(permutation)
    transformed_q = self.online_net(
        torch.as_tensor(transformed_local, device=self.device), source_global.float(),
        torch.as_tensor(transformed_tactical, device=self.device),
    )
    permutation_tensor = torch.as_tensor(np.stack(permutations), device=self.device)
    aligned_q = transformed_q.gather(1, permutation_tensor.long())
    with torch.no_grad():
        source_q = self.online_net(source_local.float(), source_global.float(), source_tactical.float())
    return torch.nn.functional.smooth_l1_loss(aligned_q, source_q)


def _learn(self):
    if len(self.replay_buffer) < self.warmup_transitions or self.training_steps % self.update_every:
        return
    sampled = self.replay_buffer.sample(
        self.batch_size, self.replay_rng, self.kill_causal_fraction,
    )
    (local, global_features, tactical, actions, rewards, dones, discounts,
     next_local, next_global, next_tactical, next_masks) = _batch(sampled, self.device)
    q_values = self.online_net(local.float(), global_features.float(), tactical.float()).gather(
        1, actions.long().unsqueeze(1),
    ).squeeze(1)
    with torch.no_grad():
        online_next = self.online_net(
            next_local.float(), next_global.float(), next_tactical.float(),
        ).masked_fill(~next_masks.bool(), -1e9)
        next_actions = online_next.argmax(dim=1)
        target_next = self.target_net(
            next_local.float(), next_global.float(), next_tactical.float(),
        ).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards.float() + discounts.float() * (1.0 - dones.float()) * target_next
    td_loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
    consistency = _consistency_loss(self, local.float(), global_features.float(), tactical.float())
    weight = self.d4_lambda_final * min(1.0, self.gradient_steps / max(1, self.d4_lambda_warmup))
    loss = td_loss + weight * consistency
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.online_net.residual.parameters(), self.gradient_clip)
    self.optimizer.step()
    self.gradient_steps += 1
    if self.gradient_steps % self.target_update_every == 0:
        self.target_net.residual.load_state_dict(self.online_net.residual.state_dict())


def _features(game_state):
    if game_state is None:
        return None
    local, global_features = state_to_features(game_state)
    return local, global_features, task3_action_features(game_state)


def _emit_ready(self, terminal: bool) -> None:
    while self.n_step_queue and (terminal or len(self.n_step_queue) >= self.n_step):
        self.replay_buffer.add(aggregate_n_step(self.n_step_queue, self.n_step, self.gamma))
        self.n_step_queue.popleft()
        if not terminal:
            break


def _record(self, old, action, new, reward, done, events):
    state = _features(old)
    if state is None or action not in ACTIONS:
        return
    next_state = None if done else _features(new)
    next_mask = None if next_state is None else legal_action_mask(new)
    episode_id, start_step = int(old["round"]), int(old["step"])
    self.n_step_queue.append(RawTransition(
        state=state, action=ACTIONS.index(action), reward=reward,
        next_state=next_state, done=done, next_mask=next_mask,
        episode_id=episode_id, start_step=start_step,
    ))
    _emit_ready(self, done)
    kill_count = events.count(e.KILLED_OPPONENT)
    if kill_count:
        self.kill_events_seen += kill_count
        self.replay_buffer.mark_kill_window(episode_id, start_step, self.kill_causal_window)
    self.training_steps += 1
    progress = min(1.0, self.training_steps / self.epsilon_decay_steps)
    self.epsilon = self.epsilon_start + progress * (self.epsilon_final - self.epsilon_start)
    _learn(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _record(
        self, old_game_state, self_action, new_game_state,
        _reward(events) + _shaping(old_game_state, new_game_state), False, events,
    )


def _save(self):
    payload = {
        "online_net": self.online_net.state_dict(),
        "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "completed_rounds": self.completed_rounds,
        "training_steps": self.training_steps,
        "gradient_steps": self.gradient_steps,
        "epsilon": self.epsilon,
        "architecture": architecture_name(self.task3_config),
        "config_sha256": self.task3_config_sha256,
        "parent_sha256": self.task3_config["parent_sha256"],
        "arm": "D",
        "reward_profile": "official-aligned",
        "action_features": True,
        "n_step": 3,
        "sampling_profile": self.task3_config["sampling_profile"],
        "kill_causal_fraction": self.kill_causal_fraction,
        "kill_causal_window": self.kill_causal_window,
        "kill_events_seen": self.kill_events_seen,
        "replay_diagnostics": self.replay_buffer.diagnostics(),
    }
    temporary = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(CHECKPOINT_PATH)


def end_of_round(self, last_game_state, last_action, events):
    _record(self, last_game_state, last_action, None, _reward(events), True, events)
    if self.n_step_queue:
        raise RuntimeError("n-step queue did not flush at episode end")
    self.completed_rounds += 1
    if self.completed_rounds % self.checkpoint_every_rounds == 0:
        _save(self)
