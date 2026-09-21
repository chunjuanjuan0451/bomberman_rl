"""Uniform-replay n=8 Double-DQN training for the clean CNN curriculum."""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import events as e

from .callbacks import _load
from .features import ACTIONS, legal_action_mask, state_to_features
from .network import ARCHITECTURE, torch
from .replay import ReplayBuffer, Transition
from .rewards import OFFICIAL, SHAPING_NAMES, official_components, reward_total


@dataclass
class RawTransition:
    state: tuple[np.ndarray, np.ndarray]
    action: int
    reward: float
    components: dict[str, float]
    next_state: tuple[np.ndarray, np.ndarray] | None
    done: bool
    next_mask: np.ndarray | None
    episode_id: int
    step: int
    events: tuple[str, ...]


def aggregate_n_step(queue, n_step: int, gamma: float) -> Transition:
    window = list(queue)[: int(n_step)]
    if not window:
        raise ValueError("empty n-step queue")
    first = window[0]
    reward = 0.0
    final = first
    used = 0
    for index, raw in enumerate(window):
        if raw.episode_id != first.episode_id:
            raise ValueError("n-step return crossed an episode")
        reward += gamma ** index * raw.reward
        final = raw
        used = index + 1
        if raw.done:
            break
    return Transition(first.state, first.action, reward, final.next_state, final.done,
                      final.next_mask, gamma ** used, used)


def setup_training(self) -> None:
    if self.cnn_mode != "train":
        raise RuntimeError("training callback loaded outside train mode")
    hyper = self.cnn_protocol["learning"]
    self.gamma = float(hyper["gamma"]); self.n_step = int(hyper["n_step"])
    self.batch_size = int(hyper["batch_size"]); self.update_every = int(hyper["update_every"])
    self.warmup = int(hyper["warmup_transitions"]); self.target_every_steps = int(hyper["target_sync_environment_steps"])
    self.checkpoint_every = int(hyper["checkpoint_every_rounds"])
    self.target_net = copy.deepcopy(self.online_net).to(self.device); self.target_net.eval()
    self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=float(hyper["learning_rate"]))
    self.replay_buffer = ReplayBuffer(int(hyper["replay_capacity"]))
    self.replay_rng = np.random.default_rng(np.random.SeedSequence([self.agent_seed, 0xC118]))
    self.n_step_queue = deque()
    self.stage_rounds = 0
    self.training_steps = 0
    self.gradient_steps = 0
    self.epsilon = float(hyper["epsilon_start"])
    if self.resume_checkpoint is not None:
        checkpoint = self.resume_checkpoint
        self.target_net.load_state_dict(checkpoint["target_net"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = int(checkpoint["training_steps"])
        self.gradient_steps = int(checkpoint["gradient_steps"])
        self.epsilon = float(checkpoint["epsilon"])
    names = list(OFFICIAL) + list(SHAPING_NAMES)
    self.diagnostics = {
        "raw_transitions": 0, "matured_targets": 0, "gradient_updates": 0,
        "sampled_transitions": 0, "return_steps_histogram": {str(i): 0 for i in range(1, self.n_step + 1)},
        "event_counts": {name: 0 for name in OFFICIAL},
        "reward_component_counts": {name: 0 for name in names},
        "reward_component_sums": {name: 0.0 for name in names},
        "action_counts": {name: 0 for name in ACTIONS},
        "survivor_terminal_merges": 0, "dead_terminal_appends": 0, "per_round": [],
    }
    self._round = None


def _batch(items, device):
    spatial = np.stack([item.state[0] for item in items]); scalars = np.stack([item.state[1] for item in items])
    actions = np.asarray([item.action for item in items], dtype=np.int64)
    rewards = np.asarray([item.reward for item in items], dtype=np.float32)
    dones = np.asarray([item.done for item in items], dtype=np.float32)
    discounts = np.asarray([item.bootstrap_discount for item in items], dtype=np.float32)
    next_spatial = np.stack([item.next_state[0] if item.next_state else np.zeros_like(item.state[0]) for item in items])
    next_scalars = np.stack([item.next_state[1] if item.next_state else np.zeros_like(item.state[1]) for item in items])
    masks = np.stack([item.next_mask if item.next_mask is not None else np.zeros(len(ACTIONS), bool) for item in items])
    return (
        torch.as_tensor(spatial, dtype=torch.float32, device=device).div_(255.0),
        torch.as_tensor(scalars, dtype=torch.float32, device=device),
        torch.as_tensor(actions, device=device), torch.as_tensor(rewards, device=device),
        torch.as_tensor(dones, device=device), torch.as_tensor(discounts, device=device),
        torch.as_tensor(next_spatial, dtype=torch.float32, device=device).div_(255.0),
        torch.as_tensor(next_scalars, dtype=torch.float32, device=device),
        torch.as_tensor(masks, device=device),
    )


def _learn(self) -> None:
    if len(self.replay_buffer) < max(self.warmup, self.batch_size) or self.training_steps % self.update_every:
        return
    batch = _batch(self.replay_buffer.sample(self.batch_size, self.replay_rng), self.device)
    spatial, scalars, actions, rewards, dones, discounts, next_spatial, next_scalars, masks = batch
    q = self.online_net(spatial, scalars).gather(1, actions.long().unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        online_next = self.online_net(next_spatial, next_scalars).masked_fill(~masks.bool(), -1e9)
        next_actions = online_next.argmax(1)
        target_next = self.target_net(next_spatial, next_scalars).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + discounts * (1.0 - dones) * target_next
    loss = torch.nn.functional.smooth_l1_loss(q, target)
    self.optimizer.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), float(self.cnn_protocol["learning"]["gradient_clip"]))
    self.optimizer.step(); self.gradient_steps += 1
    self.diagnostics["gradient_updates"] = self.gradient_steps
    self.diagnostics["sampled_transitions"] += self.batch_size
    if self.training_steps % self.target_every_steps < self.update_every:
        self.target_net.load_state_dict(self.online_net.state_dict())


def _raw(self, old, action, new, events, done) -> RawTransition | None:
    state = state_to_features(old)
    if state is None or action not in ACTIONS:
        return None
    reward, components = reward_total(old, None if done else new, action, events, self.cnn_stage)
    next_state = None if done else state_to_features(new)
    next_mask = None if done else legal_action_mask(new)
    return RawTransition(state, ACTIONS.index(action), reward, components, next_state, done, next_mask,
                         int(old["round"]), int(old["step"]), tuple(events))


def _add_target(self) -> None:
    item = aggregate_n_step(self.n_step_queue, self.n_step, self.gamma)
    self.replay_buffer.add(item); self.n_step_queue.popleft()
    self.diagnostics["matured_targets"] += 1
    self.diagnostics["return_steps_histogram"][str(item.return_steps)] += 1


def _ensure_round(self, raw: RawTransition) -> None:
    if self._round is None:
        self._round = {"round": raw.episode_id, "transitions": 0, "optimizer_updates": 0,
                       "actions": {name: 0 for name in ACTIONS}, "reward_sum": 0.0}


def _account(self, raw: RawTransition) -> None:
    _ensure_round(self, raw); d = self.diagnostics
    d["raw_transitions"] += 1; d["action_counts"][ACTIONS[raw.action]] += 1
    self._round["transitions"] += 1; self._round["actions"][ACTIONS[raw.action]] += 1
    self._round["reward_sum"] += raw.reward
    for event in raw.events:
        if event in d["event_counts"]: d["event_counts"][event] += 1
    for name, value in raw.components.items():
        d["reward_component_counts"][name] += 1; d["reward_component_sums"][name] += value


def _append(self, raw: RawTransition) -> None:
    before = self.gradient_steps; _account(self, raw); self.n_step_queue.append(raw)
    self.training_steps += 1
    decay = float(self.cnn_protocol["learning"]["epsilon_decay_steps"])
    progress = min(1.0, self.training_steps / decay)
    start = float(self.cnn_protocol["learning"]["epsilon_start"]); end = float(self.cnn_protocol["learning"]["epsilon_final"])
    self.epsilon = start + progress * (end - start)
    # Keep one look-ahead transition pending.  The engine can report
    # SURVIVED_ROUND only in end_of_round after it already reported the final
    # ordinary transition; maturing at exactly n here would permanently miss
    # that terminal reward in one n-step target.
    if not raw.done and len(self.n_step_queue) > self.n_step: _add_target(self)
    if raw.done:
        while self.n_step_queue: _add_target(self)
    _learn(self); self._round["optimizer_updates"] += self.gradient_steps - before


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    raw = _raw(self, old_game_state, self_action, new_game_state, events, False)
    if raw is not None: _append(self, raw)


def _merge_survivor(self, state, action, events) -> None:
    if not self.n_step_queue: raise RuntimeError("survivor has no pending transition")
    raw = self.n_step_queue[-1]
    if (raw.episode_id, raw.step, raw.action) != (int(state["round"]), int(state["step"]), ACTIONS.index(action)):
        raise RuntimeError("survivor transition identity mismatch")
    added = list(events).count(e.SURVIVED_ROUND) - list(raw.events).count(e.SURVIVED_ROUND)
    if added != 1: raise RuntimeError("survivor event delta mismatch")
    old_components = raw.components
    components = official_components(tuple(events))
    components.update({name: value for name, value in old_components.items() if name in SHAPING_NAMES})
    reward = float(sum(components.values()))
    # The transition was already accounted before SURVIVED_ROUND arrived.
    for name, value in old_components.items():
        self.diagnostics["reward_component_counts"][name] -= 1
        self.diagnostics["reward_component_sums"][name] -= value
    for name, value in components.items():
        self.diagnostics["reward_component_counts"][name] += 1
        self.diagnostics["reward_component_sums"][name] += value
    self.diagnostics["event_counts"][e.SURVIVED_ROUND] += 1
    self._round["reward_sum"] += reward - raw.reward
    raw.reward = reward; raw.components = components; raw.events = tuple(events)
    raw.done = True; raw.next_state = None; raw.next_mask = None
    self.diagnostics["survivor_terminal_merges"] += 1
    while self.n_step_queue: _add_target(self)


def _save(self) -> None:
    directory = Path(os.environ["MODEL_A_CNN_CHECKPOINT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"round-{self.stage_rounds:04d}.pt"
    payload = {
        "architecture": ARCHITECTURE, "online_net": self.online_net.state_dict(),
        "target_net": self.target_net.state_dict(), "optimizer": self.optimizer.state_dict(),
        "protocol_sha256": self.cnn_protocol_sha256, "stage": self.cnn_stage,
        "replica": self.cnn_replica, "stage_rounds": self.stage_rounds,
        "completed_total_rounds": self.completed_rounds, "training_steps": self.training_steps,
        "gradient_steps": self.gradient_steps, "epsilon": self.epsilon,
        "parent_checkpoint": None if self.parent_checkpoint_path is None else str(self.parent_checkpoint_path),
        "parent_sha256": self.parent_checkpoint_sha256,
        "n_step": self.n_step, "device_used": str(self.device),
        "training_diagnostics": copy.deepcopy(self.diagnostics),
    }
    temporary = path.with_suffix(".pt.tmp"); torch.save(payload, temporary); temporary.replace(path)


def end_of_round(self, last_game_state, last_action, events):
    identity = bool(self.n_step_queue) and (
        self.n_step_queue[-1].episode_id == int(last_game_state["round"])
        and self.n_step_queue[-1].step == int(last_game_state["step"])
        and self.n_step_queue[-1].action == ACTIONS.index(last_action)
    )
    if identity:
        _merge_survivor(self, last_game_state, last_action, events)
    else:
        raw = _raw(self, last_game_state, last_action, None, events, True)
        if raw is None: raise RuntimeError("terminal transition missing")
        self.diagnostics["dead_terminal_appends"] += 1; _append(self, raw)
    if self.n_step_queue: raise RuntimeError("n-step queue crossed episode boundary")
    self.stage_rounds += 1; self.completed_rounds += 1
    self.diagnostics["per_round"].append(self._round); self._round = None
    if self.stage_rounds % self.checkpoint_every == 0: _save(self)
