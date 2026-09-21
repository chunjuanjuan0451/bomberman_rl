"""Residual-only uniform n=8 Double-DQN training for the rescue pilot."""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import events as e

from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask
from agent_code.model_a_v4_global_resource.train import (
    REWARD_BY_EVENT,
    _new_round,
    _record_evaluation,
    _reward,
    _shaping,
)
from .config import architecture_name
from .features import combined_features
from .network import torch
from .replay import ReplayBuffer, Transition


@dataclass
class RawTransition:
    state: object
    action: int
    reward: float
    state_shaping: float
    next_state: object
    done: bool
    next_mask: object
    episode_id: int
    start_step: int
    events: tuple[str, ...]


def aggregate_n_step(queue, n_step: int, gamma: float) -> Transition:
    window = list(queue)[: int(n_step)]
    if not window:
        raise ValueError("cannot aggregate an empty n-step queue")
    first = window[0]
    if any(raw.episode_id != first.episode_id for raw in window):
        raise ValueError("n-step return crossed an episode boundary")
    reward = 0.0
    final = first
    used = 0
    for index, raw in enumerate(window):
        reward += (float(gamma) ** index) * float(raw.reward)
        final = raw
        used = index + 1
        if raw.done:
            break
    return Transition(
        state=first.state,
        action=first.action,
        reward=reward,
        next_state=final.next_state,
        done=final.done,
        next_mask=final.next_mask,
        bootstrap_discount=float(gamma) ** used,
        return_steps=used,
    )


def setup_training(self):
    if self.global_run_mode == "evaluate":
        self.evaluation_diagnostic_path = Path(os.environ["MODEL_A_GLOBAL_DIAGNOSTIC_PATH"])
        self.evaluation_rounds = []
        self.evaluation_updates = 0
        return
    hyper = self.global_protocol["learning_contract"]
    self.gamma = float(hyper["gamma"])
    self.n_step = int(hyper["n_step"])
    self.batch_size = int(hyper["batch_size"])
    self.warmup_transitions = int(hyper["warmup_transitions"])
    self.update_every = int(hyper["update_every"])
    self.target_update_every = int(hyper["target_update_every"])
    self.checkpoint_every_rounds = int(hyper["checkpoint_every_rounds"])
    self.gradient_clip = float(hyper["gradient_clip"])
    self.epsilon = float(hyper["epsilon"])
    self.target_net = copy.deepcopy(self.online_net).to(self.device)
    self.target_net.freeze_base()
    parameters = [*self.online_net.resource_encoder.parameters(), *self.online_net.resource_head.parameters()]
    self.optimizer = torch.optim.Adam(parameters, lr=float(hyper["learning_rate"]))
    self.replay_buffer = ReplayBuffer(int(hyper["replay_capacity"]))
    self.replay_rng = np.random.default_rng(np.random.SeedSequence([self.agent_seed, 0x141]))
    self.n_step_queue = deque()
    self.training_steps = self.gradient_steps = 0
    self.training_diagnostics = {
        "raw_transitions": 0,
        "matured_targets": 0,
        "gradient_updates": 0,
        "sampled_transitions": 0,
        "return_steps_histogram": {str(i): 0 for i in range(1, self.n_step + 1)},
        "survivor_terminal_merges": 0,
        "dead_terminal_appends": 0,
        "per_round": [],
        "event_counts": {name: 0 for name in REWARD_BY_EVENT},
    }
    self._round_diagnostic = None


def _batch(transitions, device):
    local = np.stack([item.state[0] for item in transitions])
    glob = np.stack([item.state[1] for item in transitions])
    resource = np.stack([item.state[2] for item in transitions])
    actions = np.asarray([item.action for item in transitions], dtype=np.int64)
    rewards = np.asarray([item.reward for item in transitions], dtype=np.float32)
    dones = np.asarray([item.done for item in transitions], dtype=np.float32)
    discounts = np.asarray([item.bootstrap_discount for item in transitions], dtype=np.float32)
    next_local = np.stack([
        item.next_state[0] if item.next_state is not None else np.zeros_like(item.state[0])
        for item in transitions
    ])
    next_glob = np.stack([
        item.next_state[1] if item.next_state is not None else np.zeros_like(item.state[1])
        for item in transitions
    ])
    next_resource = np.stack([
        item.next_state[2] if item.next_state is not None else np.zeros_like(item.state[2])
        for item in transitions
    ])
    next_masks = np.stack([
        item.next_mask if item.next_state is not None else np.zeros(len(ACTIONS), dtype=bool)
        for item in transitions
    ])
    values = (
        local, glob, resource, actions, rewards, dones, discounts,
        next_local, next_glob, next_resource, next_masks,
    )
    return tuple(torch.as_tensor(value, device=device) for value in values)


def _learn(self) -> None:
    if len(self.replay_buffer) < self.warmup_transitions or self.training_steps % self.update_every:
        return
    transitions = self.replay_buffer.sample(self.batch_size, self.replay_rng)
    values = _batch(transitions, self.device)
    (local, glob, resource, actions, rewards, dones, discounts,
     next_local, next_glob, next_resource, next_masks) = values
    q_values = self.online_net(local.float(), glob.float(), resource.float()).gather(
        1, actions.long().unsqueeze(1),
    ).squeeze(1)
    with torch.no_grad():
        online_next = self.online_net(next_local.float(), next_glob.float(), next_resource.float())
        online_next = online_next.masked_fill(~next_masks.bool(), -1e9)
        next_actions = online_next.argmax(dim=1)
        target_next = self.target_net(next_local.float(), next_glob.float(), next_resource.float()).gather(
            1, next_actions.unsqueeze(1),
        ).squeeze(1)
        targets = rewards.float() + discounts.float() * (1.0 - dones.float()) * target_next
    loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [*self.online_net.resource_encoder.parameters(), *self.online_net.resource_head.parameters()],
        self.gradient_clip,
    )
    self.optimizer.step()
    self.gradient_steps += 1
    self.training_diagnostics["gradient_updates"] = self.gradient_steps
    self.training_diagnostics["sampled_transitions"] += self.batch_size
    if self.gradient_steps % self.target_update_every == 0:
        self.target_net.resource_encoder.load_state_dict(self.online_net.resource_encoder.state_dict())
        self.target_net.resource_head.load_state_dict(self.online_net.resource_head.state_dict())


def _raw(self, old, action: str, new, events, done: bool) -> RawTransition | None:
    state = combined_features(old, self.global_arm)
    if state is None or action not in ACTIONS:
        return None
    shaping = 0.0 if done else float(_shaping(old, new))
    next_state = None if done else combined_features(new, self.global_arm)
    next_mask = None if next_state is None else legal_action_mask(new)
    return RawTransition(
        state=state,
        action=ACTIONS.index(action),
        reward=float(_reward(events)) + shaping,
        state_shaping=shaping,
        next_state=next_state,
        done=done,
        next_mask=next_mask,
        episode_id=int(old["round"]),
        start_step=int(old["step"]),
        events=tuple(events),
    )


def _add_target(self) -> None:
    transition = aggregate_n_step(self.n_step_queue, self.n_step, self.gamma)
    self.replay_buffer.add(transition)
    diagnostics = self.training_diagnostics
    diagnostics["matured_targets"] += 1
    diagnostics["return_steps_histogram"][str(transition.return_steps)] += 1
    self.n_step_queue.popleft()


def _append_training_raw(self, raw: RawTransition) -> None:
    if self._round_diagnostic is None:
        self._round_diagnostic = _new_round({"round": raw.episode_id})
    before_updates = self.gradient_steps
    self.n_step_queue.append(raw)
    diagnostics = self.training_diagnostics
    diagnostics["raw_transitions"] += 1
    self.training_steps += 1
    self._round_diagnostic["transitions"] += 1
    self._round_diagnostic["actions"][ACTIONS[raw.action]] += 1
    if e.COIN_COLLECTED in raw.events:
        self._round_diagnostic["coins"] += raw.events.count(e.COIN_COLLECTED)
        if self._round_diagnostic["first_coin_step"] is None:
            self._round_diagnostic["first_coin_step"] = raw.start_step
    for event in raw.events:
        if event in diagnostics["event_counts"]:
            diagnostics["event_counts"][event] += 1
    if not raw.done and len(self.n_step_queue) >= self.n_step:
        _add_target(self)
    if raw.done:
        while self.n_step_queue:
            _add_target(self)
    _learn(self)
    self._round_diagnostic["optimizer_updates"] += self.gradient_steps - before_updates


def _merge_training_survivor(self, last_state, last_action: str, events) -> None:
    if not self.n_step_queue:
        raise RuntimeError("n8 survivor has no pending transition")
    raw = self.n_step_queue[-1]
    expected = (int(last_state["round"]), int(last_state["step"]), ACTIONS.index(last_action))
    observed = (raw.episode_id, raw.start_step, raw.action)
    if expected != observed:
        raise RuntimeError(f"n8 survivor identity mismatch: {expected} != {observed}")
    final_events = tuple(events)
    old_counts = {name: raw.events.count(name) for name in set(raw.events) | set(final_events)}
    new_counts = {name: final_events.count(name) for name in old_counts}
    delta = {name: new_counts[name] - old_counts[name] for name in old_counts if new_counts[name] != old_counts[name]}
    if delta != {e.SURVIVED_ROUND: 1}:
        raise RuntimeError(f"n8 survivor event delta changed: {delta}")
    raw.events = final_events
    raw.reward = float(_reward(final_events)) + raw.state_shaping
    raw.done = True
    raw.next_state = None
    raw.next_mask = None
    self.training_diagnostics["event_counts"][e.SURVIVED_ROUND] += 1
    self.training_diagnostics["survivor_terminal_merges"] += 1
    while self.n_step_queue:
        _add_target(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if self.global_run_mode == "evaluate":
        _record_evaluation(self, old_game_state, self_action, events)
        return
    raw = _raw(self, old_game_state, self_action, new_game_state, events, False)
    if raw is not None:
        _append_training_raw(self, raw)


def _save(self) -> None:
    endpoint = Path(os.environ["MODEL_A_GLOBAL_CHECKPOINT_PATH"])
    output = endpoint.with_name(f"round-{self.completed_rounds:04d}.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "online_net": self.online_net.state_dict(),
        "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "completed_rounds": self.completed_rounds,
        "training_steps": self.training_steps,
        "gradient_steps": self.gradient_steps,
        "epsilon": self.epsilon,
        "n_step": self.n_step,
        "architecture": architecture_name(),
        "protocol_sha256": self.global_protocol_sha256,
        "parent_sha256": self.global_protocol["frozen_v4"]["sha256"],
        "arm": self.global_arm,
        "replica": self.global_replica,
        "single_training_variable": self.global_protocol["single_training_variable"],
        "training_diagnostics": copy.deepcopy(self.training_diagnostics),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def _write_evaluation(self) -> None:
    latencies = [
        item["first_coin_step"]
        for item in self.evaluation_rounds
        if item["first_coin_step"] is not None
    ]
    payload = {
        "schema_version": 1,
        "kind": "model-a-v4-global-resource-n8-pilot-evaluation-diagnostic",
        "protocol_sha256": self.global_protocol_sha256,
        "arm": self.global_arm,
        "replica": self.global_replica,
        "policy_updates": self.evaluation_updates,
        "rounds": self.evaluation_rounds,
        "first_coin": {
            "observed_rounds": len(latencies),
            "missing_rounds": len(self.evaluation_rounds) - len(latencies),
            "mean_step_when_observed": float(np.mean(latencies)) if latencies else None,
        },
    }
    path = self.evaluation_diagnostic_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def end_of_round(self, last_game_state, last_action, events):
    if self.global_run_mode == "evaluate":
        _record_evaluation(self, last_game_state, last_action, events)
        _write_evaluation(self)
        return
    identity_matches = bool(self.n_step_queue) and (
        self.n_step_queue[-1].episode_id == int(last_game_state["round"])
        and self.n_step_queue[-1].start_step == int(last_game_state["step"])
        and self.n_step_queue[-1].action == ACTIONS.index(last_action)
    )
    if identity_matches:
        _merge_training_survivor(self, last_game_state, last_action, events)
    else:
        raw = _raw(self, last_game_state, last_action, None, events, True)
        if raw is None:
            raise RuntimeError("n8 terminal transition missing")
        self.training_diagnostics["dead_terminal_appends"] += 1
        _append_training_raw(self, raw)
    if self.n_step_queue:
        raise RuntimeError("n8 queue crossed an episode boundary")
    self.completed_rounds += 1
    assert self._round_diagnostic is not None
    self.training_diagnostics["per_round"].append(self._round_diagnostic)
    self._round_diagnostic = None
    if self.completed_rounds % self.checkpoint_every_rounds == 0:
        _save(self)
