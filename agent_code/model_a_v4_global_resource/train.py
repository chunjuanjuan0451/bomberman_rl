"""Residual-only one-step Double-DQN training and evaluation diagnostics."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import events as e

from agent_code.model_a_dqn.features import ACTIONS, danger_time_map, legal_action_mask
from agent_code.model_a_dqn.replay_buffer import ReplayBuffer, Transition
from .config import architecture_name
from .features import combined_features
from .network import torch


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


def _reward(events) -> float:
    return sum(REWARD_BY_EVENT.get(event, 0.0) for event in events)


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


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def setup_training(self):
    if self.global_run_mode == "evaluate":
        self.evaluation_diagnostic_path = Path(os.environ["MODEL_A_GLOBAL_DIAGNOSTIC_PATH"])
        self.evaluation_rounds = []
        self.evaluation_updates = 0
        return
    hyper = self.global_protocol["learning_contract"]
    self.gamma = float(hyper["gamma"])
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
    self.replay_rng = np.random.default_rng(np.random.SeedSequence([self.agent_seed, 0x140]))
    self.training_steps = self.gradient_steps = 0
    self.training_diagnostics = {
        "raw_transitions": 0,
        "gradient_updates": 0,
        "sampled_transitions": 0,
        "per_round": [],
        "event_counts": {name: 0 for name in REWARD_BY_EVENT},
    }
    self._round_diagnostic = None


def _batch(transitions, device):
    local = np.stack([item.state[0] for item in transitions])
    global_features = np.stack([item.state[1] for item in transitions])
    resource = np.stack([item.state[2] for item in transitions])
    actions = np.asarray([item.action for item in transitions], dtype=np.int64)
    rewards = np.asarray([item.reward for item in transitions], dtype=np.float32)
    dones = np.asarray([item.done for item in transitions], dtype=np.float32)
    next_local = np.stack([item.next_state[0] if item.next_state is not None else np.zeros_like(item.state[0]) for item in transitions])
    next_global = np.stack([item.next_state[1] if item.next_state is not None else np.zeros_like(item.state[1]) for item in transitions])
    next_resource = np.stack([item.next_state[2] if item.next_state is not None else np.zeros_like(item.state[2]) for item in transitions])
    next_masks = np.stack([item.next_mask if item.next_state is not None else np.zeros(len(ACTIONS), dtype=bool) for item in transitions])
    return tuple(torch.as_tensor(value, device=device) for value in (
        local, global_features, resource, actions, rewards, dones,
        next_local, next_global, next_resource, next_masks,
    ))


def _learn(self) -> None:
    if len(self.replay_buffer) < self.warmup_transitions or self.training_steps % self.update_every:
        return
    transitions = self.replay_buffer.sample(self.batch_size, self.replay_rng)
    values = _batch(transitions, self.device)
    local, glob, resource, actions, rewards, dones, next_local, next_glob, next_resource, next_masks = values
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
        targets = rewards.float() + self.gamma * (1.0 - dones.float()) * target_next
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


def _new_round(state) -> dict:
    return {
        "round": int(state["round"]), "transitions": 0, "optimizer_updates": 0,
        "first_coin_step": None, "coins": 0,
        "actions": {action: 0 for action in ACTIONS},
    }


def _record_training(self, old, action, new, events, done) -> None:
    state = combined_features(old, self.global_arm)
    if state is None or action not in ACTIONS:
        return
    if self._round_diagnostic is None:
        self._round_diagnostic = _new_round(old)
    before_updates = self.gradient_steps
    next_state = None if done else combined_features(new, self.global_arm)
    next_mask = None if next_state is None else legal_action_mask(new)
    reward = _reward(events) + (_shaping(old, new) if not done else 0.0)
    self.replay_buffer.add(Transition(state, ACTIONS.index(action), reward, next_state, done, next_mask))
    self.training_steps += 1
    self.training_diagnostics["raw_transitions"] += 1
    self._round_diagnostic["transitions"] += 1
    self._round_diagnostic["actions"][action] += 1
    if e.COIN_COLLECTED in events:
        self._round_diagnostic["coins"] += events.count(e.COIN_COLLECTED)
        if self._round_diagnostic["first_coin_step"] is None:
            self._round_diagnostic["first_coin_step"] = int(old["step"])
    for event in events:
        if event in self.training_diagnostics["event_counts"]:
            self.training_diagnostics["event_counts"][event] += 1
    _learn(self)
    self._round_diagnostic["optimizer_updates"] += self.gradient_steps - before_updates


def _record_evaluation(self, state, action, events) -> None:
    if not self.evaluation_rounds or self.evaluation_rounds[-1]["round"] != int(state["round"]):
        self.evaluation_rounds.append(_new_round(state))
    item = self.evaluation_rounds[-1]
    item["transitions"] += 1
    item["actions"][action] += 1
    if e.COIN_COLLECTED in events:
        item["coins"] += events.count(e.COIN_COLLECTED)
        if item["first_coin_step"] is None:
            item["first_coin_step"] = int(state["step"])


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if self.global_run_mode == "evaluate":
        _record_evaluation(self, old_game_state, self_action, events)
        return
    _record_training(self, old_game_state, self_action, new_game_state, events, False)


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
        "architecture": architecture_name(),
        "protocol_sha256": self.global_protocol_sha256,
        "parent_sha256": self.global_protocol["frozen_v4"]["sha256"],
        "arm": self.global_arm,
        "replica": self.global_replica,
        "single_training_variable": self.global_protocol["single_training_variable"],
        "training_diagnostics": self.training_diagnostics,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def _write_evaluation(self) -> None:
    latencies = [item["first_coin_step"] for item in self.evaluation_rounds if item["first_coin_step"] is not None]
    payload = {
        "schema_version": 1,
        "kind": "model-a-v4-global-resource-stage1-evaluation-diagnostic",
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
    _atomic_json(self.evaluation_diagnostic_path, payload)


def end_of_round(self, last_game_state, last_action, events):
    if self.global_run_mode == "evaluate":
        _record_evaluation(self, last_game_state, last_action, events)
        _write_evaluation(self)
        return
    _record_training(self, last_game_state, last_action, None, events, True)
    self.completed_rounds += 1
    assert self._round_diagnostic is not None
    self.training_diagnostics["per_round"].append(self._round_diagnostic)
    self._round_diagnostic = None
    if self.completed_rounds % self.checkpoint_every_rounds == 0:
        _save(self)
