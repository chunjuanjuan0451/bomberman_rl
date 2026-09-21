"""Exact-v4 training with one arm-specific conditional idle-WAIT reward."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from agent_code.model_a_dqn import callbacks as base_callbacks
from agent_code.model_a_dqn import train as base_train
from agent_code.model_a_dqn.features import ACTIONS, INF_TIME, danger_time_map, legal_action_mask
from agent_code.model_a_dqn.network import torch


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("control", "idle-wait-001")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _idle_wait(state: dict | None, action: str | None) -> bool:
    if state is None or action != "WAIT" or state["bombs"]:
        return False
    position = tuple(state["self"][3])
    if np.asarray(state["explosion_map"])[position] > 0:
        return False
    legal = legal_action_mask(state)
    return bool(np.any(legal[:4])) and int(danger_time_map(state)[position]) == INF_TIME


def _augment_checkpoint(self) -> None:
    endpoint = Path(base_callbacks.CHECKPOINT_PATH).resolve()
    payload = _load(endpoint)
    payload.update({
        "protocol_sha256": self.idle_protocol_sha256,
        "parent_sha256": self.idle_protocol["parent_checkpoint"]["sha256"],
        "arm": self.idle_arm,
        "replica": self.idle_replica,
        "experiment_completed_rounds": self.idle_completed_rounds,
        "idle_wait_penalty": self.idle_wait_penalty,
        "idle_wait_definition": self.idle_protocol["single_changed_variable"]["definition"],
        "experiment_training_diagnostics": copy.deepcopy(self.idle_diagnostics),
    })
    temporary = endpoint.with_suffix(endpoint.suffix + ".meta.tmp")
    torch.save(payload, temporary)
    temporary.replace(endpoint)


def setup_training(self) -> None:
    base_train.setup_training(self)
    protocol_path = Path(os.environ["MODEL_A_IDLE_PROTOCOL_PATH"]).resolve()
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    if (
        protocol.get("kind") != "model-a-v4-idle-wait-ab"
        or protocol.get("protocol_id") != "model-a-v4-idle-wait-ab-s146000"
    ):
        raise RuntimeError("invalid idle-WAIT A/B protocol")
    arm = os.environ.get("MODEL_A_IDLE_ARM", "")
    replica = os.environ.get("MODEL_A_IDLE_REPLICA", "")
    if arm not in ARMS or replica not in protocol["training"]["replicas"]:
        raise RuntimeError("unregistered idle-WAIT arm or replica")
    endpoint = Path(base_callbacks.CHECKPOINT_PATH).resolve()
    if not endpoint.is_file() or _sha256(endpoint) != protocol["parent_checkpoint"]["sha256"]:
        raise RuntimeError("idle-WAIT A/B must start from the exact frozen-v4 parent")
    self.idle_protocol = protocol
    self.idle_protocol_sha256 = hashlib.sha256(raw).hexdigest()
    self.idle_arm = arm
    self.idle_replica = replica
    self.idle_wait_penalty = 0.0 if arm == "control" else float(
        protocol["single_changed_variable"]["candidate_penalty"],
    )
    self.idle_completed_rounds = 0
    self.idle_diagnostics = {
        "raw_transitions": 0,
        "action_counts": {action: 0 for action in ACTIONS},
        "eligible_idle_waits": 0,
        "penalized_idle_waits": 0,
        "idle_wait_reward_contribution": 0.0,
        "per_round": [],
    }
    self._idle_round = None


def _record_diagnostic(self, state: dict, action: str, eligible: bool) -> None:
    if self._idle_round is None:
        self._idle_round = {
            "experiment_round": self.idle_completed_rounds + 1,
            "transitions": 0,
            "action_counts": {item: 0 for item in ACTIONS},
            "eligible_idle_waits": 0,
            "penalized_idle_waits": 0,
        }
    self.idle_diagnostics["raw_transitions"] += 1
    self.idle_diagnostics["action_counts"][action] += 1
    self._idle_round["transitions"] += 1
    self._idle_round["action_counts"][action] += 1
    if eligible:
        self.idle_diagnostics["eligible_idle_waits"] += 1
        self._idle_round["eligible_idle_waits"] += 1
        if self.idle_wait_penalty:
            self.idle_diagnostics["penalized_idle_waits"] += 1
            self._idle_round["penalized_idle_waits"] += 1
            self.idle_diagnostics["idle_wait_reward_contribution"] += self.idle_wait_penalty


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    eligible = _idle_wait(old_game_state, self_action)
    reward = (
        base_train.reward_from_events(events)
        + base_train._state_shaping(old_game_state, new_game_state)
        + (self.idle_wait_penalty if eligible else 0.0)
    )
    _record_diagnostic(self, old_game_state, self_action, eligible)
    base_train._record(self, old_game_state, self_action, new_game_state, reward, done=False)


def end_of_round(self, last_game_state, last_action, events):
    eligible = _idle_wait(last_game_state, last_action)
    reward = base_train.reward_from_events(events) + (self.idle_wait_penalty if eligible else 0.0)
    _record_diagnostic(self, last_game_state, last_action, eligible)
    base_train._record(self, last_game_state, last_action, None, reward, done=True)
    self.completed_rounds += 1
    self.idle_completed_rounds += 1
    self.idle_diagnostics["per_round"].append(self._idle_round)
    self._idle_round = None
    if self.idle_completed_rounds % 50 == 0:
        base_train._save_checkpoint(self)
        _augment_checkpoint(self)
