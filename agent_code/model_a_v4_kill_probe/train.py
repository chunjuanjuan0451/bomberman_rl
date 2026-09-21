"""Trajectory recording callbacks; deliberately performs no policy learning."""

from __future__ import annotations

import json

import numpy as np
import events as e

from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import torch


def setup_training(self):
    if self.trace_path.exists():
        raise RuntimeError(f"refusing to overwrite kill-probe trace: {self.trace_path}")
    self._trace = {
        "latent": [], "action": [], "episode": [], "step": [], "legal_mask": [],
        "q_values": [], "kill_event": [], "self_event": [], "got_killed_event": [],
    }
    self._seen_transition_keys = set()
    self._episode_index = 0
    self._expected_rounds = int(self.protocol["collection"]["rounds_per_case"])


def _record(self, game_state, action: str, events) -> None:
    if game_state is None or action not in ACTIONS:
        return
    key = (int(game_state["round"]), int(game_state["step"]), action)
    if key in self._seen_transition_keys:
        return
    self._seen_transition_keys.add(key)
    local, global_features = state_to_features(game_state)
    with torch.no_grad():
        local_tensor = torch.as_tensor(local[None], dtype=torch.float32, device=self.device)
        global_tensor = torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device)
        combined = torch.cat((local_tensor.flatten(start_dim=1), global_tensor), dim=1)
        latent = self.online_net.encoder(combined)[0].cpu().numpy().astype(np.float32)
        q_values = self.online_net(local_tensor, global_tensor)[0].cpu().numpy().astype(np.float32)
    self._trace["latent"].append(latent)
    self._trace["action"].append(ACTIONS.index(action))
    self._trace["episode"].append(self._episode_index)
    self._trace["step"].append(int(game_state["step"]))
    self._trace["legal_mask"].append(legal_action_mask(game_state).astype(np.bool_))
    self._trace["q_values"].append(q_values)
    self._trace["kill_event"].append(e.KILLED_OPPONENT in events)
    self._trace["self_event"].append(e.KILLED_SELF in events)
    self._trace["got_killed_event"].append(e.GOT_KILLED in events)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _record(self, old_game_state, self_action, events)


def _save(self) -> None:
    if not self._trace["latent"]:
        raise RuntimeError("kill-probe trace is empty")
    self.trace_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = self.trace_path.with_suffix(self.trace_path.suffix + ".tmp")
    arrays = {
        "latent": np.stack(self._trace["latent"]).astype(np.float32),
        "action": np.asarray(self._trace["action"], dtype=np.int8),
        "episode": np.asarray(self._trace["episode"], dtype=np.int16),
        "step": np.asarray(self._trace["step"], dtype=np.int16),
        "legal_mask": np.stack(self._trace["legal_mask"]).astype(np.bool_),
        "q_values": np.stack(self._trace["q_values"]).astype(np.float32),
        "kill_event": np.asarray(self._trace["kill_event"], dtype=np.bool_),
        "self_event": np.asarray(self._trace["self_event"], dtype=np.bool_),
        "got_killed_event": np.asarray(self._trace["got_killed_event"], dtype=np.bool_),
        "metadata": np.asarray(json.dumps({
            "schema_version": 1,
            "kind": "model-a-v4-kill-probe-trace",
            "protocol_sha256": self.protocol_sha256,
            "case": self.case_label,
            "parent_sha256": self.parent_sha256,
            "rounds": self._expected_rounds,
            "collection_epsilon": self.collection_epsilon,
        }, sort_keys=True)),
    }
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(self.trace_path)


def end_of_round(self, last_game_state, last_action, events):
    _record(self, last_game_state, last_action, events)
    self._episode_index += 1
    if self._episode_index == self._expected_rounds:
        _save(self)
    elif self._episode_index > self._expected_rounds:
        raise RuntimeError("kill-probe collector exceeded its registered round budget")
