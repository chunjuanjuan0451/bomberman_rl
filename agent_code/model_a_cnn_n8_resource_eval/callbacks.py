"""Evaluate one immutable s165 resource A/B checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from agent_code.model_a_cnn_n8.callbacks import _load, _tensors
from agent_code.model_a_cnn_n8.features import ACTIONS, state_to_features
from agent_code.model_a_cnn_n8.network import ARCHITECTURE, FullBoardDuelingCNN, torch
from agent_code.model_a_cnn_n8_league_train.callbacks import effective_action_mask


def setup(self) -> None:
    protocol = json.loads(Path(os.environ["CNN_RESOURCE_PROTOCOL_PATH"]).read_text(encoding="utf-8"))
    if protocol.get("kind") != "model-a-cnn-n8-resource-competition-ab":
        raise RuntimeError("invalid resource A/B evaluation protocol")
    path = Path(os.environ["CNN_RESOURCE_EVAL_CHECKPOINT"]).resolve()
    expected = os.environ["CNN_RESOURCE_EVAL_CHECKPOINT_SHA256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("resource A/B evaluation checkpoint hash mismatch")
    payload = _load(path)
    if payload.get("architecture") != ARCHITECTURE or payload.get("stage") != "task4":
        raise RuntimeError("resource A/B evaluation checkpoint identity mismatch")
    self.device = torch.device("cpu")
    torch.set_num_threads(1)
    self.online_net = FullBoardDuelingCNN().to(self.device)
    self.online_net.load_state_dict(payload["online_net"], strict=True)
    self.online_net.eval()
    self.rng = np.random.default_rng(int(os.environ["CNN_RESOURCE_EVAL_AGENT_SEED"]))
    self._active_bomb = False
    self._active_bomb_position = None


def act(self, game_state: dict) -> str:
    bomb_positions = {tuple(position) for position, _timer in game_state["bombs"]}
    if self._active_bomb and self._active_bomb_position not in bomb_positions:
        self._active_bomb = False
        self._active_bomb_position = None
    indices = np.flatnonzero(effective_action_mask(game_state, self._active_bomb))
    if not indices.size:
        return "WAIT"
    features = state_to_features(game_state)
    assert features is not None
    spatial, scalars = _tensors(features, self.device)
    with torch.no_grad():
        q_values = self.online_net(spatial, scalars)[0].detach().cpu().numpy()
    best = q_values[indices].max()
    action = ACTIONS[int(self.rng.choice(indices[np.isclose(q_values[indices], best)]))]
    if action == "BOMB" and game_state["self"][2]:
        self._active_bomb = True
        self._active_bomb_position = tuple(game_state["self"][3])
    return action


__all__ = ["act", "setup"]
