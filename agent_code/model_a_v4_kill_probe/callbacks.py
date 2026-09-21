"""Greedy/epsilon collector using a byte-frozen source-r2 network."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE, _load_network_state
from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import DuelingDQN, torch

from .config import CASES, ROOT, load_protocol, sha256_file, trace_path


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for model_a_v4_kill_probe")
    return value


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "online_net" not in payload:
        raise RuntimeError(f"invalid source-r2 checkpoint: {path}")
    return payload


def setup(self):
    if torch is None:
        raise RuntimeError("model_a_v4_kill_probe requires PyTorch")
    if not self.train:
        raise RuntimeError("kill-probe collector must be invoked as the sole training-mode agent")
    torch.set_num_threads(1)
    protocol_path = Path(_required("MODEL_A_KILL_PROBE_PROTOCOL_PATH")).resolve()
    self.protocol, self.protocol_sha256 = load_protocol(protocol_path)
    self.case_label = _required("MODEL_A_KILL_PROBE_CASE")
    if self.case_label not in CASES:
        raise RuntimeError("invalid kill-probe collection case")
    expected = self.protocol["collection"]["cases"][self.case_label]
    self.agent_seed = int(_required("MODEL_A_KILL_PROBE_SEED"))
    if self.agent_seed != int(expected["agent_seed"]):
        raise RuntimeError("kill-probe agent seed mismatch")
    self.rng = np.random.default_rng(self.agent_seed)
    self.collection_epsilon = float(self.protocol["collection"]["epsilon"])
    self.device = torch.device("cpu")
    torch.manual_seed(self.agent_seed)
    self.online_net = DuelingDQN().to(self.device)
    parent = Path(_required("MODEL_A_KILL_PROBE_PARENT_PATH")).resolve()
    bound = (ROOT / self.protocol["source_parent"]["path"]).resolve()
    if parent != bound or sha256_file(parent) != self.protocol["source_parent"]["sha256"]:
        raise RuntimeError("kill-probe source-r2 binding mismatch")
    payload = _torch_load(parent)
    lineage = self.protocol["source_parent"]["lineage"]
    for key, value in {
        "architecture": MODEL_ARCHITECTURE,
        "protocol_sha256": lineage["protocol_sha256"],
        "arm": "curriculum",
        "replica": "r2",
        "stage_id": "task2",
    }.items():
        if payload.get(key) != value:
            raise RuntimeError(f"kill-probe source-r2 {key} mismatch")
    if _load_network_state(self.online_net, payload["online_net"]):
        raise RuntimeError("kill-probe source-r2 unexpectedly required schema migration")
    self.online_net.eval()
    for parameter in self.online_net.parameters():
        parameter.requires_grad = False
    self.parent_sha256 = sha256_file(parent)
    self.trace_path = trace_path(self.protocol, self.case_label)


def _q_values(self, game_state: dict):
    local, global_features = state_to_features(game_state)
    with torch.no_grad():
        local_tensor = torch.as_tensor(local[None], dtype=torch.float32, device=self.device)
        global_tensor = torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device)
        return self.online_net(local_tensor, global_tensor)[0].cpu().numpy()


def act(self, game_state: dict) -> str:
    legal_indices = np.flatnonzero(legal_action_mask(game_state))
    if legal_indices.size == 0:
        return "WAIT"
    if self.rng.random() < self.collection_epsilon:
        return ACTIONS[int(self.rng.choice(legal_indices))]
    q_values = _q_values(self, game_state)
    best = q_values[legal_indices].max()
    choices = legal_indices[np.isclose(q_values[legal_indices], best)]
    return ACTIONS[int(self.rng.choice(choices))]
