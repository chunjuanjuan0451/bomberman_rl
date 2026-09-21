"""Exact-v4 policy with isolated Task1 epsilon-floor gate checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from agent_code.model_a_dqn import callbacks as v4_callbacks
from agent_code.model_a_dqn.callbacks import ACTIONS, MODEL_ARCHITECTURE
from agent_code.model_a_dqn.network import DuelingDQN, torch

from .config import checkpoint_path, load_protocol


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "online_net" not in payload:
        raise RuntimeError(f"invalid epsilon-floor checkpoint: {path}")
    return payload


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for model_a_v4_epsilon_gate")
    return value


def setup(self):
    if torch is None:
        raise RuntimeError("model_a_v4_epsilon_gate requires PyTorch")
    torch.set_num_threads(1)
    protocol_path = Path(_required_environment("MODEL_A_V4EF_PROTOCOL_PATH"))
    self.protocol, self.protocol_sha256 = load_protocol(protocol_path)
    self.replica = _required_environment("MODEL_A_V4EF_REPLICA")
    expected_output = checkpoint_path(self.protocol, self.replica)
    supplied_output = Path(_required_environment("MODEL_A_V4EF_CHECKPOINT_PATH")).resolve()
    if supplied_output != expected_output:
        raise RuntimeError("checkpoint path does not match epsilon-floor protocol identity")
    self.checkpoint_path = supplied_output
    supplied_seed = int(_required_environment("MODEL_A_V4EF_SEED"))
    expected_seed = int(self.protocol["replicas"][self.replica]["agent_seed"])
    if self.train and supplied_seed != expected_seed:
        raise RuntimeError("agent seed does not match epsilon-floor protocol")
    self.agent_seed = supplied_seed
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    torch.manual_seed(self.agent_seed)
    self.online_net = DuelingDQN().to(self.device)
    self.resume_checkpoint = None
    self.feature_schema_migrated = False
    self.completed_rounds = 0
    self.action_counts = {action: 0 for action in ACTIONS}
    if self.train:
        if self.checkpoint_path.exists():
            raise RuntimeError(f"refusing to overwrite epsilon-floor checkpoint: {self.checkpoint_path}")
    else:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"epsilon-floor checkpoint is missing: {self.checkpoint_path}")
        payload = _torch_load(self.checkpoint_path)
        expected = {
            "architecture": MODEL_ARCHITECTURE,
            "protocol_sha256": self.protocol_sha256,
            "replica": self.replica,
            "completed_rounds": 400,
            "epsilon_floor": 0.15,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"evaluation checkpoint {key} mismatch")
        self.online_net.load_state_dict(payload["online_net"], strict=True)
        self.completed_rounds = int(payload["completed_rounds"])


def act(self, game_state: dict) -> str:
    action = v4_callbacks.act(self, game_state)
    if self.train:
        self.action_counts[action] += 1
    return action
