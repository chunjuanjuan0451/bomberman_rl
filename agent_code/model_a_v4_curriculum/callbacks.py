"""Exact-v4 callbacks with isolated, lineage-checked experiment checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import DuelingDQN, torch

from .config import checkpoint_path, load_protocol, parent_stage, sha256_file


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "online_net" not in payload:
        raise RuntimeError(f"invalid clean-v4 checkpoint: {path}")
    return payload


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for model_a_v4_curriculum")
    return value


def _validate_parent(self, payload: dict, expected_parent: Path, expected_stage: str) -> None:
    expected = {
        "architecture": MODEL_ARCHITECTURE,
        "protocol_sha256": self.protocol_sha256,
        "arm": self.arm,
        "replica": self.replica,
        "stage_id": expected_stage,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"parent checkpoint {key} mismatch: {expected_parent}")
    if payload.get("checkpoint_sha256") is not None:
        raise RuntimeError("checkpoint must not contain a self-referential hash")


def setup(self):
    if torch is None:
        raise RuntimeError("model_a_v4_curriculum requires PyTorch")
    torch.set_num_threads(1)
    protocol_path = Path(_required_environment("MODEL_A_V4C_PROTOCOL_PATH"))
    self.protocol, self.protocol_sha256 = load_protocol(protocol_path)
    self.arm = _required_environment("MODEL_A_V4C_ARM")
    self.replica = _required_environment("MODEL_A_V4C_REPLICA")
    self.stage_id = _required_environment("MODEL_A_V4C_STAGE")
    expected_output = checkpoint_path(self.protocol, self.arm, self.replica, self.stage_id)
    supplied_output = Path(_required_environment("MODEL_A_V4C_CHECKPOINT_PATH")).resolve()
    if supplied_output != expected_output:
        raise RuntimeError("checkpoint path does not match protocol identity")
    self.checkpoint_path = supplied_output
    stage = self.protocol["replicas"][self.replica]["stages"][self.stage_id]
    supplied_seed = int(_required_environment("MODEL_A_V4C_SEED"))
    if self.train and supplied_seed != int(stage["agent_seed"]):
        raise RuntimeError("agent seed does not match protocol")
    self.agent_seed = supplied_seed
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    torch.manual_seed(self.agent_seed)
    self.online_net = DuelingDQN().to(self.device)
    self.resume_checkpoint = None
    self.feature_schema_migrated = False
    self.parent_sha256 = None
    previous = parent_stage(self.stage_id)
    self.stage_start_rounds = 0

    if self.train:
        if self.checkpoint_path.exists():
            raise RuntimeError(f"refusing to overwrite clean-v4 checkpoint: {self.checkpoint_path}")
        if previous is None:
            if os.environ.get("MODEL_A_V4C_PARENT_PATH"):
                raise RuntimeError("Task1 must start from random initialization")
            self.completed_rounds = 0
        else:
            expected_parent = checkpoint_path(self.protocol, self.arm, self.replica, previous)
            supplied_parent = Path(_required_environment("MODEL_A_V4C_PARENT_PATH")).resolve()
            if supplied_parent != expected_parent or not expected_parent.is_file():
                raise RuntimeError("missing or incorrect parent checkpoint")
            payload = _torch_load(expected_parent)
            _validate_parent(self, payload, expected_parent, previous)
            self.online_net.load_state_dict(payload["online_net"], strict=True)
            self.resume_checkpoint = payload
            self.completed_rounds = int(payload.get("completed_rounds", 0))
            self.stage_start_rounds = self.completed_rounds
            self.parent_sha256 = sha256_file(expected_parent)
    else:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"clean-v4 checkpoint is missing: {self.checkpoint_path}")
        payload = _torch_load(self.checkpoint_path)
        expected = {
            "architecture": MODEL_ARCHITECTURE,
            "protocol_sha256": self.protocol_sha256,
            "arm": self.arm,
            "replica": self.replica,
            "stage_id": self.stage_id,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"evaluation checkpoint {key} mismatch")
        self.online_net.load_state_dict(payload["online_net"], strict=True)
        self.completed_rounds = int(payload.get("completed_rounds", 0))


def act(self, game_state: dict) -> str:
    """Byte-for-byte policy logic of final v4, with an isolated RNG."""
    legal_indices = np.flatnonzero(legal_action_mask(game_state))
    if legal_indices.size == 0:
        return "WAIT"
    epsilon = getattr(self, "epsilon", 0.0)
    if self.train and self.rng.random() < epsilon:
        return ACTIONS[int(self.rng.choice(legal_indices))]
    features = state_to_features(game_state)
    assert features is not None
    local, global_features = features
    with torch.no_grad():
        q_values = self.online_net(
            torch.as_tensor(local[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
        )[0].cpu().numpy()
    best_value = q_values[legal_indices].max()
    best_indices = legal_indices[np.isclose(q_values[legal_indices], best_value)]
    return ACTIONS[int(self.rng.choice(best_indices))]
