"""Exact-v4 callbacks for direct Task4C training."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from agent_code.model_a_dqn import callbacks as v4_callbacks
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.network import DuelingDQN, torch

from .config import (
    REPLICAS, ROOT, SNAPSHOT_ROUNDS, final_checkpoint_path, load_protocol,
    selected_checkpoint_path, sha256_file, snapshot_path,
)


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "online_net" not in payload:
        raise RuntimeError(f"invalid Task4C checkpoint: {path}")
    return payload


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for model_a_v4_task4c")
    return value


def _validate_candidate(self, payload: dict, path: Path) -> None:
    expected = {
        "architecture": MODEL_ARCHITECTURE,
        "protocol_sha256": self.protocol_sha256,
        "replica": self.replica,
        "stage_id": "task4c_three_rule",
        "parent_sha256": self.protocol["source_parent"]["sha256"],
        "training_variable": "task4c_episode_distribution_only",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Task4C checkpoint {key} mismatch: {path}")


def setup(self):
    if torch is None:
        raise RuntimeError("model_a_v4_task4c requires PyTorch")
    torch.set_num_threads(1)
    protocol_path = Path(_required("MODEL_A_V4T4C_PROTOCOL_PATH"))
    self.protocol, self.protocol_sha256 = load_protocol(protocol_path)
    self.replica = _required("MODEL_A_V4T4C_REPLICA")
    if self.replica not in REPLICAS:
        raise RuntimeError("invalid Task4C replica")
    supplied_seed = int(_required("MODEL_A_V4T4C_SEED"))
    expected_seed = int(self.protocol["training"]["seeds"][self.replica]["agent_seed"])
    if self.train and supplied_seed != expected_seed:
        raise RuntimeError("Task4C agent seed mismatch")
    self.agent_seed = supplied_seed
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    torch.manual_seed(self.agent_seed)
    self.online_net = DuelingDQN().to(self.device)
    self.resume_checkpoint = None
    self.feature_schema_migrated = False
    supplied_checkpoint = Path(_required("MODEL_A_V4T4C_CHECKPOINT_PATH")).resolve()

    if self.train:
        expected_output = final_checkpoint_path(self.protocol, self.replica)
        if supplied_checkpoint != expected_output or supplied_checkpoint.exists():
            raise RuntimeError("Task4C output path mismatch or overwrite attempt")
        parent = Path(_required("MODEL_A_V4T4C_PARENT_PATH")).resolve()
        expected_parent = (ROOT / self.protocol["source_parent"]["path"]).resolve()
        if parent != expected_parent or sha256_file(parent) != self.protocol["source_parent"]["sha256"]:
            raise RuntimeError("Task4C source-r2 parent mismatch")
        payload = _torch_load(parent)
        lineage = self.protocol["source_parent"]["lineage"]
        expected_lineage = {
            "architecture": MODEL_ARCHITECTURE,
            "protocol_sha256": lineage["protocol_sha256"],
            "arm": "curriculum",
            "replica": "r2",
            "stage_id": "task2",
        }
        for key, value in expected_lineage.items():
            if payload.get(key) != value:
                raise RuntimeError(f"Task4C source parent {key} mismatch")
        self.online_net.load_state_dict(payload["online_net"], strict=True)
        self.resume_checkpoint = payload
        self.completed_rounds = int(payload.get("completed_rounds", 0))
        self.stage_start_rounds = self.completed_rounds
        self.parent_sha256 = sha256_file(parent)
        self.checkpoint_path = supplied_checkpoint
    else:
        if not supplied_checkpoint.is_file():
            raise FileNotFoundError(f"Task4C evaluation checkpoint missing: {supplied_checkpoint}")
        payload = _torch_load(supplied_checkpoint)
        _validate_candidate(self, payload, supplied_checkpoint)
        allowed = {
            final_checkpoint_path(self.protocol, self.replica),
            selected_checkpoint_path(self.protocol),
            *{snapshot_path(self.protocol, self.replica, value) for value in SNAPSHOT_ROUNDS},
        }
        if supplied_checkpoint not in allowed:
            raise RuntimeError("Task4C evaluation checkpoint is outside the immutable artifact set")
        self.online_net.load_state_dict(payload["online_net"], strict=True)
        self.completed_rounds = int(payload.get("completed_rounds", 0))


def act(self, game_state: dict) -> str:
    return v4_callbacks.act(self, game_state)
