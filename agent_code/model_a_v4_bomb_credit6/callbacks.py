"""Exact-v4 callbacks for the corrected six-transition BOMB-credit A/B."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from agent_code.model_a_dqn import callbacks as v4_callbacks
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.network import DuelingDQN, torch

from .config import (
    ARMS, REPLICAS, ROOT, SNAPSHOT_ROUNDS, final_checkpoint_path, load_protocol,
    sha256_file, snapshot_path,
)


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for model_a_v4_bomb_credit6")
    return value


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "online_net" not in payload:
        raise RuntimeError(f"invalid BOMB-credit6 checkpoint: {path}")
    return payload


def _validate_candidate(self, payload: dict, path: Path) -> None:
    expected = {
        "architecture": MODEL_ARCHITECTURE,
        "protocol_sha256": self.protocol_sha256,
        "arm": self.arm,
        "replica": self.replica,
        "stage_id": "task4c_bomb_credit6",
        "parent_sha256": self.protocol["source_parent"]["sha256"],
        "training_variable": "bomb_transition_return_horizon_1_vs_6",
        "bomb_return_steps": 1 if self.arm == "control" else 6,
        "non_bomb_return_steps": 1,
        "bomb_maturation_steps": 6,
        "terminal_event_delivery": "posthumous owned-bomb kills attach to the last terminal transition",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"BOMB-credit6 checkpoint {key} mismatch: {path}")


def setup(self):
    if torch is None:
        raise RuntimeError("model_a_v4_bomb_credit6 requires PyTorch")
    torch.set_num_threads(1)
    protocol_path = Path(_required("MODEL_A_BOMB_CREDIT6_PROTOCOL_PATH"))
    self.protocol, self.protocol_sha256 = load_protocol(protocol_path)
    self.arm = _required("MODEL_A_BOMB_CREDIT6_ARM")
    self.replica = _required("MODEL_A_BOMB_CREDIT6_REPLICA")
    if self.arm not in ARMS or self.replica not in REPLICAS:
        raise RuntimeError("invalid BOMB-credit6 arm or replica")
    supplied_seed = int(_required("MODEL_A_BOMB_CREDIT6_SEED"))
    expected_seed = int(self.protocol["training"]["seeds_by_replica"][self.replica]["agent_seed"])
    if self.train and supplied_seed != expected_seed:
        raise RuntimeError("BOMB-credit6 agent seed mismatch")
    self.agent_seed = supplied_seed
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    torch.manual_seed(self.agent_seed)
    self.online_net = DuelingDQN().to(self.device)
    self.resume_checkpoint = None
    self.feature_schema_migrated = False
    supplied_checkpoint = Path(_required("MODEL_A_BOMB_CREDIT6_CHECKPOINT_PATH")).resolve()
    if self.train:
        expected_output = final_checkpoint_path(self.protocol, self.arm, self.replica)
        if supplied_checkpoint != expected_output or supplied_checkpoint.exists():
            raise RuntimeError("BOMB-credit6 output path mismatch or overwrite attempt")
        parent = Path(_required("MODEL_A_BOMB_CREDIT6_PARENT_PATH")).resolve()
        expected_parent = (ROOT / self.protocol["source_parent"]["path"]).resolve()
        if parent != expected_parent or sha256_file(parent) != self.protocol["source_parent"]["sha256"]:
            raise RuntimeError("BOMB-credit6 source-r2 parent mismatch")
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
                raise RuntimeError(f"BOMB-credit6 source parent {key} mismatch")
        self.online_net.load_state_dict(payload["online_net"], strict=True)
        self.resume_checkpoint = payload
        self.completed_rounds = int(payload.get("completed_rounds", 0))
        self.stage_start_rounds = self.completed_rounds
        self.parent_sha256 = sha256_file(parent)
        self.checkpoint_path = supplied_checkpoint
    else:
        if not supplied_checkpoint.is_file():
            raise FileNotFoundError(f"BOMB-credit6 evaluation checkpoint missing: {supplied_checkpoint}")
        payload = _torch_load(supplied_checkpoint)
        _validate_candidate(self, payload, supplied_checkpoint)
        allowed = {
            final_checkpoint_path(self.protocol, self.arm, self.replica),
            *{
                snapshot_path(self.protocol, self.arm, self.replica, value)
                for value in SNAPSHOT_ROUNDS
            },
        }
        if supplied_checkpoint not in allowed:
            raise RuntimeError("BOMB-credit6 checkpoint is outside the immutable artifact set")
        self.online_net.load_state_dict(payload["online_net"], strict=True)
        self.completed_rounds = int(payload.get("completed_rounds", 0))


def act(self, game_state: dict) -> str:
    return v4_callbacks.act(self, game_state)
