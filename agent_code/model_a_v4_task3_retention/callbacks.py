"""Exact-v4 callbacks for isolated retention-aware Task3 branches."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from agent_code.model_a_dqn import callbacks as v4_callbacks
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.network import DuelingDQN, torch

from .config import (
    ROOT, final_checkpoint_path, load_protocol, selected_checkpoint_path, sha256_file,
    snapshot_path,
)


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "online_net" not in payload:
        raise RuntimeError(f"invalid Task3 retention checkpoint: {path}")
    return payload


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for model_a_v4_task3_retention")
    return value


def _validate_new_checkpoint(self, payload: dict, path: Path, expected_stage: str | None = None) -> None:
    expected = {
        "architecture": MODEL_ARCHITECTURE,
        "protocol_sha256": self.protocol_sha256,
        "branch": self.branch,
        "stage_id": self.stage_id if expected_stage is None else expected_stage,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Task3 checkpoint {key} mismatch: {path}")


def setup(self):
    if torch is None:
        raise RuntimeError("model_a_v4_task3_retention requires PyTorch")
    torch.set_num_threads(1)
    protocol_path = Path(_required_environment("MODEL_A_V4T3R_PROTOCOL_PATH"))
    self.protocol, self.protocol_sha256 = load_protocol(protocol_path)
    self.branch = _required_environment("MODEL_A_V4T3R_BRANCH")
    self.stage_id = _required_environment("MODEL_A_V4T3R_STAGE")
    supplied_seed = int(_required_environment("MODEL_A_V4T3R_SEED"))
    expected_seed = int(self.protocol["training"][self.stage_id]["branches"][self.branch]["agent_seed"])
    if self.train and supplied_seed != expected_seed:
        raise RuntimeError("Task3 branch agent seed mismatch")
    self.agent_seed = supplied_seed
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    torch.manual_seed(self.agent_seed)
    self.online_net = DuelingDQN().to(self.device)
    self.resume_checkpoint = None
    self.feature_schema_migrated = False
    supplied_checkpoint = Path(_required_environment("MODEL_A_V4T3R_CHECKPOINT_PATH")).resolve()

    if self.train:
        expected_output = final_checkpoint_path(self.protocol, self.branch, self.stage_id)
        if supplied_checkpoint != expected_output:
            raise RuntimeError("Task3 training output path mismatch")
        if supplied_checkpoint.exists():
            raise RuntimeError(f"refusing to overwrite Task3 final checkpoint: {supplied_checkpoint}")
        self.checkpoint_path = supplied_checkpoint
        parent = Path(_required_environment("MODEL_A_V4T3R_PARENT_PATH")).resolve()
        if self.stage_id == "task3_peaceful":
            expected_parent = (ROOT / self.protocol["source_parent"]["checkpoint"]["path"]).resolve()
            if parent != expected_parent or sha256_file(parent) != self.protocol["source_parent"]["checkpoint"]["sha256"]:
                raise RuntimeError("Task3 peaceful source parent mismatch")
            payload = _torch_load(parent)
            if (
                payload.get("protocol_sha256") != self.protocol["source_parent"]["clean_protocol"]["sha256"]
                or payload.get("arm") != "curriculum"
                or payload.get("replica") != "r2"
                or payload.get("stage_id") != "task2"
            ):
                raise RuntimeError("Task3 peaceful parent metadata mismatch")
        else:
            expected_parent = selected_checkpoint_path(self.protocol, self.branch, "task3_peaceful")
            if parent != expected_parent or not parent.is_file():
                raise RuntimeError("Task3 coin selected-parent path mismatch")
            payload = _torch_load(parent)
            _validate_new_checkpoint(self, payload, parent, "task3_peaceful")
            if payload.get("stage_id") != "task3_peaceful" or not payload.get("selected_by_inner_validation"):
                raise RuntimeError("Task3 coin parent was not selected by inner validation")
        self.online_net.load_state_dict(payload["online_net"], strict=True)
        self.resume_checkpoint = payload
        self.completed_rounds = int(payload.get("completed_rounds", 0))
        self.stage_start_rounds = self.completed_rounds
        self.parent_sha256 = sha256_file(parent)
    else:
        self.checkpoint_path = supplied_checkpoint
        if not supplied_checkpoint.is_file():
            raise FileNotFoundError(f"Task3 evaluation checkpoint missing: {supplied_checkpoint}")
        payload = _torch_load(supplied_checkpoint)
        _validate_new_checkpoint(self, payload, supplied_checkpoint)
        allowed = {
            final_checkpoint_path(self.protocol, self.branch, self.stage_id),
            selected_checkpoint_path(self.protocol, self.branch, self.stage_id),
            *{
                snapshot_path(self.protocol, self.branch, self.stage_id, round_number)
                for round_number in range(50, 601, 50)
            },
        }
        if supplied_checkpoint not in allowed:
            raise RuntimeError("Task3 evaluation checkpoint is outside the frozen artifact set")
        self.online_net.load_state_dict(payload["online_net"], strict=True)
        self.completed_rounds = int(payload.get("completed_rounds", 0))


def act(self, game_state: dict) -> str:
    return v4_callbacks.act(self, game_state)
