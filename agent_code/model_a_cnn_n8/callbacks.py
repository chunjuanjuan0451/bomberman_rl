"""Inference callbacks for the end-to-end CNN curriculum agent."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .features import ACTIONS, legal_action_mask, state_to_features
from .network import ARCHITECTURE, FullBoardDuelingCNN, torch


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _protocol() -> tuple[dict, Path, str]:
    path = Path(os.environ["MODEL_A_CNN_PROTOCOL_PATH"]).resolve()
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("kind") != "model-a-cnn-n8-clean-curriculum":
        raise RuntimeError("invalid CNN curriculum protocol")
    return protocol, path, _sha256_bytes(raw)


def setup(self) -> None:
    if torch is None:
        raise RuntimeError("model_a_cnn_n8 requires PyTorch")
    torch.set_num_threads(1)
    self.cnn_protocol, self.cnn_protocol_path, self.cnn_protocol_sha256 = _protocol()
    self.cnn_mode = os.environ.get("MODEL_A_CNN_MODE", "evaluate")
    self.cnn_stage = os.environ.get("MODEL_A_CNN_STAGE", "task1")
    self.cnn_replica = os.environ.get("MODEL_A_CNN_REPLICA", "")
    if self.cnn_mode not in ("train", "evaluate") or self.cnn_stage not in self.cnn_protocol["stage_order"]:
        raise RuntimeError("invalid CNN mode or stage")
    self.agent_seed = int(os.environ["MODEL_A_CNN_SEED"])
    torch.manual_seed(self.agent_seed)
    self.rng = np.random.default_rng(self.agent_seed)
    if self.train and self.cnn_mode == "train":
        device_name = self.cnn_protocol["hardware"]["training_device"]
        if device_name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("formal CNN training requires available MPS")
        self.device = torch.device(device_name)
    else:
        self.device = torch.device("cpu")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    self.completed_rounds = 0
    checkpoint_text = os.environ.get("MODEL_A_CNN_CHECKPOINT_PATH")
    self.resume_checkpoint = None
    self.parent_checkpoint_path = None
    self.parent_checkpoint_sha256 = None
    if self.cnn_mode == "evaluate" or os.environ.get("MODEL_A_CNN_RESUME") == "1":
        if not checkpoint_text:
            raise RuntimeError("CNN checkpoint path is required")
        checkpoint_path = Path(checkpoint_text).resolve()
        checkpoint = _load(checkpoint_path)
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise RuntimeError("CNN checkpoint architecture mismatch")
        if self.cnn_mode == "train":
            expected_parent_stage = {
                "task2": "task1", "task3_peaceful": "task2",
                "task3_coin": "task3_peaceful", "task4": "task3_coin",
            }.get(self.cnn_stage)
            if expected_parent_stage is None or checkpoint.get("stage") != expected_parent_stage:
                raise RuntimeError("CNN parent stage mismatch")
            if checkpoint.get("replica") != self.cnn_replica:
                raise RuntimeError("CNN parent replica mismatch")
            actual_parent_hash = _sha256_file(checkpoint_path)
            if actual_parent_hash != os.environ.get("MODEL_A_CNN_PARENT_SHA256"):
                raise RuntimeError("CNN parent checkpoint hash mismatch")
            self.parent_checkpoint_path = checkpoint_path
            self.parent_checkpoint_sha256 = actual_parent_hash
        self.online_net.load_state_dict(checkpoint["online_net"], strict=True)
        self.completed_rounds = int(checkpoint.get("completed_total_rounds", 0))
        self.resume_checkpoint = checkpoint if self.cnn_mode == "train" else None
    elif self.cnn_stage != "task1":
        raise RuntimeError("only Task1 may start without a parent checkpoint")
    if self.cnn_mode == "evaluate":
        self.online_net.eval()


def _tensors(features, device):
    spatial, scalars = features
    spatial_tensor = torch.as_tensor(spatial[None], dtype=torch.float32, device=device).div_(255.0)
    scalar_tensor = torch.as_tensor(scalars[None], dtype=torch.float32, device=device)
    return spatial_tensor, scalar_tensor


def act(self, game_state: dict) -> str:
    legal = legal_action_mask(game_state)
    legal_indices = np.flatnonzero(legal)
    if not legal_indices.size:
        return "WAIT"
    epsilon = float(getattr(self, "epsilon", 0.0)) if self.cnn_mode == "train" else 0.0
    if epsilon and self.rng.random() < epsilon:
        return ACTIONS[int(self.rng.choice(legal_indices))]
    features = state_to_features(game_state)
    assert features is not None
    spatial, scalars = _tensors(features, self.device)
    with torch.no_grad():
        q_values = self.online_net(spatial, scalars)[0].detach().cpu().numpy()
    best = q_values[legal_indices].max()
    candidates = legal_indices[np.isclose(q_values[legal_indices], best)]
    return ACTIONS[int(self.rng.choice(candidates))]


__all__ = ["_load", "_sha256_file", "act", "setup"]
