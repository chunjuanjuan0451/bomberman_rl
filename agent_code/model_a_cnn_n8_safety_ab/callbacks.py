"""Inference and setup callbacks for the CNN safety replay A/B."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from agent_code.model_a_cnn_n8.features import ACTIONS, legal_action_mask, state_to_features
from agent_code.model_a_cnn_n8.network import ARCHITECTURE, FullBoardDuelingCNN, torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def setup(self) -> None:
    if torch is None:
        raise RuntimeError("CNN safety A/B requires PyTorch")
    torch.set_num_threads(1)
    protocol_path = Path(os.environ["CNN_SAFETY_AB_PROTOCOL_PATH"]).resolve()
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("kind") != "model-a-cnn-n8-safety-replay-ab":
        raise RuntimeError("invalid CNN safety A/B protocol")
    self.safety_protocol = protocol
    self.safety_protocol_path = protocol_path
    self.safety_protocol_sha256 = hashlib.sha256(raw).hexdigest()
    self.safety_mode = os.environ.get("CNN_SAFETY_AB_MODE", "evaluate")
    self.safety_arm = os.environ.get("CNN_SAFETY_AB_ARM", "")
    self.safety_replica = os.environ.get("CNN_SAFETY_AB_REPLICA", "")
    if self.safety_mode not in ("train", "evaluate"):
        raise RuntimeError("invalid CNN safety A/B mode")
    if self.safety_mode == "train" and self.safety_arm not in protocol["arms"]:
        raise RuntimeError("invalid CNN safety A/B arm")
    self.agent_seed = int(os.environ["CNN_SAFETY_AB_SEED"])
    torch.manual_seed(self.agent_seed)
    self.rng = np.random.default_rng(self.agent_seed)
    if self.train and self.safety_mode == "train":
        device_name = protocol["hardware"]["training_device"]
        if device_name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("formal CNN safety training requires MPS")
        self.device = torch.device(device_name)
    else:
        self.device = torch.device("cpu")
    checkpoint_path = Path(os.environ["CNN_SAFETY_AB_CHECKPOINT_PATH"]).resolve()
    checkpoint = _load(checkpoint_path)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise RuntimeError("CNN safety checkpoint architecture mismatch")
    if _sha256(checkpoint_path) != os.environ["CNN_SAFETY_AB_CHECKPOINT_SHA256"]:
        raise RuntimeError("CNN safety checkpoint hash mismatch")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    self.online_net.load_state_dict(checkpoint["online_net"], strict=True)
    self.online_net.eval() if self.safety_mode == "evaluate" else self.online_net.train()
    self.safety_checkpoint_path = checkpoint_path
    self.safety_checkpoint_sha256 = _sha256(checkpoint_path)
    self.resume_checkpoint = checkpoint if self.safety_mode == "train" else None
    self.completed_rounds = int(checkpoint.get("completed_total_rounds", 0))


def act(self, game_state: dict) -> str:
    legal = legal_action_mask(game_state)
    indices = np.flatnonzero(legal)
    if not indices.size:
        return "WAIT"
    epsilon = float(getattr(self, "epsilon", 0.0)) if self.safety_mode == "train" else 0.0
    if epsilon and self.rng.random() < epsilon:
        return ACTIONS[int(self.rng.choice(indices))]
    spatial, scalars = state_to_features(game_state)
    spatial_tensor = torch.as_tensor(spatial[None], dtype=torch.float32, device=self.device).div_(255.0)
    scalar_tensor = torch.as_tensor(scalars[None], dtype=torch.float32, device=self.device)
    with torch.no_grad():
        values = self.online_net(spatial_tensor, scalar_tensor)[0].detach().cpu().numpy()
    best = values[indices].max()
    candidates = indices[np.isclose(values[indices], best)]
    return ACTIONS[int(self.rng.choice(candidates))]
