"""Resume the frozen s162 control checkpoint for the s166 crate-reward A/B."""

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


KIND = "model-a-cnn-n8-crate-reward-ab"


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(self) -> None:
    if torch is None:
        raise RuntimeError("crate-reward A/B requires PyTorch")
    torch.set_num_threads(1)
    protocol_path = Path(_required("CNN_CRATE_PROTOCOL_PATH")).resolve()
    raw = protocol_path.read_bytes()
    self.cnn_protocol = json.loads(raw)
    self.cnn_protocol_path = protocol_path
    self.cnn_protocol_sha256 = hashlib.sha256(raw).hexdigest()
    if self.cnn_protocol.get("kind") != KIND:
        raise RuntimeError("invalid crate-reward A/B protocol")
    self.cnn_mode = os.environ.get("CNN_CRATE_MODE", "evaluate")
    self.cnn_stage = "task4"
    self.cnn_arm = _required("CNN_CRATE_ARM")
    self.cnn_replica = _required("CNN_CRATE_REPLICA")
    if self.cnn_arm not in self.cnn_protocol["arms"]:
        raise RuntimeError("unknown crate-reward A/B arm")
    if self.cnn_replica not in self.cnn_protocol["replicas"]:
        raise RuntimeError("unknown crate-reward A/B replica")
    self.cnn_crate_reward = float(self.cnn_protocol["arms"][self.cnn_arm]["crate_destroyed_reward"])
    self.agent_seed = int(_required("CNN_CRATE_AGENT_SEED"))
    if self.agent_seed != int(self.cnn_protocol["replicas"][self.cnn_replica]["agent_seed"]):
        raise RuntimeError("crate-reward A/B learner seed mismatch")
    torch.manual_seed(self.agent_seed)
    self.rng = np.random.default_rng(self.agent_seed)
    if self.train and self.cnn_mode == "train":
        device_name = self.cnn_protocol["hardware"]["training_device"]
        if device_name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("formal crate-reward A/B requires MPS")
        self.device = torch.device(device_name)
    else:
        self.device = torch.device("cpu")
    checkpoint_path = Path(_required("CNN_CRATE_CHECKPOINT_PATH")).resolve()
    parent = self.cnn_protocol["parent"]
    if self.cnn_mode == "train":
        expected = (Path(__file__).resolve().parents[2] / parent["path"]).resolve()
        if checkpoint_path != expected or _sha256(checkpoint_path) != parent["sha256"]:
            raise RuntimeError("crate-reward A/B parent checkpoint drift")
    checkpoint = _load(checkpoint_path)
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("stage") != "task4":
        raise RuntimeError("crate-reward A/B checkpoint identity mismatch")
    if self.cnn_mode == "train":
        expected_identity = {
            "arm": "control", "replica": "r2", "stage_rounds": 150,
            "completed_total_rounds": 2900, "n_step": 8,
        }
        if any(checkpoint.get(key) != value for key, value in expected_identity.items()):
            raise RuntimeError("crate-reward A/B parent lineage mismatch")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    self.online_net.load_state_dict(checkpoint["online_net"], strict=True)
    self.completed_rounds = int(checkpoint["completed_total_rounds"])
    self.resume_checkpoint = checkpoint if self.cnn_mode == "train" else None
    self.parent_checkpoint_path = checkpoint_path if self.cnn_mode == "train" else None
    self.parent_checkpoint_sha256 = parent["sha256"] if self.cnn_mode == "train" else None
    self._active_bomb = False
    if self.cnn_mode != "train":
        self.online_net.eval()


def act(self, game_state: dict) -> str:
    indices = np.flatnonzero(effective_action_mask(game_state, self._active_bomb))
    if not indices.size:
        return "WAIT"
    epsilon = float(getattr(self, "epsilon", 0.0)) if self.cnn_mode == "train" else 0.0
    if epsilon and self.rng.random() < epsilon:
        return ACTIONS[int(self.rng.choice(indices))]
    features = state_to_features(game_state)
    assert features is not None
    spatial, scalars = _tensors(features, self.device)
    with torch.no_grad():
        q_values = self.online_net(spatial, scalars)[0].detach().cpu().numpy()
    best = q_values[indices].max()
    return ACTIONS[int(self.rng.choice(indices[np.isclose(q_values[indices], best)]))]


__all__ = ["act", "setup"]
