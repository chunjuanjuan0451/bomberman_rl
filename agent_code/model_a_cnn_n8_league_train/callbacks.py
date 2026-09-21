"""CNN continuation with the frozen immediate-collision mask."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from agent_code.model_a_cnn_n8.callbacks import _load, _tensors
from agent_code.model_a_cnn_n8.features import ACTIONS, state_to_features
from agent_code.model_a_cnn_n8.network import ARCHITECTURE, FullBoardDuelingCNN, torch
from agent_code.model_a_cnn_n8_postbomb_movement_audit.callbacks import immediate_collision_action_details
from agent_code.model_a_dqn.features import legal_action_mask


KIND = "model-a-cnn-n8-opponent-league-ab"


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for league training")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effective_action_mask(game_state: dict, active_own_bomb: bool) -> np.ndarray:
    """Return the exact action set used by behaviour and DDQN bootstrapping."""
    legal = legal_action_mask(game_state)
    if not active_own_bomb:
        return legal
    robust = np.asarray(immediate_collision_action_details(game_state)["robust_action_mask"], dtype=bool)
    filtered = legal & robust
    return filtered if filtered.any() else legal


def setup(self) -> None:
    if torch is None:
        raise RuntimeError("league training requires PyTorch")
    torch.set_num_threads(1)
    protocol_path = Path(_required("CNN_LEAGUE_PROTOCOL_PATH")).resolve()
    raw = protocol_path.read_bytes()
    self.cnn_protocol = json.loads(raw)
    self.cnn_protocol_path = protocol_path
    self.cnn_protocol_sha256 = hashlib.sha256(raw).hexdigest()
    if self.cnn_protocol.get("kind") != KIND:
        raise RuntimeError("invalid league protocol")
    self.cnn_mode = os.environ.get("CNN_LEAGUE_MODE", "evaluate")
    self.cnn_stage = "task4"
    self.cnn_arm = _required("CNN_LEAGUE_ARM")
    self.cnn_replica = _required("CNN_LEAGUE_REPLICA")
    if self.cnn_arm not in self.cnn_protocol["arms"] or self.cnn_replica not in self.cnn_protocol["replicas"]:
        raise RuntimeError("unknown league arm or replica")
    self.agent_seed = int(_required("CNN_LEAGUE_AGENT_SEED"))
    expected_seed = int(self.cnn_protocol["replicas"][self.cnn_replica]["agent_seed"])
    if self.agent_seed != expected_seed:
        raise RuntimeError("league learner seed mismatch")
    torch.manual_seed(self.agent_seed)
    self.rng = np.random.default_rng(self.agent_seed)
    if self.train and self.cnn_mode == "train":
        device_name = self.cnn_protocol["hardware"]["training_device"]
        if device_name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("formal league training requires MPS")
        self.device = torch.device(device_name)
    else:
        self.device = torch.device("cpu")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    checkpoint_path = Path(_required("CNN_LEAGUE_CHECKPOINT_PATH")).resolve()
    parent = self.cnn_protocol["parent"]
    if self.cnn_mode == "train":
        if checkpoint_path != (Path(__file__).resolve().parents[2] / parent["path"]).resolve():
            raise RuntimeError("league parent path mismatch")
        if _sha256(checkpoint_path) != parent["sha256"]:
            raise RuntimeError("league parent hash mismatch")
    checkpoint = _load(checkpoint_path)
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("stage") != "task4":
        raise RuntimeError("league checkpoint identity mismatch")
    if self.cnn_mode == "train" and (checkpoint.get("replica") != "r3" or checkpoint.get("stage_rounds") != 800):
        raise RuntimeError("league parent identity mismatch")
    if self.cnn_mode == "evaluate" and checkpoint_path.name != "round-0800.pt":
        if (checkpoint.get("protocol_sha256") != self.cnn_protocol_sha256
                or checkpoint.get("replica") != self.cnn_replica
                or checkpoint.get("arm") != self.cnn_arm):
            raise RuntimeError("league candidate lineage mismatch")
    self.online_net.load_state_dict(checkpoint["online_net"], strict=True)
    self.completed_rounds = int(checkpoint["completed_total_rounds"])
    self.resume_checkpoint = checkpoint if self.cnn_mode == "train" else None
    self.parent_checkpoint_path = checkpoint_path if self.cnn_mode == "train" else None
    self.parent_checkpoint_sha256 = parent["sha256"] if self.cnn_mode == "train" else None
    self._active_bomb = False
    if self.cnn_mode == "evaluate":
        self.online_net.eval()


def act(self, game_state: dict) -> str:
    legal = effective_action_mask(game_state, self._active_bomb)
    indices = np.flatnonzero(legal)
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


__all__ = ["act", "effective_action_mask", "setup"]
