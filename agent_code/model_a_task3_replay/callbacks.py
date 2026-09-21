"""Callbacks for the isolated Task-3 replay-sampling agent."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from agent_code.model_a_task3.features import task3_action_features
from agent_code.model_a_task3.network import Task3DQN, torch
from agent_code.model_a_v6.features import ACTIONS, legal_action_mask, state_to_features

from .config import architecture_name, load_config


CHECKPOINT_PATH = Path(os.environ.get("MODEL_A_TASK3_REPLAY_CHECKPOINT_PATH", "model_a_task3_replay.pt"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parent_path(config: dict, config_path: Path) -> Path:
    path = Path(config["parent_checkpoint"])
    return path if path.is_absolute() else config_path.parents[2] / path


def _load_parent(model, config: dict, config_path: Path) -> Path:
    parent = _parent_path(config, config_path).resolve()
    if _sha256(parent) != config["parent_sha256"]:
        raise RuntimeError(f"frozen v4 parent hash mismatch: {parent}")
    checkpoint = _torch_load(parent)
    model.base.load_state_dict(checkpoint["online_net"], strict=True)
    model.freeze_base()
    return parent


def setup(self):
    if torch is None:
        raise RuntimeError("Task-3 replay agent requires PyTorch")
    torch.set_num_threads(1)
    self.task3_config, self.task3_config_path, self.task3_config_sha256 = load_config()
    hyper = self.task3_config["hyperparameters"]
    seed_text = os.environ.get("MODEL_A_TASK3_REPLAY_SEED")
    if seed_text is None:
        raise RuntimeError("MODEL_A_TASK3_REPLAY_SEED is required")
    self.agent_seed = int(seed_text)
    torch.manual_seed(self.agent_seed)
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = Task3DQN(hyper["delta_cap"], True).to(self.device)
    self.parent_checkpoint_path = _load_parent(self.online_net, self.task3_config, self.task3_config_path)
    if os.environ.get("MODEL_A_TASK3_REPLAY_RESUME", "") == "1":
        raise RuntimeError("Task-3 replay runs forbid resume")
    if self.train:
        self.completed_rounds = 0
        return
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"explicit Task-3 replay checkpoint is required: {CHECKPOINT_PATH}")
    checkpoint = _torch_load(CHECKPOINT_PATH)
    checks = {
        "architecture": architecture_name(self.task3_config),
        "config_sha256": self.task3_config_sha256,
        "parent_sha256": self.task3_config["parent_sha256"],
        "sampling_profile": self.task3_config["sampling_profile"],
        "kill_causal_fraction": self.task3_config["kill_causal_fraction"],
    }
    for key, expected in checks.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"Task-3 replay checkpoint {key} mismatch")
    self.online_net.load_state_dict(checkpoint["online_net"], strict=True)
    self.online_net.freeze_base()
    self.completed_rounds = int(checkpoint.get("completed_rounds", 0))


def act(self, game_state: dict) -> str:
    legal_indices = np.flatnonzero(legal_action_mask(game_state))
    if legal_indices.size == 0:
        return "WAIT"
    epsilon = getattr(self, "epsilon", 0.0)
    if self.train and self.rng.random() < epsilon:
        return ACTIONS[int(self.rng.choice(legal_indices))]
    local, global_features = state_to_features(game_state)
    action_features = task3_action_features(game_state)
    with torch.no_grad():
        q_values = self.online_net(
            torch.as_tensor(local[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(action_features[None], dtype=torch.float32, device=self.device),
        )[0].cpu().numpy()
    best = q_values[legal_indices].max()
    candidates = legal_indices[np.isclose(q_values[legal_indices], best)]
    return ACTIONS[int(self.rng.choice(candidates))]
