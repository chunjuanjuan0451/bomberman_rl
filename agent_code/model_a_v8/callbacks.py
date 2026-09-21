"""Callbacks for frozen-v4 rollout-distilled residual experiments."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from agent_code.model_a_v6.features import ACTIONS, legal_action_mask, state_to_features
from .config import architecture_name, load_config
from .network import StableDQN, torch

CHECKPOINT_PATH = Path(os.environ.get("MODEL_A_V8_CHECKPOINT_PATH", "model_a_v8.pt"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def setup(self):
    if torch is None:
        raise RuntimeError("Model A v8 requires PyTorch")
    torch.set_num_threads(1)
    self.v8_config, self.v8_config_path, self.v8_config_sha256 = load_config()
    hyper = self.v8_config["hyperparameters"]
    seed_text = os.environ.get("MODEL_A_V8_SEED")
    self.agent_seed = None if seed_text is None else int(seed_text)
    if self.agent_seed is not None:
        torch.manual_seed(self.agent_seed)
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = StableDQN(hyper["delta_cap"]).to(self.device)
    parent = Path(self.v8_config["parent_checkpoint"])
    if not parent.is_absolute():
        parent = self.v8_config_path.parents[2] / parent
    parent = parent.resolve()
    if _sha256(parent) != self.v8_config["parent_sha256"]:
        raise RuntimeError("Frozen v4 parent hash mismatch")
    self.online_net.base.load_state_dict(_load(parent)["online_net"], strict=True)
    self.online_net.freeze_base()
    if os.environ.get("MODEL_A_V8_RESUME", "") == "1":
        raise RuntimeError("v8 forbids resume")
    if self.train:
        self.completed_rounds = 0
        return
    checkpoint = _load(CHECKPOINT_PATH)
    if checkpoint.get("architecture") != architecture_name():
        raise RuntimeError("v8 checkpoint architecture mismatch")
    if checkpoint.get("config_sha256") != self.v8_config_sha256:
        raise RuntimeError("v8 checkpoint/config mismatch")
    if checkpoint.get("parent_sha256") != self.v8_config["parent_sha256"]:
        raise RuntimeError("v8 checkpoint parent mismatch")
    self.online_net.load_state_dict(checkpoint["online_net"], strict=True)
    self.online_net.freeze_base()
    self.online_net.eval()
    self.completed_rounds = int(checkpoint.get("completed_rounds", 0))


def act(self, game_state):
    legal = np.flatnonzero(legal_action_mask(game_state))
    if not legal.size:
        return "WAIT"
    if self.train and self.rng.random() < getattr(self, "epsilon", 0.0):
        return ACTIONS[int(self.rng.choice(legal))]
    local, global_features = state_to_features(game_state)
    with torch.no_grad():
        q = self.online_net(
            torch.as_tensor(local[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
        )[0].cpu().numpy()
    best = q[legal].max()
    return ACTIONS[int(self.rng.choice(legal[np.isclose(q[legal], best)]))]
