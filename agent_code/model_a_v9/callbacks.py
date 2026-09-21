"""Callback entry points for Model A v9."""

from collections import deque
import os
from pathlib import Path

import numpy as np

from .config import architecture_name, load_config
from .features import ACTIONS, HISTORY_LENGTH, dynamic_frame, legal_action_mask, state_to_features
from .network import V9Network, torch

CHECKPOINT_PATH = Path(os.environ.get("MODEL_A_V9_CHECKPOINT_PATH", "model_a_v9.pt"))


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def setup(self):
    if torch is None:
        raise RuntimeError("Model A v9 requires PyTorch")
    torch.set_num_threads(4 if self.train else 1)
    self.v9_config, self.v9_config_path, self.v9_config_sha256 = load_config()
    self.agent_seed = int(os.environ.get("MODEL_A_V9_SEED", self.v9_config["agent_seed"]))
    torch.manual_seed(self.agent_seed)
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = V9Network(self.v9_config["hyperparameters"]["n_quantiles"]).to(self.device)
    self.frame_history = deque(maxlen=HISTORY_LENGTH)
    self.history_round = None
    self.last_v9_features = self.last_v9_state_key = None
    if os.environ.get("MODEL_A_V9_RESUME", "") == "1":
        raise RuntimeError("v9 resume is disabled in the initial run")
    if self.train:
        self.completed_rounds = 0
        return
    checkpoint = _load(CHECKPOINT_PATH)
    if checkpoint.get("architecture") != architecture_name() or checkpoint.get("config_sha256") != self.v9_config_sha256:
        raise RuntimeError("v9 checkpoint metadata mismatch")
    self.online_net.load_state_dict(checkpoint["online_net"], strict=True)
    self.online_net.eval()
    self.completed_rounds = int(checkpoint.get("completed_rounds", 0))


def _features_and_advance(self, game_state):
    if self.history_round != game_state["round"]:
        self.frame_history.clear()
        self.history_round = game_state["round"]
    features = state_to_features(game_state, self.frame_history)
    self.last_v9_features = features
    self.last_v9_state_key = game_state["round"], game_state["step"]
    self.frame_history.append(dynamic_frame(game_state))
    return features


def act(self, game_state):
    legal = np.flatnonzero(legal_action_mask(game_state))
    if not legal.size:
        return "WAIT"
    features = _features_and_advance(self, game_state)
    if self.train and self.rng.random() < getattr(self, "epsilon", 0.0):
        return ACTIONS[int(self.rng.choice(legal))]
    spatial, global_features = features
    with torch.no_grad():
        q = self.online_net.q_values(
            torch.as_tensor(spatial[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
        )[0].cpu().numpy()
    best = q[legal].max()
    return ACTIONS[int(self.rng.choice(legal[np.isclose(q[legal], best)]))]
