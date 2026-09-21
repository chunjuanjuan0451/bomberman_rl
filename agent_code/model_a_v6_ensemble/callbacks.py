"""Callbacks for the inference-only multi-seed residual ensemble."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from agent_code.model_a_v6.features import ACTIONS, legal_action_mask, state_to_features
from .config import architecture_name, load_config, sha256_file
from .network import EnsembleDQN, torch

BUNDLE_PATH = Path(os.environ.get("MODEL_A_V6_ENSEMBLE_CHECKPOINT_PATH", "model_a_v6_ensemble.pt"))


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _repository_root(config_path: Path) -> Path:
    return config_path.parents[2]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def setup(self):
    if torch is None:
        raise RuntimeError("v6 ensemble requires PyTorch")
    if self.train:
        raise RuntimeError("v6 ensemble is inference-only")
    torch.set_num_threads(1)
    config, config_path, config_sha256 = load_config()
    seed_text = os.environ.get("MODEL_A_V6_ENSEMBLE_SEED")
    self.agent_seed = None if seed_text is None else int(seed_text)
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = EnsembleDQN(len(config["members"]), config["delta_cap"]).to(self.device)
    root = _repository_root(config_path)
    parent_path = _resolve(root, config["parent_checkpoint"])
    parent = _torch_load(parent_path)
    self.online_net.base.load_state_dict(parent["online_net"], strict=True)
    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError(f"explicit v6 ensemble bundle is required: {BUNDLE_PATH}")
    bundle = _torch_load(BUNDLE_PATH)
    expected_hashes = [member["sha256"] for member in config["members"]]
    if bundle.get("architecture") != architecture_name():
        raise RuntimeError("v6 ensemble bundle architecture mismatch")
    if bundle.get("config_sha256") != config_sha256:
        raise RuntimeError("v6 ensemble bundle config hash mismatch")
    if bundle.get("parent_sha256") != config["parent_sha256"]:
        raise RuntimeError("v6 ensemble bundle parent hash mismatch")
    if bundle.get("member_sha256") != expected_hashes:
        raise RuntimeError("v6 ensemble member hash list mismatch")
    residual_states = bundle.get("residual_states", [])
    if len(residual_states) != len(self.online_net.residual_heads):
        raise RuntimeError("v6 ensemble residual count mismatch")
    for head, state in zip(self.online_net.residual_heads, residual_states):
        head.load_state_dict(state, strict=True)
    self.online_net.freeze_all()
    self.completed_rounds = 0


def act(self, game_state: dict) -> str:
    legal_indices = np.flatnonzero(legal_action_mask(game_state))
    if legal_indices.size == 0:
        return "WAIT"
    local, global_features = state_to_features(game_state)
    with torch.no_grad():
        q_values = self.online_net(
            torch.as_tensor(local[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
        )[0].cpu().numpy()
    best = q_values[legal_indices].max()
    candidates = legal_indices[np.isclose(q_values[legal_indices], best)]
    return ACTIONS[int(self.rng.choice(candidates))]
