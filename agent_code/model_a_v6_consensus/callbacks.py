"""Callbacks for action-majority consensus with frozen-v4 fallback."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from agent_code.model_a_v6.features import ACTIONS, legal_action_mask, state_to_features
from agent_code.model_a_v6_ensemble.network import EnsembleDQN, torch
from .config import architecture_name, load_config

BUNDLE_PATH = Path(os.environ.get("MODEL_A_V6_CONSENSUS_CHECKPOINT_PATH", "model_a_v6_consensus.pt"))


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _best_action(q_values, legal_indices, rng) -> int:
    best = q_values[legal_indices].max()
    candidates = legal_indices[np.isclose(q_values[legal_indices], best)]
    return int(rng.choice(candidates))


def select_consensus_action(base_q, member_q, legal_indices, rng) -> tuple[int, bool, tuple[int, ...]]:
    base_action = _best_action(base_q, legal_indices, rng)
    member_actions = tuple(_best_action(values, legal_indices, rng) for values in member_q)
    counts = np.bincount(member_actions, minlength=len(ACTIONS))
    winner = int(counts.argmax())
    if counts[winner] >= 2:
        return winner, winner != base_action, member_actions
    return base_action, False, member_actions


def setup(self):
    if torch is None:
        raise RuntimeError("v6 consensus requires PyTorch")
    if self.train:
        raise RuntimeError("v6 consensus is inference-only")
    torch.set_num_threads(1)
    config, config_path, config_sha256 = load_config()
    seed_text = os.environ.get("MODEL_A_V6_CONSENSUS_SEED")
    self.agent_seed = None if seed_text is None else int(seed_text)
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = EnsembleDQN(3, config["delta_cap"]).to(self.device)
    root = config_path.parents[2]
    parent = _torch_load(_resolve(root, config["parent_checkpoint"]))
    self.online_net.base.load_state_dict(parent["online_net"], strict=True)
    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError(f"explicit v6 consensus bundle is required: {BUNDLE_PATH}")
    bundle = _torch_load(BUNDLE_PATH)
    if bundle.get("architecture") != architecture_name():
        raise RuntimeError("v6 consensus bundle architecture mismatch")
    if bundle.get("config_sha256") != config_sha256:
        raise RuntimeError("v6 consensus bundle config hash mismatch")
    if bundle.get("parent_sha256") != config["parent_sha256"]:
        raise RuntimeError("v6 consensus bundle parent hash mismatch")
    if bundle.get("member_sha256") != [member["sha256"] for member in config["members"]]:
        raise RuntimeError("v6 consensus member hash list mismatch")
    states = bundle.get("residual_states", [])
    if len(states) != 3:
        raise RuntimeError("v6 consensus residual count mismatch")
    for head, state in zip(self.online_net.residual_heads, states):
        head.load_state_dict(state, strict=True)
    self.online_net.freeze_all()
    self.completed_rounds = 0


def act(self, game_state: dict) -> str:
    legal_indices = np.flatnonzero(legal_action_mask(game_state))
    if legal_indices.size == 0:
        return "WAIT"
    local, global_features = state_to_features(game_state)
    with torch.no_grad():
        final_q, mean_delta, member_deltas = self.online_net(
            torch.as_tensor(local[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
            return_delta=True,
        )
    base_q = (final_q - mean_delta)[0].cpu().numpy()
    member_q = (base_q[None, :] + member_deltas[:, 0].cpu().numpy())
    action, _, _ = select_consensus_action(base_q, member_q, legal_indices, self.rng)
    return ACTIONS[action]
