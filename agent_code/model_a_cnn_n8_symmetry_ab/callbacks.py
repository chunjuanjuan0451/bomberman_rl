"""Read-only D4 test-time inference A/B for the frozen control-r2 checkpoint.

The control arm executes the current single-view CNN policy.  The treatment
arm evaluates all eight square symmetries, maps every Q vector back to the
original action coordinates, and averages before applying the unchanged
collision-consistent action mask.
"""

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
from agent_code.model_a_v6.symmetry import (
    ACTION_PERMUTATIONS, SYMMETRY_NAMES, transform_game_state,
)


KIND = "model-a-cnn-n8-d4-symmetry-ab"


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for D4 symmetry A/B")
    return value


def single_view_q_values(network, game_state: dict, device) -> np.ndarray:
    features = state_to_features(game_state)
    assert features is not None
    spatial, scalars = _tensors(features, device)
    with torch.no_grad():
        return network(spatial, scalars)[0].detach().cpu().numpy()


def aligned_d4_q_matrix(network, game_state: dict, device) -> np.ndarray:
    """Return eight Q vectors expressed in the original action coordinates."""
    feature_views = [state_to_features(transform_game_state(game_state, symmetry))
                     for symmetry in range(len(SYMMETRY_NAMES))]
    assert all(features is not None for features in feature_views)
    spatial = np.stack([features[0] for features in feature_views])
    scalars = np.stack([features[1] for features in feature_views])
    with torch.no_grad():
        q_views = network(
            torch.as_tensor(spatial, dtype=torch.float32, device=device).div_(255.0),
            torch.as_tensor(scalars, dtype=torch.float32, device=device),
        ).detach().cpu().numpy()
    return np.stack([
        q_views[symmetry, ACTION_PERMUTATIONS[symmetry]]
        for symmetry in range(len(SYMMETRY_NAMES))
    ])


def d4_ensemble_q_values(network, game_state: dict, device) -> tuple[np.ndarray, np.ndarray]:
    aligned = aligned_d4_q_matrix(network, game_state, device)
    return aligned.mean(axis=0), aligned.std(axis=0)


def setup(self) -> None:
    if torch is None:
        raise RuntimeError("D4 symmetry A/B requires PyTorch")
    if self.train:
        raise RuntimeError("D4 symmetry A/B is evaluation-only")
    protocol_path = Path(_required("CNN_D4_PROTOCOL")).resolve()
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("kind") != KIND:
        raise RuntimeError("invalid D4 symmetry A/B protocol")
    self.audit_protocol = protocol
    self.audit_protocol_sha256 = hashlib.sha256(raw).hexdigest()
    self.audit_case = _required("CNN_D4_CASE")
    self.audit_arm = _required("CNN_D4_ARM")
    if self.audit_arm not in {"control", "treatment"}:
        raise RuntimeError("invalid D4 symmetry A/B arm")
    cases = protocol["collection"]["cases"]
    if self.audit_case not in cases:
        raise RuntimeError("invalid D4 symmetry A/B case")
    self.agent_seed = int(_required("CNN_D4_AGENT_SEED"))
    if self.agent_seed != int(cases[self.audit_case]["agent_seed"]):
        raise RuntimeError("D4 symmetry A/B agent seed mismatch")

    checkpoint = Path(_required("CNN_D4_CHECKPOINT")).resolve()
    expected = protocol["checkpoint"]
    expected_path = (Path(__file__).resolve().parents[2] / expected["path"]).resolve()
    if checkpoint != expected_path or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected["sha256"]:
        raise RuntimeError("D4 symmetry A/B checkpoint binding mismatch")
    payload = _load(checkpoint)
    identity = {
        "architecture": ARCHITECTURE,
        "stage": "task4",
        "arm": "control",
        "replica": "r2",
        "stage_rounds": 150,
    }
    if any(payload.get(key) != value for key, value in identity.items()):
        raise RuntimeError("D4 symmetry A/B checkpoint identity mismatch")

    torch.set_num_threads(1)
    self.device = torch.device("cpu")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    self.online_net.load_state_dict(payload["online_net"], strict=True)
    self.online_net.eval()
    for parameter in self.online_net.parameters():
        parameter.requires_grad = False
    self.rng = np.random.default_rng(self.agent_seed)
    self.audit_checkpoint_sha256 = expected["sha256"]
    self._active_bomb = False
    self._active_bomb_position = None


def act(self, game_state: dict) -> str:
    bomb_positions = {tuple(position) for position, _timer in game_state["bombs"]}
    if self._active_bomb and self._active_bomb_position not in bomb_positions:
        self._active_bomb = False
        self._active_bomb_position = None

    legal = effective_action_mask(game_state, self._active_bomb)
    indices = np.flatnonzero(legal)
    if not indices.size:
        indices = np.asarray([ACTIONS.index("WAIT")], dtype=np.int64)
    base_q = single_view_q_values(self.online_net, game_state, self.device)
    q_values = base_q
    if self.audit_arm == "treatment":
        q_values, _ = d4_ensemble_q_values(self.online_net, game_state, self.device)

    best = q_values[indices].max()
    candidates = indices[np.isclose(q_values[indices], best)]
    chosen_index = int(self.rng.choice(candidates))
    action = ACTIONS[chosen_index]
    if action == "BOMB" and game_state["self"][2]:
        self._active_bomb = True
        self._active_bomb_position = tuple(game_state["self"][3])
    return action


__all__ = [
    "act", "aligned_d4_q_matrix", "d4_ensemble_q_values", "setup",
    "single_view_q_values",
]
