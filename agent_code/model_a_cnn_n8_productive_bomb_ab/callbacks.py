"""Frozen control-r2 policy with an optional productive-bomb gate.

The treatment is implemented from the official environment semantics and this
project's existing ``blast_positions`` helper.  It does not import external
agent source code, weights, actions, or labels.
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
from agent_code.model_a_dqn.features import blast_positions


KIND = "model-a-cnn-n8-productive-bomb-ab"
BOMB_INDEX = ACTIONS.index("BOMB")


def bomb_has_productive_target(game_state: dict) -> bool:
    """Whether a prospective bomb can hit a current crate or opponent."""
    origin = tuple(game_state["self"][3])
    blast = set(blast_positions(np.asarray(game_state["field"]), origin))
    opponents = {tuple(other[3]) for other in game_state["others"]}
    return bool(blast & opponents) or any(game_state["field"][cell] == 1 for cell in blast)


def productive_bomb_mask(game_state: dict, base_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Remove only an unproductive BOMB; retain the base mask on empty fallback."""
    result = np.asarray(base_mask, dtype=bool).copy()
    productive = bomb_has_productive_target(game_state)
    veto_condition = bool(result[BOMB_INDEX] and not productive)
    fallback = False
    if veto_condition:
        result[BOMB_INDEX] = False
        if not result.any():
            result = np.asarray(base_mask, dtype=bool).copy()
            fallback = True
    return result, {
        "bomb_legal": bool(base_mask[BOMB_INDEX]),
        "bomb_productive": bool(productive),
        "veto_condition": veto_condition,
        "fallback": fallback,
    }


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for productive-bomb A/B")
    return value


def setup(self) -> None:
    if torch is None:
        raise RuntimeError("productive-bomb A/B requires PyTorch")
    if not self.train:
        raise RuntimeError("productive-bomb A/B uses train mode only for read-only tracing")
    protocol_path = Path(_required("CNN_PRODUCTIVE_BOMB_PROTOCOL")).resolve()
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("kind") != KIND:
        raise RuntimeError("invalid productive-bomb A/B protocol")
    self.audit_protocol = protocol
    self.audit_protocol_sha256 = hashlib.sha256(raw).hexdigest()
    self.audit_case = _required("CNN_PRODUCTIVE_BOMB_CASE")
    self.audit_arm = _required("CNN_PRODUCTIVE_BOMB_ARM")
    if self.audit_arm not in {"control", "treatment"}:
        raise RuntimeError("invalid productive-bomb A/B arm")
    cases = protocol["collection"]["cases"]
    if self.audit_case not in cases:
        raise RuntimeError("invalid productive-bomb A/B case")
    self.agent_seed = int(_required("CNN_PRODUCTIVE_BOMB_AGENT_SEED"))
    if self.agent_seed != int(cases[self.audit_case]["agent_seed"]):
        raise RuntimeError("productive-bomb A/B agent seed mismatch")

    checkpoint = Path(_required("CNN_PRODUCTIVE_BOMB_CHECKPOINT")).resolve()
    expected = protocol["checkpoint"]
    expected_path = (Path(__file__).resolve().parents[2] / expected["path"]).resolve()
    if checkpoint != expected_path or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected["sha256"]:
        raise RuntimeError("productive-bomb A/B checkpoint binding mismatch")
    payload = _load(checkpoint)
    identity = {
        "architecture": ARCHITECTURE,
        "stage": "task4",
        "arm": "control",
        "replica": "r2",
        "stage_rounds": 150,
    }
    if any(payload.get(key) != value for key, value in identity.items()):
        raise RuntimeError("productive-bomb A/B checkpoint identity mismatch")

    torch.set_num_threads(1)
    self.device = torch.device("cpu")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    self.online_net.load_state_dict(payload["online_net"], strict=True)
    self.online_net.eval()
    for parameter in self.online_net.parameters():
        parameter.requires_grad = False
    self.rng = np.random.default_rng(self.agent_seed)
    self.audit_checkpoint_sha256 = expected["sha256"]
    self.audit_trace_path = (
        Path(__file__).resolve().parents[2] / protocol["trace_directory"]
        / f"{self.audit_case}-{self.audit_arm}.json"
    )


def act(self, game_state: dict) -> str:
    if self._pending_decision is not None:
        raise RuntimeError("previous productive-bomb decision was not consumed")

    bomb_positions = {tuple(position) for position, _timer in game_state["bombs"]}
    if self._active_bomb and self._active_bomb_position not in bomb_positions:
        self._active_bomb = False
        self._active_bomb_position = None

    base = effective_action_mask(game_state, self._active_bomb)
    base_indices = np.flatnonzero(base)
    q_values = np.full(len(ACTIONS), np.nan, dtype=np.float32)
    if not base_indices.size:
        original_index = ACTIONS.index("WAIT")
    else:
        features = state_to_features(game_state)
        assert features is not None
        spatial, scalars = _tensors(features, self.device)
        with torch.no_grad():
            q_values = self.online_net(spatial, scalars)[0].detach().cpu().numpy()
        best = q_values[base_indices].max()
        original_index = int(self.rng.choice(base_indices[np.isclose(q_values[base_indices], best)]))

    diagnostic_mask, diagnostic = productive_bomb_mask(game_state, base)
    effective = diagnostic_mask if self.audit_arm == "treatment" else base
    changed = bool(effective.any() and not effective[original_index])
    chosen_index = original_index
    if changed:
        indices = np.flatnonzero(effective)
        best = q_values[indices].max()
        chosen_index = int(self.rng.choice(indices[np.isclose(q_values[indices], best)]))
    action = ACTIONS[chosen_index]

    intervention = bool(self.audit_arm == "treatment" and diagnostic["veto_condition"])
    self._pending_decision = {
        "round": int(game_state["round"]),
        "step": int(game_state["step"]),
        "action": action,
        "original_action": ACTIONS[original_index],
        "bomb_legal": diagnostic["bomb_legal"],
        "bomb_productive": diagnostic["bomb_productive"],
        "veto_condition": diagnostic["veto_condition"],
        "restriction_available": intervention and not diagnostic["fallback"],
        "fallback_used": intervention and diagnostic["fallback"],
        "action_changed": changed,
    }
    if action == "BOMB" and game_state["self"][2]:
        self._active_bomb = True
        self._active_bomb_position = tuple(game_state["self"][3])
    return action


__all__ = ["act", "bomb_has_productive_target", "productive_bomb_mask", "setup"]
