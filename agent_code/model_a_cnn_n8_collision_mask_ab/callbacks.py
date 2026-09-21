"""Frozen CNN with an optional one-step post-bomb collision filter."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from agent_code.model_a_cnn_n8.callbacks import _load, _tensors
from agent_code.model_a_cnn_n8.features import ACTIONS, state_to_features
from agent_code.model_a_cnn_n8.network import ARCHITECTURE, FullBoardDuelingCNN, torch
from agent_code.model_a_cnn_n8_postbomb_movement_audit.callbacks import immediate_collision_action_details
from agent_code.model_a_dqn.features import legal_action_mask

from .config import ROOT, load_protocol, sha256_file, trace_path


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for collision-mask A/B")
    return value


def setup(self) -> None:
    if torch is None:
        raise RuntimeError("collision-mask A/B requires PyTorch")
    if not self.train:
        raise RuntimeError("collision-mask A/B agent must run in training mode for tracing")
    torch.set_num_threads(1)
    self.audit_protocol, self.audit_protocol_path, self.audit_protocol_sha256 = load_protocol()
    self.audit_case = _required("CNN_COLLISION_AB_CASE")
    self.audit_arm = _required("CNN_COLLISION_AB_ARM")
    if self.audit_arm not in {"control", "treatment"}:
        raise RuntimeError("invalid collision-mask A/B arm")
    blocks = self.audit_protocol["collection"]["blocks"]
    if self.audit_case not in blocks:
        raise RuntimeError("invalid collision-mask A/B case")
    block = blocks[self.audit_case]
    self.agent_seed = int(_required("CNN_COLLISION_AB_AGENT_SEED"))
    expected_agent_seed = block.get("agent_seed", block.get("cnn_agent_seed"))
    if expected_agent_seed is None or self.agent_seed != int(expected_agent_seed):
        raise RuntimeError("collision-mask A/B agent seed mismatch")
    self.rng = np.random.default_rng(self.agent_seed)
    torch.manual_seed(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    checkpoint = Path(_required("CNN_COLLISION_AB_CHECKPOINT")).resolve()
    expected = self.audit_protocol["checkpoint"]
    if checkpoint != (ROOT / expected["path"]).resolve() or sha256_file(checkpoint) != expected["sha256"]:
        raise RuntimeError("collision-mask A/B checkpoint binding mismatch")
    payload = _load(checkpoint)
    for key, value in {"architecture": ARCHITECTURE, "stage": "task4", "replica": "r3", "stage_rounds": 800}.items():
        if payload.get(key) != value:
            raise RuntimeError(f"collision-mask A/B checkpoint identity mismatch: {key}")
    self.online_net.load_state_dict(payload["online_net"], strict=True)
    self.online_net.eval()
    for parameter in self.online_net.parameters():
        parameter.requires_grad = False
    self.audit_checkpoint_sha256 = sha256_file(checkpoint)
    self.audit_trace_path = trace_path(self.audit_protocol, self.audit_case, self.audit_arm)


def act(self, game_state: dict) -> str:
    if self._pending_decision is not None:
        raise RuntimeError("previous collision-mask A/B decision was not consumed")
    legal = legal_action_mask(game_state)
    legal_indices = np.flatnonzero(legal)
    q_values = np.full(len(ACTIONS), np.nan, dtype=np.float32)
    if not legal_indices.size:
        original_index = ACTIONS.index("WAIT")
    else:
        features = state_to_features(game_state)
        assert features is not None
        spatial, scalars = _tensors(features, self.device)
        with torch.no_grad():
            q_values = self.online_net(spatial, scalars)[0].detach().cpu().numpy()
        best = q_values[legal_indices].max()
        original_candidates = legal_indices[np.isclose(q_values[legal_indices], best)]
        original_index = int(self.rng.choice(original_candidates))

    effective = legal.copy()
    active_decision = self._active_bomb
    restriction_available = False
    fallback_used = False
    diagnostic = None
    if self.audit_arm == "treatment" and active_decision:
        diagnostic = immediate_collision_action_details(game_state)
        robust = np.asarray(diagnostic["robust_action_mask"], dtype=bool)
        filtered = legal & robust
        if filtered.any():
            effective = filtered
            restriction_available = not np.array_equal(effective, legal)
        else:
            fallback_used = True

    chosen_index = original_index
    action_changed = False
    if effective.any() and not effective[original_index]:
        effective_indices = np.flatnonzero(effective)
        best = q_values[effective_indices].max()
        candidates = effective_indices[np.isclose(q_values[effective_indices], best)]
        chosen_index = int(self.rng.choice(candidates))
        action_changed = True
    action = ACTIONS[chosen_index]
    self._pending_decision = {
        "round": int(game_state["round"]),
        "step": int(game_state["step"]),
        "action": action,
        "original_action": ACTIONS[original_index],
        "active_own_bomb": bool(active_decision),
        "restriction_available": bool(restriction_available),
        "fallback_used": bool(fallback_used),
        "action_changed": bool(action_changed),
        "robust_action_count": None if diagnostic is None else diagnostic["robust_action_count"],
    }
    return action


__all__ = ["act", "setup"]
