"""Frozen-v4 callbacks that record WAIT Q gaps without changing actions."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path

import numpy as np

from agent_code.model_a_dqn import callbacks as base
from agent_code.model_a_dqn.features import ACTIONS, danger_time_map, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import torch


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _flush(self) -> None:
    output = getattr(self, "wait_audit_output", None)
    payload = getattr(self, "wait_audit", None)
    if output is not None and payload is not None:
        _atomic_json(output, payload)


def setup(self) -> None:
    if self.train:
        raise RuntimeError("WAIT Q-gap audit is evaluation-only")
    base.setup(self)
    protocol_path = Path(os.environ["MODEL_A_WAIT_AUDIT_PROTOCOL_PATH"]).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("kind") != "model-a-v4-wait-qgap-audit" or protocol.get("training_allowed") is not False:
        raise RuntimeError("invalid WAIT Q-gap audit protocol")
    self.wait_audit_output = Path(os.environ["MODEL_A_WAIT_AUDIT_OUTPUT"]).resolve()
    self.wait_audit = {
        "schema_version": 1,
        "kind": "model-a-v4-wait-qgap-case-diagnostic",
        "protocol_id": protocol["protocol_id"],
        "case_id": os.environ["MODEL_A_WAIT_AUDIT_CASE_ID"],
        "policy_updates": 0,
        "actions_changed_by_audit": 0,
        "steps": 0,
        "action_counts": {action: 0 for action in ACTIONS},
        "wait_records": [],
    }
    self.wait_audit_previous_action = None
    self.wait_audit_previous_position = None
    atexit.register(_flush, self)


def act(self, game_state: dict) -> str:
    legal = legal_action_mask(game_state)
    legal_indices = np.flatnonzero(legal)
    if not legal_indices.size:
        action = "WAIT"
        q_values = np.full(len(ACTIONS), np.nan, dtype=np.float32)
    else:
        features = state_to_features(game_state)
        assert features is not None
        local, global_features = features
        with torch.no_grad():
            q_values = self.online_net(
                torch.as_tensor(local[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
            )[0].cpu().numpy()
        best_value = q_values[legal_indices].max()
        candidates = legal_indices[np.isclose(q_values[legal_indices], best_value)]
        action = ACTIONS[int(self.rng.choice(candidates))]

    position = tuple(game_state["self"][3])
    audit = self.wait_audit
    audit["steps"] += 1
    audit["action_counts"][action] += 1
    if action == "WAIT":
        safe_moves = np.flatnonzero(legal[:4])
        best_move_index = None
        best_move_q = None
        gap = None
        if safe_moves.size:
            best_move_q = float(q_values[safe_moves].max())
            best_candidates = safe_moves[np.isclose(q_values[safe_moves], best_move_q)]
            best_move_index = int(best_candidates[0])
            gap = float(q_values[4] - best_move_q)
        danger = danger_time_map(game_state)
        audit["wait_records"].append({
            "round": int(game_state["round"]),
            "step": int(game_state["step"]),
            "q_wait": None if not np.isfinite(q_values[4]) else float(q_values[4]),
            "best_safe_move": None if best_move_index is None else ACTIONS[best_move_index],
            "best_safe_move_q": best_move_q,
            "q_gap": gap,
            "safe_move_count": int(safe_moves.size),
            "current_danger_time": int(danger[position]),
            "bombs_present": len(game_state["bombs"]),
            "coins_remaining": len(game_state["coins"]),
            "repeat_wait_same_position": bool(
                self.wait_audit_previous_action == "WAIT"
                and self.wait_audit_previous_position == position
            ),
        })
    self.wait_audit_previous_action = action
    self.wait_audit_previous_position = position
    if int(game_state["step"]) % 50 == 0:
        _flush(self)
    return action
