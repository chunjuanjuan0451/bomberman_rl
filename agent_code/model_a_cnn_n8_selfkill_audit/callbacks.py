"""Frozen CNN inference with passive escape-risk instrumentation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from agent_code.model_a_cnn_n8.callbacks import _load, _tensors
from agent_code.model_a_cnn_n8.features import ACTIONS, state_to_features
from agent_code.model_a_cnn_n8.network import ARCHITECTURE, FullBoardDuelingCNN, torch
from agent_code.model_a_dqn.features import (
    BOMB_TIMER,
    DELTAS,
    INF_TIME,
    _blocked_positions,
    _can_survive_from,
    _escape_options_after_bomb,
    _hazard_schedule,
    _in_bounds,
    _physical_destination,
    danger_time_map,
    legal_action_mask,
)

from .config import ROOT, load_protocol, sha256_file, trace_path


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for model_a_cnn_n8_selfkill_audit")
    return value


def _nearest_opponent_distance(game_state: dict) -> int:
    position = tuple(game_state["self"][3])
    if not game_state["others"]:
        return -1
    return min(
        abs(int(other[3][0]) - position[0]) + abs(int(other[3][1]) - position[1])
        for other in game_state["others"]
    )


def _bomb_first_moves(game_state: dict) -> list[tuple[int, int]]:
    start = tuple(game_state["self"][3])
    hazards, horizon = _hazard_schedule(game_state, (start, BOMB_TIMER))
    blocked = _blocked_positions(game_state) | {start}
    result = []
    for dx, dy in DELTAS.values():
        first = (start[0] + dx, start[1] + dy)
        if not _in_bounds(game_state["field"], first) or game_state["field"][first] != 0:
            continue
        if first in blocked or 2 in hazards.get(first, set()):
            continue
        if _can_survive_from(game_state, first, 2, hazards, horizon, {start}):
            result.append(first)
    return result


def _survival_diagnostics(game_state: dict, legal: np.ndarray) -> dict:
    position = tuple(game_state["self"][3])
    hazards, horizon = _hazard_schedule(game_state)
    survivable_moves = []
    for action in ACTIONS[:4]:
        destination = _physical_destination(game_state, action)
        survivable_moves.append(bool(
            destination is not None
            and _can_survive_from(game_state, destination, 1, hazards, horizon)
        ))
    wait_survives = bool(_can_survive_from(game_state, position, 1, hazards, horizon))
    fallback_used = not any(survivable_moves) and not wait_survives
    danger = int(danger_time_map(game_state)[position])
    first_moves = _bomb_first_moves(game_state) if bool(legal[5]) else []
    opponents = [tuple(other[3]) for other in game_state["others"]]
    contested = [
        any(abs(first[0] - other[0]) + abs(first[1] - other[1]) <= 1 for other in opponents)
        for first in first_moves
    ]
    shortest_escape = _escape_options_after_bomb(game_state)[1] if bool(legal[5]) else INF_TIME
    return {
        "position": list(position),
        "current_danger_time": danger,
        "known_survivable_nonbomb_actions": int(sum(survivable_moves) + wait_survives),
        "survivable_move_mask": survivable_moves,
        "wait_survives": wait_survives,
        "fallback_used": fallback_used,
        "bomb_safe_first_moves": [list(item) for item in first_moves],
        "bomb_safe_first_move_count": len(first_moves),
        "bomb_contestable_first_moves": contested,
        "bomb_any_first_move_contestable": any(contested),
        "bomb_shortest_escape_moves": None if shortest_escape >= INF_TIME else int(shortest_escape),
        "nearest_opponent_distance": _nearest_opponent_distance(game_state),
        "visible_bombs": [[list(position), int(timer)] for position, timer in game_state["bombs"]],
        "opponent_positions": [list(item) for item in opponents],
    }


def setup(self) -> None:
    if torch is None:
        raise RuntimeError("CNN self-kill audit requires PyTorch")
    if not self.train:
        raise RuntimeError("CNN self-kill audit collector must be the training-mode agent")
    torch.set_num_threads(1)
    self.audit_protocol, self.audit_protocol_path, self.audit_protocol_sha256 = load_protocol()
    self.audit_case = _required("CNN_SELFKILL_AUDIT_CASE")
    cases = self.audit_protocol["collection"]["cases"]
    if self.audit_case not in cases:
        raise RuntimeError("invalid CNN self-kill audit case")
    case = cases[self.audit_case]
    self.agent_seed = int(_required("CNN_SELFKILL_AUDIT_AGENT_SEED"))
    if self.agent_seed != int(case["agent_seed"]):
        raise RuntimeError("CNN self-kill audit agent seed mismatch")
    self.rng = np.random.default_rng(self.agent_seed)
    torch.manual_seed(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    checkpoint = Path(_required("CNN_SELFKILL_AUDIT_CHECKPOINT")).resolve()
    expected = self.audit_protocol["checkpoint"]
    if checkpoint != (ROOT / expected["path"]).resolve() or sha256_file(checkpoint) != expected["sha256"]:
        raise RuntimeError("CNN self-kill audit checkpoint binding mismatch")
    payload = _load(checkpoint)
    for key, value in {
        "architecture": ARCHITECTURE,
        "stage": "task4",
        "replica": "r3",
        "stage_rounds": 800,
    }.items():
        if payload.get(key) != value:
            raise RuntimeError(f"CNN self-kill audit checkpoint identity mismatch: {key}")
    self.online_net.load_state_dict(payload["online_net"], strict=True)
    self.online_net.eval()
    for parameter in self.online_net.parameters():
        parameter.requires_grad = False
    self.audit_checkpoint_sha256 = sha256_file(checkpoint)
    self.audit_trace_path = trace_path(self.audit_protocol, self.audit_case)


def act(self, game_state: dict) -> str:
    if self._pending_decision is not None:
        raise RuntimeError("previous CNN self-kill decision was not consumed")
    legal = legal_action_mask(game_state)
    legal_indices = np.flatnonzero(legal)
    if not legal_indices.size:
        action_index = ACTIONS.index("WAIT")
        q_values = np.full(len(ACTIONS), np.nan, dtype=np.float32)
    else:
        features = state_to_features(game_state)
        assert features is not None
        spatial, scalars = _tensors(features, self.device)
        with torch.no_grad():
            q_values = self.online_net(spatial, scalars)[0].detach().cpu().numpy()
        best = q_values[legal_indices].max()
        candidates = legal_indices[np.isclose(q_values[legal_indices], best)]
        action_index = int(self.rng.choice(candidates))
    ordered = sorted((float(q_values[index]), int(index)) for index in legal_indices)[::-1]
    q_gap = None if len(ordered) < 2 else ordered[0][0] - ordered[1][0]
    row = {
        "round": int(game_state["round"]),
        "step": int(game_state["step"]),
        "action": ACTIONS[action_index],
        "legal_mask": legal.astype(bool).tolist(),
        "q_values": [None if not np.isfinite(value) else float(value) for value in q_values],
        "chosen_q_gap": q_gap,
        "bomb_available": bool(game_state["self"][2]),
        **_survival_diagnostics(game_state, legal),
    }
    self._pending_decision = row
    return ACTIONS[action_index]


__all__ = ["act", "setup"]
