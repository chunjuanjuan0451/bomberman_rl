"""Frozen CNN inference plus a passive adversarial opponent-reachability test."""

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
    _escape_options_after_bomb,
    _hazard_schedule,
    _in_bounds,
    blast_positions,
    danger_time_map,
    legal_action_mask,
)

from .config import ROOT, load_protocol, sha256_file, trace_path


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for robust-bomb audit")
    return value


def _opponent_reachability(game_state: dict, horizon: int,
                           extra_blocked: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Over-approximate every tile any opponent can occupy after each action."""
    field = game_state["field"]
    blocked = _blocked_positions(game_state) | extra_blocked
    positions = {tuple(other[3]) for other in game_state["others"]}
    timeline = [set(positions)]
    for _ in range(horizon):
        next_positions = set()
        for position in positions:
            candidates = [position]
            candidates.extend((position[0] + dx, position[1] + dy) for dx, dy in DELTAS.values())
            for candidate in candidates:
                if not _in_bounds(field, candidate) or field[candidate] != 0:
                    continue
                # Opponents may remain on their current bomb tile but cannot enter another bomb.
                if candidate in blocked and candidate != position:
                    continue
                next_positions.add(candidate)
        positions = next_positions
        timeline.append(set(positions))
    return timeline


def robust_bomb_escape_details(game_state: dict) -> dict:
    """Test whether one escape path survives all possible opponent occupancies.

    The action at offset one is the BOMB placement itself. The first escape move
    lands at offset two, matching the existing bomb-mask timing convention.
    This is evaluated passively and never changes the selected action.
    """
    start = tuple(game_state["self"][3])
    hazards, horizon = _hazard_schedule(game_state, (start, BOMB_TIMER))
    own_blast = set(blast_positions(game_state["field"], start))
    blocked = _blocked_positions(game_state) | {start}
    opponent_reach = _opponent_reachability(game_state, horizon, {start})
    states = {(start, False, None)}
    minimum_frontier = len(states)
    for time in range(2, horizon + 1):
        next_states = set()
        for position, left_blast, first_move in states:
            # The existing mask suppresses WAIT after a fresh bomb whenever an
            # escape move exists, so the first post-placement action must move.
            candidates = [] if time == 2 else [position]
            candidates.extend((position[0] + dx, position[1] + dy) for dx, dy in DELTAS.values())
            for candidate in candidates:
                if not _in_bounds(game_state["field"], candidate) or game_state["field"][candidate] != 0:
                    continue
                if candidate in blocked and candidate != position:
                    continue
                if time in hazards.get(candidate, set()):
                    continue
                if candidate in opponent_reach[time]:
                    continue
                escaped = left_blast or candidate not in own_blast
                route_first_move = candidate if time == 2 else first_move
                next_states.add((candidate, escaped, route_first_move))
        states = next_states
        minimum_frontier = min(minimum_frontier, len(states))
        if not states:
            break
    surviving = [state for state in states if state[1]]
    robust_first_moves = {state[2] for state in surviving}
    allowed = bool(surviving)
    safe_first_moves, shortest = _escape_options_after_bomb(game_state)
    nearest = -1
    if game_state["others"]:
        nearest = min(
            abs(int(other[3][0]) - start[0]) + abs(int(other[3][1]) - start[1])
            for other in game_state["others"]
        )
    return {
        "robust_bomb_allowed": allowed,
        "robust_first_move_count": len(robust_first_moves),
        "robust_minimum_frontier": minimum_frontier,
        "base_safe_first_move_count": int(safe_first_moves),
        "base_shortest_escape_moves": None if shortest >= INF_TIME else int(shortest),
        "nearest_opponent_distance": nearest,
        "opponent_count": len(game_state["others"]),
        "position": [int(start[0]), int(start[1])],
        "current_danger_time": int(danger_time_map(game_state)[start]),
    }


def setup(self) -> None:
    if torch is None:
        raise RuntimeError("robust-bomb audit requires PyTorch")
    if not self.train:
        raise RuntimeError("robust-bomb collector must be the training-mode agent")
    torch.set_num_threads(1)
    self.audit_protocol, self.audit_protocol_path, self.audit_protocol_sha256 = load_protocol()
    self.audit_case = _required("CNN_ROBUST_BOMB_AUDIT_CASE")
    cases = self.audit_protocol["collection"]["cases"]
    if self.audit_case not in cases:
        raise RuntimeError("invalid robust-bomb audit case")
    case = cases[self.audit_case]
    self.agent_seed = int(_required("CNN_ROBUST_BOMB_AUDIT_AGENT_SEED"))
    if self.agent_seed != int(case["agent_seed"]):
        raise RuntimeError("robust-bomb audit agent seed mismatch")
    self.rng = np.random.default_rng(self.agent_seed)
    torch.manual_seed(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    checkpoint = Path(_required("CNN_ROBUST_BOMB_AUDIT_CHECKPOINT")).resolve()
    expected = self.audit_protocol["checkpoint"]
    if checkpoint != (ROOT / expected["path"]).resolve() or sha256_file(checkpoint) != expected["sha256"]:
        raise RuntimeError("robust-bomb audit checkpoint binding mismatch")
    payload = _load(checkpoint)
    for key, value in {"architecture": ARCHITECTURE, "stage": "task4", "replica": "r3", "stage_rounds": 800}.items():
        if payload.get(key) != value:
            raise RuntimeError(f"robust-bomb checkpoint identity mismatch: {key}")
    self.online_net.load_state_dict(payload["online_net"], strict=True)
    self.online_net.eval()
    for parameter in self.online_net.parameters():
        parameter.requires_grad = False
    self.audit_checkpoint_sha256 = sha256_file(checkpoint)
    self.audit_trace_path = trace_path(self.audit_protocol, self.audit_case)


def act(self, game_state: dict) -> str:
    if self._pending_decision is not None:
        raise RuntimeError("previous robust-bomb decision was not consumed")
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
    action = ACTIONS[action_index]
    diagnostic = robust_bomb_escape_details(game_state) if action == "BOMB" else None
    ordered = sorted((float(q_values[index]), int(index)) for index in legal_indices)[::-1]
    self._pending_decision = {
        "round": int(game_state["round"]),
        "step": int(game_state["step"]),
        "action": action,
        "diagnostic": diagnostic,
        "q_gap": None if len(ordered) < 2 else ordered[0][0] - ordered[1][0],
        "bomb_q": None if not np.isfinite(q_values[5]) else float(q_values[5]),
    }
    return action


__all__ = ["act", "robust_bomb_escape_details", "setup"]
