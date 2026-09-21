"""Frozen CNN inference plus passive post-bomb robust-movement labels."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from agent_code.model_a_cnn_n8.callbacks import _load, _tensors
from agent_code.model_a_cnn_n8.features import ACTIONS, state_to_features
from agent_code.model_a_cnn_n8.network import ARCHITECTURE, FullBoardDuelingCNN, torch
from agent_code.model_a_cnn_n8_robust_bomb_audit.callbacks import _opponent_reachability
from agent_code.model_a_dqn.features import (
    DELTAS,
    _blocked_positions,
    _can_survive_from,
    _hazard_schedule,
    _in_bounds,
    _physical_destination,
    legal_action_mask,
)

from .config import ROOT, load_protocol, sha256_file, trace_path


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for post-bomb movement audit")
    return value


def robust_postbomb_action_details(game_state: dict) -> dict:
    """Return actions retaining a full robust route through known hazards.

    This evaluates the actual current state after our bomb has been placed. An
    action is robust only if its destination is safe at offset one and at least
    one continuation survives the full known hazard horizon while avoiding all
    cells any currently visible opponent could occupy at each matching offset.
    """
    hazards, horizon = _hazard_schedule(game_state)
    blocked = _blocked_positions(game_state)
    opponent_reach = _opponent_reachability(game_state, horizon, set())
    robust_actions: list[str] = []
    frontier_by_action: dict[str, int] = {}
    for action in ACTIONS[:5]:
        destination = _physical_destination(game_state, action)
        if destination is None:
            continue
        if 1 in hazards.get(destination, set()) or destination in opponent_reach[1]:
            continue
        states = {destination}
        minimum_frontier = 1
        for time in range(2, horizon + 1):
            next_states = set()
            for position in states:
                candidates = [position]
                candidates.extend((position[0] + dx, position[1] + dy) for dx, dy in DELTAS.values())
                for candidate in candidates:
                    if not _in_bounds(game_state["field"], candidate):
                        continue
                    if game_state["field"][candidate] != 0:
                        continue
                    if candidate in blocked and candidate != position:
                        continue
                    if time in hazards.get(candidate, set()):
                        continue
                    if candidate in opponent_reach[time]:
                        continue
                    next_states.add(candidate)
            states = next_states
            minimum_frontier = min(minimum_frontier, len(states))
            if not states:
                break
        if states:
            robust_actions.append(action)
            frontier_by_action[action] = minimum_frontier
    return {
        "robust_actions": robust_actions,
        "robust_action_count": len(robust_actions),
        "robust_action_mask": [action in robust_actions for action in ACTIONS],
        "robust_minimum_frontier_by_action": frontier_by_action,
        "known_hazard_horizon": int(horizon),
    }


def immediate_collision_action_details(game_state: dict) -> dict:
    """Return ordinary survivable actions excluding next-step opponent reach.

    Unlike the full-horizon rule, opponent reachability is applied only to the
    next destination. Continuations use the existing known-explosion survival
    search, preserving aggressive routes after the immediate collision window.
    """
    hazards, horizon = _hazard_schedule(game_state)
    opponent_reach = _opponent_reachability(game_state, 1, set())
    robust_actions: list[str] = []
    for action in ACTIONS[:5]:
        destination = _physical_destination(game_state, action)
        if destination is None:
            continue
        if 1 in hazards.get(destination, set()) or destination in opponent_reach[1]:
            continue
        if _can_survive_from(game_state, destination, 1, hazards, horizon):
            robust_actions.append(action)
    return {
        "robust_actions": robust_actions,
        "robust_action_count": len(robust_actions),
        "robust_action_mask": [action in robust_actions for action in ACTIONS],
        "robust_minimum_frontier_by_action": {action: 1 for action in robust_actions},
        "known_hazard_horizon": int(horizon),
    }


def setup(self) -> None:
    if torch is None:
        raise RuntimeError("post-bomb movement audit requires PyTorch")
    if not self.train:
        raise RuntimeError("post-bomb movement collector must run in training mode")
    torch.set_num_threads(1)
    self.audit_protocol, self.audit_protocol_path, self.audit_protocol_sha256 = load_protocol()
    self.audit_case = _required("CNN_POSTBOMB_MOVEMENT_AUDIT_CASE")
    cases = self.audit_protocol["collection"]["cases"]
    if self.audit_case not in cases:
        raise RuntimeError("invalid post-bomb movement audit case")
    case = cases[self.audit_case]
    self.agent_seed = int(_required("CNN_POSTBOMB_MOVEMENT_AUDIT_AGENT_SEED"))
    if self.agent_seed != int(case["agent_seed"]):
        raise RuntimeError("post-bomb movement audit agent seed mismatch")
    self.rng = np.random.default_rng(self.agent_seed)
    torch.manual_seed(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    checkpoint = Path(_required("CNN_POSTBOMB_MOVEMENT_AUDIT_CHECKPOINT")).resolve()
    expected = self.audit_protocol["checkpoint"]
    if checkpoint != (ROOT / expected["path"]).resolve() or sha256_file(checkpoint) != expected["sha256"]:
        raise RuntimeError("post-bomb movement checkpoint binding mismatch")
    payload = _load(checkpoint)
    for key, value in {"architecture": ARCHITECTURE, "stage": "task4", "replica": "r3", "stage_rounds": 800}.items():
        if payload.get(key) != value:
            raise RuntimeError(f"post-bomb movement checkpoint identity mismatch: {key}")
    self.online_net.load_state_dict(payload["online_net"], strict=True)
    self.online_net.eval()
    for parameter in self.online_net.parameters():
        parameter.requires_grad = False
    self.audit_checkpoint_sha256 = sha256_file(checkpoint)
    self.audit_trace_path = trace_path(self.audit_protocol, self.audit_case)


def act(self, game_state: dict) -> str:
    if self._pending_decision is not None:
        raise RuntimeError("previous post-bomb movement decision was not consumed")
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
    diagnostic = None
    if self._active_bomb is not None:
        rule_name = self.audit_protocol["candidate_rule"]["name"]
        if rule_name == "full_horizon_postbomb_movement_reachability":
            diagnostic = robust_postbomb_action_details(game_state)
        elif rule_name == "immediate_opponent_collision_then_known_hazard_survival":
            diagnostic = immediate_collision_action_details(game_state)
        else:
            raise RuntimeError(f"unsupported post-bomb movement rule: {rule_name}")
    self._pending_decision = {
        "round": int(game_state["round"]),
        "step": int(game_state["step"]),
        "action": action,
        "position": [int(v) for v in game_state["self"][3]],
        "diagnostic": diagnostic,
        "chosen_action_robust": None if diagnostic is None else action in diagnostic["robust_actions"],
        "avoidable_deviation": None if diagnostic is None else bool(
            diagnostic["robust_action_count"] > 0 and action not in diagnostic["robust_actions"]
        ),
    }
    return action


__all__ = ["act", "immediate_collision_action_details", "robust_postbomb_action_details", "setup"]
