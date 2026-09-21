"""Frozen source-r2 policy with passive reward and attack-chain instrumentation."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np

from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE, _load_network_state
from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import DuelingDQN, torch
from agent_code.model_a_v7b.tactical import evaluate_bomb

from .config import CASES, ROOT, case_stratum, load_protocol, sha256_file, trace_path


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for model_a_v4_system_audit")
    return value


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "online_net" not in payload or "target_net" not in payload:
        raise RuntimeError(f"invalid source-r2 audit checkpoint: {path}")
    return payload


def _q_values(network, game_state: dict, device) -> tuple[np.ndarray, np.ndarray]:
    local, global_features = state_to_features(game_state)
    with torch.no_grad():
        values = network(
            torch.as_tensor(local[None], dtype=torch.float32, device=device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=device),
        )[0].cpu().numpy().astype(np.float32)
    return values, global_features.astype(np.float32)


def _nearest_distance(game_state: dict | None) -> float:
    if game_state is None or not game_state.get("others"):
        return -1.0
    x, y = game_state["self"][3]
    return float(min(abs(other[3][0] - x) + abs(other[3][1] - y) for other in game_state["others"]))


def setup(self):
    if torch is None:
        raise RuntimeError("model_a_v4_system_audit requires PyTorch")
    if not self.train:
        raise RuntimeError("training-system collector must be the sole training-mode agent")
    torch.set_num_threads(1)
    protocol_path = Path(_required("MODEL_A_SYSTEM_AUDIT_PROTOCOL_PATH")).resolve()
    self.protocol, self.protocol_sha256 = load_protocol(protocol_path)
    self.case_label = _required("MODEL_A_SYSTEM_AUDIT_CASE")
    if self.case_label not in CASES:
        raise RuntimeError("invalid training-system audit case")
    self.stratum = case_stratum(self.protocol, self.case_label)
    expected = self.protocol["collection"]["cases"][self.case_label]
    self.agent_seed = int(_required("MODEL_A_SYSTEM_AUDIT_SEED"))
    if self.agent_seed != int(expected["agent_seed"]):
        raise RuntimeError("training-system audit agent seed mismatch")
    self.rng = np.random.default_rng(self.agent_seed)
    self.collection_epsilon = float(self.protocol["collection"]["epsilon"])
    self.oracle_deadline_seconds = float(self.protocol["collection"]["oracle_deadline_seconds"])
    self.device = torch.device("cpu")
    torch.manual_seed(self.agent_seed)
    self.online_net = DuelingDQN().to(self.device)
    self.target_net = DuelingDQN().to(self.device)
    parent = Path(_required("MODEL_A_SYSTEM_AUDIT_PARENT_PATH")).resolve()
    bound = (ROOT / self.protocol["source_parent"]["path"]).resolve()
    if parent != bound or sha256_file(parent) != self.protocol["source_parent"]["sha256"]:
        raise RuntimeError("training-system audit source-r2 binding mismatch")
    payload = _torch_load(parent)
    lineage = self.protocol["source_parent"]["lineage"]
    for key, value in {
        "architecture": MODEL_ARCHITECTURE, "protocol_sha256": lineage["protocol_sha256"],
        "arm": "curriculum", "replica": "r2", "stage_id": "task2",
    }.items():
        if payload.get(key) != value:
            raise RuntimeError(f"training-system source parent {key} mismatch")
    if _load_network_state(self.online_net, payload["online_net"]):
        raise RuntimeError("training-system online network required schema migration")
    if _load_network_state(self.target_net, payload["target_net"]):
        raise RuntimeError("training-system target network required schema migration")
    self.online_net.eval(); self.target_net.eval()
    for network in (self.online_net, self.target_net):
        for parameter in network.parameters():
            parameter.requires_grad = False
    self.parent_sha256 = sha256_file(parent)
    self.trace_path = trace_path(self.protocol, self.case_label)


def act(self, game_state: dict) -> str:
    if self._pending_decision is not None:
        raise RuntimeError("training-system event callback did not consume previous decision")
    mask = legal_action_mask(game_state)
    legal_indices = np.flatnonzero(mask)
    online_q, global_features = _q_values(self.online_net, game_state, self.device)
    target_q, _ = _q_values(self.target_net, game_state, self.device)
    if legal_indices.size == 0:
        action_index = ACTIONS.index("WAIT")
    elif self.rng.random() < self.collection_epsilon:
        action_index = int(self.rng.choice(legal_indices))
    else:
        best = online_q[legal_indices].max()
        choices = legal_indices[np.isclose(online_q[legal_indices], best)]
        action_index = int(self.rng.choice(choices))
    bomb_index = ACTIONS.index("BOMB")
    bomb_legal = bool(mask[bomb_index])
    tactics = None
    oracle_timeout = False
    if bomb_legal:
        tactics = evaluate_bomb(game_state, perf_counter() + self.oracle_deadline_seconds)
        oracle_timeout = tactics is None
    self._pending_decision = {
        "round": int(game_state["round"]), "step": int(game_state["step"]),
        "action": action_index, "legal_mask": mask.astype(np.bool_),
        "online_q_values": online_q, "target_q_values": target_q,
        "global_features": global_features,
        "opponent_count": len(game_state["others"]),
        "nearest_opponent_distance": _nearest_distance(game_state),
        "bomb_legal": bomb_legal, "oracle_evaluated": bool(bomb_legal and tactics is not None),
        "oracle_timeout": oracle_timeout,
        "guaranteed_traps": 0 if tactics is None else int(tactics.guaranteed_traps),
        "affected_opponents": 0 if tactics is None else int(tactics.affected_opponents),
        "max_space_reduction": 0.0 if tactics is None else float(tactics.max_space_reduction),
        "own_bottleneck": 0 if tactics is None else int(tactics.own_bottleneck),
        "own_terminal_positions": 0 if tactics is None else int(tactics.own_terminal_positions),
    }
    return ACTIONS[action_index]
