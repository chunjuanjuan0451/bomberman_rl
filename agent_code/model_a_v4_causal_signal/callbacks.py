"""Frozen source-r2 behavior with a passive counterfactual BOMB oracle."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np

from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE, _load_network_state
from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import DuelingDQN, torch
from agent_code.model_a_v7b.tactical import BombTactics, evaluate_bomb

from .config import CASES, ROOT, case_stratum, load_protocol, sha256_file, trace_path


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for model_a_v4_causal_signal")
    return value


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "online_net" not in payload:
        raise RuntimeError(f"invalid source-r2 checkpoint: {path}")
    return payload


def strict_opportunity(tactics: BombTactics | None) -> bool:
    """Use the exact conservative condition from the terminal v7b gate."""
    return bool(
        tactics is not None
        and tactics.guaranteed_traps >= 1
        and tactics.own_bottleneck >= 2
        and tactics.own_terminal_positions >= 3
    )


def setup(self):
    if torch is None:
        raise RuntimeError("model_a_v4_causal_signal requires PyTorch")
    if not self.train:
        raise RuntimeError("counterfactual signal collector must be the sole training-mode agent")
    torch.set_num_threads(1)
    protocol_path = Path(_required("MODEL_A_CAUSAL_SIGNAL_PROTOCOL_PATH")).resolve()
    self.protocol, self.protocol_sha256 = load_protocol(protocol_path)
    self.case_label = _required("MODEL_A_CAUSAL_SIGNAL_CASE")
    if self.case_label not in CASES:
        raise RuntimeError("invalid counterfactual signal collection case")
    self.stratum = case_stratum(self.protocol, self.case_label)
    expected = self.protocol["collection"]["cases"][self.case_label]
    self.agent_seed = int(_required("MODEL_A_CAUSAL_SIGNAL_SEED"))
    if self.agent_seed != int(expected["agent_seed"]):
        raise RuntimeError("counterfactual signal agent seed mismatch")
    self.rng = np.random.default_rng(self.agent_seed)
    self.collection_epsilon = float(self.protocol["collection"]["epsilon"])
    self.oracle_deadline_seconds = float(self.protocol["collection"]["oracle_deadline_seconds"])
    self.device = torch.device("cpu")
    torch.manual_seed(self.agent_seed)
    self.online_net = DuelingDQN().to(self.device)
    parent = Path(_required("MODEL_A_CAUSAL_SIGNAL_PARENT_PATH")).resolve()
    bound = (ROOT / self.protocol["source_parent"]["path"]).resolve()
    if parent != bound or sha256_file(parent) != self.protocol["source_parent"]["sha256"]:
        raise RuntimeError("counterfactual signal source-r2 binding mismatch")
    payload = _torch_load(parent)
    lineage = self.protocol["source_parent"]["lineage"]
    for key, value in {
        "architecture": MODEL_ARCHITECTURE,
        "protocol_sha256": lineage["protocol_sha256"],
        "arm": "curriculum",
        "replica": "r2",
        "stage_id": "task2",
    }.items():
        if payload.get(key) != value:
            raise RuntimeError(f"counterfactual signal source-r2 {key} mismatch")
    if _load_network_state(self.online_net, payload["online_net"]):
        raise RuntimeError("counterfactual signal source-r2 unexpectedly required schema migration")
    self.online_net.eval()
    for parameter in self.online_net.parameters():
        parameter.requires_grad = False
    self.parent_sha256 = sha256_file(parent)
    self.trace_path = trace_path(self.protocol, self.case_label)


def _q_values(self, game_state: dict) -> np.ndarray:
    local, global_features = state_to_features(game_state)
    with torch.no_grad():
        local_tensor = torch.as_tensor(local[None], dtype=torch.float32, device=self.device)
        global_tensor = torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device)
        return self.online_net(local_tensor, global_tensor)[0].cpu().numpy()


def act(self, game_state: dict) -> str:
    if self._pending_decision is not None:
        raise RuntimeError("counterfactual signal event callback did not consume the previous decision")
    mask = legal_action_mask(game_state)
    legal_indices = np.flatnonzero(mask)
    if legal_indices.size == 0:
        action_index = ACTIONS.index("WAIT")
    elif self.rng.random() < self.collection_epsilon:
        action_index = int(self.rng.choice(legal_indices))
    else:
        q_values = _q_values(self, game_state)
        best = q_values[legal_indices].max()
        choices = legal_indices[np.isclose(q_values[legal_indices], best)]
        action_index = int(self.rng.choice(choices))

    bomb_index = ACTIONS.index("BOMB")
    bomb_legal = bool(mask[bomb_index])
    tactics = None
    oracle_timeout = False
    if bomb_legal:
        tactics = evaluate_bomb(game_state, perf_counter() + self.oracle_deadline_seconds)
        oracle_timeout = tactics is None
    strict = bomb_legal and strict_opportunity(tactics)
    self._pending_decision = {
        "round": int(game_state["round"]),
        "step": int(game_state["step"]),
        "action": action_index,
        "legal_mask": mask.astype(np.bool_),
        "bomb_legal": bomb_legal,
        "oracle_evaluated": bool(bomb_legal and tactics is not None),
        "oracle_timeout": oracle_timeout,
        "strict_opportunity": strict,
        "guaranteed_traps": 0 if tactics is None else int(tactics.guaranteed_traps),
        "affected_opponents": 0 if tactics is None else int(tactics.affected_opponents),
        "max_space_reduction": 0.0 if tactics is None else float(tactics.max_space_reduction),
        "own_bottleneck": 0 if tactics is None else int(tactics.own_bottleneck),
        "own_terminal_positions": 0 if tactics is None else int(tactics.own_terminal_positions),
    }
    return ACTIONS[action_index]
