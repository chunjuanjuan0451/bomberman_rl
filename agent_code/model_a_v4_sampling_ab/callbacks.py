"""Callbacks for matched n=8 replay training and passive endpoint evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np

from agent_code.model_a_dqn import callbacks as v4_callbacks
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE, _load_network_state
from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import DuelingDQN, torch
from agent_code.model_a_v7b.tactical import evaluate_bomb

from .config import ARMS, REPLICAS, ROOT, checkpoint_path, load_protocol, sha256_file


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for model_a_v4_sampling_ab")
    return value


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "online_net" not in payload:
        raise RuntimeError(f"invalid sampling-distribution checkpoint: {path}")
    return payload


def _validate_source(payload: dict, protocol: dict) -> None:
    lineage = protocol["source_parent"]["lineage"]
    expected = {
        "architecture": MODEL_ARCHITECTURE, "protocol_sha256": lineage["protocol_sha256"],
        "arm": "curriculum", "replica": "r2", "stage_id": "task2",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"sampling-distribution source-r2 {key} mismatch")


def _validate_endpoint(payload: dict, protocol_hash: str, arm: str, replica: str) -> None:
    expected = {
        "architecture": MODEL_ARCHITECTURE, "protocol_sha256": protocol_hash,
        "arm": arm, "replica": replica, "stage_id": "task4c_sampling_ab",
        "stage_completed_rounds": 100, "all_action_return_horizon": 8,
        "sampling_mode": arm,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"sampling-distribution endpoint {key} mismatch")


def _load_bound_network(self, path: Path, label: str) -> dict:
    payload = _torch_load(path)
    if label == "source-r2":
        if sha256_file(path) != self.protocol["source_parent"]["sha256"]:
            raise RuntimeError("sampling-distribution source-r2 hash mismatch")
        _validate_source(payload, self.protocol)
    elif label == "v4":
        if sha256_file(path) != self.protocol["frozen_v4_baseline"]["sha256"]:
            raise RuntimeError("sampling-distribution frozen-v4 hash mismatch")
    else:
        arm, replica = label.rsplit("-", 1)
        if arm not in ARMS or replica not in REPLICAS or path != checkpoint_path(self.protocol, arm, replica):
            raise RuntimeError("sampling-distribution endpoint identity mismatch")
        _validate_endpoint(payload, self.protocol_sha256, arm, replica)
    if _load_network_state(self.online_net, payload["online_net"]):
        raise RuntimeError("sampling-distribution checkpoint required schema migration")
    return payload


def setup(self):
    if torch is None:
        raise RuntimeError("model_a_v4_sampling_ab requires PyTorch")
    if not self.train:
        raise RuntimeError("sampling-distribution agent uses training mode for event delivery")
    torch.set_num_threads(1)
    self.protocol, self.protocol_sha256 = load_protocol(Path(_required("MODEL_A_SAMPLING_PROTOCOL_PATH")))
    self.run_mode = _required("MODEL_A_SAMPLING_RUN_MODE")
    if self.run_mode not in {"train", "evaluate"}:
        raise RuntimeError("invalid sampling-distribution run mode")
    self.agent_seed = int(_required("MODEL_A_SAMPLING_SEED"))
    self.rng = np.random.default_rng(self.agent_seed)
    torch.manual_seed(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = DuelingDQN().to(self.device)
    if self.run_mode == "train":
        self.arm = _required("MODEL_A_SAMPLING_ARM")
        self.replica = _required("MODEL_A_SAMPLING_REPLICA")
        if self.arm not in ARMS or self.replica not in REPLICAS:
            raise RuntimeError("invalid sampling-distribution training identity")
        expected_seed = int(self.protocol["training"]["seeds_by_replica"][self.replica]["agent_seed"])
        if self.agent_seed != expected_seed:
            raise RuntimeError("sampling-distribution training seed mismatch")
        self.checkpoint_path = Path(_required("MODEL_A_SAMPLING_CHECKPOINT_PATH")).resolve()
        if self.checkpoint_path != checkpoint_path(self.protocol, self.arm, self.replica) or self.checkpoint_path.exists():
            raise RuntimeError("sampling-distribution output mismatch or overwrite attempt")
        parent = Path(_required("MODEL_A_SAMPLING_PARENT_PATH")).resolve()
        expected_parent = (ROOT / self.protocol["source_parent"]["path"]).resolve()
        if parent != expected_parent:
            raise RuntimeError("sampling-distribution parent path mismatch")
        self.resume_checkpoint = _load_bound_network(self, parent, "source-r2")
        self.parent_sha256 = sha256_file(parent)
        self.completed_rounds = int(self.resume_checkpoint.get("completed_rounds", 0))
        self.stage_start_rounds = self.completed_rounds
    else:
        self.eval_label = _required("MODEL_A_SAMPLING_EVAL_LABEL")
        self.eval_stratum = _required("MODEL_A_SAMPLING_EVAL_STRATUM")
        path = Path(_required("MODEL_A_SAMPLING_CHECKPOINT_PATH")).resolve()
        _load_bound_network(self, path, self.eval_label)
        self.completed_rounds = 0
    self.online_net.eval() if self.run_mode == "evaluate" else self.online_net.train()


def _greedy_action(self, game_state: dict) -> int:
    mask = legal_action_mask(game_state)
    legal = np.flatnonzero(mask)
    if not len(legal):
        return ACTIONS.index("WAIT")
    local, global_features = state_to_features(game_state)
    with torch.no_grad():
        q = self.online_net(
            torch.as_tensor(local[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
        )[0].cpu().numpy()
    best = q[legal].max()
    choices = legal[np.isclose(q[legal], best)]
    return int(self.rng.choice(choices))


def _nearest_distance(game_state: dict | None) -> float:
    if game_state is None or not game_state.get("others"):
        return -1.0
    x, y = game_state["self"][3]
    return float(min(abs(other[3][0] - x) + abs(other[3][1] - y) for other in game_state["others"]))


def act(self, game_state: dict) -> str:
    if self.run_mode == "train":
        return v4_callbacks.act(self, game_state)
    if self._pending_eval_decision is not None:
        raise RuntimeError("sampling-distribution evaluation did not consume previous decision")
    action = _greedy_action(self, game_state)
    mask = legal_action_mask(game_state)
    bomb_legal = bool(mask[ACTIONS.index("BOMB")])
    tactics = evaluate_bomb(game_state, perf_counter() + self.eval_oracle_deadline) if bomb_legal else None
    self._pending_eval_decision = {
        "round": int(game_state["round"]), "step": int(game_state["step"]), "action": action,
        "nearest": _nearest_distance(game_state), "bomb_legal": bomb_legal,
        "oracle_evaluated": bool(bomb_legal and tactics is not None),
        "oracle_timeout": bool(bomb_legal and tactics is None),
        "affected_opponents": 0 if tactics is None else int(tactics.affected_opponents),
        "max_space_reduction": 0.0 if tactics is None else float(tactics.max_space_reduction),
        "guaranteed_traps": 0 if tactics is None else int(tactics.guaranteed_traps),
    }
    return ACTIONS[action]
