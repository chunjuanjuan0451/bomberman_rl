"""Shared frozen-v4 loader for the s139000 opponent-shift evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from agent_code.model_a_dqn import callbacks as v4_callbacks
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE, _load_network_state
from agent_code.model_a_dqn.network import DuelingDQN, torch


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_KIND = "model-a-v4-opponent-shift"


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for opponent-shift evaluation")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "online_net" not in payload:
        raise RuntimeError(f"invalid frozen opponent-shift checkpoint: {path}")
    return payload


def setup_frozen(self, label: str) -> None:
    if self.train:
        raise RuntimeError("s139000 wrappers are evaluation-only")
    if torch is None:
        raise RuntimeError("s139000 requires PyTorch")
    protocol_path = Path(_required("MODEL_A_OPPONENT_SHIFT_PROTOCOL_PATH")).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("kind") != PROTOCOL_KIND or protocol.get("training_allowed") is not False:
        raise RuntimeError("invalid opponent-shift protocol")
    case_id = _required("MODEL_A_OPPONENT_SHIFT_CASE_ID")
    cases = {case["case_id"]: case for case in protocol["evaluation"]["cases"]}
    if case_id not in cases:
        raise RuntimeError(f"unknown opponent-shift case: {case_id}")
    if label not in protocol["checkpoint_inventory"]:
        raise RuntimeError(f"unregistered opponent-shift label: {label}")
    item = protocol["checkpoint_inventory"][label]
    checkpoint = (ROOT / item["path"]).resolve()
    if not checkpoint.is_file() or _sha256(checkpoint) != item["sha256"]:
        raise RuntimeError(f"opponent-shift checkpoint mismatch: {label}")
    payload = _torch_load(checkpoint)
    if payload.get("architecture") not in (None, MODEL_ARCHITECTURE):
        raise RuntimeError(f"opponent-shift architecture mismatch: {label}")

    self.agent_seed = int(cases[case_id]["agent_seeds"][label])
    self.rng = np.random.default_rng(self.agent_seed)
    torch.manual_seed(self.agent_seed)
    torch.set_num_threads(1)
    self.device = torch.device("cpu")
    self.online_net = DuelingDQN().to(self.device)
    if _load_network_state(self.online_net, payload["online_net"]):
        raise RuntimeError(f"opponent-shift checkpoint unexpectedly required migration: {label}")
    self.online_net.eval()
    self.completed_rounds = int(payload.get("completed_rounds", 0))
    self.epsilon = 0.0


def act_frozen(self, game_state: dict) -> str:
    return v4_callbacks.act(self, game_state)
