"""Callbacks for the frozen-v4 global-resource n=8 rescue pilot."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask
from .config import ARMS, ENDPOINT_ROUND, architecture_name, load_protocol
from .features import combined_features
from .network import GlobalResourceDQN, torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_base(model, parent: Path, expected_sha256: str) -> None:
    if _sha256(parent) != expected_sha256:
        raise RuntimeError("frozen-v4 parent hash mismatch")
    payload = _torch_load(parent)
    model.base.load_state_dict(payload["online_net"], strict=True)
    model.freeze_base()


def setup(self):
    if torch is None:
        raise RuntimeError("global-resource n8 pilot requires PyTorch")
    torch.set_num_threads(1)
    self.global_protocol, self.global_protocol_path, self.global_protocol_sha256 = load_protocol()
    self.global_arm = os.environ.get("MODEL_A_GLOBAL_ARM", "")
    self.global_replica = os.environ.get("MODEL_A_GLOBAL_REPLICA", "")
    self.global_run_mode = os.environ.get("MODEL_A_GLOBAL_RUN_MODE", "")
    if self.global_arm not in (*ARMS, "frozen-v4"):
        raise RuntimeError("MODEL_A_GLOBAL_ARM must identify a preregistered arm")
    if self.global_run_mode not in ("train", "evaluate"):
        raise RuntimeError("MODEL_A_GLOBAL_RUN_MODE must be train or evaluate")
    if self.global_run_mode == "train" and (
        self.global_arm not in ARMS
        or self.global_replica not in self.global_protocol["training"]["replicas"]
    ):
        raise RuntimeError("training requires a residual arm and registered replica")
    self.agent_seed = int(os.environ["MODEL_A_GLOBAL_SEED"])
    torch.manual_seed(self.agent_seed)
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    hyper = self.global_protocol["learning_contract"]
    self.online_net = GlobalResourceDQN(float(hyper["delta_cap"])).to(self.device)
    parent = Path(self.global_protocol["frozen_v4"]["path"])
    if not parent.is_absolute():
        parent = self.global_protocol_path.parents[2] / parent
    self.parent_checkpoint_path = parent.resolve()
    _load_base(self.online_net, self.parent_checkpoint_path, self.global_protocol["frozen_v4"]["sha256"])
    self.completed_rounds = 0
    if self.global_run_mode == "train":
        if os.environ.get("MODEL_A_GLOBAL_RESUME", "") == "1":
            raise RuntimeError("global-resource n8 pilot forbids resume")
        return
    if self.global_arm == "frozen-v4":
        self.online_net.eval()
        return
    checkpoint_path = Path(os.environ["MODEL_A_GLOBAL_CHECKPOINT_PATH"])
    payload = _torch_load(checkpoint_path)
    expected = {
        "architecture": architecture_name(),
        "protocol_sha256": self.global_protocol_sha256,
        "arm": self.global_arm,
        "replica": self.global_replica,
        "completed_rounds": ENDPOINT_ROUND,
        "parent_sha256": self.global_protocol["frozen_v4"]["sha256"],
        "n_step": 8,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("global-resource n8 checkpoint metadata mismatch")
    self.online_net.load_state_dict(payload["online_net"], strict=True)
    self.online_net.freeze_base()
    self.online_net.eval()
    self.completed_rounds = int(payload["completed_rounds"])


def act(self, game_state: dict) -> str:
    legal_indices = np.flatnonzero(legal_action_mask(game_state))
    if not legal_indices.size:
        return "WAIT"
    epsilon = getattr(self, "epsilon", 0.0) if self.global_run_mode == "train" else 0.0
    if epsilon and self.rng.random() < epsilon:
        return ACTIONS[int(self.rng.choice(legal_indices))]
    local, global_features, resource = combined_features(game_state, self.global_arm)
    with torch.no_grad():
        q_values = self.online_net(
            torch.as_tensor(local[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(resource[None], dtype=torch.float32, device=self.device),
        )[0].cpu().numpy()
    best = q_values[legal_indices].max()
    candidates = legal_indices[np.isclose(q_values[legal_indices], best)]
    return ACTIONS[int(self.rng.choice(candidates))]
