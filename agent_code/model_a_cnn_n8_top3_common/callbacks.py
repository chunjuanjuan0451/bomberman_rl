"""Load the frozen Task4 CNN with or without the confirmed collision filter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from agent_code.model_a_cnn_n8.callbacks import _load, _tensors
from agent_code.model_a_cnn_n8.features import ACTIONS, state_to_features
from agent_code.model_a_cnn_n8.network import ARCHITECTURE, FullBoardDuelingCNN, torch
from agent_code.model_a_cnn_n8_postbomb_movement_audit.callbacks import immediate_collision_action_details
from agent_code.model_a_dqn.features import legal_action_mask


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_KIND = "model-a-cnn-n8-top3-external-match"


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for the top-three direct match")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_agent(self, identity: str, collision_filter: bool) -> None:
    if torch is None:
        raise RuntimeError("top-three CNN evaluation requires PyTorch")
    if self.train:
        raise RuntimeError("top-three CNN agents are evaluation-only")
    torch.set_num_threads(1)
    protocol_path = Path(_required("CNN_TOP3_MATCH_PROTOCOL_PATH")).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("kind") != PROTOCOL_KIND or protocol.get("training_allowed") is not False:
        raise RuntimeError("invalid top-three match protocol")
    case_id = _required("CNN_TOP3_MATCH_CASE_ID")
    cases = {case["case_id"]: case for case in protocol["evaluation"]["cases"]}
    if case_id not in cases or identity not in protocol["labels"]:
        raise RuntimeError("unknown top-three match case or identity")
    checkpoint = ROOT / protocol["checkpoint_inventory"][identity]["path"]
    expected = protocol["checkpoint_inventory"][identity]
    if not checkpoint.is_file() or _sha256(checkpoint) != expected["sha256"]:
        raise RuntimeError(f"top-three checkpoint mismatch: {identity}")
    payload = _load(checkpoint)
    required = {"architecture": ARCHITECTURE, "stage": "task4", "replica": "r3", "stage_rounds": 800}
    if any(payload.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"top-three CNN lineage mismatch: {identity}")
    self.device = torch.device("cpu")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    self.online_net.load_state_dict(payload["online_net"], strict=True)
    self.online_net.eval()
    for parameter in self.online_net.parameters():
        parameter.requires_grad = False
    seed = int(cases[case_id]["agent_seeds"][identity])
    self.rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    self.top3_collision_filter = bool(collision_filter)


def act_agent(self, game_state: dict) -> str:
    legal = legal_action_mask(game_state)
    legal_indices = np.flatnonzero(legal)
    if not legal_indices.size:
        return "WAIT"
    features = state_to_features(game_state)
    assert features is not None
    spatial, scalars = _tensors(features, self.device)
    with torch.no_grad():
        q_values = self.online_net(spatial, scalars)[0].detach().cpu().numpy()
    best = q_values[legal_indices].max()
    original_candidates = legal_indices[np.isclose(q_values[legal_indices], best)]
    original_index = int(self.rng.choice(original_candidates))

    # `self[2]` is the environment's can_bomb flag.  In the default classic
    # setting it is false exactly while this agent's sole bomb is active.
    effective = legal
    if self.top3_collision_filter and not bool(game_state["self"][2]):
        robust = np.asarray(immediate_collision_action_details(game_state)["robust_action_mask"], dtype=bool)
        filtered = legal & robust
        if filtered.any():
            effective = filtered
    chosen = original_index
    if not effective[original_index]:
        effective_indices = np.flatnonzero(effective)
        best = q_values[effective_indices].max()
        candidates = effective_indices[np.isclose(q_values[effective_indices], best)]
        chosen = int(self.rng.choice(candidates))
    return ACTIONS[chosen]


__all__ = ["act_agent", "setup_agent"]
