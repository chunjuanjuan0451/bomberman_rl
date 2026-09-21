"""Load one of the frozen s162000 direct-match identities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from agent_code.model_a_cnn_n8.callbacks import _load, _tensors
from agent_code.model_a_cnn_n8.features import ACTIONS, state_to_features
from agent_code.model_a_cnn_n8.network import ARCHITECTURE, FullBoardDuelingCNN, torch
from agent_code.model_a_cnn_n8_league_train.callbacks import effective_action_mask


def setup_agent(self, identity: str) -> None:
    if self.train:
        raise RuntimeError("league match agents are evaluation-only")
    protocol = json.loads(Path(os.environ["CNN_LEAGUE_PROTOCOL_PATH"]).read_text(encoding="utf-8"))
    case_id = os.environ["CNN_LEAGUE_MATCH_CASE"]
    case = next(case for case in protocol["blind_confirmation"]["cases"] if case["case_id"] == case_id)
    path = Path(os.environ[f"CNN_LEAGUE_MATCH_{identity.upper().replace('-', '_')}_CHECKPOINT"]).resolve()
    payload = _load(path)
    if payload.get("architecture") != ARCHITECTURE:
        raise RuntimeError("league match checkpoint architecture mismatch")
    self.device = torch.device("cpu")
    torch.set_num_threads(1)
    self.online_net = FullBoardDuelingCNN().to(self.device)
    self.online_net.load_state_dict(payload["online_net"], strict=True)
    self.online_net.eval()
    self.rng = np.random.default_rng(int(case["agent_seeds"][identity]))
    self._active_bomb = False
    self._active_bomb_position = None


def act_agent(self, game_state: dict) -> str:
    bomb_positions = {tuple(position) for position, _timer in game_state["bombs"]}
    if self._active_bomb and self._active_bomb_position not in bomb_positions:
        self._active_bomb = False
        self._active_bomb_position = None
    indices = np.flatnonzero(effective_action_mask(game_state, self._active_bomb))
    if not indices.size:
        return "WAIT"
    features = state_to_features(game_state)
    assert features is not None
    spatial, scalars = _tensors(features, self.device)
    with torch.no_grad():
        q = self.online_net(spatial, scalars)[0].detach().cpu().numpy()
    best = q[indices].max()
    action = ACTIONS[int(self.rng.choice(indices[np.isclose(q[indices], best)]))]
    if action == "BOMB" and game_state["self"][2]:
        self._active_bomb = True
        self._active_bomb_position = tuple(game_state["self"][3])
    return action


__all__ = ["act_agent", "setup_agent"]
