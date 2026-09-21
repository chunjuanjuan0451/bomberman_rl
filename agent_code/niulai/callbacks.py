"""Inference callbacks for the final D4 CNN agent."""

from pathlib import Path

import numpy as np
import torch

from .features import ACTIONS, state_to_features
from .mask import effective_action_mask
from .network import FullBoardDuelingCNN
from .symmetry import ACTION_PERMUTATIONS, SYMMETRY_NAMES, transform_game_state


MODEL_PATH = Path(__file__).with_name("model.pt")


def _load_model(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def setup(self):
    torch.set_num_threads(1)
    self.device = torch.device("cpu")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    checkpoint = _load_model(MODEL_PATH)
    self.online_net.load_state_dict(checkpoint["online_net"], strict=True)
    self.online_net.eval()
    self.rng = np.random.default_rng(20260917)
    self.active_bomb_position = None


def d4_q_values(network, game_state, device):
    views = [state_to_features(transform_game_state(game_state, symmetry))
             for symmetry in range(len(SYMMETRY_NAMES))]
    spatial = np.stack([view[0] for view in views])
    scalars = np.stack([view[1] for view in views])
    with torch.no_grad():
        q_views = network(
            torch.as_tensor(spatial, dtype=torch.float32, device=device).div_(255.0),
            torch.as_tensor(scalars, dtype=torch.float32, device=device),
        ).cpu().numpy()
    aligned = np.stack([
        q_views[symmetry, ACTION_PERMUTATIONS[symmetry]]
        for symmetry in range(len(SYMMETRY_NAMES))
    ])
    return aligned.mean(axis=0)


def act(self, game_state):
    bomb_positions = {tuple(position) for position, _ in game_state["bombs"]}
    if self.active_bomb_position is not None and self.active_bomb_position not in bomb_positions:
        self.active_bomb_position = None

    legal = effective_action_mask(game_state, self.active_bomb_position is not None)
    indices = np.flatnonzero(legal)
    if not indices.size:
        indices = np.asarray([ACTIONS.index("WAIT")], dtype=np.int64)

    q_values = d4_q_values(self.online_net, game_state, self.device)
    best = q_values[indices].max()
    candidates = indices[np.isclose(q_values[indices], best)]
    action = ACTIONS[int(self.rng.choice(candidates))]
    if action == "BOMB" and game_state["self"][2]:
        self.active_bomb_position = tuple(game_state["self"][3])
    return action
