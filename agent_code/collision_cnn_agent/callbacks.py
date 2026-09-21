"""Tournament inference entry points for the standalone collision CNN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .features import ACTIONS, collision_filtered_mask, legal_action_mask, state_to_features
from .network import ARCHITECTURE, FullBoardDuelingCNN


MODEL_PATH = Path(__file__).with_name("model.pt")


def _load_model(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def setup(self) -> None:
    torch.set_num_threads(1)
    self.device = torch.device("cpu")
    self.online_net = FullBoardDuelingCNN().to(self.device)
    checkpoint = _load_model(MODEL_PATH)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise RuntimeError("collision CNN checkpoint architecture mismatch")
    self.online_net.load_state_dict(checkpoint["online_net"], strict=True)
    self.online_net.eval()
    for parameter in self.online_net.parameters():
        parameter.requires_grad = False
    self.rng = np.random.default_rng(20260917)


def act(self, game_state: dict) -> str:
    legal = legal_action_mask(game_state)
    legal_indices = np.flatnonzero(legal)
    if not legal_indices.size:
        return "WAIT"
    spatial, scalars = state_to_features(game_state)
    with torch.inference_mode():
        q_values = self.online_net(
            torch.as_tensor(spatial[None], dtype=torch.float32, device=self.device).div_(255.0),
            torch.as_tensor(scalars[None], dtype=torch.float32, device=self.device),
        )[0].cpu().numpy()
    best = q_values[legal_indices].max()
    original = legal_indices[np.isclose(q_values[legal_indices], best)]
    original_index = int(self.rng.choice(original))
    effective = collision_filtered_mask(game_state, legal)
    chosen = original_index
    if not effective[original_index]:
        effective_indices = np.flatnonzero(effective)
        best = q_values[effective_indices].max()
        candidates = effective_indices[np.isclose(q_values[effective_indices], best)]
        chosen = int(self.rng.choice(candidates))
    return ACTIONS[chosen]
