"""Inference-only v7a: frozen v4 policy with an anytime safety planner."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np

from agent_code.model_a_dqn.callbacks import _load_checkpoint, _load_network_state
from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import DuelingDQN, torch
from .planner import select_action

CHECKPOINT_PATH = Path(os.environ.get(
    "MODEL_A_V7A_CHECKPOINT_PATH",
    Path(__file__).parents[1] / "model_a_dqn" / "model_a.pt",
))
PLANNER_BUDGET_SECONDS = 0.35


def setup(self):
    if torch is None:
        raise RuntimeError("Model A v7a requires PyTorch")
    if self.train:
        raise RuntimeError("Model A v7a is inference-only")
    torch.set_num_threads(1)
    seed_text = os.environ.get("MODEL_A_V7A_SEED")
    self.agent_seed = None if seed_text is None else int(seed_text)
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = DuelingDQN().to(self.device)
    checkpoint = _load_checkpoint(CHECKPOINT_PATH)
    _load_network_state(self.online_net, checkpoint["online_net"])
    self.online_net.eval()
    self.completed_rounds = int(checkpoint.get("completed_rounds", 0))


def act(self, game_state: dict) -> str:
    # Compute the known-good v4 decision first so every later stage can abort.
    fallback_mask = legal_action_mask(game_state)
    features = state_to_features(game_state)
    assert features is not None
    local, global_features = features
    with torch.no_grad():
        q_values = self.online_net(
            torch.as_tensor(local[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
        )[0].cpu().numpy()
    action, diagnostics = select_action(
        game_state, q_values, fallback_mask, self.rng,
        deadline=perf_counter() + PLANNER_BUDGET_SECONDS,
    )
    self.last_planner_diagnostics = diagnostics
    return ACTIONS[action]
