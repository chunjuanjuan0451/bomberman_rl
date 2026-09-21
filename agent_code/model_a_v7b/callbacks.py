"""Frozen v4 with a deadline-aware, opponent-conditioned bomb gate."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np

from agent_code.model_a_dqn.callbacks import _load_checkpoint, _load_network_state
from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import DuelingDQN, torch
from .tactical import maybe_override_with_bomb

CHECKPOINT_PATH = Path(os.environ.get(
    "MODEL_A_V7B_CHECKPOINT_PATH",
    Path(__file__).parents[1] / "model_a_dqn" / "model_a.pt",
))
PLANNER_BUDGET_SECONDS = 0.35


def setup(self):
    if torch is None:
        raise RuntimeError("Model A v7b requires PyTorch")
    if self.train:
        raise RuntimeError("Model A v7b is inference-only")
    torch.set_num_threads(1)
    seed_text = os.environ.get("MODEL_A_V7B_SEED")
    self.agent_seed = None if seed_text is None else int(seed_text)
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = DuelingDQN().to(self.device)
    checkpoint = _load_checkpoint(CHECKPOINT_PATH)
    _load_network_state(self.online_net, checkpoint["online_net"])
    self.online_net.eval()
    self.completed_rounds = int(checkpoint.get("completed_rounds", 0))
    self.v7b_overrides = 0


def act(self, game_state: dict) -> str:
    mask = legal_action_mask(game_state)
    legal = np.flatnonzero(mask)
    if not legal.size:
        return "WAIT"
    local, global_features = state_to_features(game_state)
    with torch.no_grad():
        q_values = self.online_net(
            torch.as_tensor(local[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
        )[0].cpu().numpy()
    best = q_values[legal].max()
    tied = legal[np.isclose(q_values[legal], best)]
    baseline = int(self.rng.choice(tied))
    selected, tactics, reason = maybe_override_with_bomb(
        game_state, baseline, mask, perf_counter() + PLANNER_BUDGET_SECONDS,
    )
    if tactics is not None and tactics.affected_opponents:
        self.logger.debug("v7b bomb candidate (%s): %s", reason, tactics)
    if selected != baseline:
        self.v7b_overrides += 1
        self.logger.debug("v7b BOMB override (%s): %s", reason, tactics)
    return ACTIONS[selected]
