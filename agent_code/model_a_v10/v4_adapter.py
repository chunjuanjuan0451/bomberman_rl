"""Read-only adapter from the frozen v4 DQN to the v10 search interface."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from agent_code.model_a_dqn.callbacks import _load_checkpoint, _load_network_state
from agent_code.model_a_dqn.features import ACTIONS, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import DuelingDQN, torch

from .interfaces import NetworkOutput
from .simulator import SimState


def game_state_from_sim_state(state: SimState, player: int = 0) -> dict:
    """Project a simulated public state into the exact v4 feature contract."""
    controlled = state.agents[player]
    others = [
        (agent.name, agent.score, agent.bombs_left, (agent.x, agent.y))
        for index, agent in enumerate(state.agents)
        if index != player and agent.alive
    ]
    return {
        "round": 0,
        "step": state.step_count,
        "field": state.field,
        "self": (controlled.name, controlled.score, controlled.bombs_left,
                 (controlled.x, controlled.y)),
        "others": others,
        "bombs": [((bomb.x, bomb.y), bomb.timer) for bomb in state.bombs],
        "coins": [position for position, visible in state.coins.items() if visible],
        "explosion_map": state.explosion_map(),
        "user_input": "WAIT",
    }


class FrozenV4Policy:
    """Load v4 immutably and expose its masked Q policy without retraining."""

    def __init__(self, checkpoint_path: Path, expected_sha256: str, *,
                 seed: int = 0, temperature: float = 0.25):
        if torch is None:
            raise RuntimeError("The frozen v4 adapter requires PyTorch")
        checkpoint_path = Path(checkpoint_path)
        actual = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError("frozen v4 checkpoint hash mismatch")
        if temperature <= 0.0:
            raise ValueError("v4 policy temperature must be positive")
        torch.set_num_threads(1)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = actual
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(int(seed))
        self.model = DuelingDQN().to(torch.device("cpu"))
        checkpoint = _load_checkpoint(checkpoint_path)
        _load_network_state(self.model, checkpoint["online_net"])
        self.model.eval()

    def q_values(self, game_state: dict) -> np.ndarray:
        features = state_to_features(game_state)
        if features is None:
            return np.zeros(len(ACTIONS), dtype=np.float32)
        local, global_features = features
        with torch.no_grad():
            result = self.model(
                torch.as_tensor(local[None], dtype=torch.float32),
                torch.as_tensor(global_features[None], dtype=torch.float32),
            )[0].cpu().numpy()
        return np.asarray(result, dtype=np.float64)

    def policy(self, game_state: dict) -> dict[str, float]:
        legal = np.flatnonzero(legal_action_mask(game_state))
        if not len(legal):
            return {"WAIT": 1.0}
        logits = self.q_values(game_state)[legal] / self.temperature
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        return {ACTIONS[int(index)]: float(weight) for index, weight in zip(legal, weights)}

    def greedy_action(self, game_state: dict) -> str:
        """Match the official v4 callback, including seeded Q-tie handling."""
        legal = np.flatnonzero(legal_action_mask(game_state))
        if not len(legal):
            return "WAIT"
        q_values = self.q_values(game_state)
        best = q_values[legal].max()
        tied = legal[np.isclose(q_values[legal], best)]
        return ACTIONS[int(self.rng.choice(tied))]


class V4AnchoredNetwork:
    """Use v4 only for root/rollout policy and retain v10's auxiliary heads."""

    def __init__(self, anchor: FrozenV4Policy, auxiliary_network):
        self.anchor = anchor
        self.auxiliary_network = auxiliary_network

    def evaluate(self, state: SimState, player: int) -> NetworkOutput:
        auxiliary = self.auxiliary_network.evaluate(state, player)
        if not state.agents[player].alive:
            return NetworkOutput({}, -1.0, auxiliary.opponent_policy, 1.0)
        return NetworkOutput(
            self.anchor.policy(game_state_from_sim_state(state, player)),
            auxiliary.value,
            auxiliary.opponent_policy,
            auxiliary.risk,
        )

    def opponent_action_policy(self, state: SimState, player: int,
                               root_player: int) -> dict[str, float]:
        return self.auxiliary_network.opponent_action_policy(state, player, root_player)


class FrozenV4RootPolicyTransform:
    """Replace only the search root prior; leaf evaluation stays inexpensive."""

    def __init__(self, anchor: FrozenV4Policy):
        self.anchor = anchor
        self.call_count = 0

    def __call__(self, state: SimState, player: int,
                 _base_policy: dict[str, float]) -> dict[str, float]:
        self.call_count += 1
        return self.anchor.policy(game_state_from_sim_state(state, player))
