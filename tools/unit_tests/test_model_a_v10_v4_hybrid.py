"""Regression tests for the no-training v10.12 frozen-v4 hybrid."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_code.model_a_dqn.features import ACTIONS
from agent_code.model_a_v10.v4_hybrid import (
    BombOwnershipTracker,
    V4AnchoredHybridController,
    owned_state_from_game_state,
)
from agent_code.model_a_v10.hybrid_search import HybridSearchResult
from agent_code.model_a_v10.v4_adapter import FrozenV4Policy
from agent_code.model_a_v10.callbacks import setup
from experiments.evaluate import referenced_checkpoints
from tools.v10_v4_hybrid_gate import load_protocol


ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "agent_code/model_a_dqn/model_a.pt"
V4_SHA = "d5fe215b1f909149e1eefa2ee6ac00d7365e61343b333ba91b900e11e911d239"
CANDIDATE = ROOT / "experiments/configs/v10.12-v4-tactical-hybrid-candidate-s106300.json"
GATE = ROOT / "experiments/configs/v10.12-v4-tactical-hybrid-gate-s106300.json"


def _game_state(*, step: int = 1, self_position=(3, 3), bombs_left=True,
                others=(), bombs=(), explosion_map=None):
    field = -np.ones((9, 9), dtype=int)
    field[1:-1, 1:-1] = 0
    return {
        "round": 7,
        "step": step,
        "field": field,
        "self": ("root", 0, bombs_left, self_position),
        "others": list(others),
        "bombs": list(bombs),
        "coins": [(6, 6)],
        "explosion_map": (np.zeros_like(field) if explosion_map is None else explosion_map),
        "user_input": "WAIT",
    }


def test_frozen_v4_adapter_greedy_action_is_masked_q_argmax():
    policy = FrozenV4Policy(V4, V4_SHA, seed=19)
    game_state = _game_state()
    q_values = policy.q_values(game_state)
    action = policy.greedy_action(game_state)
    distribution = policy.policy(game_state)
    assert action in distribution
    assert np.isclose(sum(distribution.values()), 1.0)
    assert np.isclose(q_values[ACTIONS.index(action)],
                      max(q_values[ACTIONS.index(item)] for item in distribution))


def test_bomb_ownership_tracker_attributes_self_and_opponent_new_bombs():
    tracker = BombOwnershipTracker()
    first = _game_state(others=(("other", 0, True, (5, 5)),))
    tracker.observe(first)
    second = _game_state(
        step=2, self_position=(3, 2), bombs_left=False,
        others=(("other", 0, False, (5, 4)),),
        bombs=(((3, 3), 3), ((5, 5), 3)),
    )
    owners, explosions = tracker.observe(second, last_self_action="BOMB")
    assert owners == {(3, 3): 0, (5, 5): 1}
    assert not explosions
    state = owned_state_from_game_state(second, owners, {})
    assert [(bomb.x, bomb.y, bomb.owner) for bomb in state.bombs] == [
        (3, 3, 0), (5, 5, 1)]


class _Anchor:
    def __init__(self, action: str):
        self.action = action

    def greedy_action(self, _game_state):
        return self.action


class _StubHybrid(V4AnchoredHybridController):
    def __init__(self, result: HybridSearchResult, anchor="RIGHT"):
        super().__init__(anchor_policy=_Anchor(anchor), minimum_action_visits=8,
                         minimum_value_advantage=0.10)
        self.stub_result = result

    def _select_tactical(self, state, seed):
        self.last_result = self.stub_result
        return self.stub_result


def _result(*, proposed="UP", proposed_visits=10, anchor_visits=10,
            proposed_value=0.3, anchor_value=0.1):
    return HybridSearchResult(
        proposed, "RIGHT", {"UP": 0.4, "RIGHT": 0.6},
        {"UP": proposed_visits, "RIGHT": anchor_visits},
        {"UP": proposed_value, "RIGHT": anchor_value},
        (proposed,), 0.0, 20, 4, "simulation-limit", 1.0,
    )


def test_hybrid_is_exact_v4_off_gate_and_requires_override_evidence():
    quiet = _game_state()
    controller = _StubHybrid(_result())
    assert controller.act(quiet) == "RIGHT"
    assert controller.last_result is None
    assert controller.last_override_reason == "v4-default"

    tactical = _game_state(others=(("other", 0, True, (5, 3)),))
    controller = _StubHybrid(_result(proposed_visits=7))
    assert controller.act(tactical) == "RIGHT"
    assert controller.last_override_reason == "insufficient-override-evidence"
    controller = _StubHybrid(_result())
    assert controller.act(tactical) == "UP"
    assert controller.last_override_reason == "value-qualified-override"


def test_candidate_config_binds_v4_and_is_no_training():
    config = json.loads(CANDIDATE.read_text())
    assert config["training"] is False
    assert config["v4_tactical_hybrid"]["minimum_completed_simulations"] == 32
    assert config["v4_tactical_hybrid"]["minimum_action_visits"] == 8
    assert config["v4_tactical_hybrid"]["minimum_value_advantage"] == 0.10
    references = referenced_checkpoints(CANDIDATE)
    assert references == [{
        "role": "frozen_v4_anchor",
        "path": "agent_code/model_a_dqn/model_a.pt",
        "sha256": V4_SHA,
    }]


def test_official_callback_and_gate_load_the_frozen_hybrid_contract():
    names = ("MODEL_A_V10_CHECKPOINT_PATH", "MODEL_A_V10_CONFIG_PATH", "MODEL_A_V10_SEED")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ[names[0]] = str(
            ROOT / "experiments/checkpoints/model-a-v10-centered-blend-coin-heaven-s103004.npz")
        os.environ[names[1]] = str(CANDIDATE)
        os.environ[names[2]] = "106300"
        holder = SimpleNamespace()
        setup(holder)
        assert isinstance(holder.v10_controller, V4AnchoredHybridController)
        assert holder.v10_controller.budget.tactical_min_simulations == 32
        protocol = load_protocol(GATE)
        assert protocol["stages"]["confirmation"]["seeds"] == list(range(106302, 106308))
        assert protocol["training"] is False and protocol["automatic_training"] is False
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
