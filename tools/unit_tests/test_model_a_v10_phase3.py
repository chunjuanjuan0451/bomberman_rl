"""Phase-3 preflight tests: no long training or persistent project artifacts."""

from __future__ import annotations

import warnings

import numpy as np

from agent_code.model_a_v10.interfaces import (ACTIONS, BlendedExpertNetwork, HeuristicNetwork,
                                                LinearExpertNetwork)
from agent_code.model_a_v10.runtime import state_features
from agent_code.model_a_v10.search import StochasticPUCT
from agent_code.model_a_v10.simulator import SimAgent
from tools.v10_head_confirmation import evaluate_gate, summarize
from tools.v10_phase3_train import (blended_network, initial_state, iteration_head_weights,
                                    model_diagnostics, play_episode, train,
                                    validation_head_weights)


def _config() -> dict:
    return {"scenario": "coin-heaven", "search_horizon_plies": 3,
            "search_simulations": 4, "max_steps": 12,
            "epochs_per_iteration": 3, "batch_size": 8, "learning_rate": 5e-5,
            "risk_horizon_plies": 8}


def test_v10_phase3_train_and_inference_share_feature_preprocessing():
    state = initial_state(109001, 12)
    model = LinearExpertNetwork(int(np.prod(state_features(state).shape)), 109001)
    raw = state_features(state).reshape(1, -1)
    policy, value, opponent, risk = model.batch_outputs(raw)
    output = model.evaluate(state, 0)
    legal_indices = [ACTIONS.index(action) for action in output.policy]
    expected = policy[0, legal_indices] / policy[0, legal_indices].sum()
    assert np.allclose(expected, list(output.policy.values()))
    assert np.isclose(value[0], output.value) and np.isclose(risk[0], output.risk)
    assert np.isfinite(opponent).all()


def test_v10_phase3_features_are_agent_centered():
    state = initial_state(109000, 12)
    free = [tuple(map(int, position)) for position in np.argwhere(state.field == 0)
            if tuple(map(int, position)) != (state.agents[0].x, state.agents[0].y)]
    state.agents.append(SimAgent("opponent", *free[-1]))
    features = state_features(state)
    center = (features.shape[1] // 2, features.shape[2] // 2)
    assert features[3, center[0], center[1]] == 1.0
    assert np.count_nonzero(features[3] > 0.5) == 1


def test_v10_phase3_micro_update_is_finite_and_warning_free():
    config = _config()
    state = initial_state(109002, config["max_steps"])
    records, terminal = play_episode(state, HeuristicNetwork(), config, 109002)
    assert records and terminal.ended
    assert all(0.0 <= float(record["outcome"]) <= 1.0 for record in records)
    model = LinearExpertNetwork(int(np.prod(state_features(state).shape)), 109002)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        losses = train(model, records, config, np.random.default_rng(109099))
        diagnostics = model_diagnostics(model, records, config["batch_size"])
        result = model.evaluate(state, 0)
    assert all(np.isfinite(value) for value in losses.values())
    assert all(np.isfinite(value) for value in diagnostics.values())
    assert 0.0 <= losses["wait_fraction_used"] <= 1.0
    assert all(np.isfinite(value) for value in result.policy.values())
    assert model.parameter_health()


def test_v10_phase3_opponent_head_is_a_normalized_legal_distribution():
    state = initial_state(109003, 12)
    # Add a second agent on a known free tile so the opponent mask is exercised.
    free = [tuple(map(int, position)) for position in np.argwhere(state.field == 0)
            if tuple(map(int, position)) != (state.agents[0].x, state.agents[0].y)]
    state.agents.append(SimAgent("opponent", *free[-1]))
    model = LinearExpertNetwork(int(np.prod(state_features(state).shape)), 109003)
    policy = model.opponent_action_policy(state, 1, 0)
    assert policy and np.isclose(sum(policy.values()), 1.0)
    assert all(np.isfinite(value) and value >= 0 for value in policy.values())


def test_v10_phase3_coin_heaven_masks_bomb_and_blend_is_normalized():
    state = initial_state(109004, 12)
    model = LinearExpertNetwork(int(np.prod(state_features(state).shape)), 109004)
    blended = BlendedExpertNetwork(model, 0.1)
    output = blended.evaluate(state, 0)
    assert np.isclose(sum(output.policy.values()), 1.0)
    result = StochasticPUCT(blended, forbidden_root_actions=("BOMB",)).search(
        state, 0, simulations=4, seed=109004)
    assert "BOMB" not in result.raw_prior and result.action != "BOMB"
    weights = {"policy": 0.0, "value": 0.1, "opponent": 0.0, "risk": 0.0}
    value_only = blended_network(model, weights)
    teacher_output = HeuristicNetwork().evaluate(state, 0)
    value_output = value_only.evaluate(state, 0)
    assert value_output.policy == teacher_output.policy
    config = _config() | {
        "iterations": 2,
        "policy_student_weight_schedule": [0.0, 0.0],
        "value_student_weight_schedule": [0.1, 0.1],
        "opponent_student_weight_schedule": [0.0, 0.0],
        "risk_student_weight_schedule": [0.0, 0.0],
        "validation_policy_student_weight": 0.0,
        "validation_value_student_weight": 0.1,
        "validation_opponent_student_weight": 0.0,
        "validation_risk_student_weight": 0.0,
    }
    assert iteration_head_weights(config, 1) == weights
    assert validation_head_weights(config) == weights
    rows = [{"results": {"teacher": {"score": 10, "steps": 12, "survived": True},
                         "candidate": {"score": 20, "steps": 10, "survived": True}}}]
    aggregates = summarize(rows, ["teacher", "candidate"], 12)
    gate_config = {"baseline_variant": "teacher", "candidate_variant": "candidate",
                   "gates": {"minimum_paired_wins": 1, "minimum_mean_gain_ratio": 1.15,
                             "maximum_step_limit_delta": 0}}
    assert evaluate_gate(rows, aggregates, gate_config)["passed"]
