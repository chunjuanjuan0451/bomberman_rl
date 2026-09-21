"""Deterministic nonlinear policy-head and data-path tests."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from agent_code.model_a_v10.callbacks import setup
from agent_code.model_a_v10.policy_head import (NonlinearPolicyHead, RootPolicyBlender,
                                                 masked_policy, prepare_policy_features)
from agent_code.model_a_v10.search import StochasticPUCT
from agent_code.model_a_v10.simulator import SimAgent, SimState


def _features(n=3):
    values = np.zeros((n, 5, 17, 17), dtype=np.float32)
    values[:, 3, 8, 8] = 1.0
    return values


def _state():
    field = -np.ones((17, 17), dtype=int)
    field[1:-1, 1:-1] = 0
    return SimState(field, [SimAgent("root", 8, 8), SimAgent("other", 14, 14)])


def test_policy_head_forward_is_deterministic_and_finite():
    torch.manual_seed(9); model = NonlinearPolicyHead(); x = torch.from_numpy(prepare_policy_features(_features()))
    with torch.no_grad(): first = model(x); second = model(x)
    assert torch.equal(first, second) and torch.isfinite(first).all()


def test_policy_features_are_centered_and_scaled_once():
    values = _features(1); values[0, 0, 8, 8] = 4.0
    prepared = prepare_policy_features(values)
    assert prepared.shape == (1, 1445) and prepared[0, 8 * 17 + 8] == 1.0


def test_masked_policy_is_normalized_and_excludes_illegal_actions():
    logits = torch.zeros((1, 6)); probabilities = masked_policy(logits, [0, 1, 4])
    assert torch.isfinite(probabilities).all() and torch.allclose(probabilities.sum(1), torch.ones(1))
    assert probabilities[0, 2] == 0 and probabilities[0, 5] == 0


def test_policy_head_schema_rejects_wrong_dimensions():
    try:
        NonlinearPolicyHead(1444)
    except ValueError:
        return
    raise AssertionError("schema mismatch was accepted")


def test_policy_head_single_update_changes_only_policy_parameters():
    torch.manual_seed(3); model = NonlinearPolicyHead(); before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4); x = torch.from_numpy(prepare_policy_features(_features(2)))
    y = torch.zeros((2, 6)); y[:, 0] = 1
    loss = (-y * torch.log_softmax(model(x), dim=1)).sum(1).mean(); loss.backward(); optimizer.step()
    assert all(torch.isfinite(v).all() for v in model.state_dict().values()) and any(not torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_zero_weight_is_exact_search_noop_and_skips_forward():
    state = _state()
    reference = StochasticPUCT().search(state, 0, simulations=8, seed=17)
    blender = RootPolicyBlender(NonlinearPolicyHead(), weight=0.0)
    candidate = StochasticPUCT(root_policy_transform=blender).search(
        state, 0, simulations=8, seed=17)
    assert candidate.action == reference.action
    assert candidate.raw_prior == reference.raw_prior
    assert candidate.root_visits == reference.root_visits
    assert blender.forward_calls == 0


def test_positive_weight_calls_policy_head_once_per_search_and_masks_bomb():
    state = _state()
    blender = RootPolicyBlender(NonlinearPolicyHead(), weight=0.1)
    result = StochasticPUCT(
        forbidden_root_actions=("BOMB",), root_policy_transform=blender,
    ).search(state, 0, simulations=12, seed=18)
    assert blender.forward_calls == 1
    assert "BOMB" not in result.raw_prior
    assert set(result.root_visits) == set(result.raw_prior)
    assert abs(sum(result.raw_prior.values()) - 1.0) < 1e-7


def test_official_callback_resolves_repository_relative_paths_from_agent_cwd():
    root = Path(__file__).resolve().parents[2]
    previous_cwd = Path.cwd()
    previous = {name: os.environ.get(name) for name in
                ("MODEL_A_V10_CHECKPOINT_PATH", "MODEL_A_V10_CONFIG_PATH")}
    try:
        os.chdir(root / "agent_code" / "model_a_v10")
        os.environ["MODEL_A_V10_CHECKPOINT_PATH"] = (
            "experiments/checkpoints/model-a-v10-centered-blend-coin-heaven-s103004.npz")
        os.environ["MODEL_A_V10_CONFIG_PATH"] = (
            "experiments/configs/v10-phase3.3-policy-s103009-candidate.json")
        holder = SimpleNamespace()
        setup(holder)
        assert isinstance(holder.v10_controller.searcher.root_policy_transform, RootPolicyBlender)
    finally:
        os.chdir(previous_cwd)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_nonfinite_optional_head_falls_back_to_legal_expert_prior():
    class NonfinitePolicy(NonlinearPolicyHead):
        def forward(self, values):
            return torch.full((values.shape[0], 6), float("nan"))

    state = _state()
    baseline = StochasticPUCT().search(state, 0, simulations=4, seed=19)
    blender = RootPolicyBlender(NonfinitePolicy(), weight=0.1)
    result = StochasticPUCT(root_policy_transform=blender).search(
        state, 0, simulations=4, seed=19)
    assert result.raw_prior == baseline.raw_prior
    assert result.action == baseline.action
    assert blender.forward_calls == 1 and blender.failure_count == 1
