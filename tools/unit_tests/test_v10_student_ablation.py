"""No-execution tests for the v10.7 student-only ablation route."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from agent_code.model_a_v10.interfaces import ACTIONS
from agent_code.model_a_v10.residual_policy import (
    PlannerAwareGatedResidualPolicy,
    PlannerAwareResidualPolicy,
    PlannerResidualTransform,
    TargetAwareResidualPolicy,
    load_residual_checkpoint,
)
from agent_code.model_a_v10.simulator import SimAgent, SimState
from tools.v10_student_ablation import (
    balanced_order,
    gate_result,
    load_config,
    ranking_loss,
    trajectory_path,
    training_loss,
)


ROOT = Path(__file__).resolve().parents[2]


def _optimizer_config():
    return {
        "search_confidence_margin": 0.15, "soft_label_weight": 1.0,
        "hard_correction_weight": 0.25, "planner_agreement_kl_weight": 0.75,
        "agreement_rank_weight": 1.0, "wait_agreement_rank_weight": 2.0,
        "correction_rank_weight": 0.25, "rank_margin": 0.02,
        "residual_l2_weight": 0.02, "preservation_residual_weight": 0.1,
        "gate_bce_weight": 0.5, "gate_preservation_weight": 0.25,
    }


def _batch():
    n = 12
    target = np.zeros((n, 6), dtype=np.float32)
    planner = np.zeros_like(target)
    for index in range(n):
        action = index % 6
        target[index, action] = 0.9
        target[index, (action + 1) % 6] = 0.1
        planner[index, (action if index % 2 else (action + 1) % 6)] = 0.6
        planner[index, action] += 0.4
    return {
        "features": np.zeros((n, 8, 33, 33), dtype=np.float32),
        "visit_policy": target, "planner_prior": planner,
        "safe_mask": np.ones_like(target),
    }


def test_planner_aware_heads_start_as_exact_zero_residuals():
    features = torch.randn((3, 8, 33, 33))
    planner = torch.softmax(torch.randn((3, 6)), dim=-1)
    for model in (PlannerAwareResidualPolicy(8, 1),
                  PlannerAwareGatedResidualPolicy(8, 1)):
        assert torch.equal(model(features, planner), torch.zeros((3, 6)))
    try:
        PlannerAwareResidualPolicy(8, 1)(features, planner[:, :5])
    except ValueError:
        pass
    else:
        raise AssertionError("planner-aware head accepted a malformed prior")


def test_planner_aware_checkpoint_dispatch_is_strict():
    for model in (PlannerAwareResidualPolicy(8, 1),
                  PlannerAwareGatedResidualPolicy(8, 1)):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            metadata = {"architecture": model.architecture, "channels": 8,
                        "blocks": 1, "bound": 0.5}
            model.save_checkpoint(path, metadata)
            loaded, loaded_metadata = load_residual_checkpoint(path)
            assert isinstance(loaded, type(model)) and loaded_metadata == metadata


def test_root_transform_supplies_the_exact_planner_prior_to_new_heads():
    class Recorder(torch.nn.Module):
        requires_planner_prior = True
        input_channels = 8

        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, features, planner_prior):
            self.seen = planner_prior.detach().clone()
            return torch.zeros_like(planner_prior)

    field = -np.ones((9, 9), dtype=int)
    field[1:-1, 1:-1] = 0
    state = SimState(field, [SimAgent("root", 4, 4)])
    prior = {"UP": 0.2, "RIGHT": 0.3, "WAIT": 0.5}
    recorder = Recorder()
    output = PlannerResidualTransform(recorder)(state, 0, prior)
    expected = torch.zeros((1, 6))
    for action, probability in prior.items():
        expected[0, ACTIONS.index(action)] = probability
    assert torch.equal(recorder.seen, expected)
    assert all(abs(output[action] - probability) < 1e-7
               for action, probability in prior.items())


def test_three_arm_loss_is_finite_and_gate_penalizes_wait_regression():
    batch = _batch()
    for model in (TargetAwareResidualPolicy(8, 1),
                  PlannerAwareResidualPolicy(8, 1),
                  PlannerAwareGatedResidualPolicy(8, 1)):
        loss = training_loss(model, batch, _optimizer_config())
        assert torch.isfinite(loss)
        loss.backward()
    measurement = {
        "baseline": {"cross_entropy": 1.0, "nonwait_top1_accuracy": 0.5,
                     "wait_top1_accuracy": 0.8, "bomb_top1_accuracy": 1.0,
                     "high_confidence_cross_entropy": 1.0,
                     "planner_agreement_top1_accuracy": 1.0},
        "candidate": {"cross_entropy": 0.9, "nonwait_top1_accuracy": 0.55,
                      "wait_top1_accuracy": 0.7, "bomb_top1_accuracy": 1.0,
                      "high_confidence_cross_entropy": 0.9,
                      "planner_agreement_top1_accuracy": 0.99,
                      "unsafe_probability_mass": 0.0,
                      "residual_state_std_mean": 0.01,
                      "residual_saturation_fraction": 0.0},
    }
    gates = {"minimum_relative_ce_gain": 0.05,
             "minimum_nonwait_top1_gain_points": 0.03,
             "maximum_wait_top1_drop_points": 0.02,
             "maximum_bomb_top1_drop_points": 0.0,
             "minimum_high_confidence_relative_ce_gain": 0.05,
             "maximum_planner_agreement_top1_drop_points": 0.01,
             "maximum_unsafe_probability": 0.0,
             "minimum_residual_state_std_mean": 0.001,
             "maximum_residual_saturation_fraction": 0.05}
    assert not gate_result(measurement, gates)["passed"]


def test_ranking_loss_and_balanced_sampling_are_deterministic():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    action = torch.tensor([0, 0])
    mask = torch.tensor([True, True])
    assert float(ranking_loss(logits, action, mask, 0.1)) > 0.0
    data = {"visit_policy": np.eye(6, dtype=np.float32)[
        np.asarray([0] * 60 + [1] * 20 + [2] * 10 + [3] * 5 + [4] * 3 + [5] * 2)]}
    first = balanced_order(data, 17, 0.5)
    second = balanced_order(data, 17, 0.5)
    assert np.array_equal(first, second)
    sampled = data["visit_policy"][first].argmax(-1)
    assert int((sampled == 5).sum()) > 2


def test_registered_ablation_excludes_the_seen_outer_validation():
    config = load_config(
        ROOT / "experiments/configs/v10.7-student-three-arm-s106000.json")
    assert config["inner_train_episode_seed_range"] == [105200, 105229]
    assert config["inner_dev_episode_seed_range"] == [105230, 105237]
    assert config["excluded_outer_validation_seed_range"] == [105238, 105247]
    assert config["automatic_full_training"] is False
    assert trajectory_path(config, 105237).name == "episode-038-s105237.npz"
