"""Protocol tests for the corrected v10.2 bounded residual route."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from agent_code.model_a_v10.callbacks import setup
from agent_code.model_a_v10.interfaces import ACTIONS, MOVES, HeuristicNetwork
from agent_code.model_a_v10.planner import ClassicTacticalPlanner
from agent_code.model_a_v10.residual_policy import (
    ACTION_D4, D4_TRANSFORMS, ActionConditionedResidualPolicy,
    BoundedResidualPolicy, PlannerResidualTransform, TargetAwareResidualPolicy,
    load_residual_checkpoint,
    padded_centered_features, planner_residual_policy, transform_action_vector,
    transform_features,
)
from agent_code.model_a_v10.runtime import state_from_game_state
from agent_code.model_a_v10.simulator import SimAgent, SimBomb, SimState
from agent_code.model_a_v10.tactics import tactical_cases
from tools.v10_classic_residual_train import (
    REQUIRED_SHARD_KEYS, augment_batch, finalize_episode_targets, initial_state, load_config,
    masked_metrics, observer_game_state, offline_gate_metrics, residual_training_loss,
    teacher_action, teacher_visit_policy, train_model, visit_policy,
    sample_decision,
)


ROOT = Path(__file__).resolve().parents[2]


def _open_state(root=(1, 1)):
    field = -np.ones((17, 17), dtype=int)
    field[1:-1, 1:-1] = 0
    return SimState(field, [SimAgent("root", *root), SimAgent("other", 15, 15)])


def test_padded_schema_keeps_full_corner_board_and_wall_padding():
    state = _open_state((1, 1))
    features = padded_centered_features(state, safe_actions=("WAIT", "RIGHT"))
    assert features.shape == (7, 33, 33)
    assert features[3, 16, 16] == 1
    assert features[4, 30, 30] == 1
    assert features[0, 31, 31] == -1 and features[0, 32, 32] == -1


def test_safety_channel_is_exactly_planner_allowed_destinations():
    case = next(case for case in tactical_cases() if case.name == "false_kill")
    base = HeuristicNetwork().evaluate(case.state, 0).policy
    plan = ClassicTacticalPlanner().plan(case.state, 0, base)
    features = padded_centered_features(case.state, safe_actions=tuple(plan.prior))
    expected = {(16, 16)}
    for action in plan.prior:
        if action in MOVES:
            dx, dy, _ = MOVES[action]
            expected.add((16 + dx, 16 + dy))
    actual = set(map(tuple, np.argwhere(features[6] > 0)))
    assert actual == expected and "BOMB" not in plan.prior


def test_all_eight_d4_transforms_align_spatial_moves_and_action_vectors():
    for transform in D4_TRANSFORMS:
        for action in ("UP", "RIGHT", "DOWN", "LEFT"):
            features = np.zeros((7, 33, 33), dtype=np.float32)
            dx, dy, _ = MOVES[action]
            features[6, 16 + dx, 16 + dy] = 1
            vector = np.zeros(6, dtype=np.float32)
            vector[ACTIONS.index(action)] = 1
            transformed_vector = transform_action_vector(vector, transform)
            transformed_action = ACTIONS[int(np.argmax(transformed_vector))]
            tx, ty, _ = MOVES[transformed_action]
            transformed_features = transform_features(features, transform)
            assert transformed_features[6, 16 + tx, 16 + ty] == 1
            assert ACTION_D4[transform][ACTIONS.index(action)] == transformed_action


def test_target_distance_channel_is_wall_aware_and_d4_equivariant():
    field = -np.ones((9, 9), dtype=int)
    field[1:-1, 1:-1] = 0
    field[4, 2:7] = -1
    field[4, 6] = 0
    state = SimState(field, [SimAgent("root", 2, 4)])
    target = (6, 4)
    features = padded_centered_features(
        state, safe_actions=("WAIT",), crate_target=target,
        include_target_distance=True)
    assert features.shape == (8, 33, 33)
    # The wall forces a detour, so the root is more than four steps away.
    assert 0.0 < features[7, 16, 16] < 1.0 - 4.0 / 16.0
    for transform in D4_TRANSFORMS:
        transformed = transform_features(features, transform)
        assert transformed.shape == (8, 33, 33)
        assert np.isclose(transformed[7].max(), 1.0)


def test_zero_residual_is_exact_planner_policy_and_mask_is_hard():
    prior = torch.tensor([[0.1, 0.6, 0.0, 0.0, 0.3, 0.0]])
    safe = torch.tensor([[1, 1, 0, 0, 1, 0]], dtype=torch.float32)
    output = planner_residual_policy(prior, safe, torch.zeros_like(prior))
    assert torch.allclose(output, prior, atol=1e-7, rtol=0)
    assert float((output * (1 - safe)).sum()) == 0.0


def test_model_is_zero_initialized_and_strictly_bounded():
    model = BoundedResidualPolicy()
    values = torch.randn((4, 7, 33, 33))
    assert torch.equal(model(values), torch.zeros((4, 6)))
    with torch.no_grad():
        model.head.bias.fill_(100)
    assert float(model(values).detach().abs().max()) <= 0.5


def test_action_conditioned_model_is_zero_initialized_bounded_and_loadable():
    model = ActionConditionedResidualPolicy()
    values = torch.randn((3, 7, 33, 33))
    assert torch.equal(model(values), torch.zeros((3, 6)))
    with torch.no_grad():
        model.move_output.weight.fill_(10.0)
        model.stationary_output.weight.fill_(10.0)
    assert float(model(values).detach().abs().max()) <= 0.5
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "v3.pt"
        metadata = {"architecture": model.architecture, "channels": 32,
                    "blocks": 3, "bound": 0.5}
        model.save_checkpoint(path, metadata)
        loaded, loaded_metadata = load_residual_checkpoint(path)
        assert isinstance(loaded, ActionConditionedResidualPolicy)
        assert loaded_metadata == metadata


def test_target_aware_model_is_zero_initialized_and_checkpoint_dispatches():
    model = TargetAwareResidualPolicy(channels=8, blocks=1)
    values = torch.randn((2, 8, 33, 33))
    assert torch.equal(model(values), torch.zeros((2, 6)))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "v4.pt"
        metadata = {"architecture": model.architecture, "channels": 8,
                    "blocks": 1, "bound": 0.5}
        model.save_checkpoint(path, metadata)
        loaded, _ = load_residual_checkpoint(path)
        assert isinstance(loaded, TargetAwareResidualPolicy)


def test_root_transform_adds_residual_to_planner_and_runs_once():
    state = _open_state((8, 8))
    planner_prior = ClassicTacticalPlanner().plan(
        state, 0, HeuristicNetwork().evaluate(state, 0).policy).prior
    transform = PlannerResidualTransform(BoundedResidualPolicy())
    output = transform(state, 0, planner_prior)
    assert transform.forward_calls == 1
    assert all(abs(output[action] - planner_prior[action]) < 1e-7 for action in output)
    assert set(output) == set(planner_prior)


def test_observer_projection_removes_hidden_coin_and_bomb_owner():
    state = _open_state((3, 3))
    state.coins = {(4, 4): False, (5, 5): True}
    state.bombs = [SimBomb(3, 3, 0, 2)]
    public = observer_game_state(state)
    observed = state_from_game_state(public)
    assert public["coins"] == [(5, 5)]
    assert set(observed.coins) == {(5, 5)}
    assert observed.bombs[0].owner == 1


def test_training_augmentation_synchronizes_every_action_field():
    n = 32
    features = np.zeros((n, 7, 33, 33), dtype=np.float32)
    target = np.zeros((n, 6), dtype=np.float32)
    target[:, ACTIONS.index("UP")] = 1
    features[:, 6, 16, 15] = 1
    batch = {
        "features": features, "visit_policy": target.copy(),
        "planner_prior": target.copy(), "safe_mask": target.copy(),
        "planner_action_scores": target.copy(),
        "search_action": np.full(n, ACTIONS.index("UP"), dtype=np.int64),
        "opponent_actions": np.full((n, 3), ACTIONS.index("UP"), dtype=np.int64),
    }
    transformed = augment_batch(batch, 1.0, np.random.default_rng(4))
    for index in range(n):
        action = ACTIONS[int(transformed["search_action"][index])]
        assert np.argmax(transformed["visit_policy"][index]) == ACTIONS.index(action)
        assert np.argmax(transformed["planner_prior"][index]) == ACTIONS.index(action)
        assert np.argmax(transformed["safe_mask"][index]) == ACTIONS.index(action)
        assert np.all(transformed["opponent_actions"][index] == ACTIONS.index(action))
        dx, dy, _ = MOVES[action]
        assert transformed["features"][index, 6, 16 + dx, 16 + dy] == 1


def test_visit_target_and_registered_config_enforce_search_protocol():
    target = visit_policy({"UP": 32, "RIGHT": 96})
    assert target[ACTIONS.index("UP")] == 0.25
    assert target[ACTIONS.index("RIGHT")] == 0.75
    config = load_config(ROOT / "experiments/configs/v10.2-classic-residual-policy-s103132.json")
    assert config["teacher_search_simulations"] == 128
    assert config["training_seed"] == 103132 and config["protocol_version"] == 2
    assert REQUIRED_SHARD_KEYS == set(config.get("required_shard_keys", REQUIRED_SHARD_KEYS))
    task2 = load_config(ROOT / "experiments/configs/v10.3-task2-action-residual-s103252.json")
    assert task2["opponent_count"] == 0 and task2["scenario"] == "classic"
    state = initial_state(task2["training_seed"], task2["max_steps"], opponent_count=0)
    assert len(state.agents) == 1 and not state.ended
    # Source-bound v10.5/v10.6 configs are immutable historical runs and are
    # intentionally no longer executable after the next protocol is added.


def test_zero_residual_metrics_equal_planner_and_future_death_is_delayed():
    planner = np.asarray([[0.7, 0.3, 0, 0, 0, 0]], dtype=np.float32)
    data = {
        "features": np.zeros((1, 7, 33, 33), dtype=np.float32),
        "visit_policy": np.asarray([[0.5, 0.5, 0, 0, 0, 0]], dtype=np.float32),
        "planner_prior": planner,
        "safe_mask": (planner > 0).astype(np.float32),
    }
    baseline, candidate = masked_metrics(BoundedResidualPolicy(), data)
    assert abs(baseline["cross_entropy"] - candidate["cross_entropy"]) < 1e-7
    records = [{"step": np.int64(1), "score_at_state": np.int64(2)},
               {"step": np.int64(2), "score_at_state": np.int64(4)}]
    finalize_episode_targets(records, death_step=10, final_score=7)
    assert records[0]["future_death"] == 0 and records[1]["future_death"] == 1
    assert records[0]["return_target"] == 5 and records[1]["return_target"] == 3


def test_training_step_uses_combined_planner_residual_without_nonfinite_values():
    n = 16
    planner = np.zeros((n, 6), dtype=np.float32)
    planner[:, :2] = 0.5
    target = np.zeros_like(planner)
    target[:, 0] = 0.2
    target[:, 1] = 0.8
    data = {
        "features": np.random.default_rng(3).normal(size=(n, 7, 33, 33)).astype(np.float32),
        "visit_policy": target, "planner_prior": planner,
        "safe_mask": (planner > 0).astype(np.float32),
        "planner_action_scores": np.zeros_like(planner),
        "search_action": np.full(n, 1, dtype=np.int64),
        "opponent_actions": np.zeros((n, 3), dtype=np.int64),
        "return_target": np.zeros(n, dtype=np.float32),
        "future_death": np.zeros(n, dtype=np.float32),
        "tactical": np.zeros(n, dtype=bool), "step": np.arange(n),
    }
    config = {
        "training_seed": 5,
        "model": {"kind": "depthwise-residual-policy-v2", "channels": 4,
                  "blocks": 1, "residual_logit_bound": 0.5},
        "optimizer": {"name": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
                      "batch_size": 8, "epochs": 1, "patience": 1, "min_delta": 1e-4,
                      "gradient_clip": 1.0, "planner_kl_anchor": 0.02,
                      "random_d4_probability": 0.5},
    }
    model, history = train_model(config, data, data)
    assert len(history) == 1 and np.isfinite(history[0]["train_loss"])
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())


def test_confidence_gated_loss_and_slice_gates_are_finite_and_binding():
    n = 4
    planner = np.tile(np.asarray([[0.6, 0.4, 0, 0, 0, 0]], dtype=np.float32), (n, 1))
    target = np.tile(np.asarray([[0.1, 0.9, 0, 0, 0, 0]], dtype=np.float32), (n, 1))
    batch = {
        "features": np.zeros((n, 8, 33, 33), dtype=np.float32),
        "visit_policy": target, "planner_prior": planner,
        "safe_mask": (planner > 0).astype(np.float32),
    }
    optimizer = {
        "loss_kind": "confidence-gated-residual-v1", "search_confidence_margin": 0.15,
        "soft_label_weight": 1.0, "hard_correction_weight": 0.25,
        "preservation_kl_weight": 0.5, "correction_margin_weight": 0.1,
        "nonwait_margin": 0.05, "residual_l2_weight": 0.02,
        "preservation_residual_weight": 0.1,
    }
    loss = residual_training_loss(TargetAwareResidualPolicy(8, 1), batch, optimizer)
    assert torch.isfinite(loss)
    baseline = {"cross_entropy": 1.0, "non_wait_top1_accuracy": 0.5,
                "wait_top1_accuracy": 0.8, "bomb_top1_accuracy": 1.0,
                "high_confidence_cross_entropy": 1.0}
    candidate = {"cross_entropy": 0.9, "non_wait_top1_accuracy": 0.55,
                 "wait_top1_accuracy": 0.79, "bomb_top1_accuracy": 1.0,
                 "high_confidence_cross_entropy": 0.9,
                 "unsafe_probability_mass": 0.0, "residual_state_std_mean": 0.01}
    gates = {"minimum_relative_ce_gain": 0.05,
             "minimum_nonwait_top1_gain_points": 0.03,
             "maximum_unsafe_probability": 0.0,
             "minimum_residual_state_std_mean": 0.001,
             "maximum_wait_top1_drop_points": 0.02,
             "maximum_bomb_top1_drop_points": 0.0,
             "minimum_high_confidence_relative_ce_gain": 0.05}
    assert offline_gate_metrics(baseline, candidate, gates)["passed"]
    candidate["bomb_top1_accuracy"] = 0.9
    assert not offline_gate_metrics(baseline, candidate, gates)["passed"]


def test_stride_sampling_does_not_falsely_mark_a_state_tactical():
    assert sample_decision(False, 8, 4) == (True, False)
    assert sample_decision(False, 9, 4) == (False, False)
    assert sample_decision(True, 9, 4) == (True, True)
    assert sample_decision(True, 9, 4, tactical_stride=2) == (False, True)
    assert sample_decision(True, 10, 4, tactical_stride=2) == (True, True)


def test_productive_bomb_protection_binds_action_and_supervised_target():
    plan = SimpleNamespace(
        prior={"BOMB": 0.8, "WAIT": 0.2},
        bomb_utility=1.8,
        safe_actions=("BOMB", "WAIT"),
    )
    action, protected = teacher_action(plan, "WAIT", True)
    target = teacher_visit_policy({"WAIT": 120, "BOMB": 8}, action, protected)
    assert protected and action == "BOMB"
    assert target[ACTIONS.index("BOMB")] == 1.0 and target.sum() == 1.0

    action, protected = teacher_action(plan, "WAIT", False)
    target = teacher_visit_policy({"WAIT": 120, "BOMB": 8}, action, protected)
    assert not protected and action == "WAIT"
    assert target[ACTIONS.index("WAIT")] == 120 / 128


def test_official_callback_loads_bound_residual_after_planner():
    primary = ROOT / "experiments/checkpoints/model-a-v10-centered-blend-coin-heaven-s103004.npz"
    base_config = json.loads((ROOT / "experiments/configs/v10.1-classic-planner-s109602.json").read_text())
    old = {name: os.environ.get(name) for name in
           ("MODEL_A_V10_CHECKPOINT_PATH", "MODEL_A_V10_CONFIG_PATH")}
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        residual_path = directory / "residual.pt"
        model = TargetAwareResidualPolicy()
        metadata = {"architecture": model.architecture,
                    "channels": 32, "blocks": 3, "bound": 0.5,
                    "config_sha256": "test", "persistent_target_enabled": True,
                    "productive_bomb_protection": True}
        model.save_checkpoint(residual_path, metadata)
        base_config["classic_planner"]["persistent_target_enabled"] = True
        base_config["classic_planner"]["productive_bomb_protection"] = True
        base_config["residual_policy"] = {
            "enabled": True, "checkpoint_path": str(residual_path),
            "checkpoint_sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
            "expected_metadata": {"config_sha256": "test"},
        }
        config_path = directory / "inference.json"
        config_path.write_text(json.dumps(base_config))
        try:
            os.environ["MODEL_A_V10_CHECKPOINT_PATH"] = str(primary)
            os.environ["MODEL_A_V10_CONFIG_PATH"] = str(config_path)
            holder = SimpleNamespace()
            setup(holder)
            state = _open_state((8, 8))
            state.field[10, 8] = 1
            target = holder.v10_controller.searcher.root_policy_transform.prepare(
                state, 0, round_id=1)
            assert target is not None
            assert holder.v10_planner_transform.crate_target == target
            assert holder.v10_residual_transform.crate_target == target
            holder.v10_controller.searcher.search(state, 0, simulations=0, deadline_ms=0)
            assert holder.v10_residual_transform.forward_calls == 1
        finally:
            for name, value in old.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
