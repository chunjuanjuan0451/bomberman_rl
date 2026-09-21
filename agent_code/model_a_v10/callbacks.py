"""Official inference callback for heuristic or explicitly configured v10."""

import hashlib
import json
import os
from pathlib import Path

from .interfaces import BlendedExpertNetwork, LinearExpertNetwork
from .planner import ClassicTacticalPlanner, PlannerRootPolicyTransform
from .runtime import Phase2Controller, SearchBudget, search_budget_for_profile
from .repetition import RepetitionContext


REQUIRED_STUDENT_HEADS = {"policy", "value", "opponent", "risk"}


class _RootPolicyPipeline:
    """Apply independently testable root-only policy transforms in order."""

    def __init__(self, *transforms):
        self.transforms = tuple(transform for transform in transforms if transform is not None)

    def __call__(self, state, player, policy):
        result = policy
        for transform in self.transforms:
            result = transform(state, player, result)
        return result

    def prepare(self, state, player, round_id=None):
        target = None
        for transform in self.transforms:
            if hasattr(transform, "prepare"):
                candidate = transform.prepare(state, player, round_id)
                if candidate is not None:
                    target = candidate
            elif hasattr(transform, "set_crate_target"):
                transform.set_crate_target(target)
        return target

    def protect_action(self, search_action):
        action, protected = search_action, False
        for transform in self.transforms:
            if hasattr(transform, "protect_action"):
                action, applied = transform.protect_action(action)
                protected = protected or applied
        return action, protected


def validated_student_head_weights(config: dict) -> dict[str, float]:
    """Validate the exact JSON schema consumed by the official callback."""
    weights = config.get("student_head_weights", {})
    if set(weights) != REQUIRED_STUDENT_HEADS:
        raise ValueError("v10 inference config requires all student head weights")
    result = {name: float(weights[name]) for name in REQUIRED_STUDENT_HEADS}
    if not all(value >= 0.0 for value in result.values()):
        raise ValueError("v10 inference student head weights must be non-negative")
    return result


def setup(self):
    checkpoint_text = os.environ.get("MODEL_A_V10_CHECKPOINT_PATH")
    config_text = os.environ.get("MODEL_A_V10_CONFIG_PATH")
    seed = int(os.environ.get("MODEL_A_V10_SEED", "0"))
    if checkpoint_text is None and config_text is None:
        self.v10_controller = Phase2Controller(seed_offset=seed)
        return
    if checkpoint_text is None or config_text is None:
        raise ValueError("v10 learned inference requires both checkpoint and config")
    repository_root = Path(__file__).resolve().parents[2]
    checkpoint, config_path = Path(checkpoint_text), Path(config_text)
    # The official agent backend runs callbacks from an agent-specific working
    # directory.  Environment paths are repository-relative by contract, not
    # process-CWD-relative.
    if not checkpoint.is_absolute():
        checkpoint = repository_root / checkpoint
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    config = json.loads(config_path.read_text())
    actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_hash != config.get("checkpoint_sha256"):
        raise ValueError("v10 inference checkpoint hash mismatch")
    weights = validated_student_head_weights(config)
    student = LinearExpertNetwork.load(checkpoint)
    network = BlendedExpertNetwork(
        student, policy_weight=float(weights["policy"]), value_weight=float(weights["value"]),
        opponent_weight=float(weights["opponent"]), risk_weight=float(weights["risk"]),
    )
    anchor_policy = None
    anchor_transform = None
    anchor_config = config.get("v4_anchor", {})
    if anchor_config.get("enabled", False):
        from .v4_adapter import FrozenV4Policy, FrozenV4RootPolicyTransform
        anchor_path = Path(anchor_config["checkpoint_path"])
        if not anchor_path.is_absolute():
            anchor_path = repository_root / anchor_path
        anchor_policy = FrozenV4Policy(
            anchor_path, str(anchor_config["checkpoint_sha256"]), seed=seed,
            temperature=float(anchor_config.get("policy_temperature", 0.25)),
        )
        anchor_transform = FrozenV4RootPolicyTransform(anchor_policy)
    forbidden = ("BOMB",) if config.get("scenario") == "coin-heaven" else ()
    repetition = config.get("repetition", {})
    budget = search_budget_for_profile(config.get("search_budget_profile", "research-m4-v1"))
    context = RepetitionContext(
        enabled=bool(repetition.get("enabled", False)),
        history_window=int(repetition.get("history_window", 16)),
        activation_steps=int(repetition.get("activation_steps", 8)),
        lambda_=float(repetition.get("lambda", 0.6)),
        minimum_multiplier=float(repetition.get("minimum_multiplier", 0.05)),
    ) if repetition else None
    planner_config = config.get("classic_planner", {})
    planner_transform = None
    if planner_config.get("enabled", False):
        if config.get("scenario") != "classic":
            raise ValueError("v10 classic planner may only be enabled for scenario='classic'")
        planner_transform = PlannerRootPolicyTransform(ClassicTacticalPlanner(
            horizon=int(planner_config.get("horizon", 8)),
            prior_weight=float(planner_config.get("prior_weight", 0.80)),
            temperature=float(planner_config.get("temperature", 0.85)),
            crate_frontier_weight=float(planner_config.get("crate_frontier_weight", 0.32)),
            bomb_efficiency_penalty=float(planner_config.get("bomb_efficiency_penalty", 0.0)),
        ),
            persistent_target_enabled=bool(
                planner_config.get("persistent_target_enabled", False)),
            productive_bomb_protection=bool(
                planner_config.get("productive_bomb_protection", False)),
        )
    self.v10_planner_transform = planner_transform
    residual_config = config.get("residual_policy", {})
    residual_transform = None
    if residual_config.get("enabled", False):
        if planner_transform is None:
            raise ValueError("v10 residual policy requires the classic planner")
        import torch
        from .residual_policy import PlannerResidualTransform, load_residual_checkpoint
        if hasattr(torch.backends, "nnpack"):
            torch.backends.nnpack.set_flags(False)
        residual_path = Path(residual_config["checkpoint_path"])
        if not residual_path.is_absolute():
            residual_path = repository_root / residual_path
        residual_hash = hashlib.sha256(residual_path.read_bytes()).hexdigest()
        if residual_hash != residual_config.get("checkpoint_sha256"):
            raise ValueError("v10 residual policy checkpoint hash mismatch")
        residual_model, residual_metadata = load_residual_checkpoint(
            residual_path, residual_config.get("expected_metadata", {}))
        if residual_metadata.get("architecture") in {
                "target-aware-action-conditioned-residual-policy-v4",
                "planner-aware-action-conditioned-residual-policy-v5",
                "planner-aware-gated-residual-policy-v6",
        }:
            if (not planner_config.get("persistent_target_enabled", False)
                    or not planner_config.get("productive_bomb_protection", False)):
                raise ValueError(
                    "target-aware residual requires persistent target and BOMB protection")
            if (residual_metadata.get("persistent_target_enabled") is not True
                    or residual_metadata.get("productive_bomb_protection") is not True):
                raise ValueError("target-aware residual checkpoint lacks planner-state binding")
        residual_transform = PlannerResidualTransform(residual_model)
    self.v10_residual_transform = residual_transform
    nonlinear = config.get("nonlinear_policy", {})
    nonlinear_transform = None
    if nonlinear.get("enabled", False):
        if residual_transform is not None:
            raise ValueError("v10 residual and legacy nonlinear policies are mutually exclusive")
        # Keep the NumPy-only v10.1 planner usable even when optional PyTorch
        # is absent.  Tournament images still validate PyTorch separately.
        from .policy_head import NonlinearPolicyHead, RootPolicyBlender
        nonlinear_path = Path(nonlinear["checkpoint_path"])
        if not nonlinear_path.is_absolute():
            nonlinear_path = repository_root / nonlinear_path
        nonlinear_hash = hashlib.sha256(nonlinear_path.read_bytes()).hexdigest()
        if nonlinear_hash != nonlinear.get("checkpoint_sha256"):
            raise ValueError("v10 nonlinear policy checkpoint hash mismatch")
        expected = nonlinear.get("expected_metadata", {})
        policy_model, _ = NonlinearPolicyHead.load_checkpoint(nonlinear_path, expected)
        nonlinear_transform = RootPolicyBlender(
            policy_model, weight=float(nonlinear.get("weight", 0.0)))
    transforms = tuple(transform for transform in
                       (anchor_transform, planner_transform, residual_transform, nonlinear_transform)
                       if transform is not None)
    root_policy_transform = (None if not transforms else transforms[0] if len(transforms) == 1
                             else _RootPolicyPipeline(*transforms))
    hybrid_config = config.get("v4_tactical_hybrid", {})
    if hybrid_config.get("enabled", False):
        from .v4_hybrid import V4AnchoredHybridController
        if anchor_policy is None:
            raise ValueError("v4 tactical hybrid requires an enabled v4 anchor")
        if config.get("scenario") != "classic" or planner_transform is None:
            raise ValueError("v4 tactical hybrid requires the classic planner")
        controller_class = V4AnchoredHybridController
        budget = SearchBudget(
            budget.normal_simulations, budget.tactical_simulations,
            budget.normal_min_simulations,
            int(hybrid_config.get("minimum_completed_simulations", 32)),
            budget.normal_deadline_ms, budget.tactical_deadline_ms,
        )
        controller_options = {
            "anchor_policy": anchor_policy,
            "minimum_action_visits": int(hybrid_config.get("minimum_action_visits", 8)),
            "minimum_value_advantage": float(
                hybrid_config.get("minimum_value_advantage", 0.10)),
            "crate_progress_weight": float(config.get("crate_progress_weight", 0.0)),
        }
    else:
        controller_class = Phase2Controller
        controller_options = {}
    self.v10_controller = controller_class(
        network, forbidden_root_actions=forbidden, seed_offset=seed,
        repetition_context=context, budget=budget,
        root_policy_transform=root_policy_transform,
        **controller_options)


def act(self, game_state: dict) -> str:
    return self.v10_controller.act(game_state)
