"""Pre-registered v10.2 planner-residual training; dry-run by default."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v10.interfaces import (  # noqa: E402
    ACTIONS, BlendedExpertNetwork, LinearExpertNetwork,
)
from agent_code.model_a_v10.planner import (  # noqa: E402
    ClassicTacticalPlanner, PlannerRootPolicyTransform,
)
from agent_code.model_a_v10.residual_policy import (  # noqa: E402
    ActionConditionedResidualPolicy, BoundedResidualPolicy, D4_TRANSFORMS,
    TargetAwareResidualPolicy, padded_centered_features,
    planner_residual_policy, transform_action_vector, transform_features,
)
from agent_code.model_a_v10.runtime import state_from_game_state  # noqa: E402
from agent_code.model_a_v10.search import StochasticPUCT  # noqa: E402
from agent_code.model_a_v10.simulator import SimAgent, SimState, step  # noqa: E402
from environment import BombeRLeWorld  # noqa: E402


REQUIRED_SHARD_KEYS = {
    "features", "visit_policy", "planner_prior", "safe_mask",
    "planner_action_scores", "search_action", "opponent_actions",
    "return_target", "future_death", "tactical", "step",
}
SOURCE_PATHS = (
    "agent_code/model_a_v10/callbacks.py",
    "agent_code/model_a_v10/interfaces.py",
    "agent_code/model_a_v10/planner.py",
    "agent_code/model_a_v10/residual_policy.py",
    "agent_code/model_a_v10/runtime.py",
    "agent_code/model_a_v10/search.py",
    "agent_code/model_a_v10/simulator.py",
    "environment.py",
    "tools/v10_classic_residual_train.py",
    "tools/v10_residual_latency.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    return {path: sha256(ROOT / path) for path in SOURCE_PATHS}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    run_id = config.get("run_id")
    profiles = {
        "model-a-v10.2-classic-residual-policy-s103132": {
            "protocol_version": 2, "training_seed": 103132,
            "episodes": 120, "train_episodes": 96, "validation_episodes": 24,
            "training_episode_seed_range": [103132, 103227],
            "validation_episode_seed_range": [103228, 103251],
            "opponent_count": 3, "opponent_distribution": "seeded-random-agent-v1",
            "model_kind": "depthwise-residual-policy-v2",
        },
        "model-a-v10.3-task2-action-residual-s103252": {
            "protocol_version": 3, "training_seed": 103252,
            "episodes": 48, "train_episodes": 38, "validation_episodes": 10,
            "training_episode_seed_range": [103252, 103289],
            "validation_episode_seed_range": [103290, 103299],
            "opponent_count": 0, "opponent_distribution": "none",
            "model_kind": "action-conditioned-residual-policy-v3",
        },
        "model-a-v10.4-task2-frontier-residual-s103372": {
            "protocol_version": 4, "training_seed": 103372,
            "episodes": 48, "train_episodes": 38, "validation_episodes": 10,
            "training_episode_seed_range": [103372, 103409],
            "validation_episode_seed_range": [103410, 103419],
            "opponent_count": 0, "opponent_distribution": "none",
            "model_kind": "action-conditioned-residual-policy-v3",
        },
        "model-a-v10.5-task2-persistent-teacher-residual-s105000": {
            "protocol_version": 5, "training_seed": 105000,
            "episodes": 48, "train_episodes": 38, "validation_episodes": 10,
            "training_episode_seed_range": [105000, 105037],
            "validation_episode_seed_range": [105038, 105047],
            "opponent_count": 0, "opponent_distribution": "none",
            "model_kind": "action-conditioned-residual-policy-v3",
        },
        "model-a-v10.6-task2-target-aware-correction-s105200": {
            "protocol_version": 6, "training_seed": 105200,
            "episodes": 48, "train_episodes": 38, "validation_episodes": 10,
            "training_episode_seed_range": [105200, 105237],
            "validation_episode_seed_range": [105238, 105247],
            "opponent_count": 0, "opponent_distribution": "none",
            "model_kind": "target-aware-action-conditioned-residual-policy-v4",
            "feature_schema": "padded-centered-33x33-target-distance-v3",
        },
    }
    if run_id not in profiles:
        raise ValueError(f"unsupported residual training protocol: {run_id!r}")
    profile = profiles[run_id]
    exact = {
        "protocol_version": profile["protocol_version"],
        "scenario": "classic",
        "episodes": profile["episodes"],
        "train_episodes": profile["train_episodes"],
        "validation_episodes": profile["validation_episodes"],
        "opponent_count": profile["opponent_count"],
        "opponent_distribution": profile["opponent_distribution"],
        "feature_schema": profile.get("feature_schema", "padded-centered-33x33-v2"),
        "teacher_search_simulations": 128,
        "search_horizon_plies": 8,
        "state_sample_stride": 4,
        "always_sample_tactical": True,
        "training": True,
        "h2h": False,
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise ValueError(f"{run_id} requires {key}={expected!r}")
    if config.get("training_seed") != profile["training_seed"]:
        raise ValueError("training-seed binding mismatch")
    if config["episodes"] != config["train_episodes"] + config["validation_episodes"]:
        raise ValueError("episode split mismatch")
    if config.get("training_episode_seed_range") != profile["training_episode_seed_range"]:
        raise ValueError("training episode seed range mismatch")
    if config.get("validation_episode_seed_range") != profile["validation_episode_seed_range"]:
        raise ValueError("validation episode seed range mismatch")
    if set(config.get("required_shard_keys", ())) != REQUIRED_SHARD_KEYS:
        raise ValueError("required trajectory shard schema mismatch")
    model = config["model"]
    if model != {"kind": profile["model_kind"], "channels": 32,
                  "blocks": 3, "residual_logit_bound": 0.5,
                  "zero_initialized_output": True}:
        raise ValueError("residual model configuration mismatch")
    expected_feature_channels = [
        "field_with_wall_padding", "bomb_timer", "explosion", "self",
        "opponents", "visible_coins_only", "planner_allowed_destinations",
    ]
    if profile["protocol_version"] == 6:
        expected_feature_channels.append("persistent_target_wall_distance")
    if config.get("feature_channels") != expected_feature_channels:
        raise ValueError("residual feature-channel binding mismatch")
    if profile["protocol_version"] in (3, 4, 5, 6):
        if config.get("teacher_crate_progress_weight") != 0.08:
            raise ValueError("task2 teacher crate-progress weight mismatch")
        if profile["protocol_version"] in (4, 5, 6):
            if config.get("teacher_crate_frontier_weight") != 0.32:
                raise ValueError("task2 teacher crate-frontier weight mismatch")
            if config.get("teacher_bomb_efficiency_penalty") != 0.0:
                raise ValueError("task2 teacher bomb-efficiency penalty mismatch")
        expected_tactical_stride = 4 if profile["protocol_version"] in (5, 6) else 2
        if config.get("tactical_sample_stride") != expected_tactical_stride:
            raise ValueError("task2 tactical sampling stride mismatch")
        if profile["protocol_version"] in (5, 6):
            if config.get("teacher_productive_bomb_protection") is not True:
                raise ValueError("v10.5 requires productive-BOMB protection")
            if config.get("teacher_persistent_target") is not True:
                raise ValueError("v10.5 requires the selected persistent target arm")
            gate_path = ROOT / config["teacher_gate_report"]
            if sha256(gate_path) != config.get("teacher_gate_report_sha256"):
                raise ValueError("v10.5 teacher gate report hash mismatch")
            gate_report = json.loads(gate_path.read_text())
            if not gate_report.get("teacher_gate", {}).get("passed"):
                raise ValueError("v10.5 teacher gate did not pass")
            if gate_report["teacher_gate"].get("selected_arm") != "persistent_target":
                raise ValueError("v10.5 teacher gate selected a different arm")
            if config.get("teacher_selected_arm") != "persistent_target":
                raise ValueError("v10.5 config selected a different teacher arm")
            preregistration_path = ROOT / config["teacher_gate_preregistration"]
            if sha256(preregistration_path) != config.get(
                    "teacher_gate_preregistration_sha256"):
                raise ValueError("v10.5 teacher preregistration hash mismatch")
            training_sources = config.get("training_source_sha256", {})
            required_training_sources = {
                "agent_code/model_a_v10/planner.py",
                "agent_code/model_a_v10/residual_policy.py",
                "agent_code/model_a_v10/search.py",
                "tools/v10_classic_residual_train.py",
            }
            if profile["protocol_version"] == 6:
                required_training_sources.update({
                    "agent_code/model_a_v10/callbacks.py",
                    "agent_code/model_a_v10/runtime.py",
                })
            if set(training_sources) != required_training_sources:
                raise ValueError("v10.5 training source bindings are incomplete")
            for relative, expected in training_sources.items():
                if sha256(ROOT / relative) != expected:
                    raise ValueError(f"v10.5 training source hash mismatch: {relative}")
        expected_optimizer = {
            "name": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.0001,
            "batch_size": 128, "epochs": 30, "patience": 5, "min_delta": 0.0001,
            "gradient_clip": 1.0, "planner_kl_anchor": 0.2,
            "random_d4_probability": 0.5, "soft_label_weight": 0.25,
            "nonwait_sample_weight": 2.0, "correction_sample_weight": 2.0,
            "hard_label_weight": 1.0, "nonwait_margin_weight": 0.5,
            "nonwait_margin": 0.05, "search_confidence_margin": 0.15,
            "residual_l2_weight": 0.02,
        }
        if profile["protocol_version"] == 6:
            expected_optimizer = {
                "name": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.0001,
                "batch_size": 128, "epochs": 30, "patience": 5,
                "min_delta": 0.0001, "gradient_clip": 1.0,
                "loss_kind": "confidence-gated-residual-v1",
                "random_d4_probability": 0.5,
                "soft_label_weight": 1.0,
                "hard_correction_weight": 0.25,
                "preservation_kl_weight": 0.5,
                "correction_margin_weight": 0.1,
                "nonwait_margin": 0.05,
                "search_confidence_margin": 0.15,
                "residual_l2_weight": 0.02,
                "preservation_residual_weight": 0.1,
            }
        if config.get("optimizer") != expected_optimizer:
            raise ValueError("task2 optimizer/loss configuration mismatch")
        if config.get("maximum_labeled_states") != 12000:
            raise ValueError("task2 label cap mismatch")
        if profile["protocol_version"] == 6:
            expected_offline_gate = {
                "minimum_relative_ce_gain": 0.05,
                "minimum_nonwait_top1_gain_points": 0.03,
                "maximum_unsafe_probability": 0.0,
                "minimum_residual_state_std_mean": 0.001,
                "maximum_wait_top1_drop_points": 0.02,
                "maximum_bomb_top1_drop_points": 0.0,
                "minimum_high_confidence_relative_ce_gain": 0.05,
                "maximum_root_forward_p99_ms": 5.0,
            }
            if config.get("offline_gate") != expected_offline_gate:
                raise ValueError("v10.6 offline gate binding mismatch")
    return config


def observer_game_state(state: SimState) -> dict:
    """Expose only fields available to the official controlled agent."""
    controlled = state.agents[0]
    return {
        "round": 1,
        "step": state.step_count,
        "field": state.field.copy(),
        "self": (controlled.name, controlled.score, controlled.bombs_left,
                 (controlled.x, controlled.y)),
        "others": [(agent.name, agent.score, agent.bombs_left, (agent.x, agent.y))
                   for agent in state.agents[1:] if agent.alive],
        "bombs": [((bomb.x, bomb.y), bomb.timer) for bomb in state.bombs],
        "coins": [position for position, collectable in state.coins.items() if collectable],
        "explosion_map": state.explosion_map(),
        "user_input": "WAIT",
    }


def initial_state(seed: int, max_steps: int, opponent_count: int = 3) -> SimState:
    agents = [SimAgent("model_a_v10", 1, 1)] + [
        SimAgent(f"seeded_random_agent_{index}", 1, 1) for index in range(opponent_count)
    ]
    carrier = type("Carrier", (), {})()
    carrier.rng = np.random.default_rng(seed)
    carrier.args = type("Args", (), {"scenario": "classic"})()
    carrier.agents = agents
    field, coins, positioned = BombeRLeWorld.build_arena(carrier)
    return SimState(field, positioned, {(coin.x, coin.y): coin.collectable for coin in coins},
                    max_steps=max_steps)


def random_action(rng: np.random.Generator) -> str:
    return str(rng.choice(("RIGHT", "LEFT", "UP", "DOWN", "BOMB"),
                          p=(0.23, 0.23, 0.23, 0.23, 0.08)))


def policy_array(policy: dict[str, float]) -> np.ndarray:
    result = np.zeros(len(ACTIONS), dtype=np.float32)
    for action, probability in policy.items():
        result[ACTIONS.index(action)] = float(probability)
    return result


def visit_policy(visits: dict[str, int]) -> np.ndarray:
    total = sum(visits.values())
    if total <= 0:
        raise ValueError("teacher search returned no root visits")
    return policy_array({action: count / total for action, count in visits.items()})


def teacher_action(
    plan,
    search_action: str,
    productive_bomb_protection: bool,
) -> tuple[str, bool]:
    """Optionally retain a planner-safe productive bomb over a search override."""
    greedy_action = max(plan.prior, key=plan.prior.get)
    protected = bool(
        productive_bomb_protection
        and greedy_action == "BOMB"
        and plan.bomb_utility > 0.0
        and "BOMB" in plan.safe_actions
        and search_action != "BOMB"
    )
    return ("BOMB" if protected else search_action), protected


def teacher_visit_policy(visits: dict[str, int], action: str, protected: bool) -> np.ndarray:
    """Bind the supervised target to a protected teacher action when one is used."""
    if not protected:
        return visit_policy(visits)
    return policy_array({action: 1.0})


def build_teacher(config: dict):
    checkpoint = ROOT / config["teacher_checkpoint"]
    planner_config = ROOT / config["planner_inference_config"]
    if sha256(checkpoint) != config["teacher_checkpoint_sha256"]:
        raise ValueError("teacher checkpoint hash mismatch")
    if sha256(planner_config) != config["planner_inference_config_sha256"]:
        raise ValueError("planner config hash mismatch")
    deployed = json.loads(planner_config.read_text())
    planner_settings = deployed["classic_planner"]
    student = LinearExpertNetwork.load(checkpoint)
    network = BlendedExpertNetwork(student, policy_weight=0.0, value_weight=0.1,
                                   opponent_weight=0.0, risk_weight=0.0)
    planner = ClassicTacticalPlanner(
        horizon=int(planner_settings["horizon"]),
        prior_weight=float(planner_settings["prior_weight"]),
        temperature=float(planner_settings["temperature"]),
        crate_frontier_weight=float(config.get("teacher_crate_frontier_weight", 0.32)),
        bomb_efficiency_penalty=float(config.get("teacher_bomb_efficiency_penalty", 0.0)),
    )
    transform = PlannerRootPolicyTransform(
        planner,
        persistent_target_enabled=bool(config.get("teacher_persistent_target", False)),
        productive_bomb_protection=bool(
            config.get("teacher_productive_bomb_protection", False)),
    )
    searcher = StochasticPUCT(
                              network, horizon=config["search_horizon_plies"],
                              crate_progress_weight=float(
                                  config.get("teacher_crate_progress_weight", 0.0)),
                              root_policy_transform=transform)
    return network, planner, transform, searcher


def sample_decision(plan_tactical: bool, step_count: int, stride: int,
                    tactical_stride: int = 1) -> tuple[bool, bool]:
    """Keep sampling cadence separate from the factual tactical label."""
    return bool(step_count % stride == 0
                or (plan_tactical and step_count % tactical_stride == 0)), bool(plan_tactical)


def finalize_episode_targets(records: list[dict], death_step: int | None,
                             final_score: int) -> None:
    """Attach return and eight-step future-death labels after the episode."""
    for record in records:
        record["return_target"] = np.float32(final_score - int(record.pop("score_at_state")))
        record["future_death"] = np.float32(
            death_step is not None and death_step <= int(record["step"]) + 8)


def collect_episode(seed: int, config: dict, teacher, *, return_summary: bool = False):
    network, planner, transform, searcher = teacher
    opponent_count = int(config["opponent_count"])
    state = initial_state(seed, int(config.get("max_steps", 400)), opponent_count)
    rng = np.random.default_rng(seed + 991)
    records: list[dict] = []
    while (not state.ended and state.agents[0].alive
           and (opponent_count == 0 or any(agent.alive for agent in state.agents[1:]))):
        observed = state_from_game_state(observer_game_state(state))
        base = network.evaluate(observed, 0).policy
        persistent_target = transform.prepare(observed, 0, round_id=1)
        plan = planner.plan(observed, 0, base, crate_target=persistent_target)
        should_label, tactical_label = sample_decision(
            plan.tactical, state.step_count, int(config["state_sample_stride"]),
            int(config.get("tactical_sample_stride", 1)))
        opponent_actions = [random_action(rng) if agent.alive else "WAIT"
                            for agent in state.agents[1:]]
        if should_label:
            result = searcher.search(
                observed, 0, simulations=int(config["teacher_search_simulations"]),
                seed=seed * 1000 + state.step_count, deadline_ms=None,
            )
            final_plan = transform.last_plan
            if final_plan is None or result.simulations != config["teacher_search_simulations"]:
                raise RuntimeError("teacher did not complete the fixed search budget")
            safe = np.zeros(len(ACTIONS), dtype=np.float32)
            for action in result.raw_prior:
                safe[ACTIONS.index(action)] = 1.0
            scores = np.full(len(ACTIONS), -1e9, dtype=np.float32)
            for action, score in final_plan.action_scores.items():
                scores[ACTIONS.index(action)] = float(score)
            root_action, protected = transform.protect_action(result.action)
            target = teacher_visit_policy(result.root_visits, root_action, protected)
            if float((target * (1.0 - safe)).sum()) != 0.0:
                raise RuntimeError("teacher visit mass escaped the planner safety mask")
            records.append({
                "features": padded_centered_features(
                    observed, 0, safe_actions=tuple(result.raw_prior),
                    crate_target=final_plan.crate_target,
                    include_target_distance=(
                        config["model"]["kind"] == TargetAwareResidualPolicy.architecture),
                ),
                "visit_policy": target,
                "planner_prior": policy_array(result.raw_prior),
                "safe_mask": safe,
                "planner_action_scores": scores,
                "search_action": np.int64(ACTIONS.index(root_action)),
                "opponent_actions": np.asarray(
                    [ACTIONS.index(action) for action in opponent_actions], dtype=np.int64),
                "return_target": np.float32(0.0),
                "future_death": np.float32(0.0),
                "tactical": np.bool_(tactical_label),
                "step": np.int64(state.step_count),
                "score_at_state": np.int64(state.agents[0].score),
            })
        else:
            root_action = max(plan.prior, key=plan.prior.get)
        joint = [root_action, *opponent_actions]
        active = state.active_ids()
        state = step(state, joint, tuple(int(value) for value in rng.permutation(len(active))))

    death_step = state.step_count if not state.agents[0].alive else None
    final_score = state.agents[0].score
    finalize_episode_targets(records, death_step, final_score)
    if not records:
        raise ValueError("episode produced no labels")
    shard = {key: np.asarray([record[key] for record in records]) for key in REQUIRED_SHARD_KEYS}
    summary = {
        "seed": seed, "steps": state.step_count, "score": final_score,
        "survived": death_step is None,
        "cleared_task2_board": bool(
            opponent_count == 0 and not np.any(state.field == 1)
            and not any(state.coins.values())),
        "crates_remaining": int(np.count_nonzero(state.field == 1)),
        "collectable_coins_remaining": int(sum(state.coins.values())),
        "labels": len(records),
    }
    return (shard, summary) if return_summary else shard


def concatenate(shards: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if any(set(shard) != REQUIRED_SHARD_KEYS for shard in shards):
        raise ValueError("trajectory shard schema mismatch")
    return {key: np.concatenate([shard[key] for shard in shards])
            for key in sorted(REQUIRED_SHARD_KEYS)}


def validate_dataset(data: dict[str, np.ndarray], opponent_count: int,
                     feature_channels: int = 7) -> None:
    count = len(data["features"])
    if data["features"].shape != (count, feature_channels, 33, 33):
        raise ValueError("residual feature dataset shape mismatch")
    if data["opponent_actions"].shape != (count, opponent_count):
        raise ValueError("opponent-action dataset shape mismatch")
    for key in ("visit_policy", "planner_prior", "safe_mask", "planner_action_scores"):
        if data[key].shape != (count, len(ACTIONS)):
            raise ValueError(f"{key} dataset shape mismatch")
    finite_keys = ("features", "visit_policy", "planner_prior", "safe_mask",
                   "return_target", "future_death")
    if any(not np.isfinite(data[key]).all() for key in finite_keys):
        raise FloatingPointError("non-finite residual training data")
    if not np.allclose(data["visit_policy"].sum(1), 1.0, atol=1e-6):
        raise ValueError("visit policies are not normalized")
    if not np.allclose(data["planner_prior"].sum(1), 1.0, atol=1e-6):
        raise ValueError("planner priors are not normalized")
    if np.any(data["safe_mask"].sum(1) <= 0):
        raise ValueError("a training state has no planner-allowed action")
    if float((data["visit_policy"] * (1.0 - data["safe_mask"])).sum()) != 0.0:
        raise ValueError("visit policy contains unsafe probability")
    if np.any((data["opponent_actions"] < 0) | (data["opponent_actions"] >= len(ACTIONS))):
        raise ValueError("invalid opponent action index")


def augment_batch(batch: dict[str, np.ndarray], probability: float,
                  rng: np.random.Generator) -> dict[str, np.ndarray]:
    result = {key: value.copy() for key, value in batch.items()}
    for index in range(len(result["features"])):
        if rng.random() >= probability:
            continue
        transform = str(rng.choice(D4_TRANSFORMS[1:]))
        result["features"][index] = transform_features(result["features"][index], transform)
        for key in ("visit_policy", "planner_prior", "safe_mask", "planner_action_scores"):
            result[key][index] = transform_action_vector(result[key][index], transform)
        one_hot = np.zeros(len(ACTIONS), dtype=np.int64)
        one_hot[int(result["search_action"][index])] = 1
        result["search_action"][index] = int(np.argmax(transform_action_vector(one_hot, transform)))
        if "opponent_actions" in result:
            for opponent in range(result["opponent_actions"].shape[1]):
                one_hot.fill(0)
                one_hot[int(result["opponent_actions"][index, opponent])] = 1
                result["opponent_actions"][index, opponent] = int(
                    np.argmax(transform_action_vector(one_hot, transform)))
    return result


def masked_metrics(model, data: dict[str, np.ndarray]) -> tuple[dict, dict]:
    features = torch.from_numpy(data["features"].astype(np.float32))
    target = torch.from_numpy(data["visit_policy"].astype(np.float32))
    planner = torch.from_numpy(data["planner_prior"].astype(np.float32))
    safe = torch.from_numpy(data["safe_mask"].astype(np.float32))
    with torch.inference_mode():
        residual = model(features)
        candidate = planner_residual_policy(planner, safe, residual)
        baseline = planner / planner.sum(dim=-1, keepdim=True)
    baseline_ce = float(-(target * torch.log(baseline.clamp_min(1e-12))).sum(-1).mean())
    candidate_ce = float(-(target * torch.log(candidate.clamp_min(1e-12))).sum(-1).mean())
    target_action = target.argmax(-1)
    non_wait = target_action != ACTIONS.index("WAIT")
    if not torch.any(non_wait):
        raise ValueError("validation data contains no non-WAIT teacher target")
    baseline_accuracy = float((baseline.argmax(-1)[non_wait] == target_action[non_wait]).float().mean())
    candidate_accuracy = float((candidate.argmax(-1)[non_wait] == target_action[non_wait]).float().mean())
    top_two = torch.topk(target, k=2, dim=-1).values
    confident = (top_two[:, 0] - top_two[:, 1]) >= 0.15
    wait = target_action == ACTIONS.index("WAIT")
    bomb = target_action == ACTIONS.index("BOMB")

    def subset_ce(probabilities, mask):
        if not torch.any(mask):
            return None
        return float(-(target[mask] * torch.log(
            probabilities[mask].clamp_min(1e-12))).sum(-1).mean())

    def subset_accuracy(probabilities, mask):
        if not torch.any(mask):
            return None
        return float((probabilities.argmax(-1)[mask] == target_action[mask]).float().mean())

    unsafe_mass = float((candidate * (1.0 - safe)).sum())
    residual_abs = residual.abs()
    return ({"cross_entropy": baseline_ce, "non_wait_top1_accuracy": baseline_accuracy,
             "wait_top1_accuracy": subset_accuracy(baseline, wait),
             "bomb_top1_accuracy": subset_accuracy(baseline, bomb),
             "high_confidence_cross_entropy": subset_ce(baseline, confident),
             "low_confidence_cross_entropy": subset_ce(baseline, ~confident)},
            {"cross_entropy": candidate_ce, "non_wait_top1_accuracy": candidate_accuracy,
             "wait_top1_accuracy": subset_accuracy(candidate, wait),
             "bomb_top1_accuracy": subset_accuracy(candidate, bomb),
             "high_confidence_cross_entropy": subset_ce(candidate, confident),
             "low_confidence_cross_entropy": subset_ce(candidate, ~confident),
             "wait_prediction_fraction": float(
                 (candidate.argmax(-1) == ACTIONS.index("WAIT")).float().mean()),
             "unsafe_probability_mass": unsafe_mass,
             "residual_abs_max": float(residual_abs.max()),
             "residual_abs_p50": float(torch.quantile(residual_abs.flatten(), 0.50)),
             "residual_abs_p90": float(torch.quantile(residual_abs.flatten(), 0.90)),
             "residual_abs_p99": float(torch.quantile(residual_abs.flatten(), 0.99)),
             "residual_saturation_fraction": float(
                 (residual_abs >= 0.45).float().mean()),
             "residual_action_flip_fraction": float(
                 (candidate.argmax(-1) != baseline.argmax(-1)).float().mean()),
             "residual_state_std_mean": float(residual.std(dim=0, unbiased=False).mean())})


def offline_gate_metrics(baseline: dict, candidate: dict, gates: dict) -> dict:
    """Evaluate the pre-registered aggregate and teacher-action slice gates."""
    relative_ce_gain = ((baseline["cross_entropy"] - candidate["cross_entropy"])
                        / max(baseline["cross_entropy"], 1e-12))
    nonwait_gain = (candidate["non_wait_top1_accuracy"]
                    - baseline["non_wait_top1_accuracy"])

    def accuracy_drop(key: str) -> float | None:
        if baseline.get(key) is None or candidate.get(key) is None:
            return None
        return float(baseline[key] - candidate[key])

    def relative_ce_slice_gain(key: str) -> float | None:
        if baseline.get(key) is None or candidate.get(key) is None:
            return None
        return float((baseline[key] - candidate[key]) / max(baseline[key], 1e-12))

    wait_drop = accuracy_drop("wait_top1_accuracy")
    bomb_drop = accuracy_drop("bomb_top1_accuracy")
    high_confidence_gain = relative_ce_slice_gain("high_confidence_cross_entropy")
    required_slices_present = bool(
        ("maximum_wait_top1_drop_points" not in gates or wait_drop is not None)
        and ("maximum_bomb_top1_drop_points" not in gates or bomb_drop is not None)
        and ("minimum_high_confidence_relative_ce_gain" not in gates
             or high_confidence_gain is not None)
    )
    passed = bool(
        required_slices_present
        and relative_ce_gain >= gates.get("minimum_relative_ce_gain", float("-inf"))
        and nonwait_gain >= gates.get("minimum_nonwait_top1_gain_points", float("-inf"))
        and candidate["unsafe_probability_mass"]
        <= gates.get("maximum_unsafe_probability", float("inf"))
        and candidate["residual_state_std_mean"]
        >= gates.get("minimum_residual_state_std_mean", 0.0)
        and (wait_drop is None or wait_drop
             <= gates.get("maximum_wait_top1_drop_points", float("inf")))
        and (bomb_drop is None or bomb_drop
             <= gates.get("maximum_bomb_top1_drop_points", float("inf")))
        and (high_confidence_gain is None or high_confidence_gain
             >= gates.get("minimum_high_confidence_relative_ce_gain", float("-inf")))
    )
    return {
        "passed": passed,
        "relative_ce_gain": relative_ce_gain,
        "non_wait_top1_gain_points": nonwait_gain,
        "wait_top1_drop_points": wait_drop,
        "bomb_top1_drop_points": bomb_drop,
        "high_confidence_relative_ce_gain": high_confidence_gain,
        "unsafe_probability_mass": candidate["unsafe_probability_mass"],
        "residual_state_std_mean": candidate["residual_state_std_mean"],
    }


def residual_training_loss(model, batch: dict[str, np.ndarray], optimizer_config: dict):
    features = torch.from_numpy(batch["features"].astype(np.float32))
    target = torch.from_numpy(batch["visit_policy"].astype(np.float32))
    planner = torch.from_numpy(batch["planner_prior"].astype(np.float32))
    safe = torch.from_numpy(batch["safe_mask"].astype(np.float32))
    residual = model(features)
    candidate = planner_residual_policy(planner, safe, residual)
    log_candidate = torch.log(candidate.clamp_min(1e-12))
    target_action = target.argmax(-1)
    planner_action = planner.argmax(-1)
    top_two = torch.topk(target, k=2, dim=-1).values
    confident = (top_two[:, 0] - top_two[:, 1]
                 >= float(optimizer_config.get("search_confidence_margin", 0.0)))
    if optimizer_config.get("loss_kind") == "confidence-gated-residual-v1":
        correction = confident & (target_action != planner_action)
        preservation = ~correction
        soft_ce = -(target * log_candidate).sum(-1).mean()
        hard_correction = (
            -log_candidate[correction].gather(
                1, target_action[correction][:, None]).mean()
            if torch.any(correction) else soft_ce.new_zeros(())
        )
        planner_probability = planner / planner.sum(dim=-1, keepdim=True)
        planner_kl = (planner_probability * (
            torch.log(planner_probability.clamp_min(1e-12))
            - log_candidate)).sum(-1)
        preservation_kl = (planner_kl[preservation].mean()
                           if torch.any(preservation) else soft_ce.new_zeros(()))
        correction_nonwait = correction & (target_action != ACTIONS.index("WAIT"))
        if torch.any(correction_nonwait):
            target_logp = log_candidate.gather(1, target_action[:, None]).squeeze(1)
            wait_logp = log_candidate[:, ACTIONS.index("WAIT")]
            correction_margin = torch.relu(
                wait_logp[correction_nonwait] - target_logp[correction_nonwait]
                + float(optimizer_config.get("nonwait_margin", 0.0))).mean()
        else:
            correction_margin = soft_ce.new_zeros(())
        preservation_residual = (residual[preservation].square().mean()
                                 if torch.any(preservation) else soft_ce.new_zeros(()))
        return (
            float(optimizer_config["soft_label_weight"]) * soft_ce
            + float(optimizer_config["hard_correction_weight"]) * hard_correction
            + float(optimizer_config["preservation_kl_weight"]) * preservation_kl
            + float(optimizer_config["correction_margin_weight"]) * correction_margin
            + float(optimizer_config["residual_l2_weight"]) * residual.square().mean()
            + float(optimizer_config["preservation_residual_weight"])
            * preservation_residual
        )
    supervised_action = torch.where(confident, target_action, planner_action)
    non_wait = supervised_action != ACTIONS.index("WAIT")
    correction = supervised_action != planner_action
    sample_weight = torch.ones_like(target_action, dtype=torch.float32)
    sample_weight[non_wait] = float(optimizer_config.get("nonwait_sample_weight", 1.0))
    sample_weight[correction] *= float(optimizer_config.get("correction_sample_weight", 1.0))
    soft_ce = -(target * log_candidate).sum(-1)
    soft_ce = (soft_ce * sample_weight).sum() / sample_weight.sum()
    hard_ce = -log_candidate.gather(1, supervised_action[:, None]).squeeze(1)
    hard_ce = (hard_ce * sample_weight).sum() / sample_weight.sum()
    if torch.any(non_wait):
        target_logp = log_candidate.gather(1, supervised_action[:, None]).squeeze(1)
        wait_logp = log_candidate[:, ACTIONS.index("WAIT")]
        margin = torch.relu(wait_logp[non_wait] - target_logp[non_wait]
                            + float(optimizer_config.get("nonwait_margin", 0.0))).mean()
    else:
        margin = soft_ce.new_zeros(())
    planner_probability = planner / planner.sum(dim=-1, keepdim=True)
    anchor = (planner_probability * (
        torch.log(planner_probability.clamp_min(1e-12)) - log_candidate)).sum(-1).mean()
    loss = (float(optimizer_config.get("soft_label_weight", 1.0)) * soft_ce
            + float(optimizer_config.get("hard_label_weight", 0.0)) * hard_ce
            + float(optimizer_config.get("nonwait_margin_weight", 0.0)) * margin
            + float(optimizer_config["planner_kl_anchor"]) * anchor
            + float(optimizer_config.get("residual_l2_weight", 0.0))
            * residual.square().mean())
    return loss


def train_model(config: dict, train: dict[str, np.ndarray], validation: dict[str, np.ndarray]):
    model_config = config["model"]
    model_classes = {
        BoundedResidualPolicy.architecture: BoundedResidualPolicy,
        ActionConditionedResidualPolicy.architecture: ActionConditionedResidualPolicy,
        TargetAwareResidualPolicy.architecture: TargetAwareResidualPolicy,
    }
    model_class = model_classes[model_config["kind"]]
    model = model_class(model_config["channels"], model_config["blocks"],
                        model_config["residual_logit_bound"])
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config["learning_rate"],
                                  weight_decay=optimizer_config["weight_decay"])
    augmentation_rng = np.random.default_rng(config["training_seed"] + 7001)
    validation_baseline, _ = masked_metrics(model, validation)
    gates = config.get("offline_gate", {})
    best_protected_ce = validation_baseline["cross_entropy"]
    best_protected_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()}
    best_eligible_ce = float("inf")
    best_eligible_state = None
    patience, history = 0, []
    for epoch in range(1, int(optimizer_config["epochs"]) + 1):
        model.train()
        order = np.random.default_rng(config["training_seed"] + epoch).permutation(len(train["features"]))
        total_loss = total_grad_norm = 0.0
        clipped_batches = batch_count = 0
        for start in range(0, len(order), int(optimizer_config["batch_size"])):
            indices = order[start:start + int(optimizer_config["batch_size"])]
            batch = augment_batch({key: value[indices] for key, value in train.items()},
                                  float(optimizer_config["random_d4_probability"]), augmentation_rng)
            loss = residual_training_loss(model, batch, optimizer_config)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite residual-policy loss")
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), optimizer_config["gradient_clip"]))
            total_grad_norm += gradient_norm
            clipped_batches += int(gradient_norm > optimizer_config["gradient_clip"])
            batch_count += 1
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        model.eval()
        _, candidate_metrics = masked_metrics(model, validation)
        validation_ce = candidate_metrics["cross_entropy"]
        with torch.inference_mode():
            validation_loss = float(residual_training_loss(
                model, validation, optimizer_config).detach())
        gate_result = offline_gate_metrics(validation_baseline, candidate_metrics, gates)
        relative_ce_gain = gate_result["relative_ce_gain"]
        nonwait_gain = gate_result["non_wait_top1_gain_points"]
        eligible = gate_result["passed"]
        protected = bool(
            nonwait_gain >= 0.0
            and candidate_metrics["unsafe_probability_mass"]
            <= gates.get("maximum_unsafe_probability", float("inf")))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train["features"]),
                        "gradient_norm_mean_preclip": total_grad_norm / max(batch_count, 1),
                        "gradient_clip_fraction": clipped_batches / max(batch_count, 1),
                        "validation_ce": validation_ce,
                        "validation_nonwait_top1": candidate_metrics["non_wait_top1_accuracy"],
                        "validation_relative_ce_gain": relative_ce_gain,
                        "validation_nonwait_gain_points": nonwait_gain,
                        "validation_gate_eligible": eligible,
                        "validation_nonwait_protected": protected,
                        "validation_objective": validation_loss})
        improved = False
        if eligible and validation_ce < best_eligible_ce - float(optimizer_config["min_delta"]):
            best_eligible_ce = validation_ce
            best_eligible_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()}
            improved = True
        elif (best_eligible_state is None and protected
              and validation_ce < best_protected_ce - float(optimizer_config["min_delta"])):
            best_protected_ce = validation_ce
            best_protected_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()}
            improved = True
        if improved:
            patience = 0
        else:
            patience += 1
        if patience >= int(optimizer_config["patience"]):
            break
    best_state = best_eligible_state if best_eligible_state is not None else best_protected_state
    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def artifact_paths(config: dict) -> dict[str, Path]:
    return {name: ROOT / config[key] for name, key in {
        "checkpoint": "checkpoint_path", "trajectories": "trajectory_dir",
        "manifest": "manifest_path", "metrics": "metrics_path",
        "latency": "latency_report_path",
    }.items()}


def execute(config_path: Path, config: dict) -> int:
    paths = artifact_paths(config)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite residual artifacts: {existing}")
    config_hash = sha256(config_path)
    initial_sources = source_hashes()
    manifest = {
        "run_id": config["run_id"], "status": "collecting",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash, "source_sha256": initial_sources,
        "teacher_checkpoint_sha256": config["teacher_checkpoint_sha256"],
        "planner_inference_config_sha256": config["planner_inference_config_sha256"],
        "episodes": config["episodes"],
        "split": {"train_episodes": config["train_episodes"],
                  "validation_episodes": config["validation_episodes"]},
    }
    atomic_json(paths["manifest"], manifest)
    paths["trajectories"].mkdir(parents=True, exist_ok=False)
    try:
        random.seed(config["training_seed"])
        np.random.seed(config["training_seed"])
        torch.manual_seed(config["training_seed"])
        teacher = build_teacher(config)
        shards, shard_paths, episode_summaries = [], [], []
        for episode in range(config["episodes"]):
            seed = config["training_seed"] + episode
            shard, episode_summary = collect_episode(seed, config, teacher, return_summary=True)
            shard_path = paths["trajectories"] / f"episode-{episode + 1:03d}-s{seed}.npz"
            np.savez_compressed(shard_path, **shard)
            shards.append(shard)
            shard_paths.append(shard_path)
            episode_summaries.append(episode_summary)
            manifest.update(episodes_completed=episode + 1,
                            labels=sum(len(item["features"]) for item in shards))
            atomic_json(paths["manifest"], manifest)
        total_labels = sum(len(item["features"]) for item in shards)
        if total_labels > config["maximum_labeled_states"]:
            raise ValueError("collected labels exceed the pre-registered cap; do not truncate episode split")
        train_shards = shards[:config["train_episodes"]]
        validation_shards = shards[config["train_episodes"]:]
        train, validation = concatenate(train_shards), concatenate(validation_shards)
        if not len(train["features"]) or not len(validation["features"]):
            raise ValueError("empty train or validation episode split")
        feature_channels = 8 if config["model"]["kind"] == TargetAwareResidualPolicy.architecture else 7
        validate_dataset(train, config["opponent_count"], feature_channels)
        validate_dataset(validation, config["opponent_count"], feature_channels)
        dataset_digest = hashlib.sha256()
        for shard_path in shard_paths:
            dataset_digest.update(shard_path.name.encode())
            dataset_digest.update(shard_path.read_bytes())
        dataset_hash = dataset_digest.hexdigest()
        model, history = train_model(config, train, validation)
        baseline, candidate = masked_metrics(model, validation)
        gates = config["offline_gate"]
        gate_result = offline_gate_metrics(baseline, candidate, gates)
        offline_passed = gate_result["passed"]
        metadata = {
            "run_id": config["run_id"],
            "architecture": model.architecture, "channels": config["model"]["channels"],
            "blocks": config["model"]["blocks"],
            "bound": config["model"]["residual_logit_bound"],
            "training_seed": config["training_seed"], "config_sha256": config_hash,
            "dataset_sha256": dataset_hash,
            "teacher_checkpoint_sha256": config["teacher_checkpoint_sha256"],
            "planner_inference_config_sha256": config["planner_inference_config_sha256"],
            "feature_schema": config["feature_schema"],
            "persistent_target_enabled": bool(config.get("teacher_persistent_target", False)),
            "productive_bomb_protection": bool(
                config.get("teacher_productive_bomb_protection", False)),
        }
        model.save_checkpoint(paths["checkpoint"], metadata)
        metrics = {
            "run_id": config["run_id"],
            "status": "awaiting_official_latency" if offline_passed else "offline_gate_failed",
            "config_sha256": config_hash, "dataset_sha256": dataset_hash,
            "checkpoint_sha256": sha256(paths["checkpoint"]),
            "split": {"train_states": len(train["features"]),
                      "validation_states": len(validation["features"])},
            "teacher_episodes": {
                "mean_score": float(np.mean([item["score"] for item in episode_summaries])),
                "mean_steps": float(np.mean([item["steps"] for item in episode_summaries])),
                "survival_rate": float(np.mean([item["survived"] for item in episode_summaries])),
                "task2_clear_rate": float(np.mean(
                    [item["cleared_task2_board"] for item in episode_summaries])),
                "mean_crates_remaining": float(np.mean(
                    [item["crates_remaining"] for item in episode_summaries])),
                "per_episode": episode_summaries,
            },
            "training": {"epochs_completed": len(history), "history": history},
            "baseline": baseline, "candidate": candidate,
            "gate": {**gate_result, "passed": False,
                     "offline_passed": offline_passed,
                     "official_latency_passed": None},
        }
        if source_hashes() != initial_sources:
            raise RuntimeError("source changed during residual training execution")
        atomic_json(paths["metrics"], metrics)
        manifest.update(status=metrics["status"], episodes_completed=config["episodes"],
                        labels=total_labels, dataset_sha256=dataset_hash,
                        checkpoint_sha256=metrics["checkpoint_sha256"],
                        artifacts={name: str(path) for name, path in paths.items()},
                        ended_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(paths["manifest"], manifest)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    except Exception as exception:
        manifest.update(status="failed", error=str(exception), traceback=traceback.format_exc(),
                        ended_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(paths["manifest"], manifest)
        raise


def finalize(config_path: Path, config: dict, report_path: Path) -> int:
    paths = artifact_paths(config)
    metrics = json.loads(paths["metrics"].read_text())
    report = json.loads(report_path.read_text())
    if metrics["status"] != "awaiting_official_latency" or not metrics["gate"]["offline_passed"]:
        raise ValueError("offline gate did not authorize latency finalization")
    if metrics.get("config_sha256") != sha256(config_path):
        raise ValueError("training config changed after execution")
    if sha256(paths["checkpoint"]) != metrics.get("checkpoint_sha256"):
        raise ValueError("checkpoint changed after execution")
    if report_path.resolve() != paths["latency"].resolve():
        raise ValueError("latency report path is not the pre-registered artifact")
    if report.get("checkpoint_sha256") != metrics["checkpoint_sha256"]:
        raise ValueError("latency report checkpoint mismatch")
    if report.get("config_sha256") != metrics["config_sha256"]:
        raise ValueError("latency report config mismatch")
    if report.get("dataset_sha256") != metrics["dataset_sha256"]:
        raise ValueError("latency report dataset mismatch")
    if report.get("feature_schema") != config["feature_schema"]:
        raise ValueError("latency report feature schema mismatch")
    if (report.get("kind") != "v10-residual-root-forward-latency-v1"
            or report.get("architecture") != config["model"]["kind"]
            or report.get("run_id") != config["run_id"]
            or report.get("samples", 0) < 100
            or report.get("torch_num_threads") != 1):
        raise ValueError("latency report protocol mismatch")
    if report.get("machine") != "x86_64" or report.get("torch") != "2.10.0":
        raise ValueError("latency report is not from the official amd64 PyTorch environment")
    if not np.isfinite(report.get("p99_ms", float("nan"))):
        raise ValueError("latency report p99 is not finite")
    manifest = json.loads(paths["manifest"].read_text())
    if manifest.get("source_sha256") != source_hashes():
        raise ValueError("source changed after training")
    if manifest.get("teacher_checkpoint_sha256") != sha256(resolve(config["teacher_checkpoint"])):
        raise ValueError("teacher checkpoint changed after training")
    if manifest.get("planner_inference_config_sha256") != sha256(resolve(config["planner_inference_config"])):
        raise ValueError("planner config changed after training")
    threshold = config["offline_gate"]["maximum_root_forward_p99_ms"]
    latency_passed = bool(report["p99_ms"] < threshold)
    metrics["official_latency"] = report
    metrics["gate"]["official_latency_passed"] = latency_passed
    metrics["gate"]["passed"] = bool(metrics["gate"]["offline_passed"] and latency_passed)
    metrics["status"] = "completed" if metrics["gate"]["passed"] else "latency_gate_failed"
    atomic_json(paths["metrics"], metrics)
    manifest["status"] = metrics["status"]
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(paths["manifest"], manifest)
    print(json.dumps(metrics["gate"], indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--finalize-official-latency", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.finalize_official_latency:
        return finalize(args.config, config, args.finalize_official_latency)
    if args.execute:
        return execute(args.config, config)
    plan = {"mode": "dry-run", "run_id": config["run_id"],
            "config_sha256": sha256(args.config),
            "training_seeds": [config["training_seed"],
                               config["training_seed"] + config["episodes"] - 1],
            "fixed_teacher_simulations": config["teacher_search_simulations"],
            "artifacts": {name: str(path) for name, path in artifact_paths(config).items()}}
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
