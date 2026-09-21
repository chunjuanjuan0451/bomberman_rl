"""Pre-registered v10.7 student-only three-arm ablation; dry-run by default."""

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

from agent_code.model_a_v10.interfaces import ACTIONS
from agent_code.model_a_v10.residual_policy import (
    D4_TRANSFORMS,
    PlannerAwareGatedResidualPolicy,
    PlannerAwareResidualPolicy,
    TargetAwareResidualPolicy,
    planner_residual_policy,
    residual_forward,
    transform_action_vector,
    transform_features,
)


REQUIRED_KEYS = {
    "features", "visit_policy", "planner_prior", "safe_mask",
    "planner_action_scores", "search_action", "opponent_actions",
    "return_target", "future_death", "tactical", "step",
}
MODEL_CLASSES = {
    TargetAwareResidualPolicy.architecture: TargetAwareResidualPolicy,
    PlannerAwareResidualPolicy.architecture: PlannerAwareResidualPolicy,
    PlannerAwareGatedResidualPolicy.architecture: PlannerAwareGatedResidualPolicy,
}
SOURCE_PATHS = (
    "agent_code/model_a_v10/residual_policy.py",
    "tools/v10_student_ablation.py",
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
    exact = {
        "protocol_version": 1,
        "run_id": "model-a-v10.7-student-three-arm-s106000",
        "source_run_id": "model-a-v10.6-task2-target-aware-correction-s105200",
        "source_dataset_sha256":
            "8fe08dc0696bd8f09e49af1b62844ed00159a22e2f5147c38113dabe8b32a632",
        "source_config":
            "experiments/configs/v10.6-task2-target-aware-correction-s105200.json",
        "source_manifest":
            "experiments/logs/runs/model-a-v10.6-task2-target-aware-correction-s105200.json",
        "source_trajectory_dir": (
            "experiments/logs/v10.6/"
            "model-a-v10.6-task2-target-aware-correction-s105200-trajectories"),
        "development_episode_seed_range": [105200, 105237],
        "inner_train_episode_seed_range": [105200, 105229],
        "inner_dev_episode_seed_range": [105230, 105237],
        "excluded_outer_validation_seed_range": [105238, 105247],
        "model_seeds": [106000, 106001, 106002],
        "environment_rollout": False,
        "student_only_training": True,
        "automatic_full_training": False,
        "selection_rule":
            "all-three-seeds-pass-then-highest-mean-relative-ce-gain",
        "checkpoint_dir": "experiments/checkpoints/v10.7-student-three-arm-s106000",
        "manifest_path":
            "experiments/logs/runs/model-a-v10.7-student-three-arm-s106000.json",
        "report_path":
            "experiments/logs/diagnostics/v10.7-student-three-arm-s106000.json",
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise ValueError(f"v10.7 student ablation requires {key}={expected!r}")
    expected_arms = [
        {"id": "A_loss_repair", "architecture": TargetAwareResidualPolicy.architecture},
        {"id": "B_planner_aware", "architecture": PlannerAwareResidualPolicy.architecture},
        {"id": "C_selective_gate", "architecture": PlannerAwareGatedResidualPolicy.architecture},
    ]
    if config.get("arms") != expected_arms:
        raise ValueError("v10.7 three-arm binding mismatch")
    if config.get("model") != {
            "channels": 32, "blocks": 3, "residual_logit_bound": 0.5,
            "feature_schema": "padded-centered-33x33-target-distance-v3"}:
        raise ValueError("v10.7 model binding mismatch")
    expected_optimizer = {
        "name": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.0001,
        "batch_size": 128, "epochs": 40, "patience": 8, "min_delta": 0.0001,
        "gradient_clip": 1.0, "random_d4_probability": 0.5,
        "action_balance_power": 0.5, "search_confidence_margin": 0.15,
        "soft_label_weight": 1.0, "hard_correction_weight": 0.25,
        "planner_agreement_kl_weight": 0.75,
        "agreement_rank_weight": 1.0, "wait_agreement_rank_weight": 2.0,
        "correction_rank_weight": 0.25, "rank_margin": 0.02,
        "residual_l2_weight": 0.02, "preservation_residual_weight": 0.1,
        "gate_bce_weight": 0.5, "gate_preservation_weight": 0.25,
    }
    if config.get("optimizer") != expected_optimizer:
        raise ValueError("v10.7 optimizer/loss binding mismatch")
    expected_gate = {
        "minimum_relative_ce_gain": 0.05,
        "minimum_nonwait_top1_gain_points": 0.03,
        "maximum_wait_top1_drop_points": 0.02,
        "maximum_bomb_top1_drop_points": 0.0,
        "minimum_high_confidence_relative_ce_gain": 0.05,
        "maximum_planner_agreement_top1_drop_points": 0.01,
        "maximum_unsafe_probability": 0.0,
        "minimum_residual_state_std_mean": 0.001,
        "maximum_residual_saturation_fraction": 0.05,
        "all_three_model_seeds_must_pass": True,
    }
    if config.get("inner_dev_gate") != expected_gate:
        raise ValueError("v10.7 inner-dev gate binding mismatch")
    source_config = ROOT / config["source_config"]
    source_manifest = ROOT / config["source_manifest"]
    if sha256(source_config) != config.get("source_config_sha256"):
        raise ValueError("v10.7 source config hash mismatch")
    if sha256(source_manifest) != config.get("source_manifest_sha256"):
        raise ValueError("v10.7 source manifest hash mismatch")
    manifest = json.loads(source_manifest.read_text())
    if (manifest.get("run_id") != config["source_run_id"]
            or manifest.get("dataset_sha256") != config["source_dataset_sha256"]
            or manifest.get("episodes_completed") != 48):
        raise ValueError("v10.7 source manifest binding mismatch")
    if config.get("source_sha256") != source_hashes():
        raise ValueError("v10.7 source-code binding mismatch")
    return config


def trajectory_path(config: dict, seed: int) -> Path:
    first_seed = 105200
    episode = seed - first_seed + 1
    return ROOT / config["source_trajectory_dir"] / f"episode-{episode:03d}-s{seed}.npz"


def verify_source_dataset(config: dict) -> None:
    directory = ROOT / config["source_trajectory_dir"]
    paths = sorted(directory.glob("episode-*.npz"))
    if len(paths) != 48:
        raise ValueError("v10.7 source trajectory count mismatch")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != config["source_dataset_sha256"]:
        raise ValueError("v10.7 source trajectory digest mismatch")


def load_seed_range(config: dict, seed_range: list[int]) -> dict[str, np.ndarray]:
    seeds = range(int(seed_range[0]), int(seed_range[1]) + 1)
    shards = []
    for seed in seeds:
        path = trajectory_path(config, seed)
        with np.load(path, allow_pickle=False) as payload:
            shard = {key: payload[key] for key in payload.files}
        if set(shard) != REQUIRED_KEYS:
            raise ValueError(f"trajectory schema mismatch: {path.name}")
        shards.append(shard)
    data = {key: np.concatenate([shard[key] for shard in shards])
            for key in sorted(REQUIRED_KEYS)}
    count = len(data["features"])
    if data["features"].shape != (count, 8, 33, 33):
        raise ValueError("v10.7 requires target-aware source features")
    if data["opponent_actions"].shape != (count, 0):
        raise ValueError("v10.7 requires Task 2 single-agent source data")
    for key in ("visit_policy", "planner_prior", "safe_mask"):
        if data[key].shape != (count, len(ACTIONS)):
            raise ValueError(f"v10.7 invalid {key} shape")
    if not np.allclose(data["visit_policy"].sum(1), 1.0, atol=1e-6):
        raise ValueError("v10.7 source targets are not normalized")
    if float((data["visit_policy"] * (1.0 - data["safe_mask"])).sum()) != 0.0:
        raise ValueError("v10.7 source target violates the safety mask")
    return data


def augment_batch(batch: dict[str, np.ndarray], probability: float,
                  rng: np.random.Generator) -> dict[str, np.ndarray]:
    result = {key: value.copy() for key, value in batch.items()}
    for index in range(len(result["features"])):
        if rng.random() >= probability:
            continue
        transform = str(rng.choice(D4_TRANSFORMS[1:]))
        result["features"][index] = transform_features(result["features"][index], transform)
        for key in ("visit_policy", "planner_prior", "safe_mask"):
            result[key][index] = transform_action_vector(result[key][index], transform)
    return result


def model_output(model, features: torch.Tensor, planner: torch.Tensor):
    if isinstance(model, PlannerAwareGatedResidualPolicy):
        residual, gate_logit, raw_residual = model.forward_with_aux(features, planner)
        return residual, gate_logit, raw_residual
    residual = residual_forward(model, features, planner)
    return residual, None, residual


def correction_masks(target: torch.Tensor, planner: torch.Tensor,
                     confidence_margin: float):
    target_action = target.argmax(-1)
    planner_action = planner.argmax(-1)
    top_two = torch.topk(target, k=2, dim=-1).values
    confident = (top_two[:, 0] - top_two[:, 1]) >= confidence_margin
    correction = confident & (target_action != planner_action)
    agreement = target_action == planner_action
    return target_action, planner_action, confident, correction, agreement


def ranking_loss(logits: torch.Tensor, action: torch.Tensor,
                 mask: torch.Tensor, margin: float) -> torch.Tensor:
    if not torch.any(mask):
        return logits.new_zeros(())
    selected = logits.gather(1, action[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, action[:, None], float("-inf"))
    strongest = competitors.max(dim=-1).values
    return torch.relu(strongest[mask] - selected[mask] + margin).mean()


def training_loss(model, batch: dict[str, np.ndarray], optimizer_config: dict):
    features = torch.from_numpy(batch["features"].astype(np.float32))
    target = torch.from_numpy(batch["visit_policy"].astype(np.float32))
    planner = torch.from_numpy(batch["planner_prior"].astype(np.float32))
    safe = torch.from_numpy(batch["safe_mask"].astype(np.float32))
    residual, gate_logit, raw_residual = model_output(model, features, planner)
    candidate = planner_residual_policy(planner, safe, residual)
    log_candidate = torch.log(candidate.clamp_min(1e-12))
    target_action, planner_action, _, correction, agreement = correction_masks(
        target, planner, float(optimizer_config["search_confidence_margin"]))
    preservation = ~correction
    soft_ce = -(target * log_candidate).sum(-1).mean()
    hard_correction = (-log_candidate[correction].gather(
        1, target_action[correction][:, None]).mean()
        if torch.any(correction) else soft_ce.new_zeros(()))
    planner_probability = planner / planner.sum(dim=-1, keepdim=True)
    planner_kl = (planner_probability * (
        torch.log(planner_probability.clamp_min(1e-12)) - log_candidate)).sum(-1)
    agreement_kl = (planner_kl[agreement].mean()
                    if torch.any(agreement) else soft_ce.new_zeros(()))
    combined_logits = (torch.log(planner.clamp_min(1e-12)) + residual).masked_fill(
        safe <= 0, float("-inf"))
    agreement_rank = ranking_loss(
        combined_logits, planner_action, agreement,
        float(optimizer_config["rank_margin"]))
    wait_agreement = agreement & (target_action == ACTIONS.index("WAIT"))
    wait_rank = ranking_loss(
        combined_logits, planner_action, wait_agreement,
        float(optimizer_config["rank_margin"]))
    correction_rank = ranking_loss(
        combined_logits, target_action, correction,
        float(optimizer_config["rank_margin"]))
    preservation_residual = (raw_residual[preservation].square().mean()
                             if torch.any(preservation) else soft_ce.new_zeros(()))
    loss = (
        float(optimizer_config["soft_label_weight"]) * soft_ce
        + float(optimizer_config["hard_correction_weight"]) * hard_correction
        + float(optimizer_config["planner_agreement_kl_weight"]) * agreement_kl
        + float(optimizer_config["agreement_rank_weight"]) * agreement_rank
        + float(optimizer_config["wait_agreement_rank_weight"]) * wait_rank
        + float(optimizer_config["correction_rank_weight"]) * correction_rank
        + float(optimizer_config["residual_l2_weight"]) * raw_residual.square().mean()
        + float(optimizer_config["preservation_residual_weight"])
        * preservation_residual
    )
    if gate_logit is not None:
        labels = correction.float()
        positive = labels.sum().clamp_min(1.0)
        negative = (1.0 - labels).sum().clamp_min(1.0)
        positive_weight = (negative / positive).detach()
        gate_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            gate_logit, labels, pos_weight=positive_weight)
        gate_preservation = torch.sigmoid(gate_logit[preservation]).mean()
        loss = (loss + float(optimizer_config["gate_bce_weight"]) * gate_bce
                + float(optimizer_config["gate_preservation_weight"])
                * gate_preservation)
    return loss


def metrics(model, data: dict[str, np.ndarray], confidence_margin: float):
    features = torch.from_numpy(data["features"].astype(np.float32))
    target = torch.from_numpy(data["visit_policy"].astype(np.float32))
    planner = torch.from_numpy(data["planner_prior"].astype(np.float32))
    safe = torch.from_numpy(data["safe_mask"].astype(np.float32))
    with torch.inference_mode():
        residual, gate_logit, raw_residual = model_output(model, features, planner)
        candidate = planner_residual_policy(planner, safe, residual)
        baseline = planner / planner.sum(dim=-1, keepdim=True)
    target_action, planner_action, confident, correction, agreement = correction_masks(
        target, planner, confidence_margin)
    wait = target_action == ACTIONS.index("WAIT")
    bomb = target_action == ACTIONS.index("BOMB")
    nonwait = ~wait

    def cross_entropy(probability, mask):
        if not torch.any(mask):
            return None
        return float(-(target[mask] * torch.log(
            probability[mask].clamp_min(1e-12))).sum(-1).mean())

    def accuracy(probability, mask):
        if not torch.any(mask):
            return None
        return float((probability.argmax(-1)[mask] == target_action[mask]).float().mean())

    all_states = torch.ones(len(target), dtype=torch.bool)
    result = {
        "states": len(target),
        "baseline": {
            "cross_entropy": cross_entropy(baseline, all_states),
            "nonwait_top1_accuracy": accuracy(baseline, nonwait),
            "wait_top1_accuracy": accuracy(baseline, wait),
            "bomb_top1_accuracy": accuracy(baseline, bomb),
            "high_confidence_cross_entropy": cross_entropy(baseline, confident),
            "planner_agreement_top1_accuracy": accuracy(baseline, agreement),
        },
        "candidate": {
            "cross_entropy": cross_entropy(candidate, all_states),
            "nonwait_top1_accuracy": accuracy(candidate, nonwait),
            "wait_top1_accuracy": accuracy(candidate, wait),
            "bomb_top1_accuracy": accuracy(candidate, bomb),
            "high_confidence_cross_entropy": cross_entropy(candidate, confident),
            "planner_agreement_top1_accuracy": accuracy(candidate, agreement),
            "unsafe_probability_mass": float((candidate * (1.0 - safe)).sum()),
            "residual_state_std_mean": float(residual.std(dim=0, unbiased=False).mean()),
            "residual_saturation_fraction": float((raw_residual.abs() >= 0.45).float().mean()),
            "residual_abs_max": float(raw_residual.abs().max()),
            "action_flip_fraction": float(
                (candidate.argmax(-1) != baseline.argmax(-1)).float().mean()),
            "harmful_planner_agreement_flip_fraction": float(
                (candidate.argmax(-1)[agreement] != planner_action[agreement]).float().mean()),
        },
        "slices": {
            "wait_states": int(wait.sum()), "bomb_states": int(bomb.sum()),
            "high_confidence_states": int(confident.sum()),
            "planner_agreement_states": int(agreement.sum()),
            "confident_correction_states": int(correction.sum()),
        },
    }
    if gate_logit is not None:
        gate = torch.sigmoid(gate_logit)
        result["candidate"].update(
            gate_mean=float(gate.mean()),
            gate_correction_mean=(float(gate[correction].mean())
                                  if torch.any(correction) else None),
            gate_preservation_mean=(float(gate[~correction].mean())
                                    if torch.any(~correction) else None),
        )
    return result


def gate_result(measurement: dict, gates: dict) -> dict:
    baseline, candidate = measurement["baseline"], measurement["candidate"]
    relative_ce_gain = ((baseline["cross_entropy"] - candidate["cross_entropy"])
                        / max(baseline["cross_entropy"], 1e-12))
    nonwait_gain = (candidate["nonwait_top1_accuracy"]
                    - baseline["nonwait_top1_accuracy"])
    wait_drop = baseline["wait_top1_accuracy"] - candidate["wait_top1_accuracy"]
    bomb_drop = baseline["bomb_top1_accuracy"] - candidate["bomb_top1_accuracy"]
    high_confidence_gain = (
        (baseline["high_confidence_cross_entropy"]
         - candidate["high_confidence_cross_entropy"])
        / max(baseline["high_confidence_cross_entropy"], 1e-12))
    agreement_drop = (baseline["planner_agreement_top1_accuracy"]
                      - candidate["planner_agreement_top1_accuracy"])
    passed = bool(
        relative_ce_gain >= gates["minimum_relative_ce_gain"]
        and nonwait_gain >= gates["minimum_nonwait_top1_gain_points"]
        and wait_drop <= gates["maximum_wait_top1_drop_points"]
        and bomb_drop <= gates["maximum_bomb_top1_drop_points"]
        and high_confidence_gain >= gates["minimum_high_confidence_relative_ce_gain"]
        and agreement_drop <= gates["maximum_planner_agreement_top1_drop_points"]
        and candidate["unsafe_probability_mass"]
        <= gates["maximum_unsafe_probability"]
        and candidate["residual_state_std_mean"]
        >= gates["minimum_residual_state_std_mean"]
        and candidate["residual_saturation_fraction"]
        <= gates["maximum_residual_saturation_fraction"]
    )
    return {
        "passed": passed, "relative_ce_gain": relative_ce_gain,
        "nonwait_top1_gain_points": nonwait_gain,
        "wait_top1_drop_points": wait_drop, "bomb_top1_drop_points": bomb_drop,
        "high_confidence_relative_ce_gain": high_confidence_gain,
        "planner_agreement_top1_drop_points": agreement_drop,
        "unsafe_probability_mass": candidate["unsafe_probability_mass"],
        "residual_state_std_mean": candidate["residual_state_std_mean"],
        "residual_saturation_fraction": candidate["residual_saturation_fraction"],
    }


def balanced_order(data: dict[str, np.ndarray], seed: int, power: float) -> np.ndarray:
    actions = data["visit_policy"].argmax(-1)
    counts = np.bincount(actions, minlength=len(ACTIONS)).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError("v10.7 inner train is missing a target action class")
    class_weight = np.power(counts.max() / counts, power)
    probability = class_weight[actions]
    probability /= probability.sum()
    return np.random.default_rng(seed).choice(
        len(actions), size=len(actions), replace=True, p=probability)


def train_one(config: dict, arm: dict, seed: int,
              train: dict[str, np.ndarray], dev: dict[str, np.ndarray]):
    model_config, optimizer_config = config["model"], config["optimizer"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MODEL_CLASSES[arm["architecture"]](
        model_config["channels"], model_config["blocks"],
        model_config["residual_logit_bound"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"])
    augmentation_rng = np.random.default_rng(seed + 7001)
    baseline_measurement = metrics(
        model, dev, float(optimizer_config["search_confidence_margin"]))
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_measurement = baseline_measurement
    best_gate = gate_result(best_measurement, config["inner_dev_gate"])
    best_epoch = 0
    best_eligible_ce = float("inf")
    best_protected_ce = baseline_measurement["baseline"]["cross_entropy"]
    raw_best_ce = best_protected_ce
    patience = 0
    history = []
    for epoch in range(1, int(optimizer_config["epochs"]) + 1):
        model.train()
        order = balanced_order(
            train, seed + epoch, float(optimizer_config["action_balance_power"]))
        total_loss = 0.0
        for start in range(0, len(order), int(optimizer_config["batch_size"])):
            indices = order[start:start + int(optimizer_config["batch_size"])]
            batch = augment_batch(
                {key: value[indices] for key, value in train.items()},
                float(optimizer_config["random_d4_probability"]), augmentation_rng)
            loss = training_loss(model, batch, optimizer_config)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite v10.7 ablation loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(optimizer_config["gradient_clip"]))
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        model.eval()
        measurement = metrics(
            model, dev, float(optimizer_config["search_confidence_margin"]))
        gate = gate_result(measurement, config["inner_dev_gate"])
        validation_ce = measurement["candidate"]["cross_entropy"]
        protected = bool(
            gate["wait_top1_drop_points"]
            <= config["inner_dev_gate"]["maximum_wait_top1_drop_points"]
            and gate["bomb_top1_drop_points"]
            <= config["inner_dev_gate"]["maximum_bomb_top1_drop_points"]
            and gate["planner_agreement_top1_drop_points"]
            <= config["inner_dev_gate"]["maximum_planner_agreement_top1_drop_points"]
            and gate["unsafe_probability_mass"]
            <= config["inner_dev_gate"]["maximum_unsafe_probability"])
        selected = False
        if gate["passed"] and validation_ce < best_eligible_ce:
            best_eligible_ce, selected = validation_ce, True
        elif best_eligible_ce == float("inf") and protected and validation_ce < best_protected_ce:
            best_protected_ce, selected = validation_ce, True
        if selected:
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()}
            best_measurement, best_gate, best_epoch = measurement, gate, epoch
        if validation_ce < raw_best_ce - float(optimizer_config["min_delta"]):
            raw_best_ce, patience = validation_ce, 0
        else:
            patience += 1
        history.append({
            "epoch": epoch, "train_loss": total_loss / len(order),
            "validation_ce": validation_ce, **gate,
            "protected": protected, "selected_checkpoint": selected,
        })
        if patience >= int(optimizer_config["patience"]):
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "arm": arm["id"], "architecture": arm["architecture"], "model_seed": seed,
        "selected_epoch": best_epoch, "epochs_completed": len(history),
        "metrics": best_measurement, "gate": best_gate, "history": history,
    }


def aggregate_arm(runs: list[dict]) -> dict:
    gate_keys = (
        "relative_ce_gain", "nonwait_top1_gain_points", "wait_top1_drop_points",
        "bomb_top1_drop_points", "high_confidence_relative_ce_gain",
        "planner_agreement_top1_drop_points", "residual_saturation_fraction",
    )
    return {
        "arm": runs[0]["arm"], "architecture": runs[0]["architecture"],
        "all_model_seeds_passed": all(run["gate"]["passed"] for run in runs),
        "passed_model_seeds": sum(run["gate"]["passed"] for run in runs),
        "mean": {key: float(np.mean([run["gate"][key] for run in runs]))
                 for key in gate_keys},
        "worst": {
            "relative_ce_gain": min(run["gate"]["relative_ce_gain"] for run in runs),
            "nonwait_top1_gain_points": min(
                run["gate"]["nonwait_top1_gain_points"] for run in runs),
            "wait_top1_drop_points": max(
                run["gate"]["wait_top1_drop_points"] for run in runs),
            "bomb_top1_drop_points": max(
                run["gate"]["bomb_top1_drop_points"] for run in runs),
            "planner_agreement_top1_drop_points": max(
                run["gate"]["planner_agreement_top1_drop_points"] for run in runs),
        },
    }


def artifact_paths(config: dict) -> dict[str, Path]:
    return {
        "checkpoint_dir": ROOT / config["checkpoint_dir"],
        "manifest": ROOT / config["manifest_path"],
        "report": ROOT / config["report_path"],
    }


def execute(config_path: Path, config: dict) -> int:
    paths = artifact_paths(config)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite v10.7 ablation artifacts: {existing}")
    config_hash = sha256(config_path)
    initial_sources = source_hashes()
    manifest = {
        "run_id": config["run_id"], "status": "validating_source",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash, "source_sha256": initial_sources,
        "source_dataset_sha256": config["source_dataset_sha256"],
        "automatic_full_training": False,
    }
    atomic_json(paths["manifest"], manifest)
    try:
        torch.set_num_threads(1)
        verify_source_dataset(config)
        train = load_seed_range(config, config["inner_train_episode_seed_range"])
        dev = load_seed_range(config, config["inner_dev_episode_seed_range"])
        paths["checkpoint_dir"].mkdir(parents=True, exist_ok=False)
        manifest.update(status="training_student_arms",
                        inner_train_states=len(train["features"]),
                        inner_dev_states=len(dev["features"]))
        atomic_json(paths["manifest"], manifest)
        runs = []
        for arm in config["arms"]:
            for seed in config["model_seeds"]:
                model, result = train_one(config, arm, int(seed), train, dev)
                checkpoint = paths["checkpoint_dir"] / f"{arm['id']}-s{seed}.pt"
                metadata = {
                    "run_id": config["run_id"], "arm": arm["id"],
                    "architecture": arm["architecture"], "channels": config["model"]["channels"],
                    "blocks": config["model"]["blocks"],
                    "bound": config["model"]["residual_logit_bound"], "model_seed": seed,
                    "config_sha256": config_hash,
                    "source_dataset_sha256": config["source_dataset_sha256"],
                    "inner_train_episode_seed_range": config["inner_train_episode_seed_range"],
                    "inner_dev_episode_seed_range": config["inner_dev_episode_seed_range"],
                    "outer_validation_used": False, "diagnostic_only": True,
                    "persistent_target_enabled": True,
                    "productive_bomb_protection": True,
                }
                model.save_checkpoint(checkpoint, metadata)
                result["checkpoint_path"] = str(checkpoint)
                result["checkpoint_sha256"] = sha256(checkpoint)
                runs.append(result)
                manifest.update(runs_completed=len(runs))
                atomic_json(paths["manifest"], manifest)
        aggregates = [aggregate_arm([
            run for run in runs if run["arm"] == arm["id"]]) for arm in config["arms"]]
        eligible = [item for item in aggregates if item["all_model_seeds_passed"]]
        selected = (max(eligible, key=lambda item: item["mean"]["relative_ce_gain"])["arm"]
                    if eligible else None)
        report = {
            "run_id": config["run_id"],
            "status": "arm_selected" if selected else "no_arm_selected",
            "config_sha256": config_hash,
            "source_dataset_sha256": config["source_dataset_sha256"],
            "data_isolation": {
                "inner_train_episode_seed_range": config["inner_train_episode_seed_range"],
                "inner_dev_episode_seed_range": config["inner_dev_episode_seed_range"],
                "excluded_outer_validation_seed_range":
                    config["excluded_outer_validation_seed_range"],
                "outer_validation_used": False,
            },
            "runs": runs, "arm_aggregates": aggregates, "selected_arm": selected,
            "automatic_full_training_started": False,
        }
        if source_hashes() != initial_sources:
            raise RuntimeError("source changed during v10.7 student ablation")
        atomic_json(paths["report"], report)
        manifest.update(
            status=report["status"], runs_completed=len(runs), selected_arm=selected,
            report_sha256=sha256(paths["report"]),
            ended_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(paths["manifest"], manifest)
        print(json.dumps({"status": report["status"], "selected_arm": selected,
                          "arm_aggregates": aggregates}, indent=2, sort_keys=True))
        return 0
    except Exception as exception:
        manifest.update(status="failed", error=str(exception), traceback=traceback.format_exc(),
                        ended_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(paths["manifest"], manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.execute:
        return execute(args.config, config)
    verify_source_dataset(config)
    plan = {
        "mode": "dry-run", "run_id": config["run_id"],
        "config_sha256": sha256(args.config), "arms": config["arms"],
        "model_seeds": config["model_seeds"],
        "inner_train_episode_seed_range": config["inner_train_episode_seed_range"],
        "inner_dev_episode_seed_range": config["inner_dev_episode_seed_range"],
        "excluded_outer_validation_seed_range": config["excluded_outer_validation_seed_range"],
        "automatic_full_training": False,
        "artifacts": {name: str(path) for name, path in artifact_paths(config).items()},
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
