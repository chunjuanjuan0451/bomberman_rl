"""Strict immutable configuration contract for the Task-3 four-arm study."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


ARM_SPECS = {
    "A": {"reward_profile": "legacy", "action_features": False, "n_step": 1},
    "B": {"reward_profile": "official-aligned", "action_features": False, "n_step": 1},
    "C": {"reward_profile": "official-aligned", "action_features": True, "n_step": 1},
    "D": {"reward_profile": "official-aligned", "action_features": True, "n_step": 3},
}

REQUIRED_HYPERPARAMETERS = {
    "gamma", "learning_rate", "batch_size", "replay_capacity",
    "warmup_transitions", "update_every", "target_update_every",
    "epsilon_start", "epsilon_final", "epsilon_decay_steps",
    "checkpoint_every_rounds", "gradient_clip", "delta_cap",
    "d4_batch_fraction", "d4_lambda_final", "d4_lambda_warmup_updates",
}


def load_config(path: str | Path | None = None) -> tuple[dict, Path, str]:
    if path is None:
        path = os.environ.get("MODEL_A_TASK3_CONFIG_PATH")
    if not path:
        raise RuntimeError("MODEL_A_TASK3_CONFIG_PATH is required")
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1 or config.get("agent") != "model_a_task3":
        raise ValueError("Task-3 config requires schema_version=1 and agent='model_a_task3'")
    arm = config.get("arm")
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown Task-3 arm: {arm!r}")
    expected = ARM_SPECS[arm]
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Task-3 arm {arm} requires {key}={value!r}")
    if config.get("task") != 3 or config.get("includes_task4", True):
        raise ValueError("Task-3 config may not include Task 4")
    if config.get("agents") != ["model_a_task3", "task3_curriculum_agent"]:
        raise ValueError("Task-3 training must use exactly one curriculum opponent")
    if config.get("scenario") != "classic" or int(config.get("rounds", 0)) != 300:
        raise ValueError("Task-3 pilot requires 300 classic rounds")
    if config.get("resume") or not config.get("enabled_for_training"):
        raise ValueError("Task-3 pilot requires a fresh explicitly enabled run")
    hyper = config.get("hyperparameters", {})
    if set(hyper) != REQUIRED_HYPERPARAMETERS:
        raise ValueError("Task-3 hyperparameter keys do not match the schema")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in hyper.values()):
        raise ValueError("Task-3 hyperparameters must be finite numbers")
    if not 0 <= hyper["epsilon_final"] <= hyper["epsilon_start"] <= 1:
        raise ValueError("invalid epsilon schedule")
    if hyper["delta_cap"] <= 0 or not 0 <= hyper["d4_batch_fraction"] <= 1:
        raise ValueError("invalid residual or D4 settings")
    if hyper["warmup_transitions"] < hyper["batch_size"]:
        raise ValueError("warmup must be at least one batch")
    parent = Path(config["parent_checkpoint"])
    if not parent.is_absolute():
        parent = resolved.parents[2] / parent
    if not parent.is_file() or len(str(config.get("parent_sha256", ""))) != 64:
        raise ValueError("a real frozen-v4 parent checkpoint and SHA-256 are required")
    return config, resolved, hashlib.sha256(raw).hexdigest()


def architecture_name(config: dict) -> str:
    residual = "action6" if config["action_features"] else "state6"
    return f"model-a-task3-frozen-v4-{residual}-n{config['n_step']}"
