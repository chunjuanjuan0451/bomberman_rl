"""Immutable contract for the Task-3 training-distribution experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


REQUIRED_HYPERPARAMETERS = {
    "gamma", "learning_rate", "batch_size", "replay_capacity",
    "warmup_transitions", "update_every", "target_update_every",
    "epsilon_start", "epsilon_final", "epsilon_decay_steps",
    "checkpoint_every_rounds", "gradient_clip", "delta_cap",
    "d4_batch_fraction", "d4_lambda_final", "d4_lambda_warmup_updates",
}
DISTRIBUTIONS = {
    "classic-control": "classic",
    "kill-rich": "classic",
}


def load_config(path: str | Path | None = None) -> tuple[dict, Path, str]:
    if path is None:
        path = os.environ.get("MODEL_A_TASK3_KILLRICH_CONFIG_PATH")
    if not path:
        raise RuntimeError("MODEL_A_TASK3_KILLRICH_CONFIG_PATH is required")
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1 or config.get("agent") != "model_a_task3_killrich":
        raise ValueError("kill-rich config requires schema_version=1 and the isolated agent")
    if config.get("task") != 3 or config.get("includes_task4", True):
        raise ValueError("kill-rich experiment must remain isolated to Task 3")
    if config.get("agents") != ["model_a_task3_killrich", "task3_curriculum_agent"]:
        raise ValueError("training requires exactly one unchanged curriculum opponent")
    distribution = config.get("training_distribution")
    if distribution not in DISTRIBUTIONS or config.get("scenario") != DISTRIBUTIONS[distribution]:
        raise ValueError("training distribution and scenario do not match")
    if int(config.get("rounds", 0)) != 300:
        raise ValueError("the matched experiment requires exactly 300 rounds")
    if config.get("resume") or not config.get("enabled_for_training"):
        raise ValueError("a fresh explicitly enabled run is required")
    fixed = {
        "arm": "D", "reward_profile": "official-aligned",
        "action_features": True, "n_step": 3, "sampling_profile": "uniform",
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f"training-distribution experiment freezes {key}={value!r}")
    hyper = config.get("hyperparameters", {})
    if set(hyper) != REQUIRED_HYPERPARAMETERS:
        raise ValueError("hyperparameters do not match the frozen D schema")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in hyper.values()):
        raise ValueError("hyperparameters must be finite numbers")
    if int(hyper["batch_size"]) != 64 or int(hyper["replay_capacity"]) != 50000:
        raise ValueError("uniform replay is frozen at batch 64 and capacity 50000")
    if not 0 <= hyper["epsilon_final"] <= hyper["epsilon_start"] <= 1:
        raise ValueError("invalid epsilon schedule")
    if hyper["delta_cap"] != 0.25 or not 0 <= hyper["d4_batch_fraction"] <= 1:
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
    return "model-a-task3-killrich-frozen-v4-action6-n3"
