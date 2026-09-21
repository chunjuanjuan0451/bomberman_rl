"""Validated, immutable experiment configuration for Model A v6."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


VARIANT_SPECS = {
    "v6a-control": {"augmentation": "none", "tactical_residual": False},
    "v6b-random-d4": {"augmentation": "random_d4", "tactical_residual": False},
    "v6c-tactical-residual": {"augmentation": "none", "tactical_residual": True},
    "v6d-combined": {"augmentation": "random_d4", "tactical_residual": True},
}
REQUIRED_HYPERPARAMETERS = {
    "gamma", "learning_rate", "batch_size", "replay_capacity",
    "warmup_transitions", "update_every", "target_update_every",
    "epsilon_start", "epsilon_final", "epsilon_decay_steps",
    "checkpoint_every_rounds",
}
REQUIRED_REWARDS = {
    "COIN_COLLECTED", "CRATE_DESTROYED", "KILLED_OPPONENT", "KILLED_SELF",
    "GOT_KILLED", "INVALID_ACTION", "BOMB_DROPPED", "SURVIVED_ROUND",
}


def load_v6_config(path: str | Path | None = None) -> tuple[dict, Path, str]:
    """Load a v6 JSON config and return it with its path and SHA-256."""
    if path is None:
        path = os.environ.get("MODEL_A_V6_CONFIG_PATH")
    if not path:
        raise RuntimeError("MODEL_A_V6_CONFIG_PATH is required for model_a_v6")
    config_path = Path(path).resolve()
    raw = config_path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("v6 config must contain one JSON object")
    if config.get("schema_version") != 1 or config.get("agent") != "model_a_v6":
        raise ValueError("v6 config requires schema_version=1 and agent='model_a_v6'")
    variant = config.get("variant")
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown v6 variant: {variant!r}")
    expected = VARIANT_SPECS[variant]
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"{variant} requires {key}={value!r}, got {config.get(key)!r}"
            )
    hyperparameters = config.get("hyperparameters", {})
    if set(hyperparameters) != REQUIRED_HYPERPARAMETERS:
        raise ValueError("v6 hyperparameter keys do not match the frozen schema")
    rewards = config.get("rewards", {})
    if set(rewards) != REQUIRED_REWARDS:
        raise ValueError("v6 reward keys do not match the frozen schema")
    shaping = config.get("shaping", {})
    if set(shaping) != {"danger_exit_reward", "danger_entry_penalty"}:
        raise ValueError("v6 shaping keys do not match the frozen schema")
    numeric_values = [*hyperparameters.values(), *rewards.values(), *shaping.values()]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric_values):
        raise ValueError("v6 numeric configuration values must be finite")
    if not 0.0 <= hyperparameters["epsilon_final"] <= hyperparameters["epsilon_start"] <= 1.0:
        raise ValueError("v6 epsilon schedule is invalid")
    if hyperparameters["warmup_transitions"] < hyperparameters["batch_size"]:
        raise ValueError("v6 warmup must be at least one batch")
    return config, config_path, hashlib.sha256(raw).hexdigest()


def architecture_name(config: dict) -> str:
    suffix = "tactical4" if config["tactical_residual"] else "base"
    return f"model-a-v6-mlp-global7-{suffix}"
