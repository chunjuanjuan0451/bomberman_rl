"""Strict configuration loader for stable v6 experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

VARIANT_SPECS = {
    "v6s1-residual-control": False,
    "v6s2-d4-consistency": True,
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
        path = os.environ.get("MODEL_A_V6_STABLE_CONFIG_PATH")
    if not path:
        raise RuntimeError("MODEL_A_V6_STABLE_CONFIG_PATH is required")
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1 or config.get("agent") != "model_a_v6_stable":
        raise ValueError("stable v6 config requires schema_version=1 and agent='model_a_v6_stable'")
    variant = config.get("variant")
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown stable v6 variant: {variant!r}")
    if bool(config.get("d4_consistency")) != VARIANT_SPECS[variant]:
        raise ValueError(f"{variant} has an invalid d4_consistency setting")
    hyper = config.get("hyperparameters", {})
    if set(hyper) != REQUIRED_HYPERPARAMETERS:
        raise ValueError("stable v6 hyperparameter keys do not match the schema")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in hyper.values()):
        raise ValueError("stable v6 hyperparameters must be finite numbers")
    if not 0 <= hyper["epsilon_final"] <= hyper["epsilon_start"] <= 1:
        raise ValueError("invalid epsilon schedule")
    if not 0 <= hyper["d4_batch_fraction"] <= 1 or hyper["delta_cap"] <= 0:
        raise ValueError("invalid D4 fraction or residual cap")
    if hyper["warmup_transitions"] < hyper["batch_size"]:
        raise ValueError("warmup must be at least one batch")
    parent = Path(config["parent_checkpoint"])
    if not parent.is_absolute():
        parent = resolved.parents[2] / parent
    if not parent.is_file() or len(str(config.get("parent_sha256", ""))) != 64:
        raise ValueError("a real parent checkpoint and SHA-256 are required")
    return config, resolved, hashlib.sha256(raw).hexdigest()


def architecture_name() -> str:
    return "model-a-v6-stable-frozen-v4-residual6"
