"""Strict configuration for rollout-distilled residual training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

REQUIRED = {
    "gamma", "learning_rate", "batch_size", "replay_capacity", "warmup_transitions",
    "update_every", "target_update_every", "epsilon_start", "epsilon_final",
    "epsilon_decay_steps", "checkpoint_every_rounds", "gradient_clip", "delta_cap",
    "d4_batch_fraction", "d4_lambda_final", "d4_lambda_warmup_updates",
    "rollout_lambda", "behavior_kl_lambda", "behavior_temperature",
}


def load_config(path=None):
    path = path or os.environ.get("MODEL_A_V8_CONFIG_PATH")
    if not path:
        raise RuntimeError("MODEL_A_V8_CONFIG_PATH is required")
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1 or config.get("agent") != "model_a_v8":
        raise ValueError("v8 config requires schema_version=1 and agent='model_a_v8'")
    if config.get("variant") != "v8-rollout-distill" or not config.get("rollout_distillation"):
        raise ValueError("v8 requires the rollout-distillation variant")
    hyper = config.get("hyperparameters", {})
    if set(hyper) != REQUIRED:
        raise ValueError("v8 hyperparameter keys do not match the schema")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in hyper.values()):
        raise ValueError("v8 hyperparameters must be finite numbers")
    if not 0 <= hyper["epsilon_final"] <= hyper["epsilon_start"] <= 1:
        raise ValueError("invalid epsilon schedule")
    if hyper["delta_cap"] <= 0 or hyper["rollout_lambda"] < 0 or hyper["behavior_kl_lambda"] < 0:
        raise ValueError("invalid residual or regularization weight")
    parent = Path(config["parent_checkpoint"])
    if not parent.is_absolute():
        parent = resolved.parents[2] / parent
    if not parent.is_file() or len(str(config.get("parent_sha256", ""))) != 64:
        raise ValueError("a real frozen parent and SHA-256 are required")
    return config, resolved, hashlib.sha256(raw).hexdigest()


def architecture_name():
    return "model-a-v8-frozen-v4-rollout-residual6"
