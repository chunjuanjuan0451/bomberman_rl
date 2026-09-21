"""Configuration validation for Model A v9."""

import hashlib
import json
import math
import os
from pathlib import Path

REQUIRED = {
    "gamma", "learning_rate", "batch_size", "replay_capacity", "warmup_transitions",
    "update_every", "target_update_every", "epsilon_start", "epsilon_final",
    "epsilon_decay_steps", "checkpoint_every_rounds", "gradient_clip", "n_step",
    "n_quantiles", "per_alpha", "per_beta_start", "per_beta_steps", "risk_lambda",
}


def load_config(path=None):
    path = path or os.environ.get("MODEL_A_V9_CONFIG_PATH")
    if not path:
        raise RuntimeError("MODEL_A_V9_CONFIG_PATH is required")
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1 or config.get("agent") != "model_a_v9":
        raise ValueError("v9 config requires schema_version=1 and agent='model_a_v9'")
    if config.get("variant") != "v9-fullboard-qr-dqn":
        raise ValueError("unknown v9 variant")
    hyper = config.get("hyperparameters", {})
    if set(hyper) != REQUIRED or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in hyper.values()):
        raise ValueError("invalid v9 hyperparameter schema")
    if int(hyper["n_quantiles"]) != 32 or int(hyper["n_step"]) < 1:
        raise ValueError("v9 requires 32 quantiles and positive n-step")
    if hyper["warmup_transitions"] < hyper["batch_size"]:
        raise ValueError("warmup must cover a batch")
    return config, resolved, hashlib.sha256(raw).hexdigest()


def architecture_name():
    return "model-a-v9-fullboard-history2-cnn-qr32-risk"
