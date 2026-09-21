"""Strict configuration for the inference-only v6 residual ensemble."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(repository_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def load_config(path: str | Path | None = None) -> tuple[dict, Path, str]:
    if path is None:
        path = os.environ.get("MODEL_A_V6_ENSEMBLE_CONFIG_PATH")
    if not path:
        raise RuntimeError("MODEL_A_V6_ENSEMBLE_CONFIG_PATH is required")
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1 or config.get("agent") != "model_a_v6_ensemble":
        raise ValueError("v6 ensemble requires schema_version=1 and agent='model_a_v6_ensemble'")
    if config.get("variant") != "v6s3-mean-residual-ensemble":
        raise ValueError("unknown v6 ensemble variant")
    if config.get("aggregation") != "mean":
        raise ValueError("v6s3 requires mean residual aggregation")
    delta_cap = config.get("delta_cap")
    if not isinstance(delta_cap, (int, float)) or delta_cap <= 0:
        raise ValueError("delta_cap must be positive")
    repository_root = resolved.parents[2]
    parent = _resolve(repository_root, config.get("parent_checkpoint", ""))
    if not parent.is_file() or sha256_file(parent) != config.get("parent_sha256"):
        raise ValueError("frozen v4 parent file or hash mismatch")
    members = config.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("v6 ensemble requires at least two members")
    seen = set()
    for member in members:
        if set(member) != {"training_seed", "checkpoint", "sha256", "config", "config_sha256"}:
            raise ValueError("invalid ensemble member schema")
        training_seed = int(member["training_seed"])
        if training_seed in seen:
            raise ValueError("ensemble training seeds must be unique")
        seen.add(training_seed)
        checkpoint = _resolve(repository_root, member["checkpoint"])
        member_config = _resolve(repository_root, member["config"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != member["sha256"]:
            raise ValueError(f"ensemble member checkpoint hash mismatch: {training_seed}")
        if not member_config.is_file() or sha256_file(member_config) != member["config_sha256"]:
            raise ValueError(f"ensemble member config hash mismatch: {training_seed}")
    return config, resolved, hashlib.sha256(raw).hexdigest()


def architecture_name() -> str:
    return "model-a-v6s3-mean-residual-ensemble-v1"
