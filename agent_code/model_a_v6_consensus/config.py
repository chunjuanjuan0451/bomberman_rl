"""Strict configuration for v6s4 majority-consensus inference."""

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


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path | None = None) -> tuple[dict, Path, str]:
    if path is None:
        path = os.environ.get("MODEL_A_V6_CONSENSUS_CONFIG_PATH")
    if not path:
        raise RuntimeError("MODEL_A_V6_CONSENSUS_CONFIG_PATH is required")
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1 or config.get("agent") != "model_a_v6_consensus":
        raise ValueError("v6 consensus requires schema_version=1 and the consensus agent")
    if config.get("variant") != "v6s4a-action-majority" or config.get("aggregation") != "action_majority":
        raise ValueError("unknown v6 consensus variant")
    if config.get("fallback") != "frozen_v4":
        raise ValueError("v6s4a must fall back to frozen v4")
    delta_cap = config.get("delta_cap")
    if not isinstance(delta_cap, (int, float)) or delta_cap <= 0:
        raise ValueError("delta_cap must be positive")
    root = resolved.parents[2]
    parent = _resolve(root, config.get("parent_checkpoint", ""))
    if not parent.is_file() or sha256_file(parent) != config.get("parent_sha256"):
        raise ValueError("frozen v4 parent file or hash mismatch")
    members = config.get("members")
    if not isinstance(members, list) or len(members) != 3:
        raise ValueError("v6s4a requires exactly three members")
    seen = set()
    for member in members:
        if set(member) != {"training_seed", "checkpoint", "sha256", "config", "config_sha256"}:
            raise ValueError("invalid consensus member schema")
        seed = int(member["training_seed"])
        if seed in seen:
            raise ValueError("consensus training seeds must be unique")
        seen.add(seed)
        checkpoint = _resolve(root, member["checkpoint"])
        member_config = _resolve(root, member["config"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != member["sha256"]:
            raise ValueError(f"consensus member checkpoint hash mismatch: {seed}")
        if not member_config.is_file() or sha256_file(member_config) != member["config_sha256"]:
            raise ValueError(f"consensus member config hash mismatch: {seed}")
    return config, resolved, hashlib.sha256(raw).hexdigest()


def architecture_name() -> str:
    return "model-a-v6s4a-action-majority-v1"
