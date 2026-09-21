"""Strict protocol loading for the v4 global-resource stage-1 experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("local7", "full33")
EVALUATION_ARMS = ("frozen-v4", *ARMS)
REPLICAS = ("r1", "r2", "r3")
RESOURCE_SHAPE = (2, 33, 33)
ENDPOINT_ROUND = 400

REQUIRED_HYPERPARAMETERS = {
    "gamma", "learning_rate", "batch_size", "replay_capacity",
    "warmup_transitions", "update_every", "target_update_every",
    "epsilon", "checkpoint_every_rounds", "gradient_clip", "delta_cap",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(protocol_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else protocol_path.parents[2] / path


def load_protocol(path: str | Path | None = None) -> tuple[dict, Path, str]:
    path = path or os.environ.get("MODEL_A_GLOBAL_PROTOCOL_PATH")
    if not path:
        raise RuntimeError("MODEL_A_GLOBAL_PROTOCOL_PATH is required")
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema_version") != 1:
        raise ValueError("global-resource protocol requires schema_version=1")
    if protocol.get("protocol_id") != "model-a-v4-global-resource-stage1-s140000":
        raise ValueError("unexpected global-resource protocol_id")
    if protocol.get("single_training_variable") != "resource_observation_extent_local7_vs_full33":
        raise ValueError("global-resource single-variable contract changed")
    if tuple(protocol.get("training", {}).get("arms", ())) != ARMS:
        raise ValueError("global-resource arm order changed")
    if tuple(protocol.get("training", {}).get("replicas", ())) != REPLICAS:
        raise ValueError("global-resource replica order changed")
    if int(protocol["training"].get("rounds_per_arm", -1)) != ENDPOINT_ROUND:
        raise ValueError("global-resource endpoint must remain round 400")
    if protocol["training"].get("scenario") != "classic" or protocol["training"].get("opponents") != []:
        raise ValueError("stage 1 must train only in opponent-free Task 2 classic")
    hyper = protocol.get("learning_contract", {})
    if set(hyper) != REQUIRED_HYPERPARAMETERS:
        raise ValueError("global-resource hyperparameter schema changed")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in hyper.values()):
        raise ValueError("global-resource hyperparameters must be finite numbers")
    if hyper["epsilon"] != 0.10 or hyper["delta_cap"] != 0.5:
        raise ValueError("fixed epsilon or residual cap changed")
    if hyper["checkpoint_every_rounds"] != 100 or hyper["warmup_transitions"] < hyper["batch_size"]:
        raise ValueError("invalid global-resource checkpoint or warmup contract")
    parent = _resolve(resolved, protocol["frozen_v4"]["path"])
    if not parent.is_file() or sha256_file(parent) != protocol["frozen_v4"]["sha256"]:
        raise ValueError("frozen-v4 checkpoint missing or hash mismatch")
    for stratum in ("task1", "task2"):
        spec = protocol["evaluation"]["strata"].get(stratum, {})
        if int(spec.get("rounds_per_case", -1)) != 25 or len(spec.get("cases", ())) != 2:
            raise ValueError(f"{stratum} evaluation contract changed")
        if spec.get("opponents") != []:
            raise ValueError(f"{stratum} must remain opponent-free")
    return protocol, resolved, hashlib.sha256(raw).hexdigest()


def checkpoint_path(protocol: dict, arm: str, replica: str) -> Path:
    if arm not in ARMS or replica not in REPLICAS:
        raise ValueError("unknown global-resource arm or replica")
    return ROOT / protocol["checkpoint_directory"] / arm / replica / "round-0400.pt"


def milestone_path(protocol: dict, arm: str, replica: str, round_number: int) -> Path:
    if round_number not in (100, 200, 300, 400):
        raise ValueError("only preregistered 100-round milestones are allowed")
    return ROOT / protocol["checkpoint_directory"] / arm / replica / f"round-{round_number:04d}.pt"


def architecture_name() -> str:
    return "model-a-v4-frozen-global-resource-residual-v1"
