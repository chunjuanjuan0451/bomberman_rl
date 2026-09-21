"""Strict protocol loading for the n=8 global-resource rescue pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("local7", "full33")
REPLICAS = ("r1", "r2", "r3")
RESOURCE_SHAPE = (2, 33, 33)
ENDPOINT_ROUND = 200
MILESTONES = (100, 200)

REQUIRED_HYPERPARAMETERS = {
    "gamma", "learning_rate", "batch_size", "replay_capacity",
    "warmup_transitions", "update_every", "target_update_every",
    "epsilon", "checkpoint_every_rounds", "gradient_clip", "delta_cap", "n_step",
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
        raise ValueError("global-resource n8 protocol requires schema_version=1")
    if protocol.get("protocol_id") != "model-a-v4-global-resource-n8-pilot-s141000":
        raise ValueError("unexpected global-resource n8 protocol_id")
    if protocol.get("single_training_variable") != "resource_observation_extent_local7_vs_full33_at_fixed_n8":
        raise ValueError("global-resource n8 single-variable contract changed")
    if tuple(protocol.get("training", {}).get("arms", ())) != ARMS:
        raise ValueError("global-resource n8 arm order changed")
    if tuple(protocol.get("training", {}).get("replicas", ())) != REPLICAS:
        raise ValueError("global-resource n8 replica order changed")
    if int(protocol["training"].get("rounds_per_arm", -1)) != ENDPOINT_ROUND:
        raise ValueError("global-resource n8 endpoint must remain round 200")
    if tuple(protocol["training"].get("checkpoint_rounds", ())) != MILESTONES:
        raise ValueError("global-resource n8 milestones changed")
    if protocol["training"].get("scenario") != "classic" or protocol["training"].get("opponents") != []:
        raise ValueError("global-resource n8 pilot must train in opponent-free Task 2")
    hyper = protocol.get("learning_contract", {})
    if set(hyper) != REQUIRED_HYPERPARAMETERS:
        raise ValueError("global-resource n8 hyperparameter schema changed")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in hyper.values()):
        raise ValueError("global-resource n8 hyperparameters must be finite")
    if hyper["n_step"] != 8 or hyper["epsilon"] != 0.10 or hyper["delta_cap"] != 0.5:
        raise ValueError("fixed n-step, epsilon, or residual cap changed")
    if hyper["checkpoint_every_rounds"] != 100 or hyper["warmup_transitions"] < hyper["batch_size"]:
        raise ValueError("invalid checkpoint or warmup contract")
    parent = _resolve(resolved, protocol["frozen_v4"]["path"])
    if not parent.is_file() or sha256_file(parent) != protocol["frozen_v4"]["sha256"]:
        raise ValueError("frozen-v4 checkpoint missing or hash mismatch")
    expected_cases = {"task1": (20, 1), "task2": (25, 2)}
    for stratum, (rounds, cases) in expected_cases.items():
        spec = protocol["evaluation"]["strata"].get(stratum, {})
        if int(spec.get("rounds_per_case", -1)) != rounds or len(spec.get("cases", ())) != cases:
            raise ValueError(f"{stratum} evaluation contract changed")
        if spec.get("opponents") != []:
            raise ValueError(f"{stratum} must remain opponent-free")
    return protocol, resolved, hashlib.sha256(raw).hexdigest()


def checkpoint_path(protocol: dict, arm: str, replica: str) -> Path:
    if arm not in ARMS or replica not in REPLICAS:
        raise ValueError("unknown global-resource n8 arm or replica")
    return ROOT / protocol["checkpoint_directory"] / arm / replica / "round-0200.pt"


def milestone_path(protocol: dict, arm: str, replica: str, round_number: int) -> Path:
    if round_number not in MILESTONES:
        raise ValueError("only preregistered n8 milestones are allowed")
    return ROOT / protocol["checkpoint_directory"] / arm / replica / f"round-{round_number:04d}.pt"


def architecture_name() -> str:
    return "model-a-v4-frozen-global-resource-residual-n8-v1"
