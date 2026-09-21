"""Protocol helpers for the collision-mask paired A/B."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> tuple[dict, Path, str]:
    value = os.environ.get("CNN_COLLISION_AB_PROTOCOL")
    if not value:
        raise RuntimeError("CNN_COLLISION_AB_PROTOCOL is required")
    path = Path(value).resolve()
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("kind") != "model-a-cnn-n8-collision-mask-ab":
        raise RuntimeError("invalid collision-mask A/B protocol")
    return protocol, path, hashlib.sha256(raw).hexdigest()


def trace_path(protocol: dict, case_label: str, arm: str) -> Path:
    return ROOT / protocol["trace_directory"] / f"{case_label}-{arm}.json"


__all__ = ["ROOT", "load_protocol", "sha256_file", "trace_path"]
