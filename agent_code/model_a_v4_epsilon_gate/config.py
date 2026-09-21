"""Frozen contract helpers for the exact-v4 Task1 epsilon-floor signal gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPLICAS = ("r1", "r2", "r3")
SOURCE_PATHS = {
    "agent_code/model_a_dqn/callbacks.py",
    "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py",
    "agent_code/model_a_dqn/train.py",
    "agent_code/model_a_dqn/replay_buffer.py",
    "agent_code/model_a_v4_curriculum/config.py",
    "agent_code/model_a_v4_curriculum/callbacks.py",
    "agent_code/model_a_v4_epsilon_gate/config.py",
    "agent_code/model_a_v4_epsilon_gate/callbacks.py",
    "agent_code/model_a_v4_epsilon_gate/train.py",
    "tools/v4_task1_epsilon_floor_gate.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("epsilon-floor gate requires schema_version=1")
    if protocol.get("kind") != "model-a-v4-task1-epsilon-floor-gate":
        raise ValueError("wrong epsilon-floor protocol kind")
    if tuple(protocol.get("replicas", ())) != REPLICAS:
        raise ValueError("exactly the ordered replicas r1, r2, r3 are required")
    if float(protocol.get("epsilon", {}).get("floor", -1)) != 0.15:
        raise ValueError("the preregistered epsilon floor must be 0.15")
    if protocol.get("epsilon", {}).get("semantics") != "original 0.95/80000 linear slope, clamped at 0.15":
        raise ValueError("epsilon clamp semantics changed")
    if int(protocol.get("training_rounds_per_replica", 0)) != 400:
        raise ValueError("each replica must train for exactly 400 rounds")
    for replica, spec in protocol["replicas"].items():
        if set(spec) != {"world_seed", "agent_seed", "opponent_seed"}:
            raise ValueError(f"invalid seed tuple for {replica}")
    training_seeds = [
        int(value)
        for spec in protocol["replicas"].values()
        for value in spec.values()
    ]
    if len(training_seeds) != len(set(training_seeds)):
        raise ValueError("training seeds must be unique")
    evaluation = protocol.get("evaluation", {})
    if tuple(evaluation) != ("paired_seen", "fresh_confirmation"):
        raise ValueError("evaluation suites changed")
    for suite, spec in evaluation.items():
        if int(spec.get("rounds_per_case", 0)) != 25 or len(spec.get("cases", [])) != 2:
            raise ValueError(f"invalid evaluation budget for {suite}")
        for case in spec["cases"]:
            if set(case) != {"world_seed", "agent_seed", "opponent_seed"}:
                raise ValueError(f"invalid evaluation seed tuple for {suite}")
    fresh_seeds = {
        int(value)
        for case in evaluation["fresh_confirmation"]["cases"]
        for value in case.values()
    }
    paired_seeds = {
        int(value)
        for case in evaluation["paired_seen"]["cases"]
        for value in case.values()
    }
    if len(fresh_seeds) != 6 or fresh_seeds & set(training_seeds) or fresh_seeds & paired_seeds:
        raise ValueError("fresh confirmation seeds must be unique and previously unused")
    bindings = protocol.get("source_bindings", {})
    if set(bindings) != SOURCE_PATHS:
        raise ValueError("source binding set mismatch")
    for relative_path, expected_hash in bindings.items():
        source = ROOT / relative_path
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"source binding mismatch: {relative_path}")
    baseline = protocol.get("baseline", {})
    report = ROOT / baseline.get("task1_report_path", "missing")
    if not report.is_file() or sha256_file(report) != baseline.get("task1_report_sha256"):
        raise ValueError("frozen s114000 Task1 report binding mismatch")
    old_protocol = ROOT / baseline.get("protocol_path", "missing")
    if not old_protocol.is_file() or sha256_file(old_protocol) != baseline.get("protocol_sha256"):
        raise ValueError("frozen s114000 protocol binding mismatch")
    for replica in REPLICAS:
        checkpoint = ROOT / baseline["curriculum_checkpoints"][replica]["path"]
        expected = baseline["curriculum_checkpoints"][replica]["sha256"]
        if not checkpoint.is_file() or sha256_file(checkpoint) != expected:
            raise ValueError(f"frozen s114000 checkpoint binding mismatch: {replica}")
    frozen_v4 = ROOT / protocol.get("frozen_v4_baseline", {}).get("checkpoint_path", "missing")
    if (
        not frozen_v4.is_file()
        or sha256_file(frozen_v4)
        != protocol.get("frozen_v4_baseline", {}).get("checkpoint_sha256")
    ):
        raise ValueError("frozen v4 checkpoint binding mismatch")
    return protocol, sha256_file(path)


def checkpoint_path(protocol: dict, replica: str) -> Path:
    if replica not in REPLICAS:
        raise ValueError("invalid replica")
    return (ROOT / protocol["checkpoint_directory"] / replica / "task1.pt").resolve()


def snapshot_path(protocol: dict, replica: str, completed_rounds: int) -> Path:
    if replica not in REPLICAS or completed_rounds not in range(50, 401, 50):
        raise ValueError("invalid diagnostic snapshot identity")
    return (
        ROOT / protocol["snapshot_directory"] / replica
        / f"round-{completed_rounds:04d}.pt"
    ).resolve()
