"""Immutable contract for the source-r2 kill-head learnability audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASES = ("c1", "c2", "c3", "c4", "c5", "c6")
FOLDS = ("f1", "f2", "f3")
SOURCE_PATHS = {
    "environment.py",
    "settings.py",
    "agent_code/model_a_dqn/callbacks.py",
    "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py",
    "agent_code/model_a_v4_kill_probe/config.py",
    "agent_code/model_a_v4_kill_probe/callbacks.py",
    "agent_code/model_a_v4_kill_probe/train.py",
    "agent_code/seeded_rule_based_agent/callbacks.py",
    "tools/v4_task4_kill_probe_audit.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _register(seen: set[int], case: dict, identity: str) -> None:
    if set(case) != {"world_seed", "agent_seed", "opponent_seed"}:
        raise ValueError(f"invalid seed tuple: {identity}")
    for raw in case.values():
        value = int(raw)
        if value <= 0 or value in seen:
            raise ValueError(f"seeds must be positive and globally unique: {identity}")
        seen.add(value)


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-kill-probe-audit":
        raise ValueError("wrong kill-probe protocol")
    if protocol.get("scope") != "task4c_offline_learnability_only":
        raise ValueError("kill-probe audit may not train or select a policy")
    collection = protocol.get("collection", {})
    if (
        collection.get("scenario") != "classic"
        or collection.get("opponents") != ["seeded_rule_based_agent"] * 3
        or int(collection.get("rounds_per_case", 0)) != 50
        or float(collection.get("epsilon", -1)) != 0.10
        or tuple(collection.get("cases", {})) != CASES
    ):
        raise ValueError("kill-probe collection contract changed")
    seen: set[int] = set()
    for label, case in collection["cases"].items():
        _register(seen, case, f"collection/{label}")
    if len(seen) != 18:
        raise ValueError("kill-probe audit must register exactly 18 unique seeds")

    labeling = protocol.get("labeling", {})
    if labeling != {
        "horizon_steps": 6,
        "positive": "KILLED_OPPONENT occurs in the current or next five recorded transitions of the same episode",
        "risk": "KILLED_SELF occurs in the current or next five recorded transitions of the same episode; GOT_KILLED is not double-counted",
        "terminal_transition_deduplication": "one row per observed (round, step, action)",
    }:
        raise ValueError("kill-probe labels changed")

    probe = protocol.get("probe", {})
    expected_folds = {
        "f1": {"validation": ["c1", "c2"], "training": ["c3", "c4", "c5", "c6"]},
        "f2": {"validation": ["c3", "c4"], "training": ["c1", "c2", "c5", "c6"]},
        "f3": {"validation": ["c5", "c6"], "training": ["c1", "c2", "c3", "c4"]},
    }
    expected_probe = {
        "latent_size": 96,
        "action_count": 6,
        "hidden_size": 48,
        "epochs": 40,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "positive_weight_cap": 100.0,
        "top_fraction": 0.10,
        "folds": expected_folds,
        "probe_seeds": {"f1": 428101, "f2": 428102, "f3": 428103},
        "controls": ["action_conditioned", "state_only_matched_capacity"],
    }
    if probe != expected_probe:
        raise ValueError("kill-probe training contract changed")

    expected_gate = {
        "minimum_total_distinct_kill_events": 30,
        "minimum_validation_distinct_kill_events_per_fold": 8,
        "minimum_action_auprc_lift_over_prevalence": 2.0,
        "minimum_top_decile_kill_lift": 2.0,
        "minimum_action_vs_state_auprc_ratio": 1.10,
        "minimum_top_decile_kill_self_odds_lift": 1.25,
        "minimum_supportive_folds": 2,
        "pass_decision": "kill_head_signal_supported_stop_before_policy_training",
        "fail_decision": "kill_head_signal_not_supported_stop",
    }
    if protocol.get("decision_gate") != expected_gate:
        raise ValueError("kill-probe decision gate changed")
    if protocol.get("automatic_followup") is not False:
        raise ValueError("kill-probe audit must always stop")

    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("kill-probe source binding set mismatch")
    for relative_path, expected_hash in protocol["source_bindings"].items():
        source = ROOT / relative_path
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"kill-probe source binding mismatch: {relative_path}")
    for field in ("source_parent", "previous_task4c_report"):
        item = protocol.get(field, {})
        artifact = ROOT / item.get("path", "missing")
        if not artifact.is_file() or sha256_file(artifact) != item.get("sha256"):
            raise ValueError(f"kill-probe bound artifact mismatch: {field}")
    previous = json.loads((ROOT / protocol["previous_task4c_report"]["path"]).read_text(encoding="utf-8"))
    if previous.get("result", {}).get("decision") != "task4c_training_signal_not_reproducible_stop":
        raise ValueError("kill-probe audit is not bound to the terminal s127000 failure")
    return protocol, sha256_file(path)


def trace_path(protocol: dict, case: str) -> Path:
    if case not in CASES:
        raise ValueError("unknown collection case")
    return (ROOT / protocol["trace_directory"] / f"{case}.npz").resolve()
