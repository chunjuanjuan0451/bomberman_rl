"""Collect source-r2 trajectories and audit an action-conditioned kill probe.

This is a diagnostic-only, terminal experiment.  It freezes source-r2, records
Task4C trajectories with mild legal-action exploration, and trains fixed-fold
offline probes.  No probe participates in game decisions and no policy
checkpoint is produced.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v4_kill_probe.config import (  # noqa: E402
    CASES, FOLDS, load_protocol, sha256_file, trace_path,
)
from tools.v4_task4_duel_training import atomic_json, load_completed, metrics_by_agent, relative, seed_values  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-kill-probe-audit-s128000.json"
COLLECTION_KIND = "model-a-v4-kill-probe-collection"
REPORT_KIND = "model-a-v4-kill-probe-audit-report"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collection_manifest_path(protocol: dict, case: str) -> Path:
    return ROOT / protocol["collection_manifest_directory"] / f"{case}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def registered_game_seeds(protocol: dict) -> set[int]:
    result = set()
    for case in protocol["collection"]["cases"].values():
        result.update(int(value) for value in case.values())
    return result


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = registered_game_seeds(protocol)
    excluded_roots = {
        (ROOT / protocol["trace_directory"]).resolve(),
        (ROOT / protocol["collection_manifest_directory"]).resolve(),
    }
    excluded_files = {protocol_path.resolve(), report_path(protocol).resolve()}
    conflicts = []
    for path in (ROOT / "experiments").rglob("*.json"):
        resolved = path.resolve()
        if resolved in excluded_files or any(root == resolved or root in resolved.parents for root in excluded_roots):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overlap = sorted(registered & seed_values(payload))
        if overlap:
            conflicts.append({"path": relative(path), "seeds": overlap})
    if conflicts:
        raise ValueError(f"registered s128000 collection seeds were already used: {conflicts}")


def _trace_payload(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"kill-probe trace missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def validate_trace(protocol: dict, protocol_hash: str, case: str) -> dict:
    path = trace_path(protocol, case)
    payload = _trace_payload(path)
    required = {
        "latent", "action", "episode", "step", "legal_mask", "q_values",
        "kill_event", "self_event", "got_killed_event", "metadata",
    }
    if set(payload) != required:
        raise RuntimeError(f"kill-probe trace fields changed: {case}")
    metadata = json.loads(str(payload["metadata"].item()))
    expected = {
        "schema_version": 1,
        "kind": "model-a-v4-kill-probe-trace",
        "protocol_sha256": protocol_hash,
        "case": case,
        "parent_sha256": protocol["source_parent"]["sha256"],
        "rounds": int(protocol["collection"]["rounds_per_case"]),
        "collection_epsilon": float(protocol["collection"]["epsilon"]),
    }
    if metadata != expected:
        raise RuntimeError(f"kill-probe trace metadata changed: {case}")
    rows = len(payload["action"])
    if rows <= 0 or payload["latent"].shape != (rows, 96):
        raise RuntimeError(f"kill-probe latent shape mismatch: {case}")
    if payload["legal_mask"].shape != (rows, 6) or payload["q_values"].shape != (rows, 6):
        raise RuntimeError(f"kill-probe action tensor shape mismatch: {case}")
    for key in ("episode", "step", "kill_event", "self_event", "got_killed_event"):
        if len(payload[key]) != rows:
            raise RuntimeError(f"kill-probe row count mismatch: {case}/{key}")
    if set(np.unique(payload["episode"]).tolist()) != set(range(50)):
        raise RuntimeError(f"kill-probe episode coverage mismatch: {case}")
    action = payload["action"].astype(np.int64)
    if np.any(action < 0) or np.any(action >= 6):
        raise RuntimeError(f"kill-probe action index mismatch: {case}")
    if not np.all(payload["legal_mask"][np.arange(rows), action]):
        raise RuntimeError(f"kill-probe recorded an action outside its policy mask: {case}")
    if np.any(payload["self_event"] & ~payload["got_killed_event"]):
        raise RuntimeError(f"self-kill event did not include GOT_KILLED: {case}")
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "rounds": 50,
        "distinct_kill_events": int(payload["kill_event"].sum()),
        "distinct_self_kill_events": int(payload["self_event"].sum()),
    }


def collect_one(protocol_path: Path, protocol: dict, protocol_hash: str, case: str) -> dict:
    manifest = collection_manifest_path(protocol, case)
    existing = load_completed(manifest, COLLECTION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed kill-probe collection protocol drift: {case}")
        trace = validate_trace(protocol, protocol_hash, case)
        if trace["sha256"] != existing["trace"]["sha256"]:
            raise RuntimeError(f"completed kill-probe trace drift: {case}")
        return existing
    output = trace_path(protocol, case)
    stats_path = manifest.with_suffix(".stats.json")
    if output.exists() or stats_path.exists() or manifest.exists():
        raise RuntimeError(f"refusing orphaned kill-probe collection artifacts: {case}")
    spec = protocol["collection"]
    seeds = spec["cases"][case]
    parent = (ROOT / protocol["source_parent"]["path"]).resolve()
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_kill_probe", *spec["opponents"],
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_case"]),
        "--seed", str(seeds["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides = {
        "MODEL_A_KILL_PROBE_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_KILL_PROBE_PARENT_PATH": str(parent),
        "MODEL_A_KILL_PROBE_CASE": case,
        "MODEL_A_KILL_PROBE_SEED": str(seeds["agent_seed"]),
        "TASK4_RULE_SEED": str(seeds["opponent_seed"]),
    }
    record = {
        "schema_version": 1,
        "kind": COLLECTION_KIND,
        "status": "running",
        "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "case": case,
        "collection": {**seeds, "scenario": spec["scenario"], "opponents": spec["opponents"],
                       "rounds": spec["rounds_per_case"], "epsilon": spec["epsilon"]},
        "parent_checkpoint": {"path": relative(parent), "sha256": sha256_file(parent)},
        "command": command,
        "environment_overrides": overrides,
        "trace": {"path": relative(output), "sha256": None},
        "artifacts": {"raw_stats": relative(stats_path)},
        "policy_updates": 0,
    }
    atomic_json(manifest, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats_path.is_file():
            raise RuntimeError("kill-probe collection process or stats failed")
        trace = validate_trace(protocol, protocol_hash, case)
        record["trace"] = trace
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats_path.read_text(encoding="utf-8")))
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"kill-probe collection failed: {case}: {record.get('error')}")
    return record


def window_labels(episodes: np.ndarray, events: np.ndarray, horizon: int) -> np.ndarray:
    """Mark rows whose current/future same-episode horizon contains an event."""
    episodes = np.asarray(episodes)
    events = np.asarray(events, dtype=np.bool_)
    result = np.zeros(len(events), dtype=np.bool_)
    for episode in np.unique(episodes):
        indices = np.flatnonzero(episodes == episode)
        event_positions = np.flatnonzero(events[indices])
        for position in event_positions:
            start = max(0, int(position) - horizon + 1)
            result[indices[start:position + 1]] = True
    return result


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order].astype(np.float64)
    ranked_scores = scores[order]
    true_positives = np.cumsum(ranked_labels)
    # Evaluate once after every complete tie group.  This avoids making AP
    # depend on the original row order when a probe emits identical scores.
    group_ends = np.r_[np.flatnonzero(np.diff(ranked_scores) != 0), len(ranked_scores) - 1]
    precision = true_positives[group_ends] / (group_ends + 1)
    recall = true_positives[group_ends] / positives
    recall_gain = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_gain * precision))


def ranking_metrics(kill: np.ndarray, self_risk: np.ndarray, scores: np.ndarray, top_fraction: float) -> dict:
    kill = np.asarray(kill, dtype=np.bool_)
    self_risk = np.asarray(self_risk, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    count = len(kill)
    top_count = max(1, int(math.ceil(count * top_fraction)))
    top = np.argsort(-scores, kind="mergesort")[:top_count]
    prevalence = float(kill.mean())
    top_kill_rate = float(kill[top].mean())
    top_self_rate = float(self_risk[top].mean())
    overall_self_rate = float(self_risk.mean())
    overall_odds = (float(kill.sum()) + 0.5) / (float(self_risk.sum()) + 0.5)
    top_odds = (float(kill[top].sum()) + 0.5) / (float(self_risk[top].sum()) + 0.5)
    return {
        "rows": count,
        "positive_windows": int(kill.sum()),
        "self_risk_windows": int(self_risk.sum()),
        "prevalence": prevalence,
        "auprc": average_precision(kill, scores),
        "auprc_lift_over_prevalence": average_precision(kill, scores) / max(prevalence, 1e-12),
        "top_count": top_count,
        "top_kill_rate": top_kill_rate,
        "top_self_risk_rate": top_self_rate,
        "overall_self_risk_rate": overall_self_rate,
        "top_kill_lift": top_kill_rate / max(prevalence, 1e-12),
        "top_kill_self_odds_lift": top_odds / max(overall_odds, 1e-12),
    }


def _probe_scores(
    train_latent: np.ndarray, train_action: np.ndarray, train_labels: np.ndarray,
    validation_latent: np.ndarray, validation_action: np.ndarray,
    protocol: dict, seed: int, action_conditioned: bool,
) -> np.ndarray:
    from agent_code.model_a_dqn.network import torch
    if torch is None:
        raise RuntimeError("kill-probe analysis requires PyTorch")
    hyper = protocol["probe"]
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    mean = train_latent.mean(axis=0, keepdims=True)
    std = train_latent.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    def inputs(latent, action):
        scaled = ((latent - mean) / std).astype(np.float32)
        one_hot = np.zeros((len(action), 6), dtype=np.float32)
        if action_conditioned:
            one_hot[np.arange(len(action)), action.astype(np.int64)] = 1.0
        return np.concatenate((scaled, one_hot), axis=1)

    x_train = torch.as_tensor(inputs(train_latent, train_action))
    y_train = torch.as_tensor(train_labels.astype(np.float32))
    x_validation = torch.as_tensor(inputs(validation_latent, validation_action))
    model = torch.nn.Sequential(
        torch.nn.Linear(102, int(hyper["hidden_size"])), torch.nn.ReLU(),
        torch.nn.Linear(int(hyper["hidden_size"]), 1),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(hyper["learning_rate"]), weight_decay=float(hyper["weight_decay"]),
    )
    positive = max(1, int(train_labels.sum()))
    negative = max(1, len(train_labels) - positive)
    weight = min(float(hyper["positive_weight_cap"]), negative / positive)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    generator = torch.Generator().manual_seed(seed + 10_000)
    batch_size = int(hyper["batch_size"])
    model.train()
    for _ in range(int(hyper["epochs"])):
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            loss = criterion(model(x_train[index]).squeeze(1), y_train[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x_validation).squeeze(1)).cpu().numpy()


def _load_labeled_cases(protocol: dict) -> dict[str, dict[str, np.ndarray]]:
    horizon = int(protocol["labeling"]["horizon_steps"])
    result = {}
    for case in CASES:
        payload = _trace_payload(trace_path(protocol, case))
        result[case] = {
            "latent": payload["latent"].astype(np.float32),
            "action": payload["action"].astype(np.int64),
            "kill": window_labels(payload["episode"], payload["kill_event"], horizon),
            "self_risk": window_labels(payload["episode"], payload["self_event"], horizon),
            "kill_event": payload["kill_event"].astype(np.bool_),
        }
    return result


def _concat(cases: dict, labels: list[str], field: str) -> np.ndarray:
    return np.concatenate([cases[label][field] for label in labels], axis=0)


def analyze(protocol: dict) -> tuple[dict, dict]:
    cases = _load_labeled_cases(protocol)
    gate = protocol["decision_gate"]
    folds = {}
    pooled = {"kill": [], "self_risk": [], "action_scores": [], "state_scores": []}
    supportive_count = 0
    for fold in FOLDS:
        spec = protocol["probe"]["folds"][fold]
        train_labels, validation_labels = spec["training"], spec["validation"]
        train_latent = _concat(cases, train_labels, "latent")
        train_action = _concat(cases, train_labels, "action")
        train_kill = _concat(cases, train_labels, "kill")
        validation_latent = _concat(cases, validation_labels, "latent")
        validation_action = _concat(cases, validation_labels, "action")
        validation_kill = _concat(cases, validation_labels, "kill")
        validation_self = _concat(cases, validation_labels, "self_risk")
        seed = int(protocol["probe"]["probe_seeds"][fold])
        action_scores = _probe_scores(
            train_latent, train_action, train_kill, validation_latent, validation_action,
            protocol, seed, True,
        )
        state_scores = _probe_scores(
            train_latent, train_action, train_kill, validation_latent, validation_action,
            protocol, seed, False,
        )
        action_metrics = ranking_metrics(
            validation_kill, validation_self, action_scores, float(protocol["probe"]["top_fraction"]),
        )
        state_metrics = ranking_metrics(
            validation_kill, validation_self, state_scores, float(protocol["probe"]["top_fraction"]),
        )
        distinct_validation_kills = sum(int(cases[label]["kill_event"].sum()) for label in validation_labels)
        action_vs_state = action_metrics["auprc"] / max(state_metrics["auprc"], 1e-12)
        checks = {
            "validation_event_coverage": distinct_validation_kills >= gate["minimum_validation_distinct_kill_events_per_fold"],
            "action_auprc_lift": action_metrics["auprc_lift_over_prevalence"] >= gate["minimum_action_auprc_lift_over_prevalence"],
            "top_decile_kill_lift": action_metrics["top_kill_lift"] >= gate["minimum_top_decile_kill_lift"],
            "action_specificity": action_vs_state >= gate["minimum_action_vs_state_auprc_ratio"],
            "kill_vs_self_discrimination": action_metrics["top_kill_self_odds_lift"] >= gate["minimum_top_decile_kill_self_odds_lift"],
        }
        supportive = all(checks.values())
        supportive_count += int(supportive)
        folds[fold] = {
            "training_cases": train_labels,
            "validation_cases": validation_labels,
            "probe_seed": seed,
            "training_rows": len(train_kill),
            "training_positive_windows": int(train_kill.sum()),
            "validation_distinct_kill_events": distinct_validation_kills,
            "action_conditioned": action_metrics,
            "state_only_matched_capacity": state_metrics,
            "action_vs_state_auprc_ratio": action_vs_state,
            "checks": checks,
            "supportive": supportive,
        }
        pooled["kill"].append(validation_kill)
        pooled["self_risk"].append(validation_self)
        pooled["action_scores"].append(action_scores)
        pooled["state_scores"].append(state_scores)

    pooled_action = ranking_metrics(
        np.concatenate(pooled["kill"]), np.concatenate(pooled["self_risk"]),
        np.concatenate(pooled["action_scores"]), float(protocol["probe"]["top_fraction"]),
    )
    pooled_state = ranking_metrics(
        np.concatenate(pooled["kill"]), np.concatenate(pooled["self_risk"]),
        np.concatenate(pooled["state_scores"]), float(protocol["probe"]["top_fraction"]),
    )
    total_distinct_kills = sum(int(cases[label]["kill_event"].sum()) for label in CASES)
    pooled_ratio = pooled_action["auprc"] / max(pooled_state["auprc"], 1e-12)
    checks = {
        "total_event_coverage": total_distinct_kills >= gate["minimum_total_distinct_kill_events"],
        "supportive_fold_count": supportive_count >= gate["minimum_supportive_folds"],
        "pooled_action_auprc_lift": pooled_action["auprc_lift_over_prevalence"] >= gate["minimum_action_auprc_lift_over_prevalence"],
        "pooled_top_decile_kill_lift": pooled_action["top_kill_lift"] >= gate["minimum_top_decile_kill_lift"],
        "pooled_action_specificity": pooled_ratio >= gate["minimum_action_vs_state_auprc_ratio"],
        "pooled_kill_vs_self_discrimination": pooled_action["top_kill_self_odds_lift"] >= gate["minimum_top_decile_kill_self_odds_lift"],
    }
    passed = all(checks.values())
    result = {
        "passed": passed,
        "decision": gate["pass_decision"] if passed else gate["fail_decision"],
        "checks": checks,
        "supportive_folds": supportive_count,
        "supportive_folds_required": gate["minimum_supportive_folds"],
        "total_distinct_kill_events": total_distinct_kills,
        "pooled": {
            "action_conditioned": pooled_action,
            "state_only_matched_capacity": pooled_state,
            "action_vs_state_auprc_ratio": pooled_ratio,
        },
        "policy_training_started": False,
        "policy_checkpoint_created": False,
        "task4_complete": False,
        "automatic_followup_started": False,
    }
    return folds, result


def write_report(protocol_path: Path, protocol: dict, protocol_hash: str, collections: dict, folds: dict, result: dict) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed kill-probe report protocol drift")
        return existing, sha256_file(path)
    record = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "scope": protocol["scope"],
        "source_parent": protocol["source_parent"],
        "collections": {
            case: {"path": relative(collection_manifest_path(protocol, case)),
                   "sha256": sha256_file(collection_manifest_path(protocol, case)),
                   "trace": collections[case]["trace"]}
            for case in CASES
        },
        "folds": folds,
        "result": result,
        "awaiting_user_instruction": True,
        "source_checkpoint_modified": False,
        "default_checkpoint_modified": False,
    }
    atomic_json(path, record)
    return record, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "scope": protocol["scope"],
        "parent": "source-r2",
        "collection_cases": len(CASES),
        "rounds_per_case": int(protocol["collection"]["rounds_per_case"]),
        "total_collection_rounds": len(CASES) * int(protocol["collection"]["rounds_per_case"]),
        "collection_epsilon": float(protocol["collection"]["epsilon"]),
        "cross_validation_folds": len(FOLDS),
        "policy_updates": 0,
        "policy_training_started": False,
        "policy_checkpoint_created": False,
        "formal_collection_started": False,
        "automatic_followup_started": False,
        "writes_terminal_report_and_stops": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol, protocol_hash = load_protocol(protocol_path)
    assert_registered_seeds_untouched(protocol_path, protocol)
    if not args.execute:
        print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
        return 0
    existing = load_completed(report_path(protocol), REPORT_KIND)
    if existing is not None:
        print(json.dumps({
            "status": existing["status"], "report": relative(report_path(protocol)),
            "report_sha256": sha256_file(report_path(protocol)), **existing["result"],
        }, indent=2, sort_keys=True))
        return 0
    source = ROOT / protocol["source_parent"]["path"]
    before = sha256_file(source)
    collections = {case: collect_one(protocol_path.resolve(), protocol, protocol_hash, case) for case in CASES}
    if sha256_file(source) != before or before != protocol["source_parent"]["sha256"]:
        raise RuntimeError("source-r2 changed during the diagnostic collection")
    folds, result = analyze(protocol)
    report, report_hash = write_report(protocol_path.resolve(), protocol, protocol_hash, collections, folds, result)
    print(json.dumps({
        "status": report["status"], "report": relative(report_path(protocol)),
        "report_sha256": report_hash, **report["result"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
