"""Run the terminal source-r2 counterfactual causal-signal gate.

The runner performs only Phase 1 passive collection.  The Phase 0 Oracle is
covered by unit tests and source bindings.  No action is overridden, no policy
parameter is updated, and no checkpoint is created.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_dqn.features import ACTIONS  # noqa: E402
from agent_code.model_a_v4_causal_signal.config import (  # noqa: E402
    CASES, STRATA, case_stratum, load_protocol, sha256_file, trace_path,
)
from tools.v4_task4_duel_training import (  # noqa: E402
    atomic_json, load_completed, metrics_by_agent, relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-counterfactual-signal-s129000.json"
COLLECTION_KIND = "model-a-v4-counterfactual-signal-collection"
REPORT_KIND = "model-a-v4-counterfactual-signal-report"
TRACE_FIELDS = {
    "round", "step", "action", "legal_mask", "bomb_legal", "oracle_evaluated",
    "oracle_timeout", "strict_opportunity", "guaranteed_traps", "affected_opponents",
    "max_space_reduction", "own_bottleneck", "own_terminal_positions", "kill_event",
    "self_event", "got_killed_event", "metadata",
}


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
        raise ValueError(f"registered s129000 seeds were already used: {conflicts}")


def _trace_payload(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"counterfactual signal trace missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def attempt_outcomes(
    rounds: np.ndarray,
    steps: np.ndarray,
    strict: np.ndarray,
    actions: np.ndarray,
    kill_events: np.ndarray,
    self_events: np.ndarray,
    kill_horizon: int,
    self_horizon: int,
) -> list[dict[str, object]]:
    """Return same-episode realized outcomes for every strict BOMB attempt."""
    rounds = np.asarray(rounds)
    steps = np.asarray(steps)
    attempts = np.flatnonzero(np.asarray(strict, dtype=np.bool_) & (np.asarray(actions) == ACTIONS.index("BOMB")))
    result = []
    for index in attempts:
        same_round = rounds == rounds[index]
        kill_window = same_round & (steps >= steps[index]) & (steps < steps[index] + kill_horizon)
        self_window = same_round & (steps >= steps[index]) & (steps < steps[index] + self_horizon)
        realized_kills = int(np.asarray(kill_events, dtype=np.bool_)[kill_window].sum())
        self_risk = bool(np.asarray(self_events, dtype=np.bool_)[self_window].any())
        result.append({
            "row": int(index),
            "round": int(rounds[index]),
            "step": int(steps[index]),
            "realized_kills": realized_kills,
            "kill_realized": realized_kills > 0,
            "self_risk": self_risk,
            "successful_causal_segment": realized_kills > 0 and not self_risk,
        })
    return result


def summarize_payload(protocol: dict, payload: dict[str, np.ndarray]) -> dict:
    strict = payload["strict_opportunity"].astype(np.bool_)
    actions = payload["action"].astype(np.int64)
    bomb = actions == ACTIONS.index("BOMB")
    outcomes = attempt_outcomes(
        payload["round"], payload["step"], strict, actions,
        payload["kill_event"], payload["self_event"],
        int(protocol["labeling"]["kill_horizon_steps"]),
        int(protocol["labeling"]["self_risk_horizon_steps"]),
    )
    attempts = len(outcomes)
    kills = sum(int(row["kill_realized"]) for row in outcomes)
    self_risks = sum(int(row["self_risk"]) for row in outcomes)
    successful = sum(int(row["successful_causal_segment"]) for row in outcomes)
    bomb_legal = int(payload["bomb_legal"].sum())
    timeouts = int(payload["oracle_timeout"].sum())
    return {
        "rows": len(actions),
        "rounds": len(np.unique(payload["round"])),
        "bomb_legal_decisions": bomb_legal,
        "oracle_evaluated_decisions": int(payload["oracle_evaluated"].sum()),
        "oracle_timeouts": timeouts,
        "oracle_timeout_fraction": timeouts / bomb_legal if bomb_legal else 0.0,
        "strict_opportunities": int(strict.sum()),
        "attempts": attempts,
        "missed_opportunities": int((strict & ~bomb).sum()),
        "kill_realized_attempts": kills,
        "self_risk_attempts": self_risks,
        "successful_causal_segments": successful,
        "kill_realization_rate": kills / attempts if attempts else 0.0,
        "self_risk_rate": self_risks / attempts if attempts else 0.0,
        "distinct_kill_events": int(payload["kill_event"].sum()),
        "distinct_self_kill_events": int(payload["self_event"].sum()),
        "attempt_outcomes": outcomes,
    }


def validate_trace(protocol: dict, protocol_hash: str, case: str) -> dict:
    path = trace_path(protocol, case)
    payload = _trace_payload(path)
    if set(payload) != TRACE_FIELDS:
        raise RuntimeError(f"counterfactual signal trace fields changed: {case}")
    stratum = case_stratum(protocol, case)
    metadata = json.loads(str(payload["metadata"].item()))
    expected = {
        "schema_version": 1,
        "kind": "model-a-v4-counterfactual-signal-trace",
        "protocol_sha256": protocol_hash,
        "case": case,
        "stratum": stratum,
        "parent_sha256": protocol["source_parent"]["sha256"],
        "rounds": int(protocol["collection"]["rounds_per_case"]),
        "collection_epsilon": float(protocol["collection"]["epsilon"]),
        "oracle_deadline_seconds": float(protocol["collection"]["oracle_deadline_seconds"]),
        "policy_updates": 0,
    }
    if metadata != expected:
        raise RuntimeError(f"counterfactual signal trace metadata changed: {case}")
    rows = len(payload["action"])
    if rows <= 0 or payload["legal_mask"].shape != (rows, len(ACTIONS)):
        raise RuntimeError(f"counterfactual signal trace shape mismatch: {case}")
    for key, value in payload.items():
        if key not in {"metadata", "legal_mask"} and len(value) != rows:
            raise RuntimeError(f"counterfactual signal row count mismatch: {case}/{key}")
    if len(np.unique(payload["round"])) != int(protocol["collection"]["rounds_per_case"]):
        raise RuntimeError(f"counterfactual signal round coverage mismatch: {case}")
    action = payload["action"].astype(np.int64)
    if np.any(action < 0) or np.any(action >= len(ACTIONS)):
        raise RuntimeError(f"counterfactual signal action index mismatch: {case}")
    if not np.all(payload["legal_mask"][np.arange(rows), action]):
        raise RuntimeError(f"counterfactual signal recorded an action outside its policy mask: {case}")
    if np.any(payload["self_event"] & ~payload["got_killed_event"]):
        raise RuntimeError(f"self-kill event did not include GOT_KILLED: {case}")
    if np.any(payload["strict_opportunity"] & ~payload["bomb_legal"]):
        raise RuntimeError(f"strict opportunity was not BOMB-legal: {case}")
    strict = payload["strict_opportunity"].astype(np.bool_)
    if np.any(strict & (
        (payload["guaranteed_traps"] < 1)
        | (payload["own_bottleneck"] < 2)
        | (payload["own_terminal_positions"] < 3)
    )):
        raise RuntimeError(f"strict opportunity violated conservative Oracle conditions: {case}")
    summary = summarize_payload(protocol, payload)
    return {"path": relative(path), "sha256": sha256_file(path), "stratum": stratum, **summary}


def collect_one(protocol_path: Path, protocol: dict, protocol_hash: str, case: str) -> dict:
    manifest = collection_manifest_path(protocol, case)
    existing = load_completed(manifest, COLLECTION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed counterfactual signal collection protocol drift: {case}")
        trace = validate_trace(protocol, protocol_hash, case)
        if trace["sha256"] != existing["trace"]["sha256"]:
            raise RuntimeError(f"completed counterfactual signal trace drift: {case}")
        return existing
    output = trace_path(protocol, case)
    stats_path = manifest.with_suffix(".stats.json")
    if output.exists() or stats_path.exists() or manifest.exists():
        raise RuntimeError(f"refusing orphaned counterfactual signal artifacts: {case}")
    collection = protocol["collection"]
    stratum = case_stratum(protocol, case)
    opponents = collection["strata"][stratum]["opponents"]
    seeds = collection["cases"][case]
    parent = (ROOT / protocol["source_parent"]["path"]).resolve()
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_causal_signal", *opponents,
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", collection["scenario"], "--n-rounds", str(collection["rounds_per_case"]),
        "--seed", str(seeds["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides = {
        "MODEL_A_CAUSAL_SIGNAL_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_CAUSAL_SIGNAL_PARENT_PATH": str(parent),
        "MODEL_A_CAUSAL_SIGNAL_CASE": case,
        "MODEL_A_CAUSAL_SIGNAL_SEED": str(seeds["agent_seed"]),
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
        "stratum": stratum,
        "collection": {**seeds, "scenario": collection["scenario"], "opponents": opponents,
                       "rounds": collection["rounds_per_case"], "epsilon": collection["epsilon"]},
        "parent_checkpoint": {"path": relative(parent), "sha256": sha256_file(parent)},
        "command": command,
        "environment_overrides": overrides,
        "trace": {"path": relative(output), "sha256": None},
        "artifacts": {"raw_stats": relative(stats_path)},
        "policy_updates": 0,
        "actions_overridden": 0,
    }
    atomic_json(manifest, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats_path.is_file():
            raise RuntimeError("counterfactual signal collection process or stats failed")
        record["trace"] = validate_trace(protocol, protocol_hash, case)
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats_path.read_text(encoding="utf-8")))
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"counterfactual signal collection failed: {case}: {record.get('error')}")
    return record


def _sum_summaries(summaries: list[dict]) -> dict:
    additive = (
        "rows", "rounds", "bomb_legal_decisions", "oracle_evaluated_decisions",
        "oracle_timeouts", "strict_opportunities", "attempts", "missed_opportunities",
        "kill_realized_attempts", "self_risk_attempts", "successful_causal_segments",
        "distinct_kill_events", "distinct_self_kill_events",
    )
    result = {key: sum(int(item[key]) for item in summaries) for key in additive}
    result["oracle_timeout_fraction"] = (
        result["oracle_timeouts"] / result["bomb_legal_decisions"]
        if result["bomb_legal_decisions"] else 0.0
    )
    result["kill_realization_rate"] = (
        result["kill_realized_attempts"] / result["attempts"] if result["attempts"] else 0.0
    )
    result["self_risk_rate"] = (
        result["self_risk_attempts"] / result["attempts"] if result["attempts"] else 0.0
    )
    return result


def analyze(protocol: dict, collections: dict) -> dict:
    summaries = {case: collections[case]["trace"] for case in CASES}
    strata = {
        stratum: _sum_summaries([summaries[case] for case in protocol["collection"]["strata"][stratum]["cases"]])
        for stratum in STRATA
    }
    pooled = _sum_summaries(list(summaries.values()))
    gate = protocol["decision_gate"]
    checks = {
        "total_attempt_coverage": pooled["attempts"] >= gate["minimum_total_attempts"],
        "total_successful_segment_coverage": pooled["successful_causal_segments"] >= gate["minimum_total_successful_causal_segments"],
        "kill_realization_rate": pooled["kill_realization_rate"] >= gate["minimum_kill_realization_rate"],
        "self_risk_rate": pooled["self_risk_rate"] <= gate["maximum_self_risk_rate"],
        "per_stratum_successful_segment_coverage": all(
            strata[name]["successful_causal_segments"] >= gate["minimum_successful_causal_segments_per_stratum"]
            for name in STRATA
        ),
        "oracle_timeout_fraction": pooled["oracle_timeout_fraction"] <= gate["maximum_oracle_timeout_fraction"],
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "decision": gate["pass_decision"] if passed else gate["fail_decision"],
        "checks": checks,
        "cases": summaries,
        "strata": strata,
        "pooled": pooled,
        "actions_overridden": 0,
        "policy_updates": 0,
        "policy_training_started": False,
        "policy_checkpoint_created": False,
        "phase2_training_started": False,
        "task4_complete": False,
        "automatic_followup_started": False,
    }


def write_report(protocol_path: Path, protocol: dict, protocol_hash: str, collections: dict, result: dict) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed counterfactual signal report protocol drift")
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
            case: {
                "path": relative(collection_manifest_path(protocol, case)),
                "sha256": sha256_file(collection_manifest_path(protocol, case)),
                "trace": collections[case]["trace"],
            }
            for case in CASES
        },
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
        "architecture": "exact-v4",
        "phase0_oracle_tests_required": True,
        "collection_cases": len(CASES),
        "strata": list(STRATA),
        "rounds_per_case": int(protocol["collection"]["rounds_per_case"]),
        "total_collection_rounds": len(CASES) * int(protocol["collection"]["rounds_per_case"]),
        "collection_epsilon": float(protocol["collection"]["epsilon"]),
        "actions_overridden": 0,
        "policy_updates": 0,
        "policy_training_started": False,
        "policy_checkpoint_created": False,
        "phase2_training_started": False,
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
    collections = {
        case: collect_one(protocol_path.resolve(), protocol, protocol_hash, case)
        for case in CASES
    }
    if sha256_file(source) != before or before != protocol["source_parent"]["sha256"]:
        raise RuntimeError("source-r2 changed during passive signal collection")
    result = analyze(protocol, collections)
    report, report_hash = write_report(protocol_path.resolve(), protocol, protocol_hash, collections, result)
    print(json.dumps({
        "status": report["status"], "report": relative(report_path(protocol)),
        "report_sha256": report_hash, **report["result"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
