"""Evaluate the adjudicated source-r2 Task3 checkpoint at the Task4 boundary.

The gate compares source-r2 and frozen v4 against one and three deterministic
copies of the supplied full-strength rule-based agent.  It never trains and
never launches a follow-up.  Its only decision is whether Task4 training is
needed or the unchanged source-r2 policy is already competitive.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v4_curriculum.config import (  # noqa: E402
    load_protocol as load_clean_protocol,
    sha256_file,
)
from tools.v4_clean_curriculum import validate_checkpoint as validate_clean_checkpoint  # noqa: E402
from tools.v4_task3_source_r2_final_confirmation import (  # noqa: E402
    aggregate,
    atomic_json,
    load_completed,
    metrics_by_agent,
    relative,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task4-rule-baseline-gate-s123000.json"
LABELS = ("v4", "source-r2")
STRATA = ("task4_rule_duel", "task4_three_rule")
EVALUATION_KIND = "model-a-v4-task4-rule-baseline-evaluation"
REPORT_KIND = "model-a-v4-task4-rule-baseline-report"
EXPECTED_CASES = {
    "task4_rule_duel": tuple(
        {"world_seed": 123100 + index, "agent_seed": 223100 + index, "opponent_seed": 323100 + index}
        for index in range(1, 5)
    ),
    "task4_three_rule": tuple(
        {"world_seed": 123200 + index, "agent_seed": 223200 + index, "opponent_seed": 323200 + index}
        for index in range(1, 5)
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seed_values(payload: object) -> set[int]:
    found: set[int] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "seed" or key.endswith("_seed") or key.endswith("_seeds"):
                values = value if isinstance(value, list) else [value]
                found.update(item for item in values if isinstance(item, int))
            found.update(seed_values(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(seed_values(value))
    return found


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = {
        int(value)
        for cases in EXPECTED_CASES.values()
        for case in cases
        for value in case.values()
    }
    excluded_root = (ROOT / protocol["evaluation_manifest_directory"]).resolve()
    excluded_files = {protocol_path.resolve(), (ROOT / protocol["report_path"]).resolve()}
    conflicts = []
    for path in (ROOT / "experiments").rglob("*.json"):
        resolved = path.resolve()
        if resolved in excluded_files or excluded_root in resolved.parents:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overlap = sorted(registered & seed_values(payload))
        if overlap:
            conflicts.append({"path": relative(path), "seeds": overlap})
    if conflicts:
        raise ValueError(f"registered s123000 seeds were already used: {conflicts}")


def load_protocol(path: Path) -> tuple[dict, str, dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-task4-rule-baseline-gate":
        raise ValueError("wrong Task4 rule-baseline protocol")
    if tuple(protocol.get("labels", ())) != LABELS or tuple(protocol.get("evaluation", {}).get("strata", {})) != STRATA:
        raise ValueError("Task4 labels or strata changed")
    if protocol.get("training_allowed") is not False or protocol.get("automatic_followup") is not False:
        raise ValueError("Task4 baseline gate must remain evaluation-only and terminal")
    expected_opponents = {
        "task4_rule_duel": ["seeded_rule_based_agent"],
        "task4_three_rule": ["seeded_rule_based_agent"] * 3,
    }
    seeds = []
    for stratum, spec in protocol["evaluation"]["strata"].items():
        if spec.get("scenario") != "classic" or spec.get("opponents") != expected_opponents[stratum]:
            raise ValueError(f"Task4 evaluation distribution changed: {stratum}")
        if int(spec.get("rounds_per_case", 0)) != 25 or tuple(spec.get("cases", ())) != EXPECTED_CASES[stratum]:
            raise ValueError(f"Task4 case budget or seeds changed: {stratum}")
        for case in spec["cases"]:
            seeds.extend(int(value) for value in case.values())
    if len(seeds) != 24 or len(set(seeds)) != 24:
        raise ValueError("Task4 registered seeds must be globally unique")
    assert_registered_seeds_untouched(path, protocol)

    for name, expected_hash in protocol.get("source_bindings", {}).items():
        source = ROOT / name
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"Task4 source binding mismatch: {name}")
    if set(protocol.get("source_bindings", {})) != {
        "tools/v4_task4_rule_baseline_gate.py",
        "tools/v4_task3_source_r2_final_confirmation.py",
    }:
        raise ValueError("Task4 source binding set mismatch")

    audit_item = protocol["task3_adjudication"]["audit_report"]
    audit_path = ROOT / audit_item["path"]
    if not audit_path.is_file() or sha256_file(audit_path) != audit_item["sha256"]:
        raise ValueError("Task3 boundary-audit report binding mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "completed"
        or audit.get("result", {}).get("decision") != "dynamic_collision_boundary_confirmed"
        or audit.get("result", {}).get("task3_adjudicated_for_task4_progression") is not True
    ):
        raise ValueError("Task3 boundary was not adjudicated for Task4 progression")
    selection_item = protocol["task3_adjudication"]["manifest"]
    selection_path = ROOT / selection_item["path"]
    if not selection_path.is_file() or sha256_file(selection_path) != selection_item["sha256"]:
        raise ValueError("Task3 adjudication-manifest binding mismatch")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("adjudication") != "accept_source_r2_for_task4_progression_after_dynamic_collision_boundary_audit":
        raise ValueError("Task3 adjudication decision mismatch")

    clean_item = protocol["source_lineage"]["clean_protocol"]
    clean_protocol, clean_hash = load_clean_protocol(ROOT / clean_item["path"])
    if clean_hash != clean_item["sha256"]:
        raise ValueError("clean-v4 protocol hash mismatch")
    validate_clean_checkpoint(clean_protocol, clean_hash, "curriculum", "r2", "task2")
    inventory = protocol["checkpoint_inventory"]
    for key in ("v4", "source_r2", "adjudicated_task3"):
        item = inventory[key]
        checkpoint = ROOT / item["path"]
        if not checkpoint.is_file() or sha256_file(checkpoint) != item["sha256"]:
            raise ValueError(f"Task4 checkpoint binding mismatch: {key}")
    if inventory["source_r2"]["sha256"] != inventory["adjudicated_task3"]["sha256"]:
        raise ValueError("adjudicated Task3 checkpoint is not byte-identical to source-r2")

    expected_rule = {
        "maximum_candidate_score_gap_to_v4": 0.25,
        "minimum_candidate_kills_delta_to_v4": 0.0,
        "maximum_candidate_suicide_increase_over_v4": 0.05,
        "minimum_duel_score_margin_over_rule": 0.0,
        "minimum_three_rule_score_margin_over_rule_mean": 0.0,
    }
    for key, value in expected_rule.items():
        if protocol.get("decision_rule", {}).get(key) != value:
            raise ValueError(f"Task4 baseline decision rule changed: {key}")
    return protocol, sha256_file(path), clean_protocol, clean_hash


def checkpoint_identity(protocol: dict, label: str) -> tuple[str, Path, dict[str, str]]:
    inventory = protocol["checkpoint_inventory"]
    if label == "v4":
        return "model_a_dqn", ROOT / inventory["v4"]["path"], {}
    return "model_a_v4_curriculum", ROOT / inventory["source_r2"]["path"], {
        "MODEL_A_V4C_PROTOCOL_PATH": str(ROOT / protocol["source_lineage"]["clean_protocol"]["path"]),
        "MODEL_A_V4C_ARM": "curriculum",
        "MODEL_A_V4C_REPLICA": "r2",
        "MODEL_A_V4C_STAGE": "task2",
    }


def manifest_path(protocol: dict, stratum: str, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / stratum / f"{label}-s{world_seed}.json"


def evaluate_one(protocol_path: Path, protocol: dict, protocol_hash: str, stratum: str, spec: dict, label: str, case: dict) -> dict:
    target, checkpoint, overrides = checkpoint_identity(protocol, label)
    checkpoint = checkpoint.resolve()
    expected_hash = protocol["checkpoint_inventory"]["v4" if label == "v4" else "source_r2"]["sha256"]
    if sha256_file(checkpoint) != expected_hash:
        raise RuntimeError(f"Task4 checkpoint drift: {label}")
    output = manifest_path(protocol, stratum, label, int(case["world_seed"]))
    expected = {
        **case,
        "scenario": spec["scenario"],
        "opponents": spec["opponents"],
        "rounds": int(spec["rounds_per_case"]),
        "target_agent": target,
    }
    existing = load_completed(output, EVALUATION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or existing.get("evaluation") != expected:
            raise RuntimeError(f"completed Task4 evaluation drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned Task4 stats: {stats_path}")
    command = [
        sys.executable, "main.py", "play", "--agents", target, *spec["opponents"],
        "--train", "0", "--continue-without-training", "--no-gui", "--scenario", "classic",
        "--n-rounds", str(spec["rounds_per_case"]), "--seed", str(case["world_seed"]),
        "--save-stats", str(stats_path),
    ]
    overrides.update({
        "MODEL_A_CHECKPOINT_PATH" if label == "v4" else "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_SEED" if label == "v4" else "MODEL_A_V4C_SEED": str(case["agent_seed"]),
        "TASK4_RULE_SEED": str(case["opponent_seed"]),
    })
    record = {
        "schema_version": 1,
        "kind": EVALUATION_KIND,
        "status": "running",
        "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "stratum": stratum,
        "label": label,
        "evaluation": expected,
        "checkpoint": {"path": relative(checkpoint), "sha256": expected_hash},
        "command": command,
        "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(output, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    if completed.returncode == 0 and stats_path.is_file() and sha256_file(checkpoint) == expected_hash:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        record["metrics_by_agent"] = metrics_by_agent(stats)
        record["target_metrics"] = record["metrics_by_agent"].get(target)
        record["status"] = "completed" if record["target_metrics"] else "failed"
    else:
        record["status"] = "failed"
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task4 baseline evaluation failed: {stratum}/{label}/{case['world_seed']}")
    return record


def opponent_summary(runs: list[dict], target_name: str) -> dict:
    by_name: dict[str, dict[str, int]] = {}
    for run in runs:
        for name, metrics in run["metrics_by_agent"].items():
            if name == target_name:
                continue
            totals = by_name.setdefault(name, {"rounds": 0, "score": 0, "kills": 0, "suicides": 0})
            for key in totals:
                totals[key] += int(metrics[key])
    summaries = {
        name: {
            **totals,
            "score_per_round": totals["score"] / totals["rounds"],
            "kills_per_round": totals["kills"] / totals["rounds"],
            "suicides_per_round": totals["suicides"] / totals["rounds"],
        }
        for name, totals in by_name.items()
    }
    agent_rounds = sum(item["rounds"] for item in summaries.values())
    return {
        "by_agent": summaries,
        "opponent_count": len(summaries),
        "mean_score_per_agent_round": sum(item["score"] for item in summaries.values()) / agent_rounds,
        "mean_kills_per_agent_round": sum(item["kills"] for item in summaries.values()) / agent_rounds,
        "best_instance_score_per_round": max(item["score_per_round"] for item in summaries.values()),
    }


def run_evaluations(protocol_path: Path, protocol: dict, protocol_hash: str) -> tuple[dict, dict]:
    rows = {}
    manifests = {}
    for stratum, spec in protocol["evaluation"]["strata"].items():
        rows[stratum] = {}
        manifests[stratum] = {}
        for label in LABELS:
            target, _, _ = checkpoint_identity(protocol, label)
            runs = [evaluate_one(protocol_path, protocol, protocol_hash, stratum, spec, label, case) for case in spec["cases"]]
            rows[stratum][label] = {
                "target": aggregate(runs),
                "opponents": opponent_summary(runs, target),
            }
            manifests[stratum][label] = [
                relative(manifest_path(protocol, stratum, label, int(case["world_seed"])))
                for case in spec["cases"]
            ]
    return rows, manifests


def decide(protocol: dict, rows: dict) -> dict:
    rule = protocol["decision_rule"]
    candidate = {stratum: rows[stratum]["source-r2"] for stratum in STRATA}
    v4 = {stratum: rows[stratum]["v4"] for stratum in STRATA}
    per_stratum = {}
    for stratum in STRATA:
        per_stratum[stratum] = {
            "score_noninferior_to_v4": candidate[stratum]["target"]["score_per_round"] + 1e-12 >= v4[stratum]["target"]["score_per_round"] - float(rule["maximum_candidate_score_gap_to_v4"]),
            "kills_not_below_v4": candidate[stratum]["target"]["kills_per_round"] + 1e-12 >= v4[stratum]["target"]["kills_per_round"] + float(rule["minimum_candidate_kills_delta_to_v4"]),
            "suicide_safe_vs_v4": candidate[stratum]["target"]["suicides_per_round"] <= v4[stratum]["target"]["suicides_per_round"] + float(rule["maximum_candidate_suicide_increase_over_v4"]) + 1e-12,
        }
    duel_margin = candidate["task4_rule_duel"]["target"]["score_per_round"] - candidate["task4_rule_duel"]["opponents"]["mean_score_per_agent_round"]
    three_margin = candidate["task4_three_rule"]["target"]["score_per_round"] - candidate["task4_three_rule"]["opponents"]["mean_score_per_agent_round"]
    gates = {
        "duel_beats_rule": duel_margin + 1e-12 >= float(rule["minimum_duel_score_margin_over_rule"]),
        "three_rule_beats_rule_mean": three_margin + 1e-12 >= float(rule["minimum_three_rule_score_margin_over_rule_mean"]),
        "candidate_v4_score": all(item["score_noninferior_to_v4"] for item in per_stratum.values()),
        "candidate_v4_kills": all(item["kills_not_below_v4"] for item in per_stratum.values()),
        "candidate_v4_suicide": all(item["suicide_safe_vs_v4"] for item in per_stratum.values()),
    }
    competitive = all(gates.values())
    return {
        "decision": "source_r2_already_task4_competitive" if competitive else "task4_training_required",
        "already_competitive": competitive,
        "gates": gates,
        "per_stratum_gates": per_stratum,
        "source_r2_rule_score_margins": {
            "task4_rule_duel": duel_margin,
            "task4_three_rule": three_margin,
        },
        "source_r2_minus_v4": {
            stratum: {
                metric: candidate[stratum]["target"][metric] - v4[stratum]["target"][metric]
                for metric in ("score_per_round", "kills_per_round", "suicides_per_round", "invalid_actions_per_round")
            }
            for stratum in STRATA
        },
        "training_started": False,
        "automatic_followup_started": False,
        "task4_started": True,
        "task4_phase": "baseline_gate",
    }


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def write_report(protocol_path: Path, protocol: dict, protocol_hash: str, rows: dict, manifests: dict) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed Task4 baseline report protocol drift")
        return existing, sha256_file(path)
    report = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "rows": rows,
        "evaluation_manifests": manifests,
        "result": decide(protocol, rows),
        "awaiting_user_instruction": True,
        "source_artifacts_modified": False,
        "default_checkpoint_modified": False,
    }
    atomic_json(path, report)
    return report, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    rounds = len(LABELS) * sum(
        len(spec["cases"]) * int(spec["rounds_per_case"])
        for spec in protocol["evaluation"]["strata"].values()
    )
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "labels": list(LABELS),
        "strata": list(STRATA),
        "total_evaluation_rounds": rounds,
        "training_started": False,
        "automatic_followup_started": False,
        "task4_started": False,
        "formal_evaluation_started": False,
        "writes_terminal_report_and_stops": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol, protocol_hash, _, _ = load_protocol(protocol_path)
    if not args.execute:
        print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
        return 0
    existing = load_completed(report_path(protocol), REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed Task4 baseline report protocol drift")
        report, report_hash = existing, sha256_file(report_path(protocol))
    else:
        rows, manifests = run_evaluations(protocol_path.resolve(), protocol, protocol_hash)
        report, report_hash = write_report(protocol_path.resolve(), protocol, protocol_hash, rows, manifests)
    print(json.dumps({
        "status": report["status"],
        "report": relative(report_path(protocol)),
        "report_sha256": report_hash,
        **report["result"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
