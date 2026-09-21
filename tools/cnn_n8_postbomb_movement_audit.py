"""Collect passive post-bomb robust-movement labels and official outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_cnn_n8.network import ARCHITECTURE  # noqa: E402
from tools.cnn_n8_task1 import _metrics, _torch_load, atomic_json, relative, sha256_file, utc_now  # noqa: E402
from tools.cnn_n8_task4 import combine_metrics  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-postbomb-movement-audit-s157000.json"
KIND_MANIFEST = "model-a-cnn-n8-postbomb-movement-audit-case"
KIND_REPORT = "model-a-cnn-n8-postbomb-movement-audit-report"


def _seed_values(node) -> list[int]:
    values = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("_seed") and isinstance(value, int):
                values.append(value)
            else:
                values.extend(_seed_values(value))
    elif isinstance(node, list):
        for value in node:
            values.extend(_seed_values(value))
    return values


def load_protocol(path: Path) -> tuple[dict, Path, str]:
    path = path.resolve()
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("kind") != "model-a-cnn-n8-postbomb-movement-audit":
        raise RuntimeError("wrong post-bomb movement audit protocol")
    if protocol.get("training_allowed") or protocol.get("automatic_followup"):
        raise RuntimeError("post-bomb movement audit must prohibit training and follow-up")
    rule = protocol["candidate_rule"]
    allowed_rules = {
        "full_horizon_postbomb_movement_reachability",
        "immediate_opponent_collision_then_known_hazard_survival",
    }
    if (rule["name"] not in allowed_rules or rule["action_override"]
            or rule["horizon_sweep"] or rule["threshold_sweep"]):
        raise RuntimeError("post-bomb movement candidate rule changed")
    collection = protocol["collection"]
    if (collection["rounds_per_case"] != 50 or collection["policy_updates"] != 0
            or collection["actions_overridden"] != 0 or len(collection["cases"]) != 4):
        raise RuntimeError("post-bomb movement collection contract changed")
    seeds = _seed_values(collection["cases"])
    if len(seeds) != 16 or len(set(seeds)) != 16 or min(seeds) < 157000:
        raise RuntimeError("post-bomb movement audit requires 16 unique 157xxx+ seeds")
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    if sha256_file(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("post-bomb movement checkpoint drift")
    payload = _torch_load(checkpoint)
    expected = {"architecture": ARCHITECTURE, "stage": "task4", "replica": "r3", "stage_rounds": 800}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("post-bomb movement checkpoint identity mismatch")
    return protocol, path, hashlib.sha256(raw).hexdigest()


def _overrides(protocol: dict, protocol_path: Path, label: str, case: dict) -> dict[str, str]:
    result = {
        "CNN_POSTBOMB_MOVEMENT_AUDIT_PROTOCOL": str(protocol_path),
        "CNN_POSTBOMB_MOVEMENT_AUDIT_CASE": label,
        "CNN_POSTBOMB_MOVEMENT_AUDIT_AGENT_SEED": str(case["agent_seed"]),
        "CNN_POSTBOMB_MOVEMENT_AUDIT_CHECKPOINT": str((ROOT / protocol["checkpoint"]["path"]).resolve()),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if "rule_seed" in case:
        result["TASK4_RULE_SEED"] = str(case["rule_seed"])
    if "coin_seed" in case:
        result["TASK3_OPPONENT_SEED"] = str(case["coin_seed"])
    if "random_seed" in case:
        result["SEEDED_RANDOM_AGENT_SEED"] = str(case["random_seed"])
    return result


def _validate_trace(protocol: dict, protocol_hash: str, label: str, path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "kind": "model-a-cnn-n8-postbomb-movement-trace",
        "protocol_sha256": protocol_hash,
        "case": label,
        "checkpoint_sha256": protocol["checkpoint"]["sha256"],
        "rounds": protocol["collection"]["rounds_per_case"],
        "policy_updates": 0,
        "actions_overridden": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"invalid post-bomb movement trace {label}: {key}")
    if payload.get("placement_count") != len(payload.get("placements", [])):
        raise RuntimeError(f"post-bomb movement placement count mismatch: {label}")
    for row in payload["placements"]:
        if row["audited_decision_count"] != len(row["decisions"]):
            raise RuntimeError(f"post-bomb movement decision count mismatch: {label}")
        for decision in row["decisions"]:
            diagnostic = decision["diagnostic"]
            if diagnostic["robust_action_count"] != len(diagnostic["robust_actions"]):
                raise RuntimeError(f"post-bomb movement robust count mismatch: {label}")
            if diagnostic["robust_action_count"] > 5:
                raise RuntimeError(f"impossible post-bomb robust action count: {label}")
    return payload


def collect_case(protocol: dict, protocol_path: Path, protocol_hash: str,
                 label: str, case: dict) -> dict:
    manifest = ROOT / protocol["manifest_directory"] / f"{label}.json"
    trace = ROOT / protocol["trace_directory"] / f"{label}.json"
    stats = manifest.with_suffix(".stats.json")
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol_sha256") == protocol_hash:
            _validate_trace(protocol, protocol_hash, label, trace)
            return existing
        raise RuntimeError(f"orphaned post-bomb movement manifest: {relative(manifest)}")
    if trace.exists() or stats.exists():
        raise RuntimeError(f"orphaned post-bomb movement artifact: {label}")
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_cnn_n8_postbomb_movement_audit",
        *case["opponents"], "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", "classic", "--n-rounds", str(protocol["collection"]["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = _overrides(protocol, protocol_path, label, case)
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    checkpoint_hash = sha256_file(checkpoint)
    record = {
        "schema_version": 1, "kind": KIND_MANIFEST,
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
        "case_label": label, "case": case, "status": "running",
        "started_at_utc": utc_now(), "command": command,
        "environment_overrides": overrides, "checkpoint_sha256_before": checkpoint_hash,
        "trace_path": relative(trace), "raw_stats": relative(stats),
    }
    atomic_json(manifest, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not trace.is_file() or not stats.is_file():
            raise RuntimeError("post-bomb movement collector process failed")
        if sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("post-bomb movement audit mutated checkpoint")
        trace_payload = _validate_trace(protocol, protocol_hash, label, trace)
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"]["model_a_cnn_n8_postbomb_movement_audit"]
        target_metrics = _metrics(raw)
        rows = trace_payload["placements"]
        if (len(rows) != target_metrics["bombs"]
                or sum(row["selfkill"] for row in rows) != target_metrics["suicides"]
                or sum(row["kills"] for row in rows) != target_metrics["kills"]):
            raise RuntimeError("post-bomb movement trace/outcome accounting mismatch")
        record.update({
            "status": "completed", "checkpoint_sha256_after": checkpoint_hash,
            "trace_sha256": sha256_file(trace), "placement_count": len(rows),
            "target_metrics": target_metrics,
        })
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"post-bomb movement collection failed: {label}: {record.get('error')}")
    return record


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def summarize(protocol: dict, protocol_hash: str, manifests: dict[str, dict]) -> dict:
    placements = []
    per_case = {}
    for label, manifest in manifests.items():
        trace = _validate_trace(protocol, protocol_hash, label, ROOT / manifest["trace_path"])
        rows = trace["placements"]
        placements.extend({**row, "case": label} for row in rows)
        per_case[label] = {
            "stratum": manifest["case"]["stratum"], "metrics": manifest["target_metrics"],
            "placement_count": len(rows), "trace_sha256": manifest["trace_sha256"],
        }
    resolved = [row for row in placements if row["resolution"] == "exploded"]
    censored = [row for row in placements if row["resolution"] != "exploded"]
    selfkills = [row for row in resolved if row["selfkill"]]
    safe = [row for row in resolved if not row["selfkill"]]
    kills = [row for row in resolved if row["kills"] > 0]
    flagged = [row for row in resolved if row["has_avoidable_deviation"]]
    selfkill_flagged = [row for row in selfkills if row["has_avoidable_deviation"]]
    safe_flagged = [row for row in safe if row["has_avoidable_deviation"]]
    kill_flagged = [row for row in kills if row["has_avoidable_deviation"]]
    audited_decisions = sum(row["audited_decision_count"] for row in resolved)
    deviations = sum(row["avoidable_deviation_count"] for row in resolved)
    no_robust = sum(row["no_robust_action_count"] for row in resolved)
    metrics = {
        "placements": len(placements), "resolved_placements": len(resolved),
        "censored_placements": len(censored), "selfkill_placements": len(selfkills),
        "safe_placements": len(safe), "kill_placements": len(kills),
        "audited_postbomb_decisions": audited_decisions,
        "avoidable_deviation_decisions": deviations,
        "no_robust_action_decisions": no_robust,
        "flagged_resolved_placements": len(flagged),
        "flagged_selfkill_placements": len(selfkill_flagged),
        "flagged_safe_placements": len(safe_flagged),
        "flagged_kill_placements": len(kill_flagged),
        "decision_intervention_fraction": _fraction(deviations, audited_decisions),
        "overall_placement_intervention_fraction": _fraction(len(flagged), len(resolved)),
        "selfkill_capture_fraction": _fraction(len(selfkill_flagged), len(selfkills)),
        "safe_placement_intervention_fraction": _fraction(len(safe_flagged), len(safe)),
        "kill_placement_intervention_fraction": _fraction(len(kill_flagged), len(kills)),
    }
    gate = protocol["decision_rule"]
    checks = {
        "enough_resolved_placements": len(resolved) >= gate["minimum_resolved_placements"],
        "enough_selfkill_placements": len(selfkills) >= gate["minimum_selfkill_placements"],
        "selfkill_capture": metrics["selfkill_capture_fraction"] is not None
        and metrics["selfkill_capture_fraction"] >= gate["minimum_selfkill_capture_fraction"],
        "safe_placement_preservation": metrics["safe_placement_intervention_fraction"] is not None
        and metrics["safe_placement_intervention_fraction"] <= gate["maximum_safe_placement_intervention_fraction"],
        "kill_placement_preservation": metrics["kill_placement_intervention_fraction"] is not None
        and metrics["kill_placement_intervention_fraction"] <= gate["maximum_kill_placement_intervention_fraction"],
        "overall_placement_preservation": metrics["overall_placement_intervention_fraction"] is not None
        and metrics["overall_placement_intervention_fraction"] <= gate["maximum_overall_placement_intervention_fraction"],
    }
    passed = all(checks.values())
    strata = {}
    for stratum in ("task4_rule", "task4_mixed"):
        rows = [item["target_metrics"] for item in manifests.values() if item["case"]["stratum"] == stratum]
        strata[stratum] = combine_metrics(rows)
    decision = ("postbomb_movement_signal_supported_stop_before_ab" if passed
                else "postbomb_movement_signal_not_supported_stop")
    return {
        "schema_version": 1, "kind": KIND_REPORT, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash, "status": "completed", "completed_at_utc": utc_now(),
        "checkpoint": protocol["checkpoint"], "candidate_rule": protocol["candidate_rule"],
        "environment_rounds": 200, "stratum_metrics": strata,
        "combined_metrics": combine_metrics(list(strata.values())), "per_case": per_case,
        "counterfactual_metrics": metrics,
        "gate": {"checks": checks, "passed": passed, "thresholds": gate},
        "result": {"decision": decision, "training_started": False, "actions_overridden": False,
                   "checkpoint_modified": False, "automatic_followup_started": False},
        "awaiting_user_instruction": True,
    }


def execute(protocol: dict, protocol_path: Path, protocol_hash: str) -> dict:
    manifests = {label: collect_case(protocol, protocol_path, protocol_hash, label, case)
                 for label, case in protocol["collection"]["cases"].items()}
    report = summarize(protocol, protocol_hash, manifests)
    output = ROOT / protocol["report_path"]
    if output.exists():
        raise RuntimeError(f"refusing to overwrite post-bomb movement report: {relative(output)}")
    atomic_json(output, report)
    return report


def dry_run(protocol: dict, protocol_hash: str) -> dict:
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash, "checkpoint": protocol["checkpoint"],
        "candidate_rule": protocol["candidate_rule"], "environment_rounds": 200,
        "training_started": False, "actions_overridden": False, "automatic_followup_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol, protocol_path, protocol_hash = load_protocol(args.protocol)
    result = execute(protocol, protocol_path, protocol_hash) if args.execute else dry_run(protocol, protocol_hash)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
