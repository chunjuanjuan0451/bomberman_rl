"""Run and summarize a passive self-kill audit of the frozen CNN candidate."""

from __future__ import annotations

import argparse
from collections import Counter
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


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-selfkill-audit-s154000.json"
KIND_MANIFEST = "model-a-cnn-n8-selfkill-audit-case"
KIND_REPORT = "model-a-cnn-n8-selfkill-audit-report"
DYNAMIC_CATEGORIES = {
    "fragile_contestable_bomb",
    "route_became_unavoidable",
    "mask_predicted_safe_but_died",
}


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
    if protocol.get("kind") != "model-a-cnn-n8-selfkill-audit":
        raise RuntimeError("wrong CNN self-kill audit protocol")
    if protocol.get("training_allowed") or protocol.get("automatic_followup"):
        raise RuntimeError("self-kill audit must prohibit training and follow-up")
    if protocol["collection"] != {**protocol["collection"], "rounds_per_case": 50,
                                   "trace_window_steps": 8, "policy_updates": 0,
                                   "actions_overridden": 0}:
        raise RuntimeError("self-kill collection contract changed")
    cases = protocol["collection"]["cases"]
    if len(cases) != 4 or sum(int(item["stratum"] == "task4_rule") for item in cases.values()) != 2:
        raise RuntimeError("self-kill audit must contain two cases per Task4 stratum")
    seeds = _seed_values(cases)
    if len(seeds) != 16 or len(set(seeds)) != 16 or min(seeds) < 154000:
        raise RuntimeError("self-kill audit requires 16 unique 154xxx+ seed values")
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    if sha256_file(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("self-kill audit checkpoint drift")
    payload = _torch_load(checkpoint)
    expected = {"architecture": ARCHITECTURE, "stage": "task4", "replica": "r3", "stage_rounds": 800}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("self-kill audit checkpoint identity mismatch")
    return protocol, path, hashlib.sha256(raw).hexdigest()


def _overrides(protocol: dict, protocol_path: Path, label: str, case: dict) -> dict[str, str]:
    result = {
        "CNN_SELFKILL_AUDIT_PROTOCOL": str(protocol_path),
        "CNN_SELFKILL_AUDIT_CASE": label,
        "CNN_SELFKILL_AUDIT_AGENT_SEED": str(case["agent_seed"]),
        "CNN_SELFKILL_AUDIT_CHECKPOINT": str((ROOT / protocol["checkpoint"]["path"]).resolve()),
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
        "kind": "model-a-cnn-n8-selfkill-trace",
        "protocol_sha256": protocol_hash,
        "case": label,
        "checkpoint_sha256": protocol["checkpoint"]["sha256"],
        "rounds": protocol["collection"]["rounds_per_case"],
        "policy_updates": 0,
        "actions_overridden": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"invalid self-kill trace {label}: {key}")
    if payload.get("selfkill_count") != len(payload.get("selfkill_cases", [])):
        raise RuntimeError(f"self-kill trace count mismatch: {label}")
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
        raise RuntimeError(f"orphaned self-kill manifest: {relative(manifest)}")
    if trace.exists() or stats.exists():
        raise RuntimeError(f"orphaned self-kill artifact: {label}")
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_cnn_n8_selfkill_audit",
        *case["opponents"], "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", "classic", "--n-rounds", str(protocol["collection"]["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = _overrides(protocol, protocol_path, label, case)
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    checkpoint_hash = sha256_file(checkpoint)
    record = {
        "schema_version": 1,
        "kind": KIND_MANIFEST,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "case_label": label,
        "case": case,
        "status": "running",
        "started_at_utc": utc_now(),
        "command": command,
        "environment_overrides": overrides,
        "checkpoint_sha256_before": checkpoint_hash,
        "trace_path": relative(trace),
        "raw_stats": relative(stats),
    }
    atomic_json(manifest, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not trace.is_file() or not stats.is_file():
            raise RuntimeError("self-kill collector process failed")
        if sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("self-kill audit mutated the checkpoint")
        trace_payload = _validate_trace(protocol, protocol_hash, label, trace)
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"]["model_a_cnn_n8_selfkill_audit"]
        record.update({
            "status": "completed",
            "checkpoint_sha256_after": checkpoint_hash,
            "trace_sha256": sha256_file(trace),
            "selfkill_count": trace_payload["selfkill_count"],
            "target_metrics": _metrics(raw),
        })
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"self-kill collection failed: {label}: {record.get('error')}")
    return record


def classify_case(case: dict, recent_window: int) -> dict:
    rows = case["window"]
    terminal = rows[-1]
    placements = [
        row for row in rows
        if row["action"] == "BOMB" and "BOMB_DROPPED" in row["events"]
        and terminal["step"] - row["step"] <= recent_window
    ]
    placement = placements[-1] if placements else None
    flags = {
        "recent_bomb_found": placement is not None,
        "terminal_fallback_used": bool(terminal["fallback_used"]),
        "terminal_mask_predicted_survival": terminal["known_survivable_nonbomb_actions"] > 0,
        "route_became_unavoidable": False,
    }
    if placement is None:
        category = "missing_recent_own_bomb"
    else:
        placement_index = rows.index(placement)
        flags["route_became_unavoidable"] = any(
            row["fallback_used"] for row in rows[placement_index + 1:]
        )
        first_count = int(placement["bomb_safe_first_move_count"])
        contestable = bool(placement["bomb_any_first_move_contestable"])
        flags.update({
            "placement_first_move_count": first_count,
            "placement_contestable": contestable,
            "placement_q_gap": placement["chosen_q_gap"],
        })
        if first_count == 1 and contestable:
            category = "fragile_contestable_bomb"
        elif first_count == 1:
            category = "fragile_single_exit_bomb"
        elif flags["route_became_unavoidable"]:
            category = "route_became_unavoidable"
        elif flags["terminal_mask_predicted_survival"]:
            category = "mask_predicted_safe_but_died"
        else:
            category = "other"
    return {
        "case": case.get("case"),
        "round": case["round"],
        "terminal_step": case["terminal_step"],
        "category": category,
        "flags": flags,
        "window": rows,
    }


def summarize(protocol: dict, protocol_hash: str, manifests: dict[str, dict]) -> dict:
    classified = []
    per_case = {}
    for label, manifest in manifests.items():
        trace_path = ROOT / manifest["trace_path"]
        trace = _validate_trace(protocol, protocol_hash, label, trace_path)
        items = []
        for item in trace["selfkill_cases"]:
            item = dict(item)
            item["case"] = label
            result = classify_case(item, int(protocol["classification"]["recent_bomb_window_steps"]))
            items.append(result)
            classified.append(result)
        per_case[label] = {
            "stratum": manifest["case"]["stratum"],
            "metrics": manifest["target_metrics"],
            "selfkill_count": len(items),
            "category_counts": dict(Counter(item["category"] for item in items)),
            "trace_sha256": manifest["trace_sha256"],
        }
    counts = Counter(item["category"] for item in classified)
    total = len(classified)
    dynamic = sum(counts[name] for name in DYNAMIC_CATEGORIES)
    minimum = int(protocol["classification"]["minimum_selfkill_cases"])
    threshold = float(protocol["classification"]["dynamic_gap_fraction"])
    if total < minimum:
        decision = "insufficient_selfkill_cases_inconclusive"
    elif dynamic / total >= threshold:
        decision = "dynamic_safety_gap_supported"
    else:
        decision = "dynamic_safety_gap_not_dominant"
    strata = {}
    for stratum in ("task4_rule", "task4_mixed"):
        rows = [item["target_metrics"] for item in manifests.values() if item["case"]["stratum"] == stratum]
        strata[stratum] = combine_metrics(rows)
    return {
        "schema_version": 1,
        "kind": KIND_REPORT,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "checkpoint": protocol["checkpoint"],
        "environment_rounds": sum(item["target_metrics"]["rounds"] for item in manifests.values()),
        "stratum_metrics": strata,
        "combined_metrics": combine_metrics(list(strata.values())),
        "per_case": per_case,
        "classification": {
            "total_selfkills": total,
            "category_counts": dict(counts),
            "dynamic_categories": sorted(DYNAMIC_CATEGORIES),
            "dynamic_case_count": dynamic,
            "dynamic_case_fraction": 0.0 if not total else dynamic / total,
            "minimum_selfkill_cases": minimum,
            "dynamic_gap_fraction_threshold": threshold,
            "decision": decision,
        },
        "classified_selfkills": classified,
        "result": {
            "decision": decision,
            "training_started": False,
            "actions_overridden": False,
            "checkpoint_modified": False,
            "automatic_followup_started": False,
        },
        "awaiting_user_instruction": True,
    }


def execute(protocol: dict, protocol_path: Path, protocol_hash: str) -> dict:
    manifests = {
        label: collect_case(protocol, protocol_path, protocol_hash, label, case)
        for label, case in protocol["collection"]["cases"].items()
    }
    report = summarize(protocol, protocol_hash, manifests)
    output = ROOT / protocol["report_path"]
    if output.exists():
        raise RuntimeError(f"refusing to overwrite self-kill report: {relative(output)}")
    atomic_json(output, report)
    return report


def dry_run(protocol: dict, protocol_hash: str) -> dict:
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "checkpoint": protocol["checkpoint"],
        "cases": protocol["collection"]["cases"],
        "environment_rounds": 200,
        "training_started": False,
        "actions_overridden": False,
        "automatic_followup_started": False,
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
