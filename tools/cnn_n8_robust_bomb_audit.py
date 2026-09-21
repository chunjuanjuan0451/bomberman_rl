"""Collect passive counterfactual robust-BOMB labels and official outcomes."""

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


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-robust-bomb-audit-s155000.json"
KIND_MANIFEST = "model-a-cnn-n8-robust-bomb-audit-case"
KIND_REPORT = "model-a-cnn-n8-robust-bomb-audit-report"


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
    if protocol.get("kind") != "model-a-cnn-n8-robust-bomb-audit":
        raise RuntimeError("wrong robust-bomb audit protocol")
    if protocol.get("training_allowed") or protocol.get("automatic_followup"):
        raise RuntimeError("robust-bomb audit must prohibit training and follow-up")
    rule = protocol["candidate_rule"]
    if (rule["name"] != "full_horizon_adversarial_opponent_reachability"
            or rule["action_override"] or rule["horizon_sweep"]):
        raise RuntimeError("robust-bomb candidate rule changed")
    collection = protocol["collection"]
    if (collection["rounds_per_case"] != 50 or collection["policy_updates"] != 0
            or collection["actions_overridden"] != 0 or len(collection["cases"]) != 4):
        raise RuntimeError("robust-bomb collection contract changed")
    seeds = _seed_values(collection["cases"])
    if len(seeds) != 16 or len(set(seeds)) != 16 or min(seeds) < 155000:
        raise RuntimeError("robust-bomb audit requires 16 unique 155xxx+ seeds")
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    if sha256_file(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("robust-bomb checkpoint drift")
    payload = _torch_load(checkpoint)
    expected = {"architecture": ARCHITECTURE, "stage": "task4", "replica": "r3", "stage_rounds": 800}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("robust-bomb checkpoint identity mismatch")
    return protocol, path, hashlib.sha256(raw).hexdigest()


def _overrides(protocol: dict, protocol_path: Path, label: str, case: dict) -> dict[str, str]:
    result = {
        "CNN_ROBUST_BOMB_AUDIT_PROTOCOL": str(protocol_path),
        "CNN_ROBUST_BOMB_AUDIT_CASE": label,
        "CNN_ROBUST_BOMB_AUDIT_AGENT_SEED": str(case["agent_seed"]),
        "CNN_ROBUST_BOMB_AUDIT_CHECKPOINT": str((ROOT / protocol["checkpoint"]["path"]).resolve()),
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
        "kind": "model-a-cnn-n8-robust-bomb-trace",
        "protocol_sha256": protocol_hash,
        "case": label,
        "checkpoint_sha256": protocol["checkpoint"]["sha256"],
        "rounds": protocol["collection"]["rounds_per_case"],
        "policy_updates": 0,
        "actions_overridden": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"invalid robust-bomb trace {label}: {key}")
    if payload.get("placement_count") != len(payload.get("placements", [])):
        raise RuntimeError(f"robust-bomb placement count mismatch: {label}")
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
        raise RuntimeError(f"orphaned robust-bomb manifest: {relative(manifest)}")
    if trace.exists() or stats.exists():
        raise RuntimeError(f"orphaned robust-bomb artifact: {label}")
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_cnn_n8_robust_bomb_audit",
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
            raise RuntimeError("robust-bomb collector process failed")
        if sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("robust-bomb audit mutated the checkpoint")
        trace_payload = _validate_trace(protocol, protocol_hash, label, trace)
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"]["model_a_cnn_n8_robust_bomb_audit"]
        record.update({
            "status": "completed",
            "checkpoint_sha256_after": checkpoint_hash,
            "trace_sha256": sha256_file(trace),
            "placement_count": trace_payload["placement_count"],
            "target_metrics": _metrics(raw),
        })
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"robust-bomb collection failed: {label}: {record.get('error')}")
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
            "stratum": manifest["case"]["stratum"],
            "metrics": manifest["target_metrics"],
            "placement_count": len(rows),
            "trace_sha256": manifest["trace_sha256"],
        }
    resolved = [row for row in placements if row["resolution"] == "exploded"]
    censored = [row for row in placements if row["resolution"] != "exploded"]
    selfkills = [row for row in resolved if row["selfkill"]]
    safe = [row for row in resolved if not row["selfkill"]]
    kills = [row for row in resolved if row["kills"] > 0]
    vetoed = [row for row in resolved if not row["robust_bomb_allowed"]]
    selfkill_vetoed = [row for row in selfkills if not row["robust_bomb_allowed"]]
    safe_vetoed = [row for row in safe if not row["robust_bomb_allowed"]]
    kill_vetoed = [row for row in kills if not row["robust_bomb_allowed"]]
    metrics = {
        "placements": len(placements),
        "resolved_placements": len(resolved),
        "censored_placements": len(censored),
        "selfkill_placements": len(selfkills),
        "safe_placements": len(safe),
        "kill_placements": len(kills),
        "vetoed_resolved_placements": len(vetoed),
        "vetoed_selfkill_placements": len(selfkill_vetoed),
        "vetoed_safe_placements": len(safe_vetoed),
        "vetoed_kill_placements": len(kill_vetoed),
        "overall_veto_fraction": _fraction(len(vetoed), len(resolved)),
        "selfkill_capture_fraction": _fraction(len(selfkill_vetoed), len(selfkills)),
        "safe_bomb_veto_fraction": _fraction(len(safe_vetoed), len(safe)),
        "kill_bomb_veto_fraction": _fraction(len(kill_vetoed), len(kills)),
    }
    gate = protocol["decision_rule"]
    checks = {
        "enough_resolved_placements": len(resolved) >= gate["minimum_resolved_placements"],
        "enough_selfkill_placements": len(selfkills) >= gate["minimum_selfkill_placements"],
        "selfkill_capture": metrics["selfkill_capture_fraction"] is not None
        and metrics["selfkill_capture_fraction"] >= gate["minimum_selfkill_capture_fraction"],
        "safe_bomb_preservation": metrics["safe_bomb_veto_fraction"] is not None
        and metrics["safe_bomb_veto_fraction"] <= gate["maximum_safe_bomb_veto_fraction"],
        "kill_bomb_preservation": metrics["kill_bomb_veto_fraction"] is not None
        and metrics["kill_bomb_veto_fraction"] <= gate["maximum_kill_bomb_veto_fraction"],
        "overall_bomb_preservation": metrics["overall_veto_fraction"] is not None
        and metrics["overall_veto_fraction"] <= gate["maximum_overall_veto_fraction"],
    }
    passed = all(checks.values())
    strata = {}
    for stratum in ("task4_rule", "task4_mixed"):
        rows = [item["target_metrics"] for item in manifests.values() if item["case"]["stratum"] == stratum]
        strata[stratum] = combine_metrics(rows)
    decision = "robust_bomb_signal_supported_stop_before_ab" if passed else "robust_bomb_signal_not_supported_stop"
    return {
        "schema_version": 1,
        "kind": KIND_REPORT,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "checkpoint": protocol["checkpoint"],
        "candidate_rule": protocol["candidate_rule"],
        "environment_rounds": 200,
        "stratum_metrics": strata,
        "combined_metrics": combine_metrics(list(strata.values())),
        "per_case": per_case,
        "counterfactual_metrics": metrics,
        "gate": {"checks": checks, "passed": passed, "thresholds": gate},
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
        raise RuntimeError(f"refusing to overwrite robust-bomb report: {relative(output)}")
    atomic_json(output, report)
    return report


def dry_run(protocol: dict, protocol_hash: str) -> dict:
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "checkpoint": protocol["checkpoint"],
        "candidate_rule": protocol["candidate_rule"],
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
