"""Run the selection-free frozen-CNN immediate-collision mask utility A/B."""

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


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-collision-mask-ab-s159000.json"
KIND_MANIFEST = "model-a-cnn-n8-collision-mask-ab-case"
KIND_REPORT = "model-a-cnn-n8-collision-mask-ab-report"


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
    if protocol.get("kind") != "model-a-cnn-n8-collision-mask-ab":
        raise RuntimeError("wrong collision-mask A/B protocol")
    if protocol.get("training_allowed") or protocol.get("automatic_followup"):
        raise RuntimeError("collision-mask A/B must prohibit training and automatic follow-up")
    collection = protocol["collection"]
    if collection["rounds_per_block"] != 25 or collection["policy_updates"] != 0:
        raise RuntimeError("collision-mask A/B collection contract changed")
    blocks = collection["blocks"]
    if len(blocks) != 8 or sum(b["stratum"] == "task4_rule" for b in blocks.values()) != 4:
        raise RuntimeError("collision-mask A/B requires four blocks per stratum")
    seeds = _seed_values(blocks)
    if len(seeds) != 32 or len(set(seeds)) != 32 or min(seeds) < 159000:
        raise RuntimeError("collision-mask A/B requires 32 unique 159xxx+ seeds")
    gate = protocol["decision_rule"]
    expected_gate = {
        "minimum_score_gain_per_round": 0.10,
        "minimum_winning_blocks": 5,
        "maximum_suicide_increase_per_round": 0.0,
        "maximum_kill_loss_per_round": 0.02,
        "maximum_mean_decision_time_ms": 500.0,
    }
    if gate != expected_gate:
        raise RuntimeError("collision-mask A/B gate changed")
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    if sha256_file(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("collision-mask A/B checkpoint drift")
    payload = _torch_load(checkpoint)
    expected = {"architecture": ARCHITECTURE, "stage": "task4", "replica": "r3", "stage_rounds": 800}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("collision-mask A/B checkpoint identity mismatch")
    return protocol, path, hashlib.sha256(raw).hexdigest()


def _overrides(protocol: dict, path: Path, label: str, arm: str, block: dict) -> dict[str, str]:
    result = {
        "CNN_COLLISION_AB_PROTOCOL": str(path),
        "CNN_COLLISION_AB_CASE": label,
        "CNN_COLLISION_AB_ARM": arm,
        "CNN_COLLISION_AB_AGENT_SEED": str(block["agent_seed"]),
        "CNN_COLLISION_AB_CHECKPOINT": str((ROOT / protocol["checkpoint"]["path"]).resolve()),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    if "rule_seed" in block:
        result["TASK4_RULE_SEED"] = str(block["rule_seed"])
    if "coin_seed" in block:
        result["TASK3_OPPONENT_SEED"] = str(block["coin_seed"])
    if "random_seed" in block:
        result["SEEDED_RANDOM_AGENT_SEED"] = str(block["random_seed"])
    return result


def _validate_trace(protocol: dict, protocol_hash: str, label: str, arm: str, path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "kind": "model-a-cnn-n8-collision-mask-ab-trace",
        "protocol_sha256": protocol_hash,
        "case": label, "arm": arm,
        "checkpoint_sha256": protocol["checkpoint"]["sha256"],
        "rounds": protocol["collection"]["rounds_per_block"], "policy_updates": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"invalid collision-mask A/B trace {label}/{arm}: {key}")
    counts = payload["counts"]
    if arm == "control" and any(counts[key] for key in (
            "restriction_available_decisions", "fallback_decisions", "actions_changed")):
        raise RuntimeError(f"control collision-mask trace contains interventions: {label}")
    if counts["actions_changed"] > counts["restriction_available_decisions"]:
        raise RuntimeError(f"collision-mask changed-action count impossible: {label}/{arm}")
    return payload


def collect(protocol: dict, path: Path, protocol_hash: str, label: str, arm: str, block: dict) -> dict:
    stem = f"{label}-{arm}"
    manifest = ROOT / protocol["manifest_directory"] / f"{stem}.json"
    trace = ROOT / protocol["trace_directory"] / f"{stem}.json"
    stats = manifest.with_suffix(".stats.json")
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol_sha256") == protocol_hash:
            _validate_trace(protocol, protocol_hash, label, arm, trace)
            return existing
        raise RuntimeError(f"orphaned collision-mask A/B manifest: {relative(manifest)}")
    if trace.exists() or stats.exists():
        raise RuntimeError(f"orphaned collision-mask A/B artifact: {stem}")
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_cnn_n8_collision_mask_ab",
        *block["opponents"], "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", "classic", "--n-rounds", str(protocol["collection"]["rounds_per_block"]),
        "--seed", str(block["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = _overrides(protocol, path, label, arm, block)
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    checkpoint_hash = sha256_file(checkpoint)
    record = {
        "schema_version": 1, "kind": KIND_MANIFEST,
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
        "case_label": label, "arm": arm, "block": block, "status": "running",
        "started_at_utc": utc_now(), "command": command, "environment_overrides": overrides,
        "checkpoint_sha256_before": checkpoint_hash, "trace_path": relative(trace),
        "raw_stats": relative(stats),
    }
    atomic_json(manifest, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not trace.is_file() or not stats.is_file():
            raise RuntimeError("collision-mask A/B process failed")
        if sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("collision-mask A/B mutated checkpoint")
        trace_payload = _validate_trace(protocol, protocol_hash, label, arm, trace)
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"]["model_a_cnn_n8_collision_mask_ab"]
        record.update({
            "status": "completed", "checkpoint_sha256_after": checkpoint_hash,
            "trace_sha256": sha256_file(trace), "trace_counts": trace_payload["counts"],
            "target_metrics": _metrics(raw),
        })
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"collision-mask A/B failed: {stem}: {record.get('error')}")
    return record


def _sum_counts(items: list[dict]) -> dict:
    keys = items[0]["trace_counts"]
    return {key: sum(item["trace_counts"][key] for item in items) for key in keys}


def summarize(protocol: dict, protocol_hash: str, manifests: dict[str, dict[str, dict]]) -> dict:
    arm_items = {arm: [manifests[label][arm] for label in manifests] for arm in ("control", "treatment")}
    pooled = {arm: combine_metrics([item["target_metrics"] for item in items]) for arm, items in arm_items.items()}
    strata = {}
    for stratum in ("task4_rule", "task4_mixed"):
        strata[stratum] = {}
        for arm in ("control", "treatment"):
            selected = [m[arm]["target_metrics"] for label, m in manifests.items()
                        if protocol["collection"]["blocks"][label]["stratum"] == stratum]
            strata[stratum][arm] = combine_metrics(selected)
    paired = {}
    wins = 0
    for label, arms in manifests.items():
        control = arms["control"]["target_metrics"]
        treatment = arms["treatment"]["target_metrics"]
        delta = treatment["score_per_round"] - control["score_per_round"]
        wins += delta > 0
        paired[label] = {
            "stratum": protocol["collection"]["blocks"][label]["stratum"],
            "control": control, "treatment": treatment, "score_delta_per_round": delta,
        }
    delta = {
        key: pooled["treatment"][key] - pooled["control"][key]
        for key in ("score_per_round", "coins_per_round", "kills_per_round",
                    "suicides_per_round", "invalid_actions_per_round", "wait_fraction",
                    "mean_decision_time_ms")
    }
    gate = protocol["decision_rule"]
    checks = {
        "score_gain": delta["score_per_round"] >= gate["minimum_score_gain_per_round"],
        "block_support": wins >= gate["minimum_winning_blocks"],
        "suicide_nonincrease": delta["suicides_per_round"] <= gate["maximum_suicide_increase_per_round"],
        "kill_preservation": delta["kills_per_round"] >= -gate["maximum_kill_loss_per_round"],
        "latency": pooled["treatment"]["mean_decision_time_ms"] <= gate["maximum_mean_decision_time_ms"],
    }
    passed = all(checks.values())
    treatment_counts = _sum_counts(arm_items["treatment"])
    decision = "collision_mask_supported_freeze_candidate" if passed else "collision_mask_not_supported_retain_original"
    return {
        "schema_version": 1, "kind": KIND_REPORT, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash, "status": "completed", "completed_at_utc": utc_now(),
        "checkpoint": protocol["checkpoint"], "rounds_per_arm": 200,
        "pooled_metrics": pooled, "stratum_metrics": strata, "paired_blocks": paired,
        "paired_winning_blocks": wins, "treatment_minus_control": delta,
        "treatment_trace_counts": treatment_counts,
        "gate": {"checks": checks, "passed": passed, "thresholds": gate},
        "result": {"decision": decision, "training_started": False,
                   "checkpoint_modified": False, "automatic_followup_started": False,
                   "candidate_deployed": False},
        "awaiting_user_instruction": True,
    }


def execute(protocol: dict, path: Path, protocol_hash: str) -> dict:
    manifests = {}
    for label, block in protocol["collection"]["blocks"].items():
        manifests[label] = {
            arm: collect(protocol, path, protocol_hash, label, arm, block)
            for arm in ("control", "treatment")
        }
    report = summarize(protocol, protocol_hash, manifests)
    output = ROOT / protocol["report_path"]
    if output.exists():
        raise RuntimeError(f"refusing to overwrite collision-mask A/B report: {relative(output)}")
    atomic_json(output, report)
    return report


def dry_run(protocol: dict, protocol_hash: str) -> dict:
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash, "checkpoint": protocol["checkpoint"],
        "blocks": len(protocol["collection"]["blocks"]), "rounds_per_arm": 200,
        "training_started": False, "automatic_followup_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol, path, protocol_hash = load_protocol(args.protocol)
    result = execute(protocol, path, protocol_hash) if args.execute else dry_run(protocol, protocol_hash)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
