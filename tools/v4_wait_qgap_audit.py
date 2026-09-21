"""Run and aggregate the read-only frozen-v4 WAIT Q-gap audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-wait-qgap-audit-s144000.json"
EXPECTED_CASES = tuple(
    {
        "case_id": f"p{index + 1}",
        "world_seed": 144001 + index,
        "agent_seed": 244001 + index,
        "opponent_seed": 344001 + index,
        "target_slot": index,
    }
    for index in range(4)
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_protocol(path: Path) -> tuple[dict, Path, str]:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    raw = resolved.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("kind") != "model-a-v4-wait-qgap-audit":
        raise ValueError("wrong WAIT audit protocol kind")
    if protocol.get("protocol_id") != "model-a-v4-wait-qgap-audit-s144000":
        raise ValueError("wrong WAIT audit protocol id")
    if protocol.get("training_allowed") is not False or protocol.get("policy_override_allowed") is not False:
        raise ValueError("WAIT audit must remain read-only")
    if tuple(protocol["evaluation"]["cases"]) != EXPECTED_CASES:
        raise ValueError("WAIT audit cases changed")
    if protocol["evaluation"].get("rounds_per_case") != 25 or protocol["evaluation"].get("total_rounds") != 100:
        raise ValueError("WAIT audit budget changed")
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    if not checkpoint.is_file() or sha256_file(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise ValueError("WAIT audit checkpoint mismatch")
    return protocol, resolved, hashlib.sha256(raw).hexdigest()


def case_paths(protocol: dict, case_id: str) -> tuple[Path, Path]:
    root = ROOT / protocol["output_directory"]
    return root / f"{case_id}.json", root / f"{case_id}.stats.json"


def execute_case(protocol: dict, protocol_path: Path, protocol_hash: str, case: dict) -> dict:
    diagnostic, stats = case_paths(protocol, case["case_id"])
    if diagnostic.exists() or stats.exists():
        raise RuntimeError(f"refusing existing WAIT audit case output: {case['case_id']}")
    roster = ["seeded_rule_based_agent"] * 4
    roster[int(case["target_slot"])] = "model_a_v4_wait_qgap"
    command = [
        sys.executable, "main.py", "play", "--agents", *roster,
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", "classic", "--n-rounds", "25", "--seed", str(case["world_seed"]),
        "--save-stats", str(stats),
    ]
    checkpoint = (ROOT / protocol["checkpoint"]["path"]).resolve()
    before = sha256_file(checkpoint)
    child = os.environ.copy()
    child.update({
        "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_SEED": str(case["agent_seed"]),
        "TASK4_RULE_SEED": str(case["opponent_seed"]),
        "MODEL_A_WAIT_AUDIT_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_WAIT_AUDIT_CASE_ID": case["case_id"],
        "MODEL_A_WAIT_AUDIT_OUTPUT": str(diagnostic.resolve()),
    })
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    if (
        completed.returncode != 0
        or not diagnostic.is_file()
        or not stats.is_file()
        or sha256_file(checkpoint) != before
    ):
        raise RuntimeError(f"WAIT audit case failed: {case['case_id']}")
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    if payload.get("policy_updates") != 0 or payload.get("actions_changed_by_audit") != 0:
        raise RuntimeError("WAIT audit changed the policy")
    payload["case"] = case
    payload["protocol_sha256"] = protocol_hash
    return payload


def summarize(protocol: dict, cases: list[dict]) -> dict:
    steps = sum(int(case["steps"]) for case in cases)
    action_counts = {
        action: sum(int(case["action_counts"][action]) for case in cases)
        for action in ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
    }
    waits = [record for case in cases for record in case["wait_records"]]
    discretionary = [record for record in waits if record["q_gap"] is not None]
    forced = len(waits) - len(discretionary)
    gaps = np.asarray([float(record["q_gap"]) for record in discretionary], dtype=np.float64)
    thresholds = [float(value) for value in protocol["diagnostic"]["counterfactual_thresholds"]]
    near = float(protocol["diagnostic"]["near_tie_threshold"])
    large = float(protocol["diagnostic"]["large_preference_threshold"])
    near_share = float(np.mean(gaps <= near)) if gaps.size else 0.0
    large_share = float(np.mean(gaps >= large)) if gaps.size else 0.0
    if near_share >= float(protocol["diagnostic"]["minimum_near_tie_share_for_action_selection_diagnosis"]):
        diagnosis = "near_tie_action_selection_issue"
    elif large_share >= float(protocol["diagnostic"]["minimum_large_gap_share_for_learned_preference_diagnosis"]):
        diagnosis = "learned_wait_q_preference"
    else:
        diagnosis = "mixed_wait_mechanisms"
    return {
        "steps": steps,
        "action_counts": action_counts,
        "wait_count": len(waits),
        "wait_fraction": len(waits) / steps if steps else 0.0,
        "forced_wait_count": forced,
        "forced_wait_share": forced / len(waits) if waits else 0.0,
        "discretionary_wait_count": len(discretionary),
        "repeat_wait_same_position_count": sum(bool(record["repeat_wait_same_position"]) for record in waits),
        "repeat_wait_same_position_share": (
            sum(bool(record["repeat_wait_same_position"]) for record in waits) / len(waits) if waits else 0.0
        ),
        "q_gap": {
            "minimum": float(gaps.min()) if gaps.size else None,
            "p10": float(np.quantile(gaps, 0.10)) if gaps.size else None,
            "p25": float(np.quantile(gaps, 0.25)) if gaps.size else None,
            "median": float(np.median(gaps)) if gaps.size else None,
            "p75": float(np.quantile(gaps, 0.75)) if gaps.size else None,
            "p90": float(np.quantile(gaps, 0.90)) if gaps.size else None,
            "maximum": float(gaps.max()) if gaps.size else None,
            "mean": float(gaps.mean()) if gaps.size else None,
        },
        "near_tie_threshold": near,
        "near_tie_share_of_discretionary_waits": near_share,
        "large_preference_threshold": large,
        "large_gap_share_of_discretionary_waits": large_share,
        "counterfactual_margin_override": {
            str(threshold): {
                "eligible_waits": int(np.sum(gaps <= threshold)),
                "projected_wait_fraction_if_all_eligible_were_moves": (
                    (len(waits) - int(np.sum(gaps <= threshold))) / steps if steps else 0.0
                ),
            }
            for threshold in thresholds
        },
        "diagnosis": diagnosis,
        "training_started": False,
        "actions_changed_by_audit": 0,
        "checkpoint_modified": False,
    }


def dry_run(protocol: dict) -> dict:
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "cases": len(protocol["evaluation"]["cases"]),
        "rounds_per_case": protocol["evaluation"]["rounds_per_case"],
        "total_rounds": protocol["evaluation"]["total_rounds"],
        "training_started": False,
        "policy_override_allowed": False,
        "formal_evaluation_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol, protocol_path, protocol_hash = load_protocol(args.protocol)
    if not args.execute:
        print(json.dumps({**dry_run(protocol), "protocol_sha256": protocol_hash}, indent=2, sort_keys=True))
        return 0
    report_path = ROOT / protocol["report_path"]
    if report_path.exists() or (ROOT / protocol["output_directory"]).exists():
        raise SystemExit("WAIT audit output already exists")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cases = [execute_case(protocol, protocol_path, protocol_hash, case) for case in protocol["evaluation"]["cases"]]
    report = {
        "schema_version": 1,
        "kind": "model-a-v4-wait-qgap-audit-report",
        "status": "completed",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_path": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_hash,
        "case_diagnostics": [
            {"path": str(case_paths(protocol, case["case_id"])[0].relative_to(ROOT)),
             "sha256": sha256_file(case_paths(protocol, case["case_id"])[0])}
            for case in protocol["evaluation"]["cases"]
        ],
        "summary": summarize(protocol, cases),
    }
    atomic_json(report_path, report)
    print(json.dumps({"report": str(report_path.relative_to(ROOT)), **report["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
