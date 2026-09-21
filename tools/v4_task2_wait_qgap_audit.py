"""Read-only Task2 WAIT Q-gap audit, stratified by bomb state."""

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
DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task2-wait-qgap-audit-s145000.json"
EXPECTED_CASES = tuple(
    {"case_id": f"t2{letter}", "world_seed": 145001 + index, "agent_seed": 245001 + index}
    for index, letter in enumerate("abcd")
)
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


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
        raise ValueError("wrong Task2 WAIT audit kind")
    if protocol.get("protocol_id") != "model-a-v4-task2-wait-qgap-audit-s145000":
        raise ValueError("wrong Task2 WAIT audit protocol id")
    if any(protocol.get(key) is not False for key in (
        "training_allowed", "checkpoint_copy_allowed", "policy_override_allowed",
    )):
        raise ValueError("Task2 WAIT audit must remain read-only")
    evaluation = protocol["evaluation"]
    if (
        evaluation.get("scenario") != "classic"
        or evaluation.get("opponents") != []
        or tuple(evaluation.get("cases", ())) != EXPECTED_CASES
        or evaluation.get("rounds_per_case") != 25
        or evaluation.get("total_rounds") != 100
    ):
        raise ValueError("Task2 WAIT audit evaluation contract changed")
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    if not checkpoint.is_file() or sha256_file(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise ValueError("Task2 WAIT audit checkpoint mismatch")
    return protocol, resolved, hashlib.sha256(raw).hexdigest()


def case_paths(protocol: dict, case_id: str) -> tuple[Path, Path]:
    root = ROOT / protocol["output_directory"]
    return root / f"{case_id}.json", root / f"{case_id}.stats.json"


def execute_case(protocol: dict, protocol_path: Path, protocol_hash: str, case: dict) -> dict:
    diagnostic, stats = case_paths(protocol, case["case_id"])
    if diagnostic.exists() or stats.exists():
        raise RuntimeError(f"refusing existing Task2 WAIT audit output: {case['case_id']}")
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_wait_qgap",
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
        raise RuntimeError(f"Task2 WAIT audit case failed: {case['case_id']}")
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    if payload.get("policy_updates") != 0 or payload.get("actions_changed_by_audit") != 0:
        raise RuntimeError("Task2 WAIT audit changed the policy")
    payload["case"] = case
    payload["protocol_sha256"] = protocol_hash
    return payload


def _stratum(records: list[dict], wait_total: int) -> dict:
    discretionary = [record for record in records if record["q_gap"] is not None]
    gaps = np.asarray([float(record["q_gap"]) for record in discretionary], dtype=np.float64)
    return {
        "wait_count": len(records),
        "share_of_all_waits": len(records) / wait_total if wait_total else 0.0,
        "forced_wait_count": len(records) - len(discretionary),
        "discretionary_wait_count": len(discretionary),
        "repeat_wait_same_position_share": (
            sum(bool(record["repeat_wait_same_position"]) for record in records) / len(records)
            if records else 0.0
        ),
        "q_gap_mean": float(gaps.mean()) if gaps.size else None,
        "q_gap_median": float(np.median(gaps)) if gaps.size else None,
        "q_gap_p25": float(np.quantile(gaps, 0.25)) if gaps.size else None,
        "q_gap_p75": float(np.quantile(gaps, 0.75)) if gaps.size else None,
        "near_tie_share": float(np.mean(gaps <= 0.05)) if gaps.size else None,
        "large_gap_share": float(np.mean(gaps >= 0.2)) if gaps.size else None,
    }


def summarize(cases: list[dict]) -> dict:
    steps = sum(int(case["steps"]) for case in cases)
    actions = {
        action: sum(int(case["action_counts"][action]) for case in cases)
        for action in ACTIONS
    }
    waits = [record for case in cases for record in case["wait_records"]]
    with_bomb = [record for record in waits if record["bombs_present"] > 0]
    without_bomb = [record for record in waits if record["bombs_present"] == 0]
    idle_candidates = [
        record for record in without_bomb
        if record["q_gap"] is not None and record["current_danger_time"] == 99
    ]
    early = [record for record in waits if record["step"] <= 100]
    middle = [record for record in waits if 100 < record["step"] <= 250]
    late = [record for record in waits if record["step"] > 250]
    result = {
        "steps": steps,
        "action_counts": actions,
        "wait_count": len(waits),
        "wait_fraction": len(waits) / steps if steps else 0.0,
        "all_waits": _stratum(waits, len(waits)),
        "bomb_present_waits": _stratum(with_bomb, len(waits)),
        "no_bomb_waits": _stratum(without_bomb, len(waits)),
        "idle_wait_candidates": _stratum(idle_candidates, len(waits)),
        "early_steps_1_100": _stratum(early, len(waits)),
        "middle_steps_101_250": _stratum(middle, len(waits)),
        "late_steps_251_plus": _stratum(late, len(waits)),
        "idle_wait_candidate_fraction_of_all_steps": len(idle_candidates) / steps if steps else 0.0,
        "training_started": False,
        "actions_changed_by_audit": 0,
        "checkpoint_modified": False,
    }
    idle = result["idle_wait_candidates"]
    bomb = result["bomb_present_waits"]
    if idle["share_of_all_waits"] >= 0.25 and idle["large_gap_share"] >= 0.5:
        recommendation = "test_conditional_idle_wait_penalty_from_task2"
    elif bomb["share_of_all_waits"] >= 0.75:
        recommendation = "do_not_penalize_task2_wait_bomb_timing_dominates"
    else:
        recommendation = "task2_wait_mechanism_mixed_no_training_yet"
    result["recommendation"] = recommendation
    return result


def dry_run(protocol: dict) -> dict:
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"],
        "scenario": "classic", "opponents": [], "cases": 4,
        "rounds_per_case": 25, "total_rounds": 100,
        "training_started": False, "policy_override_allowed": False,
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
    output_root = ROOT / protocol["output_directory"]
    report_path = ROOT / protocol["report_path"]
    if output_root.exists() or report_path.exists():
        raise SystemExit("Task2 WAIT audit output already exists")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cases = [execute_case(protocol, protocol_path, protocol_hash, case) for case in protocol["evaluation"]["cases"]]
    report = {
        "schema_version": 1,
        "kind": "model-a-v4-task2-wait-qgap-audit-report",
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
        "summary": summarize(cases),
    }
    atomic_json(report_path, report)
    print(json.dumps({"report": str(report_path.relative_to(ROOT)), **report["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
