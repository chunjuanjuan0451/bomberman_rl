"""Run and mechanically decide the v10.12 frozen-v4 tactical hybrid gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    for relative, expected in protocol["bindings"].items():
        target = ROOT / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"binding mismatch: {relative}")
    if protocol.get("training") or protocol.get("automatic_training"):
        raise ValueError("v10.12 must remain evaluation-only")
    return protocol


def _manifest_path(protocol: dict, stage: str, arm: str, seed: int) -> Path:
    return ROOT / protocol["evaluation_output_directory"] / f"v10.12-{stage}-{arm}-s{seed}.json"


def _run_one(protocol: dict, stage: str, arm: str, seed: int, rounds: int) -> dict:
    manifest_path = _manifest_path(protocol, stage, arm, seed)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") != "completed":
            raise RuntimeError(f"incomplete existing evaluation: {manifest_path}")
        return existing
    common = [
        sys.executable, "experiments/evaluate.py", "--scenario", "classic",
        "--rounds", str(rounds), "--seed", str(seed), "--agent-seed", str(seed),
        "--run-id", f"v10.12-{stage}-{arm}-s{seed}",
        "--variant", f"v10.12-{arm}",
        "--output-dir", protocol["evaluation_output_directory"],
    ]
    if arm == "v4":
        command = common + [
            "--agents", "model_a_dqn", *protocol["opponents"],
            "--weight-path", "agent_code/model_a_dqn/model_a.pt",
        ]
    else:
        command = common + [
            "--agents", "model_a_v10", *protocol["opponents"],
            "--weight-path", "experiments/checkpoints/model-a-v10-centered-blend-coin-heaven-s103004.npz",
            "--agent-config", protocol["candidate_config"],
        ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"evaluation failed ({completed.returncode}): {' '.join(command)}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        raise RuntimeError(f"evaluation did not complete: {manifest_path}")
    return manifest


def summarize(rows: list[dict], arm: str) -> dict:
    metrics = [row[arm]["target_metrics"] for row in rows]
    rounds = sum(int(item["rounds"]) for item in metrics)
    steps = sum(int(item["steps"]) for item in metrics)
    return {
        "rounds": rounds,
        "score": sum(int(item["score"]) for item in metrics),
        "score_per_round": sum(int(item["score"]) for item in metrics) / rounds,
        "coins_per_round": sum(int(item["coins"]) for item in metrics) / rounds,
        "kills_per_round": sum(int(item["kills"]) for item in metrics) / rounds,
        "suicides": sum(int(item["suicides"]) for item in metrics),
        "suicides_per_round": sum(int(item["suicides"]) for item in metrics) / rounds,
        "invalid_actions": sum(int(item["invalid_actions"]) for item in metrics),
        "mean_decision_time_ms": (
            sum(float(item["mean_decision_time_ms"]) * int(item["steps"]) for item in metrics)
            / steps if steps else 0.0
        ),
        "max_seed_mean_decision_time_ms": max(
            float(item["mean_decision_time_ms"]) for item in metrics),
    }


def compare(rows: list[dict]) -> dict:
    deltas = [
        row["hybrid"]["target_metrics"]["score_per_round"]
        - row["v4"]["target_metrics"]["score_per_round"]
        for row in rows
    ]
    ordered = sorted(deltas)
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2 else
              (ordered[middle - 1] + ordered[middle]) / 2.0)
    return {
        "score_per_round_deltas": deltas,
        "mean_score_per_round_delta": sum(deltas) / len(deltas),
        "median_score_per_round_delta": median,
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "worst_score_per_round_delta": min(deltas),
    }


def evaluate_stage(rows: list[dict], gate: dict) -> dict:
    v4, hybrid, paired = summarize(rows, "v4"), summarize(rows, "hybrid"), compare(rows)
    checks = {
        "score_delta": paired["mean_score_per_round_delta"] >= float(gate["minimum_mean_score_per_round_delta"]),
        "paired_net_wins": paired["wins"] - paired["losses"] >= int(gate["minimum_paired_net_wins"]),
        "median_delta": paired["median_score_per_round_delta"] >= float(gate["minimum_median_score_per_round_delta"]),
        "worst_seed": paired["worst_score_per_round_delta"] >= -float(gate["maximum_single_seed_score_per_round_regression"]),
        "suicide_delta": hybrid["suicides_per_round"] - v4["suicides_per_round"] <= float(gate["maximum_suicides_per_round_increase"]),
        "invalid_actions": hybrid["invalid_actions"] <= v4["invalid_actions"] + int(gate["maximum_invalid_action_increase"]),
        "latency": hybrid["max_seed_mean_decision_time_ms"] < float(gate["maximum_seed_mean_decision_time_ms"]),
    }
    return {"v4": v4, "hybrid": hybrid, "paired": paired,
            "checks": checks, "passed": all(checks.values())}


def run_stage(protocol: dict, name: str) -> tuple[list[dict], dict]:
    stage = protocol["stages"][name]
    rows = []
    for seed in stage["seeds"]:
        row = {"seed": seed}
        for arm in ("v4", "hybrid"):
            manifest = _run_one(protocol, name, arm, seed, int(stage["rounds_per_seed"]))
            row[arm] = {
                "manifest": str(_manifest_path(protocol, name, arm, seed).relative_to(ROOT)),
                "manifest_sha256": sha256_file(_manifest_path(protocol, name, arm, seed)),
                "target_metrics": manifest["target_metrics"],
            }
        rows.append(row)
    return rows, evaluate_stage(rows, stage["gate"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    protocol = load_protocol(config_path)
    dry = {
        "mode": "execute" if args.execute else "dry-run",
        "run_id": protocol["run_id"],
        "stages": protocol["stages"],
        "training": False,
        "output_path": protocol["output_path"],
    }
    if not args.execute:
        print(json.dumps(dry, indent=2, sort_keys=True))
        return 0
    output = ROOT / protocol["output_path"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite report: {output}")
    pilot_rows, pilot = run_stage(protocol, "pilot")
    confirmation_rows: list[dict] = []
    confirmation = None
    if pilot["passed"]:
        confirmation_rows, confirmation = run_stage(protocol, "confirmation")
    selected = bool(confirmation and confirmation["passed"])
    decision = ("select_v4_tactical_hybrid" if selected else
                "reject_after_confirmation" if confirmation is not None else
                "reject_after_pilot")
    report = {
        "kind": "v10.12-frozen-v4-tactical-hybrid-official-gate-v1",
        "run_id": protocol["run_id"],
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": str(config_path.relative_to(ROOT)),
        "preregistration_sha256": sha256_file(config_path),
        "pilot": {"rows": pilot_rows, "result": pilot},
        "confirmation": None if confirmation is None else {
            "rows": confirmation_rows, "result": confirmation},
        "decision": decision,
        "selected_arm": "hybrid" if selected else "v4",
        "automatic_training_started": False,
        "next_step": protocol["decision_branches"][decision],
    }
    atomic_json(output, report)
    print(json.dumps({"output": str(output), "decision": decision,
                      "next_step": report["next_step"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
