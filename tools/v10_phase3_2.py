"""Official paired confirmation for the v10 history-aware root prior.

Every arm/seed is launched through experiments/evaluate.py, which in turn starts
an isolated official main.py game. This module never calls the offline v10
simulator and never trains or mutates a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_code.model_a_v10.callbacks import validated_student_head_weights


SOURCE_PATHS = (
    "agent_code/model_a_v10/callbacks.py",
    "agent_code/model_a_v10/interfaces.py",
    "agent_code/model_a_v10/repetition.py",
    "agent_code/model_a_v10/runtime.py",
    "agent_code/model_a_v10/search.py",
    "agent_code/model_a_v10/simulator.py",
    "experiments/evaluate.py",
    "tools/v10_phase3_2.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_arm_config(config: dict, *, repetition_enabled: bool, checkpoint_hash: str) -> None:
    if config.get("scenario") != "coin-heaven":
        raise ValueError("s103008 only supports official coin-heaven")
    if config.get("training") is not False or config.get("h2h") is not False:
        raise ValueError("s103008 must be inference-only and non-H2H")
    if config.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("arm config checkpoint hash mismatch")
    validated_student_head_weights(config)
    repetition = config.get("repetition", {})
    if bool(repetition.get("enabled", False)) is not repetition_enabled:
        raise ValueError("arm config repetition flag mismatch")
    if config.get("forbidden_actions") != ["BOMB"]:
        raise ValueError("coin-heaven arm must explicitly forbid BOMB")


def run_official(seed: int, variant: str, agent_config: Path, checkpoint: Path,
                 output_dir: Path) -> dict:
    run_id = f"s103008-{variant}-s{seed}"
    command = [
        sys.executable, "experiments/evaluate.py",
        "--agents", "model_a_v10",
        "--scenario", "coin-heaven",
        "--rounds", "1",
        "--seed", str(seed),
        "--agent-seed", str(seed),
        "--variant", f"v10-history-aware-{variant}",
        "--run-id", run_id,
        "--weight-path", str(checkpoint),
        "--agent-config", str(agent_config),
        "--output-dir", str(output_dir),
    ]
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    manifest_path = output_dir / f"{run_id}.json"
    if completed.returncode != 0 or not manifest_path.is_file():
        raise RuntimeError(f"official evaluation failed for {variant} seed {seed}")
    manifest = load_json(manifest_path)
    if manifest.get("status") != "completed" or manifest.get("exit_code") != 0:
        raise RuntimeError(f"official evaluation did not complete for {variant} seed {seed}")
    metrics = manifest.get("target_metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"official evaluation omitted target metrics for {variant} seed {seed}")
    return {
        "score": int(metrics["score"]),
        "steps": int(metrics["steps"]),
        "survived": int(metrics.get("suicides", 0)) == 0,
        "invalid": int(metrics.get("invalid_actions", 0)),
        "mean_decision_ms": float(metrics["mean_decision_time_ms"]),
        "manifest": str(manifest_path.relative_to(REPOSITORY_ROOT)),
        "raw_stats": manifest["artifacts"]["raw_stats"],
    }


def aggregate(rows: list[dict], variant: str, max_steps: int) -> dict:
    values = [row[variant] for row in rows]
    scores = [value["score"] for value in values]
    steps = [value["steps"] for value in values]
    decision_times = [value["mean_decision_ms"] for value in values]
    return {
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": int(min(scores)),
        "max_score": int(max(scores)),
        "full_score_count": int(sum(score == 50 for score in scores)),
        "step_limit_count": int(sum(step == max_steps for step in steps)),
        "mean_steps": float(np.mean(steps)),
        "survival_rate": float(np.mean([value["survived"] for value in values])),
        "invalid_actions": int(sum(value["invalid"] for value in values)),
        "mean_decision_ms": float(np.mean(decision_times)),
        "max_run_mean_decision_ms": float(max(decision_times)),
    }


def evaluate_gate(rows: list[dict], baseline: dict, candidate: dict, thresholds: dict) -> dict:
    wins = sum(row["candidate"]["score"] > row["baseline"]["score"] for row in rows)
    ties = sum(row["candidate"]["score"] == row["baseline"]["score"] for row in rows)
    losses = len(rows) - wins - ties
    baseline_has_limit = baseline["step_limit_count"] > 0
    latency_ok = all(
        row[variant]["mean_decision_ms"] < float(thresholds["mean_decision_ms"])
        for row in rows for variant in ("baseline", "candidate")
    )
    passed = bool(
        baseline_has_limit
        and candidate["mean_score"] >= baseline["mean_score"]
        and candidate["step_limit_count"]
            <= baseline["step_limit_count"] - int(thresholds["minimum_step_limit_reduction"])
        and candidate["mean_steps"] <= baseline["mean_steps"] * float(thresholds["mean_steps_ratio"])
        and all(
            row["candidate"]["score"] >= row["baseline"]["score"] - int(thresholds["max_score_drop"])
            for row in rows
        )
        and candidate["survival_rate"] >= baseline["survival_rate"]
        and candidate["invalid_actions"] <= baseline["invalid_actions"]
        and latency_ok
    )
    return {
        "passed": passed,
        "baseline_has_step_limit_games": baseline_has_limit,
        "paired_wins": wins,
        "paired_ties": ties,
        "paired_losses": losses,
        "all_run_mean_decision_times_within_limit": latency_ok,
        "thresholds": thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    output_path = (REPOSITORY_ROOT / config["output_path"]).resolve()
    run_dir = (REPOSITORY_ROOT / config["run_output_dir"]).resolve()
    if output_path.exists() or run_dir.exists():
        raise SystemExit("Refusing to overwrite existing s103008 output")
    if config.get("training") is not False or config.get("h2h") is not False:
        raise ValueError("s103008 must be inference-only and non-H2H")

    checkpoint = (REPOSITORY_ROOT / config["checkpoint_path"]).resolve()
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != config["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    arm_paths = {
        variant: (REPOSITORY_ROOT / config[f"{variant}_config_path"]).resolve()
        for variant in ("baseline", "candidate")
    }
    arm_configs = {variant: load_json(path) for variant, path in arm_paths.items()}
    validate_arm_config(arm_configs["baseline"], repetition_enabled=False,
                        checkpoint_hash=checkpoint_hash)
    validate_arm_config(arm_configs["candidate"], repetition_enabled=True,
                        checkpoint_hash=checkpoint_hash)
    for relative, expected in config["source_sha256"].items():
        if sha256(REPOSITORY_ROOT / relative) != expected:
            raise ValueError(f"source hash mismatch: {relative}")

    started = utc_now()
    run_dir.mkdir(parents=True)
    rows = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        baseline = run_official(seed, "baseline", arm_paths["baseline"], checkpoint, run_dir)
        candidate = run_official(seed, "candidate", arm_paths["candidate"], checkpoint, run_dir)
        rows.append({"seed": seed, "baseline": baseline, "candidate": candidate})

    max_steps = int(config["max_steps"])
    aggregates = {
        variant: aggregate(rows, variant, max_steps)
        for variant in ("baseline", "candidate")
    }
    gate = evaluate_gate(rows, aggregates["baseline"], aggregates["candidate"], config["thresholds"])
    current_hashes = {relative: sha256(REPOSITORY_ROOT / relative) for relative in SOURCE_PATHS}
    if current_hashes != config["source_sha256"]:
        raise RuntimeError("source changed during official evaluation")
    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("checkpoint changed during official evaluation")
    report = {
        "schema_version": 2,
        "kind": "v10-phase3.2-official-history-aware-confirmation",
        "status": "completed",
        "run_id": config["run_id"],
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": checkpoint_hash,
        "arm_config_sha256": {variant: sha256(path) for variant, path in arm_paths.items()},
        "source_sha256": current_hashes,
        "training": False,
        "h2h": False,
        "protocol": "official main.py; one isolated process per arm and seed",
        "rows": rows,
        "aggregates": aggregates,
        "gate": gate,
    }
    write_json(output_path, report)
    print(json.dumps({"output": str(output_path), "aggregates": aggregates, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
