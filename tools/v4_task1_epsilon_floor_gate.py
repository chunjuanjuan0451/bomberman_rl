"""Run the isolated exact-v4 Task1 epsilon-floor signal gate.

Dry-run is the default.  ``--execute`` trains three paired Task1 replicas,
runs the preregistered greedy evaluations, writes one terminal report, and
never starts Task2 or a replacement full curriculum.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v4_epsilon_gate.config import (  # noqa: E402
    REPLICAS, checkpoint_path, load_protocol, sha256_file, snapshot_path,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task1-epsilon-floor-e015-paired-s114000.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_completed(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("kind") != kind:
        raise RuntimeError(f"refusing incomplete/incompatible artifact: {path}")
    return payload


def metrics_by_agent(stats: dict) -> dict[str, dict]:
    result = {}
    for name, raw in stats.get("by_agent", {}).items():
        rounds = int(raw.get("rounds", 0))
        steps = int(raw.get("steps", 0))
        moves = int(raw.get("moves", 0))
        bombs = int(raw.get("bombs", 0))
        invalid = int(raw.get("invalid", 0))
        waits = max(0, steps - moves - bombs - invalid)
        result[name] = {
            "rounds": rounds,
            "score": int(raw.get("score", 0)),
            "score_per_round": int(raw.get("score", 0)) / rounds if rounds else 0.0,
            "coins": int(raw.get("coins", 0)),
            "coins_per_round": int(raw.get("coins", 0)) / rounds if rounds else 0.0,
            "bombs": bombs,
            "bombs_per_round": bombs / rounds if rounds else 0.0,
            "moves": moves,
            "move_fraction": moves / steps if steps else 0.0,
            "waits": waits,
            "wait_fraction": waits / steps if steps else 0.0,
            "suicides": int(raw.get("suicides", 0)),
            "suicides_per_round": int(raw.get("suicides", 0)) / rounds if rounds else 0.0,
            "invalid_actions": invalid,
            "invalid_actions_per_round": invalid / rounds if rounds else 0.0,
            "steps": steps,
        }
    return result


def training_manifest_path(protocol: dict, replica: str) -> Path:
    return ROOT / protocol["training_manifest_directory"] / f"{replica}-task1.json"


def evaluation_manifest_path(protocol: dict, suite: str, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / suite / f"{label}-s{world_seed}.json"


def validate_candidate_checkpoint(protocol: dict, protocol_hash: str, replica: str) -> dict:
    from agent_code.model_a_v4_epsilon_gate.callbacks import _torch_load

    path = checkpoint_path(protocol, replica)
    if not path.is_file():
        raise RuntimeError(f"candidate checkpoint is missing: {path}")
    payload = _torch_load(path)
    expected = {
        "architecture": "model-a-mlp-v4-global7",
        "protocol_sha256": protocol_hash,
        "replica": replica,
        "completed_rounds": 400,
        "epsilon_floor": 0.15,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"candidate checkpoint {key} mismatch: {replica}")
    if abs(float(payload.get("epsilon", -1)) - 0.15) > 1e-12:
        raise RuntimeError(f"candidate final epsilon mismatch: {replica}")
    return payload


def validate_snapshots(protocol: dict, protocol_hash: str, replica: str) -> list[dict]:
    from agent_code.model_a_v4_epsilon_gate.callbacks import _torch_load

    result = []
    for completed_rounds in range(50, 401, 50):
        path = snapshot_path(protocol, replica, completed_rounds)
        diagnostic = path.with_suffix(".json")
        if not path.is_file() or not diagnostic.is_file():
            raise RuntimeError(f"missing round-{completed_rounds} diagnostic for {replica}")
        payload = _torch_load(path)
        metadata = json.loads(diagnostic.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != protocol_hash
            or payload.get("replica") != replica
            or int(payload.get("completed_rounds", -1)) != completed_rounds
            or int(metadata.get("completed_rounds", -1)) != completed_rounds
            or abs(float(payload.get("epsilon_floor", -1)) - 0.15) > 1e-12
        ):
            raise RuntimeError(f"diagnostic snapshot contract mismatch: {path}")
        result.append({
            "round": completed_rounds,
            "checkpoint": relative(path),
            "checkpoint_sha256": sha256_file(path),
            "diagnostic": relative(diagnostic),
            "diagnostic_sha256": sha256_file(diagnostic),
            "epsilon": float(metadata["epsilon"]),
            "training_steps": int(metadata["training_steps"]),
            "gradient_steps": int(metadata["gradient_steps"]),
            "cumulative_action_counts": metadata["cumulative_action_counts"],
        })
    return result


def curve_summary(stats: dict) -> dict:
    rounds = list(stats.get("by_round", {}).values())
    if len(rounds) != 400:
        raise RuntimeError(f"expected 400 Task1 rounds, found {len(rounds)}")
    coins = [int(item.get("coins", 0)) for item in rounds]
    blocks = [sum(coins[start:start + 100]) / 100 for start in range(0, 400, 100)]
    best_previous = max(blocks[:3])
    ratio = blocks[3] / best_previous if best_previous else 0.0
    return {
        "nonoverlapping_100_round_coin_means": blocks,
        "best_preceding_100_round_mean": best_previous,
        "final_100_round_mean": blocks[3],
        "final_to_best_preceding_ratio": ratio,
    }


def parse_loss_trace(log_path: Path) -> list[dict]:
    pattern = re.compile(r"update=(\d+) replay=(\d+) loss=([0-9.eE+-]+) epsilon=([0-9.eE+-]+)")
    trace = []
    if not log_path.is_file():
        return trace
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            trace.append({
                "gradient_step": int(match.group(1)),
                "replay_size": int(match.group(2)),
                "loss": float(match.group(3)),
                # This is v4's pre-clamp value inside _record; the snapshot
                # diagnostics contain the action-selection epsilon.
                "pre_clamp_epsilon": float(match.group(4)),
            })
    return trace


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, replica: str) -> dict:
    manifest_path = training_manifest_path(protocol, replica)
    output = checkpoint_path(protocol, replica)
    existing = load_completed(manifest_path, "model-a-v4-task1-epsilon-floor-training")
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed training protocol drift: {manifest_path}")
        validate_candidate_checkpoint(protocol, protocol_hash, replica)
        validate_snapshots(protocol, protocol_hash, replica)
        if existing["checkpoint"]["sha256"] != sha256_file(output):
            raise RuntimeError(f"completed checkpoint hash drift: {replica}")
        return existing
    stats_path = manifest_path.with_suffix(".stats.json")
    captured_log = manifest_path.with_suffix(".agent.log")
    snapshot_root = ROOT / protocol["snapshot_directory"] / replica
    if output.exists() or stats_path.exists() or captured_log.exists() or snapshot_root.exists():
        raise RuntimeError(f"refusing orphaned training artifact for {replica}")
    spec = protocol["replicas"][replica]
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_epsilon_gate",
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", "coin-heaven", "--n-rounds", str(protocol["training_rounds_per_replica"]),
        "--seed", str(spec["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides = {
        "MODEL_A_V4EF_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_V4EF_CHECKPOINT_PATH": str(output),
        "MODEL_A_V4EF_REPLICA": replica,
        "MODEL_A_V4EF_SEED": str(spec["agent_seed"]),
    }
    record = {
        "schema_version": 1,
        "kind": "model-a-v4-task1-epsilon-floor-training",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "replica": replica,
        "status": "running",
        "started_at_utc": utc_now(),
        "training": {
            **spec,
            "scenario": "coin-heaven",
            "opponents": [],
            "rounds": int(protocol["training_rounds_per_replica"]),
            "epsilon_floor": 0.15,
        },
        "command": command,
        "environment_overrides": overrides,
        "checkpoint": {"path": relative(output), "sha256": None},
        "artifacts": {
            "raw_stats": relative(stats_path),
            "captured_agent_log": relative(captured_log),
        },
    }
    atomic_json(manifest_path, record)
    output.parent.mkdir(parents=True, exist_ok=True)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    source_log = ROOT / "agent_code/model_a_v4_epsilon_gate/logs/model_a_v4_epsilon_gate.log"
    if source_log.is_file():
        captured_log.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_log, captured_log)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats_path.is_file():
            raise RuntimeError("training process or stats output failed")
        payload = validate_candidate_checkpoint(protocol, protocol_hash, replica)
        snapshots = validate_snapshots(protocol, protocol_hash, replica)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        record["metrics_by_agent"] = metrics_by_agent(stats)
        record["curve"] = curve_summary(stats)
        record["snapshots"] = snapshots
        record["loss_trace"] = parse_loss_trace(captured_log)
        record["final_training_state"] = {
            "training_steps": int(payload["training_steps"]),
            "gradient_steps": int(payload["gradient_steps"]),
            "epsilon": float(payload["epsilon"]),
        }
        record["checkpoint"]["sha256"] = sha256_file(output)
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest_path, record)
    if record["status"] != "completed":
        raise RuntimeError(f"training failed for {replica}: {record.get('error')}")
    return record


def evaluation_identity(protocol: dict, label: str, case: dict) -> tuple[str, Path, dict[str, str]]:
    if label == "v4":
        checkpoint = ROOT / protocol["frozen_v4_baseline"]["checkpoint_path"]
        return "model_a_dqn", checkpoint, {
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_SEED": str(case["agent_seed"]),
        }
    family, replica = label.split("-", 1)
    if replica not in REPLICAS:
        raise ValueError(f"invalid evaluation label: {label}")
    if family == "candidate":
        checkpoint = checkpoint_path(protocol, replica)
        return "model_a_v4_epsilon_gate", checkpoint, {
            "MODEL_A_V4EF_PROTOCOL_PATH": str(ROOT / protocol["protocol_path"]),
            "MODEL_A_V4EF_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V4EF_REPLICA": replica,
            "MODEL_A_V4EF_SEED": str(case["agent_seed"]),
        }
    if family == "control":
        baseline = protocol["baseline"]
        checkpoint = ROOT / baseline["curriculum_checkpoints"][replica]["path"]
        return "model_a_v4_curriculum", checkpoint, {
            "MODEL_A_V4C_PROTOCOL_PATH": str(ROOT / baseline["protocol_path"]),
            "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V4C_ARM": "curriculum",
            "MODEL_A_V4C_REPLICA": replica,
            "MODEL_A_V4C_STAGE": "task1",
            "MODEL_A_V4C_SEED": str(case["agent_seed"]),
        }
    raise ValueError(f"invalid evaluation label: {label}")


def evaluate_one(
    protocol: dict,
    protocol_hash: str,
    suite: str,
    label: str,
    case: dict,
) -> dict:
    target, checkpoint, overrides = evaluation_identity(protocol, label, case)
    expected_hash = sha256_file(checkpoint)
    manifest_path = evaluation_manifest_path(protocol, suite, label, int(case["world_seed"]))
    existing = load_completed(manifest_path, "model-a-v4-task1-epsilon-floor-evaluation")
    expected_evaluation = {
        **case,
        "scenario": "coin-heaven",
        "opponents": [],
        "rounds": int(protocol["evaluation"][suite]["rounds_per_case"]),
        "target_agent": target,
    }
    if existing is not None:
        if (
            existing.get("protocol_sha256") != protocol_hash
            or existing.get("evaluation") != expected_evaluation
            or existing.get("checkpoint", {}).get("sha256") != expected_hash
        ):
            raise RuntimeError(f"completed evaluation drift: {manifest_path}")
        return existing
    stats_path = manifest_path.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned evaluation artifact: {stats_path}")
    command = [
        sys.executable, "main.py", "play", "--agents", target,
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", "coin-heaven", "--n-rounds", str(expected_evaluation["rounds"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    record = {
        "schema_version": 1,
        "kind": "model-a-v4-task1-epsilon-floor-evaluation",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "suite": suite,
        "label": label,
        "status": "running",
        "started_at_utc": utc_now(),
        "evaluation": expected_evaluation,
        "checkpoint": {"path": relative(checkpoint), "sha256": expected_hash},
        "command": command,
        "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest_path, record)
    child = os.environ.copy()
    child.update(overrides)
    before = sha256_file(checkpoint)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    if completed.returncode == 0 and stats_path.is_file() and sha256_file(checkpoint) == before:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        record["metrics_by_agent"] = metrics_by_agent(stats)
        record["target_metrics"] = record["metrics_by_agent"].get(target)
        record["status"] = "completed" if record["target_metrics"] else "failed"
    else:
        record["status"] = "failed"
    atomic_json(manifest_path, record)
    if record["status"] != "completed":
        raise RuntimeError(f"evaluation failed: {suite}/{label}/{case['world_seed']}")
    return record


def aggregate(manifests: list[dict]) -> dict:
    additive = (
        "rounds", "score", "coins", "bombs", "moves", "waits",
        "suicides", "invalid_actions", "steps",
    )
    totals = {key: sum(int(item["target_metrics"][key]) for item in manifests) for key in additive}
    rounds = totals["rounds"]
    steps = totals["steps"]
    return {
        **totals,
        "score_per_round": totals["score"] / rounds,
        "coins_per_round": totals["coins"] / rounds,
        "bombs_per_round": totals["bombs"] / rounds,
        "suicides_per_round": totals["suicides"] / rounds,
        "invalid_actions_per_round": totals["invalid_actions"] / rounds,
        "move_fraction": totals["moves"] / steps if steps else 0.0,
        "wait_fraction": totals["waits"] / steps if steps else 0.0,
    }


def run_evaluations(protocol: dict, protocol_hash: str) -> tuple[dict, dict]:
    rows = {}
    manifests = {}
    labels_by_suite = {
        "paired_seen": [f"candidate-{replica}" for replica in REPLICAS],
        "fresh_confirmation": [
            "v4",
            *[f"control-{replica}" for replica in REPLICAS],
            *[f"candidate-{replica}" for replica in REPLICAS],
        ],
    }
    for suite, labels in labels_by_suite.items():
        rows[suite] = {}
        manifests[suite] = {}
        cases = protocol["evaluation"][suite]["cases"]
        for label in labels:
            runs = [evaluate_one(protocol, protocol_hash, suite, label, case) for case in cases]
            rows[suite][label] = aggregate(runs)
            manifests[suite][label] = [
                relative(evaluation_manifest_path(protocol, suite, label, int(case["world_seed"])))
                for case in cases
            ]
    return rows, manifests


def decide(protocol: dict, training: dict, evaluation_rows: dict) -> dict:
    gate = protocol["gate"]
    baseline_report = json.loads(
        (ROOT / protocol["baseline"]["task1_report_path"]).read_text(encoding="utf-8")
    )
    baseline_rows = baseline_report["heldout_validation"]["rows"]["task1"]
    stable = []
    supportive = []
    per_replica = {}
    for replica in REPLICAS:
        curve = training[replica]["curve"]
        stability_ratio = float(curve["final_to_best_preceding_ratio"])
        paired_candidate = evaluation_rows["paired_seen"][f"candidate-{replica}"]["score_per_round"]
        paired_control = baseline_rows[f"curriculum-{replica}"]["score_per_round"]
        paired_delta = paired_candidate - paired_control
        is_stable = stability_ratio >= float(gate["minimum_final_to_best_100_round_ratio"])
        is_supportive = paired_delta >= float(gate["minimum_paired_score_improvement"])
        if is_stable:
            stable.append(replica)
        if is_supportive:
            supportive.append(replica)
        per_replica[replica] = {
            "curve_stable": is_stable,
            "final_to_best_100_round_ratio": stability_ratio,
            "paired_seen_candidate_score_per_round": paired_candidate,
            "paired_seen_control_score_per_round": paired_control,
            "paired_seen_improvement": paired_delta,
            "paired_supportive": is_supportive,
            "fresh_confirmation_score_per_round": evaluation_rows["fresh_confirmation"][f"candidate-{replica}"]["score_per_round"],
        }
    paired_candidate_family = sum(
        evaluation_rows["paired_seen"][f"candidate-{replica}"]["score_per_round"]
        for replica in REPLICAS
    ) / len(REPLICAS)
    paired_control_family = sum(
        baseline_rows[f"curriculum-{replica}"]["score_per_round"] for replica in REPLICAS
    ) / len(REPLICAS)
    confirmation_family = sum(
        evaluation_rows["fresh_confirmation"][f"candidate-{replica}"]["score_per_round"]
        for replica in REPLICAS
    ) / len(REPLICAS)
    catastrophic = [
        replica for replica in REPLICAS
        if per_replica[replica]["fresh_confirmation_score_per_round"]
        < float(gate["minimum_each_replica_fresh_score_per_round"])
    ]
    gates = {
        "curve_stability": len(stable) >= int(gate["minimum_stable_replicas"]),
        "paired_support": len(supportive) >= int(gate["minimum_supportive_replicas"]),
        "paired_family_improvement": (
            paired_candidate_family - paired_control_family
            >= float(gate["minimum_paired_family_score_improvement"])
        ),
        "no_catastrophic_fresh_replica": not catastrophic,
        "fresh_family_quality": confirmation_family >= float(gate["minimum_fresh_family_score_per_round"]),
    }
    passed = all(gates.values())
    return {
        "per_replica": per_replica,
        "stable_replicas": stable,
        "supportive_replicas": supportive,
        "paired_candidate_family_score_per_round": paired_candidate_family,
        "paired_control_family_score_per_round": paired_control_family,
        "paired_family_improvement": paired_candidate_family - paired_control_family,
        "fresh_candidate_family_score_per_round": confirmation_family,
        "catastrophic_fresh_replicas": catastrophic,
        "gates": gates,
        "passed": passed,
        "decision": "epsilon_floor_signal_supported" if passed else "epsilon_floor_signal_rejected",
        "automatic_followup_started": False,
        "task2_started": False,
        "matched_full_restart_started": False,
    }


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def write_report(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    training: dict,
    rows: dict,
    evaluation_manifests: dict,
) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, "model-a-v4-task1-epsilon-floor-report")
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed report protocol drift: {path}")
        return existing, sha256_file(path)
    report = {
        "schema_version": 1,
        "kind": "model-a-v4-task1-epsilon-floor-report",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "single_changed_variable": protocol["single_changed_variable"],
        "training": {
            replica: {
                "manifest": relative(training_manifest_path(protocol, replica)),
                "manifest_sha256": sha256_file(training_manifest_path(protocol, replica)),
                "checkpoint": training[replica]["checkpoint"],
                "curve": training[replica]["curve"],
                "final_training_state": training[replica]["final_training_state"],
                "snapshots": training[replica]["snapshots"],
                "loss_trace": training[replica]["loss_trace"],
            }
            for replica in REPLICAS
        },
        "evaluation_rows": rows,
        "evaluation_manifests": evaluation_manifests,
        "result": decide(protocol, training, rows),
        "awaiting_user_instruction": True,
        "default_checkpoint_modified": False,
        "automatic_followup_started": False,
    }
    atomic_json(path, report)
    return report, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "single_changed_variable": protocol["single_changed_variable"],
        "epsilon": protocol["epsilon"],
        "training": [
            {
                "replica": replica,
                **protocol["replicas"][replica],
                "scenario": "coin-heaven",
                "opponents": [],
                "rounds": int(protocol["training_rounds_per_replica"]),
                "checkpoint": relative(checkpoint_path(protocol, replica)),
            }
            for replica in REPLICAS
        ],
        "training_rounds": 3 * int(protocol["training_rounds_per_replica"]),
        "evaluation_rounds": 500,
        "paired_seen_is_diagnostic_only": True,
        "fresh_confirmation_cases": len(protocol["evaluation"]["fresh_confirmation"]["cases"]),
        "writes_terminal_report": True,
        "task2_started": False,
        "matched_full_restart_started": False,
        "formal_experiment_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    try:
        protocol, protocol_hash = load_protocol(protocol_path)
        if not args.execute:
            print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
            return 0
        existing_report = load_completed(report_path(protocol), "model-a-v4-task1-epsilon-floor-report")
        if existing_report is not None:
            if existing_report.get("protocol_sha256") != protocol_hash:
                raise RuntimeError("completed report protocol drift")
            print(json.dumps({
                "status": "completed",
                "report": relative(report_path(protocol)),
                "report_sha256": sha256_file(report_path(protocol)),
                "decision": existing_report["result"]["decision"],
                "awaiting_user_instruction": True,
                "task2_started": False,
                "matched_full_restart_started": False,
            }, indent=2, sort_keys=True))
            return 0
        training = {
            replica: train_one(protocol_path.resolve(), protocol, protocol_hash, replica)
            for replica in REPLICAS
        }
        rows, evaluation_manifests = run_evaluations(protocol, protocol_hash)
        report, report_hash = write_report(
            protocol_path.resolve(), protocol, protocol_hash,
            training, rows, evaluation_manifests,
        )
        print(json.dumps({
            "status": "completed",
            "report": relative(report_path(protocol)),
            "report_sha256": report_hash,
            "decision": report["result"]["decision"],
            "gates": report["result"]["gates"],
            "awaiting_user_instruction": True,
            "task2_started": False,
            "matched_full_restart_started": False,
        }, indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
