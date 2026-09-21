"""Run the no-training paired Task1 retention audit after clean-v4 Task2.

Dry-run is the default. ``--execute`` evaluates frozen checkpoints only,
writes a terminal diagnostic report, and never starts Task3 or any training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v4_curriculum.config import load_protocol as load_clean_protocol  # noqa: E402
from tools.v4_clean_curriculum import validate_checkpoint  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task1-retention-audit-s118100.json"
REPLICAS = ("r1", "r2", "r3")
STAGES = ("task1", "task2")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_protocol(path: Path) -> tuple[dict, str, dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("retention audit requires schema_version=1")
    if protocol.get("kind") != "model-a-v4-task1-retention-audit":
        raise ValueError("wrong retention-audit protocol kind")
    if tuple(protocol.get("replicas", ())) != REPLICAS:
        raise ValueError("retention audit requires ordered r1/r2/r3")
    evaluation = protocol.get("evaluation", {})
    if (
        evaluation.get("scenario") != "coin-heaven"
        or evaluation.get("opponents") != []
        or int(evaluation.get("rounds_per_case", 0)) != 25
        or len(evaluation.get("cases", [])) != 2
    ):
        raise ValueError("paired Task1 evaluation contract changed")
    seen = set()
    for case in evaluation["cases"]:
        if set(case) != {"world_seed", "agent_seed", "opponent_seed"}:
            raise ValueError("incomplete retention evaluation seed tuple")
        for value in case.values():
            seed = int(value)
            if seed in seen:
                raise ValueError("retention evaluation seeds must be unique")
            seen.add(seed)
    if any(seed // 1000 != prefix for seed, prefix in zip(sorted(seen), (118, 118, 218, 218, 318, 318))):
        raise ValueError("retention audit must use the preregistered 118/218/318 seed families")
    bindings = protocol.get("source_bindings", {})
    required_sources = {
        "agent_code/model_a_dqn/callbacks.py",
        "agent_code/model_a_dqn/features.py",
        "agent_code/model_a_dqn/network.py",
        "agent_code/model_a_v4_curriculum/config.py",
        "agent_code/model_a_v4_curriculum/callbacks.py",
        "tools/v4_clean_curriculum.py",
        "tools/v4_task1_retention_audit.py",
    }
    if set(bindings) != required_sources:
        raise ValueError("retention-audit source binding set mismatch")
    for relative_path, expected_hash in bindings.items():
        source = ROOT / relative_path
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"retention-audit source binding mismatch: {relative_path}")
    source = protocol["source_experiment"]
    clean_path = ROOT / source["protocol_path"]
    clean_protocol, clean_hash = load_clean_protocol(clean_path)
    if clean_hash != source["protocol_sha256"]:
        raise ValueError("clean-v4 source protocol hash mismatch")
    for key in ("task1_report", "task2_report"):
        report = ROOT / source[key]["path"]
        if not report.is_file() or sha256_file(report) != source[key]["sha256"]:
            raise ValueError(f"clean-v4 {key} binding mismatch")
    frozen = ROOT / protocol["frozen_v4_baseline"]["checkpoint_path"]
    if not frozen.is_file() or sha256_file(frozen) != protocol["frozen_v4_baseline"]["checkpoint_sha256"]:
        raise ValueError("frozen-v4 checkpoint binding mismatch")
    for replica in REPLICAS:
        for stage in STAGES:
            validate_checkpoint(clean_protocol, clean_hash, "curriculum", replica, stage)
            checkpoint = ROOT / source["checkpoints"][replica][stage]["path"]
            if sha256_file(checkpoint) != source["checkpoints"][replica][stage]["sha256"]:
                raise ValueError(f"checkpoint binding mismatch: {replica}/{stage}")
    return protocol, sha256_file(path), clean_protocol, clean_hash


def checkpoint_identity(protocol: dict, label: str) -> tuple[str, Path, str | None, str | None]:
    if label == "v4":
        return "model_a_dqn", ROOT / protocol["frozen_v4_baseline"]["checkpoint_path"], None, None
    stage, replica = label.split("-", 1)
    if stage not in STAGES or replica not in REPLICAS:
        raise ValueError(f"invalid retention label: {label}")
    path = ROOT / protocol["source_experiment"]["checkpoints"][replica][stage]["path"]
    return "model_a_v4_curriculum", path, stage, replica


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


def manifest_path(protocol: dict, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / f"{label}-s{world_seed}.json"


def evaluate_one(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    clean_protocol_hash: str,
    label: str,
    case: dict,
) -> dict:
    target, checkpoint, stage, replica = checkpoint_identity(protocol, label)
    checkpoint_hash = sha256_file(checkpoint)
    output = manifest_path(protocol, label, int(case["world_seed"]))
    expected_evaluation = {
        **case,
        "scenario": "coin-heaven",
        "opponents": [],
        "rounds": int(protocol["evaluation"]["rounds_per_case"]),
        "target_agent": target,
    }
    existing = load_completed(output, "model-a-v4-task1-retention-evaluation")
    if existing is not None:
        if (
            existing.get("protocol_sha256") != protocol_hash
            or existing.get("evaluation") != expected_evaluation
            or existing.get("checkpoint", {}).get("sha256") != checkpoint_hash
        ):
            raise RuntimeError(f"completed retention evaluation drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned retention stats: {stats_path}")
    command = [
        sys.executable, "main.py", "play", "--agents", target,
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", "coin-heaven", "--n-rounds", str(expected_evaluation["rounds"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    if label == "v4":
        overrides = {
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_SEED": str(case["agent_seed"]),
        }
    else:
        overrides = {
            "MODEL_A_V4C_PROTOCOL_PATH": str(ROOT / protocol["source_experiment"]["protocol_path"]),
            "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V4C_ARM": "curriculum",
            "MODEL_A_V4C_REPLICA": str(replica),
            "MODEL_A_V4C_STAGE": str(stage),
            "MODEL_A_V4C_SEED": str(case["agent_seed"]),
        }
    record = {
        "schema_version": 1,
        "kind": "model-a-v4-task1-retention-evaluation",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "source_clean_protocol_sha256": clean_protocol_hash,
        "label": label,
        "status": "running",
        "started_at_utc": utc_now(),
        "evaluation": expected_evaluation,
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
        "command": command,
        "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(output, record)
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
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"retention evaluation failed: {label}/{case['world_seed']}")
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
        "move_fraction": totals["moves"] / steps if steps else 0.0,
        "wait_fraction": totals["waits"] / steps if steps else 0.0,
        "suicides_per_round": totals["suicides"] / rounds,
        "invalid_actions_per_round": totals["invalid_actions"] / rounds,
    }


def run_evaluations(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    clean_protocol_hash: str,
) -> tuple[dict, dict]:
    labels = ["v4", *[f"{stage}-{replica}" for replica in REPLICAS for stage in STAGES]]
    rows = {}
    manifests = {}
    for label in labels:
        runs = [
            evaluate_one(protocol_path, protocol, protocol_hash, clean_protocol_hash, label, case)
            for case in protocol["evaluation"]["cases"]
        ]
        rows[label] = aggregate(runs)
        manifests[label] = [
            relative(manifest_path(protocol, label, int(case["world_seed"])))
            for case in protocol["evaluation"]["cases"]
        ]
    return rows, manifests


def decide(protocol: dict, rows: dict, task2_report: dict) -> dict:
    rule = protocol["decision_rule"]
    per_replica = {}
    affected = []
    joint = []
    task2_rows = task2_report["heldout_validation"]["rows"]["task2"]
    task2_v4 = float(task2_rows["v4"]["score_per_round"])
    for replica in REPLICAS:
        before = float(rows[f"task1-{replica}"]["score_per_round"])
        after = float(rows[f"task2-{replica}"]["score_per_round"])
        delta = after - before
        ratio = after / before if before else 0.0
        is_affected = (
            delta <= -float(rule["minimum_absolute_drop_per_round"])
            and ratio <= float(rule["maximum_retained_fraction"])
        )
        current_task2 = float(task2_rows[f"curriculum-{replica}"]["score_per_round"])
        is_joint = (
            after >= float(rows["v4"]["score_per_round"]) - float(rule["maximum_task1_gap_to_v4_for_joint_parent"])
            and current_task2 >= task2_v4
        )
        if is_affected:
            affected.append(replica)
        if is_joint:
            joint.append(replica)
        per_replica[replica] = {
            "task1_checkpoint_score_per_round": before,
            "task2_checkpoint_score_per_round": after,
            "absolute_retention_delta": delta,
            "retained_fraction": ratio,
            "forgetting_threshold_met": is_affected,
            "existing_task2_score_per_round": current_task2,
            "joint_parent_gate": is_joint,
        }
    before_family = sum(per_replica[r]["task1_checkpoint_score_per_round"] for r in REPLICAS) / 3
    after_family = sum(per_replica[r]["task2_checkpoint_score_per_round"] for r in REPLICAS) / 3
    family_delta = after_family - before_family
    gates = {
        "affected_replica_count": len(affected) >= int(rule["minimum_affected_replicas"]),
        "family_drop": family_delta <= -float(rule["minimum_family_drop_per_round"]),
    }
    confirmed = all(gates.values())
    if confirmed and joint:
        decision = "retention_failure_confirmed_with_joint_parent"
    elif confirmed:
        decision = "retention_failure_confirmed_no_joint_parent"
    else:
        decision = "retention_failure_not_confirmed"
    return {
        "per_replica": per_replica,
        "affected_replicas": affected,
        "task1_checkpoint_family_score_per_round": before_family,
        "task2_checkpoint_family_score_per_round": after_family,
        "family_retention_delta": family_delta,
        "gates": gates,
        "forgetting_confirmed": confirmed,
        "joint_parent_candidates": joint,
        "decision": decision,
        "automatic_checkpoint_selected": False,
        "training_started": False,
        "task3_started": False,
        "automatic_followup_started": False,
    }


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def write_report(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    rows: dict,
    manifests: dict,
) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, "model-a-v4-task1-retention-audit-report")
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed retention report protocol drift")
        return existing, sha256_file(path)
    task2_report = json.loads(
        (ROOT / protocol["source_experiment"]["task2_report"]["path"]).read_text(encoding="utf-8")
    )
    report = {
        "schema_version": 1,
        "kind": "model-a-v4-task1-retention-audit-report",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "rows": rows,
        "evaluation_manifests": manifests,
        "result": decide(protocol, rows, task2_report),
        "awaiting_user_instruction": True,
        "training_started": False,
        "task3_started": False,
        "automatic_followup_started": False,
        "default_checkpoint_modified": False,
    }
    atomic_json(path, report)
    return report, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    labels = ["v4", *[f"{stage}-{replica}" for replica in REPLICAS for stage in STAGES]]
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "labels": labels,
        "cases": protocol["evaluation"]["cases"],
        "rounds_per_case": int(protocol["evaluation"]["rounds_per_case"]),
        "total_evaluation_rounds": len(labels) * len(protocol["evaluation"]["cases"]) * int(protocol["evaluation"]["rounds_per_case"]),
        "paired_same_seeds": True,
        "training_started": False,
        "task3_started": False,
        "automatic_followup_started": False,
        "formal_evaluation_started": False,
        "writes_terminal_report_and_stops": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    try:
        protocol, protocol_hash, _, clean_hash = load_protocol(protocol_path)
        if not args.execute:
            print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
            return 0
        existing = load_completed(report_path(protocol), "model-a-v4-task1-retention-audit-report")
        if existing is not None:
            if existing.get("protocol_sha256") != protocol_hash:
                raise RuntimeError("completed retention report protocol drift")
            print(json.dumps({
                "status": "completed",
                "report": relative(report_path(protocol)),
                "report_sha256": sha256_file(report_path(protocol)),
                "decision": existing["result"]["decision"],
                "awaiting_user_instruction": True,
                "training_started": False,
                "task3_started": False,
            }, indent=2, sort_keys=True))
            return 0
        rows, manifests = run_evaluations(protocol_path.resolve(), protocol, protocol_hash, clean_hash)
        report, report_hash = write_report(protocol_path.resolve(), protocol, protocol_hash, rows, manifests)
        print(json.dumps({
            "status": "completed",
            "report": relative(report_path(protocol)),
            "report_sha256": report_hash,
            "decision": report["result"]["decision"],
            "gates": report["result"]["gates"],
            "joint_parent_candidates": report["result"]["joint_parent_candidates"],
            "awaiting_user_instruction": True,
            "training_started": False,
            "task3_started": False,
        }, indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
