"""Train, evaluate, and mechanically screen the Task-3 four-arm pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def metrics_by_agent(stats: dict) -> dict[str, dict]:
    result = {}
    for name, raw in stats.get("by_agent", {}).items():
        rounds = int(raw.get("rounds", 0))
        steps = int(raw.get("steps", 0))
        result[name] = {
            "rounds": rounds,
            "score": int(raw.get("score", 0)),
            "score_per_round": int(raw.get("score", 0)) / rounds if rounds else 0.0,
            "coins": int(raw.get("coins", 0)),
            "kills": int(raw.get("kills", 0)),
            "crates": int(raw.get("crates", 0)),
            "bombs": int(raw.get("bombs", 0)),
            "suicides": int(raw.get("suicides", 0)),
            "invalid_actions": int(raw.get("invalid", 0)),
            "steps": steps,
            "mean_decision_time_ms": 1000.0 * float(raw.get("time", 0.0)) / steps if steps else 0.0,
        }
    return result


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("task") != 3 or protocol.get("includes_task4", True):
        raise ValueError("four-arm protocol must remain isolated to Task 3")
    if not protocol.get("training") or protocol.get("automatic_replication"):
        raise ValueError("pilot training must be enabled and automatic replication disabled")
    if list(protocol.get("arms", {})) != ["A", "B", "C", "D"]:
        raise ValueError("four-arm protocol requires ordered arms A, B, C, D")
    for relative_path, expected in protocol["bindings"].items():
        target = ROOT / relative_path
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"binding mismatch: {relative_path}")
    v4_path = ROOT / protocol["v4"]["checkpoint_path"]
    if not v4_path.is_file() or sha256_file(v4_path) != protocol["v4"]["checkpoint_sha256"]:
        raise ValueError("frozen v4 checkpoint mismatch")
    from agent_code.model_a_task3.config import load_config
    run_ids, checkpoints = set(), set()
    training_seed_tuples = set()
    for arm, spec in protocol["arms"].items():
        config_path = ROOT / spec["config_path"]
        config, _, config_hash = load_config(config_path)
        if config["arm"] != arm or config_hash != spec["config_sha256"]:
            raise ValueError(f"arm config mismatch: {arm}")
        if config["parent_sha256"] != protocol["v4"]["checkpoint_sha256"]:
            raise ValueError(f"arm parent mismatch: {arm}")
        run_ids.add(config["run_id"])
        checkpoints.add(config["checkpoint_path"])
        training_seed_tuples.add((config["seed"], config["agent_seed"], config["opponent_seed"]))
    if len(run_ids) != 4 or len(checkpoints) != 4 or len(training_seed_tuples) != 1:
        raise ValueError("arms require unique artifacts and one shared paired training seed tuple")
    if set(protocol.get("strata", {})) != {"peaceful", "coin_collector"}:
        raise ValueError("pilot requires separate peaceful and coin_collector strata")
    seeds = []
    for name, stratum in protocol["strata"].items():
        if int(stratum.get("rounds_per_seed", 0)) <= 0 or not stratum.get("cases"):
            raise ValueError(f"invalid evaluation budget for {name}")
        for case in stratum["cases"]:
            if set(case) != {"world_seed", "agent_seed", "opponent_seed"}:
                raise ValueError(f"incomplete seed tuple for {name}")
            seeds.append(int(case["world_seed"]))
    if len(seeds) != len(set(seeds)):
        raise ValueError("evaluation world seeds must be unique across strata")
    return protocol


def _load_completed(path: Path, expected_kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("kind") != expected_kind:
        raise RuntimeError(f"refusing incomplete or incompatible artifact: {path}")
    return payload


def train_one(protocol: dict, arm: str) -> dict:
    spec = protocol["arms"][arm]
    config_path = (ROOT / spec["config_path"]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = (ROOT / config["checkpoint_path"]).resolve()
    manifest_path = ROOT / "experiments/logs/runs" / f"{config['run_id']}.json"
    stats_path = manifest_path.with_suffix(".stats.json")
    existing = _load_completed(manifest_path, "task3-training")
    if existing is not None:
        if existing.get("config_sha256") != spec["config_sha256"]:
            raise RuntimeError(f"training config mismatch for arm {arm}")
        if not checkpoint.is_file() or sha256_file(checkpoint) != existing["checkpoint"]["sha256"]:
            raise RuntimeError(f"checkpoint mismatch for arm {arm}")
        return existing
    if checkpoint.exists() or stats_path.exists():
        raise RuntimeError(f"refusing orphaned Task-3 training artifact for arm {arm}")
    environment_overrides = {
        "MODEL_A_TASK3_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_TASK3_CONFIG_PATH": str(config_path),
        "MODEL_A_TASK3_SEED": str(config["agent_seed"]),
        "MODEL_A_TASK3_RESUME": "0",
        "TASK3_CURRICULUM_SEED": str(config["opponent_seed"]),
    }
    environment = os.environ.copy()
    environment.update(environment_overrides)
    command = [
        sys.executable, "main.py", "play", "--agents", *config["agents"],
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", "classic", "--n-rounds", str(config["rounds"]),
        "--seed", str(config["seed"]), "--save-stats", str(stats_path),
    ]
    manifest = {
        "schema_version": 1,
        "kind": "task3-training",
        "run_id": config["run_id"],
        "arm": arm,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "config": config,
        "checkpoint": {"path": relative(checkpoint), "sha256": None},
        "environment_overrides": environment_overrides,
        "command": command,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest_path, manifest)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["exit_code"] = completed.returncode
    if completed.returncode == 0 and checkpoint.is_file() and stats_path.is_file():
        try:
            from agent_code.model_a_task3.callbacks import _torch_load
            from agent_code.model_a_task3.config import architecture_name
            checkpoint_payload = _torch_load(checkpoint)
            valid = (
                checkpoint_payload.get("arm") == arm
                and checkpoint_payload.get("architecture") == architecture_name(config)
                and checkpoint_payload.get("config_sha256") == spec["config_sha256"]
                and checkpoint_payload.get("parent_sha256") == config["parent_sha256"]
                and int(checkpoint_payload.get("completed_rounds", 0)) == int(config["rounds"])
            )
        except Exception as exc:
            valid = False
            manifest["checkpoint_validation_error"] = f"{type(exc).__name__}: {exc}"
        if valid:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            manifest["metrics_by_agent"] = metrics_by_agent(stats)
            manifest["checkpoint"]["sha256"] = sha256_file(checkpoint)
            manifest["status"] = "completed"
        else:
            manifest["status"] = "failed"
            manifest["error"] = "checkpoint metadata validation failed"
    else:
        manifest["status"] = "failed"
    atomic_json(manifest_path, manifest)
    if manifest["status"] != "completed":
        raise RuntimeError(f"Task-3 training failed for arm {arm}")
    return manifest


def _evaluation_manifest(protocol: dict, stratum: str, label: str, seed: int) -> Path:
    return ROOT / protocol["evaluation_output_directory"] / f"task3-four-arm-{stratum}-{label}-s{seed}.json"


def evaluate_one(protocol: dict, stratum: str, label: str, case: dict, checkpoints: dict) -> dict:
    seed = int(case["world_seed"])
    manifest_path = _evaluation_manifest(protocol, stratum, label, seed)
    existing = _load_completed(manifest_path, "task3-four-arm-evaluation")
    target = "model_a_dqn" if label == "v4" else "model_a_task3"
    checkpoint = (
        ROOT / protocol["v4"]["checkpoint_path"] if label == "v4" else checkpoints[label]
    ).resolve()
    config_path = None if label == "v4" else (ROOT / protocol["arms"][label]["config_path"]).resolve()
    expected_checkpoint_hash = (
        protocol["v4"]["checkpoint_sha256"] if label == "v4" else sha256_file(checkpoint)
    )
    expected_evaluation = {
        "target_agent": target,
        "agents": [target, protocol["strata"][stratum]["opponent"]],
        "scenario": "classic",
        "rounds": int(protocol["strata"][stratum]["rounds_per_seed"]),
        "seed": seed,
        "agent_seed": int(case["agent_seed"]),
        "opponent_seed": int(case["opponent_seed"]),
        "training_agents": 0,
        "continue_without_training": True,
    }
    if existing is not None:
        if existing.get("evaluation") != expected_evaluation:
            raise RuntimeError(f"evaluation contract mismatch: {manifest_path}")
        if existing.get("checkpoint", {}).get("sha256") != expected_checkpoint_hash:
            raise RuntimeError(f"evaluation checkpoint mismatch: {manifest_path}")
        return existing
    stats_path = manifest_path.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned evaluation stats: {stats_path}")
    environment_overrides = {"TASK3_OPPONENT_SEED": str(case["opponent_seed"])}
    if label == "v4":
        environment_overrides.update({
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_SEED": str(case["agent_seed"]),
        })
    else:
        environment_overrides.update({
            "MODEL_A_TASK3_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_TASK3_CONFIG_PATH": str(config_path),
            "MODEL_A_TASK3_SEED": str(case["agent_seed"]),
            "MODEL_A_TASK3_RESUME": "0",
        })
    environment = os.environ.copy()
    environment.update(environment_overrides)
    command = [
        sys.executable, "main.py", "play", "--agents", target,
        protocol["strata"][stratum]["opponent"], "--train", "0",
        "--continue-without-training", "--no-gui", "--scenario", "classic",
        "--n-rounds", str(protocol["strata"][stratum]["rounds_per_seed"]),
        "--seed", str(seed), "--save-stats", str(stats_path),
    ]
    manifest = {
        "schema_version": 1,
        "kind": "task3-four-arm-evaluation",
        "run_id": f"task3-four-arm-{stratum}-{label}-s{seed}",
        "label": label,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {"path": relative(checkpoint), "sha256": expected_checkpoint_hash},
        "agent_config": None if config_path is None else {
            "path": relative(config_path), "sha256": sha256_file(config_path),
        },
        "evaluation": expected_evaluation,
        "environment_overrides": environment_overrides,
        "command": command,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest_path, manifest)
    checkpoint_hash_before = sha256_file(checkpoint)
    config_hash_before = None if config_path is None else sha256_file(config_path)
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["exit_code"] = completed.returncode
    if completed.returncode == 0 and stats_path.is_file():
        unchanged = sha256_file(checkpoint) == checkpoint_hash_before
        unchanged &= config_path is None or sha256_file(config_path) == config_hash_before
        if unchanged:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            manifest["metrics_by_agent"] = metrics_by_agent(stats)
            manifest["target_metrics"] = manifest["metrics_by_agent"].get(target)
            manifest["status"] = "completed"
        else:
            manifest["status"] = "failed"
            manifest["error"] = "evaluation modified a frozen artifact"
    else:
        manifest["status"] = "failed"
    atomic_json(manifest_path, manifest)
    if manifest["status"] != "completed":
        raise RuntimeError(f"Task-3 evaluation failed for {stratum}/{label}/{seed}")
    return manifest


def summarize(rows: list[dict], label: str) -> dict:
    metrics = [row[label]["target_metrics"] for row in rows]
    rounds = sum(int(item["rounds"]) for item in metrics)
    steps = sum(int(item["steps"]) for item in metrics)
    keys = ("score", "coins", "kills", "crates", "bombs", "suicides", "invalid_actions")
    totals = {key: sum(int(item[key]) for item in metrics) for key in keys}
    result = {"rounds": rounds, **totals}
    result.update({f"{key}_per_round": totals[key] / rounds for key in keys})
    result["mean_decision_time_ms"] = (
        sum(float(item["mean_decision_time_ms"]) * int(item["steps"]) for item in metrics) / steps
        if steps else 0.0
    )
    result["max_seed_mean_decision_time_ms"] = max(
        float(item["mean_decision_time_ms"]) for item in metrics
    )
    return result


def compare(rows: list[dict], arm: str) -> dict:
    score_deltas, kill_deltas = [], []
    for row in rows:
        baseline, candidate = row["v4"]["target_metrics"], row[arm]["target_metrics"]
        score_deltas.append(float(candidate["score_per_round"]) - float(baseline["score_per_round"]))
        kill_deltas.append(
            int(candidate["kills"]) / int(candidate["rounds"])
            - int(baseline["kills"]) / int(baseline["rounds"])
        )
    return {
        "score_per_round_deltas": score_deltas,
        "kills_per_round_deltas": kill_deltas,
        "wins": sum(value > 0 for value in score_deltas),
        "ties": sum(value == 0 for value in score_deltas),
        "losses": sum(value < 0 for value in score_deltas),
        "worst_score_per_round_delta": min(score_deltas),
    }


def evaluate_arm(rows_by_stratum: dict, arm: str, gate: dict) -> dict:
    strata, all_rows = {}, []
    for name, rows in rows_by_stratum.items():
        all_rows.extend(rows)
        base, candidate = summarize(rows, "v4"), summarize(rows, arm)
        deltas = {
            key: candidate[f"{key}_per_round"] - base[f"{key}_per_round"]
            for key in ("score", "coins", "kills", "suicides", "invalid_actions")
        }
        checks = {
            "score_floor": deltas["score"] >= float(gate["minimum_stratum_score_per_round_delta"]),
            "kill_floor": deltas["kills"] >= float(gate["minimum_stratum_kills_per_round_delta"]),
            "suicide_control": deltas["suicides"] <= float(gate["maximum_stratum_suicides_per_round_increase"]),
            "invalid_control": deltas["invalid_actions"] <= float(gate["maximum_stratum_invalid_actions_per_round_increase"]),
        }
        strata[name] = {
            "v4": base, "candidate": candidate, "deltas_per_round": deltas,
            "paired": compare(rows, arm), "checks": checks, "passed": all(checks.values()),
        }
    base, candidate = summarize(all_rows, "v4"), summarize(all_rows, arm)
    comparison = compare(all_rows, arm)
    deltas = {
        key: candidate[f"{key}_per_round"] - base[f"{key}_per_round"]
        for key in ("score", "coins", "kills", "suicides", "invalid_actions")
    }
    checks = {
        "all_strata_pass": all(item["passed"] for item in strata.values()),
        "official_score_gain": deltas["score"] >= float(gate["minimum_overall_score_per_round_delta"]),
        "kill_gain": deltas["kills"] >= float(gate["minimum_overall_kills_per_round_delta"]),
        "coin_retention": deltas["coins"] >= -float(gate["maximum_overall_coins_per_round_regression"]),
        "paired_net_wins": comparison["wins"] - comparison["losses"] >= int(gate["minimum_paired_net_wins"]),
        "worst_cell": comparison["worst_score_per_round_delta"] >= -float(gate["maximum_single_cell_score_per_round_regression"]),
        "latency": candidate["max_seed_mean_decision_time_ms"] < float(gate["maximum_seed_mean_decision_time_ms"]),
    }
    return {
        "strata": strata,
        "overall": {"v4": base, "candidate": candidate, "deltas_per_round": deltas, "paired": comparison},
        "checks": checks,
        "passed": all(checks.values()),
    }


def select_arm(results: dict, tolerance: float) -> str | None:
    eligible = [arm for arm in ("A", "B", "C", "D") if results[arm]["passed"]]
    if not eligible:
        return None
    best = max(results[arm]["overall"]["deltas_per_round"]["score"] for arm in eligible)
    near_best = [
        arm for arm in eligible
        if results[arm]["overall"]["deltas_per_round"]["score"] >= best - tolerance
    ]
    return min(near_best, key=("A", "B", "C", "D").index)


def run(protocol: dict) -> tuple[dict, dict, dict]:
    training = {arm: train_one(protocol, arm) for arm in protocol["arms"]}
    checkpoints = {
        arm: (ROOT / training[arm]["checkpoint"]["path"]).resolve() for arm in training
    }
    rows_by_stratum = {}
    for stratum, settings in protocol["strata"].items():
        rows = []
        for case in settings["cases"]:
            row = {"seed_tuple": case}
            for label in ("v4", "A", "B", "C", "D"):
                manifest = evaluate_one(protocol, stratum, label, case, checkpoints)
                path = _evaluation_manifest(protocol, stratum, label, int(case["world_seed"]))
                row[label] = {
                    "manifest": relative(path),
                    "manifest_sha256": sha256_file(path),
                    "target_metrics": manifest["target_metrics"],
                }
            rows.append(row)
        rows_by_stratum[stratum] = rows
    results = {
        arm: evaluate_arm(rows_by_stratum, arm, protocol["pilot_gate"])
        for arm in protocol["arms"]
    }
    selected = select_arm(results, float(protocol["selection"]["simplicity_tolerance_score_per_round"]))
    return training, rows_by_stratum, {"arms": results, "selected_arm": selected}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    try:
        protocol = load_protocol(config_path)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    if not args.execute:
        evaluation_cases = sum(len(item["cases"]) for item in protocol["strata"].values())
        print(json.dumps({
            "mode": "dry-run",
            "run_id": protocol["run_id"],
            "task": 3,
            "task4": False,
            "training_arms": list(protocol["arms"]),
            "training_rounds": sum(
                json.loads((ROOT / item["config_path"]).read_text())["rounds"]
                for item in protocol["arms"].values()
            ),
            "evaluation_count": evaluation_cases * 5,
            "evaluation_rounds": sum(
                len(item["cases"]) * int(item["rounds_per_seed"]) * 5
                for item in protocol["strata"].values()
            ),
            "automatic_replication": False,
            "output_path": protocol["output_path"],
        }, indent=2, sort_keys=True))
        return 0
    output = ROOT / protocol["output_path"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite report: {output}")
    training, rows, result = run(protocol)
    selected = result["selected_arm"]
    decision = "advance_arm_for_multiseed" if selected is not None else "no_arm_selected"
    report = {
        "kind": "task3-four-arm-pilot-v1",
        "run_id": protocol["run_id"],
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": relative(config_path),
        "preregistration_sha256": sha256_file(config_path),
        "training_manifests": {
            arm: {
                "path": f"experiments/logs/runs/{manifest['run_id']}.json",
                "sha256": sha256_file(ROOT / "experiments/logs/runs" / f"{manifest['run_id']}.json"),
                "checkpoint": manifest["checkpoint"],
            }
            for arm, manifest in training.items()
        },
        "rows_by_stratum": rows,
        "result": result,
        "decision": decision,
        "selected_arm": selected,
        "automatic_replication_started": False,
        "task4_started": False,
        "next_step": protocol["decision_branches"][decision],
    }
    atomic_json(output, report)
    print(json.dumps({
        "output": str(output), "decision": decision,
        "selected_arm": selected, "next_step": report["next_step"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
