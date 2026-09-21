"""Run the preregistered, evaluation-only Task-3 v6s2 promotion gate."""

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


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _metrics_by_agent(stats: dict) -> dict[str, dict]:
    metrics = {}
    for name, raw in stats.get("by_agent", {}).items():
        rounds = int(raw.get("rounds", 0))
        steps = int(raw.get("steps", 0))
        metrics[name] = {
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
    return metrics


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("task") != 3 or protocol.get("includes_task4", True):
        raise ValueError("protocol must be isolated to Task 3")
    if protocol.get("training") or protocol.get("automatic_training"):
        raise ValueError("Task-3 checkpoint gate must remain evaluation-only")
    if protocol.get("candidate_selection_rule") != "locked-first-pilot-checkpoint-s7501":
        raise ValueError("candidate selection must remain locked to v6s2-s7501")
    if set(protocol.get("strata", {})) != {"peaceful", "coin_collector"}:
        raise ValueError("Task-3 gate requires separate peaceful and coin_collector strata")
    all_world_seeds = []
    for name, stratum in protocol["strata"].items():
        if int(stratum["rounds_per_seed"]) <= 0 or not stratum["cases"]:
            raise ValueError(f"invalid evaluation budget for {name}")
        for case in stratum["cases"]:
            if set(case) != {"world_seed", "agent_seed", "opponent_seed"}:
                raise ValueError(f"incomplete seed tuple for {name}")
            all_world_seeds.append(int(case["world_seed"]))
    if len(all_world_seeds) != len(set(all_world_seeds)):
        raise ValueError("world seeds must be unique across Task-3 strata")
    for relative, expected in protocol["bindings"].items():
        target = ROOT / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"binding mismatch: {relative}")
    return protocol


def _manifest_path(protocol: dict, stratum: str, arm: str, seed: int) -> Path:
    filename = f"task3-{stratum}-{arm}-s{seed}.json"
    return ROOT / protocol["evaluation_output_directory"] / filename


def _expected_evaluation(protocol: dict, stratum: str, arm: str, case: dict) -> dict:
    target = protocol["arms"][arm]["agent"]
    return {
        "target_agent": target,
        "agents": [target, protocol["strata"][stratum]["opponent"]],
        "scenario": "classic",
        "rounds": int(protocol["strata"][stratum]["rounds_per_seed"]),
        "seed": int(case["world_seed"]),
        "agent_seed": int(case["agent_seed"]),
        "opponent_seed": int(case["opponent_seed"]),
        "training_agents": 0,
        "continue_without_training": True,
    }


def _validate_manifest(protocol: dict, stratum: str, arm: str, case: dict, manifest: dict) -> None:
    if manifest.get("status") != "completed":
        raise RuntimeError(f"incomplete existing evaluation for {stratum}/{arm}")
    if manifest.get("evaluation") != _expected_evaluation(protocol, stratum, arm, case):
        raise RuntimeError(f"evaluation contract mismatch for {stratum}/{arm}")
    expected_hash = protocol["arms"][arm]["checkpoint_sha256"]
    if manifest.get("checkpoint", {}).get("sha256") != expected_hash:
        raise RuntimeError(f"checkpoint mismatch for {stratum}/{arm}")
    if arm == "candidate":
        expected_config = protocol["arms"][arm]["config_sha256"]
        if manifest.get("agent_config", {}).get("sha256") != expected_config:
            raise RuntimeError("candidate config mismatch")
    environment = manifest.get("environment_overrides", {})
    if environment.get("TASK3_OPPONENT_SEED") != str(case["opponent_seed"]):
        raise RuntimeError(f"opponent seed mismatch for {stratum}/{arm}")
    metrics = manifest.get("target_metrics")
    required = {
        "rounds", "score", "score_per_round", "coins", "kills", "crates", "bombs",
        "suicides", "invalid_actions", "steps", "mean_decision_time_ms",
    }
    if not isinstance(metrics, dict) or not required.issubset(metrics):
        raise RuntimeError(f"incomplete target metrics for {stratum}/{arm}")


def _run_one(protocol: dict, stratum: str, arm: str, case: dict) -> dict:
    seed = int(case["world_seed"])
    manifest_path = _manifest_path(protocol, stratum, arm, seed)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(protocol, stratum, arm, case, manifest)
        return manifest
    arm_config = protocol["arms"][arm]
    target = arm_config["agent"]
    stats_path = manifest_path.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"orphaned stats file would be overwritten: {stats_path}")
    checkpoint = (ROOT / arm_config["checkpoint_path"]).resolve()
    config_path = None
    environment_overrides = {"TASK3_OPPONENT_SEED": str(case["opponent_seed"])}
    if arm == "v4":
        environment_overrides.update({
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_SEED": str(case["agent_seed"]),
        })
    else:
        config_path = (ROOT / arm_config["config_path"]).resolve()
        environment_overrides.update({
            "MODEL_A_V6_STABLE_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V6_STABLE_CONFIG_PATH": str(config_path),
            "MODEL_A_V6_STABLE_SEED": str(case["agent_seed"]),
            "MODEL_A_V6_STABLE_RESUME": "0",
        })
    child_environment = os.environ.copy()
    child_environment.update(environment_overrides)
    command = [
        sys.executable, "main.py", "play", "--agents", target,
        protocol["strata"][stratum]["opponent"], "--train", "0",
        "--continue-without-training", "--no-gui", "--scenario", "classic",
        "--n-rounds", str(protocol["strata"][stratum]["rounds_per_seed"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "kind": "task3-evaluation",
        "run_id": f"task3-{stratum}-{arm}-s{seed}",
        "variant": f"task3-{stratum}-{arm}",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository_commit": commit or None,
        "checkpoint": {"path": _relative(checkpoint), "sha256": sha256_file(checkpoint)},
        "agent_config": None if config_path is None else {
            "path": _relative(config_path), "sha256": sha256_file(config_path),
        },
        "evaluation": _expected_evaluation(protocol, stratum, arm, case),
        "environment_overrides": environment_overrides,
        "command": command,
        "artifacts": {"raw_stats": _relative(stats_path)},
    }
    atomic_json(manifest_path, manifest)
    checkpoint_hash_before = sha256_file(checkpoint)
    config_hash_before = None if config_path is None else sha256_file(config_path)
    completed = subprocess.run(command, cwd=ROOT, env=child_environment, check=False)
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["exit_code"] = completed.returncode
    if completed.returncode == 0 and stats_path.is_file():
        if sha256_file(checkpoint) != checkpoint_hash_before:
            manifest["status"] = "failed"
            manifest["error"] = "evaluation modified the checkpoint"
        elif config_path is not None and sha256_file(config_path) != config_hash_before:
            manifest["status"] = "failed"
            manifest["error"] = "evaluation modified the candidate config"
        else:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            manifest["metrics_by_agent"] = _metrics_by_agent(stats)
            manifest["target_metrics"] = manifest["metrics_by_agent"].get(target)
            manifest["status"] = "completed"
    else:
        manifest["status"] = "failed"
    atomic_json(manifest_path, manifest)
    if manifest["status"] != "completed":
        raise RuntimeError(f"evaluation failed ({completed.returncode}): {' '.join(command)}")
    _validate_manifest(protocol, stratum, arm, case, manifest)
    return manifest


def summarize(rows: list[dict], arm: str) -> dict:
    metrics = [row[arm]["target_metrics"] for row in rows]
    rounds = sum(int(item["rounds"]) for item in metrics)
    steps = sum(int(item["steps"]) for item in metrics)
    totals = {
        key: sum(int(item[key]) for item in metrics)
        for key in ("score", "coins", "kills", "crates", "bombs", "suicides", "invalid_actions")
    }
    summary = {"rounds": rounds, **totals}
    for key in ("score", "coins", "kills", "crates", "bombs", "suicides", "invalid_actions"):
        summary[f"{key}_per_round"] = totals[key] / rounds if rounds else 0.0
    summary["mean_decision_time_ms"] = (
        sum(float(item["mean_decision_time_ms"]) * int(item["steps"]) for item in metrics) / steps
        if steps else 0.0
    )
    summary["max_seed_mean_decision_time_ms"] = max(
        float(item["mean_decision_time_ms"]) for item in metrics
    )
    return summary


def compare(rows: list[dict]) -> dict:
    score_deltas = []
    kill_deltas = []
    for row in rows:
        v4 = row["v4"]["target_metrics"]
        candidate = row["candidate"]["target_metrics"]
        score_deltas.append(float(candidate["score_per_round"]) - float(v4["score_per_round"]))
        kill_deltas.append(
            int(candidate["kills"]) / int(candidate["rounds"])
            - int(v4["kills"]) / int(v4["rounds"])
        )
    return {
        "score_per_round_deltas": score_deltas,
        "kills_per_round_deltas": kill_deltas,
        "wins": sum(delta > 0 for delta in score_deltas),
        "ties": sum(delta == 0 for delta in score_deltas),
        "losses": sum(delta < 0 for delta in score_deltas),
        "worst_score_per_round_delta": min(score_deltas),
    }


def evaluate_gate(rows_by_stratum: dict[str, list[dict]], gate: dict) -> dict:
    stratum_results = {}
    all_rows = []
    for name, rows in rows_by_stratum.items():
        all_rows.extend(rows)
        v4 = summarize(rows, "v4")
        candidate = summarize(rows, "candidate")
        comparison = compare(rows)
        score_delta = candidate["score_per_round"] - v4["score_per_round"]
        kill_delta = candidate["kills_per_round"] - v4["kills_per_round"]
        checks = {
            "score_non_regression": score_delta >= float(gate["minimum_stratum_score_per_round_delta"]),
            "kill_non_regression": kill_delta >= float(gate["minimum_stratum_kills_per_round_delta"]),
        }
        stratum_results[name] = {
            "v4": v4,
            "candidate": candidate,
            "paired": comparison,
            "score_per_round_delta": score_delta,
            "kills_per_round_delta": kill_delta,
            "checks": checks,
            "passed": all(checks.values()),
        }

    v4 = summarize(all_rows, "v4")
    candidate = summarize(all_rows, "candidate")
    comparison = compare(all_rows)
    deltas = {
        key: candidate[f"{key}_per_round"] - v4[f"{key}_per_round"]
        for key in ("score", "coins", "kills", "crates", "bombs", "suicides", "invalid_actions")
    }
    checks = {
        "all_strata_pass": all(item["passed"] for item in stratum_results.values()),
        "official_score_gain": deltas["score"] >= float(gate["minimum_overall_score_per_round_delta"]),
        "kill_gain": deltas["kills"] >= float(gate["minimum_overall_kills_per_round_delta"]),
        "coin_retention": deltas["coins"] >= -float(gate["maximum_overall_coins_per_round_regression"]),
        "paired_net_wins": comparison["wins"] - comparison["losses"] >= int(gate["minimum_paired_net_wins"]),
        "worst_cell": comparison["worst_score_per_round_delta"] >= -float(gate["maximum_single_cell_score_per_round_regression"]),
        "suicide_control": deltas["suicides"] <= float(gate["maximum_suicides_per_round_increase"]),
        "invalid_control": deltas["invalid_actions"] <= float(gate["maximum_invalid_actions_per_round_increase"]),
        "latency": candidate["max_seed_mean_decision_time_ms"] < float(gate["maximum_seed_mean_decision_time_ms"]),
    }
    return {
        "strata": stratum_results,
        "overall": {"v4": v4, "candidate": candidate, "deltas_per_round": deltas, "paired": comparison},
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(protocol: dict) -> tuple[dict[str, list[dict]], dict]:
    rows_by_stratum = {}
    for stratum, settings in protocol["strata"].items():
        rows = []
        for case in settings["cases"]:
            row = {"seed_tuple": case}
            for arm in ("v4", "candidate"):
                manifest = _run_one(protocol, stratum, arm, case)
                path = _manifest_path(protocol, stratum, arm, int(case["world_seed"]))
                row[arm] = {
                    "manifest": str(path.relative_to(ROOT)),
                    "manifest_sha256": sha256_file(path),
                    "target_metrics": manifest["target_metrics"],
                }
            rows.append(row)
        rows_by_stratum[stratum] = rows
    return rows_by_stratum, evaluate_gate(rows_by_stratum, protocol["gate"])


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
        case_count = sum(len(item["cases"]) for item in protocol["strata"].values())
        print(json.dumps({
            "mode": "dry-run",
            "run_id": protocol["run_id"],
            "task": 3,
            "strata": protocol["strata"],
            "evaluation_count": case_count * 2,
            "round_count": sum(
                len(item["cases"]) * int(item["rounds_per_seed"]) * 2
                for item in protocol["strata"].values()
            ),
            "training": False,
            "task4": False,
            "output_path": protocol["output_path"],
        }, indent=2, sort_keys=True))
        return 0

    output = ROOT / protocol["output_path"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite report: {output}")
    rows_by_stratum, result = run(protocol)
    decision = "select_v6s2_task3_incumbent" if result["passed"] else "retain_v4_prepare_task3_training"
    report = {
        "kind": "task3-v6s2-evaluation-only-promotion-gate-v1",
        "run_id": protocol["run_id"],
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": str(config_path.relative_to(ROOT)),
        "preregistration_sha256": sha256_file(config_path),
        "rows_by_stratum": rows_by_stratum,
        "result": result,
        "decision": decision,
        "selected_arm": "candidate" if result["passed"] else "v4",
        "automatic_training_started": False,
        "task4_started": False,
        "next_step": protocol["decision_branches"][decision],
    }
    atomic_json(output, report)
    print(json.dumps({
        "output": str(output), "decision": decision, "next_step": report["next_step"]
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
