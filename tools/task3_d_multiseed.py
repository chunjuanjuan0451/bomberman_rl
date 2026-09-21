"""Train and confirm three fresh replicas of the selected Task-3 D recipe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.task3_four_arm import (  # noqa: E402
    atomic_json, metrics_by_agent, relative, sha256_file, summarize,
)


REPLICAS = ("R1", "R2", "R3")
METRIC_KEYS = ("score", "coins", "kills", "suicides", "invalid_actions")


def _threshold(gate: dict, key: str) -> Fraction:
    value = gate[key]
    if not isinstance(value, str) or "/" not in value:
        raise ValueError(f"exact threshold {key} must be a rational string")
    result = Fraction(value)
    if result.denominator <= 0:
        raise ValueError(f"invalid exact threshold: {key}")
    return result


def _rate(summary: dict, key: str) -> Fraction:
    rounds = int(summary["rounds"])
    if rounds <= 0:
        raise ValueError("cannot compute a rate without rounds")
    return Fraction(int(summary[key]), rounds)


def _delta(base: dict, candidate: dict, key: str) -> Fraction:
    return _rate(candidate, key) - _rate(base, key)


def _deltas(base: dict, candidate: dict) -> tuple[dict, dict]:
    exact = {key: _delta(base, candidate, key) for key in METRIC_KEYS}
    return (
        {key: float(value) for key, value in exact.items()},
        {key: str(value) for key, value in exact.items()},
    )


def _algorithm_contract(config: dict) -> dict:
    return {
        "arm": config["arm"],
        "reward_profile": config["reward_profile"],
        "action_features": config["action_features"],
        "n_step": config["n_step"],
        "agents": config["agents"],
        "scenario": config["scenario"],
        "rounds": config["rounds"],
        "resume": config["resume"],
        "parent_checkpoint": config["parent_checkpoint"],
        "parent_sha256": config["parent_sha256"],
        "hyperparameters": config["hyperparameters"],
    }


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("kind") != "task3-d-multiseed-confirmation-v1":
        raise ValueError("unexpected Task-3 confirmation protocol kind")
    if protocol.get("task") != 3 or protocol.get("includes_task4", True):
        raise ValueError("D confirmation must remain isolated to Task 3")
    if not protocol.get("training") or protocol.get("automatic_followup"):
        raise ValueError("confirmation requires training and forbids automatic follow-up")
    if list(protocol.get("replicas", {})) != list(REPLICAS):
        raise ValueError("confirmation requires ordered replicas R1, R2, R3")
    for relative_path, expected in protocol["bindings"].items():
        target = ROOT / relative_path
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"binding mismatch: {relative_path}")

    discovery = protocol["discovery"]
    report_path = ROOT / discovery["report_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if sha256_file(report_path) != discovery["report_sha256"]:
        raise ValueError("discovery report hash mismatch")
    if report.get("selected_arm") != "D" or report.get("decision") != "advance_arm_for_multiseed":
        raise ValueError("discovery report does not select D")
    erratum_path = ROOT / discovery["erratum_path"]
    if sha256_file(erratum_path) != discovery["erratum_sha256"]:
        raise ValueError("discovery erratum hash mismatch")
    discovery_checkpoint = ROOT / discovery["checkpoint_path"]
    if sha256_file(discovery_checkpoint) != discovery["checkpoint_sha256"]:
        raise ValueError("discovery D checkpoint hash mismatch")
    v4_path = ROOT / protocol["v4"]["checkpoint_path"]
    if sha256_file(v4_path) != protocol["v4"]["checkpoint_sha256"]:
        raise ValueError("frozen v4 checkpoint hash mismatch")

    from agent_code.model_a_task3.config import load_config
    pilot_config, _, pilot_hash = load_config(ROOT / discovery["config_path"])
    if pilot_hash != discovery["config_sha256"] or pilot_config["arm"] != "D":
        raise ValueError("pilot D config mismatch")
    pilot_contract = _algorithm_contract(pilot_config)
    run_ids, checkpoints, training_seeds = set(), set(), set()
    for replica, spec in protocol["replicas"].items():
        config, _, config_hash = load_config(ROOT / spec["config_path"])
        if config_hash != spec["config_sha256"] or _algorithm_contract(config) != pilot_contract:
            raise ValueError(f"replica {replica} changes the selected D recipe")
        run_ids.add(config["run_id"])
        checkpoints.add(config["checkpoint_path"])
        training_seeds.add((config["seed"], config["agent_seed"], config["opponent_seed"]))
    if len(run_ids) != 3 or len(checkpoints) != 3 or len(training_seeds) != 3:
        raise ValueError("replica run IDs, checkpoints, and training seed tuples must be unique")

    if set(protocol.get("strata", {})) != {"peaceful", "coin_collector"}:
        raise ValueError("confirmation requires separate peaceful and coin_collector strata")
    world_seeds, agent_seeds, opponent_seeds = [], [], []
    for name, stratum in protocol["strata"].items():
        if int(stratum.get("rounds_per_seed", 0)) <= 0 or len(stratum.get("cases", [])) != 6:
            raise ValueError(f"{name} requires six nonempty evaluation cases")
        for case in stratum["cases"]:
            if set(case) != {"world_seed", "agent_seed", "opponent_seed"}:
                raise ValueError(f"incomplete evaluation seed tuple for {name}")
            world_seeds.append(int(case["world_seed"]))
            agent_seeds.append(int(case["agent_seed"]))
            opponent_seeds.append(int(case["opponent_seed"]))
    if not (len(set(world_seeds)) == len(set(agent_seeds)) == len(set(opponent_seeds)) == 12):
        raise ValueError("all confirmation evaluation seeds must be unique")
    excluded = set(int(value) for value in protocol["excluded_world_seeds"])
    if excluded.intersection(world_seeds) or excluded.intersection(seed[0] for seed in training_seeds):
        raise ValueError("confirmation reuses a discovery world seed")

    exact_keys = {
        "minimum_stratum_score_delta", "minimum_stratum_kills_delta",
        "maximum_stratum_suicides_increase", "maximum_stratum_invalid_increase",
        "minimum_overall_score_delta", "minimum_overall_kills_delta",
        "maximum_overall_coin_regression", "maximum_single_cell_score_regression",
        "maximum_overall_suicides_increase", "maximum_overall_invalid_increase",
    }
    for section in ("replica_gate", "replica_floor_gate", "pooled_gate"):
        gate = protocol[section]
        for key in exact_keys.intersection(gate):
            _threshold(gate, key)
    if int(protocol["pooled_gate"]["minimum_supportive_replicas"]) != 2:
        raise ValueError("confirmation requires at least two supportive replicas")
    if protocol.get("selection", {}).get("order") != list(REPLICAS):
        raise ValueError("replica selection order must be fixed as R1, R2, R3")
    return protocol


def _load_completed(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != kind or payload.get("status") != "completed":
        raise RuntimeError(f"refusing incomplete or incompatible artifact: {path}")
    return payload


def _validate_checkpoint(path: Path, config: dict, config_hash: str) -> bool:
    from agent_code.model_a_task3.callbacks import _torch_load
    from agent_code.model_a_task3.config import architecture_name
    payload = _torch_load(path)
    return (
        payload.get("arm") == "D"
        and payload.get("architecture") == architecture_name(config)
        and payload.get("config_sha256") == config_hash
        and payload.get("parent_sha256") == config["parent_sha256"]
        and int(payload.get("completed_rounds", 0)) == int(config["rounds"])
    )


def train_one(protocol: dict, replica: str) -> dict:
    spec = protocol["replicas"][replica]
    config_path = (ROOT / spec["config_path"]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256_file(config_path)
    checkpoint = (ROOT / config["checkpoint_path"]).resolve()
    manifest_path = ROOT / "experiments/logs/runs" / f"{config['run_id']}.json"
    stats_path = manifest_path.with_suffix(".stats.json")
    existing = _load_completed(manifest_path, "task3-d-confirm-training")
    if existing is not None:
        if existing.get("replica") != replica or existing.get("config_sha256") != config_hash:
            raise RuntimeError(f"training manifest mismatch for {replica}")
        if not checkpoint.is_file() or sha256_file(checkpoint) != existing["checkpoint"]["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch for {replica}")
        if not _validate_checkpoint(checkpoint, config, config_hash):
            raise RuntimeError(f"checkpoint metadata mismatch for {replica}")
        return existing
    if checkpoint.exists() or stats_path.exists():
        raise RuntimeError(f"refusing orphaned confirmation artifact for {replica}")

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
        "schema_version": 1, "kind": "task3-d-confirm-training",
        "run_id": config["run_id"], "replica": replica, "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": relative(config_path), "config_sha256": config_hash,
        "config": config, "checkpoint": {"path": relative(checkpoint), "sha256": None},
        "environment_overrides": environment_overrides, "command": command,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest_path, manifest)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["exit_code"] = completed.returncode
    if completed.returncode == 0 and checkpoint.is_file() and stats_path.is_file():
        try:
            valid = _validate_checkpoint(checkpoint, config, config_hash)
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
        raise RuntimeError(f"Task-3 D confirmation training failed for {replica}")
    return manifest


def _evaluation_manifest(protocol: dict, stratum: str, label: str, seed: int) -> Path:
    name = f"task3-d-confirm-{stratum}-{label}-s{seed}.json"
    return ROOT / protocol["evaluation_output_directory"] / name


def evaluate_one(
    protocol: dict, stratum: str, label: str, case: dict, checkpoints: dict,
) -> dict:
    seed = int(case["world_seed"])
    manifest_path = _evaluation_manifest(protocol, stratum, label, seed)
    target = "model_a_dqn" if label == "v4" else "model_a_task3"
    checkpoint = (
        ROOT / protocol["v4"]["checkpoint_path"] if label == "v4" else checkpoints[label]
    ).resolve()
    config_path = None if label == "v4" else (
        ROOT / protocol["replicas"][label]["config_path"]
    ).resolve()
    checkpoint_hash = (
        protocol["v4"]["checkpoint_sha256"] if label == "v4" else sha256_file(checkpoint)
    )
    expected = {
        "target_agent": target,
        "agents": [target, protocol["strata"][stratum]["opponent"]],
        "scenario": "classic",
        "rounds": int(protocol["strata"][stratum]["rounds_per_seed"]),
        "seed": seed, "agent_seed": int(case["agent_seed"]),
        "opponent_seed": int(case["opponent_seed"]), "training_agents": 0,
        "continue_without_training": True,
    }
    existing = _load_completed(manifest_path, "task3-d-confirm-evaluation")
    if existing is not None:
        if existing.get("evaluation") != expected:
            raise RuntimeError(f"evaluation contract mismatch: {manifest_path}")
        if existing.get("checkpoint", {}).get("sha256") != checkpoint_hash:
            raise RuntimeError(f"evaluation checkpoint mismatch: {manifest_path}")
        expected_config_hash = None if config_path is None else sha256_file(config_path)
        actual_config_hash = None if config_path is None else existing.get("agent_config", {}).get("sha256")
        if actual_config_hash != expected_config_hash:
            raise RuntimeError(f"evaluation config mismatch: {manifest_path}")
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
        "--n-rounds", str(expected["rounds"]), "--seed", str(seed),
        "--save-stats", str(stats_path),
    ]
    manifest = {
        "schema_version": 1, "kind": "task3-d-confirm-evaluation",
        "run_id": f"task3-d-confirm-{stratum}-{label}-s{seed}",
        "label": label, "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
        "agent_config": None if config_path is None else {
            "path": relative(config_path), "sha256": sha256_file(config_path),
        },
        "evaluation": expected, "environment_overrides": environment_overrides,
        "command": command, "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest_path, manifest)
    checkpoint_before = sha256_file(checkpoint)
    config_before = None if config_path is None else sha256_file(config_path)
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["exit_code"] = completed.returncode
    if completed.returncode == 0 and stats_path.is_file():
        unchanged = sha256_file(checkpoint) == checkpoint_before
        unchanged &= config_path is None or sha256_file(config_path) == config_before
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
        raise RuntimeError(f"Task-3 D evaluation failed for {stratum}/{label}/{seed}")
    return manifest


def compare(rows: list[dict], label: str) -> dict:
    score_deltas, kill_deltas = [], []
    for row in rows:
        base, candidate = row["v4"]["target_metrics"], row[label]["target_metrics"]
        score_deltas.append(
            Fraction(int(candidate["score"]), int(candidate["rounds"]))
            - Fraction(int(base["score"]), int(base["rounds"]))
        )
        kill_deltas.append(
            Fraction(int(candidate["kills"]), int(candidate["rounds"]))
            - Fraction(int(base["kills"]), int(base["rounds"]))
        )
    return {
        "score_per_round_deltas": [float(value) for value in score_deltas],
        "exact_score_per_round_deltas": [str(value) for value in score_deltas],
        "kills_per_round_deltas": [float(value) for value in kill_deltas],
        "exact_kills_per_round_deltas": [str(value) for value in kill_deltas],
        "wins": sum(value > 0 for value in score_deltas),
        "ties": sum(value == 0 for value in score_deltas),
        "losses": sum(value < 0 for value in score_deltas),
        "worst_score_per_round_delta": float(min(score_deltas)),
        "exact_worst_score_per_round_delta": str(min(score_deltas)),
    }


def _stratum_result(rows: list[dict], label: str, gate: dict) -> dict:
    base, candidate = summarize(rows, "v4"), summarize(rows, label)
    deltas, exact = _deltas(base, candidate)
    checks = {
        "score_floor": _delta(base, candidate, "score") >= _threshold(gate, "minimum_stratum_score_delta"),
        "kill_floor": _delta(base, candidate, "kills") >= _threshold(gate, "minimum_stratum_kills_delta"),
        "suicide_control": _delta(base, candidate, "suicides") <= _threshold(gate, "maximum_stratum_suicides_increase"),
        "invalid_control": _delta(base, candidate, "invalid_actions") <= _threshold(gate, "maximum_stratum_invalid_increase"),
    }
    return {
        "v4": base, "candidate": candidate, "deltas_per_round": deltas,
        "exact_deltas_per_round": exact, "paired": compare(rows, label),
        "checks": checks, "passed": all(checks.values()),
    }


def evaluate_replica(rows_by_stratum: dict, label: str, gate: dict) -> dict:
    strata = {
        name: _stratum_result(rows, label, gate)
        for name, rows in rows_by_stratum.items()
    }
    all_rows = [row for rows in rows_by_stratum.values() for row in rows]
    base, candidate = summarize(all_rows, "v4"), summarize(all_rows, label)
    deltas, exact = _deltas(base, candidate)
    paired = compare(all_rows, label)
    checks = {
        "all_strata_pass": all(item["passed"] for item in strata.values()),
        "official_score_gain": _delta(base, candidate, "score") >= _threshold(gate, "minimum_overall_score_delta"),
        "kill_non_regression": _delta(base, candidate, "kills") >= _threshold(gate, "minimum_overall_kills_delta"),
        "coin_retention": _delta(base, candidate, "coins") >= -_threshold(gate, "maximum_overall_coin_regression"),
        "paired_net_wins": paired["wins"] - paired["losses"] >= int(gate["minimum_paired_net_wins"]),
        "worst_cell": Fraction(paired["exact_worst_score_per_round_delta"]) >= -_threshold(gate, "maximum_single_cell_score_regression"),
        "latency": candidate["max_seed_mean_decision_time_ms"] < float(gate["maximum_seed_mean_decision_time_ms"]),
    }
    return {
        "strata": strata,
        "overall": {
            "v4": base, "candidate": candidate, "deltas_per_round": deltas,
            "exact_deltas_per_round": exact, "paired": paired,
        },
        "checks": checks, "supportive": all(checks.values()),
    }


def _passes_floor(replica_result: dict, gate: dict) -> tuple[bool, dict]:
    overall = replica_result["overall"]
    base, candidate = overall["v4"], overall["candidate"]
    checks = {
        "score_floor": _delta(base, candidate, "score") >= _threshold(gate, "minimum_overall_score_delta"),
        "kill_floor": _delta(base, candidate, "kills") >= _threshold(gate, "minimum_overall_kills_delta"),
    }
    return all(checks.values()), checks


def evaluate_pooled(rows_by_stratum: dict, replicas: dict, protocol: dict) -> dict:
    pooled_rows = {}
    for stratum, rows in rows_by_stratum.items():
        pooled_rows[stratum] = []
        for replica in REPLICAS:
            for row in rows:
                pooled_rows[stratum].append({
                    "v4": row["v4"], "candidate": row[replica],
                    "replica": replica, "seed_tuple": row["seed_tuple"],
                })
    gate = protocol["pooled_gate"]
    strata = {
        name: _stratum_result(rows, "candidate", gate)
        for name, rows in pooled_rows.items()
    }
    all_rows = [row for rows in pooled_rows.values() for row in rows]
    base, candidate = summarize(all_rows, "v4"), summarize(all_rows, "candidate")
    deltas, exact = _deltas(base, candidate)
    paired = compare(all_rows, "candidate")
    floor_results = {
        replica: _passes_floor(result, protocol["replica_floor_gate"])
        for replica, result in replicas.items()
    }
    supportive = [replica for replica in REPLICAS if replicas[replica]["supportive"]]
    checks = {
        "all_pooled_strata_pass": all(item["passed"] for item in strata.values()),
        "official_score_gain": _delta(base, candidate, "score") >= _threshold(gate, "minimum_overall_score_delta"),
        "kill_gain": _delta(base, candidate, "kills") >= _threshold(gate, "minimum_overall_kills_delta"),
        "coin_retention": _delta(base, candidate, "coins") >= -_threshold(gate, "maximum_overall_coin_regression"),
        "suicide_control": _delta(base, candidate, "suicides") <= _threshold(gate, "maximum_overall_suicides_increase"),
        "invalid_control": _delta(base, candidate, "invalid_actions") <= _threshold(gate, "maximum_overall_invalid_increase"),
        "paired_net_wins": paired["wins"] - paired["losses"] >= int(gate["minimum_paired_net_wins"]),
        "worst_cell": Fraction(paired["exact_worst_score_per_round_delta"]) >= -_threshold(gate, "maximum_single_cell_score_regression"),
        "latency": candidate["max_seed_mean_decision_time_ms"] < float(gate["maximum_seed_mean_decision_time_ms"]),
        "supportive_replica_count": len(supportive) >= int(gate["minimum_supportive_replicas"]),
        "all_replica_floors": all(item[0] for item in floor_results.values()),
    }
    selected = supportive[0] if all(checks.values()) else None
    return {
        "strata": strata,
        "overall": {
            "v4": base, "candidate": candidate, "deltas_per_round": deltas,
            "exact_deltas_per_round": exact, "paired": paired,
        },
        "supportive_replicas": supportive,
        "supportive_replica_count": len(supportive),
        "replica_floor_checks": {
            replica: {"passed": passed, "checks": checks_for_replica}
            for replica, (passed, checks_for_replica) in floor_results.items()
        },
        "checks": checks, "passed": all(checks.values()),
        "selected_replica": selected,
    }


def run(protocol: dict) -> tuple[dict, dict, dict]:
    training = {replica: train_one(protocol, replica) for replica in REPLICAS}
    checkpoints = {
        replica: (ROOT / training[replica]["checkpoint"]["path"]).resolve()
        for replica in REPLICAS
    }
    rows_by_stratum = {}
    for stratum, settings in protocol["strata"].items():
        rows = []
        for case in settings["cases"]:
            row = {"seed_tuple": case}
            for label in ("v4", *REPLICAS):
                manifest = evaluate_one(protocol, stratum, label, case, checkpoints)
                path = _evaluation_manifest(protocol, stratum, label, int(case["world_seed"]))
                row[label] = {
                    "manifest": relative(path), "manifest_sha256": sha256_file(path),
                    "target_metrics": manifest["target_metrics"],
                }
            rows.append(row)
        rows_by_stratum[stratum] = rows
    replicas = {
        replica: evaluate_replica(rows_by_stratum, replica, protocol["replica_gate"])
        for replica in REPLICAS
    }
    pooled = evaluate_pooled(rows_by_stratum, replicas, protocol)
    return training, rows_by_stratum, {"replicas": replicas, "pooled": pooled}


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
        cases = sum(len(item["cases"]) for item in protocol["strata"].values())
        print(json.dumps({
            "mode": "dry-run", "run_id": protocol["run_id"], "task": 3,
            "task4": False, "training_replicas": list(REPLICAS),
            "training_rounds": sum(
                json.loads((ROOT / spec["config_path"]).read_text())["rounds"]
                for spec in protocol["replicas"].values()
            ),
            "evaluation_count": cases * 4,
            "evaluation_rounds": sum(
                len(item["cases"]) * int(item["rounds_per_seed"]) * 4
                for item in protocol["strata"].values()
            ),
            "discovery_data_in_confirmation": False,
            "automatic_followup": False, "output_path": protocol["output_path"],
        }, indent=2, sort_keys=True))
        return 0

    output = ROOT / protocol["output_path"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite report: {output}")
    training, rows, result = run(protocol)
    selected = result["pooled"]["selected_replica"]
    decision = "confirm_d_for_official_opponent_gate" if selected else "d_not_confirmed_continue_task3"
    report = {
        "schema_version": 1, "kind": "task3-d-multiseed-confirmation-v1",
        "run_id": protocol["run_id"], "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": relative(config_path),
        "preregistration_sha256": sha256_file(config_path),
        "discovery_report_used_for_metrics": False,
        "training_manifests": {
            replica: {
                "path": f"experiments/logs/runs/{manifest['run_id']}.json",
                "sha256": sha256_file(ROOT / "experiments/logs/runs" / f"{manifest['run_id']}.json"),
                "checkpoint": manifest["checkpoint"],
            }
            for replica, manifest in training.items()
        },
        "rows_by_stratum": rows, "result": result, "decision": decision,
        "selected_replica": selected, "automatic_followup_started": False,
        "official_unmodified_opponent_gate_started": False, "task4_started": False,
        "next_step": protocol["decision_branches"][decision],
    }
    atomic_json(output, report)
    print(json.dumps({
        "output": str(output), "decision": decision,
        "selected_replica": selected, "next_step": report["next_step"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
