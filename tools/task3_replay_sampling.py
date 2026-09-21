"""Run the preregistered matched-seed Task-3 replay-sampling experiment."""

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

from tools.task3_four_arm import atomic_json, metrics_by_agent, relative, sha256_file, summarize  # noqa: E402


REPLICAS = ("R1", "R2", "R3")
LABELS = ("v4", "C1", "S1", "C2", "S2", "C3", "S3")
METRICS = ("score", "coins", "kills", "suicides", "invalid_actions")


def _threshold(gate: dict, key: str) -> Fraction:
    value = gate[key]
    if not isinstance(value, str) or "/" not in value:
        raise ValueError(f"exact threshold {key} must be a rational string")
    return Fraction(value)


def _rate(summary: dict, key: str) -> Fraction:
    return Fraction(int(summary[key]), int(summary["rounds"]))


def _delta(base: dict, candidate: dict, key: str) -> Fraction:
    return _rate(candidate, key) - _rate(base, key)


def _contract(config: dict) -> dict:
    return {
        key: config[key] for key in (
            "agent", "agents", "task", "includes_task4", "scenario", "rounds",
            "resume", "enabled_for_training", "arm", "reward_profile",
            "action_features", "n_step", "kill_causal_window", "parent_checkpoint",
            "parent_sha256", "hyperparameters",
        )
    }


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("kind") != "task3-replay-sampling-matched-v1":
        raise ValueError("unexpected replay-sampling protocol kind")
    if protocol.get("task") != 3 or protocol.get("includes_task4", True):
        raise ValueError("replay-sampling experiment must remain isolated to Task 3")
    if not protocol.get("training") or protocol.get("automatic_followup"):
        raise ValueError("formal training is required and automatic follow-up is forbidden")
    if list(protocol.get("replicas", {})) != list(REPLICAS):
        raise ValueError("ordered replicas R1, R2, R3 are required")
    for relative_path, expected in protocol["bindings"].items():
        target = ROOT / relative_path
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"binding mismatch: {relative_path}")
    v4 = ROOT / protocol["v4"]["checkpoint_path"]
    if not v4.is_file() or sha256_file(v4) != protocol["v4"]["checkpoint_sha256"]:
        raise ValueError("frozen v4 checkpoint mismatch")

    from agent_code.model_a_task3_replay.config import load_config
    contracts, run_ids, checkpoints, training_tuples = [], set(), set(), []
    for index, (replica, pair) in enumerate(protocol["replicas"].items(), start=1):
        pair_configs = {}
        for profile in ("control", "stratified"):
            spec = pair[profile]
            config, _, config_hash = load_config(ROOT / spec["config_path"])
            if config_hash != spec["config_sha256"]:
                raise ValueError(f"config hash mismatch: {replica}/{profile}")
            expected_profile = "uniform" if profile == "control" else "kill-causal-25"
            expected_fraction = 0.0 if profile == "control" else 0.25
            if config["sampling_profile"] != expected_profile or config["kill_causal_fraction"] != expected_fraction:
                raise ValueError(f"sampling treatment mismatch: {replica}/{profile}")
            if config["parent_sha256"] != protocol["v4"]["checkpoint_sha256"]:
                raise ValueError(f"parent mismatch: {replica}/{profile}")
            pair_configs[profile] = config
            contracts.append(_contract(config))
            if config["run_id"] in run_ids or config["checkpoint_path"] in checkpoints:
                raise ValueError("training artifacts must be unique")
            run_ids.add(config["run_id"])
            checkpoints.add(config["checkpoint_path"])
        control, stratified = pair_configs["control"], pair_configs["stratified"]
        if _contract(control) != _contract(stratified):
            raise ValueError(f"paired algorithms differ beyond replay sampling: {replica}")
        seed_tuple = (control["seed"], control["agent_seed"], control["opponent_seed"])
        if seed_tuple != (stratified["seed"], stratified["agent_seed"], stratified["opponent_seed"]):
            raise ValueError(f"training seeds are not matched: {replica}")
        training_tuples.append(seed_tuple)
        if pair.get("labels") != {"control": f"C{index}", "stratified": f"S{index}"}:
            raise ValueError(f"unexpected fixed labels: {replica}")
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("replicas alter the frozen D algorithm")
    if len(set(training_tuples)) != 3:
        raise ValueError("three fresh training seed tuples are required")

    if set(protocol.get("strata", {})) != {"peaceful", "coin_collector"}:
        raise ValueError("peaceful and coin_collector strata are required")
    evaluation_tuples = []
    for name, stratum in protocol["strata"].items():
        if int(stratum.get("rounds_per_seed", 0)) != 50 or len(stratum.get("cases", [])) != 4:
            raise ValueError(f"{name} must contain four 50-round cases")
        for case in stratum["cases"]:
            if set(case) != {"world_seed", "agent_seed", "opponent_seed"}:
                raise ValueError(f"incomplete evaluation seed tuple: {name}")
            evaluation_tuples.append(tuple(int(case[key]) for key in ("world_seed", "agent_seed", "opponent_seed")))
    if len(set(evaluation_tuples)) != 8:
        raise ValueError("all evaluation seed tuples must be unique")
    floor = int(protocol["fresh_seed_floor"])
    if any(any(value < floor for value in item) for item in training_tuples + evaluation_tuples):
        raise ValueError("a preregistered fresh seed is below the freshness floor")

    expected_budget = {"training_runs": 6, "training_rounds": 1800, "evaluation_runs": 56, "evaluation_rounds": 2800}
    if protocol.get("formal_budget") != expected_budget:
        raise ValueError("formal budget must be exactly 6/1800 training and 56/2800 evaluation")
    for section in ("vs_v4_gate", "vs_control_gate", "replica_gate", "replica_v4_floor"):
        for key, value in protocol[section].items():
            if key.startswith(("minimum_", "maximum_")) and key not in {
                "minimum_paired_net_wins", "minimum_supportive_replicas", "maximum_seed_mean_decision_time_ms",
            }:
                if not isinstance(value, str) or "/" not in value:
                    raise ValueError(f"{section}.{key} must be an exact rational string")
                Fraction(value)
    if protocol["selection"] != {"order": ["S1", "S2", "S3"], "rule": "lowest-index-supportive"}:
        raise ValueError("selection order must be fixed before execution")
    return protocol


def _load_completed(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != kind or payload.get("status") != "completed":
        raise RuntimeError(f"refusing incomplete or incompatible artifact: {path}")
    return payload


def _validate_checkpoint(path: Path, config: dict, config_hash: str) -> tuple[bool, dict]:
    from agent_code.model_a_task3_replay.callbacks import _torch_load
    from agent_code.model_a_task3_replay.config import architecture_name
    payload = _torch_load(path)
    valid = (
        payload.get("architecture") == architecture_name(config)
        and payload.get("config_sha256") == config_hash
        and payload.get("parent_sha256") == config["parent_sha256"]
        and payload.get("sampling_profile") == config["sampling_profile"]
        and payload.get("kill_causal_fraction") == config["kill_causal_fraction"]
        and payload.get("kill_causal_window") == config["kill_causal_window"]
        and int(payload.get("completed_rounds", 0)) == int(config["rounds"])
    )
    diagnostics = {
        "kill_events_seen": int(payload.get("kill_events_seen", 0)),
        "replay": payload.get("replay_diagnostics", {}),
    }
    return valid, diagnostics


def train_one(protocol: dict, replica: str, profile: str) -> dict:
    spec = protocol["replicas"][replica][profile]
    config_path = (ROOT / spec["config_path"]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256_file(config_path)
    checkpoint = (ROOT / config["checkpoint_path"]).resolve()
    manifest_path = ROOT / "experiments/logs/runs" / f"{config['run_id']}.json"
    stats_path = manifest_path.with_suffix(".stats.json")
    existing = _load_completed(manifest_path, "task3-replay-sampling-training")
    if existing is not None:
        if (
            existing.get("config_sha256") != config_hash
            or existing.get("replica") != replica
            or existing.get("profile") != profile
        ):
            raise RuntimeError(f"training config mismatch: {replica}/{profile}")
        if not checkpoint.is_file() or sha256_file(checkpoint) != existing["checkpoint"]["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {replica}/{profile}")
        valid, diagnostics = _validate_checkpoint(checkpoint, config, config_hash)
        if not valid or existing.get("replay_diagnostics") != diagnostics:
            raise RuntimeError(f"checkpoint metadata mismatch: {replica}/{profile}")
        return existing
    if checkpoint.exists() or stats_path.exists():
        raise RuntimeError(f"refusing orphaned training artifact: {replica}/{profile}")

    overrides = {
        "MODEL_A_TASK3_REPLAY_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_TASK3_REPLAY_CONFIG_PATH": str(config_path),
        "MODEL_A_TASK3_REPLAY_SEED": str(config["agent_seed"]),
        "MODEL_A_TASK3_REPLAY_RESUME": "0",
        "TASK3_CURRICULUM_SEED": str(config["opponent_seed"]),
    }
    environment = os.environ.copy()
    environment.update(overrides)
    command = [
        sys.executable, "main.py", "play", "--agents", *config["agents"],
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", "classic", "--n-rounds", str(config["rounds"]),
        "--seed", str(config["seed"]), "--save-stats", str(stats_path),
    ]
    manifest = {
        "schema_version": 1, "kind": "task3-replay-sampling-training",
        "run_id": config["run_id"], "replica": replica, "profile": profile,
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": relative(config_path), "config_sha256": config_hash, "config": config,
        "checkpoint": {"path": relative(checkpoint), "sha256": None},
        "environment_overrides": overrides, "command": command,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest_path, manifest)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["exit_code"] = completed.returncode
    if completed.returncode == 0 and checkpoint.is_file() and stats_path.is_file():
        try:
            valid, diagnostics = _validate_checkpoint(checkpoint, config, config_hash)
        except Exception as exc:
            valid, diagnostics = False, {}
            manifest["checkpoint_validation_error"] = f"{type(exc).__name__}: {exc}"
        if valid:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            manifest["metrics_by_agent"] = metrics_by_agent(stats)
            manifest["replay_diagnostics"] = diagnostics
            manifest["checkpoint"]["sha256"] = sha256_file(checkpoint)
            manifest["status"] = "completed"
        else:
            manifest["status"] = "failed"
            manifest["error"] = "checkpoint metadata validation failed"
    else:
        manifest["status"] = "failed"
    atomic_json(manifest_path, manifest)
    if manifest["status"] != "completed":
        raise RuntimeError(f"training failed: {replica}/{profile}")
    return manifest


def _evaluation_manifest(protocol: dict, stratum: str, label: str, seed: int) -> Path:
    return ROOT / protocol["evaluation_output_directory"] / f"task3-replay-{stratum}-{label}-s{seed}.json"


def evaluate_one(protocol: dict, stratum: str, label: str, case: dict, checkpoints: dict) -> dict:
    seed = int(case["world_seed"])
    manifest_path = _evaluation_manifest(protocol, stratum, label, seed)
    target = "model_a_dqn" if label == "v4" else "model_a_task3_replay"
    checkpoint = (ROOT / protocol["v4"]["checkpoint_path"]).resolve() if label == "v4" else checkpoints[label]
    config_path = None
    if label != "v4":
        index, profile = int(label[1]), "control" if label[0] == "C" else "stratified"
        config_path = (ROOT / protocol["replicas"][f"R{index}"][profile]["config_path"]).resolve()
    checkpoint_hash = protocol["v4"]["checkpoint_sha256"] if label == "v4" else sha256_file(checkpoint)
    expected = {
        "target_agent": target, "agents": [target, protocol["strata"][stratum]["opponent"]],
        "scenario": "classic", "rounds": int(protocol["strata"][stratum]["rounds_per_seed"]),
        "seed": seed, "agent_seed": int(case["agent_seed"]),
        "opponent_seed": int(case["opponent_seed"]), "training_agents": 0,
        "continue_without_training": True,
    }
    existing = _load_completed(manifest_path, "task3-replay-sampling-evaluation")
    if existing is not None:
        if existing.get("evaluation") != expected or existing.get("checkpoint", {}).get("sha256") != checkpoint_hash:
            raise RuntimeError(f"evaluation contract mismatch: {manifest_path}")
        expected_config_hash = None if config_path is None else sha256_file(config_path)
        actual_config_hash = None if config_path is None else existing.get("agent_config", {}).get("sha256")
        if actual_config_hash != expected_config_hash:
            raise RuntimeError(f"evaluation config mismatch: {manifest_path}")
        return existing
    stats_path = manifest_path.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned evaluation stats: {stats_path}")
    overrides = {"TASK3_OPPONENT_SEED": str(case["opponent_seed"])}
    if label == "v4":
        overrides.update({"MODEL_A_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_SEED": str(case["agent_seed"])})
    else:
        overrides.update({
            "MODEL_A_TASK3_REPLAY_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_TASK3_REPLAY_CONFIG_PATH": str(config_path),
            "MODEL_A_TASK3_REPLAY_SEED": str(case["agent_seed"]),
            "MODEL_A_TASK3_REPLAY_RESUME": "0",
        })
    environment = os.environ.copy()
    environment.update(overrides)
    command = [
        sys.executable, "main.py", "play", "--agents", target,
        protocol["strata"][stratum]["opponent"], "--train", "0",
        "--continue-without-training", "--no-gui", "--scenario", "classic",
        "--n-rounds", str(expected["rounds"]), "--seed", str(seed),
        "--save-stats", str(stats_path),
    ]
    manifest = {
        "schema_version": 1, "kind": "task3-replay-sampling-evaluation",
        "run_id": f"task3-replay-{stratum}-{label}-s{seed}", "label": label,
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
        "agent_config": None if config_path is None else {"path": relative(config_path), "sha256": sha256_file(config_path)},
        "evaluation": expected, "environment_overrides": overrides, "command": command,
        "artifacts": {"raw_stats": relative(stats_path)},
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
        raise RuntimeError(f"evaluation failed: {stratum}/{label}/{seed}")
    return manifest


def _comparison(rows: list[dict], base_label: str, candidate_label: str) -> dict:
    score_deltas, kill_deltas = [], []
    for row in rows:
        base = row[base_label]["target_metrics"]
        candidate = row[candidate_label]["target_metrics"]
        score_deltas.append(Fraction(candidate["score"], candidate["rounds"]) - Fraction(base["score"], base["rounds"]))
        kill_deltas.append(Fraction(candidate["kills"], candidate["rounds"]) - Fraction(base["kills"], base["rounds"]))
    return {
        "exact_score_per_round_deltas": [str(value) for value in score_deltas],
        "score_per_round_deltas": [float(value) for value in score_deltas],
        "exact_kills_per_round_deltas": [str(value) for value in kill_deltas],
        "kills_per_round_deltas": [float(value) for value in kill_deltas],
        "wins": sum(value > 0 for value in score_deltas),
        "ties": sum(value == 0 for value in score_deltas),
        "losses": sum(value < 0 for value in score_deltas),
        "exact_worst_score_per_round_delta": str(min(score_deltas)),
    }


def _axis(rows_by_stratum: dict, base_label: str, candidate_label: str, gate: dict) -> dict:
    strata, all_rows = {}, []
    for name, rows in rows_by_stratum.items():
        all_rows.extend(rows)
        base, candidate = summarize(rows, base_label), summarize(rows, candidate_label)
        exact = {key: _delta(base, candidate, key) for key in METRICS}
        checks = {
            "score_floor": exact["score"] >= _threshold(gate, "minimum_stratum_score_delta"),
            "kill_floor": exact["kills"] >= _threshold(gate, "minimum_stratum_kills_delta"),
            "suicide_control": exact["suicides"] <= _threshold(gate, "maximum_stratum_suicides_increase"),
            "invalid_control": exact["invalid_actions"] <= _threshold(gate, "maximum_stratum_invalid_increase"),
        }
        strata[name] = {
            "base": base, "candidate": candidate,
            "exact_deltas_per_round": {key: str(value) for key, value in exact.items()},
            "deltas_per_round": {key: float(value) for key, value in exact.items()},
            "paired": _comparison(rows, base_label, candidate_label),
            "checks": checks, "passed": all(checks.values()),
        }
    base, candidate = summarize(all_rows, base_label), summarize(all_rows, candidate_label)
    exact = {key: _delta(base, candidate, key) for key in METRICS}
    paired = _comparison(all_rows, base_label, candidate_label)
    checks = {
        "all_strata_pass": all(item["passed"] for item in strata.values()),
        "score": exact["score"] >= _threshold(gate, "minimum_overall_score_delta"),
        "kills": exact["kills"] >= _threshold(gate, "minimum_overall_kills_delta"),
        "coin_retention": exact["coins"] >= -_threshold(gate, "maximum_overall_coin_regression"),
        "suicide_control": exact["suicides"] <= _threshold(gate, "maximum_overall_suicides_increase"),
        "invalid_control": exact["invalid_actions"] <= _threshold(gate, "maximum_overall_invalid_increase"),
        "paired_net_wins": paired["wins"] - paired["losses"] >= int(gate["minimum_paired_net_wins"]),
        "worst_cell": Fraction(paired["exact_worst_score_per_round_delta"]) >= -_threshold(gate, "maximum_single_cell_score_regression"),
        "latency": candidate["max_seed_mean_decision_time_ms"] < float(gate["maximum_seed_mean_decision_time_ms"]),
    }
    return {
        "strata": strata,
        "overall": {
            "base": base, "candidate": candidate,
            "exact_deltas_per_round": {key: str(value) for key, value in exact.items()},
            "deltas_per_round": {key: float(value) for key, value in exact.items()}, "paired": paired,
        },
        "checks": checks, "passed": all(checks.values()),
    }


def _profile_rows(rows_by_stratum: dict, base_for_replica, candidate_for_replica) -> dict:
    result = {}
    for stratum, rows in rows_by_stratum.items():
        result[stratum] = []
        for replica in REPLICAS:
            base_label, candidate_label = base_for_replica(replica), candidate_for_replica(replica)
            for row in rows:
                result[stratum].append({
                    "seed_tuple": row["seed_tuple"], "replica": replica,
                    "base": row[base_label], "candidate": row[candidate_label],
                })
    return result


def _replica_support(rows_by_stratum: dict, replica: str, protocol: dict) -> dict:
    index = int(replica[1])
    matched_rows = {
        name: [{"base": row[f"C{index}"], "candidate": row[f"S{index}"], "seed_tuple": row["seed_tuple"]} for row in rows]
        for name, rows in rows_by_stratum.items()
    }
    v4_rows = {
        name: [{"base": row["v4"], "candidate": row[f"S{index}"], "seed_tuple": row["seed_tuple"]} for row in rows]
        for name, rows in rows_by_stratum.items()
    }
    matched = _axis(matched_rows, "base", "candidate", protocol["replica_gate"])
    v4 = _axis(v4_rows, "base", "candidate", protocol["replica_v4_floor"])
    kill_delta = Fraction(matched["overall"]["exact_deltas_per_round"]["kills"])
    checks = {
        "matched_kill_strict_gain": kill_delta > 0,
        "matched_gate": matched["passed"],
        "v4_floor": v4["passed"],
    }
    return {"matched": matched, "vs_v4": v4, "checks": checks, "supportive": all(checks.values())}


def run(protocol: dict) -> tuple[dict, dict, dict]:
    training, checkpoints = {}, {}
    for replica in REPLICAS:
        for profile in ("control", "stratified"):
            label = protocol["replicas"][replica]["labels"][profile]
            manifest = train_one(protocol, replica, profile)
            training[label] = manifest
            checkpoints[label] = (ROOT / manifest["checkpoint"]["path"]).resolve()
    rows_by_stratum = {}
    for stratum, settings in protocol["strata"].items():
        rows = []
        for case in settings["cases"]:
            row = {"seed_tuple": case}
            for label in LABELS:
                manifest = evaluate_one(protocol, stratum, label, case, checkpoints)
                path = _evaluation_manifest(protocol, stratum, label, int(case["world_seed"]))
                row[label] = {
                    "manifest": relative(path), "manifest_sha256": sha256_file(path),
                    "target_metrics": manifest["target_metrics"],
                }
            rows.append(row)
        rows_by_stratum[stratum] = rows

    vs_v4_rows = _profile_rows(rows_by_stratum, lambda _: "v4", lambda replica: f"S{replica[1]}")
    vs_control_rows = _profile_rows(rows_by_stratum, lambda replica: f"C{replica[1]}", lambda replica: f"S{replica[1]}")
    vs_v4 = _axis(vs_v4_rows, "base", "candidate", protocol["vs_v4_gate"])
    vs_control = _axis(vs_control_rows, "base", "candidate", protocol["vs_control_gate"])
    replicas = {replica: _replica_support(rows_by_stratum, replica, protocol) for replica in REPLICAS}
    supportive = [f"S{replica[1]}" for replica in REPLICAS if replicas[replica]["supportive"]]

    diagnostic_checks = {}
    for label, manifest in training.items():
        diagnostics = manifest["replay_diagnostics"]
        replay = diagnostics["replay"]
        if label.startswith("C"):
            diagnostic_checks[label] = replay.get("realized_causal_draws", -1) == 0
        else:
            diagnostic_checks[label] = (
                diagnostics.get("kill_events_seen", 0) > 0
                and replay.get("realized_causal_draws", 0) > 0
            )
    checks = {
        "candidate_vs_v4": vs_v4["passed"],
        "candidate_vs_matched_control": vs_control["passed"],
        "minimum_supportive_replicas": len(supportive) >= int(protocol["vs_control_gate"]["minimum_supportive_replicas"]),
        "sampling_intervention_delivered": all(diagnostic_checks.values()),
    }
    selected = next((label for label in protocol["selection"]["order"] if label in supportive), None) if all(checks.values()) else None
    result = {
        "vs_v4": vs_v4, "vs_matched_control": vs_control, "replicas": replicas,
        "supportive_candidates": supportive, "diagnostic_checks": diagnostic_checks,
        "checks": checks, "passed": all(checks.values()), "selected_candidate": selected,
    }
    return training, rows_by_stratum, result


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
        print(json.dumps({
            "mode": "dry-run", "run_id": protocol["run_id"], "task": 3, "task4": False,
            **protocol["formal_budget"], "labels": list(LABELS),
            "only_variable": "replay sampling: uniform 0/64 vs kill-causal 16/64",
            "automatic_followup": False, "output_path": protocol["output_path"],
        }, indent=2, sort_keys=True))
        return 0
    output = ROOT / protocol["output_path"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite report: {output}")
    training, rows, result = run(protocol)
    selected = result["selected_candidate"]
    decision = "replay_sampling_confirmed" if selected else "replay_sampling_not_confirmed_stop"
    report = {
        "schema_version": 1, "kind": protocol["kind"], "run_id": protocol["run_id"],
        "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": relative(config_path), "preregistration_sha256": sha256_file(config_path),
        "training_manifests": {
            label: {
                "path": f"experiments/logs/runs/{manifest['run_id']}.json",
                "sha256": sha256_file(ROOT / "experiments/logs/runs" / f"{manifest['run_id']}.json"),
                "checkpoint": manifest["checkpoint"], "replay_diagnostics": manifest["replay_diagnostics"],
            } for label, manifest in training.items()
        },
        "rows_by_stratum": rows, "result": result, "decision": decision,
        "selected_candidate": selected, "automatic_followup_started": False,
        "task4_started": False, "next_step": protocol["decision_branches"][decision],
    }
    atomic_json(output, report)
    print(json.dumps({
        "output": str(output), "decision": decision,
        "selected_candidate": selected, "next_step": report["next_step"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
