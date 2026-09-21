"""Run the preregistered Task-3 kill-rich training-distribution experiment."""

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
from tools.task3_replay_sampling import _axis  # noqa: E402


REPLICAS = ("R1", "R2", "R3")
LABELS = ("v4", "C1", "K1", "C2", "K2", "C3", "K3")


def _contract(config: dict) -> dict:
    return {
        key: config[key] for key in (
            "agent", "agents", "task", "includes_task4", "rounds", "resume",
            "enabled_for_training", "arm", "reward_profile", "action_features",
            "n_step", "sampling_profile", "parent_checkpoint", "parent_sha256",
            "hyperparameters",
        )
    }


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("kind") != "task3-training-distribution-matched-v1":
        raise ValueError("unexpected training-distribution protocol kind")
    if protocol.get("task") != 3 or protocol.get("includes_task4", True):
        raise ValueError("the experiment must remain isolated to Task 3")
    if not protocol.get("training") or protocol.get("automatic_followup"):
        raise ValueError("formal training is required and automatic follow-up is forbidden")
    for relative_path, expected in protocol["bindings"].items():
        target = ROOT / relative_path
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"binding mismatch: {relative_path}")
    v4 = ROOT / protocol["v4"]["checkpoint_path"]
    if not v4.is_file() or sha256_file(v4) != protocol["v4"]["checkpoint_sha256"]:
        raise ValueError("frozen v4 checkpoint mismatch")

    from agent_code.model_a_task3_killrich.config import load_config
    contracts, seed_tuples, run_ids, checkpoints = [], [], set(), set()
    if list(protocol.get("replicas", {})) != list(REPLICAS):
        raise ValueError("ordered replicas R1, R2, R3 are required")
    for index, (replica, pair) in enumerate(protocol["replicas"].items(), start=1):
        loaded = {}
        for profile in ("control", "killrich"):
            spec = pair[profile]
            config, _, digest = load_config(ROOT / spec["config_path"])
            if digest != spec["config_sha256"]:
                raise ValueError(f"config hash mismatch: {replica}/{profile}")
            expected_distribution = "classic-control" if profile == "control" else "kill-rich"
            if config["training_distribution"] != expected_distribution:
                raise ValueError(f"distribution mismatch: {replica}/{profile}")
            if config["parent_sha256"] != protocol["v4"]["checkpoint_sha256"]:
                raise ValueError(f"parent mismatch: {replica}/{profile}")
            contracts.append(_contract(config))
            loaded[profile] = config
            if config["run_id"] in run_ids or config["checkpoint_path"] in checkpoints:
                raise ValueError("training artifacts must be unique")
            run_ids.add(config["run_id"])
            checkpoints.add(config["checkpoint_path"])
        control, candidate = loaded["control"], loaded["killrich"]
        if _contract(control) != _contract(candidate):
            raise ValueError(f"paired algorithms differ beyond training distribution: {replica}")
        seed_tuple = tuple(control[key] for key in ("seed", "agent_seed", "opponent_seed"))
        if seed_tuple != tuple(candidate[key] for key in ("seed", "agent_seed", "opponent_seed")):
            raise ValueError(f"training seeds are not matched: {replica}")
        seed_tuples.append(seed_tuple)
        if pair.get("labels") != {"control": f"C{index}", "killrich": f"K{index}"}:
            raise ValueError(f"unexpected labels: {replica}")
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("replicas alter the frozen D algorithm")
    if len(set(seed_tuples)) != 3:
        raise ValueError("three fresh training seed tuples are required")

    signal_cases = protocol.get("signal_gate", {}).get("cases", [])
    if len(signal_cases) != 8 or int(protocol["signal_gate"].get("rounds_per_case", 0)) != 25:
        raise ValueError("signal gate requires eight matched 25-round cases")
    evaluation_tuples = []
    if set(protocol.get("strata", {})) != {"peaceful", "coin_collector"}:
        raise ValueError("two official evaluation strata are required")
    for name, stratum in protocol["strata"].items():
        if int(stratum.get("rounds_per_seed", 0)) != 50 or len(stratum.get("cases", [])) != 4:
            raise ValueError(f"{name} must contain four 50-round cases")
        evaluation_tuples.extend(
            tuple(int(case[key]) for key in ("world_seed", "agent_seed", "opponent_seed"))
            for case in stratum["cases"]
        )
    all_tuples = seed_tuples + [
        tuple(int(case[key]) for key in ("world_seed", "agent_seed", "opponent_seed"))
        for case in signal_cases
    ] + evaluation_tuples
    if len(set(all_tuples)) != len(all_tuples):
        raise ValueError("all formal seed tuples must be unique")
    floor = int(protocol["fresh_seed_floor"])
    if any(any(value < floor for value in item) for item in all_tuples):
        raise ValueError("a preregistered seed is below the freshness floor")
    expected_budget = {
        "signal_runs": 16, "signal_rounds": 400,
        "conditional_training_runs": 6, "conditional_training_rounds": 1800,
        "conditional_evaluation_runs": 56, "conditional_evaluation_rounds": 2800,
    }
    if protocol.get("formal_budget") != expected_budget:
        raise ValueError("formal budget mismatch")
    return protocol


def _load_completed(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != kind or payload.get("status") != "completed":
        raise RuntimeError(f"refusing incomplete or incompatible artifact: {path}")
    return payload


def _signal_manifest(protocol: dict, distribution: str, seed: int) -> Path:
    return ROOT / protocol["signal_output_directory"] / f"task3-distribution-signal-{distribution}-s{seed}.json"


def signal_one(protocol: dict, distribution: str, case: dict) -> dict:
    seed = int(case["world_seed"])
    manifest_path = _signal_manifest(protocol, distribution, seed)
    rounds = int(protocol["signal_gate"]["rounds_per_case"])
    expected = {
        "target_agent": "model_a_dqn",
        "agents": ["model_a_dqn", "task3_curriculum_agent"],
        "scenario": "classic", "training_distribution": distribution,
        "rounds": rounds, "seed": seed,
        "agent_seed": int(case["agent_seed"]),
        "opponent_seed": int(case["opponent_seed"]),
        "training_agents": 0, "continue_without_training": True,
    }
    existing = _load_completed(manifest_path, "task3-training-distribution-signal")
    if existing is not None:
        if existing.get("evaluation") != expected:
            raise RuntimeError(f"signal contract mismatch: {manifest_path}")
        return existing
    stats_path = manifest_path.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned signal stats: {stats_path}")
    checkpoint = (ROOT / protocol["v4"]["checkpoint_path"]).resolve()
    overrides = {
        "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_SEED": str(case["agent_seed"]),
        "TASK3_CURRICULUM_SEED": str(case["opponent_seed"]),
        "TASK3_TRAINING_DISTRIBUTION": distribution,
    }
    environment = os.environ.copy()
    environment.update(overrides)
    command = [
        sys.executable, "tools/task3_distribution_play.py", "play", "--agents", *expected["agents"],
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", "classic", "--n-rounds", str(rounds), "--seed", str(seed),
        "--save-stats", str(stats_path),
    ]
    manifest = {
        "schema_version": 1, "kind": "task3-training-distribution-signal",
        "run_id": manifest_path.stem, "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {"path": relative(checkpoint), "sha256": sha256_file(checkpoint)},
        "evaluation": expected, "environment_overrides": overrides, "command": command,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest_path, manifest)
    before = sha256_file(checkpoint)
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["exit_code"] = completed.returncode
    if completed.returncode == 0 and stats_path.is_file() and sha256_file(checkpoint) == before:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        manifest["metrics_by_agent"] = metrics_by_agent(stats)
        manifest["target_metrics"] = manifest["metrics_by_agent"].get("model_a_dqn")
        manifest["status"] = "completed"
    else:
        manifest["status"] = "failed"
    atomic_json(manifest_path, manifest)
    if manifest["status"] != "completed":
        raise RuntimeError(f"signal run failed: {distribution}/{seed}")
    return manifest


def run_signal_gate(protocol: dict) -> dict:
    rows = []
    for case in protocol["signal_gate"]["cases"]:
        control = signal_one(protocol, "classic-control", case)
        candidate = signal_one(protocol, "kill-rich", case)
        rows.append({
            "seed_tuple": case,
            "control": {"manifest": relative(_signal_manifest(protocol, "classic-control", int(case["world_seed"]))),
                        "target_metrics": control["target_metrics"]},
            "killrich": {"manifest": relative(_signal_manifest(protocol, "kill-rich", int(case["world_seed"]))),
                         "target_metrics": candidate["target_metrics"]},
        })
    control = summarize(rows, "control")
    candidate = summarize(rows, "killrich")
    control["steps"] = sum(int(row["control"]["target_metrics"]["steps"]) for row in rows)
    candidate["steps"] = sum(int(row["killrich"]["target_metrics"]["steps"]) for row in rows)
    control_rate = Fraction(int(control["kills"]), max(1, int(control["steps"])))
    candidate_rate = Fraction(int(candidate["kills"]), max(1, int(candidate["steps"])))
    gate = protocol["signal_gate"]
    positive_cases = sum(row["killrich"]["target_metrics"]["kills"] > 0 for row in rows)
    checks = {
        "minimum_kills": int(candidate["kills"]) >= int(gate["minimum_candidate_kills"]),
        "minimum_positive_cases": positive_cases >= int(gate["minimum_positive_cases"]),
        "absolute_kill_step_rate": candidate_rate >= Fraction(gate["minimum_kill_step_rate"]),
        "relative_kill_step_rate": candidate_rate >= int(gate["minimum_rate_multiplier"]) * control_rate,
        "suicide_ceiling": Fraction(int(candidate["suicides"]), int(candidate["rounds"]))
                           <= Fraction(gate["maximum_suicides_per_round"]),
    }
    return {
        "rows": rows, "control": control, "candidate": candidate,
        "exact_control_kills_per_step": str(control_rate),
        "exact_candidate_kills_per_step": str(candidate_rate),
        "positive_candidate_cases": positive_cases,
        "checks": checks, "passed": all(checks.values()),
    }


def _validate_checkpoint(path: Path, config: dict, config_hash: str) -> bool:
    from agent_code.model_a_task3_killrich.callbacks import _torch_load
    from agent_code.model_a_task3_killrich.config import architecture_name
    payload = _torch_load(path)
    return (
        payload.get("architecture") == architecture_name(config)
        and payload.get("config_sha256") == config_hash
        and payload.get("parent_sha256") == config["parent_sha256"]
        and payload.get("training_distribution") == config["training_distribution"]
        and payload.get("sampling_profile") == "uniform"
        and int(payload.get("completed_rounds", 0)) == int(config["rounds"])
    )


def train_one(protocol: dict, replica: str, profile: str) -> dict:
    spec = protocol["replicas"][replica][profile]
    config_path = (ROOT / spec["config_path"]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256_file(config_path)
    checkpoint = (ROOT / config["checkpoint_path"]).resolve()
    manifest_path = ROOT / "experiments/logs/runs" / f"{config['run_id']}.json"
    stats_path = manifest_path.with_suffix(".stats.json")
    existing = _load_completed(manifest_path, "task3-training-distribution-training")
    if existing is not None:
        if existing.get("config_sha256") != config_hash or not checkpoint.is_file():
            raise RuntimeError(f"training artifact mismatch: {replica}/{profile}")
        if sha256_file(checkpoint) != existing["checkpoint"]["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {replica}/{profile}")
        if not _validate_checkpoint(checkpoint, config, config_hash):
            raise RuntimeError(f"checkpoint metadata mismatch: {replica}/{profile}")
        return existing
    if checkpoint.exists() or stats_path.exists():
        raise RuntimeError(f"refusing orphaned training artifact: {replica}/{profile}")
    overrides = {
        "MODEL_A_TASK3_KILLRICH_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_TASK3_KILLRICH_CONFIG_PATH": str(config_path),
        "MODEL_A_TASK3_KILLRICH_SEED": str(config["agent_seed"]),
        "MODEL_A_TASK3_KILLRICH_RESUME": "0",
        "TASK3_CURRICULUM_SEED": str(config["opponent_seed"]),
        "TASK3_TRAINING_DISTRIBUTION": config["training_distribution"],
    }
    environment = os.environ.copy()
    environment.update(overrides)
    command = [
        sys.executable, "tools/task3_distribution_play.py", "play", "--agents", *config["agents"],
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", config["scenario"], "--n-rounds", str(config["rounds"]),
        "--seed", str(config["seed"]), "--save-stats", str(stats_path),
    ]
    manifest = {
        "schema_version": 1, "kind": "task3-training-distribution-training",
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
    valid = completed.returncode == 0 and checkpoint.is_file() and stats_path.is_file()
    if valid:
        valid = _validate_checkpoint(checkpoint, config, config_hash)
    if valid:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        manifest["metrics_by_agent"] = metrics_by_agent(stats)
        manifest["checkpoint"]["sha256"] = sha256_file(checkpoint)
        manifest["status"] = "completed"
    else:
        manifest["status"] = "failed"
    atomic_json(manifest_path, manifest)
    if manifest["status"] != "completed":
        raise RuntimeError(f"training failed: {replica}/{profile}")
    return manifest


def _evaluation_manifest(protocol: dict, stratum: str, label: str, seed: int) -> Path:
    return ROOT / protocol["evaluation_output_directory"] / f"task3-distribution-{stratum}-{label}-s{seed}.json"


def evaluate_one(protocol: dict, stratum: str, label: str, case: dict, checkpoints: dict) -> dict:
    seed = int(case["world_seed"])
    manifest_path = _evaluation_manifest(protocol, stratum, label, seed)
    target = "model_a_dqn" if label == "v4" else "model_a_task3_killrich"
    checkpoint = (ROOT / protocol["v4"]["checkpoint_path"]).resolve() if label == "v4" else checkpoints[label]
    config_path = None
    if label != "v4":
        index = int(label[1])
        profile = "control" if label[0] == "C" else "killrich"
        config_path = (ROOT / protocol["replicas"][f"R{index}"][profile]["config_path"]).resolve()
    expected = {
        "target_agent": target, "agents": [target, protocol["strata"][stratum]["opponent"]],
        "scenario": "classic", "rounds": int(protocol["strata"][stratum]["rounds_per_seed"]),
        "seed": seed, "agent_seed": int(case["agent_seed"]),
        "opponent_seed": int(case["opponent_seed"]), "training_agents": 0,
        "continue_without_training": True,
    }
    existing = _load_completed(manifest_path, "task3-training-distribution-evaluation")
    if existing is not None:
        if existing.get("evaluation") != expected or existing.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint):
            raise RuntimeError(f"evaluation contract mismatch: {manifest_path}")
        return existing
    stats_path = manifest_path.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned evaluation stats: {stats_path}")
    overrides = {"TASK3_OPPONENT_SEED": str(case["opponent_seed"])}
    if label == "v4":
        overrides.update({"MODEL_A_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_SEED": str(case["agent_seed"])})
    else:
        overrides.update({
            "MODEL_A_TASK3_KILLRICH_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_TASK3_KILLRICH_CONFIG_PATH": str(config_path),
            "MODEL_A_TASK3_KILLRICH_SEED": str(case["agent_seed"]),
            "MODEL_A_TASK3_KILLRICH_RESUME": "0",
        })
    environment = os.environ.copy()
    environment.update(overrides)
    command = [
        sys.executable, "main.py", "play", "--agents", *expected["agents"],
        "--train", "0", "--continue-without-training", "--no-gui", "--scenario", "classic",
        "--n-rounds", str(expected["rounds"]), "--seed", str(seed), "--save-stats", str(stats_path),
    ]
    manifest = {
        "schema_version": 1, "kind": "task3-training-distribution-evaluation",
        "run_id": manifest_path.stem, "label": label, "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {"path": relative(checkpoint), "sha256": sha256_file(checkpoint)},
        "agent_config": None if config_path is None else {"path": relative(config_path), "sha256": sha256_file(config_path)},
        "evaluation": expected, "environment_overrides": overrides, "command": command,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest_path, manifest)
    before = sha256_file(checkpoint)
    config_before = None if config_path is None else sha256_file(config_path)
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["exit_code"] = completed.returncode
    unchanged = sha256_file(checkpoint) == before
    unchanged &= config_path is None or sha256_file(config_path) == config_before
    if completed.returncode == 0 and stats_path.is_file() and unchanged:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        manifest["metrics_by_agent"] = metrics_by_agent(stats)
        manifest["target_metrics"] = manifest["metrics_by_agent"].get(target)
        manifest["status"] = "completed"
    else:
        manifest["status"] = "failed"
    atomic_json(manifest_path, manifest)
    if manifest["status"] != "completed":
        raise RuntimeError(f"evaluation failed: {stratum}/{label}/{seed}")
    return manifest


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
        name: [{"base": row[f"C{index}"], "candidate": row[f"K{index}"], "seed_tuple": row["seed_tuple"]} for row in rows]
        for name, rows in rows_by_stratum.items()
    }
    v4_rows = {
        name: [{"base": row["v4"], "candidate": row[f"K{index}"], "seed_tuple": row["seed_tuple"]} for row in rows]
        for name, rows in rows_by_stratum.items()
    }
    matched = _axis(matched_rows, "base", "candidate", protocol["replica_gate"])
    v4 = _axis(v4_rows, "base", "candidate", protocol["replica_v4_floor"])
    checks = {
        "matched_kill_strict_gain": Fraction(matched["overall"]["exact_deltas_per_round"]["kills"]) > 0,
        "matched_gate": matched["passed"], "v4_floor": v4["passed"],
    }
    return {"matched": matched, "vs_v4": v4, "checks": checks, "supportive": all(checks.values())}


def run_after_signal(protocol: dict) -> tuple[dict, dict, dict]:
    training, checkpoints = {}, {}
    for replica in REPLICAS:
        for profile in ("control", "killrich"):
            label = protocol["replicas"][replica]["labels"][profile]
            manifest = train_one(protocol, replica, profile)
            training[label] = manifest
            checkpoints[label] = (ROOT / manifest["checkpoint"]["path"]).resolve()
    rows_by_stratum = {}
    for stratum, stratum_settings in protocol["strata"].items():
        rows = []
        for case in stratum_settings["cases"]:
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
    vs_v4 = _axis(
        _profile_rows(rows_by_stratum, lambda _: "v4", lambda r: f"K{r[1]}"),
        "base", "candidate", protocol["vs_v4_gate"],
    )
    vs_control = _axis(
        _profile_rows(rows_by_stratum, lambda r: f"C{r[1]}", lambda r: f"K{r[1]}"),
        "base", "candidate", protocol["vs_control_gate"],
    )
    replicas = {replica: _replica_support(rows_by_stratum, replica, protocol) for replica in REPLICAS}
    supportive = [f"K{replica[1]}" for replica in REPLICAS if replicas[replica]["supportive"]]
    checks = {
        "candidate_vs_v4": vs_v4["passed"],
        "candidate_vs_matched_control": vs_control["passed"],
        "minimum_supportive_replicas": len(supportive) >= int(protocol["minimum_supportive_replicas"]),
    }
    selected = next((label for label in protocol["selection"]["order"] if label in supportive), None) if all(checks.values()) else None
    result = {
        "vs_v4": vs_v4, "vs_matched_control": vs_control, "replicas": replicas,
        "supportive_candidates": supportive, "checks": checks,
        "passed": all(checks.values()), "selected_candidate": selected,
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
            "only_variable": "training episode/state distribution: classic-control vs kill-rich",
            "signal_gate_before_training": True, "automatic_followup": False,
            "output_path": protocol["output_path"],
        }, indent=2, sort_keys=True))
        return 0
    output = ROOT / protocol["output_path"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite report: {output}")
    signal = run_signal_gate(protocol)
    if signal["passed"]:
        training, rows, result = run_after_signal(protocol)
        selected = result["selected_candidate"]
        decision = "killrich_distribution_confirmed" if selected else "killrich_distribution_not_confirmed_stop"
    else:
        training, rows, selected = {}, {}, None
        result = {"passed": False, "selected_candidate": None, "checks": {"signal_gate": False}}
        decision = "environment_signal_failed_stop"
    report = {
        "schema_version": 1, "kind": protocol["kind"], "run_id": protocol["run_id"],
        "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": relative(config_path), "preregistration_sha256": sha256_file(config_path),
        "signal_gate": signal,
        "training_manifests": {
            label: {
                "path": f"experiments/logs/runs/{manifest['run_id']}.json",
                "sha256": sha256_file(ROOT / "experiments/logs/runs" / f"{manifest['run_id']}.json"),
                "checkpoint": manifest["checkpoint"],
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
