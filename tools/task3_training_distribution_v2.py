"""Run the corrected Task-3 resolved-duel training-distribution experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import task3_training_distribution as base  # noqa: E402
from tools.task3_four_arm import atomic_json, metrics_by_agent, relative, sha256_file  # noqa: E402


def load_protocol(path: Path) -> dict:
    protocol = base.load_protocol(path)
    if protocol.get("protocol_revision") != "resolved-duel-v2":
        raise ValueError("corrected runner requires protocol_revision='resolved-duel-v2'")
    expected = {
        "duel_signal_failed_stop", "killrich_duel_distribution_confirmed",
        "killrich_duel_distribution_not_confirmed_stop",
    }
    if set(protocol.get("decision_branches", {})) != expected:
        raise ValueError("unexpected corrected decision branches")
    return protocol


def signal_one(protocol: dict, distribution: str, case: dict) -> dict:
    seed = int(case["world_seed"])
    manifest_path = base._signal_manifest(protocol, distribution, seed)
    rounds = int(protocol["signal_gate"]["rounds_per_case"])
    expected = {
        "target_agent": "model_a_dqn",
        "agents": ["model_a_dqn", "task3_curriculum_agent"],
        "scenario": "classic", "training_distribution": distribution,
        "rounds": rounds, "seed": seed, "agent_seed": int(case["agent_seed"]),
        "opponent_seed": int(case["opponent_seed"]),
        "training_agents": 0, "continue_without_training": True,
        "killrich_end_on_elimination": distribution == "kill-rich",
    }
    existing = base._load_completed(manifest_path, "task3-training-distribution-v2-signal")
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
        sys.executable, "tools/task3_distribution_play_v2.py", "play",
        "--agents", *expected["agents"], "--train", "0",
        "--continue-without-training", "--no-gui", "--scenario", "classic",
        "--n-rounds", str(rounds), "--seed", str(seed),
        "--save-stats", str(stats_path),
    ]
    manifest = {
        "schema_version": 1, "kind": "task3-training-distribution-v2-signal",
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


def train_one(protocol: dict, replica: str, profile: str) -> dict:
    spec = protocol["replicas"][replica][profile]
    config_path = (ROOT / spec["config_path"]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256_file(config_path)
    checkpoint = (ROOT / config["checkpoint_path"]).resolve()
    manifest_path = ROOT / "experiments/logs/runs" / f"{config['run_id']}.json"
    stats_path = manifest_path.with_suffix(".stats.json")
    existing = base._load_completed(manifest_path, "task3-training-distribution-v2-training")
    if existing is not None:
        if existing.get("config_sha256") != config_hash or not checkpoint.is_file():
            raise RuntimeError(f"training artifact mismatch: {replica}/{profile}")
        if sha256_file(checkpoint) != existing["checkpoint"]["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {replica}/{profile}")
        if not base._validate_checkpoint(checkpoint, config, config_hash):
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
        sys.executable, "tools/task3_distribution_play_v2.py", "play",
        "--agents", *config["agents"], "--train", "1",
        "--continue-without-training", "--no-gui", "--scenario", "classic",
        "--n-rounds", str(config["rounds"]), "--seed", str(config["seed"]),
        "--save-stats", str(stats_path),
    ]
    manifest = {
        "schema_version": 1, "kind": "task3-training-distribution-v2-training",
        "run_id": config["run_id"], "replica": replica, "profile": profile,
        "killrich_end_on_elimination": profile == "killrich", "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": relative(config_path), "config_sha256": config_hash,
        "config": config, "checkpoint": {"path": relative(checkpoint), "sha256": None},
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
        valid = base._validate_checkpoint(checkpoint, config, config_hash)
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
            "mode": "dry-run", "run_id": protocol["run_id"], "revision": "resolved-duel-v2",
            "task": 3, "task4": False, **protocol["formal_budget"],
            "labels": list(base.LABELS),
            "only_change_from_v1": "end kill-rich episode immediately when either duel agent is eliminated",
            "signal_gate_before_training": True, "automatic_followup": False,
            "output_path": protocol["output_path"],
        }, indent=2, sort_keys=True))
        return 0
    output = ROOT / protocol["output_path"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite report: {output}")
    base.signal_one = signal_one
    base.train_one = train_one
    signal = base.run_signal_gate(protocol)
    if signal["passed"]:
        training, rows, result = base.run_after_signal(protocol)
        selected = result["selected_candidate"]
        decision = ("killrich_duel_distribution_confirmed" if selected
                    else "killrich_duel_distribution_not_confirmed_stop")
    else:
        training, rows, selected = {}, {}, None
        result = {"passed": False, "selected_candidate": None, "checks": {"signal_gate": False}}
        decision = "duel_signal_failed_stop"
    report = {
        "schema_version": 1, "kind": protocol["kind"], "protocol_revision": "resolved-duel-v2",
        "run_id": protocol["run_id"], "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
