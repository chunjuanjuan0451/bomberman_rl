"""Validate and optionally execute one isolated Model A training run.

The command is dry-run by default. Pass ``--execute`` explicitly to launch
training. This script never writes the tournament-default ``model_a.pt``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .evaluate import REPOSITORY_ROOT, _relative, _utc_now, _write_json, repository_state, sha256_file
except ImportError:  # Direct invocation: python experiments/train_model_a.py
    from evaluate import REPOSITORY_ROOT, _relative, _utc_now, _write_json, repository_state, sha256_file


DEFAULT_CHECKPOINT = (REPOSITORY_ROOT / "agent_code/model_a_dqn/model_a.pt").resolve()
DEFAULT_RUN_DIRECTORY = REPOSITORY_ROOT / "experiments/logs/runs"
REQUIRED_FIELDS = {
    "run_id", "agents", "scenario", "rounds", "seed", "agent_seed", "checkpoint_path",
}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Training config must contain one JSON object")
    missing = sorted(REQUIRED_FIELDS - set(config))
    if missing:
        raise ValueError(f"Training config is missing fields: {', '.join(missing)}")
    agent = config.get("agent", "model_a_dqn")
    if agent not in ("model_a_dqn", "model_a_v6", "model_a_v6_stable", "model_a_v8", "model_a_v9"):
        raise ValueError("Training runner does not support the requested agent")
    agents = config["agents"]
    if not isinstance(agents, list) or not agents or agents[0] != agent:
        raise ValueError(f"agents must be a non-empty list beginning with {agent}")
    if int(config["rounds"]) <= 0:
        raise ValueError("rounds must be positive")
    if config.get("resume", False):
        raise ValueError("Fresh-run infrastructure forbids resume; use a separately reviewed runner")
    if not config.get("enabled_for_training", True):
        raise ValueError(f"Training is disabled for {config.get('variant', config['run_id'])}")
    if agent == "model_a_v6":
        try:
            if str(REPOSITORY_ROOT) not in sys.path:
                sys.path.insert(0, str(REPOSITORY_ROOT))
            from agent_code.model_a_v6.config import load_v6_config
            load_v6_config(path)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid model_a_v6 config: {exc}") from exc
    if agent == "model_a_v6_stable":
        try:
            if str(REPOSITORY_ROOT) not in sys.path:
                sys.path.insert(0, str(REPOSITORY_ROOT))
            from agent_code.model_a_v6_stable.config import load_config as load_stable_config
            load_stable_config(path)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid model_a_v6_stable config: {exc}") from exc
    if agent == "model_a_v8":
        try:
            if str(REPOSITORY_ROOT) not in sys.path:
                sys.path.insert(0, str(REPOSITORY_ROOT))
            from agent_code.model_a_v8.config import load_config as load_v8_config
            load_v8_config(path)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid model_a_v8 config: {exc}") from exc
    if agent == "model_a_v9":
        try:
            if str(REPOSITORY_ROOT) not in sys.path:
                sys.path.insert(0, str(REPOSITORY_ROOT))
            from agent_code.model_a_v9.config import load_config as load_v9_config
            load_v9_config(path)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid model_a_v9 config: {exc}") from exc
    return config


def resolve_checkpoint(config: dict) -> Path:
    checkpoint = Path(config["checkpoint_path"])
    if not checkpoint.is_absolute():
        checkpoint = REPOSITORY_ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    if checkpoint == DEFAULT_CHECKPOINT:
        raise ValueError("Training may not overwrite the tournament-default model_a.pt")
    try:
        checkpoint.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError("Training checkpoint must remain inside the repository") from exc
    return checkpoint


def training_command(config: dict, stats_path: Path) -> list[str]:
    return [
        sys.executable, "main.py", "play", "--agents", *config["agents"],
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", str(config["scenario"]), "--n-rounds", str(config["rounds"]),
        "--seed", str(config["seed"]), "--save-stats", str(stats_path),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--execute", action="store_true", help="Actually launch training; default is validation only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    try:
        config = load_config(config_path)
        checkpoint = resolve_checkpoint(config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    run_directory = DEFAULT_RUN_DIRECTORY
    run_id = str(config["run_id"])
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None:
        raise SystemExit("run_id may contain only letters, digits, dots, underscores and hyphens")
    manifest_path = run_directory / f"{run_id}.json"
    stats_path = run_directory / f"{run_id}.stats.json"
    command = training_command(config, stats_path)
    if checkpoint.exists():
        raise SystemExit(f"Refusing to overwrite existing checkpoint: {checkpoint}")
    if manifest_path.exists() or stats_path.exists():
        raise SystemExit(f"Refusing to reuse run_id: {run_id}")

    summary = {
        "config": _relative(config_path),
        "checkpoint": _relative(checkpoint),
        "command": command,
        "execute": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(summary, indent=2))
        return 0

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if config.get("agent", "model_a_dqn") == "model_a_v6":
        environment_overrides = {
            "MODEL_A_V6_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V6_CONFIG_PATH": str(config_path),
            "MODEL_A_V6_SEED": str(config["agent_seed"]),
            "MODEL_A_V6_RESUME": "0",
        }
    elif config.get("agent") == "model_a_v6_stable":
        environment_overrides = {
            "MODEL_A_V6_STABLE_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V6_STABLE_CONFIG_PATH": str(config_path),
            "MODEL_A_V6_STABLE_SEED": str(config["agent_seed"]),
            "MODEL_A_V6_STABLE_RESUME": "0",
        }
    elif config.get("agent") == "model_a_v8":
        environment_overrides = {
            "MODEL_A_V8_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V8_CONFIG_PATH": str(config_path),
            "MODEL_A_V8_SEED": str(config["agent_seed"]),
            "MODEL_A_V8_RESUME": "0",
        }
    elif config.get("agent") == "model_a_v9":
        environment_overrides = {
            "MODEL_A_V9_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V9_CONFIG_PATH": str(config_path),
            "MODEL_A_V9_SEED": str(config["agent_seed"]),
            "MODEL_A_V9_RESUME": "0",
        }
    else:
        environment_overrides = {
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_SEED": str(config["agent_seed"]),
            "MODEL_A_RESUME": "0",
        }
    environment.update(environment_overrides)
    manifest = {
        "schema_version": 1,
        "kind": "training",
        "run_id": run_id,
        "status": "running",
        "started_at_utc": _utc_now(),
        "repository": repository_state(run_directory),
        "config_path": _relative(config_path),
        "config_sha256": sha256_file(config_path),
        "config": config,
        "checkpoint": {"path": _relative(checkpoint), "sha256": None},
        "artifacts": {"raw_stats": _relative(stats_path)},
        "command": command,
        "environment_overrides": environment_overrides,
    }
    _write_json(manifest_path, manifest)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=False)
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["exit_code"] = completed.returncode
    manifest["status"] = "completed" if completed.returncode == 0 and checkpoint.is_file() else "failed"
    if checkpoint.is_file():
        manifest["checkpoint"]["sha256"] = sha256_file(checkpoint)
    _write_json(manifest_path, manifest)
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
