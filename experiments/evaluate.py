"""Run one evaluation and write a reproducible manifest beside its raw stats."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "experiments" / "logs" / "evaluations"
DEFAULT_WEIGHT_FILES = {
    "model_a_dqn": Path("agent_code/model_a_dqn/model_a.pt"),
    "model_a_v7a": Path("agent_code/model_a_dqn/model_a.pt"),
    "model_a_v7b": Path("agent_code/model_a_dqn/model_a.pt"),
    "model_b_linear": Path("agent_code/model_b_linear/weights.npy"),
}
WEIGHT_ENVIRONMENT_BY_AGENT = {
    "model_a_dqn": "MODEL_A_CHECKPOINT_PATH",
    "model_a_v6": "MODEL_A_V6_CHECKPOINT_PATH",
    "model_a_v6_stable": "MODEL_A_V6_STABLE_CHECKPOINT_PATH",
    "model_a_v6_ensemble": "MODEL_A_V6_ENSEMBLE_CHECKPOINT_PATH",
    "model_a_v6_consensus": "MODEL_A_V6_CONSENSUS_CHECKPOINT_PATH",
    "model_a_v7a": "MODEL_A_V7A_CHECKPOINT_PATH",
    "model_a_v7b": "MODEL_A_V7B_CHECKPOINT_PATH",
    "model_a_v8": "MODEL_A_V8_CHECKPOINT_PATH",
    "model_a_v9": "MODEL_A_V9_CHECKPOINT_PATH",
    "model_a_v10": "MODEL_A_V10_CHECKPOINT_PATH",
    "model_b_linear": "MODEL_B_WEIGHTS_PATH",
}
SEED_ENVIRONMENT_BY_AGENT = {
    "model_a_dqn": "MODEL_A_SEED",
    "model_a_v6": "MODEL_A_V6_SEED",
    "model_a_v6_stable": "MODEL_A_V6_STABLE_SEED",
    "model_a_v6_ensemble": "MODEL_A_V6_ENSEMBLE_SEED",
    "model_a_v6_consensus": "MODEL_A_V6_CONSENSUS_SEED",
    "model_a_v7a": "MODEL_A_V7A_SEED",
    "model_a_v7b": "MODEL_A_V7B_SEED",
    "model_a_v8": "MODEL_A_V8_SEED",
    "model_a_v9": "MODEL_A_V9_SEED",
    "model_a_v10": "MODEL_A_V10_SEED",
    "model_b_linear": "MODEL_B_SEED",
}
CONFIG_ENVIRONMENT_BY_AGENT = {
    "model_a_v6": "MODEL_A_V6_CONFIG_PATH",
    "model_a_v6_stable": "MODEL_A_V6_STABLE_CONFIG_PATH",
    "model_a_v6_ensemble": "MODEL_A_V6_ENSEMBLE_CONFIG_PATH",
    "model_a_v6_consensus": "MODEL_A_V6_CONSENSUS_CONFIG_PATH",
    "model_a_v8": "MODEL_A_V8_CONFIG_PATH",
    "model_a_v9": "MODEL_A_V9_CONFIG_PATH",
    "model_a_v10": "MODEL_A_V10_CONFIG_PATH",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    # Preserve porcelain status' leading column; stripping it corrupts the
    # first dirty path (for example " M file" became "ile").
    return completed.stdout.rstrip("\r\n") if completed.returncode == 0 else None


def repository_state(excluded_directory: Path) -> dict:
    """Report source-tree changes without counting generated evaluation files."""
    status = _git("status", "--porcelain", "--untracked-files=all") or ""
    excluded_directory = excluded_directory.resolve()
    dirty_paths = []
    for line in status.splitlines():
        relative_path = line[3:].split(" -> ")[-1]
        candidate = (REPOSITORY_ROOT / relative_path).resolve()
        try:
            candidate.relative_to(excluded_directory)
        except ValueError:
            dirty_paths.append(relative_path)
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def metrics_by_agent(stats: dict) -> dict[str, dict[str, float | int]]:
    metrics = {}
    for name, raw in stats.get("by_agent", {}).items():
        rounds = int(raw.get("rounds", 0))
        steps = int(raw.get("steps", 0))
        metrics[name] = {
            "rounds": rounds,
            "score": int(raw.get("score", 0)),
            "score_per_round": float(raw.get("score", 0)) / rounds if rounds else 0.0,
            "coins": int(raw.get("coins", 0)),
            "kills": int(raw.get("kills", 0)),
            "suicides": int(raw.get("suicides", 0)),
            "invalid_actions": int(raw.get("invalid", 0)),
            "steps": steps,
            "mean_decision_time_ms": 1000.0 * float(raw.get("time", 0.0)) / steps if steps else 0.0,
        }
    return metrics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def referenced_checkpoints(agent_config: Path | None) -> list[dict[str, str]]:
    """Resolve secondary model files bound by an immutable agent config."""
    if agent_config is None:
        return []
    config = json.loads(agent_config.read_text(encoding="utf-8"))
    references = []
    for field, role in (("nonlinear_policy", "nonlinear_root_policy"),
                        ("residual_policy", "bounded_residual_root_policy"),
                        ("v4_anchor", "frozen_v4_anchor")):
        binding = config.get(field, {})
        if not binding.get("enabled", False):
            continue
        path = Path(binding["checkpoint_path"])
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        if not path.is_file():
            raise ValueError(f"Referenced checkpoint does not exist: {path}")
        actual = sha256_file(path)
        if actual != binding.get("checkpoint_sha256"):
            raise ValueError(f"Referenced {field} checkpoint hash mismatch")
        references.append({"role": role, "path": _relative(path), "sha256": actual})
    return references


def evaluation_environment(
    target_agent: str,
    agent_seed: int,
    weight_path: Path | None,
    base_environment: dict[str, str] | None = None,
    agent_config: Path | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build the child environment and the reproducible manifest subset."""
    seed_environment = SEED_ENVIRONMENT_BY_AGENT.get(target_agent)
    if seed_environment is None:
        raise ValueError(f"Agent seed injection is not supported for {target_agent!r}")
    weight_environment = WEIGHT_ENVIRONMENT_BY_AGENT.get(target_agent)
    if weight_path is not None and weight_environment is None:
        raise ValueError(f"Checkpoint injection is not supported for {target_agent!r}")
    child_environment = dict(os.environ if base_environment is None else base_environment)
    overrides = {seed_environment: str(agent_seed)}
    child_environment[seed_environment] = str(agent_seed)
    if weight_path is not None:
        resolved = str(weight_path.resolve())
        child_environment[weight_environment] = resolved
        overrides[weight_environment] = resolved
    config_environment = CONFIG_ENVIRONMENT_BY_AGENT.get(target_agent)
    if config_environment is not None and agent_config is None:
        raise ValueError(f"An explicit agent config is required for {target_agent!r}")
    if agent_config is not None:
        if config_environment is None:
            raise ValueError(f"Agent config injection is not supported for {target_agent!r}")
        resolved_config = str(agent_config.resolve())
        child_environment[config_environment] = resolved_config
        overrides[config_environment] = resolved_config
    return child_environment, overrides


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", nargs="+", required=True, help="Agents in play order; the first is evaluated.")
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--agent-seed", type=int,
        help="Seed for the evaluated agent's private RNG; defaults to --seed.",
    )
    parser.add_argument(
        "--opponent-seed", type=int,
        help="Required deterministic seed when seeded_random_agent is present.",
    )
    parser.add_argument("--variant", required=True, help="Stable experimental variant name used for grouping.")
    parser.add_argument("--run-id", help="Optional unique id; generated when omitted.")
    parser.add_argument("--weight-path", type=Path, help="Actual weight file loaded by the evaluated agent.")
    parser.add_argument("--agent-config", type=Path, help="Immutable agent JSON config, required by model_a_v6.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    target_agent = args.agents[0]
    agent_seed = args.seed if args.agent_seed is None else args.agent_seed
    uses_seeded_opponent = "seeded_random_agent" in args.agents[1:]
    if uses_seeded_opponent and args.opponent_seed is None:
        raise SystemExit("--opponent-seed is required with seeded_random_agent")
    if args.opponent_seed is not None and not uses_seeded_opponent:
        raise SystemExit("--opponent-seed requires seeded_random_agent")
    run_id = args.run_id or (
        f"eval-{target_agent}-{args.scenario}-s{args.seed}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None:
        raise SystemExit("--run-id may contain only letters, digits, dots, underscores and hyphens")
    output_dir = args.output_dir.resolve()
    stats_path = output_dir / f"{run_id}.stats.json"
    manifest_path = output_dir / f"{run_id}.json"
    if stats_path.exists() or manifest_path.exists():
        raise SystemExit(f"Refusing to overwrite an existing evaluation: {run_id}")
    weight_path = args.weight_path
    if weight_path is None and target_agent in DEFAULT_WEIGHT_FILES:
        weight_path = REPOSITORY_ROOT / DEFAULT_WEIGHT_FILES[target_agent]
    elif weight_path is not None and not weight_path.is_absolute():
        weight_path = REPOSITORY_ROOT / weight_path
    if weight_path is not None and not weight_path.is_file():
        raise SystemExit(f"Weight file does not exist: {weight_path}")
    if target_agent == "model_a_v6" and weight_path is None:
        raise SystemExit("model_a_v6 requires an explicit --weight-path")
    agent_config = args.agent_config
    if agent_config is not None and not agent_config.is_absolute():
        agent_config = REPOSITORY_ROOT / agent_config
    if agent_config is not None and not agent_config.is_file():
        raise SystemExit(f"Agent config does not exist: {agent_config}")
    try:
        secondary_checkpoints = referenced_checkpoints(agent_config)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    try:
        child_environment, environment_overrides = evaluation_environment(
            target_agent, agent_seed, weight_path, agent_config=agent_config,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.opponent_seed is not None:
        child_environment["SEEDED_RANDOM_AGENT_SEED"] = str(args.opponent_seed)
        environment_overrides["SEEDED_RANDOM_AGENT_SEED"] = str(args.opponent_seed)

    command = [
        sys.executable, "main.py", "play", "--agents", *args.agents,
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", args.scenario, "--n-rounds", str(args.rounds),
        "--seed", str(args.seed), "--save-stats", str(stats_path),
    ]
    manifest = {
        "schema_version": 1,
        "kind": "evaluation",
        "run_id": run_id,
        "variant": args.variant,
        "status": "running",
        "started_at_utc": _utc_now(),
        "repository": repository_state(output_dir),
        "checkpoint": None if weight_path is None else {
            "path": _relative(weight_path),
            "sha256": sha256_file(weight_path),
        },
        "agent_config": None if agent_config is None else {
            "path": _relative(agent_config),
            "sha256": sha256_file(agent_config),
        },
        "referenced_checkpoints": secondary_checkpoints,
        "evaluation": {
            "target_agent": target_agent,
            "agents": args.agents,
            "scenario": args.scenario,
            "rounds": args.rounds,
            "seed": args.seed,
            "agent_seed": agent_seed,
            "opponent_seed": args.opponent_seed,
            "training_agents": 0,
            "continue_without_training": True,
        },
        "command": command,
        "artifacts": {"raw_stats": _relative(stats_path)},
        "environment_overrides": environment_overrides,
    }
    _write_json(manifest_path, manifest)
    checkpoint_hash_before = None if weight_path is None else sha256_file(weight_path)
    config_hash_before = None if agent_config is None else sha256_file(agent_config)
    secondary_hashes_before = {entry["path"]: entry["sha256"] for entry in secondary_checkpoints}
    completed = subprocess.run(
        command, cwd=REPOSITORY_ROOT, env=child_environment, check=False,
    )
    manifest["ended_at_utc"] = _utc_now()
    manifest["exit_code"] = completed.returncode
    if completed.returncode == 0 and stats_path.is_file():
        if weight_path is not None and sha256_file(weight_path) != checkpoint_hash_before:
            manifest["status"] = "failed"
            manifest["error"] = "Evaluation modified the checkpoint or weights file"
            _write_json(manifest_path, manifest)
            return 1
        if agent_config is not None and sha256_file(agent_config) != config_hash_before:
            manifest["status"] = "failed"
            manifest["error"] = "Evaluation modified the agent config file"
            _write_json(manifest_path, manifest)
            return 1
        if any(sha256_file(REPOSITORY_ROOT / path) != expected
               for path, expected in secondary_hashes_before.items()):
            manifest["status"] = "failed"
            manifest["error"] = "Evaluation modified a referenced checkpoint"
            _write_json(manifest_path, manifest)
            return 1
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        manifest["metrics_by_agent"] = metrics_by_agent(stats)
        manifest["target_metrics"] = manifest["metrics_by_agent"].get(target_agent)
        manifest["status"] = "completed"
    else:
        manifest["status"] = "failed"
    _write_json(manifest_path, manifest)
    print(manifest_path)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
