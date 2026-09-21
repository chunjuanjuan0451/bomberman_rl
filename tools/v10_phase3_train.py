"""Phase-3 coin-heaven Expert Iteration; search supplies every policy target."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_code.model_a_v10.interfaces import (ACTIONS, BlendedExpertNetwork, HeuristicNetwork,
                                                LinearExpertNetwork)
from agent_code.model_a_v10.search import StochasticPUCT
from agent_code.model_a_v10.simulator import SimAgent, SimState, step
from agent_code.model_a_v10.runtime import state_features
from environment import BombeRLeWorld

HEADS = ("policy", "value", "opponent", "risk")
PROVENANCE_FILES = (
    "agent_code/model_a_v10/interfaces.py", "agent_code/model_a_v10/runtime.py",
    "agent_code/model_a_v10/search.py", "agent_code/model_a_v10/simulator.py",
    "tools/v10_phase3_train.py",
)


def source_hashes(root: Path, files=PROVENANCE_FILES) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def _validate_weights(weights: dict[str, float], label: str) -> dict[str, float]:
    invalid = {name: weight for name, weight in weights.items() if not 0.0 <= float(weight) <= 1.0}
    if invalid:
        raise ValueError(f"{label} weights must stay in [0, 1]: {invalid}")
    return {name: float(weight) for name, weight in weights.items()}


def iteration_head_weights(config: dict, iteration: int) -> dict[str, float] | None:
    """Resolve legacy uniform or explicit per-head self-play schedules."""
    legacy = config.get("student_weight_schedule")
    explicit = any(f"{head}_student_weight_schedule" in config for head in HEADS)
    if legacy is None and not explicit:
        return None
    weights = {}
    for head in HEADS:
        schedule = config.get(f"{head}_student_weight_schedule", legacy)
        if schedule is None:
            raise ValueError(f"missing {head}_student_weight_schedule")
        if len(schedule) != int(config["iterations"]):
            raise ValueError(f"{head} student schedule must match iterations")
        weights[head] = float(schedule[iteration])
    return _validate_weights(weights, f"iteration {iteration + 1}")


def validation_head_weights(config: dict) -> dict[str, float]:
    legacy = float(config.get("validation_student_weight", 0.1))
    return _validate_weights({
        head: float(config.get(f"validation_{head}_student_weight", legacy)) for head in HEADS
    }, "validation")


def blended_network(model: LinearExpertNetwork, weights: dict[str, float]) -> BlendedExpertNetwork:
    return BlendedExpertNetwork(
        model, policy_weight=weights["policy"], value_weight=weights["value"],
        opponent_weight=weights["opponent"], risk_weight=weights["risk"],
    )


def load(path: Path) -> dict:
    config = json.loads(path.read_text())
    required = {"run_id", "scenario", "episodes", "iterations", "episodes_per_iteration", "max_steps",
                "search_simulations", "seed", "validation_seeds", "learning_rate", "epochs_per_iteration",
                "batch_size", "checkpoint_path", "trajectory_path"}
    missing = required - config.keys()
    if missing or config["scenario"] != "coin-heaven" or not config.get("training") or config.get("h2h"):
        raise ValueError(f"Invalid Phase-3 config; missing={sorted(missing)}")
    if config["episodes"] != config["iterations"] * config["episodes_per_iteration"]:
        raise ValueError("episodes must equal iterations * episodes_per_iteration")
    for iteration in range(int(config["iterations"])):
        iteration_head_weights(config, iteration)
    maximum_fraction = float(config.get("max_wait_fraction", 1.0))
    if not 0.0 < maximum_fraction <= 1.0:
        raise ValueError("max_wait_fraction must be in (0, 1]")
    validation_head_weights(config)
    if bool(config.get("initial_checkpoint")) != bool(config.get("initial_checkpoint_sha256")):
        raise ValueError("initial_checkpoint and initial_checkpoint_sha256 must be provided together")
    return config


def initial_state(seed: int, max_steps: int) -> SimState:
    carrier = SimpleNamespace(rng=np.random.default_rng(seed), args=SimpleNamespace(scenario="coin-heaven"),
                              agents=[SimAgent("expert", 1, 1)])
    field, coins, agents = BombeRLeWorld.build_arena(carrier)
    return SimState(field, agents, {(coin.x, coin.y): coin.collectable for coin in coins}, max_steps=max_steps)


def policy_target(visits: dict[str, int]) -> np.ndarray:
    target = np.zeros(len(ACTIONS), dtype=np.float32)
    total = sum(visits.values())
    for action, count in visits.items():
        target[ACTIONS.index(action)] = count / max(1, total)
    return target


def play_episode(state: SimState, network, config: dict, episode_seed: int) -> tuple[list[dict], SimState]:
    forbidden = ("BOMB",) if config["scenario"] == "coin-heaven" else ()
    searcher = StochasticPUCT(network, horizon=int(config["search_horizon_plies"]),
                             forbidden_root_actions=forbidden)
    records: list[dict] = []
    current = state
    while not current.ended:
        result = searcher.search(current, 0, simulations=int(config["search_simulations"]),
                                 seed=episode_seed + current.step_count)
        records.append({"features": state_features(current).astype(np.float16),
                        "visit_policy": policy_target(result.root_visits),
                        "opponent_action": -1, "search_action": result.action,
                        "simulations": result.simulations, "risk_prediction": result.risk,
                        "score_at_state": current.agents[0].score,
                        "step_at_state": current.step_count})
        current = step(current, [result.action], [0])
    score = current.agents[0].score
    normalizer = max(1, len(state.coins))
    death = np.float32(not current.agents[0].alive)
    risk_horizon = int(config.get("risk_horizon_plies", 8))
    for record in records:
        # Non-negative return-to-go avoids labelling a partially successful
        # coin-heaven episode as a loss and gives late states a local target.
        record["outcome"] = np.float32(np.clip(
            (score - int(record["score_at_state"])) / normalizer, 0.0, 1.0))
        record["future_death"] = np.float32(
            bool(death) and current.step_count - int(record["step_at_state"]) <= risk_horizon)
    return records, current


def train(model: LinearExpertNetwork, records: list[dict], config: dict, rng) -> dict[str, float]:
    losses: list[tuple[int, dict[str, float]]] = []
    used = 0
    for _ in range(int(config["epochs_per_iteration"])):
        non_wait = np.asarray([index for index, record in enumerate(records)
                               if record["search_action"] != "WAIT"], dtype=int)
        wait = np.asarray([index for index, record in enumerate(records)
                           if record["search_action"] == "WAIT"], dtype=int)
        maximum_fraction = float(config.get("max_wait_fraction", 1.0))
        if not non_wait.size and maximum_fraction < 1.0:
            raise ValueError("WAIT balancing cannot train an all-WAIT dataset")
        if non_wait.size and maximum_fraction < 1.0:
            maximum_wait = int(maximum_fraction / max(1e-9, 1.0 - maximum_fraction) * non_wait.size)
            if wait.size > maximum_wait:
                wait = rng.choice(wait, maximum_wait, replace=False)
        order = rng.permutation(np.concatenate((non_wait, wait)))
        used += len(order)
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start:start + int(config["batch_size"])]
            if not len(indices):
                continue
            batch = [records[int(index)] for index in indices]
            features = np.stack([record["features"].astype(np.float32) for record in batch])
            policies = np.stack([record["visit_policy"] for record in batch])
            values = np.asarray([record["outcome"] for record in batch], dtype=np.float32)
            risk = np.asarray([record["future_death"] for record in batch], dtype=np.float32)
            losses.append((len(batch), model.train_batch(
                features, policies, values, np.zeros_like(policies),
                np.zeros(len(batch), dtype=np.float32), risk, float(config["learning_rate"]))))
    if not losses:
        raise ValueError("Phase-3 training received no records")
    total_weight = sum(weight for weight, _ in losses)
    result = {key: float(sum(weight * loss[key] for weight, loss in losses) / total_weight)
              for key in losses[0][1]}
    result.update({"samples_used": float(used),
                   "wait_fraction_used": float(len(wait) / max(1, len(order)))})
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"non-finite Phase-3 aggregate: {result}")
    return result


def model_diagnostics(model: LinearExpertNetwork, records: list[dict], batch_size: int) -> dict[str, float]:
    """Report unbalanced policy behaviour and value fit on all collected states."""
    policies, values = [], []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        features = np.stack([record["features"].astype(np.float32) for record in batch])
        policy, value, _, _ = model.batch_outputs(features)
        policies.append(policy)
        values.append(value)
    policy = np.concatenate(policies)
    value = np.concatenate(values)
    targets = np.stack([record["visit_policy"] for record in records])
    actions = np.asarray([ACTIONS.index(record["search_action"]) for record in records])
    predictions = np.argmax(policy, axis=1)
    non_wait = actions != ACTIONS.index("WAIT")
    outcomes = np.asarray([record["outcome"] for record in records], dtype=float)
    correlation = (float(np.corrcoef(value, outcomes)[0, 1])
                   if np.std(value) > 0 and np.std(outcomes) > 0 else 0.0)
    result = {
        "policy_ce_all": float(-np.mean(np.sum(targets * np.log(policy + 1e-12), axis=1))),
        "policy_top1_accuracy": float(np.mean(predictions == actions)),
        "non_wait_top1_accuracy": float(np.mean(predictions[non_wait] == actions[non_wait]))
        if np.any(non_wait) else 0.0,
        "raw_wait_argmax_fraction": float(np.mean(predictions == ACTIONS.index("WAIT"))),
        "value_mse_all": float(np.mean((value - outcomes) ** 2)),
        "value_target_correlation": correlation,
    }
    if not all(np.isfinite(metric) for metric in result.values()):
        raise FloatingPointError(f"non-finite v10 diagnostics: {result}")
    return result


def save_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        features=np.stack([r["features"] for r in records]),
                        visit_policy=np.stack([r["visit_policy"] for r in records]),
                        outcome=np.asarray([r["outcome"] for r in records], dtype=np.float32),
                        opponent_action=np.asarray([r["opponent_action"] for r in records], dtype=np.int8),
                        future_death=np.asarray([r["future_death"] for r in records], dtype=np.float32),
                        search_action=np.asarray([ACTIONS.index(r["search_action"]) for r in records], dtype=np.int8))


def write_manifest(path: Path, summary: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validation(model: LinearExpertNetwork, config: dict) -> dict[str, object]:
    rows = []
    blend_weights = validation_head_weights(config)
    blended = blended_network(model, blend_weights)
    for seed in config["validation_seeds"]:
        learned_records, learned = play_episode(
            initial_state(int(seed), int(config["max_steps"])), model, config, int(seed))
        _, heuristic = play_episode(
            initial_state(int(seed), int(config["max_steps"])), HeuristicNetwork(), config, int(seed))
        _, blend = play_episode(
            initial_state(int(seed), int(config["max_steps"])), blended, config, int(seed))
        raw = initial_state(int(seed), int(config["max_steps"]))
        while not raw.ended:
            policy = model.evaluate(raw, 0).policy
            policy.pop("BOMB", None)
            action = max(policy, key=lambda item: (policy[item], item)) if policy else "WAIT"
            raw = step(raw, [action], [0])
        rows.append({"seed": seed,
                     "learned_search_score": learned.agents[0].score,
                     "learned_raw_score": raw.agents[0].score,
                     "blended_search_score": blend.agents[0].score,
                     "heuristic_search_score": heuristic.agents[0].score,
                     "survived": learned.agents[0].alive, "steps": learned.step_count,
                     "transitions": len(learned_records)})
    learned_scores = [row["learned_search_score"] for row in rows]
    raw_scores = [row["learned_raw_score"] for row in rows]
    heuristic_scores = [row["heuristic_search_score"] for row in rows]
    blended_scores = [row["blended_search_score"] for row in rows]
    return {"rows": rows, "mean_learned_search_score": float(np.mean(learned_scores)),
            "mean_learned_raw_score": float(np.mean(raw_scores)),
            "mean_blended_search_score": float(np.mean(blended_scores)),
            "mean_heuristic_search_score": float(np.mean(heuristic_scores)),
            "search_gain_over_raw": float(np.mean(learned_scores) - np.mean(raw_scores)),
            "blend_retention_vs_heuristic": float(np.mean(blended_scores) / max(1e-9, np.mean(heuristic_scores))),
            "validation_student_weights": blend_weights,
            "survival_rate": float(np.mean([row["survived"] for row in rows]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load(config_path)
    root = Path.cwd()
    checkpoint, trajectories = (root / config["checkpoint_path"]), (root / config["trajectory_path"])
    manifest = root / "experiments/logs/runs" / f"{config['run_id']}.json"
    parts = trajectories.with_suffix("")
    milestones = [checkpoint.with_name(f"{checkpoint.stem}-iteration-{index + 1:02d}{checkpoint.suffix}")
                  for index in range(int(config["iterations"]))]
    if (checkpoint.exists() or trajectories.exists() or manifest.exists() or parts.exists()
            or any(path.exists() for path in milestones)):
        raise SystemExit("Refusing to overwrite existing Phase-3 artifact/run ID")
    input_size = int(np.prod(state_features(initial_state(int(config["seed"]), int(config["max_steps"]))).shape))
    initial_path = root / config["initial_checkpoint"] if config.get("initial_checkpoint") else None
    initial_hash = hashlib.sha256(initial_path.read_bytes()).hexdigest() if initial_path else None
    if initial_path and initial_hash != config["initial_checkpoint_sha256"]:
        raise ValueError("initial v10 checkpoint hash mismatch")
    model = (LinearExpertNetwork.load(initial_path) if initial_path
             else LinearExpertNetwork(input_size, int(config["seed"])))
    if model.value_w.shape[0] != input_size:
        raise ValueError("initial v10 checkpoint feature size mismatch")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    trajectories.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    summary = {"kind": "v10-phase3-expert-iteration", "status": "running", "run_id": config["run_id"],
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
               "config_sha256": config_hash, "training": "search-supervised from first transition",
               "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "source_sha256": source_hashes(root)}
    if initial_hash:
        summary["initial_checkpoint_sha256"] = initial_hash
    summary["progress"] = {"episodes_completed": 0, "iterations_completed": 0,
                           "trajectory_parts": [], "parameter_health": {}}
    write_manifest(manifest, summary)
    rng = np.random.default_rng(int(config["seed"]) + 99)
    all_records: list[dict] = []
    iterations = []
    try:
        parts.mkdir(parents=True)
        for iteration in range(int(config["iterations"])):
            weights = iteration_head_weights(config, iteration)
            if weights is not None:
                network = blended_network(model, weights)
                network_name = "blend-" + "-".join(
                    f"{head[0]}{weights[head]:.3f}" for head in HEADS)
            else:
                network = HeuristicNetwork() if iteration == 0 else model
                network_name = "heuristic" if iteration == 0 else "learned"
            episodes = []
            start = iteration * int(config["episodes_per_iteration"])
            for offset in range(int(config["episodes_per_iteration"])):
                episode_seed = int(config["seed"]) + start + offset
                records, terminal = play_episode(initial_state(episode_seed, int(config["max_steps"])),
                                                 network, config, episode_seed)
                all_records.extend(records)
                row = {"seed": episode_seed, "score": terminal.agents[0].score,
                       "survived": terminal.agents[0].alive, "steps": terminal.step_count,
                       "transitions": len(records)}
                episodes.append(row)
                part = parts / f"episode-{start + offset + 1:05d}-s{episode_seed}.npz"
                save_records(part, records)
                summary["progress"].update({"episodes_completed": start + offset + 1,
                                            "last_episode": row})
                summary["progress"]["trajectory_parts"].append(str(part))
                write_manifest(manifest, summary)
            losses = train(model, all_records, config, rng)
            milestone = checkpoint.with_name(f"{checkpoint.stem}-iteration-{iteration + 1:02d}{checkpoint.suffix}")
            model.save(milestone)
            iteration_row = {"iteration": iteration + 1,
                             "self_play_network": network_name,
                             "episodes": episodes, "losses": losses,
                             "model_diagnostics": model_diagnostics(
                                 model, all_records, int(config["batch_size"])),
                             "parameter_health": model.parameter_health(),
                             "transitions_total": len(all_records), "checkpoint": str(milestone)}
            iterations.append(iteration_row)
            summary["progress"].update({"iterations_completed": iteration + 1,
                                        "parameter_health": iteration_row["parameter_health"],
                                        "last_losses": losses})
            summary["iterations"] = iterations
            write_manifest(manifest, summary)
        save_records(trajectories, all_records)
        model.save(checkpoint)
        if initial_path and hashlib.sha256(initial_path.read_bytes()).hexdigest() != initial_hash:
            raise RuntimeError("initial v10 checkpoint changed during training")
        summary.update({"status": "completed", "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                        "machine": {"python": sys.version, "platform": platform.platform()},
                        "validation": validation(model, config),
                        "artifacts": {"checkpoint": str(checkpoint), "trajectories": str(trajectories),
                                      "trajectory_parts": str(parts)}})
        write_manifest(manifest, summary)
    except Exception as exc:
        summary.update({"status": "failed", "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                        "failure": {"type": type(exc).__name__, "message": str(exc),
                                    "traceback": traceback.format_exc()},
                        "iterations": iterations})
        write_manifest(manifest, summary)
        raise
    print(json.dumps({"run_id": config["run_id"], "transitions": len(all_records), "validation": summary["validation"],
                      "checkpoint": str(checkpoint), "trajectory": str(trajectories)}, indent=2))


if __name__ == "__main__":
    main()
