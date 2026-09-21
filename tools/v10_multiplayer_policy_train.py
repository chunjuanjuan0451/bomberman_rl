"""Collect seeded multiplayer search targets and distil a fresh v10 policy head."""

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
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_code.model_a_v10.interfaces import ACTIONS, BlendedExpertNetwork, LinearExpertNetwork
from agent_code.model_a_v10.policy_head import NonlinearPolicyHead, prepare_policy_features
from agent_code.model_a_v10.runtime import opponent_aware_state_features
from agent_code.model_a_v10.search import StochasticPUCT
from agent_code.model_a_v10.simulator import SimAgent, SimState, step
from environment import BombeRLeWorld


RANDOM_ACTIONS = np.asarray(("RIGHT", "LEFT", "UP", "DOWN", "BOMB"))
RANDOM_PROBABILITIES = np.asarray((0.23, 0.23, 0.23, 0.23, 0.08))
REQUIRED = {
    "run_id", "scenario", "training_seed", "episodes", "train_episodes",
    "validation_episodes", "opponent_count", "max_steps", "teacher_checkpoint",
    "teacher_checkpoint_sha256", "teacher_search_simulations", "search_horizon_plies",
    "teacher_head_weights", "opponent_distribution", "feature_schema", "forbidden_root_actions",
    "epochs", "patience", "min_delta", "learning_rate", "weight_decay", "batch_size",
    "wait_cap", "minimum_opponent_visible_fraction", "checkpoint_path", "trajectory_dir",
    "manifest_path", "metrics_path", "training", "h2h",
}
PROVENANCE = (
    "agent_code/model_a_v10/interfaces.py", "agent_code/model_a_v10/policy_head.py",
    "agent_code/model_a_v10/runtime.py", "agent_code/model_a_v10/search.py",
    "agent_code/model_a_v10/simulator.py", "agent_code/seeded_random_agent/callbacks.py",
    "tools/v10_multiplayer_policy_train.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    missing = REQUIRED - config.keys()
    if missing:
        raise ValueError(f"missing multiplayer distillation fields: {sorted(missing)}")
    if config["scenario"] != "coin-heaven" or not config["training"] or config["h2h"]:
        raise ValueError("s103011 is training-only coin-heaven and never H2H")
    if int(config["opponent_count"]) != 3:
        raise ValueError("official-distribution collection requires exactly three opponents")
    if config["opponent_distribution"] != "seeded-random-agent-v1":
        raise ValueError("unexpected opponent action distribution")
    if config["feature_schema"] != "centered-opponent-projection-v1":
        raise ValueError("unexpected multiplayer policy feature schema")
    if config["forbidden_root_actions"] != ["BOMB"]:
        raise ValueError("coin-heaven teacher must mask root BOMB")
    expected_weights = {"policy": 0.0, "value": 0.1, "opponent": 0.0, "risk": 0.0}
    if config["teacher_head_weights"] != expected_weights:
        raise ValueError("s103011 teacher head weights are immutable")
    if int(config["episodes"]) != int(config["train_episodes"]) + int(config["validation_episodes"]):
        raise ValueError("train and validation episode counts must sum to episodes")
    for name in ("episodes", "train_episodes", "validation_episodes", "max_steps",
                 "teacher_search_simulations", "search_horizon_plies", "epochs", "patience",
                 "batch_size"):
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("wait_cap", "minimum_opponent_visible_fraction"):
        if not 0.0 < float(config[name]) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    return config


def initial_multiplayer_state(seed: int, max_steps: int, opponent_count: int = 3) -> SimState:
    agents = [SimAgent("model_a_v10", 1, 1)] + [
        SimAgent(f"seeded_random_agent_{index}", 1, 1) for index in range(opponent_count)
    ]
    carrier = SimpleNamespace(
        rng=np.random.default_rng(seed), args=SimpleNamespace(scenario="coin-heaven"), agents=agents,
    )
    field, coins, positioned_agents = BombeRLeWorld.build_arena(carrier)
    return SimState(
        field, positioned_agents, {(coin.x, coin.y): coin.collectable for coin in coins},
        max_steps=max_steps,
    )


def seeded_opponent_rng(base_seed: int, index: int) -> np.random.Generator:
    """Match seeded_random_agent setup for opponent suffixes 0..2."""
    return np.random.default_rng(np.random.SeedSequence([base_seed, index]))


def sample_official_random_action(rng: np.random.Generator) -> str:
    """Match random_agent exactly, including actions that the world may reject."""
    return str(rng.choice(RANDOM_ACTIONS, p=RANDOM_PROBABILITIES))


class OfficialRandomOpponentPUCT(StochasticPUCT):
    """Use the official random-agent proposal distribution inside teacher search."""

    def _sample_opponent_action(self, state, player, root_player, rng) -> str:
        del state, player, root_player
        return sample_official_random_action(rng)


def policy_array(policy: dict[str, float]) -> np.ndarray:
    values = np.zeros(len(ACTIONS), dtype=np.float32)
    for action, probability in policy.items():
        values[ACTIONS.index(action)] = float(probability)
    total = float(values.sum())
    if not np.isfinite(values).all() or total <= 0:
        raise FloatingPointError("invalid policy array")
    return values / total


def visit_target(visits: dict[str, int]) -> np.ndarray:
    return policy_array({action: float(count) for action, count in visits.items()})


def collect_episode(state: SimState, teacher, config: dict, episode_seed: int) -> tuple[dict, SimState]:
    searcher = OfficialRandomOpponentPUCT(
        teacher, horizon=int(config["search_horizon_plies"]), forbidden_root_actions=("BOMB",),
    )
    opponent_rngs = [seeded_opponent_rng(episode_seed, index)
                     for index in range(int(config["opponent_count"]))]
    world_rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 991]))
    rows = {"features": [], "visit_policy": [], "base_prior": [], "legal_mask": [],
            "search_action": [], "opponent_actions": []}
    current = state
    # Stop when the multiplayer interaction ends.  Continuing a surviving
    # root through hundreds of solo cleanup steps would recreate the exact
    # distribution mismatch this run is designed to remove.
    while (not current.ended and current.agents[0].alive
           and any(agent.alive for agent in current.agents[1:])):
        features = opponent_aware_state_features(current, 0)
        base = dict(teacher.evaluate(current, 0).policy)
        base.pop("BOMB", None)
        result = searcher.search(
            current, 0, simulations=int(config["teacher_search_simulations"]),
            seed=episode_seed * (int(config["max_steps"]) + 1) + current.step_count,
        )
        legal = np.zeros(len(ACTIONS), dtype=bool)
        for action in result.raw_prior:
            legal[ACTIONS.index(action)] = True
        proposed = [sample_official_random_action(rng) if current.agents[index + 1].alive else "WAIT"
                    for index, rng in enumerate(opponent_rngs)]
        rows["features"].append(features.astype(np.float16))
        rows["visit_policy"].append(visit_target(result.root_visits))
        rows["base_prior"].append(policy_array(base))
        rows["legal_mask"].append(legal)
        rows["search_action"].append(ACTIONS.index(result.action))
        rows["opponent_actions"].append([ACTIONS.index(action) for action in proposed])
        joint = [result.action] + proposed
        active = current.active_ids()
        permutation = tuple(int(value) for value in world_rng.permutation(len(active)))
        current = step(current, joint, permutation)
    if not rows["features"]:
        raise ValueError(f"episode {episode_seed} produced no root states")
    arrays = {
        "features": np.stack(rows["features"]),
        "visit_policy": np.stack(rows["visit_policy"]),
        "base_prior": np.stack(rows["base_prior"]),
        "legal_mask": np.stack(rows["legal_mask"]),
        "search_action": np.asarray(rows["search_action"], dtype=np.int8),
        "opponent_actions": np.asarray(rows["opponent_actions"], dtype=np.int8),
    }
    return arrays, current


def save_shard(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def load_shards(paths: list[Path]) -> dict[str, np.ndarray]:
    keys = ("features", "visit_policy", "base_prior", "legal_mask", "search_action",
            "opponent_actions")
    loaded = {key: [] for key in keys}
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            if set(shard.files) != set(keys):
                raise ValueError(f"unexpected trajectory schema: {path}")
            for key in keys:
                loaded[key].append(shard[key])
    return {key: np.concatenate(values) for key, values in loaded.items()}


def dataset_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def opponent_visible_fraction(features: np.ndarray) -> float:
    occupancy = np.asarray(features)[:, 3]
    return float(np.mean(np.any(occupancy < -0.5, axis=(1, 2))))


def balanced_indices(actions: np.ndarray, wait_cap: float,
                     rng: np.random.Generator) -> np.ndarray:
    wait_index = ACTIONS.index("WAIT")
    non_wait = np.flatnonzero(actions != wait_index)
    wait = np.flatnonzero(actions == wait_index)
    if not len(non_wait):
        raise ValueError("cannot balance an all-WAIT training set")
    maximum_wait = int(wait_cap / (1.0 - wait_cap) * len(non_wait)) if wait_cap < 1.0 else len(wait)
    if len(wait) > maximum_wait:
        wait = rng.choice(wait, maximum_wait, replace=False)
    return rng.permutation(np.concatenate((non_wait, wait)))


def masked_log_probabilities(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    if logits.shape != legal.shape or not torch.all(legal.any(dim=1)):
        raise ValueError("each training row must contain at least one legal action")
    return torch.log_softmax(logits.masked_fill(~legal, -1e9), dim=1)


def probabilities(model: NonlinearPolicyHead, data: dict[str, np.ndarray], batch_size: int) -> np.ndarray:
    model.eval()
    features = prepare_policy_features(data["features"])
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(features[start:start + batch_size])
            legal = torch.from_numpy(data["legal_mask"][start:start + batch_size].astype(bool))
            outputs.append(torch.exp(masked_log_probabilities(model(x), legal)).numpy())
    return np.concatenate(outputs)


def policy_metrics(target: np.ndarray, probability: np.ndarray,
                   target_actions: np.ndarray) -> dict[str, float]:
    prediction = probability.argmax(axis=1)
    non_wait = target_actions != ACTIONS.index("WAIT")
    return {
        "cross_entropy": float(-np.mean(np.sum(target * np.log(probability + 1e-12), axis=1))),
        "top1_accuracy": float(np.mean(prediction == target_actions)),
        "non_wait_top1_accuracy": float(np.mean(prediction[non_wait] == target_actions[non_wait]))
        if np.any(non_wait) else 0.0,
        "wait_prediction_fraction": float(np.mean(prediction == ACTIONS.index("WAIT"))),
    }


def train_head(train_data: dict[str, np.ndarray], validation_data: dict[str, np.ndarray],
               config: dict) -> tuple[NonlinearPolicyHead, dict]:
    seed = int(config["training_seed"])
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = NonlinearPolicyHead()
    model.feature_schema = "centered-opponent-projection-v1"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    x_train = torch.from_numpy(prepare_policy_features(train_data["features"]))
    y_train = torch.from_numpy(train_data["visit_policy"].astype(np.float32))
    legal_train = torch.from_numpy(train_data["legal_mask"].astype(bool))
    x_validation = torch.from_numpy(prepare_policy_features(validation_data["features"]))
    y_validation = torch.from_numpy(validation_data["visit_policy"].astype(np.float32))
    legal_validation = torch.from_numpy(validation_data["legal_mask"].astype(bool))
    rng = np.random.default_rng(seed + 17)
    best_state = None
    best_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        order = balanced_indices(train_data["search_action"], float(config["wait_cap"]), rng)
        weighted_loss = 0.0
        for start in range(0, len(order), int(config["batch_size"])):
            indices = torch.from_numpy(order[start:start + int(config["batch_size"])]).long()
            optimizer.zero_grad(set_to_none=True)
            log_probability = masked_log_probabilities(model(x_train[indices]), legal_train[indices])
            loss = -(y_train[indices] * log_probability).sum(dim=1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite multiplayer policy loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weighted_loss += float(loss.detach()) * len(indices)
        model.eval()
        with torch.inference_mode():
            validation_loss = float((-(y_validation * masked_log_probabilities(
                model(x_validation), legal_validation)).sum(dim=1).mean()).item())
        history.append({"epoch": epoch, "train_ce": weighted_loss / len(order),
                        "validation_ce": validation_loss, "samples_used": len(order)})
        if validation_loss < best_loss - float(config["min_delta"]):
            best_loss = validation_loss
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    if best_state is None:
        raise RuntimeError("training did not produce a finite best checkpoint")
    model.load_state_dict(best_state)
    if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise FloatingPointError("non-finite policy-head parameters")
    return model.eval(), {"best_validation_ce": best_loss, "epochs_completed": len(history),
                          "history": history}


def output_paths(root: Path, config: dict) -> dict[str, Path]:
    return {name: root / config[key] for name, key in (
        ("checkpoint", "checkpoint_path"), ("trajectory_dir", "trajectory_dir"),
        ("manifest", "manifest_path"), ("metrics", "metrics_path"),
    )}


def validate_inputs(root: Path, config_path: Path, config: dict) -> tuple[Path, dict[str, Path]]:
    teacher = root / config["teacher_checkpoint"]
    if not teacher.is_file() or sha256(teacher) != config["teacher_checkpoint_sha256"]:
        raise ValueError("frozen teacher checkpoint hash mismatch")
    paths = output_paths(root, config)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite s103011 artifacts: {existing}")
    if sha256(config_path) == "":
        raise AssertionError("unreachable config digest")
    return teacher, paths


def run(root: Path, config_path: Path, config: dict) -> dict:
    teacher_path, paths = validate_inputs(root, config_path, config)
    config_hash = sha256(config_path)
    teacher_hash = sha256(teacher_path)
    teacher_model = LinearExpertNetwork.load(teacher_path)
    weights = config["teacher_head_weights"]
    teacher = BlendedExpertNetwork(
        teacher_model, policy_weight=float(weights["policy"]), value_weight=float(weights["value"]),
        opponent_weight=float(weights["opponent"]), risk_weight=float(weights["risk"]),
    )
    source_hashes = {path: sha256(root / path) for path in PROVENANCE}
    manifest = {
        "kind": "v10-multiplayer-policy-distillation", "status": "running",
        "run_id": config["run_id"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "config_sha256": config_hash,
        "teacher_checkpoint_sha256": teacher_hash, "source_sha256": source_hashes,
        "progress": {"episodes_completed": 0, "trajectory_shards": []},
    }
    paths["trajectory_dir"].mkdir(parents=True)
    atomic_json(paths["manifest"], manifest)
    shard_paths = []
    episode_rows = []
    try:
        for offset in range(int(config["episodes"])):
            episode_seed = int(config["training_seed"]) + offset
            arrays, terminal = collect_episode(
                initial_multiplayer_state(episode_seed, int(config["max_steps"])),
                teacher, config, episode_seed,
            )
            shard = paths["trajectory_dir"] / f"episode-{offset + 1:05d}-s{episode_seed}.npz"
            save_shard(shard, arrays)
            shard_paths.append(shard)
            row = {"episode": offset + 1, "seed": episode_seed, "transitions": len(arrays["features"]),
                   "score": terminal.agents[0].score, "survived": terminal.agents[0].alive,
                   "steps": terminal.step_count,
                   "opponent_visible_fraction": opponent_visible_fraction(arrays["features"])}
            episode_rows.append(row)
            manifest["progress"] = {"episodes_completed": offset + 1,
                                    "trajectory_shards": [str(path) for path in shard_paths],
                                    "last_episode": row}
            atomic_json(paths["manifest"], manifest)
        split = int(config["train_episodes"])
        train_data = load_shards(shard_paths[:split])
        validation_data = load_shards(shard_paths[split:])
        visible = opponent_visible_fraction(np.concatenate(
            (train_data["features"], validation_data["features"])))
        if visible < float(config["minimum_opponent_visible_fraction"]):
            raise ValueError(f"opponent-aware dataset gate failed: {visible:.6f}")
        data_hash = dataset_sha256(shard_paths)
        model, training = train_head(train_data, validation_data, config)
        candidate = policy_metrics(
            validation_data["visit_policy"], probabilities(model, validation_data,
                                                             int(config["batch_size"])),
            validation_data["search_action"],
        )
        baseline = policy_metrics(
            validation_data["visit_policy"], validation_data["base_prior"],
            validation_data["search_action"],
        )
        relative_ce_gain = (baseline["cross_entropy"] - candidate["cross_entropy"]) / baseline["cross_entropy"]
        non_wait_gain = candidate["non_wait_top1_accuracy"] - baseline["non_wait_top1_accuracy"]
        gate = {
            "passed": bool(relative_ce_gain >= 0.05 and non_wait_gain >= 0.05
                           and candidate["wait_prediction_fraction"] <= 0.35),
            "minimum_relative_ce_gain": 0.05, "relative_ce_gain": float(relative_ce_gain),
            "minimum_non_wait_top1_gain": 0.05, "non_wait_top1_gain": float(non_wait_gain),
            "maximum_wait_prediction_fraction": 0.35,
        }
        metadata = {
            "architecture": NonlinearPolicyHead.architecture, "input_size": 1445,
            "action_count": len(ACTIONS), "training_seed": int(config["training_seed"]),
            "config_sha256": config_hash, "data_sha256": data_hash,
            "teacher_checkpoint_sha256": teacher_hash, "opponent_count": 3,
            "opponent_distribution": "seeded-random-agent-v1",
            "feature_schema": "centered-opponent-projection-v1",
        }
        paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
        temporary_checkpoint = paths["checkpoint"].with_suffix(".pt.tmp")
        NonlinearPolicyHead.save_checkpoint(temporary_checkpoint, model, metadata)
        temporary_checkpoint.replace(paths["checkpoint"])
        metrics = {
            "kind": "v10-multiplayer-policy-offline-gate", "run_id": config["run_id"],
            "status": "completed", "split": {"train_episodes": split,
                                                "validation_episodes": int(config["validation_episodes"]),
                                                "train_states": len(train_data["features"]),
                                                "validation_states": len(validation_data["features"])},
            "opponent_visible_fraction": visible, "baseline": baseline, "candidate": candidate,
            "training": training, "gate": gate, "data_sha256": data_hash,
            "checkpoint_sha256": sha256(paths["checkpoint"]),
        }
        atomic_json(paths["metrics"], metrics)
        if sha256(teacher_path) != teacher_hash:
            raise RuntimeError("frozen teacher changed during training")
        manifest.update({
            "status": "completed", "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "machine": {"python": sys.version, "torch": torch.__version__,
                        "platform": platform.platform()},
            "repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "episodes": episode_rows, "data_sha256": data_hash, "offline_gate": gate,
            "artifacts": {name: str(path) for name, path in paths.items()},
        })
        atomic_json(paths["manifest"], manifest)
        return metrics
    except Exception as exc:
        manifest.update({"status": "failed", "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                         "failure": {"type": type(exc).__name__, "message": str(exc),
                                     "traceback": traceback.format_exc()}})
        atomic_json(paths["manifest"], manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--execute", action="store_true", help="Create data and train; default is dry-run.")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = load_config(config_path)
    teacher, paths = validate_inputs(root, config_path, config)
    plan = {"mode": "execute" if args.execute else "dry-run", "run_id": config["run_id"],
            "teacher": str(teacher), "episodes": config["episodes"],
            "split": [config["train_episodes"], config["validation_episodes"]],
            "opponents": config["opponent_count"], "simulations_per_state": config["teacher_search_simulations"],
            "outputs": {name: str(path) for name, path in paths.items()}}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    metrics = run(root, config_path, config)
    print(json.dumps({"run_id": config["run_id"], "gate": metrics["gate"],
                      "baseline": metrics["baseline"], "candidate": metrics["candidate"],
                      "metrics": str(paths["metrics"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
