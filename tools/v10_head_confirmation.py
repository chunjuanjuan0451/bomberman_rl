"""Paired, inference-only confirmation gate for v10 per-head blending."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agent_code.model_a_v10.interfaces import HeuristicNetwork, LinearExpertNetwork
from tools.v10_phase3_train import (HEADS, PROVENANCE_FILES, blended_network, initial_state,
                                    play_episode, source_hashes)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict], variants: list[str], max_steps: int) -> dict[str, dict]:
    result = {}
    for variant in variants:
        scores = [row["results"][variant]["score"] for row in rows]
        steps = [row["results"][variant]["steps"] for row in rows]
        result[variant] = {
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "min_score": int(min(scores)),
            "max_score": int(max(scores)),
            "full_score_count": int(sum(score == 50 for score in scores)),
            "step_limit_count": int(sum(step == max_steps for step in steps)),
            "survival_rate": float(np.mean([
                row["results"][variant]["survived"] for row in rows
            ])),
        }
    return result


def evaluate_gate(rows: list[dict], aggregates: dict[str, dict], config: dict) -> dict:
    baseline = config["baseline_variant"]
    candidate = config["candidate_variant"]
    gates = config["gates"]
    wins = sum(row["results"][candidate]["score"] > row["results"][baseline]["score"]
               for row in rows)
    ties = sum(row["results"][candidate]["score"] == row["results"][baseline]["score"]
               for row in rows)
    baseline_mean = aggregates[baseline]["mean_score"]
    candidate_mean = aggregates[candidate]["mean_score"]
    gain_ratio = candidate_mean / max(1e-9, baseline_mean)
    step_limit_delta = (aggregates[candidate]["step_limit_count"]
                        - aggregates[baseline]["step_limit_count"])
    passed = (wins >= int(gates["minimum_paired_wins"])
              and gain_ratio >= float(gates["minimum_mean_gain_ratio"])
              and step_limit_delta <= int(gates["maximum_step_limit_delta"]))
    return {
        "passed": bool(passed), "baseline": baseline, "candidate": candidate,
        "paired_wins": int(wins), "paired_ties": int(ties),
        "mean_gain_ratio": float(gain_ratio), "step_limit_count_delta": int(step_limit_delta),
        "thresholds": gates,
    }


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    required = {"run_id", "scenario", "checkpoint_path", "checkpoint_sha256", "seeds",
                "variants", "baseline_variant", "candidate_variant", "max_steps",
                "search_simulations", "search_horizon_plies", "output_path", "gates"}
    missing = required - config.keys()
    if missing or config["scenario"] != "coin-heaven" or config.get("training") or config.get("h2h"):
        raise ValueError(f"invalid v10 confirmation config; missing={sorted(missing)}")
    names = [variant["name"] for variant in config["variants"]]
    if len(names) != len(set(names)) or config["baseline_variant"] not in names \
            or config["candidate_variant"] not in names:
        raise ValueError("confirmation variants must be unique and include baseline/candidate")
    if len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("confirmation seeds must be unique")
    for variant in config["variants"]:
        if variant["kind"] == "heuristic":
            continue
        if variant["kind"] != "blend" or set(variant.get("weights", {})) != set(HEADS):
            raise ValueError(f"invalid confirmation variant: {variant}")
        if any(not 0.0 <= float(weight) <= 1.0 for weight in variant["weights"].values()):
            raise ValueError(f"variant weights outside [0, 1]: {variant}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    root = Path.cwd()
    checkpoint = root / config["checkpoint_path"]
    output = root / config["output_path"]
    if output.exists():
        raise SystemExit("Refusing to overwrite existing v10 confirmation output")
    actual_checkpoint_hash = sha256(checkpoint)
    if actual_checkpoint_hash != config["checkpoint_sha256"]:
        raise ValueError("v10 confirmation checkpoint hash mismatch")
    student = LinearExpertNetwork.load(checkpoint)
    networks = {}
    for variant in config["variants"]:
        networks[variant["name"]] = (HeuristicNetwork() if variant["kind"] == "heuristic"
                                     else blended_network(student, variant["weights"]))
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for seed in config["seeds"]:
        results = {}
        for name, network in networks.items():
            _, terminal = play_episode(initial_state(int(seed), int(config["max_steps"])),
                                       network, config, int(seed))
            results[name] = {"score": terminal.agents[0].score,
                             "steps": terminal.step_count,
                             "survived": terminal.agents[0].alive}
        rows.append({"seed": int(seed), "results": results})
    names = list(networks)
    aggregates = summarize(rows, names, int(config["max_steps"]))
    report = {
        "kind": "v10-head-confirmation", "status": "completed", "run_id": config["run_id"],
        "started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": config, "config_sha256": sha256(config_path),
        "source_sha256": source_hashes(root, PROVENANCE_FILES + ("tools/v10_head_confirmation.py",)),
        "checkpoint_sha256": actual_checkpoint_hash, "rows": rows, "aggregates": aggregates,
        "gate": evaluate_gate(rows, aggregates, config),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "aggregates": aggregates, "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
