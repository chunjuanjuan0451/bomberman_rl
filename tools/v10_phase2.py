"""Reproducible Phase-2 budget-gate benchmark on official-world snapshots."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np

from agent_code.model_a_v10.interfaces import HeuristicNetwork
from agent_code.model_a_v10.official_adapter import OfficialReference
from agent_code.model_a_v10.runtime import (NORMAL_SIMULATIONS, TACTICAL_SIMULATIONS,
                                             NORMAL_MIN_SIMULATIONS, TACTICAL_MIN_SIMULATIONS,
                                             Phase2Controller, is_tactical, state_features,
                                             search_budget_for_profile, state_from_game_state)
from agent_code.model_a_v10.simulator import SimAgent, SimState, step
from agent_code.model_a_v10.tactics import evaluate_tactics
from environment import BombeRLeWorld


def _official_initial_state(seed: int) -> SimState:
    """Use the original classic arena generator without starting agent workers."""
    carrier = SimpleNamespace(
        rng=np.random.default_rng(seed), args=SimpleNamespace(scenario="classic"),
        agents=[SimAgent(f"a{index}", 1, 1) for index in range(4)],
    )
    arena, coins, agents = BombeRLeWorld.build_arena(carrier)
    return SimState(arena, agents, {(coin.x, coin.y): coin.collectable for coin in coins})


def fixed_official_states(samples: int, seed: int) -> list[SimState]:
    """Collect fixed snapshots via official build_arena + GenericWorld methods."""
    rng = np.random.default_rng(seed)
    states: list[SimState] = []
    world_number = 0
    while len(states) < samples:
        reference = OfficialReference(_official_initial_state(seed + world_number))
        world_number += 1
        for _ in range(80):
            snapshot = reference.as_sim_state()
            if snapshot.agents[0].alive:
                states.append(snapshot)
                if len(states) == samples:
                    return states
            if not reference.running:
                break
            actions = [str(rng.choice(("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")))
                       for _ in reference.agents]
            reference.step_once(actions, rng.permutation(len(reference.active_agents)).tolist())
    return states


def _game_state(state: SimState) -> dict:
    root = state.agents[0]
    return {"round": 1, "step": state.step_count, "field": state.field.copy(),
            "self": (root.name, root.score, root.bombs_left, (root.x, root.y)),
            "others": [(agent.name, agent.score, agent.bombs_left, (agent.x, agent.y))
                       for agent in state.agents[1:] if agent.alive],
            "bombs": [((bomb.x, bomb.y), bomb.timer) for bomb in state.bombs],
            "coins": [position for position, visible in state.coins.items() if visible],
            "explosion_map": state.explosion_map(), "user_input": "WAIT"}


def _percentiles(values: list[int]) -> dict[str, float]:
    milliseconds = np.asarray(values, dtype=float) / 1_000_000.0
    return {"p50_ms": float(np.percentile(milliseconds, 50)), "p95_ms": float(np.percentile(milliseconds, 95)),
            "p99_ms": float(np.percentile(milliseconds, 99)), "max_ms": float(milliseconds.max())}


def benchmark(samples: int, seed: int, budget_profile: str = "research-m4-v1") -> dict[str, object]:
    states = fixed_official_states(samples, seed)
    budget = search_budget_for_profile(budget_profile)
    controller = Phase2Controller(budget=budget)
    network = HeuristicNetwork()
    rng = np.random.default_rng(seed + 1)
    feature_ns: list[int] = []
    network_ns: list[int] = []
    simulation_ns: list[int] = []
    tree_ns: list[int] = []
    total_ns: list[int] = []
    rows: list[dict[str, object]] = []
    for index, state in enumerate(states):
        game_state = _game_state(state)
        started = perf_counter_ns()
        converted = state_from_game_state(game_state)
        state_features(converted)
        feature_ns.append(perf_counter_ns() - started)
        started = perf_counter_ns()
        network.evaluate(converted, 0)
        network_ns.append(perf_counter_ns() - started)
        active = converted.active_ids()
        actions = [str(rng.choice(("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")))
                   for _ in converted.agents]
        started = perf_counter_ns()
        step(converted, actions, rng.permutation(len(active)).tolist())
        simulation_ns.append(perf_counter_ns() - started)
        tactical = is_tactical(converted)
        started = perf_counter_ns()
        search_result = controller.select(converted, seed=seed + index)
        tree_ns.append(perf_counter_ns() - started)
        started = perf_counter_ns()
        action = controller.act(game_state)
        total_ns.append(perf_counter_ns() - started)
        rows.append({"index": index, "tactical": tactical, "action": action,
                     "simulations": search_result.simulations, "max_depth": search_result.max_depth,
                     "reason": search_result.reason, "risk": search_result.risk,
                     "under_minimum": controller.last_under_minimum,
                     "feature_ns": feature_ns[-1], "network_ns": network_ns[-1],
                     "simulation_ns": simulation_ns[-1], "tree_ns": tree_ns[-1], "total_ns": total_ns[-1]})
    tactical_rows = [row for row in rows if row["tactical"]]
    normal_rows = [row for row in rows if not row["tactical"]]
    return {"state_source": "official BombeRLeWorld.build_arena plus unmodified GenericWorld transition methods",
            "state_seed": seed, "samples": samples,
            "feature": _percentiles(feature_ns), "network_interface": _percentiles(network_ns),
            "simulation_step": _percentiles(simulation_ns), "tree_search": _percentiles(tree_ns),
            "total_act": _percentiles(total_ns),
            "completed_simulations": {"min": min(row["simulations"] for row in rows),
                                        "max": max(row["simulations"] for row in rows),
                                        "p50": float(np.percentile([row["simulations"] for row in rows], 50)),
                                        "p95": float(np.percentile([row["simulations"] for row in rows], 95))},
            "deadline_stop_count": sum(row["reason"] == "deadline" for row in rows),
            "under_minimum_count": sum(bool(row["under_minimum"]) for row in rows),
            "budget_profile": budget_profile,
            "normal": {"states": len(normal_rows), "simulation_cap": budget.normal_simulations,
                       "minimum_simulations": budget.normal_min_simulations,
                       "deadline_ms": budget.normal_deadline_ms},
            "tactical": {"states": len(tactical_rows), "simulation_cap": budget.tactical_simulations,
                         "minimum_simulations": budget.tactical_min_simulations,
                         "deadline_ms": budget.tactical_deadline_ms},
            "raw_samples": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=102001)
    parser.add_argument("--budget-profile", choices=("research-m4-v1", "ryzen-safe-v1"),
                        default="research-m4-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"phase": "v10-phase2", "created_at": datetime.now(timezone.utc).isoformat(),
              "configuration": vars(args) | {"output": str(args.output)},
              "machine": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                          "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                          "thread_configuration": {name: os.environ.get(name) for name in
                                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
              "benchmark": benchmark(args.samples, args.seed, args.budget_profile),
              "tactical_ablation": evaluate_tactics(50)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = json.loads(json.dumps(result))
    summary["benchmark"]["raw_samples"] = f"{args.samples} values retained in {args.output}"
    print(json.dumps(summary, indent=2, sort_keys=True))
    total = result["benchmark"]["total_act"]
    if (not result["tactical_ablation"]["passed"] or result["benchmark"]["under_minimum_count"]
            or total["p99_ms"] >= 400 or total["max_ms"] >= 450):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
