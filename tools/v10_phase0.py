"""Reproducible v10 Phase-0 differential test and throughput benchmark."""

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

import numpy as np

from agent_code.model_a_v10.official_adapter import equivalent, official_step
from agent_code.model_a_v10.simulator import SimAgent, SimBomb, SimExplosion, SimState, step


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB", "INVALID")


def random_state(rng: np.random.Generator) -> SimState:
    size = 17
    field = np.zeros((size, size), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    for x in range(size):
        for y in range(size):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = -1
    free = list(map(tuple, np.argwhere(field == 0)))
    rng.shuffle(free)
    count = int(rng.integers(1, 5))
    positions = free[:count]
    for x, y in free[count: count + int(rng.integers(0, 35))]:
        field[x, y] = 1
    agents = [SimAgent(f"a{i}", int(x), int(y), int(rng.integers(0, 6)), bool(rng.integers(0, 2)))
              for i, (x, y) in enumerate(positions)]
    candidates = [p for p in free[count:] if field[p] == 0]
    rng.shuffle(candidates)
    bombs = [SimBomb(int(x), int(y), int(rng.integers(0, count)), int(rng.integers(0, 5)))
             for x, y in candidates[:int(rng.integers(0, 4))]]
    coins: dict[tuple[int, int], bool] = {}
    for x, y in free[count + 4: count + 4 + int(rng.integers(0, 10))]:
        coins[(int(x), int(y))] = bool(field[x, y] == 0 and rng.integers(0, 2))
    explosions = []
    if rng.integers(0, 5) == 0:
        owner = int(rng.integers(0, count))
        x, y = positions[int(rng.integers(0, count))]
        explosions.append(SimExplosion(((int(x), int(y)),), owner, int(rng.integers(1, 3)), 0))
    return SimState(field, agents, coins, bombs, explosions)


def differential(samples: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mismatches: list[dict[str, object]] = []
    for sample in range(samples):
        source = random_state(rng)
        actions = [str(rng.choice(ACTIONS)) for _ in source.agents]
        permutation = rng.permutation(len(source.agents)).tolist()
        simulated = step(source, actions, permutation)
        official = official_step(source, actions, permutation)
        equal, field = equivalent(simulated, official)
        if not equal:
            mismatches.append({"sample": sample, "field": field, "actions": actions,
                               "permutation": permutation})
    return {"samples": samples, "seed": seed, "mismatch_count": len(mismatches),
            "mismatches": mismatches}


def benchmark(samples: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    for x in range(17):
        for y in range(17):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = -1
    source = SimState(field, [SimAgent("a0", 1, 1), SimAgent("a1", 1, 15),
                              SimAgent("a2", 15, 1), SimAgent("a3", 15, 15)])
    timings = np.empty(samples, dtype=np.int64)
    for index in range(samples):
        actions = [str(rng.choice(ACTIONS[:-1])) for _ in source.agents]
        permutation = rng.permutation(4).tolist()
        started = perf_counter_ns()
        step(source, actions, permutation)
        timings[index] = perf_counter_ns() - started
    milliseconds = timings / 1_000_000.0
    p50, p95, p99 = (float(np.percentile(milliseconds, q)) for q in (50, 95, 99))
    steps_per_second = float(1_000_000_000.0 / timings.mean())
    conservative_rollouts = int((0.350 * steps_per_second) // 8)
    return {"samples": samples, "seed": seed, "p50_ms": p50, "p95_ms": p95,
            "p99_ms": p99, "max_ms": float(milliseconds.max()),
            "steps_per_second": steps_per_second, "eight_ply_rollouts_in_350ms": conservative_rollouts,
            "rollout_estimate_basis": "mean step throughput / 8 plies",
            "benchmark_state": "fixed 17x17 classic wall layout; four live agents; copy-on-step; random joint actions and permutations",
            "raw_step_timings_ns": timings.tolist()}


def machine_info() -> dict[str, object]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "git_commit": commit,
            "thread_configuration": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                                     "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                                     "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--benchmark-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=101001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"phase": "v10-phase0", "created_at": datetime.now(timezone.utc).isoformat(),
              "configuration": vars(args) | {"output": str(args.output) if args.output else None},
              "machine": machine_info(), "differential": differential(args.samples, args.seed),
              "benchmark": benchmark(args.benchmark_samples, args.seed + 1)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    display_result = json.loads(json.dumps(result))
    display_result["benchmark"]["raw_step_timings_ns"] = (
        f"{len(result['benchmark']['raw_step_timings_ns'])} values retained in {args.output}"
        if args.output else f"{len(result['benchmark']['raw_step_timings_ns'])} values (use --output to retain)"
    )
    print(json.dumps(display_result, indent=2, sort_keys=True))
    if result["differential"]["mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
