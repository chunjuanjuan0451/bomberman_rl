"""Pre-registered fixed-seed, zero-training Task 2 teacher diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v10_classic_residual_train import (  # noqa: E402
    build_teacher, initial_state, load_config, observer_game_state,
    sample_decision, state_from_game_state, teacher_action,
)
from agent_code.model_a_v10.simulator import step  # noqa: E402


PREREGISTRATION_KIND = "v10-task2-teacher-three-arm-preregistration-v1"
REPORT_KIND = "v10-task2-teacher-three-arm-report-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part) for part in value.split(",") if part]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty, unique comma-separated list")
    return seeds


def _lag_repeat_fraction(positions: list[tuple[int, int]], lag: int) -> float:
    if len(positions) <= lag:
        return 0.0
    return float(np.mean([
        positions[index] == positions[index - lag]
        for index in range(lag, len(positions))
    ]))


def _longest_no_crate_progress(trace: list[dict], final_crates: int) -> int:
    crates = [row["crates_remaining"] for row in trace] + [final_crates]
    longest = current = 0
    for before, after in zip(crates, crates[1:]):
        if after < before:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def rollout(seed: int, config: dict, arm: dict) -> dict:
    """Run one teacher episode and retain every decision, but no training labels."""
    network, planner, transform, searcher = build_teacher(config)
    state = initial_state(seed, int(config.get("max_steps", 400)), 0)
    previous_target = None
    trace, search_calls = [], 0
    while not state.ended and state.agents[0].alive:
        observed = state_from_game_state(observer_game_state(state))
        base = network.evaluate(observed, 0).policy
        transform.persistent_target_enabled = bool(arm["persistent_target"])
        transform.productive_bomb_protection = bool(
            arm["productive_bomb_protection"])
        persistent_target = transform.prepare(observed, 0, round_id=1)
        target_changed = bool(
            previous_target is not None and persistent_target != previous_target)
        previous_target = persistent_target
        transform.set_crate_target(persistent_target)
        plan = planner.plan(observed, 0, base, crate_target=persistent_target)
        search_now, _ = sample_decision(
            plan.tactical and arm["tactical_stride"] > 0,
            state.step_count,
            int(config["state_sample_stride"]),
            max(int(arm["tactical_stride"]), 1),
        )
        greedy_action = max(plan.prior, key=plan.prior.get)
        search_action = None
        action = greedy_action
        protected = False
        if search_now:
            result = searcher.search(
                observed, 0,
                simulations=int(config["teacher_search_simulations"]),
                seed=seed * 1000 + state.step_count,
                deadline_ms=None,
            )
            search_action = result.action
            action, protected = transform.protect_action(search_action)
            search_calls += 1
        agent = state.agents[0]
        trace.append({
            "step": state.step_count,
            "position": [agent.x, agent.y],
            "action": action,
            "greedy_action": greedy_action,
            "search_action": search_action,
            "search_invoked": search_now,
            "productive_bomb_protected": protected,
            "tactical": plan.tactical,
            "bomb_utility": plan.bomb_utility,
            "crate_target": (None if plan.crate_target is None
                             else list(plan.crate_target)),
            "crate_target_distance": plan.crate_target_distance,
            "target_changed": target_changed,
            "crates_remaining": int(np.count_nonzero(state.field == 1)),
        })
        state = step(state, [action], (0,))

    positions = [tuple(row["position"]) for row in trace]
    repeats = sum(position in positions[:index]
                  for index, position in enumerate(positions))
    initial_crates = trace[0]["crates_remaining"] if trace else 0
    final_crates = int(np.count_nonzero(state.field == 1))
    search_rows = [row for row in trace if row["search_invoked"]]
    search_overrides = sum(
        row["search_action"] != row["greedy_action"] for row in search_rows)
    raw_productive_suppressions = sum(
        row["greedy_action"] == "BOMB"
        and row["bomb_utility"] > 0.0
        and row["search_action"] != "BOMB"
        for row in search_rows
    )
    productive_suppressions = sum(
        row["greedy_action"] == "BOMB"
        and row["bomb_utility"] > 0.0
        and row["action"] != "BOMB"
        for row in search_rows
    )
    bombs = sum(row["action"] == "BOMB" for row in trace)
    target_distances = [row["crate_target_distance"] for row in trace
                        if row["crate_target_distance"] is not None]
    return {
        "seed": seed,
        "steps": state.step_count,
        "score": state.agents[0].score,
        "survived": state.agents[0].alive,
        "initial_crates": initial_crates,
        "crates_remaining": final_crates,
        "crates_destroyed": initial_crates - final_crates,
        "search_calls": search_calls,
        "search_override_rate": search_overrides / max(len(search_rows), 1),
        "raw_productive_bomb_suppressions": raw_productive_suppressions,
        "productive_bomb_suppressions": productive_suppressions,
        "productive_bomb_protections": sum(
            row["productive_bomb_protected"] for row in trace),
        "bomb_actions": bombs,
        "crates_destroyed_per_bomb": ((initial_crates - final_crates) / max(bombs, 1)),
        "repeated_position_fraction": repeats / max(len(trace), 1),
        "lag2_repeat_fraction": _lag_repeat_fraction(positions, 2),
        "lag4_repeat_fraction": _lag_repeat_fraction(positions, 4),
        "lag8_repeat_fraction": _lag_repeat_fraction(positions, 8),
        "longest_no_crate_progress_steps": _longest_no_crate_progress(
            trace, final_crates),
        "target_switches": sum(row["target_changed"] for row in trace),
        "mean_target_distance": (None if not target_distances
                                 else float(np.mean(target_distances))),
        "trace": trace,
    }


def summarize(episodes: list[dict]) -> dict:
    def mean(key: str) -> float:
        return float(np.mean([item[key] for item in episodes]))

    curves = {}
    for step_index in (0, 100, 200, 300, 399):
        values = [item["trace"][step_index]["crates_remaining"]
                  for item in episodes if len(item["trace"]) > step_index]
        curves[f"step_{step_index}"] = None if not values else float(np.mean(values))
    target_distances = [item["mean_target_distance"] for item in episodes
                        if item["mean_target_distance"] is not None]
    return {
        "episodes": len(episodes),
        "mean_score": mean("score"),
        "survival_rate": mean("survived"),
        "cleared_rate": float(np.mean([
            item["crates_remaining"] == 0 for item in episodes])),
        "mean_crates_remaining": mean("crates_remaining"),
        "max_crates_remaining": max(item["crates_remaining"] for item in episodes),
        "mean_crates_destroyed": mean("crates_destroyed"),
        "mean_search_calls": mean("search_calls"),
        "mean_search_override_rate": mean("search_override_rate"),
        "total_raw_productive_bomb_suppressions": sum(
            item["raw_productive_bomb_suppressions"] for item in episodes),
        "total_productive_bomb_suppressions": sum(
            item["productive_bomb_suppressions"] for item in episodes),
        "total_productive_bomb_protections": sum(
            item["productive_bomb_protections"] for item in episodes),
        "mean_bomb_actions": mean("bomb_actions"),
        "mean_crates_destroyed_per_bomb": mean("crates_destroyed_per_bomb"),
        "mean_repeated_position_fraction": mean("repeated_position_fraction"),
        "mean_lag2_repeat_fraction": mean("lag2_repeat_fraction"),
        "mean_lag4_repeat_fraction": mean("lag4_repeat_fraction"),
        "mean_lag8_repeat_fraction": mean("lag8_repeat_fraction"),
        "mean_longest_no_crate_progress_steps": mean(
            "longest_no_crate_progress_steps"),
        "mean_target_switches": mean("target_switches"),
        "mean_target_distance": (None if not target_distances
                                 else float(np.mean(target_distances))),
        "mean_crates_remaining_curve": curves,
    }


def _paired_comparison(candidate: list[dict], reference: list[dict]) -> dict:
    reference_by_seed = {item["seed"]: item for item in reference}
    deltas = [item["crates_remaining"]
              - reference_by_seed[item["seed"]]["crates_remaining"]
              for item in candidate]
    return {
        "wins": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta > 0 for delta in deltas),
        "mean_crates_remaining_delta": float(np.mean(deltas)),
        "maximum_crates_remaining_regression": max(deltas),
    }


def evaluate_gate(variants: dict, gate: dict) -> dict:
    control_name = gate["control_arm"]
    control = variants[control_name]
    evaluations = {}
    for name in gate["candidate_arms"]:
        candidate = variants[name]
        summary = candidate["summary"]
        paired = _paired_comparison(candidate["episodes"], control["episodes"])
        checks = {
            "survival": summary["survival_rate"] >= gate["minimum_survival_rate"],
            "clear_rate": summary["cleared_rate"] >= gate["minimum_clear_rate"],
            "mean_crates": summary["mean_crates_remaining"]
            <= gate["maximum_mean_crates_remaining"],
            "worst_seed_crates": summary["max_crates_remaining"]
            <= gate["maximum_worst_seed_crates_remaining"],
            "paired_wins": paired["wins"] >= gate["minimum_paired_wins_vs_control"],
            "relative_mean_improvement": summary["mean_crates_remaining"]
            <= control["summary"]["mean_crates_remaining"]
            * (1.0 - gate["minimum_relative_mean_crate_improvement"]),
            "maximum_seed_regression": paired["maximum_crates_remaining_regression"]
            <= gate["maximum_single_seed_crate_regression"],
            "productive_bomb_suppression":
            summary["total_productive_bomb_suppressions"] == 0,
        }
        evaluations[name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "paired_vs_control": paired,
        }

    eligible = [name for name in gate["candidate_arms"]
                if evaluations[name]["passed"]]
    selected = None
    selection_reason = "no candidate passed the teacher gate"
    simpler, persistent = gate["candidate_arms"]
    if simpler in eligible:
        selected = simpler
        selection_reason = "productive-BOMB protection passed; prefer simpler arm"
        if persistent in eligible:
            paired = _paired_comparison(
                variants[persistent]["episodes"], variants[simpler]["episodes"])
            clear_gain = (variants[persistent]["summary"]["cleared_rate"]
                          > variants[simpler]["summary"]["cleared_rate"])
            material_mean_gain = (
                variants[persistent]["summary"]["mean_crates_remaining"]
                <= variants[simpler]["summary"]["mean_crates_remaining"]
                - gate["persistent_minimum_absolute_mean_crate_gain"]
                and paired["wins"] >= gate["persistent_minimum_paired_wins_vs_bomb_protection"]
            )
            if clear_gain or material_mean_gain:
                selected = persistent
                selection_reason = "persistent target added material incremental improvement"
    elif persistent in eligible:
        selected = persistent
        selection_reason = "only the cumulative persistent-target arm passed"
    return {
        "passed": selected is not None,
        "selected_arm": selected,
        "selection_reason": selection_reason,
        "candidate_evaluations": evaluations,
    }


def load_preregistration(path: Path) -> tuple[dict, dict]:
    protocol = json.loads(path.read_text())
    if protocol.get("kind") != PREREGISTRATION_KIND:
        raise ValueError("not a three-arm teacher preregistration")
    seeds = protocol.get("seeds", [])
    if len(seeds) != 12 or len(set(seeds)) != 12:
        raise ValueError("three-arm protocol requires exactly 12 unique seeds")
    expected_arms = [
        {"name": "stride4_control", "tactical_stride": 4,
         "productive_bomb_protection": False, "persistent_target": False},
        {"name": "productive_bomb_protection", "tactical_stride": 4,
         "productive_bomb_protection": True, "persistent_target": False},
        {"name": "persistent_target", "tactical_stride": 4,
         "productive_bomb_protection": True, "persistent_target": True},
    ]
    if protocol.get("arms") != expected_arms:
        raise ValueError("three-arm definitions do not match the frozen protocol")
    for relative, expected in protocol.get("source_sha256", {}).items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"source hash mismatch: {relative}")
    base_path = ROOT / protocol["base_training_config"]
    if sha256(base_path) != protocol["base_training_config_sha256"]:
        raise ValueError("base training config hash mismatch")
    base_config = load_config(base_path)
    output_path = ROOT / protocol["output_path"]
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic report: {output_path}")
    return protocol, base_config


def run_preregistered(path: Path, execute: bool) -> int:
    protocol, config = load_preregistration(path)
    plan = {
        "mode": "execute" if execute else "dry-run",
        "run_id": protocol["run_id"],
        "seeds": protocol["seeds"],
        "arms": protocol["arms"],
        "output_path": protocol["output_path"],
        "preregistration_sha256": sha256(path),
    }
    print(json.dumps(plan, indent=2))
    if not execute:
        return 0
    variants = {}
    for arm in protocol["arms"]:
        episodes = [rollout(seed, config, arm) for seed in protocol["seeds"]]
        variants[arm["name"]] = {
            "definition": arm,
            "summary": summarize(episodes),
            "episodes": episodes,
        }
    gate = evaluate_gate(variants, protocol["teacher_gate"])
    payload = {
        "kind": REPORT_KIND,
        "run_id": protocol["run_id"],
        "preregistration": str(path.relative_to(ROOT)),
        "preregistration_sha256": sha256(path),
        "base_training_config": protocol["base_training_config"],
        "base_training_config_sha256": protocol["base_training_config_sha256"],
        "seeds": protocol["seeds"],
        "variants": variants,
        "teacher_gate": gate,
    }
    output_path = ROOT / protocol["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        payload, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ) + "\n")
    print(json.dumps({
        "summaries": {name: value["summary"] for name, value in variants.items()},
        "teacher_gate": gate,
    }, indent=2))
    return 0


def run_legacy(args: argparse.Namespace, config: dict) -> int:
    if args.seeds is None or args.output is None:
        raise ValueError("legacy mode requires --seeds and --output")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic report: {args.output}")
    seeds = parse_seeds(args.seeds)
    variants = {}
    for tactical_stride in parse_seeds(args.tactical_strides):
        arm = {"name": f"tactical_stride_{tactical_stride}",
               "tactical_stride": tactical_stride,
               "productive_bomb_protection": False, "persistent_target": False}
        episodes = [rollout(seed, config, arm) for seed in seeds]
        variants[arm["name"]] = {
            "definition": arm, "summary": summarize(episodes), "episodes": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "kind": "v10-task2-teacher-diagnostic-v2", "config": str(args.config),
        "seeds": seeds, "variants": variants,
    }, indent=2) + "\n")
    print(json.dumps({key: value["summary"] for key, value in variants.items()}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--seeds", help="legacy fixed comma-separated seeds")
    parser.add_argument("--tactical-strides", default="2,4",
                        help="legacy comma-separated tactical search strides")
    parser.add_argument("--output", type=Path, help="legacy output path")
    parser.add_argument("--execute", action="store_true",
                        help="execute a preregistered protocol; default is dry-run")
    args = parser.parse_args(argv)
    payload = json.loads(args.config.read_text())
    if payload.get("kind") == PREREGISTRATION_KIND:
        return run_preregistered(args.config.resolve(), args.execute)
    return run_legacy(args, load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
