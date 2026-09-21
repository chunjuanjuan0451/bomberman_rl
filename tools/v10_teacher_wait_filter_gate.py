"""Pre-registered Task 2 teacher gate for low-margin WAIT corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v10.interfaces import ACTIONS  # noqa: E402
from agent_code.model_a_v10.simulator import step  # noqa: E402
from tools.v10_classic_residual_train import (  # noqa: E402
    build_teacher,
    initial_state,
    observer_game_state,
    state_from_game_state,
    visit_policy,
)
from tools.v10_teacher_repeatability import atomic_json, observer_state_sha256  # noqa: E402


PREREGISTRATION_KIND = "v10-task2-teacher-wait-filter-preregistration-v1"
REPORT_KIND = "v10-task2-teacher-wait-filter-report-v1"
EXPECTED_RUN_ID = "v10.10-task2-teacher-wait-filter-s106160"
EXPECTED_SEEDS = list(range(106160, 106172))
EXPECTED_ARMS = [
    {"name": "control_128", "low_margin_wait_filter": False},
    {"name": "candidate_wait_margin_015", "low_margin_wait_filter": True},
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visit_margin(visits: dict[str, int]) -> float:
    policy = visit_policy(visits).astype(np.float64)
    ordered = np.sort(policy)
    return float(ordered[-1] - ordered[-2])


def apply_wait_filter(
    planner_action: str,
    protected_search_action: str,
    search_margin: float,
    *,
    enabled: bool,
    threshold: float,
) -> tuple[str, bool]:
    """Preserve a non-WAIT planner action over a low-margin WAIT correction."""
    applied = bool(
        enabled
        and protected_search_action == "WAIT"
        and planner_action != "WAIT"
        and search_margin < threshold
    )
    return (planner_action if applied else protected_search_action), applied


def lag_repeat_fraction(positions: list[tuple[int, int]], lag: int) -> float:
    if len(positions) <= lag:
        return 0.0
    return float(np.mean([
        positions[index] == positions[index - lag]
        for index in range(lag, len(positions))
    ]))


def longest_no_crate_progress(trace: list[dict], final_crates: int) -> int:
    crates = [row["crates_remaining"] for row in trace] + [final_crates]
    longest = current = 0
    for before, after in zip(crates, crates[1:]):
        if after < before:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def rollout(seed: int, base: dict, arm: dict, protocol: dict) -> dict:
    """Run one deterministic teacher episode; never collect training labels."""
    network, planner, transform, searcher = build_teacher(base)
    max_steps = int(protocol["controlled_variables"]["maximum_steps"])
    stride = int(protocol["state_sample_stride"])
    simulations = int(protocol["teacher_search_simulations"])
    threshold = float(protocol["wait_filter_margin_threshold"])
    state = initial_state(seed, max_steps, 0)
    previous_target = None
    trace: list[dict] = []
    action_counts = Counter()

    while not state.ended and state.agents[0].alive:
        observed = state_from_game_state(observer_game_state(state))
        base_policy = network.evaluate(observed, 0).policy
        target = transform.prepare(observed, 0, round_id=1)
        target_changed = previous_target is not None and target != previous_target
        previous_target = target
        plan = planner.plan(observed, 0, base_policy, crate_target=target)
        planner_action = max(plan.prior, key=plan.prior.get)
        search_invoked = state.step_count % stride == 0
        search_action = None
        protected_search_action = None
        search_margin = None
        productive_bomb_protected = False
        wait_filter_applied = False
        action = planner_action
        state_hash = None

        if search_invoked:
            state_hash = observer_state_sha256(observed)
            result = searcher.search(
                observed,
                0,
                simulations=simulations,
                seed=seed * 1000 + state.step_count,
                deadline_ms=None,
            )
            if result.simulations != simulations:
                raise RuntimeError("teacher did not complete its fixed search budget")
            search_action = result.action
            search_margin = visit_margin(result.root_visits)
            protected_search_action, productive_bomb_protected = (
                transform.protect_action(search_action)
            )
            action, wait_filter_applied = apply_wait_filter(
                planner_action,
                protected_search_action,
                search_margin,
                enabled=bool(arm["low_margin_wait_filter"]),
                threshold=threshold,
            )

        agent = state.agents[0]
        action_counts[action] += 1
        trace.append({
            "step": state.step_count,
            "observer_state_sha256": state_hash,
            "position": [agent.x, agent.y],
            "action": action,
            "planner_action": planner_action,
            "search_action": search_action,
            "protected_search_action": protected_search_action,
            "search_margin": search_margin,
            "search_invoked": search_invoked,
            "productive_bomb_protected": productive_bomb_protected,
            "wait_filter_applied": wait_filter_applied,
            "bomb_utility": plan.bomb_utility,
            "crate_target": None if target is None else list(target),
            "target_changed": bool(target_changed),
            "crates_remaining": int(np.count_nonzero(state.field == 1)),
            "collectable_coins_remaining": int(sum(state.coins.values())),
        })
        state = step(state, [action], (0,))

    initial_crates = trace[0]["crates_remaining"] if trace else 0
    final_crates = int(np.count_nonzero(state.field == 1))
    final_coins = int(sum(state.coins.values()))
    search_rows = [row for row in trace if row["search_invoked"]]
    positions = [tuple(row["position"]) for row in trace]
    bombs = action_counts["BOMB"]
    low_margin_wait_overrides = [
        row for row in search_rows
        if row["action"] == "WAIT"
        and row["planner_action"] != "WAIT"
        and row["search_margin"] < threshold
    ]
    return {
        "seed": seed,
        "steps": state.step_count,
        "score": state.agents[0].score,
        "survived": state.agents[0].alive,
        "cleared_task2_board": bool(final_crates == 0 and final_coins == 0),
        "initial_crates": initial_crates,
        "crates_remaining": final_crates,
        "collectable_coins_remaining": final_coins,
        "crates_destroyed": initial_crates - final_crates,
        "action_counts": {action: action_counts[action] for action in ACTIONS},
        "wait_action_fraction": action_counts["WAIT"] / max(len(trace), 1),
        "bomb_actions": bombs,
        "crates_destroyed_per_bomb": (
            (initial_crates - final_crates) / max(bombs, 1)
        ),
        "search_calls": len(search_rows),
        "search_cadence_valid": (
            len(search_rows) == (len(trace) + stride - 1) // stride
        ),
        "raw_search_wait_fraction": float(np.mean([
            row["search_action"] == "WAIT" for row in search_rows
        ])) if search_rows else 0.0,
        "search_wait_fraction": float(np.mean([
            row["action"] == "WAIT" for row in search_rows
        ])) if search_rows else 0.0,
        "search_planner_agreement": float(np.mean([
            row["action"] == row["planner_action"] for row in search_rows
        ])) if search_rows else 0.0,
        "search_override_fraction": float(np.mean([
            row["action"] != row["planner_action"] for row in search_rows
        ])) if search_rows else 0.0,
        "search_low_margin_fraction": float(np.mean([
            row["search_margin"] < threshold for row in search_rows
        ])) if search_rows else 0.0,
        "mean_search_margin": float(np.mean([
            row["search_margin"] for row in search_rows
        ])) if search_rows else 0.0,
        "low_margin_wait_override_fraction": (
            len(low_margin_wait_overrides) / max(len(search_rows), 1)
        ),
        "wait_filter_activations": sum(
            row["wait_filter_applied"] for row in search_rows
        ),
        "productive_bomb_protections": sum(
            row["productive_bomb_protected"] for row in search_rows
        ),
        "productive_bomb_suppressions": sum(
            row["planner_action"] == "BOMB"
            and row["bomb_utility"] > 0.0
            and row["action"] != "BOMB"
            for row in search_rows
        ),
        "repeated_position_fraction": sum(
            position in positions[:index]
            for index, position in enumerate(positions)
        ) / max(len(positions), 1),
        "lag4_repeat_fraction": lag_repeat_fraction(positions, 4),
        "longest_no_crate_progress_steps": longest_no_crate_progress(
            trace, final_crates
        ),
        "target_switches": sum(row["target_changed"] for row in trace),
        "trace": trace,
    }


def summarize(episodes: list[dict]) -> dict:
    def mean(key: str) -> float:
        return float(np.mean([episode[key] for episode in episodes]))

    return {
        "episodes": len(episodes),
        "mean_score": mean("score"),
        "survival_rate": mean("survived"),
        "clear_count": sum(
            episode["cleared_task2_board"] for episode in episodes
        ),
        "clear_rate": mean("cleared_task2_board"),
        "mean_crates_remaining": mean("crates_remaining"),
        "maximum_crates_remaining": max(
            episode["crates_remaining"] for episode in episodes
        ),
        "mean_collectable_coins_remaining": mean(
            "collectable_coins_remaining"
        ),
        "mean_steps": mean("steps"),
        "mean_wait_action_fraction": mean("wait_action_fraction"),
        "mean_bomb_actions": mean("bomb_actions"),
        "mean_crates_destroyed_per_bomb": mean("crates_destroyed_per_bomb"),
        "mean_search_calls": mean("search_calls"),
        "all_search_cadence_valid": all(
            episode["search_cadence_valid"] for episode in episodes
        ),
        "mean_raw_search_wait_fraction": mean("raw_search_wait_fraction"),
        "mean_search_wait_fraction": mean("search_wait_fraction"),
        "mean_search_planner_agreement": mean("search_planner_agreement"),
        "mean_search_override_fraction": mean("search_override_fraction"),
        "mean_search_low_margin_fraction": mean("search_low_margin_fraction"),
        "mean_search_margin": mean("mean_search_margin"),
        "mean_low_margin_wait_override_fraction": mean(
            "low_margin_wait_override_fraction"
        ),
        "total_wait_filter_activations": sum(
            episode["wait_filter_activations"] for episode in episodes
        ),
        "total_productive_bomb_protections": sum(
            episode["productive_bomb_protections"] for episode in episodes
        ),
        "total_productive_bomb_suppressions": sum(
            episode["productive_bomb_suppressions"] for episode in episodes
        ),
        "mean_repeated_position_fraction": mean("repeated_position_fraction"),
        "mean_lag4_repeat_fraction": mean("lag4_repeat_fraction"),
        "mean_longest_no_crate_progress_steps": mean(
            "longest_no_crate_progress_steps"
        ),
        "mean_target_switches": mean("target_switches"),
    }


def paired_comparison(candidate: list[dict], control: list[dict]) -> dict:
    control_by_seed = {episode["seed"]: episode for episode in control}
    pairs = []
    for episode in candidate:
        reference = control_by_seed[episode["seed"]]
        pairs.append({
            "seed": episode["seed"],
            "candidate_crates_delta": (
                episode["crates_remaining"] - reference["crates_remaining"]
            ),
            "candidate_score_delta": episode["score"] - reference["score"],
            "candidate_steps_delta": episode["steps"] - reference["steps"],
            "control_cleared": reference["cleared_task2_board"],
            "candidate_cleared": episode["cleared_task2_board"],
        })
    crate_deltas = [pair["candidate_crates_delta"] for pair in pairs]
    score_deltas = [pair["candidate_score_delta"] for pair in pairs]
    return {
        "wins": sum(delta < 0 for delta in crate_deltas),
        "ties": sum(delta == 0 for delta in crate_deltas),
        "losses": sum(delta > 0 for delta in crate_deltas),
        "mean_crates_remaining_delta": float(np.mean(crate_deltas)),
        "maximum_single_seed_crate_regression": max(crate_deltas),
        "mean_score_delta": float(np.mean(score_deltas)),
        "clear_upgrades": sum(
            not pair["control_cleared"] and pair["candidate_cleared"]
            for pair in pairs
        ),
        "clear_downgrades": sum(
            pair["control_cleared"] and not pair["candidate_cleared"]
            for pair in pairs
        ),
        "per_seed": pairs,
    }


def evaluate_gate(variants: dict, gate: dict, branches: dict) -> dict:
    control = variants[gate["control_arm"]]
    candidate = variants[gate["candidate_arm"]]
    baseline = control["summary"]
    summary = candidate["summary"]
    paired = paired_comparison(candidate["episodes"], control["episodes"])
    wait_reduction = (
        baseline["mean_search_wait_fraction"]
        - summary["mean_search_wait_fraction"]
    )
    planner_gain = (
        summary["mean_search_planner_agreement"]
        - baseline["mean_search_planner_agreement"]
    )
    safeguards = {
        "survival": summary["survival_rate"] >= gate["minimum_survival_rate"],
        "clear_noninferiority": summary["clear_count"] >= baseline["clear_count"],
        "no_clear_downgrade": paired["clear_downgrades"] == 0,
        "mean_crates_noninferiority": (
            summary["mean_crates_remaining"]
            <= baseline["mean_crates_remaining"]
            + gate["maximum_mean_crate_regression"]
        ),
        "mean_score_noninferiority": (
            summary["mean_score"]
            >= baseline["mean_score"] - gate["maximum_mean_score_regression"]
        ),
        "single_seed_crate_regression": (
            paired["maximum_single_seed_crate_regression"]
            <= gate["maximum_single_seed_crate_regression"]
        ),
        "worst_seed_crates": (
            summary["maximum_crates_remaining"]
            <= gate["maximum_candidate_worst_seed_crates"]
        ),
        "productive_bomb_suppression": (
            summary["total_productive_bomb_suppressions"] == 0
        ),
        "search_cadence": (
            summary["all_search_cadence_valid"]
            and baseline["all_search_cadence_valid"]
        ),
        "low_margin_wait_override_eliminated": (
            summary["mean_low_margin_wait_override_fraction"] == 0.0
        ),
    }
    target_repair = {
        "minimum_interventions": (
            summary["total_wait_filter_activations"]
            >= gate["minimum_wait_filter_activations"]
        ),
        "search_wait_reduction": (
            wait_reduction >= gate["minimum_search_wait_fraction_reduction"]
        ),
        "planner_agreement_gain": (
            planner_gain >= gate["minimum_search_planner_agreement_gain"]
        ),
    }
    safeguards_passed = all(safeguards.values())
    target_repair_passed = all(target_repair.values())
    if safeguards_passed and target_repair_passed:
        decision = "select_wait_filter_teacher"
    elif safeguards_passed:
        decision = "keep_control_no_target_repair"
    else:
        decision = "reject_wait_filter_regression"
    return {
        "passed": decision == "select_wait_filter_teacher",
        "decision": decision,
        "selected_arm": (
            "candidate_wait_margin_015"
            if decision == "select_wait_filter_teacher"
            else "control_128"
        ),
        "safeguards_passed": safeguards_passed,
        "target_repair_passed": target_repair_passed,
        "safeguard_checks": safeguards,
        "target_repair_checks": target_repair,
        "effect_sizes": {
            "mean_search_wait_fraction_reduction": wait_reduction,
            "mean_search_planner_agreement_gain": planner_gain,
        },
        "paired": paired,
        "next_step": branches[decision],
    }


def load_preregistration(
    path: Path, *, allow_existing_output: bool = False
) -> tuple[dict, dict]:
    protocol = json.loads(path.read_text())
    exact = {
        "kind": PREREGISTRATION_KIND,
        "run_id": EXPECTED_RUN_ID,
        "episode_seeds": EXPECTED_SEEDS,
        "arms": EXPECTED_ARMS,
        "opponent_count": 0,
        "teacher_search_simulations": 128,
        "state_sample_stride": 4,
        "wait_filter_margin_threshold": 0.15,
        "automatic_training": False,
        "training": False,
        "h2h": False,
    }
    for key, expected in exact.items():
        if protocol.get(key) != expected:
            raise ValueError(f"teacher WAIT-filter protocol requires {key}={expected!r}")
    for relative, expected in protocol.get("source_sha256", {}).items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"source hash mismatch: {relative}")

    predecessor_path = ROOT / protocol["predecessor_report"]
    if sha256(predecessor_path) != protocol["predecessor_report_sha256"]:
        raise ValueError("v10.9 predecessor report hash mismatch")
    predecessor = json.loads(predecessor_path.read_text())
    gate = predecessor.get("teacher_gate", {})
    if gate.get("decision") != "reject_256_regression":
        raise ValueError("WAIT-filter gate requires v10.9 budget rejection")
    if gate.get("selected_arm") != "control_128":
        raise ValueError("WAIT-filter gate requires the 128-simulation teacher")

    repeatability_path = ROOT / protocol["repeatability_report"]
    if sha256(repeatability_path) != protocol["repeatability_report_sha256"]:
        raise ValueError("repeatability report hash mismatch")
    repeatability = json.loads(repeatability_path.read_text())
    if repeatability.get("decision_gate", {}).get("decision") != (
        "search_budget_instability"
    ):
        raise ValueError("WAIT-filter gate requires registered instability evidence")

    base_path = ROOT / protocol["base_teacher_config"]
    if sha256(base_path) != protocol["base_teacher_config_sha256"]:
        raise ValueError("base teacher config hash mismatch")
    base = json.loads(base_path.read_text())
    required_base = {
        "opponent_count": 0,
        "teacher_search_simulations": 128,
        "teacher_persistent_target": True,
        "teacher_productive_bomb_protection": True,
        "teacher_crate_progress_weight": 0.08,
        "teacher_crate_frontier_weight": 0.32,
        "search_horizon_plies": 8,
    }
    for key, expected in required_base.items():
        if base.get(key) != expected:
            raise ValueError(f"base teacher config requires {key}={expected!r}")

    output = ROOT / protocol["output_path"]
    if output.exists() and not allow_existing_output:
        raise FileExistsError(f"refusing to overwrite WAIT-filter report: {output}")
    return protocol, base


def run(path: Path, execute: bool) -> int:
    protocol, base = load_preregistration(path)
    print(json.dumps({
        "mode": "execute" if execute else "dry-run",
        "run_id": protocol["run_id"],
        "episode_seeds": protocol["episode_seeds"],
        "arms": protocol["arms"],
        "teacher_search_simulations": protocol["teacher_search_simulations"],
        "state_sample_stride": protocol["state_sample_stride"],
        "wait_filter_margin_threshold": protocol["wait_filter_margin_threshold"],
        "automatic_training": protocol["automatic_training"],
        "output_path": protocol["output_path"],
        "preregistration_sha256": sha256(path),
    }, indent=2))
    if not execute:
        return 0

    variants = {}
    for arm in protocol["arms"]:
        episodes = [
            rollout(seed, base, arm, protocol)
            for seed in protocol["episode_seeds"]
        ]
        variants[arm["name"]] = {
            "definition": arm,
            "summary": summarize(episodes),
            "episodes": episodes,
        }
    gate = evaluate_gate(
        variants,
        protocol["teacher_gate"],
        protocol["decision_branches"],
    )
    payload = {
        "kind": REPORT_KIND,
        "run_id": protocol["run_id"],
        "preregistration": str(path.relative_to(ROOT)),
        "preregistration_sha256": sha256(path),
        "predecessor_report": protocol["predecessor_report"],
        "predecessor_report_sha256": protocol["predecessor_report_sha256"],
        "repeatability_report": protocol["repeatability_report"],
        "repeatability_report_sha256": protocol["repeatability_report_sha256"],
        "base_teacher_config": protocol["base_teacher_config"],
        "base_teacher_config_sha256": protocol["base_teacher_config_sha256"],
        "episode_seeds": protocol["episode_seeds"],
        "variants": variants,
        "teacher_gate": gate,
        "automatic_training_started": False,
    }
    atomic_json(ROOT / protocol["output_path"], payload)
    print(json.dumps({
        "summaries": {
            name: value["summary"] for name, value in variants.items()
        },
        "teacher_gate": gate,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the preregistered teacher gate; default is dry-run",
    )
    args = parser.parse_args(argv)
    return run(args.config.resolve(), args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
