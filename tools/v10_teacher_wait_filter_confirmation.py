"""Pre-registered confirmation gate for the Task 2 WAIT-filter teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v10_teacher_repeatability import atomic_json  # noqa: E402
from tools.v10_teacher_wait_filter_gate import (  # noqa: E402
    paired_comparison,
    rollout,
    summarize,
)


PREREGISTRATION_KIND = (
    "v10-task2-teacher-wait-filter-confirmation-preregistration-v1"
)
REPORT_KIND = "v10-task2-teacher-wait-filter-confirmation-report-v1"
EXPECTED_RUN_ID = "v10.11-task2-teacher-wait-filter-confirmation-s106180"
EXPECTED_SEEDS = list(range(106180, 106204))
EXPECTED_ARMS = [
    {"name": "control_128", "low_margin_wait_filter": False},
    {"name": "candidate_wait_margin_015", "low_margin_wait_filter": True},
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_gate(variants: dict, gate: dict, branches: dict) -> dict:
    control = variants[gate["control_arm"]]
    candidate = variants[gate["candidate_arm"]]
    baseline = control["summary"]
    summary = candidate["summary"]
    paired = paired_comparison(candidate["episodes"], control["episodes"])
    effects = {
        "clear_count_difference": (
            summary["clear_count"] - baseline["clear_count"]
        ),
        "mean_crates_remaining_improvement": (
            baseline["mean_crates_remaining"]
            - summary["mean_crates_remaining"]
        ),
        "paired_net_wins": paired["wins"] - paired["losses"],
        "mean_score_difference": summary["mean_score"] - baseline["mean_score"],
        "mean_collectable_coins_remaining_difference": (
            summary["mean_collectable_coins_remaining"]
            - baseline["mean_collectable_coins_remaining"]
        ),
        "mean_search_wait_fraction_reduction": (
            baseline["mean_search_wait_fraction"]
            - summary["mean_search_wait_fraction"]
        ),
        "mean_search_planner_agreement_gain": (
            summary["mean_search_planner_agreement"]
            - baseline["mean_search_planner_agreement"]
        ),
    }
    safeguards = {
        "survival": summary["survival_rate"] >= gate["minimum_survival_rate"],
        "aggregate_clear_noninferiority": (
            effects["clear_count_difference"]
            >= gate["minimum_clear_count_difference"]
        ),
        "mean_score_noninferiority": (
            effects["mean_score_difference"]
            >= -gate["maximum_mean_score_regression"]
        ),
        "mean_collectable_coins_noninferiority": (
            effects["mean_collectable_coins_remaining_difference"]
            <= gate["maximum_mean_collectable_coin_regression"]
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
    confirmation = {
        "minimum_interventions": (
            summary["total_wait_filter_activations"]
            >= gate["minimum_wait_filter_activations"]
        ),
        "search_wait_reduction": (
            effects["mean_search_wait_fraction_reduction"]
            >= gate["minimum_search_wait_fraction_reduction"]
        ),
        "planner_agreement_gain": (
            effects["mean_search_planner_agreement_gain"]
            >= gate["minimum_search_planner_agreement_gain"]
        ),
        "mean_crate_improvement": (
            effects["mean_crates_remaining_improvement"]
            >= gate["minimum_mean_crate_improvement"]
        ),
        "paired_net_wins": (
            effects["paired_net_wins"] >= gate["minimum_paired_net_wins"]
        ),
    }
    safeguards_passed = all(safeguards.values())
    confirmation_passed = all(confirmation.values())
    if safeguards_passed and confirmation_passed:
        decision = "select_confirmed_wait_filter_teacher"
    elif safeguards_passed:
        decision = "keep_control_no_confirmed_benefit"
    else:
        decision = "reject_wait_filter_confirmation_regression"
    return {
        "passed": decision == "select_confirmed_wait_filter_teacher",
        "decision": decision,
        "selected_arm": (
            "candidate_wait_margin_015"
            if decision == "select_confirmed_wait_filter_teacher"
            else "control_128"
        ),
        "safeguards_passed": safeguards_passed,
        "confirmation_passed": confirmation_passed,
        "safeguard_checks": safeguards,
        "confirmation_checks": confirmation,
        "effect_sizes": effects,
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
            raise ValueError(
                f"teacher WAIT-filter confirmation requires {key}={expected!r}"
            )
    for relative, expected in protocol.get("source_sha256", {}).items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"source hash mismatch: {relative}")

    discovery_path = ROOT / protocol["discovery_report"]
    if sha256(discovery_path) != protocol["discovery_report_sha256"]:
        raise ValueError("v10.10 discovery report hash mismatch")
    discovery = json.loads(discovery_path.read_text())
    discovery_gate = discovery.get("teacher_gate", {})
    required_discovery = {
        "decision": "reject_wait_filter_regression",
        "selected_arm": "control_128",
        "target_repair_passed": True,
    }
    for key, expected in required_discovery.items():
        if discovery_gate.get(key) != expected:
            raise ValueError(f"v10.10 discovery report requires {key}={expected!r}")
    if discovery_gate.get("paired", {}).get("mean_crates_remaining_delta", 0.0) >= 0:
        raise ValueError("v10.10 did not provide positive replication evidence")

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
        raise FileExistsError(
            f"refusing to overwrite WAIT-filter confirmation report: {output}"
        )
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
        "discovery_report": protocol["discovery_report"],
        "discovery_report_sha256": protocol["discovery_report_sha256"],
        "base_teacher_config": protocol["base_teacher_config"],
        "base_teacher_config_sha256": protocol["base_teacher_config_sha256"],
        "episode_seeds": protocol["episode_seeds"],
        "variants": variants,
        "teacher_gate": gate,
        "automatic_training_started": False,
        "discovery_data_used_for_selection": False,
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
        help="run the preregistered confirmation; default is dry-run",
    )
    args = parser.parse_args(argv)
    return run(args.config.resolve(), args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
