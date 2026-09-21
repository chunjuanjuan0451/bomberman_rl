"""Protocol tests for the v10.9 paired teacher search-budget gate."""

import json
from pathlib import Path

from tools.v10_teacher_budget_gate import (
    EXPECTED_ARMS,
    EXPECTED_SEEDS,
    evaluate_gate,
    load_preregistration,
    paired_comparison,
    summarize,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/configs/v10.9-task2-teacher-budget-gate-s106140.json"


def _episode(seed: int, crates: int, *, cleared: bool = False) -> dict:
    return {
        "seed": seed,
        "score": 9,
        "survived": True,
        "cleared_task2_board": cleared,
        "crates_remaining": crates,
        "collectable_coins_remaining": 0 if cleared else 1,
        "steps": 380 if cleared else 400,
        "wait_action_fraction": 0.2,
        "bomb_actions": 9,
        "crates_destroyed_per_bomb": 10.0,
        "search_calls": 95 if cleared else 100,
        "search_cadence_valid": True,
        "search_wait_fraction": 0.2,
        "search_planner_agreement": 0.8,
        "search_low_margin_fraction": 0.3,
        "mean_search_margin": 0.2,
        "productive_bomb_protections": 1,
        "productive_bomb_suppressions": 0,
        "repeated_position_fraction": 0.2,
        "lag4_repeat_fraction": 0.1,
        "longest_no_crate_progress_steps": 40,
        "target_switches": 10,
    }


def _variants(candidate: list[dict], control: list[dict]) -> dict:
    return {
        "control_128": {"episodes": control, "summary": summarize(control)},
        "candidate_256": {"episodes": candidate, "summary": summarize(candidate)},
    }


def test_budget_preregistration_is_source_and_instability_bound():
    protocol, base = load_preregistration(CONFIG, allow_existing_output=True)
    assert protocol["episode_seeds"] == EXPECTED_SEEDS
    assert protocol["arms"] == EXPECTED_ARMS
    assert protocol["required_repeatability_decision"] == "search_budget_instability"
    assert protocol["automatic_training"] is False and protocol["training"] is False
    assert base["teacher_persistent_target"] is True
    assert base["teacher_productive_bomb_protection"] is True


def test_paired_comparison_preserves_seed_alignment():
    control = [_episode(seed, 3) for seed in EXPECTED_SEEDS]
    candidate = [_episode(seed, 2) for seed in reversed(EXPECTED_SEEDS)]
    paired = paired_comparison(candidate, control)
    assert paired["wins"] == 12 and paired["losses"] == 0
    assert paired["mean_crates_remaining_delta"] == -1.0
    assert [row["seed"] for row in paired["per_seed"]] == list(reversed(EXPECTED_SEEDS))


def test_gate_distinguishes_material_gain_no_gain_and_regression():
    config = json.loads(CONFIG.read_text())
    control = [_episode(seed, 3) for seed in EXPECTED_SEEDS]
    improved = [_episode(seed, 0 if index == 0 else 2, cleared=index == 0)
                for index, seed in enumerate(EXPECTED_SEEDS)]
    gate = evaluate_gate(
        _variants(improved, control), config["teacher_gate"],
        config["decision_branches"])
    assert gate["decision"] == "select_256_teacher" and gate["passed"]

    gate = evaluate_gate(
        _variants([_episode(seed, 3) for seed in EXPECTED_SEEDS], control),
        config["teacher_gate"], config["decision_branches"])
    assert gate["decision"] == "keep_128_no_material_gain" and not gate["passed"]

    regressed = [_episode(seed, 7 if index == 0 else 3)
                 for index, seed in enumerate(EXPECTED_SEEDS)]
    gate = evaluate_gate(
        _variants(regressed, control), config["teacher_gate"],
        config["decision_branches"])
    assert gate["decision"] == "reject_256_regression" and not gate["passed"]
