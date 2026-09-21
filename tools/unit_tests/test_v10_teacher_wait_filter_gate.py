"""Protocol tests for the v10.10 low-margin WAIT teacher gate."""

import json
from pathlib import Path

from tools.v10_teacher_wait_filter_gate import (
    EXPECTED_ARMS,
    EXPECTED_SEEDS,
    apply_wait_filter,
    evaluate_gate,
    load_preregistration,
    summarize,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/configs/v10.10-task2-teacher-wait-filter-s106160.json"
)


def _episode(
    seed: int,
    crates: int,
    *,
    cleared: bool = False,
    search_wait: float = 0.10,
    planner_agreement: float = 0.80,
    activations: int = 0,
    low_margin_wait_overrides: float = 0.02,
) -> dict:
    return {
        "seed": seed,
        "score": 9,
        "survived": True,
        "cleared_task2_board": cleared,
        "crates_remaining": crates,
        "collectable_coins_remaining": 0 if cleared else 1,
        "steps": 380 if cleared else 400,
        "wait_action_fraction": 0.08,
        "bomb_actions": 9,
        "crates_destroyed_per_bomb": 10.0,
        "search_calls": 100,
        "search_cadence_valid": True,
        "raw_search_wait_fraction": 0.10,
        "search_wait_fraction": search_wait,
        "search_planner_agreement": planner_agreement,
        "search_override_fraction": 1.0 - planner_agreement,
        "search_low_margin_fraction": 0.30,
        "mean_search_margin": 0.20,
        "low_margin_wait_override_fraction": low_margin_wait_overrides,
        "wait_filter_activations": activations,
        "productive_bomb_protections": 1,
        "productive_bomb_suppressions": 0,
        "repeated_position_fraction": 0.20,
        "lag4_repeat_fraction": 0.10,
        "longest_no_crate_progress_steps": 40,
        "target_switches": 10,
    }


def _variants(candidate: list[dict], control: list[dict]) -> dict:
    return {
        "control_128": {"episodes": control, "summary": summarize(control)},
        "candidate_wait_margin_015": {
            "episodes": candidate,
            "summary": summarize(candidate),
        },
    }


def test_wait_filter_is_narrow_and_uses_strict_registered_threshold():
    assert apply_wait_filter(
        "RIGHT", "WAIT", 0.149, enabled=True, threshold=0.15
    ) == ("RIGHT", True)
    assert apply_wait_filter(
        "RIGHT", "WAIT", 0.15, enabled=True, threshold=0.15
    ) == ("WAIT", False)
    assert apply_wait_filter(
        "WAIT", "WAIT", 0.01, enabled=True, threshold=0.15
    ) == ("WAIT", False)
    assert apply_wait_filter(
        "RIGHT", "LEFT", 0.01, enabled=True, threshold=0.15
    ) == ("LEFT", False)
    assert apply_wait_filter(
        "RIGHT", "WAIT", 0.01, enabled=False, threshold=0.15
    ) == ("WAIT", False)


def test_preregistration_is_predecessor_bound_and_training_disabled():
    protocol, base = load_preregistration(CONFIG, allow_existing_output=True)
    assert protocol["episode_seeds"] == EXPECTED_SEEDS
    assert protocol["arms"] == EXPECTED_ARMS
    assert protocol["teacher_search_simulations"] == 128
    assert protocol["wait_filter_margin_threshold"] == 0.15
    assert protocol["automatic_training"] is False
    assert protocol["training"] is False
    assert base["teacher_persistent_target"] is True
    assert base["teacher_productive_bomb_protection"] is True


def test_gate_selects_only_a_material_safe_target_repair():
    config = json.loads(CONFIG.read_text())
    control = [_episode(seed, 3) for seed in EXPECTED_SEEDS]
    candidate = [
        _episode(
            seed,
            3,
            search_wait=0.08,
            planner_agreement=0.82,
            activations=1,
            low_margin_wait_overrides=0.0,
        )
        for seed in EXPECTED_SEEDS
    ]
    gate = evaluate_gate(
        _variants(candidate, control),
        config["teacher_gate"],
        config["decision_branches"],
    )
    assert gate["decision"] == "select_wait_filter_teacher"
    assert gate["passed"]
    assert gate["effect_sizes"]["mean_search_wait_fraction_reduction"] >= 0.01
    assert gate["effect_sizes"]["mean_search_planner_agreement_gain"] >= 0.01


def test_gate_distinguishes_no_repair_from_regression():
    config = json.loads(CONFIG.read_text())
    control = [_episode(seed, 3) for seed in EXPECTED_SEEDS]
    no_repair = [
        _episode(
            seed,
            3,
            search_wait=0.095,
            planner_agreement=0.805,
            activations=1,
            low_margin_wait_overrides=0.0,
        )
        for seed in EXPECTED_SEEDS
    ]
    gate = evaluate_gate(
        _variants(no_repair, control),
        config["teacher_gate"],
        config["decision_branches"],
    )
    assert gate["decision"] == "keep_control_no_target_repair"
    assert not gate["passed"]

    regressed = list(no_repair)
    regressed[0] = _episode(
        EXPECTED_SEEDS[0],
        7,
        search_wait=0.08,
        planner_agreement=0.82,
        activations=1,
        low_margin_wait_overrides=0.0,
    )
    gate = evaluate_gate(
        _variants(regressed, control),
        config["teacher_gate"],
        config["decision_branches"],
    )
    assert gate["decision"] == "reject_wait_filter_regression"
    assert not gate["passed"]
