"""Protocol tests for the v10.11 WAIT-filter confirmation gate."""

import json
from pathlib import Path

from tools.v10_teacher_wait_filter_confirmation import (
    EXPECTED_ARMS,
    EXPECTED_SEEDS,
    evaluate_gate,
    load_preregistration,
)
from tools.v10_teacher_wait_filter_gate import summarize


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/configs/"
    / "v10.11-task2-teacher-wait-filter-confirmation-s106180.json"
)


def _episode(
    seed: int,
    crates: int,
    *,
    cleared: bool = False,
    survived: bool = True,
    search_wait: float = 0.10,
    planner_agreement: float = 0.80,
    activations: int = 0,
    low_margin_wait_overrides: float = 0.02,
) -> dict:
    return {
        "seed": seed,
        "score": 9,
        "survived": survived,
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


def _candidate(seed: int, crates: int, *, cleared: bool = False) -> dict:
    return _episode(
        seed,
        crates,
        cleared=cleared,
        search_wait=0.08,
        planner_agreement=0.82,
        activations=1,
        low_margin_wait_overrides=0.0,
    )


def _variants(candidate: list[dict], control: list[dict]) -> dict:
    return {
        "control_128": {"episodes": control, "summary": summarize(control)},
        "candidate_wait_margin_015": {
            "episodes": candidate,
            "summary": summarize(candidate),
        },
    }


def test_confirmation_preregistration_is_discovery_bound_and_fresh():
    protocol, base = load_preregistration(CONFIG, allow_existing_output=True)
    assert protocol["episode_seeds"] == EXPECTED_SEEDS
    assert protocol["arms"] == EXPECTED_ARMS
    assert len(EXPECTED_SEEDS) == 24
    assert not set(EXPECTED_SEEDS) & set(range(106160, 106172))
    assert protocol["teacher_search_simulations"] == 128
    assert protocol["wait_filter_margin_threshold"] == 0.15
    assert protocol["registered_discovery_interpretation"][
        "discovery_data_used_for_selection"
    ] is False
    assert protocol["automatic_training"] is False
    assert protocol["training"] is False
    assert base["teacher_persistent_target"] is True
    assert base["teacher_productive_bomb_protection"] is True


def test_confirmation_allows_clear_identity_swap_when_aggregate_is_safe():
    config = json.loads(CONFIG.read_text())
    control = [
        _episode(seed, 0 if index == 0 else 3, cleared=index == 0)
        for index, seed in enumerate(EXPECTED_SEEDS)
    ]
    candidate = [
        _candidate(
            seed,
            1 if index == 0 else (0 if index == 1 else 2),
            cleared=index == 1,
        )
        for index, seed in enumerate(EXPECTED_SEEDS)
    ]
    gate = evaluate_gate(
        _variants(candidate, control),
        config["teacher_gate"],
        config["decision_branches"],
    )
    assert gate["paired"]["clear_upgrades"] == 1
    assert gate["paired"]["clear_downgrades"] == 1
    assert gate["safeguard_checks"]["aggregate_clear_noninferiority"]
    assert gate["decision"] == "select_confirmed_wait_filter_teacher"
    assert gate["passed"]


def test_confirmation_requires_crate_and_paired_material_benefit():
    config = json.loads(CONFIG.read_text())
    control = [_episode(seed, 3) for seed in EXPECTED_SEEDS]
    candidate = [_candidate(seed, 3) for seed in EXPECTED_SEEDS]
    gate = evaluate_gate(
        _variants(candidate, control),
        config["teacher_gate"],
        config["decision_branches"],
    )
    assert gate["safeguards_passed"]
    assert not gate["confirmation_checks"]["mean_crate_improvement"]
    assert not gate["confirmation_checks"]["paired_net_wins"]
    assert gate["decision"] == "keep_control_no_confirmed_benefit"
    assert not gate["passed"]


def test_confirmation_rejects_aggregate_clear_regression():
    config = json.loads(CONFIG.read_text())
    control = [
        _episode(seed, 0 if index == 0 else 3, cleared=index == 0)
        for index, seed in enumerate(EXPECTED_SEEDS)
    ]
    candidate = [_candidate(seed, 2) for seed in EXPECTED_SEEDS]
    gate = evaluate_gate(
        _variants(candidate, control),
        config["teacher_gate"],
        config["decision_branches"],
    )
    assert not gate["safeguard_checks"]["aggregate_clear_noninferiority"]
    assert gate["decision"] == "reject_wait_filter_confirmation_regression"
    assert not gate["passed"]
