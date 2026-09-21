"""Protocol tests for the v10.8 teacher-target repeatability diagnostic."""

import json
import tempfile
from pathlib import Path

import numpy as np

from tools.v10_teacher_repeatability import (
    EXPECTED_SEEDS,
    EXPECTED_STRATA,
    atomic_json,
    classify_stratum,
    initial_state,
    jensen_shannon,
    load_preregistration,
    observer_state_sha256,
    summarize,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/configs/v10.8.1-task2-teacher-repeatability-s106120.json"


def _row(stratum: str) -> dict:
    return {
        "stratum": stratum,
        "mean_js_to_consensus": 0.0,
        "search_action_modal_fraction": 1.0,
        "protected_action_modal_fraction": 1.0,
        "target_action_modal_fraction": 1.0,
        "reference_target_matches_repeat_mode": True,
        "confident_correction_label_modal_fraction": 1.0,
        "budget_probes": {
            "64": {
                "js_to_mean_128": 0.03,
                "protected_action_matches_128_mode": True,
                "confident_correction_matches_128_mode": True,
            },
            "128": {
                "js_to_mean_128": 0.0,
                "protected_action_matches_128_mode": True,
                "confident_correction_matches_128_mode": True,
            },
            "256": {
                "js_to_mean_128": 0.01,
                "protected_action_matches_128_mode": True,
                "confident_correction_matches_128_mode": True,
            },
        },
    }


def test_js_and_frozen_stratification_are_well_defined():
    left = np.asarray([0.75, 0.25, 0, 0, 0, 0], dtype=float)
    right = np.asarray([0.25, 0.75, 0, 0, 0, 0], dtype=float)
    assert jensen_shannon(left, left) == 0.0
    assert 0.0 < jensen_shannon(left, right) < np.log(2.0)
    assert classify_stratum("WAIT", "UP", 0.8, 0.15) == "wait_target"
    assert classify_stratum("BOMB", "BOMB", 0.8, 0.15) == "bomb_target"
    assert (classify_stratum("RIGHT", "UP", 0.2, 0.15)
            == "confident_correction_move")
    assert classify_stratum("RIGHT", "RIGHT", 0.01, 0.15) == "preservation_move"
    assert classify_stratum("RIGHT", "UP", 0.14, 0.15) is None


def test_registered_protocol_is_source_bound_and_training_disabled():
    protocol, base = load_preregistration(CONFIG, allow_existing_output=True)
    assert protocol["episode_seeds"] == EXPECTED_SEEDS
    assert tuple(protocol["strata"]) == EXPECTED_STRATA
    assert protocol["automatic_training"] is False and protocol["training"] is False
    assert protocol["repeat_count"] == 8
    assert protocol["budget_probes"] == [64, 128, 256]
    assert base["teacher_persistent_target"] is True
    assert base["teacher_productive_bomb_protection"] is True


def test_real_observer_state_and_atomic_report_accept_numpy_scalars():
    state = initial_state(106120, 400, opponent_count=0)
    digest = observer_state_sha256(state)
    assert len(digest) == 64 and digest == observer_state_sha256(state.copy())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "report.json"
        atomic_json(path, {
            "scalar": np.int64(7),
            "array": np.asarray([1, 2], dtype=np.int64),
        })
        assert json.loads(path.read_text()) == {"array": [1, 2], "scalar": 7}


def test_mechanical_decision_branches_separate_coverage_seed_and_budget():
    config = json.loads(CONFIG.read_text())
    rows = [_row(stratum) for stratum in EXPECTED_STRATA for _ in range(12)]
    available = {stratum: 12 for stratum in EXPECTED_STRATA}
    _, gate = summarize(rows, config, available)
    assert gate["decision"] == "teacher_targets_repeatable" and gate["passed"]

    sparse = [row for row in rows if row["stratum"] != "wait_target"]
    _, gate = summarize(sparse, config, available)
    assert gate["decision"] == "insufficient_stratum_coverage"

    noisy = [dict(row) for row in rows]
    for index in range(8):
        noisy[index] = dict(noisy[index], mean_js_to_consensus=0.8)
    _, gate = summarize(noisy, config, available)
    assert gate["decision"] == "search_seed_instability"

    unstable_budget = [_row(stratum) for stratum in EXPECTED_STRATA for _ in range(12)]
    for row in unstable_budget[:12]:
        row["budget_probes"]["256"]["protected_action_matches_128_mode"] = False
    _, gate = summarize(unstable_budget, config, available)
    assert gate["decision"] == "search_budget_instability"
