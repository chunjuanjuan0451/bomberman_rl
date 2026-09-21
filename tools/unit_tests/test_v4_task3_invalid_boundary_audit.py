"""Contract tests for the Task3 dynamic-collision boundary audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.v4_task3_invalid_boundary_audit import decision_legality, dry_run, load_protocol


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task3-invalid-boundary-audit-s122000.json"


def _state() -> dict:
    field = np.zeros((5, 5), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "field": field,
        "self": ("target", 0, True, (1, 1)),
        "others": [("opponent", 0, True, (3, 1))],
        "bombs": [],
    }


def test_boundary_protocol_replays_only_revealed_coin_cases_without_training():
    protocol, _, report = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["revealed_cases_only"] == [122401, 122402]
    assert summary["total_diagnostic_rounds"] == 100
    assert summary["formal_decision_override"] is False
    assert summary["efficacy_reselection_allowed"] is False
    assert summary["training_started"] is False
    assert summary["task4_started"] is False
    assert report["result"]["decision"] == "source_r2_rejected_task3_unresolved_stop"


def test_decision_legality_marks_a_free_destination_as_legal_before_execution():
    legal, destination, reason = decision_legality(_state(), "RIGHT")
    assert legal is True
    assert destination == (2, 1)
    assert reason == "free_at_decision"


def test_decision_legality_rejects_a_destination_occupied_in_the_observed_state():
    state = _state()
    state["others"] = [("opponent", 0, True, (2, 1))]
    legal, destination, reason = decision_legality(state, "RIGHT")
    assert legal is False
    assert destination == (2, 1)
    assert reason == "agent_occupied"
