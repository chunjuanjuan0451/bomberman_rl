"""Contract tests for the sole-candidate source-r2 Task3 confirmation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.v4_task3_source_r2_final_confirmation import (
    EXPECTED_CASES,
    LABELS,
    STRATA,
    decide,
    dry_run,
    freeze_confirmed_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task3-source-r2-final-confirmation-s122000.json"


def _historical_protocol() -> tuple[dict, str]:
    """Load the completed protocol without rerunning its one-time seed-freshness audit."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    return protocol, hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def test_protocol_fixes_source_r2_as_the_only_candidate_on_24_fresh_seed_values():
    protocol, _ = _historical_protocol()
    assert tuple(protocol["labels"]) == LABELS
    assert protocol["fixed_candidate"] == "source-r2"
    assert protocol["candidate_count"] == 1
    assert protocol["selection_free"] is True
    assert protocol["training_allowed"] is False
    assert tuple(protocol["evaluation"]["strata"]) == STRATA
    actual = {
        value
        for spec in protocol["evaluation"]["strata"].values()
        for case in spec["cases"]
        for value in case.values()
    }
    expected = {
        value
        for cases in EXPECTED_CASES.values()
        for case in cases
        for value in case.values()
    }
    assert actual == expected
    assert len(actual) == 24


def test_dry_run_is_exactly_400_rounds_without_training_or_task4():
    protocol, _ = _historical_protocol()
    summary = dry_run(protocol)
    assert summary["labels"] == list(LABELS)
    assert summary["fixed_candidate"] == "source-r2"
    assert summary["total_evaluation_rounds"] == 400
    assert summary["uses_fresh_s122000_seeds"] is True
    assert summary["formal_evaluation_started"] is False
    assert summary["training_started"] is False
    assert summary["checkpoint_copy_started"] is False
    assert summary["conditional_checkpoint_freeze_on_pass"] is True
    assert summary["task4_started"] is False


def _row(score: float, kills: float = 0.0, suicide: float = 0.0, invalid: float = 0.0) -> dict:
    return {
        "score_per_round": score,
        "coins_per_round": score - 5.0 * kills,
        "kills_per_round": kills,
        "suicides_per_round": suicide,
        "invalid_actions_per_round": invalid,
    }


def _passing_rows() -> dict:
    return {
        "task1": {"v4": _row(10.0), "source-r2": _row(9.1)},
        "task2": {"v4": _row(3.2), "source-r2": _row(3.0)},
        "task3_peaceful": {
            "v4": _row(2.0, 0.1, 0.01),
            "source-r2": _row(2.3, 0.2, 0.03),
        },
        "task3_coin": {
            "v4": _row(3.1, 0.05, 0.02),
            "source-r2": _row(3.0, 0.1, 0.04),
        },
    }


def test_unchanged_nine_gates_confirm_task3_and_terminate_search():
    protocol, _ = _historical_protocol()
    result = decide(protocol, _passing_rows())
    assert result["passed"] is True
    assert all(result["gates"].values())
    assert result["decision"] == "source_r2_confirmed_task3_complete"
    assert result["task3_complete"] is True
    assert result["task3_search_terminated"] is True
    assert result["task4_started"] is False


def test_coin_failure_rejects_source_r2_and_leaves_task3_unresolved():
    protocol, _ = _historical_protocol()
    rows = _passing_rows()
    rows["task3_coin"]["source-r2"] = _row(2.0, 0.0)
    result = decide(protocol, rows)
    assert result["gates"]["coin_score"] is False
    assert result["gates"]["coin_kills_absolute"] is False
    assert result["passed"] is False
    assert result["decision"] == "source_r2_rejected_task3_unresolved_stop"
    assert result["task3_complete"] is False
    assert result["automatic_checkpoint_copied"] is False


def test_pass_freezes_a_byte_identical_dedicated_checkpoint_without_replacing_default():
    protocol, protocol_hash = _historical_protocol()
    isolated = deepcopy(protocol)
    with TemporaryDirectory(prefix="v4t3-source-r2-confirm-") as directory:
        temporary = Path(directory)
        isolated["confirmed_checkpoint"]["path"] = str(temporary / "task3.pt")
        isolated["confirmed_checkpoint"]["selection_manifest_path"] = str(temporary / "selection.json")
        frozen = freeze_confirmed_checkpoint(
            isolated,
            protocol_hash,
            {"decision": "source_r2_confirmed_task3_complete"},
        )
        assert frozen["sha256"] == isolated["checkpoint_inventory"]["source_r2"]["sha256"]
        assert Path(frozen["path"]).is_file()
        assert Path(frozen["selection_manifest_path"]).is_file()
        assert isolated["checkpoint_inventory"]["v4"]["path"] == "agent_code/model_a_dqn/model_a.pt"
