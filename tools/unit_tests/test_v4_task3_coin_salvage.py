"""Contract tests for the evaluation-only Task3 coin snapshot salvage audit."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from tools.v4_task3_coin_salvage import (
    BRANCHES, ROUNDS, candidate_label, decide, dry_run, load_protocol, shortlist,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task3-coin-salvage-s121000.json"


def test_protocol_binds_all_existing_snapshots_and_uses_fresh_split_seeds():
    protocol, _, source, _ = load_protocol(PROTOCOL_PATH)
    assert sum(len(items) for items in protocol["checkpoint_inventory"]["coin_snapshots"].values()) == 36
    assert protocol["source_experiment"]["source_outer_seeds_used"] is False
    fresh = []
    for suite in (protocol["screening"]["strata"], protocol["selection"]["strata"]):
        for spec in suite.values():
            for case in spec["cases"]:
                fresh.extend(case.values())
    assert len(fresh) == len(set(fresh))
    assert all(value not in {120101, 220101, 320101} for value in fresh)
    assert source["report_path"] == "experiments/logs/evaluations/model-a-v4-task3-retention-s119000.json"


def test_dry_run_is_small_evaluation_only_and_preserves_outer():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["coin_snapshots"] == 36
    assert summary["screening_evaluation_rounds"] == 820
    assert summary["maximum_selection_evaluation_rounds"] == 1120
    assert summary["maximum_total_evaluation_rounds"] == 1940
    assert summary["source_outer_seeds_used"] is False
    assert summary["training_started"] is False
    assert summary["outer_evaluation_started"] is False
    assert summary["task4_started"] is False


def test_screening_shortlist_keeps_score_kill_and_balanced_extremes():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    rows = {}
    for branch in BRANCHES:
        for stage_round in ROUNDS:
            rows[candidate_label(branch, stage_round)] = {
                "score_per_round": 1.0, "kills_per_round": 0.1,
            }
    rows["b1-r0050"] = {"score_per_round": 5.0, "kills_per_round": 0.1}
    rows["b1-r0100"] = {"score_per_round": 1.0, "kills_per_round": 0.5}
    rows["b1-r0150"] = {"score_per_round": 4.0, "kills_per_round": 0.4}
    selected, diagnostics = shortlist(protocol, {"task3_coin": rows})
    assert selected["b1"] == ["b1-r0050", "b1-r0100", "b1-r0150"]
    assert diagnostics["b1"]["reason_to_label"] == {
        "highest_score": "b1-r0050",
        "highest_kills": "b1-r0100",
        "highest_balanced_achievement": "b1-r0150",
    }
    assert all(len(selected[branch]) <= 3 for branch in BRANCHES)


def test_selection_is_provisional_and_retention_aware():
    protocol, _, _, _ = load_protocol(PROTOCOL_PATH)
    shortlists = {branch: [candidate_label(branch, 50)] for branch in BRANCHES}
    labels = [shortlists[branch][0] for branch in BRANCHES]
    rows = {stratum: {} for stratum in ("task1", "task2", "task3_peaceful", "task3_coin")}
    for index, branch in enumerate(BRANCHES):
        parent = f"parent-{branch}"
        rows["task1"][parent] = {"score_per_round": 10.0}
        rows["task2"][parent] = {"score_per_round": 4.0}
        rows["task3_peaceful"][parent] = {"score_per_round": 5.0, "kills_per_round": 0.2}
        fraction = (0.9, 0.8, 0.75)[index]
        label = labels[index]
        rows["task1"][label] = {"score_per_round": 10.0 * fraction}
        rows["task2"][label] = {"score_per_round": 4.0 * fraction}
        rows["task3_peaceful"][label] = {
            "score_per_round": 5.0 * fraction,
            "kills_per_round": 0.2 * fraction,
        }
        rows["task3_coin"][label] = {"score_per_round": 3.0, "kills_per_round": 0.2}
    rows["task3_coin"]["v4"] = {"score_per_round": 2.8, "kills_per_round": 0.1}
    result = decide(protocol, shortlists, rows)
    assert result["decision"] == "provisional_coin_snapshot_selected"
    assert result["provisional_selected_label"] == "b1-r0050"
    assert result["requires_selection_free_outer_confirmation"] is True
    assert result["automatic_checkpoint_copied"] is False
    assert result["training_started"] is False
    assert result["task4_started"] is False


def test_protocol_rejects_seed_reuse_with_source_outer_suite():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["screening"]["strata"]["task3_coin"]["cases"][0]["world_seed"] = 120101
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "protocol.json"
        path.write_text(json.dumps(protocol), encoding="utf-8")
        try:
            load_protocol(path)
        except ValueError as exc:
            assert "overlap" in str(exc)
        else:
            raise AssertionError("source outer seed reuse must fail closed")
