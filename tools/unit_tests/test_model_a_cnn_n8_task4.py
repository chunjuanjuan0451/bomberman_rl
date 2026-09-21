"""Contract tests for the unified CNN Task4 course."""

import hashlib
import json
from pathlib import Path

from tools.cnn_n8_task4 import load_protocol, rank_course_champions, select_task4_milestone, validate_lineage


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-task4-s151000.json"


def _seeds(node):
    values = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("_seed") and isinstance(value, int):
                values.append(value)
            else:
                values.extend(_seeds(value))
    elif isinstance(node, list):
        for value in node:
            values.extend(_seeds(value))
    return values


def test_cnn_task4_uses_selected_task3c_parents_and_all_course_champions():
    protocol, _ = load_protocol(PROTOCOL)
    validate_lineage(protocol)
    report = json.loads((ROOT / protocol["course_reports"]["task3_coin"]["path"]).read_text(encoding="utf-8"))
    assert report["selection"]["selected_common_milestone"] == 150
    assert protocol["parents"] == report["selected_checkpoints"]
    assert sum(len(rows) for rows in protocol["course_champions"].values()) == 12


def test_cnn_task4_seeds_are_unique_and_new():
    protocol, _ = load_protocol(PROTOCOL)
    values = _seeds(protocol)
    old = "".join(path.read_text(encoding="utf-8") for path in (
        ROOT / "experiments/configs/model-a-cnn-n8-task1-s147000.json",
        ROOT / "experiments/configs/model-a-cnn-n8-task2-s148000.json",
        ROOT / "experiments/configs/model-a-cnn-n8-task3-peaceful-s149000.json",
        ROOT / "experiments/configs/model-a-cnn-n8-task3-coin-s150000.json",
    ))
    assert len(values) == len(set(values))
    assert not any(str(seed) in old for seed in values)


def test_cnn_task4_reports_and_schedule_are_frozen():
    protocol, _ = load_protocol(PROTOCOL)
    for spec in protocol["course_reports"].values():
        path = ROOT / spec["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
    assert protocol["learning"]["checkpoint_every_rounds"] == 400
    assert protocol["task4"]["milestones"] == [400, 800, 1200, 1600]
    assert protocol["task4"]["total_training_rounds"] == 4800
    assert protocol["validation"]["total_rounds"] == 8400


def test_cnn_task4_has_one_training_roster_and_two_selection_strata():
    protocol, _ = load_protocol(PROTOCOL)
    rule = "seeded_rule_based_agent"
    assert protocol["task4"]["opponents"] == [rule, rule, rule]
    assert protocol["validation"]["task4_rule_selection"]["opponents"] == [rule, rule, rule]
    assert protocol["validation"]["task4_mixed_selection"]["opponents"] == [
        rule, "seeded_coin_collector_agent", "seeded_random_agent",
    ]


def test_cnn_task4_selection_combines_both_strata_and_uses_one_shared_milestone():
    protocol, _ = load_protocol(PROTOCOL)

    def row(score):
        return {"rounds": 100, "score": score, "coins": score, "kills": 0, "crates": 0,
                "bombs": 0, "moves": 100, "waits": 0, "invalid_actions": 0, "suicides": 0,
                "steps": 100, "mean_decision_time_ms": 1.0}

    rule_rows = {"frozen-v4": row(1), "source-r2": row(1)}
    mixed_rows = {"frozen-v4": row(1), "source-r2": row(1)}
    totals = {400: (30, 30), 800: (40, 40), 1200: (39, 39), 1600: (38, 38)}
    for milestone, (rule_score, mixed_score) in totals.items():
        for replica in ("r1", "r2", "r3"):
            label = f"task4-{replica}-round{milestone:04d}"
            rule_rows[label] = row(rule_score)
            mixed_rows[label] = row(mixed_score)
    selection, _ = select_task4_milestone(rule_rows, mixed_rows, protocol)
    assert selection["selected_common_milestone"] == 800


def test_cnn_task4_course_ranking_does_not_assume_the_latest_course_wins():
    base = {"rounds": 200, "score": 500, "score_per_round": 2.5, "coins_per_round": 2.0,
            "suicides_per_round": 0.1, "wait_fraction": 0.1, "mean_decision_time_ms": 1.0}
    rows = {
        "task2-r1-selected": dict(base, score=600, score_per_round=3.0),
        "task4-r1-round0400": base,
        "frozen-v4": base,
        "source-r2": base,
    }
    assert rank_course_champions(rows)["selected_label"] == "task2-r1-selected"
