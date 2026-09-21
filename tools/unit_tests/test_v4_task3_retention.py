"""Contract tests for retention-aware exact-v4 Task3 training."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

from agent_code.model_a_dqn.network import DuelingDQN as V4Network
from agent_code.model_a_v4_task3_retention import train as retention_train
from agent_code.model_a_v4_task3_retention.config import BRANCHES, STAGES, load_protocol
from tools.v4_task3_retention import dry_run, selection_score


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task3-retention-s119000.json"


def test_protocol_freezes_separate_task3_stages_and_globally_unique_seeds():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert tuple(protocol["branches"]) == BRANCHES
    assert tuple(protocol["stage_order"]) == STAGES
    assert protocol["inner_selection"]["current_task_metrics"] == [
        "score_per_round", "kills_per_round",
    ]
    assert protocol["inner_selection"]["stages"]["task3_coin"]["prior_retention_metrics"]["task3_peaceful"] == [
        "score_per_round", "kills_per_round",
    ]
    seeds = []
    for stage in STAGES:
        for case in protocol["training"][stage]["branches"].values():
            seeds.extend(case.values())
    for suite in (protocol["inner_selection"]["strata"], protocol["outer_evaluation"]["strata"]):
        for spec in suite.values():
            for case in spec["cases"]:
                seeds.extend(case.values())
    assert len(seeds) == len(set(seeds))


def test_protocol_rejects_training_and_evaluation_seed_reuse():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["inner_selection"]["strata"]["task1"]["cases"][0]["world_seed"] = 119001
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "protocol.json"
        path.write_text(json.dumps(protocol), encoding="utf-8")
        try:
            load_protocol(path)
        except ValueError as exc:
            assert "globally unique" in str(exc)
        else:
            raise AssertionError("training/evaluation seed overlap must fail closed")


def test_agent_reuses_exact_v4_network_and_learning_callbacks():
    from agent_code.model_a_v4_task3_retention.callbacks import DuelingDQN as Task3Network
    from agent_code.model_a_dqn import train as v4_train

    assert Task3Network is V4Network
    assert retention_train.v4 is v4_train
    assert retention_train.setup_training.__code__.co_names == ("v4", "setup_training")
    assert retention_train.game_events_occurred.__code__.co_names == (
        "v4", "game_events_occurred",
    )


def test_peaceful_evaluation_setup_loads_selected_checkpoint_without_writing():
    from agent_code.model_a_v4_task3_retention import callbacks

    protocol, _ = load_protocol(PROTOCOL_PATH)
    checkpoint = ROOT / protocol["checkpoint_directory"] / "b1/task3_peaceful/selected.pt"
    before = checkpoint.stat().st_mtime_ns
    values = {
        "MODEL_A_V4T3R_PROTOCOL_PATH": str(PROTOCOL_PATH),
        "MODEL_A_V4T3R_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_V4T3R_BRANCH": "b1",
        "MODEL_A_V4T3R_STAGE": "task3_peaceful",
        "MODEL_A_V4T3R_SEED": str(protocol["training"]["task3_peaceful"]["branches"]["b1"]["agent_seed"]),
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        agent = SimpleNamespace(train=False, logger=logging.getLogger("task3-retention-test"))
        callbacks.setup(agent)
        assert agent.stage_id == "task3_peaceful"
        assert agent.completed_rounds > 0
        assert checkpoint.stat().st_mtime_ns == before
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def test_selection_requires_both_current_score_and_kills_near_best():
    accepted = selection_score(
        {"score_per_round": 9.0, "kills_per_round": 0.18},
        {"score_per_round": 10.0, "kills_per_round": 0.20},
        {"task1": 20.0, "task2": 3.0},
        {"task1": 21.0, "task2": 3.0},
        0.9,
    )
    rejected = selection_score(
        {"score_per_round": 10.0, "kills_per_round": 0.10},
        {"score_per_round": 10.0, "kills_per_round": 0.20},
        {"task1": 20.0, "task2": 3.0},
        {"task1": 21.0, "task2": 3.0},
        0.9,
    )
    assert accepted["current_task_eligible"] is True
    assert rejected["current_task_metric_gates"] == {
        "score_per_round": True, "kills_per_round": False,
    }
    assert rejected["current_task_eligible"] is False


def test_dry_run_stops_after_selection_free_outer_evaluation():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["training_rounds"] == 3600
    assert summary["inner_evaluation_rounds"] == 2670
    assert summary["outer_evaluation_rounds"] == 1000
    assert summary["formal_training_started"] is False
    assert summary["formal_evaluation_started"] is False
    assert summary["task4_started"] is False
    assert summary["writes_terminal_report_and_stops"] is True
