"""Contract tests for split exact-v4 Task4A."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

from agent_code.model_a_dqn.network import DuelingDQN as V4Network
from agent_code.model_a_v4_task4_split import train as split_train
from agent_code.model_a_v4_task4_split.config import (
    REPLICAS, SNAPSHOT_ROUNDS, STRATA, final_checkpoint_path, load_protocol,
)
from tools.v4_task4_split_a import dry_run, inner_score, outer_decision


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task4-split-a-s125000.json"


def _row(score=3.0, kills=0.1, suicide=0.1, opponent_score=2.5):
    return {
        "target": {
            "score_per_round": score,
            "kills_per_round": kills,
            "suicides_per_round": suicide,
            "invalid_actions_per_round": 0.0,
        },
        "opponents": {"mean_score_per_agent_round": opponent_score},
    }


def test_protocol_splits_task4_and_executes_only_open_rule_first():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert protocol["task4_subcourse_order"] == [
        "task4a_open_rule", "task4b_classic_duel", "task4c_three_rule",
    ]
    assert protocol["active_subcourse"] == "task4a_open_rule"
    assert tuple(protocol["replicas"]) == REPLICAS
    assert tuple(protocol["snapshot_rounds"]) == SNAPSHOT_ROUNDS
    assert tuple(protocol["inner_selection"]["strata"]) == STRATA
    assert protocol["training"]["scenario"] == "coin-heaven"
    assert protocol["training"]["opponents"] == ["seeded_rule_based_agent"]
    assert protocol["automatic_next_subcourse"] is False


def test_task4a_protocol_uses_93_globally_unique_seeds():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    seeds = []
    for case in protocol["training"]["seeds"].values():
        seeds.extend(case.values())
    for suite in ("inner_selection", "outer_confirmation"):
        for spec in protocol[suite]["strata"].values():
            for case in spec["cases"]:
                seeds.extend(case.values())
    assert len(seeds) == 93
    assert len(seeds) == len(set(seeds))


def test_task4a_agent_reuses_exact_v4_learning_code():
    from agent_code.model_a_v4_task4_split.callbacks import DuelingDQN as SplitNetwork
    from agent_code.model_a_dqn import train as v4_train

    assert SplitNetwork is V4Network
    assert split_train.v4 is v4_train
    assert split_train.setup_training.__code__.co_names == ("v4", "setup_training")
    assert split_train.game_events_occurred.__code__.co_names == (
        "v4", "game_events_occurred",
    )


def test_task4a_resumes_source_optimizer_and_epsilon_with_empty_replay():
    from agent_code.model_a_v4_task4_split import callbacks

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
        protocol["checkpoint_directory"] = str(Path(directory) / "checkpoints")
        temporary_protocol = Path(directory) / "protocol.json"
        temporary_protocol.write_text(json.dumps(protocol), encoding="utf-8")
        output = final_checkpoint_path(protocol, "r1")
        values = {
            "MODEL_A_V4T4S_PROTOCOL_PATH": str(temporary_protocol),
            "MODEL_A_V4T4S_CHECKPOINT_PATH": str(output),
            "MODEL_A_V4T4S_PARENT_PATH": str(ROOT / protocol["source_parent"]["path"]),
            "MODEL_A_V4T4S_REPLICA": "r1",
            "MODEL_A_V4T4S_SEED": str(protocol["training"]["seeds"]["r1"]["agent_seed"]),
        }
        previous = {name: os.environ.get(name) for name in values}
        os.environ.update(values)
        try:
            agent = SimpleNamespace(train=True, logger=logging.getLogger("task4a-test"))
            callbacks.setup(agent)
            split_train.setup_training(agent)
            assert abs(agent.epsilon - 0.05) < 1e-12
            assert agent.completed_rounds == 1200
            assert agent.training_steps == 480870
            assert len(agent.replay_buffer) == 0
            assert not output.exists()
        finally:
            for name, old_value in previous.items():
                if old_value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old_value


def test_task4a_dry_run_stops_before_task4b():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["total_training_rounds"] == 600
    assert summary["snapshots"] == 24
    assert summary["inner_evaluation_rounds"] == 3640
    assert summary["maximum_outer_evaluation_rounds"] == 1050
    assert summary["formal_training_started"] is False
    assert summary["formal_evaluation_started"] is False
    assert summary["automatic_task4b_started"] is False
    assert summary["writes_terminal_report_and_stops"] is True


def test_inner_support_requires_open_rule_mastery_and_all_retention():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    source = {name: _row() for name in STRATA}
    candidate = {name: _row() for name in STRATA}
    candidate["task4a_open_rule"] = _row(score=3.5, opponent_score=3.4)
    accepted = inner_score(protocol, candidate, source)
    assert accepted["retention_eligible"] is True
    assert accepted["supportive"] is True
    candidate["task4b_classic_duel"] = _row(score=2.7)
    rejected = inner_score(protocol, candidate, source)
    assert rejected["retention_gates"]["classic_duel_score"] is False
    assert rejected["supportive"] is False


def test_outer_passes_task4a_only_and_requires_separate_task4b():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    v4 = {name: _row() for name in STRATA}
    source = {name: _row() for name in STRATA}
    candidate = {name: _row() for name in STRATA}
    candidate["task1"] = _row(score=2.7)
    candidate["task2"] = _row(score=2.75)
    candidate["task3_peaceful"] = _row(score=3.25, kills=0.1, suicide=0.15)
    candidate["task3_coin"] = _row(score=2.75, kills=0.1, suicide=0.15)
    candidate["task4a_open_rule"] = _row(score=4.0, suicide=0.15, opponent_score=4.0)
    result = outer_decision(protocol, {"v4": v4, "source-r2": source, "candidate": candidate})
    assert result["passed"] is True
    assert result["task4a_complete"] is True
    assert result["task4_complete"] is False
    assert result["next_subcourse"] == "task4b_classic_duel"
    candidate["task4a_open_rule"]["target"]["score_per_round"] = 3.9
    result = outer_decision(protocol, {"v4": v4, "source-r2": source, "candidate": candidate})
    assert result["passed"] is False
    assert result["current_subcourse_gates"]["open_rule_beats_rule"] is False
