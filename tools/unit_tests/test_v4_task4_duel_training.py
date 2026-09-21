"""Contract tests for the single-variable exact-v4 Task4 duel course."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

from agent_code.model_a_dqn.network import DuelingDQN as V4Network
from agent_code.model_a_v4_task4_duel import train as task4_train
from agent_code.model_a_v4_task4_duel.config import (
    REPLICAS, SNAPSHOT_ROUNDS, STRATA, load_protocol,
)
from tools.v4_task4_duel_training import dry_run, inner_score, outer_decision


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task4-duel-training-s124000.json"


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


def test_protocol_changes_only_duel_opponent_sampling_and_uses_fresh_suites():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert tuple(protocol["replicas"]) == REPLICAS
    assert tuple(protocol["snapshot_rounds"]) == SNAPSHOT_ROUNDS
    assert tuple(protocol["inner_selection"]["strata"]) == STRATA
    assert tuple(protocol["outer_confirmation"]["strata"]) == STRATA
    assert protocol["single_training_variable"] == "opponent_sampling_distribution"
    assert protocol["training"]["opponents"] == ["seeded_rule_based_agent"]
    assert protocol["learning_contract"]["epsilon"].endswith("unchanged 0.05 floor")
    seeds = []
    for case in protocol["training"]["seeds"].values():
        seeds.extend(case.values())
    for suite in ("inner_selection", "outer_confirmation"):
        for spec in protocol[suite]["strata"].values():
            for case in spec["cases"]:
                seeds.extend(case.values())
    assert len(seeds) == 81
    assert len(seeds) == len(set(seeds))


def test_protocol_rejects_training_and_selection_seed_reuse():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["inner_selection"]["strata"]["task1"]["cases"][0]["world_seed"] = 124101
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "protocol.json"
        path.write_text(json.dumps(protocol), encoding="utf-8")
        try:
            load_protocol(path)
        except ValueError as exc:
            assert "globally unique" in str(exc)
        else:
            raise AssertionError("Task4 seed reuse must fail closed")


def test_task4_agent_reuses_exact_v4_network_optimizer_reward_and_replay_callbacks():
    from agent_code.model_a_v4_task4_duel.callbacks import DuelingDQN as Task4Network
    from agent_code.model_a_dqn import train as v4_train

    assert Task4Network is V4Network
    assert task4_train.v4 is v4_train
    assert task4_train.setup_training.__code__.co_names == ("v4", "setup_training")
    assert task4_train.game_events_occurred.__code__.co_names == (
        "v4", "game_events_occurred",
    )


def test_training_setup_resumes_source_optimizer_and_epsilon_but_resets_replay():
    from agent_code.model_a_v4_task4_duel import callbacks
    from agent_code.model_a_v4_task4_duel.config import final_checkpoint_path

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    replica = "r1"
    with tempfile.TemporaryDirectory() as directory:
        protocol["checkpoint_directory"] = str(Path(directory) / "checkpoints")
        temporary_protocol = Path(directory) / "protocol.json"
        temporary_protocol.write_text(json.dumps(protocol), encoding="utf-8")
        output = final_checkpoint_path(protocol, replica)
        values = {
            "MODEL_A_V4T4D_PROTOCOL_PATH": str(temporary_protocol),
            "MODEL_A_V4T4D_CHECKPOINT_PATH": str(output),
            "MODEL_A_V4T4D_PARENT_PATH": str(ROOT / protocol["source_parent"]["path"]),
            "MODEL_A_V4T4D_REPLICA": replica,
            "MODEL_A_V4T4D_SEED": str(protocol["training"]["seeds"][replica]["agent_seed"]),
        }
        previous = {name: os.environ.get(name) for name in values}
        os.environ.update(values)
        try:
            agent = SimpleNamespace(train=True, logger=logging.getLogger("task4-duel-test"))
            callbacks.setup(agent)
            task4_train.setup_training(agent)
            assert abs(agent.epsilon - 0.05) < 1e-12
            assert agent.completed_rounds == 1200
            assert agent.training_steps == 480870
            assert len(agent.replay_buffer) == 0
            assert agent.parent_sha256 == protocol["source_parent"]["sha256"]
            assert not output.exists()
        finally:
            for name, old_value in previous.items():
                if old_value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old_value


def test_dry_run_preregisters_three_repeats_snapshot_selection_and_fresh_outer():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["total_training_rounds"] == 1800
    assert summary["snapshots"] == 18
    assert summary["inner_evaluation_rounds"] == 2400
    assert summary["maximum_outer_evaluation_rounds"] == 900
    assert summary["formal_training_started"] is False
    assert summary["formal_evaluation_started"] is False
    assert summary["automatic_followup_started"] is False
    assert summary["writes_terminal_report_and_stops"] is True


def test_inner_support_requires_duel_progress_and_all_retention_gates():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    source = {name: _row() for name in STRATA}
    candidate = {name: _row() for name in STRATA}
    candidate["task4_rule_duel"] = _row(score=3.5, suicide=0.0)
    accepted = inner_score(protocol, candidate, source)
    assert accepted["retention_eligible"] is True
    assert accepted["supportive"] is True
    candidate["task3_coin"] = _row(score=2.4, kills=0.1)
    rejected = inner_score(protocol, candidate, source)
    assert rejected["retention_gates"]["coin_score"] is False
    assert rejected["supportive"] is False


def test_outer_requires_task1_to_task4_and_both_rule_matchups_to_pass():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    v4 = {name: _row() for name in STRATA}
    source = {name: _row() for name in STRATA}
    candidate = {name: _row() for name in STRATA}
    candidate["task1"] = _row(score=2.7)
    candidate["task2"] = _row(score=2.75)
    candidate["task3_peaceful"] = _row(score=3.25, kills=0.1, suicide=0.15)
    candidate["task3_coin"] = _row(score=2.75, kills=0.1, suicide=0.15)
    candidate["task4_rule_duel"] = _row(score=4.0, kills=0.1, suicide=0.15, opponent_score=4.0)
    candidate["task4_three_rule"] = _row(score=3.0, kills=0.1, suicide=0.15, opponent_score=3.0)
    result = outer_decision(protocol, {"v4": v4, "source-r2": source, "candidate": candidate})
    assert result["passed"] is True
    assert result["decision"] == "task4_duel_training_confirmed"
    candidate["task4_rule_duel"]["target"]["score_per_round"] = 3.99
    result = outer_decision(protocol, {"v4": v4, "source-r2": source, "candidate": candidate})
    assert result["passed"] is False
    assert result["task4_gates"]["task4_rule_duel"]["beats_rule"] is False
    assert result["decision"] == "task4_duel_candidate_rejected_stop"
