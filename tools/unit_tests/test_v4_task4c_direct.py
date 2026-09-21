"""Contract tests for direct exact-v4 Task4C training."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

from agent_code.model_a_dqn.network import DuelingDQN as V4Network
from agent_code.model_a_v4_task4c import train as task4c_train
from agent_code.model_a_v4_task4c.config import (
    OUTER_STRATA, REPLICAS, SNAPSHOT_ROUNDS, final_checkpoint_path, load_protocol,
)
from tools.v4_task4c_direct import dry_run, inner_score, outer_decision


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task4c-direct-s127000.json"


def _row(score=3.0, kills=0.1, suicide=0.1, opponent_score=2.8):
    return {
        "target": {
            "score_per_round": score,
            "kills_per_round": kills,
            "suicides_per_round": suicide,
            "invalid_actions_per_round": 0.0,
        },
        "opponents": {"mean_score_per_agent_round": opponent_score},
    }


def test_task4c_protocol_changes_only_final_three_rule_distribution():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert protocol["active_subcourse"] == "task4c_three_rule"
    assert tuple(protocol["replicas"]) == REPLICAS
    assert tuple(protocol["snapshot_rounds"]) == SNAPSHOT_ROUNDS
    assert protocol["training"]["scenario"] == "classic"
    assert protocol["training"]["opponents"] == ["seeded_rule_based_agent"] * 3
    assert protocol["source_parent"]["lineage"]["replica"] == "r2"
    assert tuple(protocol["outer_confirmation"]["strata"]) == OUTER_STRATA


def test_task4c_protocol_registers_exactly_75_unique_seeds():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    seeds = []
    for case in protocol["training"]["seeds"].values():
        seeds.extend(case.values())
    for cases in protocol["inner_signal"]["cases_by_replica"].values():
        for case in cases:
            seeds.extend(case.values())
    for spec in protocol["outer_confirmation"]["strata"].values():
        for case in spec["cases"]:
            seeds.extend(case.values())
    assert len(seeds) == 75
    assert len(set(seeds)) == 75


def test_task4c_reuses_exact_v4_learner_and_resumes_source_state():
    from agent_code.model_a_v4_task4c import callbacks
    from agent_code.model_a_v4_task4c.callbacks import DuelingDQN as Task4CNetwork
    from agent_code.model_a_dqn import train as v4_train

    assert Task4CNetwork is V4Network
    assert task4c_train.v4 is v4_train
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
        protocol["checkpoint_directory"] = str(Path(directory) / "checkpoints")
        temporary_protocol = Path(directory) / "protocol.json"
        temporary_protocol.write_text(json.dumps(protocol), encoding="utf-8")
        output = final_checkpoint_path(protocol, "r1")
        values = {
            "MODEL_A_V4T4C_PROTOCOL_PATH": str(temporary_protocol),
            "MODEL_A_V4T4C_CHECKPOINT_PATH": str(output),
            "MODEL_A_V4T4C_PARENT_PATH": str(ROOT / protocol["source_parent"]["path"]),
            "MODEL_A_V4T4C_REPLICA": "r1",
            "MODEL_A_V4T4C_SEED": str(protocol["training"]["seeds"]["r1"]["agent_seed"]),
        }
        previous = {name: os.environ.get(name) for name in values}
        os.environ.update(values)
        try:
            agent = SimpleNamespace(train=True, logger=logging.getLogger("task4c-test"))
            callbacks.setup(agent)
            task4c_train.setup_training(agent)
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


def test_task4c_dry_run_has_cheap_signal_gate_before_conditional_outer():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["parent"] == "source-r2"
    assert summary["total_training_rounds"] == 600
    assert summary["snapshots"] == 24
    assert summary["inner_signal_evaluation_rounds"] == 600
    assert summary["maximum_outer_evaluation_rounds"] == 1200
    assert summary["formal_training_started"] is False


def test_inner_signal_requires_score_kills_safety_and_rule_mastery():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    source = _row(score=3.0, kills=0.1, suicide=0.2)
    v4 = _row(score=3.1, kills=0.08, suicide=0.1)
    candidate = _row(score=3.3, kills=0.12, suicide=0.15, opponent_score=3.2)
    result = inner_score(protocol, candidate, source, v4)
    assert result["supportive"] is True
    candidate["target"]["score_per_round"] = 3.0
    assert inner_score(protocol, candidate, source, v4)["supportive"] is False


def test_outer_pass_requires_prior_retention_and_replicated_task4c_gain():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    rows = {
        "v4": {name: _row(score=3.0) for name in OUTER_STRATA},
        "source-r2": {name: _row(score=3.0) for name in OUTER_STRATA},
        "candidate": {name: _row(score=3.0) for name in OUTER_STRATA},
    }
    rows["candidate"]["task4c_three_rule"] = _row(score=3.3, kills=0.12, suicide=0.1, opponent_score=3.2)
    case_runs = {label: {"task4c_three_rule": []} for label in ("v4", "source-r2", "candidate")}
    for _ in range(4):
        case_runs["v4"]["task4c_three_rule"].append({"target_metrics": {"score_per_round": 3.0}})
        case_runs["source-r2"]["task4c_three_rule"].append({"target_metrics": {"score_per_round": 3.0}})
        case_runs["candidate"]["task4c_three_rule"].append({"target_metrics": {"score_per_round": 3.3}})
    result = outer_decision(protocol, rows, case_runs)
    assert result["passed"] is True
    assert result["task4_complete"] is True
    rows["candidate"]["task3_peaceful"]["target"]["score_per_round"] = 2.0
    result = outer_decision(protocol, rows, case_runs)
    assert result["passed"] is False
    assert result["prior_course_gates"]["peaceful_score"] is False
