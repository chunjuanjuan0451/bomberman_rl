"""Contracts for the CNN n=8 safety replay fine-tune A/B."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import events as e

from agent_code.model_a_cnn_n8 import rewards as base_rewards
from agent_code.model_a_cnn_n8_safety_ab import train as safety_train
from agent_code.model_a_cnn_n8_safety_ab.replay import ReplayBuffer, Transition
from tools.cnn_n8_safety_replay_ab import (
    ARMS, MILESTONES, REPLICAS, decide_task4, dry_run, load_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-safety-replay-ab-s153000.json"


def _transition(index: int, kill: bool = False, self_kill: bool = False) -> Transition:
    state = (np.zeros((11, 33, 33), dtype=np.uint8), np.zeros(6, dtype=np.float32))
    return Transition(state, 0, float(index), None, True, None, 0.99, 1,
                      1, index, kill, self_kill)


def _row(score: float, suicide: float, kills: float = 0.1, invalid: float = 0.5) -> dict:
    rounds, steps = 100, 1000
    values = {
        "rounds": rounds, "score": round(score * rounds), "coins": round((score - 5 * kills) * rounds),
        "kills": round(kills * rounds), "crates": 0, "bombs": 100,
        "moves": 800, "waits": 50, "invalid_actions": round(invalid * rounds),
        "suicides": round(suicide * rounds), "steps": steps,
        "decision_time_seconds": 1.0, "mean_decision_time_ms": 1.0,
    }
    for key in ("score", "coins", "kills", "crates", "bombs", "moves", "waits", "invalid_actions", "suicides"):
        values[f"{key}_per_round"] = values[key] / rounds
    values["wait_fraction"] = values["waits"] / steps
    return values


def _task4_rows(treatment_suicide: float = 0.14) -> dict:
    labels = ["parent-cnn", "frozen-v4", "source-r2"]
    labels += [f"{arm}-{replica}-round{milestone:04d}"
               for arm in ARMS for replica in REPLICAS for milestone in MILESTONES]
    result = {stratum: {} for stratum in ("task4_rule", "task4_mixed")}
    for stratum in result:
        for label in labels:
            if label.startswith("uniform_n8"):
                result[stratum][label] = _row(3.15, 0.24)
            elif label.startswith("balanced_safety_n8"):
                result[stratum][label] = _row(3.12, treatment_suicide)
            elif label == "parent-cnn":
                result[stratum][label] = _row(3.15, 0.25)
            else:
                result[stratum][label] = _row(2.9, 0.14)
    return result


def test_protocol_freezes_one_variable_and_paired_budget():
    protocol, _, _ = load_protocol(PROTOCOL)
    assert protocol["single_training_variable"] == "replay_sampling_uniform_vs_reserved_8_kill_8_self"
    assert protocol["learning"]["reward_contract_changed"] is False
    assert protocol["reward_contract"]["changed_from_parent"] is False
    assert protocol["training"]["total_environment_rounds"] == 1200
    assert protocol["evaluation"]["maximum_environment_rounds_including_training"] == 3300
    assert protocol["training"]["paired_seeds"]["r1"]["world_seed"] == 153101


def test_replay_reserves_exact_slots_and_uses_unique_batch_items():
    replay = ReplayBuffer(500)
    for index in range(300):
        replay.add(_transition(index, kill=index < 20, self_kill=20 <= index < 40))
    items, diagnostics = replay.sample(256, np.random.default_rng(7), 8, 8)
    assert len(items) == 256
    assert len({item.start_step for item in items}) == 256
    assert diagnostics["fulfilled_kill_slots"] == 8
    assert diagnostics["fulfilled_self_slots"] == 8
    assert diagnostics["fallback_uniform_slots"] == 0
    assert diagnostics["realized_kill_chain_samples"] >= 8
    assert diagnostics["realized_self_chain_samples"] >= 8


def test_nstep_tags_kill_and_self_chains_without_crossing_episode():
    state = (np.zeros((11, 33, 33), dtype=np.uint8), np.zeros(6, dtype=np.float32))
    raw = []
    for step in range(8):
        events = (e.KILLED_OPPONENT,) if step == 4 else ()
        raw.append(safety_train.RawTransition(state, 0, 0.0, {}, state, False,
                                              np.ones(6, dtype=bool), 1, step, events))
    assert safety_train.aggregate_n_step(deque(raw), 8, 0.99).kill_chain is True
    raw[6].events = (e.KILLED_SELF,)
    item = safety_train.aggregate_n_step(deque(raw), 8, 0.99)
    assert item.kill_chain is False
    assert item.self_chain is True


def test_safety_train_reuses_exact_parent_reward_function():
    assert safety_train.reward_total is base_rewards.reward_total
    assert safety_train.OFFICIAL[e.KILLED_SELF] == -8.0
    assert safety_train.OFFICIAL[e.KILLED_OPPONENT] == 5.0


def test_decision_selects_supported_balanced_candidate_and_rejects_weak_signal():
    protocol, _, _ = load_protocol(PROTOCOL)
    passed = decide_task4(protocol, _task4_rows())
    assert passed["supported"] is True
    assert passed["selected_milestone"] == 100
    assert passed["selected_label"].startswith("balanced_safety_n8-")
    failed = decide_task4(protocol, _task4_rows(treatment_suicide=0.23))
    assert failed["supported"] is False
    assert failed["decision"] == "safety_replay_not_supported_stop"


def test_dry_run_starts_nothing_and_stops_before_confirmation():
    protocol, _, protocol_hash = load_protocol(PROTOCOL)
    result = dry_run(protocol, protocol_hash, {})
    assert result["maximum_environment_rounds"] == 3300
    assert result["formal_training_started"] is False
    assert result["formal_evaluation_started"] is False
    assert result["selection_started"] is False
    assert result["checkpoint_copy_allowed"] is False
    assert result["automatic_followup_started"] is False
