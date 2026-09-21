"""Contracts for Phase 0/1 source-r2 counterfactual signal audit."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from agent_code.model_a_dqn.features import ACTIONS
from agent_code.model_a_v4_causal_signal import callbacks
from agent_code.model_a_v4_causal_signal.config import CASES, STRATA, load_protocol
from agent_code.model_a_v7b.tactical import evaluate_bomb
from tools.v4_task4_counterfactual_signal import attempt_outcomes, dry_run, summarize_payload


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-counterfactual-signal-s129000.json"


def _state(trapped: bool = True) -> dict:
    field = -np.ones((9, 9), dtype=int)
    field[1:8, 1:8] = 0
    if trapped:
        for x in (5, 6, 7):
            field[x, 3] = -1
            field[x, 5] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("me", 0, True, (4, 4)),
        "others": [("other", 0, True, (7, 4))],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
        "user_input": None,
    }


def test_counterfactual_signal_protocol_is_passive_and_uses_two_strata():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert protocol["scope"] == "phase0_oracle_validation_and_phase1_passive_signal_only"
    assert tuple(protocol["collection"]["cases"]) == CASES
    assert tuple(protocol["collection"]["strata"]) == STRATA
    assert protocol["collection"]["policy_updates"] == 0
    assert protocol["phase2_training_enabled"] is False
    assert protocol["automatic_followup"] is False
    seeds = [value for case in protocol["collection"]["cases"].values() for value in case.values()]
    assert len(seeds) == 24 and len(set(seeds)) == 24


def test_counterfactual_signal_dry_run_stops_before_replay_training():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["parent"] == "source-r2"
    assert summary["architecture"] == "exact-v4"
    assert summary["total_collection_rounds"] == 400
    assert summary["actions_overridden"] == 0
    assert summary["policy_updates"] == 0
    assert summary["phase2_training_started"] is False


def test_phase0_oracle_requires_new_causal_trap_and_robust_own_escape():
    trapped = evaluate_bomb(_state(True), perf_counter() + 1.0)
    assert callbacks.strict_opportunity(trapped)

    open_arena = evaluate_bomb(_state(False), perf_counter() + 1.0)
    assert not callbacks.strict_opportunity(open_arena)

    already_doomed = _state(True)
    # Model an already lethal observed explosion over the opponent's only two
    # reachable cells.  The counterfactual must not credit our new bomb when
    # the no-new-bomb frontier is already empty.
    already_doomed["explosion_map"][7, 4] = 1
    already_doomed["explosion_map"][6, 4] = 1
    doomed_tactics = evaluate_bomb(already_doomed, perf_counter() + 1.0)
    assert doomed_tactics is not None
    assert doomed_tactics.guaranteed_traps == 0
    assert not callbacks.strict_opportunity(doomed_tactics)


def test_attempt_outcomes_use_same_episode_and_separate_kill_and_self_windows():
    rounds = np.asarray([1, 1, 1, 1, 2, 2, 2])
    steps = np.asarray([1, 2, 6, 8, 1, 2, 3])
    strict = np.asarray([True, False, False, False, True, False, False])
    actions = np.asarray([5, 0, 0, 0, 5, 0, 0])
    kills = np.asarray([False, False, True, False, False, False, False])
    self_events = np.asarray([False, False, False, True, False, False, True])
    outcomes = attempt_outcomes(rounds, steps, strict, actions, kills, self_events, 6, 8)
    assert len(outcomes) == 2
    assert outcomes[0]["kill_realized"] is True
    assert outcomes[0]["self_risk"] is True
    assert outcomes[0]["successful_causal_segment"] is False
    assert outcomes[1]["kill_realized"] is False
    assert outcomes[1]["self_risk"] is True


def test_summary_distinguishes_attempts_misses_and_successes():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    payload = {
        "round": np.asarray([1, 1, 2, 2]),
        "step": np.asarray([1, 5, 1, 5]),
        "action": np.asarray([5, 0, 0, 0]),
        "bomb_legal": np.asarray([True, False, True, False]),
        "oracle_evaluated": np.asarray([True, False, True, False]),
        "oracle_timeout": np.zeros(4, dtype=bool),
        "strict_opportunity": np.asarray([True, False, True, False]),
        "kill_event": np.asarray([False, True, False, False]),
        "self_event": np.zeros(4, dtype=bool),
    }
    summary = summarize_payload(protocol, payload)
    assert summary["strict_opportunities"] == 2
    assert summary["attempts"] == 1
    assert summary["missed_opportunities"] == 1
    assert summary["successful_causal_segments"] == 1
    assert summary["kill_realization_rate"] == 1.0


def test_passive_collector_loads_frozen_source_r2_without_output_checkpoint():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    case = "duel_c1"
    values = {
        "MODEL_A_CAUSAL_SIGNAL_PROTOCOL_PATH": str(PROTOCOL_PATH),
        "MODEL_A_CAUSAL_SIGNAL_PARENT_PATH": str(ROOT / protocol["source_parent"]["path"]),
        "MODEL_A_CAUSAL_SIGNAL_CASE": case,
        "MODEL_A_CAUSAL_SIGNAL_SEED": str(protocol["collection"]["cases"][case]["agent_seed"]),
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        expected_trace = ROOT / protocol["trace_directory"] / f"{case}.npz"
        trace_existed_before = expected_trace.exists()
        agent = SimpleNamespace(train=True, logger=logging.getLogger("causal-signal-test"))
        callbacks.setup(agent)
        assert all(not parameter.requires_grad for parameter in agent.online_net.parameters())
        assert agent.parent_sha256 == protocol["source_parent"]["sha256"]
        # Setup is read-only.  The formal trace may already exist after the
        # completed s129000 run, so assert that setup did not create/remove it.
        assert agent.trace_path == expected_trace.resolve()
        assert agent.trace_path.exists() == trace_existed_before
        assert not hasattr(agent, "optimizer")
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value
