"""Tests for the passive Phase A / offline-only Phase B audit."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import events as e
from agent_code.model_a_dqn.features import ACTIONS
from agent_code.model_a_v4_system_audit.config import EVENT_NAMES, load_protocol
from agent_code.model_a_v4_system_audit import train as collector
from tools import v4_training_system_audit as audit


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-v4-training-system-audit-s135000.json"


def _payload(rows: int = 6) -> dict[str, np.ndarray]:
    events = np.zeros((rows, len(EVENT_NAMES)), dtype=np.int16)
    rewards = np.zeros_like(events, dtype=np.float32)
    legal = np.ones((rows, len(ACTIONS)), dtype=np.bool_)
    actions = np.full(rows, ACTIONS.index("WAIT"), dtype=np.int8)
    actions[0] = ACTIONS.index("BOMB")
    events[0, EVENT_NAMES.index("BOMB_DROPPED")] = 1
    rewards[0, EVENT_NAMES.index("BOMB_DROPPED")] = -0.05
    if rows > 4:
        events[4, EVENT_NAMES.index("KILLED_OPPONENT")] = 1
        rewards[4, EVENT_NAMES.index("KILLED_OPPONENT")] = 12.0
    return {
        "round": np.ones(rows, dtype=np.int16),
        "step": np.arange(1, rows + 1, dtype=np.int16),
        "action": actions,
        "legal_mask": legal,
        "online_q_values": np.zeros((rows, len(ACTIONS)), dtype=np.float32),
        "target_q_values": np.zeros((rows, len(ACTIONS)), dtype=np.float32),
        "global_features": np.zeros((rows, 7), dtype=np.float32),
        "opponent_count": np.ones(rows, dtype=np.int8),
        "nearest_opponent_distance": np.full(rows, 3.0, dtype=np.float32),
        "next_nearest_opponent_distance": np.full(rows, 3.0, dtype=np.float32),
        "approach_step": np.zeros(rows, dtype=np.bool_),
        "bomb_legal": np.ones(rows, dtype=np.bool_),
        "oracle_evaluated": np.ones(rows, dtype=np.bool_),
        "oracle_timeout": np.zeros(rows, dtype=np.bool_),
        "guaranteed_traps": np.zeros(rows, dtype=np.int8),
        "affected_opponents": np.asarray([1, 0, 0, 0, 0, 0], dtype=np.int8),
        "max_space_reduction": np.asarray([0.7, 0, 0, 0, 0, 0], dtype=np.float32),
        "own_bottleneck": np.zeros(rows, dtype=np.int16),
        "own_terminal_positions": np.zeros(rows, dtype=np.int16),
        "event_counts": events,
        "event_reward_components": rewards,
        "state_shaping": np.zeros(rows, dtype=np.float32),
        "total_reward": rewards.sum(axis=1),
    }


def test_protocol_is_passive_and_terminal():
    protocol, _ = load_protocol(PROTOCOL)
    assert protocol["training_enabled"] is False
    assert protocol["automatic_followup"] is False
    assert protocol["decision"] == "diagnostic_complete_stop_before_any_training"
    assert audit.dry_run(protocol)["optimizer_created"] is False
    assert audit.dry_run(protocol)["phase_a"]["total_rounds"] == 200


def test_horizon_boundary_credits_bomb_only_after_lag_four():
    payload = _payload()
    n1 = audit.horizon_audit([payload], 1, 0.99, 64)
    n4 = audit.horizon_audit([payload], 4, 0.99, 64)
    n8 = audit.horizon_audit([payload], 8, 0.99, 64)
    assert n1["kill_bearing_targets"] == 1
    assert n4["kill_bearing_targets"] == 4
    assert n8["kill_bearing_targets"] == 5
    assert n4["linked_bomb_kill_coverage"] == 0.0
    assert n8["linked_bomb_kill_coverage"] == 1.0
    assert n8["kill_target_start_stages"]["bomb"] == 1


def test_reward_ledger_separates_kill_self_and_bomb():
    payload = _payload()
    payload["event_counts"][5, EVENT_NAMES.index("KILLED_SELF")] = 1
    payload["event_reward_components"][5, EVENT_NAMES.index("KILLED_SELF")] = -20.0
    payload["total_reward"] = payload["event_reward_components"].sum(axis=1)
    ledger = audit.reward_ledger([payload])["by_official_event"]
    assert ledger["KILLED_OPPONENT"]["signed_contribution"] == 12.0
    assert ledger["KILLED_SELF"]["signed_contribution"] == -20.0
    assert np.isclose(ledger["BOMB_DROPPED"]["signed_contribution"], -0.05)


def _trace_lists(payload: dict[str, np.ndarray]) -> dict[str, list]:
    result = {}
    for field in collector.TRACE_FIELDS:
        value = payload[field][0]
        result[field] = [value.copy() if isinstance(value, np.ndarray) else value.item() if hasattr(value, "item") else value]
    return result


def test_max_step_survivor_merges_survival_without_duplicating_final_events():
    payload = _payload(1)
    agent = SimpleNamespace(
        _trace=_trace_lists(payload), _pending_decision=None,
        _completed_rounds=0, _expected_rounds=2,
    )
    collector.end_of_round(
        agent, {"round": 1, "step": 1}, "BOMB",
        [e.BOMB_DROPPED, e.SURVIVED_ROUND],
    )
    counts = agent._trace["event_counts"][-1]
    assert counts[EVENT_NAMES.index("BOMB_DROPPED")] == 1
    assert counts[EVENT_NAMES.index("SURVIVED_ROUND")] == 1
    assert np.isclose(agent._trace["total_reward"][-1], 2.95)
    assert len(agent._trace["action"]) == 1


def test_dead_agent_records_pending_terminal_transition_once():
    pending = {
        "round": 1, "step": 7, "action": ACTIONS.index("WAIT"),
        "legal_mask": np.ones(len(ACTIONS), dtype=np.bool_),
        "online_q_values": np.zeros(len(ACTIONS), dtype=np.float32),
        "target_q_values": np.zeros(len(ACTIONS), dtype=np.float32),
        "global_features": np.zeros(7, dtype=np.float32),
        "opponent_count": 1, "nearest_opponent_distance": 2.0,
        "bomb_legal": False, "oracle_evaluated": False, "oracle_timeout": False,
        "guaranteed_traps": 0, "affected_opponents": 0, "max_space_reduction": 0.0,
        "own_bottleneck": 0, "own_terminal_positions": 0,
    }
    agent = SimpleNamespace(
        _trace={field: [] for field in collector.TRACE_FIELDS},
        _pending_decision=pending, _seen_transition_keys=set(),
        _completed_rounds=0, _expected_rounds=2,
    )
    state = {"round": 1, "step": 7, "self": ("me", 0, False, (1, 1))}
    collector.end_of_round(agent, state, "WAIT", [e.GOT_KILLED])
    assert len(agent._trace["action"]) == 1
    assert agent._trace["event_counts"][0][EVENT_NAMES.index("GOT_KILLED")] == 1
    assert agent._trace["total_reward"][0] == -20.0
    assert agent._pending_decision is None


def test_collector_and_runner_do_not_construct_optimizer_or_override_action():
    paths = [
        ROOT / "agent_code/model_a_v4_system_audit/callbacks.py",
        ROOT / "agent_code/model_a_v4_system_audit/train.py",
        ROOT / "tools/v4_training_system_audit.py",
    ]
    identifiers = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        identifiers.update(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))
        identifiers.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
    assert "optimizer" not in identifiers
    assert "backward" not in identifiers
