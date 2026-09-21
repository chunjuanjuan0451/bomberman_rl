"""Contracts for the safe-kill-only BOMB reward-500 signal gate."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

import events as e
from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.features import ACTIONS
from agent_code.model_a_v4_bomb_credit6.replay import ReplayBuffer
from agent_code.model_a_v4_bomb_safe500 import callbacks, train
from agent_code.model_a_v4_bomb_safe500.config import load_protocol
from tools.v4_task4_bomb_safe500 import decide, dry_run, registered_seeds


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task4-bomb-safe500-s133000.json"


def _raw(step: int, reward: float, events=(), *, done: bool = False):
    state = (np.full((2, 2), step, dtype=np.float32), np.full(3, step, dtype=np.float32))
    next_state = None if done else (
        np.full((2, 2), step + 1, dtype=np.float32),
        np.full(3, step + 1, dtype=np.float32),
    )
    return train.RawTransition(
        state=state,
        action=ACTIONS.index("BOMB") if step == 10 else ACTIONS.index("WAIT"),
        reward=reward,
        next_state=next_state,
        done=done,
        next_mask=None if done else np.ones(len(ACTIONS), dtype=bool),
        episode_id=1,
        start_step=step,
        events=tuple(events),
    )


def _diagnostics():
    return {
        "raw_transitions": 0,
        "non_bomb_one_step_targets": 0,
        "bomb_targets_matured": 0,
        "bomb_targets_full_six_step_observed": 0,
        "bomb_targets_terminal_truncated": 0,
        "bomb_target_return_steps_histogram": {str(i): 0 for i in range(1, 7)},
        "bomb_target_kill_events_included": 0,
        "bomb_target_self_events_included": 0,
        "observed_kill_events": 0,
        "observed_self_events": 0,
        "uniquely_attributed_kill_events": 0,
        "uniquely_attributed_self_events": 0,
        "outcome_lag_histogram": {"4": 0, "5": 0, "terminal_deferred": 0},
        "terminal_deferred_kill_events": 0,
        "attribution_errors": 0,
        "bomb_target_safe_kill_events": 0,
        "bomb_target_trade_kill_events": 0,
        "bomb_target_self_without_kill_events": 0,
        "bomb_target_suppressed_trade_reward_events": 0,
    }


def _agent(arm: str):
    return SimpleNamespace(arm=arm, replay_buffer=ReplayBuffer(), target_diagnostics=_diagnostics())


def _assert_nested_equal(left, right):
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    elif hasattr(left, "detach"):
        assert left.detach().equal(right.detach())
    else:
        assert left == right


def test_safe_kill_is_the_only_reward_scaled_to_500():
    window = [_raw(10 + i, 12.0 if i == 4 else 0.0,
                   (e.KILLED_OPPONENT,) if i == 4 else (), done=i == 5) for i in range(6)]
    control, candidate = _agent("safe12"), _agent("safe500")
    train._mature_bomb(control, window)
    train._mature_bomb(candidate, window)
    control_reward = control.replay_buffer._items[0].reward
    candidate_reward = candidate.replay_buffer._items[0].reward
    assert np.isclose(control_reward, (v4.GAMMA ** 4) * 12.0)
    assert np.isclose(candidate_reward, (v4.GAMMA ** 4) * 500.0)
    assert control.target_diagnostics["bomb_target_safe_kill_events"] == 1
    assert candidate.target_diagnostics["bomb_target_safe_kill_events"] == 1


def test_kill_plus_self_kill_gets_zero_kill_reward_in_both_arms():
    window = [
        _raw(10 + i, 12.0 if i == 4 else -40.0 if i == 5 else 0.0,
             (e.KILLED_OPPONENT,) if i == 4 else (e.KILLED_SELF, e.GOT_KILLED) if i == 5 else (),
             done=i == 5)
        for i in range(6)
    ]
    control, candidate = _agent("safe12"), _agent("safe500")
    train._mature_bomb(control, window)
    train._mature_bomb(candidate, window)
    expected = (v4.GAMMA ** 5) * -40.0
    assert np.isclose(control.replay_buffer._items[0].reward, expected)
    assert np.isclose(candidate.replay_buffer._items[0].reward, expected)
    for agent in (control, candidate):
        assert agent.target_diagnostics["bomb_target_trade_kill_events"] == 1
        assert agent.target_diagnostics["bomb_target_suppressed_trade_reward_events"] == 1
        assert agent.target_diagnostics["bomb_target_safe_kill_events"] == 0


def test_protocol_is_fresh_conditional_terminal_gate():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert protocol["single_training_variable"] == "safe_bomb_kill_reward_12_vs_500"
    assert len(registered_seeds(protocol)) == 21
    summary = dry_run(protocol)
    assert summary["snapshot_rounds"] == [5, 10, 15, 25]
    assert summary["total_training_rounds"] == 150
    assert summary["conditional_evaluation_rounds"] == 800
    assert summary["maximum_total_game_rounds"] == 950
    assert summary["trade_kill_reward_both_arms"] == 0.0
    assert summary["minimum_divergent_pairs_before_evaluation"] == 2
    assert summary["automatic_followup_started"] is False


def test_no_replicated_parameter_effect_stops_before_evaluation():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    effect = {
        "evaluation_authorized": False,
        "endpoint_divergent_pairs": 1,
        "pairs": {},
    }
    result = decide(protocol, {}, {}, effect)
    assert result["decision"] == "reward500_no_replicated_parameter_effect_stop"
    assert result["evaluation_started"] is False
    assert result["automatic_followup_started"] is False


def test_both_arms_restore_identical_parent_before_safe_kill_signal():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    base = {
        "MODEL_A_BOMB_SAFE500_PROTOCOL_PATH": str(PROTOCOL_PATH),
        "MODEL_A_BOMB_SAFE500_CHECKPOINT_PATH": "",
        "MODEL_A_BOMB_SAFE500_PARENT_PATH": str(ROOT / protocol["source_parent"]["path"]),
        "MODEL_A_BOMB_SAFE500_REPLICA": "r1",
        "MODEL_A_BOMB_SAFE500_SEED": str(protocol["training"]["seeds_by_replica"]["r1"]["agent_seed"]),
    }
    keys = (*base, "MODEL_A_BOMB_SAFE500_ARM")
    previous = {key: os.environ.get(key) for key in keys}
    original_final_checkpoint_path = callbacks.final_checkpoint_path
    agents = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            callbacks.final_checkpoint_path = lambda _protocol, arm, replica: root / arm / replica / "final.pt"
            for arm in ("safe12", "safe500"):
                values = dict(base)
                values["MODEL_A_BOMB_SAFE500_ARM"] = arm
                values["MODEL_A_BOMB_SAFE500_CHECKPOINT_PATH"] = str(root / arm / "r1" / "final.pt")
                os.environ.update(values)
                agent = SimpleNamespace(train=True, logger=logging.getLogger(f"bomb-safe500-{arm}"))
                callbacks.setup(agent)
                train.setup_training(agent)
                agents[arm] = agent
            for left, right in zip(agents["safe12"].online_net.parameters(), agents["safe500"].online_net.parameters()):
                assert left.detach().equal(right.detach())
            _assert_nested_equal(agents["safe12"].optimizer.state_dict(), agents["safe500"].optimizer.state_dict())
            assert len(agents["safe12"].replay_buffer) == len(agents["safe500"].replay_buffer) == 0
            assert abs(agents["safe12"].epsilon - 0.05) < 1e-12
            assert abs(agents["safe500"].epsilon - 0.05) < 1e-12
    finally:
        callbacks.final_checkpoint_path = original_final_checkpoint_path
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
