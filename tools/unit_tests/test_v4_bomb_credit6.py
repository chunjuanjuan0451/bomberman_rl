"""Contracts for the corrected six-transition BOMB-credit experiment."""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

import events as e
import settings as s
from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.features import ACTIONS
from agent_code.model_a_v4_bomb_credit6 import callbacks, train
from agent_code.model_a_v4_bomb_credit6.config import load_protocol
from agent_code.model_a_v4_bomb_credit6.replay import ReplayBuffer
from tools.v4_bomb_credit_lifecycle_audit import build_report
from tools.v4_task4_bomb_credit6 import dry_run, registered_seeds


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task4-bomb-credit6-s131000.json"


def _raw(step: int, reward: float, events=(), *, done: bool = False, episode: int = 1):
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
        episode_id=episode,
        start_step=step,
        events=tuple(events),
    )


def _game_state(step: int) -> dict:
    field = -np.ones((17, 17), dtype=int)
    field[1:16, 1:16] = 0
    return {
        "round": 1,
        "step": step,
        "field": field,
        "self": ("me", 0, True, (8, 8)),
        "others": [],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
        "user_input": None,
    }


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
    }


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


def test_lifecycle_audit_freezes_the_real_lag_five_failure_without_new_games():
    # The live game log is intentionally mutable and changes after later
    # experiments.  Once the lifecycle audit is frozen, test that immutable
    # report rather than trying to reconstruct it from the current log tail.
    report = json.loads((ROOT / "experiments/logs/diagnostics/model-a-v4-bomb-credit-lifecycle-s131000.json").read_text(encoding="utf-8"))
    assert report["failure"]["bomb_step"] == 58
    assert report["failure"]["explosion_and_kill_step"] == 62
    assert report["failure"]["lingering_self_kill_step"] == 63
    assert report["environment_semantics"]["allowed_official_outcome_lags"] == [4, 5]
    assert report["environment_semantics"]["minimum_complete_return_horizon"] == 6
    assert report["new_game_rounds"] == report["policy_updates"] == 0
    assert s.BOMB_TIMER == 4 and s.EXPLOSION_TIMER == 2


def test_six_step_return_includes_explosion_and_lingering_flame_rewards():
    window = [
        _raw(10 + i, float(i + 1),
             (e.KILLED_OPPONENT,) if i == 4 else (e.KILLED_SELF,) if i == 5 else (),
             done=i == 5)
        for i in range(6)
    ]
    control = train.aggregate_return(window, 1, v4.GAMMA)
    credit = train.aggregate_return(window, 6, v4.GAMMA)
    assert control.reward == 1.0 and control.return_steps == 1
    assert np.isclose(control.bootstrap_discount, v4.GAMMA)
    assert np.isclose(credit.reward, sum((v4.GAMMA ** i) * (i + 1) for i in range(6)))
    assert np.isclose(credit.bootstrap_discount, v4.GAMMA ** 6)
    assert credit.return_steps == 6 and credit.done is True


def test_attribution_accepts_lag_four_and_five_but_rejects_other_lags():
    agent = SimpleNamespace(_pending_bombs=[[_raw(10, 0.0)]], target_diagnostics=_diagnostics())
    train._assert_outcome_attribution(agent, _raw(14, 1.0, (e.KILLED_OPPONENT,)))
    train._assert_outcome_attribution(agent, _raw(15, -1.0, (e.KILLED_SELF,), done=True))
    assert agent.target_diagnostics["outcome_lag_histogram"] == {"4": 1, "5": 1, "terminal_deferred": 0}
    assert agent.target_diagnostics["uniquely_attributed_kill_events"] == 1
    assert agent.target_diagnostics["uniquely_attributed_self_events"] == 1
    try:
        train._assert_outcome_attribution(agent, _raw(16, 0.0, (e.KILLED_SELF,)))
    except RuntimeError as exc:
        assert "lag-4/5" in str(exc)
    else:
        raise AssertionError("lag-six outcome attribution was accepted")


def test_credit_matures_after_six_and_includes_both_official_outcomes():
    window = [
        _raw(10 + i, 0.0,
             (e.KILLED_OPPONENT,) if i == 4 else (e.KILLED_SELF,) if i == 5 else (),
             done=i == 5)
        for i in range(6)
    ]
    control = SimpleNamespace(arm="control", replay_buffer=ReplayBuffer(), target_diagnostics=_diagnostics())
    credit = SimpleNamespace(arm="credit", replay_buffer=ReplayBuffer(), target_diagnostics=_diagnostics())
    train._mature_bomb(control, window)
    train._mature_bomb(credit, window)
    assert control.replay_buffer._items[0].return_steps == 1
    assert control.target_diagnostics["bomb_target_kill_events_included"] == 0
    assert control.target_diagnostics["bomb_target_self_events_included"] == 0
    assert credit.replay_buffer._items[0].return_steps == 6
    assert credit.target_diagnostics["bomb_target_kill_events_included"] == 1
    assert credit.target_diagnostics["bomb_target_self_events_included"] == 1


def test_record_replays_the_s130_lifecycle_without_losing_end_of_round_self_kill():
    agent = SimpleNamespace(
        arm="credit",
        replay_buffer=ReplayBuffer(),
        target_diagnostics=_diagnostics(),
        _pending_bombs=[],
        training_steps=0,
        epsilon=0.05,
    )
    sequence = (
        (10, "BOMB", 0.0, (e.BOMB_DROPPED,), False),
        (11, "WAIT", 0.0, (), False),
        (12, "WAIT", 0.0, (), False),
        (13, "WAIT", 0.0, (), False),
        (14, "WAIT", 12.0, (e.KILLED_OPPONENT,), False),
        (15, "LEFT", -20.0, (e.KILLED_SELF, e.GOT_KILLED), True),
    )
    for step, action, reward, events, done in sequence:
        train._record(
            agent, _game_state(step), action,
            None if done else _game_state(step + 1), reward, done, events,
        )
    assert not agent._pending_bombs
    assert len(agent.replay_buffer) == 6
    assert agent.target_diagnostics["outcome_lag_histogram"] == {"4": 1, "5": 1, "terminal_deferred": 0}
    assert agent.target_diagnostics["bomb_target_kill_events_included"] == 1
    assert agent.target_diagnostics["bomb_target_self_events_included"] == 1


def test_posthumous_owned_bomb_kill_is_attached_to_terminal_transition():
    agent = SimpleNamespace(
        arm="credit", replay_buffer=ReplayBuffer(), target_diagnostics=_diagnostics(),
        _pending_bombs=[], training_steps=0, epsilon=0.05,
    )
    # The agent drops a bomb and is killed two steps later.  The official
    # environment may keep its bomb alive, add KILLED_OPPONENT later, and only
    # deliver that event at end_of_round on this last transition.
    sequence = (
        (10, "BOMB", 0.0, (e.BOMB_DROPPED,), False),
        (11, "WAIT", 0.0, (), False),
        (12, "WAIT", 12.0, (e.GOT_KILLED, e.KILLED_OPPONENT), True),
    )
    for step, action, reward, events, done in sequence:
        train._record(
            agent, _game_state(step), action,
            None if done else _game_state(step + 1), reward, done, events,
        )
    assert not agent._pending_bombs
    assert agent.target_diagnostics["terminal_deferred_kill_events"] == 1
    assert agent.target_diagnostics["outcome_lag_histogram"]["terminal_deferred"] == 1
    assert agent.target_diagnostics["bomb_target_kill_events_included"] == 1


def test_terminal_before_lingering_frame_truncates_without_crossing_episode():
    window = [_raw(10 + i, float(i), done=i == 4) for i in range(5)]
    result = train.aggregate_return(window, 6, v4.GAMMA)
    assert result.done is True and result.return_steps == 5 and result.next_state is None
    try:
        train.aggregate_return([window[0], _raw(11, 0.0, episode=2)], 6, v4.GAMMA)
    except ValueError as exc:
        assert "cross episodes" in str(exc)
    else:
        raise AssertionError("cross-episode return was accepted")


def test_protocol_uses_fresh_s131_seeds_and_stops_after_fixed_endpoint():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert protocol["single_training_variable"] == "bomb_transition_return_horizon_1_vs_6"
    assert len(registered_seeds(protocol)) == 33
    summary = dry_run(protocol)
    assert summary["total_training_rounds"] == 600
    assert summary["evaluation_rounds"] == 1600
    assert summary["maximum_total_game_rounds"] == 2200
    assert summary["allowed_outcome_lags"] == [4, 5]
    assert summary["failed_s130000_reused"] is False
    assert summary["task4b_started"] is False
    assert summary["distillation_started"] is False


def test_both_corrected_arms_restore_identical_parent_before_bomb_maturation():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    base = {
        "MODEL_A_BOMB_CREDIT6_PROTOCOL_PATH": str(PROTOCOL_PATH),
        "MODEL_A_BOMB_CREDIT6_CHECKPOINT_PATH": "",
        "MODEL_A_BOMB_CREDIT6_PARENT_PATH": str(ROOT / protocol["source_parent"]["path"]),
        "MODEL_A_BOMB_CREDIT6_REPLICA": "r1",
        "MODEL_A_BOMB_CREDIT6_SEED": str(protocol["training"]["seeds_by_replica"]["r1"]["agent_seed"]),
    }
    previous = {key: os.environ.get(key) for key in (*base, "MODEL_A_BOMB_CREDIT6_ARM")}
    original_final_checkpoint_path = callbacks.final_checkpoint_path
    agents = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            callbacks.final_checkpoint_path = lambda _protocol, arm, replica: root / arm / replica / "final.pt"
            for arm in ("control", "credit"):
                values = dict(base)
                values["MODEL_A_BOMB_CREDIT6_ARM"] = arm
                values["MODEL_A_BOMB_CREDIT6_CHECKPOINT_PATH"] = str(root / arm / "r1" / "final.pt")
                os.environ.update(values)
                agent = SimpleNamespace(train=True, logger=logging.getLogger(f"bomb-credit6-{arm}"))
                callbacks.setup(agent)
                train.setup_training(agent)
                agents[arm] = agent
            for left, right in zip(agents["control"].online_net.parameters(), agents["credit"].online_net.parameters()):
                assert left.detach().equal(right.detach())
            _assert_nested_equal(agents["control"].optimizer.state_dict(), agents["credit"].optimizer.state_dict())
            assert len(agents["control"].replay_buffer) == len(agents["credit"].replay_buffer) == 0
            assert agents["control"].epsilon == agents["credit"].epsilon
    finally:
        callbacks.final_checkpoint_path = original_final_checkpoint_path
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
