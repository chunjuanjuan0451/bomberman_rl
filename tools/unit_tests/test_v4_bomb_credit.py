"""Contracts for the source-r2 BOMB temporal-credit experiment."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import events as e
import settings as s
from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.features import ACTIONS
from agent_code.model_a_v4_bomb_credit import callbacks, train
from agent_code.model_a_v4_bomb_credit.config import load_protocol
from agent_code.model_a_v4_bomb_credit.replay import ReplayBuffer
from tools.v4_bomb_credit_attribution_audit import attribute_events
from tools.v4_task4_bomb_credit import decide, dry_run, registered_seeds


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task4-bomb-credit-s130000.json"


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


def _diagnostics():
    return {
        "raw_transitions": 0,
        "non_bomb_one_step_targets": 0,
        "bomb_targets_matured": 0,
        "bomb_targets_full_five_step_observed": 0,
        "bomb_targets_terminal_truncated": 0,
        "bomb_target_return_steps_histogram": {str(i): 0 for i in range(1, 6)},
        "bomb_target_kill_events_included": 0,
        "bomb_target_self_events_included": 0,
        "observed_kill_events": 0,
        "observed_self_events": 0,
        "uniquely_attributed_kill_events": 0,
        "uniquely_attributed_self_events": 0,
        "attribution_errors": 0,
    }


def test_stage0_audit_and_environment_fix_the_five_transition_horizon():
    report = attribute_events(
        np.asarray([1, 1, 1, 1, 1]),
        np.asarray([10, 11, 12, 13, 14]),
        np.asarray([ACTIONS.index("BOMB"), 0, 0, 0, 0]),
        np.asarray([False, False, False, False, True]),
    )
    assert s.BOMB_TIMER == 4
    match = report["matches"][0]
    assert match["event_step"] - match["expected_bomb_step"] == 4
    assert report["unique_matches"] == report["events"] == 1
    # Five rewards r_s..r_(s+4) are required to include an event four step
    # indices after the causative action.
    assert train.BOMB_MATURATION_STEPS == 5


def test_control_and_credit_differ_only_in_bomb_return_horizon():
    window = [_raw(10 + i, float(i + 1), (e.KILLED_OPPONENT,) if i == 4 else ()) for i in range(5)]
    control = train.aggregate_return(window, 1, v4.GAMMA)
    credit = train.aggregate_return(window, 5, v4.GAMMA)
    assert control.reward == 1.0
    assert np.isclose(control.bootstrap_discount, v4.GAMMA)
    assert control.return_steps == 1
    assert np.isclose(credit.reward, sum((v4.GAMMA ** i) * (i + 1) for i in range(5)))
    assert np.isclose(credit.bootstrap_discount, v4.GAMMA ** 5)
    assert credit.return_steps == 5
    assert np.array_equal(control.state[0], credit.state[0])
    assert control.action == credit.action == ACTIONS.index("BOMB")


def test_terminal_bomb_return_truncates_and_never_crosses_episode():
    window = [_raw(10, 1.0), _raw(11, 2.0), _raw(12, 3.0, done=True)]
    result = train.aggregate_return(window, 5, v4.GAMMA)
    assert result.done is True
    assert result.next_state is None
    assert result.return_steps == 3
    assert np.isclose(result.bootstrap_discount, v4.GAMMA ** 3)
    try:
        train.aggregate_return([window[0], _raw(11, 2.0, episode=2)], 5, v4.GAMMA)
    except ValueError as exc:
        assert "cross episodes" in str(exc)
    else:
        raise AssertionError("cross-episode return was accepted")


def test_official_kill_is_uniquely_attributed_to_step_minus_four_bomb():
    pending = [[_raw(10, 0.0)]]
    agent = SimpleNamespace(_pending_bombs=pending, target_diagnostics=_diagnostics())
    outcome = _raw(14, 1.0, (e.KILLED_OPPONENT,))
    train._assert_outcome_attribution(agent, outcome)
    assert agent.target_diagnostics["observed_kill_events"] == 1
    assert agent.target_diagnostics["uniquely_attributed_kill_events"] == 1

    wrong_episode = _raw(14, 1.0, (e.KILLED_SELF,), episode=2)
    try:
        train._assert_outcome_attribution(agent, wrong_episode)
    except RuntimeError as exc:
        assert "did not uniquely match" in str(exc)
    else:
        raise AssertionError("wrong-episode outcome attribution was accepted")


def test_matured_control_excludes_and_credit_includes_delayed_official_events():
    window = [_raw(10 + i, 0.0, (e.KILLED_OPPONENT, e.KILLED_OPPONENT, e.KILLED_SELF) if i == 4 else ()) for i in range(5)]
    agents = {}
    for arm in ("control", "credit"):
        agents[arm] = SimpleNamespace(
            arm=arm,
            replay_buffer=ReplayBuffer(),
            target_diagnostics=_diagnostics(),
        )
        train._mature_bomb(agents[arm], window)
    control, credit = agents["control"], agents["credit"]
    assert control.replay_buffer._items[0].return_steps == 1
    assert control.target_diagnostics["bomb_target_kill_events_included"] == 0
    assert control.target_diagnostics["bomb_target_self_events_included"] == 0
    assert credit.replay_buffer._items[0].return_steps == 5
    assert credit.target_diagnostics["bomb_target_kill_events_included"] == 2
    assert credit.target_diagnostics["bomb_target_self_events_included"] == 1


def test_non_bomb_transition_is_field_identical_across_arms():
    raw = _raw(11, 2.5)
    left = train.aggregate_return([raw], 1, v4.GAMMA)
    right = train.aggregate_return([copy.deepcopy(raw)], 1, v4.GAMMA)
    assert left.action == right.action
    assert left.reward == right.reward
    assert left.done == right.done
    assert left.bootstrap_discount == right.bootstrap_discount
    assert left.return_steps == right.return_steps == 1
    assert left.episode_id == right.episode_id
    assert left.start_step == right.start_step
    assert np.array_equal(left.state[0], right.state[0])
    assert np.array_equal(left.state[1], right.state[1])
    assert np.array_equal(left.next_state[0], right.next_state[0])
    assert np.array_equal(left.next_state[1], right.next_state[1])
    assert np.array_equal(left.next_mask, right.next_mask)


def test_protocol_is_frozen_single_variable_and_dry_run_stops():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert protocol["single_training_variable"] == "bomb_transition_return_horizon_1_vs_5"
    assert protocol["training"]["rounds_per_arm"] == 100
    assert len(registered_seeds(protocol)) == 33
    summary = dry_run(protocol)
    assert summary["total_training_rounds"] == 600
    assert summary["evaluation_rounds"] == 1600
    assert summary["maximum_total_game_rounds"] == 2200
    assert summary["formal_training_started"] is False
    assert summary["task4b_started"] is False
    assert summary["distillation_started"] is False
    assert summary["writes_terminal_report_and_stops"] is True


def test_decision_requires_replicated_external_gain_and_real_credit_arm_kills():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    rows = {}
    base = {
        "score_per_round": 4.0,
        "kills_per_round": 0.1,
        "suicides_per_round": 0.1,
        "invalid_actions_per_round": 0.0,
    }
    for label in ("v4", "source-r2", *[f"{arm}-{replica}" for arm in ("control", "credit") for replica in ("r1", "r2", "r3")]):
        rows[label] = {stratum: {"target": dict(base)} for stratum in protocol["evaluation"]["strata"]}
    for replica in ("r1", "r2", "r3"):
        rows[f"control-{replica}"]["task4c_three_rule"]["target"].update(
            score_per_round=2.0, kills_per_round=0.1, suicides_per_round=0.1,
        )
        rows[f"credit-{replica}"]["task4c_three_rule"]["target"].update(
            score_per_round=2.0, kills_per_round=0.2, suicides_per_round=0.1,
        )
    rows["source-r2"]["task4c_three_rule"]["target"].update(
        score_per_round=2.0, kills_per_round=0.1, suicides_per_round=0.1,
    )
    training = {arm: {} for arm in ("control", "credit")}
    for arm in training:
        for replica in ("r1", "r2", "r3"):
            diagnostics = _diagnostics()
            diagnostics["observed_kill_events"] = 1
            diagnostics["uniquely_attributed_kill_events"] = 1
            if arm == "credit":
                diagnostics["bomb_target_kill_events_included"] = 1
            training[arm][replica] = {"snapshots": [{"target_diagnostics": diagnostics}]}
    result = decide(protocol, rows, training)
    assert result["passed"] is True
    assert result["supportive_pairs"] == 3
    assert result["task4b_started"] is False

    no_kill = training["credit"]["r1"]["snapshots"][-1]["target_diagnostics"]
    no_kill["observed_kill_events"] = 0
    no_kill["uniquely_attributed_kill_events"] = 0
    no_kill["bomb_target_kill_events_included"] = 0
    assert decide(protocol, rows, training)["passed"] is False


def test_both_arms_restore_identical_parent_before_first_matured_bomb():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    base = {
        "MODEL_A_BOMB_CREDIT_PROTOCOL_PATH": str(PROTOCOL_PATH),
        "MODEL_A_BOMB_CREDIT_CHECKPOINT_PATH": "",
        "MODEL_A_BOMB_CREDIT_PARENT_PATH": str(ROOT / protocol["source_parent"]["path"]),
        # s130000 control/r1 is now preserved as a failed formal artifact.
        # Use untouched r2 only to verify read-only setup semantics.
        "MODEL_A_BOMB_CREDIT_REPLICA": "r2",
        "MODEL_A_BOMB_CREDIT_SEED": str(protocol["training"]["seeds_by_replica"]["r2"]["agent_seed"]),
    }
    agents = {}
    previous = {key: os.environ.get(key) for key in (*base, "MODEL_A_BOMB_CREDIT_ARM")}
    try:
        for arm in ("control", "credit"):
            values = dict(base)
            values["MODEL_A_BOMB_CREDIT_ARM"] = arm
            values["MODEL_A_BOMB_CREDIT_CHECKPOINT_PATH"] = str(
                ROOT / protocol["checkpoint_directory"] / arm / "r2" / "final.pt"
            )
            os.environ.update(values)
            agent = SimpleNamespace(train=True, logger=logging.getLogger(f"bomb-credit-{arm}"))
            callbacks.setup(agent)
            train.setup_training(agent)
            agents[arm] = agent
        for left, right in zip(agents["control"].online_net.parameters(), agents["credit"].online_net.parameters()):
            assert left.detach().equal(right.detach())
        _assert_nested_equal(
            agents["control"].optimizer.state_dict(),
            agents["credit"].optimizer.state_dict(),
        )
        assert len(agents["control"].replay_buffer) == len(agents["credit"].replay_buffer) == 0
        assert np.isclose(agents["control"].epsilon, 0.05)
        assert agents["control"].epsilon == agents["credit"].epsilon
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
