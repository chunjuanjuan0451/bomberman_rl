"""Contracts for the n=8 replay-sampling-only matched A/B."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

import events as e
from agent_code.model_a_dqn.features import ACTIONS
from agent_code.model_a_v4_sampling_ab import callbacks, train
from agent_code.model_a_v4_sampling_ab.config import ARMS, REPLICAS, load_protocol
from agent_code.model_a_v4_sampling_ab.replay import ReplayBuffer, Transition
from tools.v4_sampling_distribution_ab import decide, dry_run, registered_seeds


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-v4-sampling-distribution-ab-s136000.json"


def _raw(step: int, reward: float = 0.0, events=(), *, done: bool = False, episode: int = 1):
    state = (np.full((2, 2), step, dtype=np.float32), np.full(3, step, dtype=np.float32))
    next_state = None if done else (
        np.full((2, 2), step + 1, dtype=np.float32),
        np.full(3, step + 1, dtype=np.float32),
    )
    return train.RawTransition(
        state=state, action=ACTIONS.index("BOMB") if step == 1 else ACTIONS.index("WAIT"),
        reward=reward, state_shaping=0.0, next_state=next_state, done=done,
        next_mask=None if done else np.ones(len(ACTIONS), dtype=bool),
        episode_id=episode, start_step=step, events=tuple(events),
    )


def _transition(index: int, *, kill: bool = False, self_kill: bool = False) -> Transition:
    return Transition(
        state=index, action=0, reward=float(index), next_state=index + 1, done=False,
        next_mask=None, bootstrap_discount=0.99, return_steps=8,
        episode_id=1, start_step=index, kill_chain=kill, self_chain=self_kill,
    )


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


def _queue_agent():
    return SimpleNamespace(
        arm="uniform_n8", replay_buffer=ReplayBuffer(100),
        training_rng=np.random.default_rng(1), training_steps=480_870, epsilon=0.05,
        _episode_raw=[],
        training_diagnostics={
            "raw_transitions": 0, "matured_targets": 0,
            "return_steps_histogram": {str(i): 0 for i in range(1, 9)},
            "kill_chain_targets_created": 0, "self_chain_targets_created": 0,
            "observed_kill_events": 0, "observed_self_events": 0,
            "survivor_terminal_merges": 0, "dead_terminal_appends": 0,
            "gradient_updates": 0, "sampled_batches": 0,
            "requested_kill_slots": 0, "fulfilled_kill_slots": 0,
            "requested_self_slots": 0, "fulfilled_self_slots": 0,
            "fallback_uniform_slots": 0, "realized_kill_chain_samples": 0,
            "realized_self_chain_samples": 0,
        },
    )


def test_protocol_is_fresh_fixed_endpoint_sampling_only_and_terminal():
    protocol, _ = load_protocol(PROTOCOL)
    assert len(registered_seeds(protocol)) == 33
    summary = dry_run(protocol)
    assert summary["arms"] == list(ARMS)
    assert summary["replicas"] == list(REPLICAS)
    assert summary["all_action_return_horizon_both_arms"] == 8
    assert summary["total_training_rounds"] == 600
    assert summary["conditional_evaluation_rounds"] == 1600
    assert summary["maximum_total_game_rounds"] == 2200
    assert summary["fixed_endpoint_round"] == 100 and summary["snapshot_search"] is False
    assert summary["new_reward_added"] is False
    assert summary["automatic_followup_started"] is False


def test_n8_return_credits_clean_kill_and_keeps_trade_only_negative_bucket():
    clean = [_raw(step, 12.0 if step == 5 else 0.0,
                  (e.KILLED_OPPONENT,) if step == 5 else ()) for step in range(1, 9)]
    clean_target = train.aggregate_return(clean)
    assert clean_target.return_steps == 8
    assert np.isclose(clean_target.reward, (0.99 ** 4) * 12.0)
    assert clean_target.kill_chain is True and clean_target.self_chain is False

    trade = [_raw(
        step, 12.0 if step == 5 else -40.0 if step == 6 else 0.0,
        (e.KILLED_OPPONENT,) if step == 5 else (e.KILLED_SELF, e.GOT_KILLED) if step == 6 else (),
        done=step == 6,
    ) for step in range(1, 7)]
    trade_target = train.aggregate_return(trade)
    assert trade_target.return_steps == 6 and trade_target.done is True
    assert trade_target.kill_chain is False and trade_target.self_chain is True


def test_n8_return_rejects_cross_episode_and_stops_at_terminal():
    window = [_raw(1, 1.0), _raw(2, 2.0, done=True), _raw(3, 999.0)]
    target = train.aggregate_return(window)
    assert target.return_steps == 2
    assert np.isclose(target.reward, 1.0 + 0.99 * 2.0)
    crossed = [_raw(1, episode=1), _raw(2, episode=2)]
    try:
        train.aggregate_return(crossed)
    except ValueError as exc:
        assert "crossed an episode" in str(exc)
    else:
        raise AssertionError("cross-episode return was accepted")


def test_dead_terminal_flushes_every_raw_transition_exactly_once():
    agent = _queue_agent()
    for step in range(1, 11):
        train._append_training_raw(agent, _raw(step, done=step == 10))
    d = agent.training_diagnostics
    assert d["raw_transitions"] == d["matured_targets"] == 10
    assert sum(d["return_steps_histogram"].values()) == 10
    assert d["return_steps_histogram"]["8"] == 3
    assert not agent._episode_raw


def test_survivor_terminal_merges_only_the_new_survival_event():
    agent = _queue_agent()
    raw = _raw(1, events=(e.WAITED,))
    train._append_training_raw(agent, raw)
    train._merge_training_survivor(
        agent, {"round": 1, "step": 1}, "BOMB",
        [e.WAITED, e.SURVIVED_ROUND],
    )
    d = agent.training_diagnostics
    assert d["raw_transitions"] == d["matured_targets"] == 1
    assert d["survivor_terminal_merges"] == 1
    assert agent.replay_buffer._items[0].done is True
    assert not agent._episode_raw


def test_stratified_batch_reserves_two_clean_kills_and_two_self_kills():
    replay = ReplayBuffer(200)
    for index in range(80):
        replay.add(_transition(index, kill=index in (4, 14), self_kill=index in (24, 34)))
    batch, stats = replay.sample_stratified(64, 2, 2, np.random.default_rng(136000))
    assert len(batch) == len({item.start_step for item in batch}) == 64
    assert stats["fulfilled_kill_slots"] == 2 and stats["fulfilled_self_slots"] == 2
    assert stats["fallback_uniform_slots"] == 0
    assert stats["realized_kill_chain_samples"] >= 2
    assert stats["realized_self_chain_samples"] >= 2


def test_reserved_shortages_fall_back_to_unique_uniform_and_are_audited():
    replay = ReplayBuffer(100)
    for index in range(70):
        replay.add(_transition(index, kill=index == 3))
    batch, stats = replay.sample_stratified(64, 2, 2, np.random.default_rng(7))
    assert len(batch) == len({item.start_step for item in batch}) == 64
    assert stats["fulfilled_kill_slots"] == 1 and stats["fulfilled_self_slots"] == 0
    assert stats["fallback_uniform_slots"] == 3


def test_circular_overwrite_removes_stale_sampling_membership():
    replay = ReplayBuffer(4)
    replay.add(_transition(0, kill=True))
    replay.add(_transition(1, self_kill=True))
    replay.add(_transition(2))
    replay.add(_transition(3))
    assert replay.kill_items == replay.self_items == 1
    replay.add(_transition(4))
    replay.add(_transition(5))
    assert replay.kill_items == replay.self_items == 0


def test_no_replicated_parameter_effect_skips_all_evaluation():
    protocol, _ = load_protocol(PROTOCOL)
    effect = {"evaluation_authorized": False, "endpoint_divergent_pairs": 1, "pairs": {}}
    result = decide(protocol, None, {}, effect)
    assert result["decision"] == "sampling_intervention_no_parameter_effect_stop"
    assert result["evaluation_started"] is False
    assert result["automatic_followup_started"] is False


def test_both_arms_restore_identical_parent_before_sampling_changes():
    protocol, _ = load_protocol(PROTOCOL)
    base = {
        "MODEL_A_SAMPLING_PROTOCOL_PATH": str(PROTOCOL),
        "MODEL_A_SAMPLING_RUN_MODE": "train",
        "MODEL_A_SAMPLING_REPLICA": "r1",
        "MODEL_A_SAMPLING_SEED": str(protocol["training"]["seeds_by_replica"]["r1"]["agent_seed"]),
        "MODEL_A_SAMPLING_PARENT_PATH": str(ROOT / protocol["source_parent"]["path"]),
    }
    keys = (*base, "MODEL_A_SAMPLING_ARM", "MODEL_A_SAMPLING_CHECKPOINT_PATH")
    previous = {key: os.environ.get(key) for key in keys}
    original_checkpoint_path = callbacks.checkpoint_path
    agents = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            callbacks.checkpoint_path = lambda _protocol, arm, replica: root / arm / replica / "round-0100.pt"
            for arm in ARMS:
                output = root / arm / "r1" / "round-0100.pt"
                os.environ.update({
                    **base, "MODEL_A_SAMPLING_ARM": arm,
                    "MODEL_A_SAMPLING_CHECKPOINT_PATH": str(output),
                })
                agent = SimpleNamespace(train=True, logger=logging.getLogger(f"sampling-{arm}"))
                callbacks.setup(agent)
                train.setup_training(agent)
                agents[arm] = agent
            for left, right in zip(agents["uniform_n8"].online_net.parameters(), agents["stratified_n8"].online_net.parameters()):
                assert left.detach().equal(right.detach())
            _assert_nested_equal(
                agents["uniform_n8"].optimizer.state_dict(),
                agents["stratified_n8"].optimizer.state_dict(),
            )
            assert len(agents["uniform_n8"].replay_buffer) == len(agents["stratified_n8"].replay_buffer) == 0
            assert abs(agents["uniform_n8"].epsilon - 0.05) < 1e-12
            assert abs(agents["stratified_n8"].epsilon - 0.05) < 1e-12
    finally:
        callbacks.checkpoint_path = original_checkpoint_path
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
