"""Regression tests for the single-variable replay-sampling experiment."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from agent_code.model_a_task3_replay.config import load_config
from agent_code.model_a_task3_replay.replay import ReplayBuffer, Transition
from agent_code.model_a_task3_replay.train import RawTransition, aggregate_n_step
from tools.task3_replay_sampling import load_protocol


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/task3-replay-sampling-matched-s110000.json"


def _transition(episode: int, step: int) -> Transition:
    return Transition(step, step % 6, float(step), step + 1, False, None, 0.99, episode, step)


def test_uniform_and_no_tag_fallback_use_exact_legacy_rng_path():
    buffer = ReplayBuffer(100)
    for step in range(1, 81):
        buffer.add(_transition(1, step))
    expected_rng = np.random.default_rng(210000)
    expected = expected_rng.choice(80, size=64, replace=False).tolist()

    control = [item.start_step - 1 for item in buffer.sample(64, np.random.default_rng(210000), 0.0)]
    fallback = [item.start_step - 1 for item in buffer.sample(64, np.random.default_rng(210000), 0.25)]
    assert control == expected == fallback
    assert buffer.uniform_fallback_calls == 1


def test_kill_window_tags_past_and_future_n_step_starts_without_duplication():
    buffer = ReplayBuffer(100)
    for step in range(1, 21):
        buffer.add(_transition(7, step))
    buffer.mark_kill_window(7, 20, 12)
    assert {buffer._items[index].start_step for index in buffer._tagged} == set(range(9, 21))
    # In real 3-step emission, t-1 and t may arrive after the kill callback.
    future_buffer = ReplayBuffer(100)
    for step in range(1, 19):
        future_buffer.add(_transition(7, step))
    future_buffer.mark_kill_window(7, 20, 12)
    future_buffer.add(_transition(7, 19))
    future_buffer.add(_transition(7, 20))
    assert {future_buffer._items[index].start_step for index in future_buffer._tagged} == set(range(9, 21))
    assert len(future_buffer) == 20


def test_stratified_batch_is_exactly_sixteen_causal_when_available():
    buffer = ReplayBuffer(200)
    for step in range(1, 101):
        buffer.add(_transition(3, step))
    buffer.mark_kill_window(3, 80, 20)
    sampled = buffer.sample(64, np.random.default_rng(9), 0.25)
    tagged_steps = set(range(61, 81))
    assert sum(item.start_step in tagged_steps for item in sampled) == 16
    assert len({item.start_step for item in sampled}) == 64
    assert buffer.requested_causal_draws == buffer.realized_causal_draws == 16


def test_circular_overwrite_removes_stale_tag_membership():
    buffer = ReplayBuffer(4)
    for step in range(1, 5):
        buffer.add(_transition(1, step))
    buffer.mark_kill_window(1, 4, 2)
    assert buffer.tagged_count == 2
    for step in range(1, 5):
        buffer.add(_transition(2, step))
    assert buffer.tagged_count == 0


def test_n_step_aggregation_preserves_start_metadata():
    queue = deque([
        RawTransition("s0", 0, 1.0, "s1", False, "m1", 12, 30),
        RawTransition("s1", 1, 2.0, "s2", False, "m2", 12, 31),
        RawTransition("s2", 2, 3.0, "s3", False, "m3", 12, 32),
    ])
    result = aggregate_n_step(queue, 3, 0.9)
    assert result.episode_id == 12 and result.start_step == 30
    assert np.isclose(result.reward, 1 + 0.9 * 2 + 0.81 * 3)


def test_configs_are_matched_and_change_only_sampling_distribution():
    for index, seed in enumerate((110000, 110001, 110002), start=1):
        control, _, _ = load_config(ROOT / f"experiments/configs/task3-replay-control-r{index}-s{seed}.json")
        stratified, _, _ = load_config(ROOT / f"experiments/configs/task3-replay-stratified-r{index}-s{seed}.json")
        ignored = {"run_id", "variant", "checkpoint_path", "sampling_profile", "kill_causal_fraction"}
        assert {key: value for key, value in control.items() if key not in ignored} == {
            key: value for key, value in stratified.items() if key not in ignored
        }
        assert control["kill_causal_fraction"] == 0.0
        assert stratified["kill_causal_fraction"] == 0.25


def test_protocol_is_single_shot_task3_only_and_exactly_budgeted():
    protocol = load_protocol(PROTOCOL)
    assert protocol["task"] == 3 and protocol["includes_task4"] is False
    assert protocol["automatic_followup"] is False
    assert protocol["formal_budget"] == {
        "training_runs": 6, "training_rounds": 1800,
        "evaluation_runs": 56, "evaluation_rounds": 2800,
    }
    assert protocol["selection"]["order"] == ["S1", "S2", "S3"]
    assert protocol["vs_control_gate"]["minimum_overall_kills_delta"] == "1/100"
    assert protocol["vs_v4_gate"]["minimum_overall_score_delta"] == "1/10"
    assert "ratios/windows" in protocol["decision_branches"]["replay_sampling_not_confirmed_stop"]
    output = ROOT / protocol["output_path"]
    if output.exists():
        report = json.loads(output.read_text())
        assert report["status"] == "completed"
        assert report["decision"] in {
            "replay_sampling_confirmed", "replay_sampling_not_confirmed_stop",
        }
        assert report["automatic_followup_started"] is False
        assert report["task4_started"] is False
    assert len(json.loads(PROTOCOL.read_text())["bindings"]) == 10
