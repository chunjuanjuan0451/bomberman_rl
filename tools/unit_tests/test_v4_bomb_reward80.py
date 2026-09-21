"""Contracts for the terminal six-step BOMB reward-scale signal gate."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import events as e
from agent_code.model_a_dqn import train as v4
from agent_code.model_a_v4_bomb_reward80 import callbacks, train
from agent_code.model_a_v4_bomb_reward80.config import load_protocol
from tools.v4_task4_bomb_reward80 import dry_run, registered_seeds


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-task4-bomb-reward80-s132000.json"


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


def test_reward80_changes_only_killed_opponent_value():
    reward12 = SimpleNamespace(arm="reward12")
    reward80 = SimpleNamespace(arm="reward80")
    unchanged = [e.COIN_COLLECTED, e.CRATE_DESTROYED, e.KILLED_SELF, e.GOT_KILLED, e.INVALID_ACTION]
    assert train.reward_from_events(reward12, unchanged) == v4.reward_from_events(unchanged)
    assert train.reward_from_events(reward80, unchanged) == v4.reward_from_events(unchanged)
    assert train.reward_from_events(reward12, [e.KILLED_OPPONENT]) == 12.0
    assert train.reward_from_events(reward80, [e.KILLED_OPPONENT]) == 80.0
    assert train.reward_from_events(reward12, [e.KILLED_OPPONENT, e.KILLED_SELF, e.GOT_KILLED]) == -28.0
    assert train.reward_from_events(reward80, [e.KILLED_OPPONENT, e.KILLED_SELF, e.GOT_KILLED]) == 40.0


def test_protocol_is_fresh_terminal_25_round_signal_gate():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert protocol["single_training_variable"] == "killed_opponent_reward_12_vs_80"
    assert len(registered_seeds(protocol)) == 21
    summary = dry_run(protocol)
    assert summary["snapshot_rounds"] == [5, 10, 15, 25]
    assert summary["total_training_rounds"] == 150
    assert summary["evaluation_rounds"] == 800
    assert summary["maximum_total_game_rounds"] == 950
    assert summary["bomb_return_steps_both_arms"] == 6
    assert summary["expansion_started"] is False
    assert summary["automatic_followup_started"] is False


def test_matched_arms_restore_identical_parent_and_optimizer():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    base = {
        "MODEL_A_BOMB_REWARD80_PROTOCOL_PATH": str(PROTOCOL_PATH),
        "MODEL_A_BOMB_REWARD80_CHECKPOINT_PATH": "",
        "MODEL_A_BOMB_REWARD80_PARENT_PATH": str(ROOT / protocol["source_parent"]["path"]),
        "MODEL_A_BOMB_REWARD80_REPLICA": "r1",
        "MODEL_A_BOMB_REWARD80_SEED": str(protocol["training"]["seeds_by_replica"]["r1"]["agent_seed"]),
    }
    keys = (*base, "MODEL_A_BOMB_REWARD80_ARM")
    previous = {key: os.environ.get(key) for key in keys}
    original_final_checkpoint_path = callbacks.final_checkpoint_path
    agents = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            callbacks.final_checkpoint_path = lambda _protocol, arm, replica: root / arm / replica / "final.pt"
            for arm in ("reward12", "reward80"):
                values = dict(base)
                values["MODEL_A_BOMB_REWARD80_ARM"] = arm
                values["MODEL_A_BOMB_REWARD80_CHECKPOINT_PATH"] = str(root / arm / "r1" / "final.pt")
                os.environ.update(values)
                agent = SimpleNamespace(train=True, logger=logging.getLogger(f"bomb-reward80-{arm}"))
                callbacks.setup(agent)
                train.setup_training(agent)
                agents[arm] = agent
            assert agents["reward12"].bomb_return_steps == agents["reward80"].bomb_return_steps == 6
            assert abs(agents["reward12"].epsilon - 0.05) < 1e-12
            assert abs(agents["reward80"].epsilon - 0.05) < 1e-12
            for left, right in zip(agents["reward12"].online_net.parameters(), agents["reward80"].online_net.parameters()):
                assert left.detach().equal(right.detach())
            _assert_nested_equal(agents["reward12"].optimizer.state_dict(), agents["reward80"].optimizer.state_dict())
            assert len(agents["reward12"].replay_buffer) == len(agents["reward80"].replay_buffer) == 0
    finally:
        callbacks.final_checkpoint_path = original_final_checkpoint_path
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_reward_profiles_are_bound_into_checkpoint_payload():
    for arm, expected in (("reward12", 12.0), ("reward80", 80.0)):
        agent = SimpleNamespace(
            arm=arm, replica="r1", online_net=SimpleNamespace(state_dict=lambda: {}),
            target_net=SimpleNamespace(state_dict=lambda: {}), optimizer=SimpleNamespace(state_dict=lambda: {}),
            completed_rounds=25, stage_start_rounds=0, training_steps=1, gradient_steps=1,
            epsilon=0.05, agent_seed=232001, protocol_sha256="x", parent_sha256="y",
            protocol={"reward_profiles": {arm: {"KILLED_OPPONENT": expected}}}, target_diagnostics={},
        )
        payload = train._payload(agent)
        assert payload["kill_reward"] == expected
        assert payload["bomb_return_steps"] == 6
        assert payload["training_variable"] == "killed_opponent_reward_12_vs_80"
