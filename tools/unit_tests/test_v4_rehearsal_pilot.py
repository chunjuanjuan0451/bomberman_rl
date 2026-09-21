"""Contracts for the fixed 25% source-task rehearsal pilot."""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

import events as e
from agent_code.model_a_dqn.features import ACTIONS
from agent_code.model_a_dqn.network import DuelingDQN, torch
from agent_code.model_a_v4_rehearsal import train
from agent_code.model_a_v4_rehearsal.config import ARMS, OLD_STRATA, REPLICAS, load_protocol
from agent_code.model_a_v4_rehearsal.replay import ReplayBuffer, Transition, load_dataset, pack_dataset, sample_fixed, unpack_dataset
from agent_code.model_a_v4_sampling_ab.train import RawTransition, aggregate_return
from tools.v4_rehearsal_pilot import dry_run, registered_seeds, validation_decision


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-v4-rehearsal-pilot-s137000.json"


def _transition(index: int, *, done: bool = False, kill: bool = False, self_kill: bool = False) -> Transition:
    state = (np.full((2, 2), index, dtype=np.float32), np.full(3, index, dtype=np.float32))
    next_state = None if done else (np.full((2, 2), index + 1, dtype=np.float32), np.full(3, index + 1, dtype=np.float32))
    return Transition(state, index % 6, float(index), next_state, done, None if done else np.ones(6, dtype=bool), 0.99 ** 8, 8, 1, index, kill, self_kill)


def _network_transition(index: int, *, kill: bool = False, self_kill: bool = False) -> Transition:
    state = (np.full((4, 7, 7), index / 100, dtype=np.float32), np.full(7, index / 100, dtype=np.float32))
    next_state = (np.full((4, 7, 7), (index + 1) / 100, dtype=np.float32), np.full(7, (index + 1) / 100, dtype=np.float32))
    return Transition(state, index % 6, float(index % 5), next_state, False, np.ones(6, dtype=bool), 0.99 ** 8, 8, 1, index, kill, self_kill)


def _raw(step: int, *, done: bool = False):
    state = (np.full((2, 2), step, dtype=np.float32), np.full(3, step, dtype=np.float32))
    next_state = None if done else (np.full((2, 2), step + 1, dtype=np.float32), np.full(3, step + 1, dtype=np.float32))
    return RawTransition(state, ACTIONS.index("WAIT"), 0.0, 0.0, next_state, done, None if done else np.ones(6, dtype=bool), 1, step, ())


def _queue_agent():
    return SimpleNamespace(
        run_mode="collect", dataset_transitions=[], training_steps=0, gradient_steps=0,
        _episode_raw=[], training_diagnostics={
            "raw_transitions": 0, "matured_targets": 0,
            "return_steps_histogram": {str(i): 0 for i in range(1, 9)},
            "kill_chain_targets_created": 0, "self_chain_targets_created": 0,
            "observed_kill_events": 0, "observed_self_events": 0,
            "survivor_terminal_merges": 0, "dead_terminal_appends": 0,
            "gradient_updates": 0, "sampled_batches": 0, "per_round": [],
        },
    )


def test_protocol_skips_redundant_stages_and_has_one_rehearsal_ratio():
    protocol, _ = load_protocol(PROTOCOL)
    summary = dry_run(protocol)
    assert len(registered_seeds(protocol)) == 93
    assert summary["stage0"].startswith("complete_")
    assert summary["stage1"].startswith("satisfied_by_completed_s136000")
    assert summary["rehearsal_batch"] == {
        "task4_uniform": 44, "task4_kill": 2, "task4_self": 2,
        "task1": 4, "task2": 4, "task3_peaceful": 4, "task3_coin": 4,
    }
    assert sum(summary["rehearsal_batch"].values()) == 64
    assert summary["maximum_total_rounds"] == 2900
    assert summary["automatic_followup_started"] is False


def test_dataset_round_trip_uses_tensor_only_checkpoint():
    transitions = [_transition(1), _transition(2, done=True, kill=True)]
    metadata = {"kind": "model-a-v4-rehearsal-dataset", "protocol_sha256": "abc", "replica": "r1", "stratum": "task1", "source_sha256": "source", "all_action_return_horizon": 8}
    payload = pack_dataset(transitions, metadata)
    restored = unpack_dataset(payload)
    assert len(restored) == 2 and restored[1].done and restored[1].kill_chain
    assert np.array_equal(restored[0].state[0], transitions[0].state[0])
    import torch
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dataset.pt"
        torch.save(payload, path)
        loaded = load_dataset(path, metadata)
        assert len(loaded) == 2 and loaded[1].next_state is None


def test_fixed_rehearsal_draw_is_unique_and_refuses_shortage():
    pool = [_transition(i) for i in range(10)]
    selected = sample_fixed(pool, 4, np.random.default_rng(137000))
    assert len(selected) == len({item.start_step for item in selected}) == 4
    try:
        sample_fixed(pool, 11, np.random.default_rng(1))
    except RuntimeError as exc:
        assert "cannot fill" in str(exc)
    else:
        raise AssertionError("rehearsal shortage was silently accepted")


def test_collection_n8_queue_flushes_without_optimizer():
    agent = _queue_agent()
    for step in range(1, 11):
        train._append_raw(agent, _raw(step, done=step == 10))
    assert agent.training_diagnostics["raw_transitions"] == 10
    assert agent.training_diagnostics["matured_targets"] == 10
    assert len(agent.dataset_transitions) == 10 and not agent._episode_raw
    assert agent.training_diagnostics["gradient_updates"] == 0


def test_rehearsal_uses_existing_clean_kill_and_self_definitions():
    window = [_raw(i) for i in range(1, 9)]
    window[4].events = (e.KILLED_OPPONENT,)
    clean = aggregate_return(window)
    assert clean.kill_chain and not clean.self_chain
    window[5].events = (e.KILLED_SELF, e.GOT_KILLED)
    trade = aggregate_return(window)
    assert not trade.kill_chain and trade.self_chain


def test_no_parameter_effect_stops_before_validation():
    protocol, _ = load_protocol(PROTOCOL)
    result = validation_decision(protocol, None, {"evaluation_authorized": False, "endpoint_divergent_pairs": 1, "pairs": {}})
    assert result["decision"] == "rehearsal_no_parameter_effect_stop"
    assert result["evaluation_started"] is False


def test_rehearsal_code_adds_no_reward_or_attack_module():
    paths = [
        ROOT / "agent_code/model_a_v4_rehearsal/callbacks.py",
        ROOT / "agent_code/model_a_v4_rehearsal/train.py",
        ROOT / "agent_code/model_a_v4_rehearsal/replay.py",
    ]
    assigned_constants = set()
    class_names = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        class_names.update(node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                assigned_constants.update(target.id for target in targets if isinstance(target, ast.Name) and target.id.isupper())
    assert not any("REWARD" in name for name in assigned_constants)
    assert not any(name.lower().endswith(("option", "gate", "head")) for name in class_names)


def test_control_and_rehearsal_batches_are_matched_at_64():
    protocol, _ = load_protocol(PROTOCOL)
    control = protocol["learning_contract"]["no_rehearsal_batch"]
    rehearsal = protocol["learning_contract"]["rehearsal25_batch"]
    assert control == {"task4_uniform": 60, "task4_kill": 2, "task4_self": 2, "old_task": 0}
    assert sum(control.values()) == sum(rehearsal.values()) == 64
    assert rehearsal["task4_kill"] == control["task4_kill"] == 2
    assert rehearsal["task4_self"] == control["task4_self"] == 2
    assert tuple(OLD_STRATA) == ("task1", "task2", "task3_peaceful", "task3_coin")
    assert tuple(ARMS) == ("no_rehearsal", "rehearsal25") and len(REPLICAS) == 3


def test_both_sampling_paths_execute_one_equal_size_optimizer_update():
    original_warmup = train.v4.WARMUP_TRANSITIONS
    try:
        train.v4.WARMUP_TRANSITIONS = 64
        for arm in ARMS:
            replay = ReplayBuffer(100)
            for i in range(70):
                replay.add(_network_transition(i, kill=i in (2, 12), self_kill=i in (22, 32)))
            online = DuelingDQN(); target = DuelingDQN(); target.load_state_dict(online.state_dict())
            diagnostics = {
                "gradient_updates": 0, "sampled_batches": 0,
                "requested_kill_slots": 0, "fulfilled_kill_slots": 0,
                "requested_self_slots": 0, "fulfilled_self_slots": 0,
                "fallback_uniform_slots": 0, "realized_kill_chain_samples": 0,
                "realized_self_chain_samples": 0, "requested_rehearsal_slots": 0,
                "fulfilled_rehearsal_slots": 0,
                "rehearsal_by_stratum": {name: 0 for name in OLD_STRATA},
            }
            agent = SimpleNamespace(
                arm=arm, replay_buffer=replay, training_steps=100,
                training_rng=np.random.default_rng(9), online_net=online, target_net=target,
                optimizer=torch.optim.Adam(online.parameters(), lr=3e-4), device=torch.device("cpu"),
                gradient_steps=0, training_diagnostics=diagnostics,
                rehearsal_pools={name: [_network_transition(100 + i) for i in range(10)] for name in OLD_STRATA},
            )
            train._learn(agent)
            assert diagnostics["gradient_updates"] == diagnostics["sampled_batches"] == 1
            expected = 16 if arm == "rehearsal25" else 0
            assert diagnostics["requested_rehearsal_slots"] == diagnostics["fulfilled_rehearsal_slots"] == expected
            assert sum(diagnostics["rehearsal_by_stratum"].values()) == expected
    finally:
        train.v4.WARMUP_TRANSITIONS = original_warmup
