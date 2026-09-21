"""Contract tests for the terminal source-r2 kill-probe audit."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from agent_code.model_a_v4_kill_probe import callbacks
from agent_code.model_a_v4_kill_probe.config import CASES, FOLDS, load_protocol
from tools.v4_task4_kill_probe_audit import average_precision, dry_run, ranking_metrics, window_labels


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/configs/model-a-v4-kill-probe-audit-s128000.json"


def test_kill_probe_protocol_is_diagnostic_only_and_uses_fresh_grouped_cases():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    assert protocol["scope"] == "task4c_offline_learnability_only"
    assert tuple(protocol["collection"]["cases"]) == CASES
    assert tuple(protocol["probe"]["folds"]) == FOLDS
    assert protocol["collection"]["policy_updates"] == 0
    assert protocol["automatic_followup"] is False
    seeds = [value for case in protocol["collection"]["cases"].values() for value in case.values()]
    assert len(seeds) == 18
    assert len(set(seeds)) == 18


def test_kill_probe_dry_run_never_starts_policy_training():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    summary = dry_run(protocol)
    assert summary["total_collection_rounds"] == 300
    assert summary["cross_validation_folds"] == 3
    assert summary["policy_updates"] == 0
    assert summary["policy_training_started"] is False
    assert summary["policy_checkpoint_created"] is False


def test_window_labels_do_not_cross_episode_boundaries():
    episodes = np.asarray([0, 0, 0, 1, 1, 1])
    events = np.asarray([False, False, True, False, False, False])
    labels = window_labels(episodes, events, horizon=2)
    assert labels.tolist() == [False, True, True, False, False, False]


def test_ranking_metrics_use_natural_prevalence_and_report_risk():
    kill = np.asarray([True, False, False, False, True, False, False, False, False, False])
    risk = np.asarray([False, True, False, False, False, False, False, False, False, False])
    scores = np.asarray([1.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    metrics = ranking_metrics(kill, risk, scores, top_fraction=0.2)
    assert metrics["prevalence"] == 0.2
    assert metrics["top_kill_rate"] == 1.0
    assert metrics["top_kill_lift"] == 5.0
    assert metrics["top_self_risk_rate"] == 0.0
    assert average_precision(kill, np.zeros(len(kill))) == 0.2


def test_collector_loads_and_freezes_source_r2_without_an_output_checkpoint():
    protocol, _ = load_protocol(PROTOCOL_PATH)
    case = "c1"
    with tempfile.TemporaryDirectory() as directory:
        protocol_path = PROTOCOL_PATH
        trace_root = Path(directory) / "traces"
        # The protocol itself remains immutable; only the collector's output binding
        # is redirected after setup so this test cannot create a formal artifact.
        values = {
            "MODEL_A_KILL_PROBE_PROTOCOL_PATH": str(protocol_path),
            "MODEL_A_KILL_PROBE_PARENT_PATH": str(ROOT / protocol["source_parent"]["path"]),
            "MODEL_A_KILL_PROBE_CASE": case,
            "MODEL_A_KILL_PROBE_SEED": str(protocol["collection"]["cases"][case]["agent_seed"]),
        }
        previous = {name: os.environ.get(name) for name in values}
        os.environ.update(values)
        try:
            agent = SimpleNamespace(train=True, logger=logging.getLogger("kill-probe-test"))
            callbacks.setup(agent)
            agent.trace_path = trace_root / "c1.npz"
            assert all(not parameter.requires_grad for parameter in agent.online_net.parameters())
            assert agent.parent_sha256 == protocol["source_parent"]["sha256"]
            assert not agent.trace_path.exists()
        finally:
            for name, old_value in previous.items():
                if old_value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old_value
