"""Regression tests for corrected resolved-duel Task-3 distribution v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_code.model_a_task3_killrich.config import load_config
from tools.task3_distribution_world_v2 import Task3ResolvedDuelWorld
from tools.task3_training_distribution_v2 import load_protocol


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/task3-training-distribution-v2-matched-s113000.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_report_remains_bound_and_untouched():
    report = ROOT / "experiments/logs/evaluations/task3-training-distribution-matched-s112000.json"
    assert _sha256(report) == "dd7f21f9e1ca052eaf5feff1e226cada78cfcf5afb91d01e8011eccd45c6184b"


def test_resolved_duel_ends_on_elimination_but_control_keeps_official_rule():
    world = Task3ResolvedDuelWorld.__new__(Task3ResolvedDuelWorld)
    world.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    world.active_agents = [SimpleNamespace(train=False)]
    world.agents = list(world.active_agents)
    world.arena = np.ones((17, 17), dtype=int)
    world.coins = []
    world.bombs = []
    world.explosions = []
    world.step = 1
    world.args = SimpleNamespace(continue_without_training=True)

    os.environ["TASK3_TRAINING_DISTRIBUTION"] = "kill-rich"
    try:
        assert world.time_to_stop() is True
    finally:
        os.environ.pop("TASK3_TRAINING_DISTRIBUTION", None)

    os.environ["TASK3_TRAINING_DISTRIBUTION"] = "classic-control"
    try:
        assert world.time_to_stop() is False
    finally:
        os.environ.pop("TASK3_TRAINING_DISTRIBUTION", None)


def test_v2_pairs_keep_the_same_algorithm_and_fresh_matched_seeds():
    for index, seed in enumerate((113100, 113101, 113102), start=1):
        control, _, _ = load_config(ROOT / f"experiments/configs/task3-distribution-v2-control-r{index}-s{seed}.json")
        candidate, _, _ = load_config(ROOT / f"experiments/configs/task3-distribution-v2-killrich-r{index}-s{seed}.json")
        ignored = {"training_distribution", "run_id", "variant", "checkpoint_path"}
        assert {k: v for k, v in control.items() if k not in ignored} == {
            k: v for k, v in candidate.items() if k not in ignored
        }
        assert (control["seed"], control["agent_seed"], control["opponent_seed"]) == (
            candidate["seed"], candidate["agent_seed"], candidate["opponent_seed"],
        )
        assert control["sampling_profile"] == candidate["sampling_profile"] == "uniform"


def test_v2_protocol_corrects_only_termination_and_uses_feasible_fresh_gate():
    protocol = load_protocol(PROTOCOL)
    assert protocol["protocol_revision"] == "resolved-duel-v2"
    assert protocol["only_variable"]["only_change_from_v1"] == (
        "end the training-only kill-rich episode on first elimination"
    )
    gate = protocol["signal_gate"]
    assert gate["minimum_candidate_kills"] == 60
    assert gate["minimum_positive_cases"] == 6
    assert gate["minimum_kill_step_rate"] == "1/1000"
    assert gate["minimum_rate_multiplier"] == 3
    assert protocol["automatic_followup"] is False
    assert protocol["includes_task4"] is False
    # This test is used both before and after the preregistered run.  Once the
    # run exists, validate its terminal lifecycle guarantees instead of
    # retaining the now-stale pre-run assertion that the report must be absent.
    output = ROOT / protocol["output_path"]
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "completed"
        assert report["protocol_revision"] == "resolved-duel-v2"
        assert report["automatic_followup_started"] is False
        assert report["task4_started"] is False
