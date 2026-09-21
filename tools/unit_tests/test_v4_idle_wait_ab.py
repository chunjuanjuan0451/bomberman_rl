"""Contract tests for the conditional idle-WAIT short-training A/B."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest

import numpy as np

from agent_code.model_a_v4_idle_wait.train import _idle_wait
from tools import v4_idle_wait_ab as experiment


def state() -> dict:
    field = -np.ones((17, 17), dtype=int)
    field[1:16, 1:16] = 0
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("me", 0, True, (5, 5)), "others": [], "bombs": [], "coins": [],
        "explosion_map": np.zeros_like(field), "user_input": None,
    }


def metrics(score: int, waits: int, crates: int = 100, bombs: int = 50) -> dict:
    return {
        "rounds": 100, "score": score, "score_per_round": score / 100,
        "coins": score, "coins_per_round": score / 100, "kills": 0,
        "crates": crates, "crates_per_round": crates / 100,
        "bombs": bombs, "bombs_per_round": bombs / 100,
        "moves": 1000 - waits, "move_fraction": (1000 - waits) / 1000,
        "waits": waits, "wait_fraction": waits / 1000,
        "suicides": 0, "suicides_per_round": 0.0,
        "invalid_actions": 0, "invalid_actions_per_round": 0.0, "steps": 1000,
    }


class IdleWaitDefinitionTests(unittest.TestCase):
    def test_only_safe_bombless_discretionary_wait_is_eligible(self):
        base = state()
        self.assertTrue(_idle_wait(base, "WAIT"))
        self.assertFalse(_idle_wait(base, "UP"))
        with_bomb = copy.deepcopy(base); with_bomb["bombs"] = [((10, 10), 3)]
        self.assertFalse(_idle_wait(with_bomb, "WAIT"))
        exploding = copy.deepcopy(base); exploding["explosion_map"][5, 5] = 1
        self.assertFalse(_idle_wait(exploding, "WAIT"))

    def test_forced_wait_is_not_eligible(self):
        blocked = state()
        x, y = blocked["self"][3]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            blocked["field"][x + dx, y + dy] = -1
        self.assertFalse(_idle_wait(blocked, "WAIT"))


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(experiment.DEFAULT_PROTOCOL.read_text(encoding="utf-8"))

    def rows(self, candidate_score: int = 120, candidate_waits: int = 300) -> dict:
        rows = {"task1": {}, "task2": {}}
        rows["task1"]["frozen-v4"] = metrics(100, 400)
        rows["task2"]["frozen-v4"] = metrics(100, 400)
        for replica in experiment.REPLICAS:
            rows["task1"][f"control-{replica}"] = metrics(100, 400)
            rows["task1"][f"idle-wait-001-{replica}"] = metrics(100, 300)
            rows["task2"][f"control-{replica}"] = metrics(100, 400)
            rows["task2"][f"idle-wait-001-{replica}"] = metrics(candidate_score, candidate_waits)
        return rows

    def test_passing_signal_requires_pooled_and_replica_support(self):
        result = experiment.decide(self.protocol, self.rows())
        self.assertTrue(result["passed"])
        self.assertEqual(result["supportive_replica_pairs"], 3)
        self.assertIn("stop_before_competition_confirmation", result["decision"])

    def test_wait_reduction_cannot_be_replaced_by_score_alone(self):
        result = experiment.decide(self.protocol, self.rows(candidate_score=140, candidate_waits=390))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["task2_wait_reduction"])


class DryRunTests(unittest.TestCase):
    def test_registered_protocol_dry_run_starts_nothing(self):
        completed = subprocess.run(
            [sys.executable, str(experiment.ROOT / "tools/v4_idle_wait_ab.py")],
            cwd=experiment.ROOT, text=True, capture_output=True, check=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["training_rounds"], 600)
        self.assertEqual(payload["evaluation_rounds"], 875)
        self.assertFalse(payload["formal_execution_started"])
        self.assertFalse(payload["checkpoint_selection_allowed"])


if __name__ == "__main__":
    unittest.main()
