"""Regression tests for the Task-3 D-only multiseed confirmation."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from agent_code.model_a_task3.config import load_config
from tools.task3_d_multiseed import (
    REPLICAS, _delta, evaluate_pooled, evaluate_replica, load_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/task3-d-multiseed-confirmation-s108300.json"


def test_task3_d_confirmation_protocol_is_fresh_isolated_and_budgeted():
    protocol = load_protocol(PROTOCOL)
    assert protocol["task"] == 3 and protocol["includes_task4"] is False
    assert protocol["training"] is True and protocol["automatic_followup"] is False
    assert protocol["discovery_data_in_confirmation"] is False
    assert tuple(protocol["replicas"]) == REPLICAS
    configs = [
        load_config(ROOT / spec["config_path"])[0]
        for spec in protocol["replicas"].values()
    ]
    assert sum(config["rounds"] for config in configs) == 900
    assert len({(c["seed"], c["agent_seed"], c["opponent_seed"]) for c in configs}) == 3
    assert all(c["arm"] == "D" and c["action_features"] and c["n_step"] == 3 for c in configs)
    cases = [case for stratum in protocol["strata"].values() for case in stratum["cases"]]
    assert len(cases) == 12 and len({case["world_seed"] for case in cases}) == 12
    assert sum(
        len(stratum["cases"]) * stratum["rounds_per_seed"] * 4
        for stratum in protocol["strata"].values()
    ) == 2400


def test_task3_exact_rate_gate_accepts_mathematical_equality():
    base = {"rounds": 300, "kills": 21}
    candidate = {"rounds": 300, "kills": 27}
    assert _delta(base, candidate, "kills") == Fraction(1, 50)
    assert _delta(base, candidate, "kills") >= Fraction("1/50")


def _metrics(score: int, coins: int, kills: int, suicides: int = 2) -> dict:
    return {
        "rounds": 50, "score": score, "score_per_round": score / 50,
        "coins": coins, "kills": kills, "crates": 90, "bombs": 55,
        "suicides": suicides, "invalid_actions": 2, "steps": 1_000,
        "mean_decision_time_ms": 0.5,
    }


def _synthetic_rows() -> dict:
    rows = {}
    for stratum_index, stratum in enumerate(("peaceful", "coin_collector")):
        rows[stratum] = []
        for seed in range(6):
            row = {
                "seed_tuple": {"world_seed": 900_000 + 100 * stratum_index + seed},
                "v4": {"target_metrics": _metrics(100, 75, 5)},
            }
            for replica in REPLICAS:
                row[replica] = {"target_metrics": _metrics(110, 77, 6)}
            rows[stratum].append(row)
    return rows


def test_task3_d_confirmation_requires_replicas_and_pooled_exact_gate():
    protocol = load_protocol(PROTOCOL)
    rows = _synthetic_rows()
    replicas = {
        replica: evaluate_replica(rows, replica, protocol["replica_gate"])
        for replica in REPLICAS
    }
    assert all(result["supportive"] for result in replicas.values())
    pooled = evaluate_pooled(rows, replicas, protocol)
    assert pooled["passed"]
    assert pooled["supportive_replica_count"] == 3
    assert pooled["selected_replica"] == "R1"
    assert pooled["overall"]["exact_deltas_per_round"]["kills"] == "1/50"


def test_task3_four_arm_erratum_preserves_exact_decision_provenance():
    path = ROOT / "experiments/logs/evaluations/task3-four-arm-pilot-s108000-DECISION_ERRATUM.json"
    erratum = json.loads(path.read_text())
    assert erratum["exact_evidence"]["exact_delta"] == "1/50"
    assert erratum["exact_evidence"]["comparison"] == "equal_pass"
    assert erratum["corrected_selected_arm"] == "D"
    assert erratum["future_rule"].startswith("All subsequent rate gates must use integer")
