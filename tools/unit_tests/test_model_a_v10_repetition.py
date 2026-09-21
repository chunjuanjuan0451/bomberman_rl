"""Deterministic history-aware root-prior tests."""

import json
from pathlib import Path

import numpy as np

from agent_code.model_a_v10.callbacks import validated_student_head_weights
from agent_code.model_a_v10.interfaces import HeuristicNetwork
from agent_code.model_a_v10.repetition import RepetitionContext
from agent_code.model_a_v10.search import StochasticPUCT
from agent_code.model_a_v10.simulator import SimAgent, SimState
from tools.v10_phase3_2 import evaluate_gate, sha256, validate_arm_config


def _state():
    field = -np.ones((9, 9), dtype=int)
    field[1:-1, 1:-1] = 0
    return SimState(field, [SimAgent("root", 4, 4)])


def _context(steps=0, positions=((4, 4),)):
    ctx = RepetitionContext()
    ctx.round_id, ctx.last_score, ctx.steps_since_score = 1, 0, steps
    for pos in positions:
        ctx.positions.append(pos)
    return ctx


def test_repetition_inactive_prior_is_unchanged_before_eight_steps():
    state = _state(); base = StochasticPUCT(HeuristicNetwork())._root_policy(state, 0)
    adjusted = StochasticPUCT(HeuristicNetwork(), repetition_context=_context(7, ((4, 4),) * 8))._root_policy(state, 0, apply_history=True)
    assert adjusted.keys() == base.keys() and np.allclose(list(adjusted.values()), list(base.values()))


def test_repetition_penalizes_repeated_candidate_after_activation():
    state = _state(); ctx = _context(8, ((3, 4),) * 8)
    values = ctx.multipliers(state, 0, ("LEFT", "RIGHT"))
    assert values["LEFT"] < 1.0 and values["RIGHT"] == 1.0


def test_unvisited_direction_is_not_penalized():
    values = _context(8, ((4, 3),) * 16).multipliers(_state(), 0, ("RIGHT",))
    assert values["RIGHT"] == 1.0


def test_score_clears_history_and_reactivates_counter():
    ctx = _context(12, ((4, 3),) * 8)
    ctx.update(1, 1, (4, 4))
    assert ctx.steps_since_score == 0 and list(ctx.positions) == [(4, 4)]
    assert ctx.multipliers(_state(), 0, ("LEFT",))["LEFT"] == 1.0


def test_round_change_clears_history():
    ctx = _context(12, ((3, 4),) * 8)
    ctx.update(2, 0, (4, 4))
    assert ctx.steps_since_score == 0 and list(ctx.positions) == [(4, 4)]


def test_history_only_changes_root_prior_not_deeper_policy():
    state = _state(); ctx = _context(8, ((3, 4),) * 8)
    searcher = StochasticPUCT(HeuristicNetwork(), repetition_context=ctx)
    root = searcher._root_policy(state, 0, apply_history=True)
    deeper = searcher._root_policy(state, 0, apply_history=False)
    assert root != deeper and abs(sum(root.values()) - 1.0) < 1e-12


def test_disabled_context_matches_baseline_search_exactly():
    state = _state()
    baseline = StochasticPUCT(HeuristicNetwork()).search(state, 0, simulations=20, seed=123)
    disabled = StochasticPUCT(HeuristicNetwork(), repetition_context=RepetitionContext(enabled=False)).search(state, 0, simulations=20, seed=123)
    assert baseline.action == disabled.action
    assert baseline.root_visits == disabled.root_visits


def test_official_callback_config_schema_rejects_offline_head_key():
    valid = {"student_head_weights": {"policy": 0, "value": 0.1, "opponent": 0, "risk": 0}}
    assert validated_student_head_weights(valid)["value"] == 0.1
    try:
        validated_student_head_weights({"head_weights": valid["student_head_weights"]})
    except ValueError:
        pass
    else:
        raise AssertionError("offline-only head_weights key must not pass official validation")


def test_official_gate_checks_latency_for_both_arms():
    rows = [{
        "baseline": {"score": 40, "mean_decision_ms": 451.0},
        "candidate": {"score": 50, "mean_decision_ms": 20.0},
    }]
    baseline = {"mean_score": 40.0, "step_limit_count": 1, "mean_steps": 400.0,
                "survival_rate": 1.0, "invalid_actions": 0}
    candidate = {"mean_score": 50.0, "step_limit_count": 0, "mean_steps": 120.0,
                 "survival_rate": 1.0, "invalid_actions": 0}
    thresholds = {"mean_decision_ms": 450.0, "minimum_step_limit_reduction": 1,
                  "mean_steps_ratio": 0.95, "max_score_drop": 5}
    assert not evaluate_gate(rows, baseline, candidate, thresholds)["passed"]


def test_s103008_arm_configs_match_official_callback_schema():
    root = Path(__file__).resolve().parents[2]
    checkpoint = root / "experiments/checkpoints/model-a-v10-centered-blend-coin-heaven-s103004.npz"
    for variant, enabled in (("baseline", False), ("candidate", True)):
        path = root / f"experiments/configs/v10-phase3.2-official-s103008-{variant}.json"
        validate_arm_config(json.loads(path.read_text()), repetition_enabled=enabled,
                            checkpoint_hash=sha256(checkpoint))
