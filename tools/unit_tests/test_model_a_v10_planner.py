"""No-training regression tests for the v10.1 classic tactical planner."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_code.model_a_v10.callbacks import setup
from agent_code.model_a_v10.interfaces import HeuristicNetwork
from agent_code.model_a_v10.planner import ClassicTacticalPlanner, PlannerRootPolicyTransform
from agent_code.model_a_v10.search import StochasticPUCT
from agent_code.model_a_v10.simulator import SimAgent, SimState
from agent_code.model_a_v10.tactics import tactical_cases


def _case(name):
    return next(case for case in tactical_cases() if case.name == name)


def _plan(name):
    case = _case(name)
    base = HeuristicNetwork().evaluate(case.state, case.root_player).policy
    return ClassicTacticalPlanner().plan(case.state, case.root_player, base)


def test_planner_prefers_the_only_timely_escape_route():
    plan = _plan("unique_escape")
    assert max(plan.prior, key=plan.prior.get) == "RIGHT"
    assert plan.tactical and abs(sum(plan.prior.values()) - 1.0) < 1e-12


def test_planner_removes_bomb_when_root_cannot_survive_own_blast():
    plan = _plan("false_kill")
    assert plan.survival_width["BOMB"] == 0
    assert "BOMB" not in plan.prior


def test_planner_bombs_only_a_genuinely_trapped_opponent():
    forced = _plan("forced_kill")
    conflict = _plan("opponent_exit_conflict")
    assert forced.bomb_utility >= 4.5 and max(forced.prior, key=forced.prior.get) == "BOMB"
    assert conflict.bomb_utility == 0.0 and max(conflict.prior, key=conflict.prior.get) in ("UP", "DOWN")


def test_planner_moves_toward_visible_coin_without_bombs():
    field = -np.ones((9, 9), dtype=int)
    field[1:-1, 1:-1] = 0
    state = SimState(field, [SimAgent("root", 3, 3, bombs_left=False)], coins={(6, 3): True})
    base = HeuristicNetwork().evaluate(state, 0).policy
    plan = ClassicTacticalPlanner().plan(state, 0, base)
    assert max(plan.prior, key=plan.prior.get) == "RIGHT"


def test_task2_frontier_prefers_a_dense_bomb_square_over_one_adjacent_crate():
    field = -np.ones((11, 11), dtype=int)
    field[1:-1, 1:-1] = 0
    field[4, 5] = 1  # Immediate one-crate blast at the root.
    field[7, 4:7] = 1  # Reaching (6, 5) enables a three-crate blast.
    state = SimState(field, [SimAgent("root", 3, 5)])
    base = HeuristicNetwork().evaluate(state, 0).policy
    plan = ClassicTacticalPlanner(crate_frontier_weight=0.85,
                                  bomb_efficiency_penalty=1.0).plan(state, 0, base)
    # The adjacent crate blocks RIGHT itself; either vertical first step leads
    # around it to a three-crate frontier.  The important regression is that
    # this is preferred to spending a bomb for the single immediate crate.
    assert max(plan.prior, key=plan.prior.get) in ("UP", "DOWN")
    assert plan.action_scores["BOMB"] < max(plan.action_scores["UP"],
                                              plan.action_scores["DOWN"])


def test_persistent_crate_target_is_retained_and_drives_wall_aware_progress():
    field = -np.ones((11, 11), dtype=int)
    field[1:-1, 1:-1] = 0
    field[7, 4:7] = 1
    state = SimState(field, [SimAgent("root", 2, 5)])
    planner = ClassicTacticalPlanner()
    target = planner.select_persistent_crate_target(state, 0, None)
    assert target is not None
    moved = SimState(field, [SimAgent("root", 3, 5)])
    assert planner.select_persistent_crate_target(moved, 0, target) == target
    base = HeuristicNetwork().evaluate(moved, 0).policy
    plan = planner.plan(moved, 0, base, crate_target=target)
    assert plan.crate_target == target
    assert plan.crate_target_distance is not None


def test_shared_planner_transform_resets_target_by_round_and_protects_bomb():
    case = _case("forced_kill")
    transform = PlannerRootPolicyTransform(
        ClassicTacticalPlanner(), persistent_target_enabled=True,
        productive_bomb_protection=True)
    target = transform.prepare(case.state, case.root_player, round_id=7)
    base = HeuristicNetwork().evaluate(case.state, case.root_player).policy
    transform(case.state, case.root_player, base)
    assert transform.crate_target == target
    assert transform.last_plan is not None
    assert max(transform.last_plan.prior, key=transform.last_plan.prior.get) == "BOMB"
    action, protected = transform.protect_action("WAIT")
    assert action == "BOMB" and protected and transform.protected_bomb_count == 1
    transform.prepare(case.state, case.root_player, round_id=8)
    assert transform.round_id == 8


def test_planner_transform_runs_once_at_root_and_supplies_fallback_mask():
    case = _case("false_kill")
    transform = PlannerRootPolicyTransform(ClassicTacticalPlanner())
    result = StochasticPUCT(root_policy_transform=transform).search(
        case.state, case.root_player, simulations=12, seed=7)
    assert transform.call_count == 1
    assert "BOMB" not in result.raw_prior
    assert result.fallback in transform.last_plan.safe_actions


def test_planner_rejects_unsafe_configuration():
    for kwargs in ({"horizon": 4}, {"prior_weight": 1.1}, {"temperature": 0.0}):
        try:
            ClassicTacticalPlanner(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid planner settings accepted: {kwargs}")


def test_official_callback_loads_planner_and_utilized_budget_config():
    root = Path(__file__).resolve().parents[2]
    names = ("MODEL_A_V10_CHECKPOINT_PATH", "MODEL_A_V10_CONFIG_PATH")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ[names[0]] = str(root / "experiments/checkpoints/model-a-v10-centered-blend-coin-heaven-s103004.npz")
        os.environ[names[1]] = str(root / "experiments/configs/v10.1-classic-planner-s109602.json")
        holder = SimpleNamespace()
        setup(holder)
        assert isinstance(holder.v10_planner_transform, PlannerRootPolicyTransform)
        assert holder.v10_controller.budget.normal_deadline_ms == 240.0
        assert holder.v10_controller.budget.tactical_deadline_ms == 400.0
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
