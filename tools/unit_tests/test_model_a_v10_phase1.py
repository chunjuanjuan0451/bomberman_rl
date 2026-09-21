"""Tests for the no-training Phase-1 stochastic PUCT smoke implementation."""

from __future__ import annotations

from agent_code.model_a_v10.interfaces import HeuristicNetwork, legal_actions
from agent_code.model_a_v10.search import StochasticPUCT
from agent_code.model_a_v10.tactics import evaluate_tactics, tactical_cases


def _case(name):
    return next(case for case in tactical_cases() if case.name == name)


def test_v10_phase1_network_interface_has_all_four_heads_and_legal_policy():
    case = _case("unique_escape")
    output = HeuristicNetwork().evaluate(case.state, case.root_player)
    assert set(output.policy) == set(legal_actions(case.state, case.root_player))
    assert abs(sum(output.policy.values()) - 1.0) < 1e-12
    assert isinstance(output.value, float) and isinstance(output.risk, float)
    assert output.opponent_policy


def test_v10_phase1_fixed_seed_is_exactly_reproducible():
    case = _case("opponent_exit_conflict")
    searcher = StochasticPUCT()
    first = searcher.search(case.state, case.root_player, simulations=50, seed=case.seed)
    second = searcher.search(case.state, case.root_player, simulations=50, seed=case.seed)
    assert first.action == second.action and first.action in ("UP", "DOWN")
    assert first.root_visits == second.root_visits
    assert first.principal_variation == second.principal_variation


def test_v10_phase1_deadline_retains_immediate_legal_fallback():
    case = _case("unique_escape")
    result = StochasticPUCT().search(case.state, case.root_player, simulations=50, seed=case.seed, deadline_ms=0)
    assert result.simulations == 0 and result.reason == "deadline"
    assert result.action == result.fallback
    assert result.action in legal_actions(case.state, case.root_player)


def test_v10_phase1_tactics_cover_required_escape_kill_and_false_kill_cases():
    result = evaluate_tactics(50)
    by_name = {row["name"]: row for row in result["cases"]}
    assert result["passed"]
    assert by_name["unique_escape"]["search"]["action"] == "RIGHT"
    assert by_name["forced_kill"]["search"]["action"] == "BOMB"
    assert by_name["false_kill"]["search"]["action"] != "BOMB"
    assert by_name["opponent_exit_conflict"]["search"]["fallback"] == "RIGHT"
    assert by_name["opponent_exit_conflict"]["search"]["action"] in ("UP", "DOWN")
    assert by_name["opponent_exit_conflict"]["search_terminal_alive"]
    assert not by_name["opponent_exit_conflict"]["raw_terminal_alive"]


def test_v10_phase1_records_visits_risk_and_eight_ply_completion():
    case = _case("coin_kill_tradeoff")
    result = StochasticPUCT().search(case.state, case.root_player, simulations=50, seed=case.seed)
    assert result.simulations == 50 and sum(result.root_visits.values()) == 50
    assert result.max_depth == 8
    assert 0.0 <= result.risk <= 1.0
    assert result.principal_variation
