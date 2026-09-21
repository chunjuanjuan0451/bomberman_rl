"""Deterministic semantics tests for the v10 Phase-0 simulator."""

from __future__ import annotations

import numpy as np

from agent_code.model_a_v10.official_adapter import equivalent, official_step
from agent_code.model_a_v10.simulator import SimAgent, SimBomb, SimExplosion, SimState, step


def _field(size: int = 9) -> np.ndarray:
    field = -np.ones((size, size), dtype=int)
    field[1:-1, 1:-1] = 0
    return field


def _state(*agents: SimAgent, field: np.ndarray | None = None, **kwargs) -> SimState:
    return SimState(_field() if field is None else field, list(agents), **kwargs)


def test_v10_movement_and_wait_match_official():
    source = _state(SimAgent("a", 3, 3), SimAgent("b", 6, 6))
    result = step(source, ["UP", "WAIT"], [0, 1])
    official = official_step(source, ["UP", "WAIT"], [0, 1])
    assert result.agents[0].x == 3 and result.agents[0].y == 2
    assert result.agents[0].events == ["MOVED_UP"]
    assert result.agents[1].events == ["WAITED"]
    assert equivalent(result, official)[0]


def test_v10_random_permutation_resolves_occupied_target_sequentially():
    source = _state(SimAgent("a", 2, 3), SimAgent("b", 4, 3))
    first = step(source, ["RIGHT", "LEFT"], [0, 1])
    second = step(source, ["RIGHT", "LEFT"], [1, 0])
    assert (first.agents[0].x, first.agents[1].x) == (3, 4)
    assert first.agents[1].events == ["INVALID_ACTION"]
    assert (second.agents[0].x, second.agents[1].x) == (2, 3)
    assert second.agents[0].events == ["INVALID_ACTION"]
    assert equivalent(first, official_step(source, ["RIGHT", "LEFT"], [0, 1]))[0]
    assert equivalent(second, official_step(source, ["RIGHT", "LEFT"], [1, 0]))[0]


def test_v10_illegal_action_and_unavailable_bomb_are_invalid():
    source = _state(SimAgent("a", 1, 1, bombs_left=False), SimAgent("b", 6, 6))
    result = step(source, ["LEFT", "BOMB"], [0, 1])
    assert result.agents[0].events == ["INVALID_ACTION"]
    assert result.agents[1].events == ["BOMB_DROPPED"]
    assert equivalent(result, official_step(source, ["LEFT", "BOMB"], [0, 1]))[0]


def test_v10_bomb_countdown_and_explosion_lifecycle():
    source = _state(SimAgent("a", 3, 3), SimAgent("b", 7, 7))
    placed = step(source, ["BOMB", "WAIT"], [0, 1])
    assert placed.bombs[0].timer == 3 and not placed.agents[0].bombs_left
    current = step(placed, ["RIGHT", "WAIT"], [0, 1])
    current = step(current, ["UP", "WAIT"], [0, 1])
    current = step(current, ["WAIT", "WAIT"], [0, 1])
    assert current.bombs[0].timer == 0
    exploded = step(current, ["WAIT", "WAIT"], [0, 1])
    assert not exploded.bombs and exploded.explosions[0].stage == 0
    assert exploded.explosion_map()[3, 3] == 1
    lingering = step(exploded, ["WAIT", "WAIT"], [0, 1])
    assert lingering.explosions[0].stage == 0 and lingering.explosion_map()[3, 3] == 0
    smoke = step(lingering, ["WAIT", "WAIT"], [0, 1])
    assert smoke.explosions[0].stage == 1 and smoke.agents[0].bombs_left


def test_v10_official_does_not_chain_trigger_nearby_bombs():
    # Despite the common Bomberman convention, this exact environment counts
    # each timer independently; the test locks that official behavior down.
    source = _state(SimAgent("a", 1, 1), SimAgent("b", 7, 7),
                    bombs=[SimBomb(3, 3, 0, 0), SimBomb(4, 3, 1, 3)])
    result = step(source, ["WAIT", "WAIT"], [0, 1])
    assert len(result.explosions) == 1
    assert len(result.bombs) == 1 and result.bombs[0].timer == 2
    assert equivalent(result, official_step(source, ["WAIT", "WAIT"], [0, 1]))[0]


def test_v10_blast_destroys_crate_reveals_coin_and_collects_next_step():
    field = _field()
    field[4, 3] = 1
    source = _state(SimAgent("a", 3, 2), SimAgent("b", 7, 7), field=field,
                    coins={(4, 3): False}, bombs=[SimBomb(3, 3, 0, 0)])
    result = step(source, ["WAIT", "WAIT"], [0, 1])
    assert result.field[4, 3] == 0 and result.coins[(4, 3)]
    assert "CRATE_DESTROYED" in result.agents[0].events
    assert "COIN_FOUND" in result.agents[0].events
    assert equivalent(result, official_step(source, ["WAIT", "WAIT"], [0, 1]))[0]


def test_v10_explosion_death_scores_owner_and_notifies_survivors():
    source = _state(SimAgent("a", 1, 1), SimAgent("b", 3, 4), SimAgent("c", 7, 7),
                    bombs=[SimBomb(3, 3, 0, 0)])
    result = step(source, ["WAIT", "WAIT", "WAIT"], [0, 1, 2])
    assert not result.agents[1].alive
    assert result.agents[0].score == 5
    assert "KILLED_OPPONENT" in result.agents[0].events
    assert result.agents[1].events[-1] == "GOT_KILLED"
    assert "OPPONENT_ELIMINATED" in result.agents[2].events
    assert equivalent(result, official_step(source, ["WAIT", "WAIT", "WAIT"], [0, 1, 2]))[0]


def test_v10_terminal_adds_survived_round():
    source = _state(SimAgent("a", 3, 3))
    result = step(source, ["WAIT"], [0])
    assert result.ended and result.agents[0].events == ["WAITED", "SURVIVED_ROUND"]
    assert equivalent(result, official_step(source, ["WAIT"], [0]))[0]


def test_v10_rejects_non_permutation():
    source = _state(SimAgent("a", 3, 3), SimAgent("b", 5, 5))
    try:
        step(source, ["WAIT", "WAIT"], [0, 0])
    except ValueError as exc:
        assert "permutation" in str(exc)
    else:
        raise AssertionError("invalid permutation was accepted")
