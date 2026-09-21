"""Bounded potential-based reward shaping for Model B."""

from __future__ import annotations

from .features import _max_board_distance, shortest_safe_coin_distance

GAMMA = 0.99
POTENTIAL_SCALE = 0.15


def potential(game_state: dict | None) -> float:
    """Return a bounded negative potential based on safe coin distance."""
    if game_state is None:
        return 0.0
    distance = shortest_safe_coin_distance(game_state)
    if distance is None:
        return 0.0
    return -POTENTIAL_SCALE * min(1.0, distance / _max_board_distance(game_state["field"]))


def shaping_reward(old_game_state: dict | None, new_game_state: dict | None) -> float:
    return GAMMA * potential(new_game_state) - potential(old_game_state)
