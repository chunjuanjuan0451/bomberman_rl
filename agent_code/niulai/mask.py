"""Keep the trained agent's survival and collision mask."""

import numpy as np

from .safety import (
    ACTIONS, DELTAS, _blocked_positions, _can_survive_from,
    _hazard_schedule, _in_bounds, _physical_destination, legal_action_mask,
)


def _opponent_next_positions(game_state):
    field = game_state["field"]
    blocked = _blocked_positions(game_state)
    positions = {tuple(other[3]) for other in game_state["others"]}
    next_positions = set()
    for position in positions:
        candidates = [position]
        candidates.extend((position[0] + dx, position[1] + dy) for dx, dy in DELTAS.values())
        for candidate in candidates:
            if not _in_bounds(field, candidate) or field[candidate] != 0:
                continue
            if candidate in blocked and candidate != position:
                continue
            next_positions.add(candidate)
    return next_positions


def _collision_safe_actions(game_state):
    hazards, horizon = _hazard_schedule(game_state)
    opponent_reach = _opponent_next_positions(game_state)
    safe = np.zeros(len(ACTIONS), dtype=bool)
    for index, action in enumerate(ACTIONS[:5]):
        destination = _physical_destination(game_state, action)
        if destination is None:
            continue
        if 1 in hazards.get(destination, set()) or destination in opponent_reach:
            continue
        if _can_survive_from(game_state, destination, 1, hazards, horizon):
            safe[index] = True
    return safe


def effective_action_mask(game_state, active_own_bomb):
    legal = legal_action_mask(game_state)
    if not active_own_bomb:
        return legal
    filtered = legal & _collision_safe_actions(game_state)
    return filtered if filtered.any() else legal
