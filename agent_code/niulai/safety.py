"""Bomb timing and legal action checks."""

from __future__ import annotations

import numpy as np

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
MOVE_ACTIONS = ACTIONS[:4]
DELTAS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
INF_TIME = 99
BLAST_POWER = 3
BOMB_TIMER = 4
EXPLOSION_DURATION = 2


def _position(game_state: dict) -> tuple[int, int]:
    return tuple(game_state["self"][3])


def _in_bounds(field: np.ndarray, position: tuple[int, int]) -> bool:
    x, y = position
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def _blocked_positions(game_state: dict) -> set[tuple[int, int]]:
    return {tuple(position) for position, _ in game_state["bombs"]} | {
        tuple(other[3]) for other in game_state["others"]
    }


def blast_positions(field: np.ndarray, origin: tuple[int, int], power: int = BLAST_POWER) -> list[tuple[int, int]]:
    """Cells hit by a bomb; stone walls stop the blast."""
    cells = [origin]
    for dx, dy in DELTAS.values():
        for distance in range(1, power + 1):
            candidate = (origin[0] + dx * distance, origin[1] + dy * distance)
            if not _in_bounds(field, candidate) or field[candidate] == -1:
                break
            cells.append(candidate)
    return cells


def danger_time_map(game_state: dict) -> np.ndarray:
    """Earliest known explosion time at each tile."""
    field = game_state["field"]
    danger = np.full(field.shape, INF_TIME, dtype=np.int16)
    danger[np.asarray(game_state["explosion_map"]) > 0] = 0
    for position, timer in game_state["bombs"]:
        for cell in blast_positions(field, tuple(position)):
            danger[cell] = min(danger[cell], int(timer))
    return danger


def _hazard_schedule(
    game_state: dict, extra_bomb: tuple[tuple[int, int], int] | None = None,
) -> tuple[dict[tuple[int, int], set[int]], int]:
    """List the future steps when each tile is dangerous."""
    field = game_state["field"]
    hazards: dict[tuple[int, int], set[int]] = {}
    explosion_map = np.asarray(game_state["explosion_map"])
    for x, y in np.argwhere(explosion_map > 0):
        hazards.setdefault((int(x), int(y)), set()).add(1)
    bombs = [(tuple(position), int(timer)) for position, timer in game_state["bombs"]]
    if extra_bomb is not None:
        bombs.append(extra_bomb)
    horizon = 1
    for position, timer in bombs:
        explosion_step = timer + 1
        horizon = max(horizon, explosion_step + EXPLOSION_DURATION - 1)
        for cell in blast_positions(field, position):
            hazards.setdefault(cell, set()).update(
                range(explosion_step, explosion_step + EXPLOSION_DURATION)
            )
    return hazards, horizon


def _physical_destination(
    game_state: dict, action: str, position: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    position = _position(game_state) if position is None else position
    if action == "WAIT":
        return position
    dx, dy = DELTAS[action]
    candidate = (position[0] + dx, position[1] + dy)
    field = game_state["field"]
    if not _in_bounds(field, candidate) or field[candidate] != 0:
        return None
    if candidate in _blocked_positions(game_state):
        return None
    return candidate


def _can_survive_from(
    game_state: dict,
    start: tuple[int, int],
    start_time: int,
    hazards: dict[tuple[int, int], set[int]],
    horizon: int,
    extra_blocked: set[tuple[int, int]] | None = None,
) -> bool:
    """Check whether at least one route survives the known explosions."""
    if start_time in hazards.get(start, set()):
        return False
    field = game_state["field"]
    blocked = _blocked_positions(game_state) | (extra_blocked or set())
    positions = {start}
    for time in range(start_time + 1, horizon + 1):
        next_positions = set()
        for position in positions:
            candidates = [position]
            candidates.extend((position[0] + dx, position[1] + dy) for dx, dy in DELTAS.values())
            for candidate in candidates:
                if not _in_bounds(field, candidate) or field[candidate] != 0:
                    continue
                if candidate in blocked and candidate != position:
                    continue
                if time in hazards.get(candidate, set()):
                    continue
                next_positions.add(candidate)
        if not next_positions:
            return False
        positions = next_positions
    return True


def _escape_options_after_bomb(game_state: dict) -> tuple[int, int]:
    """Count safe first moves and find the shortest exit from our blast."""
    from collections import deque

    start = _position(game_state)
    hazards, horizon = _hazard_schedule(game_state, (start, BOMB_TIMER))
    own_blast = set(blast_positions(game_state["field"], start))
    blocked = _blocked_positions(game_state) | {start}
    safe_first_moves = 0
    shortest_escape = INF_TIME
    for dx, dy in DELTAS.values():
        first = (start[0] + dx, start[1] + dy)
        if not _in_bounds(game_state["field"], first) or game_state["field"][first] != 0:
            continue
        if first in blocked or 2 in hazards.get(first, set()):
            continue
        if _can_survive_from(game_state, first, 2, hazards, horizon, {start}):
            safe_first_moves += 1

    queue = deque([(start, 0)])
    visited = {(start, 1)}
    while queue:
        position, moves = queue.popleft()
        time = moves + 1
        if moves and position not in own_blast:
            shortest_escape = moves
            break
        if moves == BOMB_TIMER - 1:
            continue
        for dx, dy in DELTAS.values():
            candidate = (position[0] + dx, position[1] + dy)
            next_time = time + 1
            state = (candidate, next_time)
            if state in visited or not _in_bounds(game_state["field"], candidate):
                continue
            if game_state["field"][candidate] != 0 or candidate in blocked:
                continue
            if next_time in hazards.get(candidate, set()):
                continue
            visited.add(state)
            queue.append((candidate, moves + 1))
    return safe_first_moves, shortest_escape


def _can_escape_after_bomb(game_state: dict) -> bool:
    """Require an exit before the bomb explodes."""
    if not game_state["self"][2]:
        return False
    start = _position(game_state)
    hazards, horizon = _hazard_schedule(game_state, (start, BOMB_TIMER))
    if 1 in hazards.get(start, set()):
        return False
    if not _can_survive_from(game_state, start, 1, hazards, horizon, {start}):
        return False
    safe_first_moves, shortest_escape = _escape_options_after_bomb(game_state)
    return safe_first_moves >= 1 and shortest_escape <= BOMB_TIMER - 1


def legal_action_mask(game_state: dict) -> np.ndarray:
    """Allow moves with a survival route and bombs with an exit."""
    x, y = _position(game_state)
    physical_moves = []
    for action in MOVE_ACTIONS:
        physical_moves.append(_physical_destination(game_state, action) is not None)
    hazards, horizon = _hazard_schedule(game_state)
    moves = []
    destinations = []
    for action, physical in zip(MOVE_ACTIONS, physical_moves):
        destination = _physical_destination(game_state, action) if physical else None
        destinations.append(destination)
        moves.append(
            bool(destination is not None and _can_survive_from(
                game_state, destination, 1, hazards, horizon
            ))
        )
    wait_survives = _can_survive_from(game_state, (x, y), 1, hazards, horizon)
    danger = danger_time_map(game_state)
    low_risk_moves = [
        allowed and danger[destination] > 1
        for allowed, destination in zip(moves, destinations)
    ]
    if any(low_risk_moves):
        moves = low_risk_moves
    threatened = danger[x, y] < INF_TIME
    if threatened and any(moves):
        wait_survives = False
        opponent_positions = [tuple(other[3]) for other in game_state["others"]]
        contested = [
            allowed and any(abs(destination[0] - ox) + abs(destination[1] - oy) == 1
                            for ox, oy in opponent_positions)
            for allowed, destination in zip(moves, destinations)
        ]
        if any(allowed and not risky for allowed, risky in zip(moves, contested)):
            moves = [allowed and not risky for allowed, risky in zip(moves, contested)]
    if not any(moves) and not wait_survives:
        moves = physical_moves
        wait_survives = True
    wait_allowed = wait_survives
    bomb_allowed = danger[x, y] > 1 and _can_escape_after_bomb(game_state)
    return np.asarray([*moves, wait_allowed, bomb_allowed], dtype=bool)
