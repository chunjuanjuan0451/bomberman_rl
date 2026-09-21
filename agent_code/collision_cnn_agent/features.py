"""Self-contained full-board features and survival/collision action masks."""

from __future__ import annotations

from collections import deque

import numpy as np


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
MOVE_ACTIONS = ACTIONS[:4]
DELTAS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
INF_TIME = 99
BLAST_POWER = 3
BOMB_TIMER = 4
EXPLOSION_DURATION = 2
SPATIAL_SHAPE = (11, 33, 33)
CENTER = 16


def _position(game_state: dict) -> tuple[int, int]:
    return tuple(game_state["self"][3])


def _in_bounds(field: np.ndarray, position: tuple[int, int]) -> bool:
    x, y = position
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def _blocked_positions(game_state: dict) -> set[tuple[int, int]]:
    return {tuple(position) for position, _ in game_state["bombs"]} | {
        tuple(other[3]) for other in game_state["others"]
    }


def blast_positions(field: np.ndarray, origin: tuple[int, int]) -> list[tuple[int, int]]:
    cells = [origin]
    for dx, dy in DELTAS.values():
        for distance in range(1, BLAST_POWER + 1):
            cell = (origin[0] + dx * distance, origin[1] + dy * distance)
            if not _in_bounds(field, cell) or field[cell] == -1:
                break
            cells.append(cell)
    return cells


def danger_time_map(game_state: dict) -> np.ndarray:
    field = np.asarray(game_state["field"])
    danger = np.full(field.shape, INF_TIME, dtype=np.int16)
    danger[np.asarray(game_state["explosion_map"]) > 0] = 0
    for position, timer in game_state["bombs"]:
        for cell in blast_positions(field, tuple(position)):
            danger[cell] = min(danger[cell], int(timer))
    return danger


def _hazard_schedule(game_state: dict, extra_bomb=None):
    field = np.asarray(game_state["field"])
    hazards: dict[tuple[int, int], set[int]] = {}
    for x, y in np.argwhere(np.asarray(game_state["explosion_map"]) > 0):
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


def _physical_destination(game_state: dict, action: str, position=None):
    position = _position(game_state) if position is None else position
    if action == "WAIT":
        return position
    dx, dy = DELTAS[action]
    candidate = (position[0] + dx, position[1] + dy)
    field = np.asarray(game_state["field"])
    if not _in_bounds(field, candidate) or field[candidate] != 0:
        return None
    if candidate in _blocked_positions(game_state):
        return None
    return candidate


def _can_survive_from(game_state: dict, start, start_time: int, hazards, horizon: int, extra_blocked=None) -> bool:
    if start_time in hazards.get(start, set()):
        return False
    field = np.asarray(game_state["field"])
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
    start = _position(game_state)
    hazards, horizon = _hazard_schedule(game_state, (start, BOMB_TIMER))
    own_blast = set(blast_positions(np.asarray(game_state["field"]), start))
    blocked = _blocked_positions(game_state) | {start}
    safe_first_moves = 0
    for dx, dy in DELTAS.values():
        first = (start[0] + dx, start[1] + dy)
        field = np.asarray(game_state["field"])
        if not _in_bounds(field, first) or field[first] != 0 or first in blocked:
            continue
        if 2 in hazards.get(first, set()):
            continue
        if _can_survive_from(game_state, first, 2, hazards, horizon, {start}):
            safe_first_moves += 1
    shortest_escape = INF_TIME
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
            field = np.asarray(game_state["field"])
            if state in visited or not _in_bounds(field, candidate):
                continue
            if field[candidate] != 0 or candidate in blocked or next_time in hazards.get(candidate, set()):
                continue
            visited.add(state)
            queue.append((candidate, moves + 1))
    return safe_first_moves, shortest_escape


def _can_escape_after_bomb(game_state: dict) -> bool:
    if not bool(game_state["self"][2]):
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
    x, y = _position(game_state)
    physical_moves = [_physical_destination(game_state, action) is not None for action in MOVE_ACTIONS]
    hazards, horizon = _hazard_schedule(game_state)
    destinations = [_physical_destination(game_state, action) if physical else None
                    for action, physical in zip(MOVE_ACTIONS, physical_moves)]
    moves = [bool(destination is not None and _can_survive_from(game_state, destination, 1, hazards, horizon))
             for destination in destinations]
    wait_survives = _can_survive_from(game_state, (x, y), 1, hazards, horizon)
    danger = danger_time_map(game_state)
    low_risk = [allowed and danger[destination] > 1
                for allowed, destination in zip(moves, destinations)]
    if any(low_risk):
        moves = low_risk
    if danger[x, y] < INF_TIME and any(moves):
        wait_survives = False
        opponent_positions = [tuple(other[3]) for other in game_state["others"]]
        contested = [allowed and any(abs(destination[0] - ox) + abs(destination[1] - oy) == 1
                                     for ox, oy in opponent_positions)
                     for allowed, destination in zip(moves, destinations)]
        if any(allowed and not risky for allowed, risky in zip(moves, contested)):
            moves = [allowed and not risky for allowed, risky in zip(moves, contested)]
    if not any(moves) and not wait_survives:
        moves = physical_moves
        wait_survives = True
    bomb_allowed = danger[x, y] > 1 and _can_escape_after_bomb(game_state)
    return np.asarray([*moves, wait_survives, bomb_allowed], dtype=bool)


def _opponent_reachability(game_state: dict) -> set[tuple[int, int]]:
    field = np.asarray(game_state["field"])
    blocked = _blocked_positions(game_state)
    positions = {tuple(other[3]) for other in game_state["others"]}
    reachable = set()
    for position in positions:
        candidates = [position]
        candidates.extend((position[0] + dx, position[1] + dy) for dx, dy in DELTAS.values())
        for candidate in candidates:
            if not _in_bounds(field, candidate) or field[candidate] != 0:
                continue
            if candidate in blocked and candidate != position:
                continue
            reachable.add(candidate)
    return reachable


def collision_filtered_mask(game_state: dict, base_mask: np.ndarray) -> np.ndarray:
    """Apply the s159 one-step collision filter while the own bomb is active."""
    if bool(game_state["self"][2]):
        return base_mask
    hazards, horizon = _hazard_schedule(game_state)
    opponent_reach = _opponent_reachability(game_state)
    robust = np.zeros(len(ACTIONS), dtype=bool)
    for index, action in enumerate(ACTIONS[:5]):
        destination = _physical_destination(game_state, action)
        if destination is None:
            continue
        if 1 in hazards.get(destination, set()) or destination in opponent_reach:
            continue
        if _can_survive_from(game_state, destination, 1, hazards, horizon):
            robust[index] = True
    filtered = base_mask & robust
    return filtered if filtered.any() else base_mask


def _canvas(position: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    return CENTER + cell[1] - position[1], CENTER + cell[0] - position[0]


def state_to_features(game_state: dict) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(game_state["field"])
    position = _position(game_state)
    spatial = np.zeros(SPATIAL_SHAPE, dtype=np.uint8)
    spatial[0] = 255
    danger = danger_time_map(game_state)
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            row, column = _canvas(position, (x, y))
            spatial[0, row, column] = 255 if field[x, y] == -1 else 0
            spatial[1, row, column] = 255 if field[x, y] == 1 else 0
            if danger[x, y] < INF_TIME:
                urgency = max(0.0, min(1.0, (5.0 - float(danger[x, y])) / 5.0))
                spatial[10, row, column] = int(round(255.0 * urgency))
    for coin in game_state["coins"]:
        row, column = _canvas(position, tuple(coin))
        spatial[2, row, column] = 255
    spatial[3, CENTER, CENTER] = 255
    for other in game_state["others"]:
        row, column = _canvas(position, tuple(other[3]))
        spatial[4, row, column] = 255
    for bomb_position, timer in game_state["bombs"]:
        channel = 5 + max(1, min(4, int(timer))) - 1
        row, column = _canvas(position, tuple(bomb_position))
        spatial[channel, row, column] = 255
    for x, y in np.argwhere(np.asarray(game_state["explosion_map"]) > 0):
        row, column = _canvas(position, (int(x), int(y)))
        spatial[9, row, column] = 255
    current_danger = int(danger[position])
    danger_scalar = 0.0 if current_danger >= INF_TIME else max(0.0, (5.0 - current_danger) / 5.0)
    scalars = np.asarray([
        float(bool(game_state["self"][2])),
        min(1.0, float(game_state.get("step", 0)) / 400.0),
        float(np.tanh(float(game_state["self"][1]) / 10.0)),
        min(1.0, len(game_state["coins"]) / 20.0),
        min(1.0, len(game_state["others"]) / 3.0),
        danger_scalar,
    ], dtype=np.float32)
    return spatial, scalars
