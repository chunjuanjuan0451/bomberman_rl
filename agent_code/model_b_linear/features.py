"""Deterministic, BFS-based state features for the Model B linear Q-learner."""

from __future__ import annotations

from collections import deque

import numpy as np

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
MOVE_ACTIONS = ACTIONS[:4]
DELTAS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}

# 6 legal-action bits, 4 coin-direction scores, 4 escape-area scores, and 5 globals.
FEATURE_SIZE = 19
INF_TIME = 99
MAX_ESCAPE_STEPS = 5


def _position(game_state: dict) -> tuple[int, int]:
    return tuple(game_state["self"][3])


def _blocked_positions(game_state: dict) -> set[tuple[int, int]]:
    bombs = {tuple(position) for position, _ in game_state["bombs"]}
    opponents = {tuple(other[3]) for other in game_state["others"]}
    return bombs | opponents


def _in_bounds(field: np.ndarray, position: tuple[int, int]) -> bool:
    x, y = position
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def _blast_positions(field: np.ndarray, origin: tuple[int, int], power: int = 3) -> list[tuple[int, int]]:
    """Return blast cells, stopping only at stone walls like the environment."""
    x, y = origin
    positions = [origin]
    for dx, dy in DELTAS.values():
        for distance in range(1, power + 1):
            candidate = (x + dx * distance, y + dy * distance)
            if not _in_bounds(field, candidate) or field[candidate] == -1:
                break
            positions.append(candidate)
    return positions


def danger_time_map(game_state: dict) -> np.ndarray:
    """Earliest known explosion time per cell; ``0`` means exploding now."""
    field = game_state["field"]
    danger = np.full(field.shape, INF_TIME, dtype=np.int16)
    danger[np.asarray(game_state["explosion_map"]) > 0] = 0
    for position, timer in game_state["bombs"]:
        for blast_cell in _blast_positions(field, tuple(position)):
            danger[blast_cell] = min(danger[blast_cell], int(timer))
    return danger


def legal_action_mask(game_state: dict | None) -> np.ndarray:
    """Physical legality mask in ``ACTIONS`` order; danger remains learnable."""
    if game_state is None:
        return np.zeros(len(ACTIONS), dtype=bool)
    field = game_state["field"]
    x, y = _position(game_state)
    blocked = _blocked_positions(game_state)
    legal = []
    for action in MOVE_ACTIONS:
        dx, dy = DELTAS[action]
        candidate = (x + dx, y + dy)
        legal.append(_in_bounds(field, candidate) and field[candidate] == 0 and candidate not in blocked)
    legal.extend((True, bool(game_state["self"][2])))
    return np.asarray(legal, dtype=bool)


def _safe_traversable(game_state: dict, position: tuple[int, int], arrival_step: int, danger: np.ndarray) -> bool:
    field = game_state["field"]
    if not _in_bounds(field, position) or field[position] != 0 or position in _blocked_positions(game_state):
        return False
    return danger[position] > arrival_step


def _shortest_safe_coin_distance_from(
    game_state: dict, start: tuple[int, int], danger: np.ndarray, start_arrival: int = 0
) -> int | None:
    """BFS distance to a coin after arriving at ``start`` at a known time."""
    if not game_state["coins"]:
        return None
    targets = {tuple(coin) for coin in game_state["coins"]}
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, start_arrival)])
    visited = {start}
    while queue:
        position, arrival = queue.popleft()
        if position in targets:
            return arrival - start_arrival
        for dx, dy in DELTAS.values():
            candidate = (position[0] + dx, position[1] + dy)
            next_arrival = arrival + 1
            if candidate not in visited and _safe_traversable(game_state, candidate, next_arrival, danger):
                visited.add(candidate)
                queue.append((candidate, next_arrival))
    return None


def shortest_safe_coin_distance(game_state: dict | None) -> int | None:
    """Shortest time-aware BFS distance to a currently visible coin."""
    if game_state is None:
        return None
    return _shortest_safe_coin_distance_from(game_state, _position(game_state), danger_time_map(game_state))


def _safe_area_size(game_state: dict, start: tuple[int, int], danger: np.ndarray) -> int:
    if not _safe_traversable(game_state, start, 1, danger):
        return 0
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 1)])
    visited = {start}
    while queue:
        position, distance = queue.popleft()
        if distance >= MAX_ESCAPE_STEPS:
            continue
        for dx, dy in DELTAS.values():
            candidate = (position[0] + dx, position[1] + dy)
            next_distance = distance + 1
            if candidate not in visited and _safe_traversable(game_state, candidate, next_distance, danger):
                visited.add(candidate)
                queue.append((candidate, next_distance))
    return len(visited)


def _can_safely_bomb(game_state: dict, danger: np.ndarray, legal: np.ndarray) -> float:
    """Approximate whether a bomb can be placed while preserving an escape route."""
    if not game_state["self"][2]:
        return 0.0
    simulated_danger = danger.copy()
    simulated_danger[_position(game_state)] = min(simulated_danger[_position(game_state)], 4)
    x, y = _position(game_state)
    for index, action in enumerate(MOVE_ACTIONS):
        if legal[index]:
            dx, dy = DELTAS[action]
            if _safe_area_size(game_state, (x + dx, y + dy), simulated_danger) >= 2:
                return 1.0
    return 0.0


def _max_board_distance(field: np.ndarray) -> float:
    return float(max(1, field.shape[0] + field.shape[1] - 2))


def state_to_features(game_state: dict | None) -> np.ndarray | None:
    """Return a fixed ``float32`` vector for Model B's linear action values."""
    if game_state is None:
        return None
    field = game_state["field"]
    position = _position(game_state)
    danger = danger_time_map(game_state)
    legal = legal_action_mask(game_state).astype(np.float32)
    coin_scores, escape_scores = [], []
    for index, action in enumerate(MOVE_ACTIONS):
        dx, dy = DELTAS[action]
        candidate = (position[0] + dx, position[1] + dy)
        if not legal[index] or not _safe_traversable(game_state, candidate, 1, danger):
            coin_scores.append(0.0)
            escape_scores.append(0.0)
        else:
            distance = _shortest_safe_coin_distance_from(game_state, candidate, danger, start_arrival=1)
            coin_scores.append(0.0 if distance is None else 1.0 - min(distance, _max_board_distance(field)) / _max_board_distance(field))
            escape_scores.append(min(1.0, _safe_area_size(game_state, candidate, danger) / 12.0))

    nearest_coin = shortest_safe_coin_distance(game_state)
    max_distance = _max_board_distance(field)
    nearest_coin_score = 0.0 if nearest_coin is None else 1.0 - min(nearest_coin, max_distance) / max_distance
    adjacent_crate = float(any(
        _in_bounds(field, (position[0] + dx, position[1] + dy))
        and field[position[0] + dx, position[1] + dy] == 1
        for dx, dy in DELTAS.values()
    ))
    globals_ = np.asarray(
        [float(danger[position] <= 1), _can_safely_bomb(game_state, danger, legal), nearest_coin_score,
         adjacent_crate, min(1.0, len(game_state["coins"]) / 9.0)], dtype=np.float32,
    )
    return np.concatenate((legal, np.asarray(coin_scores, dtype=np.float32),
                           np.asarray(escape_scores, dtype=np.float32), globals_)).astype(np.float32)
