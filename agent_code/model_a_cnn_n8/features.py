"""Agent-centred full-board features for the clean CNN curriculum."""

from __future__ import annotations

from collections import deque

import numpy as np

from agent_code.model_a_dqn.features import (
    ACTIONS, DELTAS, INF_TIME, danger_time_map, legal_action_mask,
)


SPATIAL_SHAPE = (11, 33, 33)
SCALAR_SIZE = 6
CENTER = 16


def _canvas(position: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    """Return conventional (row=y, column=x) agent-centred coordinates."""
    return CENTER + cell[1] - position[1], CENTER + cell[0] - position[0]


def state_to_features(game_state: dict | None) -> tuple[np.ndarray, np.ndarray] | None:
    if game_state is None:
        return None
    field = np.asarray(game_state["field"])
    position = tuple(game_state["self"][3])
    spatial = np.zeros(SPATIAL_SHAPE, dtype=np.uint8)
    spatial[0] = 255  # Outside the translated board is a rigid wall.
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
    explosion = np.asarray(game_state["explosion_map"])
    for x, y in np.argwhere(explosion > 0):
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


def _resource_targets(game_state: dict) -> list[tuple[int, int]]:
    coins = [tuple(item) for item in game_state["coins"]]
    if coins:
        return coins
    field = np.asarray(game_state["field"])
    targets: set[tuple[int, int]] = set()
    for x, y in np.argwhere(field == 1):
        for dx, dy in DELTAS.values():
            candidate = (int(x) + dx, int(y) + dy)
            if 0 <= candidate[0] < field.shape[0] and 0 <= candidate[1] < field.shape[1] and field[candidate] == 0:
                targets.add(candidate)
    return sorted(targets)


def resource_distance(game_state: dict | None) -> int | None:
    """Shortest static-board distance to a visible coin or crate frontier."""
    if game_state is None:
        return None
    start = tuple(game_state["self"][3])
    targets = set(_resource_targets(game_state))
    if not targets:
        return None
    field = np.asarray(game_state["field"])
    blocked = {tuple(position) for position, _ in game_state["bombs"]}
    blocked.update(tuple(other[3]) for other in game_state["others"])
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        position, distance = queue.popleft()
        if position in targets:
            return distance
        for dx, dy in DELTAS.values():
            candidate = (position[0] + dx, position[1] + dy)
            if candidate in seen or candidate in blocked:
                continue
            if not (0 <= candidate[0] < field.shape[0] and 0 <= candidate[1] < field.shape[1]):
                continue
            if field[candidate] != 0:
                continue
            seen.add(candidate)
            queue.append((candidate, distance + 1))
    return None


def idle_wait(game_state: dict | None, action: str | None) -> bool:
    if game_state is None or action != "WAIT" or game_state["bombs"]:
        return False
    position = tuple(game_state["self"][3])
    if np.asarray(game_state["explosion_map"])[position] > 0:
        return False
    legal = legal_action_mask(game_state)
    return bool(np.any(legal[:4])) and int(danger_time_map(game_state)[position]) == INF_TIME


__all__ = [
    "ACTIONS", "CENTER", "SCALAR_SIZE", "SPATIAL_SHAPE", "idle_wait",
    "legal_action_mask", "resource_distance", "state_to_features",
]
