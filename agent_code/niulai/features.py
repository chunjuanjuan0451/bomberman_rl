"""Board representation used by the CNN."""

from __future__ import annotations

import numpy as np

from .safety import ACTIONS, INF_TIME, danger_time_map


SPATIAL_SHAPE = (11, 33, 33)
CENTER = 16


def _canvas(position: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    return CENTER + cell[1] - position[1], CENTER + cell[0] - position[0]


def state_to_features(game_state: dict) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(game_state["field"])
    position = tuple(game_state["self"][3])
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
