"""Rotations and reflections of a game state."""

from __future__ import annotations

import copy

import numpy as np

from .safety import ACTIONS, DELTAS


SYMMETRY_NAMES = (
    "identity", "rotate_90", "rotate_180", "rotate_270",
    "reflect", "reflect_rotate_90", "reflect_rotate_180", "reflect_rotate_270",
)
MATRICES = np.asarray(
    [((1, 0), (0, 1)), ((0, -1), (1, 0)), ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
     ((-1, 0), (0, 1)), ((0, -1), (-1, 0)), ((1, 0), (0, -1)), ((0, 1), (1, 0))],
    dtype=np.int8,
)


def _action_permutations() -> np.ndarray:
    by_delta = {delta: ACTIONS.index(action) for action, delta in DELTAS.items()}
    result = np.empty((len(SYMMETRY_NAMES), len(ACTIONS)), dtype=np.int64)
    for symmetry, matrix in enumerate(MATRICES):
        for action_index, action in enumerate(ACTIONS):
            result[symmetry, action_index] = (
                action_index if action not in DELTAS else by_delta[tuple(matrix @ DELTAS[action])]
            )
    return result


ACTION_PERMUTATIONS = _action_permutations()


def transform_position(position: tuple[int, int], size: int, symmetry: int) -> tuple[int, int]:
    centre = size // 2
    x, y = MATRICES[symmetry] @ (position[0] - centre, position[1] - centre)
    return int(x) + centre, int(y) + centre


def transform_array(array: np.ndarray, symmetry: int) -> np.ndarray:
    size = array.shape[0]
    x, y = np.indices((size, size))
    centre = size // 2
    coordinates = np.stack((x - centre, y - centre), axis=0).reshape(2, -1)
    mapped = MATRICES[symmetry] @ coordinates
    result = np.empty_like(array)
    result[(mapped[0] + centre).reshape(size, size), (mapped[1] + centre).reshape(size, size)] = array
    return result


def transform_game_state(game_state: dict, symmetry: int) -> dict:
    state = copy.copy(game_state)
    field = np.asarray(game_state["field"])
    size = field.shape[0]
    state["field"] = transform_array(field, symmetry)
    state["explosion_map"] = transform_array(np.asarray(game_state["explosion_map"]), symmetry)
    state["bombs"] = [
        (transform_position(tuple(position), size, symmetry), timer)
        for position, timer in game_state["bombs"]
    ]
    state["coins"] = [transform_position(tuple(position), size, symmetry) for position in game_state["coins"]]
    own = list(game_state["self"])
    own[3] = transform_position(tuple(own[3]), size, symmetry)
    state["self"] = tuple(own)
    others = []
    for other in game_state["others"]:
        transformed = list(other)
        transformed[3] = transform_position(tuple(transformed[3]), size, symmetry)
        others.append(tuple(transformed))
    state["others"] = others
    return state
