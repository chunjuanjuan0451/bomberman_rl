"""Exact D4 transforms for random single-view replay augmentation."""

from __future__ import annotations

import copy

import numpy as np

from .features import ACTIONS, DELTAS


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
    if array.ndim != 2 or array.shape[1] != size:
        raise ValueError("D4 transform requires a square 2D array")
    x, y = np.indices((size, size))
    centre = size // 2
    coordinates = np.stack((x - centre, y - centre), axis=0).reshape(2, -1)
    mapped = MATRICES[symmetry] @ coordinates
    result = np.empty_like(array)
    result[(mapped[0] + centre).reshape(size, size), (mapped[1] + centre).reshape(size, size)] = array
    return result


def transform_local(local: np.ndarray, symmetry: int) -> np.ndarray:
    result = np.empty_like(local)
    for channel in range(local.shape[-3]):
        result[..., channel, :, :] = transform_array(local[..., channel, :, :], symmetry)
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


def augment_transition_arrays_random(
    local, global_features, tactical, actions, rewards, dones,
    next_local, next_global, next_tactical, next_masks, rng,
):
    """Apply one independently sampled D4 view per transition."""
    symmetries = rng.integers(len(SYMMETRY_NAMES), size=len(local))
    out_local = np.empty_like(local)
    out_tactical = np.empty_like(tactical)
    out_actions = np.empty_like(actions)
    out_next_local = np.empty_like(next_local)
    out_next_tactical = np.empty_like(next_tactical)
    out_masks = np.empty_like(next_masks)
    for index, symmetry in enumerate(symmetries):
        permutation = ACTION_PERMUTATIONS[symmetry]
        out_local[index] = transform_local(local[index], int(symmetry))
        out_tactical[index, permutation] = tactical[index]
        out_actions[index] = permutation[actions[index]]
        out_next_local[index] = transform_local(next_local[index], int(symmetry))
        out_next_tactical[index, permutation] = next_tactical[index]
        out_masks[index, permutation] = next_masks[index]
    return (
        out_local, global_features, out_tactical, out_actions, rewards, dones,
        out_next_local, next_global, out_next_tactical, out_masks,
    )
