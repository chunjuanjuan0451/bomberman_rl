"""Resource-only full-map features; no opponent, attack, target, or history labels."""

from __future__ import annotations

import numpy as np

from agent_code.model_a_dqn.features import state_to_features as v4_state_to_features
from .config import RESOURCE_SHAPE


BOARD = 33
CENTRE = BOARD // 2
LOCAL_RADIUS = 3


def resource_features(game_state: dict | None, arm: str) -> np.ndarray | None:
    """Return wall-padded, agent-centred field and visible-coin channels.

    Both arms have exactly the same shape. ``local7`` masks every observer-board
    cell outside radius three; ``full33`` retains the complete 17x17 board.
    """
    if game_state is None:
        return None
    if arm not in ("local7", "full33", "frozen-v4"):
        raise ValueError(f"unknown resource feature arm: {arm}")
    field = np.asarray(game_state["field"])
    px, py = map(int, game_state["self"][3])
    out = np.zeros(RESOURCE_SHAPE, dtype=np.float32)
    out[0].fill(-1.0)
    offset_x, offset_y = CENTRE - px, CENTRE - py
    local_only = arm in ("local7", "frozen-v4")
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            rx, ry = x - px, y - py
            if local_only and (abs(rx) > LOCAL_RADIUS or abs(ry) > LOCAL_RADIUS):
                continue
            out[0, x + offset_x, y + offset_y] = float(field[x, y])
    for x, y in game_state["coins"]:
        rx, ry = int(x) - px, int(y) - py
        if local_only and (abs(rx) > LOCAL_RADIUS or abs(ry) > LOCAL_RADIUS):
            continue
        out[1, int(x) + offset_x, int(y) + offset_y] = 1.0
    return out


def combined_features(game_state: dict | None, arm: str):
    base = v4_state_to_features(game_state)
    resource = resource_features(game_state, arm)
    if base is None or resource is None:
        return None
    return base[0], base[1], resource
