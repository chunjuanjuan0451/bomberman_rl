"""Full-board, short-history feature extraction for v9."""

from __future__ import annotations

import numpy as np

from agent_code.model_a_dqn.features import (
    ACTIONS, DELTAS, INF_TIME, _physical_destination, danger_time_map, legal_action_mask,
)

BOARD_SIZE = 17
SPATIAL_CHANNELS = 16
GLOBAL_SIZE = 8
HISTORY_LENGTH = 2


def dynamic_frame(game_state):
    shape = game_state["field"].shape
    frame = np.zeros((3, *shape), dtype=np.float32)
    frame[0][tuple(game_state["self"][3])] = 1.0
    for other in game_state["others"]:
        frame[1][tuple(other[3])] = 1.0
    for position, timer in game_state["bombs"]:
        frame[2][tuple(position)] = (int(timer) + 1) / 5.0
    return frame


def state_to_features(game_state, history=()):
    if game_state is None:
        return None
    field = np.asarray(game_state["field"])
    if field.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"v9 expects a {BOARD_SIZE}x{BOARD_SIZE} board")
    spatial = np.zeros((SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    spatial[0] = field == -1
    spatial[1] = field == 1
    for coin in game_state["coins"]:
        spatial[2][tuple(coin)] = 1.0
    current = dynamic_frame(game_state)
    spatial[3:6] = current
    spatial[6] = np.asarray(game_state["explosion_map"]) > 0
    danger = danger_time_map(game_state)
    spatial[7] = np.where(danger < INF_TIME, np.maximum(0.0, 1.0 - danger / 5.0), 0.0)
    past = list(history)[-HISTORY_LENGTH:]
    while len(past) < HISTORY_LENGTH:
        past.insert(0, current)
    spatial[8:11] = past[-1]
    spatial[11:14] = past[-2]
    coordinates = np.linspace(-1.0, 1.0, BOARD_SIZE, dtype=np.float32)
    spatial[14] = coordinates[:, None]
    spatial[15] = coordinates[None, :]

    position = tuple(game_state["self"][3])
    legal = legal_action_mask(game_state)
    adjacent_crate = any(
        field[position[0] + dx, position[1] + dy] == 1 for dx, dy in DELTAS.values()
    )
    score = float(game_state["self"][1])
    global_features = np.asarray([
        float(game_state["self"][2]), len(game_state["coins"]) / 9.0,
        len(game_state["others"]) / 3.0, min(1.0, game_state["step"] / 400.0),
        float(danger[position] <= 2), float(adjacent_crate), np.tanh(score / 10.0),
        float(legal[:5].sum()) / 5.0,
    ], dtype=np.float32)
    return spatial, global_features


def counterfactual_risk_labels(game_state):
    """Exact known-hazard labels; physically invalid actions are ignored."""
    physical = np.zeros(len(ACTIONS), dtype=bool)
    for index, action in enumerate(ACTIONS[:4]):
        physical[index] = _physical_destination(game_state, action) is not None
    physical[4] = True
    physical[5] = bool(game_state["self"][2])
    safe = legal_action_mask(game_state)
    return (physical & ~safe).astype(np.float32), physical
