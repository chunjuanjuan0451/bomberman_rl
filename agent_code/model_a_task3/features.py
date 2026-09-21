"""Action-conditioned Task-3 features layered over the frozen v4 schema."""

from __future__ import annotations

import numpy as np

from agent_code.model_a_v6.features import (
    ACTIONS, BOMB_TIMER, DELTAS, INF_TIME, _blocked_positions,
    _hazard_schedule, _in_bounds, _path_distances, _physical_destination,
    _position, _survival_space, blast_positions, legal_action_mask,
    state_to_features,
)


ACTION_FEATURE_SIZE = 6


def _attack_tiles(field: np.ndarray, opponents: list[tuple[int, int]], blocked: set) -> set:
    """Free bomb origins whose blast line currently contains an opponent."""
    targets = set()
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            origin = (x, y)
            if field[origin] != 0 or origin in blocked:
                continue
            blast = set(blast_positions(field, origin))
            if any(opponent in blast for opponent in opponents):
                targets.add(origin)
    return targets


def _nearest_distance(field, origin, targets, blocked) -> int:
    if not targets:
        return INF_TIME
    distances = _path_distances(field, origin, blocked - {origin})
    return min((distances.get(target, INF_TIME) for target in targets), default=INF_TIME)


def _bomb_attack_value(game_state: dict, origin: tuple[int, int]) -> tuple[float, float]:
    """Return direct-line and opponent-escape-pressure values for a bomb origin."""
    field = game_state["field"]
    opponents = [tuple(other[3]) for other in game_state["others"]]
    blast = set(blast_positions(field, origin))
    exposed = [opponent for opponent in opponents if opponent in blast]
    if not exposed:
        return 0.0, 0.0
    static_blocked = {tuple(position) for position, _ in game_state["bombs"]}
    static_blocked |= set(opponents) | {origin}
    pressure = []
    for opponent in exposed:
        positions = {opponent}
        blocked = static_blocked - {opponent}
        for _ in range(BOMB_TIMER - 1):
            next_positions = set(positions)
            for position in positions:
                for dx, dy in DELTAS.values():
                    candidate = position[0] + dx, position[1] + dy
                    if candidate in blocked:
                        continue
                    if _in_bounds(field, candidate) and field[candidate] == 0:
                        next_positions.add(candidate)
            positions = next_positions
        safe_positions = positions - blast
        pressure.append(1.0 if not safe_positions else 1.0 / (1.0 + len(safe_positions)))
    return 1.0, float(max(pressure))


def task3_action_features(game_state: dict | None) -> np.ndarray:
    """Return six bounded values for every action, without choosing an action.

    Columns are progress toward a reachable attack tile, line-of-fire setup,
    opponent escape pressure, own survival space, coin progress, and crate yield.
    """
    values = np.zeros((len(ACTIONS), ACTION_FEATURE_SIZE), dtype=np.float32)
    if game_state is None:
        return values
    field = game_state["field"]
    start = _position(game_state)
    legal = legal_action_mask(game_state)
    opponents = [tuple(other[3]) for other in game_state["others"]]
    coins = [tuple(coin) for coin in game_state["coins"]]
    blocked = _blocked_positions(game_state)
    attack_tiles = _attack_tiles(field, opponents, blocked - {start})
    start_attack_distance = _nearest_distance(field, start, attack_tiles, blocked)
    start_coin_distance = _nearest_distance(field, start, set(coins), blocked)
    hazards, horizon = _hazard_schedule(game_state)
    bombs_available = bool(game_state["self"][2])

    for index, action in enumerate(ACTIONS):
        if not legal[index]:
            continue
        destination = start if action in ("WAIT", "BOMB") else _physical_destination(game_state, action)
        if destination is None:
            continue
        attack_distance = _nearest_distance(field, destination, attack_tiles, blocked)
        if start_attack_distance < INF_TIME and attack_distance < INF_TIME:
            values[index, 0] = float(np.clip(start_attack_distance - attack_distance, -1, 1))

        if bombs_available:
            direct, pressure = _bomb_attack_value(game_state, destination)
            values[index, 1] = direct
            values[index, 2] = pressure
            values[index, 5] = min(
                1.0,
                sum(field[position] == 1 for position in blast_positions(field, destination)) / 4.0,
            )

        if action == "BOMB":
            bomb_hazards, bomb_horizon = _hazard_schedule(game_state, (start, BOMB_TIMER))
            values[index, 3] = _survival_space(
                game_state, start, 1, bomb_hazards, bomb_horizon, {start},
            )
        else:
            values[index, 3] = _survival_space(game_state, destination, 1, hazards, horizon)

        coin_distance = _nearest_distance(field, destination, set(coins), blocked)
        if start_coin_distance < INF_TIME and coin_distance < INF_TIME:
            values[index, 4] = float(np.clip(start_coin_distance - coin_distance, -1, 1))
    return values
