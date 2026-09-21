"""Feature extraction and time-aware safe action masking for Model A.

The representation deliberately uses only information provided in the official
``game_state``.  It is local enough to be compact, while the danger channel
and the safe-bomb mask encode the two constraints that matter most in classic:
do not stand in an imminent blast and do not place a bomb without an exit.
"""

from __future__ import annotations

import numpy as np

LOCAL_SHAPE = (4, 7, 7)
GLOBAL_SIZE = 7
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
    """Cells hit by one bomb, matching the official environment semantics.

    In this framework crates are destroyed by a blast but do not block it;
    only stone walls stop propagation.
    """
    cells = [origin]
    for dx, dy in DELTAS.values():
        for distance in range(1, power + 1):
            candidate = (origin[0] + dx * distance, origin[1] + dy * distance)
            if not _in_bounds(field, candidate) or field[candidate] == -1:
                break
            cells.append(candidate)
    return cells


def danger_time_map(game_state: dict) -> np.ndarray:
    """Earliest explosion time for every cell; ``INF_TIME`` means no known risk."""
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
    """Return dangerous action offsets for all known explosions.

    A bomb shown with timer ``t`` explodes after action ``t + 1``.  The
    explosion is lethal on that action and the following one.  Existing
    explosions with a positive explosion-map value remain lethal after the
    current action.
    """
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
    """Return the destination of a physically executable move or wait."""
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
    """Search space-time states until every known explosion has passed."""
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
                # An agent may remain on a bomb it already occupies, but it
                # cannot enter any bomb tile after leaving it.
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
    """Return (safe first moves, shortest moves to leave the new blast)."""
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
        # Require leaving before the explosion action (time 5).
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


def _can_escape_after_bomb(game_state: dict, danger: np.ndarray | None = None) -> bool:
    """Require a survivable bomb escape with one action of timing slack.

    The full space-time route must survive every known explosion and leave the
    new bomb's blast within three moves.  A stricter two-route rule was tested
    but suppressed useful crate-clearing bombs too aggressively; dynamic exit
    contention is instead handled on each subsequent action.
    """
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


def state_to_features(game_state: dict | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a ``[4, 7, 7]`` local tensor and seven bounded global scalars.

    Channels are: static board (wall=-1, crate=1), blast urgency, bomb timer,
    and entities (coin=1, opponent=-1).  Global features preserve the original
    four inputs, followed by opponent proximity, confinement and whether a
    safe bomb placed here would directly hit an opponent.
    """
    if game_state is None:
        return None
    field = game_state["field"]
    x, y = _position(game_state)
    danger = danger_time_map(game_state)
    local = np.zeros(LOCAL_SHAPE, dtype=np.float32)
    for local_x, board_x in enumerate(range(x - 3, x + 4)):
        for local_y, board_y in enumerate(range(y - 3, y + 4)):
            if not _in_bounds(field, (board_x, board_y)):
                local[0, local_x, local_y] = -1.0
                continue
            tile = field[board_x, board_y]
            local[0, local_x, local_y] = float(tile)  # -1 wall, 0 free, 1 crate
            time = danger[board_x, board_y]
            if time < INF_TIME:
                local[1, local_x, local_y] = max(0.0, 1.0 - float(time) / 5.0)
    for (bomb_x, bomb_y), timer in game_state["bombs"]:
        if abs(bomb_x - x) <= 3 and abs(bomb_y - y) <= 3:
            local[2, bomb_x - x + 3, bomb_y - y + 3] = float(timer) / 4.0
    for coin_x, coin_y in game_state["coins"]:
        if abs(coin_x - x) <= 3 and abs(coin_y - y) <= 3:
            local[3, coin_x - x + 3, coin_y - y + 3] = 1.0
    for other in game_state["others"]:
        other_x, other_y = other[3]
        if abs(other_x - x) <= 3 and abs(other_y - y) <= 3:
            local[3, other_x - x + 3, other_y - y + 3] = -1.0
    legal = legal_action_mask(game_state)
    adjacent_crate = any(
        _in_bounds(field, (x + dx, y + dy)) and field[x + dx, y + dy] == 1
        for dx, dy in DELTAS.values()
    )
    opponents = [tuple(other[3]) for other in game_state["others"]]
    opponent_proximity = 0.0
    opponent_confinement = 0.0
    direct_attack = 0.0
    if opponents:
        nearest = min(opponents, key=lambda position: abs(position[0] - x) + abs(position[1] - y))
        distance = abs(nearest[0] - x) + abs(nearest[1] - y)
        opponent_proximity = max(0.0, 1.0 - min(distance, 12) / 12.0)
        occupied = _blocked_positions(game_state) | {(x, y)}
        free_exits = sum(
            _in_bounds(field, (nearest[0] + dx, nearest[1] + dy))
            and field[nearest[0] + dx, nearest[1] + dy] == 0
            and (nearest[0] + dx, nearest[1] + dy) not in occupied
            for dx, dy in DELTAS.values()
        )
        opponent_confinement = 1.0 - free_exits / 4.0
        if legal[5]:
            own_blast = set(blast_positions(field, (x, y)))
            direct_attack = float(any(position in own_blast for position in opponents))
    global_features = np.asarray(
        [float(game_state["self"][2]), float(danger[x, y] <= 1),
         min(1.0, len(game_state["coins"]) / 12.0), float(adjacent_crate or legal[5]),
         opponent_proximity, opponent_confinement, direct_attack],
        dtype=np.float32,
    )
    return local, global_features


def legal_action_mask(game_state: dict | None) -> np.ndarray:
    """Mask physically invalid actions and actions without a survival path.

    Each move is checked through the last known explosion in space-time.  When
    threatened, waiting is suppressed if moving preserves survival slack, and
    exits an opponent could enter simultaneously are avoided when an
    uncontested survivable exit exists.  Physical actions are retained as a
    last resort if the known state has no survivable continuation.
    """
    if game_state is None:
        return np.zeros(len(ACTIONS), dtype=bool)
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
    bomb_allowed = danger[x, y] > 1 and _can_escape_after_bomb(game_state, danger)
    return np.asarray([*moves, wait_allowed, bomb_allowed], dtype=bool)
