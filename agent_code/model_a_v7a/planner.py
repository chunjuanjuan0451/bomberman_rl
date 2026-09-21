"""Deadline-aware, exact-hazard safety planning for Model A v7a.

The planner deliberately changes only safety decisions.  The frozen v4 Q
network remains responsible for collection and offence; this module resolves
escape emergencies and rejects routes that do not survive the complete known
bomb schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from agent_code.model_a_dqn.features import (
    ACTIONS,
    BOMB_TIMER,
    DELTAS,
    EXPLOSION_DURATION,
    blast_positions,
)

MIN_HORIZON = 6
MAX_HORIZON = 10
PATH_COUNT_CAP = 4096


@dataclass(frozen=True)
class SafetyProfile:
    """Completed search result for one first action."""

    survived: bool
    bottleneck: int
    terminal_positions: int
    terminal_routes: int
    terminal_mobility: int
    horizon: int

    @property
    def rank(self) -> tuple[int, int, int, int]:
        # Clip large open regions so safety does not overwhelm the learned
        # policy after an adequate amount of route redundancy is reached.
        return (
            int(self.survived),
            min(self.bottleneck, 3),
            min(self.terminal_positions, 10),
            min(self.terminal_mobility, 20),
        )


def _position(game_state: dict) -> tuple[int, int]:
    return tuple(game_state["self"][3])


def _in_bounds(field: np.ndarray, position: tuple[int, int]) -> bool:
    return 0 <= position[0] < field.shape[0] and 0 <= position[1] < field.shape[1]


def _bomb_schedule(
    game_state: dict, place_bomb: bool,
) -> tuple[list[tuple[tuple[int, int], int]], dict[tuple[int, int], set[int]], dict[tuple[int, int], int], int]:
    """Compile bomb occupancy, lethal offsets and crate clearing times.

    The local framework does not implement chain reactions and crates do not
    stop blast propagation.  A visible timer t explodes after action t+1; a
    bomb placed by the root BOMB action explodes on action offset 5.
    """
    field = game_state["field"]
    bombs = [(tuple(position), int(timer) + 1) for position, timer in game_state["bombs"]]
    if place_bomb:
        bombs.append((_position(game_state), BOMB_TIMER + 1))

    hazards: dict[tuple[int, int], set[int]] = {}
    for x, y in np.argwhere(np.asarray(game_state["explosion_map"]) > 0):
        hazards.setdefault((int(x), int(y)), set()).add(1)

    crate_clear: dict[tuple[int, int], int] = {}
    last_hazard = 1
    for origin, explosion_step in bombs:
        last_hazard = max(last_hazard, explosion_step + EXPLOSION_DURATION - 1)
        cells = blast_positions(field, origin)
        for cell in cells:
            hazards.setdefault(cell, set()).update(
                range(explosion_step, explosion_step + EXPLOSION_DURATION)
            )
            if field[cell] == 1:
                crate_clear[cell] = min(crate_clear.get(cell, explosion_step), explosion_step)
    return bombs, hazards, crate_clear, last_hazard


def _root_destination(game_state: dict, action: str) -> tuple[int, int] | None:
    start = _position(game_state)
    if action in ("WAIT", "BOMB"):
        if action == "BOMB" and not bool(game_state["self"][2]):
            return None
        return start
    dx, dy = DELTAS[action]
    destination = start[0] + dx, start[1] + dy
    field = game_state["field"]
    occupied = {tuple(position) for position, _ in game_state["bombs"]}
    occupied.update(tuple(other[3]) for other in game_state["others"])
    if not _in_bounds(field, destination) or field[destination] != 0 or destination in occupied:
        return None
    return destination


def _traversable(
    field: np.ndarray,
    candidate: tuple[int, int],
    previous: tuple[int, int],
    time: int,
    bombs: list[tuple[tuple[int, int], int]],
    crate_clear: dict[tuple[int, int], int],
) -> bool:
    if not _in_bounds(field, candidate) or field[candidate] == -1:
        return False
    if field[candidate] == 1 and time <= crate_clear.get(candidate, MAX_HORIZON + 1):
        return False
    for bomb_position, explosion_step in bombs:
        if candidate == bomb_position and time <= explosion_step and candidate != previous:
            return False
    return True


def profile_action(
    game_state: dict,
    action: str,
    deadline: float,
) -> SafetyProfile | None:
    """Exhaustively propagate all safe own paths for one root action.

    ``None`` means the action is physically impossible or the deadline was
    reached before a complete profile could be produced.
    """
    destination = _root_destination(game_state, action)
    if destination is None:
        return None
    bombs, hazards, crate_clear, last_hazard = _bomb_schedule(game_state, action == "BOMB")
    horizon = min(MAX_HORIZON, max(MIN_HORIZON, last_hazard + 1))
    if 1 in hazards.get(destination, set()):
        return SafetyProfile(False, 0, 0, 0, 0, horizon)

    field = game_state["field"]
    frontier: dict[tuple[int, int], int] = {destination: 1}
    bottleneck = MAX_HORIZON * field.size
    for time in range(2, horizon + 1):
        if perf_counter() >= deadline:
            return None
        next_frontier: dict[tuple[int, int], int] = {}
        for position, routes in frontier.items():
            candidates = [position]
            candidates.extend((position[0] + dx, position[1] + dy) for dx, dy in DELTAS.values())
            for candidate in candidates:
                if not _traversable(field, candidate, position, time, bombs, crate_clear):
                    continue
                if time in hazards.get(candidate, set()):
                    continue
                next_frontier[candidate] = min(
                    PATH_COUNT_CAP, next_frontier.get(candidate, 0) + routes,
                )
        if not next_frontier:
            return SafetyProfile(False, 0, 0, 0, 0, horizon)
        frontier = next_frontier
        bottleneck = min(bottleneck, len(frontier))

    mobility = 0
    for position in frontier:
        for dx, dy in ((0, 0), *DELTAS.values()):
            candidate = position[0] + dx, position[1] + dy
            if _traversable(field, candidate, position, horizon + 1, bombs, crate_clear):
                mobility += 1
    return SafetyProfile(
        True,
        bottleneck if bottleneck < MAX_HORIZON * field.size else 1,
        len(frontier),
        min(PATH_COUNT_CAP, sum(frontier.values())),
        mobility,
        horizon,
    )


def _q_choice(q_values: np.ndarray, indices: list[int], rng) -> int:
    values = q_values[indices]
    best = values.max()
    tied = [index for index in indices if np.isclose(q_values[index], best)]
    if len(tied) == 1:
        return tied[0]
    return int(rng.choice(tied))


def select_action(
    game_state: dict,
    q_values: np.ndarray,
    fallback_mask: np.ndarray,
    rng,
    deadline: float,
) -> tuple[int, dict[str, object]]:
    """Select an action, retaining a precomputed v4 fallback at all times."""
    fallback_indices = np.flatnonzero(fallback_mask).tolist()
    fallback = _q_choice(q_values, fallback_indices, rng) if fallback_indices else ACTIONS.index("WAIT")
    profiles: dict[int, SafetyProfile] = {}
    for index, action in enumerate(ACTIONS):
        profile = profile_action(game_state, action, deadline)
        if profile is None:
            if perf_counter() >= deadline:
                return fallback, {"fallback": True, "reason": "deadline"}
            continue
        profiles[index] = profile

    # v7a is a conservative post-processor: never re-enable an action rejected
    # by the battle-tested v4 mask.  The exact search may understand a crate
    # opening that the old mask does not, but exploiting that belongs in a
    # separate offensive experiment rather than this safety-only ablation.
    safe = [
        index for index, profile in profiles.items()
        if profile.survived and bool(fallback_mask[index])
    ]
    if not safe:
        return fallback, {"fallback": True, "reason": "no-proven-safe-action"}
    learned_choice = _q_choice(q_values, safe, rng)

    start = _position(game_state)
    _, hazards, _, _ = _bomb_schedule(game_state, False)
    # Do not start escape steering merely because a timer-4 bomb can
    # eventually hit this tile.  v4 is often intentionally finishing a useful
    # action before leaving; robustness intervention is reserved for the two
    # genuinely urgent action offsets.
    emergency = any(time <= 2 for time in hazards.get(start, ()))
    selected = learned_choice
    reason = "v4-q"
    if emergency:
        best_rank = max(profiles[index].rank for index in safe)
        if profiles[learned_choice].rank < best_rank:
            robust = [index for index in safe if profiles[index].rank == best_rank]
            selected = _q_choice(q_values, robust, rng)
            reason = "emergency-robustness"
    elif learned_choice == ACTIONS.index("BOMB"):
        bomb_profile = profiles[learned_choice]
        alternatives = [index for index in safe if index != learned_choice]
        # Veto only genuinely single-route bombs.  This deliberately avoids
        # the broad bomb suppression that hurt the earlier v4.5 experiment.
        if alternatives and bomb_profile.bottleneck <= 1 and bomb_profile.terminal_positions <= 1:
            selected = _q_choice(q_values, alternatives, rng)
            reason = "single-route-bomb-veto"

    return selected, {
        "fallback": False,
        "reason": reason,
        "safe_actions": tuple(ACTIONS[index] for index in safe),
        "profile": profiles[selected],
    }
