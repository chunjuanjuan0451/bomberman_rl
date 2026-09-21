"""Counterfactual opponent rollouts used by the v7b bomb gate."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from agent_code.model_a_dqn.features import ACTIONS, DELTAS, blast_positions
from agent_code.model_a_v7a.planner import _bomb_schedule, _traversable, profile_action


@dataclass(frozen=True)
class BombTactics:
    guaranteed_traps: int
    affected_opponents: int
    max_space_reduction: float
    crates_destroyed: int
    own_bottleneck: int
    own_terminal_positions: int


def _opponent_frontier(game_state: dict, start: tuple[int, int], place_bomb: bool, horizon: int):
    """Return reachable positions after every step, assuming optimal escape."""
    field = game_state["field"]
    bombs, hazards, crate_clear, _ = _bomb_schedule(game_state, place_bomb)
    frontier = {start}
    history = []
    for time in range(1, horizon + 1):
        next_frontier = set()
        for position in frontier:
            candidates = [position]
            candidates.extend((position[0] + dx, position[1] + dy) for dx, dy in DELTAS.values())
            for candidate in candidates:
                if not _traversable(field, candidate, position, time, bombs, crate_clear):
                    continue
                # On the root action the opponent cannot move into our still
                # occupied bomb tile. Afterwards our route is unknown, so
                # treating the tile as free is conservative for a trap claim.
                if time == 1 and candidate == tuple(game_state["self"][3]):
                    continue
                if time in hazards.get(candidate, set()):
                    continue
                next_frontier.add(candidate)
        frontier = next_frontier
        history.append(frontier)
        if not frontier:
            history.extend([set()] * (horizon - time))
            break
    return history


def evaluate_bomb(game_state: dict, deadline: float) -> BombTactics | None:
    """Compare every opponent's survival with and without a new own bomb."""
    if perf_counter() >= deadline or not bool(game_state["self"][2]):
        return None
    own_profile = profile_action(game_state, "BOMB", deadline)
    if own_profile is None or not own_profile.survived:
        return None

    origin = tuple(game_state["self"][3])
    own_blast = set(blast_positions(game_state["field"], origin))
    crates = sum(game_state["field"][position] == 1 for position in own_blast)
    horizon = own_profile.horizon
    guaranteed_traps = 0
    affected = 0
    reductions = []
    for other in game_state["others"]:
        if perf_counter() >= deadline:
            return None
        start = tuple(other[3])
        without = _opponent_frontier(game_state, start, False, horizon)
        with_bomb = _opponent_frontier(game_state, start, True, horizon)
        without_terminal = without[-1] if without else set()
        with_terminal = with_bomb[-1] if with_bomb else set()
        if without_terminal and not with_terminal:
            guaranteed_traps += 1
        # Offset five is when the new bomb explodes. Measuring only after the
        # full horizon hides meaningful containment because surviving paths
        # rapidly fan out again once the blast clears.
        critical_index = min(4, len(without) - 1)
        without_critical = without[critical_index] if without else set()
        with_critical = with_bomb[critical_index] if with_bomb else set()
        base_space = max(1, len(without_critical))
        reduction = max(0.0, 1.0 - len(with_critical) / base_space)
        reductions.append(reduction)
        if start in own_blast or reduction > 0.0:
            affected += 1
    return BombTactics(
        guaranteed_traps=guaranteed_traps,
        affected_opponents=affected,
        max_space_reduction=max(reductions, default=0.0),
        crates_destroyed=int(crates),
        own_bottleneck=own_profile.bottleneck,
        own_terminal_positions=own_profile.terminal_positions,
    )


def maybe_override_with_bomb(
    game_state: dict,
    baseline_action: int,
    fallback_mask,
    deadline: float,
) -> tuple[int, BombTactics | None, str]:
    """Add BOMB only for a high-confidence trap; never suppress v4 bombs."""
    bomb_index = ACTIONS.index("BOMB")
    if baseline_action == bomb_index or not bool(fallback_mask[bomb_index]):
        return baseline_action, None, "v4"
    tactics = evaluate_bomb(game_state, deadline)
    if tactics is None:
        return baseline_action, None, "deadline-or-unsafe"
    robust_escape = tactics.own_bottleneck >= 2 and tactics.own_terminal_positions >= 3
    if tactics.guaranteed_traps >= 1 and robust_escape:
        return bomb_index, tactics, "guaranteed-trap"
    return baseline_action, tactics, "no-high-confidence-trap"
