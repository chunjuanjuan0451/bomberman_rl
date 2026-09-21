"""Root-only classic-mode tactical planner.

The planner deliberately stays independent of learned weights.  It uses the
official simulator for a short time-expanded survival proof, then combines
that safety mask with simple coin, crate and opponent objectives.  It is only
called once per real action, never at every MCTS node.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp

import numpy as np

from .interfaces import MOVES, legal_actions
from .simulator import BOMB_POWER, SimState, _blast_coords, step


@dataclass(frozen=True, slots=True)
class PlannerOutput:
    prior: dict[str, float]
    safe_actions: tuple[str, ...]
    action_scores: dict[str, float]
    survival_width: dict[str, int]
    bomb_utility: float
    tactical: bool
    crate_target: tuple[int, int] | None
    crate_target_distance: int | None


def _joint_wait_step(state: SimState, player: int, action: str) -> SimState:
    actions = ["WAIT"] * len(state.agents)
    actions[player] = action
    return step(state, actions, tuple(range(len(state.active_ids()))))


def _dynamic_signature(state: SimState, player: int) -> tuple[object, ...]:
    agent = state.agents[player]
    bombs = tuple(sorted((bomb.x, bomb.y, bomb.owner, bomb.timer, bomb.power)
                         for bomb in state.bombs))
    explosions = tuple((explosion.coords, explosion.owner, explosion.timer, explosion.stage)
                       for explosion in state.explosions)
    return (agent.x, agent.y, agent.bombs_left, bombs, explosions,
            state.field.tobytes(), state.ended)


def _open_tile(state: SimState, position: tuple[int, int], player: int) -> bool:
    x, y = position
    if state.field[x, y] != 0:
        return False
    if any((bomb.x, bomb.y) == position for bomb in state.bombs):
        return False
    return not any(index != player and other.alive and (other.x, other.y) == position
                   for index, other in enumerate(state.agents))


def _distance_map(state: SimState, targets: set[tuple[int, int]], player: int) -> dict[tuple[int, int], int]:
    """Multi-source shortest paths over currently traversable tiles."""
    distances: dict[tuple[int, int], int] = {}
    queue: deque[tuple[int, int]] = deque()
    for target in targets:
        if _open_tile(state, target, player):
            distances[target] = 0
            queue.append(target)
    while queue:
        x, y = queue.popleft()
        for dx, dy, _ in MOVES.values():
            position = (x + dx, y + dy)
            if position not in distances and _open_tile(state, position, player):
                distances[position] = distances[(x, y)] + 1
                queue.append(position)
    return distances


def _crate_targets(state: SimState, player: int) -> set[tuple[int, int]]:
    """Free squares from which a radius-three bomb can hit a crate."""
    crates = set(map(tuple, np.argwhere(state.field == 1)))
    targets: set[tuple[int, int]] = set()
    for x, y in map(tuple, np.argwhere(state.field == 0)):
        position = (int(x), int(y))
        if (_open_tile(state, position, player)
                and crates.intersection(_blast_coords(state.field, *position, BOMB_POWER))):
            targets.add(position)
    return targets


def _bomb_yields(state: SimState, player: int) -> dict[tuple[int, int], int]:
    """Map every reachable bomb square to the crates hit by its blast."""
    result: dict[tuple[int, int], int] = {}
    for position in _crate_targets(state, player):
        result[position] = sum(
            state.field[coord] == 1
            for coord in _blast_coords(state.field, *position, BOMB_POWER))
    return result


class ClassicTacticalPlanner:
    """Short-horizon deterministic safety plus classic-mode target scoring.

    Opponents wait in the survival proof.  Their one-step reachable squares
    are separately penalised, so ``safe_actions`` means simulator-safe under
    the documented model rather than an adversarial guarantee.
    """

    def __init__(self, *, horizon: int = 8, prior_weight: float = 0.80,
                 temperature: float = 0.85, crate_frontier_weight: float = 0.32,
                 bomb_efficiency_penalty: float = 0.0):
        if horizon < 5:
            raise ValueError("classic planner horizon must be at least 5")
        if not 0.0 <= prior_weight <= 1.0:
            raise ValueError("classic planner prior_weight must be in [0, 1]")
        if temperature <= 0.0:
            raise ValueError("classic planner temperature must be positive")
        if crate_frontier_weight < 0.0 or bomb_efficiency_penalty < 0.0:
            raise ValueError("crate-frontier parameters must be non-negative")
        self.horizon = int(horizon)
        self.prior_weight = float(prior_weight)
        self.temperature = float(temperature)
        self.crate_frontier_weight = float(crate_frontier_weight)
        self.bomb_efficiency_penalty = float(bomb_efficiency_penalty)

    def _survival_width(self, state: SimState, player: int, first_action: str) -> int:
        successor = _joint_wait_step(state, player, first_action)
        if not successor.agents[player].alive:
            return 0
        return self._survival_after(successor, player, self.horizon - 1)

    @staticmethod
    def _survival_after(state: SimState, player: int, steps: int) -> int:
        """Count distinct surviving frontier states after ``steps`` plies."""
        successor = state
        if successor.ended:
            return 1
        frontier = [successor]
        for _ in range(steps):
            next_states: dict[tuple[object, ...], SimState] = {}
            for current in frontier:
                for action in legal_actions(current, player):
                    if action == "BOMB":
                        continue
                    candidate = _joint_wait_step(current, player, action)
                    if not candidate.agents[player].alive:
                        continue
                    if candidate.ended:
                        return max(1, len(next_states))
                    next_states.setdefault(_dynamic_signature(candidate, player), candidate)
            if not next_states:
                return 0
            frontier = list(next_states.values())
        return len(frontier)

    @staticmethod
    def _destination(state: SimState, player: int, action: str) -> tuple[int, int]:
        agent = state.agents[player]
        if action not in MOVES:
            return agent.x, agent.y
        dx, dy, _ = MOVES[action]
        return agent.x + dx, agent.y + dy

    def _bomb_utility(self, state: SimState, player: int, best_yield: int = 0,
                      prefer_frontier: bool = False) -> float:
        agent = state.agents[player]
        blast = set(_blast_coords(state.field, agent.x, agent.y, BOMB_POWER))
        crates = sum(state.field[position] == 1 for position in blast)
        # Do not withhold a productive local bomb merely because a denser
        # frontier exists somewhere else on the map.  That caused the teacher
        # to shuttle between distant equal-valued frontiers without clearing
        # either.  The efficiency penalty is meaningful only once that better
        # frontier is already nearby and reachable.
        shortfall = max(0, best_yield - crates) if prefer_frontier else 0
        utility = 1.8 * crates - self.bomb_efficiency_penalty * shortfall
        planted = _joint_wait_step(state, player, "BOMB")
        for index, other in enumerate(state.agents):
            if index == player or not other.alive or (other.x, other.y) not in blast:
                continue
            escape_width = self._survival_after(planted, index, self.horizon - 1)
            if escape_width == 0:
                utility += 4.5
            elif escape_width <= 2:
                utility += 0.35
        return float(utility)

    def select_persistent_crate_target(
        self,
        state: SimState,
        player: int,
        previous_target: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        """Retain a reachable useful bomb square, otherwise choose one deterministically."""
        yields = _bomb_yields(state, player)
        origin = (state.agents[player].x, state.agents[player].y)
        if previous_target in yields:
            previous_distance = _distance_map(state, {previous_target}, player)
            if origin in previous_distance:
                return previous_target
        origin_distance = _distance_map(state, {origin}, player)
        reachable = [target for target in yields if target in origin_distance]
        if not reachable:
            return None
        return min(reachable, key=lambda target: (
            -yields[target], origin_distance[target], target[0], target[1]))

    def plan(self, state: SimState, player: int, base_policy: dict[str, float],
             crate_target: tuple[int, int] | None = None) -> PlannerOutput:
        legal = tuple(action for action in legal_actions(state, player) if action in base_policy)
        if not legal:
            # Match the searcher's historical terminal-state sentinel.  The
            # official engine never requests an action from a dead agent, but
            # keeping the transform normalized makes diagnostic calls safe.
            return PlannerOutput({"WAIT": 1.0}, (), {}, {}, 0.0, True, None, None)

        widths = {action: self._survival_width(state, player, action) for action in legal}
        proven_safe = tuple(action for action in legal if widths[action] > 0)
        allowed = proven_safe or legal
        agent = state.agents[player]
        origin = (agent.x, agent.y)
        coins = {position for position, visible in state.coins.items() if visible}
        coin_distance = _distance_map(state, coins, player)
        yields = _bomb_yields(state, player)
        best_yield = max(yields.values(), default=0)
        # A one-crate blast can be sensible only after the dense frontiers are
        # exhausted.  Selecting positions within one crate of the maximum
        # gives an efficient local frontier while retaining isolated endgame
        # crates when max_yield falls to one.
        high_yield = max(1, best_yield - 1)
        frontier_targets = {position for position, value in yields.items()
                            if value >= high_yield}
        all_crate_distance = _distance_map(state, set(yields), player)
        frontier_distance = _distance_map(state, frontier_targets, player)
        # A global distance potential is unstable on symmetric boards: after
        # one step, another far frontier can become the equally closest one,
        # producing an UP/DOWN shuttle.  Prefer a dense frontier only inside
        # a short local planning window; otherwise make guaranteed progress
        # toward the nearest useful bomb square.
        frontier_nearby = frontier_distance.get(origin, 99) <= 4
        # A single deterministic frontier anchor prevents the nearest-target
        # field from changing sides after every move on a symmetric board.
        # Use manhattan distance only for choosing between equal-yield anchors;
        # the actual movement potential remains the exact wall-aware map.
        active_crate_target = None
        target_distance = None
        persistent_distance = (_distance_map(state, {crate_target}, player)
                               if crate_target in yields else {})
        if crate_target is not None and origin in persistent_distance:
            crate_distance = persistent_distance
            active_crate_target = crate_target
            target_distance = persistent_distance[origin]
        elif self.crate_frontier_weight > 0.32:
            anchor = max(frontier_targets, key=lambda position: (
                yields[position], -abs(position[0] - origin[0]) - abs(position[1] - origin[1]),
                -position[0], -position[1]), default=None)
            anchor_distance = (_distance_map(state, {anchor}, player)
                               if anchor is not None else {})
            # A non-empty map can still describe a disconnected component.
            # Only use the anchor potential when the controlled agent itself
            # can reach it; otherwise retain the nearest reachable bomb square.
            crate_distance = (anchor_distance
                              if origin in anchor_distance else all_crate_distance)
        else:
            # Preserve the established v10.1/v10.3 nearest-useful-square
            # behaviour for all older configurations.
            crate_distance = all_crate_distance
        bomb_utility = (self._bomb_utility(state, player, best_yield, frontier_nearby)
                        if "BOMB" in legal else 0.0)
        opponent_reach: set[tuple[int, int]] = set()
        near_opponent = False
        for index, other in enumerate(state.agents):
            if index == player or not other.alive:
                continue
            distance = abs(agent.x - other.x) + abs(agent.y - other.y)
            near_opponent |= distance <= 3
            opponent_reach.add((other.x, other.y))
            for action in legal_actions(state, index):
                opponent_reach.add(self._destination(state, index, action))

        scores: dict[str, float] = {}
        current_coin = coin_distance.get(origin, 99)
        current_crate = crate_distance.get(origin, 99)
        for action in allowed:
            destination = self._destination(state, player, action)
            score = 0.10 * np.log1p(widths[action])
            if coins:
                score += 0.95 * (current_coin - coin_distance.get(destination, 99))
            if crate_distance:
                score += self.crate_frontier_weight * (current_crate - crate_distance.get(destination, 99))
            if destination in opponent_reach:
                score -= 0.65
            if action == "WAIT":
                score -= 0.18
            if action == "BOMB":
                score += bomb_utility - (1.15 if bomb_utility <= 0.0 else 0.0)
            scores[action] = float(score)

        maximum = max(scores.values())
        planner_weights = {action: exp((score - maximum) / self.temperature)
                           for action, score in scores.items()}
        planner_total = sum(planner_weights.values())
        base_total = sum(max(0.0, base_policy.get(action, 0.0)) for action in allowed)
        prior = {}
        for action in allowed:
            planner_probability = planner_weights[action] / planner_total
            base_probability = (max(0.0, base_policy.get(action, 0.0)) / base_total
                                if base_total > 0.0 else 1.0 / len(allowed))
            prior[action] = ((1.0 - self.prior_weight) * base_probability
                             + self.prior_weight * planner_probability)
        total = sum(prior.values())
        prior = {action: value / total for action, value in prior.items()}
        # A bomb elsewhere on the board is routine in classic and previously
        # made almost every state "tactical".  Reserve the long budget and
        # always-sample path for an actual safety restriction, a live
        # explosion, nearby opponent pressure, or a useful bomb opportunity.
        bomb_threat = any(origin in _blast_coords(
            state.field, bomb.x, bomb.y, bomb.power) for bomb in state.bombs)
        tactical = (len(proven_safe) < len(legal) or bool(state.explosions)
                    or bomb_threat or near_opponent or bomb_utility > 0.0)
        return PlannerOutput(
            prior, proven_safe, scores, widths, bomb_utility, tactical,
            active_crate_target, target_distance,
        )


class PlannerRootPolicyTransform:
    """Search transform with inspectable diagnostics for tests/benchmarks."""

    def __init__(self, planner: ClassicTacticalPlanner, *,
                 persistent_target_enabled: bool = False,
                 productive_bomb_protection: bool = False):
        self.planner = planner
        self.call_count = 0
        self.last_plan: PlannerOutput | None = None
        self.crate_target: tuple[int, int] | None = None
        self.persistent_target_enabled = bool(persistent_target_enabled)
        self.productive_bomb_protection = bool(productive_bomb_protection)
        self.round_id: int | None = None
        self.protected_bomb_count = 0

    def set_crate_target(self, target: tuple[int, int] | None) -> None:
        self.crate_target = target

    def prepare(self, state: SimState, player: int, round_id: int | None = None):
        """Advance the shared per-round target context before a real action."""
        if round_id is not None and round_id != self.round_id:
            self.crate_target = None
            self.round_id = int(round_id)
        if self.persistent_target_enabled:
            self.crate_target = self.planner.select_persistent_crate_target(
                state, player, self.crate_target)
        else:
            self.crate_target = None
        return self.crate_target

    def protect_action(self, search_action: str) -> tuple[str, bool]:
        """Apply the same safe productive-bomb decision used by the teacher."""
        plan = self.last_plan
        if plan is None:
            raise RuntimeError("planner action protection requires a completed root plan")
        greedy_action = max(plan.prior, key=plan.prior.get)
        protected = bool(
            self.productive_bomb_protection
            and greedy_action == "BOMB"
            and plan.bomb_utility > 0.0
            and "BOMB" in plan.safe_actions
            and search_action != "BOMB"
        )
        self.protected_bomb_count += int(protected)
        return ("BOMB" if protected else search_action), protected

    def __call__(self, state: SimState, player: int,
                 base_policy: dict[str, float]) -> dict[str, float]:
        self.call_count += 1
        self.last_plan = self.planner.plan(
            state, player, base_policy, crate_target=self.crate_target)
        return self.last_plan.prior
