"""Small state-copy-friendly simulator matching the supplied environment.py.

Coordinate order is ``(x, y)`` throughout.  In particular, this module keeps
the original engine's sequential action application after actions are sampled
and does *not* add a chain-reaction rule the official engine does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


BOMB_TIMER = 4
BOMB_POWER = 3
EXPLOSION_TIMER = 2
MAX_STEPS = 400
MOVES = {"UP": (0, -1, "MOVED_UP"), "DOWN": (0, 1, "MOVED_DOWN"),
         "LEFT": (-1, 0, "MOVED_LEFT"), "RIGHT": (1, 0, "MOVED_RIGHT")}


@dataclass(slots=True)
class SimAgent:
    name: str
    x: int
    y: int
    score: int = 0
    bombs_left: bool = True
    alive: bool = True
    events: list[str] = field(default_factory=list)

    def copy(self) -> "SimAgent":
        return SimAgent(self.name, self.x, self.y, self.score, self.bombs_left, self.alive, [])


@dataclass(slots=True)
class SimBomb:
    x: int
    y: int
    owner: int
    timer: int = BOMB_TIMER
    power: int = BOMB_POWER

    def copy(self) -> "SimBomb":
        return SimBomb(self.x, self.y, self.owner, self.timer, self.power)


@dataclass(slots=True)
class SimExplosion:
    coords: tuple[tuple[int, int], ...]
    owner: int
    timer: int = EXPLOSION_TIMER
    stage: int | None = 0

    def copy(self) -> "SimExplosion":
        return SimExplosion(self.coords, self.owner, self.timer, self.stage)

    @property
    def dangerous(self) -> bool:
        return self.stage == 0


@dataclass(slots=True)
class SimState:
    field: np.ndarray
    agents: list[SimAgent]
    coins: dict[tuple[int, int], bool] = field(default_factory=dict)
    bombs: list[SimBomb] = field(default_factory=list)
    explosions: list[SimExplosion] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = MAX_STEPS
    ended: bool = False

    def copy(self) -> "SimState":
        return SimState(self.field.copy(), [a.copy() for a in self.agents], dict(self.coins),
                        [b.copy() for b in self.bombs], [e.copy() for e in self.explosions],
                        self.step_count, self.max_steps, self.ended)

    def active_ids(self) -> list[int]:
        return [i for i, agent in enumerate(self.agents) if agent.alive]

    def explosion_map(self) -> np.ndarray:
        result = np.zeros(self.field.shape, dtype=int)
        for explosion in self.explosions:
            if explosion.dangerous:
                value = explosion.timer - 1
                for x, y in explosion.coords:
                    result[x, y] = max(result[x, y], value)
        return result


def _blast_coords(field: np.ndarray, x: int, y: int, power: int) -> tuple[tuple[int, int], ...]:
    result = [(x, y)]
    # This intentionally mirrors items.Bomb.get_blast_coords: crates are
    # included but do not stop propagation in this project version.
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, power + 1):
            xx, yy = x + dx * distance, y + dy * distance
            if field[xx, yy] == -1:
                break
            result.append((xx, yy))
    return tuple(result)


def _free(state: SimState, x: int, y: int) -> bool:
    if state.field[x, y] != 0:
        return False
    return not any(bomb.x == x and bomb.y == y for bomb in state.bombs) and not any(
        agent.alive and agent.x == x and agent.y == y for agent in state.agents
    )


def _perform_action(state: SimState, agent_id: int, action: str) -> None:
    agent = state.agents[agent_id]
    move = MOVES.get(action)
    if move is not None:
        dx, dy, event = move
        if _free(state, agent.x + dx, agent.y + dy):
            agent.x += dx
            agent.y += dy
            agent.events.append(event)
        else:
            agent.events.append("INVALID_ACTION")
    elif action == "BOMB" and agent.bombs_left:
        state.bombs.append(SimBomb(agent.x, agent.y, agent_id))
        agent.bombs_left = False
        agent.events.append("BOMB_DROPPED")
    elif action == "WAIT":
        agent.events.append("WAITED")
    else:
        agent.events.append("INVALID_ACTION")


def _collect_coins(state: SimState) -> None:
    for position, collectable in tuple(state.coins.items()):
        if not collectable:
            continue
        for agent in state.agents:
            if agent.alive and (agent.x, agent.y) == position:
                state.coins[position] = False
                agent.score += 1
                agent.events.append("COIN_COLLECTED")


def _update_explosions(state: SimState) -> None:
    remaining: list[SimExplosion] = []
    for explosion in state.explosions:
        explosion.timer -= 1
        if explosion.timer <= 0:
            next_stage = (explosion.stage if explosion.stage is not None else 1) + 1
            if next_stage >= 2:
                explosion.stage = None
            else:
                explosion.stage = next_stage
                explosion.timer = 2  # len(Explosion.ASSETS[stage])
                if explosion.stage == 1:
                    state.agents[explosion.owner].bombs_left = True
        if explosion.stage is not None:
            remaining.append(explosion)
    state.explosions = remaining


def _update_bombs(state: SimState) -> None:
    remaining: list[SimBomb] = []
    for bomb in state.bombs:
        if bomb.timer <= 0:
            owner = state.agents[bomb.owner]
            owner.events.append("BOMB_EXPLODED")
            coords = _blast_coords(state.field, bomb.x, bomb.y, bomb.power)
            for coord in coords:
                if state.field[coord] == 1:
                    state.field[coord] = 0
                    owner.events.append("CRATE_DESTROYED")
                    if coord in state.coins:
                        state.coins[coord] = True
                        owner.events.append("COIN_FOUND")
            state.explosions.append(SimExplosion(coords, bomb.owner))
        else:
            bomb.timer -= 1
            remaining.append(bomb)
    state.bombs = remaining


def _evaluate_explosions(state: SimState) -> None:
    hit: set[int] = set()
    for explosion in state.explosions:
        if not explosion.dangerous:
            continue
        blast = set(explosion.coords)
        owner = state.agents[explosion.owner]
        for agent_id, agent in enumerate(state.agents):
            if agent.alive and (agent.x, agent.y) in blast:
                hit.add(agent_id)
                if agent_id == explosion.owner:
                    agent.events.append("KILLED_SELF")
                else:
                    owner.score += 5
                    owner.events.append("KILLED_OPPONENT")
    # The official engine uses a set here.  Iterating agent order keeps the
    # same observable event multiset while making copied simulation stable.
    for agent_id in sorted(hit):
        state.agents[agent_id].alive = False
        state.agents[agent_id].events.append("GOT_KILLED")
        for other_id, other in enumerate(state.agents):
            if other_id != agent_id and other.alive:
                other.events.append("OPPONENT_ELIMINATED")


def _terminal(state: SimState) -> bool:
    active = state.active_ids()
    if not active:
        return True
    if (len(active) == 1 and not np.any(state.field == 1) and not any(state.coins.values())
            and not state.bombs and not state.explosions):
        return True
    return state.step_count >= state.max_steps


def step_inplace(
    state: SimState,
    actions: Mapping[int, str] | Sequence[str],
    permutation: Sequence[int],
) -> SimState:
    """Run one official-order game step, mutating and returning ``state``.

    ``permutation`` indexes the active-agent snapshot at action execution
    time, exactly like ``BombeRLeWorld.poll_and_run_agents``.
    """
    if state.ended:
        raise ValueError("Cannot step a terminal state")
    state.step_count += 1
    active = state.active_ids()
    for agent in state.agents:
        agent.events = []
    if sorted(permutation) != list(range(len(active))):
        raise ValueError("permutation must index every active agent exactly once")
    for position in permutation:
        agent_id = active[position]
        action = actions[agent_id] if isinstance(actions, Mapping) else actions[agent_id]
        _perform_action(state, agent_id, action)
    _collect_coins(state)
    _update_explosions(state)
    _update_bombs(state)
    _evaluate_explosions(state)
    state.ended = _terminal(state)
    if state.ended:
        for agent_id in state.active_ids():
            state.agents[agent_id].events.append("SURVIVED_ROUND")
    return state


def step(state: SimState, actions: Mapping[int, str] | Sequence[str], permutation: Sequence[int]) -> SimState:
    """Copy-on-step entry point used by search nodes and benchmarks."""
    return step_inplace(state.copy(), actions, permutation)


def observable(state: SimState) -> dict[str, object]:
    """Canonical public state used by unit and differential tests."""
    return {
        "field": state.field.copy(), "coins": tuple(sorted(state.coins.items())),
        "agents": tuple((a.name, a.score, a.bombs_left, a.alive, a.x, a.y) for a in state.agents),
        "bombs": tuple((b.x, b.y, b.owner, b.timer, b.power) for b in state.bombs),
        "explosions": tuple((e.coords, e.owner, e.timer, e.stage) for e in state.explosions),
        "explosion_map": state.explosion_map(),
        "events": tuple(tuple(a.events) for a in state.agents),
        "step": state.step_count, "ended": state.ended,
    }
