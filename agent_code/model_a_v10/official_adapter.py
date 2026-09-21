"""Differential-test adapter that invokes the unmodified official methods.

It deliberately uses ``environment.GenericWorld`` methods rather than a
second handwritten reference implementation.  Lightweight stand-ins satisfy
the attributes those methods use; no official source is changed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

import environment
from items import Bomb, Coin, Explosion

from .simulator import SimAgent, SimBomb, SimExplosion, SimState, observable


class _ReferenceAgent:
    def __init__(self, sim_agent: SimAgent, index: int):
        self.name = sim_agent.name
        self.index = index
        self.x, self.y = sim_agent.x, sim_agent.y
        self.score = sim_agent.score
        self.bombs_left = sim_agent.bombs_left
        self.dead = not sim_agent.alive
        self.events: list[str] = []
        self.trophies: list[object] = []
        self.bomb_sprite = None
        self.avatar = environment.Trophy.coin_trophy

    def add_event(self, event: str) -> None:
        self.events.append(event)

    def update_score(self, amount: int) -> None:
        self.score += amount


class OfficialReference:
    """State wrapper whose step calls the supplied official environment code."""

    # ``perform_agent_action`` calls this helper through ``self``.
    tile_is_free = environment.GenericWorld.tile_is_free

    def __init__(self, state: SimState):
        self.arena = state.field.copy()
        self.agents = [_ReferenceAgent(agent, i) for i, agent in enumerate(state.agents)]
        self.active_agents = [agent for agent in self.agents if not agent.dead]
        self.coins = [Coin(position, collectable) for position, collectable in state.coins.items()]
        self.bombs = [Bomb((bomb.x, bomb.y), self.agents[bomb.owner], bomb.timer, bomb.power, None)
                      for bomb in state.bombs]
        self.explosions = []
        for explosion in state.explosions:
            reference = Explosion(list(explosion.coords), [], self.agents[explosion.owner], explosion.timer)
            reference.stage = explosion.stage
            self.explosions.append(reference)
        self.step = state.step_count
        self.max_steps = state.max_steps
        self.running = not state.ended
        self.logger = logging.getLogger("v10-official-reference")

    def step_once(self, actions: Mapping[int, str] | Sequence[str], permutation: Sequence[int]) -> None:
        if not self.running:
            raise ValueError("Cannot step a terminal state")
        self.step += 1
        for agent in self.agents:
            agent.events = []
        active = list(self.active_agents)
        if sorted(permutation) != list(range(len(active))):
            raise ValueError("permutation must index every active agent exactly once")
        for position in permutation:
            agent = active[position]
            action = actions[agent.index] if isinstance(actions, Mapping) else actions[agent.index]
            environment.GenericWorld.perform_agent_action(self, agent, action)
        environment.GenericWorld.collect_coins(self)
        environment.GenericWorld.update_explosions(self)
        environment.GenericWorld.update_bombs(self)
        environment.GenericWorld.evaluate_explosions(self)
        if self._time_to_stop():
            self.running = False
            for agent in self.active_agents:
                agent.add_event("SURVIVED_ROUND")

    def _time_to_stop(self) -> bool:
        if not self.active_agents:
            return True
        if (len(self.active_agents) == 1 and not np.any(self.arena == 1)
                and not any(coin.collectable for coin in self.coins)
                and not self.bombs and not self.explosions):
            return True
        return self.step >= self.max_steps

    def as_sim_state(self) -> SimState:
        return from_official_world(self)


def from_official_world(world) -> SimState:
    """Copy a real ``BombeRLeWorld`` (or the lightweight reference) to SimState.

    It is intentionally a one-way snapshot adapter: the official world remains
    unmodified and owns its Agents, Bombs, Coins, and Explosions.
    """
    agents = [SimAgent(agent.name, agent.x, agent.y, agent.score, agent.bombs_left,
                       not agent.dead, list(agent.events)) for agent in world.agents]
    coins = {(coin.x, coin.y): coin.collectable for coin in world.coins}
    owner_index = {id(agent): index for index, agent in enumerate(world.agents)}
    bombs = [SimBomb(bomb.x, bomb.y, owner_index[id(bomb.owner)], bomb.timer, bomb.power)
             for bomb in world.bombs]
    explosions = [SimExplosion(tuple(explosion.blast_coords), owner_index[id(explosion.owner)],
                               explosion.timer, explosion.stage) for explosion in world.explosions]
    return SimState(world.arena.copy(), agents, coins, bombs, explosions, world.step,
                    getattr(world, "max_steps", 400), not world.running)


def official_step(state: SimState, actions: Mapping[int, str] | Sequence[str], permutation: Sequence[int]) -> SimState:
    reference = OfficialReference(state)
    reference.step_once(actions, permutation)
    return reference.as_sim_state()


def equivalent(simulated: SimState, official: SimState) -> tuple[bool, str | None]:
    """Compare every Phase-0 observable, normalizing unordered death events."""
    left, right = observable(simulated), observable(official)
    for key in ("field", "explosion_map"):
        if not np.array_equal(left[key], right[key]):
            return False, key
    for key in ("coins", "agents", "bombs", "explosions", "step", "ended"):
        if left[key] != right[key]:
            return False, key
    left_events = tuple(tuple(sorted(events)) for events in left["events"])
    right_events = tuple(tuple(sorted(events)) for events in right["events"])
    if left_events != right_events:
        return False, "events"
    return True, None
