"""Phase-2 runtime: observer-state adapter, budget policy, and legal act path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interfaces import HeuristicNetwork
from .search import SearchResult, StochasticPUCT
from .simulator import SimAgent, SimBomb, SimExplosion, SimState, _blast_coords
from .repetition import RepetitionContext


NORMAL_MIN_SIMULATIONS = 50
TACTICAL_MIN_SIMULATIONS = 200
# High caps make inference deadline-driven instead of stopping after only a
# few dozen simulations.  Training remains fixed-simulation and reproducible.
NORMAL_SIMULATIONS = 10_000
TACTICAL_SIMULATIONS = 10_000
NORMAL_DEADLINE_MS = 55.0
TACTICAL_DEADLINE_MS = 220.0


@dataclass(frozen=True)
class SearchBudget:
    """Wall-clock policy kept separate from learned model parameters."""

    normal_simulations: int
    tactical_simulations: int
    normal_min_simulations: int
    tactical_min_simulations: int
    normal_deadline_ms: float
    tactical_deadline_ms: float


DEFAULT_SEARCH_BUDGET = SearchBudget(
    NORMAL_SIMULATIONS, TACTICAL_SIMULATIONS,
    NORMAL_MIN_SIMULATIONS, TACTICAL_MIN_SIMULATIONS,
    NORMAL_DEADLINE_MS, TACTICAL_DEADLINE_MS,
)

# The official allowance is 500 ms on one Ryzen 5 2600 thread.  This profile
# spends more of that allowance than the M4-oriented research default while
# retaining 200 ms for IPC, scheduling jitter, and one in-flight simulation.
RYZEN_SAFE_SEARCH_BUDGET = SearchBudget(
    NORMAL_SIMULATIONS, TACTICAL_SIMULATIONS,
    NORMAL_MIN_SIMULATIONS, TACTICAL_MIN_SIMULATIONS,
    90.0, 300.0,
)

# v10.1 spends most of the published 500 ms allowance while keeping a 100 ms
# envelope for callback IPC, scheduling jitter and one in-flight simulation.
# The root planner is included in these deadlines.
RYZEN_UTILIZED_SEARCH_BUDGET = SearchBudget(
    NORMAL_SIMULATIONS, TACTICAL_SIMULATIONS,
    50, 100,
    240.0, 400.0,
)

SEARCH_BUDGET_PROFILES = {
    "research-m4-v1": DEFAULT_SEARCH_BUDGET,
    "ryzen-safe-v1": RYZEN_SAFE_SEARCH_BUDGET,
    "ryzen-utilized-v2": RYZEN_UTILIZED_SEARCH_BUDGET,
}


def search_budget_for_profile(name: str) -> SearchBudget:
    try:
        return SEARCH_BUDGET_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown v10 search budget profile: {name!r}") from exc


def state_from_game_state(game_state: dict) -> SimState:
    """Copy one official observer state without reading any private engine data.

    Official observer states do not reveal bomb ownership.  Unknown bombs are
    conservatively attributed to the first opponent so their blast remains
    exact while root does not receive speculative kill credit.
    """
    name, score, bombs_left, position = game_state["self"]
    agents = [SimAgent(name, *position, score=score, bombs_left=bool(bombs_left))]
    for other_name, other_score, other_bombs_left, other_position in game_state["others"]:
        agents.append(SimAgent(other_name, *other_position, score=other_score,
                               bombs_left=bool(other_bombs_left)))
    owner = 1 if len(agents) > 1 else 0
    bombs = [SimBomb(int(position[0]), int(position[1]), owner, int(timer))
             for position, timer in game_state["bombs"]]
    explosions = []
    explosion_map = np.asarray(game_state["explosion_map"])
    for x, y in np.argwhere(explosion_map > 0):
        explosions.append(SimExplosion(((int(x), int(y)),), owner, int(explosion_map[x, y]) + 1, 0))
    return SimState(np.asarray(game_state["field"]).copy(), agents,
                    {tuple(position): True for position in game_state["coins"]}, bombs, explosions,
                    step_count=int(game_state.get("step", 0)))


def state_features(state: SimState, player: int = 0) -> np.ndarray:
    """Agent-centred full-board schema shared by training and inference."""
    controlled = state.agents[player]
    field = state.field.astype(np.float32, copy=True)
    bomb_timer = np.zeros(state.field.shape, dtype=np.float32)
    for bomb in state.bombs:
        bomb_timer[bomb.x, bomb.y] = max(bomb_timer[bomb.x, bomb.y], (bomb.timer + 1) / 5.0)
    occupancy = np.zeros(state.field.shape, dtype=np.float32)
    for index, observed in enumerate(state.agents):
        if observed.alive:
            occupancy[observed.x, observed.y] = 1.0 if index == player else -1.0
    coins = np.zeros(state.field.shape, dtype=np.float32)
    for position, visible in state.coins.items():
        if visible:
            coins[position] = 1.0
    features = np.stack((field, bomb_timer, state.explosion_map().astype(np.float32), occupancy, coins))
    return center_feature_maps(features, (controlled.x, controlled.y))


def opponent_aware_state_features(state: SimState, player: int = 0) -> np.ndarray:
    """Centered schema that retains the direction of off-crop opponents.

    The original centered 17x17 representation clips far-away agents when the
    controlled player is away from the board centre.  Preserve exact markers
    that fit and project only off-crop opponents onto the corresponding edge.
    This keeps the established five-channel/input-size contract while ensuring
    multiplayer policy data is not observationally identical to solo data.
    """
    result = state_features(state, player)
    controlled = state.agents[player]
    centre_x, centre_y = result.shape[1] // 2, result.shape[2] // 2
    dx, dy = centre_x - controlled.x, centre_y - controlled.y
    for index, other in enumerate(state.agents):
        if index == player or not other.alive:
            continue
        projected_x = int(np.clip(other.x + dx, 0, result.shape[1] - 1))
        projected_y = int(np.clip(other.y + dy, 0, result.shape[2] - 1))
        result[3, projected_x, projected_y] = -1.0
    return result


def center_feature_maps(features: np.ndarray,
                        player_position: tuple[int, int] | None = None) -> np.ndarray:
    """Translate maps without wrap-around so the controlled agent is central.

    ``player_position`` is optional for persisted trajectories: the +1 marker
    in occupancy channel 3 is then used.  Both one sample ``[C,X,Y]`` and a
    batch ``[B,C,X,Y]`` are accepted.
    """
    values = np.asarray(features)
    batched = values.ndim == 4
    if values.ndim not in (3, 4):
        raise ValueError("v10 spatial features must have shape [C,X,Y] or [B,C,X,Y]")
    samples = values if batched else values[None, ...]
    result = np.zeros_like(samples)
    result[:, 0] = -1
    for index, sample in enumerate(samples):
        if player_position is None:
            positions = np.argwhere(sample[3] > 0.5)
            if len(positions) != 1:
                raise ValueError("v10 features require exactly one controlled-agent marker")
            px, py = map(int, positions[0])
        else:
            px, py = player_position
        width, height = sample.shape[1:]
        dx, dy = width // 2 - px, height // 2 - py
        src_x0, src_x1 = max(0, -dx), min(width, width - dx)
        src_y0, src_y1 = max(0, -dy), min(height, height - dy)
        dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
        dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
        result[index, :, dst_x0:dst_x1, dst_y0:dst_y1] = sample[:, src_x0:src_x1, src_y0:src_y1]
    return result if batched else result[0]


def is_tactical(state: SimState, player: int = 0) -> bool:
    agent = state.agents[player]
    if state.explosion_map()[agent.x, agent.y] > 0:
        return True
    for bomb in state.bombs:
        if bomb.timer <= 2 and (agent.x, agent.y) in _blast_coords(state.field, bomb.x, bomb.y, bomb.power):
            return True
    return any(other.alive and abs(agent.x - other.x) + abs(agent.y - other.y) <= 1
               for index, other in enumerate(state.agents) if index != player)


class Phase2Controller:
    """Search-on action path with distinct normal and tactical budgets."""

    def __init__(self, network=None, *, forbidden_root_actions: tuple[str, ...] = (),
                 seed_offset: int = 0, repetition_context: RepetitionContext | None = None,
                 budget: SearchBudget = DEFAULT_SEARCH_BUDGET,
                 root_policy_transform=None):
        self.network = network or HeuristicNetwork()
        self.searcher = StochasticPUCT(
            self.network, forbidden_root_actions=forbidden_root_actions,
            repetition_context=repetition_context,
            root_policy_transform=root_policy_transform)
        self.repetition_context = repetition_context
        self.seed_offset = int(seed_offset)
        self.budget = budget
        self.last_result: SearchResult | None = None
        self.last_tactical = False
        self.last_under_minimum = False
        self.last_action = "WAIT"
        self.last_productive_bomb_protected = False

    def select(self, state: SimState, *, seed: int = 0) -> SearchResult:
        self.last_tactical = is_tactical(state)
        simulations = (self.budget.tactical_simulations if self.last_tactical
                       else self.budget.normal_simulations)
        deadline = (self.budget.tactical_deadline_ms if self.last_tactical
                    else self.budget.normal_deadline_ms)
        self.last_result = self.searcher.search(state, 0, simulations=simulations, seed=seed,
                                                deadline_ms=deadline)
        minimum = (self.budget.tactical_min_simulations if self.last_tactical
                   else self.budget.normal_min_simulations)
        # Never sacrifice the already-computed legal fallback in production;
        # the benchmark gate, rather than the callback, rejects slow machines.
        self.last_under_minimum = self.last_result.simulations < minimum
        return self.last_result

    def act(self, game_state: dict) -> str:
        state = state_from_game_state(game_state)
        if self.repetition_context is not None:
            self.repetition_context.update(
                int(game_state.get("round", 0)), int(game_state["self"][1]),
                tuple(int(value) for value in game_state["self"][3]),
            )
        # Materialize the schema now so future learned heads can share exactly
        # this callback path; Phase 2 times it explicitly.
        state_features(state)
        transform = self.searcher.root_policy_transform
        if transform is not None and hasattr(transform, "prepare"):
            transform.prepare(
                state, 0, round_id=int(game_state.get("round", 0)))
        seed = ((int(game_state.get("round", 0)) << 16)
                + int(game_state.get("step", 0)) + self.seed_offset)
        action = self.select(state, seed=seed).action
        protected = False
        if transform is not None and hasattr(transform, "protect_action"):
            action, protected = transform.protect_action(action)
        self.last_action = action
        self.last_productive_bomb_protected = protected
        return action
