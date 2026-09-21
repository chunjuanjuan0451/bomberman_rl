"""Frozen-v4 tactical overlay, isolated from historical v10 runtime semantics."""

from __future__ import annotations

import numpy as np

from .hybrid_search import ValueReportingPUCT
from .runtime import DEFAULT_SEARCH_BUDGET, Phase2Controller, SearchBudget, state_from_game_state
from .simulator import SimState, _blast_coords


class BombOwnershipTracker:
    """Infer public bomb ownership across observer frames without engine access."""

    def __init__(self):
        self.round_id: int | None = None
        self.previous_step = -1
        self.previous_agents: dict[str, tuple[tuple[int, int], bool]] = {}
        self.previous_bombs: dict[tuple[int, int], tuple[int, str | None]] = {}
        self.explosion_owners: dict[tuple[int, int], str] = {}

    def reset(self) -> None:
        self.previous_step = -1
        self.previous_agents.clear()
        self.previous_bombs.clear()
        self.explosion_owners.clear()

    def observe(self, game_state: dict, *, last_self_action: str = "WAIT") -> tuple[dict, dict]:
        round_id = int(game_state.get("round", 0))
        step_id = int(game_state.get("step", 0))
        if self.round_id != round_id or step_id <= self.previous_step:
            self.reset()
            self.round_id = round_id
        observed = [game_state["self"], *game_state["others"]]
        current_agents = {
            str(name): (tuple(position), bool(bombs_left))
            for name, _score, bombs_left, position in observed
        }
        self_name = str(game_state["self"][0])
        current_bombs: dict[tuple[int, int], tuple[int, str | None]] = {}
        for raw_position, raw_timer in game_state["bombs"]:
            position, timer = tuple(raw_position), int(raw_timer)
            owner = self.previous_bombs.get(position, (timer, None))[1]
            if owner is None:
                candidates = []
                if (last_self_action == "BOMB" and self_name in self.previous_agents
                        and self.previous_agents[self_name][0] == position):
                    candidates.append(self_name)
                for name, (previous_position, previously_available) in self.previous_agents.items():
                    current = current_agents.get(name)
                    if (previous_position == position and previously_available
                            and current is not None and not current[1]):
                        candidates.append(name)
                if len(set(candidates)) == 1:
                    owner = candidates[0]
            current_bombs[position] = (timer, owner)

        explosion_map = np.asarray(game_state["explosion_map"])
        active_cells = {(int(x), int(y)) for x, y in np.argwhere(explosion_map > 0)}
        owners = {position: name for position, name in self.explosion_owners.items()
                  if position in active_cells}
        field = np.asarray(game_state["field"])
        for position, (_timer, owner) in self.previous_bombs.items():
            if position in current_bombs or owner is None:
                continue
            for cell in _blast_coords(field, position[0], position[1], 3):
                if cell in active_cells:
                    owners[cell] = owner
        self.explosion_owners = owners

        name_to_index = {str(agent[0]): index for index, agent in enumerate(observed)}
        fallback = 1 if len(observed) > 1 else 0
        bomb_indices = {position: name_to_index.get(owner, fallback)
                        for position, (_timer, owner) in current_bombs.items()}
        explosion_indices = {position: name_to_index.get(owner, fallback)
                             for position, owner in owners.items()}
        self.previous_agents = current_agents
        self.previous_bombs = current_bombs
        self.previous_step = step_id
        return bomb_indices, explosion_indices


def owned_state_from_game_state(game_state: dict, bomb_owners: dict,
                                explosion_owners: dict) -> SimState:
    state = state_from_game_state(game_state)
    for bomb in state.bombs:
        bomb.owner = int(bomb_owners.get((bomb.x, bomb.y), bomb.owner))
    for explosion in state.explosions:
        position = explosion.coords[0]
        explosion.owner = int(explosion_owners.get(position, explosion.owner))
    return state


def hybrid_tactical_reason(state: SimState, anchor_action: str,
                           player: int = 0) -> str | None:
    agent = state.agents[player]
    if state.explosion_map()[agent.x, agent.y] > 0:
        return "active-explosion"
    for bomb in state.bombs:
        if bomb.timer <= 2 and (agent.x, agent.y) in _blast_coords(
                state.field, bomb.x, bomb.y, bomb.power):
            return "imminent-blast"
    if any(other.alive and abs(agent.x - other.x) + abs(agent.y - other.y) <= 3
           for index, other in enumerate(state.agents) if index != player):
        return "opponent-within-3"
    if anchor_action == "BOMB":
        return "v4-bomb-proposal"
    return None


class V4AnchoredHybridController(Phase2Controller):
    """Return v4 by default and admit only well-supported tactical changes."""

    def __init__(self, network=None, *, anchor_policy, minimum_action_visits: int = 8,
                 minimum_value_advantage: float = 0.10,
                 crate_progress_weight: float = 0.0,
                 forbidden_root_actions: tuple[str, ...] = (), seed_offset: int = 0,
                 repetition_context=None, budget: SearchBudget = DEFAULT_SEARCH_BUDGET,
                 root_policy_transform=None):
        super().__init__(
            network, forbidden_root_actions=forbidden_root_actions,
            seed_offset=seed_offset, repetition_context=repetition_context,
            budget=budget, root_policy_transform=root_policy_transform)
        self.searcher = ValueReportingPUCT(
            self.network, forbidden_root_actions=forbidden_root_actions,
            repetition_context=repetition_context,
            root_policy_transform=root_policy_transform,
            crate_progress_weight=crate_progress_weight)
        if minimum_action_visits < 1 or minimum_value_advantage < 0.0:
            raise ValueError("invalid v4 hybrid override thresholds")
        self.anchor_policy = anchor_policy
        self.minimum_action_visits = int(minimum_action_visits)
        self.minimum_value_advantage = float(minimum_value_advantage)
        self.ownership_tracker = BombOwnershipTracker()
        self.last_anchor_action = "WAIT"
        self.last_gate_reason: str | None = None
        self.last_override_reason = "v4-default"

    def _select_tactical(self, state: SimState, seed: int):
        self.last_tactical = True
        self.last_result = self.searcher.search(
            state, 0, simulations=self.budget.tactical_simulations, seed=seed,
            deadline_ms=self.budget.tactical_deadline_ms)
        self.last_under_minimum = (
            self.last_result.simulations < self.budget.tactical_min_simulations)
        return self.last_result

    def act(self, game_state: dict) -> str:
        bomb_owners, explosion_owners = self.ownership_tracker.observe(
            game_state, last_self_action=self.last_action)
        state = owned_state_from_game_state(game_state, bomb_owners, explosion_owners)
        if self.repetition_context is not None:
            self.repetition_context.update(
                int(game_state.get("round", 0)), int(game_state["self"][1]),
                tuple(int(value) for value in game_state["self"][3]))
        self.last_anchor_action = self.anchor_policy.greedy_action(game_state)
        self.last_gate_reason = hybrid_tactical_reason(state, self.last_anchor_action)
        if self.last_gate_reason is None:
            self.last_result = None
            self.last_action = self.last_anchor_action
            self.last_override_reason = "v4-default"
            self.last_productive_bomb_protected = False
            return self.last_action

        transform = self.searcher.root_policy_transform
        if transform is not None and hasattr(transform, "prepare"):
            transform.prepare(state, 0, round_id=int(game_state.get("round", 0)))
        seed = ((int(game_state.get("round", 0)) << 16)
                + int(game_state.get("step", 0)) + self.seed_offset)
        result = self._select_tactical(state, seed)
        proposed = result.action
        protected = False
        if transform is not None and hasattr(transform, "protect_action"):
            proposed, protected = transform.protect_action(proposed)

        anchor = self.last_anchor_action
        if protected:
            action, reason = proposed, "productive-bomb-protection"
        elif anchor not in result.raw_prior:
            action, reason = result.fallback, "planner-safety-fallback"
        elif proposed == anchor:
            action, reason = anchor, "search-agrees"
        else:
            proposed_visits = result.root_visits.get(proposed, 0)
            anchor_visits = result.root_visits.get(anchor, 0)
            advantage = result.root_values.get(proposed, 0.0) - result.root_values.get(anchor, 0.0)
            if (proposed_visits >= self.minimum_action_visits
                    and anchor_visits >= self.minimum_action_visits
                    and advantage >= self.minimum_value_advantage):
                action, reason = proposed, "value-qualified-override"
            else:
                action, reason = anchor, "insufficient-override-evidence"
        self.last_action = action
        self.last_override_reason = reason
        self.last_productive_bomb_protected = protected
        return action
