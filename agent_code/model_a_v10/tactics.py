"""Fixed, hand-verifiable Phase-1 tactical positions and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interfaces import HeuristicNetwork
from .search import SearchResult, StochasticPUCT
from .simulator import SimAgent, SimBomb, SimState, step


@dataclass(frozen=True, slots=True)
class TacticalCase:
    name: str
    description: str
    state: SimState
    root_player: int
    seed: int
    expected_action: str | None = None
    forbidden_action: str | None = None


def _field() -> np.ndarray:
    field = -np.ones((9, 9), dtype=int)
    field[1:-1, 1:-1] = 0
    return field


def tactical_cases() -> tuple[TacticalCase, ...]:
    # Only RIGHT lets the player turn away from the imminent vertical blast.
    escape = _field()
    escape[2, 3] = -1
    escape[3, 4] = -1
    unique_escape = TacticalCase(
        "unique_escape", "A timer-1 bomb makes RIGHT then DOWN the only surviving route.",
        SimState(escape, [SimAgent("root", 3, 3)], bombs=[SimBomb(3, 2, 0, 1)]), 0, 1201,
        expected_action="RIGHT",
    )

    # Raw prior heads toward the visible coin via the contested exit.  The
    # sampled-opponent search instead takes the upper lane.
    conflict = _field()
    conflict[3, 2] = conflict[3, 4] = -1
    exit_conflict = TacticalCase(
        "opponent_exit_conflict", "The opponent can occupy the coin-side exit first.",
        SimState(conflict, [SimAgent("root", 2, 3), SimAgent("opponent", 4, 3)], coins={(5, 3): True}),
        0, 1,
    )

    kill = _field()
    kill[5, 2] = kill[5, 4] = kill[6, 3] = -1
    forced_kill = TacticalCase(
        "forced_kill", "Adjacent opponent has no exit; BOMB has an escape route for root.",
        SimState(kill, [SimAgent("root", 4, 3), SimAgent("opponent", 5, 3)]), 0, 1,
        expected_action="BOMB",
    )

    false_kill_field = _field()
    for position in ((3, 2), (3, 4), (4, 3), (2, 2), (2, 4), (1, 2), (1, 4)):
        false_kill_field[position] = -1
    false_kill = TacticalCase(
        "false_kill", "Bombing the adjacent opponent leaves root in a one-cell blast corridor.",
        SimState(false_kill_field, [SimAgent("root", 3, 3), SimAgent("opponent", 4, 3)]), 0, 1,
        forbidden_action="BOMB",
    )

    chain = _field()
    no_chain = TacticalCase(
        "official_no_chain", "Adjacent bombs remain independently timed in environment.py.",
        SimState(chain, [SimAgent("root", 1, 1), SimAgent("opponent", 7, 7)],
                 bombs=[SimBomb(3, 3, 1, 0), SimBomb(4, 3, 1, 3)]), 0, 1205,
    )

    tradeoff = _field()
    tradeoff[5, 2] = tradeoff[5, 4] = tradeoff[6, 3] = -1
    coin_kill = TacticalCase(
        "coin_kill_tradeoff", "A nearby coin is less valuable than a certain trapped-opponent kill.",
        SimState(tradeoff, [SimAgent("root", 4, 3), SimAgent("opponent", 5, 3)], coins={(3, 3): True}),
        0, 1206, expected_action="BOMB",
    )
    return unique_escape, exit_conflict, forced_kill, false_kill, no_chain, coin_kill


def rollout_raw_policy(state: SimState, root_player: int, first_action: str, plies: int = 8) -> SimState:
    """Deterministic raw-policy continuation used only for the tactical ablation."""
    current = state
    network = HeuristicNetwork()
    for depth in range(plies):
        if current.ended or not current.agents[root_player].alive:
            break
        policy = network.evaluate(current, root_player).policy
        action = first_action if depth == 0 else max(policy, key=lambda item: (policy[item], item))
        joint = ["WAIT"] * len(current.agents)
        joint[root_player] = action
        current = step(current, joint, list(range(len(current.active_ids()))))
    return current


def rollout_search_policy(state: SimState, root_player: int, seed: int, simulations: int, plies: int = 8) -> SimState:
    """Re-search at every ply: the actual search-on side of the ablation."""
    current = state
    searcher = StochasticPUCT()
    for depth in range(plies):
        if current.ended or not current.agents[root_player].alive:
            break
        result = searcher.search(current, root_player, simulations=simulations, seed=seed + depth)
        joint = ["WAIT"] * len(current.agents)
        joint[root_player] = result.action
        current = step(current, joint, list(range(len(current.active_ids()))))
    return current


def evaluate_tactics(simulations: int = 50) -> dict[str, object]:
    searcher = StochasticPUCT()
    rows: list[dict[str, object]] = []
    for case in tactical_cases():
        result = searcher.search(case.state, case.root_player, simulations=simulations, seed=case.seed)
        raw_terminal = rollout_raw_policy(case.state, case.root_player, result.fallback)
        search_terminal = rollout_search_policy(case.state, case.root_player, case.seed, simulations)
        passed = ((case.expected_action is None or result.action == case.expected_action)
                  and (case.forbidden_action is None or result.action != case.forbidden_action))
        if case.name == "opponent_exit_conflict":
            passed = passed and result.action in ("UP", "DOWN") and result.action != result.fallback
        rows.append({"name": case.name, "description": case.description, "expected_action": case.expected_action,
                     "forbidden_action": case.forbidden_action, "passed": passed,
                     "raw_terminal_alive": raw_terminal.agents[case.root_player].alive,
                     "search_terminal_alive": search_terminal.agents[case.root_player].alive,
                     "search": result.as_dict()})
    return {"simulations": simulations, "cases": rows,
            "passed": all(bool(row["passed"]) for row in rows)}
