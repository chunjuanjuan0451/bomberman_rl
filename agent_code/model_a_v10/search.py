"""Deterministic-seed stochastic PUCT used for the Phase-1 smoke tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np

from .interfaces import ACTIONS, HeuristicNetwork, legal_actions
from .simulator import SimState, step
from .repetition import RepetitionContext


RootPolicyTransform = Callable[[SimState, int, dict[str, float]], dict[str, float]]


@dataclass(slots=True)
class ActionStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class Node:
    state: SimState
    depth: int
    actions: dict[str, ActionStats]
    visits: int = 0
    value_sum: float = 0.0
    children: dict[tuple[str, tuple[str, ...], tuple[int, ...]], "Node"] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    action: str
    fallback: str
    raw_prior: dict[str, float]
    root_visits: dict[str, int]
    principal_variation: tuple[str, ...]
    risk: float
    simulations: int
    max_depth: int
    reason: str
    elapsed_ms: float

    def as_dict(self) -> dict[str, object]:
        return {"action": self.action, "fallback": self.fallback, "raw_prior": self.raw_prior,
                "root_visits": self.root_visits, "principal_variation": list(self.principal_variation),
                "risk": self.risk, "simulations": self.simulations, "max_depth": self.max_depth,
                "reason": self.reason, "elapsed_ms": self.elapsed_ms}


class StochasticPUCT:
    """Joint-opponent PUCT with sampled official action order chance nodes."""

    def __init__(self, network: HeuristicNetwork | None = None, *, c_puct: float = 1.35,
                 rule_prior_mix: float = 0.70, horizon: int = 8,
                 risk_penalty: float = 0.25, widening_scale: float = 1.5,
                 crate_progress_weight: float = 0.0,
                 forbidden_root_actions: tuple[str, ...] = (),
                 repetition_context: RepetitionContext | None = None,
                 root_policy_transform: RootPolicyTransform | None = None):
        self.network = network or HeuristicNetwork()
        self.rule_network = HeuristicNetwork()
        self.c_puct = c_puct
        self.rule_prior_mix = rule_prior_mix
        self.horizon = horizon
        self.risk_penalty = risk_penalty
        self.widening_scale = widening_scale
        self.crate_progress_weight = float(crate_progress_weight)
        self.forbidden_root_actions = frozenset(forbidden_root_actions)
        self.repetition_context = repetition_context
        self.root_policy_transform = root_policy_transform

    def _normalize_policy(self, policy: dict[str, float]) -> dict[str, float]:
        filtered = {action: float(value) for action, value in policy.items()
                    if action not in self.forbidden_root_actions}
        total = sum(filtered.values())
        return ({action: value / total for action, value in filtered.items()}
                if total > 0.0 else {"WAIT": 1.0})

    def _validate_transformed_policy(self, policy: dict[str, float]) -> dict[str, float]:
        result = {action: float(value) for action, value in policy.items()
                  if action not in self.forbidden_root_actions}
        total = sum(result.values())
        if (not result or not np.isfinite(list(result.values())).all()
                or any(value < 0.0 for value in result.values()) or total <= 0.0):
            raise ValueError("root policy transform returned an invalid policy")
        # Preserve a weight-zero transform bit-for-bit.  This is important for
        # paired ablations; harmless floating summation error is not rescaled.
        if abs(total - 1.0) <= 1e-12:
            return result
        return {action: value / total for action, value in result.items()}

    def _root_policy(self, state: SimState, root_player: int, *, apply_history: bool = False) -> dict[str, float]:
        result = self._normalize_policy(self.network.evaluate(state, root_player).policy)
        if apply_history and self.repetition_context is not None:
            result = self.repetition_context.adjust_root_prior(state, root_player, result)
        return result

    def _node(self, state: SimState, root_player: int, depth: int, *, root: bool = False) -> Node:
        policy = self._root_policy(state, root_player, apply_history=root)
        return Node(state, depth, {action: ActionStats(prior) for action, prior in policy.items()})

    def _fallback(self, state: SimState, root_player: int, raw_prior: dict[str, float]) -> str:
        # Immediate exact check ensures fallback never knowingly walks into a
        # blast that resolves on this very game step.
        candidates = sorted(raw_prior, key=lambda action: (-raw_prior[action], action))
        for action in candidates:
            joint = ["WAIT"] * len(state.agents)
            joint[root_player] = action
            successor = step(state, joint, list(range(len(state.active_ids()))))
            if successor.agents[root_player].alive:
                return action
        return candidates[0] if candidates else "WAIT"

    def _select(self, node: Node) -> str:
        scale = self.c_puct * np.sqrt(max(1, node.visits))
        return max(node.actions, key=lambda action: (
            node.actions[action].mean + scale * node.actions[action].prior / (1 + node.actions[action].visits),
            node.actions[action].prior, action,
        ))

    def _sample_opponent_action(self, state: SimState, player: int, root_player: int, rng) -> str:
        legal = legal_actions(state, player)
        if not legal:
            return "WAIT"
        rule = self.rule_network.opponent_action_policy(state, player, root_player)
        learned = self.network.opponent_action_policy(state, player, root_player)
        probabilities = np.asarray([self.rule_prior_mix * rule.get(action, 0.0) +
                                    (1.0 - self.rule_prior_mix) * learned.get(action, 0.0)
                                    for action in legal], dtype=float)
        if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
            probabilities = np.ones(len(legal), dtype=float)
        probabilities /= probabilities.sum()
        return str(rng.choice(legal, p=probabilities))

    def _value(self, state: SimState, root_player: int, initial_score: int,
               initial_crates: int) -> float:
        output = self.network.evaluate(state, root_player)
        agent = state.agents[root_player]
        if not agent.alive:
            return -1.0
        score_delta = agent.score - initial_score
        crate_delta = initial_crates - int(np.count_nonzero(state.field == 1))
        opponents_alive = sum(agent.alive for index, agent in enumerate(state.agents) if index != root_player)
        terminal_bonus = 0.35 if state.ended and opponents_alive == 0 else 0.0
        return float(np.clip(output.value - self.risk_penalty * output.risk
                             + 0.16 * score_delta
                             + self.crate_progress_weight * crate_delta
                             + terminal_bonus, -1.0, 1.0))

    def _chance_child(self, node: Node, action: str, root_player: int,
                      key: tuple[str, tuple[str, ...], tuple[int, ...]],
                      successor: SimState, rng) -> tuple[Node, bool]:
        """Progressively widen stochastic outcomes so the tree can deepen.

        Without a widening cap, joint opponent actions times the official
        permutation create thousands of unique children and almost every
        simulation stops at depth one.  Existing sampled outcomes represent
        the chance distribution once the cap is reached.
        """
        existing = node.children.get(key)
        if existing is not None:
            return existing, False
        outcomes = [child for child_key, child in node.children.items() if child_key[0] == action]
        edge_visits = node.actions[action].visits
        limit = max(1, int(self.widening_scale * np.sqrt(edge_visits + 1)))
        if len(outcomes) < limit:
            child = self._node(successor, root_player, node.depth + 1)
            node.children[key] = child
            return child, True
        weights = np.asarray([child.visits + 1 for child in outcomes], dtype=float)
        weights /= weights.sum()
        return outcomes[int(rng.choice(len(outcomes), p=weights))], False

    def _principal_variation(self, root: Node) -> tuple[str, ...]:
        result: list[str] = []
        node = root
        while node.actions and len(result) < self.horizon:
            action = max(node.actions, key=lambda item: (node.actions[item].visits, node.actions[item].mean, item))
            result.append(action)
            successors = [(child.visits, child) for key, child in node.children.items() if key[0] == action]
            if not successors:
                break
            node = max(successors, key=lambda pair: pair[0])[1]
        return tuple(result)

    def _rollout_value(self, state: SimState, root_player: int, depth: int, rng,
                       initial_score: int, initial_crates: int) -> tuple[float, int, bool]:
        """Complete a newly expanded chance child to the fixed Phase-1 horizon."""
        current = state
        while (depth < self.horizon and not current.ended and current.agents[root_player].alive):
            policy = self._root_policy(current, root_player)
            if not policy:
                break
            root_action = max(policy, key=lambda action: (policy[action], action))
            active = current.active_ids()
            joint = ["WAIT"] * len(current.agents)
            joint[root_player] = root_action
            for player in active:
                if player != root_player:
                    joint[player] = self._sample_opponent_action(current, player, root_player, rng)
            current = step(current, joint, tuple(int(value) for value in rng.permutation(len(active))))
            depth += 1
        return (self._value(current, root_player, initial_score, initial_crates), depth,
                not current.agents[root_player].alive)

    def search(self, state: SimState, root_player: int, *, simulations: int = 50,
               seed: int = 0, deadline_ms: float | None = None) -> SearchResult:
        started = perf_counter()
        deadline = float("inf") if deadline_ms is None else started + deadline_ms / 1000.0
        output = self.network.evaluate(state, root_player)
        raw_prior = self._normalize_policy(output.policy)
        if self.root_policy_transform is not None:
            raw_prior = self._validate_transformed_policy(
                self.root_policy_transform(state, root_player, dict(raw_prior)))
        if not raw_prior:
            return SearchResult("WAIT", "WAIT", {}, {}, (), 1.0, 0, 0, "root-dead", 0.0)
        fallback = self._fallback(state, root_player, raw_prior)
        root_prior = raw_prior
        if self.repetition_context is not None:
            root_prior = self.repetition_context.adjust_root_prior(state, root_player, root_prior)
        root = Node(state, 0, {action: ActionStats(prior)
                              for action, prior in root_prior.items()})
        rng = np.random.default_rng(seed)
        initial_score = state.agents[root_player].score
        initial_crates = int(np.count_nonzero(state.field == 1))
        completed, max_depth, deaths, reason = 0, 0, 0, "simulation-limit"
        for _ in range(simulations):
            if perf_counter() >= deadline:
                reason = "deadline"
                break
            node = root
            trace: list[tuple[Node, str]] = []
            leaf_died = False
            while True:
                if node.depth >= self.horizon or node.state.ended or not node.state.agents[root_player].alive:
                    value = self._value(node.state, root_player, initial_score, initial_crates)
                    leaf_died = not node.state.agents[root_player].alive
                    break
                action = self._select(node)
                active = node.state.active_ids()
                joint = ["WAIT"] * len(node.state.agents)
                joint[root_player] = action
                for player in active:
                    if player != root_player:
                        joint[player] = self._sample_opponent_action(node.state, player, root_player, rng)
                permutation = tuple(int(value) for value in rng.permutation(len(active)))
                successor = step(node.state, joint, permutation)
                key = (action, tuple(joint), permutation)
                trace.append((node, action))
                child, created = self._chance_child(node, action, root_player, key, successor, rng)
                if created:
                    value, rollout_depth, leaf_died = self._rollout_value(
                        successor, root_player, child.depth, rng, initial_score, initial_crates,
                    )
                    max_depth = max(max_depth, rollout_depth)
                    break
                node = child
            if leaf_died:
                deaths += 1
            for visited, action in trace:
                visited.visits += 1
                visited.value_sum += value
                edge = visited.actions[action]
                edge.visits += 1
                edge.value_sum += value
            completed += 1
        if not completed:
            return SearchResult(fallback, fallback, raw_prior, {action: 0 for action in raw_prior}, (),
                                output.risk, 0, 0, reason, (perf_counter() - started) * 1000.0)
        chosen = max(root.actions, key=lambda action: (root.actions[action].visits,
                                                        root.actions[action].mean, raw_prior[action], action))
        elapsed = (perf_counter() - started) * 1000.0
        return SearchResult(chosen, fallback, raw_prior,
                            {action: stats.visits for action, stats in root.actions.items()},
                            self._principal_variation(root), deaths / completed, completed, max_depth,
                            reason, elapsed)
