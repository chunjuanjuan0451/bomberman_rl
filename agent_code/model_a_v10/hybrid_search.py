"""Value-reporting PUCT variant isolated to the v10.12 hybrid experiment."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .search import ActionStats, Node, StochasticPUCT
from .simulator import step


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    action: str
    fallback: str
    raw_prior: dict[str, float]
    root_visits: dict[str, int]
    root_values: dict[str, float]
    principal_variation: tuple[str, ...]
    risk: float
    simulations: int
    max_depth: int
    reason: str
    elapsed_ms: float


class ValueReportingPUCT(StochasticPUCT):
    """Keep the established search semantics and expose root action means."""

    def search(self, state, root_player: int, *, simulations: int = 50,
               seed: int = 0, deadline_ms: float | None = None) -> HybridSearchResult:
        started = perf_counter()
        deadline = float("inf") if deadline_ms is None else started + deadline_ms / 1000.0
        output = self.network.evaluate(state, root_player)
        raw_prior = self._normalize_policy(output.policy)
        if self.root_policy_transform is not None:
            raw_prior = self._validate_transformed_policy(
                self.root_policy_transform(state, root_player, dict(raw_prior)))
        if not raw_prior:
            return HybridSearchResult(
                "WAIT", "WAIT", {}, {}, {}, (), 1.0, 0, 0, "root-dead", 0.0)
        fallback = self._fallback(state, root_player, raw_prior)
        root_prior = raw_prior
        if self.repetition_context is not None:
            root_prior = self.repetition_context.adjust_root_prior(
                state, root_player, root_prior)
        root = Node(state, 0, {
            action: ActionStats(prior) for action, prior in root_prior.items()})
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
                if (node.depth >= self.horizon or node.state.ended
                        or not node.state.agents[root_player].alive):
                    value = self._value(
                        node.state, root_player, initial_score, initial_crates)
                    leaf_died = not node.state.agents[root_player].alive
                    break
                action = self._select(node)
                active = node.state.active_ids()
                joint = ["WAIT"] * len(node.state.agents)
                joint[root_player] = action
                for player in active:
                    if player != root_player:
                        joint[player] = self._sample_opponent_action(
                            node.state, player, root_player, rng)
                permutation = tuple(int(value) for value in rng.permutation(len(active)))
                successor = step(node.state, joint, permutation)
                key = (action, tuple(joint), permutation)
                trace.append((node, action))
                child, created = self._chance_child(
                    node, action, root_player, key, successor, rng)
                if created:
                    value, rollout_depth, leaf_died = self._rollout_value(
                        successor, root_player, child.depth, rng,
                        initial_score, initial_crates)
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
        visits = {action: stats.visits for action, stats in root.actions.items()}
        values = {action: stats.mean for action, stats in root.actions.items()}
        if not completed:
            return HybridSearchResult(
                fallback, fallback, raw_prior, visits, values, (), output.risk,
                0, 0, reason, (perf_counter() - started) * 1000.0)
        chosen = max(root.actions, key=lambda action: (
            root.actions[action].visits, root.actions[action].mean,
            raw_prior[action], action))
        return HybridSearchResult(
            chosen, fallback, raw_prior, visits, values,
            self._principal_variation(root), deaths / completed, completed, max_depth,
            reason, (perf_counter() - started) * 1000.0)
