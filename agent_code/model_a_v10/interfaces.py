"""Inference boundary for v10 search.

Phase 1 deliberately uses this deterministic heuristic implementation rather
than loading any v4/v9 weight.  A learned multi-head model can later implement
the same ``evaluate`` interface without changing the search contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulator import MOVES, SimState, _blast_coords


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def legal_actions(state: SimState, player: int) -> tuple[str, ...]:
    agent = state.agents[player]
    if not agent.alive:
        return ()
    occupied = {(bomb.x, bomb.y) for bomb in state.bombs}
    occupied.update((other.x, other.y) for other in state.agents if other.alive)
    legal = ["WAIT"]
    for action, (dx, dy, _) in MOVES.items():
        x, y = agent.x + dx, agent.y + dy
        if state.field[x, y] == 0 and (x, y) not in occupied:
            legal.append(action)
    if agent.bombs_left:
        legal.append("BOMB")
    return tuple(legal)


@dataclass(frozen=True, slots=True)
class NetworkOutput:
    """The four heads required by the v10 search interface."""

    policy: dict[str, float]
    value: float
    opponent_policy: dict[str, float]
    risk: float


class HeuristicNetwork:
    """Explainable, no-weight Phase-1 substitute for a learned network."""

    def _policy(self, state: SimState, player: int, target_player: int | None = None) -> dict[str, float]:
        legal = legal_actions(state, player)
        if not legal:
            return {}
        agent = state.agents[player]
        coins = [position for position, visible in state.coins.items() if visible]
        result = {action: 1.0 for action in legal}
        if coins:
            before = min(abs(agent.x - x) + abs(agent.y - y) for x, y in coins)
            for action in legal:
                if action in MOVES:
                    dx, dy, _ = MOVES[action]
                    after = min(abs(agent.x + dx - x) + abs(agent.y + dy - y) for x, y in coins)
                    if after < before:
                        result[action] += 2.0
        if target_player is not None and state.agents[target_player].alive:
            target = state.agents[target_player]
            before = abs(agent.x - target.x) + abs(agent.y - target.y)
            for action in legal:
                if action in MOVES:
                    dx, dy, _ = MOVES[action]
                    after = abs(agent.x + dx - target.x) + abs(agent.y + dy - target.y)
                    if after < before:
                        result[action] += 2.0
        adjacent_crate = any(state.field[agent.x + dx, agent.y + dy] == 1
                             for dx, dy, _ in MOVES.values())
        adjacent_opponent = any(other.alive and abs(agent.x - other.x) + abs(agent.y - other.y) == 1
                                for other in state.agents if other is not agent)
        if "BOMB" in result and (adjacent_crate or adjacent_opponent):
            result["BOMB"] += 3.0
        for bomb in state.bombs:
            if bomb.timer > 1:
                continue
            blast = set(_blast_coords(state.field, bomb.x, bomb.y, bomb.power))
            for action in legal:
                if action == "WAIT" or action == "BOMB":
                    position = (agent.x, agent.y)
                else:
                    dx, dy, _ = MOVES[action]
                    position = (agent.x + dx, agent.y + dy)
                if position in blast:
                    result[action] *= 0.03
        total = sum(result.values())
        return {action: score / total for action, score in result.items()}

    def evaluate(self, state: SimState, player: int) -> NetworkOutput:
        agent = state.agents[player]
        policy = self._policy(state, player)
        opponent_policy = dict(policy)
        if not agent.alive:
            return NetworkOutput(policy, -1.0, opponent_policy, 1.0)
        # Value is deliberately bounded and only state-observable: score,
        # current survival and available mobility.  It is not a learned value.
        mobility = max(0, len(legal_actions(state, player)) - 2)
        nearest_coin = min((abs(agent.x - x) + abs(agent.y - y)
                            for (x, y), visible in state.coins.items() if visible), default=8)
        value = float(np.clip(0.08 * agent.score + 0.05 * mobility - 0.02 * nearest_coin, -1.0, 1.0))
        risk = 0.0
        for explosion in state.explosions:
            if explosion.dangerous and (agent.x, agent.y) in explosion.coords:
                risk = 1.0
        for bomb in state.bombs:
            if bomb.timer <= 1 and (agent.x, agent.y) in _blast_coords(state.field, bomb.x, bomb.y, bomb.power):
                risk = max(risk, 0.75)
        return NetworkOutput(policy, value, opponent_policy, risk)

    def opponent_action_policy(self, state: SimState, player: int,
                               root_player: int) -> dict[str, float]:
        """Rule prior for one concrete opponent, from that opponent's position."""
        return self._policy(state, player, target_player=root_player)


class LinearExpertNetwork:
    """Small trainable four-head full-board baseline for Expert Iteration.

    It purposefully keeps the same inference interface as ``HeuristicNetwork``
    so every collected target is generated by search, never a standalone
    policy.  The linear form makes CPU self-play reproducible and inexpensive
    during the first curriculum stage; it is not a replacement for Phase-4
    model selection.
    """

    def __init__(self, input_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.policy_w = rng.normal(0, 0.002, (input_size, len(ACTIONS))).astype(np.float32)
        self.policy_b = np.zeros(len(ACTIONS), dtype=np.float32)
        self.value_w = np.zeros(input_size, dtype=np.float32)
        self.value_b = np.float32(0)
        self.opponent_w = rng.normal(0, 0.002, (input_size, len(ACTIONS))).astype(np.float32)
        self.opponent_b = np.zeros(len(ACTIONS), dtype=np.float32)
        self.risk_w = np.zeros(input_size, dtype=np.float32)
        self.risk_b = np.float32(0)
        self.last_health: dict[str, float] = {}

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        values = np.exp(shifted)
        return values / values.sum(axis=-1, keepdims=True)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))

    def features(self, state: SimState, player: int = 0) -> np.ndarray:
        from .runtime import state_features
        return state_features(state, player).reshape(-1).astype(np.float32, copy=False)

    @staticmethod
    def prepare_features(features: np.ndarray) -> np.ndarray:
        """The single train/inference preprocessing path.

        Float64 is intentional for the small linear bootstrap model.  It avoids
        the spurious/unstable float32 Accelerate matmul behaviour observed in
        the two aborted Phase-3 attempts while parameters remain float32.
        """
        values = np.asarray(features)
        if values.ndim in (3, 4):
            from .runtime import center_feature_maps
            values = center_feature_maps(values)
            if values.ndim == 3:
                values = values[None, ...]
            values = values.reshape(values.shape[0], -1)
        elif values.ndim == 1:
            values = values[None, :]
        elif values.ndim != 2:
            raise ValueError("unsupported v10 feature shape")
        values = values.astype(np.float64, copy=False) / 4.0
        if not np.isfinite(values).all():
            raise FloatingPointError("non-finite v10 input features")
        return values

    def batch_outputs(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = self.prepare_features(features)
        # ``einsum(optimize=False)`` avoids a reproducible Apple Accelerate
        # float-status bug in NumPy 2.2 where finite matmul results emit
        # overflow/divide-by-zero warnings and can poison subsequent calls.
        policy_logits = np.einsum("bi,ij->bj", values, self.policy_w, optimize=False) + self.policy_b
        value_logits = np.einsum("bi,i->b", values, self.value_w, optimize=False) + self.value_b
        opponent_logits = np.einsum("bi,ij->bj", values, self.opponent_w, optimize=False) + self.opponent_b
        risk_logits = np.einsum("bi,i->b", values, self.risk_w, optimize=False) + self.risk_b
        policy = self._softmax(policy_logits)
        value = np.tanh(value_logits)
        opponent = self._softmax(opponent_logits)
        risk = self._sigmoid(risk_logits)
        return policy, value, opponent, risk

    def evaluate(self, state: SimState, player: int) -> NetworkOutput:
        agent = state.agents[player]
        features = self.features(state, player)[None, :]
        policy_values, value, opponent_values, risk = self.batch_outputs(features)
        legal = legal_actions(state, player)
        if not legal:
            return NetworkOutput({}, -1.0, {}, 1.0)
        full_policy = policy_values[0]
        legal_weights = np.asarray([full_policy[ACTIONS.index(action)] for action in legal], dtype=float)
        legal_weights /= legal_weights.sum()
        policy = {action: float(weight) for action, weight in zip(legal, legal_weights)}
        opponent = {action: float(opponent_values[0, ACTIONS.index(action)]) for action in ACTIONS}
        return NetworkOutput(policy, float(value[0]) if agent.alive else -1.0, opponent,
                             float(risk[0]) if agent.alive else 1.0)

    def _policy(self, state: SimState, player: int, target_player: int | None = None) -> dict[str, float]:
        return self.evaluate(state, player).policy

    def opponent_action_policy(self, state: SimState, player: int,
                               root_player: int) -> dict[str, float]:
        """Mask the learned opponent head to the selected opponent's legal actions."""
        output = self.evaluate(state, root_player)
        legal = legal_actions(state, player)
        weights = np.asarray([output.opponent_policy.get(action, 0.0) for action in legal], dtype=float)
        if not legal:
            return {}
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(legal), dtype=float)
        weights /= weights.sum()
        return {action: float(weight) for action, weight in zip(legal, weights)}

    def parameter_health(self) -> dict[str, float]:
        parameters = {
            "policy_w": self.policy_w, "policy_b": self.policy_b,
            "value_w": self.value_w, "value_b": np.atleast_1d(self.value_b),
            "opponent_w": self.opponent_w, "opponent_b": self.opponent_b,
            "risk_w": self.risk_w, "risk_b": np.atleast_1d(self.risk_b),
        }
        bad = [name for name, values in parameters.items() if not np.isfinite(values).all()]
        if bad:
            raise FloatingPointError(f"non-finite v10 parameters: {bad}")
        return {f"max_abs_{name}": float(np.max(np.abs(values)))
                for name, values in parameters.items()}

    def train_batch(self, features: np.ndarray, policy_target: np.ndarray, value_target: np.ndarray,
                    opponent_target: np.ndarray, opponent_mask: np.ndarray, risk_target: np.ndarray,
                    learning_rate: float) -> dict[str, float]:
        features = np.asarray(features, dtype=np.float32)
        policy_target = np.asarray(policy_target, dtype=np.float64)
        value_target = np.asarray(value_target, dtype=np.float64)
        opponent_target = np.asarray(opponent_target, dtype=np.float64)
        opponent_mask = np.asarray(opponent_mask, dtype=np.float64)
        risk_target = np.asarray(risk_target, dtype=np.float64)
        for name, values in (("policy_target", policy_target), ("value_target", value_target),
                             ("opponent_target", opponent_target), ("opponent_mask", opponent_mask),
                             ("risk_target", risk_target)):
            if not np.isfinite(values).all():
                raise FloatingPointError(f"non-finite {name}")
        prepared = self.prepare_features(features)
        policy, value, opponent, risk = self.batch_outputs(features)
        count = features.shape[0]
        policy_grad = np.clip((policy - policy_target) / count, -1.0, 1.0)
        self.policy_w -= learning_rate * np.einsum(
            "bi,bj->ij", prepared, policy_grad, optimize=False).astype(np.float32)
        self.policy_b -= learning_rate * policy_grad.sum(axis=0).astype(np.float32)
        value_grad = np.clip(2.0 * (value - value_target) * (1.0 - value ** 2) / count, -1.0, 1.0)
        self.value_w -= learning_rate * np.einsum(
            "bi,b->i", prepared, value_grad, optimize=False).astype(np.float32)
        self.value_b -= np.float32(learning_rate * value_grad.sum())
        active = max(1, int(opponent_mask.sum()))
        opponent_grad = np.clip((opponent - opponent_target) * opponent_mask[:, None] / active, -1.0, 1.0)
        self.opponent_w -= learning_rate * np.einsum(
            "bi,bj->ij", prepared, opponent_grad, optimize=False).astype(np.float32)
        self.opponent_b -= learning_rate * opponent_grad.sum(axis=0).astype(np.float32)
        risk_grad = np.clip((risk - risk_target) / count, -1.0, 1.0)
        self.risk_w -= learning_rate * np.einsum(
            "bi,b->i", prepared, risk_grad, optimize=False).astype(np.float32)
        self.risk_b -= np.float32(learning_rate * risk_grad.sum())
        for parameter in (self.policy_w, self.policy_b, self.value_w,
                          self.opponent_w, self.opponent_b, self.risk_w):
            np.clip(parameter, -1.0, 1.0, out=parameter)
        self.value_b = np.float32(np.clip(self.value_b, -1.0, 1.0))
        self.risk_b = np.float32(np.clip(self.risk_b, -1.0, 1.0))
        self.last_health = self.parameter_health()
        losses = {"policy_ce": float(-np.mean(np.sum(policy_target * np.log(policy + 1e-12), axis=1))),
                "value_mse": float(np.mean((value - value_target) ** 2)),
                "opponent_ce": float(-np.sum(opponent_mask[:, None] * opponent_target * np.log(opponent + 1e-12)) / active),
                "risk_bce": float(-np.mean(risk_target * np.log(risk + 1e-12) + (1-risk_target) * np.log(1-risk + 1e-12)))}
        if not all(np.isfinite(value) for value in losses.values()):
            raise FloatingPointError(f"non-finite v10 losses: {losses}")
        return losses | self.last_health

    def save(self, path) -> None:
        np.savez_compressed(path, policy_w=self.policy_w, policy_b=self.policy_b, value_w=self.value_w,
                            value_b=self.value_b, opponent_w=self.opponent_w, opponent_b=self.opponent_b,
                            risk_w=self.risk_w, risk_b=self.risk_b)

    @classmethod
    def load(cls, path) -> "LinearExpertNetwork":
        payload = np.load(path)
        model = cls(int(payload["value_w"].shape[0]))
        for name in ("policy_w", "policy_b", "value_w", "value_b",
                     "opponent_w", "opponent_b", "risk_w", "risk_b"):
            setattr(model, name, payload[name].copy())
        model.parameter_health()
        return model


class BlendedExpertNetwork:
    """Conservative teacher/student bridge for early Expert Iteration."""

    def __init__(self, student: LinearExpertNetwork, student_weight: float | None = None, *,
                 policy_weight: float | None = None, value_weight: float | None = None,
                 opponent_weight: float | None = None, risk_weight: float | None = None):
        base = 0.0 if student_weight is None else float(student_weight)
        weights = {
            "policy": base if policy_weight is None else float(policy_weight),
            "value": base if value_weight is None else float(value_weight),
            "opponent": base if opponent_weight is None else float(opponent_weight),
            "risk": base if risk_weight is None else float(risk_weight),
        }
        invalid = {name: weight for name, weight in weights.items() if not 0.0 <= weight <= 1.0}
        if invalid:
            raise ValueError(f"student head weights must be in [0, 1]: {invalid}")
        self.student = student
        self.teacher = HeuristicNetwork()
        self.student_weight = base
        self.weights = weights

    @staticmethod
    def _mix_policy(teacher: dict[str, float], student: dict[str, float],
                    weight: float) -> dict[str, float]:
        if weight == 0.0:
            return dict(teacher)
        if weight == 1.0:
            return dict(student)
        actions = set(teacher) | set(student)
        mixed = {action: ((1.0 - weight) * teacher.get(action, 0.0)
                          + weight * student.get(action, 0.0)) for action in actions}
        total = sum(mixed.values())
        return {action: value / total for action, value in mixed.items()} if total else {}

    def evaluate(self, state: SimState, player: int) -> NetworkOutput:
        teacher = self.teacher.evaluate(state, player)
        student = self.student.evaluate(state, player)
        return NetworkOutput(
            self._mix_policy(teacher.policy, student.policy, self.weights["policy"]),
            (1.0 - self.weights["value"]) * teacher.value + self.weights["value"] * student.value,
            self._mix_policy(teacher.opponent_policy, student.opponent_policy,
                             self.weights["opponent"]),
            (1.0 - self.weights["risk"]) * teacher.risk + self.weights["risk"] * student.risk,
        )

    def opponent_action_policy(self, state: SimState, player: int,
                               root_player: int) -> dict[str, float]:
        return self._mix_policy(
            self.teacher.opponent_action_policy(state, player, root_player),
            self.student.opponent_action_policy(state, player, root_player),
            self.weights["opponent"],
        )
