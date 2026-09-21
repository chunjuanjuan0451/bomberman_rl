"""Write the passive reward/credit/attack-chain trace without policy learning."""

from __future__ import annotations

import json

import numpy as np

from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.features import ACTIONS

from .callbacks import _nearest_distance
from .config import CUSTOM_EVENTS_CONFIRMED_ABSENT, EVENT_NAMES


TRACE_FIELDS = (
    "round", "step", "action", "legal_mask", "online_q_values", "target_q_values",
    "global_features", "opponent_count", "nearest_opponent_distance",
    "next_nearest_opponent_distance", "approach_step", "bomb_legal", "oracle_evaluated",
    "oracle_timeout", "guaranteed_traps", "affected_opponents", "max_space_reduction",
    "own_bottleneck", "own_terminal_positions", "event_counts", "event_reward_components",
    "state_shaping", "total_reward",
)


def setup_training(self):
    if self.trace_path.exists():
        raise RuntimeError(f"refusing to overwrite training-system trace: {self.trace_path}")
    self._trace = {field: [] for field in TRACE_FIELDS}
    self._pending_decision = None
    self._seen_transition_keys = set()
    self._completed_rounds = 0
    self._expected_rounds = int(self.protocol["collection"]["rounds_per_case"])


def _event_arrays(events) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([events.count(name) for name in EVENT_NAMES], dtype=np.int16)
    components = np.asarray([
        count * float(v4.REWARD_BY_EVENT.get(name, 0.0))
        for name, count in zip(EVENT_NAMES, counts)
    ], dtype=np.float32)
    return counts, components


def _record(self, old_game_state, action: str, new_game_state, events) -> None:
    if old_game_state is None or action not in ACTIONS:
        return
    key = (int(old_game_state["round"]), int(old_game_state["step"]), action)
    if key in self._seen_transition_keys:
        return
    pending = self._pending_decision
    if pending is None:
        raise RuntimeError(f"missing training-system decision for transition {key}")
    expected = (pending["round"], pending["step"], ACTIONS[pending["action"]])
    if expected != key:
        raise RuntimeError(f"training-system transition mismatch: expected={expected}, observed={key}")
    counts, components = _event_arrays(tuple(events))
    shaping = float(v4._state_shaping(old_game_state, new_game_state))
    next_distance = _nearest_distance(new_game_state)
    old_distance = float(pending["nearest_opponent_distance"])
    approach = old_distance >= 0.0 and next_distance >= 0.0 and next_distance < old_distance
    row = dict(pending)
    row.update({
        "next_nearest_opponent_distance": next_distance,
        "approach_step": approach,
        "event_counts": counts,
        "event_reward_components": components,
        "state_shaping": shaping,
        "total_reward": float(components.sum()) + shaping,
    })
    self._seen_transition_keys.add(key)
    for field in TRACE_FIELDS:
        self._trace[field].append(row[field])
    self._pending_decision = None


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _record(self, old_game_state, self_action, new_game_state, events)


def _merge_survivor_terminal(self, last_game_state, last_action: str, events) -> None:
    """Merge SURVIVED_ROUND after an already delivered final per-step callback.

    The environment sends ``game_events_occurred`` to agents that remain alive,
    then appends ``SURVIVED_ROUND`` and sends the same final-step event list to
    ``end_of_round``.  Dead agents skip the former callback and are handled by
    the ordinary pending-decision path in ``end_of_round``.
    """
    if last_game_state is None or last_action not in ACTIONS or not self._trace["action"]:
        raise RuntimeError("training-system survivor terminal has no recorded final transition")
    expected = (
        int(last_game_state["round"]), int(last_game_state["step"]), ACTIONS.index(last_action),
    )
    observed = (
        int(self._trace["round"][-1]), int(self._trace["step"][-1]), int(self._trace["action"][-1]),
    )
    if observed != expected:
        raise RuntimeError(f"training-system survivor terminal mismatch: expected={expected}, observed={observed}")
    final_counts, final_components = _event_arrays(tuple(events))
    recorded_counts = np.asarray(self._trace["event_counts"][-1], dtype=np.int16)
    delta = final_counts.astype(np.int32) - recorded_counts.astype(np.int32)
    if np.any(delta < 0):
        raise RuntimeError("training-system survivor terminal lost a final-step event")
    survived_index = EVENT_NAMES.index("SURVIVED_ROUND")
    unexpected = delta.copy()
    unexpected[survived_index] = 0
    if np.any(unexpected != 0) or delta[survived_index] != 1:
        raise RuntimeError(f"training-system survivor terminal event delta changed: {delta.tolist()}")
    self._trace["event_counts"][-1] = final_counts
    self._trace["event_reward_components"][-1] = final_components
    self._trace["total_reward"][-1] = (
        float(final_components.sum()) + float(self._trace["state_shaping"][-1])
    )


def _save(self) -> None:
    if not self._trace["action"]:
        raise RuntimeError("training-system trace is empty")
    self.trace_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = self.trace_path.with_suffix(self.trace_path.suffix + ".tmp")
    matrix_fields = {"legal_mask", "online_q_values", "target_q_values", "global_features", "event_counts", "event_reward_components"}
    bool_fields = {"approach_step", "bomb_legal", "oracle_evaluated", "oracle_timeout"}
    int16_fields = {"round", "step", "own_bottleneck", "own_terminal_positions"}
    int8_fields = {"action", "opponent_count", "guaranteed_traps", "affected_opponents"}
    arrays = {}
    for field in TRACE_FIELDS:
        values = self._trace[field]
        if field in matrix_fields:
            arrays[field] = np.stack(values)
        elif field in bool_fields:
            arrays[field] = np.asarray(values, dtype=np.bool_)
        elif field in int16_fields:
            arrays[field] = np.asarray(values, dtype=np.int16)
        elif field in int8_fields:
            arrays[field] = np.asarray(values, dtype=np.int8)
        else:
            arrays[field] = np.asarray(values, dtype=np.float32)
    arrays["metadata"] = np.asarray(json.dumps({
        "schema_version": 1, "kind": "model-a-v4-training-system-trace",
        "protocol_sha256": self.protocol_sha256, "case": self.case_label,
        "stratum": self.stratum, "parent_sha256": self.parent_sha256,
        "rounds": self._expected_rounds, "collection_epsilon": self.collection_epsilon,
        "oracle_deadline_seconds": self.oracle_deadline_seconds,
        "event_names": list(EVENT_NAMES),
        "custom_events_confirmed_absent": list(CUSTOM_EVENTS_CONFIRMED_ABSENT),
        "policy_updates": 0, "actions_overridden": 0,
    }, sort_keys=True))
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(self.trace_path)


def end_of_round(self, last_game_state, last_action, events):
    if self._pending_decision is not None:
        _record(self, last_game_state, last_action, None, events)
    else:
        _merge_survivor_terminal(self, last_game_state, last_action, events)
    self._completed_rounds += 1
    if self._completed_rounds == self._expected_rounds:
        _save(self)
    elif self._completed_rounds > self._expected_rounds:
        raise RuntimeError("training-system collector exceeded registered round budget")
