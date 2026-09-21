"""Record passive Oracle decisions and official outcomes; never learn a policy."""

from __future__ import annotations

import json

import numpy as np
import events as e

from agent_code.model_a_dqn.features import ACTIONS


TRACE_FIELDS = (
    "round", "step", "action", "legal_mask", "bomb_legal", "oracle_evaluated",
    "oracle_timeout", "strict_opportunity", "guaranteed_traps", "affected_opponents",
    "max_space_reduction", "own_bottleneck", "own_terminal_positions", "kill_event",
    "self_event", "got_killed_event",
)


def setup_training(self):
    if self.trace_path.exists():
        raise RuntimeError(f"refusing to overwrite counterfactual signal trace: {self.trace_path}")
    self._trace = {field: [] for field in TRACE_FIELDS}
    self._pending_decision = None
    self._seen_transition_keys = set()
    self._completed_rounds = 0
    self._expected_rounds = int(self.protocol["collection"]["rounds_per_case"])


def _record(self, game_state, action: str, events) -> None:
    if game_state is None or action not in ACTIONS:
        return
    key = (int(game_state["round"]), int(game_state["step"]), action)
    if key in self._seen_transition_keys:
        return
    pending = self._pending_decision
    if pending is None:
        raise RuntimeError(f"missing passive Oracle decision for transition {key}")
    expected = (pending["round"], pending["step"], ACTIONS[pending["action"]])
    if expected != key:
        raise RuntimeError(f"passive Oracle transition mismatch: expected={expected}, observed={key}")
    self._seen_transition_keys.add(key)
    pending = dict(pending)
    pending.update({
        "kill_event": e.KILLED_OPPONENT in events,
        "self_event": e.KILLED_SELF in events,
        "got_killed_event": e.GOT_KILLED in events,
    })
    for field in TRACE_FIELDS:
        self._trace[field].append(pending[field])
    self._pending_decision = None


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _record(self, old_game_state, self_action, events)


def _save(self) -> None:
    if not self._trace["action"]:
        raise RuntimeError("counterfactual signal trace is empty")
    self.trace_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = self.trace_path.with_suffix(self.trace_path.suffix + ".tmp")
    arrays = {
        "round": np.asarray(self._trace["round"], dtype=np.int16),
        "step": np.asarray(self._trace["step"], dtype=np.int16),
        "action": np.asarray(self._trace["action"], dtype=np.int8),
        "legal_mask": np.stack(self._trace["legal_mask"]).astype(np.bool_),
        "bomb_legal": np.asarray(self._trace["bomb_legal"], dtype=np.bool_),
        "oracle_evaluated": np.asarray(self._trace["oracle_evaluated"], dtype=np.bool_),
        "oracle_timeout": np.asarray(self._trace["oracle_timeout"], dtype=np.bool_),
        "strict_opportunity": np.asarray(self._trace["strict_opportunity"], dtype=np.bool_),
        "guaranteed_traps": np.asarray(self._trace["guaranteed_traps"], dtype=np.int8),
        "affected_opponents": np.asarray(self._trace["affected_opponents"], dtype=np.int8),
        "max_space_reduction": np.asarray(self._trace["max_space_reduction"], dtype=np.float32),
        "own_bottleneck": np.asarray(self._trace["own_bottleneck"], dtype=np.int16),
        "own_terminal_positions": np.asarray(self._trace["own_terminal_positions"], dtype=np.int16),
        "kill_event": np.asarray(self._trace["kill_event"], dtype=np.bool_),
        "self_event": np.asarray(self._trace["self_event"], dtype=np.bool_),
        "got_killed_event": np.asarray(self._trace["got_killed_event"], dtype=np.bool_),
        "metadata": np.asarray(json.dumps({
            "schema_version": 1,
            "kind": "model-a-v4-counterfactual-signal-trace",
            "protocol_sha256": self.protocol_sha256,
            "case": self.case_label,
            "stratum": self.stratum,
            "parent_sha256": self.parent_sha256,
            "rounds": self._expected_rounds,
            "collection_epsilon": self.collection_epsilon,
            "oracle_deadline_seconds": self.oracle_deadline_seconds,
            "policy_updates": 0,
        }, sort_keys=True)),
    }
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(self.trace_path)


def end_of_round(self, last_game_state, last_action, events):
    if self._pending_decision is not None:
        _record(self, last_game_state, last_action, events)
    self._completed_rounds += 1
    if self._completed_rounds == self._expected_rounds:
        _save(self)
    elif self._completed_rounds > self._expected_rounds:
        raise RuntimeError("counterfactual signal collector exceeded its registered round budget")
