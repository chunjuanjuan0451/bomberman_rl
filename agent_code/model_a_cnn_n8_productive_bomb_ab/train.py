"""Read-only intervention tracing for the productive-bomb A/B."""

from __future__ import annotations

import json


def setup_training(self) -> None:
    if self.audit_trace_path.exists():
        raise RuntimeError(f"refusing to overwrite productive-bomb trace: {self.audit_trace_path}")
    self._pending_decision = None
    self._active_bomb = False
    self._active_bomb_position = None
    self._rounds_completed = 0
    self._expected_rounds = int(self.audit_protocol["collection"]["rounds_per_arm_case"])
    self._counts = {
        "decisions": 0,
        "bomb_legal_decisions": 0,
        "productive_bomb_legal_decisions": 0,
        "unproductive_bomb_legal_decisions": 0,
        "original_bomb_choices": 0,
        "unproductive_original_bomb_choices": 0,
        "restriction_available_decisions": 0,
        "fallback_decisions": 0,
        "actions_changed": 0,
        "changed_to_move": 0,
        "changed_to_wait": 0,
    }


def _consume(self, old_game_state, action) -> None:
    pending = self._pending_decision
    if pending is None:
        return
    expected = (pending["round"], pending["step"], pending["action"])
    observed = (int(old_game_state["round"]), int(old_game_state["step"]), action)
    if expected != observed:
        raise RuntimeError(f"productive-bomb transition mismatch: expected={expected}, observed={observed}")
    self._counts["decisions"] += 1
    if pending["bomb_legal"]:
        self._counts["bomb_legal_decisions"] += 1
        key = "productive_bomb_legal_decisions" if pending["bomb_productive"] else "unproductive_bomb_legal_decisions"
        self._counts[key] += 1
    if pending["original_action"] == "BOMB":
        self._counts["original_bomb_choices"] += 1
        if not pending["bomb_productive"]:
            self._counts["unproductive_original_bomb_choices"] += 1
    if pending["restriction_available"]:
        self._counts["restriction_available_decisions"] += 1
    if pending["fallback_used"]:
        self._counts["fallback_decisions"] += 1
    if pending["action_changed"]:
        self._counts["actions_changed"] += 1
        self._counts["changed_to_wait" if action == "WAIT" else "changed_to_move"] += 1
    self._pending_decision = None


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _consume(self, old_game_state, self_action)


def _save(self) -> None:
    self.audit_trace_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "model-a-cnn-n8-productive-bomb-ab-trace",
        "protocol_sha256": self.audit_protocol_sha256,
        "case": self.audit_case,
        "arm": self.audit_arm,
        "checkpoint_sha256": self.audit_checkpoint_sha256,
        "rounds": self._rounds_completed,
        "policy_updates": 0,
        "counts": self._counts,
    }
    temporary = self.audit_trace_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(self.audit_trace_path)


def end_of_round(self, last_game_state, last_action, events):
    if self._pending_decision is not None:
        _consume(self, last_game_state, last_action)
    self._active_bomb = False
    self._active_bomb_position = None
    self._rounds_completed += 1
    if self._rounds_completed == self._expected_rounds:
        _save(self)
    elif self._rounds_completed > self._expected_rounds:
        raise RuntimeError("productive-bomb A/B exceeded registered round budget")


__all__ = ["end_of_round", "game_events_occurred", "setup_training"]
