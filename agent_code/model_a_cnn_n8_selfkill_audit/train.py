"""Record only terminal self-kill windows; never update the frozen network."""

from __future__ import annotations

import json

import events as e


WINDOW = 8


def _json_default(value):
    """Convert NumPy scalar diagnostics without weakening trace validation."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"unsupported trace value: {type(value).__name__}")


def setup_training(self) -> None:
    if self.audit_trace_path.exists():
        raise RuntimeError(f"refusing to overwrite audit trace: {self.audit_trace_path}")
    self._pending_decision = None
    self._episode_rows = []
    self._selfkill_cases = []
    self._rounds_completed = 0
    self._event_totals = {}
    self._expected_rounds = int(self.audit_protocol["collection"]["rounds_per_case"])


def _record(self, old_game_state, action, new_game_state, events) -> None:
    if old_game_state is None or self._pending_decision is None:
        return
    pending = self._pending_decision
    expected = (pending["round"], pending["step"], pending["action"])
    observed = (int(old_game_state["round"]), int(old_game_state["step"]), action)
    if expected != observed:
        raise RuntimeError(f"CNN self-kill transition mismatch: expected={expected}, observed={observed}")
    row = dict(pending)
    row["events"] = list(events)
    row["next_position"] = None if new_game_state is None else list(new_game_state["self"][3])
    self._episode_rows.append(row)
    for name in events:
        self._event_totals[name] = self._event_totals.get(name, 0) + 1
    self._pending_decision = None


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _record(self, old_game_state, self_action, new_game_state, events)


def _merge_terminal_events(self, last_game_state, last_action, events) -> None:
    if not self._episode_rows:
        raise RuntimeError("survivor terminal has no recorded transition")
    row = self._episode_rows[-1]
    expected = (int(last_game_state["round"]), int(last_game_state["step"]), last_action)
    observed = (row["round"], row["step"], row["action"])
    if expected != observed:
        raise RuntimeError(f"CNN self-kill terminal mismatch: expected={expected}, observed={observed}")
    previous = set(row["events"])
    for name in events:
        if name not in previous:
            row["events"].append(name)
            self._event_totals[name] = self._event_totals.get(name, 0) + 1


def _save(self) -> None:
    self.audit_trace_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "model-a-cnn-n8-selfkill-trace",
        "protocol_sha256": self.audit_protocol_sha256,
        "case": self.audit_case,
        "checkpoint_sha256": self.audit_checkpoint_sha256,
        "rounds": self._rounds_completed,
        "policy_updates": 0,
        "actions_overridden": 0,
        "event_totals": self._event_totals,
        "selfkill_count": len(self._selfkill_cases),
        "selfkill_cases": self._selfkill_cases,
    }
    temporary = self.audit_trace_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(self.audit_trace_path)


def end_of_round(self, last_game_state, last_action, events):
    if self._pending_decision is not None:
        _record(self, last_game_state, last_action, None, events)
    else:
        _merge_terminal_events(self, last_game_state, last_action, events)
    if e.KILLED_SELF in events:
        self._selfkill_cases.append({
            "round": int(last_game_state["round"]),
            "terminal_step": int(last_game_state["step"]),
            "window": self._episode_rows[-WINDOW:],
        })
    self._episode_rows = []
    self._rounds_completed += 1
    if self._rounds_completed == self._expected_rounds:
        _save(self)
    elif self._rounds_completed > self._expected_rounds:
        raise RuntimeError("CNN self-kill audit exceeded registered round budget")


__all__ = ["end_of_round", "game_events_occurred", "setup_training"]
