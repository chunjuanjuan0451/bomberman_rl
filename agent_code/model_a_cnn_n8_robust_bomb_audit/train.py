"""Track outcomes of every actual bomb without learning or action overrides."""

from __future__ import annotations

import json

import events as e


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"unsupported trace value: {type(value).__name__}")


def setup_training(self) -> None:
    if self.audit_trace_path.exists():
        raise RuntimeError(f"refusing to overwrite robust-bomb trace: {self.audit_trace_path}")
    self._pending_decision = None
    self._active_bomb = None
    self._placements = []
    self._rounds_completed = 0
    self._expected_rounds = int(self.audit_protocol["collection"]["rounds_per_case"])


def _finalize_active(self, events, resolution: str) -> None:
    if self._active_bomb is None:
        return
    counts = {
        name: int(events.count(name))
        for name in (e.KILLED_SELF, e.KILLED_OPPONENT, e.CRATE_DESTROYED, e.GOT_KILLED)
    }
    record = dict(self._active_bomb)
    record.update({
        "resolution": resolution,
        "selfkill": bool(counts[e.KILLED_SELF]),
        "kills": counts[e.KILLED_OPPONENT],
        "crates": counts[e.CRATE_DESTROYED],
        "got_killed": bool(counts[e.GOT_KILLED]),
        "resolution_events": list(events),
    })
    self._placements.append(record)
    self._active_bomb = None


def _consume(self, old_game_state, action, events) -> None:
    pending = self._pending_decision
    if pending is None:
        return
    expected = (pending["round"], pending["step"], pending["action"])
    observed = (int(old_game_state["round"]), int(old_game_state["step"]), action)
    if expected != observed:
        raise RuntimeError(f"robust-bomb transition mismatch: expected={expected}, observed={observed}")
    if action == "BOMB" and e.BOMB_DROPPED in events:
        if self._active_bomb is not None:
            raise RuntimeError("new own bomb placed before previous bomb resolved")
        self._active_bomb = {
            "round": pending["round"],
            "placement_step": pending["step"],
            "q_gap": pending["q_gap"],
            "bomb_q": pending["bomb_q"],
            **pending["diagnostic"],
        }
    if e.BOMB_EXPLODED in events:
        _finalize_active(self, events, "exploded")
    self._pending_decision = None


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _consume(self, old_game_state, self_action, events)


def _save(self) -> None:
    self.audit_trace_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "model-a-cnn-n8-robust-bomb-trace",
        "protocol_sha256": self.audit_protocol_sha256,
        "case": self.audit_case,
        "checkpoint_sha256": self.audit_checkpoint_sha256,
        "rounds": self._rounds_completed,
        "policy_updates": 0,
        "actions_overridden": 0,
        "placement_count": len(self._placements),
        "placements": self._placements,
    }
    temporary = self.audit_trace_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(self.audit_trace_path)


def end_of_round(self, last_game_state, last_action, events):
    if self._pending_decision is not None:
        _consume(self, last_game_state, last_action, events)
    if self._active_bomb is not None:
        _finalize_active(self, events, "censored_round_end")
    self._rounds_completed += 1
    if self._rounds_completed == self._expected_rounds:
        _save(self)
    elif self._rounds_completed > self._expected_rounds:
        raise RuntimeError("robust-bomb audit exceeded registered round budget")


__all__ = ["end_of_round", "game_events_occurred", "setup_training"]
