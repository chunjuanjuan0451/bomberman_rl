"""Track post-bomb movement decisions and bomb outcomes without learning."""

from __future__ import annotations

import json

import events as e


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"unsupported trace value: {type(value).__name__}")


def setup_training(self) -> None:
    if self.audit_trace_path.exists():
        raise RuntimeError(f"refusing to overwrite post-bomb movement trace: {self.audit_trace_path}")
    self._pending_decision = None
    self._active_bomb = None
    self._placements = []
    self._rounds_completed = 0
    self._expected_rounds = int(self.audit_protocol["collection"]["rounds_per_case"])


def _finalize_active(self, events, resolution: str) -> None:
    if self._active_bomb is None:
        return
    record = dict(self._active_bomb)
    record.update({
        "resolution": resolution,
        "selfkill": e.KILLED_SELF in events,
        "kills": int(events.count(e.KILLED_OPPONENT)),
        "crates": int(events.count(e.CRATE_DESTROYED)),
        "got_killed": e.GOT_KILLED in events,
        "resolution_events": list(events),
        "audited_decision_count": len(record["decisions"]),
        "avoidable_deviation_count": sum(bool(row["avoidable_deviation"]) for row in record["decisions"]),
        "has_avoidable_deviation": any(bool(row["avoidable_deviation"]) for row in record["decisions"]),
        "no_robust_action_count": sum(
            row["diagnostic"]["robust_action_count"] == 0 for row in record["decisions"]
        ),
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
        raise RuntimeError(f"post-bomb movement transition mismatch: expected={expected}, observed={observed}")
    if self._active_bomb is not None and pending["diagnostic"] is not None:
        self._active_bomb["decisions"].append({
            "step": pending["step"],
            "action": action,
            "position": pending["position"],
            "chosen_action_robust": pending["chosen_action_robust"],
            "avoidable_deviation": pending["avoidable_deviation"],
            "diagnostic": pending["diagnostic"],
            "events": list(events),
        })
    if action == "BOMB" and e.BOMB_DROPPED in events:
        if self._active_bomb is not None:
            raise RuntimeError("new own bomb placed before previous bomb resolved")
        self._active_bomb = {
            "round": pending["round"],
            "placement_step": pending["step"],
            "placement_position": pending["position"],
            "decisions": [],
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
        "kind": "model-a-cnn-n8-postbomb-movement-trace",
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
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
        raise RuntimeError("post-bomb movement audit exceeded registered round budget")


__all__ = ["end_of_round", "game_events_occurred", "setup_training"]
