"""Exact-v4 learner with only the action-selection epsilon floor changed."""

from __future__ import annotations

import json
from pathlib import Path

from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.features import GLOBAL_SIZE
from agent_code.model_a_dqn.network import torch

from .config import snapshot_path


EPSILON_FLOOR = 0.15


def _apply_floor(self) -> None:
    # v4._record first computes its original 1.0 -> 0.05 / 80k schedule.
    # Clamping afterwards preserves that exact slope until it reaches 0.15.
    self.epsilon = max(EPSILON_FLOOR, float(self.epsilon))


def setup_training(self):
    v4.setup_training(self)
    _apply_floor(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = v4.reward_from_events(events) + v4._state_shaping(old_game_state, new_game_state)
    v4._record(self, old_game_state, self_action, new_game_state, reward, done=False)
    _apply_floor(self)


def _checkpoint_payload(self) -> dict:
    return {
        "online_net": self.online_net.state_dict(),
        "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "completed_rounds": self.completed_rounds,
        "training_steps": self.training_steps,
        "gradient_steps": self.gradient_steps,
        "epsilon": self.epsilon,
        "epsilon_floor": EPSILON_FLOOR,
        "global_feature_size": GLOBAL_SIZE,
        "architecture": MODEL_ARCHITECTURE,
        "agent_seed": self.agent_seed,
        "protocol_sha256": self.protocol_sha256,
        "replica": self.replica,
    }


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_diagnostic(self, snapshot: Path) -> None:
    diagnostic = snapshot.with_suffix(".json")
    if diagnostic.exists():
        raise RuntimeError(f"refusing to overwrite diagnostic snapshot: {diagnostic}")
    payload = {
        "completed_rounds": self.completed_rounds,
        "training_steps": self.training_steps,
        "gradient_steps": self.gradient_steps,
        "epsilon": self.epsilon,
        "epsilon_floor": EPSILON_FLOOR,
        "cumulative_action_counts": dict(self.action_counts),
    }
    temporary = diagnostic.with_suffix(diagnostic.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(diagnostic)


def end_of_round(self, last_game_state, last_action, events):
    v4._record(self, last_game_state, last_action, None, v4.reward_from_events(events), done=True)
    _apply_floor(self)
    self.completed_rounds += 1
    if self.completed_rounds % v4.CHECKPOINT_EVERY_ROUNDS == 0:
        payload = _checkpoint_payload(self)
        snapshot = snapshot_path(self.protocol, self.replica, self.completed_rounds)
        if snapshot.exists():
            raise RuntimeError(f"refusing to overwrite diagnostic snapshot: {snapshot}")
        _atomic_torch_save(payload, snapshot)
        _write_diagnostic(self, snapshot)
        _atomic_torch_save(payload, self.checkpoint_path)
        self.logger.info(
            "Saved epsilon-floor %s after round %d (epsilon=%.3f).",
            self.replica, self.completed_rounds, self.epsilon,
        )
