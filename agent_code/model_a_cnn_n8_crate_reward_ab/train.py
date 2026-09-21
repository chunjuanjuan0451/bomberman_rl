"""n=8 Double-DQN training with one arm-specific crate reward."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import events as e

from agent_code.model_a_cnn_n8.features import ACTIONS
from agent_code.model_a_cnn_n8.network import ARCHITECTURE, torch
from agent_code.model_a_cnn_n8.rewards import OFFICIAL
from agent_code.model_a_cnn_n8.train import _append, _merge_survivor, setup_training as _base_setup_training
from agent_code.model_a_cnn_n8_league_train.train import _advance_bomb_state, _raw, game_events_occurred


def setup_training(self) -> None:
    # Each formal arm runs in its own process. Mutating this module-level reward
    # table therefore changes exactly one registered scalar without leaking to
    # opponents, evaluation, or another arm.
    OFFICIAL[e.CRATE_DESTROYED] = float(self.cnn_crate_reward)
    _base_setup_training(self)


def _save(self) -> None:
    directory = Path(os.environ["MODEL_A_CNN_CHECKPOINT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"round-{self.stage_rounds:04d}.pt"
    payload = {
        "architecture": ARCHITECTURE, "online_net": self.online_net.state_dict(),
        "target_net": self.target_net.state_dict(), "optimizer": self.optimizer.state_dict(),
        "protocol_sha256": self.cnn_protocol_sha256, "stage": self.cnn_stage,
        "arm": self.cnn_arm, "replica": self.cnn_replica, "stage_rounds": self.stage_rounds,
        "completed_total_rounds": self.completed_rounds, "training_steps": self.training_steps,
        "gradient_steps": self.gradient_steps, "epsilon": self.epsilon,
        "parent_checkpoint": str(self.parent_checkpoint_path),
        "parent_sha256": self.parent_checkpoint_sha256,
        "n_step": self.n_step, "device_used": str(self.device),
        "collision_mask": "immediate_opponent_collision_then_known_hazard_survival",
        "crate_destroyed_reward": self.cnn_crate_reward,
        "training_diagnostics": copy.deepcopy(self.diagnostics),
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def end_of_round(self, last_game_state, last_action, events):
    identity = bool(self.n_step_queue) and (
        self.n_step_queue[-1].episode_id == int(last_game_state["round"])
        and self.n_step_queue[-1].step == int(last_game_state["step"])
        and self.n_step_queue[-1].action == ACTIONS.index(last_action)
    )
    if identity:
        _merge_survivor(self, last_game_state, last_action, events)
    else:
        _advance_bomb_state(self, last_action, events)
        raw = _raw(self, last_game_state, last_action, None, events, True)
        if raw is None:
            raise RuntimeError("terminal transition missing")
        self.diagnostics["dead_terminal_appends"] += 1
        _append(self, raw)
    if self.n_step_queue:
        raise RuntimeError("n-step queue crossed episode boundary")
    self._active_bomb = False
    self.stage_rounds += 1
    self.completed_rounds += 1
    self.diagnostics["per_round"].append(self._round)
    self._round = None
    if self.stage_rounds % self.checkpoint_every == 0:
        _save(self)


__all__ = ["end_of_round", "game_events_occurred", "setup_training"]
