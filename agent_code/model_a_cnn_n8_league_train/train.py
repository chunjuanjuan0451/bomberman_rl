"""n=8 Double-DQN training with collision-consistent target masks."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import events as e

from agent_code.model_a_cnn_n8.features import ACTIONS, state_to_features
from agent_code.model_a_cnn_n8.rewards import reward_total
from agent_code.model_a_cnn_n8.network import ARCHITECTURE, torch
from agent_code.model_a_cnn_n8.train import RawTransition, _append, _merge_survivor, setup_training

from .callbacks import effective_action_mask


def _advance_bomb_state(self, action: str, events) -> None:
    if action == "BOMB" and e.BOMB_DROPPED in events:
        self._active_bomb = True
    if e.BOMB_EXPLODED in events:
        self._active_bomb = False


def _raw(self, old, action, new, events, done) -> RawTransition | None:
    state = state_to_features(old)
    if state is None or action not in ACTIONS:
        return None
    reward, components = reward_total(old, None if done else new, action, events, self.cnn_stage)
    next_state = None if done else state_to_features(new)
    next_mask = None if done else effective_action_mask(new, self._active_bomb)
    return RawTransition(state, ACTIONS.index(action), reward, components, next_state, done, next_mask,
                         int(old["round"]), int(old["step"]), tuple(events))


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _advance_bomb_state(self, self_action, events)
    raw = _raw(self, old_game_state, self_action, new_game_state, events, False)
    if raw is not None:
        _append(self, raw)


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
