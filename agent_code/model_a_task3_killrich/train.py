"""Uniform-replay frozen-D training with distribution metadata only."""

from __future__ import annotations

from agent_code.model_a_task3.train import (
    RawTransition, _record, _reward, _shaping, aggregate_n_step,
    setup_training,
)

from .callbacks import CHECKPOINT_PATH
from .config import architecture_name
from .network import torch


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _record(
        self, old_game_state, self_action, new_game_state,
        _reward(self, events) + _shaping(old_game_state, new_game_state), False,
    )


def _save(self):
    payload = {
        "online_net": self.online_net.state_dict(),
        "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "completed_rounds": self.completed_rounds,
        "training_steps": self.training_steps,
        "gradient_steps": self.gradient_steps,
        "epsilon": self.epsilon,
        "architecture": architecture_name(self.task3_config),
        "config_sha256": self.task3_config_sha256,
        "parent_sha256": self.task3_config["parent_sha256"],
        "arm": "D",
        "reward_profile": "official-aligned",
        "action_features": True,
        "n_step": 3,
        "sampling_profile": "uniform",
        "training_distribution": self.task3_config["training_distribution"],
        "scenario": self.task3_config["scenario"],
    }
    temporary = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(CHECKPOINT_PATH)


def end_of_round(self, last_game_state, last_action, events):
    _record(self, last_game_state, last_action, None, _reward(self, events), True)
    if self.n_step_queue:
        raise RuntimeError("n-step queue did not flush at episode end")
    self.completed_rounds += 1
    if self.completed_rounds % self.checkpoint_every_rounds == 0:
        _save(self)


__all__ = [
    "RawTransition", "aggregate_n_step", "setup_training",
    "game_events_occurred", "end_of_round",
]
