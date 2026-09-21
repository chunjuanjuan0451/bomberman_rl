"""Exact final-v4 learner with immutable per-course output checkpoints."""

from __future__ import annotations

from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.features import GLOBAL_SIZE
from agent_code.model_a_dqn.network import torch


def setup_training(self):
    # This delegates optimizer, target, replay and epsilon semantics to final v4.
    v4.setup_training(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    v4.game_events_occurred(self, old_game_state, self_action, new_game_state, events)


def _save_checkpoint(self):
    torch.save(
        {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "completed_rounds": self.completed_rounds,
            "stage_start_rounds": self.stage_start_rounds,
            "stage_completed_rounds": self.completed_rounds - self.stage_start_rounds,
            "training_steps": self.training_steps,
            "gradient_steps": self.gradient_steps,
            "epsilon": self.epsilon,
            "global_feature_size": GLOBAL_SIZE,
            "architecture": MODEL_ARCHITECTURE,
            "agent_seed": self.agent_seed,
            "protocol_sha256": self.protocol_sha256,
            "arm": self.arm,
            "replica": self.replica,
            "stage_id": self.stage_id,
            "parent_sha256": self.parent_sha256,
        },
        self.checkpoint_path,
    )


def end_of_round(self, last_game_state, last_action, events):
    v4._record(self, last_game_state, last_action, None, v4.reward_from_events(events), done=True)
    self.completed_rounds += 1
    if self.completed_rounds % v4.CHECKPOINT_EVERY_ROUNDS == 0:
        _save_checkpoint(self)
        self.logger.info(
            "Saved clean-v4 %s/%s/%s after cumulative round %d.",
            self.arm, self.replica, self.stage_id, self.completed_rounds,
        )
