"""Unchanged exact-v4 learner with immutable 100-round Task4 snapshots."""

from __future__ import annotations

import shutil

from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.features import GLOBAL_SIZE
from agent_code.model_a_dqn.network import torch

from .config import snapshot_path


def setup_training(self):
    v4.setup_training(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    v4.game_events_occurred(self, old_game_state, self_action, new_game_state, events)


def _payload(self) -> dict:
    return {
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
        "replica": self.replica,
        "stage_id": "task4_duel",
        "parent_sha256": self.parent_sha256,
        "training_variable": "opponent_sampling_distribution_only",
    }


def _atomic_save(payload: dict, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def end_of_round(self, last_game_state, last_action, events):
    v4._record(self, last_game_state, last_action, None, v4.reward_from_events(events), done=True)
    self.completed_rounds += 1
    stage_round = self.completed_rounds - self.stage_start_rounds
    if stage_round in self.protocol["snapshot_rounds"]:
        snapshot = snapshot_path(self.protocol, self.replica, stage_round)
        if snapshot.exists():
            raise RuntimeError(f"refusing to overwrite Task4 snapshot: {snapshot}")
        _atomic_save(_payload(self), snapshot)
        _atomic_copy(snapshot, self.checkpoint_path)
        self.logger.info(
            "Saved Task4 duel %s snapshot at stage round %d.", self.replica, stage_round,
        )
