"""Exact-v4 training callbacks with immutable 50-round Task3 snapshots."""

from __future__ import annotations

from pathlib import Path

from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.features import GLOBAL_SIZE
from agent_code.model_a_dqn.network import torch

from .config import snapshot_path


def setup_training(self):
    v4.setup_training(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    v4.game_events_occurred(self, old_game_state, self_action, new_game_state, events)


def _checkpoint_payload(self) -> dict:
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
        "branch": self.branch,
        "stage_id": self.stage_id,
        "parent_sha256": self.parent_sha256,
        "selected_by_inner_validation": False,
    }


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def end_of_round(self, last_game_state, last_action, events):
    v4._record(self, last_game_state, last_action, None, v4.reward_from_events(events), done=True)
    self.completed_rounds += 1
    stage_round = self.completed_rounds - self.stage_start_rounds
    if stage_round % int(self.protocol["snapshot_every_rounds"]) == 0:
        payload = _checkpoint_payload(self)
        snapshot = snapshot_path(self.protocol, self.branch, self.stage_id, stage_round)
        if snapshot.exists():
            raise RuntimeError(f"refusing to overwrite Task3 snapshot: {snapshot}")
        _atomic_torch_save(payload, snapshot)
        _atomic_torch_save(payload, self.checkpoint_path)
        self.logger.info(
            "Saved Task3 retention %s/%s snapshot at stage round %d.",
            self.branch, self.stage_id, stage_round,
        )
