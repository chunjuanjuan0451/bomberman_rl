"""Six-step BOMB learner changing only KILLED_OPPONENT reward from 12 to 80."""

from __future__ import annotations

import copy

import events as e

from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.features import GLOBAL_SIZE
from agent_code.model_a_v4_bomb_credit6 import train as credit6

from .config import snapshot_path


RawTransition = credit6.RawTransition
aggregate_return = credit6.aggregate_return
_assert_outcome_attribution = credit6._assert_outcome_attribution
_mature_bomb = credit6._mature_bomb
_record = credit6._record


def kill_reward(self) -> float:
    return 12.0 if self.arm == "reward12" else 80.0


def reward_from_events(self, events: list[str]) -> float:
    return v4.reward_from_events(events) + (kill_reward(self) - v4.REWARD_BY_EVENT[e.KILLED_OPPONENT]) * events.count(e.KILLED_OPPONENT)


def setup_training(self):
    # Neither arm is named "control", so the already-audited implementation
    # gives both arms the identical six-step BOMB target.
    credit6.setup_training(self)
    if self.bomb_return_steps != 6:
        raise RuntimeError("BOMB-reward80 requires six-step BOMB targets in both arms")
    self.logger.info("BOMB-reward80 %s/%s: kill_reward=%.1f, BOMB return=6.", self.arm, self.replica, kill_reward(self))


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = reward_from_events(self, events) + v4._state_shaping(old_game_state, new_game_state)
    credit6._record(self, old_game_state, self_action, new_game_state, reward, False, events)


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
        "arm": self.arm,
        "replica": self.replica,
        "stage_id": "task4c_bomb_reward80_gate",
        "parent_sha256": self.parent_sha256,
        "training_variable": "killed_opponent_reward_12_vs_80",
        "kill_reward": kill_reward(self),
        "reward_profile": copy.deepcopy(self.protocol["reward_profiles"][self.arm]),
        "bomb_return_steps": 6,
        "non_bomb_return_steps": 1,
        "bomb_maturation_steps": credit6.BOMB_MATURATION_STEPS,
        "allowed_outcome_lags": list(credit6.OUTCOME_LAGS),
        "terminal_event_delivery": "posthumous owned-bomb kills attach to the last terminal transition",
        "target_diagnostics": copy.deepcopy(self.target_diagnostics),
    }


def end_of_round(self, last_game_state, last_action, events):
    credit6._record(self, last_game_state, last_action, None, reward_from_events(self, events), True, events)
    if self._pending_bombs:
        raise RuntimeError("BOMB-reward80 pending queue crossed an episode boundary")
    self.completed_rounds += 1
    stage_round = self.completed_rounds - self.stage_start_rounds
    if stage_round in self.protocol["snapshot_rounds"]:
        snapshot = snapshot_path(self.protocol, self.arm, self.replica, stage_round)
        if snapshot.exists():
            raise RuntimeError(f"refusing to overwrite BOMB-reward80 snapshot: {snapshot}")
        credit6._atomic_save(_payload(self), snapshot)
        credit6._atomic_copy(snapshot, self.checkpoint_path)
        self.logger.info("Saved BOMB-reward80 %s/%s round %d.", self.arm, self.replica, stage_round)
