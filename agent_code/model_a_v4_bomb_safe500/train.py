"""Six-step learner rewarding safe owned-BOMB kills separately from self-kills."""

from __future__ import annotations

import copy
from dataclasses import replace

import events as e
from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.features import ACTIONS, GLOBAL_SIZE
from agent_code.model_a_v4_bomb_credit6 import train as credit6
from .config import snapshot_path


RawTransition = credit6.RawTransition
aggregate_return = credit6.aggregate_return


def safe_kill_reward(self) -> float:
    return 12.0 if self.arm == "safe12" else 500.0


def setup_training(self):
    credit6.setup_training(self)
    if self.bomb_return_steps != 6:
        raise RuntimeError("BOMB-safe500 requires six-step BOMB targets in both arms")
    self.target_diagnostics.update({
        "bomb_target_safe_kill_events": 0,
        "bomb_target_trade_kill_events": 0,
        "bomb_target_self_without_kill_events": 0,
        "bomb_target_suppressed_trade_reward_events": 0,
    })
    self.logger.info("BOMB-safe500 %s/%s: safe kill reward=%.1f, trade kill reward=0.", self.arm, self.replica, safe_kill_reward(self))


def _mature_bomb(self, window: list[RawTransition]) -> None:
    selected = window[:min(credit6.BOMB_MATURATION_STEPS, len(window))]
    transition = aggregate_return(selected, len(selected), v4.GAMMA)
    kill_count = sum(raw.events.count(e.KILLED_OPPONENT) for raw in selected)
    self_count = sum(raw.events.count(e.KILLED_SELF) for raw in selected)
    adjusted = float(transition.reward)
    for index, raw in enumerate(selected):
        count = raw.events.count(e.KILLED_OPPONENT)
        if count:
            # Remove exact-v4 +12 first. A trade gets no kill reward; otherwise
            # insert the arm's safe-kill reward at the same discounted step.
            adjusted -= (v4.GAMMA ** index) * v4.REWARD_BY_EVENT[e.KILLED_OPPONENT] * count
            if not self_count:
                adjusted += (v4.GAMMA ** index) * safe_kill_reward(self) * count
    self.replay_buffer.add(replace(transition, reward=adjusted))
    diagnostics = self.target_diagnostics
    diagnostics["bomb_targets_matured"] += 1
    diagnostics["bomb_targets_full_six_step_observed"] += int(len(window) >= credit6.BOMB_MATURATION_STEPS)
    diagnostics["bomb_targets_terminal_truncated"] += int(len(window) < credit6.BOMB_MATURATION_STEPS)
    diagnostics["bomb_target_return_steps_histogram"][str(transition.return_steps)] += 1
    diagnostics["bomb_target_kill_events_included"] += kill_count
    diagnostics["bomb_target_self_events_included"] += self_count
    if kill_count and self_count:
        diagnostics["bomb_target_trade_kill_events"] += kill_count
        diagnostics["bomb_target_suppressed_trade_reward_events"] += kill_count
    elif kill_count:
        diagnostics["bomb_target_safe_kill_events"] += kill_count
    elif self_count:
        diagnostics["bomb_target_self_without_kill_events"] += self_count


def _record(self, old_game_state, action: str, new_game_state, reward: float, done: bool, events) -> None:
    raw = credit6._raw(old_game_state, action, new_game_state, reward, done, events)
    if raw is None:
        return
    self.target_diagnostics["raw_transitions"] += 1
    for window in self._pending_bombs:
        window.append(raw)
    if raw.action == credit6.BOMB_INDEX:
        if e.BOMB_DROPPED not in raw.events:
            raise RuntimeError("legal BOMB action did not produce BOMB_DROPPED")
        self._pending_bombs.append([raw])
    credit6._assert_outcome_attribution(self, raw)
    if raw.action != credit6.BOMB_INDEX:
        self.replay_buffer.add(aggregate_return([raw], 1, v4.GAMMA))
        self.target_diagnostics["non_bomb_one_step_targets"] += 1
    remaining = []
    for window in self._pending_bombs:
        if len(window) >= credit6.BOMB_MATURATION_STEPS or raw.done:
            _mature_bomb(self, window)
        else:
            remaining.append(window)
    self._pending_bombs = remaining
    self.training_steps += 1
    progress = min(1.0, self.training_steps / v4.EPSILON_DECAY_STEPS)
    self.epsilon = v4.EPSILON_START + progress * (v4.EPSILON_FINAL - v4.EPSILON_START)
    credit6._learn(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = v4.reward_from_events(events) + v4._state_shaping(old_game_state, new_game_state)
    _record(self, old_game_state, self_action, new_game_state, reward, False, events)


def _payload(self) -> dict:
    return {
        "online_net": self.online_net.state_dict(), "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(), "completed_rounds": self.completed_rounds,
        "stage_start_rounds": self.stage_start_rounds,
        "stage_completed_rounds": self.completed_rounds - self.stage_start_rounds,
        "training_steps": self.training_steps, "gradient_steps": self.gradient_steps,
        "epsilon": self.epsilon, "global_feature_size": GLOBAL_SIZE, "architecture": MODEL_ARCHITECTURE,
        "agent_seed": self.agent_seed, "protocol_sha256": self.protocol_sha256,
        "arm": self.arm, "replica": self.replica, "stage_id": "task4c_bomb_safe500_gate",
        "parent_sha256": self.parent_sha256, "training_variable": "safe_bomb_kill_reward_12_vs_500",
        "safe_kill_reward": safe_kill_reward(self), "trade_kill_reward": 0.0,
        "reward_profile": copy.deepcopy(self.protocol["reward_profiles"][self.arm]),
        "bomb_return_steps": 6, "non_bomb_return_steps": 1,
        "bomb_maturation_steps": credit6.BOMB_MATURATION_STEPS,
        "allowed_outcome_lags": list(credit6.OUTCOME_LAGS),
        "terminal_event_delivery": "posthumous owned-bomb kills attach to the last terminal transition",
        "target_diagnostics": copy.deepcopy(self.target_diagnostics),
    }


def end_of_round(self, last_game_state, last_action, events):
    _record(self, last_game_state, last_action, None, v4.reward_from_events(events), True, events)
    if self._pending_bombs:
        raise RuntimeError("BOMB-safe500 pending queue crossed an episode boundary")
    self.completed_rounds += 1
    stage_round = self.completed_rounds - self.stage_start_rounds
    if stage_round in self.protocol["snapshot_rounds"]:
        snapshot = snapshot_path(self.protocol, self.arm, self.replica, stage_round)
        if snapshot.exists():
            raise RuntimeError(f"refusing to overwrite BOMB-safe500 snapshot: {snapshot}")
        credit6._atomic_save(_payload(self), snapshot)
        credit6._atomic_copy(snapshot, self.checkpoint_path)
        self.logger.info("Saved BOMB-safe500 %s/%s round %d.", self.arm, self.replica, stage_round)
