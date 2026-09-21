"""Exact-v4 learner changing only BOMB return horizon from one to six."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import shutil

import numpy as np
import events as e

from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.features import ACTIONS, GLOBAL_SIZE, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import torch

from .config import snapshot_path
from .replay import ReplayBuffer, Transition


BOMB_INDEX = ACTIONS.index("BOMB")
BOMB_MATURATION_STEPS = 6
OUTCOME_LAGS = (4, 5)


@dataclass
class RawTransition:
    state: object
    action: int
    reward: float
    next_state: object
    done: bool
    next_mask: object
    episode_id: int
    start_step: int
    events: tuple[str, ...]


def aggregate_return(window: list[RawTransition], return_steps: int, gamma: float) -> Transition:
    if not window or return_steps <= 0:
        raise ValueError("return aggregation requires a non-empty window and positive horizon")
    first = window[0]
    if any(item.episode_id != first.episode_id for item in window):
        raise ValueError("return aggregation may not cross episodes")
    selected = window[: int(return_steps)]
    reward = 0.0
    final = selected[0]
    used = 0
    for index, raw in enumerate(selected):
        reward += (float(gamma) ** index) * float(raw.reward)
        final = raw
        used = index + 1
        if raw.done:
            break
    return Transition(
        state=first.state,
        action=first.action,
        reward=reward,
        next_state=final.next_state,
        done=final.done,
        next_mask=final.next_mask,
        bootstrap_discount=float(gamma) ** used,
        return_steps=used,
        episode_id=first.episode_id,
        start_step=first.start_step,
    )


def setup_training(self):
    self.target_net = copy.deepcopy(self.online_net).to(self.device)
    self.target_net.eval()
    self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=v4.LEARNING_RATE)
    self.replay_buffer = ReplayBuffer()
    self.training_rng = np.random.default_rng(self.agent_seed)
    checkpoint = self.resume_checkpoint
    if checkpoint is None:
        raise RuntimeError("BOMB-credit6 experiment must resume immutable source-r2")
    self.target_net.load_state_dict(checkpoint["target_net"], strict=True)
    self.optimizer.load_state_dict(checkpoint["optimizer"])
    self.training_steps = int(checkpoint["training_steps"])
    self.gradient_steps = int(checkpoint.get("gradient_steps", 0))
    self.epsilon = float(checkpoint["epsilon"])
    if abs(self.epsilon - v4.EPSILON_FINAL) > 1e-12:
        raise RuntimeError("BOMB-credit6 parent must be at the exact-v4 epsilon floor")
    self.bomb_return_steps = 1 if self.arm == "control" else 6
    self._pending_bombs: list[list[RawTransition]] = []
    self.target_diagnostics = {
        "raw_transitions": 0,
        "non_bomb_one_step_targets": 0,
        "bomb_targets_matured": 0,
        "bomb_targets_full_six_step_observed": 0,
        "bomb_targets_terminal_truncated": 0,
        "bomb_target_return_steps_histogram": {str(i): 0 for i in range(1, 7)},
        "bomb_target_kill_events_included": 0,
        "bomb_target_self_events_included": 0,
        "observed_kill_events": 0,
        "observed_self_events": 0,
        "uniquely_attributed_kill_events": 0,
        "uniquely_attributed_self_events": 0,
        "outcome_lag_histogram": {"4": 0, "5": 0, "terminal_deferred": 0},
        "terminal_deferred_kill_events": 0,
        "attribution_errors": 0,
    }
    self.logger.info(
        "BOMB-credit6 %s/%s: BOMB return=%d, non-BOMB return=1, maturation=6, lags=4/5.",
        self.arm, self.replica, self.bomb_return_steps,
    )


def _batch_tensors(transitions: list[Transition], device):
    local = np.stack([item.state[0] for item in transitions])
    global_features = np.stack([item.state[1] for item in transitions])
    actions = np.asarray([item.action for item in transitions], dtype=np.int64)
    rewards = np.asarray([item.reward for item in transitions], dtype=np.float32)
    dones = np.asarray([item.done for item in transitions], dtype=np.float32)
    discounts = np.asarray([item.bootstrap_discount for item in transitions], dtype=np.float32)
    next_local = np.stack([
        item.next_state[0] if item.next_state is not None else np.zeros_like(item.state[0])
        for item in transitions
    ])
    next_global = np.stack([
        item.next_state[1] if item.next_state is not None else np.zeros_like(item.state[1])
        for item in transitions
    ])
    next_masks = np.stack([
        item.next_mask if item.next_state is not None else np.zeros(len(ACTIONS), dtype=bool)
        for item in transitions
    ])
    arrays = (local, global_features, actions, rewards, dones, discounts, next_local, next_global, next_masks)
    return tuple(torch.as_tensor(value, device=device) for value in arrays)


def _learn(self):
    if len(self.replay_buffer) < v4.WARMUP_TRANSITIONS or self.training_steps % v4.UPDATE_EVERY:
        return
    transitions = self.replay_buffer.sample(v4.BATCH_SIZE, self.training_rng)
    (local, global_features, actions, rewards, dones, discounts,
     next_local, next_global, next_masks) = _batch_tensors(transitions, self.device)
    q_values = self.online_net(local.float(), global_features.float()).gather(
        1, actions.long().unsqueeze(1),
    ).squeeze(1)
    with torch.no_grad():
        online_next = self.online_net(next_local.float(), next_global.float()).masked_fill(~next_masks.bool(), -1e9)
        next_actions = online_next.argmax(dim=1)
        target_next = self.target_net(next_local.float(), next_global.float()).gather(
            1, next_actions.unsqueeze(1),
        ).squeeze(1)
        targets = rewards.float() + discounts.float() * (1.0 - dones.float()) * target_next
    loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
    self.optimizer.step()
    self.gradient_steps += 1
    if self.gradient_steps % v4.TARGET_UPDATE_EVERY == 0:
        self.target_net.load_state_dict(self.online_net.state_dict())


def _raw(old_game_state, action: str, new_game_state, reward: float, done: bool, events) -> RawTransition | None:
    state = state_to_features(old_game_state)
    if state is None or action not in ACTIONS:
        return None
    next_state = None if done else state_to_features(new_game_state)
    next_mask = None if next_state is None else legal_action_mask(new_game_state)
    return RawTransition(
        state=state,
        action=ACTIONS.index(action),
        reward=float(reward),
        next_state=next_state,
        done=bool(done),
        next_mask=next_mask,
        episode_id=int(old_game_state["round"]),
        start_step=int(old_game_state["step"]),
        events=tuple(events),
    )


def _assert_outcome_attribution(self, raw: RawTransition) -> None:
    for event, observed_key, attributed_key in (
        (e.KILLED_OPPONENT, "observed_kill_events", "uniquely_attributed_kill_events"),
        (e.KILLED_SELF, "observed_self_events", "uniquely_attributed_self_events"),
    ):
        count = raw.events.count(event)
        self.target_diagnostics[observed_key] += count
        if not count:
            continue
        matches = []
        for window in self._pending_bombs:
            if window[0].episode_id != raw.episode_id:
                continue
            lag = raw.start_step - window[0].start_step
            if lag in OUTCOME_LAGS:
                matches.append((window, str(lag)))
        if not matches and raw.done and event == e.KILLED_OPPONENT:
            # The official environment stops per-step callbacks for dead agents.
            # A bomb they already own can still kill later; that event is then
            # delivered at end_of_round on their last terminal transition.
            deferred = [
                window for window in self._pending_bombs
                if window[0].episode_id == raw.episode_id
                and 0 <= raw.start_step - window[0].start_step <= OUTCOME_LAGS[-1]
            ]
            matches = [(window, "terminal_deferred") for window in deferred]
        if len(matches) != 1:
            self.target_diagnostics["attribution_errors"] += count
            raise RuntimeError(
                f"official {event} at {raw.episode_id}/{raw.start_step} did not uniquely match lag-4/5 BOMB"
            )
        self.target_diagnostics[attributed_key] += count
        category = matches[0][1]
        self.target_diagnostics["outcome_lag_histogram"][category] += count
        if category == "terminal_deferred":
            self.target_diagnostics["terminal_deferred_kill_events"] += count


def _mature_bomb(self, window: list[RawTransition]) -> None:
    selected_steps = 1 if self.arm == "control" else min(BOMB_MATURATION_STEPS, len(window))
    transition = aggregate_return(window, selected_steps, v4.GAMMA)
    self.replay_buffer.add(transition)
    diagnostics = self.target_diagnostics
    diagnostics["bomb_targets_matured"] += 1
    diagnostics["bomb_targets_full_six_step_observed"] += int(len(window) >= BOMB_MATURATION_STEPS)
    diagnostics["bomb_targets_terminal_truncated"] += int(len(window) < BOMB_MATURATION_STEPS)
    diagnostics["bomb_target_return_steps_histogram"][str(transition.return_steps)] += 1
    selected = window[:selected_steps]
    diagnostics["bomb_target_kill_events_included"] += sum(
        raw.events.count(e.KILLED_OPPONENT) for raw in selected
    )
    diagnostics["bomb_target_self_events_included"] += sum(
        raw.events.count(e.KILLED_SELF) for raw in selected
    )


def _record(self, old_game_state, action: str, new_game_state, reward: float, done: bool, events) -> None:
    raw = _raw(old_game_state, action, new_game_state, reward, done, events)
    if raw is None:
        return
    self.target_diagnostics["raw_transitions"] += 1
    for window in self._pending_bombs:
        window.append(raw)
    if raw.action == BOMB_INDEX:
        if e.BOMB_DROPPED not in raw.events:
            raise RuntimeError("legal BOMB action did not produce BOMB_DROPPED")
        self._pending_bombs.append([raw])
    _assert_outcome_attribution(self, raw)
    if raw.action != BOMB_INDEX:
        self.replay_buffer.add(aggregate_return([raw], 1, v4.GAMMA))
        self.target_diagnostics["non_bomb_one_step_targets"] += 1
    remaining = []
    for window in self._pending_bombs:
        if len(window) >= BOMB_MATURATION_STEPS or raw.done:
            _mature_bomb(self, window)
        else:
            remaining.append(window)
    self._pending_bombs = remaining
    self.training_steps += 1
    progress = min(1.0, self.training_steps / v4.EPSILON_DECAY_STEPS)
    self.epsilon = v4.EPSILON_START + progress * (v4.EPSILON_FINAL - v4.EPSILON_START)
    _learn(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = v4.reward_from_events(events) + v4._state_shaping(old_game_state, new_game_state)
    _record(self, old_game_state, self_action, new_game_state, reward, False, events)


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
        "stage_id": "task4c_bomb_credit6",
        "parent_sha256": self.parent_sha256,
        "training_variable": "bomb_transition_return_horizon_1_vs_6",
        "bomb_return_steps": self.bomb_return_steps,
        "non_bomb_return_steps": 1,
        "bomb_maturation_steps": BOMB_MATURATION_STEPS,
        "allowed_outcome_lags": list(OUTCOME_LAGS),
        "terminal_event_delivery": "posthumous owned-bomb kills attach to the last terminal transition",
        "target_diagnostics": copy.deepcopy(self.target_diagnostics),
    }


def _atomic_save(payload: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_copy(source, destination) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def end_of_round(self, last_game_state, last_action, events):
    _record(self, last_game_state, last_action, None, v4.reward_from_events(events), True, events)
    if self._pending_bombs:
        raise RuntimeError("BOMB-credit6 pending queue crossed an episode boundary")
    self.completed_rounds += 1
    stage_round = self.completed_rounds - self.stage_start_rounds
    if stage_round in self.protocol["snapshot_rounds"]:
        snapshot = snapshot_path(self.protocol, self.arm, self.replica, stage_round)
        if snapshot.exists():
            raise RuntimeError(f"refusing to overwrite BOMB-credit6 snapshot: {snapshot}")
        _atomic_save(_payload(self), snapshot)
        _atomic_copy(snapshot, self.checkpoint_path)
        self.logger.info("Saved BOMB-credit6 %s/%s round %d.", self.arm, self.replica, stage_round)
