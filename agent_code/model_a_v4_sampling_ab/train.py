"""All-action n=8 learner with replay sampling as the sole A/B variable."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json

import numpy as np
import events as e

from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.features import ACTIONS, GLOBAL_SIZE, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import torch

from .callbacks import _nearest_distance
from .config import ROOT, checkpoint_path, evaluation_diagnostic_path
from .replay import ReplayBuffer, Transition


RETURN_HORIZON = 8
KILL_SLOTS = 2
SELF_SLOTS = 2


@dataclass
class RawTransition:
    state: object
    action: int
    reward: float
    state_shaping: float
    next_state: object
    done: bool
    next_mask: object
    episode_id: int
    start_step: int
    events: tuple[str, ...]


def aggregate_return(window: list[RawTransition], gamma: float = v4.GAMMA) -> Transition:
    if not window or len(window) > RETURN_HORIZON:
        raise ValueError("sampling-distribution return window must contain one to eight transitions")
    first = window[0]
    if any(raw.episode_id != first.episode_id for raw in window):
        raise ValueError("sampling-distribution return crossed an episode")
    reward = 0.0
    final = first
    used = 0
    selected = []
    for index, raw in enumerate(window):
        reward += (float(gamma) ** index) * float(raw.reward)
        final = raw
        selected.append(raw)
        used = index + 1
        if raw.done:
            break
    return Transition(
        state=first.state, action=first.action, reward=reward,
        next_state=final.next_state, done=final.done, next_mask=final.next_mask,
        bootstrap_discount=float(gamma) ** used, return_steps=used,
        episode_id=first.episode_id, start_step=first.start_step,
        kill_chain=(
            any(e.KILLED_OPPONENT in raw.events for raw in selected)
            and not any(e.KILLED_SELF in raw.events for raw in selected)
        ),
        self_chain=any(e.KILLED_SELF in raw.events for raw in selected),
    )


def setup_training(self):
    if self.run_mode == "evaluate":
        _setup_evaluation(self)
        return
    self.target_net = copy.deepcopy(self.online_net).to(self.device)
    self.target_net.eval()
    self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=v4.LEARNING_RATE)
    checkpoint = self.resume_checkpoint
    if checkpoint is None:
        raise RuntimeError("sampling-distribution A/B must resume immutable source-r2")
    self.target_net.load_state_dict(checkpoint["target_net"], strict=True)
    self.optimizer.load_state_dict(checkpoint["optimizer"])
    self.training_steps = int(checkpoint["training_steps"])
    self.gradient_steps = int(checkpoint.get("gradient_steps", 0))
    self.epsilon = float(checkpoint["epsilon"])
    if abs(self.epsilon - v4.EPSILON_FINAL) > 1e-12:
        raise RuntimeError("sampling-distribution parent must be at epsilon floor")
    self.replay_buffer = ReplayBuffer(capacity=50_000)
    self.training_rng = np.random.default_rng(self.agent_seed)
    self._episode_raw: list[RawTransition] = []
    self.training_diagnostics = {
        "raw_transitions": 0, "matured_targets": 0,
        "return_steps_histogram": {str(i): 0 for i in range(1, RETURN_HORIZON + 1)},
        "kill_chain_targets_created": 0, "self_chain_targets_created": 0,
        "observed_kill_events": 0, "observed_self_events": 0,
        "survivor_terminal_merges": 0, "dead_terminal_appends": 0,
        "gradient_updates": 0, "sampled_batches": 0,
        "requested_kill_slots": 0, "fulfilled_kill_slots": 0,
        "requested_self_slots": 0, "fulfilled_self_slots": 0,
        "fallback_uniform_slots": 0, "realized_kill_chain_samples": 0,
        "realized_self_chain_samples": 0,
    }


def _batch_tensors(transitions: list[Transition], device):
    local = np.stack([item.state[0] for item in transitions])
    global_features = np.stack([item.state[1] for item in transitions])
    actions = np.asarray([item.action for item in transitions], dtype=np.int64)
    rewards = np.asarray([item.reward for item in transitions], dtype=np.float32)
    dones = np.asarray([item.done for item in transitions], dtype=np.float32)
    discounts = np.asarray([item.bootstrap_discount for item in transitions], dtype=np.float32)
    next_local = np.stack([item.next_state[0] if item.next_state is not None else np.zeros_like(item.state[0]) for item in transitions])
    next_global = np.stack([item.next_state[1] if item.next_state is not None else np.zeros_like(item.state[1]) for item in transitions])
    next_masks = np.stack([item.next_mask if item.next_state is not None else np.zeros(len(ACTIONS), dtype=bool) for item in transitions])
    return tuple(torch.as_tensor(value, device=device) for value in (
        local, global_features, actions, rewards, dones, discounts, next_local, next_global, next_masks,
    ))


def _learn(self) -> None:
    if len(self.replay_buffer) < v4.WARMUP_TRANSITIONS or self.training_steps % v4.UPDATE_EVERY:
        return
    if self.arm == "uniform_n8":
        transitions, sampling = self.replay_buffer.sample_uniform(v4.BATCH_SIZE, self.training_rng)
    else:
        transitions, sampling = self.replay_buffer.sample_stratified(
            v4.BATCH_SIZE, KILL_SLOTS, SELF_SLOTS, self.training_rng,
        )
    for key, value in sampling.items():
        self.training_diagnostics[key] += int(value)
    self.training_diagnostics["sampled_batches"] += 1
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
    self.training_diagnostics["gradient_updates"] += 1
    if self.gradient_steps % v4.TARGET_UPDATE_EVERY == 0:
        self.target_net.load_state_dict(self.online_net.state_dict())


def _raw(old_game_state, action: str, new_game_state, events, done: bool) -> RawTransition | None:
    state = state_to_features(old_game_state)
    if state is None or action not in ACTIONS:
        return None
    shaping = 0.0 if done else float(v4._state_shaping(old_game_state, new_game_state))
    next_state = None if done else state_to_features(new_game_state)
    next_mask = None if next_state is None else legal_action_mask(new_game_state)
    return RawTransition(
        state=state, action=ACTIONS.index(action),
        reward=float(v4.reward_from_events(events)) + shaping,
        state_shaping=shaping, next_state=next_state, done=done, next_mask=next_mask,
        episode_id=int(old_game_state["round"]), start_step=int(old_game_state["step"]),
        events=tuple(events),
    )


def _add_target(self, window: list[RawTransition]) -> None:
    transition = aggregate_return(window)
    self.replay_buffer.add(transition)
    d = self.training_diagnostics
    d["matured_targets"] += 1
    d["return_steps_histogram"][str(transition.return_steps)] += 1
    d["kill_chain_targets_created"] += int(transition.kill_chain)
    d["self_chain_targets_created"] += int(transition.self_chain)


def _append_training_raw(self, raw: RawTransition) -> None:
    self._episode_raw.append(raw)
    d = self.training_diagnostics
    d["raw_transitions"] += 1
    d["observed_kill_events"] += raw.events.count(e.KILLED_OPPONENT)
    d["observed_self_events"] += raw.events.count(e.KILLED_SELF)
    if not raw.done and len(self._episode_raw) >= RETURN_HORIZON:
        _add_target(self, self._episode_raw[:RETURN_HORIZON])
        self._episode_raw.pop(0)
    if raw.done:
        while self._episode_raw:
            _add_target(self, self._episode_raw[:RETURN_HORIZON])
            self._episode_raw.pop(0)
    self.training_steps += 1
    progress = min(1.0, self.training_steps / v4.EPSILON_DECAY_STEPS)
    self.epsilon = v4.EPSILON_START + progress * (v4.EPSILON_FINAL - v4.EPSILON_START)
    _learn(self)


def _merge_training_survivor(self, last_game_state, last_action: str, events) -> None:
    if not self._episode_raw:
        raise RuntimeError("sampling-distribution survivor has no pending raw transition")
    raw = self._episode_raw[-1]
    expected = (int(last_game_state["round"]), int(last_game_state["step"]), ACTIONS.index(last_action))
    observed = (raw.episode_id, raw.start_step, raw.action)
    if expected != observed:
        raise RuntimeError(f"sampling-distribution survivor identity mismatch: {expected} != {observed}")
    final_events = tuple(events)
    old_counts = {name: raw.events.count(name) for name in set(raw.events) | set(final_events)}
    new_counts = {name: final_events.count(name) for name in old_counts}
    delta = {
        name: new_counts[name] - old_counts[name]
        for name in old_counts if new_counts[name] != old_counts[name]
    }
    if delta != {e.SURVIVED_ROUND: 1}:
        raise RuntimeError(f"sampling-distribution survivor event delta changed: {delta}")
    raw.events = final_events
    raw.reward = float(v4.reward_from_events(final_events)) + raw.state_shaping
    raw.done = True; raw.next_state = None; raw.next_mask = None
    self.training_diagnostics["survivor_terminal_merges"] += 1
    while self._episode_raw:
        _add_target(self, self._episode_raw[:RETURN_HORIZON])
        self._episode_raw.pop(0)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if self.run_mode == "evaluate":
        _record_evaluation(self, old_game_state, self_action, new_game_state, events)
        return
    raw = _raw(old_game_state, self_action, new_game_state, events, False)
    if raw is not None:
        _append_training_raw(self, raw)


def _checkpoint_payload(self) -> dict:
    diagnostics = copy.deepcopy(self.training_diagnostics)
    diagnostics["replay_size"] = len(self.replay_buffer)
    diagnostics["replay_kill_items"] = self.replay_buffer.kill_items
    diagnostics["replay_self_items"] = self.replay_buffer.self_items
    return {
        "online_net": self.online_net.state_dict(), "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(), "completed_rounds": self.completed_rounds,
        "stage_start_rounds": self.stage_start_rounds,
        "stage_completed_rounds": self.completed_rounds - self.stage_start_rounds,
        "training_steps": self.training_steps, "gradient_steps": self.gradient_steps,
        "epsilon": self.epsilon, "global_feature_size": GLOBAL_SIZE,
        "architecture": MODEL_ARCHITECTURE, "agent_seed": self.agent_seed,
        "protocol_sha256": self.protocol_sha256, "arm": self.arm, "replica": self.replica,
        "stage_id": "task4c_sampling_ab", "parent_sha256": self.parent_sha256,
        "single_training_variable": "replay_sampling_uniform_vs_reserved_2_kill_2_self",
        "all_action_return_horizon": RETURN_HORIZON, "sampling_mode": self.arm,
        "reserved_slots": {"kill_chain": 0 if self.arm == "uniform_n8" else KILL_SLOTS,
                           "self_kill_chain": 0 if self.arm == "uniform_n8" else SELF_SLOTS},
        "training_diagnostics": diagnostics,
    }


def _atomic_json(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_checkpoint(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def end_of_round(self, last_game_state, last_action, events):
    if self.run_mode == "evaluate":
        _end_evaluation_round(self, last_game_state, last_action, events)
        return
    if self._episode_raw and (
        self._episode_raw[-1].episode_id == int(last_game_state["round"])
        and self._episode_raw[-1].start_step == int(last_game_state["step"])
        and self._episode_raw[-1].action == ACTIONS.index(last_action)
    ):
        _merge_training_survivor(self, last_game_state, last_action, events)
    else:
        raw = _raw(last_game_state, last_action, None, events, True)
        if raw is None:
            raise RuntimeError("sampling-distribution dead terminal transition missing")
        self.training_diagnostics["dead_terminal_appends"] += 1
        _append_training_raw(self, raw)
    if self._episode_raw:
        raise RuntimeError("sampling-distribution n=8 queue crossed an episode")
    self.completed_rounds += 1
    if self.completed_rounds - self.stage_start_rounds == 100:
        if self.checkpoint_path.exists():
            raise RuntimeError("refusing to overwrite sampling-distribution endpoint")
        _atomic_checkpoint(self.checkpoint_path, _checkpoint_payload(self))
    elif self.completed_rounds - self.stage_start_rounds > 100:
        raise RuntimeError("sampling-distribution training exceeded fixed endpoint")


# Passive evaluation -------------------------------------------------------

def _setup_evaluation(self) -> None:
    self.eval_oracle_deadline = float(self.protocol["evaluation"]["oracle_deadline_seconds"])
    self.eval_world_seed = int(__import__("os").environ["MODEL_A_SAMPLING_EVAL_WORLD_SEED"])
    expected_seed = None
    for case in self.protocol["evaluation"]["strata"][self.eval_stratum]["cases"]:
        if int(case["world_seed"]) == self.eval_world_seed:
            expected_seed = int(case["agent_seed"])
    if expected_seed is None or self.agent_seed != expected_seed:
        raise RuntimeError("sampling-distribution evaluation seed mismatch")
    self.eval_output = evaluation_diagnostic_path(
        self.protocol, self.eval_stratum, self.eval_label, self.eval_world_seed,
    )
    supplied = __import__("pathlib").Path(
        __import__("os").environ["MODEL_A_SAMPLING_EVAL_DIAGNOSTIC_PATH"],
    ).resolve()
    if supplied != self.eval_output or supplied.exists():
        raise RuntimeError("sampling-distribution evaluation diagnostic path mismatch or overwrite")
    self.eval_expected_rounds = int(self.protocol["evaluation"]["strata"][self.eval_stratum]["rounds_per_case"])
    self._pending_eval_decision = None
    self._eval_history = []
    self._eval_successful_threat_bombs = set()
    self._eval_last_events: tuple[str, ...] | None = None
    self.eval_diagnostics = {
        "rounds": 0, "rows": 0, "approach_steps": 0, "bomb_legal": 0,
        "bombs": 0, "threat_bombs": 0, "trap_bombs": 0,
        "kill_events": 0, "self_kill_events": 0, "got_killed_events": 0, "survived_rounds": 0,
        "linked_kill_events": 0, "linked_threat_kill_events": 0,
        "unlinked_or_ambiguous_kill_events": 0, "oracle_evaluated": 0, "oracle_timeouts": 0,
    }


def _record_evaluation(self, old_game_state, action: str, new_game_state, events) -> None:
    pending = self._pending_eval_decision
    if pending is None:
        raise RuntimeError("sampling-distribution evaluation decision missing")
    observed = (int(old_game_state["round"]), int(old_game_state["step"]), ACTIONS.index(action))
    expected = (pending["round"], pending["step"], pending["action"])
    if observed != expected:
        raise RuntimeError(f"sampling-distribution evaluation transition mismatch: {observed} != {expected}")
    d = self.eval_diagnostics
    bomb = action == "BOMB"
    d["rows"] += 1
    d["approach_steps"] += int(
        pending["nearest"] >= 0 and _nearest_distance(new_game_state) >= 0
        and _nearest_distance(new_game_state) < pending["nearest"]
    )
    d["bomb_legal"] += int(pending["bomb_legal"])
    d["oracle_evaluated"] += int(pending["oracle_evaluated"])
    d["oracle_timeouts"] += int(pending["oracle_timeout"])
    d["bombs"] += int(bomb)
    d["threat_bombs"] += int(bomb and pending["affected_opponents"] >= 1)
    d["trap_bombs"] += int(bomb and pending["max_space_reduction"] >= 0.5)
    record = {**pending, "bomb": bomb}
    self._eval_history.append(record)
    kills = tuple(events).count(e.KILLED_OPPONENT)
    d["kill_events"] += kills
    d["self_kill_events"] += tuple(events).count(e.KILLED_SELF)
    d["got_killed_events"] += tuple(events).count(e.GOT_KILLED)
    if kills:
        matches = [
            item for item in self._eval_history
            if item["round"] == pending["round"] and item["bomb"]
            and pending["step"] - item["step"] in (4, 5)
        ]
        if not matches and new_game_state is None:
            matches = [
                item for item in self._eval_history
                if item["round"] == pending["round"] and item["bomb"]
                and 0 <= pending["step"] - item["step"] <= 5
            ]
        if len(matches) == 1:
            d["linked_kill_events"] += kills
            if matches[0]["affected_opponents"] >= 1:
                d["linked_threat_kill_events"] += kills
                self._eval_successful_threat_bombs.add((matches[0]["round"], matches[0]["step"]))
        else:
            d["unlinked_or_ambiguous_kill_events"] += kills
    self._eval_last_events = tuple(events)
    self._pending_eval_decision = None


def _end_evaluation_round(self, last_game_state, last_action, events) -> None:
    if self._pending_eval_decision is not None:
        _record_evaluation(self, last_game_state, last_action, None, events)
    else:
        final = tuple(events)
        if self._eval_last_events is None:
            raise RuntimeError("sampling-distribution survivor evaluation lacks final events")
        names = set(final) | set(self._eval_last_events)
        delta = {
            name: final.count(name) - self._eval_last_events.count(name)
            for name in names if final.count(name) != self._eval_last_events.count(name)
        }
        if delta != {e.SURVIVED_ROUND: 1}:
            raise RuntimeError(f"sampling-distribution evaluation survivor delta changed: {delta}")
        self.eval_diagnostics["survived_rounds"] += 1
    self.eval_diagnostics["rounds"] += 1
    self._eval_last_events = None
    if self.eval_diagnostics["rounds"] == self.eval_expected_rounds:
        d = dict(self.eval_diagnostics)
        d["successful_threat_bombs"] = len(self._eval_successful_threat_bombs)
        d["threat_to_kill_conversion"] = (
            d["successful_threat_bombs"] / d["threat_bombs"] if d["threat_bombs"] else 0.0
        )
        d["linked_kills_per_threat"] = (
            d["linked_threat_kill_events"] / d["threat_bombs"] if d["threat_bombs"] else 0.0
        )
        _atomic_json(self.eval_output, {
            "schema_version": 1, "kind": "model-a-v4-sampling-ab-evaluation-diagnostic",
            "protocol_sha256": self.protocol_sha256, "label": self.eval_label,
            "stratum": self.eval_stratum, "world_seed": self.eval_world_seed,
            "agent_seed": self.agent_seed, "policy_updates": 0, "actions_overridden": 0,
            "diagnostics": d,
        })
    elif self.eval_diagnostics["rounds"] > self.eval_expected_rounds:
        raise RuntimeError("sampling-distribution evaluation exceeded round budget")
