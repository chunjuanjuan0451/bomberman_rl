"""All-action n=8 learner and frozen-source rehearsal collector."""

from __future__ import annotations

import copy
import json

import numpy as np
import events as e

from agent_code.model_a_dqn import train as v4
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE
from agent_code.model_a_dqn.features import ACTIONS, GLOBAL_SIZE, legal_action_mask, state_to_features
from agent_code.model_a_dqn.network import torch
from agent_code.model_a_v4_sampling_ab.train import RawTransition, aggregate_return

from .callbacks import _nearest_distance
from .config import OLD_STRATA, ROOT, dataset_path, evaluation_diagnostic_path, sha256_file
from .replay import ReplayBuffer, Transition, load_dataset, pack_dataset, sample_fixed


RETURN_HORIZON = 8
KILL_SLOTS = 2
SELF_SLOTS = 2
REHEARSAL_PER_STRATUM = 4


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


def _base_queue(self) -> None:
    self._episode_raw: list[RawTransition] = []
    self._round_identity = None
    self._round_start_raw = 0
    self._round_start_updates = 0
    self._round_opponent_states = {str(i): 0 for i in range(4)}


def setup_training(self):
    if self.run_mode == "evaluate":
        _setup_evaluation(self)
        return
    _base_queue(self)
    if self.run_mode == "collect":
        self.dataset_transitions: list[Transition] = []
        self.collection_rounds = 0
        self.training_steps = 0
        self.gradient_steps = 0
        self.training_diagnostics = {
            "raw_transitions": 0, "matured_targets": 0,
            "return_steps_histogram": {str(i): 0 for i in range(1, 9)},
            "kill_chain_targets_created": 0, "self_chain_targets_created": 0,
            "observed_kill_events": 0, "observed_self_events": 0,
            "survivor_terminal_merges": 0, "dead_terminal_appends": 0,
            "gradient_updates": 0, "sampled_batches": 0, "per_round": [],
        }
        return
    self.target_net = copy.deepcopy(self.online_net).to(self.device)
    self.target_net.eval()
    self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=v4.LEARNING_RATE)
    checkpoint = self.resume_checkpoint
    self.target_net.load_state_dict(checkpoint["target_net"], strict=True)
    self.optimizer.load_state_dict(checkpoint["optimizer"])
    self.training_steps = int(checkpoint["training_steps"])
    self.gradient_steps = int(checkpoint.get("gradient_steps", 0))
    self.epsilon = float(checkpoint["epsilon"])
    if abs(self.epsilon - v4.EPSILON_FINAL) > 1e-12:
        raise RuntimeError("rehearsal parent must be at epsilon floor")
    self.replay_buffer = ReplayBuffer(capacity=50_000)
    self.training_rng = np.random.default_rng(self.agent_seed)
    self.rehearsal_pools = {}
    self.rehearsal_hashes = {}
    if self.arm == "rehearsal25":
        for stratum in OLD_STRATA:
            path = dataset_path(self.protocol, self.replica, stratum)
            expected = {
                "kind": "model-a-v4-rehearsal-dataset", "protocol_sha256": self.protocol_sha256,
                "replica": self.replica, "stratum": stratum,
                "source_sha256": self.protocol["source_parent"]["sha256"],
                "all_action_return_horizon": 8,
            }
            self.rehearsal_pools[stratum] = load_dataset(path, expected)
            if len(self.rehearsal_pools[stratum]) < REHEARSAL_PER_STRATUM:
                raise RuntimeError(f"rehearsal dataset too small: {stratum}")
            self.rehearsal_hashes[stratum] = sha256_file(path)
    self.training_diagnostics = {
        "raw_transitions": 0, "matured_targets": 0,
        "return_steps_histogram": {str(i): 0 for i in range(1, 9)},
        "kill_chain_targets_created": 0, "self_chain_targets_created": 0,
        "observed_kill_events": 0, "observed_self_events": 0,
        "survivor_terminal_merges": 0, "dead_terminal_appends": 0,
        "gradient_updates": 0, "sampled_batches": 0,
        "requested_kill_slots": 0, "fulfilled_kill_slots": 0,
        "requested_self_slots": 0, "fulfilled_self_slots": 0,
        "fallback_uniform_slots": 0, "realized_kill_chain_samples": 0,
        "realized_self_chain_samples": 0, "requested_rehearsal_slots": 0,
        "fulfilled_rehearsal_slots": 0,
        "rehearsal_by_stratum": {stratum: 0 for stratum in OLD_STRATA},
        "per_round": [],
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
    online_size = 64 if self.arm == "no_rehearsal" else 48
    transitions, sampling = self.replay_buffer.sample_stratified(
        online_size, KILL_SLOTS, SELF_SLOTS, self.training_rng,
    )
    rehearsal = []
    if self.arm == "rehearsal25":
        for stratum in OLD_STRATA:
            selected = sample_fixed(self.rehearsal_pools[stratum], REHEARSAL_PER_STRATUM, self.training_rng)
            rehearsal.extend(selected)
            self.training_diagnostics["rehearsal_by_stratum"][stratum] += len(selected)
        self.training_diagnostics["requested_rehearsal_slots"] += 16
        self.training_diagnostics["fulfilled_rehearsal_slots"] += len(rehearsal)
    transitions.extend(rehearsal)
    if len(transitions) != 64:
        raise RuntimeError("rehearsal learner did not construct a 64-item batch")
    for key in ("requested_kill_slots", "fulfilled_kill_slots", "requested_self_slots", "fulfilled_self_slots", "fallback_uniform_slots"):
        self.training_diagnostics[key] += int(sampling[key])
    self.training_diagnostics["realized_kill_chain_samples"] += sum(item.kill_chain for item in transitions)
    self.training_diagnostics["realized_self_chain_samples"] += sum(item.self_chain for item in transitions)
    self.training_diagnostics["sampled_batches"] += 1
    (local, global_features, actions, rewards, dones, discounts,
     next_local, next_global, next_masks) = _batch_tensors(transitions, self.device)
    q_values = self.online_net(local.float(), global_features.float()).gather(1, actions.long().unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        online_next = self.online_net(next_local.float(), next_global.float()).masked_fill(~next_masks.bool(), -1e9)
        next_actions = online_next.argmax(dim=1)
        target_next = self.target_net(next_local.float(), next_global.float()).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards.float() + discounts.float() * (1.0 - dones.float()) * target_next
    loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
    self.optimizer.zero_grad(); loss.backward()
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
        state=state, action=ACTIONS.index(action), reward=float(v4.reward_from_events(events)) + shaping,
        state_shaping=shaping, next_state=next_state, done=done, next_mask=next_mask,
        episode_id=int(old_game_state["round"]), start_step=int(old_game_state["step"]), events=tuple(events),
    )


def _add_target(self, window: list[RawTransition]) -> None:
    transition = aggregate_return(window)
    if self.run_mode == "collect":
        self.dataset_transitions.append(transition)
    else:
        self.replay_buffer.add(transition)
    d = self.training_diagnostics
    d["matured_targets"] += 1
    d["return_steps_histogram"][str(transition.return_steps)] += 1
    d["kill_chain_targets_created"] += int(transition.kill_chain)
    d["self_chain_targets_created"] += int(transition.self_chain)


def _ensure_round(self, game_state: dict) -> None:
    identity = int(game_state["round"])
    if self._round_identity is None:
        self._round_identity = identity
        self._round_start_raw = self.training_diagnostics["raw_transitions"]
        self._round_start_updates = self.training_diagnostics["gradient_updates"]
        self._round_opponent_states = {str(i): 0 for i in range(4)}
    elif self._round_identity != identity:
        raise RuntimeError("rehearsal round tracking crossed an unfinished episode")


def _append_raw(self, raw: RawTransition) -> None:
    self._episode_raw.append(raw)
    d = self.training_diagnostics
    d["raw_transitions"] += 1
    d["observed_kill_events"] += raw.events.count(e.KILLED_OPPONENT)
    d["observed_self_events"] += raw.events.count(e.KILLED_SELF)
    if not raw.done and len(self._episode_raw) >= RETURN_HORIZON:
        _add_target(self, self._episode_raw[:RETURN_HORIZON]); self._episode_raw.pop(0)
    if raw.done:
        while self._episode_raw:
            _add_target(self, self._episode_raw[:RETURN_HORIZON]); self._episode_raw.pop(0)
    self.training_steps += 1
    if self.run_mode == "train":
        progress = min(1.0, self.training_steps / v4.EPSILON_DECAY_STEPS)
        self.epsilon = v4.EPSILON_START + progress * (v4.EPSILON_FINAL - v4.EPSILON_START)
        _learn(self)


def _merge_survivor(self, state: dict, action: str, events) -> None:
    if not self._episode_raw:
        raise RuntimeError("rehearsal survivor has no pending transition")
    raw = self._episode_raw[-1]
    if (raw.episode_id, raw.start_step, raw.action) != (int(state["round"]), int(state["step"]), ACTIONS.index(action)):
        raise RuntimeError("rehearsal survivor transition identity mismatch")
    final = tuple(events)
    names = set(raw.events) | set(final)
    delta = {name: final.count(name) - raw.events.count(name) for name in names if final.count(name) != raw.events.count(name)}
    if delta != {e.SURVIVED_ROUND: 1}:
        raise RuntimeError(f"rehearsal survivor event delta changed: {delta}")
    raw.events = final
    raw.reward = float(v4.reward_from_events(final)) + raw.state_shaping
    raw.done = True; raw.next_state = None; raw.next_mask = None
    self.training_diagnostics["survivor_terminal_merges"] += 1
    while self._episode_raw:
        _add_target(self, self._episode_raw[:RETURN_HORIZON]); self._episode_raw.pop(0)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if self.run_mode == "evaluate":
        _record_evaluation(self, old_game_state, self_action, new_game_state, events)
        return
    _ensure_round(self, old_game_state)
    self._round_opponent_states[str(min(3, len(old_game_state.get("others", ())))) ] += 1
    raw = _raw(old_game_state, self_action, new_game_state, events, False)
    if raw is not None:
        _append_raw(self, raw)


def _finish_round_ledger(self, events) -> None:
    d = self.training_diagnostics
    transitions = d["raw_transitions"] - self._round_start_raw
    updates = d["gradient_updates"] - self._round_start_updates
    final = tuple(events)
    d["per_round"].append({
        "stage_round": len(d["per_round"]) + 1,
        "environment_round": self._round_identity,
        "transitions": transitions, "optimizer_updates": updates,
        "sampled_transitions": updates * 64, "episode_steps": transitions,
        "got_killed": int(e.GOT_KILLED in final), "killed_self": int(e.KILLED_SELF in final),
        "survived": int(e.SURVIVED_ROUND in final),
        "opponent_count_state_counts": dict(self._round_opponent_states),
    })
    self._round_identity = None


def _checkpoint_payload(self) -> dict:
    diagnostics = copy.deepcopy(self.training_diagnostics)
    diagnostics.update({
        "replay_size": len(self.replay_buffer), "replay_kill_items": self.replay_buffer.kill_items,
        "replay_self_items": self.replay_buffer.self_items,
    })
    return {
        "online_net": self.online_net.state_dict(), "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(), "completed_rounds": self.completed_rounds,
        "stage_start_rounds": self.stage_start_rounds,
        "stage_completed_rounds": self.completed_rounds - self.stage_start_rounds,
        "training_steps": self.training_steps, "gradient_steps": self.gradient_steps,
        "epsilon": self.epsilon, "global_feature_size": GLOBAL_SIZE, "architecture": MODEL_ARCHITECTURE,
        "agent_seed": self.agent_seed, "protocol_sha256": self.protocol_sha256,
        "arm": self.arm, "replica": self.replica, "stage_id": "task4c_rehearsal_pilot",
        "parent_sha256": self.parent_sha256,
        "single_training_variable": "fixed_25pct_source_task1_to_task3_rehearsal",
        "all_action_return_horizon": 8, "rehearsal_fraction": 0.0 if self.arm == "no_rehearsal" else 0.25,
        "rehearsal_dataset_hashes": self.rehearsal_hashes,
        "training_diagnostics": diagnostics,
    }


def end_of_round(self, last_game_state, last_action, events):
    if self.run_mode == "evaluate":
        _end_evaluation_round(self, last_game_state, last_action, events)
        return
    _ensure_round(self, last_game_state)
    if self._episode_raw and (
        self._episode_raw[-1].episode_id == int(last_game_state["round"])
        and self._episode_raw[-1].start_step == int(last_game_state["step"])
        and self._episode_raw[-1].action == ACTIONS.index(last_action)
    ):
        _merge_survivor(self, last_game_state, last_action, events)
    else:
        self._round_opponent_states[str(min(3, len(last_game_state.get("others", ())))) ] += 1
        raw = _raw(last_game_state, last_action, None, events, True)
        if raw is None:
            raise RuntimeError("rehearsal dead terminal transition missing")
        self.training_diagnostics["dead_terminal_appends"] += 1
        _append_raw(self, raw)
    if self._episode_raw:
        raise RuntimeError("rehearsal n=8 queue crossed an episode")
    _finish_round_ledger(self, events)
    if self.run_mode == "collect":
        self.collection_rounds += 1
        if self.collection_rounds == int(self.protocol["rehearsal_collection"]["rounds_per_stratum"]):
            payload = pack_dataset(self.dataset_transitions, {
                "schema_version": 1, "kind": "model-a-v4-rehearsal-dataset",
                "protocol_sha256": self.protocol_sha256, "replica": self.replica,
                "stratum": self.collection_stratum,
                "source_sha256": self.protocol["source_parent"]["sha256"],
                "all_action_return_horizon": 8, "policy_updates": 0,
                "collection_rounds": self.collection_rounds,
                "collection_diagnostics": copy.deepcopy(self.training_diagnostics),
            })
            _atomic_checkpoint(self.dataset_output, payload)
        elif self.collection_rounds > int(self.protocol["rehearsal_collection"]["rounds_per_stratum"]):
            raise RuntimeError("rehearsal collection exceeded round budget")
        return
    self.completed_rounds += 1
    stage_rounds = self.completed_rounds - self.stage_start_rounds
    if stage_rounds == 100:
        if self.checkpoint_path.exists():
            raise RuntimeError("refusing to overwrite rehearsal endpoint")
        _atomic_checkpoint(self.checkpoint_path, _checkpoint_payload(self))
    elif stage_rounds > 100:
        raise RuntimeError("rehearsal training exceeded fixed endpoint")


# Passive evaluation -------------------------------------------------------

def _setup_evaluation(self) -> None:
    import os
    self.eval_oracle_deadline = float(self.protocol["evaluation"]["oracle_deadline_seconds"])
    self.eval_world_seed = int(os.environ["MODEL_A_REHEARSAL_EVAL_WORLD_SEED"])
    expected_seed = None
    for case in self.protocol["evaluation"][self.eval_suite]["strata"][self.eval_stratum]["cases"]:
        if int(case["world_seed"]) == self.eval_world_seed:
            expected_seed = int(case["agent_seed"])
    if expected_seed is None or self.agent_seed != expected_seed:
        raise RuntimeError("rehearsal evaluation seed mismatch")
    self.eval_output = evaluation_diagnostic_path(self.protocol, self.eval_suite, self.eval_stratum, self.eval_label, self.eval_world_seed)
    supplied = __import__("pathlib").Path(os.environ["MODEL_A_REHEARSAL_EVAL_DIAGNOSTIC_PATH"]).resolve()
    if supplied != self.eval_output or supplied.exists():
        raise RuntimeError("rehearsal evaluation diagnostic path mismatch or overwrite")
    self.eval_expected_rounds = int(self.protocol["evaluation"][self.eval_suite]["strata"][self.eval_stratum]["rounds_per_case"])
    self._pending_eval_decision = None
    self._eval_history = []
    self._eval_successful_threat_bombs = set()
    self._eval_last_events = None
    self.eval_diagnostics = {
        "rounds": 0, "rows": 0, "approach_steps": 0, "bomb_legal": 0, "bombs": 0,
        "threat_bombs": 0, "trap_bombs": 0, "kill_events": 0, "self_kill_events": 0,
        "got_killed_events": 0, "survived_rounds": 0, "linked_kill_events": 0,
        "linked_threat_kill_events": 0, "unlinked_or_ambiguous_kill_events": 0,
        "oracle_evaluated": 0, "oracle_timeouts": 0,
    }


def _record_evaluation(self, old_game_state, action: str, new_game_state, events) -> None:
    pending = self._pending_eval_decision
    if pending is None:
        raise RuntimeError("rehearsal evaluation decision missing")
    if (int(old_game_state["round"]), int(old_game_state["step"]), ACTIONS.index(action)) != (pending["round"], pending["step"], pending["action"]):
        raise RuntimeError("rehearsal evaluation transition mismatch")
    d = self.eval_diagnostics; bomb = action == "BOMB"
    d["rows"] += 1
    d["approach_steps"] += int(pending["nearest"] >= 0 and _nearest_distance(new_game_state) >= 0 and _nearest_distance(new_game_state) < pending["nearest"])
    d["bomb_legal"] += int(pending["bomb_legal"]); d["oracle_evaluated"] += int(pending["oracle_evaluated"])
    d["oracle_timeouts"] += int(pending["oracle_timeout"]); d["bombs"] += int(bomb)
    d["threat_bombs"] += int(bomb and pending["affected_opponents"] >= 1)
    d["trap_bombs"] += int(bomb and pending["max_space_reduction"] >= 0.5)
    self._eval_history.append({**pending, "bomb": bomb})
    kills = tuple(events).count(e.KILLED_OPPONENT)
    d["kill_events"] += kills; d["self_kill_events"] += tuple(events).count(e.KILLED_SELF)
    d["got_killed_events"] += tuple(events).count(e.GOT_KILLED)
    if kills:
        matches = [item for item in self._eval_history if item["round"] == pending["round"] and item["bomb"] and pending["step"] - item["step"] in (4, 5)]
        if not matches and new_game_state is None:
            matches = [item for item in self._eval_history if item["round"] == pending["round"] and item["bomb"] and 0 <= pending["step"] - item["step"] <= 5]
        if len(matches) == 1:
            d["linked_kill_events"] += kills
            if matches[0]["affected_opponents"] >= 1:
                d["linked_threat_kill_events"] += kills
                self._eval_successful_threat_bombs.add((matches[0]["round"], matches[0]["step"]))
        else:
            d["unlinked_or_ambiguous_kill_events"] += kills
    self._eval_last_events = tuple(events); self._pending_eval_decision = None


def _end_evaluation_round(self, state, action, events) -> None:
    if self._pending_eval_decision is not None:
        _record_evaluation(self, state, action, None, events)
    else:
        final = tuple(events)
        if self._eval_last_events is None:
            raise RuntimeError("rehearsal survivor evaluation lacks final events")
        names = set(final) | set(self._eval_last_events)
        delta = {name: final.count(name) - self._eval_last_events.count(name) for name in names if final.count(name) != self._eval_last_events.count(name)}
        if delta != {e.SURVIVED_ROUND: 1}:
            raise RuntimeError(f"rehearsal evaluation survivor delta changed: {delta}")
        self.eval_diagnostics["survived_rounds"] += 1
    self.eval_diagnostics["rounds"] += 1; self._eval_last_events = None
    if self.eval_diagnostics["rounds"] == self.eval_expected_rounds:
        d = dict(self.eval_diagnostics)
        d["successful_threat_bombs"] = len(self._eval_successful_threat_bombs)
        d["threat_to_kill_conversion"] = d["successful_threat_bombs"] / d["threat_bombs"] if d["threat_bombs"] else 0.0
        d["linked_kills_per_threat"] = d["linked_threat_kill_events"] / d["threat_bombs"] if d["threat_bombs"] else 0.0
        _atomic_json(self.eval_output, {
            "schema_version": 1, "kind": "model-a-v4-rehearsal-evaluation-diagnostic",
            "protocol_sha256": self.protocol_sha256, "suite": self.eval_suite,
            "label": self.eval_label, "stratum": self.eval_stratum,
            "world_seed": self.eval_world_seed, "agent_seed": self.agent_seed,
            "policy_updates": 0, "actions_overridden": 0, "diagnostics": d,
        })
    elif self.eval_diagnostics["rounds"] > self.eval_expected_rounds:
        raise RuntimeError("rehearsal evaluation exceeded round budget")
