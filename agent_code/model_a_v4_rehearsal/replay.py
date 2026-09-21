"""Replay utilities for fixed 25% old-task rehearsal."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agent_code.model_a_dqn.network import torch
from agent_code.model_a_v4_sampling_ab.replay import ReplayBuffer, Transition


def pack_dataset(transitions: list[Transition], metadata: dict) -> dict:
    if not transitions:
        raise ValueError("cannot save an empty rehearsal dataset")
    local_shape = transitions[0].state[0].shape
    global_shape = transitions[0].state[1].shape
    return {
        **metadata,
        "count": len(transitions),
        "local": torch.as_tensor(np.stack([x.state[0] for x in transitions])),
        "global": torch.as_tensor(np.stack([x.state[1] for x in transitions])),
        "action": torch.as_tensor([x.action for x in transitions], dtype=torch.int64),
        "reward": torch.as_tensor([x.reward for x in transitions], dtype=torch.float32),
        "next_local": torch.as_tensor(np.stack([x.next_state[0] if x.next_state is not None else np.zeros(local_shape, dtype=np.float32) for x in transitions])),
        "next_global": torch.as_tensor(np.stack([x.next_state[1] if x.next_state is not None else np.zeros(global_shape, dtype=np.float32) for x in transitions])),
        "done": torch.as_tensor([x.done for x in transitions], dtype=torch.bool),
        "next_mask": torch.as_tensor(np.stack([x.next_mask if x.next_mask is not None else np.zeros(6, dtype=bool) for x in transitions])),
        "discount": torch.as_tensor([x.bootstrap_discount for x in transitions], dtype=torch.float32),
        "return_steps": torch.as_tensor([x.return_steps for x in transitions], dtype=torch.int16),
        "episode_id": torch.as_tensor([x.episode_id for x in transitions], dtype=torch.int32),
        "start_step": torch.as_tensor([x.start_step for x in transitions], dtype=torch.int16),
        "kill_chain": torch.as_tensor([x.kill_chain for x in transitions], dtype=torch.bool),
        "self_chain": torch.as_tensor([x.self_chain for x in transitions], dtype=torch.bool),
    }


def unpack_dataset(payload: dict) -> list[Transition]:
    count = int(payload["count"])
    fields = ("local", "global", "action", "reward", "next_local", "next_global", "done", "next_mask", "discount", "return_steps", "episode_id", "start_step", "kill_chain", "self_chain")
    if any(len(payload[field]) != count for field in fields):
        raise ValueError("rehearsal dataset tensor lengths disagree")
    result = []
    for i in range(count):
        done = bool(payload["done"][i])
        result.append(Transition(
            state=(payload["local"][i].numpy(), payload["global"][i].numpy()),
            action=int(payload["action"][i]), reward=float(payload["reward"][i]),
            next_state=None if done else (payload["next_local"][i].numpy(), payload["next_global"][i].numpy()),
            done=done, next_mask=None if done else payload["next_mask"][i].numpy().astype(bool),
            bootstrap_discount=float(payload["discount"][i]), return_steps=int(payload["return_steps"][i]),
            episode_id=int(payload["episode_id"][i]), start_step=int(payload["start_step"][i]),
            kill_chain=bool(payload["kill_chain"][i]), self_chain=bool(payload["self_chain"][i]),
        ))
    return result


def load_dataset(path: Path, expected: dict) -> list[Transition]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"rehearsal dataset {key} mismatch: {path}")
    return unpack_dataset(payload)


def sample_fixed(pool: list[Transition], count: int, rng: np.random.Generator) -> list[Transition]:
    if len(pool) < count:
        raise RuntimeError("rehearsal pool cannot fill its preregistered slots")
    indices = rng.choice(len(pool), size=int(count), replace=False)
    return [pool[int(index)] for index in indices]


__all__ = ["ReplayBuffer", "Transition", "load_dataset", "pack_dataset", "sample_fixed", "unpack_dataset"]
