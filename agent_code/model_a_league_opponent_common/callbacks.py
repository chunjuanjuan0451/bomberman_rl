"""Deterministic fixed-pool opponent scheduler for s162000."""

from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np

from agent_code.model_a_cnn_n8.callbacks import _load, _tensors
from agent_code.model_a_cnn_n8.features import ACTIONS as CNN_ACTIONS, legal_action_mask as cnn_legal, state_to_features as cnn_features
from agent_code.model_a_cnn_n8.network import ARCHITECTURE, FullBoardDuelingCNN, torch
from agent_code.model_a_dqn.callbacks import _load_checkpoint, _load_network_state
from agent_code.model_a_dqn.features import ACTIONS as MLP_ACTIONS, legal_action_mask as mlp_legal, state_to_features as mlp_features
from agent_code.model_a_dqn.network import DuelingDQN
from agent_code.seeded_coin_collector_agent.callbacks import act as coin_act
from agent_code.seeded_rule_based_agent.callbacks import act as rule_act, reset_self


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scheduled_roster(pool: list[str], schedule_seed: int, round_number: int) -> tuple[str, str, str]:
    rng = np.random.default_rng(np.random.SeedSequence([int(schedule_seed), int(round_number), 0x1EA6]))
    return tuple(str(pool[index]) for index in rng.choice(len(pool), size=3, replace=False))


def _policy_seed(base: int, slot: int, label: str) -> int:
    tag = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")
    return int(np.random.SeedSequence([base, slot, tag]).generate_state(1)[0])


def setup_agent(self, slot: int) -> None:
    protocol_path = Path(os.environ["CNN_LEAGUE_PROTOCOL_PATH"]).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("kind") != "model-a-cnn-n8-opponent-league-ab":
        raise RuntimeError("invalid opponent league protocol")
    arm = os.environ["CNN_LEAGUE_ARM"]
    replica = os.environ["CNN_LEAGUE_REPLICA"]
    self._league_pool = list(protocol["arms"][arm]["pool"])
    self._league_seed = int(protocol["replicas"][replica]["schedule_seed"])
    self._league_slot = int(slot)
    self._league_inventory = protocol["opponent_inventory"]
    self._league_contexts = {}
    self._league_base_seed = int(protocol["replicas"][replica]["opponent_seed"])
    torch.set_num_threads(1)


def _context(self, label: str):
    if label in self._league_contexts:
        return self._league_contexts[label]
    seed = _policy_seed(self._league_base_seed, self._league_slot, label)
    context = SimpleNamespace(logger=self.logger, train=False, rng=np.random.default_rng(seed))
    spec = self._league_inventory[label]
    kind = spec["kind"]
    if kind == "rule":
        context.rng = random.Random(seed)
        reset_self(context)
        context.current_round = 0
    elif kind == "coin":
        context.rng = random.Random(seed)
    elif kind == "random":
        pass
    elif kind == "cnn":
        path = ROOT / spec["path"]
        if _sha256(path) != spec["sha256"]:
            raise RuntimeError(f"league CNN checkpoint drift: {label}")
        payload = _load(path)
        if payload.get("architecture") != ARCHITECTURE:
            raise RuntimeError(f"league CNN architecture mismatch: {label}")
        context.device = torch.device("cpu")
        context.online_net = FullBoardDuelingCNN().to(context.device)
        context.online_net.load_state_dict(payload["online_net"], strict=True)
        context.online_net.eval()
    elif kind == "mlp":
        path = ROOT / spec["path"]
        if _sha256(path) != spec["sha256"]:
            raise RuntimeError(f"league MLP checkpoint drift: {label}")
        payload = _load_checkpoint(path)
        context.device = torch.device("cpu")
        context.online_net = DuelingDQN().to(context.device)
        _load_network_state(context.online_net, payload["online_net"])
        context.online_net.eval()
    else:
        raise RuntimeError(f"unknown league policy kind: {kind}")
    context.kind = kind
    self._league_contexts[label] = context
    return context


def _learned_action(context, game_state: dict, cnn: bool) -> str:
    actions, mask_fn, feature_fn = ((CNN_ACTIONS, cnn_legal, cnn_features) if cnn else
                                    (MLP_ACTIONS, mlp_legal, mlp_features))
    indices = np.flatnonzero(mask_fn(game_state))
    if not indices.size:
        return "WAIT"
    features = feature_fn(game_state)
    assert features is not None
    if cnn:
        spatial, scalars = _tensors(features, context.device)
        with torch.no_grad():
            q = context.online_net(spatial, scalars)[0].detach().cpu().numpy()
    else:
        local, global_features = features
        with torch.no_grad():
            q = context.online_net(
                torch.as_tensor(local[None], dtype=torch.float32),
                torch.as_tensor(global_features[None], dtype=torch.float32),
            )[0].cpu().numpy()
    best = q[indices].max()
    return actions[int(context.rng.choice(indices[np.isclose(q[indices], best)]))]


def act_agent(self, game_state: dict) -> str:
    label = scheduled_roster(self._league_pool, self._league_seed, int(game_state["round"]))[self._league_slot]
    context = _context(self, label)
    if context.kind == "rule":
        return rule_act(context, game_state)
    if context.kind == "coin":
        return coin_act(context, game_state)
    if context.kind == "random":
        return str(context.rng.choice(np.asarray(["RIGHT", "LEFT", "UP", "DOWN", "BOMB"]),
                                      p=np.asarray([.23, .23, .23, .23, .08])))
    return _learned_action(context, game_state, context.kind == "cnn")


__all__ = ["act_agent", "scheduled_roster", "setup_agent"]
