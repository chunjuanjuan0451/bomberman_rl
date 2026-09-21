"""Bounded root-policy residual for the v10 classic planner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .interfaces import ACTIONS, MOVES
from .simulator import SimState


CHANNELS = 7
TARGET_AWARE_CHANNELS = 8
BOARD = 33
D4_TRANSFORMS = (
    "identity", "rot90", "rot180", "rot270",
    "flip_x", "flip_y", "transpose", "anti_transpose",
)

# Each tuple maps an original action index to its transformed action name.
ACTION_D4 = {
    "identity": ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"),
    "rot90": ("RIGHT", "DOWN", "LEFT", "UP", "WAIT", "BOMB"),
    "rot180": ("DOWN", "LEFT", "UP", "RIGHT", "WAIT", "BOMB"),
    "rot270": ("LEFT", "UP", "RIGHT", "DOWN", "WAIT", "BOMB"),
    "flip_x": ("UP", "LEFT", "DOWN", "RIGHT", "WAIT", "BOMB"),
    "flip_y": ("DOWN", "RIGHT", "UP", "LEFT", "WAIT", "BOMB"),
    "transpose": ("LEFT", "DOWN", "RIGHT", "UP", "WAIT", "BOMB"),
    "anti_transpose": ("RIGHT", "UP", "LEFT", "DOWN", "WAIT", "BOMB"),
}


def padded_centered_features(
    state: SimState,
    player: int = 0,
    *,
    safe_actions: tuple[str, ...],
    crate_target: tuple[int, int] | None = None,
    include_target_distance: bool = False,
) -> np.ndarray:
    """Encode the complete 17x17 observer board without clipping.

    Padding is wall-valued, not free-valued.  Channel six contains exactly
    the destinations represented by the planner's allowed root actions.
    ``WAIT`` and ``BOMB`` both map to the centred current cell.
    """
    if not state.agents[player].alive:
        raise ValueError("controlled agent must be alive")
    if any(action not in ACTIONS for action in safe_actions):
        raise ValueError("unknown planner safe action")
    channel_count = TARGET_AWARE_CHANNELS if include_target_distance else CHANNELS
    out = np.zeros((channel_count, BOARD, BOARD), dtype=np.float32)
    out[0].fill(-1.0)
    agent = state.agents[player]
    offset_x, offset_y = BOARD // 2 - agent.x, BOARD // 2 - agent.y

    def put(channel: int, x: int, y: int, value: float) -> None:
        xx, yy = int(x + offset_x), int(y + offset_y)
        if not (0 <= xx < BOARD and 0 <= yy < BOARD):
            raise ValueError("33x33 schema clipped an observer-board cell")
        out[channel, xx, yy] = value

    explosion_map = state.explosion_map()
    for x in range(state.field.shape[0]):
        for y in range(state.field.shape[1]):
            put(0, x, y, float(state.field[x, y]))
            put(2, x, y, float(explosion_map[x, y]))
            if state.coins.get((x, y), False):
                put(5, x, y, 1.0)
    for bomb in state.bombs:
        put(1, bomb.x, bomb.y, (bomb.timer + 1) / 5.0)
    for index, other in enumerate(state.agents):
        if other.alive:
            put(3 if index == player else 4, other.x, other.y, 1.0)
    for action in safe_actions:
        dx, dy = (0, 0) if action not in MOVES else MOVES[action][:2]
        put(6, agent.x + dx, agent.y + dy, 1.0)
    if include_target_distance and crate_target is not None:
        from .planner import _distance_map
        distances = _distance_map(state, {crate_target}, player)
        normalizer = float(2 * (state.field.shape[0] - 1))
        for (x, y), distance in distances.items():
            put(7, x, y, max(0.0, 1.0 - distance / normalizer))
    return out


def transform_features(features: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(features)
    if (values.shape[-2:] != (BOARD, BOARD)
            or values.shape[-3] not in (CHANNELS, TARGET_AWARE_CHANNELS)):
        raise ValueError("residual features must end in [7|8,33,33]")
    rotations = {"identity": 0, "rot90": 1, "rot180": 2, "rot270": 3}
    if transform in rotations:
        return np.rot90(values, k=rotations[transform], axes=(-2, -1)).copy()
    if transform == "flip_x":
        return np.flip(values, axis=-2).copy()
    if transform == "flip_y":
        return np.flip(values, axis=-1).copy()
    if transform == "transpose":
        return np.swapaxes(values, -2, -1).copy()
    if transform == "anti_transpose":
        return np.flip(np.flip(np.swapaxes(values, -2, -1), axis=-2), axis=-1).copy()
    raise ValueError(f"unknown D4 transform: {transform}")


def transform_action_vector(values: np.ndarray, transform: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1] != len(ACTIONS):
        raise ValueError("action vector must have 6 entries")
    if transform not in ACTION_D4:
        raise ValueError(f"unknown D4 transform: {transform}")
    result = np.zeros_like(array)
    for source_index, transformed_action in enumerate(ACTION_D4[transform]):
        result[..., ACTIONS.index(transformed_action)] = array[..., source_index]
    return result


def planner_residual_policy(
    planner_prior: torch.Tensor,
    safe_mask: torch.Tensor,
    residual_logits: torch.Tensor,
) -> torch.Tensor:
    """Return normalized ``planner logits + residual`` with a hard mask."""
    if planner_prior.shape != residual_logits.shape or safe_mask.shape != residual_logits.shape:
        raise ValueError("planner, mask and residual shapes must match")
    if torch.any(safe_mask.sum(dim=-1) <= 0):
        raise ValueError("every residual-policy sample needs an allowed action")
    planner_logits = torch.log(planner_prior.clamp_min(1e-12))
    combined = (planner_logits + residual_logits).masked_fill(safe_mask <= 0, float("-inf"))
    return torch.softmax(combined, dim=-1)


class BoundedResidualPolicy(nn.Module):
    architecture = "depthwise-residual-policy-v2"

    def __init__(self, channels: int = 32, blocks: int = 3, bound: float = 0.5):
        super().__init__()
        if channels <= 0 or blocks <= 0 or bound <= 0:
            raise ValueError("invalid residual model dimensions")
        self.bound = float(bound)
        self.stem = nn.Sequential(nn.Conv2d(CHANNELS, channels, 3, padding=1), nn.ReLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.ReLU(), nn.Conv2d(channels, channels, 1), nn.ReLU(),
            ) for _ in range(blocks)
        ])
        self.head = nn.Linear(channels, len(ACTIONS))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(values)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        hidden = hidden.mean(dim=(-2, -1))
        return self.bound * torch.tanh(self.head(hidden))

    def save_checkpoint(self, path: Path, metadata: dict) -> None:
        torch.save({"state_dict": self.state_dict(), "metadata": dict(metadata)}, path)

    @classmethod
    def load_checkpoint(cls, path: Path, expected: dict | None = None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        if metadata.get("architecture") != cls.architecture:
            raise ValueError("residual metadata mismatch")
        if expected and any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("residual binding mismatch")
        model = cls(int(metadata["channels"]), int(metadata["blocks"]), float(metadata["bound"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        return model, metadata


class ActionConditionedResidualPolicy(nn.Module):
    """State-local residual head with a shared scorer for the four moves.

    The v2 global-average head could minimize soft CE with an almost constant
    WAIT bias.  This head retains the root and destination embeddings and uses
    one shared move scorer, so directional decisions must depend on local
    state while D4-equivalent moves share parameters.
    """

    architecture = "action-conditioned-residual-policy-v3"

    def __init__(self, channels: int = 32, blocks: int = 3, bound: float = 0.5,
                 input_channels: int = CHANNELS):
        super().__init__()
        if channels <= 0 or blocks <= 0 or bound <= 0:
            raise ValueError("invalid residual model dimensions")
        self.bound = float(bound)
        self.input_channels = int(input_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, channels, 3, padding=1), nn.ReLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.ReLU(), nn.Conv2d(channels, channels, 1), nn.ReLU(),
            ) for _ in range(blocks)
        ])
        context = channels * 4
        self.move_hidden = nn.Sequential(nn.Linear(context, channels), nn.ReLU())
        self.move_output = nn.Linear(channels, 1, bias=False)
        self.stationary_hidden = nn.Sequential(nn.Linear(channels * 2, channels), nn.ReLU())
        self.stationary_output = nn.Linear(channels, 2, bias=False)
        nn.init.zeros_(self.move_output.weight)
        nn.init.zeros_(self.stationary_output.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(values)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        center = hidden[:, :, BOARD // 2, BOARD // 2]
        global_context = hidden.mean(dim=(-2, -1))
        move_logits = []
        for action in ACTIONS[:4]:
            dx, dy, _ = MOVES[action]
            destination = hidden[:, :, BOARD // 2 + dx, BOARD // 2 + dy]
            context = torch.cat(
                (center, destination, destination - center, global_context), dim=-1)
            move_logits.append(self.move_output(self.move_hidden(context)))
        stationary_context = torch.cat((center, global_context), dim=-1)
        stationary_logits = self.stationary_output(self.stationary_hidden(stationary_context))
        logits = torch.cat((*move_logits, stationary_logits), dim=-1)
        return self.bound * torch.tanh(logits)

    def save_checkpoint(self, path: Path, metadata: dict) -> None:
        torch.save({"state_dict": self.state_dict(), "metadata": dict(metadata)}, path)

    @classmethod
    def load_checkpoint(cls, path: Path, expected: dict | None = None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        if metadata.get("architecture") != cls.architecture:
            raise ValueError("action-conditioned residual metadata mismatch")
        if expected and any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("action-conditioned residual binding mismatch")
        model = cls(int(metadata["channels"]), int(metadata["blocks"]), float(metadata["bound"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        return model, metadata


class TargetAwareResidualPolicy(ActionConditionedResidualPolicy):
    """Action-conditioned residual with a wall-aware persistent-target channel."""

    architecture = "target-aware-action-conditioned-residual-policy-v4"

    def __init__(self, channels: int = 32, blocks: int = 3, bound: float = 0.5):
        super().__init__(channels, blocks, bound, input_channels=TARGET_AWARE_CHANNELS)


class PlannerAwareResidualPolicy(ActionConditionedResidualPolicy):
    """Target-aware residual conditioned on the policy it is correcting."""

    architecture = "planner-aware-action-conditioned-residual-policy-v5"
    requires_planner_prior = True

    def __init__(self, channels: int = 32, blocks: int = 3, bound: float = 0.5):
        super().__init__(channels, blocks, bound, input_channels=TARGET_AWARE_CHANNELS)
        # Per-action planner context is [probability, clipped log probability,
        # log-margin below the planner top action, planner entropy].
        planner_features = 4
        self.move_hidden = nn.Sequential(
            nn.Linear(channels * 4 + planner_features, channels), nn.ReLU())
        self.stationary_hidden = nn.Sequential(
            nn.Linear(channels * 2 + planner_features + 2, channels), nn.ReLU())
        self.stationary_output = nn.Linear(channels, 1, bias=False)
        nn.init.zeros_(self.move_output.weight)
        nn.init.zeros_(self.stationary_output.weight)
        self.register_buffer(
            "stationary_identity", torch.eye(2, dtype=torch.float32), persistent=False)

    @staticmethod
    def planner_features(planner_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if planner_prior.ndim != 2 or planner_prior.shape[-1] != len(ACTIONS):
            raise ValueError("planner-aware residual requires [batch,6] planner prior")
        if torch.any(planner_prior < 0) or torch.any(planner_prior.sum(-1) <= 0):
            raise ValueError("planner-aware residual received an invalid planner prior")
        probability = planner_prior / planner_prior.sum(dim=-1, keepdim=True)
        log_probability = torch.log(probability.clamp_min(1e-6)).clamp_min(-12.0)
        top_log_probability = log_probability.max(dim=-1, keepdim=True).values
        entropy = -(probability * log_probability).sum(-1, keepdim=True)
        per_action = torch.stack((
            probability, log_probability, log_probability - top_log_probability,
            entropy.expand_as(probability),
        ), dim=-1)
        return per_action, torch.cat((probability, log_probability), dim=-1)

    def _forward_context(self, values: torch.Tensor, planner_prior: torch.Tensor):
        if values.ndim != 4 or values.shape[1:] != (
                TARGET_AWARE_CHANNELS, BOARD, BOARD):
            raise ValueError("planner-aware residual requires [batch,8,33,33] features")
        if values.shape[0] != planner_prior.shape[0]:
            raise ValueError("planner-aware feature/prior batch mismatch")
        hidden = self.stem(values)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        center = hidden[:, :, BOARD // 2, BOARD // 2]
        global_context = hidden.mean(dim=(-2, -1))
        per_action, global_planner = self.planner_features(planner_prior)
        move_logits = []
        for index, action in enumerate(ACTIONS[:4]):
            dx, dy, _ = MOVES[action]
            destination = hidden[:, :, BOARD // 2 + dx, BOARD // 2 + dy]
            context = torch.cat((
                center, destination, destination - center, global_context,
                per_action[:, index],
            ), dim=-1)
            move_logits.append(self.move_output(self.move_hidden(context)))
        stationary_logits = []
        stationary_board = torch.cat((center, global_context), dim=-1)
        identity = self.stationary_identity.to(values).expand(values.shape[0], -1, -1)
        for offset, index in enumerate((ACTIONS.index("WAIT"), ACTIONS.index("BOMB"))):
            context = torch.cat((
                stationary_board, per_action[:, index], identity[:, offset],
            ), dim=-1)
            stationary_logits.append(
                self.stationary_output(self.stationary_hidden(context)))
        logits = torch.cat((*move_logits, *stationary_logits), dim=-1)
        return self.bound * torch.tanh(logits), center, global_context, global_planner

    def forward(self, values: torch.Tensor, planner_prior: torch.Tensor) -> torch.Tensor:
        return self._forward_context(values, planner_prior)[0]


class PlannerAwareGatedResidualPolicy(PlannerAwareResidualPolicy):
    """Planner-aware residual with a learned state-level correction gate."""

    architecture = "planner-aware-gated-residual-policy-v6"

    def __init__(self, channels: int = 32, blocks: int = 3, bound: float = 0.5):
        super().__init__(channels, blocks, bound)
        self.gate_hidden = nn.Sequential(
            nn.Linear(channels * 2 + len(ACTIONS) * 2, channels), nn.ReLU())
        self.gate_output = nn.Linear(channels, 1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.constant_(self.gate_output.bias, -2.0)

    def forward_with_aux(self, values: torch.Tensor, planner_prior: torch.Tensor):
        raw_residual, center, global_context, global_planner = self._forward_context(
            values, planner_prior)
        gate_logit = self.gate_output(self.gate_hidden(torch.cat(
            (center, global_context, global_planner), dim=-1))).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
        return raw_residual * gate[:, None], gate_logit, raw_residual

    def forward(self, values: torch.Tensor, planner_prior: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(values, planner_prior)[0]


def residual_forward(model: nn.Module, features: torch.Tensor,
                     planner_prior: torch.Tensor) -> torch.Tensor:
    """Call legacy or planner-aware residual heads through one strict interface."""
    if getattr(model, "requires_planner_prior", False):
        return model(features, planner_prior)
    return model(features)


def load_residual_checkpoint(path: Path, expected: dict | None = None):
    """Dispatch a tensor-only residual checkpoint by bound architecture."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    architecture = payload.get("metadata", {}).get("architecture")
    classes = {
        BoundedResidualPolicy.architecture: BoundedResidualPolicy,
        ActionConditionedResidualPolicy.architecture: ActionConditionedResidualPolicy,
        TargetAwareResidualPolicy.architecture: TargetAwareResidualPolicy,
        PlannerAwareResidualPolicy.architecture: PlannerAwareResidualPolicy,
        PlannerAwareGatedResidualPolicy.architecture: PlannerAwareGatedResidualPolicy,
    }
    if architecture not in classes:
        raise ValueError(f"unsupported residual architecture: {architecture!r}")
    return classes[architecture].load_checkpoint(path, expected)


class PlannerResidualTransform:
    """Root-only transform applied after ``PlannerRootPolicyTransform``."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.forward_calls = 0
        self.crate_target: tuple[int, int] | None = None

    def set_crate_target(self, target: tuple[int, int] | None) -> None:
        self.crate_target = target

    def __call__(self, state: SimState, player: int,
                 planner_prior: dict[str, float]) -> dict[str, float]:
        allowed = tuple(action for action in ACTIONS if action in planner_prior)
        if not allowed:
            raise ValueError("planner residual received an empty prior")
        target_aware = getattr(self.model, "input_channels", CHANNELS) == TARGET_AWARE_CHANNELS
        features = padded_centered_features(
            state, player, safe_actions=allowed,
            crate_target=self.crate_target,
            include_target_distance=target_aware,
        )
        prior = torch.zeros((1, len(ACTIONS)), dtype=torch.float32)
        mask = torch.zeros_like(prior)
        for action in allowed:
            index = ACTIONS.index(action)
            prior[0, index] = float(planner_prior[action])
            mask[0, index] = 1.0
        with torch.inference_mode():
            residual = residual_forward(
                self.model, torch.from_numpy(features[None, ...]), prior)
            probabilities = planner_residual_policy(prior, mask, residual)[0].cpu().numpy()
        self.forward_calls += 1
        return {action: float(probabilities[ACTIONS.index(action)]) for action in allowed}
