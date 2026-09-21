"""Small deterministic nonlinear policy head for v10 offline distillation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .interfaces import ACTIONS
from .runtime import center_feature_maps


INPUT_SIZE = 1445


def prepare_policy_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim == 3:
        values = center_feature_maps(values)[None]
    elif values.ndim == 4:
        values = center_feature_maps(values)
    else:
        raise ValueError("policy features must be [C,X,Y] or [B,C,X,Y]")
    values = values.reshape(values.shape[0], -1).astype(np.float32) / 4.0
    if values.shape[1] != INPUT_SIZE or not np.isfinite(values).all():
        raise ValueError("invalid policy feature schema")
    return values


class NonlinearPolicyHead(nn.Module):
    architecture = "mlp-1445-128-ln-gelu-64-ln-gelu-6"

    def __init__(self, input_size: int = INPUT_SIZE, action_count: int = len(ACTIONS)):
        super().__init__()
        if input_size != INPUT_SIZE or action_count != len(ACTIONS):
            raise ValueError("nonlinear policy schema mismatch")
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, action_count),
        )
        self.feature_schema = "centered-v1"

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)

    @classmethod
    def save_checkpoint(cls, path: Path, model: "NonlinearPolicyHead", metadata: dict) -> None:
        payload = {"state_dict": model.state_dict(), "metadata": dict(metadata)}
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: Path, expected: dict | None = None) -> tuple["NonlinearPolicyHead", dict]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        required = {"architecture", "input_size", "action_count", "training_seed", "config_sha256", "data_sha256"}
        if not required.issubset(metadata) or metadata["architecture"] != cls.architecture:
            raise ValueError("nonlinear policy metadata mismatch")
        if expected and any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("nonlinear policy checkpoint binding mismatch")
        model = cls(int(metadata["input_size"]), int(metadata["action_count"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        model.feature_schema = str(metadata.get("feature_schema", "centered-v1"))
        if model.feature_schema not in ("centered-v1", "centered-opponent-projection-v1"):
            raise ValueError("unknown nonlinear policy feature schema")
        model.eval()
        return model, metadata


class RootPolicyBlender:
    """Apply the distilled policy once at the search root and blend safely.

    Deep search nodes continue to use the lightweight expert network.  This
    keeps the policy head's cost constant per official action callback.
    """

    def __init__(self, model: NonlinearPolicyHead, *, weight: float):
        self.model = model.eval()
        self.weight = float(weight)
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("nonlinear root policy weight must be in [0, 1]")
        self.forward_calls = 0
        self.failure_count = 0

    def __call__(self, state, player: int,
                 base_policy: dict[str, float]) -> dict[str, float]:
        if self.weight == 0.0 or not base_policy:
            return dict(base_policy)
        from .runtime import opponent_aware_state_features, state_features

        try:
            extractor = (opponent_aware_state_features
                         if self.model.feature_schema == "centered-opponent-projection-v1"
                         else state_features)
            prepared = torch.from_numpy(prepare_policy_features(extractor(state, player)))
            with torch.inference_mode():
                logits = self.model(prepared)
                self.forward_calls += 1
                legal_indices = [ACTIONS.index(action) for action in base_policy]
                probabilities = masked_policy(logits, legal_indices)[0].cpu().numpy()
        except (FloatingPointError, RuntimeError, ValueError):
            # A legal expert prior is already available before this optional
            # head runs, so production inference remains safe on bad numerics.
            self.failure_count += 1
            return dict(base_policy)
        learned = {action: float(probabilities[ACTIONS.index(action)])
                   for action in base_policy}
        mixed = {action: ((1.0 - self.weight) * base_policy[action]
                          + self.weight * learned[action]) for action in base_policy}
        total = sum(mixed.values())
        if not np.isfinite(list(mixed.values())).all() or total <= 0.0:
            raise FloatingPointError("invalid nonlinear root policy")
        return {action: value / total for action, value in mixed.items()}


def masked_policy(logits: torch.Tensor, legal_indices: list[int] | None = None) -> torch.Tensor:
    values = logits.clone()
    if legal_indices is not None:
        mask = torch.full_like(values, float("-inf")); mask[:, legal_indices] = values[:, legal_indices]; values = mask
    probabilities = torch.softmax(values, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("non-finite policy probabilities")
    return probabilities / probabilities.sum(dim=-1, keepdim=True)
