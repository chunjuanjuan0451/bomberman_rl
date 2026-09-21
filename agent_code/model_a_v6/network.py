"""Dueling Q-network contract for Model A."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # Allows non-training tools to inspect the repository.
    torch = None
    nn = object

from .features import ACTION_FEATURE_SIZE, GLOBAL_SIZE


if torch is None:
    class DuelingDQN:
        """Helpful import-time placeholder when PyTorch is absent."""

        def __init__(self) -> None:
            raise RuntimeError("Model A requires PyTorch; install it before training.")
else:
    class DuelingDQN(nn.Module):
        """Reserved interface: forward(local_batch, global_batch) -> [batch, 6]."""

        def __init__(self, tactical_residual: bool = False) -> None:
            super().__init__()
            self.tactical_residual = tactical_residual
            self.encoder = nn.Sequential(
                nn.Flatten(), nn.Linear(4 * 7 * 7 + GLOBAL_SIZE, 128), nn.ReLU(),
                nn.Linear(128, 96), nn.ReLU(),
            )
            self.value = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 1))
            self.advantage = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 6))
            if tactical_residual:
                self.action_residual = nn.Sequential(
                    nn.Linear(96 + ACTION_FEATURE_SIZE, 48), nn.ReLU(), nn.Linear(48, 1),
                )
                # Matched initialization: v6a and v6c start with identical base
                # Q-values when constructed under the same torch seed.
                nn.init.zeros_(self.action_residual[-1].weight)
                nn.init.zeros_(self.action_residual[-1].bias)

        def forward(self, local, global_features, action_features=None):
            x = torch.cat((local.flatten(start_dim=1), global_features), dim=1)
            x = self.encoder(x)
            advantage = self.advantage(x)
            q_values = self.value(x) + advantage - advantage.mean(dim=1, keepdim=True)
            if not self.tactical_residual:
                return q_values
            if action_features is None:
                action_features = torch.zeros(
                    (x.shape[0], 6, ACTION_FEATURE_SIZE), dtype=x.dtype, device=x.device,
                )
            expanded = x.unsqueeze(1).expand(-1, 6, -1)
            residual_input = torch.cat((expanded, action_features), dim=2).flatten(end_dim=1)
            residual = self.action_residual(residual_input).reshape(-1, 6)
            return q_values + residual
