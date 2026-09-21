"""Dueling Q-network contract for Model A."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # Allows non-training tools to inspect the repository.
    torch = None
    nn = object

from .features import GLOBAL_SIZE


if torch is None:
    class DuelingDQN:
        """Helpful import-time placeholder when PyTorch is absent."""

        def __init__(self) -> None:
            raise RuntimeError("Model A requires PyTorch; install it before training.")
else:
    class DuelingDQN(nn.Module):
        """Reserved interface: forward(local_batch, global_batch) -> [batch, 6]."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(), nn.Linear(4 * 7 * 7 + GLOBAL_SIZE, 128), nn.ReLU(),
                nn.Linear(128, 96), nn.ReLU(),
            )
            self.value = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 1))
            self.advantage = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 6))

        def forward(self, local, global_features):
            x = torch.cat((local.flatten(start_dim=1), global_features), dim=1)
            x = self.encoder(x)
            advantage = self.advantage(x)
            return self.value(x) + advantage - advantage.mean(dim=1, keepdim=True)
