"""Compact residual CNN with distributional action values and a risk head."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = object

from .features import GLOBAL_SIZE, SPATIAL_CHANNELS

N_ACTIONS = 6
N_QUANTILES = 32


if torch is not None:
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels),
            )

        def forward(self, x):
            return torch.relu(x + self.layers(x))


    class V9Network(nn.Module):
        def __init__(self, n_quantiles=N_QUANTILES):
            super().__init__()
            self.n_quantiles = int(n_quantiles)
            self.encoder = nn.Sequential(
                nn.Conv2d(SPATIAL_CHANNELS, 24, 3, padding=1), nn.ReLU(),
                ResidualBlock(24),
                nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
                ResidualBlock(32), nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.trunk = nn.Sequential(nn.Linear(32 * 4 * 4 + GLOBAL_SIZE, 96), nn.ReLU())
            self.value = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, self.n_quantiles))
            self.advantage = nn.Sequential(
                nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, N_ACTIONS * self.n_quantiles),
            )
            self.risk = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, N_ACTIONS))

        def forward(self, spatial, global_features, return_risk=False):
            latent = self.encoder(spatial).flatten(1)
            latent = self.trunk(torch.cat((latent, global_features), dim=1))
            value = self.value(latent)[:, None, :]
            advantage = self.advantage(latent).view(-1, N_ACTIONS, self.n_quantiles)
            quantiles = value + advantage - advantage.mean(dim=1, keepdim=True)
            return (quantiles, self.risk(latent)) if return_risk else quantiles

        def q_values(self, spatial, global_features):
            return self.forward(spatial, global_features).mean(dim=-1)
else:
    class V9Network:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Model A v9 requires PyTorch")
