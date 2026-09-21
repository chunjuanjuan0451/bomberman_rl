"""Compact full-board Dueling CNN sized for safe Apple-M4 training."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = object


ARCHITECTURE = "model-a-cnn-n8-full33-v1"


if torch is None:
    class FullBoardDuelingCNN:
        def __init__(self) -> None:
            raise RuntimeError("model_a_cnn_n8 requires PyTorch")
else:
    class FullBoardDuelingCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spatial_encoder = nn.Sequential(
                nn.Conv2d(11, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(32 * 9 * 9, 128), nn.ReLU(),
            )
            self.scalar_encoder = nn.Sequential(nn.Linear(6, 16), nn.ReLU())
            self.shared = nn.Sequential(nn.Linear(144, 96), nn.ReLU())
            self.value = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1))
            self.advantage = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 6))

        def forward(self, spatial, scalars):
            encoded = torch.cat((self.spatial_encoder(spatial), self.scalar_encoder(scalars)), dim=1)
            hidden = self.shared(encoded)
            advantage = self.advantage(hidden)
            return self.value(hidden) + advantage - advantage.mean(dim=1, keepdim=True)


__all__ = ["ARCHITECTURE", "FullBoardDuelingCNN", "torch"]
