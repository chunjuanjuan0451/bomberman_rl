"""Frozen v4 plus a bounded, zero-initialized resource-map residual."""

from __future__ import annotations

from agent_code.model_a_dqn.network import DuelingDQN, torch


if torch is None:
    class GlobalResourceDQN:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("global-resource experiment requires PyTorch")
else:
    from torch import nn

    class GlobalResourceDQN(nn.Module):
        def __init__(self, delta_cap: float = 0.5) -> None:
            super().__init__()
            self.delta_cap = float(delta_cap)
            self.base = DuelingDQN()
            self.resource_encoder = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(24, 24, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                nn.Linear(24 * 4 * 4, 64), nn.ReLU(),
            )
            self.resource_head = nn.Linear(64, 6)
            nn.init.zeros_(self.resource_head.weight)
            nn.init.zeros_(self.resource_head.bias)

        def freeze_base(self) -> None:
            self.base.eval()
            for parameter in self.base.parameters():
                parameter.requires_grad = False

        def forward(self, local, global_features, resource, return_delta: bool = False):
            base_q = self.base(local, global_features)
            hidden = self.resource_encoder(resource)
            delta = self.delta_cap * torch.tanh(self.resource_head(hidden))
            output = base_q + delta
            return (output, delta) if return_delta else output
