"""Frozen v4 backbone with a bounded, zero-initialized residual."""

from __future__ import annotations

from agent_code.model_a_dqn.network import DuelingDQN, torch

if torch is None:
    class StableDQN:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Stable v6 requires PyTorch")
else:
    from torch import nn

    class StableDQN(nn.Module):
        def __init__(self, delta_cap: float) -> None:
            super().__init__()
            self.delta_cap = float(delta_cap)
            self.base = DuelingDQN()
            self.residual = nn.Sequential(
                nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 6),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

        def freeze_base(self) -> None:
            self.base.eval()
            for parameter in self.base.parameters():
                parameter.requires_grad = False

        def forward(self, local, global_features, return_delta: bool = False):
            combined = torch.cat((local.flatten(start_dim=1), global_features), dim=1)
            latent = self.base.encoder(combined)
            advantage = self.base.advantage(latent)
            base_q = self.base.value(latent) + advantage - advantage.mean(dim=1, keepdim=True)
            delta = self.delta_cap * torch.tanh(self.residual(latent))
            final = base_q + delta
            return (final, delta) if return_delta else final
