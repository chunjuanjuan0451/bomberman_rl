"""Frozen-v4 network with zero-initialized bounded Task-3 residuals."""

from __future__ import annotations

from agent_code.model_a_dqn.network import DuelingDQN, torch

from .features import ACTION_FEATURE_SIZE


if torch is None:
    class Task3DQN:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Task-3 Model A requires PyTorch")
else:
    from torch import nn

    class Task3DQN(nn.Module):
        def __init__(self, delta_cap: float, use_action_features: bool) -> None:
            super().__init__()
            self.delta_cap = float(delta_cap)
            self.use_action_features = bool(use_action_features)
            self.base = DuelingDQN()
            if self.use_action_features:
                self.residual = nn.Sequential(
                    nn.Linear(96 + ACTION_FEATURE_SIZE, 48), nn.ReLU(), nn.Linear(48, 1),
                )
            else:
                self.residual = nn.Sequential(
                    nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 6),
                )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

        def freeze_base(self) -> None:
            self.base.eval()
            for parameter in self.base.parameters():
                parameter.requires_grad = False

        def forward(self, local, global_features, action_features=None, return_delta=False):
            combined = torch.cat((local.flatten(start_dim=1), global_features), dim=1)
            latent = self.base.encoder(combined)
            advantage = self.base.advantage(latent)
            base_q = self.base.value(latent) + advantage - advantage.mean(dim=1, keepdim=True)
            if self.use_action_features:
                if action_features is None:
                    action_features = torch.zeros(
                        (latent.shape[0], 6, ACTION_FEATURE_SIZE),
                        dtype=latent.dtype, device=latent.device,
                    )
                expanded = latent.unsqueeze(1).expand(-1, 6, -1)
                inputs = torch.cat((expanded, action_features), dim=2).flatten(end_dim=1)
                raw_delta = self.residual(inputs).reshape(-1, 6)
            else:
                raw_delta = self.residual(latent)
            delta = self.delta_cap * torch.tanh(raw_delta)
            final = base_q + delta
            return (final, delta) if return_delta else final
