"""A shared frozen v4 backbone with multiple averaged residual heads."""

from __future__ import annotations

from agent_code.model_a_dqn.network import DuelingDQN, torch

if torch is None:
    class EnsembleDQN:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("v6 ensemble requires PyTorch")
else:
    from torch import nn

    def residual_head() -> nn.Sequential:
        return nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 6))


    class EnsembleDQN(nn.Module):
        def __init__(self, member_count: int, delta_cap: float) -> None:
            super().__init__()
            if member_count < 2:
                raise ValueError("an ensemble needs at least two residual heads")
            self.delta_cap = float(delta_cap)
            self.base = DuelingDQN()
            self.residual_heads = nn.ModuleList([residual_head() for _ in range(member_count)])

        def freeze_all(self) -> None:
            self.eval()
            for parameter in self.parameters():
                parameter.requires_grad = False

        def forward(self, local, global_features, return_delta: bool = False):
            combined = torch.cat((local.flatten(start_dim=1), global_features), dim=1)
            latent = self.base.encoder(combined)
            advantage = self.base.advantage(latent)
            base_q = self.base.value(latent) + advantage - advantage.mean(dim=1, keepdim=True)
            member_deltas = torch.stack([
                self.delta_cap * torch.tanh(head(latent)) for head in self.residual_heads
            ])
            mean_delta = member_deltas.mean(dim=0)
            final = base_q + mean_delta
            return (final, mean_delta, member_deltas) if return_delta else final
