"""Reuse the frozen Stage-1 residual network without modification."""

from agent_code.model_a_v4_global_resource.network import (  # noqa: F401
    GlobalResourceDQN,
    torch,
)

__all__ = ["GlobalResourceDQN", "torch"]
