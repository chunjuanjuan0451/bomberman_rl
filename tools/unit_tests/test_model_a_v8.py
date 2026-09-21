import numpy as np

from agent_code.model_a_dqn.network import DuelingDQN
from agent_code.model_a_v8.config import architecture_name, load_config
from agent_code.model_a_v8.network import StableDQN, torch
from agent_code.model_a_v8.train import rollout_soft_target


def _state():
    field = -np.ones((9, 9), dtype=int)
    field[1:8, 1:8] = 0
    field[5:8, 3] = -1
    field[5:8, 5] = -1
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("me", 0, True, (4, 4)),
        "others": [("other", 0, True, (7, 4))], "bombs": [], "coins": [],
        "explosion_map": np.zeros_like(field), "user_input": None,
    }


def test_v8_config_and_architecture_are_isolated():
    config, _, digest = load_config("experiments/configs/v8-rollout-distill-s8301.json")
    assert config["rollout_distillation"] and len(digest) == 64
    assert architecture_name().startswith("model-a-v8-")


def test_v8_zero_residual_exactly_matches_frozen_v4():
    torch.manual_seed(8)
    v4 = DuelingDQN()
    torch.manual_seed(99)
    v8 = StableDQN(0.25)
    v8.base.load_state_dict(v4.state_dict())
    local = torch.randn(2, 4, 7, 7)
    glob = torch.randn(2, 7)
    with torch.no_grad():
        assert torch.equal(v4(local, glob), v8(local, glob))


def test_v8_rollout_produces_a_soft_bounded_bomb_label():
    target, mask = rollout_soft_target(_state(), 0.25)
    assert mask.sum() == 1 and mask[-1]
    assert 0 < target[-1] <= 0.25
