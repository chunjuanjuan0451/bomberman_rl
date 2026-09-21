import numpy as np

from agent_code.model_a_v4_global_resource.config import (
    ARMS, REPLICAS, RESOURCE_SHAPE, architecture_name, load_protocol,
)
from agent_code.model_a_v4_global_resource.features import combined_features, resource_features
from agent_code.model_a_v4_global_resource.network import GlobalResourceDQN, torch
from tools.v4_global_resource_stage1 import preflight, registered_seeds


def _state():
    field = -np.ones((17, 17), dtype=int)
    field[1:16, 1:16] = 0
    field[4, 4] = 1
    field[14, 14] = 1
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("me", 0, True, (1, 1)),
        "others": [("other", 0, True, (14, 13))],
        "bombs": [((13, 14), 2)], "coins": [(2, 2), (15, 15)],
        "explosion_map": np.zeros_like(field), "user_input": None,
    }


def test_resource_arms_have_identical_shape_and_different_extent():
    local = resource_features(_state(), "local7")
    full = resource_features(_state(), "full33")
    assert local.shape == full.shape == RESOURCE_SHAPE
    assert local[1].sum() == 1
    assert full[1].sum() == 2
    assert full[0, 14 - 1 + 16, 14 - 1 + 16] == 1
    assert local[0, 14 - 1 + 16, 14 - 1 + 16] == -1


def test_resource_input_excludes_opponents_bombs_and_explosions():
    state = _state()
    baseline = resource_features(state, "full33")
    changed = dict(state, others=[], bombs=[], explosion_map=np.ones_like(state["field"]))
    assert np.array_equal(baseline, resource_features(changed, "full33"))


def test_combined_features_preserve_exact_v4_inputs():
    local, glob, resource = combined_features(_state(), "full33")
    assert local.shape == (4, 7, 7)
    assert glob.shape == (7,)
    assert resource.shape == RESOURCE_SHAPE


def test_zero_initialized_resource_head_is_exactly_frozen_v4():
    model = GlobalResourceDQN(0.5)
    model.freeze_base()
    local = torch.randn(3, 4, 7, 7)
    glob = torch.randn(3, 7)
    resource = torch.randn(3, *RESOURCE_SHAPE)
    with torch.no_grad():
        base = model.base(local, glob)
        result, delta = model(local, glob, resource, True)
    assert torch.equal(base, result)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert all(not parameter.requires_grad for parameter in model.base.parameters())


def test_resource_branch_can_learn_without_base_gradients():
    model = GlobalResourceDQN(0.5)
    model.freeze_base()
    result = model(torch.randn(2, 4, 7, 7), torch.randn(2, 7), torch.randn(2, *RESOURCE_SHAPE))
    result.sum().backward()
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert model.resource_head.weight.grad is not None


def test_stage1_protocol_and_seed_contract():
    protocol, _, digest = load_protocol("experiments/configs/model-a-v4-global-resource-stage1-s140000.json")
    assert tuple(protocol["training"]["arms"]) == ARMS
    assert tuple(protocol["training"]["replicas"]) == REPLICAS
    assert protocol["reward_contract"]["new_attack_reward_added"] is False
    assert protocol["signal_gate"]["kills_used_for_gate"] is False
    assert len(registered_seeds(protocol)) == 14
    assert len(digest) == 64 and architecture_name().endswith("v1")
    assert preflight(protocol)["passed"] is True
