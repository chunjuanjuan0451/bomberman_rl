import json
from pathlib import Path

import numpy as np

from agent_code.model_a_dqn.network import DuelingDQN
from agent_code.model_a_v6.features import GLOBAL_SIZE, LOCAL_SHAPE
from agent_code.model_a_v6_stable.config import load_config
from agent_code.model_a_v6_stable.network import StableDQN, torch

ROOT = Path(__file__).resolve().parents[2]


def _parent_state():
    try:
        checkpoint = torch.load(ROOT / "agent_code/model_a_dqn/model_a.pt", map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ROOT / "agent_code/model_a_dqn/model_a.pt", map_location="cpu")
    return checkpoint["online_net"]


def test_stable_configs_are_matched_except_d4_treatment():
    control, _, _ = load_config(ROOT / "experiments/configs/v6s1-residual-control.json")
    treatment, _, _ = load_config(ROOT / "experiments/configs/v6s2-d4-consistency.json")
    assert not control["d4_consistency"] and treatment["d4_consistency"]
    ignored = {"run_id", "variant", "d4_consistency", "checkpoint_path"}
    assert {k: v for k, v in control.items() if k not in ignored and k != "hyperparameters"} == {
        k: v for k, v in treatment.items() if k not in ignored and k != "hyperparameters"
    }
    different_hyper = {
        key for key in control["hyperparameters"]
        if control["hyperparameters"][key] != treatment["hyperparameters"][key]
    }
    assert different_hyper == {"d4_batch_fraction", "d4_lambda_final"}


def test_stable_network_starts_identical_to_v4_and_bounds_delta():
    base = DuelingDQN()
    base.load_state_dict(_parent_state())
    stable = StableDQN(delta_cap=0.25)
    stable.base.load_state_dict(_parent_state())
    stable.freeze_base()
    local = torch.randn(5, *LOCAL_SHAPE)
    global_features = torch.randn(5, GLOBAL_SIZE)
    with torch.no_grad():
        initial, initial_delta = stable(local, global_features, return_delta=True)
        expected = base(local, global_features)
    assert torch.equal(initial, expected)
    assert torch.count_nonzero(initial_delta) == 0
    with torch.no_grad():
        stable.residual[-1].bias.fill_(100.0)
        _, delta = stable(local, global_features, return_delta=True)
    assert torch.all(delta <= 0.25) and torch.all(delta >= -0.25)


def test_optimizer_step_cannot_change_frozen_base():
    stable = StableDQN(delta_cap=0.25)
    stable.base.load_state_dict(_parent_state())
    stable.freeze_base()
    before = {name: value.clone() for name, value in stable.base.state_dict().items()}
    optimizer = torch.optim.Adam(stable.residual.parameters(), lr=1e-4)
    local = torch.randn(4, *LOCAL_SHAPE)
    global_features = torch.randn(4, GLOBAL_SIZE)
    loss = stable(local, global_features).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert all(torch.equal(before[name], value) for name, value in stable.base.state_dict().items())
    assert all(not parameter.requires_grad for parameter in stable.base.parameters())
