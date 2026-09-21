import numpy as np

from agent_code.model_a_v9.config import architecture_name, load_config
from agent_code.model_a_v9.features import GLOBAL_SIZE, SPATIAL_CHANNELS, counterfactual_risk_labels, state_to_features
from agent_code.model_a_v9.network import N_QUANTILES, V9Network, torch
from agent_code.model_a_v9.replay import PrioritizedReplay


def _state():
    field = -np.ones((17, 17), dtype=int)
    field[1:16, 1:16] = 0
    field[4, 5] = 1
    return {
        "round": 1, "step": 10, "field": field, "self": ("me", 0, True, (4, 4)),
        "others": [("other", 0, True, (8, 4))], "bombs": [((6, 4), 2)],
        "coins": [(3, 3)], "explosion_map": np.zeros_like(field), "user_input": None,
    }


def test_v9_features_cover_the_full_board_and_history():
    spatial, glob = state_to_features(_state())
    assert spatial.shape == (SPATIAL_CHANNELS, 17, 17)
    assert glob.shape == (GLOBAL_SIZE,)
    assert spatial[:, 16, 16].shape[0] == SPATIAL_CHANNELS


def test_v9_network_outputs_quantiles_and_action_risk():
    model = V9Network()
    quantiles, risk = model(torch.zeros(2, SPATIAL_CHANNELS, 17, 17), torch.zeros(2, GLOBAL_SIZE), True)
    assert quantiles.shape == (2, 6, N_QUANTILES)
    assert risk.shape == (2, 6)
    assert sum(p.numel() for p in model.parameters()) < 1_000_000


def test_v9_risk_labels_ignore_impossible_actions():
    risk, mask = counterfactual_risk_labels(_state())
    assert risk.shape == mask.shape == (6,)
    assert np.all(risk[~mask] == 0)


def test_v9_config_and_prioritized_replay_contract():
    config, _, digest = load_config("experiments/configs/v9-fullboard-qr-dqn-s8501.json")
    assert config["rounds"] == 2000 and len(digest) == 64
    assert architecture_name().startswith("model-a-v9-")
    replay = PrioritizedReplay(4, 0.6)
    for value in range(4):
        replay.add(value)
    items, indices, weights = replay.sample(2, 0.4, np.random.default_rng(9))
    assert len(items) == len(indices) == len(weights) == 2
