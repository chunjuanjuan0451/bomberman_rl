from pathlib import Path

import numpy as np
import torch

from agent_code.model_a_cnn_n8.features import ACTIONS, CENTER, state_to_features
from agent_code.model_a_cnn_n8_symmetry_ab.callbacks import aligned_d4_q_matrix
from agent_code.model_a_v6.symmetry import ACTION_PERMUTATIONS, SYMMETRY_NAMES, transform_game_state
from tools.cnn_n8_d4_symmetry_ab import DEFAULT_PROTOCOL, ROOT, load_protocol


class NeighborCoinNetwork(torch.nn.Module):
    """Orientation-sensitive probe whose move Q follows an adjacent coin."""

    def forward(self, spatial, scalars):
        coins = spatial[:, 2]
        moves = torch.stack((
            coins[:, CENTER - 1, CENTER],
            coins[:, CENTER, CENTER + 1],
            coins[:, CENTER + 1, CENTER],
            coins[:, CENTER, CENTER - 1],
        ), dim=1)
        stationary = torch.zeros((len(spatial), 2), dtype=spatial.dtype, device=spatial.device)
        return torch.cat((moves, stationary), dim=1)


def _state():
    field = np.zeros((17, 17), dtype=np.int8)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "round": 1, "step": 12, "field": field,
        "self": ("me", 3, True, (8, 8)),
        "others": [("other", 0, True, (12, 12))],
        "coins": [(8, 7)], "bombs": [], "explosion_map": np.zeros_like(field),
    }


def test_all_d4_action_permutations_are_bijections_and_keep_stationary_actions():
    assert ACTION_PERMUTATIONS.shape == (8, 6)
    for permutation in ACTION_PERMUTATIONS:
        assert sorted(map(int, permutation)) == list(range(6))
        assert permutation[ACTIONS.index("WAIT")] == ACTIONS.index("WAIT")
        assert permutation[ACTIONS.index("BOMB")] == ACTIONS.index("BOMB")


def test_aligned_d4_views_map_an_orientation_sensitive_policy_back_to_original_actions():
    aligned = aligned_d4_q_matrix(NeighborCoinNetwork(), _state(), torch.device("cpu"))
    assert aligned.shape == (len(SYMMETRY_NAMES), len(ACTIONS))
    assert np.allclose(aligned[:, ACTIONS.index("UP")], 1.0)
    assert np.allclose(np.delete(aligned, ACTIONS.index("UP"), axis=1), 0.0)


def test_cnn_scalar_features_are_d4_invariant():
    state = _state()
    original = state_to_features(state)[1]
    for symmetry in range(len(SYMMETRY_NAMES)):
        transformed = state_to_features(transform_game_state(state, symmetry))[1]
        assert np.array_equal(transformed, original)


def test_d4_protocol_is_frozen_rotated_and_read_only():
    protocol, digest = load_protocol(Path(DEFAULT_PROTOCOL))
    assert protocol["collection"]["rounds_per_arm"] == 160
    assert protocol["collection"]["total_environment_rounds"] == 320
    assert protocol["collection"]["policy_updates"] == 0
    for stratum in ("rule", "mixed"):
        assert {case["seat"] for case in protocol["collection"]["cases"].values()
                if case["stratum"] == stratum} == set(range(4))
    assert len(digest) == 64


def test_d4_final_protocol_uses_fresh_larger_selection_free_budget():
    path = ROOT / "experiments/configs/model-a-cnn-n8-d4-symmetry-final-s171000.json"
    protocol, digest = load_protocol(path)
    assert protocol["collection"]["rounds_per_arm_case"] == 50
    assert protocol["collection"]["rounds_per_arm"] == 400
    assert protocol["collection"]["total_environment_rounds"] == 800
    assert protocol["evidence_source"]["required_decision"] == (
        "d4_symmetry_supported_for_selection_free_confirmation_stop"
    )
    assert protocol["decision_rule"]["pass_decision"] == (
        "d4_symmetry_confirmed_as_final_internal_candidate_stop"
    )
    assert len(digest) == 64
