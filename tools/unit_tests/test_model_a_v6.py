import json
from pathlib import Path

import numpy as np

from agent_code.model_a_v6.config import VARIANT_SPECS, architecture_name, load_v6_config
from agent_code.model_a_v6.features import (
    ACTIONS,
    ACTION_FEATURE_SIZE,
    GLOBAL_SIZE,
    LOCAL_SHAPE,
    action_tactical_features,
    legal_action_mask,
    state_to_features,
)
from agent_code.model_a_v6.network import DuelingDQN, torch
from agent_code.model_a_v6.train import setup_training
from agent_code.model_a_dqn.network import DuelingDQN as V4DuelingDQN
from agent_code.model_a_v6.symmetry import (
    ACTION_PERMUTATIONS,
    MATRICES,
    SYMMETRY_NAMES,
    augment_transition_arrays_random,
    transform_game_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _state():
    field = -np.ones((9, 9), dtype=int)
    field[1:8, 1:8] = 0
    field[2, 3] = 1
    field[6, 5] = 1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("me", 0, True, (4, 4)),
        "others": [("opponent", 0, True, (7, 4))],
        "bombs": [((3, 6), 2)],
        "coins": [(6, 4), (2, 5)],
        "explosion_map": np.zeros_like(field),
        "user_input": None,
    }


def test_v6_configs_encode_exactly_one_factor_and_combined_is_disabled():
    for variant, expected in VARIANT_SPECS.items():
        path = REPOSITORY_ROOT / "experiments/configs" / f"{variant}.json"
        config, resolved, digest = load_v6_config(path)
        assert resolved == path.resolve()
        assert len(digest) == 64
        assert config["augmentation"] == expected["augmentation"]
        assert config["tactical_residual"] == expected["tactical_residual"]
        assert config["enabled_for_training"] == (variant != "v6d-combined")


def test_v6_config_rejects_a_variant_factor_mismatch():
    import tempfile

    source = REPOSITORY_ROOT / "experiments/configs/v6a-control.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["augmentation"] = "random_d4"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        try:
            load_v6_config(path)
        except ValueError as exc:
            assert "requires augmentation" in str(exc)
        else:
            raise AssertionError("A variant/config mismatch must be rejected")


def test_v6_base_and_tactical_residual_have_matched_initial_q_values():
    torch.manual_seed(123)
    control = DuelingDQN(tactical_residual=False)
    torch.manual_seed(123)
    tactical = DuelingDQN(tactical_residual=True)
    local = torch.randn(3, *LOCAL_SHAPE)
    global_features = torch.randn(3, GLOBAL_SIZE)
    action_features = torch.randn(3, len(ACTIONS), ACTION_FEATURE_SIZE)
    with torch.no_grad():
        control_q = control(local, global_features)
        tactical_q = tactical(local, global_features, action_features)
    assert torch.equal(control_q, tactical_q)
    assert architecture_name({"tactical_residual": False}) != architecture_name({"tactical_residual": True})


def test_v6a_base_network_matches_v4_architecture_and_initialization():
    torch.manual_seed(456)
    v4 = V4DuelingDQN()
    torch.manual_seed(456)
    v6a = DuelingDQN(tactical_residual=False)
    assert set(v4.state_dict()) == set(v6a.state_dict())
    assert all(torch.equal(v4.state_dict()[name], v6a.state_dict()[name]) for name in v4.state_dict())


def test_v6_tactical_features_use_path_progress_and_hypothetical_bomb():
    state = _state()
    tactical = action_tactical_features(state)
    assert tactical.shape == (len(ACTIONS), ACTION_FEATURE_SIZE)
    assert tactical[ACTIONS.index("RIGHT"), 0] > tactical[ACTIONS.index("LEFT"), 0]
    assert legal_action_mask(state)[ACTIONS.index("BOMB")]
    assert tactical[ACTIONS.index("BOMB"), 2] > 0.0
    assert tactical[ACTIONS.index("BOMB"), 3] > 0.0
    assert np.all((-1.0 <= tactical) & (tactical <= 1.0))


def test_v6_full_feature_and_mask_extraction_is_d4_equivariant():
    state = _state()
    local, global_features = state_to_features(state)
    tactical = action_tactical_features(state)
    mask = legal_action_mask(state)
    for symmetry in range(len(SYMMETRY_NAMES)):
        transformed = transform_game_state(state, symmetry)
        transformed_local, transformed_global = state_to_features(transformed)
        transformed_tactical = action_tactical_features(transformed)
        transformed_mask = legal_action_mask(transformed)
        permutation = ACTION_PERMUTATIONS[symmetry]
        # Re-extraction is the reference contract for coordinate conventions.
        from agent_code.model_a_v6.symmetry import transform_local
        expected_local = transform_local(local, symmetry)
        assert np.array_equal(transformed_local, expected_local)
        assert np.array_equal(transformed_global, global_features)
        assert np.array_equal(transformed_tactical[permutation], tactical)
        assert np.array_equal(transformed_mask[permutation], mask)


def test_v6_d4_matrices_are_unique_and_have_inverses():
    assert len({tuple(matrix.flat) for matrix in MATRICES}) == 8
    identity = np.eye(2, dtype=np.int8)
    for matrix in MATRICES:
        assert any(np.array_equal(other @ matrix, identity) for other in MATRICES)


def test_v6_random_d4_preserves_batch_size_and_aligns_actions_masks_and_tactical_rows():
    class FixedRng:
        def integers(self, high, size):
            assert high == 8 and size == 2
            return np.asarray([1, 4])

    local = np.zeros((2, *LOCAL_SHAPE), dtype=np.float32)
    tactical = np.arange(2 * len(ACTIONS) * ACTION_FEATURE_SIZE, dtype=np.float32).reshape(
        2, len(ACTIONS), ACTION_FEATURE_SIZE,
    )
    actions = np.asarray([ACTIONS.index("RIGHT"), ACTIONS.index("WAIT")])
    masks = np.zeros((2, len(ACTIONS)), dtype=bool)
    masks[np.arange(2), actions] = True
    result = augment_transition_arrays_random(
        local, np.zeros((2, GLOBAL_SIZE), dtype=np.float32), tactical,
        actions, np.zeros(2), np.zeros(2), local,
        np.zeros((2, GLOBAL_SIZE), dtype=np.float32), tactical, masks, FixedRng(),
    )
    out_local, _, out_tactical, out_actions, _, _, _, _, out_next_tactical, out_masks = result
    assert out_local.shape == local.shape
    for index, symmetry in enumerate((1, 4)):
        permutation = ACTION_PERMUTATIONS[symmetry]
        assert out_actions[index] == permutation[actions[index]]
        assert out_masks[index, out_actions[index]]
        assert np.array_equal(out_tactical[index, permutation], tactical[index])
        assert np.array_equal(out_next_tactical[index, permutation], tactical[index])


def test_v6_training_setup_is_fresh_and_replay_rng_is_matched_across_variants():
    class Logger:
        def info(self, *args, **kwargs):
            pass

    replay_sequences = []
    for variant in ("v6a-control", "v6b-random-d4", "v6c-tactical-residual"):
        config, _, digest = load_v6_config(
            REPOSITORY_ROOT / "experiments/configs" / f"{variant}.json",
        )
        holder = type("TrainingState", (), {})()
        holder.v6_config = config
        holder.v6_config_sha256 = digest
        holder.model_architecture = architecture_name(config)
        holder.agent_seed = config["agent_seed"]
        holder.device = torch.device("cpu")
        torch.manual_seed(holder.agent_seed)
        holder.online_net = DuelingDQN(config["tactical_residual"])
        holder.logger = Logger()
        setup_training(holder)
        assert holder.training_steps == 0
        assert holder.gradient_steps == 0
        assert len(holder.replay_buffer) == 0
        replay_sequences.append(holder.replay_rng.integers(10_000, size=12))
    assert np.array_equal(replay_sequences[0], replay_sequences[1])
    assert np.array_equal(replay_sequences[0], replay_sequences[2])
