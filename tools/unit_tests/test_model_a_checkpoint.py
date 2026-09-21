import numpy as np

from agent_code.model_a_dqn.callbacks import (
    CHECKPOINT_PATH,
    MODEL_ARCHITECTURE,
    _load_checkpoint,
    _load_network_state,
)
from agent_code.model_a_dqn.features import GLOBAL_SIZE, LOCAL_SHAPE
from agent_code.model_a_dqn.network import DuelingDQN, torch


def test_model_a_old_checkpoint_migration_preserves_q_values():
    old_model = DuelingDQN()
    old_state = old_model.state_dict()
    old_weight = old_state["encoder.1.weight"][:, :-3].clone()
    old_state["encoder.1.weight"] = old_weight

    migrated_model = DuelingDQN()
    assert _load_network_state(migrated_model, old_state)
    migrated_model.eval()

    local = torch.as_tensor(np.zeros((1, *LOCAL_SHAPE)), dtype=torch.float32)
    old_globals = torch.as_tensor(np.zeros((1, GLOBAL_SIZE - 3)), dtype=torch.float32)
    new_globals = torch.as_tensor(np.zeros((1, GLOBAL_SIZE)), dtype=torch.float32)
    with torch.no_grad():
        old_input = torch.cat((local.flatten(start_dim=1), old_globals), dim=1)
        old_encoded = torch.nn.functional.linear(
            old_input, old_weight, old_state["encoder.1.bias"]
        )
        old_encoded = old_model.encoder[2:](old_encoded)
        old_advantage = old_model.advantage(old_encoded)
        old_q = old_model.value(old_encoded) + old_advantage - old_advantage.mean(dim=1, keepdim=True)
        migrated_q = migrated_model(local, new_globals)

    assert torch.allclose(old_q, migrated_q)


def test_default_v4_checkpoint_matches_current_network():
    checkpoint = _load_checkpoint(CHECKPOINT_PATH)
    model = DuelingDQN()
    assert not _load_network_state(model, checkpoint["online_net"])


def test_checkpoint_architecture_metadata_is_enforced():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "wrong.pt"
        torch.save({
            "online_net": DuelingDQN().state_dict(),
            "architecture": "different-architecture",
        }, path)
        try:
            _load_checkpoint(path)
        except RuntimeError as exc:
            assert MODEL_ARCHITECTURE in str(exc)
        else:
            raise AssertionError("Architecture mismatch must be rejected")
