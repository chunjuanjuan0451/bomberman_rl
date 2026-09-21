"""Official Bomberman callback entry points for Model A."""

import os
from pathlib import Path

import numpy as np

from .features import ACTIONS, legal_action_mask, state_to_features
from .network import DuelingDQN, torch

MODEL_ARCHITECTURE = "model-a-mlp-v4-global7"
_DEFAULT_CHECKPOINT_PATH = Path(__file__).with_name("model_a.pt")
# Evaluation and training tools may isolate experimental checkpoints without
# changing the tournament default. Relative paths are resolved by main.py from
# the repository root.
CHECKPOINT_PATH = Path(os.environ.get("MODEL_A_CHECKPOINT_PATH", _DEFAULT_CHECKPOINT_PATH))


def _load_checkpoint(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "online_net" not in checkpoint:
        raise RuntimeError(f"Invalid Model A checkpoint: {path}")
    architecture = checkpoint.get("architecture")
    if architecture is not None and architecture != MODEL_ARCHITECTURE:
        raise RuntimeError(
            f"Checkpoint architecture {architecture!r} is incompatible with {MODEL_ARCHITECTURE!r}: {path}"
        )
    return checkpoint


def _load_network_state(model, state_dict: dict) -> bool:
    """Load a checkpoint, zero-extending the first layer for new features.

    Returns whether a schema migration was required.  Zero initialization of
    appended input columns preserves every Q-value produced by the old model.
    """
    current = model.state_dict()
    migrated = False
    for name, saved in state_dict.items():
        if name not in current:
            raise RuntimeError(f"Unexpected checkpoint parameter: {name}")
        if current[name].shape == saved.shape:
            current[name] = saved
            continue
        if (
            name == "encoder.1.weight"
            and current[name].shape[0] == saved.shape[0]
            and current[name].shape[1] > saved.shape[1]
        ):
            expanded = current[name].new_zeros(current[name].shape)
            expanded[:, :saved.shape[1]] = saved
            current[name] = expanded
            migrated = True
            continue
        raise RuntimeError(
            f"Incompatible checkpoint shape for {name}: {tuple(saved.shape)} -> {tuple(current[name].shape)}"
        )
    model.load_state_dict(current)
    return migrated


def setup(self):
    if torch is None:
        raise RuntimeError("Model A requires PyTorch. Install it with: python -m pip install torch")
    torch.set_num_threads(1)
    seed_text = os.environ.get("MODEL_A_SEED")
    self.agent_seed = None if seed_text is None else int(seed_text)
    if self.agent_seed is not None:
        torch.manual_seed(self.agent_seed)
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = DuelingDQN().to(self.device)
    self.resume_checkpoint = None
    self.feature_schema_migrated = False
    resume = os.environ.get("MODEL_A_RESUME", "") == "1"
    if not self.train and not CHECKPOINT_PATH.exists():
        raise FileNotFoundError("model_a.pt is required for evaluation. Train Model A first.")
    if (not self.train or resume) and CHECKPOINT_PATH.exists():
        checkpoint = _load_checkpoint(CHECKPOINT_PATH)
        self.feature_schema_migrated = _load_network_state(self.online_net, checkpoint["online_net"])
        self.resume_checkpoint = checkpoint if self.train else None
        self.completed_rounds = int(checkpoint.get("completed_rounds", 0))
        self.logger.info("Loaded Model A checkpoint from round %d.", self.completed_rounds)
        if self.feature_schema_migrated:
            self.logger.info("Zero-extended Model A input layer for appended offensive features.")
    else:
        self.completed_rounds = 0
        self.logger.info("Starting a fresh Model A Dueling Double DQN run.")


def act(self, game_state: dict) -> str:
    """Epsilon-greedy masked action selection."""
    legal_indices = np.flatnonzero(legal_action_mask(game_state))
    if legal_indices.size == 0:
        return "WAIT"
    epsilon = getattr(self, "epsilon", 0.0)
    if self.train and self.rng.random() < epsilon:
        return ACTIONS[int(self.rng.choice(legal_indices))]
    features = state_to_features(game_state)
    assert features is not None
    local, global_features = features
    with torch.no_grad():
        q_values = self.online_net(
            torch.as_tensor(local[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
        )[0].cpu().numpy()
    best_value = q_values[legal_indices].max()
    best_indices = legal_indices[np.isclose(q_values[legal_indices], best_value)]
    return ACTIONS[int(self.rng.choice(best_indices))]
