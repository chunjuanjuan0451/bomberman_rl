"""Official callback entry points for the isolated Model A v6 experiments."""

import os
from pathlib import Path

import numpy as np

from .config import architecture_name, load_v6_config
from .features import ACTIONS, action_tactical_features, legal_action_mask, state_to_features
from .network import DuelingDQN, torch

CHECKPOINT_PATH = Path(os.environ.get("MODEL_A_V6_CHECKPOINT_PATH", "model_a_v6.pt"))


def _load_checkpoint(path: Path, architecture: str, config_sha256: str):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "online_net" not in checkpoint:
        raise RuntimeError(f"Invalid Model A v6 checkpoint: {path}")
    saved_architecture = checkpoint.get("architecture")
    if saved_architecture != architecture:
        raise RuntimeError(
            f"Checkpoint architecture {saved_architecture!r} is incompatible with {architecture!r}: {path}"
        )
    saved_config_sha256 = checkpoint.get("config_sha256")
    if saved_config_sha256 != config_sha256:
        raise RuntimeError(
            f"Checkpoint config hash {saved_config_sha256!r} does not match {config_sha256!r}: {path}"
        )
    return checkpoint


def _load_network_state(model, state_dict: dict) -> None:
    """Strictly load a v6 state dict; cross-variant migration is forbidden."""
    current = model.state_dict()
    if set(current) != set(state_dict):
        missing = sorted(set(current) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(current))
        raise RuntimeError(f"Checkpoint keys differ: missing={missing}, unexpected={unexpected}")
    for name, saved in state_dict.items():
        if current[name].shape != saved.shape:
            raise RuntimeError(
                f"Incompatible checkpoint shape for {name}: {tuple(saved.shape)} -> {tuple(current[name].shape)}"
            )
    model.load_state_dict(state_dict)


def setup(self):
    if torch is None:
        raise RuntimeError("Model A v6 requires PyTorch")
    torch.set_num_threads(1)
    self.v6_config, self.v6_config_path, self.v6_config_sha256 = load_v6_config()
    self.model_architecture = architecture_name(self.v6_config)
    seed_text = os.environ.get("MODEL_A_V6_SEED")
    self.agent_seed = None if seed_text is None else int(seed_text)
    if self.agent_seed is not None:
        torch.manual_seed(self.agent_seed)
    self.rng = np.random.default_rng(self.agent_seed)
    self.device = torch.device("cpu")
    self.online_net = DuelingDQN(self.v6_config["tactical_residual"]).to(self.device)
    self.resume_checkpoint = None
    resume = os.environ.get("MODEL_A_V6_RESUME", "") == "1"
    if resume:
        raise RuntimeError("v6 experiments are fresh-run only; resume is disabled")
    if not self.train and not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Explicit Model A v6 checkpoint is required: {CHECKPOINT_PATH}")
    if not self.train:
        checkpoint = _load_checkpoint(
            CHECKPOINT_PATH, self.model_architecture, self.v6_config_sha256,
        )
        _load_network_state(self.online_net, checkpoint["online_net"])
        self.completed_rounds = int(checkpoint.get("completed_rounds", 0))
        self.logger.info("Loaded %s checkpoint from round %d.", self.v6_config["variant"], self.completed_rounds)
    else:
        self.completed_rounds = 0
        self.logger.info("Starting fresh %s training.", self.v6_config["variant"])


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
    tactical = action_tactical_features(game_state) if self.v6_config["tactical_residual"] else None
    with torch.no_grad():
        q_values = self.online_net(
            torch.as_tensor(local[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(global_features[None], dtype=torch.float32, device=self.device),
            None if tactical is None else torch.as_tensor(tactical[None], dtype=torch.float32, device=self.device),
        )[0].cpu().numpy()
    best_value = q_values[legal_indices].max()
    best_indices = legal_indices[np.isclose(q_values[legal_indices], best_value)]
    return ACTIONS[int(self.rng.choice(best_indices))]
