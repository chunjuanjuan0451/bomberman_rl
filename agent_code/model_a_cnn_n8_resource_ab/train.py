"""Reuse the frozen n=8/collision-consistent training implementation."""

from agent_code.model_a_cnn_n8_league_train.train import (
    end_of_round,
    game_events_occurred,
    setup_training,
)

__all__ = ["end_of_round", "game_events_occurred", "setup_training"]
