"""The v6 consensus policy is inference-only."""


def setup_training(self):
    raise RuntimeError("model_a_v6_consensus cannot be trained")


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    raise RuntimeError("model_a_v6_consensus cannot be trained")


def end_of_round(self, last_game_state, last_action, events):
    raise RuntimeError("model_a_v6_consensus cannot be trained")
