"""Track the deployed collision-mask bomb state without learning."""

import events as e


def setup_training(self):
    self._active_bomb = False


def _update(self, action, events):
    if action == "BOMB" and e.BOMB_DROPPED in events:
        self._active_bomb = True
    if e.BOMB_EXPLODED in events:
        self._active_bomb = False


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _update(self, self_action, events)


def end_of_round(self, last_game_state, last_action, events):
    self._active_bomb = False
