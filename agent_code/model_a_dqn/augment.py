"""Board-symmetry action transformations for Model A replay data."""

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def rotate_action(action: str, quarter_turns: int) -> str:
    """Map an action after clockwise 90-degree rotations."""
    if action not in ACTIONS[:4]:
        return action
    directions = ACTIONS[:4]
    return directions[(directions.index(action) + quarter_turns) % 4]


def flip_horizontal_action(action: str) -> str:
    return {"LEFT": "RIGHT", "RIGHT": "LEFT"}.get(action, action)
