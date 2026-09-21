"""Checkpoint registry used for curriculum training and self-play."""

from pathlib import Path
import random


class OpponentPool:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def checkpoints(self) -> list[Path]:
        return sorted(self.directory.glob("*.pt"))

    def sample(self) -> Path | None:
        candidates = self.checkpoints()
        return random.choice(candidates) if candidates else None
