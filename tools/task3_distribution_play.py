"""Launch the official CLI with the isolated Task-3 training world."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as official_main  # noqa: E402
from tools.task3_distribution_world import Task3TrainingDistributionWorld  # noqa: E402


def main() -> None:
    official_main.BombeRLeWorld = Task3TrainingDistributionWorld
    official_main.main()


if __name__ == "__main__":
    main()
