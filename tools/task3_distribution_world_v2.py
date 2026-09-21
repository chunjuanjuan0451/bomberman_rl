"""Corrected Task-3 duel distribution: stop a kill-rich round on elimination."""

from __future__ import annotations

import os

from tools.task3_distribution_world import Task3TrainingDistributionWorld


class Task3ResolvedDuelWorld(Task3TrainingDistributionWorld):
    """Remove post-combat solo tails from the training-only distribution."""

    def time_to_stop(self):
        if (os.environ.get("TASK3_TRAINING_DISTRIBUTION") == "kill-rich"
                and len(self.active_agents) <= 1):
            self.logger.info("Kill-rich duel resolved, wrap up round")
            return True
        return super().time_to_stop()
