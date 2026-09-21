from pathlib import Path

import numpy as np

from agent_code.model_a_v6_consensus.callbacks import select_consensus_action
from agent_code.model_a_v6_consensus.config import architecture_name, load_config
from tools.build_v6_consensus import build

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/configs/v6s4a-action-majority.json"


def test_consensus_config_and_bundle_pin_all_members():
    config, _, config_sha256 = load_config(CONFIG)
    bundle = build(CONFIG)
    assert config["fallback"] == "frozen_v4"
    assert bundle["architecture"] == architecture_name()
    assert bundle["config_sha256"] == config_sha256
    assert bundle["member_sha256"] == [member["sha256"] for member in config["members"]]
    assert len(bundle["residual_states"]) == 3


def test_two_of_three_members_override_base():
    base = np.array([5.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    members = np.array([
        [4.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        [4.0, 6.0, 0.0, 0.0, 0.0, 0.0],
        [6.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    ])
    action, overridden, proposals = select_consensus_action(
        base, members, np.array([0, 1]), np.random.default_rng(1),
    )
    assert action == 1 and overridden and proposals == (1, 1, 0)


def test_three_way_disagreement_falls_back_to_base():
    base = np.array([5.0, 4.0, 3.0, 0.0, 0.0, 0.0])
    members = np.array([
        [6.0, 4.0, 3.0, 0.0, 0.0, 0.0],
        [4.0, 6.0, 3.0, 0.0, 0.0, 0.0],
        [4.0, 3.0, 6.0, 0.0, 0.0, 0.0],
    ])
    action, overridden, proposals = select_consensus_action(
        base, members, np.array([0, 1, 2]), np.random.default_rng(2),
    )
    assert action == 0 and not overridden and proposals == (0, 1, 2)
