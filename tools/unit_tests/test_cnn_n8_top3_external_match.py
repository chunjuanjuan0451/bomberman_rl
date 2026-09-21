"""Contracts for the s161000 top-three direct match."""

from pathlib import Path

from tools.cnn_n8_top3_external_match import EXPECTED_CASES, LABELS, dry_run, load_protocol, registered_seeds


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-top3-external-match-s161000.json"


def test_protocol_is_small_balanced_evaluation_only_match():
    protocol, protocol_hash = load_protocol(PROTOCOL)
    assert tuple(protocol["labels"]) == LABELS
    assert tuple(protocol["evaluation"]["cases"]) == EXPECTED_CASES
    assert len(registered_seeds(protocol)) == 20
    assert protocol["training_allowed"] is False
    assert protocol["checkpoint_copy_allowed"] is False
    assert protocol["automatic_followup"] is False
    summary = dry_run(protocol, protocol_hash)
    assert summary["shared_games"] == 100
    assert summary["rounds_observed_per_identity"] == 100
    assert summary["formal_evaluation_started"] is False
    assert summary["training_started"] is False


def test_each_identity_occupies_every_roster_slot_once():
    protocol, _ = load_protocol(PROTOCOL)
    cases = protocol["evaluation"]["cases"]
    for label in LABELS:
        assert sorted(case["roster"].index(label) for case in cases) == [0, 1, 2, 3]


def test_two_cnn_variants_share_checkpoint_but_not_agent_code():
    protocol, _ = load_protocol(PROTOCOL)
    collision = protocol["checkpoint_inventory"]["collision-cnn"]
    original = protocol["checkpoint_inventory"]["original-cnn"]
    assert collision["path"] == original["path"]
    assert collision["sha256"] == original["sha256"]
    assert collision["inference_variant"] != original["inference_variant"]
