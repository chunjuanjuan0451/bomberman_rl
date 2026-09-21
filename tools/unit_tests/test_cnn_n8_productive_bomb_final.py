from pathlib import Path

from tools.cnn_n8_productive_bomb_final import DEFAULT_PROTOCOL, load_protocol


def test_productive_bomb_final_protocol_is_fixed_and_authorized():
    protocol, digest = load_protocol(Path(DEFAULT_PROTOCOL))
    assert len(protocol["collection"]["cases"]) == 8
    assert protocol["collection"]["rounds_per_arm"] == 200
    assert protocol["collection"]["policy_updates"] == 0
    assert protocol["decision_rule"]["minimum_winning_blocks_out_of_8"] == 5
    assert len(digest) == 64
