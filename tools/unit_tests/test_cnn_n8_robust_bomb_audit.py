from tools.cnn_n8_robust_bomb_audit import _fraction


def test_fraction_handles_empty_denominator():
    assert _fraction(0, 0) is None
    assert _fraction(3, 4) == 0.75
