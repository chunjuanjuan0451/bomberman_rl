from tools.cnn_n8_collision_mask_final import _delta


def test_final_delta_uses_candidate_minus_comparator():
    assert _delta({"x": 5.0}, {"x": 3.0}, "x") == 2.0
