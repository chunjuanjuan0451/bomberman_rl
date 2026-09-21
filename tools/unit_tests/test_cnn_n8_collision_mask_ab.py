from tools.cnn_n8_collision_mask_ab import _sum_counts


def test_sum_counts_combines_all_trace_fields():
    items = [
        {"trace_counts": {"decisions": 3, "actions_changed": 1}},
        {"trace_counts": {"decisions": 4, "actions_changed": 2}},
    ]
    assert _sum_counts(items) == {"decisions": 7, "actions_changed": 3}
