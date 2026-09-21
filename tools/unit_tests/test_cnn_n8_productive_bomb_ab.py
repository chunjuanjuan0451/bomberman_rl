import numpy as np

from agent_code.model_a_cnn_n8_productive_bomb_ab.callbacks import (
    BOMB_INDEX, bomb_has_productive_target, productive_bomb_mask,
)
from tools.cnn_n8_productive_bomb_ab import sum_counts


def _state():
    field = np.zeros((17, 17), dtype=np.int8)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("me", 0, True, (5, 5)), "others": [], "coins": [], "bombs": [],
        "explosion_map": np.zeros_like(field),
    }


def test_productive_bomb_detects_crate_or_opponent_in_official_blast():
    state = _state()
    assert not bomb_has_productive_target(state)
    state["field"][8, 5] = 1
    assert bomb_has_productive_target(state)
    state["field"][8, 5] = 0
    state["others"] = [("other", 0, True, (5, 2))]
    assert bomb_has_productive_target(state)


def test_productive_bomb_respects_stone_blocking():
    state = _state()
    state["field"][6, 5] = -1
    state["field"][7, 5] = 1
    assert not bomb_has_productive_target(state)


def test_treatment_removes_only_unproductive_bomb():
    state = _state()
    base = np.ones(6, dtype=bool)
    result, diagnostic = productive_bomb_mask(state, base)
    assert not result[BOMB_INDEX]
    assert result[:BOMB_INDEX].all()
    assert diagnostic["veto_condition"] and not diagnostic["fallback"]


def test_treatment_falls_back_if_bomb_is_only_action():
    state = _state()
    base = np.zeros(6, dtype=bool)
    base[BOMB_INDEX] = True
    result, diagnostic = productive_bomb_mask(state, base)
    assert result[BOMB_INDEX]
    assert diagnostic["fallback"]


def test_sum_counts_combines_trace_fields():
    rows = [
        {"trace_counts": {"decisions": 3, "actions_changed": 1}},
        {"trace_counts": {"decisions": 4, "actions_changed": 2}},
    ]
    assert sum_counts(rows) == {"decisions": 7, "actions_changed": 3}
