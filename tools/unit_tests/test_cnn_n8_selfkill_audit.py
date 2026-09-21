from tools.cnn_n8_selfkill_audit import classify_case


def _row(step, action="UP", first=2, contested=False, fallback=False, known=2, events=None):
    return {
        "step": step,
        "action": action,
        "events": list(events or []),
        "bomb_safe_first_move_count": first,
        "bomb_any_first_move_contestable": contested,
        "chosen_q_gap": 0.2,
        "fallback_used": fallback,
        "known_survivable_nonbomb_actions": known,
    }


def test_classifies_contestable_single_exit_bomb():
    case = {
        "round": 3,
        "terminal_step": 14,
        "window": [
            _row(10, "BOMB", first=1, contested=True, events=["BOMB_DROPPED"]),
            _row(14, fallback=True, known=0, events=["KILLED_SELF"]),
        ],
    }
    result = classify_case(case, 6)
    assert result["category"] == "fragile_contestable_bomb"
    assert result["flags"]["placement_first_move_count"] == 1


def test_classifies_mask_predicted_safe_terminal():
    case = {
        "round": 4,
        "terminal_step": 24,
        "window": [
            _row(20, "BOMB", first=2, events=["BOMB_DROPPED"]),
            _row(24, known=2, events=["KILLED_SELF"]),
        ],
    }
    result = classify_case(case, 6)
    assert result["category"] == "mask_predicted_safe_but_died"
