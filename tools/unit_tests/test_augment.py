from agent_code.model_a_dqn.augment import flip_horizontal_action, rotate_action


def test_clockwise_rotation_maps_right_to_down():
    assert rotate_action("RIGHT", 1) == "DOWN"


def test_horizontal_flip_maps_left_to_right():
    assert flip_horizontal_action("LEFT") == "RIGHT"
