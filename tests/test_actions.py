import numpy as np

from racing import config
from racing.actions import class_to_controls, controls_to_class


def test_round_trip_over_all_classes():
    classes = np.arange(config.N_ACTIONS)
    steer, throttle = class_to_controls(classes)
    assert np.array_equal(controls_to_class(steer, throttle), classes)


def test_the_action_table_has_nine_steering_and_five_throttle_levels():
    """Spec §6 (version 3.9): 45 classes, class = 5 × steering index + throttle index."""
    assert config.STEER_LEVELS.tolist() == [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    assert config.THROTTLE_LEVELS.tolist() == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert config.N_ACTIONS == 45


def test_class_layout():
    steer, throttle = class_to_controls([0, 4, 22, 34, 44])
    assert list(steer) == [-1.0, -1.0, 0.0, 0.5, 1.0]
    assert list(throttle) == [-1.0, 1.0, 0.0, 1.0, 1.0]


def test_continuous_values_snap_to_nearest_level():
    assert controls_to_class([0.8], [0.4])[0] == 5 * 7 + 3      # steer 0.75, half gas
    assert controls_to_class([-0.1], [-0.6])[0] == 5 * 4 + 1    # straight, half brake
