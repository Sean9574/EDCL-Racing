import numpy as np

from racing import config
from racing.actions import class_to_controls, controls_to_class


def test_round_trip_over_all_classes():
    classes = np.arange(config.N_ACTIONS)
    steer, throttle = class_to_controls(classes)
    assert np.array_equal(controls_to_class(steer, throttle), classes)


def test_class_layout():
    steer, throttle = class_to_controls([0, 2, 14])
    assert list(steer) == [-1.0, -1.0, 1.0]
    assert list(throttle) == [-1.0, 1.0, 1.0]


def test_continuous_values_snap_to_nearest_level():
    assert controls_to_class([0.8], [0.4])[0] == 3 * 4 + 1      # steer 1.0, coast
