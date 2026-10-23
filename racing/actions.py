"""Action table (spec §6): the 45 output classes and the grip fractions they stand for."""
import numpy as np

from racing import config

N_THROTTLE = len(config.THROTTLE_LEVELS)


def class_to_controls(classes):
    """Steering and throttle grip fractions for an array of action classes."""
    classes = np.asarray(classes, dtype=np.int64)
    return config.STEER_LEVELS[classes // N_THROTTLE], config.THROTTLE_LEVELS[classes % N_THROTTLE]


def controls_to_class(steer, throttle):
    """Nearest action class for arrays of continuous steering and throttle fractions."""
    steer = np.atleast_1d(np.asarray(steer, dtype=float))
    throttle = np.atleast_1d(np.asarray(throttle, dtype=float))
    s = np.argmin(np.abs(steer[:, None] - config.STEER_LEVELS[None, :]), axis=1)
    t = np.argmin(np.abs(throttle[:, None] - config.THROTTLE_LEVELS[None, :]), axis=1)
    return s * N_THROTTLE + t
