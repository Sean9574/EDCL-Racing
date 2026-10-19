"""Runs 0 and 1 (spec §3.3): run 0 starts every random generator from its seed and replays exactly; run 1 sets no
seed anywhere, so its random choices come from the operating system's randomness and two runs 1 differ. A stopped
run 1 still resumes on its own path, because its checkpoints hold the generators' state."""
import secrets

import numpy as np

from racing import config


def seeded(run):
    """Whether `run` starts its generators from its seed: only config.SEEDED_RUN does."""
    return run == config.SEEDED_RUN


def generator(run, entropy):
    """np.random.default_rng(entropy) for the seeded run, exactly as before; for any other run a generator started
    from the operating system's randomness."""
    return np.random.default_rng(entropy if seeded(run) else None)


def seed_keras(run, value):
    """keras.utils.set_random_seed(value) for the seeded run; for any other run the same call with a number from the
    operating system, so a process that trained the seeded run before does not carry its state into this one.
    Nothing is recorded."""
    from racing.backend import keras_cpu
    keras_cpu().utils.set_random_seed(value if seeded(run) else secrets.randbits(32))
