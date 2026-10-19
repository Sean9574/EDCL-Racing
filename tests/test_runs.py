import numpy as np

from racing import config, runs


def test_run_0_is_seeded_and_run_1_is_not():
    assert config.SEEDED_RUN == 0 and runs.seeded(0) and not runs.seeded(1)


def test_run_0_gets_exactly_the_seeded_generator_and_run_1_a_fresh_one():
    assert np.array_equal(runs.generator(0, 0).random(5), np.random.default_rng(0).random(5))
    assert np.array_equal(runs.generator(0, [0, 2, 31]).random(5), np.random.default_rng([0, 2, 31]).random(5))
    first, second = runs.generator(1, 1).random(5), runs.generator(1, 1).random(5)
    assert not np.array_equal(first, second)                       # the OS's randomness, no seed


def test_keras_is_seeded_with_the_value_for_run_0_and_from_the_os_for_run_1(monkeypatch):
    from racing.backend import keras_cpu
    seen = []
    monkeypatch.setattr(keras_cpu().utils, "set_random_seed", seen.append)
    runs.seed_keras(0, 1007)
    runs.seed_keras(1, 1007)
    runs.seed_keras(1, 1007)
    assert seen[0] == 1007 and seen[1] != seen[2] and all(0 <= s < 2 ** 32 for s in seen)
