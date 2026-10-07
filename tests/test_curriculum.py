import math

import numpy as np

from racing import config
from racing.curriculum import LevelTracker, draw_seeds, mutation_strength


def test_mutation_strength_endpoints_and_midpoint():
    assert math.isclose(mutation_strength(0.0), config.GA_S_MAX)
    assert math.isclose(mutation_strength(1.0), config.GA_S_MIN)
    assert math.isclose(mutation_strength(0.5), (config.GA_S_MAX + config.GA_S_MIN) / 2)


def test_level_up_on_third_consecutive_qualifying_generation():
    t = LevelTracker()
    f = config.LEVEL_UP_FRACTION
    assert not t.update(f)
    assert not t.update(min(1.0, f + 0.2))
    assert t.update(1.0)
    assert t.level == 2


def test_streak_resets_below_the_threshold():
    t = LevelTracker()
    f = config.LEVEL_UP_FRACTION
    t.update(1.0)
    t.update(1.0)
    assert not t.update(f - 0.01)
    assert not t.update(1.0)
    assert not t.update(1.0)
    assert t.update(1.0)


def test_below_threshold_never_qualifies():
    t = LevelTracker()
    for _ in range(5):
        assert not t.update(config.LEVEL_UP_FRACTION - 0.01)
    assert t.level == 1


def test_level_stops_at_max():
    t = LevelTracker(level=config.MAX_LEVEL)
    for _ in range(5):
        assert not t.update(1.0)
    assert t.level == config.MAX_LEVEL


def test_draw_seeds_are_distinct_training_seeds():
    seeds = draw_seeds(np.random.default_rng(0))
    assert len(seeds) == config.TRACKS_PER_GEN == len(set(seeds))
    assert all(s in config.TRAIN_SEEDS for s in seeds)
