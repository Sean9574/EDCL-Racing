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
    assert not t.update(1.0, 0.9)
    assert not t.update(1.0, 0.85)
    assert t.update(1.0, 0.8)
    assert t.level == 2


def test_streak_resets_after_a_non_qualifying_generation():
    t = LevelTracker()
    t.update(1.0, 0.9)
    t.update(1.0, 0.9)
    assert not t.update(0.98, 0.9)          # one kept individual failed
    assert not t.update(1.0, 0.9)
    assert not t.update(1.0, 0.9)
    assert t.update(1.0, 0.9)
    assert t.level == 2


def test_no_kept_individuals_never_qualifies():
    t = LevelTracker()
    for _ in range(5):
        assert not t.update(float("nan"), 1.0)
    assert t.level == 1


def test_level_stops_at_max():
    t = LevelTracker(level=config.MAX_LEVEL)
    for _ in range(5):
        assert not t.update(1.0, 1.0)
    assert t.level == config.MAX_LEVEL


def test_draw_seeds_are_distinct_training_seeds():
    seeds = draw_seeds(np.random.default_rng(0))
    assert len(seeds) == config.TRACKS_PER_GEN == len(set(seeds))
    assert all(s in config.TRAIN_SEEDS for s in seeds)
