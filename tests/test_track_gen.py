import numpy as np
import pytest
from scipy.spatial import cKDTree

from racing import config
from racing.track_gen import candidate_track, make_track


def test_same_seed_gives_same_track():
    assert np.array_equal(make_track(5, 4).centerline, make_track(5, 4).centerline)


def test_different_seeds_give_different_tracks():
    assert make_track(1, 4).length != make_track(2, 4).length


def test_level1_is_the_specified_circle():
    t = make_track(0, 1)
    assert np.allclose(np.linalg.norm(t.centerline, axis=1), 50.0, atol=0.05)
    assert t.width == 16.0
    assert np.all(t.curvature > 0)                       # counter-clockwise


def test_level2_oval_dimensions():
    spec = config.LEVELS[2]
    t = make_track(3, 2)
    k = np.max(np.abs(t.curvature))
    assert t.width == spec["width"]
    assert 1 / spec["radius"][1] - 0.005 <= k <= 1 / spec["radius"][0] + 0.005


@pytest.mark.parametrize("level", [3, 4])
def test_loops_respect_minimum_corner_radius(level):
    for seed in range(10):
        t = make_track(seed, level)
        assert np.max(np.abs(t.curvature)) <= 1 / config.LEVELS[level]["min_corner"] + 1e-9


@pytest.mark.parametrize("level", [3, 4])
def test_loops_never_overlap_themselves(level):
    for seed in range(10):
        t = make_track(seed, level)
        pairs = cKDTree(t.centerline).query_pairs(t.width + config.OVERLAP_MARGIN, output_type="ndarray")
        if len(pairs) == 0:
            continue
        gap = np.abs(pairs[:, 0] - pairs[:, 1]) * t.spacing
        along = np.minimum(gap, t.length - gap)
        assert np.all(along <= 3 * (t.width + config.OVERLAP_MARGIN))


@pytest.mark.parametrize("level", [3, 4])
def test_loop_lengths_are_reasonable(level):
    for seed in range(5):
        assert 300.0 < make_track(seed, level).length < 2500.0


def test_some_candidates_are_valid():
    assert any(candidate_track(0, 4, a) is not None for a in range(20))


@pytest.mark.slow
def test_every_seed_generates():
    for level in (2, 3, 4):
        for seed in config.TRAIN_SEEDS:
            make_track(seed, level)
    for seed in list(config.VAL_SEEDS) + list(config.TEST_SEEDS):
        make_track(seed, 4)
