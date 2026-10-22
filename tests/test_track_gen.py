import types

import numpy as np
import pytest
from scipy.spatial import cKDTree

from racing import config, track_gen, tracks
from racing.track_gen import (candidate_track, circuit_points, fillet_path, geometry_ok, make_track, map_width,
                               runs_clockwise)


def slow_corners(track, radius=40.0):
    """Number of separate stretches of the centerline tighter than `radius`."""
    tight = np.abs(track.curvature) > 1.0 / radius
    return int(np.sum(tight & ~np.roll(tight, 1)))


def test_same_seed_gives_same_track():
    assert np.array_equal(make_track(5, 4).centerline, make_track(5, 4).centerline)


def test_different_seeds_give_different_tracks():
    assert make_track(1, 4).length != make_track(2, 4).length


def test_easy_maps_are_gentle_loops_that_differ_by_seed():
    spec = config.LEVELS[1]
    maps = [make_track(seed, 1) for seed in range(4)]
    assert len({round(t.length, 3) for t in maps}) == 4                 # no longer one fixed circle
    for t in maps:
        assert config.WIDTH_RANGE[0] <= t.max_width <= config.WIDTH_RANGE[1]
        assert np.max(np.abs(t.curvature)) <= 1 / spec["min_corner"] + 1e-9


def test_direction_is_drawn_per_map_and_fixed_by_its_seed():
    for level in (1, 2, 3, 4):
        clockwise = [runs_clockwise(seed, level) for seed in range(200)]
        assert 0.35 < np.mean(clockwise) < 0.65
        assert clockwise == [runs_clockwise(seed, level) for seed in range(200)]


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_track_turns_the_way_its_direction_says(level):
    for seed in range(6):
        t = make_track(seed, level)
        turning = float(np.sum(t.curvature) * t.spacing)          # +2π counter-clockwise, -2π clockwise
        assert np.isclose(abs(turning), 2 * np.pi, atol=0.05)
        assert (turning < 0) == runs_clockwise(seed, level)


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_maps_respect_width_and_minimum_corner_radius(level):
    spec = config.LEVELS[level]
    for seed in range(8):
        t = make_track(seed, level)
        assert np.all(t.widths == map_width(seed, level))
        assert np.max(np.abs(t.curvature)) <= 1 / spec["min_corner"] + 1e-9


def test_the_corner_check_can_be_left_out(circle_track):
    """min_corner None checks only that separate parts of the road stay apart (a real circuit's corners are its own)."""
    assert not geometry_ok(circle_track, 60.0) and geometry_ok(circle_track, None)


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_maps_never_overlap_themselves(level):
    for seed in range(8):
        t = make_track(seed, level)
        pairs = cKDTree(t.centerline).query_pairs(t.max_width + config.OVERLAP_MARGIN, output_type="ndarray")
        if len(pairs) == 0:
            continue
        gap = np.abs(pairs[:, 0] - pairs[:, 1]) * t.spacing
        along = np.minimum(gap, t.length - gap)
        assert np.all(along <= 3 * (t.max_width + config.OVERLAP_MARGIN))


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_map_lengths_are_reasonable(level):
    for seed in range(5):
        assert 300.0 < make_track(seed, level).length < 2500.0


def test_harder_circuits_have_more_slow_corners():
    def mean_slow(level):
        return np.mean([slow_corners(make_track(seed, level)) for seed in range(8)])
    assert mean_slow(2) < mean_slow(3) < mean_slow(4)
    assert mean_slow(4) >= 8


def test_fillet_path_rounds_a_square_with_the_given_radius():
    square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    path = fillet_path(square, [20.0] * 4)
    assert np.all(np.linalg.norm(np.diff(np.vstack([path, path[:1]]), axis=0), axis=1) < 0.6)   # about 0.5 m apart
    cut = np.min(np.linalg.norm(path - np.array([100.0, 0.0]), axis=1))
    assert np.isclose(cut, 20.0 * np.sqrt(2) - 20.0, atol=0.05)   # a 20 m arc cuts a right-angle corner by r(√2 - 1)
    assert path.min() >= -1e-9 and path.max() <= 100.0 + 1e-9


def test_fillet_radius_shrinks_on_short_edges():
    thin = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]])
    path = fillet_path(thin, [100.0] * 4)                        # cannot hold 100 m arcs: at most 0.48 × 30 m
    assert np.all(np.abs(path) <= 30.0 + 1e-9) and len(path) > 50


def test_circuit_points_follow_the_spec_and_the_generator():
    spec = config.LEVELS[4]
    a = circuit_points(np.random.default_rng([3, 4, 0]), spec, 12.0)
    b = circuit_points(np.random.default_rng([3, 4, 0]), spec, 12.0)
    assert np.array_equal(a, b) and a.shape[1] == 2 and len(a) > 1000


def test_some_candidates_are_valid():
    assert any(candidate_track(0, 4, a) is not None for a in range(20))


def test_track_cache_is_bounded(monkeypatch):
    """Review focus 1: fresh maps every generation must not accumulate in the cache."""
    monkeypatch.setattr(track_gen, "candidate_track",
                        lambda seed, level, attempt=0, width=None: types.SimpleNamespace(seed=seed,
                                                                                         expert_lap_time=float("nan")))
    from racing import expert
    monkeypatch.setattr(expert, "expert_lap_time", lambda track: 10.0)
    tracks.get_track.cache_clear()
    try:
        for seed in range(10**9, 10**9 + config.TRACK_CACHE_SIZE + 10):
            tracks.get_track(seed, 1)
        info = tracks.get_track.cache_info()
        assert info.maxsize == config.TRACK_CACHE_SIZE and info.currsize == config.TRACK_CACHE_SIZE
    finally:
        tracks.get_track.cache_clear()                        # no fake tracks for later tests


CACHE_BUDGET = 100e6     # bytes of cached tracks per process: 24 workers and the main process stay under 2.5 GB


def test_a_full_cache_of_extra_hard_tracks_fits_the_memory_budget():
    """Review focus 1: an Extra hard track holds a road grid of over 10 MB, and every worker process has its own
    cache, so a full cache must stay small (WSL 2 has about half the machine's memory)."""
    track = next(t for t in (candidate_track(0, 4, a) for a in range(20)) if t is not None)
    size = sum(v.nbytes for v in vars(track).values() if isinstance(v, np.ndarray))
    assert size > 5e6 and config.TRACK_CACHE_SIZE * size <= CACHE_BUDGET


@pytest.mark.slow
def test_every_level_generates_drivable_maps():
    """Calibration (spec §4.2): 50 seeds per level are accepted and completed by the expert."""
    for level in (1, 2, 3, 4):
        for seed in range(50):
            assert tracks.get_track(seed, level).expert_lap_time > 0
    for seed in list(config.VAL_SEEDS) + list(config.TEST_SEEDS):
        assert tracks.get_track(seed, 4).expert_lap_time > 0


def test_each_map_has_its_own_width_in_the_range():
    widths = [make_track(seed, level).max_width for level in (1, 4) for seed in range(8)]
    low, high = config.WIDTH_RANGE
    assert all(low <= w <= high for w in widths) and len(set(widths)) == len(widths)


def test_a_maps_width_is_fixed_by_seed_and_level_for_every_attempt():
    assert map_width(5, 3) == map_width(5, 3) != map_width(6, 3)
    widths = {t.max_width for t in (candidate_track(5, 3, a) for a in range(30)) if t is not None}
    assert widths == {map_width(5, 3)}


def test_features_are_sized_by_the_given_width():
    spec = config.LEVELS[4]
    narrow = circuit_points(np.random.default_rng([3, 4, 0]), spec, 10.0)
    wide = circuit_points(np.random.default_rng([3, 4, 0]), spec, 16.0)
    assert not np.array_equal(narrow, wide)


def test_a_map_can_be_built_with_another_road_width():
    """The viewer rebuilds an earlier revision's maps: the same generator with that revision's road width."""
    seed, level = 3, 4
    attempt = next(a for a in range(30) if candidate_track(seed, level, a) is not None)
    same = candidate_track(seed, level, attempt, width=map_width(seed, level))
    np.testing.assert_array_equal(same.centerline, candidate_track(seed, level, attempt).centerline)
    other = next(t for t in (candidate_track(seed, level, a, width=10.0) for a in range(30)) if t is not None)
    assert np.all(other.widths == 10.0)
