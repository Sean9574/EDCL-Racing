import functools
import types

import numpy as np
import pytest
from scipy.spatial import cKDTree

from racing import config, track_gen, tracks
from racing.track_gen import (candidate_track, circuit_points, fillet_path, geometry_ok, make_track, runs_clockwise,
                               widest)


@functools.lru_cache(maxsize=None)
def built_map(seed, level):
    """make_track(seed, level) for the tests that read only its geometry: built once for all of them, without its road
    grid, which a map of Formula 1 size holds in up to about 80 MB."""
    track = make_track(seed, level)
    track.dist = np.zeros((0, 0), dtype=np.float32)
    return track


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
        assert spec["width"][0] <= np.min(t.widths) and t.max_width <= spec["width"][1]
        assert np.max(np.abs(t.curvature)) <= 1 / spec["min_corner"] + 1e-9


def test_direction_is_drawn_per_map_and_fixed_by_its_seed():
    for level in (1, 2, 3, 4, config.VALIDATION_LEVEL):
        clockwise = [runs_clockwise(seed, level) for seed in range(200)]
        assert 0.35 < np.mean(clockwise) < 0.65
        assert clockwise == [runs_clockwise(seed, level) for seed in range(200)]


@pytest.mark.parametrize("level", [1, 2, 3, 4, config.VALIDATION_LEVEL])
def test_track_turns_the_way_its_direction_says(level):
    for seed in range(6):
        t = built_map(seed, level)
        turning = float(np.sum(t.curvature) * t.spacing)          # +2π counter-clockwise, -2π clockwise
        assert np.isclose(abs(turning), 2 * np.pi, atol=0.05)
        assert (turning < 0) == runs_clockwise(seed, level)


@pytest.mark.parametrize("level", [1, 2, 3, 4, config.VALIDATION_LEVEL])
def test_maps_respect_width_and_minimum_corner_radius(level):
    spec = config.LEVELS[level]
    for seed in range(8):
        t = built_map(seed, level)
        assert np.all((t.widths >= spec["width"][0] - 1e-9) & (t.widths <= spec["width"][1] + 1e-9))
        assert np.array_equal(t.half_left, t.half_right) and t.max_width <= widest(seed, level) + 0.05
        assert np.max(np.abs(t.curvature)) <= 1 / spec["min_corner"] + 1e-9


def test_the_corner_check_can_be_left_out(circle_track):
    """min_corner None checks only that separate parts of the road stay apart (a real circuit's corners are its own)."""
    assert not geometry_ok(circle_track, 60.0) and geometry_ok(circle_track, None)


@pytest.mark.parametrize("level", [1, 2, 3, 4, config.VALIDATION_LEVEL])
def test_maps_never_overlap_themselves(level):
    for seed in range(8):
        t = built_map(seed, level)
        pairs = cKDTree(t.centerline).query_pairs(t.max_width + config.OVERLAP_MARGIN, output_type="ndarray")
        if len(pairs) == 0:
            continue
        gap = np.abs(pairs[:, 0] - pairs[:, 1]) * t.spacing
        along = np.minimum(gap, t.length - gap)
        assert np.all(along <= 3 * (t.max_width + config.OVERLAP_MARGIN))


@pytest.mark.parametrize("level", [1, 2, 3, 4, config.VALIDATION_LEVEL])
def test_map_lengths_are_reasonable(level):
    low, high = (300.0, 2500.0) if level == 4 else (1500.0, 8000.0)    # level 4: version 3.7's generated Extra hard
    for seed in range(5):
        assert low < built_map(seed, level).length < high


def test_generated_levels_climb_toward_real_circuit_length():
    """Spec §4.2: each generated level's maps are longer on average than the level's below it, up to the validation
    setting."""
    means = [np.mean([built_map(seed, level).length for seed in range(8)])
             for level in (1, 2, 3, config.VALIDATION_LEVEL)]
    assert np.all(np.diff(means) > 0.0)


def test_harder_circuits_have_more_slow_corners():
    def mean_slow(level):
        return np.mean([slow_corners(built_map(seed, level)) for seed in range(8)])
    assert mean_slow(2) < mean_slow(3) < mean_slow(config.VALIDATION_LEVEL)
    assert mean_slow(config.VALIDATION_LEVEL) >= 8


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


MAP_BUDGET = 120e6       # bytes of the largest generated map's distance map: a process keeping TRACK_CACHE_SIZE tracks
                         # holds under 1 GB of them (spec §9.10)


@pytest.mark.slow
def test_a_full_cache_of_validation_maps_fits_the_memory_budget():
    """Review focus 1: every worker process keeps up to TRACK_CACHE_SIZE built tracks. The validation setting has the
    largest generated maps, about 80 MB each in float32 (spec §9.10)."""
    sizes = [tracks.get_track(seed, config.VALIDATION_LEVEL).dist.nbytes for seed in config.VAL_SEEDS]
    assert 20e6 < max(sizes) < MAP_BUDGET


@pytest.mark.slow
def test_every_level_generates_drivable_maps():
    """Calibration (spec §4.2): 50 seeds per level and the validation setting are accepted and completed by the
    expert, and so are the held-out maps."""
    for level in (1, 2, 3, 4, config.VALIDATION_LEVEL):
        for seed in range(50):
            assert tracks.get_track(seed, level).expert_lap_time > 0
    for seed in config.VAL_SEEDS:
        assert tracks.get_track(seed, config.VALIDATION_LEVEL).expert_lap_time > 0
    for seed in config.TEST_SEEDS:
        assert tracks.get_track(seed, config.MAX_LEVEL).expert_lap_time > 0


GENERATED = (1, 2, 3, config.VALIDATION_LEVEL)


def test_the_levels_follow_table_1():
    assert config.VALIDATION_LEVEL == 5 and config.WIDTH_VARIATION == 0.2
    assert config.LEVELS[1]["width"] == (12.0, 16.0) and config.LEVELS[1]["min_corner"] == 120.0
    assert config.LEVELS[3]["main"] == 600.0 and config.LEVELS[3]["min_corner"] == 18.0
    v = config.LEVELS[config.VALIDATION_LEVEL]
    assert (v["main"], v["hairpins"], v["chicanes"], v["esses"], v["min_corner"], v["width"]) == (
        900.0, 2, (2, 3), (2, 3), 12.0, (7.5, 12.0))


@pytest.mark.parametrize("level", GENERATED)
def test_widths_vary_by_at_most_20_percent_within_the_levels_range(level):
    low, high = config.LEVELS[level]["width"]
    for seed in range(20):
        w = track_gen.map_widths(seed, level, np.linspace(0.0, 1.0, 1000, endpoint=False))
        assert np.all((w >= low) & (w <= high))
        base = np.random.default_rng([seed, level, config.WIDTH_SALT]).uniform(low, high)
        assert np.all(np.abs(w / base - 1.0) <= config.WIDTH_VARIATION + 1e-12)
        assert np.ptp(w) > 0.0 or base in (low, high)
        assert track_gen.widest(seed, level) == pytest.approx(w.max(), abs=0.05)


@pytest.mark.parametrize("level", GENERATED)
def test_maps_have_formula_1_size_and_varying_widths(level):
    lengths = {1: (2000.0, 4200.0), 2: (2500.0, 4800.0), 3: (3000.0, 6000.0),
               5: (3800.0, 7500.0)}[level]                                     # Table 1, ±20%
    for seed in (0, 1):
        track = tracks.get_track(seed, level)
        assert lengths[0] <= track.length <= lengths[1]
        assert np.max(np.abs(track.curvature)) <= 1.0 / config.LEVELS[level]["min_corner"] + 1e-9
        assert np.ptp(track.widths) > 0.0 and np.array_equal(track.half_left, track.half_right)
        assert track.max_width <= config.LEVELS[level]["width"][1] + 1e-9


def test_a_maps_widths_are_fixed_by_seed_and_level_for_every_attempt():
    first = track_gen.candidate_track(5, 2, attempt=0)
    again = track_gen.candidate_track(5, 2, attempt=0)
    assert first is None or np.array_equal(first.widths, again.widths)
    fractions = np.linspace(0.0, 1.0, 50, endpoint=False)
    assert np.array_equal(track_gen.map_widths(5, 2, fractions), track_gen.map_widths(5, 2, fractions))


def test_validation_maps_are_narrower_and_tighter_than_every_training_level():
    v = config.LEVELS[config.VALIDATION_LEVEL]
    for level in (1, 2, 3):
        low, high = config.LEVELS[level]["width"]
        assert v["width"][0] < low and v["width"][1] < high and v["min_corner"] < config.LEVELS[level]["min_corner"]
        assert v["main"] > config.LEVELS[level].get("main", 0.0)


def test_a_map_can_be_built_with_another_road_width():
    """The viewer rebuilds an earlier revision's maps: the same generator with that revision's road width."""
    seed, level = 3, 4
    attempt = next(a for a in range(30) if candidate_track(seed, level, a) is not None)
    same = candidate_track(seed, level, attempt, width=widest(seed, level))
    np.testing.assert_array_equal(same.centerline, candidate_track(seed, level, attempt).centerline)
    other = next(t for t in (candidate_track(seed, level, a, width=10.0) for a in range(30)) if t is not None)
    assert np.all(other.widths == 10.0)
