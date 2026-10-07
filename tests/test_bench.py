import numpy as np

from racing import bench, nets, population


def test_time_full_laps_runs_on_a_small_batch():
    assert bench.time_full_laps(0, 5) > 0.0


def test_estimate_with_two_processes():
    assert bench.estimate(8, 2, repeats=1) > 0.0


def test_evaluate_population_multiprocess_consistency():
    """Multi-process evaluation must match single-process on a real level-1 track."""
    weights = nets.init_population(np.random.default_rng(0), 8)
    single = population.evaluate_population(weights, [0, 1], 1)
    multi = population.evaluate_population(weights, [0, 1], 1, n_jobs=2)
    for a, b in zip(single, multi):
        assert np.array_equal(a, b)
