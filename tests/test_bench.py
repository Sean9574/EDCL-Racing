import numpy as np

from racing import bench, nets, population


def test_time_full_laps_runs_on_a_small_batch():
    assert bench.time_full_laps(0, 5) > 0.0


def test_estimate_with_two_processes():
    assert bench.estimate(8, 2) > 0.0


def test_evaluate_population_multiprocess_consistency():
    """Multi-process evaluation must match single-process on real level-1 track."""
    weights = nets.init_population(np.random.default_rng(0), 8)
    fitness_single, finished_single = population.evaluate_population(weights, [0], 1)
    fitness_multi, finished_multi = population.evaluate_population(weights, [0], 1, n_jobs=2)
    assert np.array_equal(fitness_single, fitness_multi)
    assert np.array_equal(finished_single, finished_multi)
