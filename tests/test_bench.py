import numpy as np

from racing import bench, config, nets, population


def test_time_full_laps_runs_on_a_small_batch():
    assert bench.time_full_laps(0, 5) > 0.0


def test_generation_time_with_one_and_two_processes():
    assert bench.generation_time(8, 1, repeats=1) > 0.0
    assert bench.generation_time(8, 2, repeats=1) > 0.0


def test_generation_time_uses_the_evaluation_task_division(monkeypatch):
    monkeypatch.setattr(bench, "BACKEND", "threading")            # keeps the patched function in this process
    seen = []
    monkeypatch.setattr(bench, "time_full_laps", lambda seed, n: seen.append((seed, n)) or 0.0)
    bench.generation_time(10, 6, repeats=1)
    expected = [(seed, len(idx)) for _, seed, idx in population.split_tasks(10, bench.SEEDS, 6)]
    assert sorted(seen) == sorted(expected * 2)                    # one untimed pass, one timed repeat


def test_keras_epoch_time_on_toy_data():
    rng = np.random.default_rng(0)
    x = rng.random((600, 10)).astype(np.float32)
    data = {"X_train": x[:500], "y_train": rng.integers(0, 15, 500), "X_val": x[500:],
            "y_val": rng.integers(0, 15, 100)}
    assert bench.keras_epoch_time(data, steps_per_execution=4, jit_compile=False, epochs=1) > 0.0


def test_evaluate_population_multiprocess_consistency():
    """Multi-process evaluation must match single-process on a real level-1 map."""
    cars = nets.DensePopulation(nets.init_population(np.random.default_rng(0), 8))
    single = population.evaluate_population(cars, [0, 1], 1, config.TARGET_FACTOR)
    multi = population.evaluate_population(cars, [0, 1], 1, config.TARGET_FACTOR, n_jobs=2)
    for field in ("fitness", "success", "laps"):
        assert np.array_equal(getattr(single, field), getattr(multi, field))
