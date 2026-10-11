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
    x = rng.random((600, config.N_INPUTS)).astype(np.float32)
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


def test_job_counts_stop_at_the_number_of_logical_processors():
    assert bench.job_counts(24) == [1, 4, 6, 8, 12, 16, 20, 24]
    assert bench.job_counts(20) == [1, 4, 6, 8, 12, 16, 20]
    assert bench.job_counts(1) == [1]


def test_recommend_prefers_fewer_workers_on_a_near_tie():
    assert bench.recommend({1: 6.9, 6: 1.90, 12: 1.89, 16: 2.5}) == 6
    assert bench.recommend({1: 6.9, 6: 2.4, 12: 1.9}) == 12


def test_generation_main_prints_the_command_for_the_fastest_count(monkeypatch, capsys):
    monkeypatch.setattr(bench, "get_track", lambda seed, level: None)
    monkeypatch.setattr(bench, "job_counts", lambda cpus=None: [1, 4])
    monkeypatch.setattr(bench, "generation_time", lambda pop, jobs, repeats=3: {1: 5.0, 4: 2.0}[jobs])
    assert bench.generation_main() == {1: 5.0, 4: 2.0}
    assert "export EC_RACING_JOBS=4" in capsys.readouterr().out
