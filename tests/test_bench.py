import os
import time

import numpy as np
import pytest

from racing import bench, checkpoint, config, nets, population


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


def test_generation_main_prints_the_command_for_the_fastest_count(monkeypatch, capsys, numpy_simulator):
    monkeypatch.setattr(bench, "get_track", lambda seed, level: None)
    monkeypatch.setattr(bench, "job_counts", lambda cpus=None: [1, 4])
    monkeypatch.setattr(bench, "generation_time", lambda pop, jobs, repeats=3: {1: 5.0, 4: 2.0}[jobs])
    assert bench.generation_main() == {1: 5.0, 4: 2.0}
    assert "export EC_RACING_JOBS=4" in capsys.readouterr().out


def test_compiled_generation_time_on_a_small_batch():
    from racing import fastsim
    if not fastsim.available():
        pytest.skip("no compiled simulator")
    assert bench.compiled_generation_time(8, 2, repeats=1) > 0.0


def test_generation_main_times_the_compiled_simulator_when_it_runs(monkeypatch, capsys):
    from racing import fastsim
    monkeypatch.setattr(fastsim, "available", lambda: True)
    monkeypatch.setattr(fastsim, "description", lambda: "compiled (g++ 13.3.0)")
    monkeypatch.setattr(bench, "get_track", lambda seed, level: None)
    monkeypatch.setattr(bench, "job_counts", lambda cpus=None: [1, 4])
    monkeypatch.setattr(bench, "compiled_generation_time", lambda pop, jobs, repeats=3: {1: 3.0, 4: 1.0}[jobs])
    monkeypatch.setattr(bench, "generation_time", lambda *a, **k: pytest.fail("numpy timed"))
    assert bench.generation_main() == {1: 3.0, 4: 1.0}
    out = capsys.readouterr().out
    assert "compiled (g++ 13.3.0)" in out and "thread" in out
    assert "EC_RACING_THREADS" in out and "export EC_RACING_JOBS" not in out     # training uses every processor


def test_generation_main_names_the_expert_workload_without_a_trained_population(monkeypatch, capsys, tmp_path):
    from racing import fastsim
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)                      # no runs/*/seed*/checkpoint.pkl
    monkeypatch.setattr(fastsim, "available", lambda: True)
    monkeypatch.setattr(fastsim, "description", lambda: "compiled (g++ 13.3.0)")
    monkeypatch.setattr(bench, "job_counts", lambda cpus=None: [1])
    monkeypatch.setattr(bench, "compiled_generation_time", lambda pop, jobs, repeats=3: 1.0)
    bench.generation_main()
    assert "Workload: expert cars (no trained population found; lighter than training)" in capsys.readouterr().out


def trained_checkpoint(results, method_name, n_cars, n_inputs=config.N_INPUTS, seeds=(5, 6, 7, 8)):
    """A checkpoint of n_cars dense cars in results/runs/<method_name>/seed0, with the given map seeds."""
    rng = np.random.default_rng(0)
    weights = nets.init_population(rng, n_cars, (16,), n_inputs)
    state = {"seeds": list(seeds), "method": {"weights": weights, "hidden": (16,), "activation": "relu",
                                              "setup": np.ones((n_cars, 2))}}
    path = results / "runs" / method_name / "seed0" / "checkpoint.pkl"
    checkpoint.save(path, state)
    return path


def test_the_compiled_benchmark_times_the_newest_trained_population(monkeypatch, capsys, tmp_path, circle_track):
    """With a trained dense population, the compiled benchmark runs training's time trials (spec §9.10): pop of its
    cars, repeated, over config.LAPS laps of its last 3 maps at level 4, ending at the target time."""
    from racing import fastsim, sim
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    older = trained_checkpoint(tmp_path, "ga", 3)
    os.utime(older, (time.time() - 60, time.time() - 60))
    trained_checkpoint(tmp_path, "old", 3, n_inputs=10)                 # newer, but an earlier revision's networks
    trained_checkpoint(tmp_path, "es", 4)
    os.utime(tmp_path / "runs/es/seed0/checkpoint.pkl", (time.time() - 30, time.time() - 30))
    built, calls = [], []
    monkeypatch.setattr(bench, "get_track", lambda seed, level: built.append((seed, level)) or circle_track)
    monkeypatch.setattr(fastsim, "simulate", lambda tracks, cars, limits, laps=1, grip=None, threads=1:
                        calls.append((len(tracks), cars, limits, laps, threads)))
    assert bench.compiled_generation_time(10, 3, repeats=2) >= 0.0
    assert built == [(6, config.MAX_LEVEL), (7, config.MAX_LEVEL), (8, config.MAX_LEVEL)]
    assert len(calls) == 2
    n_tracks, cars, limits, laps, threads = calls[0]
    assert (n_tracks, len(cars), laps, threads) == (3, 10, config.LAPS, 3)
    es = checkpoint.load(tmp_path / "runs/es/seed0/checkpoint.pkl")["method"]["weights"]
    assert np.array_equal(cars.weights, es[[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]])
    assert limits == [sim.target_time(circle_track, config.TARGET_FACTOR)] * 3

    monkeypatch.setattr(fastsim, "available", lambda: True)
    monkeypatch.setattr(fastsim, "description", lambda: "compiled (g++ 13.3.0)")
    monkeypatch.setattr(bench, "job_counts", lambda cpus=None: [1])
    monkeypatch.setattr(bench, "compiled_generation_time", lambda pop, jobs, repeats=3: 1.0)
    bench.generation_main()
    out = capsys.readouterr().out
    folder = tmp_path / "runs" / "es" / "seed0"
    assert f"Workload: trained population from {folder}, {config.LAPS} laps of its last 3 maps" in out

