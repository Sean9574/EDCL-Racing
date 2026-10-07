import json
import math

import numpy as np
import pandas as pd
import pytest

from racing import config, nets, population, sim


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    return circle_track


def test_evaluate_population_shapes(fake_tracks):
    w = nets.init_population(np.random.default_rng(0), 6)
    fitness, finished, at_pace = population.evaluate_population(w, [0, 1], 1)
    assert fitness.shape == finished.shape == at_pace.shape == (6,)
    assert finished.dtype == bool and at_pace.dtype == bool and not np.any(at_pace & ~finished)
    assert np.all((fitness >= 0) & (fitness < 2))


def test_split_tasks_covers_every_individual_on_every_track():
    tasks = population.split_tasks(10, [5, 6, 7], 6)
    assert [(k, seed) for k, seed, _ in tasks] == [(0, 5), (0, 5), (1, 6), (1, 6), (2, 7), (2, 7)]
    for k in range(3):
        idx = np.concatenate([i for position, _, i in tasks if position == k])
        assert np.array_equal(np.sort(idx), np.arange(10))


def test_split_tasks_with_one_job_gives_one_task_per_track():
    tasks = population.split_tasks(10, [5, 6, 7], 1)
    assert [(k, seed, len(i)) for k, seed, i in tasks] == [(0, 5, 10), (1, 6, 10), (2, 7, 10)]


def test_split_tasks_never_makes_empty_chunks():
    assert all(len(i) > 0 for _, _, i in population.split_tasks(2, [0], 8))


def test_parallel_evaluation_matches_serial(fake_tracks):
    w = nets.init_population(np.random.default_rng(1), 9)
    serial = population.evaluate_population(w, [0, 1, 2], 1)
    threaded = population.evaluate_population(w, [0, 1, 2], 1, n_jobs=6, backend="threading")
    for a, b in zip(serial, threaded):
        assert np.array_equal(a, b)


def worker_pids(n_jobs=2):
    """Process ids of joblib's worker processes. Each task pauses, so every worker takes one."""
    from joblib import Parallel, delayed

    def pid_after_pause():
        import os
        import time
        time.sleep(0.3)
        return os.getpid()

    return set(Parallel(n_jobs=n_jobs, batch_size=1)(delayed(pid_after_pause)() for _ in range(n_jobs)))


def test_fresh_workers_start_new_processes_and_clear_the_track_cache():
    from racing import tracks
    weights = nets.init_population(np.random.default_rng(5), 4)
    population.evaluate_population(weights, [0], 1, n_jobs=2)
    first = worker_pids()
    population.evaluate_population(weights, [0], 1, n_jobs=2)
    assert len(first) == 2 and worker_pids() == first          # without the helper, the workers are reused
    tracks.get_track(0, 1)
    population.fresh_workers(2)
    assert tracks.get_track.cache_info().currsize == 0
    population.evaluate_population(weights, [0], 1, n_jobs=2)
    second = worker_pids()
    assert len(second) == 2 and second.isdisjoint(first)


def test_fresh_workers_with_one_job_only_clears_the_track_cache(monkeypatch):
    from joblib.externals import loky
    from racing import tracks
    monkeypatch.setattr(loky, "get_reusable_executor", lambda **kwargs: pytest.fail("executor touched"))
    tracks.get_track(0, 1)
    population.fresh_workers(1)
    assert tracks.get_track.cache_info().currsize == 0


def test_resimulating_a_subset_reproduces_logged_fitness(fake_tracks):
    w = nets.init_population(np.random.default_rng(2), 10)
    full, _, _ = population.evaluate_population(w, [0], 1)
    part, _, _ = population.evaluate_population(w[[1, 7]], [0], 1)
    assert np.array_equal(full[[1, 7]], part)


def test_at_pace_requires_finishing_within_the_pace_factor(monkeypatch, fake_tracks):
    limit = config.PACE_FACTOR * fake_tracks.expert_lap_time

    def fake_simulate(track, policy, n):
        return sim.RunResult(fitness=np.array([1.5, 1.2, 0.4]), finished=np.array([True, True, False]),
                             crashed=np.array([False, False, True]), progress=np.array([1.0, 1.0, 0.4]),
                             finish_time=np.array([limit - 1.0, limit + 1.0, np.nan]), steps=10)

    monkeypatch.setattr(population.sim, "simulate", fake_simulate)
    w = nets.init_population(np.random.default_rng(0), 3)
    _, finished, at_pace = population.evaluate_population(w, [0], 1)
    assert list(finished) == [True, True, False] and list(at_pace) == [True, False, False]


def report(generation, n=5):
    rng = np.random.default_rng(generation)
    w = nets.init_population(rng, n)
    fitness = rng.random(n)
    return population.make_report(approach="ga", generation=generation, level=1, seeds=[0, 1, 2], weights=w,
                                  fitness=fitness, finished=fitness > 0.5, is_kept=np.zeros(n, dtype=bool),
                                  kept_finish=float("nan"), new_finish=0.4, kept_pace=float("nan"), new_pace=0.2,
                                  step_size=0.5, evals=n * (generation + 1), elapsed=1.0, leveled_up=False)


def test_logger_records_pace_and_es_fractions(tmp_path):
    logger = population.RunLogger(tmp_path)
    logger.log(report(0))
    log = pd.read_csv(tmp_path / "log.csv")
    assert list(log.columns) == population.RunLogger.FIELDS
    assert log.loc[0, "new_pace"] == 0.2 and math.isnan(log.loc[0, "strict_success"])


def test_logger_removes_a_stale_summary(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    population.RunLogger(tmp_path)
    assert not (tmp_path / "summary.json").exists()


def test_select_champion_picks_highest_validation_fitness(monkeypatch):
    candidates = np.arange(12.0).reshape(4, 3)
    monkeypatch.setattr(population, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.array([0.2, 1.4, 0.9, 1.1]), np.ones(4, bool),
                                                           np.ones(4, bool)))
    best, value = population.select_champion(candidates)
    assert np.array_equal(best, candidates[1]) and value == 1.4


def test_save_champion_writes_keras_and_float64_weights(tmp_path):
    w = nets.init_population(np.random.default_rng(3), 1)[0]
    path = population.save_champion("ga", 2, w, tmp_path)
    assert path == tmp_path / "ga" / "seed2.keras" and path.exists()
    saved = np.load(tmp_path / "ga" / "seed2.npy")
    assert saved.dtype == np.float64 and np.array_equal(saved, w)


def test_clear_champion_removes_both_files(tmp_path):
    population.save_champion("es", 0, nets.init_population(np.random.default_rng(4), 1)[0], tmp_path)
    population.clear_champion("es", 0, tmp_path)
    assert list((tmp_path / "es").iterdir()) == []


def test_finish_fractions():
    finished = np.array([True, True, False, True])
    kept = np.array([True, True, False, False])
    assert population.finish_fractions(finished, kept) == (1.0, 0.5)
    k, n = population.finish_fractions(finished, np.zeros(4, dtype=bool))
    assert math.isnan(k) and n == 0.75


def test_make_report_fields():
    r = report(0)
    assert r["best"] == r["fitness"].max() and r["worst"] == r["fitness"].min()
    assert r["hidden"] == list(config.BASELINE_HIDDEN) and r["activation"] == "relu"


def test_logger_stays_readable_after_each_generation(tmp_path):
    logger = population.RunLogger(tmp_path)
    for g in range(2):
        r = report(g)
        logger.log(r)
        log = pd.read_csv(tmp_path / "log.csv")
        assert len(log) == g + 1
        best = np.load(tmp_path / f"best_gen{g:04d}.npy")
        assert np.array_equal(best, r["weights"][np.argmax(r["fitness"])])
    logger.summary({"generations": 2})
    assert (tmp_path / "summary.json").exists()


def test_logger_removes_stale_best_networks(tmp_path):
    (tmp_path / "best_gen0007.npy").write_bytes(b"stale")
    (tmp_path / "notes.txt").write_text("kept")
    population.RunLogger(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["log.csv", "notes.txt"]
