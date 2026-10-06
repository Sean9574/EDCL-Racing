import math

import numpy as np
import pandas as pd
import pytest

from racing import config, nets, population


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    return circle_track


def test_evaluate_population_shapes(fake_tracks):
    w = nets.init_population(np.random.default_rng(0), 6)
    fitness, finished = population.evaluate_population(w, [0, 1], 1)
    assert fitness.shape == (6,) and finished.shape == (6,) and finished.dtype == bool
    assert np.all((fitness >= 0) & (fitness < 2))


def test_parallel_evaluation_matches_serial(fake_tracks):
    w = nets.init_population(np.random.default_rng(1), 9)
    serial = population.evaluate_population(w, [0], 1)
    threaded = population.evaluate_population(w, [0], 1, n_jobs=3, backend="threading")
    assert np.array_equal(serial[0], threaded[0]) and np.array_equal(serial[1], threaded[1])


def test_resimulating_a_subset_reproduces_logged_fitness(fake_tracks):
    w = nets.init_population(np.random.default_rng(2), 10)
    full, _ = population.evaluate_population(w, [0], 1)
    part, _ = population.evaluate_population(w[[1, 7]], [0], 1)
    assert np.array_equal(full[[1, 7]], part)


def test_finish_fractions():
    finished = np.array([True, True, False, True])
    kept = np.array([True, True, False, False])
    assert population.finish_fractions(finished, kept) == (1.0, 0.5)
    k, n = population.finish_fractions(finished, np.zeros(4, dtype=bool))
    assert math.isnan(k) and n == 0.75


def report(generation, n=5):
    rng = np.random.default_rng(generation)
    w = nets.init_population(rng, n)
    fitness = rng.random(n)
    return population.make_report(approach="ga", generation=generation, level=1, seeds=[0, 1, 2], weights=w,
                                  fitness=fitness, finished=fitness > 0.5, is_kept=np.zeros(n, dtype=bool),
                                  kept_finish=float("nan"), new_finish=0.4, step_size=0.5, evals=n * (generation + 1),
                                  elapsed=1.0, leveled_up=False)


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


def test_select_champion_picks_highest_validation_fitness(monkeypatch):
    candidates = np.arange(12.0).reshape(4, 3)
    monkeypatch.setattr(population, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.array([0.2, 1.4, 0.9, 1.1]), np.ones(4, bool)))
    best, value = population.select_champion(candidates)
    assert np.array_equal(best, candidates[1]) and value == 1.4


def test_save_champion(tmp_path):
    w = nets.init_population(np.random.default_rng(3), 1)[0]
    path = population.save_champion("ga", w, tmp_path)
    assert path == tmp_path / "ga.keras" and path.exists()
