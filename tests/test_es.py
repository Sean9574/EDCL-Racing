import math

import numpy as np
import pytest

from racing import config, es, population


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)


def test_one_fifth_rule():
    assert math.isclose(es.one_fifth_rule(0.1, 0.3), 0.1 / config.ES_FACTOR)
    assert math.isclose(es.one_fifth_rule(0.1, 0.1), 0.1 * config.ES_FACTOR)
    assert es.one_fifth_rule(0.1, 0.2) == 0.1
    assert es.one_fifth_rule(config.ES_SIGMA_MAX, 1.0) == config.ES_SIGMA_MAX
    assert es.one_fifth_rule(config.ES_SIGMA_MIN, 0.0) == config.ES_SIGMA_MIN


def test_offspring_come_from_their_parents():
    rng = np.random.default_rng(0)
    parents = rng.normal(size=(4, 5))
    offspring, parent_of = es.make_offspring(rng, parents, 1e-9, 12)
    assert offspring.shape == (12, 5)
    assert np.allclose(offspring, parents[parent_of], atol=1e-6)


def test_operators_solve_the_sphere_function():
    rng = np.random.default_rng(1)
    parents = rng.normal(size=(10, 20))
    sigma = 0.3
    start = np.min(np.sum(parents ** 2, axis=1))
    for _ in range(80):
        offspring, parent_of = es.make_offspring(rng, parents, sigma, 70)
        pop = np.concatenate([parents, offspring])
        fitness = -np.sum(pop ** 2, axis=1)
        sigma = es.one_fifth_rule(sigma, float(np.mean(fitness[10:] > fitness[:10][parent_of])))
        parents = es.select_parents(pop, fitness, 10)
    assert np.min(np.sum(parents ** 2, axis=1)) < start / 10


def test_budget_must_cover_one_generation(tmp_path):
    with pytest.raises(ValueError, match="smaller than one generation"):
        es.run_es(mu=4, lam=16, budget=10, out_dir=tmp_path, champion_dir=None)


def test_small_run_reports_every_generation(fake_tracks, tmp_path):
    reports = []
    result = es.run_es(mu=4, lam=16, budget=60, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert result["generations"] == len(reports) == 3 and result["evals"] == 60
    assert reports[0]["step_size"] == config.ES_SIGMA0
    assert all(r["is_kept"].sum() == 4 for r in reports)
    assert (tmp_path / "log.csv").exists()


def test_sigma_follows_the_one_fifth_rule(monkeypatch, tmp_path):
    def fake(w, seeds, level, n_jobs=1):
        fitness = np.zeros(len(w))
        fitness[4:] = 1.0                                   # every offspring beats its parent
        return fitness, np.zeros(len(w), dtype=bool)
    monkeypatch.setattr(es, "evaluate_population", fake)
    reports = []
    es.run_es(mu=4, lam=16, budget=40, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert math.isclose(reports[1]["step_size"], config.ES_SIGMA0 / config.ES_FACTOR)
