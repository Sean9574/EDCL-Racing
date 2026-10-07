import math

import numpy as np
import pandas as pd
import pytest

from racing import config, es, nets, paths, population


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
        sigma = es.one_fifth_rule(sigma, es.success_fraction(fitness[10:], fitness[:10][parent_of]))
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
        return fitness, np.zeros(len(w), dtype=bool), np.zeros(len(w), dtype=bool)
    monkeypatch.setattr(es, "evaluate_population", fake)
    reports = []
    es.run_es(mu=4, lam=16, budget=40, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert math.isclose(reports[1]["step_size"], config.ES_SIGMA0 / config.ES_FACTOR)


def test_success_fraction_counts_ties_as_half():
    child = np.array([2.0, 1.0, 1.0, 0.0])
    parent = np.array([1.0, 1.0, 0.0, 1.0])
    assert es.success_fraction(child, parent) == (1.0 + 0.5 + 1.0 + 0.0) / 4


def test_full_tie_generation_increases_sigma(monkeypatch, tmp_path):
    monkeypatch.setattr(es, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.zeros(len(w)), np.zeros(len(w), dtype=bool),
                                                           np.zeros(len(w), dtype=bool)))
    reports = []
    es.run_es(mu=4, lam=16, budget=40, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert math.isclose(reports[1]["step_size"], config.ES_SIGMA0 / config.ES_FACTOR)


def test_sigma_resets_after_a_level_up(monkeypatch, tmp_path):
    def fake(w, seeds, level, n_jobs=1):
        fitness = np.zeros(len(w))
        fitness[4:] = 1.0                                   # every offspring beats its parent
        return fitness, np.ones(len(w), dtype=bool), np.ones(len(w), dtype=bool)   # all finish at pace
    monkeypatch.setattr(es, "evaluate_population", fake)
    reports = []
    es.run_es(mu=4, lam=16, budget=100, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert [r["leveled_up"] for r in reports] == [False, False, True, False, False]
    assert reports[2]["step_size"] > config.ES_SIGMA0
    assert reports[3]["step_size"] == config.ES_SIGMA0 and reports[3]["level"] == 2
    assert math.isclose(reports[4]["step_size"], config.ES_SIGMA0 / config.ES_FACTOR)


def test_seeded_parents_start_from_the_trained_network():
    rng = np.random.default_rng(0)
    flat = rng.normal(size=431)
    parents = es.seeded_parents(rng, flat, 50)
    assert parents.shape == (50, 431) and np.array_equal(parents[0], flat)
    assert not np.any(np.all(parents[1:] == flat, axis=1))
    assert 0.05 < np.std(parents[1:] - flat) < 0.2                 # σ₀ = 0.1


def test_seeded_run_starts_from_the_initial_weights(fake_tracks, tmp_path):
    flat = nets.init_population(np.random.default_rng(1), 1)[0]
    reports = []
    result = es.run_es(mu=4, lam=16, budget=20, initial=flat, approach="es_seeded", out_dir=tmp_path,
                       on_generation=reports.append, champion_dir=None)
    assert np.array_equal(reports[0]["weights"][0], flat) and reports[0]["approach"] == "es_seeded"
    assert result["approach"] == "es_seeded" and result["seeded"]


def test_trained_baseline_reads_the_baseline_controller(tmp_path):
    from racing import imitation
    model = imitation.build_baseline()
    path = paths.champion_path("baseline", 1, champion_dir=tmp_path)
    path.parent.mkdir(parents=True)
    model.save(path)
    flat = es.trained_baseline(1, champion_dir=tmp_path)
    assert flat.shape == (431,) and np.array_equal(flat, nets.to_flat(model))


def test_trained_baseline_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="notebook 02"):
        es.trained_baseline(0, champion_dir=tmp_path)


def test_log_records_strict_successes_and_ties(monkeypatch, tmp_path):
    def fake(w, seeds, level, n_jobs=1):
        fitness = np.zeros(len(w))
        fitness[4:12] = 1.0                      # 8 of 16 offspring beat their parent (all parents score 0)
        return fitness, np.zeros(len(w), dtype=bool), np.zeros(len(w), dtype=bool)
    monkeypatch.setattr(es, "evaluate_population", fake)
    reports = []
    es.run_es(mu=4, lam=16, budget=20, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert reports[0]["strict_success"] == 0.5 and reports[0]["tie_fraction"] == 0.5
    log = pd.read_csv(tmp_path / "log.csv")
    assert list(log["strict_success"]) == [0.5] and list(log["tie_fraction"]) == [0.5]


def test_es_champion_per_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(es, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.zeros(len(w)), np.zeros(len(w), bool),
                                                           np.zeros(len(w), bool)))
    monkeypatch.setattr(es, "select_champion", lambda candidates, n_jobs=1: (candidates[0], 0.25))
    result = es.run_es(mu=4, lam=16, budget=20, seed=1, approach="es_seeded",
                       initial=nets.init_population(np.random.default_rng(2), 1)[0],
                       out_dir=tmp_path / "run", champion_dir=tmp_path / "champions")
    assert result["champion_path"] == str(paths.champion_path("es_seeded", 1, champion_dir=tmp_path / "champions"))
    assert paths.champion_path("es_seeded", 1, ".npy", tmp_path / "champions").exists()
