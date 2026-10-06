import math

import numpy as np
import pytest

from racing import config, ga, population
from racing.curriculum import mutation_strength


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)


def test_tournament_prefers_fitter_individuals():
    winners = ga.tournament(np.random.default_rng(0), np.arange(100.0), 1000)
    assert winners.mean() > 75


def test_next_generation_keeps_elite_first_and_unchanged():
    rng = np.random.default_rng(1)
    weights = rng.normal(size=(30, 8))
    fitness = rng.random(30)
    new, is_kept = ga.next_generation(rng, weights, fitness, 5, 0.1)
    top = weights[np.argsort(-fitness, kind="stable")[:5]]
    assert new.shape == weights.shape and is_kept.sum() == 5 and np.all(is_kept[:5])
    assert np.array_equal(new[:5], top)


def test_zero_strength_children_only_recombine_parent_genes():
    rng = np.random.default_rng(2)
    weights = rng.normal(size=(20, 6))
    new, _ = ga.next_generation(rng, weights, rng.random(20), 2, 0.0)
    for j in range(6):
        assert set(new[2:, j]).issubset(set(weights[:, j]))


def test_operators_solve_the_sphere_function():
    rng = np.random.default_rng(3)
    weights = rng.normal(size=(100, 20))
    start = np.min(np.sum(weights ** 2, axis=1))
    for _ in range(80):
        fitness = -np.sum(weights ** 2, axis=1)
        weights, _ = ga.next_generation(rng, weights, fitness, 5, 0.05)
    assert np.min(np.sum(weights ** 2, axis=1)) < start / 10


def test_population_must_exceed_kept(tmp_path):
    with pytest.raises(ValueError, match="larger than"):
        ga.run_ga(pop_size=40, kept=50, budget=1000, out_dir=tmp_path, champion_dir=None)


def test_budget_must_cover_one_generation(tmp_path):
    with pytest.raises(ValueError, match="smaller than one generation"):
        ga.run_ga(pop_size=20, kept=5, budget=10, out_dir=tmp_path, champion_dir=None)


def test_small_run_reports_every_generation(fake_tracks, tmp_path):
    reports = []
    result = ga.run_ga(pop_size=20, kept=5, budget=60, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert result["generations"] == len(reports) == 3 and result["evals"] == 60
    assert reports[0]["step_size"] == config.GA_S_MAX
    assert reports[1]["is_kept"].sum() == 5
    assert math.isclose(reports[1]["step_size"], mutation_strength(reports[0]["new_finish"]))
    assert (tmp_path / "log.csv").exists() and (tmp_path / "summary.json").exists()


def test_mutation_strength_follows_new_finish_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.arange(len(w), dtype=float), np.arange(len(w)) % 2 == 0))
    reports = []
    ga.run_ga(pop_size=20, kept=5, budget=40, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert math.isclose(reports[1]["step_size"], mutation_strength(0.5))


def test_level_advances_after_three_qualifying_generations(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.ones(len(w)), np.ones(len(w), dtype=bool)))
    reports = []
    ga.run_ga(pop_size=20, kept=5, budget=100, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert [r["level"] for r in reports] == [1, 1, 1, 1, 2]       # generation 0 has no kept individuals
    assert reports[3]["leveled_up"]
