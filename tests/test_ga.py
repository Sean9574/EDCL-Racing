import json
import math

import numpy as np
import pytest

from racing import config, ga, paths, population
from racing.curriculum import mutation_strength


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)


def fake_evaluation(finished, at_pace):
    def fake(w, seeds, level, n_jobs=1):
        n = len(w)
        return np.arange(n, dtype=float), np.full(n, finished), np.full(n, at_pace)
    return fake


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
    result = ga.run_ga(pop_size=20, kept=5, budget=60, out_dir=tmp_path, on_generation=reports.append,
                       champion_dir=None)
    assert result["generations"] == len(reports) == 3 and result["evals"] == 60
    assert math.isnan(reports[0]["step_size"])                    # generation 0 is not produced by mutation
    assert reports[1]["is_kept"].sum() == 5
    assert math.isclose(reports[1]["step_size"], mutation_strength(reports[0]["new_finish"]))
    assert (tmp_path / "log.csv").exists() and (tmp_path / "summary.json").exists()


def test_mutation_strength_uses_s_max_and_the_finishing_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.arange(len(w), dtype=float),
                                                           np.arange(len(w)) % 2 == 0, np.zeros(len(w), bool)))
    reports = []
    ga.run_ga(pop_size=20, kept=5, budget=40, s_max=0.2, out_dir=tmp_path, on_generation=reports.append,
              champion_dir=None)
    assert math.isclose(reports[1]["step_size"], mutation_strength(0.5, 0.2, 0.002))


def test_level_advances_after_three_generations_at_pace(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population", fake_evaluation(True, True))
    reports = []
    ga.run_ga(pop_size=20, kept=5, budget=100, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert [r["level"] for r in reports] == [1, 1, 1, 1, 2]       # generation 0 has no kept individuals
    assert reports[3]["leveled_up"]


def test_finishing_slower_than_pace_never_advances(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population", fake_evaluation(True, False))
    reports = []
    ga.run_ga(pop_size=20, kept=5, budget=100, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert [r["level"] for r in reports] == [1] * 5
    assert reports[-1]["new_finish"] == 1.0 and reports[-1]["new_pace"] == 0.0


def test_start_level(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population", fake_evaluation(False, False))
    reports = []
    result = ga.run_ga(pop_size=20, kept=5, budget=20, start_level=3, out_dir=tmp_path,
                       on_generation=reports.append, champion_dir=None)
    assert reports[0]["level"] == 3 and result["start_level"] == 3


def test_champion_per_seed_and_stale_files_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population", fake_evaluation(False, False))
    monkeypatch.setattr(ga, "select_champion", lambda candidates, n_jobs=1: (candidates[0], 0.5))
    champions = tmp_path / "champions"
    stale = paths.champion_path("ga", 3, ".npy", champions)
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    result = ga.run_ga(pop_size=20, kept=5, budget=40, seed=3, out_dir=tmp_path / "run", champion_dir=champions)
    assert result["champion_path"] == str(paths.champion_path("ga", 3, champion_dir=champions))
    assert np.load(stale).shape == (431,)
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["seed"] == 3 and summary["wall_time"] > 0 and summary["val_fitness"] == 0.5


def test_default_output_directory_is_per_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(ga, "evaluate_population", fake_evaluation(False, False))
    ga.run_ga(pop_size=20, kept=5, budget=20, seed=2, champion_dir=None)
    assert (tmp_path / "ga" / "seed2" / "log.csv").exists()
