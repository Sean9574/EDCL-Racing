import math

import numpy as np
import pytest

from racing import config, ga, paths, population
from racing.curriculum import mutation_strength
from racing.population import TrialResult


def result(fitness, success):
    fitness = np.asarray(fitness, dtype=float)
    success = np.tile(np.asarray(success, dtype=bool), (config.TRACKS_PER_GEN, 1))
    return TrialResult(fitness=fitness, success=success, laps=np.where(success, config.LAPS, 0),
                       finish_time=np.where(success, 90.0, np.nan))


def no_success(population, seeds, level, factor, n_jobs=1, backend="loky"):
    n = len(population)
    return result(np.arange(n, dtype=float), np.zeros(n))


def started(pop_size=30, **kwargs):
    method = ga.GA(pop_size=pop_size, **kwargs)
    method.initialize(np.random.default_rng(0))
    return method


def test_tournament_prefers_fitter_individuals():
    winners = ga.tournament(np.random.default_rng(0), np.arange(100.0), 1000)
    assert winners.mean() > 75


def test_initial_population_has_no_kept_cars_and_no_strength():
    method = started()
    assert method.population.weights.shape == (30, 463) and not method.is_kept.any()
    assert math.isnan(method.step_size) and method.approach == "ga" and method.log_fields() == {}


def test_advance_keeps_the_survivors_first_and_unchanged():
    method = started()
    before = method.population.weights.copy()
    kept = np.array([5, 2, 9])
    method.advance(np.random.default_rng(1), result(np.arange(30.0), np.zeros(30)), kept, None)
    assert method.population.weights.shape == before.shape
    assert np.array_equal(method.population.weights[:3], before[kept])
    assert list(np.flatnonzero(method.is_kept)) == [0, 1, 2]


def test_zero_strength_children_only_recombine_parent_genes():
    method = started(pop_size=20, hidden=(2,), s_max=0.0)
    before = method.population.weights.copy()
    fitness = np.random.default_rng(3).random(20)
    method.advance(np.random.default_rng(2), result(fitness, np.zeros(20)), np.array([0, 1]), None)
    children = method.population.weights[2:]
    for j in range(before.shape[1]):
        assert set(children[:, j]).issubset(set(before[:, j]))


def test_strength_follows_the_previous_childrens_success():
    method = started(pop_size=20, s_max=0.2)                 # generation 0: every car is a child
    method.advance(np.random.default_rng(4), result(np.arange(20.0), np.arange(20) % 2 == 0), np.arange(10), None)
    assert math.isclose(method.step_size, mutation_strength(0.5, 0.2, 0.002))
    only_kept = np.arange(20) < 10                           # the kept cars succeed, no child does
    method.advance(np.random.default_rng(5), result(np.arange(20.0), only_kept), np.arange(10), None)
    assert math.isclose(method.step_size, 0.2)


def test_operators_solve_the_sphere_function():
    method = started(pop_size=100, hidden=(1,), s_max=0.05)
    rng = np.random.default_rng(3)
    start = np.min(np.sum(method.population.weights ** 2, axis=1))
    for _ in range(80):
        fitness = -np.sum(method.population.weights ** 2, axis=1)
        method.advance(rng, result(fitness, np.zeros(100)), np.argsort(-fitness)[:5], None)
    assert np.min(np.sum(method.population.weights ** 2, axis=1)) < start / 10


def test_state_round_trip_continues_identically():
    first = started(pop_size=20)
    second = ga.GA(pop_size=20)
    second.load_state(first.state())
    trial = result(np.arange(20.0), np.arange(20) < 4)
    first.advance(np.random.default_rng(7), trial, np.arange(6), None)
    second.advance(np.random.default_rng(7), trial, np.arange(6), None)
    assert np.array_equal(first.population.weights, second.population.weights)
    assert first.step_size == second.step_size and np.array_equal(first.is_kept, second.is_kept)


def test_population_must_hold_two_cars():
    with pytest.raises(ValueError, match="at least 2"):
        ga.GA(pop_size=1)


def test_run_ga_uses_the_seed_folder_and_saves_the_champion(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(population, "evaluate_population", no_success)
    monkeypatch.setattr(population, "select_final",
                        lambda population, candidates, factor, n_jobs=1: (int(candidates[0]), 0, 0.5))
    summary = ga.run_ga(seed=3, pop_size=20, max_generations=2, champion_dir=tmp_path / "champions")
    assert (tmp_path / "runs" / "ga" / "seed3" / "log.csv").exists() and summary["approach"] == "ga"
    assert summary["champion_path"] == str(paths.champion_path("ga", 3, champion_dir=tmp_path / "champions"))
    assert np.load(paths.champion_path("ga", 3, ".npy", tmp_path / "champions")).shape == (463,)


def test_run_ga_returns_a_completed_run(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "evaluate_population", no_success)
    first = ga.run_ga(pop_size=20, max_generations=1, out_dir=tmp_path, champion_dir=None)
    again = ga.run_ga(pop_size=20, max_generations=1, out_dir=tmp_path, champion_dir=None,
                      on_generation=lambda report: pytest.fail("trained again"))
    assert again["generations"] == first["generations"] == 1


def test_strength_floor_follows_the_configured_ratio(monkeypatch):
    monkeypatch.setattr(config, "GA_S_MIN", config.GA_S_MAX / 10)
    method = started(pop_size=20, s_max=0.2)
    method.advance(np.random.default_rng(4), result(np.arange(20.0), np.ones(20, dtype=bool)), np.arange(10), None)
    assert math.isclose(method.step_size, 0.02)               # every child succeeded: the floor, s_max / 10


def log_spread(values):
    """Standard deviation of each setup column on the log scale."""
    return np.std(np.log(values), axis=0)


def test_initial_setups_spread_within_the_limits():
    method = started(pop_size=200, hidden=(1,))
    values = method.setup
    assert values.shape == (200, 2) and values is method.population.setup
    assert np.all((values[:, 0] >= config.SENSOR_LENGTH_LIMITS[0]) & (values[:, 0] <= config.SENSOR_LENGTH_LIMITS[1]))
    assert np.all((values[:, 1] >= config.STEER_GAIN_LIMITS[0]) & (values[:, 1] <= config.STEER_GAIN_LIMITS[1]))
    assert log_spread(values).min() > 0.5        # log-uniform: about 0.78 for the length, 0.63 for the sensitivity


def test_children_take_each_setup_value_from_a_parent(monkeypatch):
    monkeypatch.setattr(config, "SETUP_MUTATION_P", 0.0)
    method = started(pop_size=40, hidden=(1,))
    before = method.setup.copy()
    kept = np.array([4, 9])
    method.advance(np.random.default_rng(1), result(np.random.default_rng(2).random(40), np.zeros(40)), kept, None)
    after = method.setup
    assert np.array_equal(after[:2], before[kept])                       # kept cars keep their setup
    for column in range(2):
        assert set(after[2:, column]) <= set(before[:, column])          # every child value is a parent's value
    parents = {tuple(row) for row in before}
    assert any(tuple(row) not in parents for row in after[2:])          # crossover mixes two parents' values


def test_setup_draws_come_after_the_network_draws():
    method = started(pop_size=30, hidden=(2,), s_max=0.1)
    weights = method.population.weights.copy()
    fitness = np.random.default_rng(5).random(30)
    trial = result(fitness, np.zeros(30))
    method.advance(np.random.default_rng(6), trial, np.array([0, 1, 2]), None)
    rng, n_new, p = np.random.default_rng(6), 27, weights.shape[1]      # the network part of revision 3, by hand
    strength = mutation_strength(0.0, 0.1, 0.1 * config.GA_S_MIN / config.GA_S_MAX)
    mothers = ga.tournament(rng, fitness, n_new)
    fathers = ga.tournament(rng, fitness, n_new)
    children = np.where(rng.random((n_new, p)) < config.GA_CROSSOVER_P, weights[mothers], weights[fathers])
    mutate = rng.random((n_new, p)) < config.GA_MUTATION_P
    children = children + mutate * rng.normal(0.0, strength, (n_new, p))
    assert np.array_equal(method.population.weights[3:], children)


def test_children_setups_mutate_at_the_configured_rate():
    method = started(pop_size=2000, hidden=(1,))
    before = method.setup.copy()
    method.advance(np.random.default_rng(4), result(np.random.default_rng(3).random(2000), np.zeros(2000)),
                   np.array([0]), None)
    columns = [set(before[:, 0]), set(before[:, 1])]
    changed = np.array([[value not in columns[c] for c, value in enumerate(row)] for row in method.setup[1:]])
    assert abs(changed.mean() - config.SETUP_MUTATION_P) < 0.03


def test_state_round_trip_keeps_the_setups():
    first = started(pop_size=20)
    second = ga.GA(pop_size=20)
    second.load_state(first.state())
    assert np.array_equal(second.setup, first.setup)
    trial = result(np.arange(20.0), np.arange(20) < 4)
    first.advance(np.random.default_rng(7), trial, np.arange(6), None)
    second.advance(np.random.default_rng(7), trial, np.arange(6), None)
    assert np.array_equal(first.setup, second.setup)
