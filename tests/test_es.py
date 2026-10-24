import math

import numpy as np
import pytest

from racing import config, es, nets, paths, population
from racing.population import TrialResult


def result(fitness, success):
    fitness = np.asarray(fitness, dtype=float)
    success = np.tile(np.asarray(success, dtype=bool), (config.TRACKS_PER_GEN, 1))
    return TrialResult(fitness=fitness, success=success, laps=np.where(success, config.LAPS, 0),
                       finish_time=np.where(success, 90.0, np.nan),
                       expert_lap=np.full(config.TRACKS_PER_GEN, 20.0))


def no_success(population, seeds, level, factor, n_jobs=1, backend="loky"):
    n = len(population)
    return result(np.arange(n, dtype=float), np.zeros(n))


def first_candidate(population, candidates, factor, n_jobs=1):
    return int(candidates[0]), 0, 0.5


def started(pop_size=20, **kwargs):
    method = es.ES(pop_size=pop_size, **kwargs)
    method.initialize(np.random.default_rng(0))
    return method


def test_initial_population_and_step_size():
    method = started(parents=4)
    assert method.population.weights.shape == (20, nets.n_params()) and not method.is_kept.any()
    assert method.step_size == config.ES_SIGMA0 and method.approach == "es" and method.log_fields() == {}


def test_seeded_population_starts_from_the_trained_network():
    flat = np.random.default_rng(1).normal(size=nets.n_params())
    method = started(pop_size=50, initial=flat, approach="es_seeded")
    weights = method.population.weights
    assert np.array_equal(weights[0], flat) and not np.any(np.all(weights[1:] == flat, axis=1))
    assert 0.05 < np.std(weights[1:] - flat) < 0.2 and method.approach == "es_seeded"


def test_children_are_noisy_copies_of_the_fittest_cars():
    method = started(parents=4, sigma0=config.ES_SIGMA_MIN)          # σ is never below ES_SIGMA_MIN
    before = method.population.weights.copy()
    method.advance(np.random.default_rng(2), result(-np.arange(20.0), np.zeros(20)), np.arange(10), None)
    for child in method.population.weights[10:]:                  # cars 0-3 were the fittest
        assert np.any(np.all(np.isclose(child, before[:4], atol=10 * config.ES_SIGMA_MIN), axis=1))


def test_children_remember_their_parents():
    method = started(parents=4)
    ids = method.ids.copy()
    method.advance(np.random.default_rng(8), result(-np.arange(20.0), np.zeros(20)), np.arange(10), None)
    assert np.array_equal(method.ids[:10], ids[:10]) and set(method.ids[10:]).isdisjoint(ids)
    assert set(method.parent_ids[10:]) <= set(ids[:4])


def test_learning_phase_follows_the_one_fifth_rule():
    method = started(parents=4)
    method.advance(np.random.default_rng(3), result(-np.arange(20.0), np.zeros(20)), np.arange(10), None)
    assert method.step_size == config.ES_SIGMA0                    # no child has a parent yet
    fitness = np.zeros(20)
    fitness[10:] = 1.0                                             # every child beats its own parent
    method.advance(np.random.default_rng(4), result(fitness, np.zeros(20)), np.arange(10), None)
    assert math.isclose(method.step_size, config.ES_SIGMA0 / config.ES_FACTOR)


def test_reliability_phase_follows_the_childrens_success():
    method = started(parents=4)
    success = np.arange(20) < 15                                    # every parent, 75% of the children
    method.advance(np.random.default_rng(5), result(-np.arange(20.0), success), np.arange(10), None)
    assert math.isclose(method.step_size, config.ES_SIGMA0 * config.ES_FACTOR)
    method.advance(np.random.default_rng(6), result(-np.arange(20.0), np.ones(20)), np.arange(10), None)
    assert math.isclose(method.step_size, config.ES_SIGMA0)


def test_step_size_resets_on_a_level_up():
    method = started(parents=4)
    method.advance(np.random.default_rng(7), result(-np.arange(20.0), np.ones(20)), np.arange(10), "level_up")
    assert method.step_size == config.ES_SIGMA0


def test_state_round_trip_continues_identically():
    first = started(parents=4)
    second = es.ES(pop_size=20, parents=4)
    second.load_state(first.state())
    trial = result(-np.arange(20.0), np.arange(20) < 3)
    first.advance(np.random.default_rng(9), trial, np.arange(10), None)
    second.advance(np.random.default_rng(9), trial, np.arange(10), None)
    assert np.array_equal(first.population.weights, second.population.weights)
    assert np.array_equal(first.parent_ids, second.parent_ids) and first.step_size == second.step_size


def test_run_es_seeded_starts_from_the_initial_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "evaluate_population", no_success)
    flat = nets.init_population(np.random.default_rng(1), 1)[0]
    reports = []
    summary = es.run_es(pop_size=20, max_generations=1, initial=flat, approach="es_seeded", out_dir=tmp_path,
                        on_generation=reports.append, champion_dir=None)
    assert np.array_equal(reports[0]["population"].weights[0], flat) and reports[0]["approach"] == "es_seeded"
    assert summary["approach"] == "es_seeded"


def test_run_es_returns_a_completed_run(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "evaluate_population", no_success)
    es.run_es(pop_size=20, max_generations=1, out_dir=tmp_path, champion_dir=None)
    again = es.run_es(pop_size=20, max_generations=1, out_dir=tmp_path, champion_dir=None,
                      on_generation=lambda report: pytest.fail("trained again"))
    assert again["generations"] == 1


def test_run_es_any_architecture_round_trips_its_champion(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "evaluate_population", no_success)
    monkeypatch.setattr(population, "select_final", first_candidate)
    flat = nets.init_population(np.random.default_rng(1), 1, (8, 4))[0]
    summary = es.run_es(seed=1, pop_size=20, max_generations=1, initial=flat, approach="kt_random", hidden=(8, 4),
                        activation="tanh", out_dir=tmp_path / "run", champion_dir=tmp_path / "champions")
    model, driver = nets.load_champion("kt_random", 1, tmp_path / "champions")
    saved = np.load(paths.champion_path("kt_random", 1, ".npy", tmp_path / "champions"))
    obs = np.random.default_rng(2).random((5, config.N_INPUTS))
    assert np.array_equal(driver.act(obs), nets.from_flat(saved, (8, 4), "tanh").act(obs))
    assert nets.dense_spec(model) == ((8, 4), "tanh") and summary["approach"] == "kt_random"


def test_dagger_baseline_reads_the_refined_baseline(tmp_path):
    from racing import imitation
    model = imitation.build_baseline()
    path = paths.run_dir("baseline", 1, tmp_path) / "dagger.keras"
    path.parent.mkdir(parents=True)
    model.save(path)
    flat = es.dagger_baseline(1, runs_dir=tmp_path)
    assert flat.shape == (nets.n_params(),) and np.array_equal(flat, nets.to_flat(model))


def test_dagger_baseline_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no refined baseline for run 1 at .*notebook 03"):
        es.dagger_baseline(1, runs_dir=tmp_path)


def test_random_start_setups_spread_but_a_trained_start_uses_the_default():
    spread = started(pop_size=100).setup
    assert np.all(spread[:, 0] == config.SENSOR_START)                 # every car starts with 150 m sensors
    assert np.std(np.log(spread[:, 1])) > 0.4                          # and a log-uniform steering sensitivity
    flat = np.random.default_rng(1).normal(size=nets.n_params())
    seeded = started(pop_size=50, initial=flat, approach="es_seeded").setup
    assert np.array_equal(seeded, np.tile(config.DEFAULT_SETUP, (50, 1)))


def test_children_inherit_their_parents_setup(monkeypatch):
    monkeypatch.setattr(config, "SETUP_MUTATION_P", 0.0)
    method = started(parents=4)
    before = dict(zip(method.ids.tolist(), method.setup.copy()))
    method.advance(np.random.default_rng(8), result(-np.arange(20.0), np.zeros(20)), np.arange(10), None)
    after = method.setup
    assert np.array_equal(after[:10], np.array([before[i] for i in range(10)]))     # kept cars keep their setup
    for row, parent in zip(after[10:], method.parent_ids[10:]):
        assert np.array_equal(row, before[int(parent)])


def test_setup_mutation_rate_and_size():
    method = started(pop_size=4001, parents=1, hidden=(1,))
    method.population = nets.DensePopulation(method.population.weights, (1,), "relu",
                                             setup=np.tile(config.DEFAULT_SETUP, (4001, 1)))   # far from the limits
    method.advance(np.random.default_rng(9), result(-np.arange(4001.0), np.zeros(4001)), np.array([0]), None)
    logs = np.log(method.setup[1:] / np.asarray(config.DEFAULT_SETUP))   # every child copies car 0
    changed = logs != 0
    assert abs(changed.mean() - config.SETUP_MUTATION_P) < 0.02
    assert abs(np.std(logs[changed]) - config.SETUP_MUTATION_SD) < 0.01


def test_state_round_trip_keeps_the_setups():
    first = started(parents=4)
    second = es.ES(pop_size=20, parents=4)
    second.load_state(first.state())
    trial = result(-np.arange(20.0), np.arange(20) < 3)
    first.advance(np.random.default_rng(9), trial, np.arange(10), None)
    second.advance(np.random.default_rng(9), trial, np.arange(10), None)
    assert np.array_equal(first.setup, second.setup)


def test_setup_draws_come_after_the_network_draws():
    method = started(parents=4)
    weights = method.population.weights.copy()
    fitness = -np.arange(20.0)
    method.advance(np.random.default_rng(6), result(fitness, np.zeros(20)), np.arange(10), None)
    rng = np.random.default_rng(6)                                      # the network part of revision 3, by hand
    top = np.argsort(-fitness, kind="stable")[:4]
    chosen = top[rng.integers(0, 4, 10)]
    children = weights[chosen] + rng.normal(0.0, method.step.sigma, (10, weights.shape[1]))
    assert np.array_equal(method.population.weights[10:], children)


def test_a_boost_triples_only_the_decision_layers_step():
    """Spec §9.4: in a stall the output layer's noise is STALL_BOOST × σ and the sensing layer's σ, from the same
    draws. From zero weights every child is its noise alone."""
    plain, boosted = started(parents=4), started(parents=4)
    for method in (plain, boosted):
        method.population.weights[:] = 0.0
    trial = result(-np.arange(20.0), np.zeros(20))
    plain_rng, boosted_rng = np.random.default_rng(5), np.random.default_rng(5)
    plain.advance(plain_rng, trial, np.arange(10), None)
    boosted.advance(boosted_rng, trial, np.arange(10), None, boost=True)
    assert plain_rng.bit_generator.state == boosted_rng.bit_generator.state     # the same draws
    assert plain.step_size == boosted.step_size                                 # σ itself is unchanged
    mask = nets.decision_mask(plain.hidden)
    a, b = plain.population.weights[10:], boosted.population.weights[10:]
    assert np.array_equal(b[:, ~mask], a[:, ~mask]) and np.all(a[:, ~mask] != 0.0)    # the sensing layer as usual
    assert np.array_equal(b[:, mask], config.STALL_BOOST * a[:, mask]) and np.all(a[:, mask] != 0.0)
    assert np.array_equal(boosted.setup[:, 0], plain.setup[:, 0])               # the sensor length as usual
    assert not np.array_equal(boosted.setup[:, 1], plain.setup[:, 1])           # the steering sensitivity boosted


def test_a_boost_scales_the_last_layer_of_an_earlier_revisions_network():
    """A speed push from a network that reads fewer inputs boosts its own last layer (spec §6, §9.4)."""
    flat = nets.init_population(np.random.default_rng(1), 1, (16,), n_inputs=10)[0]
    plain = started(parents=4, initial=flat, approach="es_seeded")
    boosted = started(parents=4, initial=flat, approach="es_seeded")
    trial = result(-np.arange(20.0), np.zeros(20))
    plain.advance(np.random.default_rng(6), trial, np.arange(10), None)
    boosted.advance(np.random.default_rng(6), trial, np.arange(10), None, boost=True)
    mask = nets.decision_mask((16,), n_inputs=10)
    a, b = plain.population.weights[10:], boosted.population.weights[10:]
    assert a.shape[1] == len(mask) and np.array_equal(b[:, ~mask], a[:, ~mask])
    assert not np.any(np.isclose(b[:, mask], a[:, mask]))
