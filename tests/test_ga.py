import json
import math
import pickle

import numpy as np
import pandas as pd
import pytest

from racing import checkpoint, config, ga, nets, paths, population
from racing.curriculum import mutation_strength
from racing.population import TrialResult

STRUCTURE = ("GA_ADD_UNIT", "GA_REMOVE_UNIT", "GA_ADD_LAYER", "GA_REMOVE_LAYER")


@pytest.fixture
def fixed_shapes(monkeypatch):
    for name in STRUCTURE:
        monkeypatch.setattr(config, name, 0.0)


class Picks:
    """A stand-in random generator whose integers() returns the given picks in turn; uniform as numpy's."""
    def __init__(self, *picks):
        self.picks, self.rng = list(picks), np.random.default_rng(0)

    def integers(self, n):
        pick = self.picks.pop(0)
        assert 0 <= pick < n
        return pick

    def uniform(self, low, high, size):
        return self.rng.uniform(low, high, size)


def random_layers(hidden, seed=0, scale=0.5):
    rng = np.random.default_rng(seed)
    return ga.layers_of(rng.normal(0.0, scale, nets.n_params(hidden)), hidden)


def outputs(layers, obs):
    flat, hidden = ga.flat_of(layers)
    return nets.forward(flat[None, :].repeat(len(obs), axis=0), obs, hidden, "relu")


OBS = np.random.default_rng(9).random((50, config.N_INPUTS)) * 2 - 0.5


def result(fitness, success):
    fitness = np.asarray(fitness, dtype=float)
    success = np.tile(np.asarray(success, dtype=bool), (config.TRACKS_PER_GEN, 1))
    return TrialResult(fitness=fitness, success=success, laps=np.where(success, config.LAPS, 0),
                       finish_time=np.where(success, 90.0, np.nan))


def no_success(population, seeds, level, factor, n_jobs=1, backend="loky"):
    n = len(population)
    return result(np.arange(n, dtype=float), np.zeros(n))


def test_tournament_prefers_fitter_individuals():
    winners = ga.tournament(np.random.default_rng(0), np.arange(100.0), 1000)
    assert winners.mean() > 75


def log_spread(values):
    """Standard deviation of each setup column on the log scale."""
    return np.std(np.log(values), axis=0)


def started(pop_size=30, **kwargs):
    method = ga.GA(pop_size=pop_size, **kwargs)
    method.initialize(np.random.default_rng(0))
    return method


def test_generation_0_wires_the_inputs_directly_to_the_outputs():
    method = started()
    direct = nets.n_params(())                                         # 12 inputs wired to the outputs: 585
    assert method.population.shapes == [()] * 30 and all(w.shape == (direct,) for w in method.population.weights)
    assert not method.is_kept.any() and math.isnan(method.step_size) and method.approach == "ga"
    assert method.log_fields() == {"mean_layers": 0.0, "max_layers": 0, "mean_nodes": 0.0, "max_nodes": 0,
                                   "mean_params": float(direct), "max_params": direct}


def test_layers_and_flat_weights_convert_both_ways():
    for hidden in ((), (3,), (5, 2)):
        flat = np.random.default_rng(1).normal(size=nets.n_params(hidden))
        again, shape = ga.flat_of(ga.layers_of(flat, hidden))
        assert np.array_equal(again, flat) and shape == hidden


def test_adding_a_unit_leaves_the_outputs_unchanged():
    layers = random_layers((4, 3))
    grown = ga.add_unit(Picks(1), layers)
    assert ga.flat_of(grown)[1] == (4, 4)
    assert np.allclose(outputs(grown, OBS), outputs(layers, OBS), rtol=0, atol=1e-12)
    assert np.all(grown[2][0][3] == 0.0) and grown[1][1][3] == 0.0           # zero outgoing weights and bias
    assert ga.flat_of(ga.add_unit(Picks(), random_layers(())))[1] == ()      # no hidden layer: unchanged


def test_adding_a_layer_behind_a_hidden_layer_leaves_the_outputs_unchanged():
    layers = random_layers((4, 3))
    for depth in (1, 2):
        grown = ga.add_layer(Picks(depth), layers)
        assert ga.flat_of(grown)[1] == ((4, 4, 3) if depth == 1 else (4, 3, 3))
        assert np.allclose(outputs(grown, OBS), outputs(layers, OBS), rtol=0, atol=1e-12)


def test_the_first_hidden_layer_copies_the_inputs():
    layers = random_layers(())
    grown = ga.add_layer(Picks(0), layers)
    assert ga.flat_of(grown)[1] == (config.N_INPUTS,) and np.array_equal(grown[0][0], np.eye(config.N_INPUTS))
    positive = np.abs(OBS)                                                    # ReLU passes non-negative inputs
    assert np.allclose(outputs(grown, positive), outputs(layers, positive), rtol=0, atol=1e-12)


def test_removing_a_unit_removes_its_weights():
    layers = random_layers((4, 3))
    smaller = ga.remove_unit(Picks(0, 2), layers)                             # layer 0, unit 2
    assert ga.flat_of(smaller)[1] == (3, 3)
    assert np.array_equal(smaller[0][0], np.delete(layers[0][0], 2, axis=1))
    assert np.array_equal(smaller[1][0], np.delete(layers[1][0], 2, axis=0))


def test_removing_the_only_unit_removes_its_layer():
    layers = random_layers((1, 3))
    assert ga.flat_of(ga.remove_unit(Picks(0), layers))[1] == (3,)


def test_removing_a_layer_joins_its_neighbours_by_their_product():
    rng = np.random.default_rng(4)
    layers = [(rng.uniform(0.1, 1.0, (config.N_INPUTS, 5)), rng.uniform(0.1, 1.0, 5)),
              (rng.uniform(0.1, 1.0, (5, 4)), rng.uniform(0.1, 1.0, 4)),
              (rng.normal(size=(4, config.N_ACTIONS)), rng.normal(size=config.N_ACTIONS))]
    folded = ga.remove_layer(Picks(1), layers)                                # every unit of layer 1 is active
    assert ga.flat_of(folded)[1] == (5,)
    assert np.allclose(outputs(folded, np.abs(OBS)), outputs(layers, np.abs(OBS)), rtol=1e-12, atol=1e-12)
    assert len(ga.remove_layer(Picks(), random_layers(()))) == 1               # no hidden layer: unchanged


def test_fold_sums_in_a_fixed_order_and_matches_the_matrix_product():
    rng = np.random.default_rng(11)
    layers = [(rng.normal(size=(7, 5)), rng.normal(size=5)), (rng.normal(size=(5, 4)), rng.normal(size=4))]
    (w, b), = ga.fold(layers, 0)
    (w1, b1), (w2, b2) = layers
    assert np.allclose(w, w1 @ w2, rtol=0, atol=1e-12) and np.allclose(b, b1 @ w2 + b2, rtol=0, atol=1e-12)
    ref_w, ref_b = np.zeros((7, 4)), np.zeros(4)
    for j in range(5):
        for i in range(7):
            ref_w[i] = ref_w[i] + w1[i, j] * w2[j]
        ref_b = ref_b + b1[j] * w2[j]
    assert np.array_equal(w, ref_w) and np.array_equal(b, ref_b + b2)


def test_crossover_between_equal_architectures_mixes_every_value():
    a, b = random_layers((4,), seed=1), random_layers((4,), seed=2)
    child = ga.crossover(np.random.default_rng(3), a, b)
    flat, hidden = ga.flat_of(child)
    fa, fb = ga.flat_of(a)[0], ga.flat_of(b)[0]
    assert hidden == (4,) and np.all((flat == fa) | (flat == fb))
    assert np.any(flat == fa) and np.any(flat == fb) and not np.any(fa == fb)


def test_crossover_keeps_the_shape_parents_architecture_and_mixes_only_shared_values():
    a, b = random_layers((6, 2), seed=1), random_layers((3,), seed=2)
    child = ga.crossover(np.random.default_rng(5), a, b)
    assert ga.flat_of(child)[1] == (6, 2)
    w0, b0 = child[0]
    assert np.all((w0[:, :3] == a[0][0][:, :3]) | (w0[:, :3] == b[0][0])) and np.any(w0[:, :3] == b[0][0])
    assert np.array_equal(w0[:, 3:], a[0][0][:, 3:]) and np.array_equal(b0[3:], a[0][1][3:])
    assert np.array_equal(child[1][0], a[1][0]) and np.array_equal(child[1][1], a[1][1])  # b has no layer 1
    assert np.array_equal(child[2][0], a[2][0])               # output weights: different depths, not shared
    assert np.all((child[2][1] == a[2][1]) | (child[2][1] == b[1][1])) and np.any(child[2][1] == b[1][1])


def test_a_child_without_mutation_holds_only_parent_values(fixed_shapes):
    mother, father = random_layers((4,), seed=1), random_layers((4,), seed=2)
    flat, hidden = ga.make_child(np.random.default_rng(6), ga.flat_of(mother), ga.flat_of(father), 0.0)
    assert hidden == (4,)
    assert np.all((flat == ga.flat_of(mother)[0]) | (flat == ga.flat_of(father)[0]))


def test_structural_mutations_follow_their_rates(monkeypatch):
    for name in STRUCTURE:
        monkeypatch.setattr(config, name, 0.0)
    monkeypatch.setattr(config, "GA_ADD_LAYER", 1.0)
    rng = np.random.default_rng(7)
    parent = ga.flat_of(random_layers((3,)))
    depths = [len(ga.make_child(rng, parent, parent, 0.01)[1]) for _ in range(20)]
    assert depths == [2] * 20
    monkeypatch.setattr(config, "GA_ADD_LAYER", 0.0)
    monkeypatch.setattr(config, "GA_ADD_UNIT", 1.0)
    assert ga.make_child(rng, parent, parent, 0.01)[1] == (4,)


def test_a_population_grows_and_logs_its_size(monkeypatch):
    monkeypatch.setattr(config, "GA_ADD_LAYER", 0.5)
    monkeypatch.setattr(config, "GA_ADD_UNIT", 0.5)
    method = started(pop_size=40)
    rng = np.random.default_rng(8)
    for _ in range(4):
        method.advance(rng, result(rng.random(40), np.zeros(40)), np.arange(4), None)
    fields = method.log_fields()
    assert fields["max_layers"] >= 1 and fields["max_nodes"] >= config.N_INPUTS
    assert fields["mean_params"] > nets.n_params(())
    assert fields["max_params"] == max(len(w) for w in method.population.weights)


def test_advance_keeps_the_survivors_first_and_unchanged():
    method = started()
    before = method.population
    kept = np.array([5, 2, 9])
    method.advance(np.random.default_rng(1), result(np.arange(30.0), np.zeros(30)), kept, None)
    assert len(method.population) == 30 and list(np.flatnonzero(method.is_kept)) == [0, 1, 2]
    for row, k in enumerate(kept):
        assert np.array_equal(method.population.weights[row], before.weights[k])
        assert method.population.shapes[row] == before.shapes[k]


def test_strength_follows_the_previous_childrens_success():
    method = started(pop_size=20, s_max=0.2)
    method.advance(np.random.default_rng(4), result(np.arange(20.0), np.arange(20) % 2 == 0), np.arange(10), None)
    assert math.isclose(method.step_size, mutation_strength(0.5, 0.2, 0.002))
    only_kept = np.arange(20) < 10
    method.advance(np.random.default_rng(5), result(np.arange(20.0), only_kept), np.arange(10), None)
    assert math.isclose(method.step_size, 0.2)


def test_strength_floor_follows_the_configured_ratio(monkeypatch):
    monkeypatch.setattr(config, "GA_S_MIN", config.GA_S_MAX / 10)
    method = started(pop_size=20, s_max=0.2)
    method.advance(np.random.default_rng(4), result(np.arange(20.0), np.ones(20, dtype=bool)), np.arange(10), None)
    assert math.isclose(method.step_size, 0.02)


def test_operators_solve_the_sphere_function(fixed_shapes):
    # s_max 0.03 = 0.05/sqrt(3): 585 genes get the per-genome mutation variance 195 genes got at 0.05
    method = started(pop_size=100, s_max=0.03)
    rng = np.random.default_rng(3)
    norms = lambda: np.array([np.sum(w ** 2) for w in method.population.weights])
    start = norms().min()
    for _ in range(160):
        fitness = -norms()
        method.advance(rng, result(fitness, np.zeros(100)), np.argsort(-fitness)[:5], None)
    assert norms().min() < start / 10


def test_state_round_trip_continues_identically(monkeypatch):
    monkeypatch.setattr(config, "GA_ADD_LAYER", 0.3)
    monkeypatch.setattr(config, "GA_ADD_UNIT", 0.3)
    first = started(pop_size=20)
    rng = np.random.default_rng(2)
    first.advance(rng, result(rng.random(20), np.zeros(20)), np.arange(4), None)    # shapes now differ
    second = ga.GA(pop_size=20)
    second.load_state(pickle.loads(pickle.dumps(first.state())))            # a checkpoint is a pickle
    trial = result(np.arange(20.0), np.arange(20) < 4)
    first.advance(np.random.default_rng(7), trial, np.arange(6), None)
    second.advance(np.random.default_rng(7), trial, np.arange(6), None)
    assert first.population.shapes == second.population.shapes
    assert len(first.population.weights) == len(second.population.weights)
    assert all(np.array_equal(a, b) for a, b in zip(first.population.weights, second.population.weights))
    assert first.step_size == second.step_size and np.array_equal(first.is_kept, second.is_kept)
    assert np.array_equal(first.setup, second.setup)


def test_population_must_hold_two_cars():
    with pytest.raises(ValueError, match="at least 2"):
        ga.GA(pop_size=1)


def test_run_ga_records_the_ga_settings_and_saves_a_champion_of_its_own_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(population, "evaluate_population", no_success)
    monkeypatch.setattr(population, "select_final",
                        lambda population, candidates, factor, n_jobs=1: (int(candidates[0]), 0, 0.5))
    summary = ga.run_ga(seed=3, pop_size=20, max_generations=2, champion_dir=tmp_path / "champions")
    assert (tmp_path / "runs" / "ga" / "seed3" / "log.csv").exists() and summary["approach"] == "ga"
    assert summary["settings"] == checkpoint.settings_record("ga")
    flat = np.load(paths.champion_path("ga", 3, ".npy", tmp_path / "champions"))
    model, driver = nets.load_champion("ga", 3, tmp_path / "champions")
    assert flat.size == driver.n_params == model.count_params()
    header = (tmp_path / "runs" / "ga" / "seed3" / "log.csv").read_text().splitlines()[0].split(",")
    assert header[-6:] == ["mean_layers", "max_layers", "mean_nodes", "max_nodes", "mean_params", "max_params"]


def test_run_ga_returns_a_completed_run(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "evaluate_population", no_success)
    first = ga.run_ga(pop_size=20, max_generations=1, out_dir=tmp_path, champion_dir=None)
    again = ga.run_ga(pop_size=20, max_generations=1, out_dir=tmp_path, champion_dir=None,
                      on_generation=lambda report: pytest.fail("trained again"))
    assert again["generations"] == first["generations"] == 1


def test_a_resumed_growing_run_replays_exactly(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GA_ADD_LAYER", 0.3)
    monkeypatch.setattr(config, "GA_ADD_UNIT", 0.3)
    monkeypatch.setattr(population, "evaluate_population", no_success)
    whole = ga.run_ga(pop_size=20, max_generations=6, out_dir=tmp_path / "whole", champion_dir=None)
    ga.run_ga(pop_size=20, max_generations=3, out_dir=tmp_path / "parts", champion_dir=None)
    (tmp_path / "parts" / "summary.json").unlink()                         # continue from the checkpoint
    parts = ga.run_ga(pop_size=20, max_generations=6, out_dir=tmp_path / "parts", champion_dir=None)
    assert parts["generations"] == whole["generations"] == 6
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "parts" / "log.csv").drop(columns="elapsed"),
                                  pd.read_csv(tmp_path / "whole" / "log.csv").drop(columns="elapsed"))
    a = checkpoint.load(tmp_path / "whole" / checkpoint.CHECKPOINT_NAME)["method"]
    b = checkpoint.load(tmp_path / "parts" / checkpoint.CHECKPOINT_NAME)["method"]
    assert a["shapes"] == b["shapes"] and len(a["networks"]) == len(b["networks"])
    assert all(np.array_equal(x, y) for x, y in zip(a["networks"], b["networks"]))


def test_a_ga_run_of_version_3_6_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "evaluate_population", no_success)
    ga.run_ga(pop_size=20, max_generations=1, out_dir=tmp_path, champion_dir=None)
    summary = checkpoint.read_record(tmp_path / "summary.json")
    summary["settings"] = checkpoint.settings_record()                    # without the GA's own settings
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(checkpoint.SettingsMismatch, match="GA_ADD_LAYER"):
        ga.run_ga(pop_size=20, max_generations=1, out_dir=tmp_path, champion_dir=None)


def test_a_boost_triples_only_the_output_layers_mutation(fixed_shapes, monkeypatch):
    """Spec §9.4: in a stall the output layer of each child's own network mutates STALL_BOOST times as strongly,
    from the same draws. From zero weights every child is its mutation alone."""
    monkeypatch.setattr(config, "GA_START_HIDDEN", (4,))          # a hidden layer, so there is a sensing layer
    plain, boosted = started(pop_size=20), started(pop_size=20)
    for method in (plain, boosted):
        for w in method.population.weights:
            w[:] = 0.0
    trial = result(np.arange(20.0), np.zeros(20))
    plain_rng, boosted_rng = np.random.default_rng(5), np.random.default_rng(5)
    plain.advance(plain_rng, trial, np.arange(4), None)
    boosted.advance(boosted_rng, trial, np.arange(4), None, boost=True)
    assert plain_rng.bit_generator.state == boosted_rng.bit_generator.state
    for a, b, hidden in zip(plain.population.weights[4:], boosted.population.weights[4:], plain.population.shapes[4:]):
        mask = nets.decision_mask(hidden)
        assert np.array_equal(b[~mask], a[~mask]) and np.array_equal(b[mask], config.STALL_BOOST * a[mask])
        assert np.any(a[~mask] != 0.0) and np.any(a[mask] != 0.0)
    assert np.array_equal(boosted.setup[:, 0], plain.setup[:, 0])
    assert not np.array_equal(boosted.setup[:, 1], plain.setup[:, 1])


def test_initial_setups_spread_within_the_limits():
    method = started(pop_size=200)
    values = method.setup
    assert values.shape == (200, 2) and values is method.population.setup
    assert np.all(values[:, 0] == config.SENSOR_START)                  # every car starts with 150 m sensors
    assert np.all((values[:, 1] >= config.STEER_GAIN_LIMITS[0]) & (values[:, 1] <= config.STEER_GAIN_LIMITS[1]))
    assert log_spread(values)[1] > 0.5           # log-uniform: about 0.63 for the sensitivity


def test_children_take_each_setup_value_from_a_parent(monkeypatch):
    monkeypatch.setattr(config, "SETUP_MUTATION_P", 0.0)
    method = started(pop_size=40)
    method.setup[:, 0] = np.geomspace(40.0, 900.0, 40)                  # evolved sensor lengths, one per car
    before = method.setup.copy()
    kept = np.array([4, 9])
    method.advance(np.random.default_rng(1), result(np.random.default_rng(2).random(40), np.zeros(40)), kept, None)
    after = method.setup
    assert np.array_equal(after[:2], before[kept])                       # kept cars keep their setup
    for column in range(2):
        assert set(after[2:, column]) <= set(before[:, column])          # every child value is a parent's value
    parents = {tuple(row) for row in before}
    assert any(tuple(row) not in parents for row in after[2:])          # crossover mixes two parents' values


def test_children_setups_mutate_at_the_configured_rate():
    method = started(pop_size=2000)
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


def test_setup_draws_come_after_the_network_draws(fixed_shapes):
    method = started(pop_size=30, s_max=0.1)
    cars = method.population
    fitness = np.random.default_rng(5).random(30)
    method.advance(np.random.default_rng(6), result(fitness, np.zeros(30)), np.array([0, 1, 2]), None)
    rng, n_new = np.random.default_rng(6), 27
    strength = mutation_strength(0.0, 0.1, 0.1 * config.GA_S_MIN / config.GA_S_MAX)
    mothers = ga.tournament(rng, fitness, n_new)
    fathers = ga.tournament(rng, fitness, n_new)
    children = [ga.make_child(rng, cars.network(m), cars.network(f), strength) for m, f in zip(mothers, fathers)]
    assert len(method.population.weights[3:]) == len(children) == 27
    assert all(np.array_equal(w, c[0]) for w, c in zip(method.population.weights[3:], children))
