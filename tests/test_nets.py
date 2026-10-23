import json
import numpy as np
import pytest

from racing import config, nets, paths, population


def test_baseline_parameter_count():
    assert nets.n_params() == 973


def test_init_population_shape_and_zero_biases():
    w = nets.init_population(np.random.default_rng(0), 7)
    assert w.shape == (7, nets.n_params())
    for _, bias in nets.unflatten(w):
        assert np.all(bias == 0.0)


def test_forward_matches_manual_computation():
    rng = np.random.default_rng(1)
    w = nets.init_population(rng, 1)
    w[:, :] += rng.normal(0, 0.1, w.shape)
    obs = rng.random((1, config.N_INPUTS))
    (w1, b1), (w2, b2) = [(a[0], b[0]) for a, b in nets.unflatten(w)]
    expected = np.maximum(obs @ w1 + b1, 0) @ w2 + b2
    assert np.allclose(nets.forward(w, obs), expected)


def test_each_car_uses_its_own_weights():
    rng = np.random.default_rng(2)
    w = nets.init_population(rng, 3)
    obs = np.repeat(rng.random((1, config.N_INPUTS)), 3, axis=0)
    together = nets.forward(w, obs)
    for i in range(3):
        assert np.allclose(together[i], nets.forward(w[i:i + 1], obs[i:i + 1])[0])


def test_act_returns_valid_classes():
    rng = np.random.default_rng(3)
    classes = nets.act(nets.init_population(rng, 50), rng.random((50, config.N_INPUTS)))
    assert classes.shape == (50,) and classes.min() >= 0 and classes.max() < config.N_ACTIONS


def test_from_flat_matches_population_forward():
    rng = np.random.default_rng(4)
    w = nets.init_population(rng, 1)
    obs = rng.random((20, config.N_INPUTS))
    net = nets.from_flat(w[0])
    assert np.array_equal(net.act(obs), nets.act(np.repeat(w, 20, axis=0), obs))
    assert net.n_params == nets.n_params()


def test_to_keras_matches_numpy():
    rng = np.random.default_rng(5)
    w = nets.init_population(rng, 1)[0] + rng.normal(0, 0.2, nets.n_params())
    model = nets.to_keras(w)
    obs = rng.random((300, config.N_INPUTS)).astype(np.float32)
    keras_classes = np.argmax(model.predict(obs, verbose=0), axis=1)
    assert model.count_params() == nets.n_params()
    assert np.mean(keras_classes == nets.from_flat(w).act(obs)) >= 0.99


def test_from_keras_skips_dropout_and_keeps_activations():
    from racing.backend import keras_cpu
    keras = keras_cpu()
    model = keras.Sequential([keras.Input((config.N_INPUTS,)), keras.layers.Dense(8, activation="tanh"),
                              keras.layers.Dropout(0.2), keras.layers.Dense(4, activation="relu"),
                              keras.layers.Dense(config.N_ACTIONS, activation="softmax")])
    net = nets.from_keras(model)
    obs = np.random.default_rng(6).random((200, config.N_INPUTS)).astype(np.float32)
    assert net.activations == ["tanh", "relu"]
    assert np.mean(np.argmax(model.predict(obs, verbose=0), axis=1) == net.act(obs)) >= 0.99


def test_to_flat_inverts_to_keras():
    flat = nets.init_population(np.random.default_rng(4), 1)[0]
    back = nets.to_flat(nets.to_keras(flat))
    assert back.dtype == np.float64 and np.array_equal(back, flat.astype(np.float32).astype(np.float64))


def test_champion_seeds_in_standard_order(tmp_path):
    for approach, seed in (("ga", 1), ("es_seeded", 2), ("baseline", 0), ("ga", 0)):
        (tmp_path / approach).mkdir(exist_ok=True)
        (tmp_path / approach / f"seed{seed}.keras").write_bytes(b"")
    found = nets.champion_seeds(tmp_path)
    assert list(found) == ["baseline", "ga", "es_seeded"] and found["ga"] == [0, 1]
    assert nets.champion_seeds(tmp_path / "missing") == {}


def test_champion_seeds_ignores_other_file_names(tmp_path):
    (tmp_path / "ga").mkdir()
    for name in ("seed1.keras", "seed0_backup.keras", "seed.keras", "seed2.keras.tmp", "seedx.keras"):
        (tmp_path / "ga" / name).write_bytes(b"")
    assert nets.champion_seeds(tmp_path) == {"ga": [1]}


def test_load_champion_drives_with_float64_weights(tmp_path):
    flat = nets.init_population(np.random.default_rng(5), 1)[0]
    population.save_dense_champion("es", 0, nets.DensePopulation(flat[None, :]), 0, tmp_path)
    model, driver = nets.load_champion("es", 0, tmp_path)
    assert model.count_params() == nets.n_params()
    assert np.array_equal(np.concatenate([np.concatenate([w.ravel(), b]) for w, b in driver.layers]), flat)


def test_load_champion_without_weights_file_uses_the_keras_model(tmp_path):
    model = nets.to_keras(nets.init_population(np.random.default_rng(6), 1)[0])
    (tmp_path / "baseline").mkdir()
    model.save(tmp_path / "baseline" / "seed0.keras")
    _, driver = nets.load_champion("baseline", 0, tmp_path)
    assert driver.n_params == nets.n_params()


def test_mixed_policy_drives_each_car_with_its_own_network():
    rng = np.random.default_rng(7)
    drivers = [nets.from_flat(f) for f in nets.init_population(rng, 3)]
    obs = rng.random((3, config.N_INPUTS))
    actions = nets.mixed_policy(drivers)(obs, None)
    assert list(actions) == [drivers[i].act(obs[i:i + 1])[0] for i in range(3)]


def test_dense_population_subset_policy_and_driver_agree():
    weights = nets.init_population(np.random.default_rng(7), 5, (6,))
    cars = nets.DensePopulation(weights, (6,), "tanh")
    part = cars.subset([4, 1])
    assert len(cars) == 5 and len(part) == 2 and part.hidden == (6,) and part.activation == "tanh"
    obs = np.random.default_rng(8).random((2, config.N_INPUTS))
    assert np.array_equal(part.policy()(obs, None), nets.act(weights[[4, 1]], obs, (6,), "tanh"))
    assert np.array_equal(cars.driver(4).act(obs[:1]), part.policy()(obs, None)[:1])


def test_dense_spec_reads_hidden_sizes_and_activation():
    model = nets.to_keras(nets.init_population(np.random.default_rng(9), 1, (8, 4))[0], (8, 4), "tanh")
    assert nets.dense_spec(model) == ((8, 4), "tanh")


def test_a_network_of_an_earlier_revision_reads_only_its_inputs():
    """Revision 3.1's networks took the first 10 inputs, the rays and the speed, which this revision computes the
    same way: they drive on the first columns of this revision's observations (spec §6)."""
    rng = np.random.default_rng(0)
    hidden = (16, 8)
    old = nets.init_population(rng, 3, hidden, n_inputs=10)
    assert old.shape[1] == nets.n_params(hidden, n_inputs=10)
    assert nets.inputs_of(old.shape[1], hidden) == 10 and nets.inputs_of(nets.n_params(hidden), hidden) == 12
    obs = rng.normal(size=(3, config.N_INPUTS))
    changed = obs.copy()
    changed[:, 10:] += 5.0                                       # the inputs revision 3.1 did not have
    np.testing.assert_array_equal(nets.forward(old, obs, hidden), nets.forward(old, changed, hidden))
    net = nets.from_flat(old[0], hidden)
    assert net.n_inputs == 10 and nets.DensePopulation(old, hidden).n_inputs == 10
    np.testing.assert_array_equal(net.act(obs), net.act(changed))
    assert nets.to_keras(old[0], hidden).input_shape[-1] == 10


def test_a_weight_count_that_fits_no_network_is_refused():
    import pytest
    with pytest.raises(ValueError):
        nets.inputs_of(nets.n_params() + 1)


def test_the_decision_mask_marks_the_last_layer():
    """The output layer's weights and biases, which the stall boost mutates harder (spec §9.4)."""
    last = 16 * config.N_ACTIONS + config.N_ACTIONS
    mask = nets.decision_mask((16,))
    assert mask.dtype == bool and mask.shape == (nets.n_params((16,)),) and mask.sum() == last
    assert mask[-last:].all() and not mask[:-last].any()
    assert nets.decision_mask((64, 64)).sum() == 64 * config.N_ACTIONS + config.N_ACTIONS
    fewer = nets.decision_mask((16,), n_inputs=10)                     # an earlier revision's network
    assert fewer.shape == (nets.n_params((16,), 10),) and fewer.sum() == last and fewer[-last:].all()
    layers = nets.unflatten(np.where(mask, 1.0, 0.0)[None, :], (16,))   # the mask is exactly the last layer
    assert all(np.all(part == (i == len(layers) - 1)) for i, layer in enumerate(layers) for part in layer)


def test_shaped_outputs_equal_each_cars_own_dense_pass(shaped_cars):
    cars = shaped_cars(13)
    obs = np.random.default_rng(1).random((13, config.N_INPUTS)) * 2 - 0.5
    logits = cars.logits(obs)
    for i in range(13):
        flat, hidden = cars.network(i)
        assert np.array_equal(logits[i], nets.forward(flat[None, :], obs[i:i + 1], hidden, "relu")[0])
    assert np.array_equal(cars.policy()(obs, None), np.argmax(logits, axis=1))
    assert cars.n_inputs == config.N_INPUTS and len(cars) == 13


def test_shaped_subsets_and_drivers_keep_each_cars_architecture_and_setup(shaped_cars):
    cars = shaped_cars(12)
    part = cars.subset([5, 0, 2])
    assert part.shapes == [cars.shapes[5], cars.shapes[0], cars.shapes[2]]
    assert all(np.array_equal(a, cars.weights[i]) for a, i in zip(part.weights, (5, 0, 2)))
    assert np.array_equal(part.setup, cars.setup[[5, 0, 2]])
    obs = np.random.default_rng(2).random((1, config.N_INPUTS))
    for i in range(12):
        driver = cars.driver(i)
        assert np.array_equal(driver.act(obs), np.argmax(cars.subset([i]).logits(obs), axis=1))
        assert np.array_equal(driver.setup, cars.setup[i]) and driver.n_params == nets.n_params(cars.shapes[i])


def test_a_network_without_hidden_layers_maps_the_inputs_to_the_outputs():
    flat = np.random.default_rng(3).normal(size=nets.n_params(()))
    assert flat.size == (config.N_INPUTS + 1) * config.N_ACTIONS == 585
    obs = np.random.default_rng(4).random((3, config.N_INPUTS))
    w = flat[:config.N_INPUTS * config.N_ACTIONS].reshape(config.N_INPUTS, config.N_ACTIONS)
    b = flat[-config.N_ACTIONS:]
    cars = nets.ShapedPopulation([(), (), ()], [flat] * 3)
    assert np.allclose(cars.logits(obs), obs @ w + b, rtol=0, atol=1e-12)


def test_shaped_populations_refuse_weights_that_do_not_fit_their_architecture():
    with pytest.raises(ValueError, match="hidden layers"):
        nets.ShapedPopulation([(4,)], [np.zeros(nets.n_params(()))])
    with pytest.raises(ValueError, match="architectures"):
        nets.ShapedPopulation([(), ()], [np.zeros(nets.n_params(()))])


def test_a_dense_population_names_each_cars_network():
    cars = nets.DensePopulation(np.arange(2 * nets.n_params(), dtype=float).reshape(2, nets.n_params()))
    flat, hidden = cars.network(1)
    assert np.array_equal(flat, cars.weights[1]) and hidden == config.BASELINE_HIDDEN


def old_actions_keras(path, outputs=15):
    """A Keras controller of version 3.8 (15 outputs) saved at `path`."""
    from racing.backend import keras_cpu
    keras = keras_cpu()
    model = keras.Sequential([keras.Input((config.N_INPUTS,)), keras.layers.Dense(8, activation="relu"),
                              keras.layers.Dense(outputs, activation="softmax")])
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return model


def test_an_imitation_controller_of_15_actions_is_refused(tmp_path):
    """Spec §10.3: a controller whose output count is not config.N_ACTIONS is refused, naming it and its count."""
    old_actions_keras(paths.champion_path("baseline", 2, champion_dir=tmp_path))
    with pytest.raises(ValueError, match=r"baseline run 2 was made for 15 actions, this version has 45"):
        nets.load_champion("baseline", 2, tmp_path)


def test_a_dense_evolved_controller_of_15_actions_is_refused(tmp_path):
    old_actions_keras(paths.champion_path("ga", 1, champion_dir=tmp_path))
    np.save(paths.champion_path("ga", 1, ".npy", tmp_path), np.zeros(config.N_INPUTS * 8 + 8 + 8 * 15 + 15))
    with pytest.raises(ValueError, match="ga run 1 was made for 15 actions"):
        nets.load_champion("ga", 1, tmp_path)


def test_a_dense_weights_file_of_15_actions_is_refused_even_beside_a_current_keras_file(tmp_path):
    population.save_dense_champion("ga", 1, nets.DensePopulation(nets.init_population(np.random.default_rng(0), 1)),
                                   0, tmp_path)
    hidden = config.BASELINE_HIDDEN
    flat = np.zeros(nets.n_params(hidden) - (config.N_ACTIONS - 15) * (hidden[-1] + 1))
    np.save(paths.champion_path("ga", 1, ".npy", tmp_path), flat)
    with pytest.raises(ValueError, match="ga run 1 was made for 15 actions"):
        nets.load_champion("ga", 1, tmp_path)


def test_a_neat_genome_of_15_actions_is_refused(tmp_path):
    """A 3.8 genome has 15 output nodes (ids 12-26); 3.9 would read its hidden nodes as outputs."""
    from racing import neat
    old_actions_keras(paths.champion_path("neat", 0, champion_dir=tmp_path))
    old = neat.Genome(node_ids=np.arange(config.N_INPUTS, config.N_INPUTS + 15), bias=np.zeros(15),
                      innov=np.array([0]), src=np.array([0]), dst=np.array([config.N_INPUTS]),
                      weight=np.array([1.0]), enabled=np.array([True]))
    paths.champion_path("neat", 0, ".json", tmp_path).write_text(json.dumps(old.to_dict()))
    with pytest.raises(ValueError, match="neat run 0 was made for 15 actions"):
        nets.load_champion("neat", 0, tmp_path)


def test_a_neat_genome_of_15_actions_is_refused_beside_a_current_keras_file(tmp_path):
    from racing import neat
    neat.save_neat_champion("neat", 0, neat.initial_genome(np.random.default_rng(0), neat.Innovations()), tmp_path)
    old = neat.Genome(node_ids=np.arange(config.N_INPUTS, config.N_INPUTS + 15), bias=np.zeros(15),
                      innov=np.array([0]), src=np.array([0]), dst=np.array([config.N_INPUTS]),
                      weight=np.array([1.0]), enabled=np.array([True]))
    paths.champion_path("neat", 0, ".json", tmp_path).write_text(json.dumps(old.to_dict()))
    with pytest.raises(ValueError, match="15 actions"):
        nets.load_champion("neat", 0, tmp_path)
