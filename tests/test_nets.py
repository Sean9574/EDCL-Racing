import numpy as np

from racing import config, nets


def test_baseline_parameter_count():
    assert nets.n_params() == 431


def test_init_population_shape_and_zero_biases():
    w = nets.init_population(np.random.default_rng(0), 7)
    assert w.shape == (7, 431)
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
    assert net.n_params == 431


def test_to_keras_matches_numpy():
    rng = np.random.default_rng(5)
    w = nets.init_population(rng, 1)[0] + rng.normal(0, 0.2, 431)
    model = nets.to_keras(w)
    obs = rng.random((300, config.N_INPUTS)).astype(np.float32)
    keras_classes = np.argmax(model.predict(obs, verbose=0), axis=1)
    assert model.count_params() == 431
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


def test_load_champions_empty_directory(tmp_path):
    assert nets.load_champions(tmp_path) == {}


def test_load_champions_keeps_standard_order(tmp_path):
    w = nets.init_population(np.random.default_rng(7), 1)[0]
    nets.to_keras(w).save(tmp_path / "ga.keras")
    nets.to_keras(w).save(tmp_path / "baseline.keras")
    assert list(nets.load_champions(tmp_path)) == ["baseline", "ga"]
