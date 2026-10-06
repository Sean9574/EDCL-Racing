"""Controller networks (spec §6): flat weight vectors for populations, a batched numpy forward pass,
and conversion to and from Keras models."""
from pathlib import Path

import numpy as np

from racing import config
from racing.backend import keras_cpu


def layer_sizes(hidden=config.BASELINE_HIDDEN):
    return [config.N_INPUTS, *hidden, config.N_ACTIONS]


def n_params(hidden=config.BASELINE_HIDDEN):
    sizes = layer_sizes(hidden)
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def init_population(rng, n, hidden=config.BASELINE_HIDDEN):
    """Glorot-uniform weights and zero biases for n networks, flattened to shape (n, n_params)."""
    parts = []
    sizes = layer_sizes(hidden)
    for a, b in zip(sizes[:-1], sizes[1:]):
        limit = np.sqrt(6.0 / (a + b))
        parts.append(rng.uniform(-limit, limit, (n, a * b)))
        parts.append(np.zeros((n, b)))
    return np.concatenate(parts, axis=1)


def unflatten(flat, hidden=config.BASELINE_HIDDEN):
    """Split (n, P) flat weights into [(W (n, a, b), bias (n, b))] per layer."""
    n = flat.shape[0]
    layers, pos = [], 0
    sizes = layer_sizes(hidden)
    for a, b in zip(sizes[:-1], sizes[1:]):
        weights = flat[:, pos:pos + a * b].reshape(n, a, b)
        pos += a * b
        bias = flat[:, pos:pos + b]
        pos += b
        layers.append((weights, bias))
    if pos != flat.shape[1]:
        raise ValueError(f"expected {pos} weights per network, got {flat.shape[1]}")
    return layers


def activate(h, name):
    if name == "relu":
        return np.maximum(h, 0.0)
    if name == "tanh":
        return np.tanh(h)
    if name == "linear":
        return h
    raise ValueError(f"unsupported activation: {name}")


def forward(flat, obs, hidden=config.BASELINE_HIDDEN, activation="relu"):
    """Logits (n, N_ACTIONS) for n cars, each with its own weights: flat (n, P), obs (n, N_INPUTS)."""
    layers = unflatten(flat, hidden)
    h = obs
    for i, (weights, bias) in enumerate(layers):
        h = np.einsum("ni,nij->nj", h, weights) + bias
        if i < len(layers) - 1:
            h = activate(h, activation)
    return h


def act(flat, obs, hidden=config.BASELINE_HIDDEN, activation="relu"):
    return np.argmax(forward(flat, obs, hidden, activation), axis=1)


def population_policy(flat, hidden=config.BASELINE_HIDDEN, activation="relu"):
    """Policy for sim.simulate: car i is driven by network flat[i]."""
    def policy(obs, cars):
        return act(flat, obs, hidden, activation)
    return policy


class DenseNet:
    """A plain stack of dense layers in numpy. Drives a car with any trained controller."""

    def __init__(self, layers, activations):
        self.layers = [(np.asarray(w, dtype=float), np.asarray(b, dtype=float)) for w, b in layers]
        self.activations = list(activations)

    @property
    def n_params(self):
        return int(sum(w.size + b.size for w, b in self.layers))

    def act(self, obs):
        h = np.asarray(obs, dtype=float)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = activate(h, self.activations[i])
        return np.argmax(h, axis=1)

    def policy(self):
        return lambda obs, cars: self.act(obs)


def from_flat(flat, hidden=config.BASELINE_HIDDEN, activation="relu"):
    layers = [(w[0], b[0]) for w, b in unflatten(np.asarray(flat, dtype=float)[None, :], hidden)]
    return DenseNet(layers, [activation] * len(hidden))


def from_keras(model):
    """DenseNet with the weights of a trained Keras model. Dropout layers are inactive at inference and skipped."""
    keras = keras_cpu()
    layers, activations = [], []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            w, b = layer.get_weights()
            layers.append((w, b))
            activations.append(layer.get_config()["activation"])
    return DenseNet(layers, activations[:-1])     # the output softmax does not change the argmax


def to_keras(flat, hidden=config.BASELINE_HIDDEN, activation="relu"):
    """Keras model with the same architecture and weights as one flat weight vector."""
    keras = keras_cpu()
    model = keras.Sequential([keras.Input((config.N_INPUTS,)),
                              *[keras.layers.Dense(units, activation=activation) for units in hidden],
                              keras.layers.Dense(config.N_ACTIONS, activation="softmax")])
    weights = []
    for w, b in unflatten(np.asarray(flat, dtype=float)[None, :], hidden):
        weights += [w[0], b[0]]
    model.set_weights(weights)
    return model


def load_champions(directory=None):
    """Final controllers saved as <approach>.keras, in the order of config.APPROACHES. Missing ones are skipped."""
    directory = Path(config.CHAMPIONS_DIR if directory is None else directory)
    models = {}
    for approach in config.APPROACHES:
        path = directory / f"{approach}.keras"
        if path.exists():
            models[approach] = keras_cpu().models.load_model(path)
    return models
