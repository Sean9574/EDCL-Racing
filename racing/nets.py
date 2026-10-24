"""Controller networks (spec §6): flat weight vectors for populations, a batched numpy forward pass,
and conversion to and from Keras models."""
import json
import re
from pathlib import Path

import numpy as np

from racing import config, paths
from racing import setup as car_setup
from racing.backend import keras_cpu


def layer_sizes(hidden=config.BASELINE_HIDDEN, n_inputs=config.N_INPUTS):
    return [n_inputs, *hidden, config.N_ACTIONS]


def n_params(hidden=config.BASELINE_HIDDEN, n_inputs=config.N_INPUTS):
    sizes = layer_sizes(hidden, n_inputs)
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def decision_mask(hidden=config.BASELINE_HIDDEN, n_inputs=config.N_INPUTS):
    """True for the flat weights and biases of the last layer, the decision layer into the N_ACTIONS outputs, which
    the stall boost mutates harder (spec §9.4)."""
    sizes = layer_sizes(hidden, n_inputs)
    last = sizes[-2] * sizes[-1] + sizes[-1]
    mask = np.zeros(n_params(hidden, n_inputs), dtype=bool)
    mask[-last:] = True
    return mask


def inputs_of(n_weights, hidden=config.BASELINE_HIDDEN):
    """Inputs of a dense network with `n_weights` weights and biases: config.N_INPUTS for this revision's networks,
    fewer for an earlier revision's, which read only the first inputs (spec §6)."""
    first = (*hidden, config.N_ACTIONS)[0]                         # units of the first layer
    rest = n_params(hidden) - (config.N_INPUTS + 1) * first
    inputs, extra = divmod(int(n_weights) - rest, first)
    if extra or inputs < 2:
        raise ValueError(f"{n_weights} weights fit no network with hidden layers {tuple(hidden)}")
    return inputs - 1


def init_population(rng, n, hidden=config.BASELINE_HIDDEN, n_inputs=config.N_INPUTS):
    """Glorot-uniform weights and zero biases for n networks, flattened to shape (n, n_params)."""
    parts = []
    sizes = layer_sizes(hidden, n_inputs)
    for a, b in zip(sizes[:-1], sizes[1:]):
        limit = np.sqrt(6.0 / (a + b))
        parts.append(rng.uniform(-limit, limit, (n, a * b)))
        parts.append(np.zeros((n, b)))
    return np.concatenate(parts, axis=1)


def unflatten(flat, hidden=config.BASELINE_HIDDEN):
    """Split (n, P) flat weights into [(W (n, a, b), bias (n, b))] per layer; P gives the number of inputs."""
    n = flat.shape[0]
    layers, pos = [], 0
    sizes = layer_sizes(hidden, inputs_of(flat.shape[1], hidden))
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


def _sum_products(h, weights):
    """einsum("ni,nij->nj") (spec §5.7): each unit sums input × weight in input order, from zero.
    For a layer of one unit numpy would sum in SIMD lanes, which no sequential sum (the compiled simulator's)
    reproduces, so that sum is explicit."""
    if weights.shape[2] != 1:
        return np.einsum("ni,nij->nj", h, weights)
    total = np.zeros((h.shape[0], 1))
    for k in range(weights.shape[1]):
        total = total + h[:, k:k + 1] * weights[:, k, :]
    return total


def forward(flat, obs, hidden=config.BASELINE_HIDDEN, activation="relu"):
    """Logits (n, N_ACTIONS) for n cars, each with its own weights: flat (n, P), obs (n, N_INPUTS). A network of an
    earlier revision reads only its first inputs."""
    layers = unflatten(flat, hidden)
    h = obs[:, :layers[0][0].shape[1]]
    for i, (weights, bias) in enumerate(layers):
        h = _sum_products(h, weights) + bias
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


class DensePopulation:
    """Dense networks of one architecture, one flat weight vector and one setup row per car (spec §9.1, §9.11).
    Like every population type, it offers len(), subset(indices), policy(), driver(i) and setup."""

    def __init__(self, weights, hidden=config.BASELINE_HIDDEN, activation="relu", setup=None):
        self.weights = np.asarray(weights, dtype=float)
        self.hidden = tuple(int(units) for units in hidden)
        self.activation = activation
        self.setup = car_setup.as_rows(setup, len(self.weights))

    def __len__(self):
        return len(self.weights)

    @property
    def n_inputs(self):
        return inputs_of(self.weights.shape[1], self.hidden)

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return DensePopulation(self.weights[indices], self.hidden, self.activation, self.setup[indices])

    def policy(self):
        return population_policy(self.weights, self.hidden, self.activation)

    def driver(self, i):
        return from_flat(self.weights[int(i)], self.hidden, self.activation, self.setup[int(i)])

    def network(self, i):
        """(flat weights, hidden layer sizes) of car i."""
        return self.weights[int(i)], self.hidden


class ShapedPopulation:
    """Dense networks whose hidden layers differ from car to car (the GA, spec §9.6): one architecture, one flat
    weight vector in the layout of unflatten and one setup row per car; every network reads config.N_INPUTS inputs.
    It offers the population interface of DensePopulation. The numpy forward pass groups the cars by architecture and
    runs one pass per layer (nets.forward) for each group, so no network is padded."""

    def __init__(self, shapes, weights, activation="relu", setup=None):
        self.shapes = [tuple(int(units) for units in shape) for shape in shapes]
        self.weights = [np.asarray(w, dtype=float) for w in weights]
        if len(self.shapes) != len(self.weights):
            raise ValueError(f"{len(self.shapes)} architectures for {len(self.weights)} networks")
        for shape, w in zip(self.shapes, self.weights):
            if w.shape != (n_params(shape),):
                raise ValueError(f"a network with hidden layers {shape} has {n_params(shape)} weights, not {w.shape}")
        self.activation = activation
        self.setup = car_setup.as_rows(setup, len(self.weights))

    def __len__(self):
        return len(self.weights)

    @property
    def n_inputs(self):
        return config.N_INPUTS

    def groups(self):
        """{architecture: car indices (int64)}, in order of first appearance."""
        found = {}
        for i, shape in enumerate(self.shapes):
            found.setdefault(shape, []).append(i)
        return {shape: np.array(cars, dtype=np.int64) for shape, cars in found.items()}

    def stacks(self):
        """[(architecture, car indices, their flat weights stacked (k, P))]: what logits computes with."""
        return [(shape, cars, np.stack([self.weights[i] for i in cars])) for shape, cars in self.groups().items()]

    def logits(self, obs, stacks=None):
        """Outputs (n, N_ACTIONS) of every car's network for its own row of obs."""
        out = np.zeros((len(self), config.N_ACTIONS))
        for shape, cars, weights in self.stacks() if stacks is None else stacks:
            out[cars] = forward(weights, obs[cars], shape, self.activation)
        return out

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.int64).ravel()
        return ShapedPopulation([self.shapes[int(i)] for i in indices], [self.weights[int(i)] for i in indices],
                                self.activation, self.setup[indices])

    def policy(self):
        """Policy for sim.simulate: car i is driven by its own network; the groups are stacked once, here."""
        stacks = self.stacks()
        return lambda obs, cars: np.argmax(self.logits(obs, stacks), axis=1)

    def network(self, i):
        """(flat weights, hidden layer sizes) of car i."""
        return self.weights[int(i)], self.shapes[int(i)]

    def driver(self, i):
        return from_flat(self.weights[int(i)], self.shapes[int(i)], self.activation, self.setup[int(i)])


class DenseNet:
    """A plain stack of dense layers in numpy. Drives a car with any trained controller, with the car's setup
    (spec §9.11; revision 3's when none is given)."""

    def __init__(self, layers, activations, setup=None):
        self.layers = [(np.asarray(w, dtype=float), np.asarray(b, dtype=float)) for w, b in layers]
        self.activations = list(activations)
        self.setup = car_setup.as_row(setup)

    @property
    def n_params(self):
        return int(sum(w.size + b.size for w, b in self.layers))

    @property
    def n_inputs(self):
        return int(self.layers[0][0].shape[0])

    def act(self, obs):
        h = np.asarray(obs, dtype=float)[:, :self.n_inputs]      # an earlier revision's network: its first inputs
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = activate(h, self.activations[i])
        return np.argmax(h, axis=1)

    def policy(self):
        return lambda obs, cars: self.act(obs)


def from_flat(flat, hidden=config.BASELINE_HIDDEN, activation="relu", setup=None):
    layers = [(w[0], b[0]) for w, b in unflatten(np.asarray(flat, dtype=float)[None, :], hidden)]
    return DenseNet(layers, [activation] * len(hidden), setup)


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


def dense_spec(model):
    """(hidden sizes, hidden activation) of a dense Keras model; dropout layers are ignored."""
    keras = keras_cpu()
    dense = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    hidden = tuple(int(layer.units) for layer in dense[:-1])
    activation = dense[0].get_config()["activation"] if hidden else "relu"
    return hidden, activation


def to_keras(flat, hidden=config.BASELINE_HIDDEN, activation="relu"):
    """Keras model with the same architecture and weights as one flat weight vector."""
    keras = keras_cpu()
    model = keras.Sequential([keras.Input((inputs_of(np.size(flat), hidden),)),
                              *[keras.layers.Dense(units, activation=activation) for units in hidden],
                              keras.layers.Dense(config.N_ACTIONS, activation="softmax")])
    weights = []
    for w, b in unflatten(np.asarray(flat, dtype=float)[None, :], hidden):
        weights += [w[0], b[0]]
    model.set_weights(weights)
    return model


def to_flat(model):
    """Flat float64 weights of a dense Keras model, in the layout of unflatten (the inverse of to_keras)."""
    keras = keras_cpu()
    parts = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            w, b = layer.get_weights()
            parts += [np.asarray(w, dtype=float).ravel(), np.asarray(b, dtype=float)]
    return np.concatenate(parts)


def drivers_setup(drivers):
    """The (len(drivers), 2) setups of a list of drivers, in order, for sim.simulate (spec §9.11)."""
    return car_setup.as_rows([np.asarray(driver.setup, dtype=float) for driver in drivers], len(drivers))


def mixed_policy(drivers):
    """Policy for sim.simulate: car i is driven by drivers[i]; the networks may differ in architecture."""
    def policy(obs, cars):
        return np.array([driver.act(obs[i:i + 1])[0] for i, driver in enumerate(drivers)], dtype=np.int64)
    return policy


def champion_seeds(directory=None):
    """{approach: [run seeds]} of the final controllers saved as <directory>/<approach>/seed<k>.keras, in the
    order of config.APPROACHES. Approaches without controllers are left out; other file names are ignored."""
    directory = Path(config.CHAMPIONS_DIR if directory is None else directory)
    found = {}
    for approach in config.APPROACHES:
        seeds = []
        for path in (directory / approach).glob("seed*.keras"):
            match = re.fullmatch(r"seed(\d+)\.keras", path.name)
            if match:
                seeds.append(int(match.group(1)))
        if seeds:
            found[approach] = sorted(seeds)
    return found


def refuse_other_actions(approach, seed, outputs):
    """Raises ValueError unless a saved controller has config.N_ACTIONS outputs (spec §10.3): the 45-class table
    would misread the outputs of one made for another action set."""
    if int(outputs) != config.N_ACTIONS:
        raise ValueError(f"{approach} run {seed} was made for {int(outputs)} actions, "
                         f"this version has {config.N_ACTIONS}")


def refuse_other_inputs(approach, seed, inputs):
    """Raises ValueError unless a saved controller reads config.N_INPUTS inputs (spec §6, §10.3): one of version 3.9
    reads 12 (revision 3.1's read 10) and knows nothing of the racing line."""
    if int(inputs) != config.N_INPUTS:
        raise ValueError(f"{approach} run {seed} was made for {int(inputs)} inputs, "
                         f"this version has {config.N_INPUTS}")


def load_champion(approach, seed, directory=None):
    """(Keras model, driver) of one final controller. A NEAT controller drives with its genome (<seed>.json,
    spec §9.9); an evolved dense controller with its float64 weights (.npy) and the architecture of its Keras
    file (dense_spec); an imitation controller with the Keras weights (spec §6). The driver has the setup saved in
    <seed>.setup.json, or revision 3's without that file (spec §6). A controller whose output count is not
    config.N_ACTIONS (15 up to version 3.8), or whose input count is not config.N_INPUTS (12 in version 3.9), is
    refused with a ValueError naming it (spec §6, §10.3)."""
    keras_path = paths.champion_path(approach, seed, champion_dir=directory)
    model = keras_cpu().models.load_model(keras_path)
    refuse_other_actions(approach, seed, model.output_shape[-1])
    refuse_other_inputs(approach, seed, model.input_shape[-1])
    genome_path = paths.champion_path(approach, seed, ".json", directory)
    flat_path = paths.champion_path(approach, seed, ".npy", directory)
    if genome_path.exists():
        from racing import neat                    # neat imports evolution lazily; nets stays import-light
        genome = neat.Genome.from_dict(json.loads(genome_path.read_text()))
        refuse_other_actions(approach, seed, int(np.isin(genome.node_ids, list(neat.OUTPUT_IDS)).sum()))
        driver = neat.NeatNet(genome)
    elif flat_path.exists():
        flat = np.load(flat_path)
        hidden, activation = dense_spec(model)
        try:
            inputs_of(flat.size, hidden)         # an earlier revision's fewer inputs still fit
        except ValueError:                       # a weights file of another action set beside a current Keras file
            last = (hidden[-1] if hidden else config.N_INPUTS) + 1
            outputs = (flat.size - (n_params(hidden) - last * config.N_ACTIONS)) / last
            refuse_other_actions(approach, seed, round(outputs))
            raise
        driver = from_flat(flat, hidden, activation)
    else:
        driver = from_keras(model)
    setup_path = paths.champion_path(approach, seed, ".setup.json", directory)
    if setup_path.exists():
        driver.setup = car_setup.from_dict(json.loads(setup_path.read_text()))
    return model, driver
