"""Genetic algorithm that grows its own networks (spec §9.6). Every car carries its own architecture, starting from
the inputs wired directly to the outputs. Children come from tournament selection, a crossover that keeps the shape
parent's architecture and mixes the values both parents have, Gaussian mutation whose strength follows the share of
the previous generation's children that succeeded, and structural mutations that add and remove hidden units and
layers with no size limit. Every car also carries a setup, inherited value by value from either parent and mutated
(spec §9.11)."""
import numpy as np

from racing import config, evolution, nets, paths, population, setup
from racing.curriculum import mutation_strength
from racing.nets import ShapedPopulation


def tournament(rng, fitness, count, size=config.GA_TOURNAMENT):
    """Indices of `count` winners, each the fittest of `size` individuals drawn from the whole population."""
    entrants = rng.integers(0, len(fitness), (count, size))
    return entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]


def layers_of(flat, hidden):
    """[(W (a, b), bias (b,))] of one network, as copies the caller may change."""
    return [(w[0].copy(), b[0].copy()) for w, b in nets.unflatten(np.asarray(flat, dtype=float)[None, :], hidden)]


def flat_of(layers):
    """(flat weights in the layout of nets.unflatten, hidden layer sizes) of [(W, bias)]."""
    flat = np.concatenate([part for w, b in layers for part in (w.ravel(), b)])
    return flat, tuple(int(w.shape[1]) for w, _ in layers[:-1])


def crossover(rng, shape_parent, other):
    """A child of two networks given as [(W, bias)] (spec §9.6): the shape parent's architecture; each value both
    parents have comes from the shape parent with probability GA_CROSSOVER_P, otherwise from the other parent; every
    other value comes from the shape parent. Units are identified by their layer, counted from the inputs, and their
    position in it, and the outputs are a layer of their own: hidden layer k is shared where both parents have it,
    over the units both have; the output weights only when both have the same number of hidden layers; the output
    biases always."""
    depth, other_depth = len(shape_parent) - 1, len(other) - 1          # numbers of hidden layers
    child = []
    for k, (w, b) in enumerate(shape_parent):
        w, b = w.copy(), b.copy()
        if k < depth:
            match = other[k] if k < other_depth else None
            weights_shared = match is not None
        else:
            match = other[other_depth]
            weights_shared = other_depth == depth
        if match is not None:
            other_w, other_b = match
            if weights_shared:
                rows, cols = min(w.shape[0], other_w.shape[0]), min(w.shape[1], other_w.shape[1])
                take = rng.random((rows, cols)) < config.GA_CROSSOVER_P
                w[:rows, :cols] = np.where(take, w[:rows, :cols], other_w[:rows, :cols])
            units = min(len(b), len(other_b))
            take = rng.random(units) < config.GA_CROSSOVER_P
            b[:units] = np.where(take, b[:units], other_b[:units])
        child.append((w, b))
    return child


def add_unit(rng, layers):
    """A random hidden layer gains a unit with Glorot-uniform incoming weights, a zero bias and zero outgoing weights,
    so the network's outputs do not change (spec §9.6). A network without hidden layers is returned as it is."""
    if len(layers) == 1:
        return layers
    k = int(rng.integers(len(layers) - 1))
    (w, b), (out_w, out_b) = layers[k], layers[k + 1]
    limit = np.sqrt(6.0 / (w.shape[0] + w.shape[1] + 1))
    layers = list(layers)
    layers[k] = (np.hstack([w, rng.uniform(-limit, limit, (w.shape[0], 1))]), np.append(b, 0.0))
    layers[k + 1] = (np.vstack([out_w, np.zeros((1, out_w.shape[1]))]), out_b)
    return layers


def remove_unit(rng, layers):
    """A random unit of a random hidden layer is removed with its weights; the only unit of a layer takes the layer
    with it (fold). A network without hidden layers is returned as it is."""
    if len(layers) == 1:
        return layers
    k = int(rng.integers(len(layers) - 1))
    (w, b), (out_w, out_b) = layers[k], layers[k + 1]
    if w.shape[1] == 1:
        return fold(layers, k)
    j = int(rng.integers(w.shape[1]))
    layers = list(layers)
    layers[k] = (np.delete(w, j, axis=1), np.delete(b, j))
    layers[k + 1] = (np.delete(out_w, j, axis=0), out_b)
    return layers


def add_layer(rng, layers):
    """A hidden layer inserted at a random depth, as wide as the layer below it (the inputs for the first), with
    identity weights and zero biases: behind a hidden layer, whose ReLU outputs are never negative, the outputs do not
    change (spec §9.6)."""
    k = int(rng.integers(len(layers)))                  # the new layer becomes hidden layer k
    width = layers[k][0].shape[0]                       # the units of the layer below it
    return [*layers[:k], (np.eye(width), np.zeros(width)), *layers[k:]]


def remove_layer(rng, layers):
    """A random hidden layer removed (fold). A network without hidden layers is returned as it is."""
    if len(layers) == 1:
        return layers
    return fold(layers, int(rng.integers(len(layers) - 1)))


def fold(layers, k):
    """Hidden layer k removed and the layers on either side joined by the product of their weights: weights
    W_k W_{k+1}, biases b_k W_{k+1} + b_{k+1}, which reproduces the network wherever layer k's units are active."""
    (w1, b1), (w2, b2) = layers[k], layers[k + 1]
    return [*layers[:k], (w1 @ w2, b1 @ w2 + b2), *layers[k + 2:]]


STRUCTURAL = (("GA_ADD_UNIT", add_unit), ("GA_REMOVE_UNIT", remove_unit), ("GA_ADD_LAYER", add_layer),
              ("GA_REMOVE_LAYER", remove_layer))


def make_child(rng, shape_parent, other, strength, boost=False):
    """(flat weights, hidden layer sizes) of a child of two networks given as (flat, hidden) (spec §9.6): crossover;
    each weight and bias mutated with probability GA_MUTATION_P by N(0, strength), the output layer STALL_BOOST times
    as strongly with boost, from the same draws (spec §9.4); then the four structural mutations, each with its own
    probability, in the order of STRUCTURAL."""
    flat, hidden = flat_of(crossover(rng, layers_of(*shape_parent), layers_of(*other)))
    mutate = rng.random(flat.size) < config.GA_MUTATION_P
    noise = rng.normal(0.0, strength, flat.size)
    if boost:
        noise[nets.decision_mask(hidden)] *= config.STALL_BOOST
    layers = layers_of(flat + mutate * noise, hidden)
    for name, change in STRUCTURAL:
        if rng.random() < getattr(config, name):
            layers = change(rng, layers)
    return flat_of(layers)


class GA:
    """The GA as a method of the evolution loop (racing/evolution.py): a nets.ShapedPopulation, survivors first and
    unchanged, the final controller saved through population.save_dense_champion."""
    approach = "ga"

    def __init__(self, pop_size=config.GA_POP, activation="relu", s_max=config.GA_S_MAX):
        if pop_size < 2:
            raise ValueError(f"population size ({pop_size}) must be at least 2")
        self.pop_size = pop_size
        self.activation = activation
        self.s_max = s_max
        self.step_size = float("nan")
        self.population = None
        self.is_kept = None

    @property
    def setup(self):
        """Every car's setup, aligned with the population (spec §9.11)."""
        return self.population.setup

    def initialize(self, rng):
        hidden = tuple(config.GA_START_HIDDEN)
        weights = nets.init_population(rng, self.pop_size, hidden)
        self.population = ShapedPopulation([hidden] * self.pop_size, list(weights), self.activation,
                                           setup=setup.random(rng, self.pop_size))   # random start (spec §9.11)
        self.is_kept = np.zeros(self.pop_size, dtype=bool)
        self.step_size = float("nan")                  # generation 0 is not produced by mutation

    def advance(self, rng, result, kept, event, boost=False):
        """Survivors first and unchanged, then children by tournament, crossover, weight mutation and structural
        mutation (make_child). A child takes each setup value from either parent, then the setup mutation applies
        (spec §9.11); these draws come after the networks'. With boost (a stall, spec §9.4) the output layer and the
        setup mutate harder, from the same draws."""
        cars = self.population
        child_success = evolution.children_success(result, self.is_kept)
        self.step_size = mutation_strength(child_success, self.s_max, self.s_max * config.GA_S_MIN / config.GA_S_MAX)
        n_new = len(cars) - len(kept)
        mothers = tournament(rng, result.fitness, n_new)
        fathers = tournament(rng, result.fitness, n_new)
        children = [make_child(rng, cars.network(mother), cars.network(father), self.step_size, boost)
                    for mother, father in zip(mothers, fathers)]
        children_setups = setup.mutate(rng, setup.crossover(rng, cars.setup[mothers], cars.setup[fathers]),
                                       boost=boost)
        kept = np.asarray(kept, dtype=np.int64)
        self.population = ShapedPopulation([cars.shapes[i] for i in kept] + [hidden for _, hidden in children],
                                           [cars.weights[i] for i in kept] + [flat for flat, _ in children],
                                           self.activation, setup=np.concatenate([cars.setup[kept], children_setups]))
        self.is_kept = np.arange(len(self.population)) < len(kept)

    def state(self):
        cars = self.population
        return {"shapes": [list(shape) for shape in cars.shapes], "networks": list(cars.weights),
                "setup": cars.setup, "is_kept": self.is_kept, "step_size": self.step_size,
                "activation": self.activation}

    def load_state(self, state):
        self.activation = state["activation"]
        self.population = ShapedPopulation([tuple(shape) for shape in state["shapes"]], state["networks"],
                                           self.activation, setup=state["setup"])
        self.is_kept = state["is_kept"]
        self.step_size = state["step_size"]

    def save_champion(self, index, seed, champion_dir):
        return population.save_dense_champion(self.approach, seed, self.population, index, champion_dir)

    def log_fields(self):
        """The population's network size (spec §9.1): the mean and largest number of hidden layers, hidden units and
        parameters."""
        layers = np.array([len(shape) for shape in self.population.shapes])
        units = np.array([sum(shape) for shape in self.population.shapes])
        params = np.array([len(w) for w in self.population.weights])
        return {"mean_layers": float(layers.mean()), "max_layers": int(layers.max()),
                "mean_nodes": float(units.mean()), "max_nodes": int(units.max()),
                "mean_params": float(params.mean()), "max_params": int(params.max())}


def run_ga(seed=0, pop_size=config.GA_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1,
           on_generation=None, start_level=1, max_generations=None, s_max=config.GA_S_MAX):
    """The GA with run seed `seed` (spec §9.6). Returns the run's summary (racing/evolution.py)."""
    return evolution.run(GA(pop_size=pop_size, s_max=s_max), seed, out_dir or paths.run_dir("ga", seed),
                         champion_dir, n_jobs, on_generation, start_level, max_generations)
