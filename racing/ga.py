"""Genetic algorithm over dense network weights (spec §9.6): tournament selection, uniform crossover and
Gaussian mutation whose strength follows the share of the previous generation's children that succeeded. Every car
also carries a setup, inherited value by value from either parent and mutated (spec §9.11)."""
import numpy as np

from racing import config, evolution, nets, paths, setup
from racing.curriculum import mutation_strength
from racing.nets import DensePopulation


def tournament(rng, fitness, count, size=config.GA_TOURNAMENT):
    """Indices of `count` winners, each the fittest of `size` individuals drawn from the whole population."""
    entrants = rng.integers(0, len(fitness), (count, size))
    return entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]


class GA(evolution.DenseMethod):
    """The GA as a method of the evolution loop (racing/evolution.py)."""
    approach = "ga"

    def __init__(self, pop_size=config.GA_POP, hidden=config.BASELINE_HIDDEN, activation="relu", s_max=config.GA_S_MAX):
        super().__init__(pop_size, hidden, activation)
        self.s_max = s_max
        self.step_size = float("nan")

    def initialize(self, rng):
        weights = nets.init_population(rng, self.pop_size, self.hidden)
        self.population = DensePopulation(weights, self.hidden, self.activation,
                                          setup=setup.random(rng, self.pop_size))   # random start (spec §9.11)
        self.is_kept = np.zeros(self.pop_size, dtype=bool)
        self.step_size = float("nan")                  # generation 0 is not produced by mutation

    def advance(self, rng, result, kept, event, boost=False):
        """Survivors first and unchanged, then children by tournament, uniform crossover and Gaussian mutation. A
        child takes each setup value from either parent, then the setup mutation applies (spec §9.11); these
        draws come after the network's. With boost (a stall, spec §9.4) the noise of the decision layer is
        STALL_BOOST times as strong and the setup mutation boosted, from the same draws."""
        weights, setups = self.population.weights, self.population.setup
        n, p = weights.shape
        child_success = evolution.children_success(result, self.is_kept)
        self.step_size = mutation_strength(child_success, self.s_max, self.s_max * config.GA_S_MIN / config.GA_S_MAX)
        n_new = n - len(kept)
        mothers = tournament(rng, result.fitness, n_new)
        fathers = tournament(rng, result.fitness, n_new)
        children = np.where(rng.random((n_new, p)) < config.GA_CROSSOVER_P, weights[mothers], weights[fathers])
        mutate = rng.random((n_new, p)) < config.GA_MUTATION_P
        noise = rng.normal(0.0, self.step_size, (n_new, p))
        if boost:
            noise[:, nets.decision_mask(self.hidden, self.population.n_inputs)] *= config.STALL_BOOST
        children = children + mutate * noise
        children_setups = setup.mutate(rng, setup.crossover(rng, setups[mothers], setups[fathers]), boost=boost)
        self.assemble(weights[kept], children, setups[kept], children_setups)

    def state(self):
        return {"weights": self.population.weights, "setup": self.population.setup, "is_kept": self.is_kept,
                "step_size": self.step_size, "hidden": self.hidden, "activation": self.activation}

    def load_state(self, state):
        self.hidden, self.activation = tuple(state["hidden"]), state["activation"]
        self.population = DensePopulation(state["weights"], self.hidden, self.activation, setup=state["setup"])
        self.is_kept = state["is_kept"]
        self.step_size = state["step_size"]


def run_ga(seed=0, pop_size=config.GA_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1,
           on_generation=None, start_level=1, max_generations=None, s_max=config.GA_S_MAX):
    """The GA with run seed `seed` (spec §9.6). Returns the run's summary (racing/evolution.py)."""
    return evolution.run(GA(pop_size=pop_size, s_max=s_max), seed, out_dir or paths.run_dir("ga", seed),
                         champion_dir, n_jobs, on_generation, start_level, max_generations)
