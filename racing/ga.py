"""Genetic algorithm over dense network weights (revision 3 §5.1): tournament selection, uniform crossover and
Gaussian mutation whose strength follows the share of the previous generation's children that succeeded."""
import numpy as np

from racing import config, evolution, nets, paths
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
        self.population = DensePopulation(nets.init_population(rng, self.pop_size, self.hidden), self.hidden,
                                          self.activation)
        self.is_kept = np.zeros(self.pop_size, dtype=bool)
        self.step_size = float("nan")                  # generation 0 is not produced by mutation

    def advance(self, rng, result, kept, event):
        """Survivors first and unchanged, then children by tournament, uniform crossover and Gaussian mutation."""
        weights = self.population.weights
        n, p = weights.shape
        child_success = evolution.children_success(result, self.is_kept)
        self.step_size = mutation_strength(child_success, self.s_max, self.s_max * config.GA_S_MIN / config.GA_S_MAX)
        n_new = n - len(kept)
        mother = weights[tournament(rng, result.fitness, n_new)]
        father = weights[tournament(rng, result.fitness, n_new)]
        children = np.where(rng.random((n_new, p)) < config.GA_CROSSOVER_P, mother, father)
        mutate = rng.random((n_new, p)) < config.GA_MUTATION_P
        children = children + mutate * rng.normal(0.0, self.step_size, (n_new, p))
        self.assemble(weights[kept], children)

    def state(self):
        return {"weights": self.population.weights, "is_kept": self.is_kept, "step_size": self.step_size,
                "hidden": self.hidden, "activation": self.activation}

    def load_state(self, state):
        self.hidden, self.activation = tuple(state["hidden"]), state["activation"]
        self.population = DensePopulation(state["weights"], self.hidden, self.activation)
        self.is_kept = state["is_kept"]
        self.step_size = state["step_size"]


def run_ga(seed=0, pop_size=config.GA_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1,
           on_generation=None, start_level=1, max_generations=None, s_max=config.GA_S_MAX):
    """The GA with run seed `seed` (revision 3 §5.1). Returns the run's summary (racing/evolution.py)."""
    return evolution.run(GA(pop_size=pop_size, s_max=s_max), seed, out_dir or paths.run_dir("ga", seed),
                         champion_dir, n_jobs, on_generation, start_level, max_generations)
