"""Genetic algorithm over dense network weights (revision 3 §5.1): tournament selection, uniform crossover and
Gaussian mutation whose strength follows the share of the previous generation's children that succeeded."""
import numpy as np

from racing import config, evolution, nets, paths, population
from racing.curriculum import mutation_strength
from racing.nets import DensePopulation


def tournament(rng, fitness, count, size=config.GA_TOURNAMENT):
    """Indices of `count` winners, each the fittest of `size` individuals drawn from the whole population."""
    entrants = rng.integers(0, len(fitness), (count, size))
    return entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]


class GA:
    """The GA as a method of the evolution loop (racing/evolution.py)."""
    approach = "ga"

    def __init__(self, pop_size=config.GA_POP, hidden=config.BASELINE_HIDDEN, activation="relu", s_max=config.GA_S_MAX):
        if pop_size < 2:
            raise ValueError(f"population size ({pop_size}) must be at least 2")
        self.pop_size = pop_size
        self.hidden = tuple(hidden)
        self.activation = activation
        self.s_max = s_max
        self.population = None
        self.is_kept = None
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
        children_before = ~self.is_kept
        child_success = float(result.all_success[children_before].mean()) if children_before.any() else float("nan")
        self.step_size = mutation_strength(child_success, self.s_max, self.s_max / 100)
        n_new = n - len(kept)
        mother = weights[tournament(rng, result.fitness, n_new)]
        father = weights[tournament(rng, result.fitness, n_new)]
        children = np.where(rng.random((n_new, p)) < config.GA_CROSSOVER_P, mother, father)
        mutate = rng.random((n_new, p)) < config.GA_MUTATION_P
        children = children + mutate * rng.normal(0.0, self.step_size, (n_new, p))
        self.population = DensePopulation(np.concatenate([weights[kept], children]), self.hidden, self.activation)
        self.is_kept = np.arange(n) < len(kept)

    def state(self):
        return {"weights": self.population.weights, "is_kept": self.is_kept, "step_size": self.step_size,
                "hidden": self.hidden, "activation": self.activation}

    def load_state(self, state):
        self.hidden, self.activation = tuple(state["hidden"]), state["activation"]
        self.population = DensePopulation(state["weights"], self.hidden, self.activation)
        self.is_kept = state["is_kept"]
        self.step_size = state["step_size"]

    def save_champion(self, index, seed, champion_dir):
        return population.save_dense_champion(self.approach, seed, self.population, index, champion_dir)

    def log_fields(self):
        return {}


def run_ga(seed=0, pop_size=config.GA_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1,
           on_generation=None, start_level=1, max_generations=None, s_max=config.GA_S_MAX):
    """The GA with run seed `seed` (revision 3 §5.1). Returns the run's summary (racing/evolution.py)."""
    return evolution.run(GA(pop_size=pop_size, s_max=s_max), seed, out_dir or paths.run_dir("ga", seed),
                         champion_dir, n_jobs, on_generation, start_level, max_generations)
