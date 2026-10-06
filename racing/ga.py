"""Genetic algorithm over network weights with finish-rate mutation control (spec §9.2)."""
import time
from pathlib import Path

import numpy as np

from racing import config, nets
from racing.curriculum import LevelTracker, draw_seeds, mutation_strength
from racing.population import (RunLogger, evaluate_population, finish_fractions, make_report,
                               save_champion, select_champion)


def tournament(rng, fitness, count, size=config.GA_TOURNAMENT):
    """Indices of `count` winners, each the fittest of `size` individuals drawn from the whole population."""
    entrants = rng.integers(0, len(fitness), (count, size))
    return entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]


def next_generation(rng, weights, fitness, kept, strength):
    """The `kept` fittest unchanged (first), then children made by uniform crossover and Gaussian mutation."""
    n, p = weights.shape
    elite = weights[np.argsort(-fitness, kind="stable")[:kept]]
    n_new = n - kept
    mother = weights[tournament(rng, fitness, n_new)]
    father = weights[tournament(rng, fitness, n_new)]
    children = np.where(rng.random((n_new, p)) < config.GA_CROSSOVER_P, mother, father)
    mutate = rng.random((n_new, p)) < config.GA_MUTATION_P
    children = children + mutate * rng.normal(0.0, strength, (n_new, p))
    is_kept = np.concatenate([np.ones(kept, dtype=bool), np.zeros(n_new, dtype=bool)])
    return np.concatenate([elite, children]), is_kept


def run_ga(pop_size=config.GA_POP, kept=config.GA_KEPT, budget=config.EVAL_BUDGET, seed=0, out_dir=None,
           on_generation=None, n_jobs=1, champion_dir=config.CHAMPIONS_DIR):
    if pop_size <= kept:
        raise ValueError(f"population size ({pop_size}) must be larger than the number of kept individuals ({kept})")
    if budget < pop_size:
        raise ValueError(f"budget ({budget}) is smaller than one generation ({pop_size} evaluations)")
    rng = np.random.default_rng(seed)
    logger = RunLogger(Path(out_dir or config.RUNS_DIR / "ga"))
    tracker = LevelTracker()
    weights = nets.init_population(rng, pop_size)
    is_kept = np.zeros(pop_size, dtype=bool)
    strength = config.GA_S_MAX
    evals, generation, start = 0, 0, time.time()
    while evals + pop_size <= budget:
        level = tracker.level
        seeds = draw_seeds(rng)
        fitness, finished = evaluate_population(weights, seeds, level, n_jobs=n_jobs)
        evals += pop_size
        kept_finish, new_finish = finish_fractions(finished, is_kept)
        leveled_up = tracker.update(kept_finish, new_finish)
        report = make_report(approach="ga", generation=generation, level=level, seeds=seeds, weights=weights,
                             fitness=fitness, finished=finished, is_kept=is_kept, kept_finish=kept_finish,
                             new_finish=new_finish, step_size=strength, evals=evals,
                             elapsed=time.time() - start, leveled_up=leveled_up)
        logger.log(report)
        if on_generation is not None:
            on_generation(report)
        strength = mutation_strength(new_finish)
        weights, is_kept = next_generation(rng, weights, fitness, kept, strength)
        generation += 1
    result = {"approach": "ga", "generations": generation, "evals": evals, "level_reached": tracker.level,
              "wall_time": time.time() - start, "pop_size": pop_size, "kept": kept, "seed": seed}
    if champion_dir is not None:
        champion, val_fitness = select_champion(weights[is_kept], n_jobs=n_jobs)
        result["val_fitness"] = val_fitness
        result["champion_path"] = str(save_champion("ga", champion, champion_dir))
    logger.summary(result)
    return result
