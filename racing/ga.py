"""Genetic algorithm over network weights with finish-rate mutation control (spec §9.2)."""
import time
from pathlib import Path

import numpy as np

from racing import config, nets, paths
from racing.curriculum import LevelTracker, draw_seeds, mutation_strength
from racing.population import (RunLogger, clear_champion, finish_fractions, fresh_workers, make_report,
                               save_champion, select_champion)
from racing.population import evaluate_weights as evaluate_population   # revision 2.0 evaluation until plan Task 6


def tournament(rng, fitness, count, size=config.GA_TOURNAMENT):
    """Indices of `count` winners, each the fittest of `size` individuals drawn from the whole population."""
    entrants = rng.integers(0, len(fitness), (count, size))
    return entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]


def survivors(fitness, at_pace, kept_min, keep_max):
    """Individuals kept unchanged (spec §9.2): every one that finished at pace, fittest first, at most keep_max; then
    the fittest others until at least kept_min are kept."""
    order = np.argsort(-fitness, kind="stable")
    chosen = [int(i) for i in order if at_pace[i]][:keep_max]
    for i in order:
        if len(chosen) >= kept_min:
            break
        if not at_pace[i]:
            chosen.append(int(i))
    return np.array(chosen, dtype=np.int64)


def next_generation(rng, weights, fitness, at_pace, kept_min, keep_max, strength):
    """The survivors unchanged (first), then children made by uniform crossover and Gaussian mutation."""
    n, p = weights.shape
    keep = survivors(fitness, at_pace, kept_min, keep_max)
    n_new = n - len(keep)
    mother = weights[tournament(rng, fitness, n_new)]
    father = weights[tournament(rng, fitness, n_new)]
    children = np.where(rng.random((n_new, p)) < config.GA_CROSSOVER_P, mother, father)
    mutate = rng.random((n_new, p)) < config.GA_MUTATION_P
    children = children + mutate * rng.normal(0.0, strength, (n_new, p))
    is_kept = np.arange(n) < len(keep)
    return np.concatenate([weights[keep], children]), is_kept


def run_ga(pop_size=config.GA_POP, kept=config.GA_KEPT, budget=config.EVAL_BUDGET, seed=0, out_dir=None,
           on_generation=None, n_jobs=1, champion_dir=config.CHAMPIONS_DIR, start_level=1, s_max=config.GA_S_MAX,
           keep_max_fraction=config.GA_KEEP_MAX_FRACTION):
    if pop_size <= kept:
        raise ValueError(f"population size ({pop_size}) must be larger than the number of kept individuals ({kept})")
    keep_max = int(keep_max_fraction * pop_size)
    if keep_max < kept:
        raise ValueError(f"the population ({pop_size}) must keep at most {keep_max} individuals, "
                         f"at least {kept} are required")
    if budget < pop_size:
        raise ValueError(f"budget ({budget}) is smaller than one generation ({pop_size} evaluations)")
    fresh_workers(n_jobs)                              # every run builds its own tracks (spec §9.1)
    start = time.time()
    rng = np.random.default_rng(seed)
    if champion_dir is not None:
        clear_champion("ga", seed, champion_dir)
    logger = RunLogger(Path(out_dir or paths.run_dir("ga", seed)))
    tracker = LevelTracker(level=start_level)
    weights = nets.init_population(rng, pop_size)
    is_kept = np.zeros(pop_size, dtype=bool)
    strength = float("nan")                             # generation 0 is not produced by mutation
    s_min = s_max / 100
    evals, generation = 0, 0
    while evals + pop_size <= budget:
        level = tracker.level
        seeds = draw_seeds(rng)
        fitness, finished, at_pace = evaluate_population(weights, seeds, level, n_jobs=n_jobs)
        evals += pop_size
        kept_finish, new_finish = finish_fractions(finished, is_kept)
        kept_pace, new_pace = finish_fractions(at_pace, is_kept)
        whole_pace = float(np.mean(at_pace))
        leveled_up = tracker.update(whole_pace)
        report = make_report(approach="ga", generation=generation, level=level, seeds=seeds, weights=weights,
                             fitness=fitness, finished=finished, is_kept=is_kept, kept_finish=kept_finish,
                             new_finish=new_finish, kept_pace=kept_pace, new_pace=new_pace, whole_pace=whole_pace, step_size=strength,
                             evals=evals, elapsed=time.time() - start, leveled_up=leveled_up)
        logger.log(report)
        if on_generation is not None:
            on_generation(report)
        strength = mutation_strength(new_finish, s_max, s_min)
        weights, is_kept = next_generation(rng, weights, fitness, at_pace, kept, keep_max, strength)
        generation += 1
    result = {"approach": "ga", "seed": seed, "generations": generation, "evals": evals,
              "level_reached": tracker.level, "start_level": start_level, "pop_size": pop_size, "kept_min": kept,
              "keep_max": keep_max, "s_max": s_max}
    if champion_dir is not None:
        champion, val_fitness = select_champion(weights[is_kept], n_jobs=n_jobs)
        result["val_fitness"] = val_fitness
        result["champion_path"] = str(save_champion("ga", seed, champion, champion_dir))
    result["wall_time"] = time.time() - start          # includes the final selection (spec §9.1)
    logger.summary(result)
    return result
