"""(μ+λ) evolution strategy over network weights with the 1/5 success rule (spec §9.3), and the seeded ES that
starts from a trained network (spec §9.4)."""
import time
from pathlib import Path

import numpy as np

from racing import config, nets, paths
from racing.curriculum import LevelTracker, draw_seeds
from racing.population import (RunLogger, clear_champion, evaluate_population, finish_fractions, make_report,
                               save_champion, select_champion)


def make_offspring(rng, parents, sigma, lam):
    """λ offspring, each a random parent plus N(0, σ²) on every weight; also returns each one's parent index."""
    parent_of = rng.integers(0, len(parents), lam)
    return parents[parent_of] + rng.normal(0.0, sigma, (lam, parents.shape[1])), parent_of


def select_parents(population, fitness, mu):
    """The μ fittest of parents ∪ offspring."""
    return population[np.argsort(-fitness, kind="stable")[:mu]]


def success_fraction(child_fitness, parent_fitness):
    """Fraction of offspring that beat their own parent; a tie with the parent counts as half a success."""
    return float(np.mean(np.where(child_fitness > parent_fitness, 1.0,
                                  np.where(child_fitness == parent_fitness, 0.5, 0.0))))


def one_fifth_rule(sigma, success_rate):
    """Grow σ when more than 1/5 of offspring beat their parent, shrink it when fewer do."""
    if success_rate > 0.2:
        sigma = sigma / config.ES_FACTOR
    elif success_rate < 0.2:
        sigma = sigma * config.ES_FACTOR
    return float(np.clip(sigma, config.ES_SIGMA_MIN, config.ES_SIGMA_MAX))


def seeded_parents(rng, flat, mu, sigma=config.ES_SIGMA0):
    """The trained network itself, then μ − 1 copies with N(0, σ²) added to every weight (spec §9.4)."""
    flat = np.asarray(flat, dtype=float)
    return np.vstack([flat, flat + rng.normal(0.0, sigma, (mu - 1, len(flat)))])


def trained_baseline(seed, champion_dir=config.CHAMPIONS_DIR):
    """Flat float64 weights of the baseline controller trained with `seed` (spec §9.4)."""
    from racing.backend import keras_cpu
    path = paths.champion_path("baseline", seed, champion_dir=champion_dir)
    if not path.exists():
        raise FileNotFoundError(f"no baseline controller for seed {seed} at {path}; run notebook 02 first")
    return nets.to_flat(keras_cpu().models.load_model(path))


def run_es(mu=config.ES_MU, lam=config.ES_LAMBDA, budget=config.EVAL_BUDGET, seed=0, out_dir=None,
           on_generation=None, n_jobs=1, champion_dir=config.CHAMPIONS_DIR, start_level=1, initial=None,
           approach="es"):
    """(μ+λ)-ES. With `initial`, the flat weights of a trained network, it is the seeded ES (spec §9.4)."""
    if budget < mu + lam:
        raise ValueError(f"budget ({budget}) is smaller than one generation (μ + λ = {mu + lam} evaluations)")
    start = time.time()
    rng = np.random.default_rng(seed)
    if champion_dir is not None:
        clear_champion(approach, seed, champion_dir)
    logger = RunLogger(Path(out_dir or paths.run_dir(approach, seed)))
    tracker = LevelTracker(level=start_level)
    parents = nets.init_population(rng, mu) if initial is None else seeded_parents(rng, initial, mu)
    sigma = config.ES_SIGMA0
    evals, generation = 0, 0
    while evals + mu + lam <= budget:
        level = tracker.level
        seeds = draw_seeds(rng)
        offspring, parent_of = make_offspring(rng, parents, sigma, lam)
        individuals = np.concatenate([parents, offspring])
        is_kept = np.arange(mu + lam) < mu
        fitness, finished, at_pace = evaluate_population(individuals, seeds, level, n_jobs=n_jobs)
        evals += mu + lam
        kept_finish, new_finish = finish_fractions(finished, is_kept)
        kept_pace, new_pace = finish_fractions(at_pace, is_kept)
        leveled_up = tracker.update(kept_pace, new_pace)
        child, parent = fitness[mu:], fitness[:mu][parent_of]
        report = make_report(approach=approach, generation=generation, level=level, seeds=seeds,
                             weights=individuals, fitness=fitness, finished=finished, is_kept=is_kept,
                             kept_finish=kept_finish, new_finish=new_finish, kept_pace=kept_pace,
                             new_pace=new_pace, step_size=sigma, evals=evals, elapsed=time.time() - start,
                             leveled_up=leveled_up, strict_success=float(np.mean(child > parent)),
                             tie_fraction=float(np.mean(child == parent)))
        logger.log(report)
        if on_generation is not None:
            on_generation(report)
        sigma = one_fifth_rule(sigma, success_fraction(child, parent))
        if leveled_up:
            sigma = config.ES_SIGMA0                        # restart the step size on the harder tracks
        parents = select_parents(individuals, fitness, mu)
        generation += 1
    result = {"approach": approach, "seed": seed, "generations": generation, "evals": evals,
              "level_reached": tracker.level, "start_level": start_level, "mu": mu, "lam": lam,
              "seeded": initial is not None}
    if champion_dir is not None:
        champion, val_fitness = select_champion(parents, n_jobs=n_jobs)
        result["val_fitness"] = val_fitness
        result["champion_path"] = str(save_champion(approach, seed, champion, champion_dir))
    result["wall_time"] = time.time() - start          # includes the final selection (spec §9.1)
    logger.summary(result)
    return result
