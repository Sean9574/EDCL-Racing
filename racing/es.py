"""(μ+λ) evolution strategy over network weights with the 1/5 success rule (spec §9.3)."""
import time
from pathlib import Path

import numpy as np

from racing import config, nets
from racing.curriculum import LevelTracker, draw_seeds
from racing.population import (RunLogger, evaluate_population, finish_fractions, make_report,
                               save_champion, select_champion)


def make_offspring(rng, parents, sigma, lam):
    """λ offspring, each a random parent plus N(0, σ²) on every weight; also returns each one's parent index."""
    parent_of = rng.integers(0, len(parents), lam)
    return parents[parent_of] + rng.normal(0.0, sigma, (lam, parents.shape[1])), parent_of


def select_parents(population, fitness, mu):
    """The μ fittest of parents ∪ offspring."""
    return population[np.argsort(-fitness, kind="stable")[:mu]]


def one_fifth_rule(sigma, success_rate):
    """Grow σ when more than 1/5 of offspring beat their parent, shrink it when fewer do."""
    if success_rate > 0.2:
        sigma = sigma / config.ES_FACTOR
    elif success_rate < 0.2:
        sigma = sigma * config.ES_FACTOR
    return float(np.clip(sigma, config.ES_SIGMA_MIN, config.ES_SIGMA_MAX))


def run_es(mu=config.ES_MU, lam=config.ES_LAMBDA, budget=config.EVAL_BUDGET, seed=0, out_dir=None,
           on_generation=None, n_jobs=1, champion_dir=config.CHAMPIONS_DIR):
    if budget < mu + lam:
        raise ValueError(f"budget ({budget}) is smaller than one generation (μ + λ = {mu + lam} evaluations)")
    rng = np.random.default_rng(seed)
    logger = RunLogger(Path(out_dir or config.RUNS_DIR / "es"))
    tracker = LevelTracker()
    parents = nets.init_population(rng, mu)
    sigma = config.ES_SIGMA0
    evals, generation, start = 0, 0, time.time()
    while evals + mu + lam <= budget:
        level = tracker.level
        seeds = draw_seeds(rng)
        offspring, parent_of = make_offspring(rng, parents, sigma, lam)
        individuals = np.concatenate([parents, offspring])
        is_kept = np.arange(mu + lam) < mu
        fitness, finished = evaluate_population(individuals, seeds, level, n_jobs=n_jobs)
        evals += mu + lam
        kept_finish, new_finish = finish_fractions(finished, is_kept)
        leveled_up = tracker.update(kept_finish, new_finish)
        report = make_report(approach="es", generation=generation, level=level, seeds=seeds, weights=individuals,
                             fitness=fitness, finished=finished, is_kept=is_kept, kept_finish=kept_finish,
                             new_finish=new_finish, step_size=sigma, evals=evals,
                             elapsed=time.time() - start, leveled_up=leveled_up)
        logger.log(report)
        if on_generation is not None:
            on_generation(report)
        sigma = one_fifth_rule(sigma, float(np.mean(fitness[mu:] > fitness[:mu][parent_of])))
        parents = select_parents(individuals, fitness, mu)
        generation += 1
    result = {"approach": "es", "generations": generation, "evals": evals, "level_reached": tracker.level,
              "wall_time": time.time() - start, "mu": mu, "lam": lam, "seed": seed}
    if champion_dir is not None:
        champion, val_fitness = select_champion(parents, n_jobs=n_jobs)
        result["val_fitness"] = val_fitness
        result["champion_path"] = str(save_champion("es", champion, champion_dir))
    logger.summary(result)
    return result
