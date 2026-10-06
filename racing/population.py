"""Shared helpers for the evolutionary approaches (spec §9.1): evaluation, reports, logging, final selection."""
import csv
import json
from pathlib import Path

import numpy as np

from racing import config, nets, sim
from racing.tracks import get_track


def _evaluate_chunk(weights, seeds, level):
    total = np.zeros(len(weights))
    finished_all = np.ones(len(weights), dtype=bool)
    policy = nets.population_policy(weights)
    for seed in seeds:
        result = sim.simulate(get_track(seed, level), policy, len(weights))
        total += result.fitness
        finished_all &= result.finished
    return total / len(seeds), finished_all


def evaluate_population(weights, seeds, level, n_jobs=1, backend="loky"):
    """Mean fitness over the tracks and whether each individual finished all of them. Independent of n_jobs."""
    if n_jobs == 1 or len(weights) < 2 * n_jobs:
        return _evaluate_chunk(weights, seeds, level)
    from joblib import Parallel, delayed
    chunks = np.array_split(np.arange(len(weights)), n_jobs)
    parts = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_evaluate_chunk)(weights[c], seeds, level) for c in chunks)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def finish_fractions(finished, is_kept):
    """Fraction of kept and of new individuals that finished; nan for an empty group."""
    kept = float(finished[is_kept].mean()) if is_kept.any() else float("nan")
    new = float(finished[~is_kept].mean()) if (~is_kept).any() else float("nan")
    return kept, new


def make_report(*, approach, generation, level, seeds, weights, fitness, finished, is_kept,
                kept_finish, new_finish, step_size, evals, elapsed, leveled_up):
    """One generation's results, for logging and for the training view."""
    return {"approach": approach, "generation": int(generation), "level": int(level), "seeds": list(seeds),
            "weights": weights, "hidden": list(config.BASELINE_HIDDEN), "activation": "relu",
            "fitness": fitness, "finished": finished, "is_kept": is_kept,
            "kept_finish": kept_finish, "new_finish": new_finish, "step_size": float(step_size),
            "best": float(fitness.max()), "mean": float(fitness.mean()), "worst": float(fitness.min()),
            "evals": int(evals), "elapsed": float(elapsed), "leveled_up": bool(leveled_up)}


class RunLogger:
    """Writes one CSV row and the best network after every generation, and a summary at the end."""
    FIELDS = ["generation", "level", "evals", "best", "mean", "worst", "kept_finish", "new_finish",
              "step_size", "seeds", "elapsed", "leveled_up"]

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for stale in self.dir.glob("best_gen*.npy"):    # left over from an earlier run in the same directory
            stale.unlink()
        self.csv_path = self.dir / "log.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)

    def log(self, report):
        row = [" ".join(str(s) for s in report["seeds"]) if key == "seeds" else report[key] for key in self.FIELDS]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        best = report["weights"][int(np.argmax(report["fitness"]))]
        np.save(self.dir / f"best_gen{report['generation']:04d}.npy", best)

    def summary(self, data):
        (self.dir / "summary.json").write_text(json.dumps(data, indent=2))


def select_champion(candidates, n_jobs=1, seeds=config.VAL_SEEDS):
    """The candidate with the highest mean fitness on the validation tracks (level 4)."""
    fitness, _ = evaluate_population(candidates, list(seeds), config.MAX_LEVEL, n_jobs=n_jobs)
    best = int(np.argmax(fitness))
    return candidates[best], float(fitness[best])


def save_champion(approach, flat, champion_dir=config.CHAMPIONS_DIR):
    directory = Path(champion_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{approach}.keras"
    nets.to_keras(flat).save(path)
    return path
