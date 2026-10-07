"""Shared helpers for the evolutionary approaches (spec §9.1, revision 3 §4): time-trial evaluation, final selection
and saved controllers, and the revision 2.0 logging helpers still used by ga.py and es.py until Task 6."""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from racing import config, nets, paths, sim, tracks
from racing.tracks import get_track


def fresh_workers(n_jobs):
    """Starts an evolutionary run without built tracks, so every run pays the same cost of generating its tracks
    (spec §9.1): empties this process's track cache and, for a parallel run, shuts down joblib's reusable worker
    processes (and their track caches); the run's first parallel evaluation starts new ones."""
    tracks.get_track.cache_clear()
    if n_jobs > 1:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor(reuse=True).shutdown(wait=True)     # creating an executor starts no workers


def split_tasks(n, seeds, n_jobs):
    """(track position, track seed, individual indices) tasks for one evaluation (spec §9.5). Each track's
    individuals are split into n_jobs // len(seeds) shares (at least one), so a task drives many cars on one track."""
    shares = min(max(1, n_jobs // len(seeds)), max(n, 1))
    parts = np.array_split(np.arange(n), shares)
    return [(k, seed, part) for k, seed in enumerate(seeds) for part in parts]


@dataclass(eq=False)
class TrialResult:
    """One generation's time trials (revision 3 §4.1): one row per map, one column per car."""
    fitness: np.ndarray        # (n,) mean fitness over the maps, with the target time as the time limit
    success: np.ndarray        # (maps, n) three laps within the target time
    laps: np.ndarray           # (maps, n) laps completed within the target time
    finish_time: np.ndarray    # (maps, n) s, nan unless success

    @property
    def all_success(self):
        return self.success.all(axis=0)

    @property
    def lap_share(self):
        return self.laps.sum(axis=0) / (config.LAPS * self.laps.shape[0])

    @property
    def map_shares(self):
        return self.success.mean(axis=1)

    @property
    def min_share(self):
        return float(self.map_shares.min())

    @property
    def lap_times(self):
        return self.finish_time / config.LAPS


def _time_trial(population, seed, level, factor):
    """Fitness, success, laps in time and finish time of every car over config.LAPS laps of one map, ending at the
    target time (revision 3 §4.1)."""
    track = get_track(seed, level)
    result = sim.simulate(track, population.policy(), len(population), time_limit=sim.target_time(track, factor),
                          laps=config.LAPS)
    return (result.fitness, result.finished, np.minimum(result.laps_done, config.LAPS),
            np.where(result.finished, result.finish_time, np.nan))


def evaluate_population(population, seeds, level, factor, n_jobs=1, backend="loky"):
    """Time trials of every car on every map (revision 3 §4.1). The per-map results are combined in map order, so
    the result does not depend on n_jobs."""
    n = len(population)
    tasks = split_tasks(n, seeds, n_jobs)
    if n_jobs == 1 or len(tasks) == 1:
        parts = [_time_trial(population.subset(idx), seed, level, factor) for _, seed, idx in tasks]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_time_trial)(population.subset(idx), seed, level, factor) for _, seed, idx in tasks)
    maps = len(seeds)
    per_map = np.zeros((maps, n))
    success = np.zeros((maps, n), dtype=bool)
    laps = np.zeros((maps, n), dtype=np.int64)
    finish_time = np.full((maps, n), np.nan)
    for (k, _, idx), (fitness, done, laps_done, times) in zip(tasks, parts):
        per_map[k, idx] = fitness
        success[k, idx] = done
        laps[k, idx] = laps_done
        finish_time[k, idx] = times
    total = np.zeros(n)
    for k in range(maps):
        total += per_map[k]
    return TrialResult(fitness=total / maps, success=success, laps=laps, finish_time=finish_time)


def select_final(population, candidates, factor, n_jobs=1, seeds=config.VAL_SEEDS):
    """The final controller (revision 3 §4.6): of the candidate indices, the car with the most successes on the
    validation maps at target factor `factor`, then the highest mean fitness, then the earliest candidate. Returns
    (index into population, validation successes, mean validation fitness)."""
    candidates = np.asarray(candidates, dtype=np.int64)
    result = evaluate_population(population.subset(candidates), list(seeds), config.MAX_LEVEL, factor, n_jobs=n_jobs)
    successes = result.success.sum(axis=0)
    best = int(np.lexsort((-result.fitness, -successes))[0])
    return int(candidates[best]), int(successes[best]), float(result.fitness[best])


def save_dense_champion(approach, seed, population, index, champion_dir=config.CHAMPIONS_DIR):
    """Final dense controller (spec §6, revision 3 §6.4): a Keras file and the float64 weights (.npy); the
    architecture is the Keras file's. Returns the Keras file's path."""
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = population.weights[int(index)]
    nets.to_keras(flat, population.hidden, population.activation).save(path)
    np.save(paths.champion_path(approach, seed, ".npy", champion_dir), np.asarray(flat, dtype=float))
    return path


def clear_champion(approach, seed, champion_dir=config.CHAMPIONS_DIR):
    """Remove a final controller left by an earlier run (.keras, .npy and a NEAT genome's .json), so an interrupted
    run leaves none behind."""
    paths.remove(*(paths.champion_path(approach, seed, suffix, champion_dir) for suffix in (".keras", ".npy", ".json")))


# ----- revision 2.0 helpers, used by ga.py and es.py until the evolution engine replaces them (plan Task 6) -----

def _evaluate_track(weights, seed, level):
    """Fitness, finishing and finishing at pace (spec §9.1) of every network over config.LAPS laps of one track."""
    track = get_track(seed, level)
    result = sim.simulate(track, nets.population_policy(weights), len(weights), laps=config.LAPS)
    pace_limit = config.PACE_FACTOR * track.expert_lap_time * config.LAPS + config.TIME_EPS
    at_pace = result.finished & (np.nan_to_num(result.finish_time, nan=np.inf) <= pace_limit)
    return result.fitness, result.finished, at_pace


def evaluate_weights(weights, seeds, level, n_jobs=1, backend="loky"):
    """Mean fitness over the tracks, whether each individual finished all of them, and whether it finished all of
    them at pace. The per-track results are combined in track order, so the result does not depend on n_jobs."""
    n = len(weights)
    tasks = split_tasks(n, seeds, n_jobs)
    if n_jobs == 1 or len(tasks) == 1:
        parts = [_evaluate_track(weights[idx], seed, level) for _, seed, idx in tasks]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_evaluate_track)(weights[idx], seed, level) for _, seed, idx in tasks)
    per_track = np.zeros((len(seeds), n))
    finished = np.ones(n, dtype=bool)
    at_pace = np.ones(n, dtype=bool)
    for (k, _, idx), (fitness, done, pace) in zip(tasks, parts):
        per_track[k, idx] = fitness
        finished[idx] &= done
        at_pace[idx] &= pace
    total = np.zeros(n)
    for k in range(len(seeds)):
        total += per_track[k]
    return total / len(seeds), finished, at_pace


def finish_fractions(finished, is_kept):
    """Fraction of kept and of new individuals that finished; nan for an empty group."""
    kept = float(finished[is_kept].mean()) if is_kept.any() else float("nan")
    new = float(finished[~is_kept].mean()) if (~is_kept).any() else float("nan")
    return kept, new


def make_report(*, approach, generation, level, seeds, weights, fitness, finished, is_kept, kept_finish,
                new_finish, kept_pace, new_pace, step_size, evals, elapsed, leveled_up,
                whole_pace=float("nan"), strict_success=float("nan"), tie_fraction=float("nan")):
    """One generation's results, for logging and for the training view."""
    return {"approach": approach, "generation": int(generation), "level": int(level), "seeds": list(seeds),
            "weights": weights, "hidden": list(config.BASELINE_HIDDEN), "activation": "relu",
            "fitness": fitness, "finished": finished, "is_kept": is_kept,
            "kept_finish": kept_finish, "new_finish": new_finish, "kept_pace": kept_pace, "new_pace": new_pace,
            "whole_pace": float(whole_pace), "n_kept": int(np.sum(is_kept)),
            "step_size": float(step_size), "strict_success": float(strict_success),
            "tie_fraction": float(tie_fraction),
            "best": float(fitness.max()), "mean": float(fitness.mean()), "worst": float(fitness.min()),
            "evals": int(evals), "elapsed": float(elapsed), "leveled_up": bool(leveled_up)}


class RunLogger:
    """Writes one CSV row and the best network after every generation, and a summary at the end."""
    FIELDS = ["generation", "level", "evals", "best", "mean", "worst", "whole_pace", "n_kept", "kept_finish",
              "new_finish", "kept_pace", "new_pace", "step_size", "strict_success", "tie_fraction", "seeds", "elapsed", "leveled_up"]

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for stale in self.dir.glob("best_gen*.npy"):    # left over from an earlier run in the same directory
            stale.unlink()
        paths.remove(self.dir / "summary.json")
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
    fitness, _, _ = evaluate_weights(candidates, list(seeds), config.MAX_LEVEL, n_jobs=n_jobs)
    best = int(np.argmax(fitness))
    return candidates[best], float(fitness[best])


def save_champion(approach, seed, flat, champion_dir=config.CHAMPIONS_DIR):
    """Final controller as a Keras file and as float64 weights (spec §6). Returns the Keras file's path."""
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    nets.to_keras(flat).save(path)
    np.save(paths.champion_path(approach, seed, ".npy", champion_dir), np.asarray(flat, dtype=float))
    return path
