"""Shared helpers for the evolutionary approaches (spec §9.1-9.5): time-trial evaluation, final selection
and saved controllers."""
import json
from dataclasses import dataclass

import numpy as np

from racing import config, fastsim, nets, paths, sim, tracks
from racing import setup as car_setup
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
    """(track position, track seed, individual indices) tasks for one evaluation (spec §9.10). Each track's
    individuals are split into n_jobs // len(seeds) shares (at least one), so a task drives many cars on one track."""
    shares = min(max(1, n_jobs // len(seeds)), max(n, 1))
    parts = np.array_split(np.arange(n), shares)
    return [(k, seed, part) for k, seed in enumerate(seeds) for part in parts]


@dataclass(eq=False)
class TrialResult:
    """One generation's time trials (spec §9.2): one row per map, one column per car."""
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


def _trial_values(result):
    """(fitness, success, laps done within the target time, finish time) of a time trial's RunResult (spec §9.2)."""
    return (result.fitness, result.finished, np.minimum(result.laps_done, config.LAPS),
            np.where(result.finished, result.finish_time, np.nan))


def _time_trial(population, seed, level, factor):
    """Fitness, success, laps in time and finish time of every car over config.LAPS laps of one map, ending at the
    target time (spec §9.2). Every car drives with its own setup (spec §9.11)."""
    track = get_track(seed, level)
    return _trial_values(sim.simulate(track, population.policy(), len(population),
                                      time_limit=sim.target_time(track, factor), laps=config.LAPS,
                                      setup=population.setup))


def evaluate_population(population, seeds, level, factor, n_jobs=1, backend="loky"):
    """Time trials of every car on every map (spec §9.2). Dense and NEAT populations run on the compiled simulator
    when it is available, in one call with config.thread_count() threads (spec §3.3, §5.7); otherwise the per-map
    tasks run on numpy in n_jobs worker processes. The per-map results are combined in map order, so the result
    depends on neither."""
    n = len(population)
    if fastsim.supports(population) and fastsim.available():
        tracks = [get_track(seed, level) for seed in seeds]
        results = fastsim.simulate(tracks, population, [sim.target_time(t, factor) for t in tracks],
                                   laps=config.LAPS, threads=config.thread_count())
        tasks = [(k, seed, np.arange(n)) for k, seed in enumerate(seeds)]
        parts = [_trial_values(result) for result in results]
    else:
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
    """The final controller (spec §9.5): of the candidate indices, the car with the most successes on the
    validation maps at target factor `factor`, then the highest mean fitness, then the earliest candidate. Returns
    (index into population, validation successes, mean validation fitness)."""
    candidates = np.asarray(candidates, dtype=np.int64)
    result = evaluate_population(population.subset(candidates), list(seeds), config.MAX_LEVEL, factor, n_jobs=n_jobs)
    successes = result.success.sum(axis=0)
    best = int(np.lexsort((-result.fitness, -successes))[0])
    return int(candidates[best]), int(successes[best]), float(result.fitness[best])


def save_dense_champion(approach, seed, population, index, champion_dir=config.CHAMPIONS_DIR):
    """Final dense controller (spec §6, §9.5): a Keras file, the float64 weights (.npy) and the car's
    setup (<seed>.setup.json), of a nets.DensePopulation or nets.ShapedPopulation; the architecture is the Keras
    file's. Returns the Keras file's path."""
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat, hidden = population.network(index)
    nets.to_keras(flat, hidden, population.activation).save(path)
    np.save(paths.champion_path(approach, seed, ".npy", champion_dir), np.asarray(flat, dtype=float))
    paths.champion_path(approach, seed, ".setup.json", champion_dir).write_text(
        json.dumps(car_setup.to_dict(population.setup[int(index)])))
    return path


def clear_champion(approach, seed, champion_dir=config.CHAMPIONS_DIR):
    """Remove a final controller left by an earlier run (.keras, .npy, a NEAT genome's .json and the setup's
    .setup.json), so an interrupted run leaves none behind."""
    paths.remove(*(paths.champion_path(approach, seed, suffix, champion_dir)
                   for suffix in (".keras", ".npy", ".json", ".setup.json")))
