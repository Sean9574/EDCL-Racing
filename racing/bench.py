"""Measures the time of one evolutionary generation (spec §9.4).

Every car drives the expert policy, so all of them complete a full lap: the worst case for runtime.
Run with:  ~/miniconda3/bin/python -m racing.bench
"""
import time

import numpy as np
from joblib import Parallel, delayed

from racing import config, expert, sim
from racing.tracks import get_track

SEEDS = (0, 1, 2)


def time_full_laps(seed, n):
    """Seconds for n cars to each drive one full level-4 lap on the given training seed."""
    track = get_track(seed, config.MAX_LEVEL)
    start = time.perf_counter()
    sim.simulate(track, expert.expert_policy(track), n)
    return time.perf_counter() - start


def estimate(pop, jobs, repeats=3):
    """Wall time for one generation of `pop` cars on three tracks, split across `jobs` processes.

    For multi-process runs: times are measured after an untimed warm-up pass to pre-cache
    all three level-4 tracks in each worker. joblib dispatch and pickling overhead is included;
    one-time track construction is excluded (happens in warm-up). Times are reported as the
    minimum of `repeats` timed runs to account for system load variation."""
    if jobs == 1:
        return sum(time_full_laps(seed, pop) for seed in SEEDS)

    # Distribute pop evenly across jobs (no truncation)
    chunks = np.array_split(np.arange(pop), jobs)
    tasks = [delayed(time_full_laps)(seed, len(chunk)) for seed in SEEDS for chunk in chunks]

    # Warm-up: untimed run to cache all tracks in each worker
    Parallel(n_jobs=jobs)(tasks)

    # Timed runs: measure steady state after warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        Parallel(n_jobs=jobs)(tasks)
        times.append(time.perf_counter() - start)

    return min(times)


def main():
    for seed in SEEDS:
        get_track(seed, config.MAX_LEVEL)
    print(f"One generation of {config.GA_POP} cars, every car completing a lap on 3 level-4 tracks:")
    for jobs in (1, 4, 8, 12, 16):
        print(f"  {jobs:2d} process(es): {estimate(config.GA_POP, jobs):6.1f} s")


if __name__ == "__main__":
    main()
