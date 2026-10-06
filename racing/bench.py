"""Measures the time of one evolutionary generation (spec §9.4).

Every car drives the expert policy, so all of them complete a full lap: the worst case for runtime.
Run with:  ~/miniconda3/bin/python -m racing.bench
"""
import time

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


def estimate(pop, jobs):
    """Wall time for one generation of `pop` cars on three tracks, split across `jobs` processes."""
    if jobs == 1:
        return sum(time_full_laps(seed, pop) for seed in SEEDS)
    chunk = max(pop // jobs, 1)
    tasks = [delayed(time_full_laps)(seed, chunk) for seed in SEEDS for _ in range(jobs)]
    Parallel(n_jobs=jobs)(tasks[:jobs])                    # warm-up: each worker builds its tracks
    start = time.perf_counter()
    Parallel(n_jobs=jobs)(tasks)
    return time.perf_counter() - start


def main():
    for seed in SEEDS:
        get_track(seed, config.MAX_LEVEL)
    print(f"One generation of {config.GA_POP} cars, every car completing a lap on 3 level-4 tracks:")
    for jobs in (1, 4, 8, 12, 16):
        print(f"  {jobs:2d} process(es): {estimate(config.GA_POP, jobs):6.1f} s")


if __name__ == "__main__":
    main()
