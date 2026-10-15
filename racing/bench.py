"""Benchmarks (spec §9.10): the time of one evolutionary generation for each number of worker processes, and the
time of one Keras training epoch for each choice of steps_per_execution and jit_compile.

Run with:  python -m racing.bench              (both)
           python -m racing.bench generation   (generation time only)
           python -m racing.bench keras        (Keras only; needs results/dataset.npz)
Run on an otherwise idle machine.
"""
import os
import sys
import time

from joblib import Parallel, delayed

from racing import config, expert, fastsim, population, sim
from racing.tracks import get_track

SEEDS = (0, 1, 2)
JOB_COUNTS = (1, 4, 6, 8, 12, 16, 20, 24)      # up to a 24-thread desktop (spec §3.3)
KERAS_STEPS = (1, 4, 16, 64)
BACKEND = "loky"


def time_full_laps(seed, n):
    """Seconds for n cars to each drive one full level-4 lap on the given training seed."""
    track = get_track(seed, config.MAX_LEVEL)
    start = time.perf_counter()
    sim.simulate(track, expert.expert_policy(track), n)
    return time.perf_counter() - start


def run_tasks(tasks, jobs):
    if jobs == 1:
        return [time_full_laps(seed, n) for seed, n in tasks]
    return Parallel(n_jobs=jobs, backend=BACKEND)(delayed(time_full_laps)(seed, n) for seed, n in tasks)


def generation_time(pop, jobs, repeats=3):
    """Best of `repeats` wall times for one generation of `pop` cars on three level-4 tracks, divided into tasks
    exactly as population.evaluate_population divides them. Every car drives the expert, so every car completes a
    lap: the worst case. An untimed pass first builds the tracks in the worker processes."""
    tasks = [(seed, len(idx)) for _, seed, idx in population.split_tasks(pop, SEEDS, jobs)]
    run_tasks(tasks, jobs)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_tasks(tasks, jobs)
        times.append(time.perf_counter() - start)
    return min(times)


def keras_epoch_time(data, steps_per_execution, jit_compile, epochs=3):
    """Seconds per epoch for the baseline network with the given Keras settings, after one untimed epoch that
    traces and compiles the training function."""
    from racing import imitation
    from racing.backend import keras_cpu
    keras_cpu().utils.set_random_seed(0)
    model = imitation.build_network(list(config.BASELINE_HIDDEN), "relu", 0.0, config.LEARNING_RATE,
                                    steps_per_execution=steps_per_execution, jit_compile=jit_compile)
    args = {"x": data["X_train"], "y": data["y_train"], "validation_data": (data["X_val"], data["y_val"]),
            "batch_size": config.BATCH_SIZE, "verbose": 0}
    model.fit(epochs=1, **args)
    start = time.perf_counter()
    model.fit(epochs=epochs, **args)
    return (time.perf_counter() - start) / epochs


def job_counts(cpus=None):
    """The worker counts to time: JOB_COUNTS up to the number of logical processors."""
    if cpus is None:
        cpus = os.cpu_count() or 1
    return [jobs for jobs in JOB_COUNTS if jobs <= cpus]


def recommend(times):
    """The worker count with the shortest generation time; on a near tie (within 2%) the smaller count, which leaves
    the machine more headroom."""
    best = min(times.values())
    return min(jobs for jobs, seconds in times.items() if seconds <= best * 1.02)


def compiled_generation_time(pop, threads, repeats=3):
    """Best of `repeats` wall times for one worst-case generation on the compiled simulator (spec §5.7): pop expert
    cars, each completing a lap of each of the three level-4 tracks, in one call with `threads` threads."""
    tracks = [get_track(seed, config.MAX_LEVEL) for seed in SEEDS]
    limits = [sim.time_limit_for(track) for track in tracks]
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fastsim.simulate_expert(tracks, pop, limits, threads=threads)
        times.append(time.perf_counter() - start)
    return min(times)


def generation_main():
    """Times one worst-case generation for each worker or thread count on the simulator in use, and prints the
    command that selects the fastest."""
    for seed in SEEDS:
        get_track(seed, config.MAX_LEVEL)
    compiled = fastsim.available()
    unit = "thread(s)" if compiled else "process(es)"
    print(f"One generation of {config.GA_POP} cars, every car completing a lap on 3 level-4 tracks "
          f"({fastsim.description()} simulator):")
    times = {}
    for jobs in job_counts():
        times[jobs] = (compiled_generation_time if compiled else generation_time)(config.GA_POP, jobs)
        print(f"  {jobs:2d} {unit}: {times[jobs]:6.2f} s", flush=True)
    best = recommend(times)
    print(f"Fastest: {best} {unit} (current setting {config.N_JOBS}). Select it with:")
    print(f"  export EC_RACING_JOBS={best}")
    return times


def keras_main():
    from racing import imitation
    data = imitation.load_dataset()
    print(f"Baseline training, seconds per epoch ({len(data['y_train'])} rows, batch {config.BATCH_SIZE}):")
    for jit_compile in (True, False):
        for steps in KERAS_STEPS:
            seconds = keras_epoch_time(data, steps, jit_compile)
            print(f"  steps_per_execution {steps:3d}, jit_compile {str(jit_compile):5s}: {seconds:6.3f} s")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or "generation" in argv:
        generation_main()
    if not argv or "keras" in argv:
        keras_main()


if __name__ == "__main__":
    main()
