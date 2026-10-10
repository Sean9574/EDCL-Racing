"""One evolutionary run of the experiment, trained outside its notebook: python -m racing.train ga 1

scripts/run_notebooks.py --runs starts several of these at once, on one or more machines (scripts/fleet.py). Each
makes exactly the call of its notebook's loop (tests/test_train.py checks this), so a run trained here is the run the
notebook would train; the notebook then finds it complete and only reports it.
"""
import argparse
import sys
from pathlib import Path

from racing import config

NOTEBOOKS = {"ga": "04_ga.ipynb", "es": "05_es.ipynb", "es_seeded": "06_es_seeded.ipynb", "neat": "07_neat.ipynb"}


def train(approach, seed):
    """Train run `seed` of `approach` as its notebook does; returns the run's summary."""
    if approach == "ga":
        from racing import ga
        return ga.run_ga(seed=seed, n_jobs=config.N_JOBS)
    if approach == "es":
        from racing import es
        return es.run_es(seed=seed, n_jobs=config.N_JOBS)
    if approach == "es_seeded":
        from racing import es
        return es.run_es(seed=seed, initial=es.dagger_baseline(seed), approach="es_seeded", n_jobs=config.N_JOBS)
    if approach == "neat":
        from racing import neat
        return neat.run_neat(seed=seed, n_jobs=config.N_JOBS)
    raise ValueError(f"no training run for {approach!r}; choose one of {', '.join(NOTEBOOKS)}")


def parse_run(text):
    """'ga:1' -> ('ga', 1). Raises ValueError for an approach without a training run or a seed outside the
    experiment's run seeds."""
    approach, _, seed = str(text).strip().partition(":")
    if approach not in NOTEBOOKS:
        raise ValueError(f"{text!r}: the approach must be one of {', '.join(NOTEBOOKS)}")
    if not seed.isdigit() or int(seed) not in config.RUN_SEEDS:
        raise ValueError(f"{text!r}: the seed must be one of {', '.join(map(str, config.RUN_SEEDS))}")
    return approach, int(seed)


def parse_runs(text):
    """'ga:1,neat:0' -> [('ga', 1), ('neat', 0)], each run once."""
    runs = [parse_run(part) for part in str(text).split(",") if part.strip()]
    if not runs:
        raise ValueError("no runs given; write them as approach:seed, for example ga:1,neat:0")
    repeated = sorted({f"{a}:{s}" for a, s in runs if runs.count((a, s)) > 1})
    if repeated:
        raise ValueError(f"each run may be named once: {', '.join(repeated)}")
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train one evolutionary run of the experiment.")
    parser.add_argument("approach", choices=list(NOTEBOOKS))
    parser.add_argument("seed", type=int, choices=list(config.RUN_SEEDS))
    parser.add_argument("--status", default=None, help="this run's status file (default: results/status.json)")
    args = parser.parse_args(argv)
    if args.status is not None:
        config.STATUS_PATH = Path(args.status)          # progress.write_status reads it at every write
    summary = train(args.approach, args.seed)
    print({key: summary.get(key) for key in ("approach", "seed", "generations", "target_reduction",
                                              "val_successes", "wall_time")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
