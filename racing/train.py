"""One run of the experiment, trained outside its notebook: python -m racing.train ga 1

scripts/run_notebooks.py --runs starts several of these at once, on one or more machines (scripts/fleet.py). Each
makes exactly the call of its notebook's loop (tests/test_train.py checks this), so a run trained here is the run the
notebook would train; the notebook then finds it complete and only reports it. A run of an imitation method first
trains its own imitation network, notebook 02's call for its seed, then refines and pushes it, notebook 03's call; an
evolutionary method's run is the run of notebooks 04 to 07. A Keras Tuner search is notebook 02's search call:
python -m racing.train search kt_random (spec §3.2).
"""
import argparse
import sys
from pathlib import Path

from racing import config

NOTEBOOKS = {**{approach: "03_dagger.ipynb" for approach in config.IMITATION_APPROACHES},
             "ga": "04_ga.ipynb", "es": "05_es.ipynb", "es_seeded": "06_es_seeded.ipynb", "neat": "07_neat.ipynb"}


def train(approach, seed):
    """Train run `seed` of `approach` as its notebooks do; returns the run's summary. A run of an imitation method
    first trains its own imitation network (notebook 02's call for its seed: the baseline's training, or the retraining
    of its search's chosen architecture), then refines it by DAgger and pushes it (notebook 03's call); a step finished
    before returns its record instead of training again (spec §3.2)."""
    if approach in config.IMITATION_APPROACHES:
        from racing import dagger, imitation
        data = imitation.load_dataset()
        if approach == "baseline":
            imitation.train_baseline(data, seed=seed)
        else:
            imitation.retrain(approach, data, seed=seed)
        return dagger.run(approach, seed, data, n_jobs=config.N_JOBS)
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


def search(approach):
    """The Keras Tuner search of `approach`, notebook 02's call (spec §3.2); returns its summary. A finished search
    returns its record instead of searching again."""
    if approach not in config.TUNER_APPROACHES:
        raise ValueError(f"no search for {approach!r}; choose one of {', '.join(config.TUNER_APPROACHES)}")
    from racing import imitation
    return imitation.run_tuner(approach, imitation.load_dataset())


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
    argv = sys.argv[1:] if argv is None else list(argv)
    searching = argv[:1] == ["search"]
    parser = argparse.ArgumentParser(description="Train one run of the experiment (python -m racing.train ga 1) or run "
                                                 "one Keras Tuner search (python -m racing.train search kt_random).")
    if searching:
        parser.add_argument("search", choices=["search"])
        parser.add_argument("approach", choices=list(config.TUNER_APPROACHES))
    else:
        parser.add_argument("approach", choices=list(NOTEBOOKS))
        parser.add_argument("seed", type=int, choices=list(config.RUN_SEEDS))
    parser.add_argument("--status", default=None, help="this job's status file (default: results/status.json)")
    args = parser.parse_args(argv)
    if args.status is not None:
        config.STATUS_PATH = Path(args.status)          # progress.write_status reads it at every write
    if searching:
        summary, keys = search(args.approach), ("approach", "trials", "chosen_params", "wall_time")
    else:
        summary, keys = train(args.approach, args.seed), ("approach", "seed", "generations", "target_reduction",
                                                          "val_successes", "wall_time")
    print({key: summary.get(key) for key in keys}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
