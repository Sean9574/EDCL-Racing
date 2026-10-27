"""Pilot of the stall boost (spec §12.2): does the boost let the stalled ES run of seed 0 (revision 3.2, 11-12 October)
break its stall? Each variant continues that run from its final population and its random generator's state, with
the speed push's patience restarting at the next target (600 generations in the version 3.6 pilot; the current
default, version 3.9, is 1,000):
  boost  the stall boost as configured (spec §9.4): after 100 generations without a mastered target the decision
         layer mutates 3 times as hard and the steering sensitivity with probability 0.5 and standard deviation 0.2;
  plain  the same run with the boost set to its off values, which makes exactly the draws of a run without it.
The variants start from the same random state, so they differ only once the boost acts. Each starts over in
results/pilots/revision3.6/<variant>/ and adds a row to results/pilots/revision3.6/summary.md. The pilot is not part
of the reported results. The source run is the 15-action ES of revision 3.2, so under the 45 actions of version 3.9
the pilot exits with a message before it deletes or writes anything.

Run both variants side by side, in two terminals:
    python scripts/boost_pilot.py boost
    python scripts/boost_pilot.py plain
Options: --source PATH (the checkpoint to continue; default the archived ES seed 0 of revision 3.2), --out DIR,
--max-generations N (generations to run after the source's last; default: until the run ends, which from version
3.15 is after hardening, not when the push ends).
The source's settings record is not checked: it was made by earlier code. The pilot's own checkpoint carries the
current settings, and evolution.run resumes it.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racing import checkpoint, config, es, evaluate, evolution, nets, paths  # noqa: E402
from racing.curriculum import MasteryTracker  # noqa: E402

VARIANTS = ("boost", "plain")
SOURCE = config.RESULTS_DIR / "archive" / "revision3.2" / "runs" / "es" / "seed0" / checkpoint.CHECKPOINT_NAME
OUT = config.PILOTS_DIR / "revision3.6"
COPIED = ("generation", "evals", "rng", "seeds", "levels", "tightenings", "start_level", "tracker", "method")
COLUMNS = ("variant", "source reduction", "reduction at end", "generations", "stalls broken", "minutes")


def reduction_text(value):
    """A target reduction as a percentage; '–' before the goal."""
    return "–" if value is None or np.isnan(value) else f"{100 * value:.1f}%"


def load_source(path):
    """The source checkpoint's state; exits naming any field this pilot copies that an older checkpoint lacks."""
    state = checkpoint.load(path)
    if state is None:
        sys.exit(f"error: no checkpoint at {path}")
    missing = [name for name in COPIED if name not in state]
    if missing:
        sys.exit(f"error: the checkpoint {path} lacks {', '.join(missing)} (made by an earlier version)")
    weights = np.asarray(state["method"]["weights"])
    if weights.ndim != 2 or weights.shape[1] != nets.n_params(tuple(state["method"]["hidden"])):
        sys.exit(f"error: the networks of {path} do not fit this version's {config.N_INPUTS} inputs and "
                 f"{config.N_ACTIONS} actions (a run of an earlier revision), so the pilot cannot continue it; "
                 "nothing was deleted or written")
    return state


def start(variant, state, out):
    """Write the variant's fresh checkpoint and log in `out`; returns the method, whose population is the source's."""
    source = state["method"]
    method = es.ES(approach="es", pop_size=len(source["weights"]), hidden=source["hidden"],
                   activation=source["activation"])
    method.load_state(source)
    tracker = MasteryTracker.from_state(state["tracker"])
    tracker.outcome, tracker.since_mastered = None, 0       # the push resumes at the next target with full patience
    if variant == "plain":
        config.STALL_BOOST = 1.0                           # factors of 1 and the base setup mutation: no change
        config.STALL_STEER_P = config.SETUP_MUTATION_P
        config.STALL_STEER_SD = config.SETUP_MUTATION_SD
    paths.remove(out)                                      # a pilot always starts over
    checkpoint.save(out / checkpoint.CHECKPOINT_NAME, {
        "settings": checkpoint.settings_record(), "generation": state["generation"], "evals": state["evals"],
        "elapsed": 0.0, "rng": state["rng"], "seeds": state["seeds"], "tracker": tracker.state(),
        "method": method.state(), "levels": state["levels"], "tightenings": state["tightenings"],
        "start_level": state["start_level"]})
    evolution.start_log(out / evolution.LOG_NAME, evolution.LOG_FIELDS + list(method.log_fields()))
    return method


def markdown_row(values):
    return "| " + " | ".join(str(v) for v in values) + " |"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stall-boost pilot: continue the stalled ES run of seed 0.")
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--source", type=Path, default=SOURCE, help="checkpoint to continue")
    parser.add_argument("--out", type=Path, default=OUT, help="folder of the variants (default %(default)s)")
    parser.add_argument("--max-generations", type=int, help="generations to run after the source's last")
    args = parser.parse_args(argv)
    state = load_source(args.source)
    folder = args.out / args.variant
    method = start(args.variant, state, folder)
    limit = None if args.max_generations is None else state["generation"] + args.max_generations
    summary = evolution.run(method, 0, folder, champion_dir=None, n_jobs=config.N_JOBS, max_generations=limit)
    log = pd.read_csv(folder / evolution.LOG_NAME)
    row = (args.variant, reduction_text(MasteryTracker.from_state(state["tracker"]).target_reduction),
           reduction_text(summary["target_reduction"]), summary["generations"] - state["generation"],
           int(evaluate.stalls_broken(log)), round(summary["wall_time"] / 60, 1))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "summary.md", "a") as f:
        f.write(f"## {datetime.now().isoformat(timespec='seconds')}, {args.variant}\n\n")
        f.write(markdown_row(COLUMNS) + "\n" + markdown_row("---" for _ in COLUMNS) + "\n")
        f.write(markdown_row(row) + "\n\n")
    print(dict(zip(COLUMNS, row)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
