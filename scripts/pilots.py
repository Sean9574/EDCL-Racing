"""Pilot runs of revision 3.10 (spec §12.2, §14): short runs of each evolutionary method from Easy with seed 0,
which show whether the 90% requirement can be reached with the reported settings, now with a racing line on every
track, 18 inputs (6 about the line) and the expert driving the line at its measured grip share, before the reported
run starts. Each pilot starts over in
results/pilots/revision3.10/<name>/ and adds a row to results/pilots/revision3.10/summary.md, with the population's mean
sensor length and steering sensitivity and, for the GA, its network size at the end. Pilots are not part of the
reported results.

Run with:  python scripts/pilots.py                       (ga, es, neat and seeded; 300 generations each)
           python scripts/pilots.py ga neat               (selected pilots)
           python scripts/pilots.py --generations 500 es
The seeded pilot needs the DAgger-refined baseline of seed 0 (notebook 03) and is skipped without it; a baseline
of an earlier revision has another number of inputs, so run `python scripts/pilots.py ga es neat` before notebook 03.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racing import config, es, ga, neat, paths  # noqa: E402

PILOTS = ("ga", "es", "neat", "seeded")
NAMES = {"ga": "GA", "es": "ES", "neat": "NEAT", "seeded": "Seeded ES"}
OUT = config.PILOTS_DIR / "revision3.10"


def level_ups(log):
    """When each level was reached and the mastery goal met, for example '2@25 3@74 4@94 goal@120'."""
    marks = []
    for row in log.itertuples(index=False):
        if row.event == "level_up":
            marks.append(f"{int(row.level) + 1}@{int(row.generation)}")
        elif row.event == "goal":
            marks.append(f"goal@{int(row.generation)}")
    return " ".join(marks) or "none"


def last_value(log, column, digits):
    """The last generation's `column` rounded, or None (a blank in the table) when the log has no such column
    (spec §12.2: the network size of the GA's pilot)."""
    if column not in log or log[column].isna().all():
        return None
    value = round(float(log[column].dropna().iloc[-1]), digits)
    return int(value) if digits == 0 else value


def last_target(summary):
    """The speed push's last mastered target in percent shorter than the starting target, or None (a blank in the
    table) when no target was mastered."""
    reduction = summary.get("target_reduction")
    return None if reduction is None else round(100 * float(reduction), 1)


def run_pilot(name, out, max_generations, n_jobs, pop_size):
    common = {"seed": 0, "out_dir": out, "champion_dir": None, "n_jobs": n_jobs, "max_generations": max_generations}
    if pop_size is not None:
        common["pop_size"] = pop_size
    if name == "ga":
        return ga.run_ga(**common)
    if name == "es":
        return es.run_es(**common)
    if name == "neat":
        return neat.run_neat(**common)
    return es.run_es(initial=es.dagger_baseline(0), approach="es_seeded", **common)


def pilot(name, out_root=None, max_generations=300, n_jobs=None, pop_size=None):
    """One pilot from Easy with seed 0, summarized as one table row."""
    out_root = Path(OUT if out_root is None else out_root)
    n_jobs = config.N_JOBS if n_jobs is None else n_jobs
    if name == "seeded" and not paths.stage_path("baseline", 0, "dagger").exists():
        return {"pilot": NAMES[name], "note": "skipped: the DAgger-refined baseline of seed 0 does not exist yet "
                                              "(run notebook 03 first)"}
    out = out_root / name
    paths.remove(out)                       # a pilot always starts over; evolutionary runs otherwise resume
    summary = run_pilot(name, out, max_generations, n_jobs, pop_size)
    log = pd.read_csv(out / "log.csv")
    return {"pilot": NAMES[name], "generations": int(summary["generations"]),
            "level_reached": int(summary["level_reached"]), "level_ups": level_ups(log),
            "min_share_last20": round(float(log["min_share"].tail(20).mean()), 3),
            "sensor_last": round(float(log["sensor_mean"].iloc[-1]), 1),
            "target_reduction": last_target(summary),
            "steer_last": round(float(log["steer_mean"].iloc[-1]), 2),
            "mean hidden units": last_value(log, "mean_nodes", 1),
            "largest parameters": last_value(log, "max_params", 0),
            "minutes": round(float(summary["wall_time"]) / 60, 1)}


def markdown(frame):
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Revision 3.1 pilot runs from Easy with seed 0.")
    parser.add_argument("pilots", nargs="*", help=f"any of {', '.join(PILOTS)} (default: all)")
    parser.add_argument("--generations", type=int, default=300, help="generations per pilot (default 300)")
    args = parser.parse_args(argv)
    selected = args.pilots or list(PILOTS)
    unknown = [name for name in selected if name not in PILOTS]
    if unknown:
        print(f"error: unknown pilot(s) {', '.join(unknown)}; choose from {', '.join(PILOTS)}", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in selected:
        rows.append(pilot(name, OUT, args.generations))
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    with open(OUT / "summary.md", "a") as f:
        f.write(f"## {datetime.now().isoformat(timespec='seconds')}, {args.generations} generations\n\n")
        f.write(markdown(table) + "\n\n")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
