"""Calibration pilots (spec §3.3): the GA mutation-strength range, the ES step-size behavior under the tie rule and
the pace requirement, and the seeded ES. Each pilot runs seed 0 at the reported settings with a shortened budget,
writes to results/pilots/ and adds a row to results/pilots/summary.md. Pilots are not part of the reported results.

Run with:  ~/miniconda3/bin/python scripts/pilots.py              (all pilots)
           ~/miniconda3/bin/python scripts/pilots.py ga es       (selected pilots: ga, es, seeded)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racing import config, es, ga, imitation  # noqa: E402
from racing.expert import collect_dataset  # noqa: E402

GA_CANDIDATES = (0.5, 0.2, 0.1)


def level_ups(log):
    """Generations at which the level advanced, as text."""
    return " ".join(str(g) for g in log.loc[log["leveled_up"].astype(bool), "generation"]) or "none"


def pilot_ga(s_max, out_root, generations=100, pop_size=config.GA_POP, kept=config.GA_KEPT, n_jobs=config.N_JOBS):
    out = Path(out_root) / f"ga_smax{s_max}"
    result = ga.run_ga(pop_size=pop_size, kept=kept, budget=generations * pop_size, seed=0, out_dir=out,
                       champion_dir=None, n_jobs=n_jobs, s_max=s_max)
    log = pd.read_csv(out / "log.csv")
    return {"pilot": f"GA s_max={s_max}", "generations": result["generations"],
            "level_reached": result["level_reached"], "level_ups": level_ups(log),
            "new_finish_last": round(float(log["new_finish"].iloc[-1]), 2),
            "new_pace_last": round(float(log["new_pace"].iloc[-1]), 2),
            "best_last": round(float(log["best"].iloc[-1]), 3), "minutes": round(result["wall_time"] / 60, 1)}


def pilot_es(out_root, generations=100, mu=config.ES_MU, lam=config.ES_LAMBDA, n_jobs=config.N_JOBS, initial=None,
             name="ES", folder="es"):
    out = Path(out_root) / folder
    approach = "es" if initial is None else "es_seeded"
    result = es.run_es(mu=mu, lam=lam, budget=generations * (mu + lam), seed=0, out_dir=out, champion_dir=None,
                       n_jobs=n_jobs, initial=initial, approach=approach)
    log = pd.read_csv(out / "log.csv")
    recent = log.tail(20)
    return {"pilot": name, "generations": result["generations"], "level_reached": result["level_reached"],
            "level_ups": level_ups(log), "new_pace_last": round(float(log["new_pace"].iloc[-1]), 2),
            "sigma_min": float(log["step_size"].min()), "sigma_last": float(log["step_size"].iloc[-1]),
            "strict_success": round(float(recent["strict_success"].mean()), 3),
            "tie_fraction": round(float(recent["tie_fraction"].mean()), 3),
            "best_last": round(float(log["best"].iloc[-1]), 3), "minutes": round(result["wall_time"] / 60, 1)}


def pilot_seeded(out_root, generations=100, mu=config.ES_MU, lam=config.ES_LAMBDA, n_jobs=config.N_JOBS):
    """Trains a seed-0 baseline into the pilot folder (the reported one may not exist yet), then runs the seeded ES."""
    folder = Path(out_root) / "seeded"
    dataset = config.DATASET_PATH
    if not dataset.exists():
        dataset = folder / "dataset.npz"
        if not dataset.exists():
            collect_dataset(dataset, n_jobs=n_jobs)
    imitation.train_baseline(imitation.load_dataset(dataset), out_dir=folder / "baseline",
                             champion_dir=folder / "champions", seed=0)
    initial = es.trained_baseline(0, champion_dir=folder / "champions")
    return pilot_es(folder, generations, mu, lam, n_jobs, initial=initial, name="Seeded ES", folder="es_seeded")


def main(argv=None):
    selected = (sys.argv[1:] if argv is None else argv) or ["ga", "es", "seeded"]
    out_root = config.PILOTS_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    if "ga" in selected:
        for s_max in GA_CANDIDATES:
            rows.append(pilot_ga(s_max, out_root))
            print(rows[-1], flush=True)
    if "es" in selected:
        rows.append(pilot_es(out_root))
        print(rows[-1], flush=True)
    if "seeded" in selected:
        rows.append(pilot_seeded(out_root))
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    try:
        text = table.to_markdown(index=False)
    except ImportError:                          # pandas needs the optional tabulate package for Markdown
        text = table.to_string(index=False)
    with open(out_root / "summary.md", "a") as f:
        f.write(text + "\n\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
