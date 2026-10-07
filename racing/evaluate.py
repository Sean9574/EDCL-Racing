"""Final evaluation (spec §11): driving on held-out tracks, test accuracy, size, speed, tables and figures.
Each method has one final controller per run seed; tables report the mean and standard deviation across seeds."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from racing import config, nets, paths, sim
from racing.tracks import get_track

DRIVE_COLUMNS = ["approach", "run_seed", "track", "finished", "crashed", "lap_time", "progress", "fitness"]


def _drive_track(drivers, track_seed, level):
    """One lap for every controller on one track, simulated together (cars never interact)."""
    track = get_track(track_seed, level)
    keys = list(drivers)
    result = sim.simulate(track, nets.mixed_policy([drivers[k] for k in keys]), len(keys))
    rows = []
    for i, (approach, run_seed) in enumerate(keys):
        rows.append({"approach": approach, "run_seed": run_seed, "track": track_seed,
                     "finished": bool(result.finished[i]), "crashed": bool(result.crashed[i]),
                     "lap_time": float(result.finish_time[i]), "progress": float(result.progress[i]),
                     "fitness": float(result.fitness[i])})
    return rows


def driving_results(drivers, track_seeds, level=config.MAX_LEVEL, n_jobs=1, backend="loky"):
    """One lap per controller per track. drivers: {(approach, run seed or trial id): DenseNet}. The tracks are
    distributed across worker processes; the result does not depend on n_jobs."""
    track_seeds = list(track_seeds)
    if not drivers:
        return pd.DataFrame(columns=DRIVE_COLUMNS)
    if n_jobs == 1 or len(track_seeds) == 1:
        parts = [_drive_track(drivers, seed, level) for seed in track_seeds]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_drive_track)(drivers, seed, level) for seed in track_seeds)
    return pd.DataFrame([row for part in parts for row in part], columns=DRIVE_COLUMNS)


def best_seeds(validation):
    """For each approach, the run seed with the highest mean validation fitness; the lowest seed wins ties."""
    best = {}
    for approach in config.APPROACHES:
        sub = validation[validation["approach"] == approach]
        if sub.empty:
            continue
        means = sub.groupby("run_seed")["fitness"].mean().sort_index()
        best[approach] = int(means.index[int(np.argmax(means.to_numpy()))])
    return best


def accuracy(model, x, y):
    predicted = np.argmax(model.predict(x, batch_size=4096, verbose=0), axis=1)
    return float(np.mean(predicted == y))


def inference_time(model, x, repeats=5):
    """Seconds per sample for Keras predict on up to 10,000 rows in one batch, averaged over `repeats` runs. The
    warm-up is the same call as the timed ones, so no tracing or compilation is timed (spec §11)."""
    x = x[:10000]
    model.predict(x, batch_size=len(x), verbose=0)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(x, batch_size=len(x), verbose=0)
        times.append(time.perf_counter() - start)
    return float(np.mean(times) / len(x))


def single_sample_latency(model, x, calls=1000):
    """Median seconds for one call of the model on a single row (spec §11)."""
    rows = [x[i % len(x)][None, :] for i in range(calls)]
    model(rows[0], training=False)
    times = []
    for row in rows:
        start = time.perf_counter()
        model(row, training=False)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def metrics_per_seed(models, data, runs_dir, latency_calls=1000):
    """One row per final controller: test accuracy, size, speed, training cost and curriculum level reached."""
    rows = []
    for (approach, run_seed), model in models.items():
        summary = read_json(paths.run_dir(approach, run_seed, runs_dir) / "summary.json")
        search = float("nan")
        if approach in config.TUNER_APPROACHES:
            search = read_json(paths.search_dir(approach, runs_dir) / "summary.json").get("wall_time", float("nan"))
        rows.append({"approach": approach, "run_seed": run_seed,
                     "accuracy": accuracy(model, data["X_test"], data["y_test"]),
                     "params": int(model.count_params()),
                     "batch_us": inference_time(model, data["X_test"]) * 1e6,
                     "single_ms": single_sample_latency(model, data["X_test"], latency_calls) * 1e3,
                     "training_s": float(summary.get("wall_time", float("nan"))), "search_s": float(search),
                     "level_reached": float(summary.get("level_reached", float("nan")))})
    return pd.DataFrame(rows, columns=["approach", "run_seed", "accuracy", "params", "batch_us", "single_ms",
                                       "training_s", "search_s", "level_reached"])


def mean_std(values, digits):
    """'mean ± std' across seeds (sample standard deviation); the value alone if there is one value or all are
    equal; '–' if there is none."""
    values = [float(v) for v in values if not np.isnan(float(v))]
    if not values:
        return "–"
    if len(values) == 1 or all(v == values[0] for v in values):
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def comparison_table(metrics):
    """Spec Table 5, required comparison."""
    rows = []
    for approach in config.APPROACHES:
        sub = metrics[metrics["approach"] == approach]
        if sub.empty:
            continue
        rows.append({"Method": config.LABELS[approach], "Seeds": len(sub),
                     "Test accuracy": mean_std(sub["accuracy"], 4),
                     "Parameters": mean_std(sub["params"], 0),
                     "Batch inference (µs/sample)": mean_std(sub["batch_us"], 3),
                     "Single-sample latency (ms)": mean_std(sub["single_ms"], 3),
                     "Search time (s)": mean_std(sub["search_s"].iloc[:1], 1),
                     "Training time (s)": mean_std(sub["training_s"], 1)})
    return pd.DataFrame(rows)


def driving_table(per_track, metrics):
    """Spec Table 5, driving performance: per seed, then mean ± std across seeds."""
    rows = []
    n_tracks = per_track["track"].nunique()
    for approach in config.APPROACHES:
        sub = per_track[per_track["approach"] == approach]
        if sub.empty:
            continue
        completed, lap_time, crashes = [], [], []
        for _, runs in sub.groupby("run_seed"):
            done = runs[runs["finished"].astype(bool)]
            completed.append(float(runs["finished"].sum()))
            lap_time.append(float(done["lap_time"].mean()) if len(done) else float("nan"))
            crashes.append(float(runs["crashed"].sum()))
        level = "n/a"
        if approach in config.EVOLUTION_APPROACHES:
            levels = metrics[metrics["approach"] == approach]["level_reached"].dropna()
            level = ", ".join(str(int(v)) for v in levels) if len(levels) else "n/a"
        rows.append({"Method": config.LABELS[approach], f"Tracks completed (of {n_tracks})": mean_std(completed, 1),
                     "Mean lap time (s)": mean_std(lap_time, 2), "Crashes": mean_std(crashes, 1),
                     "Level reached (per seed)": level})
    return pd.DataFrame(rows)


def search_cost_table(runs_dir):
    """Spec Table 5, search cost of each Keras Tuner method."""
    rows = []
    for approach in config.TUNER_APPROACHES:
        summary = read_json(paths.search_dir(approach, runs_dir) / "summary.json")
        if not summary:
            continue
        rows.append({"Method": config.LABELS[approach], "Trials": summary["trials"],
                     "Configurations": summary["configurations"], "Total epochs": summary["total_epochs"],
                     "Trials at the epoch limit": summary["trials_at_limit"],
                     "Search time (s)": round(summary["wall_time"], 1)})
    return pd.DataFrame(rows)


def driving_selected_table(runs_dir=None, val_seeds=config.VAL_SEEDS, test_seeds=config.TEST_SEEDS, n_jobs=1):
    """Secondary analysis (spec §8.3): for each Keras Tuner method, the exported model (the most-trained of each
    configuration) with the highest mean validation fitness, and its driving on the test tracks."""
    from racing.backend import keras_cpu
    keras = keras_cpu()
    rows = []
    for approach in config.TUNER_APPROACHES:
        search = paths.search_dir(approach, runs_dir)
        exported = sorted((search / "models").glob("*.keras"))
        if not exported or not (search / "trials.csv").exists():
            continue
        trials = pd.read_csv(search / "trials.csv", dtype={"trial_id": str, "config_id": str})
        drivers = {(approach, path.stem): nets.from_keras(keras.models.load_model(path)) for path in exported}
        validation = driving_results(drivers, val_seeds, n_jobs=n_jobs)
        means = validation.groupby("run_seed")["fitness"].mean()
        trial_id = str(means.idxmax())
        test = driving_results({(approach, trial_id): drivers[(approach, trial_id)]}, test_seeds, n_jobs=n_jobs)
        record = trials[trials["trial_id"] == trial_id].iloc[0]
        done = test[test["finished"].astype(bool)]
        rows.append({"Method": config.LABELS[approach], "Trial": trial_id, "Parameters": int(record["params"]),
                     "Validation accuracy": round(float(record["val_accuracy"]), 4),
                     "Mean validation fitness": round(float(means.max()), 3),
                     f"Tracks completed (of {len(list(test_seeds))})": int(test["finished"].sum()),
                     "Mean lap time (s)": round(float(done["lap_time"].mean()), 2) if len(done) else float("nan"),
                     "Crashes": int(test["crashed"].sum())})
    return pd.DataFrame(rows)


def dataset_table(data_path):
    """Dataset size and collection time, a cost shared by the imitation methods and the seeded ES (spec §7.3)."""
    data_path = Path(data_path)
    info = read_json(data_path.with_name(f"{data_path.stem}_info.json"))
    rows = []
    for split, values in info.get("splits", {}).items():
        rows.append({"Split": split, "Tracks": values["tracks"], "Rows": values["rows"],
                     "Respawns": values["respawns"], "Collection time (s)": info["wall_time"],
                     "Worker processes": info["n_jobs"]})
    return pd.DataFrame(rows)


def to_markdown(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def tuner_trials(runs_dir):
    """Each search's trials table and its chosen trial's row. Trial ids stay strings: keras-tuner zero-pads them."""
    trials = {}
    for approach in config.TUNER_APPROACHES:
        search = paths.search_dir(approach, runs_dir)
        if (search / "trials.csv").exists():
            table = pd.read_csv(search / "trials.csv", dtype={"trial_id": str, "config_id": str})
            chosen = str(read_json(search / "summary.json").get("chosen_trial"))
            trials[approach] = (table, table[table["trial_id"] == chosen])
    return trials


def evolution_logs(runs_dir):
    """{approach: {run seed: log}} for the evolutionary runs whose logs exist."""
    logs = {}
    for approach in config.EVOLUTION_APPROACHES:
        found = {}
        for path in sorted((Path(runs_dir) / approach).glob("seed*/log.csv")):
            found[int(path.parent.name[len("seed"):])] = pd.read_csv(path)
        if found:
            logs[approach] = found
    return logs


def figures(runs_dir, out_dir):
    """Fitness curves and step sizes (one line per seed) and Keras Tuner trials; only figures with data are written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    logs = evolution_logs(runs_dir)
    if logs:
        fig, ax = plt.subplots(figsize=(7, 4))
        for approach, per_seed in logs.items():
            color = config.COLORS[approach]
            for k, (seed, log) in enumerate(per_seed.items()):
                ax.plot(log["evals"], log["best"], color=color, linewidth=1.0, alpha=0.85,
                        label=f"{config.LABELS[approach]} best" if k == 0 else None)
                for evals in log.loc[log["leveled_up"].astype(bool), "evals"]:
                    ax.axvline(evals, color=color, alpha=0.25, linewidth=0.7)
        ax.set_xlabel("Individual evaluations")
        ax.set_ylabel("Best fitness")
        ax.legend(frameon=False)
        fig.tight_layout()
        written.append(out_dir / "fitness_curves.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        for approach, per_seed in logs.items():
            name = "GA mutation strength" if approach == "ga" else f"{config.LABELS[approach]} step size σ"
            for k, (seed, log) in enumerate(per_seed.items()):
                ax.plot(log["generation"], log["step_size"], color=config.COLORS[approach], linewidth=1.0,
                        label=name if k == 0 else None)
        ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.legend(frameon=False)
        fig.tight_layout()
        written.append(out_dir / "step_sizes.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

    trials = tuner_trials(runs_dir)
    if trials:
        fig, ax = plt.subplots(figsize=(7, 4))
        for approach, (table, pick) in trials.items():
            color = config.COLORS[approach]
            ax.scatter(table["params"], table["val_accuracy"], color=color, s=18, alpha=0.7,
                       label=config.LABELS[approach])
            ax.scatter(pick["params"], pick["val_accuracy"], color=color, s=140, marker="*",
                       edgecolor="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Validation accuracy")
        ax.legend(frameon=False)
        fig.tight_layout()
        written.append(out_dir / "kt_trials.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)
    return written


def main(champions_dir=None, runs_dir=None, data_path=None, out_dir=None, test_seeds=config.TEST_SEEDS,
         val_seeds=config.VAL_SEEDS, n_jobs=1, latency_calls=1000):
    champions_dir = Path(champions_dir or config.CHAMPIONS_DIR)
    runs_dir = Path(runs_dir or config.RUNS_DIR)
    data_path = Path(data_path or config.DATASET_PATH)
    out_dir = Path(out_dir or config.RESULTS_DIR)
    found = nets.champion_seeds(champions_dir)
    if not found:
        raise FileNotFoundError(f"no trained controllers found in {champions_dir}")
    models, drivers = {}, {}
    for approach, seeds in found.items():
        for seed in seeds:
            models[(approach, seed)], drivers[(approach, seed)] = nets.load_champion(approach, seed, champions_dir)
    per_track = driving_results(drivers, test_seeds, n_jobs=n_jobs)
    validation = driving_results(drivers, val_seeds, n_jobs=n_jobs)
    best = best_seeds(validation)
    paths.best_seeds_path(champions_dir).write_text(json.dumps(best, indent=2))
    selected = driving_selected_table(runs_dir, val_seeds, test_seeds, n_jobs=n_jobs)
    with np.load(data_path) as d:
        data = {key: d[key] for key in d.files}
    metrics = metrics_per_seed(models, data, runs_dir, latency_calls)     # timed after all parallel work
    tables = {"comparison": comparison_table(metrics), "driving": driving_table(per_track, metrics),
              "search_cost": search_cost_table(runs_dir), "driving_selected": selected,
              "dataset": dataset_table(data_path)}
    folder = out_dir / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    per_track.to_csv(folder / "driving_per_track.csv", index=False)
    validation.to_csv(folder / "validation_per_track.csv", index=False)
    metrics.to_csv(folder / "metrics_per_seed.csv", index=False)
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
        (folder / f"{name}.md").write_text(to_markdown(frame))
    return {**tables, "per_track": per_track, "validation": validation, "metrics": metrics, "best_seeds": best,
            "figures": figures(runs_dir, out_dir / "figures")}
