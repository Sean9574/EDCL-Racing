"""Final evaluation (spec §11, revision 3 §7): three-lap time trials on held-out maps, test accuracy, size, speed and
training cost, as tables and figures. Each method has one final controller per run seed; tables report the mean and
standard deviation across seeds and lead with completion and lap times."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from racing import config, netfig, nets, paths, sim
from racing.tracks import get_track

DRIVE_COLUMNS = ["approach", "run_seed", "track", "finished", "success", "crashed", "race_time", "lap_time",
                 "progress", "fitness"]
PER_SEED_COLUMNS = ["approach", "run_seed", "maps", "finished", "success", "lap_time", "race_time", "best_lap",
                    "crashes"]
STAGES = (("imitation", "After imitation"), ("dagger", "After DAgger"), ("speed push", "After speed push"))
LEVEL_NAMES = {**{level: f"{name} cleared" for level, name in config.DIFFICULTIES.items() if level < config.MAX_LEVEL},
               config.MAX_LEVEL: "Mastery goal"}
SEED_STYLES = ("-", "--", ":")
EVENT_STYLES = {"level_up": "-", "goal": "-", "tightened": ":"}


def _drive_track(drivers, track_seed, level, factor):
    """All config.LAPS laps for every controller on one track, simulated together (cars never interact). A car
    finishes within the simulator's limit of twice the reference lap time per lap, and succeeds if it finishes
    within the target time (revision 3 §7.1)."""
    track = get_track(track_seed, level)
    keys = list(drivers)
    chosen = [drivers[k] for k in keys]
    result = sim.simulate(track, nets.mixed_policy(chosen), len(keys), laps=config.LAPS,
                          setup=nets.drivers_setup(chosen))      # every controller with its own setup (revision 3.1)
    target = sim.target_time(track, factor)
    rows = []
    for i, (approach, run_seed) in enumerate(keys):
        race_time = float(result.finish_time[i])
        finished = bool(result.finished[i])
        rows.append({"approach": approach, "run_seed": run_seed, "track": track_seed, "finished": finished,
                     "success": finished and race_time <= target + config.TIME_EPS,
                     "crashed": bool(result.crashed[i]), "race_time": race_time,
                     "lap_time": race_time / config.LAPS, "progress": float(result.progress[i]),
                     "fitness": float(result.fitness[i])})
    return rows


def driving_results(drivers, track_seeds, level=config.MAX_LEVEL, factor=config.TARGET_FACTOR, n_jobs=1,
                    backend="loky"):
    """A three-lap time trial per controller per track. drivers: {(approach, run seed or trial id): driver}. The
    tracks are distributed across worker processes; the result does not depend on n_jobs."""
    track_seeds = list(track_seeds)
    if not drivers:
        return pd.DataFrame(columns=DRIVE_COLUMNS)
    if n_jobs == 1 or len(track_seeds) == 1:
        parts = [_drive_track(drivers, seed, level, factor) for seed in track_seeds]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_drive_track)(drivers, seed, level, factor) for seed in track_seeds)
    return pd.DataFrame([row for part in parts for row in part], columns=DRIVE_COLUMNS)


def best_seeds(validation):
    """For each approach, the run seed whose controller succeeded on the most validation maps; ties go to the higher
    mean validation fitness, then to the lowest seed (revision 3 §7.1)."""
    best = {}
    for approach in config.APPROACHES:
        sub = validation[validation["approach"] == approach]
        if sub.empty:
            continue
        per_seed = sub.groupby("run_seed").agg(successes=("success", "sum"), fitness=("fitness", "mean"))
        ranked = sorted(per_seed.itertuples(), key=lambda r: (-r.successes, -r.fitness, r.Index))
        best[approach] = int(ranked[0].Index)
    return best


def same_prediction_settings(model):
    """Every controller is timed with the same prediction settings: no XLA compilation and one batch per call
    (spec §11). A loaded imitation model would otherwise keep its training settings, an evolutionary one the
    Keras default, which depends on the installed devices."""
    model.jit_compile = False
    model.steps_per_execution = 1
    model.predict_function = None              # rebuilt with these settings by the next predict call
    return model


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


def run_summary(approach, seed, runs_dir):
    """Summary of the evolutionary run that produced a final controller: the run itself for the evolutionary methods,
    the speed push for the imitation methods (revision 3 §6.4). {} if it does not exist."""
    if approach in config.IMITATION_APPROACHES:
        return read_json(paths.speed_dir(approach, seed, runs_dir) / "summary.json")
    return read_json(paths.run_dir(approach, seed, runs_dir) / "summary.json")


def dagger_record(approach, seed, runs_dir):
    """The DAgger refinement record of an imitation run (revision 3 §6.3); {} if it does not exist."""
    return read_json(paths.run_dir(approach, seed, runs_dir) / "dagger.json")


def training_seconds(approach, seed, runs_dir):
    """Training time of one final controller: the evolutionary run, or for an imitation method its initial training,
    DAgger refinement and speed push together. The Keras Tuner search is reported separately. nan without records."""
    parts = [run_summary(approach, seed, runs_dir).get("wall_time")]
    if approach in config.IMITATION_APPROACHES:
        parts += [read_json(paths.run_dir(approach, seed, runs_dir) / "summary.json").get("wall_time"),
                  dagger_record(approach, seed, runs_dir).get("wall_time")]
    values = [float(v) for v in parts if v is not None]
    return float(sum(values)) if values else float("nan")


def metrics_per_seed(models, data, runs_dir, latency_calls=1000, drivers=None):
    """One row per final controller: test accuracy, size, speed and training cost. With drivers, the parameter count
    is the driver's, so a NEAT controller counts its genome's connections and biases rather than the zero-padded
    layers of its Keras copy (revision 3 §5.4)."""
    rows = []
    for (approach, run_seed), model in models.items():
        same_prediction_settings(model)
        search = float("nan")
        if approach in config.TUNER_APPROACHES:
            search = read_json(paths.search_dir(approach, runs_dir) / "summary.json").get("wall_time", float("nan"))
        key = (approach, run_seed)
        params = drivers[key].n_params if drivers is not None and key in drivers else model.count_params()
        rows.append({"approach": approach, "run_seed": run_seed,
                     "accuracy": accuracy(model, data["X_test"], data["y_test"]),
                     "params": int(params),
                     "batch_us": inference_time(model, data["X_test"]) * 1e6,
                     "single_ms": single_sample_latency(model, data["X_test"], latency_calls) * 1e3,
                     "training_s": training_seconds(approach, run_seed, runs_dir), "search_s": float(search)})
    return pd.DataFrame(rows, columns=["approach", "run_seed", "accuracy", "params", "batch_us", "single_ms",
                                       "training_s", "search_s"])


def mean_std(values, digits):
    """'mean ± std' across seeds (sample standard deviation); the value alone if there is one value or all are
    equal; '–' if there is none."""
    values = [float(v) for v in values if not np.isnan(float(v))]
    if not values:
        return "–"
    if len(values) == 1 or all(v == values[0] for v in values):
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def seeds_note(values, digits):
    """mean_std of per-seed values, naming how many seeds it covers when some seeds have no value."""
    values = [float(v) for v in values]
    cell = mean_std(values, digits)
    covered = sum(1 for v in values if not np.isnan(v))
    if 0 < covered < len(values):
        cell += f" ({covered} of {len(values)} seeds)"
    return cell


def hours(seconds):
    """Seconds as hours with two decimals; '–' when unknown."""
    if seconds is None or np.isnan(float(seconds)):
        return "–"
    return f"{float(seconds) / 3600:.2f}"


def per_seed(per_track):
    """One row per (approach, run seed): maps driven, finished and succeeded, mean lap and race time and best lap
    time over the finished maps, and crashes."""
    rows = []
    for (approach, seed), runs in per_track.groupby(["approach", "run_seed"]):
        done = runs[runs["finished"].astype(bool)]
        rows.append({"approach": approach, "run_seed": seed, "maps": len(runs),
                     "finished": int(runs["finished"].astype(bool).sum()),
                     "success": int(runs["success"].astype(bool).sum()),
                     "lap_time": float(done["lap_time"].mean()) if len(done) else float("nan"),
                     "race_time": float(done["race_time"].mean()) if len(done) else float("nan"),
                     "best_lap": float(done["lap_time"].min()) if len(done) else float("nan"),
                     "crashes": int(runs["crashed"].astype(bool).sum())})
    return pd.DataFrame(rows, columns=PER_SEED_COLUMNS)


def reduction_cell(summary):
    """The speed push's result for one run: the target reduction reached, 'not reached' if the run never reached the
    mastery goal, '–' without a summary (revision 3 §4.4)."""
    if not summary:
        return "–"
    if not summary.get("goal_reached"):
        return "not reached"
    return f"{100 * float(summary['target_reduction']):.1f}%"


def setup_cells(drivers, approach, seeds):
    """The final controllers' sensor length (m, no decimals) and steering sensitivity (two decimals) as 'mean ± std'
    across seeds (revision 3.1 §5); '–' for both without drivers for the method."""
    rows = [np.asarray(drivers[(approach, int(s))].setup, dtype=float) for s in seeds
            if drivers and (approach, int(s)) in drivers]
    if not rows:
        return "–", "–"
    values = np.array(rows)
    return mean_std(values[:, 0], 0), mean_std(values[:, 1], 2)


def results_table(per_track, runs_dir, drivers=None):
    """Revision 3 §7.2 table 1: test maps finished and succeeded (in total and per seed), lap and race times,
    crashes, the final controllers' setups (revision 3.1 §5; drivers: {(approach, seed): driver}) and the target
    reduction of each seed's speed push."""
    seeds = per_seed(per_track)
    rows = []
    for approach in config.APPROACHES:
        sub = seeds[seeds["approach"] == approach].sort_values("run_seed")
        if sub.empty:
            continue
        total = int(sub["maps"].sum())
        best_lap = "–" if sub["best_lap"].isna().all() else f"{sub['best_lap'].min():.2f}"
        sensor, steer = setup_cells(drivers, approach, sub["run_seed"])
        rows.append({"Method": config.LABELS[approach], "Seeds": len(sub),
                     "Finished": f"{int(sub['finished'].sum())} / {total}",
                     "Finished per seed": ", ".join(str(int(v)) for v in sub["finished"]),
                     "Succeeded": f"{int(sub['success'].sum())} / {total}",
                     "Succeeded per seed": ", ".join(str(int(v)) for v in sub["success"]),
                     "Mean lap time (s)": seeds_note(sub["lap_time"], 2), "Best lap time (s)": best_lap,
                     "Mean race time (s)": seeds_note(sub["race_time"], 1),
                     "Crashes": mean_std(sub["crashes"], 1),
                     "Sensor length (m)": sensor, "Steering sensitivity": steer,
                     "Target reduction (per seed)": ", ".join(
                         reduction_cell(run_summary(approach, int(s), runs_dir)) for s in sub["run_seed"])})
    return pd.DataFrame(rows)


def head_to_head_table(per_track):
    """Revision 3 §7.2 table 2: mean lap time on the test maps that every final controller finished, and on how many
    of those maps each method had the lowest lap time (its mean over seeds)."""
    if per_track.empty:
        return pd.DataFrame()
    finished_all = per_track.groupby("track")["finished"].all()
    shared = sorted(finished_all[finished_all.astype(bool)].index)
    column = f"Mean lap time on the {len(shared)} shared maps (s)"
    on_shared = per_track[per_track["track"].isin(shared)]
    fastest = {}
    if shared:
        means = on_shared.groupby(["track", "approach"])["lap_time"].mean().unstack()
        for approach in means.idxmin(axis=1):
            fastest[approach] = fastest.get(approach, 0) + 1
    rows = []
    for approach in config.APPROACHES:
        sub = per_track[per_track["approach"] == approach]
        if sub.empty:
            continue
        mine = on_shared[on_shared["approach"] == approach]
        laps = [float(runs["lap_time"].mean()) for _, runs in mine.groupby("run_seed")]
        rows.append({"Method": config.LABELS[approach], "Seeds": int(sub["run_seed"].nunique()),
                     column: mean_std(laps, 2), "Fastest on (maps)": fastest.get(approach, 0)})
    return pd.DataFrame(rows)


def milestone_cell(summary, level):
    """When a run cleared `level` (on the last level: reached the mastery goal), as generation, evaluations and
    active time (revision 3 §1, §7.2); 'not reached' if it never did; '–' without a summary or for a level below the
    run's start level."""
    if not summary or summary.get("start_level", 1) > level:
        return "–"
    for entry in summary.get("levels", []):
        if entry.get("level") == level:
            return (f"gen {int(entry['generation']):,} · {int(entry['evals']):,} evals · "
                    f"{hours(entry['elapsed'])} h")
    return "not reached"


def end_cell(summary):
    """How a run ended: at its speed limit, stalled on a level, or stopped by a generation limit."""
    if not summary:
        return "–"
    outcome = summary.get("outcome")
    if outcome == "speed_limit":
        how = "speed limit"
    elif outcome == "stalled":
        how = f"stalled on level {summary.get('level_reached', '?')}"
    else:
        how = "stopped"
    return f"{how} · gen {int(summary.get('generations', 0)):,}"


def training_cost_table(runs_dir):
    """Revision 3 §7.2 table 3, one row per method and seed: when each level was cleared and the mastery goal
    reached, how the run ended, evaluations, and the time of each training stage. Seeds without any record are left
    out; missing records show '–' and goals never reached show 'not reached'."""
    rows = []
    for approach in config.APPROACHES:
        search = {}
        if approach in config.TUNER_APPROACHES:
            search = read_json(paths.search_dir(approach, runs_dir) / "summary.json")
        for seed in config.RUN_SEEDS:
            summary = run_summary(approach, seed, runs_dir)
            imitation, dagger = {}, {}
            if approach in config.IMITATION_APPROACHES:
                imitation = read_json(paths.run_dir(approach, seed, runs_dir) / "summary.json")
                dagger = dagger_record(approach, seed, runs_dir)
            if not (summary or imitation or dagger):
                continue
            row = {"Method": config.LABELS[approach], "Seed": seed}
            for level in range(1, config.MAX_LEVEL + 1):
                row[LEVEL_NAMES[level]] = milestone_cell(summary, level)
            row["Run end"] = end_cell(summary)
            row["Evaluations"] = f"{int(summary['evals']):,}" if "evals" in summary else "–"
            row["Search (h)"] = hours(search.get("wall_time"))
            row["Imitation training (h)"] = hours(imitation.get("wall_time"))
            row["DAgger rounds"] = str(dagger["rounds"]) if "rounds" in dagger else "–"
            row["DAgger goal"] = "–" if not dagger else ("reached" if dagger.get("goal_reached") else "not reached")
            row["DAgger (h)"] = hours(dagger.get("wall_time"))
            row["Evolution (h)"] = hours(summary.get("wall_time"))
            row["Total (h)"] = hours(training_seconds(approach, seed, runs_dir))
            rows.append(row)
    return pd.DataFrame(rows)


def stage_results(runs_dir, per_track, test_seeds=config.TEST_SEEDS, n_jobs=1):
    """Test-map results of the imitation methods after initial training and after DAgger (from their stage files)
    and after the speed push (their final controllers, taken from per_track), with a 'stage' column
    (revision 3 §6.5)."""
    from racing.backend import keras_cpu
    keras = keras_cpu()
    parts = []
    for stage in ("imitation", "dagger"):
        drivers = {}
        for approach in config.IMITATION_APPROACHES:
            for seed in config.RUN_SEEDS:
                path = paths.stage_path(approach, seed, stage, runs_dir)
                if path.exists():
                    drivers[(approach, seed)] = nets.from_keras(keras.models.load_model(path))
        if drivers:
            parts.append(driving_results(drivers, test_seeds, n_jobs=n_jobs).assign(stage=stage))
    final = per_track[per_track["approach"].isin(config.IMITATION_APPROACHES)]
    if len(final):
        parts.append(final.assign(stage="speed push"))
    if not parts:
        return pd.DataFrame(columns=[*DRIVE_COLUMNS, "stage"])
    return pd.concat(parts, ignore_index=True)


def stages_table(results):
    """Revision 3 §7.2 table 4: the imitation methods' test results after each stage."""
    rows = []
    for approach in config.IMITATION_APPROACHES:
        for stage, name in STAGES:
            sub = results[(results["approach"] == approach) & (results["stage"] == stage)]
            if sub.empty:
                continue
            seeds = per_seed(sub)
            total = int(seeds["maps"].sum())
            rows.append({"Method": config.LABELS[approach], "Stage": name, "Seeds": len(seeds),
                         "Finished": f"{int(seeds['finished'].sum())} / {total}",
                         "Succeeded": f"{int(seeds['success'].sum())} / {total}",
                         "Mean lap time (s)": seeds_note(seeds["lap_time"], 2),
                         "Crashes": mean_std(seeds["crashes"], 1)})
    return pd.DataFrame(rows)


def comparison_table(metrics):
    """The course comparison (revision 3 §7.2 table 5), measured as in spec §11."""
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
    """{approach: {run seed: log}} for every evolutionary run whose log exists: the runs of the evolutionary methods
    and the speed pushes of the imitation methods."""
    logs = {}
    for approach in config.APPROACHES:
        found = {}
        for seed in config.RUN_SEEDS:
            if approach in config.IMITATION_APPROACHES:
                path = paths.speed_dir(approach, seed, runs_dir) / "log.csv"
            else:
                path = paths.run_dir(approach, seed, runs_dir) / "log.csv"
            if path.exists():
                found[seed] = pd.read_csv(path)
        if found:
            logs[approach] = found
    return logs


def _outline(color):
    """Path effects that draw a thin dark outline around a line in a light color (NEAT's yellow), so it stays
    visible on a white background; None for darker colors."""
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as effects
    r, g, b = mcolors.to_rgb(color)
    if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.7:
        return None
    return [effects.Stroke(linewidth=2.6, foreground="0.35"), effects.Normal()]


def _line_kwargs(color):
    outline = _outline(color)
    return {"color": color} if outline is None else {"color": color, "path_effects": outline}


def _plot_runs(ax, runs, column, approach):
    """One line per seed of `column` (seeds told apart by line style), with level advancements and the mastery goal
    as solid and target reductions as dotted grey vertical lines."""
    for k, (seed, log) in enumerate(sorted(runs.items())):
        if column not in log:
            continue
        ax.plot(log["generation"], log[column], linewidth=1.2, linestyle=SEED_STYLES[k % len(SEED_STYLES)],
                label=f"seed {seed}", **_line_kwargs(config.COLORS[approach]))
        events = log["event"].fillna("").astype(str) if "event" in log else []
        for generation, event in zip(log["generation"], events):
            if event in EVENT_STYLES:
                ax.axvline(generation, color="0.6", linewidth=0.6, linestyle=EVENT_STYLES[event], zorder=0)


def _panels(plt, logs, column, ylabel, threshold=None):
    """Small multiples: one panel per method with data in `column`, one line per seed (told apart by line style),
    level advancements and the mastery goal as solid and target reductions as dotted grey vertical lines. None if no
    method has data."""
    from matplotlib.lines import Line2D
    with_data = {approach: runs for approach, runs in logs.items()
                 if any(column in log and log[column].notna().any() for log in runs.values())}
    if not with_data:
        return None
    n = len(with_data)
    ncols = min(4, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False, sharey=True)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, (approach, runs) in zip(axes.flat, with_data.items()):
        _plot_runs(ax, runs, column, approach)
        if threshold is not None:
            ax.axhline(threshold, color="0.3", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_title(config.LABELS[approach], fontsize=10)
        ax.set_xlabel("Generation", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
    for row in axes:
        row[0].set_ylabel(ylabel, fontsize=8)
    handles, _ = axes.flat[0].get_legend_handles_labels()
    handles += [Line2D([], [], color="0.6", linewidth=0.6, label="level cleared / goal"),
                Line2D([], [], color="0.6", linewidth=0.6, linestyle=":", label="target cut")]
    axes.flat[0].legend(handles=handles, frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


SETUP_PANELS = (("sensor_mean", "Sensor length (m)", config.SENSOR_LENGTH_LIMITS, 0),
                ("steer_mean", "Steering sensitivity", config.STEER_GAIN_LIMITS, 1))


def _setup_figure(plt, logs):
    """Revision 3.1 §5: for each method, the population's mean sensor length (upper panel) and mean steering
    sensitivity (lower panel) per generation, one line per seed, with level advancements and target reductions
    marked and the revision 3 setup as a dashed line. None if no log records setups."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter
    with_setup = {approach: runs for approach, runs in logs.items()
                  if any("sensor_mean" in log and log["sensor_mean"].notna().any() for log in runs.values())}
    if not with_setup:
        return None
    ncols = min(4, len(with_setup))
    groups = -(-len(with_setup) // ncols)
    fig, axes = plt.subplots(2 * groups, ncols, figsize=(3.4 * ncols, 4.6 * groups), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, (approach, runs) in enumerate(with_setup.items()):
        group, column = divmod(k, ncols)
        for row, (name, ylabel, limits, index) in enumerate(SETUP_PANELS):
            ax = axes[2 * group + row, column]
            ax.set_visible(True)
            _plot_runs(ax, runs, name, approach)
            ax.axhline(config.DEFAULT_SETUP[index], color="0.3", linewidth=0.8, linestyle="--", zorder=0)
            ax.set_yscale("log")
            ax.set_ylim(*limits)
            ax.set_yticks([limits[0], config.DEFAULT_SETUP[index], limits[1]])   # plain labels on the log axis
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f"{config.LABELS[approach]}: {ylabel[0].lower()}{ylabel[1:]}", fontsize=9)
            ax.set_xlabel("Generation", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, linewidth=0.5)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], color="0.3", linewidth=0.8, linestyle="--", label="revision 3 setup"),
                Line2D([], [], color="0.6", linewidth=0.6, label="level cleared / goal"),
                Line2D([], [], color="0.6", linewidth=0.6, linestyle=":", label="target cut")]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def figures(runs_dir, out_dir):
    """Revision 3 §7.3: the lowest map success share and the median lap time over training (one panel per method,
    one line per seed), the population's mean setup over training (revision 3.1 §5), the step sizes, and the Keras
    Tuner trials. Only figures with data are written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    logs = evolution_logs(runs_dir)
    for name, column, ylabel, threshold in (
            ("success_over_training.png", "min_share", "Lowest map success share", config.MASTERY_FRACTION),
            ("lap_time_over_training.png", "median_lap", "Median lap time (s)", None)):
        fig = _panels(plt, logs, column, ylabel, threshold)
        if fig is None:
            continue
        if column == "min_share":
            fig.axes[0].set_ylim(0.0, 1.02)                 # the y-axis is shared, so this sets every panel
        written.append(out_dir / name)
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

    fig = _setup_figure(plt, logs)
    if fig is not None:
        written.append(out_dir / "setup_over_training.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

    with_steps = {approach: runs for approach, runs in logs.items()
                  if any("step_size" in log for log in runs.values())}
    if with_steps:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for approach, runs in with_steps.items():
            if approach == "ga":
                name = "GA mutation strength"
            elif approach in config.IMITATION_APPROACHES:
                name = f"{config.LABELS[approach]} speed push σ"
            else:
                name = f"{config.LABELS[approach]} σ"
            for k, (seed, log) in enumerate(sorted(runs.items())):
                if "step_size" in log:
                    ax.plot(log["generation"], log["step_size"], linewidth=1.0, label=name if k == 0 else None,
                            **_line_kwargs(config.COLORS[approach]))
        ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.legend(frameon=False, fontsize=7)
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
    stages = stage_results(runs_dir, per_track, test_seeds, n_jobs)
    selected = driving_selected_table(runs_dir, val_seeds, test_seeds, n_jobs=n_jobs)
    with np.load(data_path) as d:
        data = {key: d[key] for key in d.files}
    metrics = metrics_per_seed(models, data, runs_dir, latency_calls, drivers)   # timed after all parallel work
    tables = {"results": results_table(per_track, runs_dir, drivers), "head_to_head": head_to_head_table(per_track),
              "training_cost": training_cost_table(runs_dir), "stages": stages_table(stages),
              "comparison": comparison_table(metrics), "search_cost": search_cost_table(runs_dir),
              "driving_selected": selected, "dataset": dataset_table(data_path)}
    folder = out_dir / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    per_track.to_csv(folder / "driving_per_track.csv", index=False)
    validation.to_csv(folder / "validation_per_track.csv", index=False)
    stages.to_csv(folder / "stages_per_track.csv", index=False)
    metrics.to_csv(folder / "metrics_per_seed.csv", index=False)
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
        (folder / f"{name}.md").write_text(to_markdown(frame))
    drawn = figures(runs_dir, out_dir / "figures")
    if best:                                            # each method's best final controller (revision 3 §7.3)
        drawn.append(netfig.draw_networks({approach: drivers[(approach, seed)] for approach, seed in best.items()},
                                          out_dir / "figures" / "networks.png"))
    return {**tables, "per_track": per_track, "validation": validation, "metrics": metrics, "best_seeds": best,
            "figures": drawn}
