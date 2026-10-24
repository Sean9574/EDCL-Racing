"""Final evaluation (spec §11): three-lap time trials on held-out maps, test accuracy, size, speed and
training cost, as tables and figures. Each method has one final controller per run; tables report each run and their
mean as 'run 0 · run 1 · run 2 · run 3 (mean ± sd)' and lead with completion and lap times."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from racing import config, expert, netfig, nets, paths, sim
from racing.track import build_track
from racing.track_gen import geometry_ok
from racing.tracks import get_track

DRIVE_COLUMNS = ["approach", "run_seed", "track", "circuit", "finished", "success", "crashed", "race_time",
                 "lap_time", "progress", "fitness", "lap_1", "lap_2", "lap_3", "flying_lap"]
PER_SEED_COLUMNS = ["approach", "run_seed", "maps", "finished", "success", "lap_time", "race_time", "best_lap",
                    "crashes"]
STAGES = (("imitation", "After imitation"), ("dagger", "After DAgger"), ("speed push", "After speed push"))
LEVEL_NAMES = {**{level: f"{name} cleared" for level, name in config.DIFFICULTIES.items() if level < config.MAX_LEVEL},
               config.MAX_LEVEL: "Mastery goal"}
SEED_STYLES = ("-", "--", ":", "-.")                       # one per run, runs 0-3 (spec §11.4)
EVENT_STYLES = {"level_up": "-", "goal": "-", "tightened": ":"}


def _drive_track(drivers, track_seed, level, factor):
    """All config.LAPS laps for every controller on one track, simulated together (cars never interact). A car
    finishes within the simulator's limit of twice the reference lap time per lap, and succeeds if it finishes
    within the target time (spec §11.1)."""
    return _drive_on(drivers, get_track(track_seed, level), track_seed, factor)


def _drive_on(drivers, track, track_seed, factor):
    """_drive_track on a given track (spec §11.1 drives rebuilt maps)."""
    keys = list(drivers)
    chosen = [drivers[k] for k in keys]
    result = sim.simulate(track, nets.mixed_policy(chosen), len(keys), laps=config.LAPS,
                          setup=nets.drivers_setup(chosen))      # every controller with its own setup (revision 3.1)
    target = sim.target_time(track, factor)
    flying = sim.flying_laps(result.lap_times)
    rows = []
    for i, (approach, run_seed) in enumerate(keys):
        race_time = float(result.finish_time[i])
        finished = bool(result.finished[i])
        rows.append({"approach": approach, "run_seed": run_seed, "track": track_seed, "circuit": track.name,
                     "finished": finished, "success": finished and race_time <= target + config.TIME_EPS,
                     "crashed": bool(result.crashed[i]), "race_time": race_time,
                     "lap_time": race_time / config.LAPS, "progress": float(result.progress[i]),
                     "fitness": float(result.fitness[i]),
                     **{f"lap_{n + 1}": float(result.lap_times[i, n]) for n in range(3)},
                     "flying_lap": float(flying[i])})
    return rows


def driving_results(drivers, track_seeds, level=config.TEST_LEVEL, factor=config.TARGET_FACTOR, n_jobs=1,
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


def widened_track(seed, width, level=config.TEST_LEVEL):
    """Map `seed` rebuilt with the same centerline and a `width` m road, timed by the expert; None if the road would
    reach another part of the map or the expert cannot drive it (spec §11.1). A test circuit's corners are its own, so
    only a generated map's are checked against its level's tightest corner."""
    track = build_track(get_track(seed, level).centerline, width / 2, width / 2, seed, level)
    min_corner = None if level == config.TEST_LEVEL else config.LEVELS[level]["min_corner"]
    if not geometry_ok(track, min_corner):
        return None
    lap_time = expert.expert_lap_time(track)
    if lap_time is None:
        return None
    track.expert_lap_time = lap_time
    return track


def _drive_widened(drivers, seed, width, level, factor):
    track = widened_track(seed, width, level)
    return [] if track is None else [{**row, "width": width} for row in _drive_on(drivers, track, seed, factor)]


def robustness_results(drivers, track_seeds, widths=config.ROBUSTNESS_WIDTHS, level=config.TEST_LEVEL,
                       factor=config.TARGET_FACTOR, n_jobs=1, backend="loky"):
    """Every controller on every map rebuilt at every width (spec §11.1), driven as a test map; maps left out
    at a width have no rows there. The maps are distributed across worker processes."""
    tasks = [(seed, width) for width in widths for seed in track_seeds]
    if n_jobs == 1:
        parts = [_drive_widened(drivers, seed, width, level, factor) for seed, width in tasks]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_drive_widened)(drivers, seed, width, level, factor) for seed, width in tasks)
    return pd.DataFrame([row for part in parts for row in part], columns=[*DRIVE_COLUMNS, "width"])


def robustness_table(frame, widths=config.ROBUSTNESS_WIDTHS):
    """Spec §11.3 table 7: per method and width, the maps finished (in total and per run) and succeeded, and the mean
    lap time on finished maps; each width's first column names how many maps it covers."""
    rows = []
    for approach in config.APPROACHES:
        sub = frame[frame["approach"] == approach]
        if sub.empty:
            continue
        row = {"Method": config.LABELS[approach]}
        for width in widths:
            cell = sub[sub["width"] == width]
            maps = frame.loc[frame["width"] == width, "track"].nunique()
            lap = cell.loc[cell["finished"].astype(bool), "lap_time"]
            row[f"{width:g} m finished ({maps} maps)"] = f"{int(cell['finished'].sum())} / {len(cell)}"
            finished = cell.groupby("run_seed", as_index=False)["finished"].sum()
            row[f"{width:g} m finished per run"] = runs_list(per_run(finished, "finished", sub["run_seed"]))
            row[f"{width:g} m succeeded"] = f"{int(cell['success'].sum())} / {len(cell)}"
            row[f"{width:g} m mean lap (s)"] = "–" if lap.empty else f"{lap.mean():.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def best_seeds(validation):
    """For each approach, its best run: the run whose controller succeeded on the most validation maps; ties go to
    the higher mean validation fitness, then to run 0 (spec §11.1). The results table names it."""
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
    the speed push for the imitation methods (spec §8.5). {} if it does not exist."""
    if approach in config.IMITATION_APPROACHES:
        return read_json(paths.speed_dir(approach, seed, runs_dir) / "summary.json")
    return read_json(paths.run_dir(approach, seed, runs_dir) / "summary.json")


def dagger_record(approach, seed, runs_dir):
    """The DAgger refinement record of an imitation run (spec §8.4); {} if it does not exist."""
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
    layers of its Keras copy (spec §9.9)."""
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


def run_order(seeds=()):
    """The runs a table lists: config.RUN_SEEDS, then any other run seeds among `seeds` (an archive of an earlier
    revision) in seed order."""
    return [*config.RUN_SEEDS, *sorted({int(s) for s in seeds} - set(config.RUN_SEEDS))]


def runs_present(approach, runs_dir):
    """run_order of the run folders of `approach` in runs_dir, so an evaluated archive lists its other runs too."""
    return run_order(paths.run_seeds(approach, runs_dir))


def per_run(frame, column, seeds=None):
    """`column` of a frame with one row per run, as one value for each run of run_order(seeds, by default the frame's
    run seeds): NaN for a run without a row."""
    seeds = frame["run_seed"] if seeds is None else seeds
    return list(frame.set_index("run_seed")[column].astype(float).reindex(run_order(seeds)))


def runs_cell(values, digits, mean_digits=None):
    """Per-run values in run order (spec §11.3): 'r0 · r1 · r2 · r3 (mean ± sd)', with '–' for a run without a
    value; the mean and the sample standard deviation cover the runs that have one ('(mean)' alone for one); a
    value shared by every run shows once; '–' if there is none. The mean has mean_digits decimals (default: digits)."""
    values = [float(v) for v in values]
    present = [v for v in values if not np.isnan(v)]
    mean_digits = digits if mean_digits is None else mean_digits
    if not present:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    if len(present) == len(values) and all(v == present[0] for v in present):
        return f"{present[0]:.{digits}f}"
    shown = " · ".join("–" if np.isnan(v) else f"{v:.{digits}f}" for v in values)
    if len(present) == 1:
        return f"{shown} ({present[0]:.{mean_digits}f})"
    return f"{shown} ({np.mean(present):.{mean_digits}f} ± {np.std(present, ddof=1):.{mean_digits}f})"


def runs_list(values):
    """Per-run counts in run order joined by ' · ', as in runs_cell but without a mean; '–' for a run without one."""
    return " · ".join("–" if np.isnan(v) else str(int(v)) for v in values)


def stalls_broken(log):
    """Stalls one run broke out of (spec §9.4): the rows of its log whose event is 'tightened' and whose previous row
    was boosted. NaN for a log without a 'boost' column (made before the stall boost)."""
    if "boost" not in log:
        return float("nan")
    boosted = log["boost"].shift(1).fillna(0) == 1
    return float(((log["event"] == "tightened") & boosted).sum())


def hours(seconds):
    """Seconds as hours with two decimals; '–' when unknown."""
    if seconds is None or np.isnan(float(seconds)):
        return "–"
    return f"{float(seconds) / 3600:.2f}"


def per_seed(per_track):
    """One row per (approach, run seed): maps driven, finished and succeeded, mean lap and race time and best lap
    time over the finished maps, best lap time (the fastest flying lap, spec §11.1), and crashes."""
    rows = []
    for (approach, seed), runs in per_track.groupby(["approach", "run_seed"]):
        done = runs[runs["finished"].astype(bool)]
        rows.append({"approach": approach, "run_seed": seed, "maps": len(runs),
                     "finished": int(runs["finished"].astype(bool).sum()),
                     "success": int(runs["success"].astype(bool).sum()),
                     "lap_time": float(done["lap_time"].mean()) if len(done) else float("nan"),
                     "race_time": float(done["race_time"].mean()) if len(done) else float("nan"),
                     "best_lap": float(runs["flying_lap"].min()) if runs["flying_lap"].notna().any() else float("nan"),
                     "crashes": int(runs["crashed"].astype(bool).sum())})
    return pd.DataFrame(rows, columns=PER_SEED_COLUMNS)


def reduction_cell(summary):
    """The speed push's result for one run: the target reduction reached, 'not reached' if the run never reached the
    mastery goal, '–' without a summary (spec §9.4)."""
    if not summary:
        return "–"
    if not summary.get("goal_reached"):
        return "not reached"
    return f"{100 * float(summary['target_reduction']):.1f}%"


def reduction_cells(summaries):
    """The runs' reduction_cell values joined with ' · ', followed by the mean and sample sd (ddof = 1) of the target
    reduction in percent over the runs that reached the goal: ' (m ± s%)' for two or more, ' (m%)' for one, nothing
    for none (spec §11.3)."""
    text = " · ".join(reduction_cell(s) for s in summaries)
    reached = [100 * float(s["target_reduction"]) for s in summaries if s and s.get("goal_reached")]
    if len(reached) >= 2:
        return f"{text} ({np.mean(reached):.1f} ± {np.std(reached, ddof=1):.1f}%)"
    if reached:
        return f"{text} ({reached[0]:.1f}%)"
    return text


def setup_cells(drivers, approach, seeds):
    """The final controllers' sensor length (m, no decimals) and steering sensitivity (two decimals) of the runs
    `seeds`, in that order, as runs_cell (spec §11.3); a run without a driver shows '–'."""
    rows = [np.asarray(drivers[(approach, int(s))].setup, dtype=float)
            if drivers and (approach, int(s)) in drivers else np.full(2, np.nan) for s in seeds]
    if not rows:
        return "–", "–"
    values = np.array(rows)
    return runs_cell(values[:, 0], 0), runs_cell(values[:, 1], 2)


def results_table(per_track, runs_dir, drivers=None, best=None):
    """Spec §11.3 table 1: test maps finished and succeeded (in total and per run), lap and race times,
    crashes, the final controllers' setups (drivers: {(approach, run seed): driver}) and the target
    reduction of each run's speed push; per-run cells read 'r0 · r1 · r2 · r3 (mean ± sd)'. With best
    ({approach: run}, as best_seeds returns), each method's best run is named (spec §11.1)."""
    runs = per_seed(per_track)
    rows = []
    for approach in config.APPROACHES:
        sub = runs[runs["approach"] == approach].sort_values("run_seed")
        if sub.empty:
            continue
        total = int(sub["maps"].sum())
        best_lap = "–" if sub["best_lap"].isna().all() else f"{sub['best_lap'].min():.2f}"
        order = run_order(sub["run_seed"])
        sensor, steer = setup_cells(drivers, approach, order)
        row = {"Method": config.LABELS[approach], "Runs": len(sub)}
        if best is not None:
            row["Best run"] = best.get(approach, "–")
        rows.append({**row,
                     "Finished": f"{int(sub['finished'].sum())} / {total}",
                     "Finished per run": runs_list(per_run(sub, "finished")),
                     "Succeeded": f"{int(sub['success'].sum())} / {total}",
                     "Succeeded per run": runs_list(per_run(sub, "success")),
                     "Mean lap time (s)": runs_cell(per_run(sub, "lap_time"), 2), "Best lap time (s)": best_lap,
                     "Mean race time (s)": runs_cell(per_run(sub, "race_time"), 1),
                     "Crashes": runs_cell(per_run(sub, "crashes"), 0, mean_digits=1),
                     "Sensor length (m)": sensor, "Steering sensitivity": steer,
                     "Target reduction (per run)": reduction_cells(
                         [run_summary(approach, s, runs_dir) for s in order])})
    return pd.DataFrame(rows)


def _lap(seconds):
    return "–" if not np.isfinite(seconds) else f"{seconds:.2f}"


def pole_laps_table(per_track):
    """Spec §11.3 table 2: per method and run, the flying lap on each test circuit, its difference to the circuit's
    pole lap in seconds and percent, and the test circuits on which the pole lap was beaten; per method a row "fastest
    run" with the fastest flying lap on each circuit over its runs (not the "Best run" chosen on validation). Lap
    times are whole steps of 0.05 s, so a lap within 0.05 s of the pole lap is not resolvable."""
    from racing import circuits
    rows = []
    for approach in config.APPROACHES:
        sub = per_track[per_track["approach"] == approach]
        if sub.empty:
            continue
        groups = [(str(seed), sub[sub["run_seed"] == seed]) for seed in run_order(sub["run_seed"].unique())]
        for run, races in groups + [("fastest run", sub)]:
            if races.empty:                                  # a run that was never driven has no row
                continue
            row, beaten = {"Method": config.LABELS[approach], "Run": run}, 0
            for name in config.TEST_CIRCUITS:
                laps = races.loc[races["circuit"] == name, "flying_lap"].astype(float)
                lap = float(laps.min()) if np.isfinite(laps).any() else float("nan")
                pole = circuits.POLE_LAPS[name].seconds
                label = circuits.NAMES[name]
                row[f"{label} flying lap (s)"] = _lap(lap)
                row[f"{label} vs pole (s)"] = "–" if np.isnan(lap) else f"{lap - pole:+.2f}"
                row[f"{label} vs pole (%)"] = "–" if np.isnan(lap) else f"{100.0 * (lap - pole) / pole:+.2f}"
                beaten += int(np.isfinite(lap) and lap < pole)
            row["Poles beaten"] = beaten
            rows.append(row)
    return pd.DataFrame(rows)


def pole_figure(per_track, path):
    """Spec §11.4: for each test circuit, each method's fastest flying lap over its runs as a horizontal bar, the
    pole lap as a vertical line. Returns the file's path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from racing import circuits
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(config.TEST_CIRCUITS), figsize=(3.4 * len(config.TEST_CIRCUITS), 3.2))
    for ax, name in zip(np.atleast_1d(axes), config.TEST_CIRCUITS):
        laps = per_track[per_track["circuit"] == name].groupby("approach")["flying_lap"].min()
        methods = [a for a in config.APPROACHES if a in laps.index and np.isfinite(laps[a])]
        pole = circuits.POLE_LAPS[name].seconds
        bars = ax.barh([config.LABELS[a] for a in methods], [laps[a] for a in methods],
                       color=[config.COLORS[a] for a in methods], edgecolor="0.3", linewidth=0.5)
        ax.bar_label(bars, labels=[f"{laps[a] - pole:+.2f} s" for a in methods], padding=2, fontsize=7)
        ax.set_xlim(0, 1.28 * max([pole] + [laps[a] for a in methods]))   # the longest bar and its gap label fit
        ax.invert_yaxis()                                    # the tables' order, first method on top
        ax.axvline(pole, color="0.1", linewidth=1.2, linestyle="--", label="pole lap")
        ax.set_title(circuits.NAMES[name], fontsize=10)
        ax.set_xlabel("Flying lap (s)", fontsize=8)
        ax.tick_params(labelsize=7)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def head_to_head_table(per_track):
    """Spec §11.3 table 2: mean lap time on the test maps that every final controller finished, per run and their
    mean, and on how many of those maps each method had the lowest lap time (its mean over runs)."""
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
        laps = mine.groupby("run_seed", as_index=False)["lap_time"].mean()
        rows.append({"Method": config.LABELS[approach], "Runs": int(sub["run_seed"].nunique()),
                     column: runs_cell(per_run(laps, "lap_time", sub["run_seed"]), 2),
                     "Fastest on (maps)": fastest.get(approach, 0)})
    return pd.DataFrame(rows)


def milestone_cell(summary, level):
    """When a run cleared `level` (on the last level: reached the mastery goal), as generation, evaluations and
    active time (spec §11.3); 'not reached' if it never did; '–' without a summary or for a level below the
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
    """Spec §11.3 table 3, one row per method and run: when each level was cleared and the mastery goal
    reached, how the run ended, evaluations, and the time of each training stage. Runs without any record are left
    out; missing records show '–' and goals never reached show 'not reached'. Stalls broken is the count for the
    run, '–' for a run without a log or a boost column."""
    logs = evolution_logs(runs_dir)
    rows = []
    for approach in config.APPROACHES:
        search = {}
        if approach in config.TUNER_APPROACHES:
            search = read_json(paths.search_dir(approach, runs_dir) / "summary.json")
        for seed in runs_present(approach, runs_dir):
            summary = run_summary(approach, seed, runs_dir)
            imitation, dagger = {}, {}
            if approach in config.IMITATION_APPROACHES:
                imitation = read_json(paths.run_dir(approach, seed, runs_dir) / "summary.json")
                dagger = dagger_record(approach, seed, runs_dir)
            if not (summary or imitation or dagger):
                continue
            row = {"Method": config.LABELS[approach], "Run": seed}
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
            log = logs.get(approach, {}).get(seed)
            stalls = float("nan") if log is None else stalls_broken(log)
            row["Stalls broken"] = "–" if np.isnan(stalls) else str(int(stalls))
            rows.append(row)
    return pd.DataFrame(rows)


def stage_results(runs_dir, per_track, test_seeds=config.TEST_TRACKS, n_jobs=1):
    """Test-map results of the imitation methods after initial training and after DAgger (from their stage files)
    and after the speed push (their final controllers, taken from per_track), with a 'stage' column
    (spec §8.6)."""
    from racing.backend import keras_cpu
    keras = keras_cpu()
    parts = []
    for stage in ("imitation", "dagger"):
        drivers = {}
        for approach in config.IMITATION_APPROACHES:
            for seed in runs_present(approach, runs_dir):
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
    """Spec §11.3 table 4: the imitation methods' test results after each stage, per run and their mean."""
    rows = []
    for approach in config.IMITATION_APPROACHES:
        for stage, name in STAGES:
            sub = results[(results["approach"] == approach) & (results["stage"] == stage)]
            if sub.empty:
                continue
            runs = per_seed(sub)
            total = int(runs["maps"].sum())
            rows.append({"Method": config.LABELS[approach], "Stage": name, "Runs": len(runs),
                         "Finished": f"{int(runs['finished'].sum())} / {total}",
                         "Succeeded": f"{int(runs['success'].sum())} / {total}",
                         "Mean lap time (s)": runs_cell(per_run(runs, "lap_time"), 2),
                         "Crashes": runs_cell(per_run(runs, "crashes"), 0, mean_digits=1)})
    return pd.DataFrame(rows)


def comparison_table(metrics):
    """The course comparison (spec §11.3 table 5), measured as in spec §11.2: per run and their mean, except the
    search time, which a Keras Tuner method spends once for all runs (spec §8.2)."""
    rows = []
    for approach in config.APPROACHES:
        sub = metrics[metrics["approach"] == approach]
        if sub.empty:
            continue
        search = float(sub["search_s"].iloc[0])
        rows.append({"Method": config.LABELS[approach], "Runs": len(sub),
                     "Test accuracy": runs_cell(per_run(sub, "accuracy"), 4),
                     "Parameters": runs_cell(per_run(sub, "params"), 0),
                     "Batch inference (µs/sample)": runs_cell(per_run(sub, "batch_us"), 3),
                     "Single-sample latency (ms)": runs_cell(per_run(sub, "single_ms"), 3),
                     "Search time (s)": "–" if np.isnan(search) else f"{search:.1f}",
                     "Training time (s)": runs_cell(per_run(sub, "training_s"), 1)})
    return pd.DataFrame(rows)


def search_cost_table(runs_dir):
    """Spec §11.3 table 6, search cost of each Keras Tuner method."""
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


def driving_selected_table(runs_dir=None, val_seeds=config.VAL_SEEDS, test_seeds=config.TEST_TRACKS, n_jobs=1):
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
        validation = driving_results(drivers, val_seeds, level=config.VALIDATION_LEVEL, n_jobs=n_jobs)
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
        for seed in runs_present(approach, runs_dir):
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
    """One line per run of `column` (runs told apart by line style), with level advancements and the mastery goal
    as solid and target reductions as dotted grey vertical lines."""
    for k, (seed, log) in enumerate(sorted(runs.items())):
        if column not in log:
            continue
        ax.plot(log["generation"], log[column], linewidth=1.2, linestyle=SEED_STYLES[k % len(SEED_STYLES)],
                label=f"run {seed}", **_line_kwargs(config.COLORS[approach]))
        events = log["event"].fillna("").astype(str) if "event" in log else []
        for generation, event in zip(log["generation"], events):
            if event in EVENT_STYLES:
                ax.axvline(generation, color="0.6", linewidth=0.6, linestyle=EVENT_STYLES[event], zorder=0)


def _panels(plt, logs, column, ylabel, threshold=None, sharey=True):
    """Small multiples: one panel per method with data in `column`, one line per run (told apart by line style),
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False,
                             sharey=sharey)
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
    """Spec §11.4: for each method, the population's mean sensor length (upper panel) and mean steering
    sensitivity (lower panel) per generation, one line per run, with level advancements and target reductions
    marked and the starting setup (150 m, sensitivity 1) as a dashed line. None if no log records setups."""
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
    handles += [Line2D([], [], color="0.3", linewidth=0.8, linestyle="--", label="starting setup"),
                Line2D([], [], color="0.6", linewidth=0.6, label="level cleared / goal"),
                Line2D([], [], color="0.6", linewidth=0.6, linestyle=":", label="target cut")]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def network_text(driver):
    """A final controller's structure (spec §11.3, table 8): a dense network's layer sizes, such as 18 → 9 → 45;
    a NEAT genome's hidden nodes and enabled connections."""
    from racing import neat
    if isinstance(driver, neat.NeatNet):
        genome = driver.genome
        return (f"{int(np.sum(genome.node_ids >= neat.FIRST_HIDDEN))} hidden nodes, "
                f"{int(np.sum(genome.enabled))} connections")
    return " → ".join(str(units) for units in (driver.n_inputs, *(w.shape[1] for w, _ in driver.layers)))


def evolved_networks_table(drivers):
    """Spec §11.3, table 8: per GA and NEAT run, the final controller's structure and parameters (§11.2)."""
    rows = [{"Method": config.LABELS[approach], "Run": seed, "Structure": network_text(drivers[(approach, seed)]),
             "Parameters": drivers[(approach, seed)].n_params}
            for approach in ("ga", "neat") for a, seed in sorted(drivers) if a == approach]
    return pd.DataFrame(rows, columns=["Method", "Run", "Structure", "Parameters"])


def figures(runs_dir, out_dir):
    """Spec §11.4: the lowest map success share and the median lap time over training (one panel per method,
    one line per run), the population's mean setup, the GA's and NEAT's mean network size over training, the
    step sizes, and the Keras Tuner trials. Only figures with data are written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    logs = evolution_logs(runs_dir)
    for name, column, ylabel, threshold in (
            ("success_over_training.png", "min_share", "Lowest map success share", config.MASTERY_FRACTION),
            ("lap_time_over_training.png", "median_lap", "Median lap time (s)", None),
            ("network_size_over_training.png", "mean_nodes", "Hidden units or nodes (mean)", None)):
        fig = _panels(plt, logs, column, ylabel, threshold, sharey=column != "mean_nodes")
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


def main(champions_dir=None, runs_dir=None, data_path=None, out_dir=None, test_seeds=config.TEST_TRACKS,
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
    validation = driving_results(drivers, val_seeds, level=config.VALIDATION_LEVEL, n_jobs=n_jobs)
    robust = robustness_results(drivers, test_seeds, n_jobs=n_jobs)
    best = best_seeds(validation)
    paths.best_seeds_path(champions_dir).write_text(json.dumps(best, indent=2))
    stages = stage_results(runs_dir, per_track, test_seeds, n_jobs)
    selected = driving_selected_table(runs_dir, val_seeds, test_seeds, n_jobs=n_jobs)
    with np.load(data_path) as d:
        data = {key: d[key] for key in d.files}
    metrics = metrics_per_seed(models, data, runs_dir, latency_calls, drivers)   # timed after all parallel work
    tables = {"results": results_table(per_track, runs_dir, drivers, best),
              "pole_laps": pole_laps_table(per_track),
              "head_to_head": head_to_head_table(per_track),
              "training_cost": training_cost_table(runs_dir), "stages": stages_table(stages),
              "comparison": comparison_table(metrics), "search_cost": search_cost_table(runs_dir),
              "driving_selected": selected, "dataset": dataset_table(data_path),
              "robustness": robustness_table(robust),
              "evolved_networks": evolved_networks_table(drivers)}
    folder = out_dir / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    per_track.to_csv(folder / "driving_per_track.csv", index=False)
    validation.to_csv(folder / "validation_per_track.csv", index=False)
    stages.to_csv(folder / "stages_per_track.csv", index=False)
    robust.to_csv(folder / "robustness_per_track.csv", index=False)
    metrics.to_csv(folder / "metrics_per_seed.csv", index=False)
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
        (folder / f"{name}.md").write_text(to_markdown(frame))
    drawn = figures(runs_dir, out_dir / "figures")
    if per_track["flying_lap"].notna().any():
        drawn.append(pole_figure(per_track, out_dir / "figures" / "pole_laps.png"))
    if best:                                            # each method's best final controller (spec §11.4)
        drawn.append(netfig.draw_networks({approach: drivers[(approach, seed)] for approach, seed in best.items()},
                                          out_dir / "figures" / "networks.png"))
    return {**tables, "per_track": per_track, "validation": validation, "metrics": metrics, "best_seeds": best,
            "robustness_per_track": robust,
            "figures": drawn}
