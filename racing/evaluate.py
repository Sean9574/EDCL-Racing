"""Final evaluation (spec §11): driving on held-out tracks, test accuracy, size and speed, tables and figures."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from racing import config, nets, sim
from racing.tracks import get_track


def driving_results(dense_nets, seeds=config.TEST_SEEDS):
    """One lap per controller per test track."""
    rows = []
    for seed in seeds:
        track = get_track(seed, config.MAX_LEVEL)
        for approach, net in dense_nets.items():
            result = sim.simulate(track, net.policy(), 1)
            rows.append({"approach": approach, "seed": seed, "finished": bool(result.finished[0]),
                         "crashed": bool(result.crashed[0]), "lap_time": float(result.finish_time[0]),
                         "progress": float(result.progress[0])})
    return pd.DataFrame(rows, columns=["approach", "seed", "finished", "crashed", "lap_time", "progress"])


def accuracy(model, x, y):
    predicted = np.argmax(model.predict(x, batch_size=4096, verbose=0), axis=1)
    return float(np.mean(predicted == y))


def inference_time(model, x, repeats=5):
    """Seconds per sample for Keras predict on up to 10,000 rows, averaged over `repeats` runs."""
    x = x[:10000]
    model.predict(x[:256], verbose=0)                  # warm-up builds the prediction function
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(x, batch_size=len(x), verbose=0)
        times.append(time.perf_counter() - start)
    return float(np.mean(times) / len(x))


def read_summary(runs_dir, approach):
    path = Path(runs_dir) / approach / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def comparison_table(models, data, runs_dir):
    """Spec Table 4, required comparison."""
    rows = []
    for approach, model in models.items():
        rows.append({"Method": config.LABELS[approach],
                     "Test accuracy": round(accuracy(model, data["X_test"], data["y_test"]), 4),
                     "Parameters": int(model.count_params()),
                     "Inference time (µs/sample)": round(inference_time(model, data["X_test"]) * 1e6, 3),
                     "Search/training time (s)": round(read_summary(runs_dir, approach).get("wall_time", float("nan")), 1)})
    return pd.DataFrame(rows)


def driving_table(results, runs_dir):
    """Spec Table 4, driving performance."""
    rows = []
    n_tracks = results["seed"].nunique()
    for approach in config.APPROACHES:
        sub = results[results["approach"] == approach]
        if sub.empty:
            continue
        done = sub[sub["finished"]]
        level = read_summary(runs_dir, approach).get("level_reached", "n/a") if approach in ("ga", "es") else "n/a"
        rows.append({"Method": config.LABELS[approach],
                     f"Tracks completed (of {n_tracks})": int(sub["finished"].sum()),
                     "Mean lap time (s)": round(float(done["lap_time"].mean()), 2) if len(done) else float("nan"),
                     "Crashes": int(sub["crashed"].sum()),
                     "Level reached": level})
    return pd.DataFrame(rows)


def to_markdown(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def figures(runs_dir, out_dir):
    """Fitness curves, step sizes and Keras Tuner trials; only the figures whose data exists are written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    logs = {a: pd.read_csv(runs_dir / a / "log.csv") for a in ("ga", "es") if (runs_dir / a / "log.csv").exists()}
    if logs:
        fig, ax = plt.subplots(figsize=(7, 4))
        for approach, log in logs.items():
            color = config.COLORS[approach]
            ax.plot(log["evals"], log["best"], color=color, label=f"{config.LABELS[approach]} best")
            ax.plot(log["evals"], log["mean"], color=color, alpha=0.45, linestyle="--",
                    label=f"{config.LABELS[approach]} mean")
            for evals in log.loc[log["leveled_up"].astype(bool), "evals"]:
                ax.axvline(evals, color=color, alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Individual evaluations")
        ax.set_ylabel("Fitness")
        ax.legend(frameon=False)
        fig.tight_layout()
        written.append(out_dir / "fitness_curves.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        for approach, log in logs.items():
            name = "GA mutation strength" if approach == "ga" else "ES step size σ"
            ax.plot(log["generation"], log["step_size"], color=config.COLORS[approach], label=name)
        ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.legend(frameon=False)
        fig.tight_layout()
        written.append(out_dir / "step_sizes.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

    tuners = ("kt_random", "kt_hyperband", "kt_bayesian")
    trials = {a: pd.read_csv(runs_dir / a / "trials.csv") for a in tuners if (runs_dir / a / "trials.csv").exists()}
    if trials:
        fig, ax = plt.subplots(figsize=(7, 4))
        for approach, table in trials.items():
            color = config.COLORS[approach]
            ax.scatter(table["params"], table["val_accuracy"], color=color, s=18, alpha=0.7,
                       label=config.LABELS[approach])
            chosen = str(read_summary(runs_dir, approach).get("chosen_trial"))
            pick = table[table["trial_id"].astype(str) == chosen]
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


def main(champions_dir=None, runs_dir=None, data_path=None, out_dir=None, test_seeds=config.TEST_SEEDS):
    champions_dir = Path(champions_dir or config.CHAMPIONS_DIR)
    runs_dir = Path(runs_dir or config.RUNS_DIR)
    data_path = Path(data_path or config.DATASET_PATH)
    out_dir = Path(out_dir or config.RESULTS_DIR)
    models = nets.load_champions(champions_dir)
    if not models:
        raise FileNotFoundError(f"no trained controllers found in {champions_dir}")
    with np.load(data_path) as d:
        data = {key: d[key] for key in d.files}
    per_track = driving_results({a: nets.from_keras(m) for a, m in models.items()}, test_seeds)
    comparison = comparison_table(models, data, runs_dir)
    driving = driving_table(per_track, runs_dir)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    per_track.to_csv(tables / "driving_per_track.csv", index=False)
    for name, frame in (("comparison", comparison), ("driving", driving)):
        frame.to_csv(tables / f"{name}.csv", index=False)
        (tables / f"{name}.md").write_text(to_markdown(frame))
    return {"comparison": comparison, "driving": driving, "per_track": per_track,
            "figures": figures(runs_dir, out_dir / "figures")}
