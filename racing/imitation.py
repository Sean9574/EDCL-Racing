"""Imitation-learning controllers (spec §8): a baseline network, three Keras Tuner searches, and retraining of the
selected architectures with each run seed. Each trained network is saved as its imitation stage (spec §8.6)."""
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from racing import checkpoint, config, nets, paths, progress, runs
from racing.backend import keras_cpu

TUNER_CLASSES = {"kt_random": "RandomSearch", "kt_hyperband": "Hyperband", "kt_bayesian": "BayesianOptimization"}


def load_dataset(path=None):
    path = config.DATASET_PATH if path is None else Path(path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def compile_model(model, learning_rate, steps_per_execution=None, jit_compile=None):
    """Adam and sparse categorical cross-entropy, with the Keras speed settings of config (spec §8.1)."""
    keras = keras_cpu()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  steps_per_execution=config.STEPS_PER_EXECUTION if steps_per_execution is None else steps_per_execution,
                  jit_compile=config.JIT_COMPILE if jit_compile is None else jit_compile)
    model.optimizer.build(model.trainable_variables)  # so checkpoints saved with optimizer state load cleanly
    return model


def build_network(units, activation, dropout, learning_rate, steps_per_execution=None, jit_compile=None):
    """Dense classifier: 12 inputs, one hidden layer per entry of `units` (each followed by dropout if dropout > 0),
    and a 45-way softmax."""
    keras = keras_cpu()
    layers = [keras.Input((config.N_INPUTS,))]
    for n in units:
        layers.append(keras.layers.Dense(n, activation=activation))
        if dropout > 0:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(config.N_ACTIONS, activation="softmax"))
    return compile_model(keras.Sequential(layers), learning_rate, steps_per_execution, jit_compile)


def build_baseline():
    """The baseline: 12 inputs, 16 ReLU units, a 45-way softmax (973 parameters)."""
    return build_network(list(config.BASELINE_HIDDEN), "relu", 0.0, config.LEARNING_RATE)


def build_hypermodel(hp):
    """Search space of spec Table 2."""
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Choice("dropout", [0.0, 0.1, 0.2])
    units = [hp.Choice(f"units_{i}", [4, 8, 16, 32, 64]) for i in range(hp.Int("layers", 1, 3))]
    return build_network(units, activation, dropout, hp.Float("learning_rate", 1e-4, 1e-2, sampling="log"))


def architecture(values):
    """Hidden-layer sizes of a searched configuration. Values of inactive layers are ignored."""
    return [int(values[f"units_{i}"]) for i in range(int(values["layers"]))]


def early_stopping():
    keras = keras_cpu()
    return keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=config.PATIENCE, restore_best_weights=True)


def epoch_reporter(on_epoch):
    """Keras callback that calls on_epoch(epoch, model, val_accuracy) after every epoch."""
    keras = keras_cpu()

    class EpochReporter(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            on_epoch(epoch, self.model, float((logs or {}).get("val_accuracy", float("nan"))))

    return EpochReporter()


def best_of(best, value):
    """The larger of two validation accuracies, ignoring a missing (nan) one."""
    return value if math.isnan(best) else best if math.isnan(value) else max(best, value)


def epoch_status(approach, seed, on_epoch=None):
    """on_epoch of a reported training run: shows the epoch and its validation accuracy in the live status line
    (spec §3.3), then calls on_epoch."""
    began, best = time.time(), [float("nan")]

    def report(epoch, model, val_accuracy):
        best[0] = best_of(best[0], val_accuracy)
        progress.write_status({"approach": approach, "seed": seed, "stage": "imitation", "epoch": epoch + 1,
                               "val_accuracy": val_accuracy, "best": best[0], "elapsed": time.time() - began})
        if on_epoch is not None:
            on_epoch(epoch, model, val_accuracy)

    return report


def fit(model, data, epochs, on_epoch=None):
    """Train with early stopping on validation accuracy. Returns the Keras history and the wall time."""
    callbacks = [early_stopping()]
    if on_epoch is not None:
        callbacks.append(epoch_reporter(on_epoch))
    start = time.time()
    history = model.fit(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                        epochs=epochs, batch_size=config.BATCH_SIZE, callbacks=callbacks, verbose=0)
    return history, time.time() - start


STAGE_FILE = "imitation.keras"     # the trained network in its run directory (spec §8.6)


def finish_run(approach, seed, model, history, wall_time, out_dir, extra=None, report=True):
    """Save the trained network as the imitation stage, then write the run's summary. The summary records the
    settings and is written last, so its presence marks a complete run (spec §3.3). The final controller of an
    imitation method comes from its speed push (spec §8.5), not from here."""
    stage = Path(out_dir) / STAGE_FILE
    model.save(stage)
    summary = {"approach": approach, "seed": seed, "epochs": len(history.history["val_accuracy"]),
               "val_accuracy": float(max(history.history["val_accuracy"])), "params": int(model.count_params()),
               "wall_time": wall_time, "stage_path": str(stage), **(extra or {}),
               "settings": checkpoint.settings_record()}
    checkpoint.write_json(Path(out_dir) / "summary.json", summary)
    if report:
        progress.milestone(f"{config.LABELS[approach]} run {seed}: imitation trained, {summary['epochs']} epochs, "
                           f"val accuracy {progress.accuracy(summary['val_accuracy'])}")
    return summary


def start_run(out_dir):
    """Remove this run's earlier summary and stage file, so an interrupted run leaves neither behind."""
    out_dir = Path(out_dir)
    paths.remove(out_dir / "summary.json", out_dir / STAGE_FILE)
    out_dir.mkdir(parents=True, exist_ok=True)


def previous_run(out_dir, resume):
    """The summary of a complete earlier run in out_dir with the current settings, or None. A summary recorded with
    other settings raises checkpoint.SettingsMismatch (spec §3.3)."""
    if not resume:
        return None
    summary = checkpoint.completed(out_dir)
    if summary is not None and (Path(out_dir) / STAGE_FILE).exists():
        return summary
    return None


def train_baseline(data, out_dir=None, on_epoch=None, epochs=config.MAX_EPOCHS, seed=0, resume=True, report=True):
    """The baseline network with `seed`. With report (the experiment, not the viewer), the live status line shows
    each epoch and the run log its end."""
    out_dir = Path(out_dir or paths.run_dir("baseline", seed))
    done = previous_run(out_dir, resume)
    if done is not None:
        return done
    start_run(out_dir)
    runs.seed_keras(seed, seed)
    model = build_baseline()
    history, wall_time = fit(model, data, epochs, epoch_status("baseline", seed, on_epoch) if report else on_epoch)
    return finish_run("baseline", seed, model, history, wall_time, out_dir,
                      extra={"learning_rate": config.LEARNING_RATE}, report=report)


def make_tuner(approach, directory, seed=0, on_trial=None, max_trials=config.KT_TRIALS,
               max_epochs=config.MAX_EPOCHS, hyperband_iterations=config.HYPERBAND_ITERATIONS,
               bayes_initial=config.KT_BAYES_INITIAL, on_status=None):
    keras_cpu()
    import keras_tuner as kt
    base = getattr(kt, TUNER_CLASSES[approach])

    class ReportingTuner(base):
        def run_trial(self, trial, *args, **kwargs):
            """Record the epochs each trial actually trains (spec §8.2): keras-tuner keeps only its best epoch."""
            histories = super().run_trial(trial, *args, **kwargs)
            self.trained_epochs[trial.trial_id] = sum(len(h.history["val_accuracy"]) for h in histories)
            return histories

        def _configure_tensorboard_dir(self, callbacks, trial, execution=0):
            """No TensorBoard callbacks are used; skip keras-tuner's import of the uninstalled tensorboard."""

        def on_trial_end(self, trial):
            super().on_trial_end(trial)
            if trial.score is not None:
                self.completed += 1
                if on_status is not None:                    # the live status line (no model needed)
                    on_status(self.completed, float(trial.score))
                if on_trial is not None:
                    on_trial(self.completed, self.load_model(trial), float(trial.score))

    common = {"hypermodel": build_hypermodel, "objective": "val_accuracy", "seed": seed,
              "directory": str(directory), "project_name": approach, "overwrite": True}
    if approach == "kt_hyperband":
        tuner = ReportingTuner(max_epochs=max_epochs, factor=config.HYPERBAND_FACTOR,
                               hyperband_iterations=hyperband_iterations, **common)
    elif approach == "kt_bayesian":
        tuner = ReportingTuner(max_trials=max_trials, num_initial_points=bayes_initial, **common)
    else:
        tuner = ReportingTuner(max_trials=max_trials, **common)
    tuner.trained_epochs = {}
    tuner.completed = 0
    return tuner


def trial_records(tuner, max_epochs):
    """One record per completed trial. A Hyperband trial that continues a promoted trial belongs to the same
    configuration (config_id is the first trial of the chain) and counts only the epochs it added."""
    trials = tuner.oracle.trials
    records = []
    for trial in trials.values():
        if trial.score is None:
            continue
        values = dict(trial.hyperparameters.values)
        config_id = trial.trial_id
        while "tuner/trial_id" in trials[config_id].hyperparameters.values:
            config_id = trials[config_id].hyperparameters.values["tuner/trial_id"]
        searched = {k: v for k, v in values.items() if not k.startswith("tuner/")}
        epochs = int(tuner.trained_epochs[trial.trial_id])
        last_epoch = int(values.get("tuner/initial_epoch", 0)) + epochs
        records.append({"trial_id": trial.trial_id, "config_id": config_id, "val_accuracy": float(trial.score),
                        "params": nets.n_params(tuple(architecture(searched))), "epochs": epochs,
                        "last_epoch": last_epoch, "at_limit": last_epoch >= max_epochs,
                        "hyperparameters": searched, "trial": trial})
    return records


def select_smallest(records, tolerance=config.SIZE_RULE_TOLERANCE):
    """Spec §8.3: fewest parameters among records within `tolerance` of the best validation accuracy."""
    best = max(r["val_accuracy"] for r in records)
    eligible = [r for r in records if r["val_accuracy"] >= best - tolerance]
    return min(eligible, key=lambda r: (r["params"], -r["val_accuracy"]))


def most_trained(records):
    """The record that reached the latest epoch in each configuration (spec §8.3, driving-based selection)."""
    best = {}
    for record in records:
        current = best.get(record["config_id"])
        if current is None or record["last_epoch"] >= current["last_epoch"]:
            best[record["config_id"]] = record
    return list(best.values())


def run_tuner(approach, data, out_dir=None, on_trial=None, seed=config.SEARCH_SEED, max_trials=config.KT_TRIALS,
              max_epochs=config.MAX_EPOCHS, hyperband_iterations=config.HYPERBAND_ITERATIONS,
              bayes_initial=config.KT_BAYES_INITIAL, export_models=True, resume=True, report=True):
    """One Keras Tuner search (spec §8.2). Writes trials.csv, summary.json and, with export_models, the most-trained
    model of every configuration. Final controllers come from retrain(). With resume, a complete search recorded with
    the current settings is returned instead of run again (spec §3.3). With report (the experiment, not the
    viewer), the live status line shows each completed trial and the run log the search's end."""
    out_dir = Path(out_dir or paths.search_dir(approach))
    if resume:
        done = checkpoint.completed(out_dir)
        if done is not None:
            return done
    paths.remove(out_dir / "trials.csv", out_dir / "summary.json", out_dir / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    keras_cpu().utils.set_random_seed(seed)
    start, best = time.time(), [float("nan")]
    trials = None if approach == "kt_hyperband" else max_trials     # Hyperband's trial count follows from its budget

    def on_status(completed, score):
        best[0] = best_of(best[0], score)
        progress.write_status({"approach": approach, "stage": "search", "trial": completed, "trials": trials,
                               "best": best[0], "elapsed": time.time() - start})

    if report:
        on_status(0, float("nan"))
    tuner = make_tuner(approach, out_dir / "tuner", seed, on_trial, max_trials, max_epochs, hyperband_iterations,
                       bayes_initial, on_status=on_status if report else None)
    tuner.search(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                 epochs=max_epochs, batch_size=config.BATCH_SIZE, callbacks=[early_stopping()], verbose=0)
    wall_time = time.time() - start
    records = trial_records(tuner, max_epochs)
    chosen = select_smallest(records)
    with open(out_dir / "trials.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "config_id", "val_accuracy", "params", "epochs", "last_epoch", "at_limit",
                         "hyperparameters"])
        for r in records:
            writer.writerow([r["trial_id"], r["config_id"], r["val_accuracy"], r["params"], r["epochs"],
                             r["last_epoch"], r["at_limit"], json.dumps(r["hyperparameters"])])
    if export_models:
        (out_dir / "models").mkdir()
        for record in most_trained(records):
            tuner.load_model(record["trial"]).save(out_dir / "models" / f"{record['trial_id']}.keras")
    summary = {"approach": approach, "seed": seed, "trials": len(records),
               "configurations": len({r["config_id"] for r in records}),
               "total_epochs": int(sum(r["epochs"] for r in records)),
               "trials_at_limit": int(sum(r["at_limit"] for r in records)), "max_epochs": max_epochs,
               "wall_time": wall_time, "best_val_accuracy": max(r["val_accuracy"] for r in records),
               "chosen_trial": chosen["trial_id"], "chosen_val_accuracy": chosen["val_accuracy"],
               "chosen_params": chosen["params"], "chosen_hyperparameters": chosen["hyperparameters"],
               "settings": checkpoint.settings_record()}
    checkpoint.write_json(out_dir / "summary.json", summary)
    if report:
        progress.milestone(f"{config.LABELS[approach]} search: {len(records)} trials, chosen network "
                           f"{chosen['params']} parameters, val accuracy {progress.accuracy(chosen['val_accuracy'])}")
    return summary


def retrain(approach, data, seed, search=None, out_dir=None, epochs=config.MAX_EPOCHS, resume=True, report=True):
    """Train the architecture selected by a search from scratch with `seed` (spec §8.3). With report, the live
    status line shows each epoch and the run log its end."""
    search = Path(search or paths.search_dir(approach))
    out_dir = Path(out_dir or paths.run_dir(approach, seed))
    done = previous_run(out_dir, resume)
    if done is not None:
        return done
    chosen = checkpoint.read_record(search / "summary.json")["chosen_hyperparameters"]
    start_run(out_dir)
    runs.seed_keras(seed, seed)
    model = build_network(architecture(chosen), chosen["activation"], float(chosen["dropout"]),
                          float(chosen["learning_rate"]))
    history, wall_time = fit(model, data, epochs, epoch_status(approach, seed) if report else None)
    return finish_run(approach, seed, model, history, wall_time, out_dir,
                      extra={"hyperparameters": chosen, "learning_rate": float(chosen["learning_rate"])},
                      report=report)
