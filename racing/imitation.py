"""Imitation-learning controllers (spec §8): a baseline network and three Keras Tuner searches."""
import csv
import json
import time
from pathlib import Path

import numpy as np

from racing import config
from racing.backend import keras_cpu

TUNER_CLASSES = {"kt_random": "RandomSearch", "kt_hyperband": "Hyperband", "kt_bayesian": "BayesianOptimization"}


def load_dataset(path=None):
    path = config.DATASET_PATH if path is None else Path(path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def compile_model(model, learning_rate):
    keras = keras_cpu()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.optimizer.build(model.trainable_variables)  # so checkpoints saved with optimizer state load cleanly
    return model


def build_baseline():
    """The 431-parameter baseline: 10 inputs, 16 ReLU units, 15-way softmax."""
    keras = keras_cpu()
    model = keras.Sequential([keras.Input((config.N_INPUTS,)),
                              keras.layers.Dense(config.BASELINE_HIDDEN[0], activation="relu"),
                              keras.layers.Dense(config.N_ACTIONS, activation="softmax")])
    return compile_model(model, config.LEARNING_RATE)


def build_hypermodel(hp):
    """Search space of spec Table 2."""
    keras = keras_cpu()
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Choice("dropout", [0.0, 0.1, 0.2])
    layers = [keras.Input((config.N_INPUTS,))]
    for i in range(hp.Int("layers", 1, 3)):
        layers.append(keras.layers.Dense(hp.Choice(f"units_{i}", [4, 8, 16, 32, 64]), activation=activation))
        if dropout > 0:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(config.N_ACTIONS, activation="softmax"))
    return compile_model(keras.Sequential(layers), hp.Float("learning_rate", 1e-4, 1e-2, sampling="log"))


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


def train_baseline(data, out_dir=None, champion_dir=config.CHAMPIONS_DIR, on_epoch=None, epochs=config.MAX_EPOCHS):
    out_dir = Path(out_dir or config.RUNS_DIR / "baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_baseline()
    callbacks = [early_stopping()]
    if on_epoch is not None:
        callbacks.append(epoch_reporter(on_epoch))
    start = time.time()
    history = model.fit(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                        epochs=epochs, batch_size=config.BATCH_SIZE, callbacks=callbacks, verbose=0)
    summary = {"approach": "baseline", "epochs": len(history.history["val_accuracy"]),
               "val_accuracy": float(max(history.history["val_accuracy"])),
               "params": int(model.count_params()), "wall_time": time.time() - start}
    if champion_dir is not None:
        Path(champion_dir).mkdir(parents=True, exist_ok=True)
        model.save(Path(champion_dir) / "baseline.keras")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def make_tuner(approach, directory, seed=0, on_trial=None, max_trials=config.KT_TRIALS,
               max_epochs=config.MAX_EPOCHS, hyperband_iterations=config.HYPERBAND_ITERATIONS):
    keras_cpu()
    import keras_tuner as kt
    base = getattr(kt, TUNER_CLASSES[approach])

    class ReportingTuner(base):
        completed = 0

        def _configure_tensorboard_dir(self, callbacks, trial, execution=0):
            """No TensorBoard callbacks are used; skip keras-tuner's import of the uninstalled tensorboard."""

        def on_trial_end(self, trial):
            super().on_trial_end(trial)
            if on_trial is not None and trial.score is not None:
                self.completed += 1
                on_trial(self.completed, self.load_model(trial), float(trial.score))

    common = {"hypermodel": build_hypermodel, "objective": "val_accuracy", "seed": seed,
              "directory": str(directory), "project_name": approach, "overwrite": True}
    if approach == "kt_hyperband":
        return ReportingTuner(max_epochs=max_epochs, factor=config.HYPERBAND_FACTOR,
                              hyperband_iterations=hyperband_iterations, **common)
    return ReportingTuner(max_trials=max_trials, **common)


def trial_records(tuner):
    """One record per completed trial: validation accuracy, parameter count, epochs trained, hyperparameters."""
    records = []
    for trial in tuner.oracle.trials.values():
        if trial.score is None:
            continue
        model = tuner.hypermodel.build(trial.hyperparameters)
        records.append({"trial_id": trial.trial_id, "val_accuracy": float(trial.score),
                        "params": int(model.count_params()),
                        "epochs": len(trial.metrics.get_history("val_accuracy")),
                        "hyperparameters": dict(trial.hyperparameters.values), "trial": trial})
    return records


def select_smallest(records, tolerance=config.SIZE_RULE_TOLERANCE):
    """Spec §8.3: fewest parameters among records within `tolerance` of the best validation accuracy."""
    best = max(r["val_accuracy"] for r in records)
    eligible = [r for r in records if r["val_accuracy"] >= best - tolerance]
    return min(eligible, key=lambda r: (r["params"], -r["val_accuracy"]))


def run_tuner(approach, data, out_dir=None, champion_dir=config.CHAMPIONS_DIR, on_trial=None, seed=0,
              max_trials=config.KT_TRIALS, max_epochs=config.MAX_EPOCHS,
              hyperband_iterations=config.HYPERBAND_ITERATIONS):
    out_dir = Path(out_dir or config.RUNS_DIR / approach)
    out_dir.mkdir(parents=True, exist_ok=True)
    tuner = make_tuner(approach, out_dir / "tuner", seed, on_trial, max_trials, max_epochs, hyperband_iterations)
    start = time.time()
    tuner.search(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                 epochs=max_epochs, batch_size=config.BATCH_SIZE, callbacks=[early_stopping()], verbose=0)
    wall_time = time.time() - start
    records = trial_records(tuner)
    chosen = select_smallest(records)
    with open(out_dir / "trials.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "val_accuracy", "params", "epochs", "hyperparameters"])
        for r in records:
            writer.writerow([r["trial_id"], r["val_accuracy"], r["params"], r["epochs"], json.dumps(r["hyperparameters"])])
    summary = {"approach": approach, "trials": len(records), "total_epochs": int(sum(r["epochs"] for r in records)),
               "wall_time": wall_time, "best_val_accuracy": max(r["val_accuracy"] for r in records),
               "chosen_trial": chosen["trial_id"], "chosen_val_accuracy": chosen["val_accuracy"],
               "chosen_params": chosen["params"]}
    if champion_dir is not None:
        Path(champion_dir).mkdir(parents=True, exist_ok=True)
        tuner.load_model(chosen["trial"]).save(Path(champion_dir) / f"{approach}.keras")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
