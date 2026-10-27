import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from racing import checkpoint, config, imitation, nets, paths, progress


def synced_writes(monkeypatch):
    """Names of the files written through checkpoint.write_json, which syncs them to disk."""
    written, real = [], checkpoint.write_json
    monkeypatch.setattr(checkpoint, "write_json",
                        lambda path, data: written.append(Path(path).name) or real(path, data))
    return written


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(config.N_INPUTS, config.N_ACTIONS))

    def make(n):
        x = rng.random((n, config.N_INPUTS)).astype(np.float32)
        return x, np.argmax(x @ w, axis=1).astype(np.int64)

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = make(3000), make(500), make(500)
    return {"X_train": x_tr, "y_train": y_tr, "X_val": x_va, "y_val": y_va, "X_test": x_te, "y_test": y_te}


def test_select_smallest_within_one_point():
    records = [{"val_accuracy": 0.90, "params": 500}, {"val_accuracy": 0.895, "params": 100},
               {"val_accuracy": 0.85, "params": 50}]
    assert imitation.select_smallest(records)["params"] == 100


def test_select_smallest_prefers_accuracy_on_equal_size():
    records = [{"val_accuracy": 0.90, "params": 100}, {"val_accuracy": 0.899, "params": 100}]
    assert imitation.select_smallest(records)["val_accuracy"] == 0.90


def test_baseline_has_1069_parameters_and_the_configured_keras_settings():
    model = imitation.build_baseline()
    assert model.count_params() == nets.n_params() == 1069
    assert model.steps_per_execution == config.STEPS_PER_EXECUTION and model.jit_compile == config.JIT_COMPILE


def test_compile_settings_can_be_overridden():
    model = imitation.build_network([8], "relu", 0.0, 1e-3, steps_per_execution=4, jit_compile=False)
    assert model.steps_per_execution == 4 and model.jit_compile is False


def test_parameter_count_follows_from_the_architecture():
    values = {"layers": 2, "units_0": 16, "units_1": 8, "units_2": 64, "activation": "tanh", "dropout": 0.1,
              "learning_rate": 1e-3}
    assert imitation.architecture(values) == [16, 8]
    model = imitation.build_network(imitation.architecture(values), "tanh", 0.1, 1e-3)
    # 18·16+16 + 16·8+8 + 8·45+45
    assert nets.n_params(tuple(imitation.architecture(values))) == model.count_params() == 845


def test_load_dataset_round_trip(tmp_path, toy_data):
    np.savez(tmp_path / "d.npz", **toy_data)
    loaded = imitation.load_dataset(tmp_path / "d.npz")
    assert set(loaded) == set(toy_data) and np.array_equal(loaded["y_val"], toy_data["y_val"])


def test_load_dataset_refuses_a_dataset_collected_with_other_settings(tmp_path, toy_data):
    from racing import checkpoint
    np.savez(tmp_path / "d.npz", **toy_data)
    record = {"settings": checkpoint.settings_record()}
    checkpoint.write_json(tmp_path / "d_info.json", record)
    assert imitation.dataset_matches(tmp_path / "d.npz") and imitation.load_dataset(tmp_path / "d.npz")
    record["settings"]["N_ACTIONS"] = 15                      # a dataset of version 3.8: labels 0-14
    checkpoint.write_json(tmp_path / "d_info.json", record)
    assert not imitation.dataset_matches(tmp_path / "d.npz")
    with pytest.raises(checkpoint.SettingsMismatch, match="N_ACTIONS"):
        imitation.load_dataset(tmp_path / "d.npz")


def test_train_baseline_learns_reports_and_saves_its_stage(toy_data, tmp_path):
    """Run 0, the seeded run, so the check is the same every time: runs 1-3 start from the operating system's
    randomness (spec §3.3), and one start in about three left run 2 below 0.2 after five epochs."""
    seen = []
    summary = imitation.train_baseline(toy_data, out_dir=tmp_path / "run",
                                       on_epoch=lambda epoch, model, acc: seen.append((epoch, acc)), epochs=5,
                                       seed=0)
    stage = tmp_path / "run" / "imitation.keras"
    assert stage.exists() and summary["stage_path"] == str(stage)
    assert len(seen) == summary["epochs"] and summary["val_accuracy"] > 0.2 and summary["seed"] == 0
    assert summary["learning_rate"] == config.LEARNING_RATE
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["params"] == nets.n_params()
    assert checkpoint.completed(tmp_path / "run")["val_accuracy"] == summary["val_accuracy"]


def test_reported_training_shows_its_progress_and_logs_its_end(toy_data, tmp_path):
    summary = imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=2, seed=1)
    status = progress.read_status()
    assert status["approach"] == "baseline" and status["seed"] == 1 and status["stage"] == "imitation"
    assert status["epoch"] == summary["epochs"] and status["best"] == pytest.approx(summary["val_accuracy"])
    assert "Baseline run 1: imitation trained" in config.RUN_LOG_PATH.read_text()


def test_viewer_training_reports_nothing(toy_data, tmp_path):
    imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, resume=False, report=False)
    assert not config.STATUS_PATH.exists() and not config.RUN_LOG_PATH.exists()


def test_a_retrained_network_shows_its_progress(monkeypatch, toy_data, tmp_path):
    search = tmp_path / "search"
    search.mkdir()
    chosen = {"layers": 1, "units_0": 8, "activation": "relu", "dropout": 0.0, "learning_rate": 0.001}
    (search / "summary.json").write_text(json.dumps({"chosen_hyperparameters": chosen}))
    imitation.retrain("kt_random", toy_data, seed=2, search=search, out_dir=tmp_path / "run", epochs=1)
    status = progress.read_status()
    assert status["approach"] == "kt_random" and status["seed"] == 2 and status["stage"] == "imitation"
    assert "KT Random run 2: imitation trained" in config.RUN_LOG_PATH.read_text()


def test_the_training_summary_is_synced_to_disk(monkeypatch, toy_data, tmp_path):
    written = synced_writes(monkeypatch)
    imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1)
    assert written == ["summary.json"]


def test_train_baseline_saves_its_stage_per_seed_and_no_final_controller(monkeypatch, toy_data, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "CHAMPIONS_DIR", tmp_path / "champions")
    imitation.train_baseline(toy_data, epochs=1, seed=1)
    assert paths.stage_path("baseline", 1, "imitation").exists()
    assert (tmp_path / "runs" / "baseline" / "seed1" / "summary.json").exists()
    assert not (tmp_path / "champions").exists()


def test_interrupted_training_leaves_no_stale_results(monkeypatch, toy_data, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text("{}")
    (run / "imitation.keras").write_bytes(b"stale")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(imitation, "fit", interrupted)
    with pytest.raises(KeyboardInterrupt):
        imitation.train_baseline(toy_data, out_dir=run, epochs=1, resume=False)
    assert not (run / "summary.json").exists() and not (run / "imitation.keras").exists()


def test_train_baseline_is_reproducible_with_a_seed(toy_data, tmp_path):
    runs = [imitation.train_baseline(toy_data, out_dir=tmp_path / f"run{i}", epochs=3, seed=0) for i in range(2)]
    assert runs[0]["val_accuracy"] == runs[1]["val_accuracy"]


def test_complete_training_is_not_repeated(monkeypatch, toy_data, tmp_path):
    first = imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0)
    monkeypatch.setattr(imitation, "fit", lambda *args, **kwargs: pytest.fail("trained again"))
    again = imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0)
    assert again == json.loads(json.dumps(first))


def test_training_without_resume_or_without_its_stage_file_trains_again(monkeypatch, toy_data, tmp_path):
    imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0)
    real_fit, calls = imitation.fit, []
    monkeypatch.setattr(imitation, "fit", lambda *args, **kwargs: calls.append(1) or real_fit(*args, **kwargs))
    imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0, resume=False)
    (tmp_path / "run" / "imitation.keras").unlink()
    imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0)
    assert len(calls) == 2


def test_training_recorded_with_other_settings_is_refused(monkeypatch, toy_data, tmp_path):
    imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0)
    monkeypatch.setattr(checkpoint, "settings_record", lambda approach=None: {"MAX_EPOCHS": 1})
    with pytest.raises(checkpoint.SettingsMismatch):
        imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0)


def test_tuner_settings(tmp_path):
    bayes = imitation.make_tuner("kt_bayesian", tmp_path / "b")
    assert bayes.oracle.num_initial_points == config.KT_BAYES_INITIAL
    band = imitation.make_tuner("kt_hyperband", tmp_path / "h")
    assert band.oracle.max_epochs == config.MAX_EPOCHS and band.oracle.hyperband_iterations == 1
    assert band.trained_epochs == {}


def test_retrain_builds_the_chosen_architecture(toy_data, tmp_path):
    search = tmp_path / "search"
    search.mkdir()
    chosen = {"layers": 2, "units_0": 8, "units_1": 4, "activation": "tanh", "dropout": 0.0, "learning_rate": 0.003}
    (search / "summary.json").write_text(json.dumps({"chosen_hyperparameters": chosen}))
    summary = imitation.retrain("kt_random", toy_data, seed=1, search=search, out_dir=tmp_path / "run", epochs=2)
    assert summary["params"] == nets.n_params((8, 4)) and summary["seed"] == 1 and summary["wall_time"] > 0
    assert summary["learning_rate"] == 0.003 and (tmp_path / "run" / "imitation.keras").exists()


def test_a_complete_search_is_not_run_again(monkeypatch, toy_data, tmp_path):
    out = tmp_path / "kt_random"
    out.mkdir()
    done = {"approach": "kt_random", "wall_time": 12.5, "settings": checkpoint.settings_record()}
    (out / "summary.json").write_text(json.dumps(done))
    monkeypatch.setattr(imitation, "make_tuner", lambda *args, **kwargs: pytest.fail("searched again"))
    assert imitation.run_tuner("kt_random", toy_data, out_dir=out)["wall_time"] == 12.5


class FakeTrial:
    def __init__(self, trial_id, values, score):
        self.trial_id, self.score = trial_id, score
        self.hyperparameters = types.SimpleNamespace(values=values)


class FakeTuner:
    """One completed trial, without Keras Tuner."""

    def __init__(self):
        values = {"layers": 1, "units_0": 8, "activation": "relu", "dropout": 0.0, "learning_rate": 0.001}
        self.oracle = types.SimpleNamespace(trials={"00": FakeTrial("00", values, 0.5)})
        self.trained_epochs = {"00": 3}

    def search(self, *args, **kwargs):
        pass


def test_the_search_summary_is_synced_to_disk(monkeypatch, toy_data, tmp_path):
    monkeypatch.setattr(imitation, "make_tuner", lambda *args, **kwargs: FakeTuner())
    written = synced_writes(monkeypatch)
    summary = imitation.run_tuner("kt_random", toy_data, out_dir=tmp_path / "search", export_models=False)
    assert written == ["summary.json"] and summary["chosen_trial"] == "00"
    assert checkpoint.completed(tmp_path / "search")["chosen_params"] == nets.n_params((8,))


def test_a_search_shows_each_trial_and_logs_its_end(monkeypatch, toy_data, tmp_path):
    def make_tuner(*args, on_status=None, **kwargs):
        tuner = FakeTuner()
        tuner.search = lambda *a, **k: on_status(1, 0.5)
        return tuner

    monkeypatch.setattr(imitation, "make_tuner", make_tuner)
    imitation.run_tuner("kt_random", toy_data, out_dir=tmp_path / "search", export_models=False)
    status = progress.read_status()
    assert status["stage"] == "search" and status["approach"] == "kt_random"
    assert status["trial"] == 1 and status["trials"] == config.KT_TRIALS and status["best"] == 0.5
    assert "KT Random search: 1 trials" in config.RUN_LOG_PATH.read_text()


def test_a_viewer_search_reports_nothing(monkeypatch, toy_data, tmp_path):
    def make_tuner(*args, on_status=None, **kwargs):
        assert on_status is None
        return FakeTuner()

    monkeypatch.setattr(imitation, "make_tuner", make_tuner)
    imitation.run_tuner("kt_random", toy_data, out_dir=tmp_path / "search", export_models=False, resume=False,
                        report=False)
    assert not config.STATUS_PATH.exists() and not config.RUN_LOG_PATH.exists()


@pytest.mark.slow
def test_a_real_search_shows_each_trial(toy_data, tmp_path):
    run_small_search("kt_random", toy_data, tmp_path)
    status = progress.read_status()
    assert status["stage"] == "search" and status["trial"] == 3 and status["trials"] == 3


def run_small_search(approach, toy_data, tmp_path, seen=None):
    return imitation.run_tuner(approach, toy_data, out_dir=tmp_path / approach,
                               on_trial=None if seen is None else (lambda n, model, acc: seen.append(n)),
                               max_trials=3, max_epochs=3, hyperband_iterations=1)


@pytest.mark.slow
@pytest.mark.parametrize("approach", ["kt_random", "kt_bayesian"])
def test_trial_epochs_are_the_epochs_trained(approach, toy_data, tmp_path):
    seen = []
    summary = run_small_search(approach, toy_data, tmp_path, seen)
    trials = pd.read_csv(tmp_path / approach / "trials.csv", dtype={"trial_id": str, "config_id": str})
    assert summary["trials"] == len(seen) == len(trials) >= 2
    assert list(trials["epochs"]) == [3] * len(trials)      # patience 5 > 3 epochs: no trial stops early
    assert summary["total_epochs"] == 3 * len(trials) and summary["trials_at_limit"] == len(trials)
    assert summary["configurations"] == len(trials) and (trials["trial_id"] == trials["config_id"]).all()
    assert len(list((tmp_path / approach / "models").glob("*.keras"))) == len(trials)
    assert checkpoint.completed(tmp_path / approach)["trials"] == summary["trials"]


@pytest.mark.slow
def test_hyperband_epochs_and_configurations(toy_data, tmp_path):
    summary = run_small_search("kt_hyperband", toy_data, tmp_path)
    trials = pd.read_csv(tmp_path / "kt_hyperband" / "trials.csv", dtype={"trial_id": str, "config_id": str})
    promoted = trials[trials["trial_id"] != trials["config_id"]]
    assert len(promoted) >= 1
    assert set(promoted["epochs"]) == {2} and set(promoted["last_epoch"]) == {3}   # continues from epoch 1 to 3
    assert summary["configurations"] == trials["config_id"].nunique() < summary["trials"]
    assert summary["total_epochs"] == int(trials["epochs"].sum())
    exported = sorted(p.stem for p in (tmp_path / "kt_hyperband" / "models").glob("*.keras"))
    assert len(exported) == summary["configurations"]


STEPS = '''
import sys
from pathlib import Path
import numpy as np
from racing import config, imitation

rng = np.random.default_rng(0)
w = rng.normal(size=(config.N_INPUTS, config.N_ACTIONS))


def make(n):
    x = rng.random((n, config.N_INPUTS)).astype(np.float32)
    return x, np.argmax(x @ w, axis=1).astype(np.int64)


(xt, yt), (xv, yv), (xs, ys) = make(3000), make(500), make(500)
data = {"X_train": xt, "y_train": yt, "X_val": xv, "y_val": yv, "X_test": xs, "y_test": ys}
out, search = Path(sys.argv[1]), Path(sys.argv[2])
for step in sys.argv[3:]:
    if step == "baseline":
        imitation.train_baseline(data, out_dir=out / "baseline", epochs=2, seed=0, report=False)
    elif step == "search":
        imitation.run_tuner("kt_random", data, out_dir=search, max_trials=2, max_epochs=2, export_models=False,
                            report=False)
    else:
        imitation.retrain("kt_random", data, seed=0, search=search, out_dir=out / "kt_random", epochs=2,
                          report=False)
'''


def _steps(out, search, *steps):
    """Run notebook 02's steps `steps` in one fresh Python process."""
    env = {**os.environ, "PYTHONPATH": str(config.PROJECT_DIR), "CUDA_VISIBLE_DEVICES": "-1",
           "TF_CPP_MIN_LOG_LEVEL": "3"}
    subprocess.run([sys.executable, "-c", STEPS, str(out), str(search), *steps], check=True, env=env,
                   cwd=config.PROJECT_DIR)


@pytest.mark.slow
def test_a_search_and_a_retraining_in_fresh_processes_equal_notebook_order(tmp_path):
    """Every step sets its random seed when it starts, so a network trained in a job of its own equals the network
    notebook 02 trains in order, after other networks, in one process (spec §3.3, §12.1)."""
    from racing.backend import keras_cpu
    _steps(tmp_path / "order", tmp_path / "order" / "search", "baseline", "search", "retrain")
    _steps(tmp_path / "jobs", tmp_path / "jobs" / "search", "search")
    _steps(tmp_path / "jobs", tmp_path / "jobs" / "search", "retrain")
    in_order = json.loads((tmp_path / "order" / "search" / "summary.json").read_text())
    alone = json.loads((tmp_path / "jobs" / "search" / "summary.json").read_text())
    for key in ("chosen_hyperparameters", "best_val_accuracy", "chosen_val_accuracy", "total_epochs", "trials"):
        assert in_order[key] == alone[key], key
    load = keras_cpu().models.load_model
    for a, b in zip(load(tmp_path / "order" / "kt_random" / "imitation.keras").get_weights(),
                    load(tmp_path / "jobs" / "kt_random" / "imitation.keras").get_weights(), strict=True):
        assert np.array_equal(a, b)


def test_every_keras_job_trains_with_the_configured_tensorflow_threads():
    """A tiny network trains fastest on few threads, and few threads stall least beside compiled runs (spec §3.3)."""
    code = ("from racing.backend import keras_cpu; keras_cpu(); import tensorflow as tf; "
            "print(tf.config.threading.get_intra_op_parallelism_threads(), "
            "tf.config.threading.get_inter_op_parallelism_threads())")
    env = {**os.environ, "PYTHONPATH": str(config.PROJECT_DIR), "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env,
                         cwd=config.PROJECT_DIR).stdout.split()
    assert out == [str(config.KERAS_THREADS), str(config.KERAS_INTER_THREADS)] == ["4", "2"]


def test_run_1_trains_another_network_every_time_even_after_run_0(toy_data, tmp_path):
    from racing.backend import keras_cpu
    load = keras_cpu().models.load_model
    for name, seed in (("a0", 0), ("a1", 1), ("b0", 0), ("b1", 1)):            # one process, run 0 then run 1
        imitation.train_baseline(toy_data, out_dir=tmp_path / name, epochs=1, seed=seed, report=False)
    w = {name: load(tmp_path / name / "imitation.keras").get_weights() for name in ("a0", "a1", "b0", "b1")}
    assert all(np.array_equal(x, y) for x, y in zip(w["a0"], w["b0"]))
    assert not all(np.array_equal(x, y) for x, y in zip(w["a1"], w["b1"]))
