import json

import numpy as np
import pandas as pd
import pytest

from racing import config, imitation, nets, paths


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(10, 15))

    def make(n):
        x = rng.random((n, 10)).astype(np.float32)
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


def test_baseline_has_431_parameters_and_the_configured_keras_settings():
    model = imitation.build_baseline()
    assert model.count_params() == 431
    assert model.steps_per_execution == config.STEPS_PER_EXECUTION and model.jit_compile == config.JIT_COMPILE


def test_compile_settings_can_be_overridden():
    model = imitation.build_network([8], "relu", 0.0, 1e-3, steps_per_execution=4, jit_compile=False)
    assert model.steps_per_execution == 4 and model.jit_compile is False


def test_parameter_count_follows_from_the_architecture():
    values = {"layers": 2, "units_0": 16, "units_1": 8, "units_2": 64, "activation": "tanh", "dropout": 0.1,
              "learning_rate": 1e-3}
    assert imitation.architecture(values) == [16, 8]
    model = imitation.build_network(imitation.architecture(values), "tanh", 0.1, 1e-3)
    assert nets.n_params(tuple(imitation.architecture(values))) == model.count_params() == 447


def test_load_dataset_round_trip(tmp_path, toy_data):
    np.savez(tmp_path / "d.npz", **toy_data)
    loaded = imitation.load_dataset(tmp_path / "d.npz")
    assert set(loaded) == set(toy_data) and np.array_equal(loaded["y_val"], toy_data["y_val"])


def test_train_baseline_learns_reports_and_saves(toy_data, tmp_path):
    seen = []
    summary = imitation.train_baseline(toy_data, out_dir=tmp_path / "run", champion_dir=tmp_path / "champions",
                                       on_epoch=lambda epoch, model, acc: seen.append((epoch, acc)), epochs=5,
                                       seed=2)
    assert paths.champion_path("baseline", 2, champion_dir=tmp_path / "champions").exists()
    assert len(seen) == summary["epochs"] and summary["val_accuracy"] > 0.2 and summary["seed"] == 2
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["params"] == 431


def test_train_baseline_default_directory_is_per_seed(monkeypatch, toy_data, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    imitation.train_baseline(toy_data, champion_dir=None, epochs=1, seed=1)
    assert (tmp_path / "baseline" / "seed1" / "summary.json").exists()


def test_interrupted_training_leaves_no_stale_results(monkeypatch, toy_data, tmp_path):
    run, champions = tmp_path / "run", tmp_path / "champions"
    run.mkdir()
    (run / "summary.json").write_text("{}")
    stale = paths.champion_path("baseline", 0, champion_dir=champions)
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(imitation, "fit", interrupted)
    with pytest.raises(KeyboardInterrupt):
        imitation.train_baseline(toy_data, out_dir=run, champion_dir=champions, epochs=1)
    assert not (run / "summary.json").exists() and not stale.exists()


def test_train_baseline_is_reproducible_with_a_seed(toy_data, tmp_path):
    runs = [imitation.train_baseline(toy_data, out_dir=tmp_path / f"run{i}", champion_dir=None, epochs=3, seed=3)
            for i in range(2)]
    assert runs[0]["val_accuracy"] == runs[1]["val_accuracy"]


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
    summary = imitation.retrain("kt_random", toy_data, seed=1, search=search, out_dir=tmp_path / "run",
                                champion_dir=tmp_path / "champions", epochs=2)
    assert summary["params"] == nets.n_params((8, 4)) and summary["seed"] == 1 and summary["wall_time"] > 0
    assert paths.champion_path("kt_random", 1, champion_dir=tmp_path / "champions").exists()


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
