import json

import numpy as np
import pytest

from racing import imitation


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


def test_baseline_has_431_parameters():
    assert imitation.build_baseline().count_params() == 431


def test_load_dataset_round_trip(tmp_path, toy_data):
    np.savez(tmp_path / "d.npz", **toy_data)
    loaded = imitation.load_dataset(tmp_path / "d.npz")
    assert set(loaded) == set(toy_data) and np.array_equal(loaded["y_val"], toy_data["y_val"])


def test_train_baseline_learns_reports_and_saves(toy_data, tmp_path):
    seen = []
    summary = imitation.train_baseline(toy_data, out_dir=tmp_path / "run", champion_dir=tmp_path / "champions",
                                       on_epoch=lambda epoch, model, acc: seen.append((epoch, acc)), epochs=5)
    assert (tmp_path / "champions" / "baseline.keras").exists()
    assert len(seen) == summary["epochs"] and summary["val_accuracy"] > 0.2
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["params"] == 431


def test_train_baseline_without_champion(toy_data, tmp_path):
    imitation.train_baseline(toy_data, out_dir=tmp_path / "run", champion_dir=None, epochs=1)
    assert not (tmp_path / "champions").exists()


@pytest.mark.slow
@pytest.mark.parametrize("approach", ["kt_random", "kt_bayesian", "kt_hyperband"])
def test_run_tuner_small(approach, toy_data, tmp_path):
    seen = []
    summary = imitation.run_tuner(approach, toy_data, out_dir=tmp_path / approach, champion_dir=tmp_path / "champions",
                                  on_trial=lambda number, model, acc: seen.append(model.count_params()),
                                  max_trials=3, max_epochs=3, hyperband_iterations=1)
    assert (tmp_path / "champions" / f"{approach}.keras").exists()
    assert summary["trials"] == len(seen) >= 2
    assert (tmp_path / approach / "trials.csv").exists()
    assert summary["chosen_params"] <= max(seen)


def test_train_baseline_is_reproducible_with_a_seed(toy_data, tmp_path):
    runs = [imitation.train_baseline(toy_data, out_dir=tmp_path / f"run{i}", champion_dir=None, epochs=3, seed=3)
            for i in range(2)]
    assert runs[0]["val_accuracy"] == runs[1]["val_accuracy"]
