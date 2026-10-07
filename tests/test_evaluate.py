import json

import numpy as np
import pandas as pd
import pytest

from racing import config, evaluate, nets, paths, population


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(evaluate, "get_track", lambda seed, level: circle_track)
    return circle_track


def drivers(n=3, seed=0):
    flats = nets.init_population(np.random.default_rng(seed), n)
    return {("ga", k): nets.from_flat(flats[k]) for k in range(n)}


def test_driving_results_columns_and_rows(fake_tracks):
    frame = evaluate.driving_results(drivers(), [1000, 1001])
    assert list(frame.columns) == evaluate.DRIVE_COLUMNS and len(frame) == 6
    assert list(frame["track"]) == [1000] * 3 + [1001] * 3 and list(frame["run_seed"][:3]) == [0, 1, 2]


def test_driving_results_in_parallel_match_serial(fake_tracks):
    serial = evaluate.driving_results(drivers(), [1000, 1001, 1002])
    threaded = evaluate.driving_results(drivers(), [1000, 1001, 1002], n_jobs=3, backend="threading")
    pd.testing.assert_frame_equal(serial, threaded)


def test_driving_together_matches_driving_alone(fake_tracks):
    together = evaluate.driving_results(drivers(), [1000])
    alone = evaluate.driving_results({("ga", 1): drivers()[("ga", 1)]}, [1000])
    assert together.iloc[1].equals(alone.iloc[0])


def test_best_seeds_highest_mean_validation_fitness_lowest_seed_on_ties():
    rows = [("ga", 0, 0.5), ("ga", 0, 0.7), ("ga", 1, 0.9), ("ga", 1, 0.9), ("es", 0, 1.0), ("es", 2, 1.0)]
    frame = pd.DataFrame([{"approach": a, "run_seed": s, "fitness": f} for a, s, f in rows])
    assert evaluate.best_seeds(frame) == {"es": 0, "ga": 1}


def test_inference_warm_up_matches_the_timed_calls():
    model = nets.to_keras(nets.init_population(np.random.default_rng(1), 1)[0])
    calls, predict = [], model.predict

    def recording_predict(x, **kwargs):
        calls.append((len(x), kwargs.get("batch_size")))
        return predict(x, **kwargs)

    model.predict = recording_predict
    x = np.random.default_rng(2).random((500, config.N_INPUTS)).astype(np.float32)
    assert 0 < evaluate.inference_time(model, x, repeats=2) < 0.01
    assert calls == [(500, 500)] * 3


def test_single_sample_latency_is_positive():
    model = nets.to_keras(nets.init_population(np.random.default_rng(3), 1)[0])
    x = np.random.default_rng(4).random((50, config.N_INPUTS)).astype(np.float32)
    assert 0 < evaluate.single_sample_latency(model, x, calls=20) < 1.0


def test_mean_std_formats():
    assert evaluate.mean_std([0.5, 0.7, 0.6], 2) == "0.60 ± 0.10"
    assert evaluate.mean_std([431, 431, 431], 0) == "431"
    assert evaluate.mean_std([1.24], 1) == "1.2"
    assert evaluate.mean_std([float("nan")], 1) == "–" and evaluate.mean_std([], 1) == "–"


def metrics_frame():
    rows = []
    for seed, acc in ((0, 0.60), (1, 0.62), (2, 0.64)):
        rows.append({"approach": "kt_random", "run_seed": seed, "accuracy": acc, "params": 447, "batch_us": 2.0,
                     "single_ms": 0.5, "training_s": 30.0 + seed, "search_s": 900.0, "level_reached": np.nan})
    for seed, level in ((0, 4), (1, 3)):
        rows.append({"approach": "ga", "run_seed": seed, "accuracy": 0.2, "params": 431, "batch_us": 2.0,
                     "single_ms": 0.5, "training_s": 600.0, "search_s": np.nan, "level_reached": level})
    return pd.DataFrame(rows)


def test_comparison_table():
    table = evaluate.comparison_table(metrics_frame())
    assert list(table["Method"]) == ["KT Random", "GA"] and list(table["Seeds"]) == [3, 2]
    assert table.loc[0, "Test accuracy"] == "0.6200 ± 0.0200" and table.loc[0, "Parameters"] == "447"
    assert table.loc[0, "Search time (s)"] == "900.0" and table.loc[1, "Search time (s)"] == "–"


def test_driving_table_aggregates_seeds():
    per_track = pd.DataFrame([
        {"approach": "ga", "run_seed": 0, "track": 1000, "finished": True, "crashed": False, "lap_time": 50.0},
        {"approach": "ga", "run_seed": 0, "track": 1001, "finished": False, "crashed": True, "lap_time": np.nan},
        {"approach": "ga", "run_seed": 1, "track": 1000, "finished": True, "crashed": False, "lap_time": 54.0},
        {"approach": "ga", "run_seed": 1, "track": 1001, "finished": True, "crashed": False, "lap_time": 60.0}])
    table = evaluate.driving_table(per_track, metrics_frame())
    row = table.iloc[0]
    assert row["Method"] == "GA" and row["Tracks completed (of 2)"] == "1.5 ± 0.7"
    assert row["Mean lap time (s)"] == "53.50 ± 4.95" and row["Level reached (per seed)"] == "4, 3"


def test_search_cost_and_dataset_tables(tmp_path):
    search = paths.search_dir("kt_hyperband", tmp_path)
    search.mkdir(parents=True)
    (search / "summary.json").write_text(json.dumps({"trials": 254, "configurations": 172, "total_epochs": 3000,
                                                     "trials_at_limit": 9, "wall_time": 1234.5}))
    table = evaluate.search_cost_table(tmp_path)
    assert list(table["Method"]) == ["KT Hyperband"] and table.loc[0, "Configurations"] == 172
    (tmp_path / "dataset_info.json").write_text(json.dumps({"wall_time": 61.0, "n_jobs": 8, "splits": {
        "train": {"tracks": 121, "rows": 90000, "respawns": 5}}}))
    data = evaluate.dataset_table(tmp_path / "dataset.npz")
    assert data.loc[0, "Collection time (s)"] == 61.0 and data.loc[0, "Rows"] == 90000


def test_figures_per_seed(tmp_path):
    runs = tmp_path / "runs"
    for seed in (0, 1):
        folder = paths.run_dir("ga", seed, runs)
        folder.mkdir(parents=True)
        pd.DataFrame({"generation": [0, 1, 2], "level": [1, 1, 2], "evals": [10, 20, 30], "best": [0.1, 0.5, 1.2],
                      "mean": [0.05, 0.2, 0.6], "step_size": [np.nan, 0.25, 0.05],
                      "leveled_up": [False, True, False]}).to_csv(folder / "log.csv", index=False)
    written = evaluate.figures(runs, tmp_path / "figures")
    assert [p.name for p in written] == ["fitness_curves.png", "step_sizes.png"]


def test_chosen_trial_star_with_zero_padded_ids(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    runs = tmp_path / "runs"
    for approach, width, chosen in (("kt_random", 2, "07"), ("kt_hyperband", 4, "0003")):
        search = paths.search_dir(approach, runs)
        search.mkdir(parents=True)
        ids = [f"{i:0{width}d}" for i in range(12)]
        pd.DataFrame({"trial_id": ids, "config_id": ids, "val_accuracy": np.linspace(0.5, 0.9, 12),
                      "params": np.arange(12) * 100 + 50, "epochs": 5}).to_csv(search / "trials.csv", index=False)
        (search / "summary.json").write_text(json.dumps({"chosen_trial": chosen}))
    trials = evaluate.tuner_trials(runs)
    assert list(trials["kt_random"][1]["params"]) == [750] and list(trials["kt_hyperband"][1]["params"]) == [350]
    shown, close = [], plt.close
    monkeypatch.setattr(plt, "close", shown.append)
    evaluate.figures(runs, tmp_path / "figures")
    scatter = shown[-1].axes[0].collections
    for fig in shown:
        close(fig)
    assert [len(c.get_offsets()) for c in scatter] == [12, 1, 12, 1]


def test_main_without_champions_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate.main(champions_dir=tmp_path, runs_dir=tmp_path, data_path=tmp_path / "none.npz", out_dir=tmp_path)


def test_main_with_partial_results(fake_tracks, tmp_path):
    champions, runs = tmp_path / "champions", tmp_path / "runs"
    rng = np.random.default_rng(5)
    for approach, seed in (("ga", 0), ("ga", 2), ("es", 1)):
        population.save_champion(approach, seed, nets.init_population(rng, 1)[0], champions)
    folder = paths.run_dir("ga", 0, runs)
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"wall_time": 12.0, "level_reached": 2}))
    np.savez(tmp_path / "dataset.npz", X_test=rng.random((40, config.N_INPUTS)).astype(np.float32),
             y_test=rng.integers(0, 15, 40))
    results = evaluate.main(champions_dir=champions, runs_dir=runs, data_path=tmp_path / "dataset.npz",
                            out_dir=tmp_path / "out", test_seeds=[1000], val_seeds=[80], latency_calls=5)
    assert list(results["comparison"]["Method"]) == ["ES", "GA"]
    assert json.loads((champions / "best_seeds.json").read_text()).keys() == {"ga", "es"}
    for name in ("comparison", "driving", "search_cost", "dataset"):
        assert (tmp_path / "out" / "tables" / f"{name}.md").exists()
