import json

import numpy as np
import pandas as pd
import pytest

from racing import config, evaluate, nets


@pytest.fixture
def models():
    rng = np.random.default_rng(0)
    return {name: nets.to_keras(nets.init_population(rng, 1)[0]) for name in ("baseline", "ga")}


def test_driving_results_rows(monkeypatch, circle_track, models):
    monkeypatch.setattr(evaluate, "get_track", lambda seed, level: circle_track)
    dense = {name: nets.from_keras(m) for name, m in models.items()}
    frame = evaluate.driving_results(dense, seeds=[1000, 1001])
    assert len(frame) == 4
    assert list(frame.columns) == ["approach", "seed", "finished", "crashed", "lap_time", "progress"]


def test_accuracy_matches_manual(models):
    x = np.random.default_rng(1).random((100, config.N_INPUTS)).astype(np.float32)
    predicted = np.argmax(models["ga"].predict(x, verbose=0), axis=1)
    y = predicted.copy()
    y[:25] = (y[:25] + 1) % config.N_ACTIONS
    assert abs(evaluate.accuracy(models["ga"], x, y) - 0.75) < 1e-9


def test_inference_time_is_positive(models):
    x = np.random.default_rng(2).random((500, config.N_INPUTS)).astype(np.float32)
    assert 0 < evaluate.inference_time(models["ga"], x, repeats=2) < 0.01


def test_tables_and_missing_summaries(tmp_path, models):
    (tmp_path / "ga").mkdir()
    (tmp_path / "ga" / "summary.json").write_text(json.dumps({"wall_time": 12.0, "level_reached": 3}))
    rng = np.random.default_rng(3)
    data = {"X_test": rng.random((50, config.N_INPUTS)).astype(np.float32), "y_test": rng.integers(0, 15, 50)}
    comparison = evaluate.comparison_table(models, data, tmp_path)
    assert list(comparison["Method"]) == ["Baseline", "GA"]
    assert np.isnan(comparison.loc[0, "Search/training time (s)"])
    results = pd.DataFrame([
        {"approach": "baseline", "seed": 1000, "finished": True, "crashed": False, "lap_time": 50.0, "progress": 1.0},
        {"approach": "ga", "seed": 1000, "finished": False, "crashed": True, "lap_time": np.nan, "progress": 0.3}])
    driving = evaluate.driving_table(results, tmp_path)
    assert list(driving["Level reached"]) == ["n/a", 3]
    assert "| Method |" in evaluate.to_markdown(driving)


def test_figures_with_partial_logs(tmp_path):
    runs = tmp_path / "runs"
    (runs / "ga").mkdir(parents=True)
    pd.DataFrame({"generation": [0, 1, 2], "level": [1, 1, 2], "evals": [10, 20, 30], "best": [0.1, 0.5, 1.2],
                  "mean": [0.05, 0.2, 0.6], "worst": [0, 0, 0], "kept_finish": [np.nan, 1, 1],
                  "new_finish": [0, 0.5, 0.9], "step_size": [0.5, 0.25, 0.05], "seeds": ["0 1 2"] * 3,
                  "elapsed": [1, 2, 3], "leveled_up": [False, True, False]}).to_csv(runs / "ga" / "log.csv", index=False)
    written = evaluate.figures(runs, tmp_path / "figures")
    assert [p.name for p in written] == ["fitness_curves.png", "step_sizes.png"]


def test_main_without_champions_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate.main(champions_dir=tmp_path, runs_dir=tmp_path, data_path=tmp_path / "none.npz", out_dir=tmp_path)
