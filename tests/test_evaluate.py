import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from racing import config, evaluate, nets, neat, paths, population, sim


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(evaluate, "get_track", lambda seed, level: circle_track)
    return circle_track


@pytest.fixture
def requested_maps(monkeypatch, circle_track):
    """Like fake_tracks, and records the (seed, level) of every map asked for."""
    requested = []
    monkeypatch.setattr(evaluate, "get_track", lambda seed, level: requested.append((seed, level)) or circle_track)
    return requested


def drivers(n=3, seed=0):
    flats = nets.init_population(np.random.default_rng(seed), n)
    return {("ga", k): nets.from_flat(flats[k]) for k in range(n)}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def drive_rows(rows):
    """A per-track frame from (approach, seed, track, finished, succeeded, crashed, lap time) tuples."""
    return pd.DataFrame([{"approach": a, "run_seed": s, "track": t, "finished": f, "success": ok, "crashed": c,
                          "race_time": lap * config.LAPS, "lap_time": lap, "progress": 1.0 if f else 0.5,
                          "fitness": 1.5 if f else 0.5, "circuit": "", "lap_1": lap, "lap_2": lap, "lap_3": lap,
                          "flying_lap": lap if f else float("nan")} for a, s, t, f, ok, c, lap in rows],
                        columns=evaluate.DRIVE_COLUMNS)


def test_controllers_drive_all_laps(monkeypatch, fake_tracks):
    seen = []
    real = evaluate.sim.simulate

    def recording(track, policy, n, laps=1, **kwargs):
        seen.append(laps)
        return real(track, policy, n, laps=laps, **kwargs)

    monkeypatch.setattr(evaluate.sim, "simulate", recording)
    frame = evaluate.driving_results(drivers(), [1000])
    assert seen == [config.LAPS]
    done = frame[frame["finished"]]
    assert np.allclose(done["lap_time"], done["race_time"] / config.LAPS)


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


def test_success_means_three_laps_within_the_target(monkeypatch, fake_tracks):
    result = sim.RunResult(fitness=np.array([1.2, 1.1, 0.4]), finished=np.array([True, True, False]),
                           crashed=np.array([False, False, True]), progress=np.array([1.0, 1.0, 0.3]),
                           finish_time=np.array([70.0, 90.0, np.nan]), steps=1, laps_done=np.array([3, 3, 0]),
                           lap_times=np.full((3, 3), np.nan))
    monkeypatch.setattr(evaluate.sim, "simulate", lambda track, policy, n, **kwargs: result)
    frame = evaluate.driving_results(drivers(), [1000])          # target: 1.3 × 20 s × 3 laps = 78 s
    assert list(frame["finished"]) == [True, True, False] and list(frame["success"]) == [True, False, False]
    assert frame["lap_time"].iloc[1] == pytest.approx(30.0)
    assert list(frame["expert_lap"]) == [fake_tracks.expert_lap_time] * 3


def test_best_seeds_fastest_reliable_run_then_most_finishes_then_highest_fitness_then_run_0(monkeypatch):
    """Spec §11.1 (version 3.11): of the runs that finish at least FINAL_MIN_SUCCESSES validation maps, the fastest
    mean lap relative to each map's expert lap over the maps they finished; if none, the most finishes, then the
    highest mean validation fitness; then the lower run number."""
    monkeypatch.setattr(config, "FINAL_MIN_SUCCESSES", 2)
    rows = [("ga", 0, True, 30.0, 20.0, 1.2), ("ga", 0, True, 30.0, 20.0, 1.2),      # reliable, 1.5 × the expert
            ("ga", 1, True, 24.0, 20.0, 1.4), ("ga", 1, True, 48.0, 40.0, 1.4),      # reliable, 1.2 ×: the pick
            ("ga", 2, True, 20.0, 20.0, 1.5), ("ga", 2, False, 200.0, 20.0, 0.3),    # fastest, 1 finish: not reliable
            ("es", 0, True, 30.0, 20.0, 1.2), ("es", 0, False, 5.0, 20.0, 0.9),      # none reliable: 1 finish each ...
            ("es", 1, True, 24.0, 20.0, 1.4), ("es", 1, False, 20.0, 20.0, 0.2),     # ... faster, lower fitness
            ("es", 2, False, 20.0, 20.0, 0.1), ("es", 2, False, 20.0, 20.0, 0.1),    # ... so run 0 wins
            ("neat", 2, False, 25.0, 20.0, 0.6), ("neat", 1, False, 25.0, 20.0, 0.4),  # no finish: the higher fitness
            ("neat", 3, False, 25.0, 20.0, 0.6),                                       # tied with run 2: the lower run
            ("baseline", 3, True, 24.0, 20.0, 1.4), ("baseline", 3, True, 24.0, 20.0, 1.4),   # reliable, equally
            ("baseline", 1, True, 24.0, 20.0, 1.4), ("baseline", 1, True, 24.0, 20.0, 1.4)]   # fast: the lower run
    frame = pd.DataFrame([{"approach": a, "run_seed": r, "finished": ok, "lap_time": lap, "expert_lap": e,
                           "fitness": f} for a, r, ok, lap, e, f in rows])
    assert evaluate.best_seeds(frame) == {"baseline": 1, "es": 0, "ga": 1, "neat": 2}


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


def test_every_controller_is_timed_with_the_same_prediction_settings():
    flat = nets.init_population(np.random.default_rng(9), 1)[0]
    trained = nets.to_keras(flat)
    trained.compile(optimizer="adam", loss="sparse_categorical_crossentropy", jit_compile=True,
                    steps_per_execution=4)
    x = np.random.default_rng(10).random((64, config.N_INPUTS)).astype(np.float32)
    trained.predict(x, verbose=0)                       # builds a predict function with the training settings
    for model in (trained, nets.to_keras(flat)):
        evaluate.same_prediction_settings(model)
        assert model.jit_compile is False and model.steps_per_execution == 1 and model.predict_function is None
        assert model.predict(x, verbose=0).shape == (64, config.N_ACTIONS)


def stub_timing(monkeypatch, order=None):
    record = order.append if order is not None else (lambda name: None)
    monkeypatch.setattr(evaluate, "same_prediction_settings", lambda model: record("settings"))
    monkeypatch.setattr(evaluate, "accuracy", lambda model, x, y: record("accuracy") or 0.5)
    monkeypatch.setattr(evaluate, "inference_time", lambda model, x: record("batch") or 1e-6)
    monkeypatch.setattr(evaluate, "single_sample_latency", lambda model, x, calls: record("single") or 1e-3)


def test_metrics_apply_the_prediction_settings_before_any_prediction(monkeypatch, tmp_path):
    order = []
    stub_timing(monkeypatch, order)
    model = nets.to_keras(nets.init_population(np.random.default_rng(11), 1)[0])
    data = {"X_test": np.zeros((4, config.N_INPUTS), np.float32), "y_test": np.zeros(4, int)}
    evaluate.metrics_per_seed({("ga", 0): model}, data, tmp_path)
    assert order == ["settings", "accuracy", "batch", "single"]


def test_metrics_count_driver_parameters_and_all_imitation_training(monkeypatch, tmp_path):
    stub_timing(monkeypatch)

    class NeatLike:
        n_params = 19

    model = nets.to_keras(nets.init_population(np.random.default_rng(13), 1)[0])
    write_json(paths.run_dir("baseline", 0, tmp_path) / "summary.json", {"wall_time": 10.0})
    write_json(paths.run_dir("baseline", 0, tmp_path) / "dagger.json", {"wall_time": 20.0, "rounds": 3})
    write_json(paths.speed_dir("baseline", 0, tmp_path) / "summary.json", {"wall_time": 30.0})
    data = {"X_test": np.zeros((4, config.N_INPUTS), np.float32), "y_test": np.zeros(4, int)}
    metrics = evaluate.metrics_per_seed({("neat", 0): model, ("baseline", 0): model}, data, tmp_path,
                                        drivers={("neat", 0): NeatLike()}).set_index("approach")
    assert metrics.loc["neat", "params"] == 19 and metrics.loc["baseline", "params"] == model.count_params()
    assert metrics.loc["baseline", "training_s"] == 60.0 and np.isnan(metrics.loc["neat", "training_s"])


def test_single_sample_latency_is_positive():
    model = nets.to_keras(nets.init_population(np.random.default_rng(3), 1)[0])
    x = np.random.default_rng(4).random((50, config.N_INPUTS)).astype(np.float32)
    assert 0 < evaluate.single_sample_latency(model, x, calls=20) < 1.0


def test_runs_cell_shows_every_run_their_mean_and_sd():
    assert evaluate.runs_cell([31, 28, 35, 30], 0) == "31 · 28 · 35 · 30 (31 ± 3)"
    assert evaluate.runs_cell([0.62, 0.60, float("nan"), 0.64], 2) == "0.62 · 0.60 · – · 0.64 (0.62 ± 0.02)"
    assert evaluate.runs_cell([52.0, float("nan"), float("nan"), float("nan")], 2) == "52.00 · – · – · – (52.00)"
    assert evaluate.runs_cell([447, 447, 447, 447], 0) == "447"
    assert evaluate.runs_cell([1, 0, 2, 1], 0, mean_digits=1) == "1 · 0 · 2 · 1 (1.0 ± 0.8)"
    assert evaluate.runs_cell([float("nan")] * 4, 1) == "–" and evaluate.runs_cell([], 1) == "–"
    assert evaluate.runs_cell([1.24], 1) == "1.2"
    assert evaluate.runs_cell([18, 16], 0) == "18 · 16 (17 ± 1)"
    assert evaluate.runs_cell([0, 0], 0, mean_digits=1) == "0"


def metrics_frame():
    rows = []
    for seed, acc in ((0, 0.60), (1, 0.62), (2, 0.64)):
        rows.append({"approach": "kt_random", "run_seed": seed, "accuracy": acc, "params": 447, "batch_us": 2.0,
                     "single_ms": 0.5, "training_s": 30.0 + seed, "search_s": 900.0})
    for seed in (0, 1):
        rows.append({"approach": "ga", "run_seed": seed, "accuracy": 0.2, "params": 431, "batch_us": 2.0,
                     "single_ms": 0.5, "training_s": 600.0, "search_s": np.nan})
    return pd.DataFrame(rows)


def test_comparison_table():
    table = evaluate.comparison_table(metrics_frame())
    assert list(table["Method"]) == ["KT Random", "GA"] and list(table["Runs"]) == [3, 2]
    assert table.loc[0, "Test accuracy"] == "0.6000 · 0.6200 · 0.6400 · – (0.6200 ± 0.0200)"
    assert table.loc[0, "Parameters"] == "447 · 447 · 447 · – (447 ± 0)"
    assert table.loc[0, "Training time (s)"] == "30.0 · 31.0 · 32.0 · – (31.0 ± 1.0)"
    assert table.loc[0, "Search time (s)"] == "900.0" and table.loc[1, "Search time (s)"] == "–"


def test_comparison_table_lists_the_runs_in_run_order():
    shuffled = metrics_frame().iloc[[2, 0, 4, 1, 3]].reset_index(drop=True)
    pd.testing.assert_frame_equal(evaluate.comparison_table(shuffled), evaluate.comparison_table(metrics_frame()))


def test_results_table_counts_finishes_successes_and_target_reduction(tmp_path):
    nan = np.nan
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0), ("ga", 0, 1001, False, False, True, nan),
                            ("ga", 1, 1000, True, False, False, 54.0), ("ga", 1, 1001, True, True, False, 60.0),
                            ("es", 0, 1000, True, True, False, 52.0), ("es", 0, 1001, False, False, True, nan)])
    write_json(paths.run_dir("ga", 0, tmp_path) / "summary.json", {"goal_reached": True, "target_reduction": 0.143})
    write_json(paths.run_dir("ga", 1, tmp_path) / "summary.json", {"goal_reached": False})
    table = evaluate.results_table(per_track, tmp_path)
    assert list(table["Method"]) == ["ES", "GA"]
    ga = table.set_index("Method").loc["GA"]
    assert ga["Runs"] == 2 and ga["Finished"] == "3 / 4" and ga["Finished per run"] == "1 · 2 · – · –"
    assert ga["Succeeded"] == "2 / 4" and ga["Succeeded per run"] == "1 · 1 · – · –"
    assert ga["Mean lap time (s)"] == "50.00 · 57.00 · – · – (53.50 ± 4.95)" and ga["Best lap time (s)"] == "50.00"
    assert ga["Mean race time (s)"] == "150.0 · 171.0 · – · – (160.5 ± 14.8)"
    assert ga["Crashes"] == "1 · 0 · – · – (0.5 ± 0.7)"
    assert ga["Target reduction (per run)"] == "14.3% · not reached · – · – (14.3%)"
    assert table.set_index("Method").loc["ES", "Target reduction (per run)"] == "– · – · – · –"


def test_results_table_shows_a_dash_for_a_run_that_finished_no_map(tmp_path):
    nan = np.nan
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0), ("ga", 0, 1001, True, False, False, 54.0),
                            ("ga", 1, 1000, False, False, True, nan), ("ga", 1, 1001, False, False, True, nan)])
    ga = evaluate.results_table(per_track, tmp_path).set_index("Method").loc["GA"]
    assert ga["Mean lap time (s)"] == "52.00 · – · – · – (52.00)"
    assert ga["Mean race time (s)"] == "156.0 · – · – · – (156.0)"
    assert ga["Crashes"] == "0 · 2 · – · – (1.0 ± 1.4)" and ga["Finished per run"] == "2 · 0 · – · –"


def test_a_run_without_rows_shows_a_dash_and_an_archives_other_runs_follow(tmp_path):
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0), ("ga", 2, 1000, True, False, False, 54.0),
                            ("es", 0, 1000, True, True, False, 52.0), ("es", 0, 1001, False, False, True, np.nan)])
    table = evaluate.results_table(per_track, tmp_path).set_index("Method")
    es, ga = table.loc["ES"], table.loc["GA"]
    assert es["Runs"] == 1 and es["Mean lap time (s)"] == "52.00 · – · – · – (52.00)"
    assert es["Crashes"] == "1 · – · – · – (1.0)"
    assert es["Finished per run"] == "1 · – · – · –" and es["Succeeded per run"] == "1 · – · – · –"
    assert ga["Mean lap time (s)"] == "50.00 · – · 54.00 · – (52.00 ± 2.83)"
    assert ga["Finished per run"] == "1 · – · 1 · –"
    assert ga["Target reduction (per run)"] == "– · – · – · –"


def test_results_table_names_each_methods_best_run(tmp_path):
    per_track = drive_rows([(a, s, 1000, True, True, False, 50.0) for a in ("ga", "es", "neat") for s in (0, 1)])
    rows = [("ga", 0, True, 30.0, 1.2), ("ga", 0, False, 30.0, 0.5), ("ga", 1, True, 30.0, 1.2),
            ("ga", 1, True, 30.0, 1.2), ("es", 0, True, 30.0, 1.2), ("es", 0, False, 30.0, 0.5),
            ("es", 1, True, 24.0, 1.4), ("es", 1, False, 24.0, 0.5),
            ("neat", 0, True, 30.0, 1.2), ("neat", 1, True, 30.0, 1.2)]   # none reliable: finishes, fitness, run 0
    validation = pd.DataFrame([{"approach": a, "run_seed": s, "finished": ok, "lap_time": lap, "expert_lap": 20.0,
                                "fitness": f} for a, s, ok, lap, f in rows])
    table = evaluate.results_table(per_track, tmp_path, best=evaluate.best_seeds(validation)).set_index("Method")
    assert table["Best run"].to_dict() == {"ES": 1, "GA": 1, "NEAT": 0}
    assert "Best run" not in evaluate.results_table(per_track, tmp_path).columns


def test_results_table_reports_the_final_setups(tmp_path):
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0), ("ga", 1, 1000, True, True, False, 54.0),
                            ("es", 0, 1000, True, True, False, 52.0)])
    finals = drivers(n=2)
    finals[("ga", 0)].setup = np.array([40.0, 2.0])
    finals[("ga", 1)].setup = np.array([60.0, 1.0])
    finals[("es", 0)] = nets.from_flat(nets.init_population(np.random.default_rng(9), 1)[0])   # the default setup
    table = evaluate.results_table(per_track, tmp_path, finals).set_index("Method")
    assert table.loc["GA", "Sensor length (m)"] == "40 · 60 · – · – (50 ± 14)"
    assert table.loc["GA", "Steering sensitivity"] == "2.00 · 1.00 · – · – (1.50 ± 0.71)"
    assert table.loc["ES", "Sensor length (m)"] == "150 · – · – · – (150)"
    assert table.loc["ES", "Steering sensitivity"] == "1.00 · – · – · – (1.00)"
    columns = list(table.columns)
    assert columns.index("Steering sensitivity") == columns.index("Sensor length (m)") + 1
    assert columns.index("Steering sensitivity") + 1 == columns.index("Target reduction (per run)")


def test_results_table_without_drivers_shows_no_setup(tmp_path):
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0)])
    table = evaluate.results_table(per_track, tmp_path).set_index("Method")
    assert table.loc["GA", "Sensor length (m)"] == "–" and table.loc["GA", "Steering sensitivity"] == "–"


def test_head_to_head_compares_each_methods_best_run_on_the_maps_they_all_finished():
    """Spec §11.3 (version 3.16): ES run 1, which finished nothing, no longer leaves the table empty."""
    nan = np.nan
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0), ("ga", 1, 1000, True, True, False, 54.0),
                            ("es", 0, 1000, True, True, False, 51.0), ("es", 1, 1000, False, False, True, nan),
                            ("ga", 0, 1001, True, True, False, 60.0), ("ga", 1, 1001, True, True, False, 58.0),
                            ("es", 0, 1001, True, True, False, 62.0), ("es", 1, 1001, False, False, True, nan),
                            ("ga", 0, 1002, True, True, False, 55.0), ("ga", 1, 1002, True, True, False, 55.0),
                            ("es", 0, 1002, False, False, True, nan), ("es", 1, 1002, False, False, True, nan)])
    table = evaluate.head_to_head_table(per_track, {"ga": 1, "es": 0}).set_index("Method")
    column = "Mean lap time on the 2 shared maps (s)"                   # map 1002: ES's best run did not finish
    assert list(table.index) == ["ES", "GA"]                          # config.APPROACHES order
    assert (table.loc["GA", "Best run"], table.loc["GA", column]) == (1, "56.00")
    assert (table.loc["ES", "Best run"], table.loc["ES", column]) == (0, "56.50")
    assert table.loc["GA", "Fastest on (maps)"] == 1 and table.loc["ES", "Fastest on (maps)"] == 1


def test_head_to_head_without_shared_maps():
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0), ("es", 0, 1000, False, False, True, np.nan)])
    table = evaluate.head_to_head_table(per_track, {"ga": 0, "es": 0})
    column = "Mean lap time on the 0 shared maps (s)"
    assert list(table[column]) == ["–", "–"] and list(table["Fastest on (maps)"]) == [0, 0]


def test_head_to_head_leaves_out_a_method_without_a_best_run():
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0), ("es", 0, 1000, True, True, False, 51.0)])
    table = evaluate.head_to_head_table(per_track, {"ga": 0})
    assert list(table["Method"]) == ["GA"] and table.loc[0, "Mean lap time on the 1 shared maps (s)"] == "50.00"


def test_a_hardened_run_ends_hardened_in_the_training_cost_table():
    """Spec §11.3, the training-cost table (version 3.15)."""
    assert evaluate.end_cell({"outcome": "hardened", "generations": 2140}) == "hardened · gen 2,140"
    ended = {"outcome": "time_limit", "generations": 2140, "hardening": {"factor": 1.25, "from_generation": 1795}}
    assert evaluate.end_cell(ended) == "ended by time · gen 2,140"


def test_training_cost_table(tmp_path):
    level = lambda n, generation, elapsed: {"level": n, "generation": generation, "evals": generation * 1000,
                                            "elapsed": elapsed}
    write_json(paths.run_dir("ga", 0, tmp_path) / "summary.json", {
        "start_level": 1, "goal_reached": True, "outcome": "speed_limit", "generations": 1200, "evals": 1_200_000,
        "wall_time": 36000.0, "level_reached": 4,
        "levels": [level(1, 25, 60.0), level(2, 80, 600.0), level(3, 120, 1800.0), level(4, 300, 7200.0)]})
    write_json(paths.run_dir("ga", 1, tmp_path) / "summary.json", {
        "start_level": 1, "goal_reached": False, "outcome": "stalled", "generations": 2100, "evals": 2_100_000,
        "wall_time": 5000.0, "level_reached": 2, "levels": [level(1, 30, 90.0)]})
    write_json(paths.run_dir("es", 3, tmp_path) / "summary.json", {
        "start_level": 1, "goal_reached": False, "outcome": "time_limit", "generations": 700, "evals": 770_000,
        "wall_time": 25000.0, "level_reached": 3, "levels": [level(1, 25, 60.0), level(2, 90, 900.0)]})
    write_json(paths.run_dir("baseline", 0, tmp_path) / "summary.json", {"wall_time": 15.0})
    write_json(paths.run_dir("baseline", 0, tmp_path) / "dagger.json",
               {"rounds": 7, "goal_reached": True, "wall_time": 3600.0})
    write_json(paths.speed_dir("baseline", 0, tmp_path) / "summary.json", {
        "start_level": 4, "goal_reached": True, "outcome": "speed_limit", "generations": 600, "evals": 330_000,
        "wall_time": 7200.0, "level_reached": 4, "levels": [level(4, 40, 900.0)]})
    write_json(paths.search_dir("kt_random", tmp_path) / "summary.json", {"wall_time": 914.2})
    write_json(paths.run_dir("kt_random", 0, tmp_path) / "summary.json", {"wall_time": 12.0})
    table = evaluate.training_cost_table(tmp_path)
    assert list(table["Method"]) == ["Baseline", "KT Random", "ES", "GA", "GA"]
    assert list(table["Run"]) == [0, 0, 3, 0, 1]
    base, kt, es3, ga0, ga1 = (table.iloc[k] for k in range(5))
    assert es3["Run end"] == "ended by time · gen 700" and es3["Hard cleared"] == "not reached"
    assert ga0["Easy cleared"] == "gen 25 · 25,000 evals · 0.02 h"
    assert ga0["Mastery goal"] == "gen 300 · 300,000 evals · 2.00 h"
    assert ga0["Run end"] == "speed limit · gen 1,200" and ga0["Evaluations"] == "1,200,000"
    assert ga0["Evolution (h)"] == "10.00" and ga0["Total (h)"] == "10.00" and ga0["Search (h)"] == "–"
    assert ga1["Medium cleared"] == "not reached" and ga1["Mastery goal"] == "not reached"
    assert ga1["Run end"] == "stalled on level 2 · gen 2,100"
    assert base["Easy cleared"] == "–" and base["Mastery goal"] == "gen 40 · 40,000 evals · 0.25 h"
    assert base["DAgger rounds"] == "7" and base["DAgger goal"] == "reached" and base["DAgger (h)"] == "1.00"
    assert base["Imitation training (h)"] == "0.00" and base["Total (h)"] == "3.00"
    assert kt["Search (h)"] == "0.25" and kt["Run end"] == "–" and kt["Mastery goal"] == "–"
    assert kt["Evaluations"] == "–" and kt["DAgger rounds"] == "–" and kt["Total (h)"] == "0.00"


def test_training_cost_table_without_records_is_empty(tmp_path):
    assert evaluate.training_cost_table(tmp_path).empty


def test_stage_results_and_table(fake_tracks, tmp_path):
    flats = nets.init_population(np.random.default_rng(12), 2)
    for stage, flat in zip(("imitation", "dagger"), flats):
        path = paths.stage_path("baseline", 0, stage, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nets.to_keras(flat).save(path)
    final = drive_rows([("baseline", 0, 1000, True, True, False, 50.0), ("ga", 0, 1000, True, True, False, 40.0)])
    results = evaluate.stage_results(tmp_path, final, test_seeds=[1000])
    assert sorted(results["stage"].unique()) == ["dagger", "imitation", "speed push"]
    assert set(results["approach"]) == {"baseline"}
    table = evaluate.stages_table(results)
    assert list(table["Stage"]) == ["After imitation", "After DAgger", "After speed push"]
    assert set(table["Method"]) == {"Baseline"} and table.loc[2, "Finished"] == "1 / 1"


def test_stage_results_without_stage_files(tmp_path):
    results = evaluate.stage_results(tmp_path, drive_rows([("ga", 0, 1000, True, True, False, 40.0)]), [1000])
    assert results.empty and "stage" in results.columns and evaluate.stages_table(results).empty


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


def write_log(folder, events, shares, laps):
    folder.mkdir(parents=True, exist_ok=True)
    n = len(shares)
    pd.DataFrame({"generation": range(n), "level": 1, "evals": np.arange(1, n + 1) * 10, "min_share": shares,
                  "median_lap": laps, "step_size": np.linspace(0.1, 0.05, n),
                  "event": events}).to_csv(folder / "log.csv", index=False)


def test_stalls_broken_counts_tightenings_right_after_a_boosted_generation(tmp_path):
    def log(boost, events):
        return pd.DataFrame({"generation": range(len(events)), "level": 4, "evals": 10, "event": events,
                             **({"boost": boost} if boost is not None else {})})
    events = [None, None, None, None, None, None, "tightened", None, None, None]
    first = log([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], events)       # boosted at 3-5, tightened at 6: one; 8-9: none
    assert evaluate.stalls_broken(first) == 1
    assert np.isnan(evaluate.stalls_broken(log(None, events)))
    for seed, frame in ((0, first), (1, log(None, events))):
        folder = paths.run_dir("ga", seed, tmp_path)
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / "log.csv", index=False)
        write_json(folder / "summary.json", {"wall_time": 60.0})
    table = evaluate.training_cost_table(tmp_path)
    assert list(table["Stalls broken"]) == ["1", "–"]


def test_figures_per_method(tmp_path):
    runs = tmp_path / "runs"
    for seed in (0, 1):
        write_log(paths.run_dir("ga", seed, runs), [None, "level_up", None, "goal"], [0.1, 0.92, 0.5, 0.95],
                  [np.nan, np.nan, 40.0, 38.5])
    write_log(paths.speed_dir("baseline", 0, runs), [None, "goal", "tightened"], [0.91, 0.93, 0.95],
              [45.0, 44.0, 42.0])
    logs = evaluate.evolution_logs(runs)
    assert sorted(logs) == ["baseline", "ga"] and sorted(logs["ga"]) == [0, 1] and sorted(logs["baseline"]) == [0]
    written = evaluate.figures(runs, tmp_path / "figures")
    assert [p.name for p in written] == ["success_over_training.png", "lap_time_over_training.png",
                                         "step_sizes.png"]


def test_setup_figure_follows_the_population_setups(tmp_path):
    runs = tmp_path / "runs"
    folder = paths.run_dir("neat", 0, runs)
    write_log(folder, [None, "level_up", None], [0.2, 0.95, 0.5], [np.nan, 40.0, 39.0])
    log = pd.read_csv(folder / "log.csv")
    log["sensor_mean"], log["sensor_std"] = [60.0, 70.0, 85.0], [20.0, 15.0, 10.0]
    log["steer_mean"], log["steer_std"] = [1.0, 1.3, 1.6], [0.5, 0.4, 0.3]
    log.to_csv(folder / "log.csv", index=False)
    written = evaluate.figures(runs, tmp_path / "figures")
    assert [p.name for p in written] == ["success_over_training.png", "lap_time_over_training.png",
                                         "setup_over_training.png", "step_sizes.png"]
    assert (tmp_path / "figures" / "setup_over_training.png").read_bytes()[:4] == b"\x89PNG"


def test_an_archives_other_runs_join_the_training_cost_the_stages_and_the_figures(fake_tracks, tmp_path):
    """An evaluated archive of an earlier revision may hold run 2: the training cost, the imitation stages and the
    training figures follow the runs present, as the results and comparison tables do (spec §11.3)."""
    for seed in (2, 0):
        write_json(paths.run_dir("ga", seed, tmp_path) / "summary.json", {
            "start_level": 1, "goal_reached": False, "outcome": "stalled", "generations": 10 + seed, "evals": 100,
            "wall_time": 60.0, "level_reached": 1, "levels": []})
        write_log(paths.run_dir("ga", seed, tmp_path), [None, "level_up"], [0.5, 0.95], [np.nan, 40.0])
    path = paths.stage_path("baseline", 2, "imitation", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nets.to_keras(nets.init_population(np.random.default_rng(3), 1)[0]).save(path)
    table = evaluate.training_cost_table(tmp_path)
    assert list(table["Run"]) == [0, 2] and list(table["Run end"]) == ["stalled on level 1 · gen 10",
                                                                      "stalled on level 1 · gen 12"]
    assert sorted(evaluate.evolution_logs(tmp_path)["ga"]) == [0, 2]
    stages = evaluate.stage_results(tmp_path, drive_rows([]), test_seeds=[1000])
    assert set(zip(stages["approach"], stages["run_seed"], stages["stage"])) == {("baseline", 2, "imitation")}


def test_light_series_colors_get_an_outline():
    assert evaluate._outline(config.COLORS["neat"]) is not None
    assert evaluate._outline(config.COLORS["ga"]) is None and evaluate._outline(config.COLORS["es"]) is None


def export_search(runs, approach, flats):
    """A search directory with one exported model per trial, as run_tuner writes it."""
    search = paths.search_dir(approach, runs)
    (search / "models").mkdir(parents=True)
    ids = [f"{i:02d}" for i in range(len(flats))]
    for trial_id, flat in zip(ids, flats):
        nets.to_keras(flat).save(search / "models" / f"{trial_id}.keras")
    pd.DataFrame({"trial_id": ids, "config_id": ids, "val_accuracy": np.linspace(0.5, 0.6, len(ids)),
                  "params": 431, "epochs": 3}).to_csv(search / "trials.csv", index=False)
    return ids


def test_driving_selected_picks_the_best_validation_driver(fake_tracks, tmp_path):
    flats = nets.init_population(np.random.default_rng(8), 4)
    ids = export_search(tmp_path, "kt_random", flats)
    table = evaluate.driving_selected_table(tmp_path, val_seeds=[80, 81], test_seeds=[1000])
    keras_drivers = {("kt_random", i): nets.from_keras(nets.to_keras(f)) for i, f in zip(ids, flats)}
    validation = evaluate.driving_results(keras_drivers, [80, 81], level=config.VALIDATION_LEVEL)
    expected = validation.groupby("run_seed")["fitness"].mean().idxmax()
    assert list(table["Method"]) == ["KT Random"] and table.loc[0, "Trial"] == expected
    assert table.loc[0, "Parameters"] == 431


def test_driving_selected_validates_on_the_validation_setting(requested_maps, tmp_path):
    """Spec §4.6, §8.3: the validation maps are seeds 80-99 at the validation setting, not at Extra hard."""
    export_search(tmp_path, "kt_random", nets.init_population(np.random.default_rng(8), 2))
    evaluate.driving_selected_table(tmp_path, val_seeds=[80, 81], test_seeds=[1000])
    assert set(requested_maps) == {(80, config.VALIDATION_LEVEL), (81, config.VALIDATION_LEVEL),
                                   (1000, config.TEST_LEVEL)}


def test_driving_selected_without_exported_models(tmp_path):
    assert evaluate.driving_selected_table(tmp_path, val_seeds=[80], test_seeds=[1000]).empty


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


def test_main_refuses_a_controller_of_another_action_set(tmp_path):
    """Spec §10.3: the evaluation of a folder holding a controller of 15 actions stops with the same message."""
    from racing.backend import keras_cpu
    keras = keras_cpu()
    model = keras.Sequential([keras.Input((config.N_INPUTS,)), keras.layers.Dense(15, activation="softmax")])
    path = paths.champion_path("baseline", 0, champion_dir=tmp_path)
    path.parent.mkdir(parents=True)
    model.save(path)
    with pytest.raises(ValueError, match="baseline run 0 was made for 15 actions, this version has 45"):
        evaluate.main(champions_dir=tmp_path, runs_dir=tmp_path, data_path=tmp_path / "none.npz", out_dir=tmp_path)


def test_main_without_champions_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate.main(champions_dir=tmp_path, runs_dir=tmp_path, data_path=tmp_path / "none.npz", out_dir=tmp_path)


def test_main_with_partial_results(requested_maps, tmp_path):
    champions, runs = tmp_path / "champions", tmp_path / "runs"
    rng = np.random.default_rng(5)
    for approach, seed in (("ga", 0), ("ga", 2), ("es", 1)):
        population.save_dense_champion(approach, seed, nets.DensePopulation(nets.init_population(rng, 1)), 0,
                                       champions)
    write_json(paths.run_dir("ga", 0, runs) / "summary.json", {
        "wall_time": 12.0, "level_reached": 2, "goal_reached": False, "outcome": "stalled", "generations": 10,
        "evals": 1000, "start_level": 1, "levels": [{"level": 1, "generation": 4, "evals": 400, "elapsed": 5.0}]})
    np.savez(tmp_path / "dataset.npz", X_test=rng.random((40, config.N_INPUTS)).astype(np.float32),
             y_test=rng.integers(0, config.N_ACTIONS, 40))
    results = evaluate.main(champions_dir=champions, runs_dir=runs, data_path=tmp_path / "dataset.npz",
                            out_dir=tmp_path / "out", test_seeds=[1000], val_seeds=[80], latency_calls=5)
    assert set(results) == {"results", "pole_laps", "head_to_head", "training_cost", "stages", "comparison", "search_cost",
                            "driving_selected", "dataset", "per_track", "validation", "metrics", "best_seeds",
                            "figures", "robustness", "robustness_per_track", "evolved_networks"}
    assert set(results["robustness_per_track"]["width"]) <= set(config.ROBUSTNESS_WIDTHS)
    assert list(results["robustness"]["Method"]) == ["ES", "GA"]
    assert list(results["results"]["Method"]) == ["ES", "GA"] and list(results["comparison"]["Method"]) == ["ES", "GA"]
    cost = results["training_cost"].set_index("Run")
    assert cost.loc[0, "Mastery goal"] == "not reached" and cost.loc[0, "Run end"] == "stalled on level 2 · gen 10"
    assert json.loads((champions / "best_seeds.json").read_text()).keys() == {"ga", "es"}
    for name in ("results", "head_to_head", "training_cost", "stages", "comparison", "search_cost",
                 "driving_selected", "dataset", "robustness"):
        assert (tmp_path / "out" / "tables" / f"{name}.md").exists()
    assert (tmp_path / "out" / "tables" / "robustness_per_track.csv").exists()
    assert results["figures"][-1] == tmp_path / "out" / "figures" / "networks.png" and results["figures"][-1].exists()
    assert results["results"].set_index("Method").loc["GA", "Sensor length (m)"] == "150 · – · 150 · – (150 ± 0)"
    assert list(results["results"]["Best run"]) == [results["best_seeds"]["es"], results["best_seeds"]["ga"]]
    assert (80, config.VALIDATION_LEVEL) in requested_maps and (80, config.MAX_LEVEL) not in requested_maps


# ----- spec §11.1: robustness table -----

def test_a_widened_map_keeps_the_centerline_and_is_timed_by_the_expert(monkeypatch, fake_tracks):
    monkeypatch.setattr(evaluate.expert, "expert_lap_time", lambda track: 21.5)
    wide = evaluate.widened_track(80, 18.0, level=2)                # the 50 m circle is within Medium's tightest corner
    assert np.all(wide.widths == 18.0) and wide.expert_lap_time == 21.5
    assert np.allclose(wide.centerline, fake_tracks.centerline, atol=1e-6)


def test_a_widened_test_circuit_has_no_corner_limit(monkeypatch, requested_maps):
    """Spec §11.1: the test circuits are rebuilt at each width by default; a real circuit's corners are its own, so
    only the separation of the road's parts is checked."""
    limits = []
    monkeypatch.setattr(evaluate, "geometry_ok", lambda track, min_corner: limits.append(min_corner) or True)
    monkeypatch.setattr(evaluate.expert, "expert_lap_time", lambda track: 21.5)
    assert evaluate.widened_track(2, 14.0).expert_lap_time == 21.5
    assert evaluate.widened_track(80, 14.0, level=2).expert_lap_time == 21.5
    assert requested_maps == [(2, config.TEST_LEVEL), (80, 2)] and limits == [None, config.LEVELS[2]["min_corner"]]


def test_a_width_the_map_cannot_take_or_the_expert_cannot_drive_is_left_out(monkeypatch, fake_tracks):
    monkeypatch.setattr(evaluate, "geometry_ok", lambda track, min_corner: False)
    assert evaluate.widened_track(80, 18.0, level=1) is None
    monkeypatch.setattr(evaluate, "geometry_ok", lambda track, min_corner: True)
    monkeypatch.setattr(evaluate.expert, "expert_lap_time", lambda track: None)
    assert evaluate.widened_track(80, 8.0, level=1) is None


def test_robustness_results_drive_every_map_at_every_width_and_skip_left_out_maps(monkeypatch, fake_tracks):
    monkeypatch.setattr(evaluate, "widened_track",
                        lambda seed, width, level=config.MAX_LEVEL: None if (seed, width) == (1001, 18.0)
                        else fake_tracks)
    frame = evaluate.robustness_results(drivers(2), [1000, 1001], widths=(10.0, 18.0))
    assert len(frame) == 2 * 3 and set(frame["width"]) == {10.0, 18.0}
    assert frame[frame["width"] == 18.0]["track"].unique().tolist() == [1000]


def test_the_robustness_table_counts_finishes_per_method_and_width():
    narrow = drive_rows([("ga", 0, 1000, True, True, False, 20.0), ("ga", 0, 1001, False, False, True, np.nan),
                         ("es", 0, 1000, True, False, False, 25.0)])
    narrow["width"] = 10.0
    wide = drive_rows([("ga", 0, 1000, True, True, False, 18.0), ("es", 0, 1000, False, False, True, np.nan)])
    wide["width"] = 18.0
    table = evaluate.robustness_table(pd.concat([narrow, wide]), (10.0, 18.0))
    ga = table[table["Method"] == "GA"].iloc[0]
    assert (ga["10 m finished (2 maps)"], ga["10 m succeeded"], ga["10 m mean lap (s)"]) == ("1 / 2", "1 / 2", "20.00")
    assert ga["18 m finished (1 maps)"] == "1 / 1"
    es = table[table["Method"] == "ES"].iloc[0]
    assert es["18 m finished (1 maps)"] == "0 / 1" and es["18 m mean lap (s)"] == "–"


def test_the_robustness_table_gives_finishes_per_run():
    frame = drive_rows([("ga", 0, 1000, True, True, False, 20.0), ("ga", 1, 1000, False, False, True, np.nan),
                        ("ga", 2, 1000, True, False, False, 22.0)])
    frame["width"] = 14.0
    ga = evaluate.robustness_table(frame, (14.0,)).iloc[0]
    assert ga["14 m finished per run"] == "1 · 0 · 1 · –" and ga["14 m finished (1 maps)"] == "2 / 3"


def test_runs_are_told_apart_by_shade_not_dashes():
    """Version 3.18: runs 0-3 share the line style and differ by shades of the method's color."""
    fig, ax = plt.subplots()
    log = pd.DataFrame({"generation": [0, 1, 2], "fitness": [1.0, 2.0, 3.0]})
    evaluate._plot_runs(ax, {run: log for run in range(4)}, "fitness", "ga")
    lines = ax.get_lines()
    plt.close(fig)
    assert {line.get_linestyle() for line in lines} == {"-"}
    assert len({line.get_color() for line in lines if line.get_linewidth() > 1}) == 4


def test_moving_average_skips_nans():
    y = np.array([1.0, np.nan, 3.0, 5.0])
    out = evaluate.moving_average(y, window=3)
    assert np.isnan(out[1]) and out[2] == pytest.approx(4.0)


def test_reduction_cells_append_mean_and_sd_over_the_runs_that_reached_the_goal():
    reached = lambda r: {"goal_reached": True, "target_reduction": r}
    two = [reached(0.276), {"goal_reached": False}, reached(0.301), reached(0.25)]
    assert evaluate.reduction_cells(two) == "27.6% · not reached · 30.1% · 25.0% (27.6 ± 2.6%)"
    one = [reached(0.276), {"goal_reached": False}, {}, {"goal_reached": False}]
    assert evaluate.reduction_cells(one) == "27.6% · not reached · – · not reached (27.6%)"
    assert evaluate.reduction_cells([{}, {"goal_reached": False}]) == "– · not reached"


def test_network_text_names_each_layer_or_the_genomes_size():
    dense = nets.from_flat(np.zeros(nets.n_params((9,))), (9,))
    assert evaluate.network_text(dense) == "18 → 9 → 45"
    assert evaluate.network_text(nets.from_flat(np.zeros(nets.n_params(())), ())) == "18 → 45"
    innovations = neat.Innovations()
    genome = neat.add_node(np.random.default_rng(0), neat.initial_genome(np.random.default_rng(0), innovations),
                           innovations)
    assert evaluate.network_text(neat.NeatNet(genome)) == "1 hidden nodes, 811 connections"


def test_the_evolved_networks_table_lists_ga_and_neat_runs_only():
    drivers = {("ga", 1): nets.from_flat(np.zeros(nets.n_params((4,))), (4,)),
               ("ga", 0): nets.from_flat(np.zeros(nets.n_params(())), ()),
               ("es", 0): nets.from_flat(np.zeros(nets.n_params())),
               ("neat", 0): neat.NeatNet(neat.initial_genome(np.random.default_rng(0), neat.Innovations()))}
    table = evaluate.evolved_networks_table(drivers)
    assert list(table.columns) == ["Method", "Run", "Structure", "Parameters"]
    assert table.values.tolist() == [["GA", 0, "18 → 45", 855], ["GA", 1, "18 → 4 → 45", nets.n_params((4,))],
                                     ["NEAT", 0, "0 hidden nodes, 810 connections", 855]]


def test_network_size_is_drawn_for_the_ga_and_neat(tmp_path):
    runs = tmp_path / "runs"
    for approach in ("ga", "neat", "es"):
        folder = paths.run_dir(approach, 0, runs)
        folder.mkdir(parents=True)
        rows = {"generation": [0, 1], "min_share": [0.1, 0.2], "median_lap": [30.0, 29.0], "event": ["", ""]}
        if approach != "es":
            rows["mean_nodes"] = [0.0, 1.5]
        pd.DataFrame(rows).to_csv(folder / "log.csv", index=False)
    written = evaluate.figures(runs, tmp_path / "figures")
    assert tmp_path / "figures" / "network_size_over_training.png" in written
    assert (tmp_path / "figures" / "network_size_over_training.png").is_file()


def pole_rows(rows):
    """per_track rows of test races: (approach, run, circuit index, lap 1, lap 2, lap 3); NaN for a lap not driven."""
    out = []
    for approach, seed, k, l1, l2, l3 in rows:
        laps = np.array([l1, l2, l3], dtype=float)
        finished = bool(np.isfinite(laps).all())
        out.append({"approach": approach, "run_seed": seed, "track": k, "circuit": config.TEST_CIRCUITS[k],
                    "finished": finished, "success": finished, "crashed": not finished,
                    "race_time": float(laps.sum()) if finished else float("nan"),
                    "lap_time": float(laps.sum()) / 3 if finished else float("nan"), "progress": 1.0,
                    "fitness": 1.5, "lap_1": l1, "lap_2": l2, "lap_3": l3,
                    "flying_lap": float(np.nanmin(laps[1:])) if np.isfinite(laps[1:]).any() else float("nan")})
    return pd.DataFrame(out, columns=evaluate.DRIVE_COLUMNS)


def test_the_pole_table_compares_each_runs_flying_lap_with_the_pole_lap():
    per_track = pole_rows([("ga", 0, 0, 95.0, 80.0, 81.0), ("ga", 1, 0, 96.0, 82.0, 79.9),
                            ("ga", 0, 3, 99.0, 86.0, np.nan), ("es", 0, 0, 97.0, np.nan, np.nan)])
    table = evaluate.pole_laps_table(per_track)
    ga0 = table[(table["Method"] == "GA") & (table["Run"] == "0")].iloc[0]
    assert ga0["Melbourne flying lap (s)"] == "80.00" and ga0["Melbourne vs pole (s)"] == "-0.49"
    assert ga0["Melbourne vs pole (%)"] == "-0.60" and ga0["Silverstone flying lap (s)"] == "86.00"
    assert ga0["Poles beaten"] == 1
    best = table[(table["Method"] == "GA") & (table["Run"] == "fastest run")].iloc[0]
    assert best["Melbourne flying lap (s)"] == "79.90" and best["Poles beaten"] == 1
    es0 = table[(table["Method"] == "ES") & (table["Run"] == "0")].iloc[0]
    assert es0["Melbourne flying lap (s)"] == "–" and es0["Poles beaten"] == 0
    assert set(table.loc[table["Method"] == "GA", "Run"]) == {"0", "1", "fastest run"}   # runs 2 and 3 were never driven


def test_a_race_row_records_every_lap_and_the_flying_lap(circle_track, monkeypatch):
    result = sim.RunResult(fitness=np.array([1.5]), finished=np.array([True]), crashed=np.array([False]),
                           progress=np.array([1.0]), finish_time=np.array([30.0]), steps=600,
                           laps_done=np.array([3]), lap_times=np.array([[12.0, 9.0, 9.5]]))
    monkeypatch.setattr(sim, "simulate", lambda *a, **k: result)
    row = evaluate._drive_on({("ga", 0): nets.from_flat(np.zeros(nets.n_params(())), ())}, circle_track, 0, 1.3)[0]
    assert (row["lap_1"], row["lap_2"], row["lap_3"], row["flying_lap"]) == (12.0, 9.0, 9.5, 9.0)


def test_the_pole_figure_is_drawn(tmp_path):
    per_track = pole_rows([("ga", 0, k, 95.0, 85.0, 86.0) for k in config.TEST_TRACKS])
    path = evaluate.pole_figure(per_track, tmp_path / "pole_laps.png")
    assert path.exists() and path.stat().st_size > 0


def test_robustness_rebuilds_the_test_circuits_at_four_widths():
    assert config.ROBUSTNESS_WIDTHS == (6.0, 10.0, 14.0, 18.0)


def test_the_pole_figure_fits_slow_bars_and_their_gap_labels(tmp_path, monkeypatch):
    """Early controllers fly 1.35-1.9 x pole: the x-limit follows the data, so every bar end and label is inside."""
    import matplotlib.pyplot as plt
    from racing import circuits
    drawn = []
    monkeypatch.setattr(plt, "close", lambda fig=None: drawn.append(fig))
    rows = []
    for k, name in enumerate(config.TEST_CIRCUITS):
        pole = circuits.POLE_LAPS[name].seconds
        for approach, factor in (("ga", 1.35), ("es", 1.6), ("neat", 1.9)):
            rows.append((approach, 0, k, 1.5 * pole, factor * pole, factor * pole + 1.0))
    evaluate.pole_figure(pole_rows(rows), tmp_path / "slow.png")
    fig = drawn[0]
    fig.canvas.draw()
    for ax in fig.axes:
        pole_x = [line.get_xdata()[0] for line in ax.lines][0]
        assert ax.get_xlim()[0] <= 0 and pole_x < ax.get_xlim()[1]
        assert len(ax.patches) == 3 and len(ax.texts) == 3
        for bar in ax.patches:
            assert bar.get_x() + bar.get_width() < ax.get_xlim()[1]
        for text in ax.texts:
            assert text.get_window_extent().x1 <= ax.bbox.x1


def test_level_ticks_are_taller_than_target_cut_ticks():
    """Version 3.18: level advancements and the mastery goal are tall ticks on the x-axis, target cuts and hardening
    (version 3.15) short ones; the ticks are drawn in axis height, so they leave the data's y-range alone."""
    fig, ax = plt.subplots()
    ax.plot([0, 20], [40.0, 60.0])
    evaluate.event_ticks(ax, [0, 5, 9, 12, 20], ["level_up", "tightened", "goal", "hardened", np.nan])
    tops = {float(segment[0, 0]): float(segment[:, 1].max())
            for collection in ax.collections for segment in collection.get_segments()}
    bottom = ax.get_ylim()[0]
    plt.close(fig)
    assert tops == {0.0: 0.045, 5.0: 0.025, 9.0: 0.045, 12.0: 0.025}
    assert bottom > 30.0


def test_redraw_figures_writes_the_same_file_names(tmp_path):
    """The figures are redrawn from a results folder's logs and tables without new runs; the networks figure needs
    the trained controllers and is not drawn."""
    runs = tmp_path / "runs"
    for seed in (0, 1):
        folder = paths.run_dir("ga", seed, runs)
        write_log(folder, [None, "level_up", "tightened", "goal"], [0.1, 0.92, 0.5, 0.95], [np.nan, 41.0, 40.0, 38.5])
        log = pd.read_csv(folder / "log.csv")
        log["mean_nodes"], log["sensor_mean"], log["steer_mean"] = [0.0, 1.0, 1.5, 2.0], 150.0, 1.2
        log.to_csv(folder / "log.csv", index=False)
    search = paths.search_dir("kt_random", runs)
    search.mkdir(parents=True)
    pd.DataFrame({"trial_id": ["00", "01"], "config_id": ["00", "01"], "val_accuracy": [0.5, 0.6],
                  "params": [100, 200], "epochs": 3}).to_csv(search / "trials.csv", index=False)
    write_json(search / "summary.json", {"chosen_trial": "01"})
    (tmp_path / "tables").mkdir()
    pole_rows([("ga", 0, k, 95.0, 85.0, 86.0) for k in config.TEST_TRACKS]).to_csv(
        tmp_path / "tables" / "driving_per_track.csv", index=False)
    written = evaluate.redraw_figures(tmp_path)
    assert sorted(p.name for p in written) == sorted([
        "success_over_training.png", "lap_time_over_training.png", "setup_over_training.png",
        "network_size_over_training.png", "step_sizes.png", "kt_trials.png", "pole_laps.png"])
    assert all(p.parent == tmp_path / "figures" and p.read_bytes()[:4] == b"\x89PNG" for p in written)
    assert not (tmp_path / "figures" / "networks.png").exists()
