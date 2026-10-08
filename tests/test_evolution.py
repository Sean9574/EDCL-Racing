import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from racing import checkpoint, config, curriculum, evolution, ga, nets, paths, population, progress, setup
from racing.population import TrialResult


def fake_evaluate(success_rate):
    """Evaluation without simulation: fitness falls with the size of a car's weights, and the fittest
    `success_rate` share of the population succeeds on every map."""
    def evaluate(population, seeds, level, factor, n_jobs=1, backend="loky"):
        weights = population.weights
        n = len(weights)
        fitness = 1.0 / (1.0 + np.sum(weights ** 2, axis=1))
        rank = np.argsort(np.argsort(-fitness, kind="stable"), kind="stable")
        success = np.tile(rank < success_rate * n, (len(seeds), 1))
        return TrialResult(fitness=fitness + success[0], success=success,
                           laps=np.where(success, config.LAPS, 1), finish_time=np.where(success, 90.0, np.nan))
    return evaluate


@pytest.fixture
def evaluation(monkeypatch):
    def use(success_rate):
        monkeypatch.setattr(population, "evaluate_population", fake_evaluate(success_rate))
    return use


def train(tmp_path, max_generations, name="run", **kwargs):
    reports = []
    kwargs.setdefault("champion_dir", None)
    summary = evolution.run(ga.GA(pop_size=20), 0, tmp_path / name, max_generations=max_generations,
                            on_generation=reports.append, **kwargs)
    return summary, reports


class Interrupt(Exception):
    pass


def interrupt_at(generation):
    def callback(report):
        if report["generation"] == generation:
            raise Interrupt
    return callback


def comparable(log_path):
    return pd.read_csv(log_path).drop(columns=["elapsed"])


def test_small_run_logs_every_generation(evaluation, tmp_path):
    evaluation(0.0)
    summary, reports = train(tmp_path, 3)
    assert summary["generations"] == len(reports) == 3 and summary["evals"] == 60 and summary["pop_size"] == 20
    assert [r["generation"] for r in reports] == [0, 1, 2] and math.isnan(reports[0]["step_size"])
    log = pd.read_csv(tmp_path / "run" / "log.csv")
    assert list(log.columns) == evolution.LOG_FIELDS and list(log["generation"]) == [0, 1, 2]
    maps = [s for r in reports for s in r["seeds"]]
    assert len(maps) == len(set(maps)) == 9 and min(maps) >= config.TRAIN_SEED_START   # fresh maps, never reused
    assert (tmp_path / "run" / checkpoint.CHECKPOINT_NAME).exists()
    saved = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert saved["settings"] == checkpoint.settings_record() and saved["goal_reached"] is False
    assert saved["champion_path"] is None and saved["level_reached"] == 1


def test_survivors_come_first_in_the_next_generation(evaluation, tmp_path):
    evaluation(0.5)
    _, reports = train(tmp_path, 2)
    kept = reports[1]["is_kept"]
    assert kept.sum() == reports[0]["n_kept"] and np.all(kept[:kept.sum()]) and not kept[kept.sum():].any()
    assert math.isnan(reports[0]["kept_success"]) and reports[0]["child_success"] == 0.5     # generation 0: all children
    assert reports[1]["kept_success"] == reports[1]["all_success"][kept].mean()    # children may outrank kept cars
    assert reports[1]["child_success"] == reports[1]["all_success"][~kept].mean()


def test_levels_advance_after_three_passing_generations(evaluation, tmp_path):
    evaluation(1.0)
    summary, reports = train(tmp_path, 7)
    assert [r["level"] for r in reports] == [1, 1, 1, 2, 2, 2, 3]
    assert [r["event"] for r in reports[:4]] == [None, None, "level_up", None]
    assert [(e["level"], e["generation"]) for e in summary["levels"]] == [(1, 2), (2, 5)]


def test_failing_generations_never_advance(evaluation, tmp_path):
    evaluation(0.85)                                         # below the 90% requirement on every map
    _, reports = train(tmp_path, 5)
    assert [r["level"] for r in reports] == [1] * 5
    assert all(r["min_share"] < config.MASTERY_FRACTION for r in reports)


def test_goal_then_the_speed_push_tightens_the_target(evaluation, tmp_path):
    evaluation(1.0)
    summary, reports = train(tmp_path, 6, start_level=config.MAX_LEVEL)
    assert reports[2]["event"] == "goal" and reports[5]["event"] == "tightened"
    assert math.isclose(reports[3]["factor"], config.TARGET_FACTOR * config.SPEED_STEP)
    assert summary["goal_reached"] and math.isclose(summary["target_reduction"], 1 - config.SPEED_STEP)
    assert summary["start_level"] == config.MAX_LEVEL and len(summary["tightenings"]) == 1
    assert [(e["level"], e["generation"]) for e in summary["levels"]] == [(config.MAX_LEVEL, 2)]   # the goal entry


class StallsAtOnce(curriculum.MasteryTracker):
    def update(self, min_share):
        super().update(min_share)
        self.outcome = "stalled"
        return "end_stalled"


def test_a_stalled_run_still_saves_a_final_controller(evaluation, monkeypatch, tmp_path):
    evaluation(0.0)
    monkeypatch.setattr(evolution, "MasteryTracker", StallsAtOnce)
    chosen = []

    def select(population, candidates, factor, n_jobs=1):
        chosen.append(factor)
        return int(candidates[0]), 3, 0.7

    monkeypatch.setattr(population, "select_final", select)
    summary, reports = train(tmp_path, None, champion_dir=tmp_path / "champions")
    assert len(reports) == 1 and reports[0]["event"] == "end_stalled"
    assert summary["goal_reached"] is False and summary["outcome"] == "stalled" and summary["level_reached"] == 1
    assert chosen == [config.TARGET_FACTOR] and summary["val_successes"] == 3
    assert summary["champion_path"] == str(paths.champion_path("ga", 0, champion_dir=tmp_path / "champions"))
    assert paths.champion_path("ga", 0, ".npy", tmp_path / "champions").exists()
    assert "GA seed 0: stopped at generation 0" in config.RUN_LOG_PATH.read_text()


def test_the_final_controller_comes_from_the_kept_cars_at_the_tightest_mastered_target(evaluation, monkeypatch,
                                                                                         tmp_path):
    evaluation(1.0)
    chosen = []

    def select(population, candidates, factor, n_jobs=1):
        chosen.append((np.asarray(candidates), factor, population))
        return int(candidates[-1]), 20, 1.9

    monkeypatch.setattr(population, "select_final", select)
    method = ga.GA(pop_size=20)
    summary = evolution.run(method, 4, tmp_path / "run", champion_dir=tmp_path / "champions",
                            start_level=config.MAX_LEVEL, max_generations=6)
    candidates, factor, cars = chosen[0]                  # not `population`: that would shadow the module above
    assert np.array_equal(candidates, np.flatnonzero(method.is_kept)) and cars is method.population
    assert math.isclose(factor, config.TARGET_FACTOR * config.SPEED_STEP)
    saved = np.load(paths.champion_path("ga", 4, ".npy", tmp_path / "champions"))
    assert np.array_equal(saved, method.population.weights[candidates[-1]])
    assert summary["val_successes"] == 20 and summary["wall_time"] > 0


def test_a_resumed_run_matches_an_uninterrupted_one(evaluation, tmp_path):
    evaluation(0.5)
    evolution.run(ga.GA(pop_size=20), 0, tmp_path / "whole", champion_dir=None, max_generations=5)
    with pytest.raises(Interrupt):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "split", champion_dir=None, max_generations=5,
                      on_generation=interrupt_at(3))
    # generation 3 was logged, but the run stopped before its checkpoint (Review Focus 2)
    assert list(pd.read_csv(tmp_path / "split" / "log.csv")["generation"]) == [0, 1, 2, 3]
    summary = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "split", champion_dir=None, max_generations=5)
    whole, split = comparable(tmp_path / "whole" / "log.csv"), comparable(tmp_path / "split" / "log.csv")
    assert list(split["generation"]) == [0, 1, 2, 3, 4]
    pd.testing.assert_frame_equal(whole, split)
    ends = [checkpoint.load(tmp_path / name / checkpoint.CHECKPOINT_NAME) for name in ("whole", "split")]
    assert np.array_equal(ends[0]["method"]["weights"], ends[1]["method"]["weights"])
    assert np.array_equal(ends[0]["method"]["setup"], ends[1]["method"]["setup"])
    assert ends[0]["rng"] == ends[1]["rng"] and summary["generations"] == 5


def test_resuming_with_changed_settings_is_refused(evaluation, monkeypatch, tmp_path):
    evaluation(0.5)
    with pytest.raises(Interrupt):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=5,
                      on_generation=interrupt_at(1))
    monkeypatch.setattr(config, "GA_MUTATION_P", 0.3)
    with pytest.raises(checkpoint.SettingsMismatch, match="GA_MUTATION_P"):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=5)


def test_a_completed_run_is_returned_without_training(evaluation, tmp_path):
    evaluation(0.5)
    first = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=2)
    again = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=2,
                          on_generation=lambda report: pytest.fail("trained again"))
    assert again == json.loads(json.dumps(first))
    assert len(pd.read_csv(tmp_path / "run" / "log.csv")) == 2


def synced_writes(monkeypatch):
    """Names of the files written through checkpoint.write_json, which syncs them to disk."""
    written, real = [], checkpoint.write_json
    monkeypatch.setattr(checkpoint, "write_json",
                        lambda path, data: written.append(Path(path).name) or real(path, data))
    return written


def test_the_summary_is_synced_to_disk(evaluation, monkeypatch, tmp_path):
    evaluation(0.5)
    written = synced_writes(monkeypatch)
    train(tmp_path, 2)
    assert written == ["summary.json"]


def test_a_summary_cut_short_by_a_reboot_is_written_again_from_the_checkpoint(evaluation, tmp_path):
    evaluation(0.5)
    first, _ = train(tmp_path, 3)
    log = (tmp_path / "run" / "log.csv").read_text()
    (tmp_path / "run" / "summary.json").write_text("")
    again, reports = train(tmp_path, 3)
    assert reports == [] and (tmp_path / "run" / "log.csv").read_text() == log
    del first["wall_time"], again["wall_time"]
    assert again == json.loads(json.dumps(first))
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["generations"] == 3


def test_experiment_runs_report_status_and_milestones(evaluation, monkeypatch, tmp_path):
    evaluation(1.0)
    monkeypatch.setattr(population, "select_final",
                        lambda population, candidates, factor, n_jobs=1: (int(candidates[0]), 1, 1.0))
    evolution.run(ga.GA(pop_size=20), 2, tmp_path / "run", champion_dir=tmp_path / "champions", max_generations=4)
    status = progress.read_status()
    assert status["approach"] == "ga" and status["seed"] == 2 and status["stage"] == "evolution"
    assert status["level"] == 2 and status["generation"] == 3 and status["min_share"] == 1.0
    log = config.RUN_LOG_PATH.read_text()
    assert "GA seed 2: started" in log and "Easy mastered at generation 2, now Medium" in log
    assert "GA seed 2: finished" in log


def test_viewer_and_pilot_runs_report_nothing(evaluation, tmp_path):
    evaluation(1.0)
    train(tmp_path, 4)
    assert not config.STATUS_PATH.exists() and not config.RUN_LOG_PATH.exists()


def test_each_run_starts_with_fresh_workers_before_its_timer(evaluation, monkeypatch, tmp_path):
    evaluation(0.0)
    events = []

    class Clock:
        @staticmethod
        def time():
            events.append("clock")
            return 0.0

    monkeypatch.setattr(population, "fresh_workers", lambda n_jobs: events.append(("fresh", n_jobs)))
    monkeypatch.setattr(evolution, "time", Clock)
    evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, n_jobs=3, max_generations=1)
    assert events[:2] == [("fresh", 3), "clock"] and events.count(("fresh", 3)) == 1


def test_truncate_log_keeps_the_rows_before_a_generation(tmp_path):
    path = tmp_path / "log.csv"
    evolution.start_log(path)
    for g in range(4):
        evolution.append_log(path, {key: g if key == "generation" else ([1] if key in ("seeds", "map_shares")
                                                                      else 0) for key in evolution.LOG_FIELDS})
    evolution.truncate_log(path, 2)
    assert list(pd.read_csv(path)["generation"]) == [0, 1]


def damaged_log(path, tail):
    """A log of generations 0-2 whose end was damaged by a crash: `tail` replaces the last bytes."""
    evolution.start_log(path)
    for g in range(3):
        evolution.append_log(path, {key: 100 + g if key == "generation" else ([1] if key in ("seeds", "map_shares")
                                                                            else 0) for key in evolution.LOG_FIELDS})
    with open(path, "ab") as f:
        f.write(tail)


def test_truncate_log_drops_a_tail_of_zero_bytes(tmp_path):
    """After a reboot the end of a file can read as zero bytes; the run must still resume."""
    path = tmp_path / "log.csv"
    damaged_log(path, b"\0" * 300)
    evolution.truncate_log(path, 103)
    assert list(pd.read_csv(path)["generation"]) == [100, 101, 102]


def test_truncate_log_drops_a_torn_row(tmp_path):
    """A row cut short ("10" of generation 103) is not kept as a row of generation 10."""
    path = tmp_path / "log.csv"
    damaged_log(path, b"10")
    evolution.truncate_log(path, 103)
    assert list(pd.read_csv(path)["generation"]) == [100, 101, 102]
    damaged_log(path, b"1\0" + b",0" * (len(evolution.LOG_FIELDS) - 1) + b"\n")     # all fields, first damaged
    evolution.truncate_log(path, 200)
    assert list(pd.read_csv(path)["generation"]) == [100, 101, 102]


def synced_files(monkeypatch, calls):
    """Record the name of every file synced to disk (os.fsync) in `calls`."""
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(Path(os.readlink(f"/proc/self/fd/{fd}")).name)
                        or real(fd))


def test_log_rows_are_synced_before_the_checkpoint(evaluation, monkeypatch, tmp_path):
    """A generation's row is on disk before its checkpoint, so a reboot cannot lose rows the checkpoint covers."""
    evaluation(0.5)
    calls, real_save = [], checkpoint.save
    synced_files(monkeypatch, calls)
    monkeypatch.setattr(checkpoint, "save", lambda path, state: calls.append("save") or real_save(path, state))
    train(tmp_path, 2)
    saves = [i for i, call in enumerate(calls) if call == "save"]
    assert len(saves) == 2 and all(calls[i - 1] == "log.csv" for i in saves)


def test_truncate_log_syncs_the_new_log_before_replacing_it(monkeypatch, tmp_path):
    path = tmp_path / "log.csv"
    damaged_log(path, b"")
    calls, real_replace = [], os.replace
    synced_files(monkeypatch, calls)
    monkeypatch.setattr(os, "replace", lambda a, b: calls.append(("replace", Path(a).name, Path(b).name))
                        or real_replace(a, b))
    evolution.truncate_log(path, 102)
    assert len(calls) == 2 and calls[1] == ("replace", calls[0], "log.csv")
    assert list(pd.read_csv(path)["generation"]) == [100, 101]


class WithSpecies(ga.GA):
    """A GA that reports one extra log column, as NEAT reports its species."""

    def log_fields(self):
        return {"species": 3}


def test_a_methods_extra_fields_are_log_columns_also_after_resuming(evaluation, tmp_path):
    evaluation(0.5)
    with pytest.raises(Interrupt):
        evolution.run(WithSpecies(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=3,
                      on_generation=interrupt_at(1))
    reports = []
    evolution.run(WithSpecies(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=3,
                  on_generation=reports.append)
    log = pd.read_csv(tmp_path / "run" / "log.csv")
    assert list(log.columns) == evolution.LOG_FIELDS + ["species"]
    assert list(log["generation"]) == [0, 1, 2] and list(log["species"]) == [3, 3, 3]
    assert [r["species"] for r in reports] == [3, 3]


@pytest.mark.slow
def test_results_do_not_depend_on_the_number_of_workers(tmp_path):
    workers = 6                                         # splits each map's cars into two tasks
    assert len(population.split_tasks(12, range(config.TRACKS_PER_GEN), workers)) == 2 * config.TRACKS_PER_GEN
    for jobs in (1, workers):
        evolution.run(ga.GA(pop_size=12), 0, tmp_path / f"jobs{jobs}", champion_dir=None, n_jobs=jobs,
                      max_generations=2)
    pd.testing.assert_frame_equal(comparable(tmp_path / "jobs1" / "log.csv"),
                                  comparable(tmp_path / f"jobs{workers}" / "log.csv"))


def test_setup_statistics_are_logged_and_reported(evaluation, tmp_path):
    evaluation(0.5)
    summary, reports = train(tmp_path, 2)
    log = pd.read_csv(tmp_path / "run" / "log.csv")
    for report, (_, row) in zip(reports, log.iterrows()):
        for key, value in setup.summary(report["population"].setup).items():
            assert math.isclose(report[key], value) and math.isclose(row[key], value)
    assert summary["final_setup"] is None                    # no final controller was saved


def test_the_evaluated_population_carries_the_methods_setups(monkeypatch, tmp_path):
    seen = []

    def evaluate(cars, seeds, level, factor, n_jobs=1, backend="loky"):
        seen.append(cars.setup.copy())
        return fake_evaluate(0.5)(cars, seeds, level, factor)

    monkeypatch.setattr(population, "evaluate_population", evaluate)
    reports = []
    evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=3,
                  on_generation=reports.append)
    assert len(seen) == 3
    for given, report in zip(seen, reports):
        assert np.array_equal(given, report["population"].setup)
    assert not np.allclose(seen[0], np.asarray(config.DEFAULT_SETUP))   # a random-start GA spreads its setups


def test_the_final_controller_keeps_its_setup(evaluation, monkeypatch, tmp_path):
    evaluation(0.5)
    monkeypatch.setattr(population, "select_final",
                        lambda cars, candidates, factor, n_jobs=1: (int(candidates[-1]), 2, 0.9))
    method = ga.GA(pop_size=20)
    summary = evolution.run(method, 5, tmp_path / "run", champion_dir=tmp_path / "champions", max_generations=2)
    expected = method.setup[int(np.flatnonzero(method.is_kept)[-1])]
    saved = json.loads(paths.champion_path("ga", 5, ".setup.json", tmp_path / "champions").read_text())
    assert np.allclose(setup.from_dict(saved), expected) and summary["final_setup"] == setup.to_dict(expected)
    _, driver = nets.load_champion("ga", 5, tmp_path / "champions")
    assert np.allclose(driver.setup, expected)


def test_resuming_with_another_setup_mutation_is_refused(evaluation, monkeypatch, tmp_path):
    evaluation(0.5)
    with pytest.raises(Interrupt):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=5,
                      on_generation=interrupt_at(1))
    monkeypatch.setattr(config, "SETUP_MUTATION_P", 0.3)
    with pytest.raises(checkpoint.SettingsMismatch, match="SETUP_MUTATION_P"):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=5)
