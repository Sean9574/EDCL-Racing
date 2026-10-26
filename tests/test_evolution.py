import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from racing import checkpoint, config, curriculum, evolution, ga, nets, paths, population, progress, setup
from racing.population import TrialResult

GA_FIELDS = ["mean_layers", "max_layers", "mean_nodes", "max_nodes", "mean_params", "max_params"]   # what the GA logs


def fake_evaluate(success_rate):
    """Evaluation without simulation: fitness falls with the size of a car's weights, and the fittest
    `success_rate` share of the population succeeds on every map."""
    def evaluate(population, seeds, level, factor, n_jobs=1, backend="loky"):
        weights = population.weights                # a list of vectors when the cars differ in shape (the GA)
        n = len(weights)
        fitness = 1.0 / (1.0 + np.array([np.sum(w ** 2) for w in weights]))
        rank = np.argsort(np.argsort(-fitness, kind="stable"), kind="stable")
        success = np.tile(rank < success_rate * n, (len(seeds), 1))
        return TrialResult(fitness=fitness + success[0], success=success,
                           laps=np.where(success, config.LAPS, 1), finish_time=np.where(success, 90.0, np.nan),
                           expert_lap=np.full(len(seeds), 20.0))
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
    assert list(log.columns) == evolution.LOG_FIELDS + GA_FIELDS and list(log["generation"]) == [0, 1, 2]
    maps = [s for r in reports for s in r["seeds"]]
    assert len(maps) == len(set(maps)) == 9 and min(maps) >= config.TRAIN_SEED_START   # fresh maps, never reused
    assert (tmp_path / "run" / checkpoint.CHECKPOINT_NAME).exists()
    saved = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert saved["settings"] == checkpoint.settings_record("ga") and saved["goal_reached"] is False
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
    for report in reports:                                # three distinct real training layouts per generation
        assert len(set(report["seeds"])) == config.TRACKS_PER_GEN and set(report["seeds"]) <= set(range(38))


class StallsAtOnce(curriculum.MasteryTracker):
    def update(self, min_share):
        super().update(min_share)
        self.outcome = "stalled"
        return "end_stalled"


def test_a_stalled_run_still_saves_a_final_controller(evaluation, monkeypatch, tmp_path):
    evaluation(0.0)
    monkeypatch.setattr(evolution, "MasteryTracker", StallsAtOnce)
    chosen = []

    def select(population, candidates, n_jobs=1):
        chosen.append(len(candidates))
        return int(candidates[0]), 3, 0.7

    monkeypatch.setattr(population, "select_final", select)
    summary, reports = train(tmp_path, None, champion_dir=tmp_path / "champions")
    assert len(reports) == 1 and reports[0]["event"] == "end_stalled"
    assert summary["goal_reached"] is False and summary["outcome"] == "stalled" and summary["level_reached"] == 1
    assert len(chosen) == 1 and summary["val_successes"] == 3
    assert summary["champion_path"] == str(paths.champion_path("ga", 0, champion_dir=tmp_path / "champions"))
    assert paths.champion_path("ga", 0, ".npy", tmp_path / "champions").exists()
    assert "GA run 0: stopped at generation 0" in config.RUN_LOG_PATH.read_text()


def test_the_final_controller_comes_from_the_kept_cars(evaluation, monkeypatch, tmp_path):
    evaluation(1.0)
    chosen = []

    def select(population, candidates, n_jobs=1):
        chosen.append((np.asarray(candidates), population))
        return int(candidates[-1]), 20, 1.9

    monkeypatch.setattr(population, "select_final", select)
    method = ga.GA(pop_size=20)
    summary = evolution.run(method, 4, tmp_path / "run", champion_dir=tmp_path / "champions",
                            start_level=config.MAX_LEVEL, max_generations=6)
    candidates, cars = chosen[0]                          # not `population`: that would shadow the module above
    assert np.array_equal(candidates, np.flatnonzero(method.is_kept)) and cars is method.population
    saved = np.load(paths.champion_path("ga", 4, ".npy", tmp_path / "champions"))
    assert np.array_equal(saved, method.population.weights[candidates[-1]])
    assert summary["val_successes"] == 20 and summary["wall_time"] > 0
    assert summary["machine"] == checkpoint.machine_record()     # runs of one experiment may come from several machines


def test_an_end_request_ends_a_run_after_one_generation_with_its_final_controller(evaluation, monkeypatch, tmp_path):
    evaluation(0.5)
    chosen = []

    def select(population, candidates, n_jobs=1):
        chosen.append(np.asarray(candidates))
        return int(candidates[0]), 18, 1.5

    monkeypatch.setattr(population, "select_final", select)
    config.END_FILE.write_text("2026-10-25T23:15:00")
    method, reports = ga.GA(pop_size=20), []
    summary = evolution.run(method, 1, tmp_path / "run", champion_dir=tmp_path / "champions",
                            on_generation=reports.append)
    assert [report["generation"] for report in reports] == [0]
    assert summary["outcome"] == "time_limit" and summary["generations"] == 1 and summary["val_successes"] == 18
    assert len(chosen) == 1 and np.array_equal(chosen[0], np.flatnonzero(method.is_kept))
    assert paths.champion_path("ga", 1, ".npy", tmp_path / "champions").exists()
    log = config.RUN_LOG_PATH.read_text()
    assert "GA run 1: ended by request at generation 1" in log and "GA run 1: finished after 1 generations" in log


def test_a_run_resumed_under_an_end_request_picks_its_final_controller_without_training(evaluation, monkeypatch,
                                                                                         tmp_path):
    evaluation(0.5)
    monkeypatch.setattr(population, "select_final",
                        lambda population, candidates, n_jobs=1: (int(candidates[0]), 2, 0.4))
    with pytest.raises(Interrupt):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=tmp_path / "champions",
                      on_generation=interrupt_at(3))
    config.END_FILE.write_text("2026-10-25T23:15:00")
    summary = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=tmp_path / "champions",
                            on_generation=lambda report: pytest.fail("trained again"))
    assert summary["outcome"] == "time_limit" and summary["generations"] == 3 and summary["val_successes"] == 2
    assert list(pd.read_csv(tmp_path / "run" / "log.csv")["generation"]) == [0, 1, 2]
    assert paths.champion_path("ga", 0, ".npy", tmp_path / "champions").exists()


def test_viewer_and_pilot_runs_ignore_an_end_request(evaluation, tmp_path):
    evaluation(0.5)
    config.END_FILE.write_text("2026-10-25T23:15:00")
    summary, reports = train(tmp_path, 4)
    assert len(reports) == 4 and summary["outcome"] is None and summary["generations"] == 4


def test_a_resumed_run_matches_an_uninterrupted_one(evaluation, tmp_path):
    evaluation(0.5)
    evolution.run(ga.GA(pop_size=20), 0, tmp_path / "whole", champion_dir=None, max_generations=5)
    with pytest.raises(Interrupt):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "split", champion_dir=None, max_generations=5,
                      on_generation=interrupt_at(3))
    # generation 3 was logged, but the run stopped before its checkpoint (spec §3.3)
    assert list(pd.read_csv(tmp_path / "split" / "log.csv")["generation"]) == [0, 1, 2, 3]
    summary = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "split", champion_dir=None, max_generations=5)
    whole, split = comparable(tmp_path / "whole" / "log.csv"), comparable(tmp_path / "split" / "log.csv")
    assert list(split["generation"]) == [0, 1, 2, 3, 4]
    pd.testing.assert_frame_equal(whole, split)
    ends = [checkpoint.load(tmp_path / name / checkpoint.CHECKPOINT_NAME) for name in ("whole", "split")]
    a, b = ends[0]["method"], ends[1]["method"]
    assert a["shapes"] == b["shapes"] and len(a["networks"]) == len(b["networks"])
    assert all(np.array_equal(x, y) for x, y in zip(a["networks"], b["networks"]))
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
                        lambda population, candidates, n_jobs=1: (int(candidates[0]), 1, 1.0))
    evolution.run(ga.GA(pop_size=20), 2, tmp_path / "run", champion_dir=tmp_path / "champions", max_generations=4)
    status = progress.read_status()
    assert status["approach"] == "ga" and status["seed"] == 2 and status["stage"] == "evolution"
    assert status["level"] == 2 and status["generation"] == 3 and status["min_share"] == 1.0
    log = config.RUN_LOG_PATH.read_text()
    assert "GA run 2: started" in log and "Easy mastered at generation 2, now Medium" in log
    assert "GA run 2: finished" in log and "final controller finishes 1 of 20 validation maps" in log


def test_viewer_and_pilot_runs_report_nothing(evaluation, tmp_path):
    evaluation(1.0)
    train(tmp_path, 4)
    assert not config.STATUS_PATH.exists() and not config.RUN_LOG_PATH.exists()


def test_a_reported_run_keeps_its_latest_generation_for_the_viewer(evaluation, monkeypatch, tmp_path):
    evaluation(0.5)
    monkeypatch.setattr(population, "select_final",
                        lambda population, candidates, n_jobs=1: (int(candidates[0]), 1, 1.0))
    _, reports = train(tmp_path, 3, champion_dir=tmp_path / "champions")
    replay = checkpoint.load(tmp_path / "run" / paths.REPLAY_NAME)
    last = reports[-1]
    assert replay["generation"] == last["generation"] == 2 and replay["seeds"] == last["seeds"]
    assert np.array_equal(replay["fitness"], last["fitness"])
    assert replay["population"].shapes == last["population"].shapes
    assert len(replay["population"].weights) == len(last["population"].weights)
    assert all(np.array_equal(a, b) for a, b in zip(replay["population"].weights, last["population"].weights))
    assert np.array_equal(replay["population"].setup, last["population"].setup)


def test_viewer_runs_write_no_replay(evaluation, tmp_path):
    evaluation(0.5)
    train(tmp_path, 2)
    assert not (tmp_path / "run" / paths.REPLAY_NAME).exists()


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
                        lambda cars, candidates, n_jobs=1: (int(candidates[-1]), 2, 0.9))
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


LOWER_IN_A_FRESH_PROCESS = """
import os, threading
from pathlib import Path
from racing.evolution import lower_priority
waiting = threading.Event()
helper = threading.Thread(target=waiting.wait)            # a thread that exists before, like numpy's
helper.start()
start = os.getpriority(os.PRIO_PROCESS, 0)
lower = min(start + 2, 19)


def niceness():
    return sorted({int(Path(f"/proc/self/task/{t}/stat").read_text().rsplit(")", 1)[1].split()[16])
                   for t in os.listdir("/proc/self/task")})


lower_priority(lower)
first = niceness()
lower_priority(start)                                     # never raised again
child = threading.Thread(target=lambda: None)             # threads started later inherit it
child.start()
child.join()
print(first == [lower], niceness() == [lower])
waiting.set()
"""


def test_an_evolution_phase_lowers_the_priority_of_every_thread_and_never_raises_it():
    """Linux keeps a priority per thread; every thread of the process is lowered, and later ones inherit it
    (spec §3.3). Run in a fresh process, so the test process keeps its priority."""
    env = {**os.environ, "PYTHONPATH": str(config.PROJECT_DIR)}
    out = subprocess.run([sys.executable, "-c", LOWER_IN_A_FRESH_PROCESS], capture_output=True, text=True, check=True,
                         env=env, cwd=config.PROJECT_DIR).stdout.split()
    assert out == ["True", "True"]


def test_a_run_lowers_its_priority_before_its_first_generation(evaluation, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(evolution, "lower_priority", lambda nice=config.EVOLUTION_NICE: calls.append(nice))
    evaluation(0.0)
    train(tmp_path, 1)
    assert calls == [config.EVOLUTION_NICE] == [19]


def test_two_starts_of_run_1_differ_and_run_0_replays(evaluation, tmp_path):
    evaluation(0.0)
    logs = {}
    for name, seed in (("a0", 0), ("b0", 0), ("a1", 1), ("b1", 1)):
        evolution.run(ga.GA(pop_size=20), seed, tmp_path / name, max_generations=2, champion_dir=None)
        logs[name] = comparable(tmp_path / name / "log.csv")
    assert logs["a0"].equals(logs["b0"])                         # run 0: the same seed, the same run
    assert not logs["a1"].equals(logs["b1"])                     # run 1: no seed, another run every time


def test_a_stopped_run_1_resumes_on_its_own_path(evaluation, tmp_path):
    import shutil
    evaluation(0.0)
    with pytest.raises(Interrupt):
        evolution.run(ga.GA(pop_size=20), 1, tmp_path / "run", max_generations=5, champion_dir=None,
                      on_generation=interrupt_at(2))
    shutil.copytree(tmp_path / "run", tmp_path / "copy")
    for name in ("run", "copy"):
        evolution.run(ga.GA(pop_size=20), 1, tmp_path / name, max_generations=5, champion_dir=None)
    assert comparable(tmp_path / "run" / "log.csv").equals(comparable(tmp_path / "copy" / "log.csv"))


# ----- the stall boost (spec §9.4, version 3.6) -----

def test_the_run_passes_the_stall_to_advance_and_logs_it(evaluation, tmp_path, monkeypatch):
    seen = []
    real = ga.GA.advance
    monkeypatch.setattr(ga.GA, "advance", lambda self, rng, result, kept, event, boost=False:
                        seen.append(boost) or real(self, rng, result, kept, event, boost=boost))
    monkeypatch.setattr(config, "STALL_GENERATIONS", 2)
    evaluation(1.0)                     # every car succeeds: the goal at generation 2, then a target every third
    _, reports = train(tmp_path, 12, start_level=config.MAX_LEVEL)
    log = pd.read_csv(tmp_path / "run" / "log.csv")
    assert list(log.columns) == evolution.LOG_FIELDS + GA_FIELDS and "boost" in log.columns
    assert list(log["boost"]) == [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]       # on after 2 generations, off when mastered
    assert seen == list(log["boost"]) == [r["boost"] for r in reports]   # what advance got is what the log says


def test_without_a_stall_the_boost_stays_off(evaluation, tmp_path):
    evaluation(1.0)
    train(tmp_path, 12, start_level=config.MAX_LEVEL)
    assert set(pd.read_csv(tmp_path / "run" / "log.csv")["boost"]) == {0}


def masters_only_the_first_target(cars, seeds, level, factor, n_jobs=1, backend="loky"):
    """Every car succeeds until the goal, none in the speed push: the push stalls and stays stalled."""
    return fake_evaluate(1.0 if factor >= config.TARGET_FACTOR else 0.0)(cars, seeds, level, factor)


def test_a_resumed_run_0_in_a_stall_replays_exactly(monkeypatch, tmp_path):
    """Review focus 2: a run 0 stopped in a stall resumes to exactly the uninterrupted run; the boost comes back from
    the checkpoint's tracker (spec §9.4)."""
    monkeypatch.setattr(config, "STALL_GENERATIONS", 2)
    monkeypatch.setattr(population, "evaluate_population", masters_only_the_first_target)
    evolution.run(ga.GA(pop_size=20), 0, tmp_path / "whole", champion_dir=None, max_generations=8,
                  start_level=config.MAX_LEVEL)
    with pytest.raises(Interrupt):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "cut", champion_dir=None, max_generations=8,
                      start_level=config.MAX_LEVEL, on_generation=interrupt_at(6))
    stopped = checkpoint.load(tmp_path / "cut" / checkpoint.CHECKPOINT_NAME)
    assert stopped["generation"] == 6 and curriculum.MasteryTracker.from_state(stopped["tracker"]).stalled
    evolution.run(ga.GA(pop_size=20), 0, tmp_path / "cut", champion_dir=None, max_generations=8,
                  start_level=config.MAX_LEVEL)
    whole, cut = comparable(tmp_path / "whole" / "log.csv"), comparable(tmp_path / "cut" / "log.csv")
    assert list(whole["boost"]) == [0, 0, 0, 0, 1, 1, 1, 1]                  # boosted before and after the cut
    pd.testing.assert_frame_equal(whole, cut)
    ends = [checkpoint.load(tmp_path / name / checkpoint.CHECKPOINT_NAME) for name in ("whole", "cut")]
    a, b = ends[0]["method"], ends[1]["method"]
    assert a["shapes"] == b["shapes"] and len(a["networks"]) == len(b["networks"])
    assert all(np.array_equal(x, y) for x, y in zip(a["networks"], b["networks"]))
    assert np.array_equal(ends[0]["method"]["setup"], ends[1]["method"]["setup"]) and ends[0]["rng"] == ends[1]["rng"]


def test_a_log_without_the_boost_column_resumes_without_it(evaluation, tmp_path):
    """A log started before version 3.6 keeps its columns; its rows simply do not record the boost."""
    import csv
    evaluation(0.5)
    with pytest.raises(Interrupt):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=4,
                      on_generation=interrupt_at(2))
    path = tmp_path / "run" / "log.csv"
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    column = rows[0].index("boost")
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(row[:column] + row[column + 1:] for row in rows)    # the log of an older run
    evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=4)
    resumed = pd.read_csv(path)
    assert "boost" not in resumed.columns and list(resumed["generation"]) == [0, 1, 2, 3]


# ----- hardening (spec §9.4, version 3.15) -----

class EndsAsBeforeVersion3_15(curriculum.MasteryTracker):
    """The tracker of version 3.14: the end of the speed push ends the run."""
    def update(self, min_share):
        event = super().update(min_share)
        if event == "end_speed":
            self.hardening, self.outcome = False, "speed_limit"
        return event


def picks_the_first_candidate(calls):
    def select(population, candidates, n_jobs=1):
        calls.append(len(candidates))
        return int(candidates[0]), 20, 1.0
    return select


def as_before_version_3_15(run_dir):
    """The run as version 3.14 left it: no hardening record, FINAL_MIN_SUCCESSES 18. Returns its summary."""
    summary = json.loads((run_dir / "summary.json").read_text())
    del summary["hardening"]
    summary["settings"]["FINAL_MIN_SUCCESSES"] = 18
    (run_dir / "summary.json").write_text(json.dumps(summary))
    state = checkpoint.load(run_dir / checkpoint.CHECKPOINT_NAME)
    del state["hardening"]
    state["settings"]["FINAL_MIN_SUCCESSES"] = 18
    checkpoint.save(run_dir / checkpoint.CHECKPOINT_NAME, state)
    return summary


def ended_at_its_speed_limit(monkeypatch, tmp_path, calls):
    """A run that reached the goal, mastered no shorter target and ended at its speed limit under version 3.14."""
    monkeypatch.setattr(config, "SPEED_PATIENCE", 3)
    monkeypatch.setattr(population, "evaluate_population", masters_only_the_first_target)
    monkeypatch.setattr(population, "select_final", picks_the_first_candidate(calls))
    monkeypatch.setattr(evolution, "MasteryTracker", EndsAsBeforeVersion3_15)
    summary = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=tmp_path / "champions",
                            start_level=config.MAX_LEVEL)
    monkeypatch.setattr(evolution, "MasteryTracker", curriculum.MasteryTracker)
    assert summary["outcome"] == "speed_limit" and summary["generations"] == 6
    return as_before_version_3_15(tmp_path / "run")


def test_a_run_ended_at_its_speed_limit_hardens_at_its_last_mastered_target(monkeypatch, tmp_path):
    calls = []
    earlier = ended_at_its_speed_limit(monkeypatch, tmp_path, calls)
    reports = []
    summary = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=tmp_path / "champions",
                            on_generation=reports.append)
    assert [r["generation"] for r in reports] == [6, 7, 8] and [r["event"] for r in reports] == [None, None, "hardened"]
    assert all(r["factor"] == config.TARGET_FACTOR and not r["speed_push"] and r["boost"] == 0 for r in reports)
    assert summary["outcome"] == "hardened" and summary["generations"] == 9 and len(calls) == 2
    hardening = summary["hardening"]
    assert (hardening["factor"], hardening["from_generation"], hardening["generation"]) == (config.TARGET_FACTOR, 6, 8)
    assert json.loads((tmp_path / "run" / evolution.EARLIER_SUMMARY_NAME).read_text()) == earlier
    log = config.RUN_LOG_PATH.read_text()
    assert "GA run 0: hardening at the target 0.0% shorter, from generation 6" in log
    assert "GA run 0: hardened at generation 8" in log


def test_a_run_interrupted_while_hardening_resumes_hardening(monkeypatch, tmp_path):
    """Review focus 1 and 2: cut before the first hardening checkpoint, the run still hardens; meanwhile it has no
    summary.json, so the fleet does not count it finished."""
    ended_at_its_speed_limit(monkeypatch, tmp_path, [])
    with pytest.raises(Interrupt):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=tmp_path / "champions",
                      on_generation=interrupt_at(6))
    assert not (tmp_path / "run" / "summary.json").exists()
    summary = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=tmp_path / "champions")
    assert summary["outcome"] == "hardened" and summary["hardening"]["from_generation"] == 6


def test_an_end_request_ends_hardening_and_the_final_pick_follows(monkeypatch, tmp_path):
    calls = []
    ended_at_its_speed_limit(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(population, "evaluate_population", fake_evaluate(0.85))      # below the 90% rule

    def end_after(report):
        if report["generation"] == 7:
            config.END_FILE.write_text("2026-10-27T07:00:00")

    summary = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=tmp_path / "champions",
                            on_generation=end_after)
    assert summary["outcome"] == "time_limit" and summary["generations"] == 8 and len(calls) == 2
    assert summary["hardening"] == {"factor": config.TARGET_FACTOR, "from_generation": 6}
    assert "GA run 0: ended by request at generation 8" in config.RUN_LOG_PATH.read_text()


def test_a_run_resumed_into_hardening_under_an_end_request_ends_at_once(monkeypatch, tmp_path):
    """Review focus 5."""
    calls = []
    ended_at_its_speed_limit(monkeypatch, tmp_path, calls)
    config.END_FILE.write_text("2026-10-27T07:00:00")
    summary = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=tmp_path / "champions",
                            on_generation=lambda report: pytest.fail("trained again"))
    assert summary["outcome"] == "time_limit" and summary["generations"] == 6 and len(calls) == 2
    assert summary["hardening"] == {"factor": config.TARGET_FACTOR, "from_generation": 6}


def test_a_run_ended_before_its_goal_continues_its_levels_and_ends_hardened_at_the_goal(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "select_final", picks_the_first_candidate([]))
    monkeypatch.setattr(population, "evaluate_population", fake_evaluate(0.5))

    def end_after(report):                                    # version 3.14's timed end, found on Hard
        if report["generation"] == 1:
            config.END_FILE.write_text("2026-10-26T07:00:00")

    evolution.run(ga.GA(pop_size=20), 3, tmp_path / "run", champion_dir=tmp_path / "champions",
                  start_level=config.MAX_LEVEL - 1, on_generation=end_after)
    as_before_version_3_15(tmp_path / "run")
    config.END_FILE.unlink()                                  # fleet.py start removes it
    monkeypatch.setattr(population, "evaluate_population", fake_evaluate(1.0))
    reports = []
    summary = evolution.run(ga.GA(pop_size=20), 3, tmp_path / "run", champion_dir=tmp_path / "champions",
                            on_generation=reports.append)
    assert [r["event"] for r in reports] == [None, None, "level_up", None, None, "hardened"]
    assert summary["outcome"] == "hardened" and summary["goal_reached"] and summary["tightenings"] == []
    assert [e["level"] for e in summary["levels"]] == [config.MAX_LEVEL - 1, config.MAX_LEVEL]
    assert summary["hardening"]["factor"] == config.TARGET_FACTOR and summary["hardening"]["from_generation"] == 2
    assert "GA run 3: hardening from generation 2, on Hard" in config.RUN_LOG_PATH.read_text()


def test_the_end_of_the_speed_push_starts_hardening(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SPEED_PATIENCE", 3)
    monkeypatch.setattr(population, "evaluate_population", masters_only_the_first_target)
    monkeypatch.setattr(population, "select_final", picks_the_first_candidate([]))
    reports = []
    summary = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=tmp_path / "champions",
                            start_level=config.MAX_LEVEL, on_generation=reports.append)
    assert [r["event"] for r in reports] == [None, None, "goal", None, None, "end_speed", None, None, "hardened"]
    assert [r["speed_push"] for r in reports] == [False] * 3 + [True] * 3 + [False] * 3
    assert summary["outcome"] == "hardened" and summary["hardening"]["from_generation"] == 6
    assert ("GA run 0: speed push ended at generation 5, hardening at the target 0.0% shorter"
            in config.RUN_LOG_PATH.read_text())
    assert progress.read_status()["stage"] == "hardening"


def test_a_run_ended_under_version_3_15_is_not_hardened_again(evaluation, monkeypatch, tmp_path):
    evaluation(0.5)
    monkeypatch.setattr(population, "select_final", picks_the_first_candidate([]))
    config.END_FILE.write_text("2026-10-27T07:00:00")
    first = evolution.run(ga.GA(pop_size=20), 1, tmp_path / "run", champion_dir=tmp_path / "champions")
    config.END_FILE.unlink()
    again = evolution.run(ga.GA(pop_size=20), 1, tmp_path / "run", champion_dir=tmp_path / "champions",
                          on_generation=lambda report: pytest.fail("trained again"))
    assert first["hardening"] is None and again == json.loads(json.dumps(first))
