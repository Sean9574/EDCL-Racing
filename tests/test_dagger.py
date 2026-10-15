import json
import os
from pathlib import Path

import numpy as np
import pytest

from racing import checkpoint, config, dagger, es, expert, imitation, nets, paths, progress, sim
from racing.actions import controls_to_class
from racing.backend import keras_cpu

FULL_RIGHT = int(controls_to_class(-1.0, 1.0)[0])     # full right lock at full throttle: leaves the circle


class Constant:
    """A driver that always takes the same action, with revision 3's setup (spec §9.11)."""

    setup = np.asarray(config.DEFAULT_SETUP, dtype=float)

    def __init__(self, action):
        self.action = action

    def act(self, obs):
        return np.full(len(obs), self.action, dtype=np.int64)

    def policy(self):
        return lambda obs, cars: self.act(obs)


class Expert:
    """The expert as a driver of one map, with revision 3's setup (spec §9.11)."""

    setup = np.asarray(config.DEFAULT_SETUP, dtype=float)

    def __init__(self, track):
        self.track = track

    def policy(self):
        return expert.expert_policy(self.track)


@pytest.fixture(autouse=True)
def private_results(monkeypatch, tmp_path):
    """Status and milestones go to the test's own folder, never to results/."""
    monkeypatch.setattr(config, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(config, "RUN_LOG_PATH", tmp_path / "run_log.txt")


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(config.N_INPUTS, 15))

    def make(n):
        x = rng.random((n, config.N_INPUTS)).astype(np.float32)
        return x, np.argmax(x @ w, axis=1).astype(np.int64)

    (x_tr, y_tr), (x_va, y_va) = make(2000), make(300)
    return {"X_train": x_tr, "y_train": y_tr, "X_val": x_va, "y_val": y_va}


@pytest.fixture
def trained_runs(tmp_path, toy_data):
    """A runs folder with a baseline imitation stage for seed 0."""
    runs = tmp_path / "runs"
    imitation.train_baseline(toy_data, out_dir=paths.run_dir("baseline", 0, runs), epochs=1, seed=0)
    return runs


def scripted(monkeypatch, checks):
    """collect returns seven rows that depend only on the maps; check returns `checks` in order and records the
    drivers it was given. Returns (maps collected per round, drivers checked)."""
    collected, drivers, queue = [], [], list(checks)

    def collect(driver, seeds, level=config.MAX_LEVEL, n_jobs=1):
        collected.append(list(seeds))
        rng = np.random.default_rng(seeds[0])
        return rng.random((7, config.N_INPUTS)).astype(np.float32), rng.integers(0, config.N_ACTIONS, 7), 1

    def check(driver, seeds, **kwargs):
        drivers.append(driver)
        return queue.pop(0)

    monkeypatch.setattr(dagger, "collect", collect)
    monkeypatch.setattr(dagger, "check", check)
    monkeypatch.setattr(config, "DAGGER_EPOCHS", 1)
    return collected, drivers


def synced_writes(monkeypatch):
    """Names of the files written through checkpoint.write_json, which syncs them to disk."""
    written, real = [], checkpoint.write_json
    monkeypatch.setattr(checkpoint, "write_json",
                        lambda path, data: written.append(Path(path).name) or real(path, data))
    return written


def test_dagger_parameters_are_recorded_settings():
    assert (config.DAGGER_MAPS, config.DAGGER_EPOCHS, config.DAGGER_HOLDOUT, config.DAGGER_GOAL,
            config.DAGGER_STREAK, config.DAGGER_PATIENCE) == (20, 30, 0.1, 18, 3, 10)
    assert {"DAGGER_MAPS", "DAGGER_EPOCHS", "DAGGER_HOLDOUT", "DAGGER_GOAL", "DAGGER_STREAK",
            "DAGGER_PATIENCE", "DAGGER_SALT"} <= set(checkpoint.SETTINGS_KEYS)


def test_labels_are_the_expert_action_in_the_states_the_car_reaches(monkeypatch, circle_track):
    monkeypatch.setattr(dagger, "get_track", lambda seed, level: circle_track)
    real_step, expected, driven = sim.step, [], []

    def step(track, cars, steer, throttle, time_limit):
        expected.append(int(controls_to_class(*expert.expert_controls(track, cars))[0]))
        driven.append((float(steer[0]), float(throttle[0])))
        real_step(track, cars, steer, throttle, time_limit)

    monkeypatch.setattr(sim, "step", step)
    x, y, respawns = dagger.drive_and_label(Constant(FULL_RIGHT), seed=5)
    assert list(y) == expected and len(x) == len(y) > 0
    assert x.dtype == np.float32 and x.shape[1] == config.N_INPUTS and y.dtype == np.int64
    assert set(driven) == {(-1.0, 1.0)}                 # the car drives itself; the expert only labels
    assert respawns >= 1                               # after a crash the car is put back and continues


def test_collect_keeps_the_rows_of_every_map_in_order(monkeypatch, circle_track):
    monkeypatch.setattr(dagger, "get_track", lambda seed, level: circle_track)
    one = dagger.drive_and_label(Constant(FULL_RIGHT), seed=1)
    x, y, respawns = dagger.collect(Constant(FULL_RIGHT), [1, 2])
    assert len(y) == 2 * len(one[1]) and np.array_equal(x[:len(one[0])], one[0]) and respawns == 2 * one[2]


def test_check_counts_time_trials_completed_within_the_target(monkeypatch, circle_track):
    monkeypatch.setattr(dagger, "get_track", lambda seed, level: circle_track)
    assert dagger.check(Constant(FULL_RIGHT), [1, 2]) == 0
    assert dagger.check(Expert(circle_track), [1, 2]) == 2                # 3 laps well within 1.3 × 3 × 20 s
    assert dagger.check(Expert(circle_track), [1], factor=0.3) == 0       # 18 s for three laps: impossible


def test_rounds_reach_the_goal_and_rows_accumulate(monkeypatch, toy_data, trained_runs):
    collected, _ = scripted(monkeypatch, [10, 18, 19, 20])
    sizes, real_train = [], dagger._train_round
    monkeypatch.setattr(dagger, "_train_round", lambda model, x, y, rng: sizes.append(len(y)) or real_train(model, x, y, rng))
    summary = dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs)
    assert summary["outcome"] == "goal" and summary["goal_reached"] and summary["rounds"] == 4
    assert [r["streak"] for r in summary["history"]] == [0, 1, 2, 3]
    assert summary["correction_rows"] == 28 and summary["final_checks"] == 20 and summary["best_checks"] == 20
    assert sizes == [len(toy_data["y_train"]) + 7 * k for k in (1, 2, 3, 4)]   # original rows plus every round
    run = paths.run_dir("baseline", 0, trained_runs)
    with np.load(run / "dagger_rows.npz") as rows:
        assert len(rows["y"]) == 28
    assert paths.stage_path("baseline", 0, "dagger", trained_runs).exists()
    assert not (run / "dagger_state.pkl").exists()
    maps = [seed for seeds in collected for seed in seeds]
    assert len(set(maps)) == len(maps) == 4 * config.DAGGER_MAPS and min(maps) >= config.TRAIN_SEED_START
    status = progress.read_status()
    assert status["stage"] == "dagger" and status["round"] == 4 and status["checks"] == 20
    assert "DAgger goal" in config.RUN_LOG_PATH.read_text()


def test_the_refinement_summary_is_synced_to_disk(monkeypatch, toy_data, trained_runs):
    scripted(monkeypatch, [18, 18, 18])
    written = synced_writes(monkeypatch)
    dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs)
    assert written == ["dagger.json"]


def test_correction_rows_are_synced_to_disk_before_they_replace_the_old_ones(monkeypatch, toy_data, trained_runs):
    scripted(monkeypatch, [18, 18, 18])
    calls, real_fsync, real_replace = [], os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(Path(os.readlink(f"/proc/self/fd/{fd}")).name)
                        or real_fsync(fd))
    monkeypatch.setattr(os, "replace", lambda a, b: calls.append(("replace", Path(a).name, Path(b).name))
                        or real_replace(a, b))
    dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs)
    replaced = [i for i, call in enumerate(calls) if isinstance(call, tuple) and call[2] == dagger.ROWS_NAME]
    assert len(replaced) == 3 and all(calls[i - 1] == calls[i][1] for i in replaced)


def test_fewer_correction_rows_than_the_checkpoint_records_is_an_error(tmp_path):
    """Rows lost in a crash would silently change the refinement; it stops and names the files instead."""
    path = tmp_path / dagger.ROWS_NAME
    x, y = dagger._load_rows(path, 0)
    assert x.shape == (0, config.N_INPUTS) and len(y) == 0
    with pytest.raises(RuntimeError, match=f"{dagger.ROWS_NAME}.*{dagger.STATE_NAME}"):
        dagger._load_rows(path, 5)                                          # missing
    np.savez(path, X=np.zeros((3, config.N_INPUTS), dtype=np.float32), y=np.zeros(3, dtype=np.int64))
    with pytest.raises(RuntimeError, match="3 correction rows.*5"):
        dagger._load_rows(path, 5)                                          # short
    path.write_bytes(b"\0" * 100)
    with pytest.raises(RuntimeError, match=dagger.STATE_NAME):
        dagger._load_rows(path, 5)                                          # zeroed by a reboot
    np.savez(path, X=np.ones((8, config.N_INPUTS), dtype=np.float32), y=np.arange(8))
    x, y = dagger._load_rows(path, 5)                                       # rows of an unfinished round: dropped
    assert len(x) == len(y) == 5 and list(y) == [0, 1, 2, 3, 4]


def test_a_stalled_refinement_keeps_the_best_network(monkeypatch, toy_data, trained_runs):
    monkeypatch.setattr(config, "DAGGER_PATIENCE", 3)
    _, drivers = scripted(monkeypatch, [5, 12, 9, 11, 10])
    summary = dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs)
    assert summary["outcome"] == "stalled" and not summary["goal_reached"] and summary["rounds"] == 5
    assert summary["best_checks"] == 12 and summary["final_checks"] == 10
    model = keras_cpu().models.load_model(paths.stage_path("baseline", 0, "dagger", trained_runs), compile=False)
    for (w, b), (best_w, best_b) in zip(nets.from_keras(model).layers, drivers[1].layers):
        assert np.allclose(w, best_w) and np.allclose(b, best_b)


def test_a_complete_refinement_is_not_repeated(monkeypatch, toy_data, trained_runs):
    scripted(monkeypatch, [18, 18, 18])
    first = dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs)
    monkeypatch.setattr(dagger, "collect", lambda *args, **kwargs: pytest.fail("refined again"))
    assert dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs) == json.loads(json.dumps(first))


def test_a_round_limit_ends_the_refinement_with_the_best_network(monkeypatch, toy_data, trained_runs):
    scripted(monkeypatch, [4, 6])
    summary = dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs, max_rounds=2)
    assert summary["outcome"] == "max_rounds" and summary["rounds"] == 2 and summary["best_checks"] == 6


def test_a_refinement_checkpoint_with_other_settings_is_refused(toy_data, trained_runs):
    run = paths.run_dir("baseline", 0, trained_runs)
    checkpoint.save(run / "dagger_state.pkl", {"settings": {"TARGET_FACTOR": 9.9}})
    with pytest.raises(checkpoint.SettingsMismatch):
        dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs)


def test_refinement_needs_the_imitation_stage(toy_data, tmp_path):
    with pytest.raises(FileNotFoundError, match="notebook 02"):
        dagger.refine("kt_random", 1, toy_data, runs_dir=tmp_path / "runs")


@pytest.mark.slow
def test_an_interrupted_refinement_resumes_to_the_same_result(monkeypatch, toy_data, tmp_path):
    """Rows and checks depend only on the maps, so both refinements see the same data; the interrupted one stops
    inside its second round, after that round's rows were saved but before its checkpoint."""
    monkeypatch.setattr(config, "DAGGER_EPOCHS", 2)

    def collect(driver, seeds, level=config.MAX_LEVEL, n_jobs=1):
        rng = np.random.default_rng(seeds[0])
        return rng.random((40, config.N_INPUTS)).astype(np.float32), rng.integers(0, config.N_ACTIONS, 40), 0

    monkeypatch.setattr(dagger, "collect", collect)
    monkeypatch.setattr(dagger, "check", lambda driver, seeds, **kwargs: 10 + seeds[0] % 7)
    results, real_train = {}, dagger._train_round
    for name in ("straight", "interrupted"):
        runs = tmp_path / name
        imitation.train_baseline(toy_data, out_dir=paths.run_dir("baseline", 0, runs), epochs=1, seed=0)
        if name == "interrupted":
            calls = []

            def flaky(*args):
                calls.append(1)
                if len(calls) == 2:
                    raise KeyboardInterrupt
                return real_train(*args)

            monkeypatch.setattr(dagger, "_train_round", flaky)
            with pytest.raises(KeyboardInterrupt):
                dagger.refine("baseline", 0, toy_data, runs_dir=runs, max_rounds=3)
            monkeypatch.setattr(dagger, "_train_round", real_train)
        results[name] = dagger.refine("baseline", 0, toy_data, runs_dir=runs, max_rounds=3)
    straight, resumed = results["straight"], results["interrupted"]
    assert [r["checks"] for r in straight["history"]] == [r["checks"] for r in resumed["history"]]
    assert straight["correction_rows"] == resumed["correction_rows"] == 120
    load = keras_cpu().models.load_model
    a = load(paths.stage_path("baseline", 0, "dagger", tmp_path / "straight"), compile=False).get_weights()
    b = load(paths.stage_path("baseline", 0, "dagger", tmp_path / "interrupted"), compile=False).get_weights()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_speed_push_grows_an_es_population_from_the_refined_network(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    model = imitation.build_network([8, 4], "tanh", 0.1, 0.003)
    stage = paths.stage_path("kt_random", 2, "dagger", runs)
    stage.parent.mkdir(parents=True)
    model.save(stage)
    calls = {}
    monkeypatch.setattr(es, "run_es", lambda **kwargs: calls.update(kwargs) or {"approach": kwargs["approach"]})
    result = dagger.speed_push("kt_random", 2, runs_dir=runs, champion_dir=tmp_path / "champions", max_generations=3)
    assert result == {"approach": "kt_random"}
    assert calls["approach"] == "kt_random" and calls["seed"] == 2 and calls["start_level"] == config.MAX_LEVEL
    assert tuple(calls["hidden"]) == (8, 4) and calls["activation"] == "tanh" and calls["max_generations"] == 3
    assert calls["out_dir"] == paths.speed_dir("kt_random", 2, runs)
    assert calls["champion_dir"] == tmp_path / "champions"
    assert np.array_equal(calls["initial"], nets.to_flat(model))


def test_run_refines_then_pushes(monkeypatch):
    order = []
    monkeypatch.setattr(dagger, "refine", lambda approach, seed, data, **kwargs: order.append("refine") or {"r": 1})
    monkeypatch.setattr(dagger, "speed_push", lambda approach, seed, **kwargs: order.append("speed") or {"s": 2})
    assert dagger.run("baseline", 1, {}) == {"approach": "baseline", "seed": 1, "dagger": {"r": 1}, "speed": {"s": 2}}
    assert order == ["refine", "speed"]
