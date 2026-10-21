import json
import numpy as np
import pytest

from racing import checkpoint, config, expert, sim, tracks
from racing.actions import class_to_controls, controls_to_class


def test_expert_finishes_the_circle(circle_track):
    lap = expert.expert_lap_time(circle_track)
    corner_speed = np.sqrt(config.EXPERT_GRIP_MARGIN * config.GRIP * 50.0)
    assert lap is not None
    assert circle_track.length / 23.0 < lap < 2.5 * circle_track.length / corner_speed


def test_expert_policy_returns_action_classes(circle_track):
    cars = sim.start_cars(circle_track, 4)
    classes = expert.expert_policy(circle_track)(sim.sense(circle_track, cars), cars)
    assert classes.shape == (4,) and classes.min() >= 0 and classes.max() < config.N_ACTIONS


def test_get_track_attaches_lap_time_and_caches():
    t = tracks.get_track(0, 1)
    assert np.isfinite(t.expert_lap_time) and t.expert_lap_time > 0
    assert tracks.get_track(0, 1) is t


@pytest.mark.parametrize("level", [2, 3, 4])
def test_expert_drives_generated_tracks(level):
    for seed in range(5):
        assert tracks.get_track(seed, level).expert_lap_time > 0


def test_collect_rows_shapes_and_labels(circle_track):
    x, y, _ = expert.collect_rows(circle_track, np.random.default_rng(0))
    assert x.shape[1] == config.N_INPUTS and len(x) == len(y) > 100
    assert y.min() >= 0 and y.max() < config.N_ACTIONS
    assert len(np.unique(y)) >= 3          # disturbances produce corrections, not one repeated action


def test_collect_dataset_writes_all_splits(tmp_path):
    path = tmp_path / "data.npz"
    data, summary = expert.collect_dataset(path, splits={"train": [(0, 1)], "val": [(0, 2)], "test": [(1, 2)]})
    assert sorted([p.name for p in tmp_path.iterdir()]) == ["data.npz", "data_info.json"]        # no temporary file left behind
    for split in ("train", "val", "test"):
        assert data[f"X_{split}"].dtype == np.float32 and data[f"y_{split}"].dtype == np.int64
        assert len(data[f"X_{split}"]) == len(data[f"y_{split}"]) > 0
        assert summary[split]["rows"] == len(data[f"y_{split}"]) and summary[split]["tracks"] == 1
        assert summary[split]["respawns"] >= 0


def test_collect_dataset_keeps_the_old_file_if_writing_fails(monkeypatch, tmp_path, circle_track):
    monkeypatch.setattr(expert, "get_track", lambda seed, level: circle_track)
    path = tmp_path / "data.npz"
    path.write_bytes(b"old")

    def broken(file, **arrays):
        file.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", broken)
    with pytest.raises(OSError, match="disk full"):
        expert.collect_dataset(path, splits={"train": [(0, 1)]})
    assert path.read_bytes() == b"old" and [p.name for p in tmp_path.iterdir()] == ["data.npz"]


def test_a_crash_does_not_end_collection(monkeypatch, circle_track):
    real_step, calls, last = sim.step, [0], {}

    def step(track, cars, steer, throttle, time_limit):
        calls[0] += 1
        if calls[0] == 100:
            cars.x[0] = 1e4                                 # off the map: the car crashes in this step
        real_step(track, cars, steer, throttle, time_limit)
        last["cars"] = cars

    monkeypatch.setattr(sim, "step", step)
    x, y, respawns = expert.collect_rows(circle_track, np.random.default_rng(0))
    cars = last["cars"]
    assert respawns >= 1 and len(y) == calls[0] > 100
    assert cars.finished[0] and not cars.crashed[0]               # a full lap of progress was covered


def test_respawn_puts_the_car_back_on_the_centerline(circle_track):
    cars = sim.start_cars(circle_track, 1)
    cars.index[:] = 3 * len(circle_track.centerline) + 40       # unwrapped index on the fourth lap
    cars.x[:], cars.speed[:] = 1e4, 0.0
    cars.crashed[:], cars.done[:] = True, True
    assert expert.respawn(circle_track, cars) == 1
    assert np.allclose([cars.x[0], cars.y[0]], circle_track.centerline[40])
    t = circle_track.tangent[40]
    assert np.isclose(cars.heading[0], np.arctan2(t[1], t[0]))
    assert np.isclose(cars.speed[0], expert.target_speed(circle_track, np.array([40]))[0])
    assert not cars.crashed[0] and not cars.done[0]


def test_collect_rows_is_deterministic(circle_track):
    first = expert.collect_rows(circle_track, np.random.default_rng(7))
    second = expert.collect_rows(circle_track, np.random.default_rng(7))
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1]) and first[2] == second[2]


def test_labels_are_the_expert_choice_during_kicks(monkeypatch, circle_track):
    real_step, expected, driven = sim.step, [], []

    def step(track, cars, steer, throttle, time_limit):
        expected.append(int(controls_to_class(*expert.expert_controls(track, cars))[0]))
        driven.append(float(steer[0]))
        real_step(track, cars, steer, throttle, time_limit)

    monkeypatch.setattr(sim, "step", step)
    _, y, _ = expert.collect_rows(circle_track, np.random.default_rng(0))
    assert list(y) == expected
    label_steer = class_to_controls(y)[0]
    assert np.any(label_steer != np.array(driven))                # some steps were driven by a kick


@pytest.mark.slow
def test_the_first_training_maps_of_every_level_build_at_their_widths():
    for level in config.LEVELS:
        for seed in range(config.TRAIN_SEED_START, config.TRAIN_SEED_START + 200):
            tracks.get_track(seed, level)


@pytest.mark.slow
def test_expert_drives_every_dataset_and_held_out_map():
    for seed, level in expert.dataset_seeds():
        tracks.get_track(seed, level)
    for seed in list(config.VAL_SEEDS) + list(config.TEST_SEEDS):
        tracks.get_track(seed, config.MAX_LEVEL)


def test_dataset_maps_are_fresh_distinct_seeds_with_the_level_mix():
    items = expert.dataset_seeds()
    assert len(items) == sum(config.DATASET_TRACKS.values()) == 121
    for level, count in config.DATASET_TRACKS.items():
        assert sum(1 for _, map_level in items if map_level == level) == count
    seeds = [seed for seed, _ in items]
    low, high = config.DATASET_SEED_RANGE
    assert len(set(seeds)) == len(seeds) and all(low <= seed < high for seed in seeds)
    assert high <= config.TRAIN_SEED_START                      # never a training map of an evolutionary run
    assert not set(seeds) & (set(config.VAL_SEEDS) | set(config.TEST_SEEDS))
    assert expert.dataset_seeds() == items                       # drawn once: the same maps every time


def test_default_splits_use_the_dataset_maps_and_the_fixed_held_out_maps():
    splits = expert.default_splits()
    assert splits["train"] == expert.dataset_seeds()
    assert splits["val"] == [(seed, config.MAX_LEVEL) for seed in config.VAL_SEEDS]
    assert splits["test"] == [(seed, config.MAX_LEVEL) for seed in config.TEST_SEEDS]


SMALL_SPLITS = {"train": [(0, 1)], "val": [(0, 1)], "test": [(0, 1)]}


def test_a_complete_dataset_is_not_collected_again(monkeypatch, tmp_path):
    first, summary = expert.collect_dataset(tmp_path / "d.npz", splits=SMALL_SPLITS)
    monkeypatch.setattr(expert, "_collect_track", lambda seed, level: pytest.fail("collected again"))
    second, again = expert.collect_dataset(tmp_path / "d.npz", splits=SMALL_SPLITS)
    assert again == summary and set(first) == set(second)
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_other_maps_or_no_resume_collect_again(monkeypatch, tmp_path):
    expert.collect_dataset(tmp_path / "d.npz", splits=SMALL_SPLITS)
    real, calls = expert._collect_track, []
    monkeypatch.setattr(expert, "_collect_track", lambda seed, level: calls.append((seed, level)) or real(seed, level))
    other = {**SMALL_SPLITS, "train": [(1, 1)]}
    expert.collect_dataset(tmp_path / "d.npz", splits=other)                  # other maps: collected
    expert.collect_dataset(tmp_path / "d.npz", splits=other, resume=False)    # no resume: collected again
    assert len(calls) == 6


def test_a_dataset_collected_with_other_settings_is_refused(monkeypatch, tmp_path):
    expert.collect_dataset(tmp_path / "d.npz", splits=SMALL_SPLITS)
    monkeypatch.setattr(checkpoint, "settings_record", lambda approach=None: {"TARGET_FACTOR": 9.9})
    with pytest.raises(checkpoint.SettingsMismatch):
        expert.collect_dataset(tmp_path / "d.npz", splits=SMALL_SPLITS)


def test_dataset_settings_are_recorded():
    assert {"DATASET_SEED_RANGE", "DATASET_TRACKS", "DATASET_SALT"} <= set(checkpoint.settings_record())


def test_parallel_collection_matches_serial(tmp_path):
    splits = {"train": [(0, 1), (1, 2)], "val": [(80, 2)], "test": [(1000, 2)]}
    serial, _ = expert.collect_dataset(tmp_path / "a.npz", splits=splits)
    parallel, _ = expert.collect_dataset(tmp_path / "b.npz", splits=splits, n_jobs=2, backend="threading")
    assert set(serial) == set(parallel)
    for key in serial:
        assert np.array_equal(serial[key], parallel[key])


def test_collection_time_is_recorded(tmp_path):
    _, summary = expert.collect_dataset(tmp_path / "d.npz", splits={"train": [(0, 1)], "val": [(0, 1)],
                                                                    "test": [(0, 1)]})
    info = json.loads((tmp_path / "d_info.json").read_text())
    assert info["wall_time"] > 0 and info["n_jobs"] == 1 and info["splits"] == summary


def test_the_dataset_record_is_synced_to_disk(monkeypatch, tmp_path):
    written, real = [], checkpoint.write_json
    monkeypatch.setattr(checkpoint, "write_json", lambda path, data: written.append(path.name) or real(path, data))
    expert.collect_dataset(tmp_path / "d.npz", splits=SMALL_SPLITS)
    assert written == ["d_info.json"]
