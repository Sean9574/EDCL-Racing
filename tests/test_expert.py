import json
import numpy as np
import pytest

from racing import checkpoint, circuits, config, expert, fastsim, sim, tracks
from racing.actions import class_to_controls, controls_to_class
from racing.track import build_track


def test_expert_finishes_the_circle(circle_track):
    lap = expert.expert_lap_time(circle_track)
    corner_speed = expert.corner_speed(1 / 50.0)                 # the circle's radius is 50 m
    assert lap is not None
    fastest = corner_speed + config.EXPERT_SPEED_BAND
    assert circle_track.length / fastest < lap < 2.5 * circle_track.length / corner_speed


def test_the_corner_speed_uses_the_grip_at_that_speed():
    """Spec §7.1: the highest v with v² κ ≤ 0.85 a(v)."""
    kappa = np.array([1 / 15.0, 1 / 60.0, 1 / 150.0, 1 / 400.0, 1 / 5000.0])
    v = expert.corner_speed(kappa)
    margin = config.EXPERT_GRIP_MARGIN
    bound = kappa < 1 / 300.0                        # gentle curves: the ceiling or nothing binds
    assert np.allclose((v * v * kappa)[~bound], margin * sim.grip_at(v)[~bound], rtol=1e-12)
    assert np.all(v * v * kappa <= margin * sim.grip_at(v) * (1 + 1e-12))
    assert v[-1] > config.TOP_SPEED                  # a 5 km radius does not limit the car


def test_the_sight_speed_stops_within_150_m():
    assert config.EXPERT_SIGHT == config.SENSOR_START == 150.0
    v = expert.sight_speed()
    assert 97.5 < v < 98.5                           # about 98 m/s (353 km/h), spec §7.1
    speeds = np.linspace(v, 0.0, 200001)             # braking at 0.85 a(v) from v covers exactly the sight
    rate = config.EXPERT_GRIP_MARGIN * sim.grip_at(speeds)
    distance = np.trapezoid(speeds / rate, -speeds) if hasattr(np, "trapezoid") else np.trapz(speeds / rate, -speeds)
    assert np.isclose(distance, 150.0, rtol=1e-4)


def test_the_target_speed_never_exceeds_the_sight_speed_or_the_top_speed(circle_track):
    here = np.arange(len(circle_track.centerline))
    target = expert.target_speed(circle_track, here)
    assert np.all(target <= expert.sight_speed()) and np.all(target <= config.TOP_SPEED)
    assert np.allclose(target, expert.corner_speed(np.abs(circle_track.curvature).max()), rtol=1e-6)


def circle(radius, width=14.0):
    """A counter-clockwise circle of the given radius: one steady corner all the way round."""
    a = np.linspace(-np.pi / 2, 1.5 * np.pi, int(2 * np.pi * radius / 0.5), endpoint=False)
    return build_track(np.column_stack([radius * np.cos(a), radius * np.sin(a)]), width / 2, width / 2, seed=0,
                       level=1)


@pytest.mark.parametrize("radius", [50.0, 80.0, 214.0, 400.0])
def test_the_expert_never_switches_between_full_throttle_and_full_brake_in_a_steady_corner(radius):
    """Spec §7.1: in a steady corner the expert settles at its target speed, its throttle proportional to the speed
    error within the band. One step's speed change is a(v)·DT, up to 2.45 m/s at 5 g, more than the 0.5 m/s band, so
    a band of 0.5 m/s alone switched between gas and brake on every step and left only 1/√2 of a(v) for turning (on circles of radius 214 m, the ends of a 420 × 300 m
    ellipse, and 400 m, near 5 g; and of 50 m, near 1.8 g)."""
    track = circle(radius)
    throttles, policy = [], expert.expert_policy(track)

    def recording(obs, cars):
        classes = policy(obs, cars)
        throttles.append(float(class_to_controls(classes)[1][0]))
        return classes

    result = sim.simulate(track, recording, 1, time_limit=config.EXPERT_VALIDATION_TIME, laps=config.LAPS)
    throttles = np.array(throttles)
    assert result.finished[0] and not np.any(throttles[1:] * throttles[:-1] == -1.0)


def test_the_expert_never_applies_gas_at_full_lock(circle_track):
    """Spec §7.1: while its steering is saturated the expert's throttle is at most 0, so the friction circle
    leaves all of a(v) for turning; below its target speed with the steering short of full lock it still applies gas."""
    cars = sim.start_cars(circle_track, 2)
    cars.speed[:] = 20.0                                   # well below the 50 m circle's corner speed of about 32 m/s
    cars.heading[0] += 1.0                                 # pointing off the line: the steering saturates
    steer, throttle = expert.expert_controls(circle_track, cars)
    assert np.all(cars.speed < expert.target_speed(circle_track, cars.index) - 1.5)
    assert abs(steer[0]) == 1.0 and throttle[0] == 0.0
    assert abs(steer[1]) < 1.0 and throttle[1] == 1.0
    cars.speed[0] = 40.0                                   # above the target: it still brakes at full lock
    assert expert.expert_controls(circle_track, cars)[1][0] == -1.0


def test_the_expert_throttle_is_its_speed_error_over_its_band():
    """Spec §7.1 (version 3.9): throttle = clip((target − speed) / band, −1, 1), quantized to the five levels; never
    above 0 at full lock."""
    track = circle(400.0)
    cars = sim.start_cars(track, 6)
    target = expert.target_speed(track, np.mod(cars.index, len(track.centerline)))
    band = np.maximum(config.EXPERT_SPEED_BAND, sim.grip_at(target) * config.DT)
    cars.speed[:] = target - band * np.array([2.0, 0.6, 0.1, -0.1, -0.6, -2.0])
    steer, throttle = expert.expert_controls(track, cars)
    assert np.all(np.abs(steer) < 1.0)
    band_now = np.maximum(config.EXPERT_SPEED_BAND, sim.grip_at(cars.speed) * config.DT)
    assert np.allclose(throttle, np.clip((target - cars.speed) / band_now, -1.0, 1.0))
    levels = class_to_controls(controls_to_class(steer, throttle))[1]
    assert levels.tolist() == [1.0, 0.5, 0.0, 0.0, -0.5, -1.0]


def test_the_expert_leaves_montreals_hairpin_without_gas_at_full_lock(monkeypatch):
    """Spec §7.1, §12.1: in Montreal's racing direction the expert left a 30 m right-hander below its target speed at
    full gas and full lock, which left 0.71 of a(v) for turning where the following curve needed about 0.83, and drifted
    off the road after 48 s. Without gas at full lock it completes the lap, alike in both simulators."""
    track = circuits.circuit_track("Montreal")
    real, saturated_below_target = expert.expert_controls, []

    def recording(track_, cars):
        steer, throttle = real(track_, cars)
        full_lock = np.abs(steer) >= 1.0
        assert not np.any(full_lock & (throttle > 0.0))
        band = np.maximum(config.EXPERT_SPEED_BAND, sim.grip_at(cars.speed) * config.DT)
        below = cars.speed < expert.target_speed(track_, np.mod(cars.index, len(track_.centerline))) - band
        saturated_below_target.append(bool(np.any(full_lock & below)))
        return steer, throttle

    monkeypatch.setattr(expert, "expert_controls", recording)
    reference = sim.simulate(track, expert.expert_policy(track), 1, time_limit=config.EXPERT_VALIDATION_TIME)
    assert reference.finished[0] and 95.0 < reference.finish_time[0] < 110.0
    assert sum(saturated_below_target) > 0                 # the rule acts on this lap
    monkeypatch.setattr(config, "SIMULATOR", "auto")
    if fastsim.available():
        compiled = fastsim.simulate_expert([track], 1, [config.EXPERT_VALIDATION_TIME])[0]
        for field in ("fitness", "finished", "crashed", "progress", "laps_done", "steps"):
            assert np.array_equal(getattr(compiled, field), getattr(reference, field)), field
        assert np.array_equal(compiled.finish_time, reference.finish_time, equal_nan=True)
        assert np.array_equal(compiled.lap_times, reference.lap_times, equal_nan=True)


def test_expert_policy_returns_action_classes(circle_track):
    cars = sim.start_cars(circle_track, 4)
    classes = expert.expert_policy(circle_track)(sim.sense(circle_track, cars), cars)
    assert classes.shape == (4,) and classes.min() >= 0 and classes.max() < config.N_ACTIONS


def test_get_track_attaches_lap_time_and_caches():
    t = tracks.get_track(0, 1)
    assert np.isfinite(t.expert_lap_time) and t.expert_lap_time > 0
    assert tracks.get_track(0, 1) is t


@pytest.mark.parametrize("level", [1, 2, 3])
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
    for level in range(1, config.MAX_LEVEL):                  # the generated levels; Extra hard is the real layouts
        for seed in range(config.TRAIN_SEED_START, config.TRAIN_SEED_START + 200):
            tracks.get_track(seed, level)


@pytest.mark.slow
def test_the_expert_drives_every_validation_map_and_the_first_maps_of_each_level():
    for seed in config.VAL_SEEDS:
        assert tracks.get_track(seed, config.VALIDATION_LEVEL).expert_lap_time > 0
    for level in (1, 2, 3):
        for seed in range(config.TRAIN_SEED_START, config.TRAIN_SEED_START + 5):
            assert tracks.get_track(seed, level).expert_lap_time > 0


@pytest.mark.slow
def test_expert_completes_every_level_with_the_half_levels():
    """Spec §7.1, §12.1: with proportional throttle the expert still completes generated maps of every level and the
    validation setting (the real layouts are covered by the existing slow test)."""
    for level, seeds in ((1, (2_000_000, 2_000_001)), (2, (2_000_000, 2_000_001)), (3, (2_000_000, 2_000_001)),
                         (config.VALIDATION_LEVEL, config.VAL_SEEDS[:2])):
        for seed in seeds:
            assert expert.expert_lap_time(tracks.get_track(seed, level)) is not None, (seed, level)


@pytest.mark.slow
def test_expert_drives_every_dataset_and_held_out_map():
    for split in expert.default_splits().values():       # the validation maps at the validation setting (spec §4.6)
        for seed, level in split:
            tracks.get_track(seed, level)


def test_dataset_maps_are_fresh_distinct_seeds_with_the_level_mix_and_the_real_layouts():
    """Spec §4.6, §7.3: 41 generated maps (1 Easy, 20 Medium, 20 Hard), then the 38 real training layouts."""
    items = expert.dataset_seeds()
    generated = items[:-circuits.N_LAYOUTS]
    assert len(generated) == sum(config.DATASET_TRACKS.values()) == 41 and len(items) == 79
    assert items[-circuits.N_LAYOUTS:] == [(k, config.MAX_LEVEL) for k in range(circuits.N_LAYOUTS)]
    for level, count in config.DATASET_TRACKS.items():
        assert sum(1 for _, map_level in generated if map_level == level) == count
    assert config.DATASET_TRACKS == {1: 1, 2: 20, 3: 20}
    seeds = [seed for seed, _ in generated]
    low, high = config.DATASET_SEED_RANGE
    assert len(set(seeds)) == len(seeds) and all(low <= seed < high for seed in seeds)
    assert high <= config.TRAIN_SEED_START                      # never a training map of an evolutionary run
    assert not set(seeds) & (set(config.VAL_SEEDS) | set(config.DISPLAY_SEEDS))
    assert expert.dataset_seeds() == items                       # drawn once: the same maps every time


def test_default_splits_use_the_dataset_maps_and_the_fixed_held_out_maps():
    splits = expert.default_splits()
    assert splits["train"] == expert.dataset_seeds()
    assert splits["val"] == [(seed, config.VALIDATION_LEVEL) for seed in config.VAL_SEEDS]
    assert splits["test"] == [(k, config.TEST_LEVEL) for k in config.TEST_TRACKS]


def test_no_test_circuit_is_in_any_training_or_validation_map():
    splits = expert.default_splits()
    assert [item for item in splits["train"] if item[1] == config.TEST_LEVEL] == []
    assert sorted(k for k, level in splits["train"] if level == config.MAX_LEVEL) == list(range(38))
    assert splits["test"] == [(k, config.TEST_LEVEL) for k in config.TEST_TRACKS]
    names = {circuits.layout(k)[0] for k in range(38)}
    assert not names & set(config.TEST_CIRCUITS)


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
