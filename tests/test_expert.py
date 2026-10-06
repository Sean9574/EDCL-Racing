import numpy as np
import pytest

from racing import config, expert, sim, tracks


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
    x, y = expert.collect_rows(circle_track, np.random.default_rng(0))
    assert x.shape[1] == config.N_INPUTS and len(x) == len(y) > 100
    assert y.min() >= 0 and y.max() < config.N_ACTIONS
    assert len(np.unique(y)) >= 3          # disturbances produce corrections, not one repeated action


def test_collect_dataset_writes_all_splits(tmp_path):
    path = tmp_path / "data.npz"
    data = expert.collect_dataset(path, splits={"train": [(0, 1)], "val": [(0, 2)], "test": [(1, 2)]})
    assert path.exists()
    for split in ("train", "val", "test"):
        assert data[f"X_{split}"].dtype == np.float32 and data[f"y_{split}"].dtype == np.int64
        assert len(data[f"X_{split}"]) == len(data[f"y_{split}"]) > 0


@pytest.mark.slow
def test_expert_drives_every_seed():
    for level, seeds in config.TRAIN_DATA_TRACKS.items():
        for seed in seeds:
            tracks.get_track(seed, level)
    for seed in config.TRAIN_SEEDS:
        tracks.get_track(seed, 2)
        tracks.get_track(seed, 3)
    for seed in list(config.VAL_SEEDS) + list(config.TEST_SEEDS):
        tracks.get_track(seed, 4)
