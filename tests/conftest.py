import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pytest

from racing.track import build_track


SHAPED_ARCHITECTURES = [(), (5,), (8, 3), (1,), (), (4, 4, 2)]


def spread_setups(rng, n):
    """n random-start setups (spec §9.11) whose sensor lengths are spread log-evenly over SENSOR_LENGTH_LIMITS, in a
    random order: a random start gives every car the same starting length, so tests that need cars of different
    lengths, such as the simulators' equality checks over every ray-iteration cap (spec §5.2, §5.7), use these."""
    from racing import config
    from racing import setup as car_setup
    rows = car_setup.random(rng, n)
    rows[:, 0] = rng.permutation(np.geomspace(*config.SENSOR_LENGTH_LIMITS, n))
    return rows


def make_shaped_cars(n, seed=0, shapes=SHAPED_ARCHITECTURES):
    """Random networks of mixed architectures (some without hidden layers) and random setups of spread sensor
    lengths."""
    from racing import nets
    rng = np.random.default_rng(seed)
    chosen = [shapes[i % len(shapes)] for i in range(n)]
    weights = [nets.init_population(rng, 1, shape)[0] + rng.normal(0.0, 0.3, nets.n_params(shape)) for shape in chosen]
    return nets.ShapedPopulation(chosen, weights, "relu", spread_setups(rng, n))


@pytest.fixture
def shaped_cars():
    """make_shaped_cars, for tests that need populations of mixed architectures."""
    return make_shaped_cars


def stadium(straight=300.0, radius=60.0, width=12.0):
    """Two straights joined by semicircles, counter-clockwise from the start of the bottom straight."""
    s = np.arange(0.0, straight, 0.5)
    a = np.arange(-np.pi / 2, np.pi / 2, 0.5 / radius)
    points = np.vstack([np.column_stack([s - straight / 2, np.full(len(s), -radius)]),
                        np.column_stack([straight / 2 + radius * np.cos(a), radius * np.sin(a)]),
                        np.column_stack([straight / 2 - s, np.full(len(s), radius)]),
                        np.column_stack([-straight / 2 - radius * np.cos(a), -radius * np.sin(a)])])
    track = build_track(points, width / 2, width / 2, seed=0, level=4)
    track.expert_lap_time = 60.0
    return track


@pytest.fixture
def circle_track():
    """A counter-clockwise circle (radius 50 m, width 16 m) with a fixed expert lap time, built without the generator."""
    a = np.linspace(-np.pi / 2, 1.5 * np.pi, 3200, endpoint=False)
    track = build_track(np.column_stack([50 * np.cos(a), 50 * np.sin(a)]), 8.0, 8.0, seed=0, level=1)
    track.expert_lap_time = 20.0
    return track


@pytest.fixture(autouse=True)
def private_progress_files(tmp_path_factory, monkeypatch):
    """The status file and the run log of every test live in a temporary folder, never in results/
    (spec §3.3)."""
    from racing import config
    folder = tmp_path_factory.mktemp("progress")
    monkeypatch.setattr(config, "STATUS_PATH", folder / "status.json")
    monkeypatch.setattr(config, "RUN_LOG_PATH", folder / "run_log.txt")
    monkeypatch.setattr(config, "STATUS_DIR", folder / "status")
    monkeypatch.setattr(config, "TRAIN_LOGS_DIR", folder / "train_logs")


@pytest.fixture(autouse=True)
def private_app_runs(tmp_path_factory, monkeypatch):
    """Viewer runs and their app cars of every test live in a temporary folder, never in results/
    (spec §10.6)."""
    from racing import config
    folder = tmp_path_factory.mktemp("app_runs")
    monkeypatch.setattr(config, "APP_RUNS_DIR", folder)
    monkeypatch.setattr(config, "APP_CHAMPIONS_DIR", folder / "champions")


@pytest.fixture
def numpy_simulator(monkeypatch):
    """Forces the numpy simulator, for tests that watch sim.simulate or the joblib task division (spec §5.7)."""
    from racing import config
    monkeypatch.setattr(config, "SIMULATOR", "numpy")


@pytest.fixture(autouse=True)
def normal_priority(monkeypatch):
    """An evolution phase lowers its process's priority (spec §3.3); tests keep theirs, so the rest of the suite runs as
    usual. tests/test_evolution.py checks the real function in a fresh process."""
    from racing import evolution
    monkeypatch.setattr(evolution, "lower_priority", lambda nice=None: None)
