import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pytest

from racing.track import build_track


@pytest.fixture
def circle_track():
    """A counter-clockwise circle (radius 50 m, width 16 m) with a fixed expert lap time, built without the generator."""
    a = np.linspace(-np.pi / 2, 1.5 * np.pi, 3200, endpoint=False)
    track = build_track(np.column_stack([50 * np.cos(a), 50 * np.sin(a)]), 16.0, seed=0, level=1)
    track.expert_lap_time = 20.0
    return track


@pytest.fixture(autouse=True)
def private_progress_files(tmp_path_factory, monkeypatch):
    """The status file and the run log of every test live in a temporary folder, never in results/
    (revision 3 §8.3)."""
    from racing import config
    folder = tmp_path_factory.mktemp("progress")
    monkeypatch.setattr(config, "STATUS_PATH", folder / "status.json")
    monkeypatch.setattr(config, "RUN_LOG_PATH", folder / "run_log.txt")


@pytest.fixture(autouse=True)
def private_app_runs(tmp_path_factory, monkeypatch):
    """Viewer runs and their app cars of every test live in a temporary folder, never in results/
    (revision 3.1 §6.1)."""
    from racing import config
    folder = tmp_path_factory.mktemp("app_runs")
    monkeypatch.setattr(config, "APP_RUNS_DIR", folder)
    monkeypatch.setattr(config, "APP_CHAMPIONS_DIR", folder / "champions")
