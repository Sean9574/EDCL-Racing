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
