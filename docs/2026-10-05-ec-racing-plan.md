# EC Racing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the EC Racing simulator, six controller-training approaches, the evaluation pipeline and the pygame viewer described in the design specification.

**Architecture:** A vectorized numpy simulator (`sim.py`) on procedurally generated tracks is shared by every component. Four controllers are trained by imitation of a scripted expert: a baseline and three Keras Tuner searches. Two are evolved by driving: a GA and a (μ+λ)-ES. All controllers share one 10-input, 15-class network interface and are stored as `.keras` files. A pygame application shows live training and races; notebooks run the reported experiments.

**Tech Stack:** Python 3.13 (Conda base), numpy 2.4, scipy 1.17, TensorFlow 2.21 / Keras 3.13, keras-tuner 1.4, pygame 2.6, joblib, pandas, matplotlib, pytest.

**Spec:** `docs/2026-10-05-ec-racing-design.md`

## Global Constraints

- Interpreter: `~/miniconda3/bin/python` (Conda base, Python 3.13.11). Every command below uses `PY=~/miniconda3/bin/python` and runs from the project root `~/Desktop/Classes/ECDL/Project`.
- Installed packages only, plus `keras-tuner`, `pygame` and `pytest` (installed in Task 1). Do not install, upgrade or reinstall TensorFlow, Keras or numpy.
- CPU only. All Keras access goes through `racing.backend.keras_cpu()`, which hides GPUs.
- Every numeric default lives in `racing/config.py`. No magic numbers elsewhere.
- Simulation uses float64 and contains no randomness inside `sim.step`. A car's result never depends on which other cars are in the batch.
- Cars collide only with track boundaries, never with each other.
- Grip is the only vehicle limit: μg = 12 m/s². There is no top speed, no steering-angle limit and no fixed acceleration rate.
- Colors: Baseline `#7F7F7F`, KT Random `#E69F00`, KT Hyperband `#56B4E9`, KT Bayesian `#CC79A7`, ES `#009E73`, GA `#D55E00`.
- Seeds: training 0–79, validation 80–99, test 1000–1019 (level 4). Test seeds are never used in training or model selection.
- Code style: plain loops and explicit conditionals, short docstrings, sparse comments. No personal names in code or documents.
- Generated data goes under `results/`, which is gitignored.
- Tests: `$PY -m pytest <test file> -q`. Tests marked `slow` run only when named explicitly or with `-m slow`.
- Commits: stage and commit only the task's own files with `git add <files> && git commit -m "<message>" -- <files>`. If git reports `index.lock`, wait two seconds and retry. Every commit message ends with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **A car that never moves.** Example: it brakes or coasts from rest. It must stop at the 5 s no-progress timeout with fitness 0, not run until the time limit. The test is in Task 3.
2. **Driving backwards from the start line, or across it.** This must never earn progress or count a lap. The test is in Task 3.
3. **Impossible population settings.** Examples: a population no larger than the kept count, a budget smaller than one generation, or a cars-shown value above the population. These must raise a clear `ValueError` or be clamped, never fail mid-run. The tests are in Tasks 8, 9 and 12.
4. **Race view with no trained controllers, or only some.** It must show a message, or race the available controllers, without crashing. The tests are in Tasks 4 and 12.
5. **Training process exits or is closed mid-run.** The viewer must report the stop instead of hanging. Logs written so far must stay readable. The tests are in Tasks 7 and 12.

## Execution Waves

At most 10 agents run at once. Implementers work in the shared working tree, touch only the files listed in their task, and run only their own test files. The only shared file that may change is `racing/config.py`, and only where a task says so. Reviewers check each task after it reports done.

| Wave | Tasks (run in parallel) | Depends on |
|---|---|---|
| 0 | Task 1 (orchestrator) | — |
| 1 | Tasks 2, 3, 4, 5 | Task 1 |
| 2 | Tasks 6, 7 | Tasks 2, 3, 4 (Task 7 tests replace `get_track` with a fixture) |
| 3 | Tasks 8, 9, 10, 11, 12 | Tasks 5, 6, 7 |
| 4 | Task 13 | All |

## File Structure

| File | Responsibility |
|---|---|
| `racing/config.py` | All default parameters, colors, paths |
| `racing/actions.py` | Action table: class ↔ (steering fraction, throttle fraction) |
| `racing/backend.py` | Keras access with GPUs hidden |
| `racing/track.py` | `Track` data structure, geometry, road grid, wall-distance lookup |
| `racing/curriculum.py` | Level-advancement rule, GA mutation-strength rule, per-generation track draw |
| `racing/track_gen.py` | Curriculum level shapes, geometric validation |
| `racing/sim.py` | Vehicle dynamics, sensors, collisions, progress, fitness, batch simulation |
| `racing/nets.py` | Flat population weights, numpy forward pass, `DenseNet`, Keras conversion, champion loading |
| `racing/expert.py` | Scripted expert driver, dataset collection |
| `racing/tracks.py` | Validated tracks with expert lap time (cached; imports deferred to avoid a cycle) |
| `racing/population.py` | Shared evolutionary helpers: evaluation, reports, logging, final selection |
| `racing/ga.py` | Genetic algorithm |
| `racing/es.py` | (μ+λ) evolution strategy |
| `racing/imitation.py` | Baseline training, Keras Tuner searches, model selection |
| `racing/evaluate.py` | Test-track driving, accuracy, inference time, tables, figures |
| `racing/app.py` | pygame viewer: training view and race view |
| `racing/bench.py` | Generation-time measurement (`python -m racing.bench`) |
| `scripts/make_notebooks.py` | Writes the five experiment notebooks |
| `notebooks/01_dataset.ipynb` … `05_evaluate.ipynb` | Reported experiments |
| `tests/…` | One test file per module |

---

### Task 1: Scaffold and shared foundations (orchestrator)

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `racing/__init__.py`, `racing/config.py`, `racing/actions.py`, `racing/backend.py`, `racing/track.py`, `racing/curriculum.py`, `racing/tracks.py`
- Test: `tests/conftest.py`, `tests/test_track.py`, `tests/test_actions.py`, `tests/test_curriculum.py`

**Interfaces:**
- Produces:
  - `config.*` constants
  - `actions.class_to_controls(classes) -> (steer, throttle)`
  - `actions.controls_to_class(steer, throttle) -> classes`
  - `backend.keras_cpu() -> keras`
  - `track.Track`
  - `track.resample_loop(points, spacing) -> (points, spacing)`
  - `track.build_track(points, width, seed, level) -> Track`
  - `track.wall_distance(track, x, y) -> ndarray`
  - `curriculum.mutation_strength(new_finish) -> float`
  - `curriculum.LevelTracker`
  - `curriculum.draw_seeds(rng) -> list[int]`
  - `tracks.get_track(seed, level) -> Track`, cached and with `expert_lap_time` set. It is tested in Task 6.
  - the `circle_track` pytest fixture

- [ ] **Step 1: Install dependencies**

```bash
PY=~/miniconda3/bin/python
$PY -m pip install keras-tuner pygame pytest
$PY -c "import keras_tuner, pygame, pytest, joblib; print('ok')"
```

Expected: `ok`. The dry run showed this installs only keras-tuner, kt-legacy, pygame, pytest and iniconfig.

- [ ] **Step 2: Write `pyproject.toml` and `requirements.txt`**

```toml
[project]
name = "ec-racing"
version = "1.0.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = ["slow: tests that take more than about ten seconds"]
addopts = "-m 'not slow'"
```

```
numpy
scipy
tensorflow
keras
keras-tuner
pygame
pytest
pandas
matplotlib
joblib
```

- [ ] **Step 3: Write `racing/__init__.py` and `racing/config.py`**

```python
"""EC Racing: evolutionary and gradient-based search for neural network driving controllers."""
```

```python
"""Default parameters of EC Racing (design specification §13). Every module reads its constants from here."""
from pathlib import Path

import numpy as np

# --- Simulation (spec §5) ---
DT = 0.05                    # s, fixed time step (20 Hz)
GRIP = 12.0                  # m/s², friction-circle limit μg; the only vehicle limit
YAW_MIN_SPEED = 1.0          # m/s, speed floor used only in the yaw-rate formula
SPEED_SCALE = 40.0           # m/s, input normalization only; not a speed limit
STALL_TIME = 5.0             # s without progress ends a run
TIME_LIMIT_FACTOR = 2.0      # lap time limit = factor × expert lap time
TIME_EPS = 1e-9              # tolerance for comparisons of accumulated simulation time

# --- Sensors (spec §5.2) ---
N_RAYS = 9
RAY_ANGLES = np.radians(np.linspace(-90.0, 90.0, N_RAYS))   # index 0 = right, 4 = ahead, 8 = left
RAY_RANGE = 50.0             # m
RAY_MIN_STEP = 0.5           # m, smallest sphere-tracing step
RAY_MAX_ITERATIONS = 48      # sphere-tracing iterations per ray at most

# --- Track geometry (spec §4) ---
CENTERLINE_SPACING = 1.0     # m
GRID_RES = 0.5               # m per road-grid cell
GRID_MARGIN = 10.0           # m of grass around the road in the grid
OVERLAP_MARGIN = 6.0         # m of grass required between separate parts of the road
MAX_TRACK_ATTEMPTS = 200
PROGRESS_BACK = 5.0          # m behind the last matched centerline point
PROGRESS_AHEAD = 30.0        # m ahead of the last matched centerline point
LEVELS = {
    1: {"kind": "circle", "radius": 50.0, "width": 16.0},
    2: {"kind": "oval", "straight": (80.0, 160.0), "radius": (25.0, 40.0), "width": 14.0},
    3: {"kind": "loop", "points": (6, 8), "radius": (80.0, 160.0), "min_corner": 40.0, "width": 14.0},
    4: {"kind": "loop", "points": (10, 16), "radius": (60.0, 200.0), "min_corner": 15.0, "width": 12.0},
}
MAX_LEVEL = 4

# --- Network interface (spec §6) ---
N_INPUTS = N_RAYS + 1
STEER_LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
THROTTLE_LEVELS = np.array([-1.0, 0.0, 1.0])               # brake, coast, gas
N_ACTIONS = len(STEER_LEVELS) * len(THROTTLE_LEVELS)       # 15
BASELINE_HIDDEN = (16,)

# --- Expert driver and dataset (spec §7) ---
EXPERT_LOOKAHEAD_BASE = 6.0  # m
EXPERT_LOOKAHEAD_TIME = 0.4  # s
EXPERT_GRIP_MARGIN = 0.85
EXPERT_SPEED_BAND = 0.5      # m/s
EXPERT_SIGHT = RAY_RANGE     # m, the expert uses no geometry beyond sensor range
EXPERT_VALIDATION_TIME = 600.0  # s, time limit when validating a track with the expert
PERTURB_INTERVAL = (2.0, 4.0)   # s between steering disturbances
PERTURB_DURATION = (0.3, 0.6)   # s per disturbance
DATASET_SALT = 99

# --- Seeds (spec §4.4) ---
TRAIN_SEEDS = range(0, 80)
VAL_SEEDS = range(80, 100)
TEST_SEEDS = range(1000, 1020)
TRAIN_DATA_TRACKS = {1: (0,), 2: tuple(range(20)), 3: tuple(range(20)), 4: tuple(TRAIN_SEEDS)}

# --- Curriculum (spec §9.1) ---
TRACKS_PER_GEN = 3
LEVEL_UP_NEW_FRACTION = 0.8
LEVEL_UP_STREAK = 3

# --- Genetic algorithm (spec §9.2) ---
GA_POP = 1000
GA_KEPT = 50
GA_TOURNAMENT = 5
GA_CROSSOVER_P = 0.5
GA_MUTATION_P = 0.1
GA_S_MAX = 0.5
GA_S_MIN = 0.005

# --- Evolution strategy (spec §9.3) ---
ES_MU = 50
ES_LAMBDA = 500
ES_SIGMA0 = 0.1
ES_FACTOR = 0.82
ES_SIGMA_MIN = 1e-4
ES_SIGMA_MAX = 1.0

# --- Budget and parallelism ---
EVAL_BUDGET = 200_000
N_JOBS = 1                   # worker processes for population evaluation; set from the Task 11 benchmark (racing/bench.py)

# --- Keras training (spec §8) ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 5
KT_TRIALS = 30
HYPERBAND_FACTOR = 3
HYPERBAND_ITERATIONS = 2
SIZE_RULE_TOLERANCE = 0.01

# --- Approaches and colors (spec §10.1) ---
APPROACHES = ("baseline", "kt_random", "kt_hyperband", "kt_bayesian", "es", "ga")
LABELS = {"baseline": "Baseline", "kt_random": "KT Random", "kt_hyperband": "KT Hyperband",
          "kt_bayesian": "KT Bayesian", "es": "ES", "ga": "GA"}
COLORS = {"baseline": "#7F7F7F", "kt_random": "#E69F00", "kt_hyperband": "#56B4E9",
          "kt_bayesian": "#CC79A7", "es": "#009E73", "ga": "#D55E00"}

# --- Application (spec §10) ---
WINDOW_SIZE = (1280, 800)
PANEL_WIDTH = 380
FPS = 60
POPULATION_PRESETS = (100, 250, 500, 1000, 2000)
SPEEDS = (1, 2, 4, 8)
DEFAULT_SHOWN = 100
KT_VIEW_SEED = 80
RACE_LAPS = 3

# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
DATASET_PATH = RESULTS_DIR / "dataset.npz"
RUNS_DIR = RESULTS_DIR / "runs"
APP_RUNS_DIR = RESULTS_DIR / "app_runs"
CHAMPIONS_DIR = RESULTS_DIR / "champions"
```

- [ ] **Step 4: Write `racing/actions.py` and `racing/backend.py`**

```python
"""Action table (spec §6): the 15 output classes and the grip fractions they stand for."""
import numpy as np

from racing import config

N_THROTTLE = len(config.THROTTLE_LEVELS)


def class_to_controls(classes):
    """Steering and throttle grip fractions for an array of action classes."""
    classes = np.asarray(classes, dtype=np.int64)
    return config.STEER_LEVELS[classes // N_THROTTLE], config.THROTTLE_LEVELS[classes % N_THROTTLE]


def controls_to_class(steer, throttle):
    """Nearest action class for arrays of continuous steering and throttle fractions."""
    steer = np.atleast_1d(np.asarray(steer, dtype=float))
    throttle = np.atleast_1d(np.asarray(throttle, dtype=float))
    s = np.argmin(np.abs(steer[:, None] - config.STEER_LEVELS[None, :]), axis=1)
    t = np.argmin(np.abs(throttle[:, None] - config.THROTTLE_LEVELS[None, :]), axis=1)
    return s * N_THROTTLE + t
```

```python
"""Keras access with GPUs hidden: the networks are small enough that the CPU is faster."""


def keras_cpu():
    """Return the keras module after hiding GPUs from TensorFlow. Safe to call repeatedly."""
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass  # TensorFlow is already initialized in this process; visibility can no longer change
    import keras
    return keras
```

- [ ] **Step 5: Write `racing/track.py`**

```python
"""Track data structure and geometry shared by the generator, the simulator and the viewer (spec §4.1)."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from racing import config


@dataclass(eq=False)
class Track:
    """A closed-loop track. Cars drive in the direction of increasing arc length."""
    seed: int
    level: int
    width: float
    centerline: np.ndarray      # (M, 2), point 0 is the start line
    tangent: np.ndarray         # (M, 2) unit direction of travel
    curvature: np.ndarray       # (M,) signed curvature in 1/m, positive for left turns
    left: np.ndarray            # (M, 2) left road edge
    right: np.ndarray           # (M, 2) right road edge
    spacing: float              # m between centerline points
    length: float               # m, equal to M * spacing
    origin: np.ndarray          # (2,) world position of the road grid's lower-left corner
    dist: np.ndarray            # (rows, cols) m from each road cell to the nearest grass cell; 0 on grass
    expert_lap_time: float = float("nan")

    @property
    def start_position(self):
        return self.centerline[0].copy()

    @property
    def start_heading(self):
        return float(np.arctan2(self.tangent[0, 1], self.tangent[0, 0]))


def resample_loop(points, spacing):
    """Resample a closed polyline (first point not repeated) to equal spacing. Returns (points, actual spacing)."""
    closed = np.vstack([points, points[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    m = max(int(round(s[-1] / spacing)), 8)
    s_new = np.arange(m) * (s[-1] / m)
    return np.column_stack([np.interp(s_new, s, closed[:, 0]), np.interp(s_new, s, closed[:, 1])]), s[-1] / m


def build_track(points, width, seed, level):
    """Build a Track from a closed loop of points in driving order (any spacing, first point not repeated)."""
    center, spacing = resample_loop(np.asarray(points, dtype=float), config.CENTERLINE_SPACING)
    m = len(center)

    segment = np.roll(center, -1, axis=0) - center
    seg_heading = np.arctan2(segment[:, 1], segment[:, 0])
    turn = np.angle(np.exp(1j * (seg_heading - np.roll(seg_heading, 1))))
    curvature = turn / spacing
    curvature = sum(np.roll(curvature, k) for k in range(-2, 3)) / 5.0   # light smoothing over ±2 points

    chord = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent = chord / np.linalg.norm(chord, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])             # points left of travel
    left = center + normal * width / 2
    right = center - normal * width / 2

    origin = center.min(axis=0) - (width / 2 + config.GRID_MARGIN)
    top = center.max(axis=0) + (width / 2 + config.GRID_MARGIN)
    cols, rows = np.ceil((top - origin) / config.GRID_RES).astype(int)
    dense, _ = resample_loop(center, config.GRID_RES / 2)
    c = np.floor((dense[:, 0] - origin[0]) / config.GRID_RES).astype(int)
    r = np.floor((dense[:, 1] - origin[1]) / config.GRID_RES).astype(int)
    on_line = np.zeros((rows, cols), dtype=bool)
    on_line[r, c] = True
    road = distance_transform_edt(~on_line) * config.GRID_RES <= width / 2
    dist = np.where(road, distance_transform_edt(road) * config.GRID_RES - config.GRID_RES / 2, 0.0)

    return Track(seed=seed, level=level, width=float(width), centerline=center, tangent=tangent,
                 curvature=curvature, left=left, right=right, spacing=spacing, length=m * spacing,
                 origin=origin, dist=dist)


def wall_distance(track, x, y):
    """Distance in m from each point to the nearest grass cell; 0 on grass or outside the grid."""
    rows, cols = track.dist.shape
    c = np.floor((np.asarray(x) - track.origin[0]) / config.GRID_RES).astype(np.int64)
    r = np.floor((np.asarray(y) - track.origin[1]) / config.GRID_RES).astype(np.int64)
    inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    return np.where(inside, track.dist[np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)], 0.0)
```

- [ ] **Step 6: Write `racing/curriculum.py`**

```python
"""Curriculum rules (spec §9.1-9.2): level advancement, GA mutation strength, per-generation tracks."""
import math

import numpy as np

from racing import config


def mutation_strength(new_finish, s_max=config.GA_S_MAX, s_min=config.GA_S_MIN):
    """High when few new individuals finish; s_min when all of them do."""
    return s_min + (s_max - s_min) * (1.0 - new_finish)


class LevelTracker:
    """Advances one level after `streak_needed` consecutive qualifying generations."""

    def __init__(self, level=1, streak_needed=config.LEVEL_UP_STREAK,
                 new_threshold=config.LEVEL_UP_NEW_FRACTION, max_level=config.MAX_LEVEL):
        self.level = level
        self.streak = 0
        self.streak_needed = streak_needed
        self.new_threshold = new_threshold
        self.max_level = max_level

    def update(self, kept_finish, new_finish):
        """Record one generation's finishing fractions. Returns True if the level advanced."""
        if self.level >= self.max_level:
            return False
        qualifies = (not math.isnan(kept_finish)) and kept_finish >= 1.0 and new_finish >= self.new_threshold
        if qualifies:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.streak_needed:
            self.level += 1
            self.streak = 0
            return True
        return False


def draw_seeds(rng):
    """Training seeds for one generation's tracks, drawn without replacement."""
    seeds = rng.choice(np.array(list(config.TRAIN_SEEDS)), config.TRACKS_PER_GEN, replace=False)
    return [int(s) for s in seeds]
```

- [ ] **Step 6b: Write `racing/tracks.py`**

The generator and the expert are imported inside the function. This lets later modules import `get_track` before Tasks 2 and 6 exist, and avoids an import cycle with `expert.py`.

```python
"""Validated tracks (spec §4.3): generated geometry that the expert can drive, with its lap time."""
import functools

from racing import config


@functools.lru_cache(maxsize=None)
def get_track(seed, level):
    """The track for (seed, level). Cached; callers must not modify it."""
    from racing import expert, track_gen
    for attempt in range(config.MAX_TRACK_ATTEMPTS):
        track = track_gen.candidate_track(seed, level, attempt)
        if track is None:
            continue
        lap_time = expert.expert_lap_time(track)
        if lap_time is not None:
            track.expert_lap_time = lap_time
            return track
    raise RuntimeError(f"no drivable track for seed {seed}, level {level}")
```

- [ ] **Step 7: Write `tests/conftest.py`**

```python
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
```

- [ ] **Step 8: Write the tests**

`tests/test_track.py`:

```python
import numpy as np

from racing.track import resample_loop, wall_distance


def test_resample_loop_is_evenly_spaced():
    a = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    points, spacing = resample_loop(np.column_stack([10 * np.cos(a), 10 * np.sin(a)]), 1.0)
    gaps = np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)
    assert abs(spacing - 1.0) < 0.05
    assert np.allclose(gaps, gaps.mean(), rtol=0.05)


def test_circle_geometry(circle_track):
    t = circle_track
    assert abs(t.length - 2 * np.pi * 50) < 1.0
    assert np.allclose(t.curvature, 1 / 50, atol=0.002)                  # counter-clockwise: left turn
    assert np.allclose(np.linalg.norm(t.left, axis=1), 42, atol=0.05)     # left of travel is the infield
    assert np.allclose(np.linalg.norm(t.right, axis=1), 58, atol=0.05)
    assert np.isclose(t.length, len(t.centerline) * t.spacing)


def test_start_pose(circle_track):
    assert np.allclose(circle_track.start_position, [0, -50], atol=1e-9)
    assert abs(circle_track.start_heading) < 0.01                         # driving +x at the bottom


def test_wall_distance(circle_track):
    d = wall_distance(circle_track, np.zeros(4), np.array([-50.0, -45.0, 0.0, -70.0]))
    assert abs(d[0] - 8.0) < 0.6      # on the centerline: half the width
    assert abs(d[1] - 3.0) < 0.6      # 5 m toward the infield
    assert d[2] == 0.0                # infield grass
    assert d[3] == 0.0                # outside the grid
```

`tests/test_actions.py`:

```python
import numpy as np

from racing import config
from racing.actions import class_to_controls, controls_to_class


def test_round_trip_over_all_classes():
    classes = np.arange(config.N_ACTIONS)
    steer, throttle = class_to_controls(classes)
    assert np.array_equal(controls_to_class(steer, throttle), classes)


def test_class_layout():
    steer, throttle = class_to_controls([0, 2, 14])
    assert list(steer) == [-1.0, -1.0, 1.0]
    assert list(throttle) == [-1.0, 1.0, 1.0]


def test_continuous_values_snap_to_nearest_level():
    assert controls_to_class([0.8], [0.4])[0] == 3 * 4 + 1      # steer 1.0, coast
```

`tests/test_curriculum.py`:

```python
import math

import numpy as np

from racing import config
from racing.curriculum import LevelTracker, draw_seeds, mutation_strength


def test_mutation_strength_endpoints_and_midpoint():
    assert math.isclose(mutation_strength(0.0), config.GA_S_MAX)
    assert math.isclose(mutation_strength(1.0), config.GA_S_MIN)
    assert math.isclose(mutation_strength(0.5), (config.GA_S_MAX + config.GA_S_MIN) / 2)


def test_level_up_on_third_consecutive_qualifying_generation():
    t = LevelTracker()
    assert not t.update(1.0, 0.9)
    assert not t.update(1.0, 0.85)
    assert t.update(1.0, 0.8)
    assert t.level == 2


def test_streak_resets_after_a_non_qualifying_generation():
    t = LevelTracker()
    t.update(1.0, 0.9)
    t.update(1.0, 0.9)
    assert not t.update(0.98, 0.9)          # one kept individual failed
    assert not t.update(1.0, 0.9)
    assert not t.update(1.0, 0.9)
    assert t.update(1.0, 0.9)
    assert t.level == 2


def test_no_kept_individuals_never_qualifies():
    t = LevelTracker()
    for _ in range(5):
        assert not t.update(float("nan"), 1.0)
    assert t.level == 1


def test_level_stops_at_max():
    t = LevelTracker(level=config.MAX_LEVEL)
    for _ in range(5):
        assert not t.update(1.0, 1.0)
    assert t.level == config.MAX_LEVEL


def test_draw_seeds_are_distinct_training_seeds():
    seeds = draw_seeds(np.random.default_rng(0))
    assert len(seeds) == config.TRACKS_PER_GEN == len(set(seeds))
    assert all(s in config.TRAIN_SEEDS for s in seeds)
```

- [ ] **Step 9: Run the tests**

Run: `$PY -m pytest tests/test_track.py tests/test_actions.py tests/test_curriculum.py -q`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml requirements.txt racing tests
git commit -m "Add project scaffold, configuration, track geometry and curriculum rules

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- pyproject.toml requirements.txt racing tests
```

---

### Task 2: Track generator

**Files:**
- Create: `racing/track_gen.py`
- Test: `tests/test_track_gen.py`

**Interfaces:**
- Consumes: `config.LEVELS`, `config.OVERLAP_MARGIN`, `config.MAX_TRACK_ATTEMPTS`, `track.build_track`.
- Produces:
  - `candidate_track(seed: int, level: int, attempt: int = 0) -> Track | None`. Returns None when the geometry fails validation.
  - `make_track(seed: int, level: int) -> Track`. The first geometrically valid attempt; raises RuntimeError after `MAX_TRACK_ATTEMPTS`.
  - `geometry_ok(track, min_corner) -> bool`.

- [ ] **Step 1: Write the failing tests** in `tests/test_track_gen.py`

```python
import numpy as np
import pytest
from scipy.spatial import cKDTree

from racing import config
from racing.track_gen import candidate_track, make_track


def test_same_seed_gives_same_track():
    assert np.array_equal(make_track(5, 4).centerline, make_track(5, 4).centerline)


def test_different_seeds_give_different_tracks():
    assert make_track(1, 4).length != make_track(2, 4).length


def test_level1_is_the_specified_circle():
    t = make_track(0, 1)
    assert np.allclose(np.linalg.norm(t.centerline, axis=1), 50.0, atol=0.05)
    assert t.width == 16.0
    assert np.all(t.curvature > 0)                       # counter-clockwise


def test_level2_oval_dimensions():
    spec = config.LEVELS[2]
    t = make_track(3, 2)
    k = np.max(np.abs(t.curvature))
    assert t.width == spec["width"]
    assert 1 / spec["radius"][1] - 0.005 <= k <= 1 / spec["radius"][0] + 0.005


@pytest.mark.parametrize("level", [3, 4])
def test_loops_respect_minimum_corner_radius(level):
    for seed in range(10):
        t = make_track(seed, level)
        assert np.max(np.abs(t.curvature)) <= 1 / config.LEVELS[level]["min_corner"] + 1e-9


@pytest.mark.parametrize("level", [3, 4])
def test_loops_never_overlap_themselves(level):
    for seed in range(10):
        t = make_track(seed, level)
        pairs = cKDTree(t.centerline).query_pairs(t.width + config.OVERLAP_MARGIN, output_type="ndarray")
        if len(pairs) == 0:
            continue
        gap = np.abs(pairs[:, 0] - pairs[:, 1]) * t.spacing
        along = np.minimum(gap, t.length - gap)
        assert np.all(along <= 3 * (t.width + config.OVERLAP_MARGIN))


@pytest.mark.parametrize("level", [3, 4])
def test_loop_lengths_are_reasonable(level):
    for seed in range(5):
        assert 300.0 < make_track(seed, level).length < 2500.0


def test_some_candidates_are_valid():
    assert any(candidate_track(0, 4, a) is not None for a in range(20))


@pytest.mark.slow
def test_every_seed_generates():
    for level in (2, 3, 4):
        for seed in config.TRAIN_SEEDS:
            make_track(seed, level)
    for seed in list(config.VAL_SEEDS) + list(config.TEST_SEEDS):
        make_track(seed, 4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_track_gen.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'racing.track_gen'`.

- [ ] **Step 3: Implement `racing/track_gen.py`**

```python
"""Procedural track generation for curriculum levels 1-4 (spec §4.2-4.3)."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from racing import config
from racing.track import build_track


def circle_points(radius, step=0.5):
    """Counter-clockwise circle starting at the bottom."""
    n = int(np.ceil(2 * np.pi * radius / step))
    a = np.linspace(-np.pi / 2, 1.5 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])


def oval_points(straight, radius, step=0.5):
    """Two straights joined by semicircles, counter-clockwise, starting mid-way along the bottom straight."""
    half = straight / 2
    points = [(x, -radius) for x in np.arange(-half, half, step)]
    points += [(half + radius * np.cos(a), radius * np.sin(a)) for a in np.arange(-np.pi / 2, np.pi / 2, step / radius)]
    points += [(x, radius) for x in np.arange(half, -half, -step)]
    points += [(-half + radius * np.cos(a), radius * np.sin(a)) for a in np.arange(np.pi / 2, 1.5 * np.pi, step / radius)]
    points = np.array(points)
    start = np.argmin(np.hypot(points[:, 0], points[:, 1] + radius))
    return np.roll(points, -start, axis=0)


def loop_points(rng, n_points, radius_range, step=0.5):
    """Control points at jittered angles and random radii, joined by a periodic cubic spline."""
    angles = (np.arange(n_points) + rng.uniform(-0.3, 0.3, n_points)) * (2 * np.pi / n_points)
    radii = rng.uniform(radius_range[0], radius_range[1], n_points)
    control = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    closed = np.vstack([control, control[:1]])
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    spline = CubicSpline(t, closed, bc_type="periodic")
    return spline(np.arange(0.0, t[-1], step))


def geometry_ok(track, min_corner):
    """True if no corner is tighter than min_corner and separate parts of the road stay apart."""
    if np.max(np.abs(track.curvature)) > 1.0 / min_corner:
        return False
    reach = track.width + config.OVERLAP_MARGIN
    pairs = cKDTree(track.centerline).query_pairs(reach, output_type="ndarray")
    if len(pairs) == 0:
        return True
    gap = np.abs(pairs[:, 0] - pairs[:, 1]) * track.spacing
    along = np.minimum(gap, track.length - gap)
    return not np.any(along > 3 * reach)


def candidate_track(seed, level, attempt=0):
    """One generation attempt for (seed, level). Returns None if the geometry is rejected."""
    spec = config.LEVELS[level]
    rng = np.random.default_rng([seed, level, attempt])
    if spec["kind"] == "circle":
        points = circle_points(spec["radius"])
    elif spec["kind"] == "oval":
        points = oval_points(rng.uniform(*spec["straight"]), rng.uniform(*spec["radius"]))
    else:
        n_points = int(rng.integers(spec["points"][0], spec["points"][1] + 1))
        points = loop_points(rng, n_points, spec["radius"])
    track = build_track(points, spec["width"], seed, level)
    if spec["kind"] == "loop" and not geometry_ok(track, spec["min_corner"]):
        return None
    return track


def make_track(seed, level):
    """The first geometrically valid track for (seed, level)."""
    for attempt in range(config.MAX_TRACK_ATTEMPTS):
        track = candidate_track(seed, level, attempt)
        if track is not None:
            return track
    raise RuntimeError(f"no valid track geometry for seed {seed}, level {level}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_track_gen.py -q`, then `$PY -m pytest tests/test_track_gen.py -q -m slow`
Expected: all pass. If any seed exhausts its attempts, or acceptance is below about 5% at level 4, adjust the generator rather than the limits:
- reduce the angular jitter (0.3)
- narrow the radius ranges in `config.LEVELS`

Report the acceptance rate (valid attempts ÷ attempts) at levels 3 and 4 for seeds 0–79. This task may edit the `LEVELS` entries in `config.py`; report every change.

- [ ] **Step 5: Commit**

```bash
git add racing/track_gen.py tests/test_track_gen.py
git commit -m "Add procedural track generator for curriculum levels 1-4

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/track_gen.py tests/test_track_gen.py
```

---

### Task 3: Simulator

**Files:**
- Create: `racing/sim.py`
- Test: `tests/test_sim.py`

**Interfaces:**
- Consumes: `config`, `actions.class_to_controls`, `track.wall_distance`, `Track` (via the `circle_track` fixture).
- Produces:
  - `Cars` dataclass. Per-car arrays: `x, y, heading, speed, index, progress, last_progress_time, laps_done, lap_start_time, last_lap, best_lap, crashed, finished, done, finish_time`. Scalars: `laps`, `time`.
  - `start_cars(track, n, laps=1) -> Cars`
  - `cast_rays(track, x, y, heading) -> (n, N_RAYS)` in m
  - `sense(track, cars) -> (n, N_INPUTS)`, zero rows for cars that are done
  - `step(track, cars, steer, throttle, time_limit) -> None`, in place
  - `fitness(cars, track, time_limit) -> (n,)`
  - `time_limit_for(track, laps=1) -> float`
  - `RunResult(fitness, finished, crashed, progress, finish_time, steps)`
  - `simulate(track, policy, n, time_limit=None, laps=1) -> RunResult`, where `policy(obs, cars) -> classes (n,)`

- [ ] **Step 1: Write the failing tests** in `tests/test_sim.py`

```python
import numpy as np

from racing import config, sim
from racing.actions import class_to_controls, controls_to_class


def constant_policy(steer, throttle):
    action = int(controls_to_class([steer], [throttle])[0])
    return lambda obs, cars: np.full(len(cars.x), action)


def test_zero_steering_drives_straight(circle_track):
    cars = sim.start_cars(circle_track, 1)
    x0, y0, h0 = cars.x[0], cars.y[0], cars.heading[0]
    for _ in range(20):
        sim.step(circle_track, cars, np.array([0.0]), np.array([1.0]), time_limit=100.0)
    assert cars.heading[0] == h0
    assert np.isclose(np.arctan2(cars.y[0] - y0, cars.x[0] - x0), h0)
    assert cars.speed[0] > 0


def test_grip_limit_is_never_exceeded(circle_track):
    cars = sim.start_cars(circle_track, 1)
    cars.speed[:] = 10.0
    h0 = cars.heading[0]
    sim.step(circle_track, cars, np.array([1.0]), np.array([1.0]), time_limit=100.0)
    a_long = (cars.speed[0] - 10.0) / config.DT
    a_lat = (cars.heading[0] - h0) / config.DT * 10.0
    assert np.hypot(a_long, a_lat) <= config.GRIP + 1e-9
    assert np.isclose(a_long, a_lat)                  # proportional scaling keeps the requested direction


def test_leaving_the_road_is_a_crash(circle_track):
    result = sim.simulate(circle_track, constant_policy(0.0, 1.0), 1)
    assert result.crashed[0] and not result.finished[0]
    assert 0.0 < result.progress[0] < 0.5


def test_car_that_never_moves_stops_at_the_stall_timeout(circle_track):
    result = sim.simulate(circle_track, constant_policy(0.0, 0.0), 1)
    assert not result.finished[0] and not result.crashed[0]
    assert result.steps * config.DT <= config.STALL_TIME + 2 * config.DT
    assert result.fitness[0] == 0.0


def test_backward_driving_earns_nothing(circle_track):
    cars = sim.start_cars(circle_track, 1)
    cars.heading[:] += np.pi
    cars.speed[:] = 10.0
    steer = np.array([-(10.0 ** 2 / 50.0) / config.GRIP])     # clockwise turn that follows the circle backwards
    while not cars.done[0]:
        sim.step(circle_track, cars, steer, np.array([0.0]), time_limit=100.0)
    assert cars.progress[0] == 0.0 and cars.laps_done[0] == 0
    assert not cars.crashed[0]                               # stopped by the stall timeout, still on the road


def test_full_lap_is_detected(circle_track):
    cars = sim.start_cars(circle_track, 1)
    cars.speed[:] = 15.0
    steer = np.array([(15.0 ** 2 / 50.0) / config.GRIP])
    while not cars.done[0]:
        sim.step(circle_track, cars, steer, np.array([0.0]), time_limit=100.0)
    assert cars.finished[0] and not cars.crashed[0]
    assert abs(cars.finish_time[0] - circle_track.length / 15.0) < 0.5
    assert np.isclose(cars.last_lap[0], cars.finish_time[0])


def test_multi_lap_race_counts_laps(circle_track):
    cars = sim.start_cars(circle_track, 1, laps=2)
    cars.speed[:] = 15.0
    steer = np.array([(15.0 ** 2 / 50.0) / config.GRIP])
    while not cars.done[0]:
        sim.step(circle_track, cars, steer, np.array([0.0]), time_limit=100.0)
    assert cars.finished[0] and cars.laps_done[0] == 2
    assert np.isfinite(cars.best_lap[0])


def test_fitness_formula(circle_track):
    cars = sim.start_cars(circle_track, 3)
    cars.progress[:] = [0.0, circle_track.length * 0.5, circle_track.length]
    cars.finished[:] = [False, False, True]
    cars.finish_time[:] = [np.nan, np.nan, 10.0]
    f = sim.fitness(cars, circle_track, time_limit=40.0)
    assert f[0] == 0.0
    assert abs(f[1] - 0.5) < 1e-12
    assert abs(f[2] - 1.75) < 1e-12


def test_rays_on_circle(circle_track):
    cars = sim.start_cars(circle_track, 1)
    rays = sim.cast_rays(circle_track, cars.x, cars.y, cars.heading)[0]
    assert abs(rays[0] - 8.0) < 0.75                        # right: outer edge
    assert abs(rays[-1] - 8.0) < 0.75                       # left: inner edge
    assert abs(rays[4] - np.sqrt(58 ** 2 - 50 ** 2)) < 0.75  # ahead: chord to the outer edge


def test_sense_is_zero_for_finished_cars(circle_track):
    cars = sim.start_cars(circle_track, 2)
    cars.done[1] = True
    obs = sim.sense(circle_track, cars)
    assert obs.shape == (2, config.N_INPUTS)
    assert np.all(obs[1] == 0.0) and np.all(obs[0, :config.N_RAYS] > 0)


def test_cars_are_independent_of_the_batch(circle_track):
    def run(speeds):
        cars = sim.start_cars(circle_track, len(speeds))
        cars.speed[:] = speeds
        for _ in range(200):
            obs = sim.sense(circle_track, cars)
            classes = np.floor(obs[:, 4] * 13 + obs[:, 9] * 7).astype(np.int64) % config.N_ACTIONS
            steer, throttle = class_to_controls(classes)
            sim.step(circle_track, cars, steer, throttle, time_limit=100.0)
        return cars

    speeds = 5.0 + np.arange(8.0)
    full, part = run(speeds), run(speeds[[2, 5]])
    for name in ("x", "y", "heading", "speed", "progress"):
        assert np.array_equal(getattr(full, name)[[2, 5]], getattr(part, name))


def test_time_limit_requires_expert_time(circle_track):
    circle_track.expert_lap_time = float("nan")
    try:
        sim.time_limit_for(circle_track)
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_sim.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'racing.sim'`.

- [ ] **Step 3: Implement `racing/sim.py`**

```python
"""Vectorized driving simulator (spec §5). Every per-car quantity is an array with one entry per car."""
from dataclasses import dataclass

import numpy as np

from racing import config
from racing.actions import class_to_controls
from racing.track import wall_distance


@dataclass(eq=False)
class Cars:
    """State of a batch of cars on one track. Cars never interact with each other."""
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    index: np.ndarray               # matched centerline point, counted without wrapping
    progress: np.ndarray            # m, furthest distance reached along the centerline
    last_progress_time: np.ndarray
    laps_done: np.ndarray
    lap_start_time: np.ndarray
    last_lap: np.ndarray            # s, nan until a lap is completed
    best_lap: np.ndarray            # s, nan until a lap is completed
    crashed: np.ndarray
    finished: np.ndarray            # completed `laps` laps
    done: np.ndarray                # crashed, finished, stalled or out of time
    finish_time: np.ndarray         # s, nan unless finished
    laps: int = 1
    time: float = 0.0


@dataclass(eq=False)
class RunResult:
    fitness: np.ndarray
    finished: np.ndarray
    crashed: np.ndarray
    progress: np.ndarray            # fraction of the run completed, 0 to 1
    finish_time: np.ndarray
    steps: int


def start_cars(track, n, laps=1):
    """n cars at rest on the start line, facing the direction of travel."""
    x0, y0 = track.start_position
    nan = np.full(n, np.nan)
    return Cars(
        x=np.full(n, x0), y=np.full(n, y0), heading=np.full(n, track.start_heading), speed=np.zeros(n),
        index=np.zeros(n, dtype=np.int64), progress=np.zeros(n), last_progress_time=np.zeros(n),
        laps_done=np.zeros(n, dtype=np.int64), lap_start_time=np.zeros(n), last_lap=nan.copy(),
        best_lap=nan.copy(), crashed=np.zeros(n, dtype=bool), finished=np.zeros(n, dtype=bool),
        done=np.zeros(n, dtype=bool), finish_time=nan.copy(), laps=laps,
    )


def cast_rays(track, x, y, heading):
    """Distance to the track boundary along each sensor ray, by sphere tracing. Returns (n, N_RAYS) in m."""
    n = len(x)
    angles = (np.asarray(heading)[:, None] + config.RAY_ANGLES[None, :]).ravel()
    dx, dy = np.cos(angles), np.sin(angles)
    ox, oy = np.repeat(x, config.N_RAYS), np.repeat(y, config.N_RAYS)
    t = np.zeros(n * config.N_RAYS)
    active = np.arange(n * config.N_RAYS)
    for _ in range(config.RAY_MAX_ITERATIONS):
        if active.size == 0:
            break
        d = wall_distance(track, ox[active] + dx[active] * t[active], oy[active] + dy[active] * t[active])
        hit = d <= 0.0
        t[active] = np.where(hit, t[active], t[active] + np.maximum(d, config.RAY_MIN_STEP))
        active = active[~hit & (t[active] < config.RAY_RANGE)]
    return np.minimum(t, config.RAY_RANGE).reshape(n, config.N_RAYS)


def sense(track, cars):
    """Network inputs: ray distances / RAY_RANGE and speed / SPEED_SCALE. Rows of cars that are done stay zero."""
    obs = np.zeros((len(cars.x), config.N_INPUTS))
    live = np.flatnonzero(~cars.done)
    if live.size:
        obs[live, :config.N_RAYS] = cast_rays(track, cars.x[live], cars.y[live], cars.heading[live]) / config.RAY_RANGE
        obs[live, config.N_RAYS] = cars.speed[live] / config.SPEED_SCALE
    return obs


def step(track, cars, steer, throttle, time_limit):
    """Advance every car that is not done by one time step. steer and throttle are grip fractions in [-1, 1]."""
    live = ~cars.done
    a_lat = np.clip(steer, -1.0, 1.0) * config.GRIP
    a_long = np.clip(throttle, -1.0, 1.0) * config.GRIP
    total = np.hypot(a_lat, a_long)
    scale = np.where(total > config.GRIP, config.GRIP / np.maximum(total, 1e-12), 1.0)
    a_lat, a_long = a_lat * scale, a_long * scale                       # friction circle

    yaw_rate = a_lat / np.maximum(cars.speed, config.YAW_MIN_SPEED)
    speed = np.maximum(cars.speed + a_long * config.DT, 0.0)
    heading = cars.heading + yaw_rate * config.DT
    cars.x = np.where(live, cars.x + speed * np.cos(heading) * config.DT, cars.x)
    cars.y = np.where(live, cars.y + speed * np.sin(heading) * config.DT, cars.y)
    cars.speed = np.where(live, speed, cars.speed)
    cars.heading = np.where(live, heading, cars.heading)
    cars.time += config.DT

    crashed = live & (wall_distance(track, cars.x, cars.y) <= 0.0)
    cars.crashed |= crashed
    cars.speed[crashed] = 0.0
    moving = live & ~crashed
    _update_progress(track, cars, moving)
    stalled = moving & (cars.time - cars.last_progress_time >= config.STALL_TIME - config.TIME_EPS)
    out_of_time = live & (cars.time >= time_limit - config.TIME_EPS)
    cars.done |= crashed | stalled | out_of_time | cars.finished


def _update_progress(track, cars, mask):
    """Match each car to the nearest centerline point near its last match; update progress, laps and finishing."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    m = len(track.centerline)
    back = int(round(config.PROGRESS_BACK / track.spacing))
    ahead = int(round(config.PROGRESS_AHEAD / track.spacing))
    candidates = cars.index[idx, None] + np.arange(-back, ahead + 1)[None, :]
    points = track.centerline[np.mod(candidates, m)]
    d2 = (points[..., 0] - cars.x[idx, None]) ** 2 + (points[..., 1] - cars.y[idx, None]) ** 2
    cars.index[idx] = candidates[np.arange(idx.size), np.argmin(d2, axis=1)]

    reached = cars.index[idx] * track.spacing
    cars.last_progress_time[idx[reached > cars.progress[idx]]] = cars.time
    cars.progress[idx] = np.maximum(cars.progress[idx], reached)

    laps_now = np.round(cars.progress[idx] / track.spacing).astype(np.int64) // m
    new_lap = laps_now > cars.laps_done[idx]
    j = idx[new_lap]
    lap_time = cars.time - cars.lap_start_time[j]
    cars.last_lap[j] = lap_time
    cars.best_lap[j] = np.fmin(cars.best_lap[j], lap_time)
    cars.lap_start_time[j] = cars.time
    cars.laps_done[j] = laps_now[new_lap]
    complete = j[cars.laps_done[j] >= cars.laps]
    cars.finished[complete] = True
    cars.finish_time[complete] = cars.time


def fitness(cars, track, time_limit):
    """Spec §5.5: completed fraction if unfinished (below 1); 1 + (1 - finish_time / time_limit) if finished."""
    fraction = np.minimum(cars.progress / (track.length * cars.laps), np.nextafter(1.0, 0.0))
    return np.where(cars.finished, 1.0 + (1.0 - cars.finish_time / time_limit), fraction)


def time_limit_for(track, laps=1):
    """Twice the expert's lap time per lap."""
    if np.isnan(track.expert_lap_time):
        raise ValueError("track has no expert lap time; pass time_limit explicitly")
    return config.TIME_LIMIT_FACTOR * track.expert_lap_time * laps


def simulate(track, policy, n, time_limit=None, laps=1):
    """Drive n cars from the start line until every car is done. policy(obs, cars) returns action classes."""
    if time_limit is None:
        time_limit = time_limit_for(track, laps)
    cars = start_cars(track, n, laps)
    steps = 0
    while not cars.done.all():
        steer, throttle = class_to_controls(policy(sense(track, cars), cars))
        step(track, cars, steer, throttle, time_limit)
        steps += 1
    return RunResult(fitness=fitness(cars, track, time_limit), finished=cars.finished.copy(),
                     crashed=cars.crashed.copy(), progress=np.minimum(cars.progress / (track.length * laps), 1.0),
                     finish_time=cars.finish_time.copy(), steps=steps)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_sim.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add racing/sim.py tests/test_sim.py
git commit -m "Add vectorized grip-limited simulator with sphere-traced sensors

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/sim.py tests/test_sim.py
```

---

### Task 4: Controller networks

**Files:**
- Create: `racing/nets.py`
- Test: `tests/test_nets.py`

**Interfaces:**
- Consumes: `config`, `backend.keras_cpu`.
- Produces:
  - Flat weights:
    - `layer_sizes(hidden)`
    - `n_params(hidden=BASELINE_HIDDEN) -> int`
    - `init_population(rng, n, hidden) -> (n, P)`
    - `unflatten(flat, hidden) -> [(W (n,a,b), b (n,b))]`
  - Numpy forward pass:
    - `activate(h, name)`
    - `forward(flat, obs, hidden, activation) -> logits (n, 15)`
    - `act(flat, obs, hidden, activation) -> classes (n,)`
    - `population_policy(flat, hidden, activation) -> policy(obs, cars)`
  - `DenseNet(layers, activations)`, with `.act(obs)`, `.policy()` and `.n_params`
  - Keras conversion and loading:
    - `from_flat(flat, hidden, activation) -> DenseNet`
    - `from_keras(model) -> DenseNet`
    - `to_keras(flat, hidden, activation) -> keras.Model`
    - `load_champions(directory=None) -> dict[approach, keras.Model]`

- [ ] **Step 1: Write the failing tests** in `tests/test_nets.py`

```python
import numpy as np

from racing import config, nets


def test_baseline_parameter_count():
    assert nets.n_params() == 431


def test_init_population_shape_and_zero_biases():
    w = nets.init_population(np.random.default_rng(0), 7)
    assert w.shape == (7, 431)
    for _, bias in nets.unflatten(w):
        assert np.all(bias == 0.0)


def test_forward_matches_manual_computation():
    rng = np.random.default_rng(1)
    w = nets.init_population(rng, 1)
    w[:, :] += rng.normal(0, 0.1, w.shape)
    obs = rng.random((1, config.N_INPUTS))
    (w1, b1), (w2, b2) = [(a[0], b[0]) for a, b in nets.unflatten(w)]
    expected = np.maximum(obs @ w1 + b1, 0) @ w2 + b2
    assert np.allclose(nets.forward(w, obs), expected)


def test_each_car_uses_its_own_weights():
    rng = np.random.default_rng(2)
    w = nets.init_population(rng, 3)
    obs = np.repeat(rng.random((1, config.N_INPUTS)), 3, axis=0)
    together = nets.forward(w, obs)
    for i in range(3):
        assert np.allclose(together[i], nets.forward(w[i:i + 1], obs[i:i + 1])[0])


def test_act_returns_valid_classes():
    rng = np.random.default_rng(3)
    classes = nets.act(nets.init_population(rng, 50), rng.random((50, config.N_INPUTS)))
    assert classes.shape == (50,) and classes.min() >= 0 and classes.max() < config.N_ACTIONS


def test_from_flat_matches_population_forward():
    rng = np.random.default_rng(4)
    w = nets.init_population(rng, 1)
    obs = rng.random((20, config.N_INPUTS))
    net = nets.from_flat(w[0])
    assert np.array_equal(net.act(obs), nets.act(np.repeat(w, 20, axis=0), obs))
    assert net.n_params == 431


def test_to_keras_matches_numpy():
    rng = np.random.default_rng(5)
    w = nets.init_population(rng, 1)[0] + rng.normal(0, 0.2, 431)
    model = nets.to_keras(w)
    obs = rng.random((300, config.N_INPUTS)).astype(np.float32)
    keras_classes = np.argmax(model.predict(obs, verbose=0), axis=1)
    assert model.count_params() == 431
    assert np.mean(keras_classes == nets.from_flat(w).act(obs)) >= 0.99


def test_from_keras_skips_dropout_and_keeps_activations():
    from racing.backend import keras_cpu
    keras = keras_cpu()
    model = keras.Sequential([keras.Input((config.N_INPUTS,)), keras.layers.Dense(8, activation="tanh"),
                              keras.layers.Dropout(0.2), keras.layers.Dense(4, activation="relu"),
                              keras.layers.Dense(config.N_ACTIONS, activation="softmax")])
    net = nets.from_keras(model)
    obs = np.random.default_rng(6).random((200, config.N_INPUTS)).astype(np.float32)
    assert net.activations == ["tanh", "relu"]
    assert np.mean(np.argmax(model.predict(obs, verbose=0), axis=1) == net.act(obs)) >= 0.99


def test_load_champions_empty_directory(tmp_path):
    assert nets.load_champions(tmp_path) == {}


def test_load_champions_keeps_standard_order(tmp_path):
    w = nets.init_population(np.random.default_rng(7), 1)[0]
    nets.to_keras(w).save(tmp_path / "ga.keras")
    nets.to_keras(w).save(tmp_path / "baseline.keras")
    assert list(nets.load_champions(tmp_path)) == ["baseline", "ga"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_nets.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'racing.nets'`.

- [ ] **Step 3: Implement `racing/nets.py`**

```python
"""Controller networks (spec §6): flat weight vectors for populations, a batched numpy forward pass,
and conversion to and from Keras models."""
from pathlib import Path

import numpy as np

from racing import config
from racing.backend import keras_cpu


def layer_sizes(hidden=config.BASELINE_HIDDEN):
    return [config.N_INPUTS, *hidden, config.N_ACTIONS]


def n_params(hidden=config.BASELINE_HIDDEN):
    sizes = layer_sizes(hidden)
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def init_population(rng, n, hidden=config.BASELINE_HIDDEN):
    """Glorot-uniform weights and zero biases for n networks, flattened to shape (n, n_params)."""
    parts = []
    sizes = layer_sizes(hidden)
    for a, b in zip(sizes[:-1], sizes[1:]):
        limit = np.sqrt(6.0 / (a + b))
        parts.append(rng.uniform(-limit, limit, (n, a * b)))
        parts.append(np.zeros((n, b)))
    return np.concatenate(parts, axis=1)


def unflatten(flat, hidden=config.BASELINE_HIDDEN):
    """Split (n, P) flat weights into [(W (n, a, b), bias (n, b))] per layer."""
    n = flat.shape[0]
    layers, pos = [], 0
    sizes = layer_sizes(hidden)
    for a, b in zip(sizes[:-1], sizes[1:]):
        weights = flat[:, pos:pos + a * b].reshape(n, a, b)
        pos += a * b
        bias = flat[:, pos:pos + b]
        pos += b
        layers.append((weights, bias))
    if pos != flat.shape[1]:
        raise ValueError(f"expected {pos} weights per network, got {flat.shape[1]}")
    return layers


def activate(h, name):
    if name == "relu":
        return np.maximum(h, 0.0)
    if name == "tanh":
        return np.tanh(h)
    if name == "linear":
        return h
    raise ValueError(f"unsupported activation: {name}")


def forward(flat, obs, hidden=config.BASELINE_HIDDEN, activation="relu"):
    """Logits (n, N_ACTIONS) for n cars, each with its own weights: flat (n, P), obs (n, N_INPUTS)."""
    layers = unflatten(flat, hidden)
    h = obs
    for i, (weights, bias) in enumerate(layers):
        h = np.einsum("ni,nij->nj", h, weights) + bias
        if i < len(layers) - 1:
            h = activate(h, activation)
    return h


def act(flat, obs, hidden=config.BASELINE_HIDDEN, activation="relu"):
    return np.argmax(forward(flat, obs, hidden, activation), axis=1)


def population_policy(flat, hidden=config.BASELINE_HIDDEN, activation="relu"):
    """Policy for sim.simulate: car i is driven by network flat[i]."""
    def policy(obs, cars):
        return act(flat, obs, hidden, activation)
    return policy


class DenseNet:
    """A plain stack of dense layers in numpy. Drives a car with any trained controller."""

    def __init__(self, layers, activations):
        self.layers = [(np.asarray(w, dtype=float), np.asarray(b, dtype=float)) for w, b in layers]
        self.activations = list(activations)

    @property
    def n_params(self):
        return int(sum(w.size + b.size for w, b in self.layers))

    def act(self, obs):
        h = np.asarray(obs, dtype=float)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = activate(h, self.activations[i])
        return np.argmax(h, axis=1)

    def policy(self):
        return lambda obs, cars: self.act(obs)


def from_flat(flat, hidden=config.BASELINE_HIDDEN, activation="relu"):
    layers = [(w[0], b[0]) for w, b in unflatten(np.asarray(flat, dtype=float)[None, :], hidden)]
    return DenseNet(layers, [activation] * len(hidden))


def from_keras(model):
    """DenseNet with the weights of a trained Keras model. Dropout layers are inactive at inference and skipped."""
    keras = keras_cpu()
    layers, activations = [], []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            w, b = layer.get_weights()
            layers.append((w, b))
            activations.append(layer.get_config()["activation"])
    return DenseNet(layers, activations[:-1])     # the output softmax does not change the argmax


def to_keras(flat, hidden=config.BASELINE_HIDDEN, activation="relu"):
    """Keras model with the same architecture and weights as one flat weight vector."""
    keras = keras_cpu()
    model = keras.Sequential([keras.Input((config.N_INPUTS,)),
                              *[keras.layers.Dense(units, activation=activation) for units in hidden],
                              keras.layers.Dense(config.N_ACTIONS, activation="softmax")])
    weights = []
    for w, b in unflatten(np.asarray(flat, dtype=float)[None, :], hidden):
        weights += [w[0], b[0]]
    model.set_weights(weights)
    return model


def load_champions(directory=None):
    """Final controllers saved as <approach>.keras, in the order of config.APPROACHES. Missing ones are skipped."""
    directory = Path(config.CHAMPIONS_DIR if directory is None else directory)
    models = {}
    for approach in config.APPROACHES:
        path = directory / f"{approach}.keras"
        if path.exists():
            models[approach] = keras_cpu().models.load_model(path)
    return models
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_nets.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add racing/nets.py tests/test_nets.py
git commit -m "Add population networks, numpy forward pass and Keras conversion

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/nets.py tests/test_nets.py
```

---

### Task 5: Imitation-learning controllers

**Files:**
- Create: `racing/imitation.py`
- Test: `tests/test_imitation.py`

**Interfaces:**
- Consumes: `config`, `backend.keras_cpu`. The dataset format from Task 6 is a dict with `X_train, y_train, X_val, y_val, X_test, y_test`: float32 inputs (k, 10), int64 labels (k,).
- Produces:
  - Data and models:
    - `load_dataset(path=None) -> dict`
    - `build_baseline() -> keras.Model`
    - `build_hypermodel(hp) -> keras.Model`
  - Baseline training: `train_baseline(data, out_dir=None, champion_dir=CHAMPIONS_DIR, on_epoch=None, epochs=MAX_EPOCHS) -> dict`. `on_epoch(epoch, model, val_accuracy)`; `champion_dir=None` skips saving.
  - Keras Tuner search: `run_tuner(approach, data, out_dir=None, champion_dir=CHAMPIONS_DIR, on_trial=None, seed=0, max_trials=KT_TRIALS, max_epochs=MAX_EPOCHS, hyperband_iterations=HYPERBAND_ITERATIONS) -> dict`. `on_trial(number, model, val_accuracy)`.
  - Model selection: `select_smallest(records, tolerance=SIZE_RULE_TOLERANCE) -> record`.
  - Run files:
    - `summary.json` in every run directory. It has `wall_time`; the tuner summary also has `chosen_trial`.
    - `trials.csv` with columns `trial_id, val_accuracy, params, epochs, hyperparameters`.

- [ ] **Step 1: Write the failing tests** in `tests/test_imitation.py`

```python
import json

import numpy as np
import pytest

from racing import imitation


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(10, 15))

    def make(n):
        x = rng.random((n, 10)).astype(np.float32)
        return x, np.argmax(x @ w, axis=1).astype(np.int64)

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = make(3000), make(500), make(500)
    return {"X_train": x_tr, "y_train": y_tr, "X_val": x_va, "y_val": y_va, "X_test": x_te, "y_test": y_te}


def test_select_smallest_within_one_point():
    records = [{"val_accuracy": 0.90, "params": 500}, {"val_accuracy": 0.895, "params": 100},
               {"val_accuracy": 0.85, "params": 50}]
    assert imitation.select_smallest(records)["params"] == 100


def test_select_smallest_prefers_accuracy_on_equal_size():
    records = [{"val_accuracy": 0.90, "params": 100}, {"val_accuracy": 0.899, "params": 100}]
    assert imitation.select_smallest(records)["val_accuracy"] == 0.90


def test_baseline_has_431_parameters():
    assert imitation.build_baseline().count_params() == 431


def test_load_dataset_round_trip(tmp_path, toy_data):
    np.savez(tmp_path / "d.npz", **toy_data)
    loaded = imitation.load_dataset(tmp_path / "d.npz")
    assert set(loaded) == set(toy_data) and np.array_equal(loaded["y_val"], toy_data["y_val"])


def test_train_baseline_learns_reports_and_saves(toy_data, tmp_path):
    seen = []
    summary = imitation.train_baseline(toy_data, out_dir=tmp_path / "run", champion_dir=tmp_path / "champions",
                                       on_epoch=lambda epoch, model, acc: seen.append((epoch, acc)), epochs=5)
    assert (tmp_path / "champions" / "baseline.keras").exists()
    assert len(seen) == summary["epochs"] and summary["val_accuracy"] > 0.2
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["params"] == 431


def test_train_baseline_without_champion(toy_data, tmp_path):
    imitation.train_baseline(toy_data, out_dir=tmp_path / "run", champion_dir=None, epochs=1)
    assert not (tmp_path / "champions").exists()


@pytest.mark.slow
@pytest.mark.parametrize("approach", ["kt_random", "kt_bayesian", "kt_hyperband"])
def test_run_tuner_small(approach, toy_data, tmp_path):
    seen = []
    summary = imitation.run_tuner(approach, toy_data, out_dir=tmp_path / approach, champion_dir=tmp_path / "champions",
                                  on_trial=lambda number, model, acc: seen.append(model.count_params()),
                                  max_trials=3, max_epochs=3, hyperband_iterations=1)
    assert (tmp_path / "champions" / f"{approach}.keras").exists()
    assert summary["trials"] == len(seen) >= 2
    assert (tmp_path / approach / "trials.csv").exists()
    assert summary["chosen_params"] <= max(seen)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_imitation.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'racing.imitation'`.

- [ ] **Step 3: Implement `racing/imitation.py`**

```python
"""Imitation-learning controllers (spec §8): a baseline network and three Keras Tuner searches."""
import csv
import json
import time
from pathlib import Path

import numpy as np

from racing import config
from racing.backend import keras_cpu

TUNER_CLASSES = {"kt_random": "RandomSearch", "kt_hyperband": "Hyperband", "kt_bayesian": "BayesianOptimization"}


def load_dataset(path=None):
    path = config.DATASET_PATH if path is None else Path(path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def compile_model(model, learning_rate):
    keras = keras_cpu()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_baseline():
    """The 431-parameter baseline: 10 inputs, 16 ReLU units, 15-way softmax."""
    keras = keras_cpu()
    model = keras.Sequential([keras.Input((config.N_INPUTS,)),
                              keras.layers.Dense(config.BASELINE_HIDDEN[0], activation="relu"),
                              keras.layers.Dense(config.N_ACTIONS, activation="softmax")])
    return compile_model(model, config.LEARNING_RATE)


def build_hypermodel(hp):
    """Search space of spec Table 2."""
    keras = keras_cpu()
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Choice("dropout", [0.0, 0.1, 0.2])
    layers = [keras.Input((config.N_INPUTS,))]
    for i in range(hp.Int("layers", 1, 3)):
        layers.append(keras.layers.Dense(hp.Choice(f"units_{i}", [4, 8, 16, 32, 64]), activation=activation))
        if dropout > 0:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(config.N_ACTIONS, activation="softmax"))
    return compile_model(keras.Sequential(layers), hp.Float("learning_rate", 1e-4, 1e-2, sampling="log"))


def early_stopping():
    keras = keras_cpu()
    return keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=config.PATIENCE, restore_best_weights=True)


def epoch_reporter(on_epoch):
    """Keras callback that calls on_epoch(epoch, model, val_accuracy) after every epoch."""
    keras = keras_cpu()

    class EpochReporter(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            on_epoch(epoch, self.model, float((logs or {}).get("val_accuracy", float("nan"))))

    return EpochReporter()


def train_baseline(data, out_dir=None, champion_dir=config.CHAMPIONS_DIR, on_epoch=None, epochs=config.MAX_EPOCHS):
    out_dir = Path(out_dir or config.RUNS_DIR / "baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_baseline()
    callbacks = [early_stopping()]
    if on_epoch is not None:
        callbacks.append(epoch_reporter(on_epoch))
    start = time.time()
    history = model.fit(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                        epochs=epochs, batch_size=config.BATCH_SIZE, callbacks=callbacks, verbose=0)
    summary = {"approach": "baseline", "epochs": len(history.history["val_accuracy"]),
               "val_accuracy": float(max(history.history["val_accuracy"])),
               "params": int(model.count_params()), "wall_time": time.time() - start}
    if champion_dir is not None:
        Path(champion_dir).mkdir(parents=True, exist_ok=True)
        model.save(Path(champion_dir) / "baseline.keras")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def make_tuner(approach, directory, seed=0, on_trial=None, max_trials=config.KT_TRIALS,
               max_epochs=config.MAX_EPOCHS, hyperband_iterations=config.HYPERBAND_ITERATIONS):
    keras_cpu()
    import keras_tuner as kt
    base = getattr(kt, TUNER_CLASSES[approach])

    class ReportingTuner(base):
        completed = 0

        def on_trial_end(self, trial):
            super().on_trial_end(trial)
            if on_trial is not None and trial.score is not None:
                self.completed += 1
                on_trial(self.completed, self.load_model(trial), float(trial.score))

    common = {"hypermodel": build_hypermodel, "objective": "val_accuracy", "seed": seed,
              "directory": str(directory), "project_name": approach, "overwrite": True}
    if approach == "kt_hyperband":
        return ReportingTuner(max_epochs=max_epochs, factor=config.HYPERBAND_FACTOR,
                              hyperband_iterations=hyperband_iterations, **common)
    return ReportingTuner(max_trials=max_trials, **common)


def trial_records(tuner):
    """One record per completed trial: validation accuracy, parameter count, epochs trained, hyperparameters."""
    records = []
    for trial in tuner.oracle.trials.values():
        if trial.score is None:
            continue
        model = tuner.hypermodel.build(trial.hyperparameters)
        records.append({"trial_id": trial.trial_id, "val_accuracy": float(trial.score),
                        "params": int(model.count_params()),
                        "epochs": len(trial.metrics.get_history("val_accuracy")),
                        "hyperparameters": dict(trial.hyperparameters.values), "trial": trial})
    return records


def select_smallest(records, tolerance=config.SIZE_RULE_TOLERANCE):
    """Spec §8.3: fewest parameters among records within `tolerance` of the best validation accuracy."""
    best = max(r["val_accuracy"] for r in records)
    eligible = [r for r in records if r["val_accuracy"] >= best - tolerance]
    return min(eligible, key=lambda r: (r["params"], -r["val_accuracy"]))


def run_tuner(approach, data, out_dir=None, champion_dir=config.CHAMPIONS_DIR, on_trial=None, seed=0,
              max_trials=config.KT_TRIALS, max_epochs=config.MAX_EPOCHS,
              hyperband_iterations=config.HYPERBAND_ITERATIONS):
    out_dir = Path(out_dir or config.RUNS_DIR / approach)
    out_dir.mkdir(parents=True, exist_ok=True)
    tuner = make_tuner(approach, out_dir / "tuner", seed, on_trial, max_trials, max_epochs, hyperband_iterations)
    start = time.time()
    tuner.search(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                 epochs=max_epochs, batch_size=config.BATCH_SIZE, callbacks=[early_stopping()], verbose=0)
    wall_time = time.time() - start
    records = trial_records(tuner)
    chosen = select_smallest(records)
    with open(out_dir / "trials.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "val_accuracy", "params", "epochs", "hyperparameters"])
        for r in records:
            writer.writerow([r["trial_id"], r["val_accuracy"], r["params"], r["epochs"], json.dumps(r["hyperparameters"])])
    summary = {"approach": approach, "trials": len(records), "total_epochs": int(sum(r["epochs"] for r in records)),
               "wall_time": wall_time, "best_val_accuracy": max(r["val_accuracy"] for r in records),
               "chosen_trial": chosen["trial_id"], "chosen_val_accuracy": chosen["val_accuracy"],
               "chosen_params": chosen["params"]}
    if champion_dir is not None:
        Path(champion_dir).mkdir(parents=True, exist_ok=True)
        tuner.load_model(chosen["trial"]).save(Path(champion_dir) / f"{approach}.keras")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_imitation.py -q`, then `$PY -m pytest tests/test_imitation.py -q -m slow`
Expected: all pass. If `on_trial_end` or `load_model` behaves differently in keras-tuner 1.4.8, adapt `ReportingTuner` and keep the `on_trial(number, model, val_accuracy)` contract. Record the adaptation in the task report.

- [ ] **Step 5: Commit**

```bash
git add racing/imitation.py tests/test_imitation.py
git commit -m "Add baseline training and Keras Tuner searches with smallest-within-1-point selection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/imitation.py tests/test_imitation.py
```

---

### Task 6: Expert driver, validated tracks and dataset

**Files:**
- Create: `racing/expert.py`
- Test: `tests/test_expert.py` (also covers `racing/tracks.py` from Task 1)

**Interfaces:**
- Consumes: `sim.start_cars`, `sim.sense`, `sim.step`, `sim.simulate`, `actions`, `tracks.get_track` (Task 1, which calls `track_gen.candidate_track` and `expert.expert_lap_time`).
- Produces:
  - Expert driver:
    - `expert_controls(track, cars) -> (steer, throttle)`, continuous
    - `expert_policy(track) -> policy(obs, cars)`
    - `expert_lap_time(track) -> float | None`
  - Dataset:
    - `collect_rows(track, rng) -> (X (k, 10), y (k,))`
    - `default_splits() -> dict`
    - `collect_dataset(path=None, splits=None) -> dict`, written as `.npz`

- [ ] **Step 1: Write the failing tests** in `tests/test_expert.py`

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_expert.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'racing.expert'`.

- [ ] **Step 3: Implement `racing/expert.py`**

```python
"""Scripted expert driver (spec §7.1) and imitation dataset collection (spec §7.2-7.3)."""
from pathlib import Path

import numpy as np

from racing import config, sim
from racing.actions import class_to_controls, controls_to_class
from racing.tracks import get_track


def expert_controls(track, cars):
    """Continuous steering and throttle fractions chosen by the expert for every car."""
    m = len(track.centerline)
    here = np.mod(cars.index, m)
    look = config.EXPERT_LOOKAHEAD_BASE + config.EXPERT_LOOKAHEAD_TIME * cars.speed
    target = track.centerline[np.mod(here + np.round(look / track.spacing).astype(np.int64), m)]
    dx, dy = target[:, 0] - cars.x, target[:, 1] - cars.y
    alpha = np.angle(np.exp(1j * (np.arctan2(dy, dx) - cars.heading)))
    path_curvature = 2.0 * np.sin(alpha) / np.maximum(np.hypot(dx, dy), 1e-6)
    steer = np.clip(cars.speed ** 2 * path_curvature / config.GRIP, -1.0, 1.0)

    sight = int(round(config.EXPERT_SIGHT / track.spacing))
    window = np.mod(here[:, None] + np.arange(sight + 1)[None, :], m)
    kappa = np.max(np.abs(track.curvature[window]), axis=1)
    corner_speed = np.sqrt(config.EXPERT_GRIP_MARGIN * config.GRIP / np.maximum(kappa, 1e-9))
    sight_speed = np.sqrt(2.0 * config.EXPERT_GRIP_MARGIN * config.GRIP * config.EXPERT_SIGHT)
    target_speed = np.minimum(corner_speed, sight_speed)
    band = config.EXPERT_SPEED_BAND
    throttle = np.where(cars.speed < target_speed - band, 1.0,
                        np.where(cars.speed > target_speed + band, -1.0, 0.0))
    return steer, throttle


def expert_policy(track):
    """Policy for sim.simulate: the expert, quantized to the 15 action classes."""
    def policy(obs, cars):
        return controls_to_class(*expert_controls(track, cars))
    return policy


def expert_lap_time(track):
    """The expert's lap time on this track, or None if it crashes, stalls or runs out of time."""
    result = sim.simulate(track, expert_policy(track), 1, time_limit=config.EXPERT_VALIDATION_TIME)
    return float(result.finish_time[0]) if result.finished[0] else None


def collect_rows(track, rng):
    """One expert lap with random steering disturbances. Labels are always the expert's own choice."""
    cars = sim.start_cars(track, 1)
    next_kick = rng.uniform(*config.PERTURB_INTERVAL)
    kick_end, kick_steer = -1.0, 0.0
    inputs, labels = [], []
    while not cars.done[0]:
        obs = sim.sense(track, cars)
        label = controls_to_class(*expert_controls(track, cars))
        inputs.append(obs[0])
        labels.append(int(label[0]))
        steer, throttle = class_to_controls(label)
        if cars.time >= next_kick:
            kick_end = cars.time + rng.uniform(*config.PERTURB_DURATION)
            kick_steer = float(rng.choice(config.STEER_LEVELS))
            next_kick = kick_end + rng.uniform(*config.PERTURB_INTERVAL)
        if cars.time < kick_end:
            steer = np.array([kick_steer])
        sim.step(track, cars, steer, throttle, config.EXPERT_VALIDATION_TIME)
    return np.array(inputs), np.array(labels, dtype=np.int64)


def default_splits():
    train = [(seed, level) for level, seeds in config.TRAIN_DATA_TRACKS.items() for seed in seeds]
    return {"train": train,
            "val": [(seed, config.MAX_LEVEL) for seed in config.VAL_SEEDS],
            "test": [(seed, config.MAX_LEVEL) for seed in config.TEST_SEEDS]}


def collect_dataset(path=None, splits=None):
    """Collect the train/val/test rows (spec §7.3) and save them as a compressed .npz file."""
    path = Path(config.DATASET_PATH if path is None else path)
    splits = default_splits() if splits is None else splits
    data = {}
    for name, items in splits.items():
        xs, ys = [], []
        for seed, level in items:
            x, y = collect_rows(get_track(seed, level), np.random.default_rng([seed, level, config.DATASET_SALT]))
            xs.append(x)
            ys.append(y)
        data[f"X_{name}"] = np.concatenate(xs).astype(np.float32)
        data[f"y_{name}"] = np.concatenate(ys).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return data
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_expert.py -q`, then `$PY -m pytest tests/test_expert.py -q -m slow`
Expected: all pass. If the quantized expert fails tracks often, calibrate the `EXPERT_*` values and the `LEVELS` geometry in `config.py`:
- This task may edit those constants directly. Report every change.
- Do not change GRIP or any other constant.

Report three numbers:
- the expert's lap time on level 1
- the median lap time on seeds 0–19 at level 4
- how many level-4 attempts the expert rejected

- [ ] **Step 5: Commit**

Include `racing/config.py` in both file lists only if this task changed it.

```bash
git add racing/expert.py tests/test_expert.py
git commit -m "Add expert driver and dataset collection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/expert.py tests/test_expert.py
```

---

### Task 7: Shared evolutionary helpers

**Files:**
- Create: `racing/population.py`
- Test: `tests/test_population.py`

**Interfaces:**
- Consumes: `sim.simulate`, `nets.population_policy`, `nets.to_keras`, `tracks.get_track` (imported as the module global `get_track`, so tests can monkeypatch `racing.population.get_track`).
- Produces:
  - Evaluation:
    - `evaluate_population(weights, seeds, level, n_jobs=1, backend="loky") -> (fitness (n,), finished_all (n,))`
    - `finish_fractions(finished, is_kept) -> (kept_finish, new_finish)`, nan when a group is empty
  - Reports: `make_report(*, approach, generation, level, seeds, weights, fitness, finished, is_kept, kept_finish, new_finish, step_size, evals, elapsed, leveled_up) -> dict`
  - Logging: `RunLogger(out_dir)`
    - `.log(report)` writes one row to `log.csv` and saves `best_genNNNN.npy`
    - `.summary(dict)` writes `summary.json`
  - Final selection:
    - `select_champion(candidates, n_jobs=1, seeds=VAL_SEEDS) -> (weights, val_fitness)`
    - `save_champion(approach, flat, champion_dir=CHAMPIONS_DIR) -> Path`
- Report dict keys (consumed by the app in Task 12): `approach, generation, level, seeds, weights, hidden, activation, fitness, finished, is_kept, kept_finish, new_finish, step_size, best, mean, worst, evals, elapsed, leveled_up`.

- [ ] **Step 1: Write the failing tests** in `tests/test_population.py`

```python
import math

import numpy as np
import pandas as pd
import pytest

from racing import config, nets, population


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    return circle_track


def test_evaluate_population_shapes(fake_tracks):
    w = nets.init_population(np.random.default_rng(0), 6)
    fitness, finished = population.evaluate_population(w, [0, 1], 1)
    assert fitness.shape == (6,) and finished.shape == (6,) and finished.dtype == bool
    assert np.all((fitness >= 0) & (fitness < 2))


def test_parallel_evaluation_matches_serial(fake_tracks):
    w = nets.init_population(np.random.default_rng(1), 9)
    serial = population.evaluate_population(w, [0], 1)
    threaded = population.evaluate_population(w, [0], 1, n_jobs=3, backend="threading")
    assert np.array_equal(serial[0], threaded[0]) and np.array_equal(serial[1], threaded[1])


def test_resimulating_a_subset_reproduces_logged_fitness(fake_tracks):
    w = nets.init_population(np.random.default_rng(2), 10)
    full, _ = population.evaluate_population(w, [0], 1)
    part, _ = population.evaluate_population(w[[1, 7]], [0], 1)
    assert np.array_equal(full[[1, 7]], part)


def test_finish_fractions():
    finished = np.array([True, True, False, True])
    kept = np.array([True, True, False, False])
    assert population.finish_fractions(finished, kept) == (1.0, 0.5)
    k, n = population.finish_fractions(finished, np.zeros(4, dtype=bool))
    assert math.isnan(k) and n == 0.75


def report(generation, n=5):
    rng = np.random.default_rng(generation)
    w = nets.init_population(rng, n)
    fitness = rng.random(n)
    return population.make_report(approach="ga", generation=generation, level=1, seeds=[0, 1, 2], weights=w,
                                  fitness=fitness, finished=fitness > 0.5, is_kept=np.zeros(n, dtype=bool),
                                  kept_finish=float("nan"), new_finish=0.4, step_size=0.5, evals=n * (generation + 1),
                                  elapsed=1.0, leveled_up=False)


def test_make_report_fields():
    r = report(0)
    assert r["best"] == r["fitness"].max() and r["worst"] == r["fitness"].min()
    assert r["hidden"] == list(config.BASELINE_HIDDEN) and r["activation"] == "relu"


def test_logger_stays_readable_after_each_generation(tmp_path):
    logger = population.RunLogger(tmp_path)
    for g in range(2):
        r = report(g)
        logger.log(r)
        log = pd.read_csv(tmp_path / "log.csv")
        assert len(log) == g + 1
        best = np.load(tmp_path / f"best_gen{g:04d}.npy")
        assert np.array_equal(best, r["weights"][np.argmax(r["fitness"])])
    logger.summary({"generations": 2})
    assert (tmp_path / "summary.json").exists()


def test_select_champion_picks_highest_validation_fitness(monkeypatch):
    candidates = np.arange(12.0).reshape(4, 3)
    monkeypatch.setattr(population, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.array([0.2, 1.4, 0.9, 1.1]), np.ones(4, bool)))
    best, value = population.select_champion(candidates)
    assert np.array_equal(best, candidates[1]) and value == 1.4


def test_save_champion(tmp_path):
    w = nets.init_population(np.random.default_rng(3), 1)[0]
    path = population.save_champion("ga", w, tmp_path)
    assert path == tmp_path / "ga.keras" and path.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_population.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'racing.population'`.

- [ ] **Step 3: Implement `racing/population.py`**

```python
"""Shared helpers for the evolutionary approaches (spec §9.1): evaluation, reports, logging, final selection."""
import csv
import json
from pathlib import Path

import numpy as np

from racing import config, nets, sim
from racing.tracks import get_track


def _evaluate_chunk(weights, seeds, level):
    total = np.zeros(len(weights))
    finished_all = np.ones(len(weights), dtype=bool)
    policy = nets.population_policy(weights)
    for seed in seeds:
        result = sim.simulate(get_track(seed, level), policy, len(weights))
        total += result.fitness
        finished_all &= result.finished
    return total / len(seeds), finished_all


def evaluate_population(weights, seeds, level, n_jobs=1, backend="loky"):
    """Mean fitness over the tracks and whether each individual finished all of them. Independent of n_jobs."""
    if n_jobs == 1 or len(weights) < 2 * n_jobs:
        return _evaluate_chunk(weights, seeds, level)
    from joblib import Parallel, delayed
    chunks = np.array_split(np.arange(len(weights)), n_jobs)
    parts = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_evaluate_chunk)(weights[c], seeds, level) for c in chunks)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def finish_fractions(finished, is_kept):
    """Fraction of kept and of new individuals that finished; nan for an empty group."""
    kept = float(finished[is_kept].mean()) if is_kept.any() else float("nan")
    new = float(finished[~is_kept].mean()) if (~is_kept).any() else float("nan")
    return kept, new


def make_report(*, approach, generation, level, seeds, weights, fitness, finished, is_kept,
                kept_finish, new_finish, step_size, evals, elapsed, leveled_up):
    """One generation's results, for logging and for the training view."""
    return {"approach": approach, "generation": int(generation), "level": int(level), "seeds": list(seeds),
            "weights": weights, "hidden": list(config.BASELINE_HIDDEN), "activation": "relu",
            "fitness": fitness, "finished": finished, "is_kept": is_kept,
            "kept_finish": kept_finish, "new_finish": new_finish, "step_size": float(step_size),
            "best": float(fitness.max()), "mean": float(fitness.mean()), "worst": float(fitness.min()),
            "evals": int(evals), "elapsed": float(elapsed), "leveled_up": bool(leveled_up)}


class RunLogger:
    """Writes one CSV row and the best network after every generation, and a summary at the end."""
    FIELDS = ["generation", "level", "evals", "best", "mean", "worst", "kept_finish", "new_finish",
              "step_size", "seeds", "elapsed", "leveled_up"]

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "log.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)

    def log(self, report):
        row = [" ".join(str(s) for s in report["seeds"]) if key == "seeds" else report[key] for key in self.FIELDS]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        best = report["weights"][int(np.argmax(report["fitness"]))]
        np.save(self.dir / f"best_gen{report['generation']:04d}.npy", best)

    def summary(self, data):
        (self.dir / "summary.json").write_text(json.dumps(data, indent=2))


def select_champion(candidates, n_jobs=1, seeds=config.VAL_SEEDS):
    """The candidate with the highest mean fitness on the validation tracks (level 4)."""
    fitness, _ = evaluate_population(candidates, list(seeds), config.MAX_LEVEL, n_jobs=n_jobs)
    best = int(np.argmax(fitness))
    return candidates[best], float(fitness[best])


def save_champion(approach, flat, champion_dir=config.CHAMPIONS_DIR):
    directory = Path(champion_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{approach}.keras"
    nets.to_keras(flat).save(path)
    return path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_population.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add racing/population.py tests/test_population.py
git commit -m "Add shared evolutionary helpers: evaluation, reports, logging and final selection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/population.py tests/test_population.py
```

---

### Task 8: Genetic algorithm

**Files:**
- Create: `racing/ga.py`
- Test: `tests/test_ga.py`

**Interfaces:**
- Consumes:
  - `nets.init_population`
  - `curriculum.LevelTracker`, `curriculum.draw_seeds`, `curriculum.mutation_strength`
  - `population.evaluate_population`, `RunLogger`, `finish_fractions`, `make_report`, `select_champion`, `save_champion`. These are imported into `racing.ga`; tests monkeypatch `racing.ga.evaluate_population`.
- Produces:
  - `tournament(rng, fitness, count, size=GA_TOURNAMENT) -> indices`
  - `next_generation(rng, weights, fitness, kept, strength) -> (weights, is_kept)`. The kept individuals come first.
  - `run_ga(pop_size=GA_POP, kept=GA_KEPT, budget=EVAL_BUDGET, seed=0, out_dir=None, on_generation=None, n_jobs=1, champion_dir=CHAMPIONS_DIR) -> dict`. Keys: `approach, generations, evals, level_reached, wall_time, pop_size, kept, seed`, plus `val_fitness, champion_path` when a champion is saved.

- [ ] **Step 1: Write the failing tests** in `tests/test_ga.py`

```python
import math

import numpy as np
import pytest

from racing import config, ga, population
from racing.curriculum import mutation_strength


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)


def test_tournament_prefers_fitter_individuals():
    winners = ga.tournament(np.random.default_rng(0), np.arange(100.0), 1000)
    assert winners.mean() > 75


def test_next_generation_keeps_elite_first_and_unchanged():
    rng = np.random.default_rng(1)
    weights = rng.normal(size=(30, 8))
    fitness = rng.random(30)
    new, is_kept = ga.next_generation(rng, weights, fitness, 5, 0.1)
    top = weights[np.argsort(-fitness, kind="stable")[:5]]
    assert new.shape == weights.shape and is_kept.sum() == 5 and np.all(is_kept[:5])
    assert np.array_equal(new[:5], top)


def test_zero_strength_children_only_recombine_parent_genes():
    rng = np.random.default_rng(2)
    weights = rng.normal(size=(20, 6))
    new, _ = ga.next_generation(rng, weights, rng.random(20), 2, 0.0)
    for j in range(6):
        assert set(new[2:, j]).issubset(set(weights[:, j]))


def test_operators_solve_the_sphere_function():
    rng = np.random.default_rng(3)
    weights = rng.normal(size=(100, 20))
    start = np.min(np.sum(weights ** 2, axis=1))
    for _ in range(80):
        fitness = -np.sum(weights ** 2, axis=1)
        weights, _ = ga.next_generation(rng, weights, fitness, 5, 0.05)
    assert np.min(np.sum(weights ** 2, axis=1)) < start / 10


def test_population_must_exceed_kept(tmp_path):
    with pytest.raises(ValueError, match="larger than"):
        ga.run_ga(pop_size=40, kept=50, budget=1000, out_dir=tmp_path, champion_dir=None)


def test_budget_must_cover_one_generation(tmp_path):
    with pytest.raises(ValueError, match="smaller than one generation"):
        ga.run_ga(pop_size=20, kept=5, budget=10, out_dir=tmp_path, champion_dir=None)


def test_small_run_reports_every_generation(fake_tracks, tmp_path):
    reports = []
    result = ga.run_ga(pop_size=20, kept=5, budget=60, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert result["generations"] == len(reports) == 3 and result["evals"] == 60
    assert reports[0]["step_size"] == config.GA_S_MAX
    assert reports[1]["is_kept"].sum() == 5
    assert math.isclose(reports[1]["step_size"], mutation_strength(reports[0]["new_finish"]))
    assert (tmp_path / "log.csv").exists() and (tmp_path / "summary.json").exists()


def test_mutation_strength_follows_new_finish_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.arange(len(w), dtype=float), np.arange(len(w)) % 2 == 0))
    reports = []
    ga.run_ga(pop_size=20, kept=5, budget=40, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert math.isclose(reports[1]["step_size"], mutation_strength(0.5))


def test_level_advances_after_three_qualifying_generations(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.ones(len(w)), np.ones(len(w), dtype=bool)))
    reports = []
    ga.run_ga(pop_size=20, kept=5, budget=100, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert [r["level"] for r in reports] == [1, 1, 1, 1, 2]       # generation 0 has no kept individuals
    assert reports[3]["leveled_up"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_ga.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'racing.ga'`.

- [ ] **Step 3: Implement `racing/ga.py`**

```python
"""Genetic algorithm over network weights with finish-rate mutation control (spec §9.2)."""
import time
from pathlib import Path

import numpy as np

from racing import config, nets
from racing.curriculum import LevelTracker, draw_seeds, mutation_strength
from racing.population import (RunLogger, evaluate_population, finish_fractions, make_report,
                               save_champion, select_champion)


def tournament(rng, fitness, count, size=config.GA_TOURNAMENT):
    """Indices of `count` winners, each the fittest of `size` individuals drawn from the whole population."""
    entrants = rng.integers(0, len(fitness), (count, size))
    return entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]


def next_generation(rng, weights, fitness, kept, strength):
    """The `kept` fittest unchanged (first), then children made by uniform crossover and Gaussian mutation."""
    n, p = weights.shape
    elite = weights[np.argsort(-fitness, kind="stable")[:kept]]
    n_new = n - kept
    mother = weights[tournament(rng, fitness, n_new)]
    father = weights[tournament(rng, fitness, n_new)]
    children = np.where(rng.random((n_new, p)) < config.GA_CROSSOVER_P, mother, father)
    mutate = rng.random((n_new, p)) < config.GA_MUTATION_P
    children = children + mutate * rng.normal(0.0, strength, (n_new, p))
    is_kept = np.concatenate([np.ones(kept, dtype=bool), np.zeros(n_new, dtype=bool)])
    return np.concatenate([elite, children]), is_kept


def run_ga(pop_size=config.GA_POP, kept=config.GA_KEPT, budget=config.EVAL_BUDGET, seed=0, out_dir=None,
           on_generation=None, n_jobs=1, champion_dir=config.CHAMPIONS_DIR):
    if pop_size <= kept:
        raise ValueError(f"population size ({pop_size}) must be larger than the number of kept individuals ({kept})")
    if budget < pop_size:
        raise ValueError(f"budget ({budget}) is smaller than one generation ({pop_size} evaluations)")
    rng = np.random.default_rng(seed)
    logger = RunLogger(Path(out_dir or config.RUNS_DIR / "ga"))
    tracker = LevelTracker()
    weights = nets.init_population(rng, pop_size)
    is_kept = np.zeros(pop_size, dtype=bool)
    strength = config.GA_S_MAX
    evals, generation, start = 0, 0, time.time()
    while evals + pop_size <= budget:
        level = tracker.level
        seeds = draw_seeds(rng)
        fitness, finished = evaluate_population(weights, seeds, level, n_jobs=n_jobs)
        evals += pop_size
        kept_finish, new_finish = finish_fractions(finished, is_kept)
        leveled_up = tracker.update(kept_finish, new_finish)
        report = make_report(approach="ga", generation=generation, level=level, seeds=seeds, weights=weights,
                             fitness=fitness, finished=finished, is_kept=is_kept, kept_finish=kept_finish,
                             new_finish=new_finish, step_size=strength, evals=evals,
                             elapsed=time.time() - start, leveled_up=leveled_up)
        logger.log(report)
        if on_generation is not None:
            on_generation(report)
        strength = mutation_strength(new_finish)
        weights, is_kept = next_generation(rng, weights, fitness, kept, strength)
        generation += 1
    result = {"approach": "ga", "generations": generation, "evals": evals, "level_reached": tracker.level,
              "wall_time": time.time() - start, "pop_size": pop_size, "kept": kept, "seed": seed}
    if champion_dir is not None:
        champion, val_fitness = select_champion(weights[is_kept], n_jobs=n_jobs)
        result["val_fitness"] = val_fitness
        result["champion_path"] = str(save_champion("ga", champion, champion_dir))
    logger.summary(result)
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_ga.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add racing/ga.py tests/test_ga.py
git commit -m "Add genetic algorithm with finish-rate mutation control and curriculum

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/ga.py tests/test_ga.py
```

---

### Task 9: Evolution strategy

**Files:**
- Create: `racing/es.py`
- Test: `tests/test_es.py`

**Interfaces:**
- Consumes: the same names as Task 8, except `mutation_strength`. Tests monkeypatch `racing.es.evaluate_population`.
- Produces:
  - `make_offspring(rng, parents, sigma, lam) -> (offspring, parent_of)`
  - `select_parents(population, fitness, mu) -> parents`
  - `one_fifth_rule(sigma, success_rate) -> float`
  - `run_es(mu=ES_MU, lam=ES_LAMBDA, budget=EVAL_BUDGET, seed=0, out_dir=None, on_generation=None, n_jobs=1, champion_dir=CHAMPIONS_DIR) -> dict`, with the same keys as `run_ga` except that `mu, lam` replace `pop_size, kept`.

- [ ] **Step 1: Write the failing tests** in `tests/test_es.py`

```python
import math

import numpy as np
import pytest

from racing import config, es, population


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)


def test_one_fifth_rule():
    assert math.isclose(es.one_fifth_rule(0.1, 0.3), 0.1 / config.ES_FACTOR)
    assert math.isclose(es.one_fifth_rule(0.1, 0.1), 0.1 * config.ES_FACTOR)
    assert es.one_fifth_rule(0.1, 0.2) == 0.1
    assert es.one_fifth_rule(config.ES_SIGMA_MAX, 1.0) == config.ES_SIGMA_MAX
    assert es.one_fifth_rule(config.ES_SIGMA_MIN, 0.0) == config.ES_SIGMA_MIN


def test_offspring_come_from_their_parents():
    rng = np.random.default_rng(0)
    parents = rng.normal(size=(4, 5))
    offspring, parent_of = es.make_offspring(rng, parents, 1e-9, 12)
    assert offspring.shape == (12, 5)
    assert np.allclose(offspring, parents[parent_of], atol=1e-6)


def test_operators_solve_the_sphere_function():
    rng = np.random.default_rng(1)
    parents = rng.normal(size=(10, 20))
    sigma = 0.3
    start = np.min(np.sum(parents ** 2, axis=1))
    for _ in range(80):
        offspring, parent_of = es.make_offspring(rng, parents, sigma, 70)
        pop = np.concatenate([parents, offspring])
        fitness = -np.sum(pop ** 2, axis=1)
        sigma = es.one_fifth_rule(sigma, float(np.mean(fitness[10:] > fitness[:10][parent_of])))
        parents = es.select_parents(pop, fitness, 10)
    assert np.min(np.sum(parents ** 2, axis=1)) < start / 10


def test_budget_must_cover_one_generation(tmp_path):
    with pytest.raises(ValueError, match="smaller than one generation"):
        es.run_es(mu=4, lam=16, budget=10, out_dir=tmp_path, champion_dir=None)


def test_small_run_reports_every_generation(fake_tracks, tmp_path):
    reports = []
    result = es.run_es(mu=4, lam=16, budget=60, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert result["generations"] == len(reports) == 3 and result["evals"] == 60
    assert reports[0]["step_size"] == config.ES_SIGMA0
    assert all(r["is_kept"].sum() == 4 for r in reports)
    assert (tmp_path / "log.csv").exists()


def test_sigma_follows_the_one_fifth_rule(monkeypatch, tmp_path):
    def fake(w, seeds, level, n_jobs=1):
        fitness = np.zeros(len(w))
        fitness[4:] = 1.0                                   # every offspring beats its parent
        return fitness, np.zeros(len(w), dtype=bool)
    monkeypatch.setattr(es, "evaluate_population", fake)
    reports = []
    es.run_es(mu=4, lam=16, budget=40, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert math.isclose(reports[1]["step_size"], config.ES_SIGMA0 / config.ES_FACTOR)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_es.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'racing.es'`.

- [ ] **Step 3: Implement `racing/es.py`**

```python
"""(μ+λ) evolution strategy over network weights with the 1/5 success rule (spec §9.3)."""
import time
from pathlib import Path

import numpy as np

from racing import config, nets
from racing.curriculum import LevelTracker, draw_seeds
from racing.population import (RunLogger, evaluate_population, finish_fractions, make_report,
                               save_champion, select_champion)


def make_offspring(rng, parents, sigma, lam):
    """λ offspring, each a random parent plus N(0, σ²) on every weight; also returns each one's parent index."""
    parent_of = rng.integers(0, len(parents), lam)
    return parents[parent_of] + rng.normal(0.0, sigma, (lam, parents.shape[1])), parent_of


def select_parents(population, fitness, mu):
    """The μ fittest of parents ∪ offspring."""
    return population[np.argsort(-fitness, kind="stable")[:mu]]


def one_fifth_rule(sigma, success_rate):
    """Grow σ when more than 1/5 of offspring beat their parent, shrink it when fewer do."""
    if success_rate > 0.2:
        sigma = sigma / config.ES_FACTOR
    elif success_rate < 0.2:
        sigma = sigma * config.ES_FACTOR
    return float(np.clip(sigma, config.ES_SIGMA_MIN, config.ES_SIGMA_MAX))


def run_es(mu=config.ES_MU, lam=config.ES_LAMBDA, budget=config.EVAL_BUDGET, seed=0, out_dir=None,
           on_generation=None, n_jobs=1, champion_dir=config.CHAMPIONS_DIR):
    if budget < mu + lam:
        raise ValueError(f"budget ({budget}) is smaller than one generation (μ + λ = {mu + lam} evaluations)")
    rng = np.random.default_rng(seed)
    logger = RunLogger(Path(out_dir or config.RUNS_DIR / "es"))
    tracker = LevelTracker()
    parents = nets.init_population(rng, mu)
    sigma = config.ES_SIGMA0
    evals, generation, start = 0, 0, time.time()
    while evals + mu + lam <= budget:
        level = tracker.level
        seeds = draw_seeds(rng)
        offspring, parent_of = make_offspring(rng, parents, sigma, lam)
        individuals = np.concatenate([parents, offspring])
        is_kept = np.arange(mu + lam) < mu
        fitness, finished = evaluate_population(individuals, seeds, level, n_jobs=n_jobs)
        evals += mu + lam
        kept_finish, new_finish = finish_fractions(finished, is_kept)
        leveled_up = tracker.update(kept_finish, new_finish)
        report = make_report(approach="es", generation=generation, level=level, seeds=seeds, weights=individuals,
                             fitness=fitness, finished=finished, is_kept=is_kept, kept_finish=kept_finish,
                             new_finish=new_finish, step_size=sigma, evals=evals,
                             elapsed=time.time() - start, leveled_up=leveled_up)
        logger.log(report)
        if on_generation is not None:
            on_generation(report)
        sigma = one_fifth_rule(sigma, float(np.mean(fitness[mu:] > fitness[:mu][parent_of])))
        parents = select_parents(individuals, fitness, mu)
        generation += 1
    result = {"approach": "es", "generations": generation, "evals": evals, "level_reached": tracker.level,
              "wall_time": time.time() - start, "mu": mu, "lam": lam, "seed": seed}
    if champion_dir is not None:
        champion, val_fitness = select_champion(parents, n_jobs=n_jobs)
        result["val_fitness"] = val_fitness
        result["champion_path"] = str(save_champion("es", champion, champion_dir))
    logger.summary(result)
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_es.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add racing/es.py tests/test_es.py
git commit -m "Add (mu+lambda) evolution strategy with the 1/5 success rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/es.py tests/test_es.py
```

---

### Task 10: Evaluation, tables and figures

**Files:**
- Create: `racing/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `nets.load_champions`, `nets.from_keras`, `nets.to_keras`, `sim.simulate`, `tracks.get_track` (module global `get_track`, monkeypatchable), the run files from Tasks 5, 8 and 9 (`summary.json`, `log.csv`, `trials.csv`).
- Produces:
  - Measurements:
    - `driving_results(dense_nets, seeds=TEST_SEEDS) -> DataFrame[approach, seed, finished, crashed, lap_time, progress]`
    - `accuracy(model, X, y) -> float`
    - `inference_time(model, X, repeats=5) -> float`, in seconds per sample
    - `read_summary(runs_dir, approach) -> dict`
  - Tables:
    - `comparison_table(models, data, runs_dir) -> DataFrame`
    - `driving_table(results, runs_dir) -> DataFrame`
    - `to_markdown(frame) -> str`
  - Figures: `figures(runs_dir, out_dir) -> list[Path]`
  - Entry point: `main(champions_dir=None, runs_dir=None, data_path=None, out_dir=None, test_seeds=TEST_SEEDS) -> dict`

- [ ] **Step 1: Write the failing tests** in `tests/test_evaluate.py`

```python
import json

import numpy as np
import pandas as pd
import pytest

from racing import config, evaluate, nets


@pytest.fixture
def models():
    rng = np.random.default_rng(0)
    return {name: nets.to_keras(nets.init_population(rng, 1)[0]) for name in ("baseline", "ga")}


def test_driving_results_rows(monkeypatch, circle_track, models):
    monkeypatch.setattr(evaluate, "get_track", lambda seed, level: circle_track)
    dense = {name: nets.from_keras(m) for name, m in models.items()}
    frame = evaluate.driving_results(dense, seeds=[1000, 1001])
    assert len(frame) == 4
    assert list(frame.columns) == ["approach", "seed", "finished", "crashed", "lap_time", "progress"]


def test_accuracy_matches_manual(models):
    x = np.random.default_rng(1).random((100, config.N_INPUTS)).astype(np.float32)
    predicted = np.argmax(models["ga"].predict(x, verbose=0), axis=1)
    y = predicted.copy()
    y[:25] = (y[:25] + 1) % config.N_ACTIONS
    assert abs(evaluate.accuracy(models["ga"], x, y) - 0.75) < 1e-9


def test_inference_time_is_positive(models):
    x = np.random.default_rng(2).random((500, config.N_INPUTS)).astype(np.float32)
    assert 0 < evaluate.inference_time(models["ga"], x, repeats=2) < 0.01


def test_tables_and_missing_summaries(tmp_path, models):
    (tmp_path / "ga").mkdir()
    (tmp_path / "ga" / "summary.json").write_text(json.dumps({"wall_time": 12.0, "level_reached": 3}))
    rng = np.random.default_rng(3)
    data = {"X_test": rng.random((50, config.N_INPUTS)).astype(np.float32), "y_test": rng.integers(0, 15, 50)}
    comparison = evaluate.comparison_table(models, data, tmp_path)
    assert list(comparison["Method"]) == ["Baseline", "GA"]
    assert np.isnan(comparison.loc[0, "Search/training time (s)"])
    results = pd.DataFrame([
        {"approach": "baseline", "seed": 1000, "finished": True, "crashed": False, "lap_time": 50.0, "progress": 1.0},
        {"approach": "ga", "seed": 1000, "finished": False, "crashed": True, "lap_time": np.nan, "progress": 0.3}])
    driving = evaluate.driving_table(results, tmp_path)
    assert list(driving["Level reached"]) == ["n/a", 3]
    assert "| Method |" in evaluate.to_markdown(driving)


def test_figures_with_partial_logs(tmp_path):
    runs = tmp_path / "runs"
    (runs / "ga").mkdir(parents=True)
    pd.DataFrame({"generation": [0, 1, 2], "level": [1, 1, 2], "evals": [10, 20, 30], "best": [0.1, 0.5, 1.2],
                  "mean": [0.05, 0.2, 0.6], "worst": [0, 0, 0], "kept_finish": [np.nan, 1, 1],
                  "new_finish": [0, 0.5, 0.9], "step_size": [0.5, 0.25, 0.05], "seeds": ["0 1 2"] * 3,
                  "elapsed": [1, 2, 3], "leveled_up": [False, True, False]}).to_csv(runs / "ga" / "log.csv", index=False)
    written = evaluate.figures(runs, tmp_path / "figures")
    assert [p.name for p in written] == ["fitness_curves.png", "step_sizes.png"]


def test_main_without_champions_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate.main(champions_dir=tmp_path, runs_dir=tmp_path, data_path=tmp_path / "none.npz", out_dir=tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_evaluate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'racing.evaluate'`.

- [ ] **Step 3: Implement `racing/evaluate.py`**

```python
"""Final evaluation (spec §11): driving on held-out tracks, test accuracy, size and speed, tables and figures."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from racing import config, nets, sim
from racing.tracks import get_track


def driving_results(dense_nets, seeds=config.TEST_SEEDS):
    """One lap per controller per test track."""
    rows = []
    for seed in seeds:
        track = get_track(seed, config.MAX_LEVEL)
        for approach, net in dense_nets.items():
            result = sim.simulate(track, net.policy(), 1)
            rows.append({"approach": approach, "seed": seed, "finished": bool(result.finished[0]),
                         "crashed": bool(result.crashed[0]), "lap_time": float(result.finish_time[0]),
                         "progress": float(result.progress[0])})
    return pd.DataFrame(rows, columns=["approach", "seed", "finished", "crashed", "lap_time", "progress"])


def accuracy(model, x, y):
    predicted = np.argmax(model.predict(x, batch_size=4096, verbose=0), axis=1)
    return float(np.mean(predicted == y))


def inference_time(model, x, repeats=5):
    """Seconds per sample for Keras predict on up to 10,000 rows, averaged over `repeats` runs."""
    x = x[:10000]
    model.predict(x[:256], verbose=0)                  # warm-up builds the prediction function
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(x, batch_size=len(x), verbose=0)
        times.append(time.perf_counter() - start)
    return float(np.mean(times) / len(x))


def read_summary(runs_dir, approach):
    path = Path(runs_dir) / approach / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def comparison_table(models, data, runs_dir):
    """Spec Table 4, required comparison."""
    rows = []
    for approach, model in models.items():
        rows.append({"Method": config.LABELS[approach],
                     "Test accuracy": round(accuracy(model, data["X_test"], data["y_test"]), 4),
                     "Parameters": int(model.count_params()),
                     "Inference time (µs/sample)": round(inference_time(model, data["X_test"]) * 1e6, 3),
                     "Search/training time (s)": round(read_summary(runs_dir, approach).get("wall_time", float("nan")), 1)})
    return pd.DataFrame(rows)


def driving_table(results, runs_dir):
    """Spec Table 4, driving performance."""
    rows = []
    n_tracks = results["seed"].nunique()
    for approach in config.APPROACHES:
        sub = results[results["approach"] == approach]
        if sub.empty:
            continue
        done = sub[sub["finished"]]
        level = read_summary(runs_dir, approach).get("level_reached", "n/a") if approach in ("ga", "es") else "n/a"
        rows.append({"Method": config.LABELS[approach],
                     f"Tracks completed (of {n_tracks})": int(sub["finished"].sum()),
                     "Mean lap time (s)": round(float(done["lap_time"].mean()), 2) if len(done) else float("nan"),
                     "Crashes": int(sub["crashed"].sum()),
                     "Level reached": level})
    return pd.DataFrame(rows)


def to_markdown(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def figures(runs_dir, out_dir):
    """Fitness curves, step sizes and Keras Tuner trials; only the figures whose data exists are written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    logs = {a: pd.read_csv(runs_dir / a / "log.csv") for a in ("ga", "es") if (runs_dir / a / "log.csv").exists()}
    if logs:
        fig, ax = plt.subplots(figsize=(7, 4))
        for approach, log in logs.items():
            color = config.COLORS[approach]
            ax.plot(log["evals"], log["best"], color=color, label=f"{config.LABELS[approach]} best")
            ax.plot(log["evals"], log["mean"], color=color, alpha=0.45, linestyle="--",
                    label=f"{config.LABELS[approach]} mean")
            for evals in log.loc[log["leveled_up"].astype(bool), "evals"]:
                ax.axvline(evals, color=color, alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Individual evaluations")
        ax.set_ylabel("Fitness")
        ax.legend(frameon=False)
        fig.tight_layout()
        written.append(out_dir / "fitness_curves.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        for approach, log in logs.items():
            name = "GA mutation strength" if approach == "ga" else "ES step size σ"
            ax.plot(log["generation"], log["step_size"], color=config.COLORS[approach], label=name)
        ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.legend(frameon=False)
        fig.tight_layout()
        written.append(out_dir / "step_sizes.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

    tuners = ("kt_random", "kt_hyperband", "kt_bayesian")
    trials = {a: pd.read_csv(runs_dir / a / "trials.csv") for a in tuners if (runs_dir / a / "trials.csv").exists()}
    if trials:
        fig, ax = plt.subplots(figsize=(7, 4))
        for approach, table in trials.items():
            color = config.COLORS[approach]
            ax.scatter(table["params"], table["val_accuracy"], color=color, s=18, alpha=0.7,
                       label=config.LABELS[approach])
            chosen = str(read_summary(runs_dir, approach).get("chosen_trial"))
            pick = table[table["trial_id"].astype(str) == chosen]
            ax.scatter(pick["params"], pick["val_accuracy"], color=color, s=140, marker="*",
                       edgecolor="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Validation accuracy")
        ax.legend(frameon=False)
        fig.tight_layout()
        written.append(out_dir / "kt_trials.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)
    return written


def main(champions_dir=None, runs_dir=None, data_path=None, out_dir=None, test_seeds=config.TEST_SEEDS):
    champions_dir = Path(champions_dir or config.CHAMPIONS_DIR)
    runs_dir = Path(runs_dir or config.RUNS_DIR)
    data_path = Path(data_path or config.DATASET_PATH)
    out_dir = Path(out_dir or config.RESULTS_DIR)
    models = nets.load_champions(champions_dir)
    if not models:
        raise FileNotFoundError(f"no trained controllers found in {champions_dir}")
    with np.load(data_path) as d:
        data = {key: d[key] for key in d.files}
    per_track = driving_results({a: nets.from_keras(m) for a, m in models.items()}, test_seeds)
    comparison = comparison_table(models, data, runs_dir)
    driving = driving_table(per_track, runs_dir)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    per_track.to_csv(tables / "driving_per_track.csv", index=False)
    for name, frame in (("comparison", comparison), ("driving", driving)):
        frame.to_csv(tables / f"{name}.csv", index=False)
        (tables / f"{name}.md").write_text(to_markdown(frame))
    return {"comparison": comparison, "driving": driving, "per_track": per_track,
            "figures": figures(runs_dir, out_dir / "figures")}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_evaluate.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add racing/evaluate.py tests/test_evaluate.py
git commit -m "Add final evaluation: test-track driving, accuracy, inference time, tables and figures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/evaluate.py tests/test_evaluate.py
```

---

### Task 11: Generation-time benchmark

**Files:**
- Create: `racing/bench.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `expert.expert_policy`, `sim.simulate`, `tracks.get_track`, `joblib`.
- Produces: `time_full_laps(seed, n) -> float` (seconds), `estimate(pop, jobs) -> float` and a printed report. The orchestrator sets `config.N_JOBS` from the result.

- [ ] **Step 1: Write the failing test** in `tests/test_bench.py`

```python
from racing import bench


def test_time_full_laps_runs_on_a_small_batch():
    assert bench.time_full_laps(0, 5) > 0.0


def test_estimate_with_two_processes():
    assert bench.estimate(8, 2) > 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_bench.py -q`
Expected: FAIL with `ImportError: cannot import name 'bench'`.

- [ ] **Step 3: Implement `racing/bench.py`**

```python
"""Measures the time of one evolutionary generation (spec §9.4).

Every car drives the expert policy, so all of them complete a full lap: the worst case for runtime.
Run with:  ~/miniconda3/bin/python -m racing.bench
"""
import time

from joblib import Parallel, delayed

from racing import config, expert, sim
from racing.tracks import get_track

SEEDS = (0, 1, 2)


def time_full_laps(seed, n):
    """Seconds for n cars to each drive one full level-4 lap on the given training seed."""
    track = get_track(seed, config.MAX_LEVEL)
    start = time.perf_counter()
    sim.simulate(track, expert.expert_policy(track), n)
    return time.perf_counter() - start


def estimate(pop, jobs):
    """Wall time for one generation of `pop` cars on three tracks, split across `jobs` processes."""
    if jobs == 1:
        return sum(time_full_laps(seed, pop) for seed in SEEDS)
    chunk = max(pop // jobs, 1)
    tasks = [delayed(time_full_laps)(seed, chunk) for seed in SEEDS for _ in range(jobs)]
    Parallel(n_jobs=jobs)(tasks[:jobs])                    # warm-up: each worker builds its tracks
    start = time.perf_counter()
    Parallel(n_jobs=jobs)(tasks)
    return time.perf_counter() - start


def main():
    for seed in SEEDS:
        get_track(seed, config.MAX_LEVEL)
    print(f"One generation of {config.GA_POP} cars, every car completing a lap on 3 level-4 tracks:")
    for jobs in (1, 4, 8, 12, 16):
        print(f"  {jobs:2d} process(es): {estimate(config.GA_POP, jobs):6.1f} s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then the benchmark**

Run: `$PY -m pytest tests/test_bench.py -q`, then `$PY -m racing.bench`
Expected: the test passes and the script prints five timings. Report the timings. The orchestrator sets `config.N_JOBS` to the fastest process count and records the 200-generation estimate in the task report.

- [ ] **Step 5: Commit**

```bash
git add racing/bench.py tests/test_bench.py
git commit -m "Add generation-time benchmark

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/bench.py tests/test_bench.py
```

---

### Task 12: Visualization application

**Files:**
- Create: `racing/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes:
  - from the simulator and networks: `sim.start_cars`, `sim.sense`, `sim.step`, `sim.time_limit_for`, `sim.cast_rays`, `nets.population_policy`, `nets.load_champions`, `nets.from_keras`, `actions.class_to_controls`
  - `tracks.get_track` (module global `get_track`, monkeypatchable)
  - the report dicts from Task 7
  - `ga.run_ga`, `es.run_es`, `imitation.train_baseline`, `imitation.run_tuner`, `expert.collect_dataset`. These are imported inside `trainer_main` only.
- Produces:
  - Pure helpers: `hex_to_rgb(code)`, `select_indices(fitness, mode, count, sample_seed=0)`, `Camera(track, rect)`, `car_polygon(sx, sy, heading, size)`, `leaderboard(playback, names)`
  - Simulation and policies: `Playback(track, policy, n, laps=1, time_limit=None)`, `report_policy(report, indices)`, `mixed_policy(dense_nets)`
  - Training process: `trainer_main(approach, size, reports, running)`
  - The window: `App(screen)` with `.receive(report)`, `.poll_reports()`, `.update(dt)`, `.draw()` and `.handle(event)`, plus `main()`
- Imitation report dicts contain `nets` (a list of `DenseNet`) instead of `weights`. All other keys match Task 7; `kept_finish`, `new_finish` and `step_size` are nan.
- Training started from the application writes to `config.APP_RUNS_DIR`. It never saves champions, so official results are never overwritten.

- [ ] **Step 1: Write the failing tests** in `tests/test_app.py`

```python
import queue

import numpy as np
import pygame
import pytest

from racing import app as viewer
from racing import config, nets


@pytest.fixture
def window(monkeypatch, circle_track):
    monkeypatch.setattr(viewer, "get_track", lambda seed, level: circle_track)
    pygame.init()
    screen = pygame.display.set_mode(config.WINDOW_SIZE)
    instance = viewer.App(screen)
    yield instance
    instance.stop_training()
    pygame.quit()


def ga_report(generation=0, n=30):
    rng = np.random.default_rng(generation)
    fitness = rng.random(n)
    return {"approach": "ga", "generation": generation, "level": 1, "seeds": [0, 1, 2],
            "weights": nets.init_population(rng, n), "hidden": [16], "activation": "relu",
            "fitness": fitness, "finished": np.zeros(n, dtype=bool), "is_kept": np.zeros(n, dtype=bool),
            "kept_finish": float("nan"), "new_finish": 0.0, "step_size": 0.5, "best": float(fitness.max()),
            "mean": float(fitness.mean()), "worst": float(fitness.min()), "evals": n, "elapsed": 1.0,
            "leveled_up": False}


def test_select_indices_modes():
    fitness = np.array([0.3, 0.9, 0.1, 0.5])
    assert list(viewer.select_indices(fitness, "best", 2)) == [1, 3]
    assert list(viewer.select_indices(fitness, "worst", 2)) == [2, 0]
    sample = viewer.select_indices(fitness, "all", 3, sample_seed=7)
    assert len(set(sample)) == 3 and np.array_equal(sample, viewer.select_indices(fitness, "all", 3, sample_seed=7))


def test_select_indices_clamps_the_count():
    fitness = np.arange(5.0)
    assert len(viewer.select_indices(fitness, "best", 0)) == 1
    assert len(viewer.select_indices(fitness, "all", 1000)) == 5


def test_camera_keeps_the_track_inside_the_view(circle_track):
    rect = pygame.Rect(0, 0, 900, 800)
    camera = viewer.Camera(circle_track, rect)
    for edge in (circle_track.left, circle_track.right):
        sx, sy = camera.to_screen(edge[:, 0], edge[:, 1])
        assert sx.min() >= 0 and sx.max() <= 900 and sy.min() >= 0 and sy.max() <= 800


def test_app_plays_a_report(window):
    window.receive(ga_report())
    for _ in range(120):
        window.update(1 / 60)
        window.draw()
    assert window.playback is not None and window.playback_generation == 0


def test_playback_advances_the_simulation(circle_track):
    playback = viewer.Playback(circle_track, lambda obs, cars: np.full(len(cars.x), 8), 3)   # class 8 = straight, gas
    playback.advance(10)
    assert abs(playback.cars.time - 10 * config.DT) < 1e-9 and np.all(playback.cars.speed > 0)


def test_cars_shown_never_exceeds_population(window):
    window.shown = 10_000
    window.receive(ga_report(n=30))
    window.update(1 / 60)
    assert len(window.indices) == 30


def test_app_moves_to_the_newer_generation(window):
    window.receive(ga_report(0))
    window.update(1 / 60)
    window.playback.cars.done[:] = True
    window.receive(ga_report(1))
    window.update(1 / 60)
    assert window.playback_generation == 1


def test_race_without_champions_shows_a_message(window, tmp_path):
    window.champions_dir = tmp_path
    window.set_mode("race")
    window.update(1 / 60)
    window.draw()
    assert "No trained controllers" in window.status


def test_dead_training_process_is_reported(window):
    class Dead:
        exitcode = 1

        def is_alive(self):
            return False

    window.process = Dead()
    window.reports = queue.Queue()
    window.poll_reports()
    assert "stopped" in window.status.lower() and window.process is None


def test_leaderboard_orders_by_distance(circle_track):
    playback = viewer.Playback(circle_track, lambda obs, cars: np.zeros(len(cars.x), dtype=int), 2, laps=3)
    playback.cars.progress[:] = [10.0, 50.0]
    playback.cars.crashed[0] = True
    rows = viewer.leaderboard(playback, ["baseline", "ga"])
    assert [r[0] for r in rows] == ["ga", "baseline"] and rows[1][4] == "crashed"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_app.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'racing.app'`.

- [ ] **Step 3: Implement `racing/app.py`**

```python
"""EC Racing viewer (spec §10): live training view and race view through one top-down camera.

Run with:  ~/miniconda3/bin/python -m racing.app
"""
import multiprocessing as mp
import os
import queue

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from racing import config, nets, sim
from racing.actions import class_to_controls
from racing.tracks import get_track

GRASS = (214, 230, 205)
ROAD = (92, 96, 104)
EDGE = (245, 245, 245)
PANEL_BG = (248, 249, 251)
INK = (29, 35, 48)
MUTED = (91, 100, 116)
BUTTON = (228, 232, 238)
BUTTON_ACTIVE = (0, 114, 178)
RAY = (255, 196, 0)


def hex_to_rgb(code):
    code = code.lstrip("#")
    return tuple(int(code[i:i + 2], 16) for i in (0, 2, 4))


def select_indices(fitness, mode, count, sample_seed=0):
    """Cars to display: a fixed random sample ('all'), the fittest ('best') or the least fit ('worst')."""
    n = len(fitness)
    count = int(min(max(count, 1), n))
    if mode == "best":
        return np.argsort(-fitness, kind="stable")[:count]
    if mode == "worst":
        return np.argsort(fitness, kind="stable")[:count]
    return np.sort(np.random.default_rng(sample_seed).choice(n, count, replace=False))


class Camera:
    """Maps world coordinates (m) to pixels so that a track fills the view rectangle."""

    def __init__(self, track, rect):
        edges = np.vstack([track.left, track.right])
        lo, hi = edges.min(axis=0), edges.max(axis=0)
        self.scale = 0.94 * min(rect.width / (hi[0] - lo[0]), rect.height / (hi[1] - lo[1]))
        self.center = (lo + hi) / 2
        self.rect = rect

    def to_screen(self, x, y):
        sx = self.rect.centerx + (np.asarray(x) - self.center[0]) * self.scale
        sy = self.rect.centery - (np.asarray(y) - self.center[1]) * self.scale
        return sx, sy


def render_track(track, camera, size):
    """Grass, road surface, edges and start line, drawn once per track."""
    surface = pygame.Surface(size)
    surface.fill(GRASS)
    lx, ly = camera.to_screen(track.left[:, 0], track.left[:, 1])
    rx, ry = camera.to_screen(track.right[:, 0], track.right[:, 1])
    m = len(lx)
    for i in range(m):
        j = (i + 1) % m
        pygame.draw.polygon(surface, ROAD, [(lx[i], ly[i]), (lx[j], ly[j]), (rx[j], ry[j]), (rx[i], ry[i])])
    pygame.draw.lines(surface, EDGE, True, list(zip(lx, ly)), 2)
    pygame.draw.lines(surface, EDGE, True, list(zip(rx, ry)), 2)
    pygame.draw.line(surface, INK, (lx[0], ly[0]), (rx[0], ry[0]), 3)
    return surface


def car_polygon(sx, sy, heading, size):
    """Triangle pointing along the heading. Screen y points down, so world rotations flip sign."""
    c, s = np.cos(heading), np.sin(heading)
    shape = [(size, 0.0), (-0.6 * size, 0.5 * size), (-0.6 * size, -0.5 * size)]
    return [(sx + px * c - py * s, sy - (px * s + py * c)) for px, py in shape]


class Playback:
    """Re-simulates a set of controllers on one track, a few steps at a time, for display."""

    def __init__(self, track, policy, n, laps=1, time_limit=None):
        self.track = track
        self.policy = policy
        self.cars = sim.start_cars(track, n, laps)
        self.time_limit = sim.time_limit_for(track, laps) if time_limit is None else time_limit

    @property
    def over(self):
        return bool(self.cars.done.all())

    def advance(self, steps):
        for _ in range(steps):
            if self.over:
                return
            steer, throttle = class_to_controls(self.policy(sim.sense(self.track, self.cars), self.cars))
            sim.step(self.track, self.cars, steer, throttle, self.time_limit)


def mixed_policy(dense_nets):
    """One car per network; each network may have its own architecture."""
    def policy(obs, cars):
        return np.array([net.act(obs[i:i + 1])[0] for i, net in enumerate(dense_nets)], dtype=np.int64)
    return policy


def report_policy(report, indices):
    """Policy for the chosen members of a training report."""
    if "weights" in report:
        return nets.population_policy(report["weights"][indices], tuple(report["hidden"]), report["activation"])
    return mixed_policy([report["nets"][i] for i in indices])


def leaderboard(playback, names):
    """Rows (name, laps completed, last lap, best lap, status), ordered by distance covered."""
    if playback is None:
        return [(name, 0, float("nan"), float("nan"), "ready") for name in names]
    cars = playback.cars
    rows = []
    for i in np.argsort(-cars.progress, kind="stable"):
        if cars.crashed[i]:
            status = "crashed"
        elif cars.finished[i]:
            status = "finished"
        elif cars.done[i]:
            status = "stopped"
        else:
            status = "racing"
        rows.append((names[i], int(cars.laps_done[i]), float(cars.last_lap[i]), float(cars.best_lap[i]), status))
    return rows


def trainer_main(approach, size, reports, running):
    """Training process: trains one approach and sends a report after every generation, trial or epoch.
    Writes to config.APP_RUNS_DIR and never saves champions."""
    from racing import es, ga, imitation
    from racing.expert import collect_dataset

    def send(report):
        running.wait()                      # training holds here while the viewer is paused
        reports.put(report)

    out_dir = config.APP_RUNS_DIR / approach
    if approach == "ga":
        ga.run_ga(pop_size=size, out_dir=out_dir, on_generation=send, champion_dir=None, n_jobs=config.N_JOBS)
        return
    if approach == "es":
        es.run_es(lam=size, out_dir=out_dir, on_generation=send, champion_dir=None, n_jobs=config.N_JOBS)
        return
    if not config.DATASET_PATH.exists():
        collect_dataset()
    data = imitation.load_dataset()
    members, scores = [], []

    def add(number, model, val_accuracy):
        members.append(nets.from_keras(model))
        scores.append(val_accuracy)
        fitness = np.array(scores)
        nan = float("nan")
        send({"approach": approach, "generation": int(number), "level": config.MAX_LEVEL,
              "seeds": [config.KT_VIEW_SEED], "nets": list(members), "fitness": fitness,
              "kept_finish": nan, "new_finish": nan, "step_size": nan, "best": float(fitness.max()),
              "mean": float(fitness.mean()), "worst": float(fitness.min()), "leveled_up": False})

    if approach == "baseline":
        imitation.train_baseline(data, out_dir=out_dir, champion_dir=None,
                                 on_epoch=lambda epoch, model, acc: add(epoch, model, acc))
    else:
        imitation.run_tuner(approach, data, out_dir=out_dir, champion_dir=None, on_trial=add)


def percent(value):
    return "–" if np.isnan(value) else f"{100 * value:.0f}%"


def number(value):
    return "–" if np.isnan(value) else f"{value:.4g}"


class App:
    """The viewer window: training view and race view."""

    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("dejavusans", 15)
        self.small = pygame.font.SysFont("dejavusans", 13)
        width, height = screen.get_size()
        self.view = pygame.Rect(0, 0, width - config.PANEL_WIDTH, height)
        self.panel = pygame.Rect(self.view.right, 0, config.PANEL_WIDTH, height)
        self.alive = True
        self.mode = "train"
        self.buttons = []
        self.slider = None
        self.dragging = False
        self.track_cache = {}
        self.speed = 1
        self.step_budget = 0.0
        self.status = "Press Play to start training."
        # training view
        self.approach = "ga"
        self.population = {"ga": config.GA_POP, "es": config.ES_LAMBDA}
        self.show = "best"
        self.shown = config.DEFAULT_SHOWN
        self.rays = False
        self.process = None
        self.reports = None
        self.running = None
        self.latest = None
        self.history = []
        self.playback = None
        self.playback_generation = None
        self.indices = None
        self.best_slot = 0
        # race view
        self.champions_dir = config.CHAMPIONS_DIR
        self.race_seed = 0
        self.race_names = []
        self.race_nets = None
        self.race = None

    # ----- training process -----
    def start_training(self):
        context = mp.get_context("spawn")
        self.reports = context.Queue()
        self.running = context.Event()
        self.running.set()
        size = self.population.get(self.approach, 0)
        self.process = context.Process(target=trainer_main, args=(self.approach, size, self.reports, self.running),
                                       daemon=True)
        self.process.start()
        self.latest, self.history, self.playback, self.playback_generation = None, [], None, None
        self.status = f"Training {config.LABELS[self.approach]}."

    def toggle_play(self):
        if self.process is None:
            self.start_training()
        elif self.running.is_set():
            self.running.clear()
            self.status = "Paused at the next generation boundary."
        else:
            self.running.set()
            self.status = f"Training {config.LABELS[self.approach]}."

    def stop_training(self):
        if self.process is not None and hasattr(self.process, "terminate") and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.process = None

    def poll_reports(self):
        if self.reports is not None:
            while True:
                try:
                    self.receive(self.reports.get_nowait())
                except queue.Empty:
                    break
        if self.process is not None and not self.process.is_alive():
            code = self.process.exitcode
            self.status = "Training finished." if code == 0 else f"Training stopped (exit code {code})."
            self.process = None

    def receive(self, report):
        self.latest = report
        self.history.append((report["generation"], report["best"], report["mean"], report["leveled_up"]))

    # ----- settings -----
    def select_approach(self, approach):
        if approach == self.approach:
            return
        self.stop_training()
        self.approach = approach
        self.latest, self.history, self.playback, self.playback_generation = None, [], None, None
        self.status = "Press Play to start training."

    def change_population(self, direction):
        presets = list(config.POPULATION_PRESETS)
        current = self.population[self.approach]
        position = presets.index(current) if current in presets else 0
        self.population[self.approach] = presets[min(max(position + direction, 0), len(presets) - 1)]

    def population_size(self):
        if self.latest is not None:
            return len(self.latest["fitness"])
        return self.population.get(self.approach, 1)

    def set_show(self, mode):
        self.show = mode
        self.playback = None

    def set_speed(self, speed):
        self.speed = speed

    def toggle_rays(self):
        self.rays = not self.rays

    def set_shown_from(self, x):
        fraction = min(max((x - self.slider.left) / self.slider.width, 0.0), 1.0)
        self.shown = 1 + int(round(fraction * (self.population_size() - 1)))
        self.playback = None

    def set_mode(self, mode):
        self.mode = mode
        self.step_budget = 0.0
        if mode == "race" and self.race_nets is None:
            self.load_race()

    # ----- playback -----
    def start_playback(self, report):
        fitness = np.asarray(report["fitness"], dtype=float)
        self.indices = select_indices(fitness, self.show, self.shown, sample_seed=report["generation"])
        track = get_track(report["seeds"][0], report["level"])
        self.playback = Playback(track, report_policy(report, self.indices), len(self.indices))
        self.best_slot = int(np.argmax(fitness[self.indices]))
        self.playback_generation = report["generation"]
        self.step_budget = 0.0

    def load_race(self):
        models = nets.load_champions(self.champions_dir)
        self.race_names = list(models)
        self.race_nets = [nets.from_keras(model) for model in models.values()]
        if not self.race_names:
            self.status = f"No trained controllers found in {self.champions_dir}."

    def start_race(self):
        if not self.race_names:
            return
        track = get_track(config.TEST_SEEDS[self.race_seed], config.MAX_LEVEL)
        self.race = Playback(track, mixed_policy(self.race_nets), len(self.race_nets), laps=config.RACE_LAPS)
        self.step_budget = 0.0

    def change_race_track(self, direction):
        self.race_seed = (self.race_seed + direction) % len(config.TEST_SEEDS)
        self.race = None

    def update(self, dt):
        if self.mode == "train":
            self.poll_reports()
            if self.latest is not None and (self.playback is None or self.playback.over):
                self.start_playback(self.latest)
            active = self.playback
        else:
            active = self.race
        if active is not None:
            self.step_budget += self.speed * dt / config.DT
            steps = int(self.step_budget)
            self.step_budget -= steps
            active.advance(steps)

    # ----- input -----
    def handle(self, event):
        if event.type == pygame.QUIT:
            self.alive = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.slider is not None and self.slider.collidepoint(event.pos):
                self.dragging = True
                self.set_shown_from(event.pos[0])
                return
            for rect, action in self.buttons:
                if rect.collidepoint(event.pos):
                    action()
                    return
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_shown_from(event.pos[0])
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and self.mode == "train":
                self.toggle_play()
            elif event.key == pygame.K_LEFT and self.mode == "race":
                self.change_race_track(-1)
            elif event.key == pygame.K_RIGHT and self.mode == "race":
                self.change_race_track(1)

    # ----- drawing -----
    def track_surface(self, track):
        key = (track.seed, track.level)
        if key not in self.track_cache:
            camera = Camera(track, self.view)
            self.track_cache[key] = (camera, render_track(track, camera, self.view.size))
        return self.track_cache[key]

    def draw(self):
        self.screen.fill(GRASS)
        self.buttons = []
        self.slider = None
        playback = self.playback if self.mode == "train" else self.race
        if playback is not None:
            camera, surface = self.track_surface(playback.track)
            self.screen.blit(surface, self.view.topleft)
            if self.mode == "train":
                self.draw_population(playback, camera)
            else:
                self.draw_racers(playback, camera)
        self.draw_panel()

    def draw_population(self, playback, camera):
        cars = playback.cars
        sx, sy = camera.to_screen(cars.x, cars.y)
        size = max(6.0, 4.5 * camera.scale)
        color = hex_to_rgb(config.COLORS[self.approach])
        overlay = pygame.Surface(self.view.size, pygame.SRCALPHA)
        for i in range(len(sx)):
            alpha = 90 if cars.done[i] else 170
            pygame.draw.polygon(overlay, color + (alpha,), car_polygon(sx[i], sy[i], cars.heading[i], size))
        self.screen.blit(overlay, self.view.topleft)
        b = self.best_slot
        if b < len(sx):
            pygame.draw.polygon(self.screen, INK, car_polygon(sx[b], sy[b], cars.heading[b], size * 1.3), 2)
            if self.rays and not cars.done[b]:
                lengths = sim.cast_rays(playback.track, cars.x[b:b + 1], cars.y[b:b + 1], cars.heading[b:b + 1])[0]
                angles = cars.heading[b] + config.RAY_ANGLES
                ex, ey = camera.to_screen(cars.x[b] + np.cos(angles) * lengths, cars.y[b] + np.sin(angles) * lengths)
                for k in range(config.N_RAYS):
                    pygame.draw.line(self.screen, RAY, (sx[b], sy[b]), (ex[k], ey[k]), 1)

    def draw_racers(self, playback, camera):
        cars = playback.cars
        sx, sy = camera.to_screen(cars.x, cars.y)
        size = max(8.0, 6.0 * camera.scale)
        for i, name in enumerate(self.race_names):
            color = hex_to_rgb(config.COLORS[name])
            if cars.crashed[i]:
                color = tuple(int(0.5 * c + 127) for c in color)          # faded
            shape = car_polygon(sx[i], sy[i], cars.heading[i], size)
            pygame.draw.polygon(self.screen, color, shape)
            pygame.draw.polygon(self.screen, INK, shape, 1)

    def text(self, string, x, y, font=None, color=INK):
        surface = (font or self.font).render(string, True, color)
        self.screen.blit(surface, (x, y))
        return y + surface.get_height() + 4

    def button(self, rect, label, action, active=False, enabled=True, color=None):
        fill = hex_to_rgb(color) if (color and active) else (BUTTON_ACTIVE if active else BUTTON)
        pygame.draw.rect(self.screen, fill if enabled else (238, 240, 243), rect, border_radius=6)
        if color and not active:
            pygame.draw.rect(self.screen, hex_to_rgb(color), pygame.Rect(rect.left, rect.top, 5, rect.height),
                             border_top_left_radius=6, border_bottom_left_radius=6)
        ink = (255, 255, 255) if active else (INK if enabled else MUTED)
        label_surface = self.small.render(label, True, ink)
        self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
        if enabled:
            self.buttons.append((rect, action))

    def draw_speed_buttons(self, x, y, w):
        y = self.text("Playback speed", x, y, self.small, MUTED)
        quarter = (w - 24) // 4
        for k, speed in enumerate(config.SPEEDS):
            self.button(pygame.Rect(x + k * (quarter + 8), y, quarter, 28), f"{speed}×",
                        lambda s=speed: self.set_speed(s), self.speed == speed)
        return y + 40

    def draw_panel(self):
        pygame.draw.rect(self.screen, PANEL_BG, self.panel)
        x, y, w = self.panel.left + 16, 14, self.panel.width - 32
        half = (w - 8) // 2
        self.button(pygame.Rect(x, y, half, 30), "Training", lambda: self.set_mode("train"), self.mode == "train")
        self.button(pygame.Rect(x + half + 8, y, half, 30), "Race", lambda: self.set_mode("race"), self.mode == "race")
        y += 44
        if self.mode == "train":
            self.draw_train_panel(x, y, w)
        else:
            self.draw_race_panel(x, y, w)
        self.text(self.status[:56], x, self.panel.bottom - 26, self.small, MUTED)

    def draw_train_panel(self, x, y, w):
        third = (w - 16) // 3
        y = self.text("Approach", x, y, self.small, MUTED)
        for k, approach in enumerate(config.APPROACHES):
            rect = pygame.Rect(x + (k % 3) * (third + 8), y + (k // 3) * 36, third, 30)
            self.button(rect, config.LABELS[approach], lambda a=approach: self.select_approach(a),
                        approach == self.approach, color=config.COLORS[approach])
        y += 80
        if self.approach in self.population:
            y = self.text("Population (λ)" if self.approach == "es" else "Population", x, y, self.small, MUTED)
            idle = self.process is None
            self.button(pygame.Rect(x, y, 30, 28), "−", lambda: self.change_population(-1), enabled=idle)
            self.text(str(self.population[self.approach]), x + 44, y + 5)
            self.button(pygame.Rect(x + 110, y, 30, 28), "+", lambda: self.change_population(1), enabled=idle)
            y += 40
        playing = self.process is not None and self.running is not None and self.running.is_set()
        self.button(pygame.Rect(x, y, w, 34), "Pause" if playing else "Play", self.toggle_play, playing)
        y += 46
        y = self.text("Show", x, y, self.small, MUTED)
        for k, mode in enumerate(("all", "best", "worst")):
            self.button(pygame.Rect(x + k * (third + 8), y, third, 28), mode.capitalize(),
                        lambda m=mode: self.set_show(m), self.show == mode)
        y += 38
        size = self.population_size()
        shown = min(self.shown, size)
        y = self.text(f"Cars shown: {shown}", x, y, self.small, MUTED)
        self.slider = pygame.Rect(x, y + 4, w, 10)
        pygame.draw.rect(self.screen, BUTTON, self.slider, border_radius=5)
        knob = x + (shown - 1) / max(size - 1, 1) * w
        pygame.draw.circle(self.screen, BUTTON_ACTIVE, (int(knob), y + 9), 8)
        y += 26
        y = self.draw_speed_buttons(x, y, w)
        self.button(pygame.Rect(x, y, w, 28), f"Sensor rays: {'on' if self.rays else 'off'}", self.toggle_rays, self.rays)
        y += 40
        y = self.draw_stats(x, y)
        self.draw_plot(pygame.Rect(x, y + 6, w, 150))

    def draw_stats(self, x, y):
        r = self.latest
        if r is None:
            return self.text("No generation completed yet.", x, y, self.small, MUTED)
        unit = "Trial" if r["approach"].startswith("kt") else ("Epoch" if r["approach"] == "baseline" else "Generation")
        size_name = "σ" if r["approach"] == "es" else "Mutation strength"
        for line in (f"Level {r['level']}    {unit} {r['generation']}",
                     f"Finished: kept {percent(r['kept_finish'])}   new {percent(r['new_finish'])}",
                     f"{size_name}: {number(r['step_size'])}",
                     f"Best {r['best']:.3f}    Mean {r['mean']:.3f}"):
            y = self.text(line, x, y)
        return y

    def draw_plot(self, rect):
        pygame.draw.rect(self.screen, (255, 255, 255), rect)
        pygame.draw.rect(self.screen, BUTTON, rect, 1)
        self.text(f"Best and mean fitness, {len(self.history)} reports", rect.left + 6, rect.top + 4, self.small, MUTED)
        if len(self.history) < 2:
            return
        gens = np.array([h[0] for h in self.history], dtype=float)
        best = np.array([h[1] for h in self.history])
        mean = np.array([h[2] for h in self.history])
        lo, hi = min(best.min(), mean.min()), max(best.max(), mean.max())
        span = hi - lo if hi > lo else 1.0
        width = max(gens[-1] - gens[0], 1.0)

        def point(g, v):
            return (rect.left + 6 + (g - gens[0]) / width * (rect.width - 12),
                    rect.bottom - 6 - (v - lo) / span * (rect.height - 30))

        for g, _, _, leveled_up in self.history:
            if leveled_up:
                px = point(g, lo)[0]
                pygame.draw.line(self.screen, MUTED, (px, rect.top + 22), (px, rect.bottom - 2), 1)
        pygame.draw.lines(self.screen, MUTED, False, [point(g, v) for g, v in zip(gens, mean)], 1)
        pygame.draw.lines(self.screen, hex_to_rgb(config.COLORS[self.approach]), False,
                          [point(g, v) for g, v in zip(gens, best)], 2)

    def draw_race_panel(self, x, y, w):
        if not self.race_names:
            self.text("No trained controllers found.", x, y, self.small, MUTED)
            return
        y = self.text("Test track", x, y, self.small, MUTED)
        self.button(pygame.Rect(x, y, 30, 28), "<", lambda: self.change_race_track(-1))
        self.text(f"seed {config.TEST_SEEDS[self.race_seed]}", x + 44, y + 5)
        self.button(pygame.Rect(x + 140, y, 30, 28), ">", lambda: self.change_race_track(1))
        y += 40
        self.button(pygame.Rect(x, y, w, 34), "Start race", self.start_race)
        y += 46
        y = self.draw_speed_buttons(x, y, w)
        y = self.text(f"Leaderboard ({config.RACE_LAPS} laps)", x, y, self.small, MUTED)
        for name, laps, last, best, status in leaderboard(self.race, self.race_names):
            pygame.draw.rect(self.screen, hex_to_rgb(config.COLORS[name]), pygame.Rect(x, y + 3, 10, 10))
            y = self.text(f"{config.LABELS[name]:<13} lap {laps}  last {number(last)}  best {number(best)}  {status}",
                          x + 16, y, self.small)


def main():
    pygame.init()
    pygame.display.set_caption("EC Racing")
    screen = pygame.display.set_mode(config.WINDOW_SIZE)
    viewer = App(screen)
    clock = pygame.time.Clock()
    try:
        while viewer.alive:
            dt = clock.tick(config.FPS) / 1000.0
            for event in pygame.event.get():
                viewer.handle(event)
            viewer.update(dt)
            viewer.draw()
            pygame.display.flip()
    finally:
        viewer.stop_training()
        pygame.quit()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_app.py -q`
Expected: all pass.

- [ ] **Step 5: Smoke-test the real window with GA training**

Run: `timeout 90 $PY -m racing.app`, with `SDL_VIDEODRIVER` unset so a real window opens. Then:
1. Press Space to start GA training.
2. Wait for the first generation to appear.
3. Close the window.

Expected: the window opens, cars appear after the first generation, and closing it exits cleanly within 2 s. If no display is available, run `SDL_VIDEODRIVER=dummy timeout 30 $PY -m racing.app` and confirm a clean exit after the timeout.

- [ ] **Step 6: Commit**

```bash
git add racing/app.py tests/test_app.py
git commit -m "Add pygame viewer with live training view and race view

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/app.py tests/test_app.py
```

---

### Task 13: Notebooks, README and end-to-end test

**Files:**
- Create: `scripts/make_notebooks.py`, `notebooks/01_dataset.ipynb` … `notebooks/05_evaluate.ipynb` (generated), `README.md`
- Test: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: every public entry point: `expert.collect_dataset`, `imitation.train_baseline`, `imitation.run_tuner`, `ga.run_ga`, `es.run_es`, `evaluate.main`.
- Produces: the experiment notebooks and the project README.

- [ ] **Step 1: Write the failing end-to-end test** in `tests/test_end_to_end.py`

```python
import json

import pytest

from racing import es, evaluate, expert, ga, imitation


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    runs, champions = tmp_path / "runs", tmp_path / "champions"
    data = expert.collect_dataset(tmp_path / "dataset.npz",
                                  splits={"train": [(0, 1), (0, 2)], "val": [(80, 4)], "test": [(1000, 4)]})
    imitation.train_baseline(data, out_dir=runs / "baseline", champion_dir=champions, epochs=2)
    ga.run_ga(pop_size=20, kept=5, budget=40, out_dir=runs / "ga", champion_dir=champions)
    es.run_es(mu=4, lam=16, budget=40, out_dir=runs / "es", champion_dir=champions)
    results = evaluate.main(champions_dir=champions, runs_dir=runs, data_path=tmp_path / "dataset.npz",
                            out_dir=tmp_path / "out", test_seeds=[1000])
    assert set(results["comparison"]["Method"]) == {"Baseline", "ES", "GA"}
    assert (tmp_path / "out" / "tables" / "comparison.md").exists()
    assert json.loads((runs / "ga" / "summary.json").read_text())["evals"] == 40


def test_notebooks_are_valid_json():
    from pathlib import Path
    folder = Path(__file__).resolve().parent.parent / "notebooks"
    names = sorted(p.name for p in folder.glob("*.ipynb"))
    assert names == ["01_dataset.ipynb", "02_imitation.ipynb", "03_ga.ipynb", "04_es.ipynb", "05_evaluate.ipynb"]
    for path in folder.glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4 and notebook["cells"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_end_to_end.py -q -m "slow or not slow"`
Expected: `test_notebooks_are_valid_json` fails because the notebooks folder is empty. The slow test passes if Tasks 1–10 are complete.

- [ ] **Step 3: Implement `scripts/make_notebooks.py` and generate the notebooks**

```python
"""Writes the five experiment notebooks as plain nbformat-4 JSON (no extra dependency).

Run with:  ~/miniconda3/bin/python scripts/make_notebooks.py
"""
import json
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent / "notebooks"
SETUP = "import sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path.cwd().parent))"

NOTEBOOKS = {
    "01_dataset.ipynb": [
        ("markdown", "# 1. Expert dataset\nCollects the imitation-learning dataset (design specification §7)."),
        ("code", SETUP),
        ("code", "from racing.expert import collect_dataset\n\ndata = collect_dataset()\n"
                 "{key: value.shape for key, value in data.items()}"),
    ],
    "02_imitation.ipynb": [
        ("markdown", "# 2. Imitation-learning controllers\nBaseline and Keras Tuner searches (design specification §8)."),
        ("code", SETUP),
        ("code", "from racing import imitation\n\ndata = imitation.load_dataset()\nimitation.train_baseline(data)"),
        ("code", "for approach in ['kt_random', 'kt_hyperband', 'kt_bayesian']:\n"
                 "    print(imitation.run_tuner(approach, data))"),
    ],
    "03_ga.ipynb": [
        ("markdown", "# 3. Genetic algorithm\nNeuroevolution with the curriculum (design specification §9.2)."),
        ("code", SETUP),
        ("code", "from racing import config, ga\n\n\ndef progress(report):\n"
                 "    if report['generation'] % 10 == 0 or report['leveled_up']:\n"
                 "        print(f\"gen {report['generation']:4d}  level {report['level']}  best {report['best']:.3f}  \"\n"
                 "              f\"mean {report['mean']:.3f}  new finish {report['new_finish']:.2f}\")\n\n\n"
                 "ga.run_ga(on_generation=progress, n_jobs=config.N_JOBS)"),
    ],
    "04_es.ipynb": [
        ("markdown", "# 4. Evolution strategy\n(μ+λ)-ES with the 1/5 success rule (design specification §9.3)."),
        ("code", SETUP),
        ("code", "from racing import config, es\n\n\ndef progress(report):\n"
                 "    if report['generation'] % 10 == 0 or report['leveled_up']:\n"
                 "        print(f\"gen {report['generation']:4d}  level {report['level']}  best {report['best']:.3f}  \"\n"
                 "              f\"mean {report['mean']:.3f}  sigma {report['step_size']:.4f}\")\n\n\n"
                 "es.run_es(on_generation=progress, n_jobs=config.N_JOBS)"),
    ],
    "05_evaluate.ipynb": [
        ("markdown", "# 5. Evaluation\nTest tracks, test accuracy, size and inference time (design specification §11)."),
        ("code", SETUP),
        ("code", "from racing import evaluate\n\nresults = evaluate.main()\nresults['comparison']"),
        ("code", "results['driving']"),
        ("code", "from IPython.display import Image, display\n\nfor path in results['figures']:\n"
                 "    display(Image(filename=str(path)))"),
    ],
}


def cell(kind, source):
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def main():
    FOLDER.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        notebook = {"cells": [cell(kind, source) for kind, source in cells],
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python"}},
                    "nbformat": 4, "nbformat_minor": 5}
        (FOLDER / name).write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    main()
```

Run: `$PY scripts/make_notebooks.py`

- [ ] **Step 4: Write `README.md`**

````markdown
# EC Racing

EC Racing compares methods for finding small neural network driving controllers. Six controllers race on
procedurally generated tracks in a top-down 2D simulator:

- **Imitation learning:** a hand-designed baseline and three Keras Tuner searches (random search, Hyperband,
  Bayesian optimization) are trained to imitate a scripted expert driver.
- **Neuroevolution:** a genetic algorithm and a (μ+λ) evolution strategy evolve network weights by driving,
  following a curriculum from a circle to fully varied tracks.

The design specification is in `docs/2026-10-05-ec-racing-design.md`, and the implementation plan is in
`docs/2026-10-05-ec-racing-plan.md`.

## Setup

The project uses the Conda base environment (Python 3.13, TensorFlow 2.21, Keras 3.13):

```bash
~/miniconda3/bin/python -m pip install keras-tuner pygame pytest
```

## Running the experiments

Run the notebooks in `notebooks/` in order:

| Notebook | Purpose |
|---|---|
| `01_dataset` | Expert dataset |
| `02_imitation` | Baseline and Keras Tuner searches |
| `03_ga` | Genetic algorithm |
| `04_es` | Evolution strategy |
| `05_evaluate` | Tables and figures |

Results are written to `results/`, which is not under version control:

| Path | Contents |
|---|---|
| `results/dataset.npz` | Expert dataset |
| `results/runs/<approach>/` | Logs, summaries, per-generation best networks |
| `results/champions/<approach>.keras` | Final controllers |
| `results/tables/` | Results tables (CSV and Markdown) |
| `results/figures/` | Figures |

## Viewer

```bash
~/miniconda3/bin/python -m racing.app
```

- **Training view:** select an approach, set the population size, and press Play (or Space). The view shows the
  latest completed generation. Training started from the viewer writes to `results/app_runs/` and never replaces
  the final controllers.
- **Race view:** the final controllers race three laps on a held-out test track. Select the track with the arrow
  keys.

## Tests

```bash
~/miniconda3/bin/python -m pytest              # fast tests
~/miniconda3/bin/python -m pytest -m slow      # slow tests: every track seed, Keras Tuner searches, end-to-end
```

## Generation-time benchmark

```bash
~/miniconda3/bin/python -m racing.bench
```

The number of worker processes used for population evaluation is `N_JOBS` in `racing/config.py`.
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_end_to_end.py -q -m "slow or not slow"`, then the full fast suite `$PY -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/make_notebooks.py notebooks README.md tests/test_end_to_end.py
git commit -m "Add experiment notebooks, README and end-to-end test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- scripts/make_notebooks.py notebooks README.md tests/test_end_to_end.py
```

---

## Spec Coverage

| Spec section | Task |
|---|---|
| §3 Architecture, module structure | 1 and all |
| §4 Tracks, curriculum levels, validation, seeds | 1, 2, 6 |
| §5 Vehicle dynamics (grip-only), sensors, collisions, progress, fitness, determinism | 1, 3 |
| §6 Network interface, Keras export | 1, 4 |
| §7 Expert driver, recovery data, dataset | 6 |
| §8 Baseline, Keras Tuner, model selection | 5 |
| §9.1 Common procedure, level advancement, budget, logging, final selection | 1, 7, 8, 9 |
| §9.2 GA, §9.3 ES | 8, 9 |
| §9.4 Computational performance | 11 |
| §10 Visualization application | 12 |
| §11 Evaluation methodology, tables, figures | 10 |
| §12 Verification and testing | every task |
| §13 Default parameters | 1 |
| §15 Software environment | 1, 13 |
