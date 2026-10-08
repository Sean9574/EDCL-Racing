# EC Racing Revision 3.1 (Evolved Car Setup) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let every evolutionary population evolve each car's sensor length (10–150 m) and steering sensitivity (⅓–3) next to its network, add a Normal/Unlimited grip selector to the viewer for display, and keep training started in the viewer (resume, New run, raceable app cars).

**Architecture:** A car's setup is a row `(sensor length m, steering sensitivity)` in an `(n, 2)` float array carried by every population and driver. The simulator takes the setup per car (ray length, steering gain) and a grip parameter that defaults to the training grip. A new `racing/setup.py` holds the shared setup rules (defaults, random draws, crossover, mutation, clipping). Methods inherit and mutate setups with these shared rules; the engine logs setup statistics; saved controllers carry a `<seed>.setup.json`. The viewer passes its selected grip only to its own display simulations.

**Tech Stack:** Python 3.13 (`~/miniconda3/bin/python`), numpy, pandas, matplotlib, joblib (loky), TensorFlow/Keras on CPU, pygame, pytest.

**Spec:** `docs/2026-10-07-ec-racing-setup-design.md` (revision 3.1), extending `docs/2026-10-07-ec-racing-mastery-design.md` (revision 3) and `docs/2026-10-05-ec-racing-design.md` (version 3.0). Executors read the 3.1 spec; the others where referenced.

## Global Constraints

- CPU only; Keras through `racing.backend.keras_cpu()`. Interpreter `~/miniconda3/bin/python`. No new packages; never install or upgrade anything.
- Tests never signal real processes and never start real training or window processes (use the existing fake process/queue patterns), and never write into the real `results/` (tests use `tmp_path`; the autouse fixture in `tests/conftest.py` already redirects the status file and run log).
- No personal names in code, docs, comments, tests or commit messages.
- Commits are path-scoped (`git add <paths>`; `git commit -m "..." -- <paths>`), and every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never `git add -A`, `git add .`, `git stash`, `git reset`, `git checkout -- .`; never stage `.vscode/`. Do not push (the controller pushes after review).
- Branch `build`. Fast suite: `~/miniconda3/bin/python -m pytest -q` (excludes `slow`); slow: `-m slow`. Every task ends with the fast suite passing.
- Results never depend on `n_jobs`; a resumed run equals an uninterrupted one; a car with the default setup `(50 m, 1)` behaves exactly as in revision 3 (same inputs, controls, trajectory, fitness).
- The training grip `config.GRIP` is used by training, evaluation and the expert. Only the viewer's display simulations may use another grip.
- Match the surrounding code: short docstrings citing the spec (`(revision 3.1 §n)`), numpy-vectorized style, constants in `racing/config.py`, existing naming.
- Every number of the 3.1 spec Table 2 lives in `racing/config.py` under the names below; code never repeats the literal.

## Review Focus

The failure modes the spec implies that per-feature tests could miss, most likely first. Each is pinned by a test in the owning task.

1. **Default setup must reproduce revision 3 exactly.** Raising the ray iteration limit for 150 m rays could change 50 m rays that previously ran out of iterations. Expect: each ray's iteration cap scales with its own length, so a 50 m ray gets exactly the revision 3 cap and identical results. *Test in Task 1.*
2. **Setups travel with cars through every path:** subset/driver, parallel evaluation tasks, checkpoints and resume, final selection, saved champions, race view, evaluation tables. Expect: a car evaluated anywhere uses its own setup; nothing silently falls back to the default. *Tests in Tasks 2 and 3.*
3. **Display grip never leaks into training.** Expect: the viewer's Unlimited grip affects only `Playback` objects it creates; the trainer process, evaluation and the expert always use `config.GRIP`. *Test in Task 5.*
4. **Resuming a saved viewer run with changed settings or code.** Expect: the panel names the problem and asks for New run; the viewer never crashes and never mixes old and new state. *Test in Task 6.*
5. **App cars never mix with reported controllers.** Expect: app champions live under `results/app_runs/champions/`, the race view lists them in their own group switched off by default, and evaluation never reads them. *Test in Task 6.*

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `racing/config.py` | Setup limits, defaults, mutation values, app grips, ray iteration cap, app champions path | 1, 2, 5, 6 |
| `racing/sim.py` | Per-car ray length and steering gain; `grip` parameter; scaled iteration cap | 1 |
| `racing/setup.py` (new) | Setup arrays: defaults, random draws, crossover, mutation, clipping, JSON form | 2 |
| `racing/nets.py`, `racing/neat.py` | Populations and drivers carry `setup`; setup files on save/load | 2 |
| `racing/population.py`, `racing/evaluate.py` (`_drive_track` only), `racing/dagger.py` | Evaluation drives each car with its setup | 2 |
| `racing/ga.py`, `racing/es.py`, `racing/neat.py`, `racing/evolution.py`, `racing/checkpoint.py` | Setup initialization, inheritance, mutation, state; log and report fields; recorded settings | 3 |
| `racing/evaluate.py`, `racing/netfig.py`, `racing/netwindow.py` | Setup columns, setup figure, titles | 4 |
| `racing/app.py` | Grip selector, per-car rays, setup stats | 5 |
| `racing/app.py`, `racing/paths.py` | Saved app runs: resume, New run, mismatch message, app champions, race view app group | 6 |
| `scripts/pilots.py`, `README.md`, `docs/2026-10-05-ec-racing-design.md` (+PDF) | Pilot setup columns, viewer notes, design specification 3.1 | 7 |

## Interface Contract

These names, signatures and semantics are fixed for all tasks.

### `racing/config.py`

```python
RAY_RANGE = 50.0                        # unchanged value: default sensor length AND the ray input scale (1.0 = 50 m)
RAY_MAX_ITERATIONS = 48                 # unchanged: the iteration cap of a RAY_RANGE ray; a ray of length L gets
                                        # ceil(RAY_MAX_ITERATIONS * L / RAY_RANGE) iterations (Review Focus 1)
SENSOR_LENGTH_LIMITS = (10.0, 150.0)    # m (revision 3.1 Table 1)
STEER_GAIN_LIMITS = (1.0 / 3.0, 3.0)
DEFAULT_SETUP = (RAY_RANGE, 1.0)        # (sensor length m, steering sensitivity) of revision 3
SETUP_MUTATION_P = 0.2                  # per value
SETUP_MUTATION_SD = 0.1                 # natural-log scale
APP_GRIPS = {"Normal": GRIP, "Unlimited": 10.0 * GRIP}   # display only (revision 3.1 §6)
APP_CHAMPIONS_DIR = APP_RUNS_DIR / "champions"           # Task 6
```
`checkpoint.SETTINGS_KEYS` gains `"SENSOR_LENGTH_LIMITS"`, `"STEER_GAIN_LIMITS"`, `"DEFAULT_SETUP"`, `"SETUP_MUTATION_P"`, `"SETUP_MUTATION_SD"` (Task 3; never `APP_GRIPS`, `APP_CHAMPIONS_DIR`, `N_JOBS`).

### `racing/sim.py` (Task 1)

```python
Cars.ray_length: np.ndarray    # (n,) m
Cars.steer_gain: np.ndarray    # (n,)
def start_cars(track, n, laps=1, setup=None) -> Cars          # setup (n, 2) or None = DEFAULT_SETUP for every car
def cast_rays(track, x, y, heading, lengths=None) -> np.ndarray   # (n, N_RAYS) m; lengths (n,) or None = RAY_RANGE;
                                                              # each ray stops at its car's length and at its own cap
def sense(track, cars) -> np.ndarray                          # rays / RAY_RANGE (may exceed 1), speed / SPEED_SCALE
def step(track, cars, steer, throttle, time_limit, grip=config.GRIP)
    # a_lat = clip(steer * cars.steer_gain, -1, 1) * grip; a_long = clip(throttle, -1, 1) * grip; friction circle at grip
def simulate(track, policy, n, time_limit=None, laps=1, setup=None, grip=config.GRIP) -> RunResult
```

### `racing/setup.py` (new, Task 2)

```python
COLUMNS = ("sensor_length", "steer_gain")
def default(n) -> np.ndarray                      # (n, 2) float64 rows of DEFAULT_SETUP
def random(rng, n) -> np.ndarray                  # (n, 2) log-uniform within the limits (one rng.uniform call)
def clip(values) -> np.ndarray                    # into SENSOR_LENGTH_LIMITS / STEER_GAIN_LIMITS
def crossover(rng, a, b) -> np.ndarray            # each value from a or b with probability 0.5 (one rng.random call)
def mutate(rng, values) -> np.ndarray             # per value: with SETUP_MUTATION_P, multiply by exp(N(0, SETUP_MUTATION_SD)); clip
                                                  # draws exactly two arrays (mask, noise) of values.shape, always
def to_dict(row) -> dict                          # {"sensor_length": float, "steer_gain": float}
def from_dict(d) -> np.ndarray                    # (2,)
def summary(values) -> dict                       # {"sensor_mean", "sensor_std", "steer_mean", "steer_std"}
def as_rows(values, n) -> np.ndarray              # (n, 2) float64 copy: None → default rows; validates the shape
def as_row(value) -> np.ndarray                  # (2,): None → DEFAULT_SETUP
```

### Populations and drivers (Task 2)

```python
nets.DensePopulation(weights, hidden=..., activation="relu", setup=None)   # .setup (n, 2), default rows when None
neat.NeatPopulation(genomes, setup=None)                                   # .setup (n, 2)
# subset(indices) keeps the matching setup rows; driver(i) returns a driver whose .setup is row i (shape (2,)).
nets.DenseNet.setup / neat.NeatNet.setup: np.ndarray (2,), default DEFAULT_SETUP
nets.load_champion(...)              # reads <seed>.setup.json when present into driver.setup
population.save_dense_champion(...)  # also writes <seed>.setup.json from population.setup[index]
neat.save_neat_champion(approach, seed, genome, champion_dir, setup=None)   # writes <seed>.setup.json (default row if None)
population.clear_champion(...)       # also removes <seed>.setup.json
def nets.drivers_setup(drivers) -> np.ndarray   # (len(drivers), 2) from each driver's .setup
```
Every simulation of a population or of drivers passes `setup=` to `sim.simulate`: `population._time_trial` (population.setup), `evaluate._drive_track` (`nets.drivers_setup`), `dagger` drives and checks (driver.setup, the default for imitation networks).

### Methods and engine (Task 3)

- **GA, ES, NEAT** expose `setup` as a read-only property returning `self.population.setup` (kept rows first, then children; it can never drift from the population), include setups in `state()`/`load_state()`, and build populations with `setup=`. Inside `racing/neat.py` the setup module is imported so that it never shadows the `NEAT.setup` property or a `setup` parameter: `from racing import setup as car_setup` (added in Task 2; Task 3 uses that name there).
- **Initial setups:** `setup.random(rng, n)` for random-start populations (GA, ES without `initial`, NEAT); `setup.default(n)` for ES with `initial` (seeded ES and speed pushes).
- **Children:** GA: `setup.crossover(rng, mother_setups, father_setups)` then `setup.mutate`; ES: parent's setup then `setup.mutate`; NEAT: fitter (or single) parent's setup then `setup.mutate`. Kept cars keep their setup. The setup draws come after each method's existing draws in `advance`, so the order of the network's random draws is unchanged within a generation.
- **Engine:** `LOG_FIELDS` gains `"sensor_mean", "sensor_std", "steer_mean", "steer_std"` (from `setup.summary(population.setup)`); the report gains these four keys; the summary gains `"final_setup": setup.to_dict(...)` of the saved champion.

### Evaluation, figures, viewer (Tasks 4–6)

```python
# evaluate.py: results_table(per_track, runs_dir, drivers=None) gains "Sensor length (m)" and "Steering sensitivity"
#   (mean ± std across seeds of the final drivers' setups; main passes its drivers); figures() also writes setup_over_training.png (population mean sensor length and steering
#   sensitivity per generation per method, level-ups and tightenings marked).
# netfig.draw_networks / netwindow titles append " · <L> m · ×<k>" for the driver's setup.
# app.Playback(track, policy, n, laps=1, time_limit=None, setup=None, grip=config.GRIP)
# App.grip_name in config.APP_GRIPS ("Normal" default); App.set_grip(name)
# Training-panel line: "Sensors <mean> m ± <std>  Steering ×<mean> ± <std>" (fits the panel; width-tested)
# Network window titles are built in app.shown_network (Task 4 edits app.py for them).
# Race panel (Task 6): one cycle button switches the controller list between reported controllers and app runs.
# Saved app runs (Task 6): trainer never clears an evolutionary app run on Play; App.new_run() deletes
#   app_runs/<approach>/seed<k> and that car's app champion files, then starts; the trainer saves the latest
#   generation's best car to config.APP_CHAMPIONS_DIR after every report (evolutionary) or at the end (imitation);
#   trainer exit code 3 = saved run unusable (settings mismatch or unreadable record) → panel message asks for New run;
#   race view lists app champions as their own group (labels "<Method> (app) s<k>"), each off by default.
```

## Tasks

| # | Task | Depends on |
|---|---|---|
| 1 | Simulator: per-car sensor length and steering sensitivity, grip parameter, scaled ray cap | — |
| 2 | Setup module; populations, drivers, evaluation paths and saved controllers carry setups | 1 |
| 3 | Methods and engine evolve setups (GA, ES, NEAT); logs, reports, summaries, settings | 2 |
| 4 | Evaluation: setup columns and figure; network figure and window titles | 3 |
| 5 | Viewer: grip selector, per-car rays, setup stats | 2, 3 |
| 6 | Viewer: saved app runs (resume, New run, mismatch message, app champions, race group) | 5 |
| 7 | Pilots, README, design specification 3.1 | 4, 6 |

---

### Task 1: Simulator: per-car sensor length and steering sensitivity, grip parameter, scaled ray cap

Revision 3.1 §2 and §6. Every car in a simulation gets its own sensor length and steering sensitivity, given as an
`(n, 2)` setup array (`None` = revision 3's setup for every car), and the grip becomes a parameter of the step that
defaults to the training grip. Each ray's sphere-tracing cap scales with its own length, so a ray of revision 3's
length (50 m) is traced exactly as in revision 3 (Review Focus 1). Ray inputs keep revision 3's scale: an input of 1.0
is 50 m, so a 150 m ray can report up to 3.0.

**Files:**
- Modify: `racing/config.py` (sensor section; new `# --- Car setup (revision 3.1 §2) ---` block with `DEFAULT_SETUP`)
- Modify: `racing/sim.py` (module docstring, `Cars`, new `setup_columns` and `ray_caps`, `start_cars`, `cast_rays`, `sense`, `step`, `simulate`)
- Test: `tests/test_sim.py`

**Interfaces:**
- Consumes: `racing.track.build_track`, `racing.track.wall_distance`, `racing.expert.expert_controls` (all unchanged).
- Produces: `config.DEFAULT_SETUP = (RAY_RANGE, 1.0)`; `sim.Cars.ray_length (n,)`, `sim.Cars.steer_gain (n,)`;
  `sim.setup_columns(setup, n) -> (lengths, gains)`; `sim.ray_caps(lengths) -> int64 array`;
  `sim.start_cars(track, n, laps=1, setup=None)`; `sim.cast_rays(track, x, y, heading, lengths=None)`;
  `sim.sense(track, cars)` (uses `cars.ray_length`); `sim.step(track, cars, steer, throttle, time_limit, grip=config.GRIP)`;
  `sim.simulate(track, policy, n, time_limit=None, laps=1, setup=None, grip=config.GRIP)`. Every existing caller
  (expert, bench, population, evaluate, dagger, app) keeps working unchanged through the defaults; Task 2 makes the
  population, evaluation and DAgger paths pass setups, Task 5 the viewer.

- [ ] **Step 1: Write the failing tests**

In `tests/test_sim.py`, change the import block at the top to:

```python
import numpy as np
import pytest

from racing import config, expert, sim
from racing.actions import class_to_controls, controls_to_class
from racing.track import build_track, wall_distance
```

and append:

```python
# ----- revision 3.1: per-car sensor length and steering sensitivity, grip as a parameter -----

def stadium(straight=300.0, radius=60.0, width=12.0):
    """Two straights joined by semicircles, counter-clockwise from the start of the bottom straight."""
    s = np.arange(0.0, straight, 0.5)
    a = np.arange(-np.pi / 2, np.pi / 2, 0.5 / radius)
    points = np.vstack([np.column_stack([s - straight / 2, np.full(len(s), -radius)]),
                        np.column_stack([straight / 2 + radius * np.cos(a), radius * np.sin(a)]),
                        np.column_stack([straight / 2 - s, np.full(len(s), radius)]),
                        np.column_stack([-straight / 2 - radius * np.cos(a), -radius * np.sin(a)])])
    track = build_track(points, width, seed=0, level=4)
    track.expert_lap_time = 60.0
    return track


def reference_cast_rays(track, x, y, heading):
    """sim.cast_rays of revision 3, verbatim (every ray RAY_RANGE long, at most RAY_MAX_ITERATIONS iterations), plus
    which rays stopped at the iteration cap."""
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
    capped = np.zeros(n * config.N_RAYS, dtype=bool)
    capped[active] = True
    return np.minimum(t, config.RAY_RANGE).reshape(n, config.N_RAYS), capped.reshape(n, config.N_RAYS)


def reference_sense(track, cars):
    """sim.sense of revision 3, verbatim."""
    obs = np.zeros((len(cars.x), config.N_INPUTS))
    live = np.flatnonzero(~cars.done)
    if live.size:
        rays, _ = reference_cast_rays(track, cars.x[live], cars.y[live], cars.heading[live])
        obs[live, :config.N_RAYS] = rays / config.RAY_RANGE
        obs[live, config.N_RAYS] = cars.speed[live] / config.SPEED_SCALE
    return obs


def reference_step(track, cars, steer, throttle, time_limit):
    """sim.step of revision 3, verbatim."""
    live = ~cars.done
    a_lat = np.clip(steer, -1.0, 1.0) * config.GRIP
    a_long = np.clip(throttle, -1.0, 1.0) * config.GRIP
    total = np.hypot(a_lat, a_long)
    scale = np.where(total > config.GRIP, config.GRIP / np.maximum(total, 1e-12), 1.0)
    a_lat, a_long = a_lat * scale, a_long * scale
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
    sim._update_progress(track, cars, moving)
    stalled = moving & (cars.time - cars.last_progress_time >= config.STALL_TIME - config.TIME_EPS)
    out_of_time = live & (cars.time >= time_limit - config.TIME_EPS)
    cars.done |= crashed | stalled | out_of_time | cars.finished


def road_poses(track, n, seed):
    """n random positions on the road with random headings."""
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(track.dist > 0)
    pick = rng.choice(len(rows), n, replace=False)
    x = track.origin[0] + (cols[pick] + 0.5) * config.GRID_RES
    y = track.origin[1] + (rows[pick] + 0.5) * config.GRID_RES
    return x, y, rng.uniform(-np.pi, np.pi, n)


def default_setups(n):
    return np.tile(np.asarray(config.DEFAULT_SETUP, dtype=float), (n, 1))


def test_the_default_setup_is_revision_3s():
    assert config.DEFAULT_SETUP == (config.RAY_RANGE, 1.0) == (50.0, 1.0)


def test_a_ray_of_revision_3s_length_keeps_revision_3s_iteration_cap():
    assert list(sim.ray_caps([10.0, 50.0, 150.0])) == [10, config.RAY_MAX_ITERATIONS, 3 * config.RAY_MAX_ITERATIONS]


def test_default_rays_are_traced_exactly_as_in_revision_3(circle_track):
    track = stadium()
    x, y, heading = road_poses(track, 2000, seed=3)
    x = np.append(x, -100.0)                   # 0.3 m from the inner edge of the bottom straight, looking along it
    y = np.append(y, -54.3)
    heading = np.append(heading, 0.0)
    expected, capped = reference_cast_rays(track, x, y, heading)
    assert capped.any()                        # the iteration cap binds for some rays (it does for the last pose)
    assert np.array_equal(sim.cast_rays(track, x, y, heading), expected)
    assert np.array_equal(sim.cast_rays(track, x, y, heading, np.full(len(x), config.RAY_RANGE)), expected)
    cx, cy, ch = road_poses(circle_track, 500, seed=4)
    assert np.array_equal(sim.cast_rays(circle_track, cx, cy, ch), reference_cast_rays(circle_track, cx, cy, ch)[0])


def test_a_default_setup_reproduces_a_revision_3_drive():
    track = stadium()
    n = 40
    new = sim.start_cars(track, n, laps=3, setup=default_setups(n))
    old = sim.start_cars(track, n, laps=3)
    rng_new, rng_old = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(900):
        obs_new, obs_old = sim.sense(track, new), reference_sense(track, old)
        assert np.array_equal(obs_new, obs_old)
        actions = []
        for cars, rng in ((new, rng_new), (old, rng_old)):
            classes = controls_to_class(*expert.expert_controls(track, cars))
            flip = rng.random(n) < 0.1                       # disturbances, so cars also crash and graze walls
            actions.append(np.where(flip, rng.integers(0, config.N_ACTIONS, n), classes))
        assert np.array_equal(actions[0], actions[1])
        steer, throttle = class_to_controls(actions[0])
        sim.step(track, new, steer, throttle, time_limit=200.0)
        reference_step(track, old, steer, throttle, time_limit=200.0)
    for name in ("x", "y", "heading", "speed", "progress", "laps_done", "crashed", "finished", "done"):
        assert np.array_equal(getattr(new, name), getattr(old, name)), name
    assert np.array_equal(new.finish_time, old.finish_time, equal_nan=True)
    assert old.crashed.any() and (old.progress > 300.0).any()            # crashes and long drives both covered


def test_simulate_with_the_default_setup_equals_simulate_without_one(circle_track):
    policy = expert.expert_policy(circle_track)
    plain = sim.simulate(circle_track, policy, 3, laps=2)
    explicit = sim.simulate(circle_track, policy, 3, laps=2, setup=default_setups(3))
    for name in ("fitness", "finished", "crashed", "progress", "laps_done"):
        assert np.array_equal(getattr(plain, name), getattr(explicit, name))
    assert np.array_equal(plain.finish_time, explicit.finish_time, equal_nan=True)


def test_rays_reach_their_cars_sensor_length():
    track = stadium()
    x, y, heading = np.full(3, -150.0), np.full(3, -60.0), np.zeros(3)     # start of the bottom straight, centered
    rays = sim.cast_rays(track, x, y, heading, np.array([10.0, 50.0, 150.0]))
    assert list(rays[:, 4]) == [10.0, 50.0, 150.0]                           # straight ahead: 300 m of road
    assert np.all(rays[0] <= 10.0) and np.all(rays <= np.array([10.0, 50.0, 150.0])[:, None])


def test_long_rays_trace_past_revision_3s_cap():
    track = stadium()
    x, y, heading = np.array([-100.0, -100.0]), np.array([-54.3, -54.3]), np.zeros(2)   # grazing the inner edge
    rays = sim.cast_rays(track, x, y, heading, np.array([50.0, 150.0]))
    _, capped = reference_cast_rays(track, x[:1], y[:1], heading[:1])
    assert capped[0, 4]                                         # revision 3 stops this grazing ray at its cap
    assert rays[1, 4] > rays[0, 4] + 40.0                       # a 150 m sensor gets three times the iterations


def test_sense_scales_rays_by_revision_3s_length():
    track = stadium()
    cars = sim.start_cars(track, 2, setup=np.array([[150.0, 1.0], [10.0, 1.0]]))
    obs = sim.sense(track, cars)
    assert obs[0, 4] == 150.0 / config.RAY_RANGE == 3.0 and obs[1, 4] == 0.2
    assert np.all(obs[:, :config.N_RAYS] <= np.array([[3.0], [0.2]]))


def test_steering_sensitivity_scales_the_steering_levels_up_to_the_grip(circle_track):
    gains = np.array([1.0 / 3.0, 1.0, 3.0, 3.0])
    cars = sim.start_cars(circle_track, 4, setup=np.column_stack([np.full(4, 50.0), gains]))
    cars.speed[:] = 10.0
    h0 = cars.heading.copy()
    sim.step(circle_track, cars, np.array([1.0, 1.0, 0.5, 1.0]), np.zeros(4), time_limit=100.0)
    a_lat = (cars.heading - h0) / config.DT * 10.0
    assert np.allclose(a_lat, config.GRIP * np.array([1.0 / 3.0, 1.0, 1.0, 1.0]))   # k × s, clipped at the grip


def test_grip_is_a_parameter_of_the_step(circle_track):
    normal, high = sim.start_cars(circle_track, 1), sim.start_cars(circle_track, 1)
    sim.step(circle_track, normal, np.zeros(1), np.ones(1), time_limit=100.0)
    sim.step(circle_track, high, np.zeros(1), np.ones(1), time_limit=100.0, grip=10 * config.GRIP)
    assert np.isclose(normal.speed[0], config.GRIP * config.DT)
    assert np.isclose(high.speed[0], 10 * config.GRIP * config.DT)
    turning = sim.start_cars(circle_track, 1)
    turning.speed[:] = 10.0
    h0 = turning.heading[0]
    sim.step(circle_track, turning, np.ones(1), np.ones(1), time_limit=100.0, grip=10 * config.GRIP)
    a_long = (turning.speed[0] - 10.0) / config.DT
    a_lat = (turning.heading[0] - h0) / config.DT * 10.0
    assert np.isclose(np.hypot(a_long, a_lat), 10 * config.GRIP)          # the friction circle at the given grip


def test_simulate_passes_the_grip_and_the_setup(monkeypatch, circle_track):
    seen = []
    real = sim.step

    def recording(track, cars, steer, throttle, time_limit, grip=config.GRIP):
        seen.append((grip, cars.ray_length.copy(), cars.steer_gain.copy()))
        real(track, cars, steer, throttle, time_limit, grip=grip)

    monkeypatch.setattr(sim, "step", recording)
    setup = np.array([[20.0, 2.0], [120.0, 0.5]])
    sim.simulate(circle_track, constant_policy(0.0, 1.0), 2, grip=7.0, setup=setup)
    assert seen and all(g == 7.0 for g, _, _ in seen)
    assert np.array_equal(seen[0][1], [20.0, 120.0]) and np.array_equal(seen[0][2], [2.0, 0.5])


def test_start_cars_rejects_a_setup_of_the_wrong_shape(circle_track):
    with pytest.raises(ValueError, match="setup"):
        sim.start_cars(circle_track, 3, setup=default_setups(2))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_sim.py -q`
Expected: 12 of the new tests FAIL (e.g. `AttributeError: module 'racing.config' has no attribute 'DEFAULT_SETUP'`,
`AttributeError: module 'racing.sim' has no attribute 'ray_caps'`,
`TypeError: start_cars() got an unexpected keyword argument 'setup'`,
`TypeError: cast_rays() takes 4 positional arguments but 5 were given`,
`TypeError: step() got an unexpected keyword argument 'grip'`); the existing `test_sim.py` tests pass.

- [ ] **Step 3: Add the setup default to `racing/config.py`**

In the sensor section, replace the `RAY_RANGE` and `RAY_MAX_ITERATIONS` lines so the section reads:

```python
# --- Sensors (spec §5.2) ---
N_RAYS = 9
RAY_ANGLES = np.radians(np.linspace(-90.0, 90.0, N_RAYS))   # index 0 = right, 4 = ahead, 8 = left
RAY_RANGE = 50.0             # m, revision 3's sensor length and the ray input scale (an input of 1.0 is 50 m)
RAY_MIN_STEP = 0.5           # m, smallest sphere-tracing step
RAY_MAX_ITERATIONS = 48      # sphere-tracing iterations of a RAY_RANGE ray; scaled with each ray's length (rev. 3.1 §2.1)

# --- Car setup (revision 3.1 §2) ---
DEFAULT_SETUP = (RAY_RANGE, 1.0)   # (sensor length m, steering sensitivity) of every car in revision 3
```

(Task 2 adds the limits and mutation values to this block.)

- [ ] **Step 4: Carry the setup through `racing/sim.py`**

1. Replace the module docstring with:

```python
"""Vectorized driving simulator (spec §5). Every per-car quantity is an array with one entry per car, including each
car's setup: its sensor length and steering sensitivity (revision 3.1 §2)."""
```

2. In `class Cars`, add two fields directly after `finish_time` (before `laps`):

```python
    ray_length: np.ndarray          # m, each car's sensor length (revision 3.1 §2.1)
    steer_gain: np.ndarray          # each car's steering sensitivity (revision 3.1 §2.2)
```

3. Replace `start_cars` with:

```python
def setup_columns(setup, n):
    """(sensor lengths, steering sensitivities) of n cars from an (n, 2) setup array, or revision 3's
    (config.DEFAULT_SETUP) for every car when setup is None."""
    if setup is None:
        length, gain = config.DEFAULT_SETUP
        return np.full(n, float(length)), np.full(n, float(gain))
    setup = np.asarray(setup, dtype=float)
    if setup.shape != (n, 2):
        raise ValueError(f"setup must have shape ({n}, 2), one (sensor length, steering sensitivity) row per car; "
                         f"got {setup.shape}")
    return setup[:, 0].copy(), setup[:, 1].copy()


def start_cars(track, n, laps=1, setup=None):
    """n cars at rest on the start line, facing the direction of travel, each with its setup (revision 3.1 §2): an
    (n, 2) array of sensor lengths and steering sensitivities, or None for revision 3's."""
    x0, y0 = track.start_position
    nan = np.full(n, np.nan)
    ray_length, steer_gain = setup_columns(setup, n)
    return Cars(
        x=np.full(n, x0), y=np.full(n, y0), heading=np.full(n, track.start_heading), speed=np.zeros(n),
        index=np.zeros(n, dtype=np.int64), progress=np.zeros(n), last_progress_time=np.zeros(n),
        laps_done=np.zeros(n, dtype=np.int64), lap_start_time=np.zeros(n), last_lap=nan.copy(),
        best_lap=nan.copy(), crashed=np.zeros(n, dtype=bool), finished=np.zeros(n, dtype=bool),
        done=np.zeros(n, dtype=bool), finish_time=nan.copy(), ray_length=ray_length, steer_gain=steer_gain,
        laps=laps,
    )
```

4. Replace `cast_rays` and `sense` with:

```python
def ray_caps(lengths):
    """Sphere-tracing iterations allowed for rays of the given lengths (revision 3.1 §2.1): RAY_MAX_ITERATIONS for a
    RAY_RANGE ray, scaled with the length, so a ray of revision 3's length is traced exactly as before."""
    return np.ceil(config.RAY_MAX_ITERATIONS * np.asarray(lengths, dtype=float) / config.RAY_RANGE).astype(np.int64)


def cast_rays(track, x, y, heading, lengths=None):
    """Distance to the track boundary along each sensor ray, by sphere tracing. Each car's rays reach at most its
    sensor length (lengths, default RAY_RANGE) and are traced for at most ray_caps(length) iterations. Returns
    (n, N_RAYS) in m."""
    n = len(x)
    lengths = np.full(n, config.RAY_RANGE) if lengths is None else np.asarray(lengths, dtype=float)
    reach = np.repeat(lengths, config.N_RAYS)
    caps = np.repeat(ray_caps(lengths), config.N_RAYS)
    angles = (np.asarray(heading)[:, None] + config.RAY_ANGLES[None, :]).ravel()
    dx, dy = np.cos(angles), np.sin(angles)
    ox, oy = np.repeat(x, config.N_RAYS), np.repeat(y, config.N_RAYS)
    t = np.zeros(n * config.N_RAYS)
    active = np.arange(n * config.N_RAYS)
    for iteration in range(int(caps.max()) if caps.size else 0):
        active = active[caps[active] > iteration]          # a ray stops at its own cap
        if active.size == 0:
            break
        d = wall_distance(track, ox[active] + dx[active] * t[active], oy[active] + dy[active] * t[active])
        hit = d <= 0.0
        t[active] = np.where(hit, t[active], t[active] + np.maximum(d, config.RAY_MIN_STEP))
        active = active[~hit & (t[active] < reach[active])]
    return np.minimum(t, reach).reshape(n, config.N_RAYS)


def sense(track, cars):
    """Network inputs: ray distances / RAY_RANGE (each car's rays reach its own sensor length, so a ray input can
    exceed 1; revision 3.1 §2.1) and speed / SPEED_SCALE. Rows of cars that are done stay zero."""
    obs = np.zeros((len(cars.x), config.N_INPUTS))
    live = np.flatnonzero(~cars.done)
    if live.size:
        rays = cast_rays(track, cars.x[live], cars.y[live], cars.heading[live], cars.ray_length[live])
        obs[live, :config.N_RAYS] = rays / config.RAY_RANGE
        obs[live, config.N_RAYS] = cars.speed[live] / config.SPEED_SCALE
    return obs
```

5. In `step`, replace the signature, docstring and the four grip lines at its top:

```python
def step(track, cars, steer, throttle, time_limit, grip=config.GRIP):
    """Advance every car that is not done by one time step. steer and throttle are grip fractions in [-1, 1]; each
    car's steering is multiplied by its steering sensitivity before the grip limit (revision 3.1 §2.2). `grip` is the
    friction-circle limit: the training grip unless the application shows another (revision 3.1 §6)."""
    live = ~cars.done
    a_lat = np.clip(steer * cars.steer_gain, -1.0, 1.0) * grip
    a_long = np.clip(throttle, -1.0, 1.0) * grip
    total = np.hypot(a_lat, a_long)
    scale = np.where(total > grip, grip / np.maximum(total, 1e-12), 1.0)
    a_lat, a_long = a_lat * scale, a_long * scale                       # friction circle
```

(The rest of `step` is unchanged. `steer * 1.0` equals `steer` exactly, so the default setup reproduces revision 3.)

6. Replace the head of `simulate` (signature, docstring and the two lines that create the cars and step them) so it
   reads:

```python
def simulate(track, policy, n, time_limit=None, laps=1, setup=None, grip=config.GRIP):
    """Drive n cars from the start line until every car is done. policy(obs, cars) returns action classes. setup:
    the cars' (n, 2) setups, None for revision 3's; grip: the friction-circle limit (revision 3.1 §2, §6)."""
    if time_limit is None:
        time_limit = time_limit_for(track, laps)
    cars = start_cars(track, n, laps, setup)
    steps = 0
    while not cars.done.all():
        steer, throttle = class_to_controls(policy(sense(track, cars), cars))
        step(track, cars, steer, throttle, time_limit, grip=grip)
        steps += 1
```

(the `return RunResult(...)` that follows is unchanged).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_sim.py -q`
Expected: all pass (the 12 new tests and the existing ones).

- [ ] **Step 6: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all passed (469 before this task + 12 = 481; nothing else changes, because every other caller uses the
defaults).

- [ ] **Step 7: Commit**

```bash
git add racing/config.py racing/sim.py tests/test_sim.py
git commit -m "Simulator: each car's sensor length and steering sensitivity, grip as a step parameter, ray caps scaled by length

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/config.py racing/sim.py tests/test_sim.py
```


---

### Task 2: Setup module; populations, drivers, evaluation paths and saved controllers carry setups

Revision 3.1 §2–§4. A new `racing/setup.py` holds the shared setup rules that every method uses (Task 3): defaults,
log-uniform random draws, crossover, mutation, clipping, the JSON form and population statistics. Every population
(`nets.DensePopulation`, `neat.NeatPopulation`) carries an `(n, 2)` `setup` array that `subset` keeps aligned, and
every driver (`nets.DenseNet`, `neat.NeatNet`) carries its `(2,)` setup. Every simulation of a population or of drivers
passes the setups to `sim.simulate` / `sim.start_cars`: the time trials of training and final selection, the
evaluation drives, and DAgger's drives and checks. A saved controller stores its setup in `<seed>.setup.json`; a
controller without that file (the imitation stage files) drives with revision 3's setup. Nothing silently falls back
to the default for a car that has its own setup (Review Focus 2): the population interface now includes `.setup`,
and the test fakes that stand in for populations and drivers gain it.

**Files:**
- Create: `racing/setup.py`, `tests/test_setup.py`
- Modify: `racing/config.py` (car setup block)
- Modify: `racing/nets.py` (`DensePopulation`, `DenseNet`, `from_flat`, new `drivers_setup`, `load_champion`)
- Modify: `racing/neat.py` (`NeatNet`, `NeatPopulation`, `save_neat_champion`)
- Modify: `racing/population.py` (`_time_trial`, `save_dense_champion`, `clear_champion`)
- Modify: `racing/evaluate.py` (`_drive_track` only)
- Modify: `racing/dagger.py` (`drive_and_label`, `_time_trial`)
- Modify: `tests/test_population.py` (`ExpertCars` fake, `fake_simulate` signature, `test_clear_champion_removes_every_file`)
- Modify: `tests/test_dagger.py` (`Constant` and `Expert` fake drivers)

**Interfaces:**
- Consumes: Task 1: `sim.start_cars(track, n, laps=1, setup=None)`, `sim.simulate(..., setup=None, grip=config.GRIP)`,
  `config.DEFAULT_SETUP`; existing `paths.champion_path(approach, seed, suffix, champion_dir)`.
- Produces (used by Tasks 3–6): `config.SENSOR_LENGTH_LIMITS`, `config.STEER_GAIN_LIMITS`, `config.SETUP_MUTATION_P`,
  `config.SETUP_MUTATION_SD`; module `racing.setup` with `COLUMNS`, `default(n)`, `as_rows(values, n)`,
  `as_row(value)`, `random(rng, n)`, `clip(values)`, `crossover(rng, a, b)`, `mutate(rng, values)`, `to_dict(row)`,
  `from_dict(d)`, `summary(values) -> {"sensor_mean", "sensor_std", "steer_mean", "steer_std"}`;
  `nets.DensePopulation(weights, hidden=..., activation="relu", setup=None)` and `neat.NeatPopulation(genomes,
  setup=None)` with `.setup (n, 2)`; `nets.DenseNet(layers, activations, setup=None)`, `nets.from_flat(flat, hidden,
  activation, setup=None)`, `neat.NeatNet(genome, setup=None)` with `.setup (2,)`; `nets.drivers_setup(drivers) ->
  (len(drivers), 2)`; `neat.save_neat_champion(approach, seed, genome, champion_dir=..., setup=None)`;
  `population.save_dense_champion` writes `<seed>.setup.json`; `nets.load_champion` reads it into `driver.setup`;
  `population.clear_champion` also removes it. Modules import the setup module as `from racing import setup as
  car_setup`, so the name never clashes with a `setup` parameter.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_setup.py`:

```python
import json

import numpy as np
import pytest

from racing import config, setup


def test_setup_parameters():
    assert config.SENSOR_LENGTH_LIMITS == (10.0, 150.0) and config.STEER_GAIN_LIMITS == (1.0 / 3.0, 3.0)
    assert (config.SETUP_MUTATION_P, config.SETUP_MUTATION_SD) == (0.2, 0.1)
    assert setup.COLUMNS == ("sensor_length", "steer_gain")


def test_default_rows_are_revision_3s_setup():
    rows = setup.default(4)
    assert rows.shape == (4, 2) and rows.dtype == np.float64
    assert np.all(rows == np.asarray(config.DEFAULT_SETUP))
    assert np.array_equal(setup.as_row(None), config.DEFAULT_SETUP)


def test_as_rows_copies_and_checks_the_shape():
    values = np.array([[20.0, 2.0], [30.0, 0.5]])
    rows = setup.as_rows(values, 2)
    values[0, 0] = 99.0
    assert rows[0, 0] == 20.0                                    # a copy, not a view
    assert np.array_equal(setup.as_rows(None, 3), setup.default(3))
    with pytest.raises(ValueError, match="setup"):
        setup.as_rows(values, 3)


def test_random_setups_are_log_uniform_within_the_limits():
    rows = setup.random(np.random.default_rng(0), 20000)
    for column, (low, high) in enumerate((config.SENSOR_LENGTH_LIMITS, config.STEER_GAIN_LIMITS)):
        values = rows[:, column]
        assert values.min() >= low and values.max() <= high
        assert abs(np.mean(values < np.sqrt(low * high)) - 0.5) < 0.02          # median at the geometric mean
        assert abs(np.mean(values < low * (high / low) ** 0.25) - 0.25) < 0.02  # a quarter in the first log quarter


def test_random_makes_one_uniform_draw():
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    setup.random(a, 5)
    b.uniform(size=(5, 2))
    assert a.bit_generator.state == b.bit_generator.state


def test_clip_keeps_values_within_the_limits():
    clipped = setup.clip(np.array([[1.0, 0.1], [500.0, 9.0], [60.0, 1.5]]))
    low = [config.SENSOR_LENGTH_LIMITS[0], config.STEER_GAIN_LIMITS[0]]
    high = [config.SENSOR_LENGTH_LIMITS[1], config.STEER_GAIN_LIMITS[1]]
    assert np.array_equal(clipped, [low, high, [60.0, 1.5]])


def test_crossover_takes_each_value_from_either_parent():
    a, b = np.full((5000, 2), 20.0), np.full((5000, 2), 80.0)
    child = setup.crossover(np.random.default_rng(2), a, b)
    assert set(np.unique(child)) == {20.0, 80.0}
    assert abs(np.mean(child == 20.0) - 0.5) < 0.02
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    setup.crossover(r1, a[:4], b[:4])
    r2.random((4, 2))
    assert r1.bit_generator.state == r2.bit_generator.state                     # one draw


def test_mutation_changes_a_fifth_of_the_values_by_about_ten_percent():
    rows = np.tile([50.0, 1.0], (20000, 1))
    mutated = setup.mutate(np.random.default_rng(4), rows)
    changed = mutated != rows
    assert abs(changed.mean() - config.SETUP_MUTATION_P) < 0.01
    ratios = np.log(mutated[changed] / rows[changed])
    assert abs(ratios.std() - config.SETUP_MUTATION_SD) < 0.005 and abs(ratios.mean()) < 0.005


def test_mutation_stays_within_the_limits_and_always_draws_two_arrays():
    edge = np.tile([config.SENSOR_LENGTH_LIMITS[1], config.STEER_GAIN_LIMITS[0]], (1000, 1))
    mutated = setup.mutate(np.random.default_rng(5), edge)
    assert np.all(mutated[:, 0] <= config.SENSOR_LENGTH_LIMITS[1])
    assert np.all(mutated[:, 1] >= config.STEER_GAIN_LIMITS[0])
    r1, r2 = np.random.default_rng(6), np.random.default_rng(6)
    setup.mutate(r1, edge[:3])
    r2.random((3, 2))
    r2.normal(0.0, config.SETUP_MUTATION_SD, (3, 2))
    assert r1.bit_generator.state == r2.bit_generator.state


def test_a_setup_round_trips_through_its_json_form():
    row = np.array([73.5, 0.62])
    d = setup.to_dict(row)
    assert d == {"sensor_length": 73.5, "steer_gain": 0.62} and json.loads(json.dumps(d)) == d
    assert np.array_equal(setup.from_dict(d), row)


def test_summary_reports_means_and_spreads():
    assert setup.summary(np.array([[40.0, 1.0], [60.0, 2.0]])) == {
        "sensor_mean": 50.0, "sensor_std": 10.0, "steer_mean": 1.5, "steer_std": 0.5}


# ----- setups travel with the cars (revision 3.1 §2, §4; Review Focus 2) -----

from racing import dagger, evaluate, nets, neat, population, sim  # noqa: E402

ROWS = np.array([[20.0, 2.0], [120.0, 0.5], [50.0, 1.0], [80.0, 3.0]])


def dense_with_setup():
    return nets.DensePopulation(nets.init_population(np.random.default_rng(0), 4), setup=ROWS)


def neat_genomes(n, seed=1):
    rng, innovations = np.random.default_rng(seed), neat.Innovations()
    return [neat.initial_genome(rng, innovations) for _ in range(n)]


def recording_simulate(monkeypatch, seen):
    """Wrap sim.simulate (one module object, seen by every caller) to record each call's setup."""
    real = sim.simulate

    def recording(track, policy, n, time_limit=None, laps=1, setup=None, grip=config.GRIP):
        seen.append(None if setup is None else np.array(setup))
        return real(track, policy, n, time_limit=time_limit, laps=laps, setup=setup, grip=grip)

    monkeypatch.setattr(sim, "simulate", recording)


def test_a_dense_population_carries_its_setup_through_subset_and_driver():
    rows = ROWS.copy()
    cars = nets.DensePopulation(nets.init_population(np.random.default_rng(0), 4), setup=rows)
    rows[0, 0] = 99.0                                            # the population keeps its own copy
    assert np.array_equal(cars.setup, ROWS)
    assert np.array_equal(cars.subset([3, 1]).setup, ROWS[[3, 1]])
    assert np.array_equal(cars.driver(1).setup, ROWS[1])
    assert np.array_equal(nets.DensePopulation(cars.weights).setup, setup.default(4))
    assert np.array_equal(nets.from_flat(cars.weights[0]).setup, config.DEFAULT_SETUP)


def test_a_neat_population_carries_its_setup_through_subset_and_driver():
    genomes = neat_genomes(4)
    cars = neat.NeatPopulation(genomes, ROWS)
    assert np.array_equal(cars.subset([2, 0]).setup, ROWS[[2, 0]])
    assert np.array_equal(cars.driver(3).setup, ROWS[3])
    assert np.array_equal(neat.NeatPopulation(genomes).setup, setup.default(4))
    assert np.array_equal(neat.NeatNet(genomes[0]).setup, config.DEFAULT_SETUP)


def test_drivers_setup_lists_each_drivers_setup_in_order():
    cars = dense_with_setup()
    drivers = [cars.driver(i) for i in (2, 0, 3)] + [neat.NeatPopulation(neat_genomes(1), ROWS[1:2]).driver(0)]
    assert np.array_equal(nets.drivers_setup(drivers), ROWS[[2, 0, 3, 1]])


def test_parallel_evaluation_gives_every_task_its_cars_setups(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    rows = setup.random(np.random.default_rng(3), 9)
    cars = nets.DensePopulation(nets.init_population(np.random.default_rng(2), 9), setup=rows)
    seen = []
    recording_simulate(monkeypatch, seen)
    serial = population.evaluate_population(cars, [0, 1, 2], 1, config.TARGET_FACTOR)
    threaded = population.evaluate_population(cars, [0, 1, 2], 1, config.TARGET_FACTOR, n_jobs=6,
                                              backend="threading")
    for field in ("fitness", "success", "laps"):
        assert np.array_equal(getattr(serial, field), getattr(threaded, field))
    assert len(seen) > 6 and all(s is not None for s in seen)    # the threaded run splits the population
    stacked = np.vstack(seen)
    for row in rows:                                              # each car's row once per map and evaluation
        assert np.sum(np.all(stacked == row, axis=1)) == 2 * 3


def test_final_selection_drives_the_candidates_with_their_setups(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    cars = dense_with_setup()
    seen = []
    recording_simulate(monkeypatch, seen)
    population.select_final(cars, [3, 1], config.TARGET_FACTOR, seeds=[80])
    assert len(seen) == 1 and np.array_equal(seen[0], ROWS[[3, 1]])


def test_a_dense_champion_round_trips_its_setup(tmp_path):
    population.save_dense_champion("ga", 0, dense_with_setup(), 3, tmp_path)
    assert json.loads((tmp_path / "ga" / "seed0.setup.json").read_text()) == setup.to_dict(ROWS[3])
    _, driver = nets.load_champion("ga", 0, tmp_path)
    assert np.array_equal(driver.setup, ROWS[3])


def test_a_neat_champion_round_trips_its_setup(tmp_path):
    genome = neat_genomes(1)[0]
    neat.save_neat_champion("neat", 1, genome, tmp_path, setup=ROWS[1])
    _, driver = nets.load_champion("neat", 1, tmp_path)
    assert isinstance(driver, neat.NeatNet) and np.array_equal(driver.setup, ROWS[1])
    neat.save_neat_champion("neat", 2, genome, tmp_path)                       # no setup given: revision 3's
    assert json.loads((tmp_path / "neat" / "seed2.setup.json").read_text()) == setup.to_dict(config.DEFAULT_SETUP)


def test_a_controller_without_a_setup_file_drives_with_revision_3s_setup(tmp_path):
    (tmp_path / "baseline").mkdir()
    nets.to_keras(nets.init_population(np.random.default_rng(5), 1)[0]).save(tmp_path / "baseline" / "seed0.keras")
    _, driver = nets.load_champion("baseline", 0, tmp_path)
    assert np.array_equal(driver.setup, config.DEFAULT_SETUP)


def test_evaluation_drives_each_controller_with_its_setup(monkeypatch, circle_track):
    monkeypatch.setattr(evaluate, "get_track", lambda seed, level: circle_track)
    cars = dense_with_setup()
    drivers = {("ga", 0): cars.driver(3), ("es", 1): cars.driver(0)}
    seen = []
    recording_simulate(monkeypatch, seen)
    evaluate.driving_results(drivers, [0], level=1)
    assert len(seen) == 1 and np.array_equal(seen[0], ROWS[[3, 0]])


def test_dagger_drives_and_checks_with_the_drivers_setup(monkeypatch, circle_track):
    monkeypatch.setattr(dagger, "get_track", lambda seed, level: circle_track)
    driver = dense_with_setup().driver(1)
    started = []
    real_start = sim.start_cars

    def recording_start(track, n, laps=1, setup=None):
        started.append(np.array(setup))
        return real_start(track, n, laps, setup)

    monkeypatch.setattr(sim, "start_cars", recording_start)
    dagger.drive_and_label(driver, 0, level=1)
    assert np.array_equal(started[0], ROWS[1:2])
    seen = []
    recording_simulate(monkeypatch, seen)
    dagger.check(driver, [0], level=1)
    assert len(seen) == 1 and np.array_equal(seen[0], ROWS[1:2])
```

In `tests/test_population.py`, add the setup module to the imports:

```python
from racing import config, expert, nets, population, sim
from racing import setup as car_setup
```

give the `ExpertCars` fake its setup (the population interface now includes `.setup`), so its `__init__` reads:

```python
    def __init__(self, track, n):
        self.track, self.n = track, n
        self.setup = car_setup.default(n)                # revision 3's sensors and steering (revision 3.1 §2)
```

in `test_time_trial_ends_at_the_target_time`, let the fake accept the new keyword:

```python
    def fake_simulate(track, policy, n, time_limit=None, laps=1, setup=None):
```

and in `test_clear_champion_removes_every_file`, replace the line that lists the saved files with:

```python
    assert sorted(p.name for p in (tmp_path / "es").iterdir()) == ["seed0.keras", "seed0.npy", "seed0.setup.json"]
```

In `tests/test_dagger.py`, give both fake drivers revision 3's setup, so the two classes start:

```python
class Constant:
    """A driver that always takes the same action, with revision 3's setup (revision 3.1 §3.1)."""

    setup = np.asarray(config.DEFAULT_SETUP, dtype=float)

    def __init__(self, action):
```

```python
class Expert:
    """The expert as a driver of one map, with revision 3's setup (revision 3.1 §3.1)."""

    setup = np.asarray(config.DEFAULT_SETUP, dtype=float)

    def __init__(self, track):
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_setup.py tests/test_population.py -q`
Expected: both files fail at collection with `ImportError: cannot import name 'setup' from 'racing'` (both import
the setup module, which does not exist yet). Once Step 4 has created it, `test_clear_champion_removes_every_file`
fails until Step 6 writes `seed0.setup.json`.

- [ ] **Step 3: Add the setup values to `racing/config.py`**

Extend the car setup block from Task 1 so it reads:

```python
# --- Car setup (revision 3.1 §2) ---
DEFAULT_SETUP = (RAY_RANGE, 1.0)   # (sensor length m, steering sensitivity) of every car in revision 3
SENSOR_LENGTH_LIMITS = (10.0, 150.0)   # m, the sensor lengths evolution may choose (revision 3.1 Table 1)
STEER_GAIN_LIMITS = (1.0 / 3.0, 3.0)   # the steering sensitivities evolution may choose
SETUP_MUTATION_P = 0.2       # chance that each setup value of a child is mutated (revision 3.1 §3.3)
SETUP_MUTATION_SD = 0.1      # standard deviation of that mutation on the natural-log scale (about ±10%)
```

(Task 3 adds these names to `checkpoint.SETTINGS_KEYS`.)

- [ ] **Step 4: Create `racing/setup.py`**

```python
"""Car setups (revision 3.1 §2-3): each car's sensor length and steering sensitivity, as one row of an (n, 2)
array next to its network. Every method draws, inherits and mutates setups with these rules, so setup evolution
favours no method. Values change on the natural-log scale, so halving and doubling are equally large steps."""
import numpy as np

from racing import config

COLUMNS = ("sensor_length", "steer_gain")


def _limits():
    """(low, high) arrays of the two columns, read from config at call time."""
    low = np.array([config.SENSOR_LENGTH_LIMITS[0], config.STEER_GAIN_LIMITS[0]], dtype=float)
    high = np.array([config.SENSOR_LENGTH_LIMITS[1], config.STEER_GAIN_LIMITS[1]], dtype=float)
    return low, high


def default(n):
    """n rows of revision 3's setup (config.DEFAULT_SETUP)."""
    return np.tile(np.asarray(config.DEFAULT_SETUP, dtype=float), (int(n), 1))


def as_rows(values, n):
    """The (n, 2) float64 setup array of n cars: a copy of `values`, or default(n) for None."""
    if values is None:
        return default(n)
    values = np.array(values, dtype=float)
    if values.shape != (int(n), 2):
        raise ValueError(f"setup must have shape ({n}, 2), one (sensor length, steering sensitivity) row per car; "
                         f"got {values.shape}")
    return values


def as_row(value):
    """The (2,) float64 setup of one car: a copy of `value`, or revision 3's for None."""
    return as_rows(None if value is None else np.asarray(value, dtype=float)[None, :], 1)[0]


def random(rng, n):
    """n setups drawn uniformly on the log scale within the limits (revision 3.1 §3.2): one rng.uniform call."""
    low, high = _limits()
    return np.exp(rng.uniform(np.log(low), np.log(high), (int(n), 2)))


def clip(values):
    """Setups moved into the limits of revision 3.1 Table 1."""
    low, high = _limits()
    return np.clip(np.asarray(values, dtype=float), low, high)


def crossover(rng, a, b):
    """Each value from setup a or setup b with equal probability (revision 3.1 §3.3): one rng.random call."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.where(rng.random(a.shape) < 0.5, a, b)


def mutate(rng, values):
    """Each value, with probability SETUP_MUTATION_P, multiplied by exp(N(0, SETUP_MUTATION_SD)), then clipped
    (revision 3.1 §3.3). Always draws two arrays of values.shape, the mask and then the noise."""
    values = np.asarray(values, dtype=float)
    mask = rng.random(values.shape) < config.SETUP_MUTATION_P
    noise = rng.normal(0.0, config.SETUP_MUTATION_SD, values.shape)
    return clip(np.where(mask, values * np.exp(noise), values))


def to_dict(row):
    """One setup as JSON-ready values: {"sensor_length": m, "steer_gain": k}."""
    return {name: float(value) for name, value in zip(COLUMNS, np.asarray(row, dtype=float))}


def from_dict(d):
    """The (2,) setup row of to_dict's form."""
    return np.array([float(d[name]) for name in COLUMNS])


def summary(values):
    """Mean and standard deviation of both setup values of a population, for logs and the application."""
    values = np.asarray(values, dtype=float)
    return {"sensor_mean": float(values[:, 0].mean()), "sensor_std": float(values[:, 0].std()),
            "steer_mean": float(values[:, 1].mean()), "steer_std": float(values[:, 1].std())}
```

- [ ] **Step 5: Populations and drivers carry their setup (`racing/nets.py`, `racing/neat.py`)**

In `racing/nets.py`:

1. Add `from racing import setup as car_setup` after `from racing import config, paths`.
2. Replace `class DensePopulation` and `class DenseNet`'s `__init__` and docstring, and `from_flat`:

```python
class DensePopulation:
    """Dense networks of one architecture, one flat weight vector and one setup row per car (revision 3 §5,
    revision 3.1 §2). Like every population type, it offers len(), subset(indices), policy(), driver(i) and setup."""

    def __init__(self, weights, hidden=config.BASELINE_HIDDEN, activation="relu", setup=None):
        self.weights = np.asarray(weights, dtype=float)
        self.hidden = tuple(int(units) for units in hidden)
        self.activation = activation
        self.setup = car_setup.as_rows(setup, len(self.weights))

    def __len__(self):
        return len(self.weights)

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return DensePopulation(self.weights[indices], self.hidden, self.activation, self.setup[indices])

    def policy(self):
        return population_policy(self.weights, self.hidden, self.activation)

    def driver(self, i):
        return from_flat(self.weights[int(i)], self.hidden, self.activation, self.setup[int(i)])


class DenseNet:
    """A plain stack of dense layers in numpy. Drives a car with any trained controller, with the car's setup
    (revision 3.1 §2; revision 3's when none is given)."""

    def __init__(self, layers, activations, setup=None):
        self.layers = [(np.asarray(w, dtype=float), np.asarray(b, dtype=float)) for w, b in layers]
        self.activations = list(activations)
        self.setup = car_setup.as_row(setup)
```

(the rest of `DenseNet` is unchanged), and

```python
def from_flat(flat, hidden=config.BASELINE_HIDDEN, activation="relu", setup=None):
    layers = [(w[0], b[0]) for w, b in unflatten(np.asarray(flat, dtype=float)[None, :], hidden)]
    return DenseNet(layers, [activation] * len(hidden), setup)
```

3. Add directly above `def mixed_policy`:

```python
def drivers_setup(drivers):
    """The (len(drivers), 2) setups of a list of drivers, in order, for sim.simulate (revision 3.1 §2)."""
    return car_setup.as_rows([np.asarray(driver.setup, dtype=float) for driver in drivers], len(drivers))
```

4. In `load_champion`, extend the docstring's last sentence with " The driver has the setup saved in
   <seed>.setup.json, or revision 3's without that file (revision 3.1 §4)." and insert before `return model, driver`:

```python
    setup_path = paths.champion_path(approach, seed, ".setup.json", directory)
    if setup_path.exists():
        driver.setup = car_setup.from_dict(json.loads(setup_path.read_text()))
```

In `racing/neat.py`:

1. Add `from racing import setup as car_setup` after `from racing import config, paths`.
2. In `class NeatNet`, replace the docstring and the first line of `__init__`:

```python
class NeatNet:
    """One NEAT controller as one masked dense matrix per depth; drives a car like nets.DenseNet, with the car's
    setup (revision 3.1 §2; revision 3's when none is given)."""

    def __init__(self, genome, setup=None):
        self.genome = genome
        self.setup = car_setup.as_row(setup)
```

(the rest of `__init__` is unchanged).
3. Replace `class NeatPopulation`'s docstring, `__init__`, `subset` and `driver`:

```python
class NeatPopulation:
    """A NEAT population with the population interface of nets.DensePopulation: one genome and one setup row per
    car (revision 3.1 §2)."""

    def __init__(self, genomes, setup=None):
        self.genomes = list(genomes)
        self.setup = car_setup.as_rows(setup, len(self.genomes))

    def __len__(self):
        return len(self.genomes)

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.int64).ravel()
        return NeatPopulation([self.genomes[int(i)] for i in indices], self.setup[indices])
```

(`policy` is unchanged) and

```python
    def driver(self, i):
        return NeatNet(self.genomes[int(i)], self.setup[int(i)])
```

4. Replace `save_neat_champion` with:

```python
def save_neat_champion(approach, seed, genome, champion_dir=config.CHAMPIONS_DIR, setup=None):
    """Final NEAT controller (revision 3 §5.4): its genome as <seed>.json, which drives, the equivalent Keras model
    as <seed>.keras, and its setup as <seed>.setup.json (revision 3.1 §4; revision 3's when setup is None). Returns
    the Keras file's path."""
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_keras(genome).save(path)
    paths.champion_path(approach, seed, ".json", champion_dir).write_text(json.dumps(genome.to_dict()))
    paths.champion_path(approach, seed, ".setup.json", champion_dir).write_text(
        json.dumps(car_setup.to_dict(car_setup.as_row(setup))))
    return path
```

(`NEAT.save_champion` keeps calling it without `setup` until Task 3 passes the champion's setup.)

- [ ] **Step 6: Every simulation passes the setups (`racing/population.py`, `racing/evaluate.py`, `racing/dagger.py`)**

In `racing/population.py`, add `import json` above `from dataclasses import dataclass` and
`from racing import setup as car_setup` after `from racing import config, nets, paths, sim, tracks`; then replace
`_time_trial`, `save_dense_champion` and `clear_champion` with:

```python
def _time_trial(population, seed, level, factor):
    """Fitness, success, laps in time and finish time of every car over config.LAPS laps of one map, ending at the
    target time (revision 3 §4.1). Every car drives with its own setup (revision 3.1 §2)."""
    track = get_track(seed, level)
    result = sim.simulate(track, population.policy(), len(population), time_limit=sim.target_time(track, factor),
                          laps=config.LAPS, setup=population.setup)
    return (result.fitness, result.finished, np.minimum(result.laps_done, config.LAPS),
            np.where(result.finished, result.finish_time, np.nan))
```

```python
def save_dense_champion(approach, seed, population, index, champion_dir=config.CHAMPIONS_DIR):
    """Final dense controller (spec §6, revision 3 §6.4): a Keras file, the float64 weights (.npy) and the car's
    setup (<seed>.setup.json, revision 3.1 §4); the architecture is the Keras file's. Returns the Keras file's path."""
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = population.weights[int(index)]
    nets.to_keras(flat, population.hidden, population.activation).save(path)
    np.save(paths.champion_path(approach, seed, ".npy", champion_dir), np.asarray(flat, dtype=float))
    paths.champion_path(approach, seed, ".setup.json", champion_dir).write_text(
        json.dumps(car_setup.to_dict(population.setup[int(index)])))
    return path


def clear_champion(approach, seed, champion_dir=config.CHAMPIONS_DIR):
    """Remove a final controller left by an earlier run (.keras, .npy, a NEAT genome's .json and the setup's
    .setup.json), so an interrupted run leaves none behind."""
    paths.remove(*(paths.champion_path(approach, seed, suffix, champion_dir)
                   for suffix in (".keras", ".npy", ".json", ".setup.json")))
```

In `racing/evaluate.py` `_drive_track`, replace the `result = sim.simulate(...)` line with:

```python
    chosen = [drivers[k] for k in keys]
    result = sim.simulate(track, nets.mixed_policy(chosen), len(keys), laps=config.LAPS,
                          setup=nets.drivers_setup(chosen))      # every controller with its own setup (revision 3.1)
```

In `racing/dagger.py` `drive_and_label`, append " The car drives with its setup (revision 3's for an imitation
network, revision 3.1 §3.1)." before "Returns" in the docstring and replace the `cars = sim.start_cars(...)` line with:

```python
    cars = sim.start_cars(track, 1, config.LAPS, setup=nets.drivers_setup([driver]))
```

and in `_time_trial` replace the `result = sim.simulate(...)` line with:

```python
    result = sim.simulate(track, driver.policy(), 1, time_limit=sim.target_time(track, factor), laps=config.LAPS,
                          setup=nets.drivers_setup([driver]))
```

The expert (`expert.expert_lap_time`, `expert.collect_rows`) and `racing/bench.py` keep calling the simulator without
a setup: the expert's sensors and steering are revision 3's (revision 3.1 §2.3). The viewer's simulations follow in
Task 5.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_setup.py tests/test_population.py tests/test_dagger.py tests/test_nets.py tests/test_neat.py tests/test_evaluate.py -q`
Expected: all pass (`tests/test_setup.py`: 21 passed).

- [ ] **Step 8: Run the fast suite, and the slow tests of the paths that changed**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all passed (481 after Task 1 + 21 = 502).
Run: `~/miniconda3/bin/python -m pytest -q -m slow tests/test_dagger.py tests/test_evolution.py tests/test_end_to_end.py tests/test_neat.py`
Expected: 4 passed (about five minutes).

- [ ] **Step 9: Commit**

```bash
git add racing/setup.py racing/config.py racing/nets.py racing/neat.py racing/population.py racing/evaluate.py \
        racing/dagger.py tests/test_setup.py tests/test_population.py tests/test_dagger.py
git commit -m "Car setups: shared setup rules; populations, drivers, evaluation, DAgger and saved controllers carry each car's setup

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/setup.py racing/config.py racing/nets.py racing/neat.py \
        racing/population.py racing/evaluate.py racing/dagger.py tests/test_setup.py tests/test_population.py \
        tests/test_dagger.py
```

---

### Task 3: Methods and engine evolve setups (GA, ES, NEAT); logs, reports, summaries, settings

Revision 3.1 §2–§5, §7. Every evolutionary population carries one setup row per car (sensor length, steering
sensitivity). This task makes GA, ES and NEAT create, inherit and mutate those rows with the shared rules of
`racing/setup.py` (Task 2), keeps them in checkpoints, and makes the engine log, report and summarise them. The order of
the network's random draws inside `advance` is unchanged: setup draws always come last, so the network part of a
generation is drawn exactly as in revision 3 (pinned by the two "draws come after" tests).

**Files:**
- Modify: `racing/checkpoint.py` (`SETTINGS_KEYS`)
- Modify: `racing/evolution.py` (imports, `LOG_FIELDS`, `Method` docstring, `DenseMethod.setup`, `DenseMethod.assemble`, the report, the summary)
- Modify: `racing/ga.py` (module docstring, imports, `GA.initialize`, `GA.advance`, `GA.state`, `GA.load_state`)
- Modify: `racing/es.py` (module docstring, imports, `ES.initialize`, `ES.advance`, `ES.state`, `ES.load_state`)
- Modify: `racing/neat.py` (`NEAT` docstring, `NEAT.setup`, `NEAT.initialize`, `NEAT.advance`, `NEAT.state`, `NEAT.load_state`, `NEAT.save_champion`)
- Test: `tests/test_checkpoint.py`, `tests/test_ga.py`, `tests/test_es.py`, `tests/test_neat.py`, `tests/test_evolution.py`

**Interfaces:**
- Consumes (Tasks 1–2, Interface Contract): `config.SENSOR_LENGTH_LIMITS`, `config.STEER_GAIN_LIMITS`,
  `config.DEFAULT_SETUP`, `config.SETUP_MUTATION_P`, `config.SETUP_MUTATION_SD`; `setup.default(n)`,
  `setup.random(rng, n)`, `setup.crossover(rng, a, b)`, `setup.mutate(rng, values)`, `setup.to_dict(row)`,
  `setup.from_dict(d)`, `setup.summary(values)`; `nets.DensePopulation(weights, hidden, activation, setup=None)` and
  `neat.NeatPopulation(genomes, setup=None)` with `.setup` (n, 2) kept by `subset`; drivers with `.setup`;
  `population.save_dense_champion` and `neat.save_neat_champion(approach, seed, genome, champion_dir, setup=None)` writing
  `<seed>.setup.json`; `nets.load_champion` reading it. Existing: `evolution.run`, `evolution.children_success`,
  `evolution.DenseMethod`, `checkpoint.settings_record`, `checkpoint.SettingsMismatch`.
- Produces: `GA.setup`, `ES.setup`, `NEAT.setup` (properties: the current population's setup, kept rows first);
  `DenseMethod.assemble(kept_weights, children, kept_setups, children_setups)`; method `state()` dicts with a `"setup"`
  entry (an (n, 2) array); `evolution.LOG_FIELDS` with `"sensor_mean", "sensor_std", "steer_mean", "steer_std"` (after
  `"median_lap"`); reports with those four keys (from `setup.summary(population.setup)` of the evaluated population);
  `summary["final_setup"]` (`setup.to_dict` of the saved controller's setup, `None` when no controller is saved);
  `checkpoint.SETTINGS_KEYS` with `"SENSOR_LENGTH_LIMITS"`, `"STEER_GAIN_LIMITS"`, `"DEFAULT_SETUP"`,
  `"SETUP_MUTATION_P"`, `"SETUP_MUTATION_SD"`.

Initial setups (revision 3.1 §3.2): GA, ES without `initial` and NEAT draw `setup.random(rng, n)` right after their initial
networks; ES with `initial` (seeded ES, speed pushes in `racing/dagger.py`) uses `setup.default(n)` (no draw, so its
initial weights stay exactly those of revision 3). Inheritance (§3.3): kept cars keep their rows; GA children
`setup.mutate(rng, setup.crossover(rng, setups[mothers], setups[fathers]))`; ES children `setup.mutate(rng,
setups[chosen])`; NEAT children `setup.mutate(rng, setups[fitter_rows])` where `fitter_rows` is each child's fitter (or
only) parent row. `racing/dagger.py` needs no change: its speed push builds `es.ES(initial=...)`, which now starts at the
default setup.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_checkpoint.py`:

```python
def test_the_setup_settings_are_recorded():
    record = checkpoint.settings_record()
    for name in ("SENSOR_LENGTH_LIMITS", "STEER_GAIN_LIMITS", "DEFAULT_SETUP", "SETUP_MUTATION_P", "SETUP_MUTATION_SD"):
        assert name in record
    assert not {"APP_GRIPS", "APP_CHAMPIONS_DIR", "N_JOBS"} & set(record)
```

Append to `tests/test_ga.py` (it already imports `math`, `numpy as np`, `config`, `ga`, `mutation_strength` and defines
`result(fitness, success)` and `started(pop_size=30, **kwargs)`):

```python
def log_spread(values):
    """Standard deviation of each setup column on the log scale."""
    return np.std(np.log(values), axis=0)


def test_initial_setups_spread_within_the_limits():
    method = started(pop_size=200, hidden=(1,))
    values = method.setup
    assert values.shape == (200, 2) and values is method.population.setup
    assert np.all((values[:, 0] >= config.SENSOR_LENGTH_LIMITS[0]) & (values[:, 0] <= config.SENSOR_LENGTH_LIMITS[1]))
    assert np.all((values[:, 1] >= config.STEER_GAIN_LIMITS[0]) & (values[:, 1] <= config.STEER_GAIN_LIMITS[1]))
    assert log_spread(values).min() > 0.5        # log-uniform: about 0.78 for the length, 0.63 for the sensitivity


def test_children_take_each_setup_value_from_a_parent(monkeypatch):
    monkeypatch.setattr(config, "SETUP_MUTATION_P", 0.0)
    method = started(pop_size=40, hidden=(1,))
    before = method.setup.copy()
    kept = np.array([4, 9])
    method.advance(np.random.default_rng(1), result(np.random.default_rng(2).random(40), np.zeros(40)), kept, None)
    after = method.setup
    assert np.array_equal(after[:2], before[kept])                       # kept cars keep their setup
    for column in range(2):
        assert set(after[2:, column]) <= set(before[:, column])          # every child value is a parent's value
    parents = {tuple(row) for row in before}
    assert any(tuple(row) not in parents for row in after[2:])          # crossover mixes two parents' values


def test_setup_draws_come_after_the_network_draws():
    method = started(pop_size=30, hidden=(2,), s_max=0.1)
    weights = method.population.weights.copy()
    fitness = np.random.default_rng(5).random(30)
    trial = result(fitness, np.zeros(30))
    method.advance(np.random.default_rng(6), trial, np.array([0, 1, 2]), None)
    rng, n_new, p = np.random.default_rng(6), 27, weights.shape[1]      # the network part of revision 3, by hand
    strength = mutation_strength(0.0, 0.1, 0.1 * config.GA_S_MIN / config.GA_S_MAX)
    mothers = ga.tournament(rng, fitness, n_new)
    fathers = ga.tournament(rng, fitness, n_new)
    children = np.where(rng.random((n_new, p)) < config.GA_CROSSOVER_P, weights[mothers], weights[fathers])
    mutate = rng.random((n_new, p)) < config.GA_MUTATION_P
    children = children + mutate * rng.normal(0.0, strength, (n_new, p))
    assert np.array_equal(method.population.weights[3:], children)


def test_children_setups_mutate_at_the_configured_rate():
    method = started(pop_size=2000, hidden=(1,))
    before = method.setup.copy()
    method.advance(np.random.default_rng(4), result(np.random.default_rng(3).random(2000), np.zeros(2000)),
                   np.array([0]), None)
    columns = [set(before[:, 0]), set(before[:, 1])]
    changed = np.array([[value not in columns[c] for c, value in enumerate(row)] for row in method.setup[1:]])
    assert abs(changed.mean() - config.SETUP_MUTATION_P) < 0.03


def test_state_round_trip_keeps_the_setups():
    first = started(pop_size=20)
    second = ga.GA(pop_size=20)
    second.load_state(first.state())
    assert np.array_equal(second.setup, first.setup)
    trial = result(np.arange(20.0), np.arange(20) < 4)
    first.advance(np.random.default_rng(7), trial, np.arange(6), None)
    second.advance(np.random.default_rng(7), trial, np.arange(6), None)
    assert np.array_equal(first.setup, second.setup)
```

Append to `tests/test_es.py` (it already imports `numpy as np`, `config`, `es`, `nets` and defines `result` and
`started(pop_size=20, **kwargs)`):

```python
def test_random_start_setups_spread_but_a_trained_start_uses_the_default():
    spread = started(pop_size=100).setup
    assert np.std(np.log(spread), axis=0).min() > 0.4
    flat = np.random.default_rng(1).normal(size=431)
    seeded = started(pop_size=50, initial=flat, approach="es_seeded").setup
    assert np.array_equal(seeded, np.tile(config.DEFAULT_SETUP, (50, 1)))


def test_children_inherit_their_parents_setup(monkeypatch):
    monkeypatch.setattr(config, "SETUP_MUTATION_P", 0.0)
    method = started(parents=4)
    before = dict(zip(method.ids.tolist(), method.setup.copy()))
    method.advance(np.random.default_rng(8), result(-np.arange(20.0), np.zeros(20)), np.arange(10), None)
    after = method.setup
    assert np.array_equal(after[:10], np.array([before[i] for i in range(10)]))     # kept cars keep their setup
    for row, parent in zip(after[10:], method.parent_ids[10:]):
        assert np.array_equal(row, before[int(parent)])


def test_setup_mutation_rate_and_size():
    method = started(pop_size=4001, parents=1, hidden=(1,))
    method.population = nets.DensePopulation(method.population.weights, (1,), "relu",
                                             setup=np.tile(config.DEFAULT_SETUP, (4001, 1)))   # far from the limits
    method.advance(np.random.default_rng(9), result(-np.arange(4001.0), np.zeros(4001)), np.array([0]), None)
    logs = np.log(method.setup[1:] / np.asarray(config.DEFAULT_SETUP))   # every child copies car 0
    changed = logs != 0
    assert abs(changed.mean() - config.SETUP_MUTATION_P) < 0.02
    assert abs(np.std(logs[changed]) - config.SETUP_MUTATION_SD) < 0.01


def test_state_round_trip_keeps_the_setups():
    first = started(parents=4)
    second = es.ES(pop_size=20, parents=4)
    second.load_state(first.state())
    trial = result(-np.arange(20.0), np.arange(20) < 3)
    first.advance(np.random.default_rng(9), trial, np.arange(10), None)
    second.advance(np.random.default_rng(9), trial, np.arange(10), None)
    assert np.array_equal(first.setup, second.setup)


def test_setup_draws_come_after_the_network_draws():
    method = started(parents=4)
    weights = method.population.weights.copy()
    fitness = -np.arange(20.0)
    method.advance(np.random.default_rng(6), result(fitness, np.zeros(20)), np.arange(10), None)
    rng = np.random.default_rng(6)                                      # the network part of revision 3, by hand
    top = np.argsort(-fitness, kind="stable")[:4]
    chosen = top[rng.integers(0, 4, 10)]
    children = weights[chosen] + rng.normal(0.0, method.step.sigma, (10, weights.shape[1]))
    assert np.array_equal(method.population.weights[10:], children)
```

Append to `tests/test_neat.py` (it already has `import pickle` and the helper `result_for(fitness, success, maps=1)`
near its end):

```python
def test_neat_children_inherit_the_setup_of_their_fitter_parent(monkeypatch):
    monkeypatch.setattr(config, "SETUP_MUTATION_P", 0.0)
    rng = np.random.default_rng(41)
    method = neat.NEAT(pop_size=30)
    method.initialize(rng)
    assert np.std(np.log(method.setup), axis=0).min() > 0.3              # random start: a wide spread
    before = dict(zip(method.ids.tolist(), method.setup.copy()))
    fitness = np.random.default_rng(42).random(30)
    method.advance(rng, result_for(fitness, np.zeros(30)), np.array([3, 7]), None)
    after = method.setup
    assert after.shape == (30, 2) and np.array_equal(after[:2], np.array([before[3], before[7]]))
    for row, parent in zip(after[2:], method.parent_ids[2:]):
        assert np.array_equal(row, before[int(parent)])


def test_neat_state_round_trip_keeps_the_setups():
    rng = np.random.default_rng(43)
    a = neat.NEAT(pop_size=20)
    a.initialize(rng)
    b = neat.NEAT(pop_size=20)
    b.load_state(pickle.loads(pickle.dumps(a.state())))
    assert np.array_equal(a.setup, b.setup)


def test_neat_champion_keeps_its_setup(tmp_path):
    import json
    from racing import paths, setup
    method = neat.NEAT(pop_size=6)
    method.initialize(np.random.default_rng(44))
    method.save_champion(4, 0, tmp_path)
    saved = json.loads(paths.champion_path("neat", 0, ".setup.json", tmp_path).read_text())
    assert np.array_equal(setup.from_dict(saved), method.setup[4])
```

In `tests/test_evolution.py`:

1. Change the import line `from racing import checkpoint, config, curriculum, evolution, ga, paths, population, progress`
   to `from racing import checkpoint, config, curriculum, evolution, ga, nets, paths, population, progress, setup`.
2. In `test_a_resumed_run_matches_an_uninterrupted_one` (Review Focus 2: resume exactness), after the line
   `assert np.array_equal(ends[0]["method"]["weights"], ends[1]["method"]["weights"])` add
   `    assert np.array_equal(ends[0]["method"]["setup"], ends[1]["method"]["setup"])`.
3. Append:

```python
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
```

(The existing slow `n_jobs` test of `tests/test_evolution.py` compares whole logs between worker counts; with the new
setup columns in the log it now also pins that setups do not depend on `n_jobs`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_ga.py tests/test_es.py tests/test_neat.py tests/test_evolution.py tests/test_checkpoint.py`
Expected: 17 failed, the rest passed. The failures are `AttributeError: 'GA' object has no attribute 'setup'` (5),
`AttributeError: 'ES' object has no attribute 'setup'` (4), `AttributeError: 'NEAT' object has no attribute 'setup'`
(3), `KeyError: 'setup'` (the resumed-run test: the checkpointed method state has no setup), `KeyError: 'sensor_mean'`,
`DID NOT RAISE SettingsMismatch`, an all-default setup seen by the evaluation, and `assert 'SENSOR_LENGTH_LIMITS' in
{...}`. The two `test_setup_draws_come_after_the_network_draws` tests already pass: they pin the revision 3 network draws,
which this task must not change.

- [ ] **Step 3: Record the setup settings**

In `racing/checkpoint.py`, in `SETTINGS_KEYS`, replace the line

```python
    "N_INPUTS", "STEER_LEVELS", "THROTTLE_LEVELS", "N_ACTIONS", "BASELINE_HIDDEN",
```

with

```python
    "N_INPUTS", "STEER_LEVELS", "THROTTLE_LEVELS", "N_ACTIONS", "BASELINE_HIDDEN",
    "SENSOR_LENGTH_LIMITS", "STEER_GAIN_LIMITS", "DEFAULT_SETUP", "SETUP_MUTATION_P", "SETUP_MUTATION_SD",
```

(Only these five names; never `APP_GRIPS`, `APP_CHAMPIONS_DIR` or `N_JOBS`. `RAY_RANGE` and `RAY_MAX_ITERATIONS` are
already recorded.)

- [ ] **Step 4: The engine logs, reports and summarises setups**

In `racing/evolution.py`:

1. Change the import line `from racing import checkpoint, config, population, progress` to
   `from racing import checkpoint, config, population, progress, setup`.
2. Replace `LOG_FIELDS` with:

```python
LOG_FIELDS = ["generation", "level", "factor", "evals", "best", "mean", "worst", "map_shares", "min_share", "n_kept",
              "kept_success", "child_success", "median_lap", "sensor_mean", "sensor_std", "steer_mean", "steer_std",
              "step_size", "streak", "event", "seeds", "elapsed"]
```

3. Replace the `Method` docstring with:

```python
    """What the loop needs from a method. `population` holds the cars carried over unchanged first; `is_kept`
    marks them; `population.setup` holds every car's setup (revision 3.1 §2). `advance` must assign a new population
    object (reports keep the evaluated one)."""
```

4. In `DenseMethod`, replace `assemble` with:

```python
    @property
    def setup(self):
        """Every car's setup, aligned with the population (revision 3.1 §2)."""
        return self.population.setup

    def assemble(self, kept_weights, children, kept_setups, children_setups):
        """The next population: the kept cars first, then the children, each with its setup (revision 3.1 §3.3);
        marks the kept cars."""
        self.population = DensePopulation(np.concatenate([kept_weights, children]), self.hidden, self.activation,
                                          setup=np.concatenate([kept_setups, children_setups]))
        self.is_kept = np.arange(len(self.population)) < len(kept_weights)
```

5. In `run`, in the report dictionary, replace `"elapsed": now, "event": event, **method.log_fields()}` with
   `"elapsed": now, "event": event, **setup.summary(method.population.setup), **method.log_fields()}` (the evaluated
   population: `advance` has not run yet).
6. In `run`, in the summary dictionary, replace `"champion_path": None}` with `"champion_path": None, "final_setup": None}`,
   and in the `summary.update(...)` after `population.select_final` replace

```python
        summary.update(val_successes=int(successes), val_fitness=float(fitness),
                       champion_path=str(method.save_champion(index, seed, champion_dir)))
```

with

```python
        summary.update(val_successes=int(successes), val_fitness=float(fitness),
                       champion_path=str(method.save_champion(index, seed, champion_dir)),
                       final_setup=setup.to_dict(method.population.setup[index]))
```

- [ ] **Step 5: GA setups**

Replace the whole of `racing/ga.py` with:

```python
"""Genetic algorithm over dense network weights (revision 3 §5.1): tournament selection, uniform crossover and
Gaussian mutation whose strength follows the share of the previous generation's children that succeeded. Every car
also carries a setup, inherited value by value from either parent and mutated (revision 3.1 §3)."""
import numpy as np

from racing import config, evolution, nets, paths, setup
from racing.curriculum import mutation_strength
from racing.nets import DensePopulation


def tournament(rng, fitness, count, size=config.GA_TOURNAMENT):
    """Indices of `count` winners, each the fittest of `size` individuals drawn from the whole population."""
    entrants = rng.integers(0, len(fitness), (count, size))
    return entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]


class GA(evolution.DenseMethod):
    """The GA as a method of the evolution loop (racing/evolution.py)."""
    approach = "ga"

    def __init__(self, pop_size=config.GA_POP, hidden=config.BASELINE_HIDDEN, activation="relu", s_max=config.GA_S_MAX):
        super().__init__(pop_size, hidden, activation)
        self.s_max = s_max
        self.step_size = float("nan")

    def initialize(self, rng):
        weights = nets.init_population(rng, self.pop_size, self.hidden)
        self.population = DensePopulation(weights, self.hidden, self.activation,
                                          setup=setup.random(rng, self.pop_size))   # random start (revision 3.1 §3.2)
        self.is_kept = np.zeros(self.pop_size, dtype=bool)
        self.step_size = float("nan")                  # generation 0 is not produced by mutation

    def advance(self, rng, result, kept, event):
        """Survivors first and unchanged, then children by tournament, uniform crossover and Gaussian mutation. A
        child takes each setup value from either parent, then the setup mutation applies (revision 3.1 §3.3); these
        draws come after the network's."""
        weights, setups = self.population.weights, self.population.setup
        n, p = weights.shape
        child_success = evolution.children_success(result, self.is_kept)
        self.step_size = mutation_strength(child_success, self.s_max, self.s_max * config.GA_S_MIN / config.GA_S_MAX)
        n_new = n - len(kept)
        mothers = tournament(rng, result.fitness, n_new)
        fathers = tournament(rng, result.fitness, n_new)
        children = np.where(rng.random((n_new, p)) < config.GA_CROSSOVER_P, weights[mothers], weights[fathers])
        mutate = rng.random((n_new, p)) < config.GA_MUTATION_P
        children = children + mutate * rng.normal(0.0, self.step_size, (n_new, p))
        children_setups = setup.mutate(rng, setup.crossover(rng, setups[mothers], setups[fathers]))
        self.assemble(weights[kept], children, setups[kept], children_setups)

    def state(self):
        return {"weights": self.population.weights, "setup": self.population.setup, "is_kept": self.is_kept,
                "step_size": self.step_size, "hidden": self.hidden, "activation": self.activation}

    def load_state(self, state):
        self.hidden, self.activation = tuple(state["hidden"]), state["activation"]
        self.population = DensePopulation(state["weights"], self.hidden, self.activation, setup=state["setup"])
        self.is_kept = state["is_kept"]
        self.step_size = state["step_size"]


def run_ga(seed=0, pop_size=config.GA_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1,
           on_generation=None, start_level=1, max_generations=None, s_max=config.GA_S_MAX):
    """The GA with run seed `seed` (revision 3 §5.1). Returns the run's summary (racing/evolution.py)."""
    return evolution.run(GA(pop_size=pop_size, s_max=s_max), seed, out_dir or paths.run_dir("ga", seed),
                         champion_dir, n_jobs, on_generation, start_level, max_generations)
```

(Before this task the file differs only in the module docstring, the `setup` import, `initialize`, `advance`, `state`
and `load_state`; `tournament`, `__init__` and `run_ga` are unchanged. If Task 1 or 2 changed anything else in
`racing/ga.py`, edit those five places instead of replacing the file.)

- [ ] **Step 6: ES setups**

In `racing/es.py`:

1. Replace the module docstring and the `from racing import ...` line with:

```python
"""Evolution strategy over dense network weights (revision 3 §5.2): each child is one of the fittest cars plus
Gaussian noise of size σ, with the two-phase step-size rule. Started from a trained network it is the seeded ES
(revision 3 §5.3) or the speed push of an imitation method (revision 3 §6.4). A child inherits its parent's setup,
which the setup mutation then changes (revision 3.1 §3)."""
import numpy as np

from racing import config, evolution, nets, paths, setup
```

2. Replace `ES.initialize` with:

```python
    def initialize(self, rng):
        if self.initial is None:
            weights = nets.init_population(rng, self.pop_size, self.hidden)
            setups = setup.random(rng, self.pop_size)           # random networks: a wide spread (revision 3.1 §3.2)
        else:
            weights = seeded_population(rng, self.initial, self.pop_size, self.sigma0)
            setups = setup.default(self.pop_size)               # the setup the network was trained with
        self.population = DensePopulation(weights, self.hidden, self.activation, setup=setups)
        self.is_kept = np.zeros(self.pop_size, dtype=bool)
        self.ids = np.arange(self.pop_size)
        self.parent_ids = np.full(self.pop_size, -1)
        self.next_id = self.pop_size
        self.step = StepSize(self.sigma0)
```

3. Replace `ES.advance`, `ES.state` and `ES.load_state` with:

```python
    def advance(self, rng, result, kept, event):
        """Adapt σ, then keep the survivors first and unchanged and add children of the ES_MU fittest cars; a child's
        setup is its parent's, mutated after the network's draws (revision 3.1 §3.3)."""
        weights, setups = self.population.weights, self.population.setup
        n, p = weights.shape
        top = np.argsort(-result.fitness, kind="stable")[:min(self.parents, n)]
        child_success = evolution.children_success(result, self.is_kept)
        self.step.update(float(result.all_success[top].mean()), child_success, self.success_rate(result.fitness))
        if event == "level_up":
            self.step.reset()                           # restart the step size on the harder maps
        n_new = n - len(kept)
        chosen = top[rng.integers(0, len(top), n_new)]
        children = weights[chosen] + rng.normal(0.0, self.step.sigma, (n_new, p))
        self.assemble(weights[kept], children, setups[kept], setup.mutate(rng, setups[chosen]))
        self.parent_ids = np.concatenate([self.parent_ids[kept], self.ids[chosen]])
        self.ids = np.concatenate([self.ids[kept], np.arange(self.next_id, self.next_id + n_new)])
        self.next_id += n_new

    def state(self):
        return {"weights": self.population.weights, "setup": self.population.setup, "is_kept": self.is_kept,
                "ids": self.ids, "parent_ids": self.parent_ids, "next_id": self.next_id, "step": self.step.state(),
                "hidden": self.hidden, "activation": self.activation}

    def load_state(self, state):
        self.hidden, self.activation = tuple(state["hidden"]), state["activation"]
        self.population = DensePopulation(state["weights"], self.hidden, self.activation, setup=state["setup"])
        self.is_kept, self.ids, self.parent_ids = state["is_kept"], state["ids"], state["parent_ids"]
        self.next_id = state["next_id"]
        self.step = StepSize.from_state(state["step"])
```

- [ ] **Step 7: NEAT setups**

In `racing/neat.py`:

1. Leave the imports unchanged: Task 2 already added `from racing import setup as car_setup`, a name that never
   shadows the `NEAT.setup` property below or the `setup` parameters of `NeatPopulation` and `save_neat_champion`.
   The code of this step uses `car_setup`.
2. Replace the `NEAT` class docstring with:

```python
    """NEAT as a method of the evolution engine (revision 3 §5.4): kept cars first, then children produced within
    species, with the two-phase step size of the ES. A child inherits the setup of its fitter (or only) parent, which
    the setup mutation then changes (revision 3.1 §3)."""
```

3. Directly after the `step_size` property of `NEAT`, add:

```python
    @property
    def setup(self):
        """Every car's setup, aligned with the population (revision 3.1 §2)."""
        return self.population.setup
```

4. In `NEAT.initialize`, replace
   `self.population = NeatPopulation([initial_genome(rng, self.innovations) for _ in range(self.pop_size)])` with:

```python
        genomes = [initial_genome(rng, self.innovations) for _ in range(self.pop_size)]
        self.population = NeatPopulation(genomes, setup=car_setup.random(rng, self.pop_size))   # revision 3.1 §3.2
```

5. In `NEAT.advance`:
   - replace the two lines

```python
        fitness, success = result.fitness, result.all_success
        genomes = self.population.genomes
```

     with

```python
        fitness, success = result.fitness, result.all_success
        genomes, setups = self.population.genomes, self.population.setup
```

     (`log_fields` has another `genomes = self.population.genomes` line; leave that one unchanged);
   - replace `new_genomes, new_parents = [], []` with `new_genomes, new_parents, fitter_rows = [], [], []`;
   - after `new_parents.append(int(self.ids[fitter]))` (inside the inner loop) add `fitter_rows.append(fitter)`;
   - replace

```python
        kept = np.asarray(kept, dtype=np.int64)
        self.population = NeatPopulation([genomes[i] for i in kept] + new_genomes)
```

     with

```python
        kept = np.asarray(kept, dtype=np.int64)
        children_setups = car_setup.mutate(rng, setups[np.asarray(fitter_rows, dtype=np.int64)])   # after the genomes
        self.population = NeatPopulation([genomes[i] for i in kept] + new_genomes,
                                         setup=np.concatenate([setups[kept], children_setups]))
```

   (`setups[np.asarray([], dtype=np.int64)]` is a `(0, 2)` array, so a generation without children works.)
6. Replace `NEAT.state`, the first two lines of `NEAT.load_state`, and `NEAT.save_champion`:

```python
    def state(self):
        return {"pop_size": self.pop_size, "genomes": [g.to_dict() for g in self.population.genomes],
                "setup": self.population.setup.copy(),
                "is_kept": self.is_kept.copy(), "ids": self.ids.copy(), "parent_ids": self.parent_ids.copy(),
                "next_id": self.next_id, "innovations": self.innovations.state(), "species": self.species.state(),
                "steps": self.steps.state()}
```

```python
    def load_state(self, state):
        self.pop_size = int(state["pop_size"])
        self.population = NeatPopulation([Genome.from_dict(g) for g in state["genomes"]],
                                         setup=np.asarray(state["setup"], dtype=float))
```

   (the remaining lines of `load_state` — `is_kept`, `ids`, `parent_ids`, `next_id`, `innovations`, `species`, `steps`
   — are unchanged), and

```python
    def save_champion(self, index, seed, champion_dir):
        return save_neat_champion(self.approach, seed, self.population.genomes[int(index)], champion_dir,
                                  setup=self.population.setup[int(index)])
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_ga.py tests/test_es.py tests/test_neat.py tests/test_evolution.py tests/test_checkpoint.py`
Expected: all passed.

- [ ] **Step 9: Run the fast suite and the slow method tests**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all passed (502 after Task 2 + 18 = 520).
Run: `~/miniconda3/bin/python -m pytest -q -m slow tests/test_evolution.py tests/test_neat.py tests/test_end_to_end.py tests/test_dagger.py`
Expected: all passed (the end-to-end test runs the speed push and the seeded ES, which now start at the default setup;
the slow evolution test compares logs, including the setup columns, between worker counts).

- [ ] **Step 10: Commit**

```bash
git add racing/checkpoint.py racing/evolution.py racing/ga.py racing/es.py racing/neat.py \
        tests/test_checkpoint.py tests/test_ga.py tests/test_es.py tests/test_neat.py tests/test_evolution.py
git commit -m "Evolved setups: GA, ES and NEAT create, inherit and mutate each car's sensor length and steering sensitivity; logged, checkpointed and saved with the final controller (revision 3.1 §3)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/checkpoint.py racing/evolution.py racing/ga.py \
        racing/es.py racing/neat.py tests/test_checkpoint.py tests/test_ga.py tests/test_es.py tests/test_neat.py \
        tests/test_evolution.py
```

---

### Task 4: Evaluation: setup columns and figure; network figure and window titles

Revision 3.1 §5. The results table reports each method's final setups, a new figure shows how the population's mean
setup moves over training, and every network title (evaluation figure and network window) names the setup. Results
stay framed by finishes and lap times; nothing is compared with the expert.

**Files:**
- Modify: `racing/evaluate.py` (`setup_cells` new, `results_table`, `_plot_runs` new and used by `_panels`,
  `SETUP_PANELS` and `_setup_figure` new, `figures`, `main`)
- Modify: `racing/netfig.py` (`setup_text` and `panel_title` new, `draw_networks`)
- Modify: `racing/app.py` (`App.shown_network`: titles only)
- Test: `tests/test_evaluate.py`, `tests/test_netfig.py`, `tests/test_app.py`

**Interfaces:**
- Consumes (Tasks 1–3): `config.DEFAULT_SETUP`, `config.SENSOR_LENGTH_LIMITS`, `config.STEER_GAIN_LIMITS`; every
  driver (`nets.DenseNet`, `neat.NeatNet`) has `.setup`, an array `(sensor length m, steering sensitivity)`, default
  `config.DEFAULT_SETUP`; `nets.load_champion` reads `<seed>.setup.json` into `driver.setup`;
  `nets.DensePopulation(weights, setup=...)` whose `driver(i).setup` is row i; the engine's `log.csv` has the columns
  `sensor_mean`, `sensor_std`, `steer_mean`, `steer_std`.
- Produces: `evaluate.results_table(per_track, runs_dir, drivers=None)` with the columns `"Sensor length (m)"`
  (mean ± std across seeds, 0 decimals) and `"Steering sensitivity"` (2 decimals), placed before
  `"Target reduction (per seed)"`, `"–"` when `drivers` is None or holds none of the method's seeds;
  `evaluate.figures` also writes `setup_over_training.png` (after `lap_time_over_training.png`, before
  `step_sizes.png`) when any log records setups; `netfig.setup_text(driver) -> str` (`" · <L> m · ×<k>"`, L with 0
  and k with 2 decimals); `netfig.panel_title(approach, driver) -> str`; network window titles end with
  `setup_text(driver)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_evaluate.py`, add after `test_results_table_counts_finishes_successes_and_target_reduction`:

```python
def test_results_table_reports_the_final_setups(tmp_path):
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0), ("ga", 1, 1000, True, True, False, 54.0),
                            ("es", 0, 1000, True, True, False, 52.0)])
    finals = drivers(n=2)
    finals[("ga", 0)].setup = np.array([40.0, 2.0])
    finals[("ga", 1)].setup = np.array([60.0, 1.0])
    finals[("es", 0)] = nets.from_flat(nets.init_population(np.random.default_rng(9), 1)[0])   # revision 3 setup
    table = evaluate.results_table(per_track, tmp_path, finals).set_index("Method")
    assert table.loc["GA", "Sensor length (m)"] == "50 ± 14" and table.loc["GA", "Steering sensitivity"] == "1.50 ± 0.71"
    assert table.loc["ES", "Sensor length (m)"] == "50" and table.loc["ES", "Steering sensitivity"] == "1.00"
    columns = list(table.columns)
    assert columns.index("Steering sensitivity") == columns.index("Sensor length (m)") + 1
    assert columns.index("Steering sensitivity") + 1 == columns.index("Target reduction (per seed)")


def test_results_table_without_drivers_shows_no_setup(tmp_path):
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0)])
    table = evaluate.results_table(per_track, tmp_path).set_index("Method")
    assert table.loc["GA", "Sensor length (m)"] == "–" and table.loc["GA", "Steering sensitivity"] == "–"
```

In `tests/test_evaluate.py`, add after `test_figures_per_method`:

```python
def test_setup_figure_follows_the_population_setups(tmp_path):
    runs = tmp_path / "runs"
    folder = paths.run_dir("neat", 0, runs)
    write_log(folder, [None, "level_up", None], [0.2, 0.95, 0.5], [np.nan, 40.0, 39.0])
    log = pd.read_csv(folder / "log.csv")
    log["sensor_mean"], log["sensor_std"] = [60.0, 70.0, 85.0], [20.0, 15.0, 10.0]
    log["steer_mean"], log["steer_std"] = [1.0, 1.3, 1.6], [0.5, 0.4, 0.3]
    log.to_csv(folder / "log.csv", index=False)
    written = evaluate.figures(runs, tmp_path / "figures")
    assert [p.name for p in written] == ["success_over_training.png", "lap_time_over_training.png",
                                         "setup_over_training.png", "step_sizes.png"]
    assert (tmp_path / "figures" / "setup_over_training.png").read_bytes()[:4] == b"\x89PNG"
```

(The existing `test_figures_per_method` stays unchanged: its logs record no setups, so no setup figure is written.)

In `tests/test_evaluate.py`, in `test_main_with_partial_results`, add at the end:

```python
    assert results["results"].set_index("Method").loc["GA", "Sensor length (m)"] == "50"
```

In `tests/test_netfig.py`, add at the end:

```python
def test_titles_name_the_method_parameters_and_setup():
    driver = dense_driver()
    assert netfig.setup_text(driver) == " · 50 m · ×1.00"
    driver.setup = np.array([123.4, 0.456])
    assert netfig.setup_text(driver) == " · 123 m · ×0.46"
    assert netfig.panel_title("ga", driver) == f"GA · {driver.n_params:,} parameters · 123 m · ×0.46"
    assert netfig.panel_title("neat", neat_driver()).endswith(" parameters · 50 m · ×1.00")
```

In `tests/test_app.py`, in `test_a_new_report_sends_exactly_one_message` (the test that asserts
`sent(net)[0]["title"] == f"GA · seed {net.run_seed} · generation 0 · ..."`), replace

```python
    assert sent(net)[0]["title"] == f"GA · seed {net.run_seed} · generation 0 · {report['population'].driver(best).n_params:,} parameters"
```

with

```python
    assert sent(net)[0]["title"] == (f"GA · seed {net.run_seed} · generation 0 · "
                                     f"{report['population'].driver(best).n_params:,} parameters · 50 m · ×1.00")
```

(If the test has another name, find it with `grep -n 'generation 0 · {report' tests/test_app.py`.) Then add after
`test_an_unchanged_report_builds_no_driver_and_the_title_uses_the_report_seed`:

```python
def test_network_window_titles_name_the_setup(net, tmp_path):
    rng = np.random.default_rng(3)
    cars = nets.DensePopulation(nets.init_population(rng, 4), setup=np.array([[120.0, 0.5]] * 4))
    net.toggle_network_window()
    net.receive(ga_report(0, n=4, population_=cars))
    net.update(1 / 60)
    assert sent(net)[-1]["title"].endswith(" parameters · 120 m · ×0.50")
    save_race_champions(tmp_path)
    net.champions_dir = tmp_path
    net.set_mode("race")
    net.update(1 / 60)
    assert sent(net)[-1]["title"].endswith(" parameters · 50 m · ×1.00")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_evaluate.py tests/test_netfig.py tests/test_app.py -k "setup or titles or exactly_one_message or main_with_partial"`
Expected: FAIL — `results_table() takes 2 positional arguments but 3 were given`, `KeyError: 'Sensor length (m)'`,
the figure list without `setup_over_training.png`, `AttributeError: module 'racing.netfig' has no attribute
'setup_text'`, and the network window titles without the setup suffix.

- [ ] **Step 3: Titles in `racing/netfig.py`**

Insert directly above `def draw_networks`:

```python
def setup_text(driver):
    """' · <sensor length> m · ×<steering sensitivity>' of a driver's setup (revision 3.1 §5); the revision 3 setup
    for a driver without one."""
    length, gain = np.asarray(getattr(driver, "setup", config.DEFAULT_SETUP), dtype=float)
    return f" · {length:.0f} m · ×{gain:.2f}"


def panel_title(approach, driver):
    """A network diagram's title: the method, its parameter count and its setup."""
    return f"{config.LABELS[approach]} · {driver.n_params:,} parameters{setup_text(driver)}"
```

In `draw_networks`, change the docstring's first sentence to `"""One panel per method in config.APPROACHES order,
titled with the method, its parameter count and its setup, written as a PNG at `path`. drivers: {approach: driver}."""`
and replace

```python
        ax.set_title(f"{config.LABELS[approach]} · {drivers[approach].n_params:,} parameters", fontsize=10)
```

with

```python
        ax.set_title(panel_title(approach, drivers[approach]), fontsize=9)
```

- [ ] **Step 4: Setup columns, setup figure and `main` in `racing/evaluate.py`**

Insert directly above `def results_table`:

```python
def setup_cells(drivers, approach, seeds):
    """The final controllers' sensor length (m, no decimals) and steering sensitivity (two decimals) as 'mean ± std'
    across seeds (revision 3.1 §5); '–' for both without drivers for the method."""
    rows = [np.asarray(drivers[(approach, int(s))].setup, dtype=float) for s in seeds
            if drivers and (approach, int(s)) in drivers]
    if not rows:
        return "–", "–"
    values = np.array(rows)
    return mean_std(values[:, 0], 0), mean_std(values[:, 1], 2)
```

Replace the whole `results_table` with:

```python
def results_table(per_track, runs_dir, drivers=None):
    """Revision 3 §7.2 table 1: test maps finished and succeeded (in total and per seed), lap and race times,
    crashes, the final controllers' setups (revision 3.1 §5; drivers: {(approach, seed): driver}) and the target
    reduction of each seed's speed push."""
    seeds = per_seed(per_track)
    rows = []
    for approach in config.APPROACHES:
        sub = seeds[seeds["approach"] == approach].sort_values("run_seed")
        if sub.empty:
            continue
        total = int(sub["maps"].sum())
        best_lap = "–" if sub["best_lap"].isna().all() else f"{sub['best_lap'].min():.2f}"
        sensor, steer = setup_cells(drivers, approach, sub["run_seed"])
        rows.append({"Method": config.LABELS[approach], "Seeds": len(sub),
                     "Finished": f"{int(sub['finished'].sum())} / {total}",
                     "Finished per seed": ", ".join(str(int(v)) for v in sub["finished"]),
                     "Succeeded": f"{int(sub['success'].sum())} / {total}",
                     "Succeeded per seed": ", ".join(str(int(v)) for v in sub["success"]),
                     "Mean lap time (s)": seeds_note(sub["lap_time"], 2), "Best lap time (s)": best_lap,
                     "Mean race time (s)": seeds_note(sub["race_time"], 1),
                     "Crashes": mean_std(sub["crashes"], 1),
                     "Sensor length (m)": sensor, "Steering sensitivity": steer,
                     "Target reduction (per seed)": ", ".join(
                         reduction_cell(run_summary(approach, int(s), runs_dir)) for s in sub["run_seed"])})
    return pd.DataFrame(rows)
```

Insert directly above `def _panels`:

```python
def _plot_runs(ax, runs, column, approach):
    """One line per seed of `column` (seeds told apart by line style), with level advancements and the mastery goal
    as solid and target reductions as dotted grey vertical lines."""
    for k, (seed, log) in enumerate(sorted(runs.items())):
        if column not in log:
            continue
        ax.plot(log["generation"], log[column], linewidth=1.2, linestyle=SEED_STYLES[k % len(SEED_STYLES)],
                label=f"seed {seed}", **_line_kwargs(config.COLORS[approach]))
        events = log["event"].fillna("").astype(str) if "event" in log else []
        for generation, event in zip(log["generation"], events):
            if event in EVENT_STYLES:
                ax.axvline(generation, color="0.6", linewidth=0.6, linestyle=EVENT_STYLES[event], zorder=0)
```

In `_panels`, replace the inner loop

```python
        for k, (seed, log) in enumerate(sorted(runs.items())):
            if column not in log:
                continue
            ax.plot(log["generation"], log[column], linewidth=1.2, linestyle=SEED_STYLES[k % len(SEED_STYLES)],
                    label=f"seed {seed}", **_line_kwargs(config.COLORS[approach]))
            events = log["event"].fillna("").astype(str) if "event" in log else []
            for generation, event in zip(log["generation"], events):
                if event in EVENT_STYLES:
                    ax.axvline(generation, color="0.6", linewidth=0.6, linestyle=EVENT_STYLES[event], zorder=0)
```

with

```python
        _plot_runs(ax, runs, column, approach)
```

Insert directly above `def figures`:

```python
SETUP_PANELS = (("sensor_mean", "Sensor length (m)", config.SENSOR_LENGTH_LIMITS, 0),
                ("steer_mean", "Steering sensitivity", config.STEER_GAIN_LIMITS, 1))


def _setup_figure(plt, logs):
    """Revision 3.1 §5: for each method, the population's mean sensor length (upper panel) and mean steering
    sensitivity (lower panel) per generation, one line per seed, with level advancements and target reductions
    marked and the revision 3 setup as a dashed line. None if no log records setups."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter
    with_setup = {approach: runs for approach, runs in logs.items()
                  if any("sensor_mean" in log and log["sensor_mean"].notna().any() for log in runs.values())}
    if not with_setup:
        return None
    ncols = min(4, len(with_setup))
    groups = -(-len(with_setup) // ncols)
    fig, axes = plt.subplots(2 * groups, ncols, figsize=(3.4 * ncols, 4.6 * groups), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, (approach, runs) in enumerate(with_setup.items()):
        group, column = divmod(k, ncols)
        for row, (name, ylabel, limits, index) in enumerate(SETUP_PANELS):
            ax = axes[2 * group + row, column]
            ax.set_visible(True)
            _plot_runs(ax, runs, name, approach)
            ax.axhline(config.DEFAULT_SETUP[index], color="0.3", linewidth=0.8, linestyle="--", zorder=0)
            ax.set_yscale("log")
            ax.set_ylim(*limits)
            ax.set_yticks([limits[0], config.DEFAULT_SETUP[index], limits[1]])   # plain labels on the log axis
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f"{config.LABELS[approach]}: {ylabel[0].lower()}{ylabel[1:]}", fontsize=9)
            ax.set_xlabel("Generation", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, linewidth=0.5)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], color="0.3", linewidth=0.8, linestyle="--", label="revision 3 setup"),
                Line2D([], [], color="0.6", linewidth=0.6, label="level cleared / goal"),
                Line2D([], [], color="0.6", linewidth=0.6, linestyle=":", label="target cut")]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=7)
    fig.tight_layout()
    return fig
```

In `figures`, change the docstring to `"""Revision 3 §7.3: the lowest map success share and the median lap time over
training (one panel per method, one line per seed), the population's mean setup over training (revision 3.1 §5), the
step sizes, and the Keras Tuner trials. Only figures with data are written."""` and insert directly after the
`for name, column, ylabel, threshold in (...)` loop (before `with_steps = ...`):

```python
    fig = _setup_figure(plt, logs)
    if fig is not None:
        written.append(out_dir / "setup_over_training.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)
```

In `main`, replace `"results": results_table(per_track, runs_dir)` with `"results": results_table(per_track, runs_dir,
drivers)`.

- [ ] **Step 5: Network window titles in `racing/app.py`**

In `App.shown_network`, replace

```python
            title = (f"{config.LABELS[approach]} · seed {seed} · {unit} {generation} · "
                     f"{driver.n_params:,} parameters")
```

with

```python
            title = (f"{config.LABELS[approach]} · seed {seed} · {unit} {generation} · "
                     f"{driver.n_params:,} parameters{netwindow.netfig.setup_text(driver)}")
```

and replace

```python
        return identity, driver, f"{config.LABELS[car[0]]} · seed {car[1]} · {driver.n_params:,} parameters"
```

with

```python
        return identity, driver, (f"{config.LABELS[car[0]]} · seed {car[1]} · {driver.n_params:,} parameters"
                                  f"{netwindow.netfig.setup_text(driver)}")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_evaluate.py tests/test_netfig.py tests/test_app.py`
Expected: all passed.
Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all passed (520 after Task 3 + 5 = 525).

- [ ] **Step 7: Look at the figures**

Run:

```bash
~/miniconda3/bin/python -c "
import numpy as np, pandas as pd, tempfile, pathlib
from racing import evaluate, paths
runs = pathlib.Path(tempfile.mkdtemp()) / 'runs'
for approach in ('ga', 'neat'):
    folder = paths.run_dir(approach, 0, runs); folder.mkdir(parents=True)
    n = 40
    pd.DataFrame({'generation': range(n), 'min_share': np.linspace(0.1, 0.95, n), 'median_lap': np.nan,
                  'sensor_mean': np.geomspace(50, 120, n), 'sensor_std': 10.0, 'steer_mean': np.geomspace(1, 1.8, n),
                  'steer_std': 0.2, 'event': [None] * 20 + ['level_up'] + [None] * 19}).to_csv(folder / 'log.csv', index=False)
print(evaluate.figures(runs, '/tmp/setup-figure-check'))
"
```

Open `/tmp/setup-figure-check/setup_over_training.png` (Read tool) and check: two rows (sensor length above, steering
sensitivity below) per method, log y-axes spanning 10–150 m and ⅓–3, a dashed line at 50 m and ×1, the level-up line,
legible titles. Adjust `figsize` in `_setup_figure` if labels overlap, then rerun Step 6.

- [ ] **Step 8: Commit**

```bash
git add racing/evaluate.py racing/netfig.py racing/app.py tests/test_evaluate.py tests/test_netfig.py tests/test_app.py
git commit -m "Evaluation: final setups in the results table, setup-over-training figure; setups in network titles

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/evaluate.py racing/netfig.py racing/app.py tests/test_evaluate.py tests/test_netfig.py tests/test_app.py
```

---

### Task 5: Viewer: grip selector, per-car rays, setup stats

Revision 3.1 §6. The viewer's own simulations (training replays and races) drive every car with its own setup and the
selected display grip (Normal 12 m/s², Unlimited 120 m/s²); training, evaluation and the expert never see the display
grip (Review Focus 3). Sensor rays are drawn at each car's own length, in the training view for the highlighted car and
in the race view for every racing car, and the training panel shows the population's mean setup.

**Files:**
- Modify: `racing/config.py` (`APP_GRIPS` in the application section)
- Modify: `racing/app.py` (`Playback`, `report_setup` new, `setup_line` new, `stats_lines`, `App.__init__`,
  `App.grip` new, `App.set_grip` new, `App.toggle_grip` new, `App.start_playback`, `App.prepare_race`,
  `App.draw_population`, `App.draw_racers`, `App.draw_view_toggles` new, `App.draw_train_panel`,
  `App.draw_race_panel`)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes (Tasks 1–3): `sim.start_cars(track, n, laps=1, setup=None)`, `sim.step(..., grip=config.GRIP)`,
  `sim.cast_rays(track, x, y, heading, lengths=None)`, `Cars.ray_length`, `Cars.steer_gain`;
  `config.GRIP`; `population.subset(indices).setup`; `nets.drivers_setup(drivers)`; engine reports carry
  `sensor_mean`, `sensor_std`, `steer_mean`, `steer_std`.
- Produces: `config.APP_GRIPS = {"Normal": GRIP, "Unlimited": 10.0 * GRIP}` (not a recorded setting);
  `app.Playback(track, policy, n, laps=1, time_limit=None, setup=None, grip=config.GRIP)` with `.grip`;
  `app.report_setup(report, indices) -> np.ndarray`; `app.setup_line(report) -> str`
  (`"Sensors <mean> m ± <std>  Steering ×<mean> ± <std>"`, the sixth line of `stats_lines` for evolutionary reports
  that carry setups); `App.grip_name` (default `"Normal"`), `App.grip` (property), `App.set_grip(name)`,
  `App.toggle_grip()`, `App.draw_view_toggles(x, y, w) -> int`.
- Note for Task 6: after this task the race panel with all eight approaches has about 12 px to spare above
  `panel.bottom - 30`; the application group of Task 6 needs a compact layout.

- [ ] **Step 1: Write the failing tests**

In `tests/test_app.py`, change the import line `from racing import config, neat, nets, paths, population` to
`from racing import config, neat, nets, paths, population, setup, sim`, and in `ga_report` replace

```python
              "evals": n, "elapsed": 1.0, "event": None}
```

with

```python
              "evals": n, "elapsed": 1.0, "event": None, "sensor_mean": 50.0, "sensor_std": 0.0, "steer_mean": 1.0,
              "steer_std": 0.0}
```

In `test_training_stats_fit_the_panel`, replace the `widest = dict(...)` statement with

```python
    widest = dict(level=4, generation=123456, map_shares=np.ones(3), min_share=1.0, streak=3, target_reduction=0.185,
                  step_size=0.0505, best=1.5843, mean=1.2329, species=20, is_kept=np.ones(1000, dtype=bool),
                  sensor_mean=150.0, sensor_std=70.0, steer_mean=3.0, steer_std=1.33)
```

Add after `test_stats_lines_show_maps_streak_and_target`:

```python
def test_stats_lines_show_the_population_setup():
    lines = viewer.stats_lines(ga_report(sensor_mean=62.4, sensor_std=18.2, steer_mean=1.4, steer_std=0.3), "ga")
    assert lines[5] == "Sensors 62 m ± 18  Steering ×1.40 ± 0.30"
    report = ga_report()
    for key in ("sensor_mean", "sensor_std", "steer_mean", "steer_std"):
        del report[key]
    assert len(viewer.stats_lines(report, "ga")) == 5        # a report without setups shows no setup line


def test_playback_drives_each_car_with_its_setup_and_the_display_grip(circle_track):
    full_left_gas = lambda obs, cars: np.full(len(cars.x), config.N_ACTIONS - 1)
    cars_setup = np.array([[30.0, 1.0], [150.0, 0.5]])
    normal = viewer.Playback(circle_track, full_left_gas, 2, setup=cars_setup)
    unlimited = viewer.Playback(circle_track, full_left_gas, 2, setup=cars_setup, grip=config.APP_GRIPS["Unlimited"])
    assert normal.grip == config.GRIP and unlimited.grip == config.APP_GRIPS["Unlimited"]
    assert np.array_equal(normal.cars.ray_length, [30.0, 150.0]) and np.array_equal(normal.cars.steer_gain, [1.0, 0.5])
    normal.advance(5)
    unlimited.advance(5)
    assert np.all(unlimited.cars.speed > 5 * normal.cars.speed)    # ten times the grip accelerates far harder


def test_the_training_replay_uses_the_report_setups_and_the_display_grip(window):
    rng = np.random.default_rng(4)
    cars = nets.DensePopulation(nets.init_population(rng, 30), setup=setup.random(rng, 30))
    window.set_grip("Unlimited")
    window.receive(ga_report(population_=cars))
    window.update(1 / 60)
    assert window.playback.grip == config.APP_GRIPS["Unlimited"]
    assert np.array_equal(window.playback.cars.ray_length, cars.subset(window.indices).setup[:, 0])
    assert np.array_equal(window.playback.cars.steer_gain, cars.subset(window.indices).setup[:, 1])
    window.set_grip("Normal")                                    # the replay restarts with the new grip
    window.update(1 / 60)
    assert window.playback.grip == config.GRIP


def test_a_stopped_replay_keeps_its_grip(window):
    window.receive(ga_report())
    window.update(1 / 60)
    window.stop()
    frozen = window.playback
    window.set_grip("Unlimited")
    window.update(1 / 60)
    assert window.playback is frozen and frozen.grip == config.GRIP and window.grip_name == "Unlimited"


def test_the_race_uses_driver_setups_and_the_display_grip(window, tmp_path):
    race_window(window, tmp_path)
    window.set_grip("Unlimited")
    window.update(1 / 60)
    drivers = [window.driver(approach, seed) for approach, seed in window.lineup]
    assert window.race.grip == config.APP_GRIPS["Unlimited"]
    assert np.array_equal(window.race.cars.ray_length, nets.drivers_setup(drivers)[:, 0])
    window.start_race()
    window.set_grip("Normal")                                    # locked while racing
    assert window.grip_name == "Unlimited" and window.race.grip == config.APP_GRIPS["Unlimited"]


def test_toggle_grip_cycles_the_display_grips(window):
    names = list(config.APP_GRIPS)
    assert window.grip_name == names[0] == "Normal" and window.grip == config.GRIP
    window.toggle_grip()
    assert window.grip_name == names[1] and window.grip == config.APP_GRIPS[names[1]]
    window.toggle_grip()
    assert window.grip_name == names[0]


def test_the_display_grip_never_reaches_training(window, monkeypatch, tmp_path):
    """Review Focus 3: the trainer process gets the same arguments whatever the display grip, the training functions
    get no grip, and the training grip itself never changes."""
    from racing import ga
    created = []

    class FakeProcess:
        def __init__(self, target, args, daemon):
            created.append(args)
            self.pid = 0

        def start(self):
            pass

        def is_alive(self):
            return False

    class FakeContext:
        Process = FakeProcess

        def Queue(self):
            return FakeQueue()

        def Event(self):
            return threading.Event()

    monkeypatch.setattr(viewer.mp, "get_context", lambda method: FakeContext())
    window.start_training()
    window.process = None
    window.set_grip("Unlimited")
    window.start_training()
    assert len(created) == 2 and created[0][:4] == created[1][:4] and len(created[1]) == len(created[0])
    assert not any(isinstance(a, float) and a == config.APP_GRIPS["Unlimited"] for a in created[1])
    calls = {}
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)       # never the real results/

    def fake_run_ga(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(ga, "run_ga", fake_run_ga)
    reports, running = queue.Queue(), threading.Event()
    running.set()
    viewer.trainer_main("ga", 30, 0, 1, reports, running)
    assert "grip" not in calls and calls["pop_size"] == 30
    assert config.GRIP == 12.0 and config.APP_GRIPS["Normal"] == config.GRIP


def test_rays_follow_each_cars_sensor_length(window, monkeypatch, tmp_path):
    seen = []
    cast = sim.cast_rays

    def recording(track, x, y, heading, lengths=None):
        seen.append(None if lengths is None else np.array(lengths))
        return cast(track, x, y, heading, lengths)

    monkeypatch.setattr(viewer.sim, "cast_rays", recording)
    rng = np.random.default_rng(6)
    cars = nets.DensePopulation(nets.init_population(rng, 30), setup=setup.random(rng, 30))
    window.rays = True
    window.receive(ga_report(population_=cars))
    window.update(1 / 60)
    seen.clear()
    window.draw()
    best = window.best_slot
    assert len(seen) == 1 and np.array_equal(seen[0], window.playback.cars.ray_length[best:best + 1])
    race_window(window, tmp_path)
    window.update(1 / 60)
    window.start_race()
    window.update(1 / 60)
    seen.clear()
    window.draw()
    live = np.flatnonzero(~window.race.cars.done)
    assert len(seen) == 1 and np.array_equal(seen[0], window.race.cars.ray_length[live])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_app.py -k "setup or grip or rays or stats"`
Expected: 8 FAIL — `TypeError: Playback.__init__() got an unexpected keyword argument 'setup'`,
`AttributeError: 'App' object has no attribute 'set_grip'` / `'toggle_grip'`, `AttributeError: module
'racing.config' has no attribute 'APP_GRIPS'`, `IndexError` for `lines[5]`, and the ray lengths recorded as `None`.

- [ ] **Step 3: `APP_GRIPS`; `Playback`, `report_setup` and `setup_line` in `racing/app.py`**

In `racing/config.py`, in the application section, directly after the `DIFFICULTIES = ...` line, add:

```python
APP_GRIPS = {"Normal": GRIP, "Unlimited": 10.0 * GRIP}   # m/s², display grip of the viewer's own simulations only
                                                         # (revision 3.1 §6); never a recorded setting
```

Do not add it to `checkpoint.SETTINGS_KEYS`.

In `racing/app.py`, replace the whole class `Playback` with:

```python
class Playback:
    """Re-simulates a set of controllers on one track, a few steps at a time, for display. Each car drives with its
    own setup and every car with the viewer's display grip (revision 3.1 §6); training never uses this class."""

    def __init__(self, track, policy, n, laps=1, time_limit=None, setup=None, grip=config.GRIP):
        self.track = track
        self.policy = policy
        self.grip = float(grip)
        self.cars = sim.start_cars(track, n, laps, setup)
        self.time_limit = sim.time_limit_for(track, laps) if time_limit is None else time_limit

    @property
    def over(self):
        return bool(self.cars.done.all())

    def advance(self, steps):
        for _ in range(steps):
            if self.over:
                return
            steer, throttle = class_to_controls(self.policy(sim.sense(self.track, self.cars), self.cars))
            sim.step(self.track, self.cars, steer, throttle, self.time_limit, grip=self.grip)
```

Insert directly after `def report_policy(...)` (after its last line):

```python
def report_setup(report, indices):
    """Setups of the chosen members of a training report: the population's rows, or each imitation network's own
    (revision 3.1 §2)."""
    if "population" in report:
        return report["population"].subset(indices).setup
    return nets.drivers_setup([report["nets"][i] for i in indices])
```

Insert directly above `def stats_lines`:

```python
def setup_line(report):
    """The population's mean and spread of sensor length and steering sensitivity (revision 3.1 §6)."""
    return (f"Sensors {report['sensor_mean']:.0f} m ± {report['sensor_std']:.0f}  "
            f"Steering ×{report['steer_mean']:.2f} ± {report['steer_std']:.2f}")
```

In `stats_lines`, replace the final

```python
    return (f"{level}    Generation {report['generation']}",
            f"Maps succeeded: {share_text(report['map_shares'])}  (need {percent(config.MASTERY_FRACTION)})",
            f"Streak {report['streak']}/{config.MASTERY_STREAK}    Target cut mastered: {mastered}",
            f"{size_name}: {number(report['step_size'])}    Kept {int(np.sum(report['is_kept']))}",
            last)
```

with

```python
    lines = (f"{level}    Generation {report['generation']}",
             f"Maps succeeded: {share_text(report['map_shares'])}  (need {percent(config.MASTERY_FRACTION)})",
             f"Streak {report['streak']}/{config.MASTERY_STREAK}    Target cut mastered: {mastered}",
             f"{size_name}: {number(report['step_size'])}    Kept {int(np.sum(report['is_kept']))}",
             last)
    return lines + (setup_line(report),) if "sensor_mean" in report else lines
```

and add to its docstring, after "the fitness (and NEAT's species).", the sentence "With setups, the population's mean
setup (revision 3.1 §6)."

- [ ] **Step 4: Display grip, setups and rays in `App`**

In `App.__init__`, directly after `self.rays = False`, add:

```python
        self.grip_name = "Normal"           # display grip of the viewer's own simulations (revision 3.1 §6)
```

Directly after `def toggle_rays(self):` and its body, add:

```python
    @property
    def grip(self):
        return config.APP_GRIPS[self.grip_name]

    def set_grip(self, name):
        """Display grip for the viewer's races and training replays (revision 3.1 §6). Training, evaluation and the
        expert always use config.GRIP. Locked while a race runs; a running replay restarts with the new grip, a
        stopped one stays frozen and the next replay uses it."""
        if name not in config.APP_GRIPS or name == self.grip_name or self.racing:
            return
        self.grip_name = name
        if self.mode == "race":
            self.reset_race()
        elif not self.halted:
            self.playback = None

    def toggle_grip(self):
        names = list(config.APP_GRIPS)
        self.set_grip(names[(names.index(self.grip_name) + 1) % len(names)])
```

In `App.start_playback`, replace

```python
        self.playback = Playback(track, report_policy(report, self.indices), len(self.indices),
                                 laps=config.LAPS)
```

with

```python
        self.playback = Playback(track, report_policy(report, self.indices), len(self.indices), laps=config.LAPS,
                                 setup=report_setup(report, self.indices), grip=self.grip)
```

In `App.prepare_race`, replace

```python
                    self.race = Playback(track, nets.mixed_policy(drivers), len(drivers), laps=config.RACE_LAPS)
```

with

```python
                    self.race = Playback(track, nets.mixed_policy(drivers), len(drivers), laps=config.RACE_LAPS,
                                         setup=nets.drivers_setup(drivers), grip=self.grip)
```

In `App.draw_population`, replace

```python
                lengths = sim.cast_rays(playback.track, cars.x[b:b + 1], cars.y[b:b + 1], cars.heading[b:b + 1])[0]
```

with

```python
                lengths = sim.cast_rays(playback.track, cars.x[b:b + 1], cars.y[b:b + 1], cars.heading[b:b + 1],
                                        lengths=cars.ray_length[b:b + 1])[0]   # this car's own sensor length
```

Replace the whole `App.draw_racers` with:

```python
    def draw_racers(self, playback, camera):
        cars = playback.cars
        sx, sy = camera.to_screen(cars.x, cars.y)
        size = max(8.0, 6.0 * camera.scale)
        if self.rays:                                   # every racing car's rays, each at its own sensor length
            live = np.flatnonzero(~cars.done)
            if live.size:
                lengths = sim.cast_rays(playback.track, cars.x[live], cars.y[live], cars.heading[live],
                                        lengths=cars.ray_length[live])
                for row, i in enumerate(live):
                    angles = cars.heading[i] + config.RAY_ANGLES
                    ex, ey = camera.to_screen(cars.x[i] + np.cos(angles) * lengths[row],
                                              cars.y[i] + np.sin(angles) * lengths[row])
                    for k in range(config.N_RAYS):
                        pygame.draw.line(self.screen, RAY, (sx[i], sy[i]), (ex[k], ey[k]), 1)
        for i, (approach, _) in enumerate(self.lineup):
            color = hex_to_rgb(config.COLORS[approach])
            if cars.crashed[i]:
                color = tuple(int(0.5 * c + 127) for c in color)          # faded
            shape = car_polygon(sx[i], sy[i], cars.heading[i], size)
            pygame.draw.polygon(self.screen, color, shape)
            pygame.draw.polygon(self.screen, INK, shape, 1)
```

Directly above `def draw_network_button`, add:

```python
    def draw_view_toggles(self, x, y, w):
        """Sensor rays and the display grip side by side (revision 3.1 §6). Returns the next free y."""
        half = (w - 8) // 2
        self.button(pygame.Rect(x, y, half, 26), f"Sensor rays: {'on' if self.rays else 'off'}", self.toggle_rays,
                    self.rays)
        self.button(pygame.Rect(x + half + 8, y, half, 26), f"Grip: {self.grip_name}", self.toggle_grip,
                    self.grip_name != "Normal", enabled=not self.racing)
        return y + 34
```

In `App.draw_train_panel`, replace

```python
        self.button(pygame.Rect(x, y, w, 26), f"Sensor rays: {'on' if self.rays else 'off'}", self.toggle_rays,
                    self.rays)
        y += 34
```

with

```python
        y = self.draw_view_toggles(x, y, w)
```

In `App.draw_race_panel`, replace

```python
        self.button(pygame.Rect(x + half + 8, y, half, 32), "Stop", self.stop_race, enabled=self.racing)
        y += 42
```

with

```python
        self.button(pygame.Rect(x + half + 8, y, half, 32), "Stop", self.stop_race, enabled=self.racing)
        y += 42
        y = self.draw_view_toggles(x, y, w)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_app.py`
Expected: all passed, including the existing panel-fit tests (`test_training_panel_fits_the_window`,
`test_a_race_of_every_method_fits_the_panel`, `test_race_draws_a_leaderboard_and_fits_the_window`,
`test_the_network_buttons_fit_both_panels`, `test_training_stats_fit_the_panel`). If
`test_training_stats_fit_the_panel` fails on the setup line, shorten `setup_line` to
`f"Sensors {mean:.0f}±{std:.0f} m  Steering ×{mean:.2f}±{std:.2f}"` and update
`test_stats_lines_show_the_population_setup` to `"Sensors 62±18 m  Steering ×1.40±0.30"`.
Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all passed (525 after Task 4 + 8 = 533).

- [ ] **Step 6: Look at the viewer**

Run (headless):

```bash
SDL_VIDEODRIVER=dummy ~/miniconda3/bin/python -c "
import sys; sys.path.insert(0, 'tests')
import numpy as np, pygame
from racing import app as viewer, config, nets, setup
from test_app import ga_report
pygame.init(); screen = pygame.display.set_mode(config.WINDOW_SIZE)
w = viewer.App(screen, background_tracks=False)
rng = np.random.default_rng(1)
cars = nets.DensePopulation(nets.init_population(rng, 40), setup=setup.random(rng, 40))
w.rays = True; w.set_grip('Unlimited')
w.receive(ga_report(population_=cars, sensor_mean=72.0, sensor_std=31.0, steer_mean=1.3, steer_std=0.6))
for _ in range(30): w.update(1 / 60)
w.draw(); pygame.image.save(screen, '/tmp/viewer-setup-check.png'); print('saved')
"
```

Open `/tmp/viewer-setup-check.png` (Read tool) and check: the "Sensor rays: on" and "Grip: Unlimited" buttons share a
row, the setup line is below the stats and inside the panel, the highlighted car's rays reach its own sensor length.

- [ ] **Step 7: Commit**

```bash
git add racing/config.py racing/app.py tests/test_app.py
git commit -m "Viewer: display grip selector (Normal/Unlimited), each car's setup in replays and races, rays at each car's sensor length, setup stats

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/config.py racing/app.py tests/test_app.py
```

---

### Task 6: Viewer: saved app runs (resume, New run, mismatch message, app champions, race group)

Revision 3.1 §6.1. Training started in the viewer is kept in `results/app_runs/<method>/seed<k>/`: for the
evolutionary methods Play continues the saved run of the selected method and seed, **New run** deletes it (and its app
car) and starts afresh, and a saved run that cannot be resumed (other settings after a code change, or a damaged file)
ends the trainer with exit code 3, after which the panel asks for New run. The trainer saves the run's best car so far to
`config.APP_CHAMPIONS_DIR` with its setup: after every report for the evolutionary methods, and whenever the validation
accuracy improves for the imitation methods (so a stopped imitation run still leaves its best network; for a run that
completes this is the network at the end of training). The race view switches its controller list between the reported
controllers and the app cars with one cycle button; app cars are labelled "<Method> (app) s<k>", start switched off,
race next to the reported controllers when switched on, and are never read by the evaluation.

**Files:**
- Modify: `racing/config.py` (`APP_CHAMPIONS_DIR`)
- Modify: `racing/app.py` (imports; `UNUSABLE_RUN`, `UNUSABLE_ERRORS`; `is_app`, `car_method`, `save_app_car`,
  `save_app_model`; `leaderboard_line`; `trainer_main`; `trainer_process`; `App.__init__`, `start_training`,
  `new_run`, `saved_run_dir`, `has_saved_run`, `saved_history`, `poll_reports`, `selected_car`, `cycle_network_car`,
  `network_car_label`, `shown_network` (race branch), `load_race`, `app_choice_label`, `cycle_app_choice`,
  `cycle_race_group`, `race_cars`, `drop_car`, `reset_race`, `driver`, `prepare_race`, `draw`, `draw_racers`,
  `draw_train_panel`, `draw_stats`, `draw_race_panel`)
- Modify: `tests/conftest.py` (autouse fixture `private_app_runs`)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes:
  - Task 2: `population.save_dense_champion(approach, seed, population, index, champion_dir)` (writes `.keras`,
    `.npy` and `<seed>.setup.json` from `population.setup[index]`); `neat.save_neat_champion(approach, seed, genome,
    champion_dir, setup=None)`; `population.clear_champion(approach, seed, champion_dir)` (also removes `.json` and
    `.setup.json`); `nets.load_champion(approach, seed, directory)` (driver with `.setup`); `nets.drivers_setup(drivers)`;
    `DensePopulation`/`NeatPopulation` with `.setup` (n, 2); `NeatPopulation.genomes`.
  - Task 4: titles built in `App.shown_network` with the setup suffix; keep that suffix unchanged.
  - Task 5: `Playback(track, policy, n, laps=1, time_limit=None, setup=None, grip=config.GRIP)`;
    `App.grip_name`; `config.APP_GRIPS`; the race panel's grip/rays row.
  - Existing: `checkpoint.SettingsMismatch`, `checkpoint.UnreadableRecord`, `checkpoint.CHECKPOINT_NAME`;
    `evolution.run` resumes from `out_dir/checkpoint.pkl` and returns a completed run's summary at once;
    `nets.champion_seeds(directory)`; `paths.champion_path`, `paths.remove`.
- Produces:
  - `config.APP_CHAMPIONS_DIR = APP_RUNS_DIR / "champions"` (not a recorded setting).
  - `app.UNUSABLE_RUN = 3`; `app.UNUSABLE_ERRORS = (checkpoint.SettingsMismatch, checkpoint.UnreadableRecord,
    EOFError, pickle.UnpicklingError)`.
  - `app.is_app(car) -> bool`; `app.car_method(car) -> str` ("GA" or "GA (app)").
  - `app.save_app_car(report, champion_dir)`; `app.save_app_model(approach, seed, model, champion_dir)`.
  - `app.leaderboard_line(position, approach, seed, laps, last, best, status, app=False)`.
  - `app.trainer_main(approach, size, seed, level, reports, running, app_runs_dir=None, app_champions_dir=None) -> int`
    (0, or `UNUSABLE_RUN`); `app.trainer_process(..., app_runs_dir=None, app_champions_dir=None)` exits with that code.
  - `App.app_runs_dir`, `App.app_champions_dir`, `App.app_available` ({approach: seeds}), `App.app_choice`
    ({approach: "off" or seed}), `App.app_lineup` ([(approach, seed, "app")]), `App.race_group` ("final" or "app"),
    `App.new_run()`, `App.has_saved_run()`, `App.saved_history()`, `App.race_cars()`, `App.drop_car(car)`,
    `App.app_choice_label(approach)`, `App.cycle_app_choice(approach)`, `App.cycle_race_group()`,
    `App.driver(approach, seed, source="final")`.
  - Race cars are `(approach, seed)` for reported controllers (unchanged) and `(approach, seed, "app")` for app cars.

- [ ] **Step 1: Write the failing tests**

In `tests/test_app.py`, change the import line `from racing import config, neat, nets, paths, population, setup, sim`
(as Task 5 left it) to:

```python
from racing import checkpoint, config, neat, nets, paths, population, setup, sim
```

Replace the last two assertion lines of `test_trainer_main_uses_app_runs_dir_and_never_saves_champions`

```python
    assert not calls["ga"]["existed"]                              # every Play starts a new run
    assert [reports.get_nowait()["generation"] for _ in range(3)] == [0, 0, 0] and reports.empty()
```

with

```python
    assert calls["ga"]["existed"]                                  # Play continues the saved run (revision 3.1 §6.1)
    assert [reports.get_nowait()["generation"] for _ in range(3)] == [0, 0, 0] and reports.empty()
```

and rename the test to `test_trainer_main_uses_app_runs_dir_and_resumes_saved_runs`. (`ga_report` reports have the
approach "ga" even for the ES and NEAT fakes; the app car they save lands in the autouse temporary folder.)

Delete `test_trainer_main_starts_each_viewer_run_afresh` (it pinned the revision 3 behavior that revision 3.1 replaces)
and add at the end of the file:

```python
# ----- saved application runs (revision 3.1 §6.1) -----
def fake_trainer_context(monkeypatch, created):
    """Training processes that record their arguments and never start (tests never start real processes)."""
    class FakeProcess:
        def __init__(self, target, args, daemon):
            created.append(args)
            self.pid = 0

        def start(self):
            pass

        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    class FakeContext:
        Process = FakeProcess

        def Queue(self):
            return FakeQueue()

        def Event(self):
            return threading.Event()

    monkeypatch.setattr(viewer.mp, "get_context", lambda method: FakeContext())


def save_app_run(window, approach="ga", seed=0):
    """A saved viewer run (checkpoint and a two-generation log) and its app car."""
    run = window.app_runs_dir / approach / f"seed{seed}"
    run.mkdir(parents=True)
    (run / checkpoint.CHECKPOINT_NAME).write_bytes(b"saved")
    (run / "log.csv").write_text("generation,min_share,event\n0,0.5,\n1,0.9,level_up\n")
    save_race_champions(window.app_champions_dir, approaches=[(approach, seed)])
    return run


def test_app_runs_and_cars_of_tests_live_in_a_temporary_folder(window):
    real = config.RESULTS_DIR / "app_runs"
    assert config.APP_RUNS_DIR != real and window.app_runs_dir == config.APP_RUNS_DIR
    assert window.app_champions_dir == config.APP_CHAMPIONS_DIR == config.APP_RUNS_DIR / "champions"


def test_trainer_main_resumes_a_saved_evolutionary_run(monkeypatch, tmp_path):
    from racing import ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    saved = tmp_path / "ga" / "seed0" / "checkpoint.pkl"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"an earlier viewer run")
    seen = []
    monkeypatch.setattr(ga, "run_ga", lambda **kwargs: seen.append((saved.exists(), kwargs["out_dir"])))
    running = threading.Event()
    running.set()
    assert viewer.trainer_main("ga", 30, 0, 1, queue.Queue(), running) == 0
    assert seen == [(True, tmp_path / "ga" / "seed0")]


def test_trainer_main_still_starts_imitation_afresh(monkeypatch, tmp_path):
    fake_imitation(monkeypatch, tmp_path)
    old = tmp_path / "app_runs" / "baseline" / "seed0" / "imitation.keras"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"an earlier viewer run")
    run_baseline_trainer()
    assert not old.exists()


def test_the_best_car_of_each_generation_is_saved_as_an_app_car(monkeypatch, tmp_path):
    from racing import ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    sent = []

    def run(**kwargs):
        for generation in (0, 1):
            sent.append(ga_report(generation))
            kwargs["on_generation"](sent[-1])

    monkeypatch.setattr(ga, "run_ga", run)
    running = threading.Event()
    running.set()
    viewer.trainer_main("ga", 30, 0, 1, queue.Queue(), running)
    last = sent[-1]
    best = int(np.argmax(last["fitness"]))
    _, driver = nets.load_champion("ga", 0, config.APP_CHAMPIONS_DIR)
    obs = np.random.default_rng(5).random((40, config.N_INPUTS))
    assert np.array_equal(driver.act(obs), last["population"].driver(best).act(obs))
    assert np.array_equal(driver.setup, last["population"].setup[best])
    assert not (config.APP_CHAMPIONS_DIR / ".staging").exists()      # files moved into place, staging removed


def test_a_neat_app_car_keeps_its_genome(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    report = neat_report(0)
    monkeypatch.setattr(neat, "run_neat", lambda **kwargs: kwargs["on_generation"](report))
    running = threading.Event()
    running.set()
    viewer.trainer_main("neat", 20, 0, 1, queue.Queue(), running)
    best = int(np.argmax(report["fitness"]))
    _, driver = nets.load_champion("neat", 0, config.APP_CHAMPIONS_DIR)
    assert isinstance(driver, neat.NeatNet)
    obs = np.random.default_rng(6).random((40, config.N_INPUTS))
    assert np.array_equal(driver.act(obs), report["population"].driver(best).act(obs))


def test_the_best_imitation_network_so_far_is_the_app_car(monkeypatch, tmp_path):
    from racing import imitation
    fake_imitation(monkeypatch, tmp_path)
    rng = np.random.default_rng(7)
    models = [nets.to_keras(nets.init_population(rng, 1)[0]) for _ in range(3)]

    def train(data, on_epoch=None, **kwargs):
        for epoch, (model, accuracy) in enumerate(zip(models, (0.4, 0.7, 0.6))):
            on_epoch(epoch, model, accuracy)

    monkeypatch.setattr(imitation, "train_baseline", train)
    run_baseline_trainer()
    _, driver = nets.load_champion("baseline", 0, config.APP_CHAMPIONS_DIR)
    obs = rng.random((40, config.N_INPUTS))
    assert np.array_equal(driver.act(obs), nets.from_keras(models[1]).act(obs))     # epoch 1 had the best accuracy


@pytest.mark.parametrize("error", [checkpoint.SettingsMismatch("GA_POP: recorded 1000, now 900"),
                                   checkpoint.UnreadableRecord("summary.json cannot be read"),
                                   EOFError("Ran out of input")])
def test_an_unusable_saved_run_ends_the_trainer_with_code_3(monkeypatch, tmp_path, capsys, error):
    from racing import ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)

    def run(**kwargs):
        raise error

    monkeypatch.setattr(ga, "run_ga", run)
    running = threading.Event()
    running.set()
    reports = queue.Queue()
    assert viewer.trainer_main("ga", 30, 0, 1, reports, running) == viewer.UNUSABLE_RUN == 3
    assert reports.empty() and "cannot be resumed" in capsys.readouterr().err


def test_the_trainer_process_exits_with_the_trainer_code(monkeypatch):
    monkeypatch.setattr(viewer.os, "setpgrp", lambda: None)                 # never changes the test's process group
    monkeypatch.setattr(viewer, "watch_parent", lambda parent, kill: None)  # the watcher signals nothing
    monkeypatch.setattr(viewer, "trainer_main", lambda *args: viewer.UNUSABLE_RUN)
    with pytest.raises(SystemExit) as stop:
        viewer.trainer_process("ga", 30, 0, 1, queue.Queue(), threading.Event())
    assert stop.value.code == 3
    monkeypatch.setattr(viewer, "trainer_main", lambda *args: 0)
    viewer.trainer_process("ga", 30, 0, 1, queue.Queue(), threading.Event())          # returns normally


def test_the_panel_asks_for_new_run_when_a_saved_run_cannot_resume(window):
    class Unusable:
        exitcode = 3

        def is_alive(self):
            return False

    window.process, window.reports = Unusable(), FakeQueue()
    window.poll_reports()
    window.draw()
    assert "New run" in window.status and len(window.status) <= 56 and window.process is None


def test_a_finished_run_says_new_run_starts_over(window):
    class Done:
        exitcode = 0

        def is_alive(self):
            return False

    window.process, window.reports = Done(), FakeQueue()
    window.poll_reports()
    assert "finished" in window.status.lower() and "New run" in window.status


def test_new_run_deletes_the_saved_run_and_its_app_car_then_starts(window, monkeypatch):
    created = []
    fake_trainer_context(monkeypatch, created)
    run = save_app_run(window)
    window.new_run()
    assert not run.exists()
    assert not paths.champion_path("ga", 0, champion_dir=window.app_champions_dir).exists()
    assert len(created) == 1 and created[0][:3] == ("ga", config.GA_POP, 0)
    assert created[0][6:] == (window.app_runs_dir, window.app_champions_dir)
    assert window.history == []
    window.new_run()                                    # ignored while training runs
    assert len(created) == 1


def test_play_resumes_a_saved_run_and_continues_its_plot(window, monkeypatch):
    created = []
    fake_trainer_context(monkeypatch, created)
    run = save_app_run(window)
    assert window.has_saved_run()
    window.toggle_play()
    assert run.exists() and len(created) == 1
    assert [(g, s, marked) for g, s, _, marked in window.history] == [(0, 0.5, False), (1, 0.9, True)]


def test_imitation_approaches_have_no_saved_run(window):
    save_app_run(window, approach="baseline")
    window.select_approach("baseline")
    assert not window.has_saved_run() and window.saved_history() == []


def test_an_idle_panel_mentions_a_saved_run(window, monkeypatch):
    save_app_run(window)
    drawn, text = [], window.text

    def recording_text(string, x, y, font=None, color=viewer.INK):
        drawn.append(string)
        return text(string, x, y, font, color)

    monkeypatch.setattr(window, "text", recording_text)
    window.draw()
    assert any(string.startswith("Saved run found") for string in drawn)


def test_the_new_run_button_is_offered_only_while_idle(window, monkeypatch):
    window.draw()
    assert any(action == window.new_run for _, action in window.buttons)
    fake_trainer_context(monkeypatch, [])
    window.toggle_play()
    window.draw()
    assert not any(action == window.new_run for _, action in window.buttons)


def app_race_window(window, tmp_path):
    save_race_champions(tmp_path)                                         # reported: es 0, ga 0, ga 1
    save_race_champions(window.app_champions_dir, approaches=[("ga", 2), ("neat", 0)])
    window.champions_dir = tmp_path
    window.set_mode("race")
    return window


def test_app_cars_form_their_own_group_switched_off_by_default(window, tmp_path):
    app_race_window(window, tmp_path)
    assert window.race_available == {"es": [0], "ga": [0, 1]}
    assert window.app_available == {"ga": [2], "neat": [0]}
    assert window.app_lineup == [] and window.app_choice_label("ga") == "off"
    window.cycle_app_choice("ga")
    assert window.app_choice_label("ga") == "seed 2"
    assert window.race_cars() == window.lineup + [("ga", 2, "app")]
    window.update(1 / 60)
    window.start_race()
    window.update(1 / 60)
    assert window.race is not None and len(window.race.cars.x) == len(window.race_cars())
    assert nets.champion_seeds(window.champions_dir) == {"es": [0], "ga": [0, 1]}     # reported set untouched


def test_the_race_group_button_switches_the_controller_list(window, tmp_path, monkeypatch):
    app_race_window(window, tmp_path)
    drawn, text = [], window.text

    def recording_text(string, x, y, font=None, color=viewer.INK):
        drawn.append(string)
        return text(string, x, y, font, color)

    monkeypatch.setattr(window, "text", recording_text)
    window.draw()
    assert "Controllers: reported ▸" in window.race_group_label() and "ES" in drawn
    window.cycle_race_group()
    drawn.clear()
    window.draw()
    assert window.race_group == "app" and "NEAT (app)" in drawn and "ES" not in drawn
    window.cycle_race_group()
    assert window.race_group == "final"


def test_without_app_cars_the_race_panel_is_unchanged(window, tmp_path):
    race_window(window, tmp_path)
    window.cycle_race_group()                                   # nothing to switch to
    assert window.app_available == {} and window.race_group == "final"


def test_the_evaluation_never_reads_app_cars():
    import inspect
    from racing import evaluate
    source = inspect.getsource(evaluate)
    assert "APP_CHAMPIONS_DIR" not in source and "APP_RUNS_DIR" not in source and "app_runs" not in source


def test_app_cars_are_labelled_in_the_network_selector_and_leaderboard(net, tmp_path):
    app_race_window(net, tmp_path)
    net.cycle_app_choice("neat")
    assert net.race_cars()[-1] == ("neat", 0, "app")
    while net.selected_car() != ("neat", 0, "app"):
        net.cycle_network_car()
    assert net.network_car_label() == "Network of: NEAT (app) s0 ▸"
    assert viewer.leaderboard_line(3, "neat", 0, 1, 31.0, 30.5, "racing", app=True).startswith("3. NEAT (app) s0  lap 1")
    assert viewer.leaderboard_line(2, "ga", 1, 0, float("nan"), float("nan"), "ready") == "2. GA s1  lap 0  –/–  ready"


def test_a_failed_app_car_is_dropped_with_a_message(window, tmp_path):
    app_race_window(window, tmp_path)
    window.cycle_app_choice("ga")
    paths.champion_path("ga", 2, champion_dir=window.app_champions_dir).write_bytes(b"not a model")
    window.drivers = {}
    window.update(1 / 60)
    assert ("ga", 2, "app") not in window.race_cars() and "GA (app) seed 2" in window.status


@pytest.mark.parametrize("group", ["final", "app"])
def test_a_race_of_every_method_and_every_app_car_fits_the_panel(window, tmp_path, monkeypatch, group):
    every = [(approach, 0) for approach in config.APPROACHES]
    save_race_champions(tmp_path, approaches=every)
    save_race_champions(window.app_champions_dir, approaches=every)
    window.champions_dir = tmp_path
    window.set_mode("race")
    for approach in config.APPROACHES:
        window.cycle_app_choice(approach)
    if group == "app":
        window.cycle_race_group()
    window.update(1 / 60)
    window.start_race()
    window.update(1 / 60)
    assert len(window.race_cars()) == 2 * len(config.APPROACHES)
    drawn, text = [], window.text

    def recording_text(string, x, y, font=None, color=viewer.INK):
        drawn.append((string, y, font or window.font))
        return text(string, x, y, font, color)

    monkeypatch.setattr(window, "text", recording_text)
    window.buttons = []
    window.draw_race_panel(window.panel.left + 16, 58, window.panel.width - 32)
    for string, y, font in drawn:
        assert y + font.get_linesize() <= window.panel.bottom - 30, string
    assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons)
    rows = [string for string, _, _ in drawn if string[:1].isdigit() and ". " in string]
    assert rows and (len(rows) == 16 or any(string.startswith("… ") for string, _, _ in drawn))
```

`race_window`, `save_race_champions`, `fake_imitation`, `run_baseline_trainer`, `ga_report`, `neat_report`,
`FakeQueue` and the `net` fixture already exist in `tests/test_app.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_app.py -q -x -k "app_run or app_car or saved_run or new_run or unusable or resumes or afresh or race_group or app_cars or trainer_process_exits or finished_run"`
Expected: FAIL — first `test_trainer_main_uses_app_runs_dir_and_resumes_saved_runs` (`assert calls["ga"]["existed"]`:
the trainer still removes the saved run). Without `-x` the other new tests fail too, for example
`test_app_runs_and_cars_of_tests_live_in_a_temporary_folder` with `AttributeError: ... has no attribute
'app_runs_dir'` and the trainer tests with `AttributeError: module 'racing.app' has no attribute 'UNUSABLE_RUN'`.

- [ ] **Step 3: Add the app champions path and keep every test's app runs private**

In `racing/config.py`, directly below `APP_RUNS_DIR = RESULTS_DIR / "app_runs"`, add:

```python
APP_CHAMPIONS_DIR = APP_RUNS_DIR / "champions"   # the best car of each viewer run (revision 3.1 §6.1); never evaluated
```

Do not add it to `checkpoint.SETTINGS_KEYS`.

In `tests/conftest.py`, append:

```python
@pytest.fixture(autouse=True)
def private_app_runs(tmp_path_factory, monkeypatch):
    """Viewer runs and their app cars of every test live in a temporary folder, never in results/
    (revision 3.1 §6.1)."""
    from racing import config
    folder = tmp_path_factory.mktemp("app_runs")
    monkeypatch.setattr(config, "APP_RUNS_DIR", folder)
    monkeypatch.setattr(config, "APP_CHAMPIONS_DIR", folder / "champions")
```

- [ ] **Step 4: Module-level code in `racing/app.py`**

1. Replace the imports block

```python
import json
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
```

with

```python
import csv
import json
import multiprocessing as mp
import os
import pickle
import queue
import signal
import sys
import threading
import time
from pathlib import Path
```

and the line `from racing import config, netwindow, nets, paths, sim` with

```python
from racing import checkpoint, config, netwindow, nets, paths, population, sim
```

2. Directly above `def select_indices(`, add:

```python
UNUSABLE_RUN = 3                    # trainer exit code: the saved viewer run cannot be resumed (revision 3.1 §6.1)
UNUSABLE_ERRORS = (checkpoint.SettingsMismatch, checkpoint.UnreadableRecord, EOFError, pickle.UnpicklingError)


def is_app(car):
    """Whether a race car is an app car (approach, seed, "app") rather than a reported controller (approach, seed)."""
    return tuple(car[2:]) == ("app",)


def car_method(car):
    """The method label of a race car: 'GA' for a reported controller, 'GA (app)' for an app car."""
    return config.LABELS[car[0]] + (" (app)" if is_app(car) else "")
```

3. Replace `leaderboard_line` with:

```python
def leaderboard_line(position, approach, seed, laps, last, best, status, app=False):
    """One leaderboard row: position, method and seed (app cars marked), laps completed, last/best lap time (s),
    status."""
    times = "/".join("–" if np.isnan(value) else f"{value:.1f}" for value in (last, best))
    label = config.LABELS[approach] + (" (app)" if app else "")
    return f"{position}. {label} s{seed}  lap {laps}  {times}  {status}"
```

4. Directly above `def trainer_main(`, add:

```python
def save_app_car(report, champion_dir):
    """The best car of an evolutionary report, saved with its setup as the run's app car (revision 3.1 §6.1): the
    files of a final controller, in `champion_dir`. They are written in a staging folder and each moved into place in
    one step, so the race view, which may load the car while training continues, never reads a half-written file."""
    from racing import neat
    cars = report["population"]
    best = int(np.argmax(report["fitness"]))
    approach, seed = report["approach"], report["seed"]
    staging = Path(champion_dir) / ".staging"        # not an approach folder, so the race view never lists it
    paths.remove(staging)
    if isinstance(cars, neat.NeatPopulation):
        neat.save_neat_champion(approach, seed, cars.genomes[best], staging, setup=cars.setup[best])
    else:
        population.save_dense_champion(approach, seed, cars, best, staging)
    target = Path(champion_dir) / approach
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted((staging / approach).iterdir()):
        os.replace(path, target / path.name)
    paths.remove(staging)


def save_app_model(approach, seed, model, champion_dir):
    """An imitation network saved as the run's app car (revision 3.1 §6.1). It has no setup file, so it drives with
    the revision 3 setup. The file is replaced in one step, so the race view never reads half a model."""
    population.clear_champion(approach, seed, champion_dir)
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.keras")
    model.save(temporary)
    os.replace(temporary, path)
```

5. Replace the whole function `trainer_main` with:

```python
def trainer_main(approach, size, seed, level, reports, running, app_runs_dir=None, app_champions_dir=None):
    """Training process: trains one approach with run seed `seed` and sends a report after every generation, trial
    or epoch (revision 3 §9). An evolutionary run continues the saved run of the same approach and seed in the app
    runs folder (App.new_run removes it to start over); imitation training always starts afresh. The run's best car
    so far is saved to the app champions folder so it can race (revision 3.1 §6.1); the reported results are never
    written. Returns 0, or UNUSABLE_RUN when the saved run cannot be resumed (other settings or a damaged file)."""
    from racing import es, ga, imitation, neat
    from racing.expert import collect_dataset

    app_runs_dir = Path(config.APP_RUNS_DIR if app_runs_dir is None else app_runs_dir)
    app_champions_dir = Path(config.APP_CHAMPIONS_DIR if app_champions_dir is None else app_champions_dir)
    out_dir = app_runs_dir / approach / f"seed{seed}"

    def send(report):
        if "population" in report:
            save_app_car(report, app_champions_dir)
        reports.put(report)
        running.wait()                      # training holds here while the viewer is paused

    common = dict(seed=seed, pop_size=size, start_level=level, out_dir=out_dir, on_generation=send,
                  champion_dir=None, n_jobs=config.N_JOBS, max_generations=None)
    if approach in config.EVOLUTION_APPROACHES:
        try:
            if approach == "ga":
                ga.run_ga(**common)
            elif approach == "neat":
                neat.run_neat(**common)
            else:
                initial = es.dagger_baseline(seed) if approach == "es_seeded" else None
                es.run_es(**common, initial=initial, approach=approach)
        except UNUSABLE_ERRORS as error:
            print(f"The saved viewer run in {out_dir} cannot be resumed: {error}", file=sys.stderr, flush=True)
            return UNUSABLE_RUN
        return 0
    paths.remove(out_dir)                   # imitation training in the viewer is never resumed
    path = config.DATASET_PATH
    if not path.exists():
        path = app_runs_dir / "dataset.npz"     # never write the reported dataset
        if not path.exists():
            collect_dataset(path)
    data = imitation.load_dataset(path)
    members, scores = [], []

    def add(number, model, val_accuracy):
        if not scores or val_accuracy > max(scores):
            save_app_model(approach, seed, model, app_champions_dir)    # the best network so far can race
        members.append(nets.from_keras(model))
        scores.append(val_accuracy)
        fitness = np.array(scores)
        send({"approach": approach, "generation": int(number), "level": level,
              "seeds": [config.KT_VIEW_SEED], "nets": list(members), "fitness": fitness,
              "best": float(fitness.max()), "mean": float(fitness.mean()), "worst": float(fitness.min()),
              "leveled_up": False})

    if approach == "baseline":                         # resume=False: always train live, never skip (revision 3 §9)
        imitation.train_baseline(data, out_dir=out_dir, seed=seed, resume=False,
                                 on_epoch=lambda epoch, model, acc: add(epoch, model, acc))
    else:
        imitation.run_tuner(approach, data, out_dir=out_dir, on_trial=add, seed=seed, export_models=False,
                            resume=False)
    return 0
```

6. Replace the whole function `trainer_process` with:

```python
def trainer_process(approach, size, seed, level, reports, running, app_runs_dir=None, app_champions_dir=None):
    """Entry point of the training process. It leads its own process group, so stopping it also stops the worker
    processes it starts (spec §10.2). A watcher thread ends the group if the viewer exits without stopping it. The
    process exits with the trainer's code, so the viewer can tell an unusable saved run (revision 3.1 §6.1)."""
    os.setpgrp()
    threading.Thread(target=watch_parent, args=(os.getppid(), os.killpg), daemon=True).start()
    code = trainer_main(approach, size, seed, level, reports, running, app_runs_dir, app_champions_dir)
    if code:
        sys.exit(code)
```

- [ ] **Step 5: Saved runs in the training view**

1. In `App.__init__`, directly after `self.runs_dir = config.RUNS_DIR ...`, add:

```python
        self.app_runs_dir = config.APP_RUNS_DIR             # saved viewer runs (revision 3.1 §6.1)
        self.app_champions_dir = config.APP_CHAMPIONS_DIR   # their best cars, raceable in the race view
```

and directly after `self.failed = set() ...` (the last line of `__init__`), add:

```python
        self.app_available = {}             # approach -> run seeds with a saved app car (revision 3.1 §6.1)
        self.app_choice = {}                # approach -> "off" or a run seed
        self.app_lineup = []                # (approach, seed, "app") of the app cars in the race
        self.race_group = "final"           # the controller list shown: "final" (reported) or "app"
```

2. In `start_training`, replace

```python
        self.process = context.Process(target=trainer_process, daemon=False,
                                       args=(self.approach, size, self.run_seed, self.difficulty, self.reports,
                                             self.running))
```

with

```python
        self.process = context.Process(target=trainer_process, daemon=False,
                                       args=(self.approach, size, self.run_seed, self.difficulty, self.reports,
                                             self.running, self.app_runs_dir, self.app_champions_dir))
```

and replace

```python
        self.latest, self.history, self.playback, self.playback_generation = None, [], None, None
        self.net_sent = None                                 # a new run's first report goes to the window
```

(the lines inside `start_training`, after `self.halted = False`) with

```python
        self.latest, self.history, self.playback, self.playback_generation = None, self.saved_history(), None, None
        self.net_sent = None                                 # a new run's first report goes to the window
```

3. Directly after `toggle_play`, add:

```python
    def new_run(self):
        """Delete the saved run of the selected approach and run seed and its app car, then start training afresh
        with the current settings (revision 3.1 §6.1). Ignored while training runs."""
        if self.process is not None:
            return
        paths.remove(self.saved_run_dir())
        population.clear_champion(self.approach, self.run_seed, self.app_champions_dir)
        self.start_training()

    def saved_run_dir(self):
        return self.app_runs_dir / self.approach / f"seed{self.run_seed}"

    def has_saved_run(self):
        """Whether Play continues a saved run of the selected evolutionary approach and run seed."""
        return (self.approach in config.EVOLUTION_APPROACHES
                and (self.saved_run_dir() / checkpoint.CHECKPOINT_NAME).exists())

    def saved_history(self):
        """The plot history of the saved run from its log (lowest map share per generation, level advancements,
        goal and target cuts marked), so a resumed run's plot continues where it stopped; empty for imitation, for
        a new run or for an unreadable log."""
        log = self.saved_run_dir() / "log.csv"
        if self.approach not in config.EVOLUTION_APPROACHES or not log.exists():
            return []
        history = []
        try:
            with open(log, newline="") as f:
                for row in csv.DictReader(f):
                    history.append((int(row["generation"]), float(row["min_share"]), float("nan"),
                                    row.get("event") in ("level_up", "goal", "tightened")))
        except (OSError, ValueError, KeyError, TypeError):
            return []
        return history
```

4. In `poll_reports`, replace

```python
            self.status = "Training finished." if code == 0 else f"Training stopped (exit code {code})."
```

with

```python
            if code == 0:
                self.status = "Training finished. New run starts over."
            elif code == UNUSABLE_RUN:
                self.status = "Saved run cannot resume: press New run to start over."
            else:
                self.status = f"Training stopped (exit code {code})."
```

5. In `draw_train_panel`, replace

```python
        half = (w - 8) // 2
        self.button(pygame.Rect(x, y, half, 32), "Pause" if playing else "Play", self.toggle_play, playing)
        self.button(pygame.Rect(x + half + 8, y, half, 32), "Stop", self.stop, enabled=not idle)
        y += 42
```

with (Play, Stop and New run share one row, so the panel does not grow):

```python
        self.button(pygame.Rect(x, y, third, 32), "Pause" if playing else "Play", self.toggle_play, playing)
        self.button(pygame.Rect(x + third + 8, y, third, 32), "Stop", self.stop, enabled=not idle)
        self.button(pygame.Rect(x + 2 * (third + 8), y, third, 32), "New run", self.new_run, enabled=idle)
        y += 42
```

(`third = (w - 16) // 3` is already defined at the top of `draw_train_panel`; if no other line of the function uses
`half`, its definition goes with the replaced lines.)

6. In `draw_stats`, replace

```python
        if r is None:
            return self.text("No generation completed yet.", x, y, self.small, MUTED)
```

with

```python
        if r is None:
            if self.process is None and self.has_saved_run():
                return self.text("Saved run found: Play resumes, New run restarts.", x, y, self.small, MUTED)
            return self.text("No generation completed yet.", x, y, self.small, MUTED)
```

- [ ] **Step 6: App cars in the race view**

1. Replace `selected_car`, `cycle_network_car` and `network_car_label` with:

```python
    def selected_car(self):
        """The race car whose network the window shows: the first car if the choice is outside the race; None for
        an empty race."""
        cars = self.race_cars()
        if not cars:
            return None
        return cars[self.net_choice if 0 <= self.net_choice < len(cars) else 0]

    def cycle_network_car(self):
        cars = self.race_cars()
        if cars:
            self.net_choice = (cars.index(self.selected_car()) + 1) % len(cars)

    def network_car_label(self):
        car = self.selected_car()
        return "Network of: –" if car is None else f"Network of: {car_method(car)} s{car[1]} ▸"
```

2. In `shown_network`, in the race branch, replace

```python
        except Exception:
            self.failed.add(car)
            self.lineup.remove(car)
            self.status = f"Controller {config.LABELS[car[0]]} seed {car[1]} could not be loaded; skipped."
            return None
```

with

```python
        except Exception:
            self.drop_car(car)
            self.status = f"Controller {car_method(car)} seed {car[1]} could not be loaded; skipped."
            return None
```

and in the race branch's returned title replace the method part `config.LABELS[car[0]]` with `car_method(car)`,
keeping the rest of the title (including the setup suffix that Task 4 added) unchanged, so the return reads:

```python
        return identity, driver, (f"{car_method(car)} · seed {car[1]} · {driver.n_params:,} parameters"
                                  f"{netwindow.netfig.setup_text(driver)}")
```

3. In `load_race`, replace

```python
        self.race_available = nets.champion_seeds(self.champions_dir)
```

with

```python
        self.race_available = nets.champion_seeds(self.champions_dir)
        self.app_available = nets.champion_seeds(self.app_champions_dir)
```

and replace

```python
        if not self.race_available:
            self.status = f"No trained controllers found in {self.champions_dir}."
        else:
            self.status = note or "Race view."
        self.reset_race()
```

with

```python
        kept_app = self.app_choice
        self.app_choice = {approach: kept_app.get(approach) if kept_app.get(approach) in seeds else "off"
                           for approach, seeds in self.app_available.items()}     # app cars start switched off
        if not self.app_available:
            self.race_group = "final"
        if not self.race_available and not self.app_available:
            self.status = f"No trained controllers found in {self.champions_dir}."
        else:
            self.status = note or "Race view."
        self.reset_race()
```

4. Directly after `cycle_choice`, add:

```python
    def app_choice_label(self, approach):
        choice = self.app_choice.get(approach, "off")
        return "off" if choice == "off" else f"seed {choice}"

    def cycle_app_choice(self, approach):
        if self.racing:
            return
        options = ["off", *self.app_available[approach]]
        current = self.app_choice.get(approach, "off")
        self.app_choice[approach] = options[(options.index(current) + 1) % len(options)]
        self.reset_race()

    def race_group_label(self):
        return f"Controllers: {'app runs' if self.race_group == 'app' else 'reported'} ▸"

    def cycle_race_group(self):
        """Switch the controller list between the reported controllers and the app cars (revision 3.1 §6.1)."""
        if self.app_available:
            self.race_group = "app" if self.race_group == "final" else "final"

    def race_cars(self):
        """The cars of the race: the chosen reported controllers (approach, seed), then the chosen app cars
        (approach, seed, "app")."""
        return self.lineup + self.app_lineup

    def drop_car(self, car):
        """Leave a car whose controller cannot be loaded out of the race."""
        self.failed.add(car)
        for cars in (self.lineup, self.app_lineup):
            if car in cars:
                cars.remove(car)
```

5. Replace `reset_race` with:

```python
    def reset_race(self):
        """Back to the map preview: the selected controllers and app cars wait on the start line."""
        self.racing = False
        self.race = None
        self.step_budget = 0.0
        selected = self.selected_car()
        self.lineup = []
        for approach in self.race_available:
            seed = self.chosen_seed(approach)
            if seed is not None and (approach, seed) not in self.failed:
                self.lineup.append((approach, seed))
        self.app_lineup = [(approach, choice, "app") for approach, choice in self.app_choice.items()
                           if choice != "off" and (approach, choice, "app") not in self.failed]
        cars = self.race_cars()
        self.net_choice = cars.index(selected) if selected in cars else 0
```

6. Replace `driver` with:

```python
    def driver(self, approach, seed, source="final"):
        """The driver of a reported controller, or of an app car (source "app"), loaded once."""
        key = (approach, seed) if source == "final" else (approach, seed, source)
        if key not in self.drivers:
            directory = self.champions_dir if source == "final" else self.app_champions_dir
            self.drivers[key] = nets.load_champion(approach, seed, directory)[1]
        return self.drivers[key]
```

7. Replace `prepare_race` with (Task 5 already passes `setup` and `grip` to `Playback`; keep exactly its arguments):

```python
    def prepare_race(self):
        """Build the preview once the track is ready: every selected car on the start line."""
        if self.race is None and self.race_cars():
            track = self.race_track()
            if track is not None:
                drivers = []
                for car in list(self.race_cars()):
                    try:
                        drivers.append(self.driver(*car))
                    except Exception:
                        self.drop_car(car)
                        self.status = f"Controller {car_method(car)} seed {car[1]} could not be loaded; skipped."
                if drivers:
                    self.race = Playback(track, nets.mixed_policy(drivers), len(drivers), laps=config.RACE_LAPS,
                                         setup=nets.drivers_setup(drivers), grip=self.grip)
```

8. In `draw`, replace

```python
            track = playback.track if playback is not None else (self.race_track() if self.race_available else None)
```

with

```python
            available = self.race_available or self.app_available
            track = playback.track if playback is not None else (self.race_track() if available else None)
```

9. In `draw_racers`, replace

```python
        for i, (approach, _) in enumerate(self.lineup):
            color = hex_to_rgb(config.COLORS[approach])
            if cars.crashed[i]:
                color = tuple(int(0.5 * c + 127) for c in color)          # faded
            shape = car_polygon(sx[i], sy[i], cars.heading[i], size)
            pygame.draw.polygon(self.screen, color, shape)
            pygame.draw.polygon(self.screen, INK, shape, 1)
```

with

```python
        for i, car in enumerate(self.race_cars()):
            color = hex_to_rgb(config.COLORS[car[0]])
            if cars.crashed[i]:
                color = tuple(int(0.5 * c + 127) for c in color)          # faded
            shape = car_polygon(sx[i], sy[i], cars.heading[i], size)
            pygame.draw.polygon(self.screen, color, shape)
            if is_app(car):                                                # app cars: a white border
                pygame.draw.polygon(self.screen, WHITE, shape, 3)
            pygame.draw.polygon(self.screen, INK, shape, 1)
```

10. In `draw_race_panel`:

- Replace the first lines

```python
        if not self.race_available:
            self.text("No trained controllers found.", x, y, self.small, MUTED)
            return
```

with

```python
        if not self.race_available and not self.app_available:
            self.text("No trained controllers found.", x, y, self.small, MUTED)
            return
```

- In the Start race button and the Network of button, replace `bool(self.lineup)` with `bool(self.race_cars())`.
- Replace the controller list, from `y = self.text("Controllers", x, y, self.small, MUTED)` through the `y += 28` of
  its loop, with (the cycle button takes the place of the "Controllers" heading, so the panel grows by at most 5 px):

```python
        if self.app_available:
            self.button(pygame.Rect(x, y, w, 22), self.race_group_label(), self.cycle_race_group)
            y += 24
        else:
            y = self.text("Controllers", x, y, self.small, MUTED)
        if self.race_group == "app" and self.app_available:
            for approach in self.app_available:
                self.swatch(approach, pygame.Rect(x, y + 8, 10, 10))
                self.text(f"{config.LABELS[approach]} (app)", x + 16, y + 5, self.small)
                self.button(pygame.Rect(x + 140, y, w - 140, 24), self.app_choice_label(approach),
                            lambda a=approach: self.cycle_app_choice(a), enabled=idle)
                y += 28
        else:
            for approach in self.race_available:
                self.swatch(approach, pygame.Rect(x, y + 8, 10, 10))
                self.text(config.LABELS[approach], x + 16, y + 5, self.small)
                self.button(pygame.Rect(x + 140, y, w - 140, 24), self.choice_label(approach),
                            lambda a=approach: self.cycle_choice(a), enabled=idle)
                y += 28
```

- Replace the leaderboard loop

```python
        for position, i, laps, last, best, status in leaderboard(self.race, len(self.lineup)):
            approach, seed = self.lineup[i]
            self.swatch(approach, pygame.Rect(x, y + 3, 10, 10))
            y = self.text(leaderboard_line(position, approach, seed, laps, last, best, status), x + 16, y,
                          self.small)
```

with (as many rows as fit above the status line; the rest are counted):

```python
        cars = self.race_cars()
        rows = leaderboard(self.race, len(cars))
        room = max((self.panel.bottom - 30 - y) // (self.small.get_linesize() + 4), 1)
        shown = rows if len(rows) <= room else rows[:room - 1]
        for position, i, laps, last, best, status in shown:
            car = cars[i]
            self.swatch(car[0], pygame.Rect(x, y + 3, 10, 10))
            y = self.text(leaderboard_line(position, car[0], car[1], laps, last, best, status, app=is_app(car)),
                          x + 16, y, self.small)
        if len(shown) < len(rows):
            self.text(f"… {len(rows) - len(shown)} more", x + 16, y, self.small, MUTED)
```

Keep every other line of `draw_race_panel` (track selector, difficulty, Task 5's grip/rays row, speed buttons, the
leaderboard heading) unchanged.

- [ ] **Step 7: Run the tests**

Run: `~/miniconda3/bin/python -m pytest tests/test_app.py -q`
Expected: all passed (the existing panel-fit test `test_a_race_of_every_method_fits_the_panel` still shows all eight
leaderboard rows, because without app cars the controller heading is unchanged).

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all passed (533 after Task 5 + 25 new − 1 deleted = 557).

Run: `git status --short results/ | head -3`
Expected: no output (no test wrote into results/).

- [ ] **Step 8: Commit**

```bash
git add racing/config.py racing/app.py tests/conftest.py tests/test_app.py
git commit -m "Viewer keeps its training: Play resumes, New run starts over, unusable saved runs are named, app cars race in their own group

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/config.py racing/app.py tests/conftest.py tests/test_app.py
```

---

### Task 7: Pilots, README, design specification 3.1

Revision 3.1 §7 and §11: the pilots are repeated with setup evolution and report the populations' final setups; the
README describes the evolved setup, the grip selector, per-car rays, saved viewer training and app cars; the design
specification becomes version 3.1 with revision 3.1 folded in; the revision 3.1 specification states the imitation app
car rule as built (Task 6); both PDFs are regenerated.

**Files:**
- Modify: `scripts/pilots.py`, `tests/test_pilots.py`
- Modify: `README.md`
- Modify: `docs/2026-10-05-ec-racing-design.md`; regenerate `docs/2026-10-05-ec-racing-design.pdf`
- Modify: `docs/2026-10-07-ec-racing-setup-design.md` (§6.1 one bullet); regenerate its PDF

**Interfaces:**
- Consumes: Task 3's log columns `sensor_mean`, `sensor_std`, `steer_mean`, `steer_std`; `config.SENSOR_LENGTH_LIMITS`,
  `config.STEER_GAIN_LIMITS`; Tasks 5–6 behavior described in the README; `scripts/md2pdf.py`.
- Produces: pilot rows with `sensor_last` and `steer_last`; pilots in `results/pilots/revision3.1/`.

- [ ] **Step 1: Write the failing pilot test**

In `tests/test_pilots.py`, add `config` to the import line (`from racing import config, nets, population, tracks`) and
replace `test_a_short_pilot_summarizes_its_run` with:

```python
@pytest.mark.parametrize("name", ["ga", "es", "neat"])
def test_a_short_pilot_summarizes_its_run(pilots, tmp_path, name):
    row = pilots.pilot(name, tmp_path, max_generations=2, n_jobs=1, pop_size=60)
    assert row["generations"] == 2 and row["level_reached"] >= 1
    assert isinstance(row["level_ups"], str) and 0.0 <= row["min_share_last20"] <= 1.0
    low, high = config.SENSOR_LENGTH_LIMITS
    assert low <= row["sensor_last"] <= high                    # the population's mean setup at the end
    low, high = config.STEER_GAIN_LIMITS
    assert round(low, 2) <= row["steer_last"] <= high
    assert (tmp_path / name / "log.csv").exists()


def test_revision_3_1_pilots_have_their_own_folder(pilots):
    assert pilots.OUT == config.PILOTS_DIR / "revision3.1"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `~/miniconda3/bin/python -m pytest tests/test_pilots.py -q`
Expected: FAIL — `KeyError: 'sensor_last'` and `assert ... == .../revision3.1`.

- [ ] **Step 3: Update `scripts/pilots.py`**

1. Replace the module docstring's first sentence block

```python
"""Pilot runs of revision 3 (§10.2): short runs of each evolutionary method from Easy with seed 0, which show
whether the 90% requirement can be reached with the reported settings before the reported run starts. Each pilot
starts over in results/pilots/revision3/<name>/ and adds a row to results/pilots/revision3/summary.md. Pilots are
not part of the reported results.
```

with

```python
"""Pilot runs of revision 3.1 (§7; revision 3 §10.2): short runs of each evolutionary method from Easy with seed 0,
which show whether the 90% requirement can be reached with the reported settings, now including the evolved car
setup, before the reported run starts. Each pilot starts over in results/pilots/revision3.1/<name>/ and adds a row to
results/pilots/revision3.1/summary.md, with the population's mean sensor length and steering sensitivity at the end.
Pilots are not part of the reported results.
```

2. Replace `OUT = config.PILOTS_DIR / "revision3"` with `OUT = config.PILOTS_DIR / "revision3.1"`.

3. In `pilot`, replace the returned dictionary

```python
    return {"pilot": NAMES[name], "generations": int(summary["generations"]),
            "level_reached": int(summary["level_reached"]), "level_ups": level_ups(log),
            "min_share_last20": round(float(log["min_share"].tail(20).mean()), 3),
            "minutes": round(float(summary["wall_time"]) / 60, 1)}
```

with

```python
    return {"pilot": NAMES[name], "generations": int(summary["generations"]),
            "level_reached": int(summary["level_reached"]), "level_ups": level_ups(log),
            "min_share_last20": round(float(log["min_share"].tail(20).mean()), 3),
            "sensor_last": round(float(log["sensor_mean"].iloc[-1]), 1),
            "steer_last": round(float(log["steer_mean"].iloc[-1]), 2),
            "minutes": round(float(summary["wall_time"]) / 60, 1)}
```

4. In `main`, replace `description="Revision 3 pilot runs from Easy with seed 0."` with
   `description="Revision 3.1 pilot runs from Easy with seed 0."`.

Run: `~/miniconda3/bin/python -m pytest tests/test_pilots.py -q`
Expected: all passed.

- [ ] **Step 4: Update `README.md`**

1. Replace

```
method runs with three seeds; results are reported as mean ± standard deviation. The design specification is
`docs/2026-10-05-ec-racing-design.md` (version 3.0); the revision 3 specification and plan are in `docs/`.
```

with

```
method runs with three seeds; results are reported as mean ± standard deviation. Every evolved car also evolves its
own setup next to its network: the length of its distance sensors (10–150 m) and its steering sensitivity (⅓–3),
judged only by success and lap time. The design specification is `docs/2026-10-05-ec-racing-design.md` (version 3.1);
the revision 3 and 3.1 specifications and plans are in `docs/`.
```

2. In the results table, replace the row

```
| `results/pilots/revision3/` | Pilot runs |
```

with

```
| `results/pilots/revision3/`, `results/pilots/revision3.1/` | Pilot runs of revision 3 and of revision 3.1 (with setup evolution) |
| `results/app_runs/<method>/seed<k>/`, `results/app_runs/champions/` | Training started in the viewer, and the best car of each such run; never part of the reported results |
```

3. In "Pilots and benchmarks", replace `The pilots write to \`results/pilots/revision3/\` and are not part of the
reported results.` with `The pilots write to \`results/pilots/revision3.1/\` and are not part of the reported results;
their summary includes the populations' final mean sensor length and steering sensitivity.`

4. In "Viewer", replace the bullets from `- **Training view:**` through `- **Race view:** ...start line.` (the first two
bullets) with:

```
- **Training view:** choose an approach, run seed, difficulty and population size, then press Play (or Space).
  Training follows the mastery rules and continues until Stop, which freezes the cars where they are. The panel shows
  the level, each map's success share against 90%, the streak, the target reduction and the population's mean sensor
  length and steering sensitivity.
- **Saved viewer training:** training started here is kept in `results/app_runs/<method>/seed<k>/`, separate from the
  reported results. For GA, ES, seeded ES and NEAT, Play continues the saved run of the selected method and seed where
  it stopped (difficulty and population size apply only to a new run); **New run** deletes it and starts over. If a
  saved run cannot be resumed (other settings after a code update, or a damaged file), the panel says so: press New
  run. Baseline and Keras Tuner training always start afresh.
- **Race view:** pick a test map with the arrow keys and a difficulty; the map is shown with the cars on the start
  line. For each method, choose its best seed, a specific seed, or off, then press Start race; Stop returns to the
  start line. The **Controllers** button switches the list to the **app runs**: the best car of each viewer training
  run (saved after every generation, or whenever an imitation network improves), labelled for example *GA (app) s1*.
  App cars start switched off, race next to the reported controllers when switched on (drawn with a white border),
  and are never part of the reported results. When many cars race, the leaderboard lists as many as fit.
- **Grip:** *Normal* (12 m/s², the training grip) or *Unlimited* (120 m/s²) sets the grip of the cars you watch in
  both views. It never affects training or results.
- **Sensor rays:** shows the outlined car's nine rays, each drawn to that car's own sensor length.
```

Keep the Network window, Difficulty and Playback speed bullets unchanged.

- [ ] **Step 5: Revision 3.1 specification: the imitation app car rule as built**

In `docs/2026-10-07-ec-racing-setup-design.md`, §6.1, replace the whole bullet

```
- **Raceable car.** After every generation (evolutionary methods) or at the end of training (imitation methods), the
  run's current best car is saved with its setup in `results/app_runs/champions/<method>/seed<k>.*`, the same files as a
  final controller. In the race view these cars appear in their own group, labelled for example **GA (app) s1**; each is
  off by default and can be switched on to race next to the reported controllers.
```

with

```
- **Raceable car.** After every generation (evolutionary methods) or whenever the validation accuracy improves
  (imitation methods, so a stopped run still leaves its best network), the run's best car so far is saved with its
  setup in `results/app_runs/champions/<method>/seed<k>.*`, the same files as a final controller. In the race view
  these cars appear in their own group, labelled for example **GA (app) s1**; each is off by default and can be
  switched on to race next to the reported controllers.
```

(If the existing line breaks differ, match on the sentence text, not the line breaks.)

- [ ] **Step 6: Design specification 3.1**

Apply these edits to `docs/2026-10-05-ec-racing-design.md`. Match on the quoted text (line breaks in the file may
differ; quoted phrases are unique in the file). Where text is inserted, insert it exactly as given.

1. **Version line.** Replace `Version 3.0 · 7 October 2026` with `Version 3.1 · 7 October 2026`.

2. **Version paragraph.** Insert before `Version 3.0 incorporates revision 3`:

```
Version 3.1 incorporates revision 3.1 (`docs/2026-10-07-ec-racing-setup-design.md`): every evolved car also
evolves its sensor length and steering sensitivity, the application can display races under a much higher grip, and
training started in the application is kept.
```

(followed by a space, so the existing sentence continues the paragraph).

3. **§3.2 module tree.** Insert after the line `│   ├── sim.py          vehicle dynamics, sensors, collisions, progress, fitness`:

```
│   ├── setup.py        evolved car setup: defaults, random draws, crossover, mutation
```

and after the line `│   ├── netfig.py       network figure`:

```
│   ├── netwindow.py    the application's network window (separate process)
```

4. **§5.1 steering.** Replace

```
  - Steering `u_s ∈ [−1, 1]` requests a lateral acceleration `a_lat = u_s × μg`.
```

with

```
  - Steering `u_s ∈ [−1, 1]` requests a lateral acceleration `a_lat = clip(k × u_s, −1, 1) × μg`, where k is the
    car's steering sensitivity (§9.11; k = 1 for every car that does not evolve its setup).
```

5. **§5.1 sensor-range consequence.** Replace

```
    Stopping within the 50 m sensor range requires a speed below about 35 m/s.
```

with

```
    Stopping within a sensor length L requires a speed below about √(2 μg L): about 35 m/s for 50 m and 60 m/s for
    150 m.
- **Grip in the application:** the visualization application can display races with a grip of 120 m/s² (§10.5);
  training, evaluation and the expert always use μg = 12 m/s².
```

6. **§5.2 Sensors.** Replace the whole section body (from `Nine rays at −90°, −67.5°, …, +90°` to `…does not limit the
vehicle.`) with:

```
Nine rays at −90°, −67.5°, …, +90° relative to the heading measure the distance to the track boundary, up to the
car's sensor length L: 50 m for every car that does not evolve its setup, and 10–150 m for evolved cars (§9.11). Rays
are cast by sphere tracing on the distance-to-boundary map, with a minimum step of 0.5 m. A ray of length L takes at
most ⌈48 × L / 50⌉ iterations, so a 50 m ray keeps its cap of 48 iterations. Each ray stops as soon as it reaches the
boundary or its length.

The input vector has 10 elements: the nine distances divided by 50 m, plus the speed divided by a reference speed of
40 m/s. The ray scale is fixed, so a ray of a car with L = 150 m can report values up to 3, and a car with L = 50 m
sees exactly the inputs of version 3.0. The reference speed only scales the input and does not limit the vehicle.
```

7. **§6 setup files.** After the sentence ending `…all driving evaluations of evolutionary controllers use these
files.`, insert:

```
An evolved controller also stores its setup (sensor length and steering sensitivity) in `<seed>.setup.json`; a
controller without this file drives with L = 50 m and k = 1.
```

8. **§9.1 setup and logging.** After the sentence ending `…NEAT starts from minimal networks and grows them.`,
insert ` Every car of these populations also carries an evolved setup (§9.11).` In the logging paragraph, replace
` NEAT also logs its number of species.` with ` Every run also logs the mean and standard deviation of the
population's sensor length and steering sensitivity, and NEAT logs its number of species.`

9. **§9.10 tracks per process.** Replace `Every process keeps at most 64 built
  tracks, so memory stays constant over runs of many thousands of generations.` (match on `at most 64 built`) with
`Every process keeps at most 8 built tracks, so memory stays constant over runs of many thousands of generations.`
Then insert as a new bullet after that bullet:

```
- **Sensor length:** tracing time grows with a ray's length, so a population that evolves long sensors drives each
  generation more slowly. Results do not depend on it.
```

10. **New §9.11.** Insert before `## 10. Visualization Application`:

```
### 9.11 Evolved car setup

Every car of an evolutionary population carries two setup values next to its network: its **sensor length** L
(10–150 m) and its **steering sensitivity** k (⅓–3). Both are stored and mutated on a logarithmic scale, so halving
and doubling are equally large changes. The setup carries no cost; it is judged only by the time trials, through
success and lap time.

- **Sensor length** limits all nine rays of the car (§5.2).
- **Steering sensitivity** multiplies the steering of every action class before the grip limit (§5.1). Below 1 it
  gives finer steering but a gentler sharpest turn; above 1 it cannot exceed the grip limit, strengthens the
  half-steering levels and saturates the full ones, so the steering becomes coarser.
- **Populations:** the GA, the ES, NEAT, the seeded ES and the speed push of each imitation method evolve their setup.
  The imitation training and DAgger use L = 50 m and k = 1, because the expert data and corrections are recorded with
  those sensors.
- **Initial setups:** populations that start from random networks (GA, ES, NEAT) draw every car's setup uniformly on
  the logarithmic scale within the ranges; populations that start from a trained network (seeded ES, speed pushes)
  start with L = 50 m and k = 1, the setup the network was trained with.
- **Inheritance and mutation**, the same for every method: a kept car keeps its setup; a child inherits each value
  from a parent (GA: from either crossover parent with equal probability; ES: from its parent; NEAT: from the fitter
  parent of a crossover or from the single parent); each inherited value is mutated with probability 0.2 by Gaussian
  noise with standard deviation 0.1 on the natural-logarithm scale (about ±10%) and clipped to its range. Setup
  mutation is independent of the network step sizes.
- **Saved controllers** store their setup (§6). The evaluation, the network figure and the application drive every
  controller with its own setup.
```

11. **§10.2 training view.** Replace the sub-bullet that begins `- Training started from the application writes to
\`results/app_runs/\` and never saves final` and ends `…be started and the view states that notebook 03 must be run
first.` with:

```
  - Training started from the application writes only to `results/app_runs/`, never to the reported results, and is
    kept there (§10.6). The seeded ES in the application starts from the DAgger-refined baseline of the selected run
    seed. If it does not exist, the seeded ES cannot be started and the view states that notebook 03 must be run
    first.
```

In the same section, replace `The fittest displayed individual is outlined, and its sensor rays can be shown.` with
`The fittest displayed individual is outlined, and its sensor rays can be shown, each drawn to the car's own sensor
length.` In the **Side panel** list, insert after `- mutation strength (GA) or σ (ES, NEAT), and the number of species
(NEAT)` the line `- the population's mean sensor length and steering sensitivity, with their standard deviations`.
In the **Controls** list, replace `ended with a stop control. After a
    stop, the approach, run seed, difficulty and population size can be changed.` (match on `ended with a stop control`)
with `ended with a stop control, which freezes the displayed cars. After a stop, the approach, run seed, difficulty and
population size can be changed.`

12. **§10.3 race view.** Insert as the last bullet of §10.3 (after `- **Playback speed:** 1×, 2×, 4× or 8×, where 1× is
twice real time.`):

```
- **App cars:** a cycle button switches the controller list between the reported controllers and the app cars, the
  best cars of training started in the application (§10.6), labelled for example *GA (app) s1*. Each app car is off by
  default and can be switched on to race next to the reported controllers, drawn with a white border; app cars never
  enter the evaluation. When many cars race, the leaderboard shows as many rows as fit and states how many more there
  are.
```

13. **New §10.5–10.7.** Insert before `## 11. Evaluation Methodology`:

```
### 10.5 Grip selector

A selector with the options **Normal** (the training grip, 12 m/s²) and **Unlimited** (10 times the training grip,
120 m/s²) sets the grip of everything the application simulates for display: the race view and the replays in the
training view. It never affects training, which always uses the training grip, so the reported results cannot depend
on it. A grip of 120 m/s² stands in for unlimited grip, because an unbounded grip would let a car turn arbitrarily far
within one time step.

### 10.6 Saved application training

Training started in the application is kept, separately from the reported results.

- **One saved run per method and seed** in `results/app_runs/<method>/seed<k>/`, checkpointed after every generation
  like a reported run. The reported results never read these folders, and `--fresh` keeps them.
- **Resume.** For the GA, ES, seeded ES and NEAT, Play continues the saved run of the selected method and seed where it
  stopped, at its own level and target; the difficulty and population settings apply only when a run starts. **New
  run** deletes that method and seed's saved run and its app car and starts afresh with the current settings. If a
  saved run was recorded with other settings or a file of it is damaged, the training process ends with exit code 3
  and the panel asks for New run.
- **Imitation methods** (baseline and Keras Tuner) are not resumed in the application: Play starts them afresh.
- **App cars.** After every generation (evolutionary methods), or whenever the validation accuracy improves
  (imitation methods), the run's best car so far is saved with its setup in `results/app_runs/champions/`, in the
  files of a final controller, so it can race in the race view (§10.3).

### 10.7 Network window

A separate window, opened from the side panel, draws a car's network in the layout of the network figure (§11.4). In
the training view it shows the best car of the latest generation (for imitation, the network of the latest epoch or
trial) and redraws with every report; in the race view a selector cycles through the cars in the race. The window runs
in its own process, can be moved and resized, and closes with the application.
```

14. **§11.3 results table.** Replace `and the target reduction reached in the speed
   push.` (match on `the target reduction reached in the speed`) with `the target reduction reached in the speed
   push, and the final controllers' sensor length and steering sensitivity.`

15. **§11.4 figures.** In the **Networks** bullet, after `…proportional to the weight's magnitude.` add ` Titles
include the controller's setup.` Insert after the **Lap time over training** bullet:

```
- **Setup over training.** For each method, the population's mean sensor length and mean steering sensitivity per
  generation, with level advancements and target reductions marked.
```

16. **§12.1 tests.** Insert before `- **Seeded ES:**`:

```
- **Car setup.** A ray stops at its car's length and a 50 m car reproduces the inputs, controls and results of version
  3.0 exactly; steering sensitivity scales the steering levels within the grip limit; initial setups, inheritance and
  mutation follow §9.11; setups survive subsets, parallel evaluation, checkpoints and saved controllers.
- **Application.** The grip selector changes only display simulations; a saved application run resumes after Play,
  New run starts afresh, an unusable saved run is named, and app cars race in their own group, switched off by
  default.
```

17. **§12.2 pilots.** Append to the section: ` Revision 3.1 repeats the pilots with setup evolution, in
\`results/pilots/revision3.1/\`, and reports the populations' final mean sensor length and steering sensitivity.`

18. **§13 parameter table.** Replace the rows

```
| Sensors | 9 rays from −90° to +90°, 50 m range |
| Sphere tracing | Minimum step 0.5 m; at most 48 iterations per ray |
```

with

```
| Sensors | 9 rays from −90° to +90°; length 50 m, or 10–150 m when evolved; inputs scaled by 50 m |
| Sphere tracing | Minimum step 0.5 m; at most ⌈48 × L / 50⌉ iterations for a ray of length L |
| Steering sensitivity | 1, or ⅓–3 when evolved |
| Car setup evolution | Log-uniform initial setups for random-start populations; 50 m and 1 for trained-start populations; per value mutation probability 0.2, Gaussian with standard deviation 0.1 on the natural-log scale, clipped |
```

replace `at most 64 built tracks per process` with `at most 8 built tracks per process`, and replace the row
`| Application playback | 1× equals twice real time |` with

```
| Application playback | 1× equals twice real time |
| Application grip | Normal 12 m/s²; Unlimited 120 m/s² (display only) |
```

19. **§14 risks.** Append after item 10:

```
11. **Sensor cost.** Long sensors cost tracing time; populations that evolve long sensors make generations slower.
    This is accepted and included in the reported training cost.
12. **Coarse steering.** Populations may settle at a steering sensitivity above 1, where steering saturates and loses
    smoothness. This is a result to be reported, not a defect.
```

Then check:

Run: `grep -n "Version 3.1\|### 9.11\|### 10.5\|### 10.6\|### 10.7\|setup.py\|at most 8 built" docs/2026-10-05-ec-racing-design.md`
Expected: each of the seven items appears (the 8-track phrase twice: §9.10 and §13).
Run: `grep -n "at most 64\|50 m range\|at most 48 iterations per ray\|never saves final" docs/2026-10-05-ec-racing-design.md`
Expected: no output.

- [ ] **Step 7: Regenerate the PDFs**

Run:

```bash
~/miniconda3/bin/python scripts/md2pdf.py docs/2026-10-05-ec-racing-design.md
~/miniconda3/bin/python scripts/md2pdf.py docs/2026-10-07-ec-racing-setup-design.md
```

Expected: each prints its PDF path. Open `docs/2026-10-05-ec-racing-design.pdf` (Read tool) on the pages with §9.11
and §13 and check that the new text and table rows render.

- [ ] **Step 8: Run the full suites**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all passed (557 after Task 6 + 1 = 558).
Run: `~/miniconda3/bin/python -m pytest -q -m slow`
Expected: all passed.

- [ ] **Step 9: Commit**

```bash
git add scripts/pilots.py tests/test_pilots.py README.md docs/2026-10-05-ec-racing-design.md \
        docs/2026-10-05-ec-racing-design.pdf docs/2026-10-07-ec-racing-setup-design.md \
        docs/2026-10-07-ec-racing-setup-design.pdf
git commit -m "Pilots report the evolved setup; README for setup, grip, saved viewer training and app cars; design specification 3.1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- scripts/pilots.py tests/test_pilots.py README.md \
        docs/2026-10-05-ec-racing-design.md docs/2026-10-05-ec-racing-design.pdf \
        docs/2026-10-07-ec-racing-setup-design.md docs/2026-10-07-ec-racing-setup-design.pdf
```

Do not push (the controller pushes after review).

