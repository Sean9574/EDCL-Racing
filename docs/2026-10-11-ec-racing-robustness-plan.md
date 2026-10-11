# EC Racing Revision 3.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Controllers robust to the road width (a width per map, 10–16 m, and two width-free inputs), a stronger speed
push, a robustness table, and a restart in which both machines train everything at once, each job waiting only for its
inputs.

**Architecture:** The map width is drawn per map in `racing/track_gen.py`; `racing/sim.py` adds two inputs; the speed
push rule lives in `racing/curriculum.py`. A new `racing/jobs.py` describes every job of the experiment (training runs and
notebooks 01–02) with the records it waits for and makes; `scripts/run_notebooks.py --runs` schedules jobs by those
records; `scripts/fleet.py` uses the same jobs. `racing/evaluate.py` adds the robustness table.

**Tech Stack:** Python 3.13, NumPy, TensorFlow/Keras 3 on the CPU, joblib, nbclient, pygame, pytest. No new packages.

**Spec:** `docs/2026-10-11-ec-racing-robustness-design.md`

## Global Constraints

- CPU only; no new packages (requirements.txt unchanged).
- WIDTH_RANGE = (10.0, 16.0) m; WIDTH_SALT = 13; WIDTH_SCALE = 20.0 m; N_INPUTS = 12 (9 rays, speed, position, width).
- SPEED_STEP = 0.98; SPEED_MASTERY_FRACTION = 0.75; SPEED_PATIENCE = 300; MASTERY_FRACTION stays 0.9 for the levels.
- ROBUSTNESS_WIDTHS = (8.0, 10.0, 14.0, 18.0) m (evaluation only, not a recorded setting).
- Position input: −1 left edge, 0 centerline, +1 right edge, relative to the driving direction, clipped to [−1, 1].
- Every setting that affects results is in `checkpoint.SETTINGS_KEYS`.
- Tests never write to the real `results/` (conftest patches the paths; new paths get the same treatment).
- No personal names in code or docs. Commits are path-scoped and end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never `git add -A`, never stage `.vscode/` or `results_laptop/`.
- The engineer never runs the app, training, notebooks or pilots; unit tests are the gate.

## Review Focus

1. A notebook job's kernel must write its status to the job's own status file (EC_RACING_STATUS reaches the kernel's
   `config.STATUS_PATH`); otherwise notebook 02's progress is invisible while running side by side. Test in Task 7.
2. After `--archive revision3.1`, no record of the old run may remain under `results/runs/`, or DAgger jobs would start
   from old imitation records. Test in Task 7.
3. Ctrl+C while some jobs still wait for inputs: the running ones stop, the waiting ones are listed as not started, and
   a rerun continues. Test in Task 7.
4. A robustness width at which the expert cannot drive a map (an 8 m road with hairpins) leaves that map out instead of
   failing the evaluation. Test in Task 4.
5. A car's matched centerline index counts laps without wrapping; the position input must wrap it. Test in Task 2.

---

### Task 1: A road width per map

**Files:**
- Modify: `racing/config.py` (LEVELS without `width`; WIDTH_RANGE, WIDTH_SALT)
- Modify: `racing/track_gen.py:70-150` (`circuit_points(rng, spec, width, step)`, `map_width`, `candidate_track`)
- Modify: `racing/checkpoint.py` (SETTINGS_KEYS += "WIDTH_RANGE", "WIDTH_SALT")
- Test: `tests/test_track_gen.py`, `tests/test_expert.py` (slow acceptance)

**Interfaces:**
- Produces: `track_gen.map_width(seed: int, level: int) -> float`; `circuit_points(rng, spec, width, step=0.5)`.

- [ ] **Step 1: Write the failing tests** (replace the two `spec["width"]` assertions at lines 30 and 55)

```python
from racing.track_gen import candidate_track, circuit_points, make_track, map_width

def test_each_map_has_its_own_width_in_the_range():
    widths = [make_track(seed, level).width for level in (1, 4) for seed in range(8)]
    low, high = config.WIDTH_RANGE
    assert all(low <= w <= high for w in widths) and len(set(widths)) == len(widths)


def test_a_maps_width_is_fixed_by_seed_and_level_for_every_attempt():
    assert map_width(5, 3) == map_width(5, 3) != map_width(6, 3)
    widths = {t.width for t in (candidate_track(5, 3, a) for a in range(30)) if t is not None}
    assert widths == {map_width(5, 3)}


def test_features_are_sized_by_the_given_width():
    spec = config.LEVELS[4]
    narrow = circuit_points(np.random.default_rng([3, 4, 0]), spec, 10.0)
    wide = circuit_points(np.random.default_rng([3, 4, 0]), spec, 16.0)
    assert not np.array_equal(narrow, wide)
```

In `test_easy_maps_are_gentle_loops_that_differ_by_seed` replace `assert t.width == spec["width"] == 16.0` with
`assert config.WIDTH_RANGE[0] <= t.width <= config.WIDTH_RANGE[1]`; in `test_maps_respect_width_and_minimum_corner_radius`
replace `assert t.width == spec["width"]` with `assert t.width == map_width(seed, level)`; in
`test_circuit_points_follow_the_spec_and_the_generator` pass `12.0` as the width.

- [ ] **Step 2: Run them to see them fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_track_gen.py -q`
Expected: FAIL (`map_width` not defined).

- [ ] **Step 3: Implement**

`racing/config.py`: delete `"width": …` from all four LEVELS entries and add after `DIRECTION_SALT`:

```python
WIDTH_RANGE = (10.0, 16.0)   # m: every map's road width is drawn from this range (revision 3.2 §2.1)
WIDTH_SALT = 13              # a map's width: default_rng([seed, level, WIDTH_SALT]), the same for every attempt
```

`racing/track_gen.py`:

```python
def map_width(seed, level):
    """Road width of map (seed, level) in m, drawn from WIDTH_RANGE by the seed and level alone, so a map keeps it on
    every generation attempt (revision 3.2 §2.1)."""
    low, high = config.WIDTH_RANGE
    return float(np.random.default_rng([seed, level, config.WIDTH_SALT]).uniform(low, high))
```

`circuit_points(rng, spec, width, step=0.5)`: replace `width = spec["width"]` by the parameter. `candidate_track`:

```python
    width = map_width(seed, level)
    if spec["kind"] == "loop":
        ...
    else:
        points = circuit_points(rng, spec, width)
    ...
    track = build_track(points, width, seed, level)
```

`racing/checkpoint.py`: append `"WIDTH_RANGE", "WIDTH_SALT"` to SETTINGS_KEYS. Grep for any other `["width"]`
(`grep -rn '\["width"\]' racing scripts tests`) and replace it with the track's own width.

Extend the slow acceptance test in `tests/test_expert.py`:

```python
@pytest.mark.slow
def test_the_first_training_maps_of_every_level_build_at_their_widths():
    for level in config.LEVELS:
        for seed in range(config.TRAIN_SEED_START, config.TRAIN_SEED_START + 200):
            tracks.get_track(seed, level)
```

- [ ] **Step 4: Run the tests**

Run: `~/miniconda3/bin/python -m pytest tests/test_track_gen.py tests/test_checkpoint.py -q`
Expected: PASS.

- [ ] **Step 5: Commit** `racing/config.py racing/track_gen.py racing/checkpoint.py tests/test_track_gen.py tests/test_expert.py`

### Task 2: Position and width inputs

**Files:**
- Modify: `racing/config.py` (WIDTH_SCALE; N_INPUTS = N_RAYS + 3)
- Modify: `racing/sim.py:106-115` (`road_position`, `sense`)
- Modify: `racing/netfig.py:16-22` (input labels)
- Modify: `racing/checkpoint.py` (SETTINGS_KEYS += "WIDTH_SCALE")
- Test: `tests/test_sim.py`, `tests/test_netfig.py`

**Interfaces:**
- Produces: `sim.road_position(track, cars, rows) -> np.ndarray` in [−1, 1]; observation columns `N_RAYS` speed,
  `N_RAYS + 1` position, `N_RAYS + 2` width / WIDTH_SCALE.

- [ ] **Step 1: Write the failing tests** (`tests/test_sim.py`)

```python
def placed(track, offset_left, index=0):
    """One car at centerline point `index`, `offset_left` m to the left of the driving direction."""
    cars = sim.start_cars(track, 1)
    i = index % len(track.centerline)
    tx, ty = track.tangent[i]
    cars.x[:] = track.centerline[i, 0] - ty * offset_left
    cars.y[:] = track.centerline[i, 1] + tx * offset_left
    cars.index[:] = index
    return cars


@pytest.mark.parametrize("reverse", [False, True])
def test_position_runs_from_the_left_edge_to_the_right_edge(circle_track, reverse):
    track = circle_track if not reverse else build_track(circle_track.centerline[::-1], circle_track.width, 0, 1)
    half = track.width / 2
    for offset, expected in ((0.0, 0.0), (half / 2, -0.5), (-half / 2, 0.5), (half * 0.999, -0.999)):
        cars = placed(track, offset)
        assert sim.road_position(track, cars, np.array([0])) == pytest.approx([expected], abs=1e-3)


def test_position_wraps_the_lap_counting_index(circle_track):
    m = len(circle_track.centerline)
    a = sim.road_position(circle_track, placed(circle_track, 2.0, index=7), np.array([0]))
    b = sim.road_position(circle_track, placed(circle_track, 2.0, index=7 + 2 * m), np.array([0]))
    assert a == pytest.approx(b)


def test_sense_adds_position_and_width_and_keeps_done_rows_zero(circle_track):
    cars = sim.start_cars(circle_track, 2)
    cars.done[1] = True
    obs = sim.sense(circle_track, cars)
    assert obs.shape == (2, 12) == (2, config.N_INPUTS)
    assert obs[0, config.N_RAYS + 1] == pytest.approx(0.0, abs=1e-6)
    assert obs[0, config.N_RAYS + 2] == pytest.approx(circle_track.width / config.WIDTH_SCALE)
    assert not obs[1].any()
```

`tests/test_netfig.py`: assert `netfig.input_labels()[-3:] == ["speed", "position", "width"]` and
`len(netfig.input_labels()) == config.N_INPUTS`.

- [ ] **Step 2: Run them to see them fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_sim.py tests/test_netfig.py -q`
Expected: FAIL (`road_position` not defined, shape (2, 10)).

- [ ] **Step 3: Implement**

`racing/config.py`:

```python
WIDTH_SCALE = 20.0           # m, input scale of the road width (revision 3.2 §3)
N_INPUTS = N_RAYS + 3        # rays, speed, position across the road, road width (revision 3.2 §3)
```

`racing/sim.py`:

```python
def road_position(track, cars, rows):
    """Position across the road of the cars `rows`: −1 at the left edge, 0 on the centerline, +1 at the right edge,
    relative to the driving direction, from each car's matched centerline point (revision 3.2 §3)."""
    i = np.asarray(cars.index[rows]) % len(track.centerline)
    tangent = track.tangent[i]
    dx = cars.x[rows] - track.centerline[i, 0]
    dy = cars.y[rows] - track.centerline[i, 1]
    left = -dx * tangent[:, 1] + dy * tangent[:, 0]          # component along the left normal
    return np.clip(-left / (track.width / 2), -1.0, 1.0)


def sense(track, cars):
    """Network inputs (revision 3.2 §3): ray distances / RAY_RANGE (each car's own sensor length, so a ray input can
    exceed 1), speed / SPEED_SCALE, position across the road (−1 left edge, +1 right edge) and road width /
    WIDTH_SCALE. Rows of cars that are done stay zero."""
    obs = np.zeros((len(cars.x), config.N_INPUTS))
    live = np.flatnonzero(~cars.done)
    if live.size:
        rays = cast_rays(track, cars.x[live], cars.y[live], cars.heading[live], cars.ray_length[live])
        obs[live, :config.N_RAYS] = rays / config.RAY_RANGE
        obs[live, config.N_RAYS] = cars.speed[live] / config.SPEED_SCALE
        obs[live, config.N_RAYS + 1] = road_position(track, cars, live)
        obs[live, config.N_RAYS + 2] = track.width / config.WIDTH_SCALE
    return obs
```

`racing/netfig.py`: `return labels + ["speed", "position", "width"]`. `racing/checkpoint.py`: append `"WIDTH_SCALE"`.
`racing/neat.py:16`: change the comment `# 10` to `# 12`.

- [ ] **Step 4: Run the whole fast suite and fix tests that hard-code 10 inputs**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: PASS after replacing literal input counts in tests by `config.N_INPUTS` (search `grep -rn "10)" tests | grep -i "input\|obs"` and NEAT tests that assume 10 input nodes).

- [ ] **Step 5: Commit** the files above.

### Task 3: A stronger speed push

**Files:**
- Modify: `racing/config.py` (SPEED_STEP 0.98, SPEED_MASTERY_FRACTION 0.75, SPEED_PATIENCE 300)
- Modify: `racing/curriculum.py:89-100` (`mastery_fraction`, `MasteryTracker.update`)
- Modify: `racing/evolution.py` (report key `speed_push`)
- Modify: `racing/progress.py:145-153` (threshold per stage)
- Modify: `racing/app.py` (stats line "need …" and plot threshold per report)
- Modify: `racing/checkpoint.py` (SETTINGS_KEYS += "SPEED_MASTERY_FRACTION")
- Test: `tests/test_curriculum.py`, `tests/test_progress.py`, `tests/test_app.py`

**Interfaces:**
- Produces: `curriculum.mastery_fraction(speed_push: bool) -> float`; generation report key `"speed_push": bool` (the
  rule the generation was judged by).

- [ ] **Step 1: Write the failing tests** (`tests/test_curriculum.py`)

```python
def at_goal():
    tracker = curriculum.MasteryTracker(level=config.MAX_LEVEL)
    for _ in range(config.MASTERY_STREAK):
        event = tracker.update(1.0)
    assert event == "goal" and tracker.goal_reached
    return tracker


def test_the_speed_push_passes_at_75_percent_and_the_levels_need_90():
    assert curriculum.mastery_fraction(False) == 0.9 and curriculum.mastery_fraction(True) == 0.75
    level = curriculum.MasteryTracker(level=1)
    level.update(0.8)
    assert level.streak == 0
    tracker = at_goal()
    tracker.update(0.75)
    assert tracker.streak == 1
    tracker.update(0.74)
    assert tracker.streak == 0


def test_each_mastered_target_is_2_percent_shorter():
    tracker = at_goal()
    first = tracker.factor
    for _ in range(config.MASTERY_STREAK):
        event = tracker.update(0.8)
    assert event == "tightened" and tracker.factor == pytest.approx(first * 0.98)


def test_the_speed_push_ends_after_300_generations_without_a_mastery():
    tracker = at_goal()
    events = [tracker.update(0.5) for _ in range(300)]
    assert events[-1] == "end_speed" and events[:-1] == [None] * 299
```

`tests/test_progress.py`: `progress.format_status(status(stage="speed push", target_reduction=0.1))` contains
`"/ 75%"`, and an evolution status contains `"/ 90%"`.

- [ ] **Step 2: Run them to see them fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_curriculum.py tests/test_progress.py -q`
Expected: FAIL (`mastery_fraction` not defined; factor × 0.95).

- [ ] **Step 3: Implement**

`racing/config.py`:

```python
SPEED_STEP = 0.98            # target factor multiplier per mastered target in the speed push (revision 3.2 §4)
SPEED_MASTERY_FRACTION = 0.75  # lowest map success share for a passing generation in the speed push (revision 3.2 §4)
SPEED_PATIENCE = 300         # generations without mastering the current target that end the speed push
```

`racing/curriculum.py`:

```python
def mastery_fraction(speed_push):
    """Lowest map success share of a passing generation: MASTERY_FRACTION on the levels, SPEED_MASTERY_FRACTION in
    the speed push (revision 3.2 §4)."""
    return config.SPEED_MASTERY_FRACTION if speed_push else config.MASTERY_FRACTION
```

and in `MasteryTracker.update`:
`self.streak = self.streak + 1 if min_share >= mastery_fraction(self.goal_reached) else 0`.

`racing/evolution.py`: before `event = tracker.update(result.min_share)` add `speed_push = tracker.goal_reached`, and
put `"speed_push": speed_push` into the report dict.

`racing/progress.py` in `format_status`:

```python
        need = config.SPEED_MASTERY_FRACTION if stage == "speed push" else config.MASTERY_FRACTION
        fill = 0.0 if math.isnan(share) else share / need
        parts += [f"gen {status.get('generation', '?')}",
                  f"worst map {percent(share)} / {percent(need)} {bar(fill)}",
```

`racing/app.py`: the stats line uses
`curriculum.mastery_fraction(report.get("speed_push", False))` instead of `config.MASTERY_FRACTION`, and the plot's
threshold line uses the same for `self.latest`. `racing/checkpoint.py`: append `"SPEED_MASTERY_FRACTION"`.

- [ ] **Step 4: Run the tests**

Run: `~/miniconda3/bin/python -m pytest tests/test_curriculum.py tests/test_progress.py tests/test_evolution.py tests/test_app.py -q`
Expected: PASS (update any test asserting factor × 0.95 or patience 500 to the configured values).

- [ ] **Step 5: Commit** the files above.

### Task 4: Robustness table

**Files:**
- Modify: `racing/config.py` (ROBUSTNESS_WIDTHS)
- Modify: `racing/evaluate.py:25-60, 677-720` (`_drive_on`, `widened_track`, `robustness_results`, `robustness_table`,
  `main`)
- Modify: `notebooks/08_evaluate.ipynb` (show the table)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `evaluate.widened_track(seed, width, level=MAX_LEVEL) -> Track | None`;
  `evaluate.robustness_results(drivers, track_seeds, widths=ROBUSTNESS_WIDTHS, level=MAX_LEVEL,
  factor=TARGET_FACTOR, n_jobs=1) -> DataFrame` (DRIVE_COLUMNS + "width");
  `evaluate.robustness_table(frame, widths) -> DataFrame`; `main(...)["robustness"]`.

- [ ] **Step 1: Write the failing tests** (`tests/test_evaluate.py`, which has `fake_tracks`, `drivers()` and
  `drive_rows()`)

```python
def test_a_widened_map_keeps_the_centerline_and_is_timed_by_the_expert(monkeypatch, fake_tracks):
    monkeypatch.setattr(evaluate.expert, "expert_lap_time", lambda track: 21.5)
    wide = evaluate.widened_track(80, 18.0, level=1)
    assert wide.width == 18.0 and wide.expert_lap_time == 21.5
    assert np.allclose(wide.centerline, fake_tracks.centerline, atol=1e-6)


def test_a_width_the_map_cannot_take_or_the_expert_cannot_drive_is_left_out(monkeypatch, fake_tracks):
    monkeypatch.setattr(evaluate, "geometry_ok", lambda track, min_corner: False)
    assert evaluate.widened_track(80, 18.0, level=1) is None
    monkeypatch.setattr(evaluate, "geometry_ok", lambda track, min_corner: True)
    monkeypatch.setattr(evaluate.expert, "expert_lap_time", lambda track: None)
    assert evaluate.widened_track(80, 8.0, level=1) is None


def test_robustness_results_drive_every_map_at_every_width_and_skip_left_out_maps(monkeypatch, fake_tracks):
    monkeypatch.setattr(evaluate, "widened_track",
                        lambda seed, width, level=config.MAX_LEVEL: None if (seed, width) == (1001, 18.0)
                        else fake_tracks)
    frame = evaluate.robustness_results(drivers(2), [1000, 1001], widths=(10.0, 18.0))
    assert len(frame) == 2 * 3 and set(frame["width"]) == {10.0, 18.0}
    assert frame[frame["width"] == 18.0]["track"].unique().tolist() == [1000]


def test_the_robustness_table_counts_finishes_per_method_and_width():
    narrow = drive_rows([("ga", 0, 1000, True, True, False, 20.0), ("ga", 0, 1001, False, False, True, np.nan),
                         ("es", 0, 1000, True, False, False, 25.0)])
    narrow["width"] = 10.0
    wide = drive_rows([("ga", 0, 1000, True, True, False, 18.0), ("es", 0, 1000, False, False, True, np.nan)])
    wide["width"] = 18.0
    table = evaluate.robustness_table(pd.concat([narrow, wide]), (10.0, 18.0))
    ga = table[table["Method"] == "GA"].iloc[0]
    assert (ga["10 m finished (2 maps)"], ga["10 m succeeded"], ga["10 m mean lap (s)"]) == ("1 / 2", "1 / 2", "20.00")
    assert ga["18 m finished (1 maps)"] == "1 / 1"
    es = table[table["Method"] == "ES"].iloc[0]
    assert es["18 m finished (1 maps)"] == "0 / 1" and es["18 m mean lap (s)"] == "–"
```

- [ ] **Step 2: Run them to see them fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_evaluate.py -q -k "widened or robustness"`
Expected: FAIL (`widened_track` not defined).

- [ ] **Step 3: Implement**

`racing/config.py`: `ROBUSTNESS_WIDTHS = (8.0, 10.0, 14.0, 18.0)   # m, widths of the robustness table (revision 3.2 §5.2); evaluation only`.

`racing/evaluate.py` (imports: `from racing import expert`, `from racing.track import build_track`,
`from racing.track_gen import geometry_ok`):

```python
def _drive_track(drivers, track_seed, level, factor):
    """All config.LAPS laps for every controller on one track ... (docstring unchanged)."""
    return _drive_on(drivers, get_track(track_seed, level), track_seed, factor)


def _drive_on(drivers, track, track_seed, factor):
    keys = list(drivers)
    chosen = [drivers[k] for k in keys]
    result = sim.simulate(track, nets.mixed_policy(chosen), len(keys), laps=config.LAPS,
                          setup=nets.drivers_setup(chosen))
    target = sim.target_time(track, factor)
    rows = []
    for i, (approach, run_seed) in enumerate(keys):
        ...  # the existing row dict, unchanged
    return rows


def widened_track(seed, width, level=config.MAX_LEVEL):
    """Map `seed` rebuilt with the same centerline and a `width` m road, timed by the expert; None if the road would
    reach another part of the map or the expert cannot drive it (revision 3.2 §5.2)."""
    track = build_track(get_track(seed, level).centerline, width, seed, level)
    if not geometry_ok(track, config.LEVELS[level]["min_corner"]):
        return None
    lap_time = expert.expert_lap_time(track)
    if lap_time is None:
        return None
    track.expert_lap_time = lap_time
    return track


def _drive_widened(drivers, seed, width, level, factor):
    track = widened_track(seed, width, level)
    return [] if track is None else [{**row, "width": width} for row in _drive_on(drivers, track, seed, factor)]


def robustness_results(drivers, track_seeds, widths=config.ROBUSTNESS_WIDTHS, level=config.MAX_LEVEL,
                       factor=config.TARGET_FACTOR, n_jobs=1, backend="loky"):
    """Every controller on every map rebuilt at every width (revision 3.2 §5.2); maps left out are absent."""
    tasks = [(seed, width) for width in widths for seed in track_seeds]
    if n_jobs == 1:
        parts = [_drive_widened(drivers, seed, width, level, factor) for seed, width in tasks]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_drive_widened)(drivers, seed, width, level, factor) for seed, width in tasks)
    return pd.DataFrame([row for part in parts for row in part], columns=[*DRIVE_COLUMNS, "width"])


def robustness_table(frame, widths=config.ROBUSTNESS_WIDTHS):
    """Per method and width: maps finished and succeeded (all seeds) and the mean lap time on finished maps; each
    column names how many maps it covers."""
    rows = []
    for approach in config.APPROACHES:
        sub = frame[frame["approach"] == approach]
        if sub.empty:
            continue
        row = {"Method": config.LABELS[approach]}
        for width in widths:
            cell = sub[sub["width"] == width]
            maps = frame[frame["width"] == width]["track"].nunique()
            total = len(cell)
            lap = cell.loc[cell["finished"], "lap_time"]
            row[f"{width:g} m finished ({maps} maps)"] = f"{int(cell['finished'].sum())} / {total}"
            row[f"{width:g} m succeeded"] = f"{int(cell['success'].sum())} / {total}"
            row[f"{width:g} m mean lap (s)"] = "–" if lap.empty else f"{lap.mean():.2f}"
        rows.append(row)
    return pd.DataFrame(rows)
```

In `main`: after `per_track = …` add `robust = robustness_results(drivers, test_seeds, n_jobs=n_jobs)`; add
`"robustness": robustness_table(robust)` to `tables`; write `robust.to_csv(folder /
"robustness_per_track.csv", index=False)`; return `"robustness_per_track": robust`.
Notebook 08 cell 2: add `'robustness'` to the tuple of displayed tables.

- [ ] **Step 4: Run the tests**

Run: `~/miniconda3/bin/python -m pytest tests/test_evaluate.py tests/test_runner.py::test_experiment_notebooks_are_valid_and_compile -q`
Expected: PASS.

- [ ] **Step 5: Commit** `racing/config.py racing/evaluate.py notebooks/08_evaluate.ipynb tests/test_evaluate.py`

### Task 5: Viewer: width in the map line, controllers of other input counts left out

**Files:**
- Modify: `racing/app.py` (`map_text`, `App.driver`, `prepare_race` message)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `nets.load_champion(approach, seed, directory) -> (keras model, driver)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_map_line_names_the_road_width(window, circle_track):
    window.receive(ga_report())
    window.update(1 / 60)
    assert window.map_text() == f"Map: seed {circle_track.seed}, level {circle_track.level}, road {circle_track.width:.1f} m"


def test_a_controller_of_another_input_count_is_left_out_of_the_race(window, tmp_path, monkeypatch):
    race_window(window, tmp_path)

    class OldModel:
        input_shape = (None, 10)

    real = viewer.nets.load_champion
    monkeypatch.setattr(viewer.nets, "load_champion",
                        lambda a, s, d=None: (OldModel(), real(a, s, d)[1]) if a == "ga" else real(a, s, d))
    window.update(1 / 60)
    window.prepare_race()
    assert all(car[0] != "ga" for car in window.race_cars())
    assert "10 inputs" in window.status
```

- [ ] **Step 2: Run them to see them fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_app.py -q -k "road_width or input_count"`
Expected: FAIL.

- [ ] **Step 3: Implement**

`map_text`: `return f"Map: seed {track.seed}, level {track.level}, road {track.width:.1f} m"`.

`App.driver`:

```python
        if key not in self.drivers:
            directory = self.champions_dir if source == "final" else self.app_champions_dir
            model, driver = nets.load_champion(approach, seed, directory)
            inputs = model.input_shape[-1]
            if inputs != config.N_INPUTS:                    # a controller of an earlier revision (revision 3.2 §7)
                raise ValueError(f"made for {inputs} inputs, this revision has {config.N_INPUTS}")
            self.drivers[key] = driver
```

`prepare_race`: `except Exception as error:` and
`self.status = f"{car_method(car)} seed {car[1]} left out: {error}."`.

- [ ] **Step 4: Run** `~/miniconda3/bin/python -m pytest tests/test_app.py -q` — Expected: PASS (adjust the one test that
  asserts the old "could not be loaded; skipped" message).

- [ ] **Step 5: Commit** `racing/app.py tests/test_app.py`

### Task 6: Jobs: every run and notebooks 01–02, with their inputs

**Files:**
- Create: `racing/jobs.py`
- Modify: `racing/train.py` (imitation methods; NOTEBOOKS for all eight)
- Modify: `racing/paths.py` (`final_record`)
- Test: `tests/test_jobs.py`, `tests/test_train.py`, `tests/test_paths.py`

**Interfaces:**
- Produces:
  - `paths.final_record(approach, seed, runs_dir=None) -> Path` (an evolutionary run's `summary.json`; an imitation
    method's `speed/summary.json`).
  - `jobs.Job` (frozen dataclass: `label: str`, `notebook: str | None`, `approach: str | None`, `seed: int | None`;
    property `is_notebook`).
  - `jobs.parse_job(text) -> Job` ("nb:01", "nb:02", or approach:seed for all eight methods); `jobs.parse_jobs(text)
    -> list[Job]` (comma list, each once).
  - `jobs.needs(job, results_dir) -> list[Path]`, `jobs.makes(job, results_dir) -> list[Path]`,
    `jobs.ready(job, results_dir) -> bool`, `jobs.unmet(job_list, results_dir) -> list[tuple[Job, Path]]` (inputs
    neither present nor made by a job of the list).
  - `train.train(approach, seed)` for all eight methods.

- [ ] **Step 1: Write the failing tests** (`tests/test_jobs.py`)

```python
from racing import config, jobs, paths


def test_jobs_are_notebooks_01_and_02_or_runs():
    assert jobs.parse_jobs("nb:01,ga:1,kt_random:2") == [jobs.parse_job("nb:01"), jobs.parse_job("ga:1"),
                                                          jobs.parse_job("kt_random:2")]
    assert jobs.parse_job("nb:02").notebook == "02_imitation.ipynb" and jobs.parse_job("nb:02").is_notebook
    with pytest.raises(ValueError):
        jobs.parse_jobs("nb:03")
    with pytest.raises(ValueError):
        jobs.parse_jobs("ga:1,ga:1")


def test_each_job_waits_for_the_record_that_makes_its_input(tmp_path):
    r = tmp_path
    assert jobs.needs(jobs.parse_job("nb:01"), r) == [] and jobs.needs(jobs.parse_job("ga:0"), r) == []
    assert jobs.needs(jobs.parse_job("nb:02"), r) == [r / "dataset_info.json"]
    assert jobs.needs(jobs.parse_job("kt_bayesian:1"), r) == [r / "runs" / "kt_bayesian" / "seed1" / "summary.json"]
    assert jobs.needs(jobs.parse_job("es_seeded:2"), r) == [r / "runs" / "baseline" / "seed2" / "dagger.json"]
    assert r / "runs" / "baseline" / "seed2" / "dagger.json" in jobs.makes(jobs.parse_job("baseline:2"), r)
    assert r / "runs" / "kt_random" / "seed0" / "summary.json" in jobs.makes(jobs.parse_job("nb:02"), r)


def test_an_input_nobody_makes_is_unmet(tmp_path):
    listed = jobs.parse_jobs("es_seeded:0,ga:1")
    assert [(j.label, p.name) for j, p in jobs.unmet(listed, tmp_path)] == [("es_seeded:0", "dagger.json")]
    assert jobs.unmet(jobs.parse_jobs("baseline:0,es_seeded:0,nb:01,nb:02"), tmp_path) == []
    made = tmp_path / "runs" / "baseline" / "seed0" / "dagger.json"
    made.parent.mkdir(parents=True)
    made.write_text("{}")
    assert jobs.unmet(listed, tmp_path) == [] and jobs.ready(listed[0], tmp_path)


def test_the_final_record_of_a_run():
    assert paths.final_record("ga", 1, "/r").as_posix() == "/r/ga/seed1/summary.json"
    assert paths.final_record("baseline", 2, "/r").as_posix() == "/r/baseline/seed2/speed/summary.json"
```

`tests/test_train.py`: extend `calls` with `dagger.run` (records positional and keyword arguments) and
`imitation.load_dataset` (returns the string `"dataset"`); parametrize the notebook-call test over all eight methods,
comparing only the recorded calls of that method (`[c for c in made if c[1]["approach"] == approach]`, with the dagger
fake recording `approach` from its first positional argument).

- [ ] **Step 2: Run them to see them fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_jobs.py tests/test_train.py tests/test_paths.py -q`
Expected: FAIL (`racing.jobs` missing).

- [ ] **Step 3: Implement**

`racing/paths.py`:

```python
def final_record(approach, seed, runs_dir=None):
    """The record a run writes when it is complete: an evolutionary run's summary, an imitation method's speed-push
    summary (revision 3.2 §6)."""
    if approach in config.EVOLUTION_APPROACHES:
        return run_dir(approach, seed, runs_dir) / "summary.json"
    return speed_dir(approach, seed, runs_dir) / "summary.json"
```

`racing/train.py`: `NOTEBOOKS` gains `{approach: "03_dagger.ipynb" for approach in config.IMITATION_APPROACHES}`, and
`train` starts with

```python
    if approach in config.IMITATION_APPROACHES:
        from racing import dagger, imitation
        data = imitation.load_dataset()
        return dagger.run(approach, seed, data, n_jobs=config.N_JOBS)
```

`racing/jobs.py`:

```python
"""Jobs of the side-by-side runner (revision 3.2 §6.3): the experiment's training runs and its notebooks 01 and 02.
A job waits for the records of the steps that make its inputs."""
from dataclasses import dataclass
from pathlib import Path

from racing import config, paths, train

NOTEBOOK_JOBS = {"01": "01_dataset.ipynb", "02": "02_imitation.ipynb"}


@dataclass(frozen=True)
class Job:
    label: str
    notebook: str = None
    approach: str = None
    seed: int = None

    @property
    def is_notebook(self):
        return self.notebook is not None


def parse_job(text):
    text = str(text).strip()
    if text.startswith("nb:"):
        number = text[3:]
        if number not in NOTEBOOK_JOBS:
            raise ValueError(f"{text!r}: notebook jobs are {', '.join('nb:' + n for n in NOTEBOOK_JOBS)}")
        return Job(label=text, notebook=NOTEBOOK_JOBS[number])
    approach, seed = train.parse_run(text)
    return Job(label=f"{approach}:{seed}", approach=approach, seed=seed)


def parse_jobs(text):
    found = [parse_job(part) for part in str(text).split(",") if part.strip()]
    if not found:
        raise ValueError("no jobs given; for example nb:01,nb:02,baseline:0,ga:1")
    repeated = sorted({j.label for j in found if found.count(j) > 1})
    if repeated:
        raise ValueError(f"each job may be named once: {', '.join(repeated)}")
    return found


def needs(job, results_dir=None):
    results = Path(config.RESULTS_DIR if results_dir is None else results_dir)
    runs = results / "runs"
    if job.notebook == NOTEBOOK_JOBS["02"]:
        return [results / config.DATASET_INFO_PATH.name]
    if job.approach in config.IMITATION_APPROACHES:
        return [paths.run_dir(job.approach, job.seed, runs) / "summary.json"]
    if job.approach == "es_seeded":
        return [paths.run_dir("baseline", job.seed, runs) / "dagger.json"]
    return []


def makes(job, results_dir=None):
    results = Path(config.RESULTS_DIR if results_dir is None else results_dir)
    runs = results / "runs"
    if job.notebook == NOTEBOOK_JOBS["01"]:
        return [results / config.DATASET_INFO_PATH.name]
    if job.notebook == NOTEBOOK_JOBS["02"]:
        return [paths.run_dir(a, s, runs) / "summary.json"
                for a in config.IMITATION_APPROACHES for s in config.RUN_SEEDS]
    made = [paths.final_record(job.approach, job.seed, runs)]
    if job.approach in config.IMITATION_APPROACHES:
        made.insert(0, paths.run_dir(job.approach, job.seed, runs) / "dagger.json")
    return made


def ready(job, results_dir=None):
    return all(path.exists() for path in needs(job, results_dir))


def unmet(job_list, results_dir=None):
    made = {path for job in job_list for path in makes(job, results_dir)}
    return [(job, path) for job in job_list for path in needs(job, results_dir)
            if not path.exists() and path not in made]
```

`train.parse_run` accepts every key of `NOTEBOOKS`, so all eight methods.

- [ ] **Step 4: Run** the three test files — Expected: PASS.
- [ ] **Step 5: Commit** `racing/jobs.py racing/train.py racing/paths.py tests/test_jobs.py tests/test_train.py tests/test_paths.py`

### Task 7: Runner: jobs that wait for their inputs, notebook jobs, `--archive`

**Files:**
- Modify: `racing/config.py` (STATUS_PATH honours `EC_RACING_STATUS`)
- Modify: `scripts/run_notebooks.py` (`--runs` with jobs, waiting and blocking; `--notebook-job NN`; `--archive NAME`)
- Modify: `tests/conftest.py` (none needed if STATUS_PATH stays patched)
- Test: `tests/test_runner.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `jobs.parse_jobs`, `jobs.needs`, `jobs.makes`, `jobs.ready`, `jobs.unmet`.
- Produces: `run_side_by_side(job_list, at_once, log_path, info_path, status_dir, logs_dir, monitor, out,
  command=job_command, results_dir=None)`; `job_command(job, status_file) -> list[str]`; `job_env(status_file) -> dict`;
  `archive_results(results_dir, name) -> Path`; CLI `--notebook-job NN` (internal) and `--archive NAME` (moves the
  earlier results and exits; nothing else runs).

- [ ] **Step 1: Write the failing tests** (`tests/test_runner.py`; `fake_jobs` replaces `fake_training`, and the
  `side_by_side` helper passes `results_dir=tmp_path / "results"` and takes jobs)

```python
def fake_jobs(tmp_path, seconds=0.2, fail=()):
    """A job_command stand-in: each job writes its start and end times, makes the records jobs.makes names under
    tmp_path/results unless it fails, and exits with 3 (printing 'boom') if its label is in `fail`."""
    results = tmp_path / "results"

    def command(job, status_file):
        name = job.label.replace(":", "-")
        made = [str(p) for p in jobs.makes(job, results)]
        code = 3 if job.label in fail else 0
        script = (f"import pathlib, sys, time\n"
                  f"open({str(tmp_path / (name + '.start'))!r}, 'w').write(repr(time.time()))\n"
                  f"time.sleep({seconds})\n"
                  f"for p in ({made!r} if {code} == 0 else []):\n"
                  f"    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)\n"
                  f"    pathlib.Path(p).write_text('{{}}')\n"
                  f"open({str(tmp_path / (name + '.end'))!r}, 'w').write(repr(time.time()))\n"
                  f"print({'boom' if code else 'ok'!r})\nsys.exit({code})\n")
        return [sys.executable, "-c", script]
    return command


def test_a_job_starts_only_when_its_input_exists_and_holds_no_slot_while_waiting(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.05)
    listed = jobs.parse_jobs("nb:01,ga:0,nb:02")
    assert side_by_side(runner, tmp_path, listed, 2, fake_jobs(tmp_path)) == 0
    start = {n: float((tmp_path / f"{n}.start").read_text()) for n in ("nb-01", "ga-0", "nb-02")}
    end01 = float((tmp_path / "nb-01.end").read_text())
    assert start["nb-02"] >= end01 and start["ga-0"] < end01            # ga:0 used the slot nb:02 could not


def test_dependants_of_a_failed_job_are_reported_and_the_rest_go_on(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.05)
    listed = jobs.parse_jobs("nb:01,nb:02,ga:0")
    assert side_by_side(runner, tmp_path, listed, 3, fake_jobs(tmp_path, fail={"nb:01"})) == 1
    log = (tmp_path / "log.txt").read_text()
    assert "FAILED nb:01" in log and "SKIPPED nb:02" in log and "done ga:0" in log


def test_a_job_whose_input_nobody_makes_is_refused_before_anything_starts(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "run_side_by_side", lambda *a, **k: pytest.fail("jobs were started"))
    assert run_main(runner, tmp_path, ["--runs", "es_seeded:1,ga:0"]) == 2
    assert "es_seeded:1" in capsys.readouterr().err


def test_a_notebook_job_writes_its_status_to_its_own_file(runner, tmp_path):
    command = runner.job_command(jobs.parse_job("nb:02"), tmp_path / "status" / "nb-02.json")
    assert command[-2:] == ["--notebook-job", "02"]
    env = runner.job_env(tmp_path / "status" / "nb-02.json")
    assert env["EC_RACING_STATUS"] == str(tmp_path / "status" / "nb-02.json")
    out = subprocess.run([sys.executable, "-c", "from racing import config; print(config.STATUS_PATH)"],
                         env=env, capture_output=True, text=True, cwd=SCRIPT.parent.parent).stdout.strip()
    assert out == str(tmp_path / "status" / "nb-02.json")


def test_archive_moves_what_fresh_would_delete(runner, tmp_path):
    results = tmp_path / "results"
    for name in ("runs/ga/seed0", "champions/ga", "pilots/x", "archive/older"):
        (results / name).mkdir(parents=True)
    (results / "run_log.txt").write_text("old")
    target = runner.archive_results(results, "revision3.1")
    assert (target / "runs" / "ga" / "seed0").is_dir() and (target / "run_log.txt").read_text() == "old"
    assert not (results / "runs").exists() and (results / "pilots" / "x").is_dir()
    with pytest.raises(FileExistsError):
        runner.archive_results(results, "revision3.1")


@pytest.mark.parametrize("extra", [["--fresh"], ["--from", "02"], ["--runs", "ga:0"]])
def test_archive_runs_nothing_else(runner, tmp_path, capsys, extra):
    three_notebooks(tmp_path)
    assert run_main(runner, tmp_path, ["--archive", "x", *extra]) == 2


def test_archive_alone_moves_the_results_and_exits(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    (tmp_path / "results" / "runs").mkdir(parents=True)
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("a notebook was run"))
    assert run_main(runner, tmp_path, ["--archive", "revision3.1"]) == 0
    assert (tmp_path / "results" / "archive" / "revision3.1" / "runs").is_dir()
```

Slow test (Review Focus 3): `stopping_runner` with jobs `ga:0, ga:1, es_seeded:0` at 2 at a time; after the stop, the
log reads `STOPPED ga:0 ga:1` and `not started: es_seeded:0`.

- [ ] **Step 2: Run them to see them fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_runner.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

`racing/config.py`: `STATUS_PATH = Path(os.environ.get("EC_RACING_STATUS") or RESULTS_DIR / "status.json")`
(comment: a job trained side by side gets its own status file, also inside a notebook's kernel).

`scripts/run_notebooks.py`:

```python
def job_command(job, status_file):
    """The process of one job: a notebook (this script with --notebook-job) or a training run (racing/train.py)."""
    if job.is_notebook:
        return [sys.executable, str(Path(__file__).resolve()), "--notebook-job", job.notebook[:2]]
    return train_command(job.approach, job.seed, status_file)


def job_env(status_file):
    return {**os.environ, **KERNEL_ENV, "PYTHONUNBUFFERED": "1", "EC_RACING_STATUS": str(status_file)}


def archive_results(results_dir, name):
    """--archive NAME: move what --fresh would delete to results/archive/NAME/, so the next start begins afresh
    (revision 3.2 §6.1)."""
    target = Path(results_dir) / "archive" / name
    if target.exists():
        raise FileExistsError(f"{target} already exists; choose another archive name")
    target.mkdir(parents=True)
    for item in FRESH_REMOVE:
        source = Path(results_dir) / item
        if source.exists():
            shutil.move(str(source), str(target / item))
    return target


def notebook_job(number, notebook_dir, log_path):
    """--notebook-job NN: execute one notebook, saving its outputs, for the side-by-side runner (which holds the
    results lock and logs the job). Returns 0, 1 if a cell failed, or STOPPED."""
    name = next(n for n in notebook_names(notebook_dir) if n.startswith(number))
    with tempfile.TemporaryDirectory() as kernels, open(Path(log_path).with_name(KERNEL_LOG), "a") as output:
        write_kernel(kernels)
        try:
            execute(Path(notebook_dir) / name, kernels, output)
        except KeyboardInterrupt:
            return STOPPED
        except Exception:
            return 1
    return 0
```

`run_side_by_side(job_list, at_once, …, command=job_command)`: same structure as now, with these changes:
- `pending` is the job list; on each pass, iterate over `pending` in order and start a job when a slot is free and
  `jobs.ready(job, results_dir)` holds (`results_dir = Path(log_path).parent` for the real run; a parameter for tests).
- a pending job is **blocked** when one of its `jobs.needs` paths is missing and every job that `makes` it has ended
  (finished or failed) without it; log `SKIPPED {label}: waits for {path name}, which {producer labels} did not make`,
  record `{"run": label, "skipped": reason}` and drop it.
- when nothing runs and nothing pending is ready or can still become ready, log the remaining jobs as skipped and stop.
- children get `env=job_env(status_file)`; status files are `status_dir / f"{label.replace(':', '-')}.json"`.
- on Ctrl+C: stop the running jobs as now and log
  `STOPPED {running labels}: run again to continue where they stopped; not started: {pending labels}`.

`main`: new options `--notebook-job` (hidden, `help=argparse.SUPPRESS`) and `--archive NAME`; `--archive` combined with
`--fresh`, `--from` or `--runs` → exit 2; `--notebook-job` returns `notebook_job(...)` before the lock; `--runs` parses
with `jobs.parse_jobs` and refuses `jobs.unmet` (exit 2, one `error:` line per job) before taking the lock; in
`locked_main`, `--archive` calls `archive_results`, prints where the results went and returns 0 without running
anything; an existing archive is an `error:` with exit 2. Remove `missing_inputs` (replaced by `jobs.unmet`).

- [ ] **Step 4: Run** `~/miniconda3/bin/python -m pytest tests/test_runner.py tests/test_config.py -q` and
  `~/miniconda3/bin/python -m pytest -q -m slow tests/test_runner.py` — Expected: PASS.
- [ ] **Step 5: Commit** `racing/config.py scripts/run_notebooks.py tests/test_runner.py tests/test_config.py`

### Task 8: Fleet on jobs, the revision 3.2 plan, `finish` from notebook 03

**Files:**
- Modify: `scripts/fleet.py` (runs as `jobs.Job`; external inputs; done checks by `paths.final_record`; `unfinished`
  over all 24 runs; `finish` with `--from 03`)
- Modify: `scripts/fleet.toml`
- Test: `tests/test_fleet.py`

**Interfaces:**
- Consumes: `jobs.parse_job`, `jobs.needs`, `jobs.makes`, `paths.final_record`.
- Produces: `external_inputs(machine) -> list[tuple[Path, str]]` (inputs the machine's jobs need that none of its own
  jobs make: local file, path relative to the project).

- [ ] **Step 1: Write the failing tests** (adapt `Fake.machines()` to `jobs.parse_job`)

```python
def test_jobs_whose_inputs_the_machine_makes_itself_start_without_them(fake):
    desktop = fake.fleet.Machine("desktop", [jobs.parse_job(t) for t in ("nb:01", "nb:02", "baseline:0",
                                                                         "es_seeded:0")], at_once=4)
    assert fake.fleet.external_inputs(desktop) == []
    server = fake.fleet.Machine("server", [jobs.parse_job("es_seeded:1")], host="me@h")
    assert [target for _, target in fake.fleet.external_inputs(server)] == ["results/runs/baseline/seed1/dagger.json"]


def test_finish_needs_all_24_runs_and_runs_the_notebooks_from_03(fake):
    fake.ready_server()
    for approach in config.APPROACHES:
        for seed in config.RUN_SEEDS:
            record = paths.final_record(approach, seed, fake.desktop / "results" / "runs")
            record.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_json(record, {"settings": checkpoint.settings_record()})
    assert fake.fleet.finish(fake.machines(), out=io.StringIO()) == 0
    assert "scripts/run_notebooks.py --from 03" in fake.session("desktop")


def test_a_dagger_run_on_a_server_counts_as_finished_only_after_its_speed_push(fake):
    script = fake.fleet.collect_script(fake.fleet.Machine("s", [jobs.parse_job("baseline:1")], host="h"))
    assert "results/runs/baseline/seed1/speed/summary.json" in script
```

`tests/test_fleet.py` other changes: `Fake.machines()` builds runs with `jobs.parse_job`; the probe test for a missing
baseline uses `external_inputs`; `finished_run` writes `paths.final_record`.

- [ ] **Step 2: Run them to see them fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_fleet.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

- `load_plan`: `runs = [jobs.parse_job(text) for text in entry.get("runs", [])]`; duplicates by `job.label`.
- `train_label(job)` → `job.label`; `train_command` joins labels.
- `external_inputs(machine)`:

```python
def external_inputs(machine):
    """(local file, path relative to the project) of every input the machine's jobs need that none of its own jobs
    make; a server must have them before its jobs start."""
    results = PROJECT / "results"
    made = {path for job in machine.runs for path in jobs.makes(job, results)}
    return [(path, path.relative_to(PROJECT).as_posix()) for job in machine.runs for path in jobs.needs(job, results)
            if path not in made]
```

  It replaces `baseline_inputs` in `setup_archive` (send those that exist; a missing one raises FleetError naming the
  job that makes it) and in `probe_script` (`[ -f target ] || echo 'missing target'`).
- `collect_script`: the done check per run job is
  `[ -f {paths.final_record(approach, seed, Path('results/runs')).as_posix()} ]`; notebook jobs are skipped.
- `unfinished()`: over `config.APPROACHES × config.RUN_SEEDS` with `paths.final_record`.
- `finish`: `command = f"{here.python} scripts/run_notebooks.py --from 03"`.

`scripts/fleet.toml`:

```toml
[machines.desktop]
at_once = 4
workers = 6
runs = ["neat:0", "neat:1", "nb:01", "nb:02",
        "baseline:0", "baseline:1", "baseline:2", "kt_random:0", "kt_random:1", "kt_random:2",
        "kt_hyperband:0", "kt_hyperband:1", "kt_hyperband:2", "kt_bayesian:0", "kt_bayesian:1", "kt_bayesian:2",
        "es_seeded:0", "es_seeded:1", "es_seeded:2"]

[machines.server]
host = "sbrainard@10.0.11.14"
ssh = "ssh.exe"
dir = "ec-racing"
at_once = 3
workers = 6
runs = ["neat:2", "ga:0", "ga:1", "ga:2", "es:0", "es:1", "es:2"]
```

(Expected work from revision 3.1's run lengths: desktop two NEAT slots of ~13 h and two slots carrying notebooks
01–02, 12 DAgger runs and 3 seeded ES, about 20 h; server one NEAT slot of ~20 h and two slots with GA and ES, about
23 h.)

- [ ] **Step 4: Run** `~/miniconda3/bin/python -m pytest tests/test_fleet.py -q` — Expected: PASS.
- [ ] **Step 5: Commit** `scripts/fleet.py scripts/fleet.toml tests/test_fleet.py`

### Task 9: Pilots, documents, full verification

**Files:**
- Modify: `scripts/pilots.py:3-25` (`OUT = config.PILOTS_DIR / "revision3.2"`, docstring)
- Modify: `README.md` (widths, inputs, speed push, robustness table, restart with `--archive`, jobs side by side,
  the new fleet plan)
- Modify: `docs/2026-10-05-ec-racing-design.md` (version 3.2: Table 1 widths, inputs, speed push values, evaluation,
  execution), regenerate its PDF with `~/miniconda3/bin/python scripts/md2pdf.py docs/2026-10-05-ec-racing-design.md`

- [ ] **Step 1:** pilots output folder and docstring; README; design specification.
- [ ] **Step 2: Fast suite** — `~/miniconda3/bin/python -m pytest -q` — Expected: PASS.
- [ ] **Step 3: Slow suites** — `~/miniconda3/bin/python -m pytest -q -m slow tests/test_runner.py tests/test_expert.py`
  — Expected: PASS; the acceptance test shows every fixed map and the first 200 training maps of every level build. If
  a map fails, raise MAX_TRACK_ATTEMPTS in `racing/config.py` and rerun.
- [ ] **Step 4: Commit and push** the files above; push `build`.

### After the plan (the user runs these)

1. **Pilots** on the desktop, after `git pull`: `python scripts/pilots.py` (GA, ES, NEAT and seeded ES, 300 generations
   each, into `results/pilots/revision3.2/`). They confirm that every method climbs the levels with 10–16 m roads.
2. **Restart** on the desktop: `python scripts/run_notebooks.py --archive revision3.1` (moves the old results), then,
   with GlobalProtect connected, `python scripts/fleet.py setup` and `python scripts/fleet.py start`.
3. **During the run:** `python scripts/fleet.py status`; when both machines are done, `python scripts/fleet.py finish`.
