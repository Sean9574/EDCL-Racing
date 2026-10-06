# EC Racing Revision 2.0: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement version 2.0 of the EC Racing design specification: corrected measurements, replication over
three seeds, the seeded ES, the pace requirement for curriculum advancement, faster evaluation, the experiment
runner, calibration pilots and the extended viewer controls.

**Architecture:** The existing `racing` package is extended in place. A new module `racing/paths.py` defines one
directory or file per approach and run seed. Population evaluation is divided into (track, share of population)
tasks. Every final controller is stored per seed, and evaluation aggregates across seeds. Two scripts are added:
`scripts/run_notebooks.py` executes the six notebooks in order, and `scripts/pilots.py` runs the calibration pilots.

**Tech Stack:** Python 3.13 (Conda base), numpy, scipy, TensorFlow 2.21 / Keras 3.13, keras-tuner 1.4.8, joblib,
pandas, matplotlib, pygame 2.6, pytest, nbclient and nbformat.

**Spec:** `docs/2026-10-05-ec-racing-design.md` (version 2.0, commit 40b0c01). The spec is binding; where this
plan and the spec disagree, the spec wins and the disagreement is reported.

## Summary

The plan contains thirteen tasks in six waves. Tasks within a wave touch disjoint files and can be implemented in
parallel; each wave starts after the previous one is complete. After the tasks, the controller runs the
benchmarks and pilots, presents the pilot results for a decision, runs the full test suite, starts the reported
run, regenerates the documents and merges the branch.

*Table 1. Tasks and waves.*

| Wave | Task | Deliverable | Files |
|---|---|---|---|
| 1 | 1 | Configuration and run paths | `racing/config.py`, `racing/paths.py`, `tests/test_paths.py` |
| 2 | 2 | Population evaluation by track, pace, champion storage | `racing/population.py`, `racing/nets.py`, `tests/test_population.py`, `tests/test_nets.py`, `tests/test_bench.py` (one test) |
| 2 | 4 | Imitation: epoch accounting, seeds, retraining, settings | `racing/imitation.py`, `tests/test_imitation.py` |
| 2 | 5 | Parallel, timed dataset collection | `racing/expert.py`, `tests/test_expert.py` |
| 3 | 3 | GA, ES and seeded ES | `racing/ga.py`, `racing/es.py`, `tests/test_ga.py`, `tests/test_es.py` |
| 3 | 8 | Generation and Keras benchmarks | `racing/bench.py`, `tests/test_bench.py` |
| 4 | 6 | Evaluation across seeds | `racing/evaluate.py`, `tests/test_evaluate.py` |
| 4 | 9 | Viewer: training view | `racing/app.py`, `tests/test_app.py` |
| 4 | 11 | Calibration pilots | `scripts/pilots.py`, `tests/test_pilots.py` |
| 5 | 7 | Driving-selected controllers | `racing/evaluate.py`, `tests/test_evaluate.py` |
| 5 | 10 | Viewer: race view | `racing/app.py`, `racing/nets.py`, `tests/test_app.py`, `tests/test_nets.py` |
| 5 | 12 | Notebooks, runner and README | `scripts/make_notebooks.py`, `scripts/run_notebooks.py`, `notebooks/*`, `README.md`, `tests/test_runner.py` |
| 6 | 13 | End-to-end test | `tests/test_end_to_end.py` |

Between waves the full suite is not expected to pass: Task 2 changes the return value of
`evaluate_population`, so `tests/test_ga.py` and `tests/test_es.py` fail until Task 3 lands, and
`tests/test_end_to_end.py` fails until Task 13. Each implementer runs only the test files of their own task.

## Global Constraints

- Interpreter: `~/miniconda3/bin/python` (Conda base, Python 3.13.11). Every command runs from the project root
  `~/Desktop/Classes/ECDL/Project`, branch `build`. Below, `PY=~/miniconda3/bin/python`.
- Packages: the base environment plus keras-tuner, pygame, pytest, nbclient and nbformat (all installed). Do not
  install, upgrade or reinstall anything, in particular TensorFlow, Keras or numpy.
- CPU only. All Keras access goes through `racing.backend.keras_cpu()`.
- Every numeric default lives in `racing/config.py`.
- Simulation uses float64 and has no randomness inside `sim.step`. A car's result never depends on which other cars
  are in the batch.
- Track seeds: training 0–79, validation 80–99, test 1000–1019. Test seeds are never used for training or
  selection. Run seeds: `config.RUN_SEEDS` = (0, 1, 2). Keras Tuner searches use `config.SEARCH_SEED` = 0.
- Colors: Baseline `#7F7F7F`, KT Random `#E69F00`, KT Hyperband `#56B4E9`, KT Bayesian `#CC79A7`, ES `#009E73`,
  GA `#D55E00`, Seeded ES `#0072B2`.
- Code style: plain loops and explicit conditionals, short docstrings, sparse comments. No personal names in code
  or documents.
- Generated data goes under `results/` (gitignored). The viewer writes only to `results/app_runs/`.
- Tests: `$PY -m pytest <test file> -q`. Tests marked `slow` run only with `-m slow` or when named.
- Other agents work in the same checkout at the same time. Touch only your task's files. Never run `git add -A`,
  `git add .`, `git stash`, `git checkout -- .`, `git reset` or anything that changes other files or the index
  globally. Commit with `git add <files> && git commit -m "<message>" -- <files>`; if git reports `index.lock`,
  wait two seconds and retry. Every commit message ends with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- Do not run long computations (benchmarks, notebooks, pilots, full training). The controller runs those.

## Review Focus

1. **Resumed run.** `scripts/run_notebooks.py --from 03` after an interrupted run must keep the results of
   notebooks 01–02 and the earlier run record entries; each later training run clears only its own outputs.
   Pinned by `test_resume_keeps_earlier_record_entries` (Task 12) and the clean-start tests of Tasks 2–4.
2. **Seeded ES without a baseline.** Pressing Play for the seeded ES before notebook 02 has produced the baseline
   must show a message, not crash. Pinned by `test_seeded_es_without_baseline_shows_a_message` (Task 9).
3. **Stopping training.** Stopping or closing the viewer while GA/ES training runs must also end the worker
   processes. Pinned by `test_stop_ends_the_trainer_process_group` (Task 9).
4. **Partial results.** Evaluation with only some methods or seeds present (for example after one failed GA seed)
   must produce tables for what exists. Pinned by `test_main_with_partial_results` (Task 6) and
   `test_race_lineup_falls_back_without_best_seeds` (Task 10).
5. **Hyperband continuation trials.** Epochs and configuration identity of promoted Hyperband trials must be
   counted from the training actually performed. Pinned by `test_hyperband_epochs_and_configurations` (Task 4).

---

### Task 1: Configuration and run paths

**Files:**
- Modify: `racing/config.py`
- Create: `racing/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (used by every later task):
  - config: `PACE_FACTOR`, `GA_S_MIN = GA_S_MAX / 100`, `RUN_SEEDS`, `SEARCH_SEED`, `MAX_EPOCHS = 150`,
    `HYPERBAND_ITERATIONS = 1`, `KT_BAYES_INITIAL = 10`, `STEPS_PER_EXECUTION`, `JIT_COMPILE`, `APPROACHES` (with
    `"es_seeded"`), `LABELS`, `COLORS`, `TUNER_APPROACHES`, `EVOLUTION_APPROACHES`, `PLAYBACK_BASE`, `DIFFICULTIES`,
    `DATASET_INFO_PATH`, `RUN_INFO_PATH`, `RUN_LOG_PATH`, `PILOTS_DIR`, `BEST_SEEDS_NAME`.
  - `paths.run_dir(approach, seed, runs_dir=None) -> Path`
  - `paths.search_dir(approach, runs_dir=None) -> Path`
  - `paths.champion_path(approach, seed, suffix=".keras", champion_dir=None) -> Path`
  - `paths.best_seeds_path(champion_dir=None) -> Path`
  - `paths.remove(*targets) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_paths.py`:

```python
from racing import config, paths


def test_run_and_search_directories(tmp_path):
    assert paths.run_dir("ga", 2, tmp_path) == tmp_path / "ga" / "seed2"
    assert paths.search_dir("kt_random", tmp_path) == tmp_path / "kt_random" / "search"


def test_default_directories_follow_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "CHAMPIONS_DIR", tmp_path / "champions")
    assert paths.run_dir("es", 0) == tmp_path / "runs" / "es" / "seed0"
    assert paths.champion_path("es", 1) == tmp_path / "champions" / "es" / "seed1.keras"
    assert paths.best_seeds_path() == tmp_path / "champions" / "best_seeds.json"


def test_champion_path_suffix(tmp_path):
    assert paths.champion_path("ga", 0, ".npy", tmp_path) == tmp_path / "ga" / "seed0.npy"


def test_remove_files_directories_and_missing(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "d" / "e").mkdir(parents=True)
    (tmp_path / "d" / "e" / "f.txt").write_text("y")
    paths.remove(tmp_path / "a.txt", tmp_path / "d", tmp_path / "missing.txt")
    assert list(tmp_path.iterdir()) == []


def test_seeded_es_is_an_approach_with_its_own_color():
    assert config.APPROACHES[-1] == "es_seeded"
    assert config.LABELS["es_seeded"] == "Seeded ES" and config.COLORS["es_seeded"] == "#0072B2"
    assert set(config.TUNER_APPROACHES) | set(config.EVOLUTION_APPROACHES) | {"baseline"} == set(config.APPROACHES)


def test_revision_parameters():
    assert config.MAX_EPOCHS == 150 and config.HYPERBAND_ITERATIONS == 1 and config.KT_BAYES_INITIAL == 10
    assert config.RUN_SEEDS == (0, 1, 2) and config.SEARCH_SEED == 0
    assert config.GA_S_MIN == config.GA_S_MAX / 100
    assert list(config.DIFFICULTIES) == [1, 2, 3, 4] and config.PLAYBACK_BASE == 2.0
    assert config.DATASET_INFO_PATH == config.DATASET_PATH.with_name(f"{config.DATASET_PATH.stem}_info.json")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_paths.py -q`
Expected: FAIL with `ImportError: cannot import name 'paths'`.

- [ ] **Step 3: Edit `racing/config.py`**

Replace the curriculum, GA, budget, Keras training, approaches, application and paths sections so that they read
as follows (the other sections stay unchanged):

```python
# --- Curriculum (spec §9.1) ---
TRACKS_PER_GEN = 3
LEVEL_UP_NEW_FRACTION = 0.8
LEVEL_UP_STREAK = 3
PACE_FACTOR = 1.3            # finishing at pace: lap time ≤ PACE_FACTOR × expert lap time; confirmed by pilots

# --- Genetic algorithm (spec §9.2) ---
GA_POP = 1000
GA_KEPT = 50
GA_TOURNAMENT = 5
GA_CROSSOVER_P = 0.5
GA_MUTATION_P = 0.1
GA_S_MAX = 0.5               # chosen by the pilot runs (scripts/pilots.py) from 0.5, 0.2 and 0.1
GA_S_MIN = GA_S_MAX / 100
```

```python
# --- Seeds (spec §4.4) ---
TRAIN_SEEDS = range(0, 80)
VAL_SEEDS = range(80, 100)
TEST_SEEDS = range(1000, 1020)
TRAIN_DATA_TRACKS = {1: (0,), 2: tuple(range(20)), 3: tuple(range(20)), 4: tuple(TRAIN_SEEDS)}
RUN_SEEDS = (0, 1, 2)        # every method is run once per seed (spec §11)
SEARCH_SEED = 0              # each Keras Tuner search runs once (spec §8.2)
```

```python
# --- Keras training (spec §8) ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 150
PATIENCE = 5
KT_TRIALS = 30
KT_BAYES_INITIAL = 10        # random trials before Bayesian optimization uses its model
HYPERBAND_FACTOR = 3
HYPERBAND_ITERATIONS = 1
SIZE_RULE_TOLERANCE = 0.01
STEPS_PER_EXECUTION = 1      # batches per call into the compiled training function; set from racing/bench.py
JIT_COMPILE = True           # XLA compilation; set from racing/bench.py

# --- Approaches and colors (spec §10.1) ---
APPROACHES = ("baseline", "kt_random", "kt_hyperband", "kt_bayesian", "es", "ga", "es_seeded")
TUNER_APPROACHES = ("kt_random", "kt_hyperband", "kt_bayesian")
EVOLUTION_APPROACHES = ("ga", "es", "es_seeded")
LABELS = {"baseline": "Baseline", "kt_random": "KT Random", "kt_hyperband": "KT Hyperband",
          "kt_bayesian": "KT Bayesian", "es": "ES", "ga": "GA", "es_seeded": "Seeded ES"}
COLORS = {"baseline": "#7F7F7F", "kt_random": "#E69F00", "kt_hyperband": "#56B4E9",
          "kt_bayesian": "#CC79A7", "es": "#009E73", "ga": "#D55E00", "es_seeded": "#0072B2"}

# --- Application (spec §10) ---
WINDOW_SIZE = (1280, 800)
PANEL_WIDTH = 380
FPS = 60
POPULATION_PRESETS = (100, 250, 500, 1000, 2000)
SPEEDS = (1, 2, 4, 8)
PLAYBACK_BASE = 2.0          # simulated seconds per real second at 1× playback
MAX_STEPS_PER_FRAME = 10     # simulation steps per drawn frame; a stalled frame cannot trigger a burst
DEFAULT_SHOWN = 100
KT_VIEW_SEED = 80
RACE_LAPS = 3
DIFFICULTIES = {1: "Easy", 2: "Medium", 3: "Hard", 4: "Extra hard"}   # curriculum level → name (spec §10.4)

# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
DATASET_PATH = RESULTS_DIR / "dataset.npz"
DATASET_INFO_PATH = RESULTS_DIR / "dataset_info.json"
RUNS_DIR = RESULTS_DIR / "runs"
APP_RUNS_DIR = RESULTS_DIR / "app_runs"
CHAMPIONS_DIR = RESULTS_DIR / "champions"
BEST_SEEDS_NAME = "best_seeds.json"
PILOTS_DIR = RESULTS_DIR / "pilots"
RUN_INFO_PATH = RESULTS_DIR / "run_info.json"
RUN_LOG_PATH = RESULTS_DIR / "run_log.txt"
```

The `EVAL_BUDGET` and `N_JOBS` lines stay as they are.

- [ ] **Step 4: Create `racing/paths.py`**

```python
"""Where each run and final controller is stored (spec §3.3, §9.1): one directory or file per approach and seed."""
import shutil
from pathlib import Path

from racing import config


def run_dir(approach, seed, runs_dir=None):
    """Logs and summary of one training run: <runs>/<approach>/seed<k>."""
    return Path(config.RUNS_DIR if runs_dir is None else runs_dir) / approach / f"seed{seed}"


def search_dir(approach, runs_dir=None):
    """The single Keras Tuner search of one method: <runs>/<approach>/search."""
    return Path(config.RUNS_DIR if runs_dir is None else runs_dir) / approach / "search"


def champion_path(approach, seed, suffix=".keras", champion_dir=None):
    """One final controller: <champions>/<approach>/seed<k>.keras, or the .npy weights next to it."""
    return Path(config.CHAMPIONS_DIR if champion_dir is None else champion_dir) / approach / f"seed{seed}{suffix}"


def best_seeds_path(champion_dir=None):
    """The record of each approach's best run seed, written by the evaluation (spec §11)."""
    return Path(config.CHAMPIONS_DIR if champion_dir is None else champion_dir) / config.BEST_SEEDS_NAME


def remove(*targets):
    """Delete the files and directories that exist; missing ones are ignored."""
    for target in targets:
        target = Path(target)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_paths.py tests/test_curriculum.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add racing/config.py racing/paths.py tests/test_paths.py
git commit -m "Add revision parameters, the seeded ES approach and per-seed run paths

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/config.py racing/paths.py tests/test_paths.py
```

---

### Task 2: Population evaluation by track, pace, champion storage

**Files:**
- Modify: `racing/population.py`, `racing/nets.py`
- Test: `tests/test_population.py`, `tests/test_nets.py`, `tests/test_bench.py` (only
  `test_evaluate_population_multiprocess_consistency`)

**Interfaces:**
- Consumes: Task 1 (`config.PACE_FACTOR`, `paths.champion_path`, `paths.remove`).
- Produces:
  - `population.split_tasks(n, seeds, n_jobs) -> list[(position, seed, indices)]`
  - `population.evaluate_population(weights, seeds, level, n_jobs=1, backend="loky") -> (fitness, finished, at_pace)`
    (three arrays of shape `(n,)`; **changed from two**)
  - `population.make_report(..., kept_pace, new_pace, ..., strict_success=nan, tie_fraction=nan)` (keyword-only)
  - `population.RunLogger.FIELDS` adds `kept_pace`, `new_pace`, `strict_success`, `tie_fraction`; the logger also
    removes a stale `summary.json`.
  - `population.select_champion(candidates, n_jobs=1, seeds=config.VAL_SEEDS) -> (flat, val_fitness)` (unchanged)
  - `population.save_champion(approach, seed, flat, champion_dir=config.CHAMPIONS_DIR) -> Path` (**new signature**;
    writes `<dir>/<approach>/seed<k>.keras` and `.npy`)
  - `population.clear_champion(approach, seed, champion_dir=config.CHAMPIONS_DIR) -> None`
  - `nets.to_flat(model) -> np.ndarray` (float64, inverse of `to_keras`)
  - `nets.champion_seeds(directory=None) -> {approach: [seeds]}` in `config.APPROACHES` order
  - `nets.load_champion(approach, seed, directory=None) -> (keras_model, DenseNet)`
  - `nets.mixed_policy(drivers) -> policy` (car *i* driven by `drivers[i]`)
  - `nets.load_champions` stays for now (Task 10 removes it).

- [ ] **Step 1: Write the failing tests**

In `tests/test_population.py`:
- add `import json` and `from racing import sim` to the imports;
- in `test_evaluate_population_shapes`, `test_resimulating_a_subset_reproduces_logged_fitness` and the `report`
  helper, unpack three values / pass the new keywords as shown below;
- replace `test_parallel_evaluation_matches_serial`, `test_select_champion_picks_highest_validation_fitness` and
  `test_save_champion`; add the new tests.

```python
def test_evaluate_population_shapes(fake_tracks):
    w = nets.init_population(np.random.default_rng(0), 6)
    fitness, finished, at_pace = population.evaluate_population(w, [0, 1], 1)
    assert fitness.shape == finished.shape == at_pace.shape == (6,)
    assert finished.dtype == bool and at_pace.dtype == bool and not np.any(at_pace & ~finished)
    assert np.all((fitness >= 0) & (fitness < 2))


def test_split_tasks_covers_every_individual_on_every_track():
    tasks = population.split_tasks(10, [5, 6, 7], 6)
    assert [(k, seed) for k, seed, _ in tasks] == [(0, 5), (0, 5), (1, 6), (1, 6), (2, 7), (2, 7)]
    for k in range(3):
        idx = np.concatenate([i for position, _, i in tasks if position == k])
        assert np.array_equal(np.sort(idx), np.arange(10))


def test_split_tasks_with_one_job_gives_one_task_per_track():
    tasks = population.split_tasks(10, [5, 6, 7], 1)
    assert [(k, seed, len(i)) for k, seed, i in tasks] == [(0, 5, 10), (1, 6, 10), (2, 7, 10)]


def test_split_tasks_never_makes_empty_chunks():
    assert all(len(i) > 0 for _, _, i in population.split_tasks(2, [0], 8))


def test_parallel_evaluation_matches_serial(fake_tracks):
    w = nets.init_population(np.random.default_rng(1), 9)
    serial = population.evaluate_population(w, [0, 1, 2], 1)
    threaded = population.evaluate_population(w, [0, 1, 2], 1, n_jobs=6, backend="threading")
    for a, b in zip(serial, threaded):
        assert np.array_equal(a, b)


def test_resimulating_a_subset_reproduces_logged_fitness(fake_tracks):
    w = nets.init_population(np.random.default_rng(2), 10)
    full, _, _ = population.evaluate_population(w, [0], 1)
    part, _, _ = population.evaluate_population(w[[1, 7]], [0], 1)
    assert np.array_equal(full[[1, 7]], part)


def test_at_pace_requires_finishing_within_the_pace_factor(monkeypatch, fake_tracks):
    limit = config.PACE_FACTOR * fake_tracks.expert_lap_time

    def fake_simulate(track, policy, n):
        return sim.RunResult(fitness=np.array([1.5, 1.2, 0.4]), finished=np.array([True, True, False]),
                             crashed=np.array([False, False, True]), progress=np.array([1.0, 1.0, 0.4]),
                             finish_time=np.array([limit - 1.0, limit + 1.0, np.nan]), steps=10)

    monkeypatch.setattr(population.sim, "simulate", fake_simulate)
    w = nets.init_population(np.random.default_rng(0), 3)
    _, finished, at_pace = population.evaluate_population(w, [0], 1)
    assert list(finished) == [True, True, False] and list(at_pace) == [True, False, False]


def report(generation, n=5):
    rng = np.random.default_rng(generation)
    w = nets.init_population(rng, n)
    fitness = rng.random(n)
    return population.make_report(approach="ga", generation=generation, level=1, seeds=[0, 1, 2], weights=w,
                                  fitness=fitness, finished=fitness > 0.5, is_kept=np.zeros(n, dtype=bool),
                                  kept_finish=float("nan"), new_finish=0.4, kept_pace=float("nan"), new_pace=0.2,
                                  step_size=0.5, evals=n * (generation + 1), elapsed=1.0, leveled_up=False)


def test_logger_records_pace_and_es_fractions(tmp_path):
    logger = population.RunLogger(tmp_path)
    logger.log(report(0))
    log = pd.read_csv(tmp_path / "log.csv")
    assert list(log.columns) == population.RunLogger.FIELDS
    assert log.loc[0, "new_pace"] == 0.2 and math.isnan(log.loc[0, "strict_success"])


def test_logger_removes_a_stale_summary(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    population.RunLogger(tmp_path)
    assert not (tmp_path / "summary.json").exists()


def test_select_champion_picks_highest_validation_fitness(monkeypatch):
    candidates = np.arange(12.0).reshape(4, 3)
    monkeypatch.setattr(population, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.array([0.2, 1.4, 0.9, 1.1]), np.ones(4, bool),
                                                           np.ones(4, bool)))
    best, value = population.select_champion(candidates)
    assert np.array_equal(best, candidates[1]) and value == 1.4


def test_save_champion_writes_keras_and_float64_weights(tmp_path):
    w = nets.init_population(np.random.default_rng(3), 1)[0]
    path = population.save_champion("ga", 2, w, tmp_path)
    assert path == tmp_path / "ga" / "seed2.keras" and path.exists()
    saved = np.load(tmp_path / "ga" / "seed2.npy")
    assert saved.dtype == np.float64 and np.array_equal(saved, w)


def test_clear_champion_removes_both_files(tmp_path):
    population.save_champion("es", 0, nets.init_population(np.random.default_rng(4), 1)[0], tmp_path)
    population.clear_champion("es", 0, tmp_path)
    assert list((tmp_path / "es").iterdir()) == []
```

`test_logger_stays_readable_after_each_generation`, `test_finish_fractions`, `test_make_report_fields` and
`test_logger_removes_stale_best_networks` stay unchanged.

In `tests/test_nets.py`, replace nothing yet; add (with `from racing import population` added to the imports):

```python
def test_to_flat_inverts_to_keras():
    flat = nets.init_population(np.random.default_rng(4), 1)[0]
    back = nets.to_flat(nets.to_keras(flat))
    assert back.dtype == np.float64 and np.array_equal(back, flat.astype(np.float32).astype(np.float64))


def test_champion_seeds_in_standard_order(tmp_path):
    for approach, seed in (("ga", 1), ("es_seeded", 2), ("baseline", 0), ("ga", 0)):
        (tmp_path / approach).mkdir(exist_ok=True)
        (tmp_path / approach / f"seed{seed}.keras").write_bytes(b"")
    found = nets.champion_seeds(tmp_path)
    assert list(found) == ["baseline", "ga", "es_seeded"] and found["ga"] == [0, 1]
    assert nets.champion_seeds(tmp_path / "missing") == {}


def test_load_champion_drives_with_float64_weights(tmp_path):
    flat = nets.init_population(np.random.default_rng(5), 1)[0]
    population.save_champion("es", 0, flat, tmp_path)
    model, driver = nets.load_champion("es", 0, tmp_path)
    assert model.count_params() == 431
    assert np.array_equal(np.concatenate([np.concatenate([w.ravel(), b]) for w, b in driver.layers]), flat)


def test_load_champion_without_weights_file_uses_the_keras_model(tmp_path):
    model = nets.to_keras(nets.init_population(np.random.default_rng(6), 1)[0])
    (tmp_path / "baseline").mkdir()
    model.save(tmp_path / "baseline" / "seed0.keras")
    _, driver = nets.load_champion("baseline", 0, tmp_path)
    assert driver.n_params == 431


def test_mixed_policy_drives_each_car_with_its_own_network():
    rng = np.random.default_rng(7)
    drivers = [nets.from_flat(f) for f in nets.init_population(rng, 3)]
    obs = rng.random((3, config.N_INPUTS))
    actions = nets.mixed_policy(drivers)(obs, None)
    assert list(actions) == [drivers[i].act(obs[i:i + 1])[0] for i in range(3)]
```

In `tests/test_bench.py`, change the last test to unpack three values:

```python
def test_evaluate_population_multiprocess_consistency():
    """Multi-process evaluation must match single-process on a real level-1 track."""
    weights = nets.init_population(np.random.default_rng(0), 8)
    single = population.evaluate_population(weights, [0, 1], 1)
    multi = population.evaluate_population(weights, [0, 1], 1, n_jobs=2)
    for a, b in zip(single, multi):
        assert np.array_equal(a, b)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_population.py tests/test_nets.py -q`
Expected: FAIL (`split_tasks`, `to_flat`, `champion_seeds`, `load_champion`, `mixed_policy`, `clear_champion` do
not exist; `evaluate_population` returns two values).

- [ ] **Step 3: Implement in `racing/population.py`**

Add `from racing import paths` to the imports and replace `_evaluate_chunk`, `evaluate_population`, `make_report`,
`RunLogger.FIELDS`, `RunLogger.__init__`, `select_champion` and `save_champion` with:

```python
def split_tasks(n, seeds, n_jobs):
    """(track position, track seed, individual indices) tasks for one evaluation (spec §9.5). Each track's
    individuals are split into n_jobs // len(seeds) shares (at least one), so a task drives many cars on one track."""
    shares = min(max(1, n_jobs // len(seeds)), max(n, 1))
    parts = np.array_split(np.arange(n), shares)
    return [(k, seed, part) for k, seed in enumerate(seeds) for part in parts]


def _evaluate_track(weights, seed, level):
    """Fitness, finishing and finishing at pace (spec §9.1) of every network on one track."""
    track = get_track(seed, level)
    result = sim.simulate(track, nets.population_policy(weights), len(weights))
    pace_limit = config.PACE_FACTOR * track.expert_lap_time + config.TIME_EPS
    at_pace = result.finished & (np.nan_to_num(result.finish_time, nan=np.inf) <= pace_limit)
    return result.fitness, result.finished, at_pace


def evaluate_population(weights, seeds, level, n_jobs=1, backend="loky"):
    """Mean fitness over the tracks, whether each individual finished all of them, and whether it finished all of
    them at pace. The per-track results are combined in track order, so the result does not depend on n_jobs."""
    n = len(weights)
    tasks = split_tasks(n, seeds, n_jobs)
    if n_jobs == 1 or len(tasks) == 1:
        parts = [_evaluate_track(weights[idx], seed, level) for _, seed, idx in tasks]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_evaluate_track)(weights[idx], seed, level) for _, seed, idx in tasks)
    per_track = np.zeros((len(seeds), n))
    finished = np.ones(n, dtype=bool)
    at_pace = np.ones(n, dtype=bool)
    for (k, _, idx), (fitness, done, pace) in zip(tasks, parts):
        per_track[k, idx] = fitness
        finished[idx] &= done
        at_pace[idx] &= pace
    total = np.zeros(n)
    for k in range(len(seeds)):
        total += per_track[k]
    return total / len(seeds), finished, at_pace


def make_report(*, approach, generation, level, seeds, weights, fitness, finished, is_kept, kept_finish,
                new_finish, kept_pace, new_pace, step_size, evals, elapsed, leveled_up,
                strict_success=float("nan"), tie_fraction=float("nan")):
    """One generation's results, for logging and for the training view."""
    return {"approach": approach, "generation": int(generation), "level": int(level), "seeds": list(seeds),
            "weights": weights, "hidden": list(config.BASELINE_HIDDEN), "activation": "relu",
            "fitness": fitness, "finished": finished, "is_kept": is_kept,
            "kept_finish": kept_finish, "new_finish": new_finish, "kept_pace": kept_pace, "new_pace": new_pace,
            "step_size": float(step_size), "strict_success": float(strict_success),
            "tie_fraction": float(tie_fraction),
            "best": float(fitness.max()), "mean": float(fitness.mean()), "worst": float(fitness.min()),
            "evals": int(evals), "elapsed": float(elapsed), "leveled_up": bool(leveled_up)}


class RunLogger:
    """Writes one CSV row and the best network after every generation, and a summary at the end."""
    FIELDS = ["generation", "level", "evals", "best", "mean", "worst", "kept_finish", "new_finish", "kept_pace",
              "new_pace", "step_size", "strict_success", "tie_fraction", "seeds", "elapsed", "leveled_up"]

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for stale in self.dir.glob("best_gen*.npy"):    # left over from an earlier run in the same directory
            stale.unlink()
        paths.remove(self.dir / "summary.json")
        self.csv_path = self.dir / "log.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)
```

(`log` and `summary` stay unchanged.)

```python
def select_champion(candidates, n_jobs=1, seeds=config.VAL_SEEDS):
    """The candidate with the highest mean fitness on the validation tracks (level 4)."""
    fitness, _, _ = evaluate_population(candidates, list(seeds), config.MAX_LEVEL, n_jobs=n_jobs)
    best = int(np.argmax(fitness))
    return candidates[best], float(fitness[best])


def save_champion(approach, seed, flat, champion_dir=config.CHAMPIONS_DIR):
    """Final controller as a Keras file and as float64 weights (spec §6). Returns the Keras file's path."""
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    nets.to_keras(flat).save(path)
    np.save(paths.champion_path(approach, seed, ".npy", champion_dir), np.asarray(flat, dtype=float))
    return path


def clear_champion(approach, seed, champion_dir=config.CHAMPIONS_DIR):
    """Remove a final controller left by an earlier run, so an interrupted run leaves none behind."""
    paths.remove(paths.champion_path(approach, seed, champion_dir=champion_dir),
                 paths.champion_path(approach, seed, ".npy", champion_dir))
```

- [ ] **Step 4: Implement in `racing/nets.py`**

Add `from racing import paths` to the imports (after `from racing import config`) and add:

```python
def to_flat(model):
    """Flat float64 weights of a dense Keras model, in the layout of unflatten (the inverse of to_keras)."""
    keras = keras_cpu()
    parts = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            w, b = layer.get_weights()
            parts += [np.asarray(w, dtype=float).ravel(), np.asarray(b, dtype=float)]
    return np.concatenate(parts)


def mixed_policy(drivers):
    """Policy for sim.simulate: car i is driven by drivers[i]; the networks may differ in architecture."""
    def policy(obs, cars):
        return np.array([driver.act(obs[i:i + 1])[0] for i, driver in enumerate(drivers)], dtype=np.int64)
    return policy


def champion_seeds(directory=None):
    """{approach: [run seeds]} of the final controllers saved as <directory>/<approach>/seed<k>.keras, in the
    order of config.APPROACHES. Approaches without controllers are left out."""
    directory = Path(config.CHAMPIONS_DIR if directory is None else directory)
    found = {}
    for approach in config.APPROACHES:
        seeds = sorted(int(p.stem[len("seed"):]) for p in (directory / approach).glob("seed*.keras"))
        if seeds:
            found[approach] = seeds
    return found


def load_champion(approach, seed, directory=None):
    """(Keras model, DenseNet driver) of one final controller. Evolutionary controllers drive with the float64
    weights saved next to the Keras file (spec §6)."""
    keras_path = paths.champion_path(approach, seed, champion_dir=directory)
    model = keras_cpu().models.load_model(keras_path)
    flat_path = paths.champion_path(approach, seed, ".npy", directory)
    driver = from_flat(np.load(flat_path)) if flat_path.exists() else from_keras(model)
    return model, driver
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_population.py tests/test_nets.py tests/test_bench.py::test_evaluate_population_multiprocess_consistency -q`
Expected: PASS. (`tests/test_ga.py` and `tests/test_es.py` now fail until Task 3; do not change them.)

- [ ] **Step 6: Commit**

```bash
git add racing/population.py racing/nets.py tests/test_population.py tests/test_nets.py tests/test_bench.py
git commit -m "Evaluate populations per track, record finishing at pace, store final controllers per seed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/population.py racing/nets.py tests/test_population.py tests/test_nets.py tests/test_bench.py
```

---

### Task 3: GA, ES and seeded ES

**Files:**
- Modify: `racing/ga.py`, `racing/es.py`
- Test: `tests/test_ga.py`, `tests/test_es.py`

**Interfaces:**
- Consumes: Task 2 (`evaluate_population` → three arrays, `make_report(kept_pace=, new_pace=, strict_success=,
  tie_fraction=)`, `save_champion(approach, seed, flat, dir)`, `clear_champion`, `nets.to_flat`), Task 1 (`paths`).
- Produces:
  - `ga.run_ga(pop_size=config.GA_POP, kept=config.GA_KEPT, budget=config.EVAL_BUDGET, seed=0, out_dir=None,
    on_generation=None, n_jobs=1, champion_dir=config.CHAMPIONS_DIR, start_level=1, s_max=config.GA_S_MAX) -> dict`
  - `es.run_es(mu=config.ES_MU, lam=config.ES_LAMBDA, budget=config.EVAL_BUDGET, seed=0, out_dir=None,
    on_generation=None, n_jobs=1, champion_dir=config.CHAMPIONS_DIR, start_level=1, initial=None, approach="es")
    -> dict`
  - `es.seeded_parents(rng, flat, mu, sigma=config.ES_SIGMA0) -> np.ndarray (mu, P)`
  - `es.trained_baseline(seed, champion_dir=config.CHAMPIONS_DIR) -> np.ndarray (431,)`; raises
    `FileNotFoundError` mentioning "notebook 02" if the baseline controller is missing.
  - Result dicts contain `approach`, `seed`, `generations`, `evals`, `level_reached`, `start_level`, `wall_time`
    (measured after final selection), and with a champion directory `val_fitness` and `champion_path`. The default
    output directory is `paths.run_dir(approach, seed)`.
  - Level advancement uses the at-pace fractions; the GA mutation strength uses the plain finishing fraction.
    The GA logs `step_size = nan` for generation 0.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_ga.py` with:

```python
import json
import math

import numpy as np
import pytest

from racing import config, ga, paths, population
from racing.curriculum import mutation_strength


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)


def fake_evaluation(finished, at_pace):
    def fake(w, seeds, level, n_jobs=1):
        n = len(w)
        return np.arange(n, dtype=float), np.full(n, finished), np.full(n, at_pace)
    return fake


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
    result = ga.run_ga(pop_size=20, kept=5, budget=60, out_dir=tmp_path, on_generation=reports.append,
                       champion_dir=None)
    assert result["generations"] == len(reports) == 3 and result["evals"] == 60
    assert math.isnan(reports[0]["step_size"])                    # generation 0 is not produced by mutation
    assert reports[1]["is_kept"].sum() == 5
    assert math.isclose(reports[1]["step_size"], mutation_strength(reports[0]["new_finish"]))
    assert (tmp_path / "log.csv").exists() and (tmp_path / "summary.json").exists()


def test_mutation_strength_uses_s_max_and_the_finishing_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.arange(len(w), dtype=float),
                                                           np.arange(len(w)) % 2 == 0, np.zeros(len(w), bool)))
    reports = []
    ga.run_ga(pop_size=20, kept=5, budget=40, s_max=0.2, out_dir=tmp_path, on_generation=reports.append,
              champion_dir=None)
    assert math.isclose(reports[1]["step_size"], mutation_strength(0.5, 0.2, 0.002))


def test_level_advances_after_three_generations_at_pace(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population", fake_evaluation(True, True))
    reports = []
    ga.run_ga(pop_size=20, kept=5, budget=100, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert [r["level"] for r in reports] == [1, 1, 1, 1, 2]       # generation 0 has no kept individuals
    assert reports[3]["leveled_up"]


def test_finishing_slower_than_pace_never_advances(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population", fake_evaluation(True, False))
    reports = []
    ga.run_ga(pop_size=20, kept=5, budget=100, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert [r["level"] for r in reports] == [1] * 5
    assert reports[-1]["new_finish"] == 1.0 and reports[-1]["new_pace"] == 0.0


def test_start_level(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population", fake_evaluation(False, False))
    reports = []
    result = ga.run_ga(pop_size=20, kept=5, budget=20, start_level=3, out_dir=tmp_path,
                       on_generation=reports.append, champion_dir=None)
    assert reports[0]["level"] == 3 and result["start_level"] == 3


def test_champion_per_seed_and_stale_files_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(ga, "evaluate_population", fake_evaluation(False, False))
    monkeypatch.setattr(ga, "select_champion", lambda candidates, n_jobs=1: (candidates[0], 0.5))
    champions = tmp_path / "champions"
    stale = paths.champion_path("ga", 3, ".npy", champions)
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    result = ga.run_ga(pop_size=20, kept=5, budget=40, seed=3, out_dir=tmp_path / "run", champion_dir=champions)
    assert result["champion_path"] == str(paths.champion_path("ga", 3, champion_dir=champions))
    assert np.load(stale).shape == (431,)
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["seed"] == 3 and summary["wall_time"] > 0 and summary["val_fitness"] == 0.5


def test_default_output_directory_is_per_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(ga, "evaluate_population", fake_evaluation(False, False))
    ga.run_ga(pop_size=20, kept=5, budget=20, seed=2, champion_dir=None)
    assert (tmp_path / "ga" / "seed2" / "log.csv").exists()
```

In `tests/test_es.py`:
- add `import pandas as pd` and `from racing import nets, paths` to the imports;
- change `test_operators_solve_the_sphere_function` to use the module's rule:
  `sigma = es.one_fifth_rule(sigma, es.success_fraction(fitness[10:], fitness[:10][parent_of]))`;
- make every fake `evaluate_population` return three arrays (append `np.zeros(len(w), dtype=bool)` as the third
  value; in `test_sigma_resets_after_a_level_up` append `np.ones(len(w), dtype=bool)` so everyone finishes at
  pace);
- add:

```python
def test_seeded_parents_start_from_the_trained_network():
    rng = np.random.default_rng(0)
    flat = rng.normal(size=431)
    parents = es.seeded_parents(rng, flat, 50)
    assert parents.shape == (50, 431) and np.array_equal(parents[0], flat)
    assert not np.any(np.all(parents[1:] == flat, axis=1))
    assert 0.05 < np.std(parents[1:] - flat) < 0.2                 # σ₀ = 0.1


def test_seeded_run_starts_from_the_initial_weights(fake_tracks, tmp_path):
    flat = nets.init_population(np.random.default_rng(1), 1)[0]
    reports = []
    result = es.run_es(mu=4, lam=16, budget=20, initial=flat, approach="es_seeded", out_dir=tmp_path,
                       on_generation=reports.append, champion_dir=None)
    assert np.array_equal(reports[0]["weights"][0], flat) and reports[0]["approach"] == "es_seeded"
    assert result["approach"] == "es_seeded" and result["seeded"]


def test_trained_baseline_reads_the_baseline_controller(tmp_path):
    from racing import imitation
    model = imitation.build_baseline()
    path = paths.champion_path("baseline", 1, champion_dir=tmp_path)
    path.parent.mkdir(parents=True)
    model.save(path)
    flat = es.trained_baseline(1, champion_dir=tmp_path)
    assert flat.shape == (431,) and np.array_equal(flat, nets.to_flat(model))


def test_trained_baseline_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="notebook 02"):
        es.trained_baseline(0, champion_dir=tmp_path)


def test_log_records_strict_successes_and_ties(monkeypatch, tmp_path):
    def fake(w, seeds, level, n_jobs=1):
        fitness = np.zeros(len(w))
        fitness[4:12] = 1.0                      # 8 of 16 offspring beat their parent (all parents score 0)
        return fitness, np.zeros(len(w), dtype=bool), np.zeros(len(w), dtype=bool)
    monkeypatch.setattr(es, "evaluate_population", fake)
    reports = []
    es.run_es(mu=4, lam=16, budget=20, out_dir=tmp_path, on_generation=reports.append, champion_dir=None)
    assert reports[0]["strict_success"] == 0.5 and reports[0]["tie_fraction"] == 0.5
    log = pd.read_csv(tmp_path / "log.csv")
    assert list(log["strict_success"]) == [0.5] and list(log["tie_fraction"]) == [0.5]


def test_es_champion_per_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(es, "evaluate_population",
                        lambda w, seeds, level, n_jobs=1: (np.zeros(len(w)), np.zeros(len(w), bool),
                                                           np.zeros(len(w), bool)))
    monkeypatch.setattr(es, "select_champion", lambda candidates, n_jobs=1: (candidates[0], 0.25))
    result = es.run_es(mu=4, lam=16, budget=20, seed=1, approach="es_seeded",
                       initial=nets.init_population(np.random.default_rng(2), 1)[0],
                       out_dir=tmp_path / "run", champion_dir=tmp_path / "champions")
    assert result["champion_path"] == str(paths.champion_path("es_seeded", 1, champion_dir=tmp_path / "champions"))
    assert paths.champion_path("es_seeded", 1, ".npy", tmp_path / "champions").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_ga.py tests/test_es.py -q`
Expected: FAIL (unexpected keyword arguments `s_max`, `start_level`, `initial`, `approach`; missing
`seeded_parents` and `trained_baseline`; three-value unpacking).

- [ ] **Step 3: Implement `racing/ga.py`**

Replace the imports and `run_ga` with:

```python
"""Genetic algorithm over network weights with finish-rate mutation control (spec §9.2)."""
import time
from pathlib import Path

import numpy as np

from racing import config, nets, paths
from racing.curriculum import LevelTracker, draw_seeds, mutation_strength
from racing.population import (RunLogger, clear_champion, evaluate_population, finish_fractions, make_report,
                               save_champion, select_champion)
```

```python
def run_ga(pop_size=config.GA_POP, kept=config.GA_KEPT, budget=config.EVAL_BUDGET, seed=0, out_dir=None,
           on_generation=None, n_jobs=1, champion_dir=config.CHAMPIONS_DIR, start_level=1, s_max=config.GA_S_MAX):
    if pop_size <= kept:
        raise ValueError(f"population size ({pop_size}) must be larger than the number of kept individuals ({kept})")
    if budget < pop_size:
        raise ValueError(f"budget ({budget}) is smaller than one generation ({pop_size} evaluations)")
    start = time.time()
    rng = np.random.default_rng(seed)
    if champion_dir is not None:
        clear_champion("ga", seed, champion_dir)
    logger = RunLogger(Path(out_dir or paths.run_dir("ga", seed)))
    tracker = LevelTracker(level=start_level)
    weights = nets.init_population(rng, pop_size)
    is_kept = np.zeros(pop_size, dtype=bool)
    strength = float("nan")                             # generation 0 is not produced by mutation
    s_min = s_max / 100
    evals, generation = 0, 0
    while evals + pop_size <= budget:
        level = tracker.level
        seeds = draw_seeds(rng)
        fitness, finished, at_pace = evaluate_population(weights, seeds, level, n_jobs=n_jobs)
        evals += pop_size
        kept_finish, new_finish = finish_fractions(finished, is_kept)
        kept_pace, new_pace = finish_fractions(at_pace, is_kept)
        leveled_up = tracker.update(kept_pace, new_pace)
        report = make_report(approach="ga", generation=generation, level=level, seeds=seeds, weights=weights,
                             fitness=fitness, finished=finished, is_kept=is_kept, kept_finish=kept_finish,
                             new_finish=new_finish, kept_pace=kept_pace, new_pace=new_pace, step_size=strength,
                             evals=evals, elapsed=time.time() - start, leveled_up=leveled_up)
        logger.log(report)
        if on_generation is not None:
            on_generation(report)
        strength = mutation_strength(new_finish, s_max, s_min)
        weights, is_kept = next_generation(rng, weights, fitness, kept, strength)
        generation += 1
    result = {"approach": "ga", "seed": seed, "generations": generation, "evals": evals,
              "level_reached": tracker.level, "start_level": start_level, "pop_size": pop_size, "kept": kept,
              "s_max": s_max}
    if champion_dir is not None:
        champion, val_fitness = select_champion(weights[is_kept], n_jobs=n_jobs)
        result["val_fitness"] = val_fitness
        result["champion_path"] = str(save_champion("ga", seed, champion, champion_dir))
    result["wall_time"] = time.time() - start          # includes the final selection (spec §9.1)
    logger.summary(result)
    return result
```

- [ ] **Step 4: Implement `racing/es.py`**

Replace the imports and `run_es`, and add `seeded_parents` and `trained_baseline`:

```python
"""(μ+λ) evolution strategy over network weights with the 1/5 success rule (spec §9.3), and the seeded ES that
starts from a trained network (spec §9.4)."""
import time
from pathlib import Path

import numpy as np

from racing import config, nets, paths
from racing.curriculum import LevelTracker, draw_seeds
from racing.population import (RunLogger, clear_champion, evaluate_population, finish_fractions, make_report,
                               save_champion, select_champion)
```

```python
def seeded_parents(rng, flat, mu, sigma=config.ES_SIGMA0):
    """The trained network itself, then μ − 1 copies with N(0, σ²) added to every weight (spec §9.4)."""
    flat = np.asarray(flat, dtype=float)
    return np.vstack([flat, flat + rng.normal(0.0, sigma, (mu - 1, len(flat)))])


def trained_baseline(seed, champion_dir=config.CHAMPIONS_DIR):
    """Flat float64 weights of the baseline controller trained with `seed` (spec §9.4)."""
    from racing.backend import keras_cpu
    path = paths.champion_path("baseline", seed, champion_dir=champion_dir)
    if not path.exists():
        raise FileNotFoundError(f"no baseline controller for seed {seed} at {path}; run notebook 02 first")
    return nets.to_flat(keras_cpu().models.load_model(path))


def run_es(mu=config.ES_MU, lam=config.ES_LAMBDA, budget=config.EVAL_BUDGET, seed=0, out_dir=None,
           on_generation=None, n_jobs=1, champion_dir=config.CHAMPIONS_DIR, start_level=1, initial=None,
           approach="es"):
    """(μ+λ)-ES. With `initial`, the flat weights of a trained network, it is the seeded ES (spec §9.4)."""
    if budget < mu + lam:
        raise ValueError(f"budget ({budget}) is smaller than one generation (μ + λ = {mu + lam} evaluations)")
    start = time.time()
    rng = np.random.default_rng(seed)
    if champion_dir is not None:
        clear_champion(approach, seed, champion_dir)
    logger = RunLogger(Path(out_dir or paths.run_dir(approach, seed)))
    tracker = LevelTracker(level=start_level)
    parents = nets.init_population(rng, mu) if initial is None else seeded_parents(rng, initial, mu)
    sigma = config.ES_SIGMA0
    evals, generation = 0, 0
    while evals + mu + lam <= budget:
        level = tracker.level
        seeds = draw_seeds(rng)
        offspring, parent_of = make_offspring(rng, parents, sigma, lam)
        individuals = np.concatenate([parents, offspring])
        is_kept = np.arange(mu + lam) < mu
        fitness, finished, at_pace = evaluate_population(individuals, seeds, level, n_jobs=n_jobs)
        evals += mu + lam
        kept_finish, new_finish = finish_fractions(finished, is_kept)
        kept_pace, new_pace = finish_fractions(at_pace, is_kept)
        leveled_up = tracker.update(kept_pace, new_pace)
        child, parent = fitness[mu:], fitness[:mu][parent_of]
        report = make_report(approach=approach, generation=generation, level=level, seeds=seeds,
                             weights=individuals, fitness=fitness, finished=finished, is_kept=is_kept,
                             kept_finish=kept_finish, new_finish=new_finish, kept_pace=kept_pace,
                             new_pace=new_pace, step_size=sigma, evals=evals, elapsed=time.time() - start,
                             leveled_up=leveled_up, strict_success=float(np.mean(child > parent)),
                             tie_fraction=float(np.mean(child == parent)))
        logger.log(report)
        if on_generation is not None:
            on_generation(report)
        sigma = one_fifth_rule(sigma, success_fraction(child, parent))
        if leveled_up:
            sigma = config.ES_SIGMA0                        # restart the step size on the harder tracks
        parents = select_parents(individuals, fitness, mu)
        generation += 1
    result = {"approach": approach, "seed": seed, "generations": generation, "evals": evals,
              "level_reached": tracker.level, "start_level": start_level, "mu": mu, "lam": lam,
              "seeded": initial is not None}
    if champion_dir is not None:
        champion, val_fitness = select_champion(parents, n_jobs=n_jobs)
        result["val_fitness"] = val_fitness
        result["champion_path"] = str(save_champion(approach, seed, champion, champion_dir))
    result["wall_time"] = time.time() - start          # includes the final selection (spec §9.1)
    logger.summary(result)
    return result
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_ga.py tests/test_es.py tests/test_curriculum.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add racing/ga.py racing/es.py tests/test_ga.py tests/test_es.py
git commit -m "Advance levels at pace, run GA and ES per seed, add the seeded ES

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/ga.py racing/es.py tests/test_ga.py tests/test_es.py
```

---

### Task 4: Imitation: epoch accounting, seeds, retraining, Keras settings

**Files:**
- Modify: `racing/imitation.py`
- Test: `tests/test_imitation.py`

**Interfaces:**
- Consumes: Task 1 (`config.MAX_EPOCHS`, `KT_BAYES_INITIAL`, `HYPERBAND_ITERATIONS`, `STEPS_PER_EXECUTION`,
  `JIT_COMPILE`, `SEARCH_SEED`, `paths`), `nets.n_params(hidden)` (existing).
- Produces:
  - `compile_model(model, learning_rate, steps_per_execution=None, jit_compile=None)` (None → config value)
  - `build_network(units, activation, dropout, learning_rate, steps_per_execution=None, jit_compile=None)`
  - `architecture(values) -> list[int]` (hidden-layer sizes of searched hyperparameter values)
  - `train_baseline(data, out_dir=None, champion_dir=config.CHAMPIONS_DIR, on_epoch=None,
    epochs=config.MAX_EPOCHS, seed=0) -> dict`; default output `paths.run_dir("baseline", seed)`; controller at
    `paths.champion_path("baseline", seed)`.
  - `make_tuner(approach, directory, seed=0, on_trial=None, max_trials=..., max_epochs=..., hyperband_iterations=...,
    bayes_initial=config.KT_BAYES_INITIAL)`; the tuner has `trained_epochs: {trial_id: epochs}`.
  - `trial_records(tuner, max_epochs) -> list[dict]` with keys `trial_id, config_id, val_accuracy, params, epochs,
    last_epoch, at_limit, hyperparameters, trial`.
  - `run_tuner(approach, data, out_dir=None, on_trial=None, seed=config.SEARCH_SEED, max_trials=config.KT_TRIALS,
    max_epochs=config.MAX_EPOCHS, hyperband_iterations=config.HYPERBAND_ITERATIONS, bayes_initial=...,
    export_models=True) -> dict`; default output `paths.search_dir(approach)`; writes `trials.csv` (columns
    `trial_id, config_id, val_accuracy, params, epochs, last_epoch, at_limit, hyperparameters`), `summary.json`
    (keys `approach, seed, trials, configurations, total_epochs, trials_at_limit, max_epochs, wall_time,
    best_val_accuracy, chosen_trial, chosen_val_accuracy, chosen_params, chosen_hyperparameters`) and, with
    `export_models`, `models/<trial_id>.keras` for the most-trained trial of every configuration. **No longer saves
    a final controller** (the `champion_dir` parameter is removed).
  - `retrain(approach, data, seed, search=None, out_dir=None, champion_dir=config.CHAMPIONS_DIR,
    epochs=config.MAX_EPOCHS) -> dict` (keys `approach, seed, epochs, val_accuracy, params, wall_time,
    hyperparameters`); default output `paths.run_dir(approach, seed)`; controller at
    `paths.champion_path(approach, seed)`.
  - Every run first removes its own `summary.json` (and `trials.csv`, `models/` for searches) and its own final
    controller.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_imitation.py` with:

```python
import json

import numpy as np
import pandas as pd
import pytest

from racing import config, imitation, nets, paths


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


def test_baseline_has_431_parameters_and_the_configured_keras_settings():
    model = imitation.build_baseline()
    assert model.count_params() == 431
    assert model.steps_per_execution == config.STEPS_PER_EXECUTION and model.jit_compile == config.JIT_COMPILE


def test_compile_settings_can_be_overridden():
    model = imitation.build_network([8], "relu", 0.0, 1e-3, steps_per_execution=4, jit_compile=False)
    assert model.steps_per_execution == 4 and model.jit_compile is False


def test_parameter_count_follows_from_the_architecture():
    values = {"layers": 2, "units_0": 16, "units_1": 8, "units_2": 64, "activation": "tanh", "dropout": 0.1,
              "learning_rate": 1e-3}
    assert imitation.architecture(values) == [16, 8]
    model = imitation.build_network(imitation.architecture(values), "tanh", 0.1, 1e-3)
    assert nets.n_params(tuple(imitation.architecture(values))) == model.count_params() == 447


def test_load_dataset_round_trip(tmp_path, toy_data):
    np.savez(tmp_path / "d.npz", **toy_data)
    loaded = imitation.load_dataset(tmp_path / "d.npz")
    assert set(loaded) == set(toy_data) and np.array_equal(loaded["y_val"], toy_data["y_val"])


def test_train_baseline_learns_reports_and_saves(toy_data, tmp_path):
    seen = []
    summary = imitation.train_baseline(toy_data, out_dir=tmp_path / "run", champion_dir=tmp_path / "champions",
                                       on_epoch=lambda epoch, model, acc: seen.append((epoch, acc)), epochs=5,
                                       seed=2)
    assert paths.champion_path("baseline", 2, champion_dir=tmp_path / "champions").exists()
    assert len(seen) == summary["epochs"] and summary["val_accuracy"] > 0.2 and summary["seed"] == 2
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["params"] == 431


def test_train_baseline_default_directory_is_per_seed(monkeypatch, toy_data, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    imitation.train_baseline(toy_data, champion_dir=None, epochs=1, seed=1)
    assert (tmp_path / "baseline" / "seed1" / "summary.json").exists()


def test_interrupted_training_leaves_no_stale_results(monkeypatch, toy_data, tmp_path):
    run, champions = tmp_path / "run", tmp_path / "champions"
    run.mkdir()
    (run / "summary.json").write_text("{}")
    stale = paths.champion_path("baseline", 0, champion_dir=champions)
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(imitation, "fit", interrupted)
    with pytest.raises(KeyboardInterrupt):
        imitation.train_baseline(toy_data, out_dir=run, champion_dir=champions, epochs=1)
    assert not (run / "summary.json").exists() and not stale.exists()


def test_train_baseline_is_reproducible_with_a_seed(toy_data, tmp_path):
    runs = [imitation.train_baseline(toy_data, out_dir=tmp_path / f"run{i}", champion_dir=None, epochs=3, seed=3)
            for i in range(2)]
    assert runs[0]["val_accuracy"] == runs[1]["val_accuracy"]


def test_tuner_settings(tmp_path):
    bayes = imitation.make_tuner("kt_bayesian", tmp_path / "b")
    assert bayes.oracle.num_initial_points == config.KT_BAYES_INITIAL
    band = imitation.make_tuner("kt_hyperband", tmp_path / "h")
    assert band.oracle.max_epochs == config.MAX_EPOCHS and band.oracle.hyperband_iterations == 1
    assert band.trained_epochs == {}


def test_retrain_builds_the_chosen_architecture(toy_data, tmp_path):
    search = tmp_path / "search"
    search.mkdir()
    chosen = {"layers": 2, "units_0": 8, "units_1": 4, "activation": "tanh", "dropout": 0.0, "learning_rate": 0.003}
    (search / "summary.json").write_text(json.dumps({"chosen_hyperparameters": chosen}))
    summary = imitation.retrain("kt_random", toy_data, seed=1, search=search, out_dir=tmp_path / "run",
                                champion_dir=tmp_path / "champions", epochs=2)
    assert summary["params"] == nets.n_params((8, 4)) and summary["seed"] == 1 and summary["wall_time"] > 0
    assert paths.champion_path("kt_random", 1, champion_dir=tmp_path / "champions").exists()


def run_small_search(approach, toy_data, tmp_path, seen=None):
    return imitation.run_tuner(approach, toy_data, out_dir=tmp_path / approach,
                               on_trial=None if seen is None else (lambda n, model, acc: seen.append(n)),
                               max_trials=3, max_epochs=3, hyperband_iterations=1)


@pytest.mark.slow
@pytest.mark.parametrize("approach", ["kt_random", "kt_bayesian"])
def test_trial_epochs_are_the_epochs_trained(approach, toy_data, tmp_path):
    seen = []
    summary = run_small_search(approach, toy_data, tmp_path, seen)
    trials = pd.read_csv(tmp_path / approach / "trials.csv", dtype={"trial_id": str, "config_id": str})
    assert summary["trials"] == len(seen) == len(trials) >= 2
    assert list(trials["epochs"]) == [3] * len(trials)      # patience 5 > 3 epochs: no trial stops early
    assert summary["total_epochs"] == 3 * len(trials) and summary["trials_at_limit"] == len(trials)
    assert summary["configurations"] == len(trials) and (trials["trial_id"] == trials["config_id"]).all()
    assert len(list((tmp_path / approach / "models").glob("*.keras"))) == len(trials)


@pytest.mark.slow
def test_hyperband_epochs_and_configurations(toy_data, tmp_path):
    summary = run_small_search("kt_hyperband", toy_data, tmp_path)
    trials = pd.read_csv(tmp_path / "kt_hyperband" / "trials.csv", dtype={"trial_id": str, "config_id": str})
    promoted = trials[trials["trial_id"] != trials["config_id"]]
    assert len(promoted) >= 1
    assert set(promoted["epochs"]) == {2} and set(promoted["last_epoch"]) == {3}   # continues from epoch 1 to 3
    assert summary["configurations"] == trials["config_id"].nunique() < summary["trials"]
    assert summary["total_epochs"] == int(trials["epochs"].sum())
    exported = sorted(p.stem for p in (tmp_path / "kt_hyperband" / "models").glob("*.keras"))
    assert len(exported) == summary["configurations"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_imitation.py -q`
Expected: FAIL (missing `build_network`, `architecture`, `retrain`, `fit`; old paths; no `trained_epochs`).

- [ ] **Step 3: Implement `racing/imitation.py`**

Replace the module with:

```python
"""Imitation-learning controllers (spec §8): a baseline network, three Keras Tuner searches, and retraining of the
selected architectures with each run seed."""
import csv
import json
import time
from pathlib import Path

import numpy as np

from racing import config, nets, paths
from racing.backend import keras_cpu

TUNER_CLASSES = {"kt_random": "RandomSearch", "kt_hyperband": "Hyperband", "kt_bayesian": "BayesianOptimization"}


def load_dataset(path=None):
    path = config.DATASET_PATH if path is None else Path(path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def compile_model(model, learning_rate, steps_per_execution=None, jit_compile=None):
    """Adam and sparse categorical cross-entropy, with the Keras speed settings of config (spec §8.1)."""
    keras = keras_cpu()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  steps_per_execution=config.STEPS_PER_EXECUTION if steps_per_execution is None else steps_per_execution,
                  jit_compile=config.JIT_COMPILE if jit_compile is None else jit_compile)
    model.optimizer.build(model.trainable_variables)  # so checkpoints saved with optimizer state load cleanly
    return model


def build_network(units, activation, dropout, learning_rate, steps_per_execution=None, jit_compile=None):
    """Dense classifier: 10 inputs, one hidden layer per entry of `units` (each followed by dropout if dropout > 0),
    and a 15-way softmax."""
    keras = keras_cpu()
    layers = [keras.Input((config.N_INPUTS,))]
    for n in units:
        layers.append(keras.layers.Dense(n, activation=activation))
        if dropout > 0:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(config.N_ACTIONS, activation="softmax"))
    return compile_model(keras.Sequential(layers), learning_rate, steps_per_execution, jit_compile)


def build_baseline():
    """The 431-parameter baseline: 10 inputs, 16 ReLU units, 15-way softmax."""
    return build_network(list(config.BASELINE_HIDDEN), "relu", 0.0, config.LEARNING_RATE)


def build_hypermodel(hp):
    """Search space of spec Table 2."""
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Choice("dropout", [0.0, 0.1, 0.2])
    units = [hp.Choice(f"units_{i}", [4, 8, 16, 32, 64]) for i in range(hp.Int("layers", 1, 3))]
    return build_network(units, activation, dropout, hp.Float("learning_rate", 1e-4, 1e-2, sampling="log"))


def architecture(values):
    """Hidden-layer sizes of a searched configuration. Values of inactive layers are ignored."""
    return [int(values[f"units_{i}"]) for i in range(int(values["layers"]))]


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


def fit(model, data, epochs, on_epoch=None):
    """Train with early stopping on validation accuracy. Returns the Keras history and the wall time."""
    callbacks = [early_stopping()]
    if on_epoch is not None:
        callbacks.append(epoch_reporter(on_epoch))
    start = time.time()
    history = model.fit(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                        epochs=epochs, batch_size=config.BATCH_SIZE, callbacks=callbacks, verbose=0)
    return history, time.time() - start


def finish_run(approach, seed, model, history, wall_time, out_dir, champion_dir, extra=None):
    """Write the summary of one training run and save its final controller."""
    summary = {"approach": approach, "seed": seed, "epochs": len(history.history["val_accuracy"]),
               "val_accuracy": float(max(history.history["val_accuracy"])), "params": int(model.count_params()),
               "wall_time": wall_time, **(extra or {})}
    if champion_dir is not None:
        path = paths.champion_path(approach, seed, champion_dir=champion_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def start_run(approach, seed, out_dir, champion_dir):
    """Remove this run's earlier summary and final controller, so an interrupted run leaves neither behind."""
    paths.remove(out_dir / "summary.json")
    if champion_dir is not None:
        paths.remove(paths.champion_path(approach, seed, champion_dir=champion_dir))
    out_dir.mkdir(parents=True, exist_ok=True)


def train_baseline(data, out_dir=None, champion_dir=config.CHAMPIONS_DIR, on_epoch=None, epochs=config.MAX_EPOCHS,
                   seed=0):
    out_dir = Path(out_dir or paths.run_dir("baseline", seed))
    start_run("baseline", seed, out_dir, champion_dir)
    keras_cpu().utils.set_random_seed(seed)
    model = build_baseline()
    history, wall_time = fit(model, data, epochs, on_epoch)
    return finish_run("baseline", seed, model, history, wall_time, out_dir, champion_dir)


def make_tuner(approach, directory, seed=0, on_trial=None, max_trials=config.KT_TRIALS,
               max_epochs=config.MAX_EPOCHS, hyperband_iterations=config.HYPERBAND_ITERATIONS,
               bayes_initial=config.KT_BAYES_INITIAL):
    keras_cpu()
    import keras_tuner as kt
    base = getattr(kt, TUNER_CLASSES[approach])

    class ReportingTuner(base):
        def run_trial(self, trial, *args, **kwargs):
            """Record the epochs each trial actually trains (spec §8.2): keras-tuner keeps only its best epoch."""
            histories = super().run_trial(trial, *args, **kwargs)
            self.trained_epochs[trial.trial_id] = sum(len(h.history["val_accuracy"]) for h in histories)
            return histories

        def _configure_tensorboard_dir(self, callbacks, trial, execution=0):
            """No TensorBoard callbacks are used; skip keras-tuner's import of the uninstalled tensorboard."""

        def on_trial_end(self, trial):
            super().on_trial_end(trial)
            if on_trial is not None and trial.score is not None:
                self.completed += 1
                on_trial(self.completed, self.load_model(trial), float(trial.score))

    common = {"hypermodel": build_hypermodel, "objective": "val_accuracy", "seed": seed,
              "directory": str(directory), "project_name": approach, "overwrite": True}
    if approach == "kt_hyperband":
        tuner = ReportingTuner(max_epochs=max_epochs, factor=config.HYPERBAND_FACTOR,
                               hyperband_iterations=hyperband_iterations, **common)
    elif approach == "kt_bayesian":
        tuner = ReportingTuner(max_trials=max_trials, num_initial_points=bayes_initial, **common)
    else:
        tuner = ReportingTuner(max_trials=max_trials, **common)
    tuner.trained_epochs = {}
    tuner.completed = 0
    return tuner


def trial_records(tuner, max_epochs):
    """One record per completed trial. A Hyperband trial that continues a promoted trial belongs to the same
    configuration (config_id is the first trial of the chain) and counts only the epochs it added."""
    trials = tuner.oracle.trials
    records = []
    for trial in trials.values():
        if trial.score is None:
            continue
        values = dict(trial.hyperparameters.values)
        config_id = trial.trial_id
        while "tuner/trial_id" in trials[config_id].hyperparameters.values:
            config_id = trials[config_id].hyperparameters.values["tuner/trial_id"]
        searched = {k: v for k, v in values.items() if not k.startswith("tuner/")}
        epochs = int(tuner.trained_epochs.get(trial.trial_id, 0))
        last_epoch = int(values.get("tuner/initial_epoch", 0)) + epochs
        records.append({"trial_id": trial.trial_id, "config_id": config_id, "val_accuracy": float(trial.score),
                        "params": nets.n_params(tuple(architecture(searched))), "epochs": epochs,
                        "last_epoch": last_epoch, "at_limit": last_epoch >= max_epochs,
                        "hyperparameters": searched, "trial": trial})
    return records


def select_smallest(records, tolerance=config.SIZE_RULE_TOLERANCE):
    """Spec §8.3: fewest parameters among records within `tolerance` of the best validation accuracy."""
    best = max(r["val_accuracy"] for r in records)
    eligible = [r for r in records if r["val_accuracy"] >= best - tolerance]
    return min(eligible, key=lambda r: (r["params"], -r["val_accuracy"]))


def most_trained(records):
    """The record that reached the latest epoch in each configuration (spec §8.3, driving-based selection)."""
    best = {}
    for record in records:
        current = best.get(record["config_id"])
        if current is None or record["last_epoch"] >= current["last_epoch"]:
            best[record["config_id"]] = record
    return list(best.values())


def run_tuner(approach, data, out_dir=None, on_trial=None, seed=config.SEARCH_SEED, max_trials=config.KT_TRIALS,
              max_epochs=config.MAX_EPOCHS, hyperband_iterations=config.HYPERBAND_ITERATIONS,
              bayes_initial=config.KT_BAYES_INITIAL, export_models=True):
    """One Keras Tuner search (spec §8.2). Writes trials.csv, summary.json and, with export_models, the most-trained
    model of every configuration. Final controllers come from retrain()."""
    out_dir = Path(out_dir or paths.search_dir(approach))
    paths.remove(out_dir / "trials.csv", out_dir / "summary.json", out_dir / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    keras_cpu().utils.set_random_seed(seed)
    tuner = make_tuner(approach, out_dir / "tuner", seed, on_trial, max_trials, max_epochs, hyperband_iterations,
                       bayes_initial)
    start = time.time()
    tuner.search(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                 epochs=max_epochs, batch_size=config.BATCH_SIZE, callbacks=[early_stopping()], verbose=0)
    wall_time = time.time() - start
    records = trial_records(tuner, max_epochs)
    chosen = select_smallest(records)
    with open(out_dir / "trials.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "config_id", "val_accuracy", "params", "epochs", "last_epoch", "at_limit",
                         "hyperparameters"])
        for r in records:
            writer.writerow([r["trial_id"], r["config_id"], r["val_accuracy"], r["params"], r["epochs"],
                             r["last_epoch"], r["at_limit"], json.dumps(r["hyperparameters"])])
    if export_models:
        (out_dir / "models").mkdir()
        for record in most_trained(records):
            tuner.load_model(record["trial"]).save(out_dir / "models" / f"{record['trial_id']}.keras")
    summary = {"approach": approach, "seed": seed, "trials": len(records),
               "configurations": len({r["config_id"] for r in records}),
               "total_epochs": int(sum(r["epochs"] for r in records)),
               "trials_at_limit": int(sum(r["at_limit"] for r in records)), "max_epochs": max_epochs,
               "wall_time": wall_time, "best_val_accuracy": max(r["val_accuracy"] for r in records),
               "chosen_trial": chosen["trial_id"], "chosen_val_accuracy": chosen["val_accuracy"],
               "chosen_params": chosen["params"], "chosen_hyperparameters": chosen["hyperparameters"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def retrain(approach, data, seed, search=None, out_dir=None, champion_dir=config.CHAMPIONS_DIR,
            epochs=config.MAX_EPOCHS):
    """Train the architecture selected by a search from scratch with `seed` (spec §8.3)."""
    search = Path(search or paths.search_dir(approach))
    chosen = json.loads((search / "summary.json").read_text())["chosen_hyperparameters"]
    out_dir = Path(out_dir or paths.run_dir(approach, seed))
    start_run(approach, seed, out_dir, champion_dir)
    keras_cpu().utils.set_random_seed(seed)
    model = build_network(architecture(chosen), chosen["activation"], float(chosen["dropout"]),
                          float(chosen["learning_rate"]))
    history, wall_time = fit(model, data, epochs)
    return finish_run(approach, seed, model, history, wall_time, out_dir, champion_dir,
                      extra={"hyperparameters": chosen})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_imitation.py -q` and then `$PY -m pytest tests/test_imitation.py -m slow -q`
Expected: both PASS. If keras-tuner's Hyperband with `max_epochs=3` produces no promoted trial, report the trial
table instead of weakening the assertion.

- [ ] **Step 5: Commit**

```bash
git add racing/imitation.py tests/test_imitation.py
git commit -m "Count trained epochs per trial, run imitation per seed, retrain selected architectures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/imitation.py tests/test_imitation.py
```

---

### Task 5: Parallel, timed dataset collection

**Files:**
- Modify: `racing/expert.py`
- Test: `tests/test_expert.py`

**Interfaces:**
- Consumes: Task 1 (`config.DATASET_INFO_PATH` naming rule).
- Produces: `collect_dataset(path=None, splits=None, n_jobs=1, backend="loky") -> (data, summary)` (unchanged
  return value). It also writes `<path stem>_info.json` next to the dataset (for the default path,
  `config.DATASET_INFO_PATH`) with keys `wall_time`, `n_jobs`, `splits`.

- [ ] **Step 1: Write the failing tests**

Add `import json` to the imports of `tests/test_expert.py` and add:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_expert.py -q -k "parallel_collection or collection_time"`
Expected: FAIL (`unexpected keyword argument 'n_jobs'`; no info file).

- [ ] **Step 3: Implement in `racing/expert.py`**

Add `import json` and `import time` to the imports. Add `_collect_track` above `collect_dataset` and replace
`collect_dataset`:

```python
def _collect_track(seed, level):
    """Rows of one track, with that track's own random number generator (spec §7.3)."""
    return collect_rows(get_track(seed, level), np.random.default_rng([seed, level, config.DATASET_SALT]))


def collect_dataset(path=None, splits=None, n_jobs=1, backend="loky"):
    """Collect the train/val/test rows (spec §7.3) and save them as a compressed .npz file. Tracks are distributed
    across worker processes; each has its own random number generator, so the rows do not depend on n_jobs.
    Returns the arrays and a summary per split (tracks, rows, respawns after crashes) and records the collection
    time in <stem>_info.json next to the dataset."""
    path = Path(config.DATASET_PATH if path is None else path)
    splits = default_splits() if splits is None else splits
    start = time.time()
    data, summary = {}, {}
    for name, items in splits.items():
        if n_jobs == 1:
            parts = [_collect_track(seed, level) for seed, level in items]
        else:
            from joblib import Parallel, delayed
            parts = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_collect_track)(seed, level) for seed, level in items)
        data[f"X_{name}"] = np.concatenate([p[0] for p in parts]).astype(np.float32)
        data[f"y_{name}"] = np.concatenate([p[1] for p in parts]).astype(np.int64)
        summary[name] = {"tracks": len(items), "rows": len(data[f"y_{name}"]),
                         "respawns": int(sum(p[2] for p in parts))}
    wall_time = time.time() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")   # same directory, so the replace is atomic
    try:
        with open(temporary, "wb") as f:
            np.savez_compressed(f, **data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    info = {"wall_time": wall_time, "n_jobs": n_jobs, "splits": summary}
    path.with_name(f"{path.stem}_info.json").write_text(json.dumps(info, indent=2))
    return data, summary
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_expert.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add racing/expert.py tests/test_expert.py
git commit -m "Collect the dataset in parallel worker processes and record the collection time

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/expert.py tests/test_expert.py
```

---

### Task 8: Generation and Keras benchmarks

**Files:**
- Modify: `racing/bench.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: Task 2 (`population.split_tasks`), Task 4 (`imitation.build_network(..., steps_per_execution=,
  jit_compile=)`).
- Produces:
  - `bench.generation_time(pop, jobs, repeats=3) -> float` (best of `repeats`, same task division as
    `evaluate_population`, worst case: every car drives the expert)
  - `bench.keras_epoch_time(data, steps_per_execution, jit_compile, epochs=3) -> float` (seconds per epoch)
  - `bench.main(argv=None)`: `python -m racing.bench` runs both, `python -m racing.bench generation` or
    `... keras` runs one.

- [ ] **Step 1: Write the failing tests**

Replace the first two tests of `tests/test_bench.py` (keep `test_evaluate_population_multiprocess_consistency`
from Task 2) with:

```python
def test_time_full_laps_runs_on_a_small_batch():
    assert bench.time_full_laps(0, 5) > 0.0


def test_generation_time_with_one_and_two_processes():
    assert bench.generation_time(8, 1, repeats=1) > 0.0
    assert bench.generation_time(8, 2, repeats=1) > 0.0


def test_generation_time_uses_the_evaluation_task_division(monkeypatch):
    monkeypatch.setattr(bench, "BACKEND", "threading")            # keeps the patched function in this process
    seen = []
    monkeypatch.setattr(bench, "time_full_laps", lambda seed, n: seen.append((seed, n)) or 0.0)
    bench.generation_time(10, 6, repeats=1)
    expected = [(seed, len(idx)) for _, seed, idx in population.split_tasks(10, bench.SEEDS, 6)]
    assert sorted(seen) == sorted(expected * 2)                    # one untimed pass, one timed repeat


def test_keras_epoch_time_on_toy_data():
    rng = np.random.default_rng(0)
    x = rng.random((600, 10)).astype(np.float32)
    data = {"X_train": x[:500], "y_train": rng.integers(0, 15, 500), "X_val": x[500:],
            "y_val": rng.integers(0, 15, 100)}
    assert bench.keras_epoch_time(data, steps_per_execution=4, jit_compile=False, epochs=1) > 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_bench.py -q`
Expected: FAIL (`generation_time`, `keras_epoch_time` and `BACKEND` do not exist).

- [ ] **Step 3: Implement `racing/bench.py`**

Replace the module with:

```python
"""Benchmarks (spec §9.5): the time of one evolutionary generation for each number of worker processes, and the
time of one Keras training epoch for each choice of steps_per_execution and jit_compile.

Run with:  ~/miniconda3/bin/python -m racing.bench              (both)
           ~/miniconda3/bin/python -m racing.bench generation   (generation time only)
           ~/miniconda3/bin/python -m racing.bench keras        (Keras only; needs results/dataset.npz)
Run on an otherwise idle machine.
"""
import sys
import time

from joblib import Parallel, delayed

from racing import config, expert, population, sim
from racing.tracks import get_track

SEEDS = (0, 1, 2)
JOB_COUNTS = (1, 3, 6, 9, 12, 15)
KERAS_STEPS = (1, 4, 16, 64)
BACKEND = "loky"


def time_full_laps(seed, n):
    """Seconds for n cars to each drive one full level-4 lap on the given training seed."""
    track = get_track(seed, config.MAX_LEVEL)
    start = time.perf_counter()
    sim.simulate(track, expert.expert_policy(track), n)
    return time.perf_counter() - start


def run_tasks(tasks, jobs):
    if jobs == 1:
        return [time_full_laps(seed, n) for seed, n in tasks]
    return Parallel(n_jobs=jobs, backend=BACKEND)(delayed(time_full_laps)(seed, n) for seed, n in tasks)


def generation_time(pop, jobs, repeats=3):
    """Best of `repeats` wall times for one generation of `pop` cars on three level-4 tracks, divided into tasks
    exactly as population.evaluate_population divides them. Every car drives the expert, so every car completes a
    lap: the worst case. An untimed pass first builds the tracks in the worker processes."""
    tasks = [(seed, len(idx)) for _, seed, idx in population.split_tasks(pop, SEEDS, jobs)]
    run_tasks(tasks, jobs)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_tasks(tasks, jobs)
        times.append(time.perf_counter() - start)
    return min(times)


def keras_epoch_time(data, steps_per_execution, jit_compile, epochs=3):
    """Seconds per epoch for the baseline network with the given Keras settings, after one untimed epoch that
    traces and compiles the training function."""
    from racing import imitation
    from racing.backend import keras_cpu
    keras_cpu().utils.set_random_seed(0)
    model = imitation.build_network(list(config.BASELINE_HIDDEN), "relu", 0.0, config.LEARNING_RATE,
                                    steps_per_execution=steps_per_execution, jit_compile=jit_compile)
    args = {"x": data["X_train"], "y": data["y_train"], "validation_data": (data["X_val"], data["y_val"]),
            "batch_size": config.BATCH_SIZE, "verbose": 0}
    model.fit(epochs=1, **args)
    start = time.perf_counter()
    model.fit(epochs=epochs, **args)
    return (time.perf_counter() - start) / epochs


def generation_main():
    for seed in SEEDS:
        get_track(seed, config.MAX_LEVEL)
    print(f"One generation of {config.GA_POP} cars, every car completing a lap on 3 level-4 tracks:")
    for jobs in JOB_COUNTS:
        print(f"  {jobs:2d} process(es): {generation_time(config.GA_POP, jobs):6.2f} s")


def keras_main():
    from racing import imitation
    data = imitation.load_dataset()
    print(f"Baseline training, seconds per epoch ({len(data['y_train'])} rows, batch {config.BATCH_SIZE}):")
    for jit_compile in (True, False):
        for steps in KERAS_STEPS:
            seconds = keras_epoch_time(data, steps, jit_compile)
            print(f"  steps_per_execution {steps:3d}, jit_compile {str(jit_compile):5s}: {seconds:6.3f} s")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or "generation" in argv:
        generation_main()
    if not argv or "keras" in argv:
        keras_main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_bench.py -q`
Expected: PASS. Do not run `python -m racing.bench`; the controller runs it on an idle machine.

- [ ] **Step 5: Commit**

```bash
git add racing/bench.py tests/test_bench.py
git commit -m "Benchmark generations with the evaluation's task division and Keras epochs per setting

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/bench.py tests/test_bench.py
```

---

### Task 6: Evaluation across seeds

**Files:**
- Modify: `racing/evaluate.py` (rewrite)
- Test: `tests/test_evaluate.py` (rewrite)

**Interfaces:**
- Consumes: Task 2 (`nets.champion_seeds`, `nets.load_champion`, `nets.mixed_policy`), Task 1 (`paths`), run
  summaries written by Tasks 3–4 (`wall_time`, `level_reached`), search summaries and `trials.csv` of Task 4,
  `<dataset stem>_info.json` of Task 5.
- Produces:
  - `driving_results(drivers, track_seeds, level=config.MAX_LEVEL, n_jobs=1, backend="loky") -> DataFrame` with
    columns `DRIVE_COLUMNS = [approach, run_seed, track, finished, crashed, lap_time, progress, fitness]`;
    `drivers` is `{(approach, key): DenseNet}` (the key is a run seed, or a trial id in Task 7).
  - `best_seeds(validation) -> {approach: seed}`
  - `inference_time(model, x, repeats=5)`, `single_sample_latency(model, x, calls=1000)`
  - `metrics_per_seed(models, data, runs_dir, latency_calls=1000) -> DataFrame`
  - `mean_std(values, digits) -> str`
  - `comparison_table(metrics)`, `driving_table(per_track, metrics)`, `search_cost_table(runs_dir)`,
    `dataset_table(data_path)`, `figures(runs_dir, out_dir)`, `tuner_trials(runs_dir)`, `to_markdown(frame)`
  - `main(champions_dir=None, runs_dir=None, data_path=None, out_dir=None, test_seeds=config.TEST_SEEDS,
    val_seeds=config.VAL_SEEDS, n_jobs=1, latency_calls=1000) -> dict` with keys `comparison, driving,
    search_cost, dataset, per_track, validation, metrics, best_seeds, figures`; writes
    `<champions>/best_seeds.json` and `<out>/tables/*.csv|md`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_evaluate.py` with:

```python
import json

import numpy as np
import pandas as pd
import pytest

from racing import config, evaluate, nets, paths, population


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(evaluate, "get_track", lambda seed, level: circle_track)
    return circle_track


def drivers(n=3, seed=0):
    flats = nets.init_population(np.random.default_rng(seed), n)
    return {("ga", k): nets.from_flat(flats[k]) for k in range(n)}


def test_driving_results_columns_and_rows(fake_tracks):
    frame = evaluate.driving_results(drivers(), [1000, 1001])
    assert list(frame.columns) == evaluate.DRIVE_COLUMNS and len(frame) == 6
    assert list(frame["track"]) == [1000] * 3 + [1001] * 3 and list(frame["run_seed"][:3]) == [0, 1, 2]


def test_driving_results_in_parallel_match_serial(fake_tracks):
    serial = evaluate.driving_results(drivers(), [1000, 1001, 1002])
    threaded = evaluate.driving_results(drivers(), [1000, 1001, 1002], n_jobs=3, backend="threading")
    pd.testing.assert_frame_equal(serial, threaded)


def test_driving_together_matches_driving_alone(fake_tracks):
    together = evaluate.driving_results(drivers(), [1000])
    alone = evaluate.driving_results({("ga", 1): drivers()[("ga", 1)]}, [1000])
    assert together.iloc[1].equals(alone.iloc[0])


def test_best_seeds_highest_mean_validation_fitness_lowest_seed_on_ties():
    rows = [("ga", 0, 0.5), ("ga", 0, 0.7), ("ga", 1, 0.9), ("ga", 1, 0.9), ("es", 0, 1.0), ("es", 2, 1.0)]
    frame = pd.DataFrame([{"approach": a, "run_seed": s, "fitness": f} for a, s, f in rows])
    assert evaluate.best_seeds(frame) == {"es": 0, "ga": 1}


def test_inference_warm_up_matches_the_timed_calls():
    model = nets.to_keras(nets.init_population(np.random.default_rng(1), 1)[0])
    calls, predict = [], model.predict

    def recording_predict(x, **kwargs):
        calls.append((len(x), kwargs.get("batch_size")))
        return predict(x, **kwargs)

    model.predict = recording_predict
    x = np.random.default_rng(2).random((500, config.N_INPUTS)).astype(np.float32)
    assert 0 < evaluate.inference_time(model, x, repeats=2) < 0.01
    assert calls == [(500, 500)] * 3


def test_single_sample_latency_is_positive():
    model = nets.to_keras(nets.init_population(np.random.default_rng(3), 1)[0])
    x = np.random.default_rng(4).random((50, config.N_INPUTS)).astype(np.float32)
    assert 0 < evaluate.single_sample_latency(model, x, calls=20) < 1.0


def test_mean_std_formats():
    assert evaluate.mean_std([0.5, 0.7, 0.6], 2) == "0.60 ± 0.10"
    assert evaluate.mean_std([431, 431, 431], 0) == "431"
    assert evaluate.mean_std([1.24], 1) == "1.2"
    assert evaluate.mean_std([float("nan")], 1) == "–" and evaluate.mean_std([], 1) == "–"


def metrics_frame():
    rows = []
    for seed, acc in ((0, 0.60), (1, 0.62), (2, 0.64)):
        rows.append({"approach": "kt_random", "run_seed": seed, "accuracy": acc, "params": 447, "batch_us": 2.0,
                     "single_ms": 0.5, "training_s": 30.0 + seed, "search_s": 900.0, "level_reached": np.nan})
    for seed, level in ((0, 4), (1, 3)):
        rows.append({"approach": "ga", "run_seed": seed, "accuracy": 0.2, "params": 431, "batch_us": 2.0,
                     "single_ms": 0.5, "training_s": 600.0, "search_s": np.nan, "level_reached": level})
    return pd.DataFrame(rows)


def test_comparison_table():
    table = evaluate.comparison_table(metrics_frame())
    assert list(table["Method"]) == ["KT Random", "GA"] and list(table["Seeds"]) == [3, 2]
    assert table.loc[0, "Test accuracy"] == "0.6200 ± 0.0200" and table.loc[0, "Parameters"] == "447"
    assert table.loc[0, "Search time (s)"] == "900.0" and table.loc[1, "Search time (s)"] == "–"


def test_driving_table_aggregates_seeds():
    per_track = pd.DataFrame([
        {"approach": "ga", "run_seed": 0, "track": 1000, "finished": True, "crashed": False, "lap_time": 50.0},
        {"approach": "ga", "run_seed": 0, "track": 1001, "finished": False, "crashed": True, "lap_time": np.nan},
        {"approach": "ga", "run_seed": 1, "track": 1000, "finished": True, "crashed": False, "lap_time": 54.0},
        {"approach": "ga", "run_seed": 1, "track": 1001, "finished": True, "crashed": False, "lap_time": 60.0}])
    table = evaluate.driving_table(per_track, metrics_frame())
    row = table.iloc[0]
    assert row["Method"] == "GA" and row["Tracks completed (of 2)"] == "1.5 ± 0.7"
    assert row["Mean lap time (s)"] == "53.50 ± 4.95" and row["Level reached (per seed)"] == "4, 3"


def test_search_cost_and_dataset_tables(tmp_path):
    search = paths.search_dir("kt_hyperband", tmp_path)
    search.mkdir(parents=True)
    (search / "summary.json").write_text(json.dumps({"trials": 254, "configurations": 172, "total_epochs": 3000,
                                                     "trials_at_limit": 9, "wall_time": 1234.5}))
    table = evaluate.search_cost_table(tmp_path)
    assert list(table["Method"]) == ["KT Hyperband"] and table.loc[0, "Configurations"] == 172
    (tmp_path / "dataset_info.json").write_text(json.dumps({"wall_time": 61.0, "n_jobs": 8, "splits": {
        "train": {"tracks": 121, "rows": 90000, "respawns": 5}}}))
    data = evaluate.dataset_table(tmp_path / "dataset.npz")
    assert data.loc[0, "Collection time (s)"] == 61.0 and data.loc[0, "Rows"] == 90000


def test_figures_per_seed(tmp_path):
    runs = tmp_path / "runs"
    for seed in (0, 1):
        folder = paths.run_dir("ga", seed, runs)
        folder.mkdir(parents=True)
        pd.DataFrame({"generation": [0, 1, 2], "level": [1, 1, 2], "evals": [10, 20, 30], "best": [0.1, 0.5, 1.2],
                      "mean": [0.05, 0.2, 0.6], "step_size": [np.nan, 0.25, 0.05],
                      "leveled_up": [False, True, False]}).to_csv(folder / "log.csv", index=False)
    written = evaluate.figures(runs, tmp_path / "figures")
    assert [p.name for p in written] == ["fitness_curves.png", "step_sizes.png"]


def test_chosen_trial_star_with_zero_padded_ids(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    runs = tmp_path / "runs"
    for approach, width, chosen in (("kt_random", 2, "07"), ("kt_hyperband", 4, "0003")):
        search = paths.search_dir(approach, runs)
        search.mkdir(parents=True)
        ids = [f"{i:0{width}d}" for i in range(12)]
        pd.DataFrame({"trial_id": ids, "config_id": ids, "val_accuracy": np.linspace(0.5, 0.9, 12),
                      "params": np.arange(12) * 100 + 50, "epochs": 5}).to_csv(search / "trials.csv", index=False)
        (search / "summary.json").write_text(json.dumps({"chosen_trial": chosen}))
    trials = evaluate.tuner_trials(runs)
    assert list(trials["kt_random"][1]["params"]) == [750] and list(trials["kt_hyperband"][1]["params"]) == [350]
    shown, close = [], plt.close
    monkeypatch.setattr(plt, "close", shown.append)
    evaluate.figures(runs, tmp_path / "figures")
    scatter = shown[-1].axes[0].collections
    for fig in shown:
        close(fig)
    assert [len(c.get_offsets()) for c in scatter] == [12, 1, 12, 1]


def test_main_without_champions_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate.main(champions_dir=tmp_path, runs_dir=tmp_path, data_path=tmp_path / "none.npz", out_dir=tmp_path)


def test_main_with_partial_results(fake_tracks, tmp_path):
    champions, runs = tmp_path / "champions", tmp_path / "runs"
    rng = np.random.default_rng(5)
    for approach, seed in (("ga", 0), ("ga", 2), ("es", 1)):
        population.save_champion(approach, seed, nets.init_population(rng, 1)[0], champions)
    folder = paths.run_dir("ga", 0, runs)
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"wall_time": 12.0, "level_reached": 2}))
    np.savez(tmp_path / "dataset.npz", X_test=rng.random((40, config.N_INPUTS)).astype(np.float32),
             y_test=rng.integers(0, 15, 40))
    results = evaluate.main(champions_dir=champions, runs_dir=runs, data_path=tmp_path / "dataset.npz",
                            out_dir=tmp_path / "out", test_seeds=[1000], val_seeds=[80], latency_calls=5)
    assert list(results["comparison"]["Method"]) == ["ES", "GA"]
    assert json.loads((champions / "best_seeds.json").read_text()).keys() == {"ga", "es"}
    for name in ("comparison", "driving", "search_cost", "dataset"):
        assert (tmp_path / "out" / "tables" / f"{name}.md").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_evaluate.py -q`
Expected: FAIL (missing `DRIVE_COLUMNS`, `best_seeds`, `single_sample_latency`, `mean_std`, …).

- [ ] **Step 3: Rewrite `racing/evaluate.py`**

```python
"""Final evaluation (spec §11): driving on held-out tracks, test accuracy, size, speed, tables and figures.
Each method has one final controller per run seed; tables report the mean and standard deviation across seeds."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from racing import config, nets, paths, sim
from racing.tracks import get_track

DRIVE_COLUMNS = ["approach", "run_seed", "track", "finished", "crashed", "lap_time", "progress", "fitness"]


def _drive_track(drivers, track_seed, level):
    """One lap for every controller on one track, simulated together (cars never interact)."""
    track = get_track(track_seed, level)
    keys = list(drivers)
    result = sim.simulate(track, nets.mixed_policy([drivers[k] for k in keys]), len(keys))
    rows = []
    for i, (approach, run_seed) in enumerate(keys):
        rows.append({"approach": approach, "run_seed": run_seed, "track": track_seed,
                     "finished": bool(result.finished[i]), "crashed": bool(result.crashed[i]),
                     "lap_time": float(result.finish_time[i]), "progress": float(result.progress[i]),
                     "fitness": float(result.fitness[i])})
    return rows


def driving_results(drivers, track_seeds, level=config.MAX_LEVEL, n_jobs=1, backend="loky"):
    """One lap per controller per track. drivers: {(approach, run seed or trial id): DenseNet}. The tracks are
    distributed across worker processes; the result does not depend on n_jobs."""
    track_seeds = list(track_seeds)
    if not drivers:
        return pd.DataFrame(columns=DRIVE_COLUMNS)
    if n_jobs == 1 or len(track_seeds) == 1:
        parts = [_drive_track(drivers, seed, level) for seed in track_seeds]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_drive_track)(drivers, seed, level) for seed in track_seeds)
    return pd.DataFrame([row for part in parts for row in part], columns=DRIVE_COLUMNS)


def best_seeds(validation):
    """For each approach, the run seed with the highest mean validation fitness; the lowest seed wins ties."""
    best = {}
    for approach in config.APPROACHES:
        sub = validation[validation["approach"] == approach]
        if sub.empty:
            continue
        means = sub.groupby("run_seed")["fitness"].mean().sort_index()
        best[approach] = int(means.index[int(np.argmax(means.to_numpy()))])
    return best


def accuracy(model, x, y):
    predicted = np.argmax(model.predict(x, batch_size=4096, verbose=0), axis=1)
    return float(np.mean(predicted == y))


def inference_time(model, x, repeats=5):
    """Seconds per sample for Keras predict on up to 10,000 rows in one batch, averaged over `repeats` runs. The
    warm-up is the same call as the timed ones, so no tracing or compilation is timed (spec §11)."""
    x = x[:10000]
    model.predict(x, batch_size=len(x), verbose=0)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(x, batch_size=len(x), verbose=0)
        times.append(time.perf_counter() - start)
    return float(np.mean(times) / len(x))


def single_sample_latency(model, x, calls=1000):
    """Median seconds for one call of the model on a single row (spec §11)."""
    rows = [x[i % len(x)][None, :] for i in range(calls)]
    model(rows[0], training=False)
    times = []
    for row in rows:
        start = time.perf_counter()
        model(row, training=False)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def metrics_per_seed(models, data, runs_dir, latency_calls=1000):
    """One row per final controller: test accuracy, size, speed, training cost and curriculum level reached."""
    rows = []
    for (approach, run_seed), model in models.items():
        summary = read_json(paths.run_dir(approach, run_seed, runs_dir) / "summary.json")
        search = float("nan")
        if approach in config.TUNER_APPROACHES:
            search = read_json(paths.search_dir(approach, runs_dir) / "summary.json").get("wall_time", float("nan"))
        rows.append({"approach": approach, "run_seed": run_seed,
                     "accuracy": accuracy(model, data["X_test"], data["y_test"]),
                     "params": int(model.count_params()),
                     "batch_us": inference_time(model, data["X_test"]) * 1e6,
                     "single_ms": single_sample_latency(model, data["X_test"], latency_calls) * 1e3,
                     "training_s": float(summary.get("wall_time", float("nan"))), "search_s": float(search),
                     "level_reached": float(summary.get("level_reached", float("nan")))})
    return pd.DataFrame(rows, columns=["approach", "run_seed", "accuracy", "params", "batch_us", "single_ms",
                                       "training_s", "search_s", "level_reached"])


def mean_std(values, digits):
    """'mean ± std' across seeds (sample standard deviation); the value alone if there is one value or all are
    equal; '–' if there is none."""
    values = [float(v) for v in values if not np.isnan(float(v))]
    if not values:
        return "–"
    if len(values) == 1 or all(v == values[0] for v in values):
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def comparison_table(metrics):
    """Spec Table 5, required comparison."""
    rows = []
    for approach in config.APPROACHES:
        sub = metrics[metrics["approach"] == approach]
        if sub.empty:
            continue
        rows.append({"Method": config.LABELS[approach], "Seeds": len(sub),
                     "Test accuracy": mean_std(sub["accuracy"], 4),
                     "Parameters": mean_std(sub["params"], 0),
                     "Batch inference (µs/sample)": mean_std(sub["batch_us"], 3),
                     "Single-sample latency (ms)": mean_std(sub["single_ms"], 3),
                     "Search time (s)": mean_std(sub["search_s"].iloc[:1], 1),
                     "Training time (s)": mean_std(sub["training_s"], 1)})
    return pd.DataFrame(rows)


def driving_table(per_track, metrics):
    """Spec Table 5, driving performance: per seed, then mean ± std across seeds."""
    rows = []
    n_tracks = per_track["track"].nunique()
    for approach in config.APPROACHES:
        sub = per_track[per_track["approach"] == approach]
        if sub.empty:
            continue
        completed, lap_time, crashes = [], [], []
        for _, runs in sub.groupby("run_seed"):
            done = runs[runs["finished"].astype(bool)]
            completed.append(float(runs["finished"].sum()))
            lap_time.append(float(done["lap_time"].mean()) if len(done) else float("nan"))
            crashes.append(float(runs["crashed"].sum()))
        level = "n/a"
        if approach in config.EVOLUTION_APPROACHES:
            levels = metrics[metrics["approach"] == approach]["level_reached"].dropna()
            level = ", ".join(str(int(v)) for v in levels) if len(levels) else "n/a"
        rows.append({"Method": config.LABELS[approach], f"Tracks completed (of {n_tracks})": mean_std(completed, 1),
                     "Mean lap time (s)": mean_std(lap_time, 2), "Crashes": mean_std(crashes, 1),
                     "Level reached (per seed)": level})
    return pd.DataFrame(rows)


def search_cost_table(runs_dir):
    """Spec Table 5, search cost of each Keras Tuner method."""
    rows = []
    for approach in config.TUNER_APPROACHES:
        summary = read_json(paths.search_dir(approach, runs_dir) / "summary.json")
        if not summary:
            continue
        rows.append({"Method": config.LABELS[approach], "Trials": summary["trials"],
                     "Configurations": summary["configurations"], "Total epochs": summary["total_epochs"],
                     "Trials at the epoch limit": summary["trials_at_limit"],
                     "Search time (s)": round(summary["wall_time"], 1)})
    return pd.DataFrame(rows)


def dataset_table(data_path):
    """Dataset size and collection time, a cost shared by the imitation methods and the seeded ES (spec §7.3)."""
    data_path = Path(data_path)
    info = read_json(data_path.with_name(f"{data_path.stem}_info.json"))
    rows = []
    for split, values in info.get("splits", {}).items():
        rows.append({"Split": split, "Tracks": values["tracks"], "Rows": values["rows"],
                     "Respawns": values["respawns"], "Collection time (s)": info["wall_time"],
                     "Worker processes": info["n_jobs"]})
    return pd.DataFrame(rows)


def to_markdown(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def tuner_trials(runs_dir):
    """Each search's trials table and its chosen trial's row. Trial ids stay strings: keras-tuner zero-pads them."""
    trials = {}
    for approach in config.TUNER_APPROACHES:
        search = paths.search_dir(approach, runs_dir)
        if (search / "trials.csv").exists():
            table = pd.read_csv(search / "trials.csv", dtype={"trial_id": str, "config_id": str})
            chosen = str(read_json(search / "summary.json").get("chosen_trial"))
            trials[approach] = (table, table[table["trial_id"] == chosen])
    return trials


def evolution_logs(runs_dir):
    """{approach: {run seed: log}} for the evolutionary runs whose logs exist."""
    logs = {}
    for approach in config.EVOLUTION_APPROACHES:
        found = {}
        for path in sorted((Path(runs_dir) / approach).glob("seed*/log.csv")):
            found[int(path.parent.name[len("seed"):])] = pd.read_csv(path)
        if found:
            logs[approach] = found
    return logs


def figures(runs_dir, out_dir):
    """Fitness curves and step sizes (one line per seed) and Keras Tuner trials; only figures with data are written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    logs = evolution_logs(runs_dir)
    if logs:
        fig, ax = plt.subplots(figsize=(7, 4))
        for approach, per_seed in logs.items():
            color = config.COLORS[approach]
            for k, (seed, log) in enumerate(per_seed.items()):
                ax.plot(log["evals"], log["best"], color=color, linewidth=1.0, alpha=0.85,
                        label=f"{config.LABELS[approach]} best" if k == 0 else None)
                for evals in log.loc[log["leveled_up"].astype(bool), "evals"]:
                    ax.axvline(evals, color=color, alpha=0.25, linewidth=0.7)
        ax.set_xlabel("Individual evaluations")
        ax.set_ylabel("Best fitness")
        ax.legend(frameon=False)
        fig.tight_layout()
        written.append(out_dir / "fitness_curves.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        for approach, per_seed in logs.items():
            name = "GA mutation strength" if approach == "ga" else f"{config.LABELS[approach]} step size σ"
            for k, (seed, log) in enumerate(per_seed.items()):
                ax.plot(log["generation"], log["step_size"], color=config.COLORS[approach], linewidth=1.0,
                        label=name if k == 0 else None)
        ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.legend(frameon=False)
        fig.tight_layout()
        written.append(out_dir / "step_sizes.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

    trials = tuner_trials(runs_dir)
    if trials:
        fig, ax = plt.subplots(figsize=(7, 4))
        for approach, (table, pick) in trials.items():
            color = config.COLORS[approach]
            ax.scatter(table["params"], table["val_accuracy"], color=color, s=18, alpha=0.7,
                       label=config.LABELS[approach])
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


def main(champions_dir=None, runs_dir=None, data_path=None, out_dir=None, test_seeds=config.TEST_SEEDS,
         val_seeds=config.VAL_SEEDS, n_jobs=1, latency_calls=1000):
    champions_dir = Path(champions_dir or config.CHAMPIONS_DIR)
    runs_dir = Path(runs_dir or config.RUNS_DIR)
    data_path = Path(data_path or config.DATASET_PATH)
    out_dir = Path(out_dir or config.RESULTS_DIR)
    found = nets.champion_seeds(champions_dir)
    if not found:
        raise FileNotFoundError(f"no trained controllers found in {champions_dir}")
    models, drivers = {}, {}
    for approach, seeds in found.items():
        for seed in seeds:
            models[(approach, seed)], drivers[(approach, seed)] = nets.load_champion(approach, seed, champions_dir)
    per_track = driving_results(drivers, test_seeds, n_jobs=n_jobs)
    validation = driving_results(drivers, val_seeds, n_jobs=n_jobs)
    best = best_seeds(validation)
    paths.best_seeds_path(champions_dir).write_text(json.dumps(best, indent=2))
    with np.load(data_path) as d:
        data = {key: d[key] for key in d.files}
    metrics = metrics_per_seed(models, data, runs_dir, latency_calls)     # timed after all parallel work
    tables = {"comparison": comparison_table(metrics), "driving": driving_table(per_track, metrics),
              "search_cost": search_cost_table(runs_dir), "dataset": dataset_table(data_path)}
    folder = out_dir / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    per_track.to_csv(folder / "driving_per_track.csv", index=False)
    validation.to_csv(folder / "validation_per_track.csv", index=False)
    metrics.to_csv(folder / "metrics_per_seed.csv", index=False)
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
        (folder / f"{name}.md").write_text(to_markdown(frame))
    return {**tables, "per_track": per_track, "validation": validation, "metrics": metrics, "best_seeds": best,
            "figures": figures(runs_dir, out_dir / "figures")}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_evaluate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add racing/evaluate.py tests/test_evaluate.py
git commit -m "Evaluate every seed's controller, record best seeds, fix inference timing, aggregate tables

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/evaluate.py tests/test_evaluate.py
```

---

### Task 7: Driving-selected controllers (secondary analysis)

**Files:**
- Modify: `racing/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: Task 6 (`driving_results`, `read_json`, `main`), Task 4 (`<search>/trials.csv` with `trial_id`,
  `config_id`, `params`, `val_accuracy`; `<search>/models/<trial_id>.keras`).
- Produces: `driving_selected_table(runs_dir=None, val_seeds=config.VAL_SEEDS, test_seeds=config.TEST_SEEDS,
  n_jobs=1) -> DataFrame`; `main()` adds the key `driving_selected` and writes `tables/driving_selected.csv|md`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluate.py`:

```python
def export_search(runs, approach, flats):
    """A search directory with one exported model per trial, as run_tuner writes it."""
    search = paths.search_dir(approach, runs)
    (search / "models").mkdir(parents=True)
    ids = [f"{i:02d}" for i in range(len(flats))]
    for trial_id, flat in zip(ids, flats):
        nets.to_keras(flat).save(search / "models" / f"{trial_id}.keras")
    pd.DataFrame({"trial_id": ids, "config_id": ids, "val_accuracy": np.linspace(0.5, 0.6, len(ids)),
                  "params": 431, "epochs": 3}).to_csv(search / "trials.csv", index=False)
    return ids


def test_driving_selected_picks_the_best_validation_driver(fake_tracks, tmp_path):
    flats = nets.init_population(np.random.default_rng(8), 4)
    ids = export_search(tmp_path, "kt_random", flats)
    table = evaluate.driving_selected_table(tmp_path, val_seeds=[80, 81], test_seeds=[1000])
    keras_drivers = {("kt_random", i): nets.from_keras(nets.to_keras(f)) for i, f in zip(ids, flats)}
    validation = evaluate.driving_results(keras_drivers, [80, 81])
    expected = validation.groupby("run_seed")["fitness"].mean().idxmax()
    assert list(table["Method"]) == ["KT Random"] and table.loc[0, "Trial"] == expected
    assert table.loc[0, "Parameters"] == 431


def test_driving_selected_without_exported_models(tmp_path):
    assert evaluate.driving_selected_table(tmp_path, val_seeds=[80], test_seeds=[1000]).empty
```

Extend `test_main_with_partial_results`: add `"driving_selected"` to the tuple of table names checked at the end.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_evaluate.py -q`
Expected: FAIL (`driving_selected_table` does not exist; no `driving_selected.md`).

- [ ] **Step 3: Implement**

Add to `racing/evaluate.py` (after `search_cost_table`):

```python
def driving_selected_table(runs_dir=None, val_seeds=config.VAL_SEEDS, test_seeds=config.TEST_SEEDS, n_jobs=1):
    """Secondary analysis (spec §8.3): for each Keras Tuner method, the exported model (the most-trained of each
    configuration) with the highest mean validation fitness, and its driving on the test tracks."""
    from racing.backend import keras_cpu
    keras = keras_cpu()
    rows = []
    for approach in config.TUNER_APPROACHES:
        search = paths.search_dir(approach, runs_dir)
        exported = sorted((search / "models").glob("*.keras"))
        if not exported or not (search / "trials.csv").exists():
            continue
        trials = pd.read_csv(search / "trials.csv", dtype={"trial_id": str, "config_id": str})
        drivers = {(approach, path.stem): nets.from_keras(keras.models.load_model(path)) for path in exported}
        validation = driving_results(drivers, val_seeds, n_jobs=n_jobs)
        means = validation.groupby("run_seed")["fitness"].mean()
        trial_id = str(means.idxmax())
        test = driving_results({(approach, trial_id): drivers[(approach, trial_id)]}, test_seeds, n_jobs=n_jobs)
        record = trials[trials["trial_id"] == trial_id].iloc[0]
        done = test[test["finished"].astype(bool)]
        rows.append({"Method": config.LABELS[approach], "Trial": trial_id, "Parameters": int(record["params"]),
                     "Validation accuracy": round(float(record["val_accuracy"]), 4),
                     "Mean validation fitness": round(float(means.max()), 3),
                     f"Tracks completed (of {len(list(test_seeds))})": int(test["finished"].sum()),
                     "Mean lap time (s)": round(float(done["lap_time"].mean()), 2) if len(done) else float("nan"),
                     "Crashes": int(test["crashed"].sum())})
    return pd.DataFrame(rows)
```

In `main`, compute it right after `best_seeds` is written (before the timed metrics, so no parallel work runs
during timing) and add it to `tables`:

```python
    selected = driving_selected_table(runs_dir, val_seeds, test_seeds, n_jobs=n_jobs)
```

```python
    tables = {"comparison": comparison_table(metrics), "driving": driving_table(per_track, metrics),
              "search_cost": search_cost_table(runs_dir), "driving_selected": selected,
              "dataset": dataset_table(data_path)}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_evaluate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add racing/evaluate.py tests/test_evaluate.py
git commit -m "Report the driving-selected controller of each Keras Tuner search

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/evaluate.py tests/test_evaluate.py
```

---

### Task 9: Viewer: training view

**Files:**
- Modify: `racing/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: Task 3 (`ga.run_ga(seed=, start_level=)`, `es.run_es(seed=, start_level=, initial=, approach=)`,
  `es.trained_baseline(seed)`), Task 4 (`imitation.train_baseline(seed=)`, `imitation.run_tuner(...,
  export_models=False)`), Task 2 (`nets.mixed_policy`), Task 1 (`config.PLAYBACK_BASE`, `DIFFICULTIES`,
  `RUN_SEEDS`, `EVOLUTION_APPROACHES`, `paths.champion_path`).
- Produces (Task 10 builds on these):
  - `trainer_main(approach, size, seed, level, reports, running)`; `trainer_process(...)` (same arguments, the
    process entry point; makes the process lead its own process group)
  - `App(screen, background_tracks=True)`; attributes `run_seed`, `difficulty`, `tracks`, `building`
  - `App.request_track(seed, level) -> Track | None`, `App.shown_track()`, `App.map_text()`, `App.stop()`,
    `App.drain_reports()`, `App.change_run_seed(direction)`, `App.set_difficulty(level)`,
    `App.draw_difficulty(x, y, w, level, action, enabled) -> y`, `App.start_playback(report, track)`
  - `BUTTON_ACTIVE` is a dark neutral `(55, 62, 78)`.
  - The race view is left as it is; Task 10 rewrites it.

- [ ] **Step 1: Write the failing tests**

In `tests/test_app.py`:
- add `import os`, `import signal`, `import time` and `from racing import paths` to the imports;
- change the `window` fixture to build `viewer.App(screen, background_tracks=False)` and to make process-group
  kills harmless:

```python
@pytest.fixture
def window(monkeypatch, circle_track):
    monkeypatch.setattr(viewer, "get_track", lambda seed, level: circle_track)
    monkeypatch.setattr(viewer.os, "killpg", lambda pgid, sig: None)     # tests never signal real processes
    pygame.init()
    screen = pygame.display.set_mode(config.WINDOW_SIZE)
    instance = viewer.App(screen, background_tracks=False)
    yield instance
    instance.process = None
    instance.stop_training()
    pygame.quit()
```

- in `ga_report`, add `"kept_pace": float("nan"), "new_pace": 0.0` to the dict;
- update every `viewer.trainer_main(...)` call to the new signature, e.g.
  `viewer.trainer_main("ga", 30, 0, 1, reports, running)` (also the `args` tuple of the thread in
  `test_pausing_still_delivers_the_finished_generation`), and `run_baseline_trainer` to
  `viewer.trainer_main("baseline", 0, 0, 1, queue.Queue(), running)`; in `fake_imitation`, make the
  `train_baseline` stub accept keywords (`lambda data, **kwargs: trained.append(data)`, unchanged);
- in `test_trainer_main_uses_app_runs_dir_and_never_saves_champions`, expect
  `calls["ga"]["out_dir"] == tmp_path / "ga" / "seed0"` and `calls["es"]["out_dir"] == tmp_path / "es" / "seed0"`;
- in `test_pause_and_resume_toggle_the_running_event`, give the `Alive` stub `pid = 0` and
  `def join(self, timeout=None): pass`;
- add:

```python
def test_trainer_main_passes_seed_level_and_seeded_start(monkeypatch, tmp_path):
    from racing import es, ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    calls = {}
    monkeypatch.setattr(ga, "run_ga", lambda **kwargs: calls.setdefault("ga", kwargs))
    monkeypatch.setattr(es, "run_es", lambda **kwargs: calls.setdefault("es", kwargs))
    monkeypatch.setattr(es, "trained_baseline", lambda seed: f"baseline seed {seed}")
    reports, running = queue.Queue(), threading.Event()
    running.set()
    viewer.trainer_main("ga", 30, 2, 3, reports, running)
    viewer.trainer_main("es_seeded", 12, 1, 2, reports, running)
    assert calls["ga"]["seed"] == 2 and calls["ga"]["start_level"] == 3
    assert calls["es"]["approach"] == "es_seeded" and calls["es"]["initial"] == "baseline seed 1"
    assert calls["es"]["out_dir"] == tmp_path / "es_seeded" / "seed1" and calls["es"]["start_level"] == 2


def test_training_process_is_not_a_daemon(window, monkeypatch):
    created = {}

    class FakeProcess:
        def __init__(self, target, args, daemon):
            created.update(target=target, args=args, daemon=daemon)
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
    window.change_run_seed(1)
    window.set_difficulty(2)
    window.start_training()
    assert created["daemon"] is False and created["target"] is viewer.trainer_process
    assert created["args"][:4] == ("ga", config.GA_POP, config.RUN_SEEDS[1], 2)


def test_stop_ends_the_trainer_process_group(window, monkeypatch):
    killed = []
    monkeypatch.setattr(viewer.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))

    class Alive:
        pid = 4321

        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    window.process, window.reports = Alive(), FakeQueue()
    window.stop()
    assert killed == [(4321, signal.SIGTERM)] and window.process is None and "Stopped" in window.status


def test_settings_are_locked_while_training(window):
    class Alive:
        pid = 0

        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    window.process = Alive()
    window.select_approach("es")
    window.change_run_seed(1)
    window.set_difficulty(3)
    window.change_population(1)
    assert window.approach == "ga" and window.run_seed == config.RUN_SEEDS[0] and window.difficulty == 1
    assert window.population["ga"] == config.GA_POP


def test_seeded_es_without_baseline_shows_a_message(window, tmp_path):
    window.champions_dir = tmp_path
    window.select_approach("es_seeded")
    window.start_training()
    assert window.process is None and "notebook 02" in window.status


def test_final_report_of_a_finished_trainer_is_shown(window):
    class Done:
        exitcode = 0

        def is_alive(self):
            return False

    class LateQueue(FakeQueue):
        """The trainer's last report becomes readable only after its process has ended."""

        def __init__(self):
            super().__init__()
            self.calls = 0

        def get_nowait(self):
            self.calls += 1
            if self.calls == 2:
                self.items.append(ga_report(5))
            return super().get_nowait()

    window.process, window.reports = Done(), LateQueue()
    window.poll_reports()
    assert window.latest["generation"] == 5 and "finished" in window.status.lower()


def test_one_x_runs_at_twice_real_time(window):
    window.receive(ga_report())
    window.update(0.0)
    window.update(0.1)
    assert abs(window.playback.cars.time - config.PLAYBACK_BASE * 0.1) < 1e-9


def test_tracks_are_built_in_the_background(monkeypatch, circle_track):
    release = threading.Event()

    def slow_track(seed, level):
        release.wait(5)
        return circle_track

    monkeypatch.setattr(viewer, "get_track", slow_track)
    pygame.init()
    app = viewer.App(pygame.display.set_mode(config.WINDOW_SIZE))
    try:
        app.receive(ga_report())
        app.update(1 / 60)
        app.draw()
        assert app.playback is None and "Map:" in app.map_text()     # the window keeps drawing meanwhile
        release.set()
        for _ in range(200):
            app.update(1 / 60)
            if app.playback is not None:
                break
            time.sleep(0.01)
        assert app.playback is not None
    finally:
        release.set()
        pygame.quit()


def test_map_text_names_the_track(window, circle_track):
    assert window.map_text() == "Map: shown with the first generation"      # GA before training
    window.receive(ga_report())
    window.update(1 / 60)
    assert window.map_text() == f"Map: seed {circle_track.seed}, level {circle_track.level}"


def test_imitation_preview_uses_the_difficulty(monkeypatch, window, circle_track):
    calls = []
    monkeypatch.setattr(viewer, "get_track", lambda seed, level: calls.append((seed, level)) or circle_track)
    window.select_approach("baseline")
    window.set_difficulty(3)
    window.draw()
    assert (config.KT_VIEW_SEED, 3) in calls


def test_training_panel_fits_the_window(window):
    for approach in config.APPROACHES:
        window.approach = approach
        window.draw()
        assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons), approach
        assert window.slider.bottom <= window.panel.bottom - 30
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_app.py -q`
Expected: FAIL (new signature of `trainer_main`, missing `trainer_process`, `stop`, `map_text`, …).

- [ ] **Step 3: Implement in `racing/app.py`**

Imports and constants:

```python
import multiprocessing as mp
import os
import queue
import signal
import threading

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from racing import config, nets, paths, sim
from racing.actions import class_to_controls
from racing.tracks import get_track
```

Set `BUTTON_ACTIVE = (55, 62, 78)` with the comment `# dark neutral: never mistaken for an approach (spec §10.1)`.

Delete the module-level `mixed_policy`; in `report_policy` and `load_race` use `nets.mixed_policy(...)`.

Replace `trainer_main` and add `trainer_process`:

```python
def trainer_main(approach, size, seed, level, reports, running):
    """Training process: trains one approach with run seed `seed`, starting at curriculum level `level`, and sends a
    report after every generation, trial or epoch. Writes to config.APP_RUNS_DIR and never saves final
    controllers (spec §10.2)."""
    from racing import es, ga, imitation
    from racing.expert import collect_dataset

    def send(report):
        reports.put(report)
        running.wait()                      # training holds here while the viewer is paused

    out_dir = config.APP_RUNS_DIR / approach / f"seed{seed}"
    if approach == "ga":
        ga.run_ga(pop_size=size, seed=seed, start_level=level, out_dir=out_dir, on_generation=send,
                  champion_dir=None, n_jobs=config.N_JOBS)
        return
    if approach in ("es", "es_seeded"):
        initial = es.trained_baseline(seed) if approach == "es_seeded" else None
        es.run_es(lam=size, seed=seed, start_level=level, initial=initial, approach=approach, out_dir=out_dir,
                  on_generation=send, champion_dir=None, n_jobs=config.N_JOBS)
        return
    path = config.DATASET_PATH
    if not path.exists():
        path = config.APP_RUNS_DIR / "dataset.npz"     # never write the reported dataset
        if not path.exists():
            collect_dataset(path)
    data = imitation.load_dataset(path)
    members, scores = [], []

    def add(number, model, val_accuracy):
        members.append(nets.from_keras(model))
        scores.append(val_accuracy)
        fitness = np.array(scores)
        nan = float("nan")
        send({"approach": approach, "generation": int(number), "level": level,
              "seeds": [config.KT_VIEW_SEED], "nets": list(members), "fitness": fitness,
              "kept_finish": nan, "new_finish": nan, "step_size": nan, "best": float(fitness.max()),
              "mean": float(fitness.mean()), "worst": float(fitness.min()), "leveled_up": False})

    if approach == "baseline":
        imitation.train_baseline(data, out_dir=out_dir, champion_dir=None, seed=seed,
                                 on_epoch=lambda epoch, model, acc: add(epoch, model, acc))
    else:
        imitation.run_tuner(approach, data, out_dir=out_dir, on_trial=add, seed=seed, export_models=False)


def trainer_process(approach, size, seed, level, reports, running):
    """Entry point of the training process. It leads its own process group, so stopping it also stops the worker
    processes it starts (spec §10.2)."""
    os.setpgrp()
    trainer_main(approach, size, seed, level, reports, running)
```

In `App.__init__`: change the signature to `def __init__(self, screen, background_tracks=True):` and add
`self.background_tracks = background_tracks`, `self.tracks = {}`, `self.building = set()`,
`self.run_seed = config.RUN_SEEDS[0]`, `self.difficulty = 1`, and
`self.population = {"ga": config.GA_POP, "es": config.ES_LAMBDA, "es_seeded": config.ES_LAMBDA}`.
Rename `self.track_cache` to `self.surfaces` (the rendered track surfaces, cached per track) in `__init__` and
`track_surface`.

Add and replace these methods:

```python
    # ----- tracks -----
    def request_track(self, seed, level):
        """The track if it is built; otherwise start building it and return None (spec §10.2). Without background
        building, the track is built at once."""
        key = (int(seed), int(level))
        if key in self.tracks:
            return self.tracks[key]
        if not self.background_tracks:
            self.tracks[key] = get_track(*key)
            return self.tracks[key]
        if key not in self.building:
            self.building.add(key)
            threading.Thread(target=self.build_track, args=(key,), daemon=True).start()
        return None

    def build_track(self, key):
        self.tracks[key] = get_track(*key)
        self.building.discard(key)

    # ----- training process -----
    def start_training(self):
        if self.approach == "es_seeded" and not paths.champion_path("baseline", self.run_seed,
                                                                   champion_dir=self.champions_dir).exists():
            self.status = "Seeded ES needs the baseline controller: run notebook 02 first."
            return
        context = mp.get_context("spawn")
        self.reports = context.Queue()
        self.running = context.Event()
        self.running.set()
        size = self.population.get(self.approach, 0)
        self.process = context.Process(target=trainer_process, daemon=False,
                                       args=(self.approach, size, self.run_seed, self.difficulty, self.reports,
                                             self.running))
        self.process.start()
        self.latest, self.history, self.playback, self.playback_generation = None, [], None, None
        self.status = f"Training {config.LABELS[self.approach]}, seed {self.run_seed}."

    def stop_training(self):
        """End the training process and the worker processes it started, and drop its report queue."""
        if self.process is not None and self.process.is_alive():
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self.process.terminate()             # the process has not yet become a group leader
            self.process.join(timeout=2)
        self.process = None
        self.discard_reports()

    def stop(self):
        """Stop control of the training view: training ends and the settings can be changed again."""
        self.stop_training()
        self.status = "Stopped. Change the settings or press Play."

    def drain_reports(self):
        if self.reports is None:
            return
        while True:
            try:
                self.receive(self.reports.get_nowait())
            except queue.Empty:
                break

    def poll_reports(self):
        self.drain_reports()
        if self.process is not None and not self.process.is_alive():
            self.drain_reports()                     # a report sent just before the process ended
            code = self.process.exitcode
            self.status = "Training finished." if code == 0 else f"Training stopped (exit code {code})."
            self.process = None
            self.discard_reports()

    # ----- settings (changeable only while no training runs) -----
    def select_approach(self, approach):
        if self.process is not None or approach == self.approach:
            return
        self.stop_training()
        self.approach = approach
        self.latest, self.history, self.playback, self.playback_generation = None, [], None, None
        self.status = "Press Play to start training."

    def change_population(self, direction):
        if self.process is not None:
            return
        presets = list(config.POPULATION_PRESETS)
        current = self.population[self.approach]
        position = presets.index(current) if current in presets else 0
        self.population[self.approach] = presets[min(max(position + direction, 0), len(presets) - 1)]

    def change_run_seed(self, direction):
        if self.process is not None:
            return
        seeds = list(config.RUN_SEEDS)
        self.run_seed = seeds[(seeds.index(self.run_seed) + direction) % len(seeds)]

    def set_difficulty(self, level):
        if self.process is not None:
            return
        self.difficulty = level
        if self.approach not in config.EVOLUTION_APPROACHES:
            self.playback = None                     # the imitation views drive the map of this level

    # ----- playback -----
    def start_playback(self, report, track):
        fitness = np.asarray(report["fitness"], dtype=float)
        self.indices = select_indices(fitness, self.show, self.shown, sample_seed=report["generation"])
        self.playback = Playback(track, report_policy(report, self.indices), len(self.indices))
        self.best_slot = int(np.argmax(fitness[self.indices]))
        self.playback_generation = report["generation"]
        self.step_budget = 0.0

    def shown_track(self):
        """The track on screen in the training view: the playback's, or for imitation approaches the map they are
        driven on, which is known before training starts."""
        if self.playback is not None:
            return self.playback.track
        if self.approach not in config.EVOLUTION_APPROACHES:
            return self.request_track(config.KT_VIEW_SEED, self.difficulty)
        return None

    def map_text(self):
        track = self.shown_track()
        if track is not None:
            return f"Map: seed {track.seed}, level {track.level}"
        if self.approach in config.EVOLUTION_APPROACHES and self.latest is None:
            return "Map: shown with the first generation"
        return "Map: building the track"

    def update(self, dt):
        self.poll_reports()
        if self.mode == "train":
            if self.latest is not None and (self.playback is None or self.playback.over):
                track = self.request_track(self.latest["seeds"][0], self.latest["level"])
                if track is not None:
                    self.start_playback(self.latest, track)
            active = self.playback
        else:
            active = self.race
        if active is not None:
            self.step_budget += self.speed * config.PLAYBACK_BASE * dt / config.DT
            steps = int(self.step_budget)
            self.step_budget = 0.0 if steps > config.MAX_STEPS_PER_FRAME else self.step_budget - steps
            steps = min(steps, config.MAX_STEPS_PER_FRAME)
            active.advance(steps)
```

`start_race` uses `self.request_track(config.TEST_SEEDS[self.race_seed], config.MAX_LEVEL)` and returns without a
race if it is `None`.

Drawing: `track_surface` caches in `self.surfaces`. `draw` shows the training view's track even without
playback:

```python
    def draw(self):
        self.screen.fill(GRASS)
        self.buttons = []
        self.slider = None
        if self.mode == "train":
            track, playback = self.shown_track(), self.playback
        else:
            playback = self.race
            track = playback.track if playback is not None else None
        if track is not None:
            camera, surface = self.track_surface(track)
            self.screen.blit(surface, self.view.topleft)
            if playback is not None and self.mode == "train":
                self.draw_population(playback, camera)
            elif playback is not None:
                self.draw_racers(playback, camera)
        self.draw_panel()
```

Panel (every control must end at least 30 px above the panel bottom, where the status line is drawn):

```python
    def draw_difficulty(self, x, y, w, level, action, enabled):
        """Easy / Medium / Hard / Extra hard buttons (spec §10.4). Returns the next free y."""
        y = self.text("Difficulty", x, y, self.small, MUTED)
        quarter = (w - 24) // 4
        for k, (value, name) in enumerate(config.DIFFICULTIES.items()):
            self.button(pygame.Rect(x + k * (quarter + 8), y, quarter, 26), name, lambda v=value: action(v),
                        level == value, enabled=enabled)
        return y + 34

    def draw_train_panel(self, x, y, w):
        idle = self.process is None
        third = (w - 16) // 3
        y = self.text("Approach", x, y, self.small, MUTED)
        for k, approach in enumerate(config.APPROACHES):
            rect = pygame.Rect(x + (k % 3) * (third + 8), y + (k // 3) * 34, third, 28)
            self.button(rect, config.LABELS[approach], lambda a=approach: self.select_approach(a),
                        approach == self.approach, enabled=idle, color=config.COLORS[approach])
        y += 3 * 34 + 4
        self.text("Run seed", x, y + 5, self.small, MUTED)
        self.button(pygame.Rect(x + 110, y, 30, 28), "<", lambda: self.change_run_seed(-1), enabled=idle)
        self.text(str(self.run_seed), x + 152, y + 5)
        self.button(pygame.Rect(x + 180, y, 30, 28), ">", lambda: self.change_run_seed(1), enabled=idle)
        y += 36
        y = self.draw_difficulty(x, y, w, self.difficulty, self.set_difficulty, idle)
        if self.approach in self.population:
            self.text("Population" if self.approach == "ga" else "Population (λ)", x, y + 5, self.small, MUTED)
            self.button(pygame.Rect(x + 110, y, 30, 28), "−", lambda: self.change_population(-1), enabled=idle)
            self.text(str(self.population[self.approach]), x + 152, y + 5)
            self.button(pygame.Rect(x + 210, y, 30, 28), "+", lambda: self.change_population(1), enabled=idle)
            y += 36
        playing = self.process is not None and self.running is not None and self.running.is_set()
        half = (w - 8) // 2
        self.button(pygame.Rect(x, y, half, 32), "Pause" if playing else "Play", self.toggle_play, playing)
        self.button(pygame.Rect(x + half + 8, y, half, 32), "Stop", self.stop, enabled=not idle)
        y += 42
        y = self.text("Show", x, y, self.small, MUTED)
        for k, mode in enumerate(("all", "best", "worst")):
            self.button(pygame.Rect(x + k * (third + 8), y, third, 26), mode.capitalize(),
                        lambda m=mode: self.set_show(m), self.show == mode)
        y += 34
        size = self.population_size()
        shown = min(self.shown, size)
        y = self.text(f"Cars shown: {shown}", x, y, self.small, MUTED)
        self.slider = pygame.Rect(x, y + 4, w, 10)
        pygame.draw.rect(self.screen, BUTTON, self.slider, border_radius=5)
        knob = x + (shown - 1) / max(size - 1, 1) * w
        pygame.draw.circle(self.screen, BUTTON_ACTIVE, (int(knob), y + 9), 8)
        y += 24
        y = self.draw_speed_buttons(x, y, w)
        self.button(pygame.Rect(x, y, w, 26), f"Sensor rays: {'on' if self.rays else 'off'}", self.toggle_rays,
                    self.rays)
        y += 34
        y = self.draw_stats(x, y)
        bottom = self.panel.bottom - 34
        if bottom - y > 50:
            self.draw_plot(pygame.Rect(x, y + 4, w, bottom - y - 4))

    def draw_stats(self, x, y):
        y = self.text(self.map_text(), x, y, self.small)
        r = self.latest
        if r is None:
            return self.text("No generation completed yet.", x, y, self.small, MUTED)
        nan = float("nan")
        unit = "Trial" if r["approach"].startswith("kt") else ("Epoch" if r["approach"] == "baseline" else "Generation")
        size_name = "Mutation strength" if r["approach"] == "ga" else "σ"
        for line in (f"Level {r['level']}    {unit} {r['generation']}",
                     f"Finished: kept {percent(r['kept_finish'])}   new {percent(r['new_finish'])}",
                     f"At pace: kept {percent(r.get('kept_pace', nan))}   new {percent(r.get('new_pace', nan))}",
                     f"{size_name}: {number(r['step_size'])}",
                     f"Best {r['best']:.3f}    Mean {r['mean']:.3f}"):
            y = self.text(line, x, y, self.small)
        return y
```

`draw_speed_buttons` makes its buttons 28 px high and returns `y + 36`. The status line stays at
`self.panel.bottom - 26`. In `main()`, nothing changes (it already calls `viewer.stop_training()` on exit).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_app.py -q`
Expected: PASS. A visual check of the window is done later by the controller.

- [ ] **Step 5: Commit**

```bash
git add racing/app.py tests/test_app.py
git commit -m "Training view: run seed, difficulty, stop control, map display, background tracks, worker processes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/app.py tests/test_app.py
```

---

### Task 10: Viewer: race view

**Files:**
- Modify: `racing/app.py`, `racing/nets.py` (remove `load_champions`)
- Test: `tests/test_app.py`, `tests/test_nets.py` (remove the two `load_champions` tests)

**Interfaces:**
- Consumes: Task 9 (`App.request_track`, `App.draw_difficulty`, `App.stop`, `PLAYBACK_BASE` handling), Task 2
  (`nets.champion_seeds`, `nets.load_champion`, `nets.mixed_policy`), Task 1 (`paths.best_seeds_path`), the
  `best_seeds.json` written by Task 6.
- Produces: race-view state `race_index`, `race_level`, `race_available`, `race_choice`, `best`, `lineup`,
  `racing`; methods `load_race`, `chosen_seed`, `choice_label`, `cycle_choice`, `reset_race`, `prepare_race`,
  `start_race`, `stop_race`, `change_race_track`, `set_race_level`; `leaderboard(playback, count)` returns rows
  `(position, car index, laps, last lap, best lap, status)`. `App.stop()` stops the race in race mode.

- [ ] **Step 1: Write the failing tests**

In `tests/test_app.py` add `import json` and `from racing import population` to the imports, delete
`stub_champions`, `test_race_runs_and_draws_a_leaderboard`, `test_champions_saved_later_are_picked_up`,
`test_step_burst_is_capped_and_budget_dropped` and `test_leaderboard_orders_by_distance`, and add:

```python
def save_race_champions(directory, best=None):
    rng = np.random.default_rng(0)
    for approach, seed in (("es", 0), ("ga", 0), ("ga", 1)):
        population.save_champion(approach, seed, nets.init_population(rng, 1)[0], directory)
    if best is not None:
        (directory / "best_seeds.json").write_text(json.dumps(best))


def race_window(window, tmp_path, best=None):
    save_race_champions(tmp_path, best)
    window.champions_dir = tmp_path
    window.set_mode("race")
    return window


def test_race_lineup_uses_the_best_seeds(window, tmp_path):
    race_window(window, tmp_path, {"ga": 1, "es": 0})
    assert window.lineup == [("es", 0), ("ga", 1)]
    assert window.choice_label("ga") == "best (seed 1)"


def test_race_lineup_falls_back_without_best_seeds(window, tmp_path):
    race_window(window, tmp_path)
    assert window.lineup == [("es", 0), ("ga", 0)]


def test_cycling_a_choice_picks_seeds_and_off(window, tmp_path):
    race_window(window, tmp_path, {"ga": 1})
    window.cycle_choice("ga")
    assert ("ga", 0) in window.lineup and window.choice_label("ga") == "seed 0"
    window.cycle_choice("ga")
    assert ("ga", 1) in window.lineup
    window.cycle_choice("ga")
    assert window.lineup == [("es", 0)] and window.choice_label("ga") == "off"
    window.cycle_choice("ga")
    assert window.lineup == [("es", 0), ("ga", 1)]


def test_preview_waits_on_the_start_line_until_started(window, tmp_path):
    race_window(window, tmp_path)
    for _ in range(5):
        window.update(1 / 60)
        window.draw()
    assert window.race is not None and window.race.cars.time == 0.0 and not window.racing
    window.start_race()
    window.update(0.1)
    assert window.racing and window.race.cars.time > 0
    window.stop()
    window.update(1 / 60)
    assert not window.racing and window.race.cars.time == 0.0


def test_race_settings_are_locked_while_racing(window, tmp_path):
    race_window(window, tmp_path)
    window.update(1 / 60)
    window.start_race()
    lineup = list(window.lineup)
    window.change_race_track(1)
    window.set_race_level(2)
    window.cycle_choice("ga")
    assert window.race_index == 0 and window.race_level == config.MAX_LEVEL and window.lineup == lineup


def test_race_difficulty_sets_the_track_level(monkeypatch, window, tmp_path, circle_track):
    calls = []
    monkeypatch.setattr(viewer, "get_track", lambda seed, level: calls.append((seed, level)) or circle_track)
    race_window(window, tmp_path)
    window.set_race_level(2)
    window.update(1 / 60)
    assert (config.TEST_SEEDS[0], 2) in calls


def test_race_draws_a_leaderboard_and_fits_the_window(window, tmp_path):
    race_window(window, tmp_path)
    window.update(1 / 60)
    window.start_race()
    for _ in range(30):
        window.update(1 / 60)
        window.draw()
    rows = viewer.leaderboard(window.race, len(window.lineup))
    assert sorted(r[0] for r in rows) == [1, 2] and sorted(r[1] for r in rows) == [0, 1]
    assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons)


def test_leaderboard_orders_by_distance(circle_track):
    playback = viewer.Playback(circle_track, lambda obs, cars: np.zeros(len(cars.x), dtype=int), 2, laps=3)
    playback.cars.progress[:] = [10.0, 50.0]
    playback.cars.crashed[0] = True
    rows = viewer.leaderboard(playback, 2)
    assert [r[1] for r in rows] == [1, 0] and rows[1][5] == "crashed" and [r[0] for r in rows] == [1, 2]
    ready = viewer.leaderboard(None, 2)
    assert [(r[0], r[1], r[2], r[5]) for r in ready] == [(1, 0, 0, "ready"), (2, 1, 0, "ready")]


def test_champions_saved_later_are_picked_up(window, tmp_path):
    window.champions_dir = tmp_path
    window.set_mode("race")
    assert "No trained controllers" in window.status
    window.set_mode("train")
    save_race_champions(tmp_path)
    window.set_mode("race")
    assert window.race_available == {"es": [0], "ga": [0, 1]}


def test_step_burst_is_capped_and_budget_dropped(window):
    window.receive(ga_report())
    window.update(0.0)
    window.update(10.0)
    assert abs(window.playback.cars.time - config.MAX_STEPS_PER_FRAME * config.DT) < 1e-9
    assert window.step_budget == 0.0
```

In `tests/test_nets.py`, delete `test_load_champions_empty_directory` and
`test_load_champions_keeps_standard_order`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_app.py tests/test_nets.py -q`
Expected: FAIL (`lineup`, `cycle_choice`, `set_race_level`, … do not exist).

- [ ] **Step 3: Implement in `racing/app.py`**

Add `import json` to the imports. Change `leaderboard`:

```python
def leaderboard(playback, count):
    """Rows (position, car index, laps completed, last lap, best lap, status), ordered by distance covered."""
    nan = float("nan")
    if playback is None:
        return [(k + 1, k, 0, nan, nan, "ready") for k in range(count)]
    cars = playback.cars
    rows = []
    for position, i in enumerate(np.argsort(-cars.progress, kind="stable")):
        if cars.crashed[i]:
            status = "crashed"
        elif cars.finished[i]:
            status = "finished"
        elif cars.done[i]:
            status = "stopped"
        else:
            status = "racing"
        rows.append((position + 1, int(i), int(cars.laps_done[i]), float(cars.last_lap[i]), float(cars.best_lap[i]),
                     status))
    return rows
```

In `App.__init__`, replace the race-view attributes with:

```python
        # race view
        self.champions_dir = config.CHAMPIONS_DIR
        self.race_index = 0                 # into config.TEST_SEEDS
        self.race_level = config.MAX_LEVEL
        self.race_available = {}            # approach -> run seeds with a saved controller
        self.race_choice = {}               # approach -> "best", a run seed, or "off"
        self.best = {}                      # approach -> best run seed recorded by the evaluation
        self.drivers = {}                   # (approach, seed) -> DenseNet
        self.lineup = []                    # (approach, seed) of the cars in the race
        self.race = None                    # Playback: the preview until the race starts
        self.racing = False
```

Replace `set_mode`, `load_race`, `start_race` and `change_race_track`, and add the other race methods:

```python
    def set_mode(self, mode):
        self.mode = mode
        self.step_budget = 0.0
        if mode == "race" and not self.race_available:
            self.load_race()
        elif mode == "train" and self.status.startswith("No trained controllers"):
            self.status = "Training view."

    def stop(self):
        """Stop control: ends training in the training view and the race in the race view."""
        if self.mode == "race":
            self.stop_race()
            return
        self.stop_training()
        self.status = "Stopped. Change the settings or press Play."

    # ----- race view -----
    def load_race(self):
        """Find the saved final controllers and each approach's best run seed (spec §10.3)."""
        self.race_available = nets.champion_seeds(self.champions_dir)
        best_path = paths.best_seeds_path(self.champions_dir)
        self.best = {}
        if best_path.exists():
            self.best = {approach: int(seed) for approach, seed in json.loads(best_path.read_text()).items()}
        self.race_choice = {approach: self.race_choice.get(approach, "best") for approach in self.race_available}
        if self.race_available:
            self.status = "Race view."
        else:
            self.status = f"No trained controllers found in {self.champions_dir}."
        self.reset_race()

    def chosen_seed(self, approach):
        """The run seed whose controller races for `approach`, or None if the approach is off. Without an
        evaluation record, 'best' falls back to the lowest saved seed."""
        choice = self.race_choice.get(approach, "best")
        if choice == "off":
            return None
        seeds = self.race_available[approach]
        if choice == "best":
            best = self.best.get(approach)
            return best if best in seeds else seeds[0]
        return choice

    def choice_label(self, approach):
        choice = self.race_choice.get(approach, "best")
        if choice == "best":
            return f"best (seed {self.chosen_seed(approach)})"
        if choice == "off":
            return "off"
        return f"seed {choice}"

    def cycle_choice(self, approach):
        if self.racing:
            return
        options = ["best", *self.race_available[approach], "off"]
        current = self.race_choice.get(approach, "best")
        self.race_choice[approach] = options[(options.index(current) + 1) % len(options)]
        self.reset_race()

    def reset_race(self):
        """Back to the map preview: the selected controllers wait on the start line."""
        self.racing = False
        self.race = None
        self.step_budget = 0.0
        self.lineup = []
        for approach in self.race_available:
            seed = self.chosen_seed(approach)
            if seed is not None:
                self.lineup.append((approach, seed))

    def driver(self, approach, seed):
        if (approach, seed) not in self.drivers:
            self.drivers[(approach, seed)] = nets.load_champion(approach, seed, self.champions_dir)[1]
        return self.drivers[(approach, seed)]

    def race_track(self):
        return self.request_track(config.TEST_SEEDS[self.race_index], self.race_level)

    def prepare_race(self):
        """Build the preview once the track is ready: every selected car on the start line."""
        if self.race is None and self.lineup:
            track = self.race_track()
            if track is not None:
                drivers = [self.driver(approach, seed) for approach, seed in self.lineup]
                self.race = Playback(track, nets.mixed_policy(drivers), len(drivers), laps=config.RACE_LAPS)

    def start_race(self):
        self.prepare_race()
        if self.race is not None:
            self.racing = True
            self.status = "Racing."

    def stop_race(self):
        self.reset_race()
        self.status = "Race stopped. Change the controllers, track or difficulty, or start again."

    def change_race_track(self, direction):
        if self.racing:
            return
        self.race_index = (self.race_index + direction) % len(config.TEST_SEEDS)
        self.reset_race()

    def set_race_level(self, level):
        if self.racing:
            return
        self.race_level = level
        self.reset_race()
```

In `update`, the race branch becomes:

```python
        else:
            self.prepare_race()
            active = self.race if self.racing else None
```

In `draw`, the race branch shows the map even before the preview exists:

```python
        else:
            playback = self.race
            track = playback.track if playback is not None else (self.race_track() if self.race_available else None)
```

Replace `draw_racers` and `draw_race_panel`:

```python
    def draw_racers(self, playback, camera):
        cars = playback.cars
        sx, sy = camera.to_screen(cars.x, cars.y)
        size = max(8.0, 6.0 * camera.scale)
        for i, (approach, _) in enumerate(self.lineup):
            color = hex_to_rgb(config.COLORS[approach])
            if cars.crashed[i]:
                color = tuple(int(0.5 * c + 127) for c in color)          # faded
            shape = car_polygon(sx[i], sy[i], cars.heading[i], size)
            pygame.draw.polygon(self.screen, color, shape)
            pygame.draw.polygon(self.screen, INK, shape, 1)

    def draw_race_panel(self, x, y, w):
        if not self.race_available:
            self.text("No trained controllers found.", x, y, self.small, MUTED)
            return
        idle = not self.racing
        self.text("Test track", x, y + 5, self.small, MUTED)
        self.button(pygame.Rect(x + 110, y, 30, 28), "<", lambda: self.change_race_track(-1), enabled=idle)
        self.text(f"seed {config.TEST_SEEDS[self.race_index]}", x + 150, y + 5)
        self.button(pygame.Rect(x + 240, y, 30, 28), ">", lambda: self.change_race_track(1), enabled=idle)
        y += 36
        y = self.draw_difficulty(x, y, w, self.race_level, self.set_race_level, idle)
        half = (w - 8) // 2
        self.button(pygame.Rect(x, y, half, 32), "Start race", self.start_race, self.racing,
                    enabled=idle and bool(self.lineup))
        self.button(pygame.Rect(x + half + 8, y, half, 32), "Stop", self.stop_race, enabled=self.racing)
        y += 42
        y = self.text("Controllers", x, y, self.small, MUTED)
        for approach in self.race_available:
            pygame.draw.rect(self.screen, hex_to_rgb(config.COLORS[approach]), pygame.Rect(x, y + 8, 10, 10))
            self.text(config.LABELS[approach], x + 16, y + 5, self.small)
            self.button(pygame.Rect(x + 140, y, w - 140, 24), self.choice_label(approach),
                        lambda a=approach: self.cycle_choice(a), enabled=idle)
            y += 28
        y += 4
        y = self.draw_speed_buttons(x, y, w)
        y = self.text(f"Leaderboard ({config.RACE_LAPS} laps)", x, y, self.small, MUTED)
        for position, i, laps, last, best, status in leaderboard(self.race, len(self.lineup)):
            approach, seed = self.lineup[i]
            pygame.draw.rect(self.screen, hex_to_rgb(config.COLORS[approach]), pygame.Rect(x, y + 3, 10, 10))
            y = self.text(f"{position}. {config.LABELS[approach]} s{seed}  lap {laps}  last {number(last)}  "
                          f"best {number(best)}  {status}", x + 16, y, self.small)
```

Keyboard: the left and right arrow keys call `change_race_track`, which ignores them while racing. Remove the
attributes `race_seed`, `race_names` and `race_nets` and every use of them.

In `racing/nets.py`, delete `load_champions` (nothing uses it any more; check with
`grep -rn load_champions racing tests scripts`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_app.py tests/test_nets.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add racing/app.py racing/nets.py tests/test_app.py tests/test_nets.py
git commit -m "Race view: map preview, difficulty, start and stop, controller choice per approach and seed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/app.py racing/nets.py tests/test_app.py tests/test_nets.py
```

---

### Task 11: Calibration pilots

**Files:**
- Create: `scripts/pilots.py`
- Test: `tests/test_pilots.py`

**Interfaces:**
- Consumes: Task 3 (`ga.run_ga(..., s_max=)`, `es.run_es(..., initial=, approach=)`, `es.trained_baseline(seed,
  champion_dir)`), Task 4 (`imitation.train_baseline`, `imitation.load_dataset`), Task 5 (`collect_dataset`).
- Produces: `pilot_ga(s_max, out_root, generations=100, pop_size=config.GA_POP, kept=config.GA_KEPT,
  n_jobs=config.N_JOBS) -> dict`, `pilot_es(out_root, generations=100, mu=config.ES_MU, lam=config.ES_LAMBDA,
  n_jobs=config.N_JOBS, initial=None) -> dict`, `pilot_seeded(out_root, generations=100, n_jobs=config.N_JOBS) ->
  dict`, `main(argv=None)`. Output: `results/pilots/<pilot>/` and `results/pilots/summary.md`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pilots.py`:

```python
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from racing import nets, population

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "pilots.py"


@pytest.fixture
def pilots(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    spec = importlib.util.spec_from_file_location("pilots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ga_pilot_summarizes_a_short_run(pilots, tmp_path):
    row = pilots.pilot_ga(0.2, tmp_path, generations=2, pop_size=20, kept=5, n_jobs=1)
    assert row["pilot"] == "GA s_max=0.2" and row["level_reached"] >= 1 and row["generations"] == 2
    assert (tmp_path / "ga_smax0.2" / "log.csv").exists()


def test_es_pilot_reports_step_size_and_ties(pilots, tmp_path):
    row = pilots.pilot_es(tmp_path, generations=2, mu=4, lam=16, n_jobs=1)
    assert row["pilot"] == "ES" and row["sigma_last"] > 0
    assert 0.0 <= row["tie_fraction"] <= 1.0 and 0.0 <= row["strict_success"] <= 1.0


def test_seeded_pilot_trains_its_own_baseline(pilots, monkeypatch, tmp_path):
    trained = []
    monkeypatch.setattr(pilots.config, "DATASET_PATH", tmp_path / "dataset.npz")
    (tmp_path / "dataset.npz").write_bytes(b"")
    monkeypatch.setattr(pilots.imitation, "train_baseline", lambda data, **kwargs: trained.append(kwargs))
    monkeypatch.setattr(pilots.imitation, "load_dataset", lambda path=None: {})
    monkeypatch.setattr(pilots.es, "trained_baseline",
                        lambda seed, champion_dir: nets.init_population(np.random.default_rng(0), 1)[0])
    row = pilots.pilot_seeded(tmp_path, generations=1, mu=4, lam=16, n_jobs=1)
    assert row["pilot"] == "Seeded ES" and trained[0]["champion_dir"] == tmp_path / "seeded" / "champions"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_pilots.py -q`
Expected: FAIL (`scripts/pilots.py` does not exist).

- [ ] **Step 3: Create `scripts/pilots.py`**

```python
"""Calibration pilots (spec §3.3): the GA mutation-strength range, the ES step-size behavior under the tie rule and
the pace requirement, and the seeded ES. Each pilot runs seed 0 at the reported settings with a shortened budget,
writes to results/pilots/ and adds a row to results/pilots/summary.md. Pilots are not part of the reported results.

Run with:  ~/miniconda3/bin/python scripts/pilots.py              (all pilots)
           ~/miniconda3/bin/python scripts/pilots.py ga es       (selected pilots: ga, es, seeded)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racing import config, es, ga, imitation  # noqa: E402
from racing.expert import collect_dataset  # noqa: E402

GA_CANDIDATES = (0.5, 0.2, 0.1)


def level_ups(log):
    """Generations at which the level advanced, as text."""
    return " ".join(str(g) for g in log.loc[log["leveled_up"].astype(bool), "generation"]) or "none"


def pilot_ga(s_max, out_root, generations=100, pop_size=config.GA_POP, kept=config.GA_KEPT, n_jobs=config.N_JOBS):
    out = Path(out_root) / f"ga_smax{s_max}"
    result = ga.run_ga(pop_size=pop_size, kept=kept, budget=generations * pop_size, seed=0, out_dir=out,
                       champion_dir=None, n_jobs=n_jobs, s_max=s_max)
    log = pd.read_csv(out / "log.csv")
    return {"pilot": f"GA s_max={s_max}", "generations": result["generations"],
            "level_reached": result["level_reached"], "level_ups": level_ups(log),
            "new_finish_last": round(float(log["new_finish"].iloc[-1]), 2),
            "new_pace_last": round(float(log["new_pace"].iloc[-1]), 2),
            "best_last": round(float(log["best"].iloc[-1]), 3), "minutes": round(result["wall_time"] / 60, 1)}


def pilot_es(out_root, generations=100, mu=config.ES_MU, lam=config.ES_LAMBDA, n_jobs=config.N_JOBS, initial=None,
             name="ES", folder="es"):
    out = Path(out_root) / folder
    approach = "es" if initial is None else "es_seeded"
    result = es.run_es(mu=mu, lam=lam, budget=generations * (mu + lam), seed=0, out_dir=out, champion_dir=None,
                       n_jobs=n_jobs, initial=initial, approach=approach)
    log = pd.read_csv(out / "log.csv")
    recent = log.tail(20)
    return {"pilot": name, "generations": result["generations"], "level_reached": result["level_reached"],
            "level_ups": level_ups(log), "new_pace_last": round(float(log["new_pace"].iloc[-1]), 2),
            "sigma_min": float(log["step_size"].min()), "sigma_last": float(log["step_size"].iloc[-1]),
            "strict_success": round(float(recent["strict_success"].mean()), 3),
            "tie_fraction": round(float(recent["tie_fraction"].mean()), 3),
            "best_last": round(float(log["best"].iloc[-1]), 3), "minutes": round(result["wall_time"] / 60, 1)}


def pilot_seeded(out_root, generations=100, mu=config.ES_MU, lam=config.ES_LAMBDA, n_jobs=config.N_JOBS):
    """Trains a seed-0 baseline into the pilot folder (the reported one may not exist yet), then runs the seeded ES."""
    folder = Path(out_root) / "seeded"
    dataset = config.DATASET_PATH
    if not dataset.exists():
        dataset = folder / "dataset.npz"
        if not dataset.exists():
            collect_dataset(dataset, n_jobs=n_jobs)
    imitation.train_baseline(imitation.load_dataset(dataset), out_dir=folder / "baseline",
                             champion_dir=folder / "champions", seed=0)
    initial = es.trained_baseline(0, champion_dir=folder / "champions")
    return pilot_es(folder, generations, mu, lam, n_jobs, initial=initial, name="Seeded ES", folder="es_seeded")


def main(argv=None):
    selected = (sys.argv[1:] if argv is None else argv) or ["ga", "es", "seeded"]
    out_root = config.PILOTS_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    if "ga" in selected:
        for s_max in GA_CANDIDATES:
            rows.append(pilot_ga(s_max, out_root))
            print(rows[-1], flush=True)
    if "es" in selected:
        rows.append(pilot_es(out_root))
        print(rows[-1], flush=True)
    if "seeded" in selected:
        rows.append(pilot_seeded(out_root))
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    try:
        text = table.to_markdown(index=False)
    except ImportError:                          # pandas needs the optional tabulate package for Markdown
        text = table.to_string(index=False)
    with open(out_root / "summary.md", "a") as f:
        f.write(text + "\n\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
```

`DataFrame.to_markdown` needs the `tabulate` package, which may be missing; the fallback writes plain text. Do not
install `tabulate`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_pilots.py -q`
Expected: PASS. Do not run the pilots themselves.

- [ ] **Step 5: Commit**

```bash
git add scripts/pilots.py tests/test_pilots.py
git commit -m "Add calibration pilots for the GA mutation range, the ES step size and the seeded ES

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- scripts/pilots.py tests/test_pilots.py
```

---

### Task 12: Notebooks, experiment runner and README

**Files:**
- Modify: `scripts/make_notebooks.py`, `README.md`
- Create: `scripts/run_notebooks.py`, `tests/test_runner.py`
- Regenerate: `notebooks/01_dataset.ipynb` … `notebooks/06_evaluate.ipynb`; delete `notebooks/05_evaluate.ipynb`

**Interfaces:**
- Consumes: every earlier task's public functions (as listed in their Interfaces blocks).
- Produces: `scripts/run_notebooks.py` with `run(notebook_dir=NOTEBOOK_DIR, start=None,
  log_path=config.RUN_LOG_PATH, info_path=config.RUN_INFO_PATH) -> int` (0 on success, 1 if a notebook failed)
  and `main(argv=None)`; notebooks `01_dataset`, `02_imitation`, `03_ga`, `04_es`, `05_es_seeded`, `06_evaluate`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_runner.py`:

```python
import importlib.util
import json
from pathlib import Path

import nbformat
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_notebooks.py"


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("run_notebooks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_notebook(path, source):
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]), path)


def test_notebook_order_and_start(runner, tmp_path):
    for name in ("02_b.ipynb", "01_a.ipynb", "10_c.ipynb", "notes.ipynb"):
        write_notebook(tmp_path / name, "1")
    assert runner.notebook_names(tmp_path) == ["01_a.ipynb", "02_b.ipynb", "10_c.ipynb"]
    assert runner.notebook_names(tmp_path, "02") == ["02_b.ipynb", "10_c.ipynb"]


@pytest.mark.slow
def test_runs_notebooks_in_order_and_records_the_run(runner, tmp_path):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    write_notebook(folder / "01_a.ipynb", "import os, sys\nprint(os.getcwd())\nprint(sys.executable)")
    write_notebook(folder / "02_b.ipynb", "print('second')")
    code = runner.run(folder, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json")
    assert code == 0
    first = nbformat.read(folder / "01_a.ipynb", as_version=4).cells[0].outputs[0]["text"].split()
    assert first == [str(folder), runner.sys.executable]
    info = json.loads((tmp_path / "info.json").read_text())
    assert [e["notebook"] for e in info["notebooks"]] == ["01_a.ipynb", "02_b.ipynb"]
    assert all(e["seconds"] >= 0 for e in info["notebooks"]) and info["n_jobs"] >= 1
    assert "done 02_b.ipynb" in (tmp_path / "log.txt").read_text()


@pytest.mark.slow
def test_resume_keeps_earlier_record_entries(runner, tmp_path):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    write_notebook(folder / "01_a.ipynb", "print(1)")
    write_notebook(folder / "02_b.ipynb", "print(2)")
    runner.run(folder, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json")
    runner.run(folder, start="02", log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json")
    info = json.loads((tmp_path / "info.json").read_text())
    assert [e["notebook"] for e in info["notebooks"]] == ["01_a.ipynb", "02_b.ipynb"]


@pytest.mark.slow
def test_a_failing_notebook_stops_the_run(runner, tmp_path):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    write_notebook(folder / "01_a.ipynb", "raise ValueError('broken')")
    write_notebook(folder / "02_b.ipynb", "print(2)")
    code = runner.run(folder, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json")
    info = json.loads((tmp_path / "info.json").read_text())
    assert code == 1 and [e["notebook"] for e in info["notebooks"]] == ["01_a.ipynb"]
    assert "failed" in info["notebooks"][0]
    assert nbformat.read(folder / "01_a.ipynb", as_version=4).cells[0].outputs     # the error is saved


def test_experiment_notebooks_are_valid_and_compile():
    import ast
    folder = Path(__file__).resolve().parent.parent / "notebooks"
    names = sorted(p.name for p in folder.glob("*.ipynb"))
    assert names == ["01_dataset.ipynb", "02_imitation.ipynb", "03_ga.ipynb", "04_es.ipynb", "05_es_seeded.ipynb",
                     "06_evaluate.ipynb"]
    for name in names:
        notebook = nbformat.read(folder / name, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_runner.py -q -m "slow or not slow"`
Expected: FAIL (`scripts/run_notebooks.py` does not exist; notebook names differ).

- [ ] **Step 3: Create `scripts/run_notebooks.py`**

```python
"""Runs the experiment notebooks in order and saves their outputs into the notebook files (spec §3.3).

Run with:      ~/miniconda3/bin/python scripts/run_notebooks.py
Resume:        ~/miniconda3/bin/python scripts/run_notebooks.py --from 03
In background: nohup ~/miniconda3/bin/python scripts/run_notebooks.py > /dev/null 2>&1 &
Progress:      tail -f results/run_log.txt
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from racing import config  # noqa: E402

NOTEBOOK_DIR = PROJECT / "notebooks"
KERNEL = "ec-racing"


def now():
    return datetime.now().isoformat(timespec="seconds")


def log(message, path):
    line = f"{now()}  {message}"
    print(line, flush=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def processor():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def power_profile():
    try:
        result = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def versions():
    from importlib.metadata import PackageNotFoundError, version
    found = {"python": platform.python_version()}
    for package in ("numpy", "tensorflow", "keras", "keras-tuner"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "missing"
    return found


def notebook_names(folder, start=None):
    """Experiment notebooks (named NN_*.ipynb) in order, from notebook `start` on."""
    names = sorted(p.name for p in Path(folder).glob("[0-9][0-9]_*.ipynb"))
    return [name for name in names if start is None or name[:2] >= start]


def execute(path, kernel_dir):
    """Run every cell of one notebook in a new kernel on this interpreter, with the notebook's folder as working
    directory, and save the outputs into the file, also when a cell fails."""
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client.manager import KernelManager
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name=KERNEL, kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir)]))
    client = NotebookClient(notebook, km=manager, timeout=None, resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)


def write_info(info, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(info, indent=2))


def run(notebook_dir=NOTEBOOK_DIR, start=None, log_path=config.RUN_LOG_PATH, info_path=config.RUN_INFO_PATH):
    """Execute the notebooks in order. A resumed run (start given) keeps the earlier notebooks' record entries."""
    from ipykernel.kernelspec import write_kernel_spec

    notebook_dir = Path(notebook_dir)
    info = {}
    if start is not None and Path(info_path).exists():
        info = json.loads(Path(info_path).read_text())
    info.update({"processor": processor(), "logical_processors": os.cpu_count(), "n_jobs": config.N_JOBS,
                 "power_profile": power_profile(), "versions": versions()})
    info.setdefault("notebooks", [])
    info.setdefault("runs", []).append({"started": now(), "from": start})
    names = notebook_names(notebook_dir, start)
    log(f"run started: {' '.join(names)} (power profile {info['power_profile']}, {config.N_JOBS} workers)", log_path)
    with tempfile.TemporaryDirectory() as kernels:
        write_kernel_spec(Path(kernels) / KERNEL)
        for name in names:
            entry = {"notebook": name, "started": now()}
            info["notebooks"] = [e for e in info["notebooks"] if e["notebook"] != name] + [entry]
            write_info(info, info_path)
            log(f"start {name}", log_path)
            began = time.time()
            try:
                execute(notebook_dir / name, kernels)
            except Exception as error:                 # a failing cell; its output is saved in the notebook
                entry["failed"] = str(error).strip().splitlines()[-1] if str(error).strip() else type(error).__name__
                entry["seconds"] = round(time.time() - began, 1)
                write_info(info, info_path)
                log(f"FAILED {name}: {entry['failed']}", log_path)
                return 1
            entry["finished"] = now()
            entry["seconds"] = round(time.time() - began, 1)
            write_info(info, info_path)
            log(f"done {name} in {entry['seconds'] / 60:.1f} min", log_path)
    log("run finished", log_path)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the EC Racing experiment notebooks in order.")
    parser.add_argument("--from", dest="start", help="number of the first notebook to run, for example 03")
    args = parser.parse_args(argv)
    return run(start=args.start)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Rewrite the notebook definitions in `scripts/make_notebooks.py`**

Change the module docstring's "five" to "six" and replace `NOTEBOOKS` with:

```python
EVOLUTION_PROGRESS = (
    "def progress(report):\n"
    "    if report['generation'] % 10 == 0 or report['leveled_up']:\n"
    "        print(f\"gen {report['generation']:4d}  level {report['level']}  best {report['best']:.3f}  \"\n"
    "              f\"new finish {report['new_finish']:.2f}  at pace {report['new_pace']:.2f}  \"\n"
    "              f\"step {report['step_size']:.4f}\")\n")

NOTEBOOKS = {
    "01_dataset.ipynb": [
        ("markdown", "# 1. Expert dataset\nCollects the imitation-learning dataset (design specification §7)."),
        ("code", SETUP),
        ("code", "import json\n\nimport pandas as pd\n\nfrom racing import config\nfrom racing.expert import "
                 "collect_dataset\n\ndata, summary = collect_dataset(n_jobs=config.N_JOBS)\n"
                 "print('collection time (s):', json.loads(config.DATASET_INFO_PATH.read_text())['wall_time'])\n"
                 "pd.DataFrame(summary).T"),
    ],
    "02_imitation.ipynb": [
        ("markdown", "# 2. Imitation-learning controllers\nBaseline with each run seed, the three Keras Tuner "
                     "searches, and retraining of each selected architecture with each run seed (design "
                     "specification §8)."),
        ("code", SETUP),
        ("code", "from racing import config, imitation\n\ndata = imitation.load_dataset()\n"
                 "for seed in config.RUN_SEEDS:\n    print(imitation.train_baseline(data, seed=seed))"),
        ("code", "for approach in config.TUNER_APPROACHES:\n    summary = imitation.run_tuner(approach, data)\n"
                 "    print({k: v for k, v in summary.items() if k != 'chosen_hyperparameters'})\n"
                 "    for seed in config.RUN_SEEDS:\n        print(imitation.retrain(approach, data, seed=seed))"),
    ],
    "03_ga.ipynb": [
        ("markdown", "# 3. Genetic algorithm\nNeuroevolution with the curriculum, once per run seed (design "
                     "specification §9.2)."),
        ("code", SETUP),
        ("code", "from racing import config, ga\n\n\n" + EVOLUTION_PROGRESS + "\n\n"
                 "for seed in config.RUN_SEEDS:\n    print(f'--- seed {seed}')\n"
                 "    print(ga.run_ga(seed=seed, on_generation=progress, n_jobs=config.N_JOBS))"),
    ],
    "04_es.ipynb": [
        ("markdown", "# 4. Evolution strategy\n(μ+λ)-ES with the 1/5 success rule, once per run seed (design "
                     "specification §9.3)."),
        ("code", SETUP),
        ("code", "from racing import config, es\n\n\n" + EVOLUTION_PROGRESS + "\n\n"
                 "for seed in config.RUN_SEEDS:\n    print(f'--- seed {seed}')\n"
                 "    print(es.run_es(seed=seed, on_generation=progress, n_jobs=config.N_JOBS))"),
    ],
    "05_es_seeded.ipynb": [
        ("markdown", "# 5. Seeded evolution strategy\nThe ES started from the baseline trained with the same run "
                     "seed (design specification §9.4)."),
        ("code", SETUP),
        ("code", "from racing import config, es\n\n\n" + EVOLUTION_PROGRESS + "\n\n"
                 "for seed in config.RUN_SEEDS:\n    print(f'--- seed {seed}')\n"
                 "    print(es.run_es(seed=seed, initial=es.trained_baseline(seed), approach='es_seeded',\n"
                 "                    on_generation=progress, n_jobs=config.N_JOBS))"),
    ],
    "06_evaluate.ipynb": [
        ("markdown", "# 6. Evaluation\nTest tracks, test accuracy, size, inference time and training cost, as "
                     "mean ± standard deviation across run seeds (design specification §11)."),
        ("code", SETUP),
        ("code", "from racing import config, evaluate\n\nresults = evaluate.main(n_jobs=config.N_JOBS)\n"
                 "results['comparison']"),
        ("code", "results['driving']"),
        ("code", "results['search_cost']"),
        ("code", "results['driving_selected']"),
        ("code", "results['dataset']"),
        ("code", "import json\n\n# run conditions recorded by scripts/run_notebooks.py (spec §3.3)\n"
                 "if config.RUN_INFO_PATH.exists():\n"
                 "    print(json.dumps(json.loads(config.RUN_INFO_PATH.read_text()), indent=1))"),
        ("code", "from IPython.display import Image, display\n\nfor path in results['figures']:\n"
                 "    display(Image(filename=str(path)))"),
    ],
}
```

Replace `cell` and `main`. Every cell gets an `id` (nbformat 4.5 requires one), and notebooks that are no longer
defined are deleted, so the old `05_evaluate.ipynb` disappears:

```python
def cell(kind, source, index):
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "id": f"cell-{index}", "metadata": {}, "source": lines}
    return {"cell_type": "code", "id": f"cell-{index}", "metadata": {}, "execution_count": None, "outputs": [],
            "source": lines}


def main():
    FOLDER.mkdir(exist_ok=True)
    for stale in FOLDER.glob("*.ipynb"):
        if stale.name not in NOTEBOOKS:
            stale.unlink()
    for name, cells in NOTEBOOKS.items():
        notebook = {"cells": [cell(kind, source, k) for k, (kind, source) in enumerate(cells)],
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python"}},
                    "nbformat": 4, "nbformat_minor": 5}
        (FOLDER / name).write_text(json.dumps(notebook, indent=1) + "\n")
```

Run: `$PY scripts/make_notebooks.py`

- [ ] **Step 5: Rewrite `README.md`**

```markdown
# EC Racing

EC Racing compares methods for finding small neural network driving controllers. Seven controllers race on
procedurally generated tracks in a top-down 2D simulator:

- **Imitation learning:** a hand-designed baseline and three Keras Tuner searches (random search, Hyperband,
  Bayesian optimization) are trained to imitate a scripted expert driver.
- **Neuroevolution:** a genetic algorithm and a (μ+λ) evolution strategy evolve network weights by driving,
  following a curriculum from a circle to fully varied tracks. A level is passed only when the cars finish at pace.
- **Hybrid:** a seeded ES starts from the trained baseline and continues by driving.

Every method runs with three seeds; results are reported as mean ± standard deviation. The design specification
is `docs/2026-10-05-ec-racing-design.md`; the implementation plans are in `docs/`.

## Setup

The project uses the Conda base environment (Python 3.13, TensorFlow 2.21, Keras 3.13):

    ~/miniconda3/bin/python -m pip install keras-tuner pygame pytest nbclient nbformat

numpy, scipy, matplotlib, pandas, joblib and IPython come from the Conda base environment.

## Running the experiment

One command runs the six notebooks in order and saves their outputs into the notebook files:

    ~/miniconda3/bin/python scripts/run_notebooks.py

The run takes several hours. To run it in the background and follow its progress:

    nohup ~/miniconda3/bin/python scripts/run_notebooks.py > /dev/null 2>&1 &
    tail -f results/run_log.txt

After an interruption, resume from a notebook, for example the GA:

    ~/miniconda3/bin/python scripts/run_notebooks.py --from 03

Run the experiment on an otherwise idle machine with a fixed power profile (`powerprofilesctl set performance`),
because training times are reported. The processor, power profile, package versions and notebook times are
recorded in `results/run_info.json`.

| Notebook | Purpose |
|---|---|
| `01_dataset` | Expert dataset |
| `02_imitation` | Baseline, Keras Tuner searches, retraining with each seed |
| `03_ga` | Genetic algorithm, three seeds |
| `04_es` | Evolution strategy, three seeds |
| `05_es_seeded` | Seeded evolution strategy, three seeds |
| `06_evaluate` | Tables and figures |

The notebooks can also be run one at a time in Jupyter or VS Code with the Conda base kernel; the working directory
must be `notebooks/`.

Results are written to `results/`, which is not under version control:

| Path | Contents |
|---|---|
| `results/dataset.npz`, `results/dataset_info.json` | Expert dataset and its collection time |
| `results/runs/<method>/seed<k>/` | Logs, summaries and per-generation best networks of each run |
| `results/runs/<method>/search/` | Keras Tuner search: trials, summary and exported models |
| `results/champions/<method>/seed<k>.keras` | Final controllers (evolutionary ones also as `.npy`) |
| `results/champions/best_seeds.json` | Each method's best seed on the validation tracks |
| `results/tables/`, `results/figures/` | Results tables (CSV and Markdown) and figures |
| `results/run_info.json`, `results/run_log.txt` | Record and log of the experiment run |

## Calibration and benchmarks

    ~/miniconda3/bin/python scripts/pilots.py          # GA mutation range, ES step size, seeded ES
    ~/miniconda3/bin/python -m racing.bench            # generation time per worker count; Keras epoch time

The pilots write to `results/pilots/`. The benchmark results set `N_JOBS`, `STEPS_PER_EXECUTION` and
`JIT_COMPILE` in `racing/config.py`.

## Viewer

    ~/miniconda3/bin/python -m racing.app

- **Training view:** choose an approach, run seed, difficulty and population size, then press Play (or Space).
  Stop ends training so the settings can be changed. The panel names the map being shown. Training started here
  writes only to `results/app_runs/`.
- **Race view:** pick a test track with the arrow keys and a difficulty; the map is shown with the cars on the
  start line. For each method, choose its best seed, a specific seed, or off, then press Start race; Stop returns
  to the start line.
- **Difficulty:** Easy (circle), Medium (oval), Hard and Extra hard (seeded loops).
- **Playback speed:** 1× is twice real time.

## Tests

    ~/miniconda3/bin/python -m pytest              # fast tests
    ~/miniconda3/bin/python -m pytest -m slow      # slow tests: every track seed, Keras Tuner, runner, end-to-end
```

(In the real file, use fenced `bash` code blocks instead of the indented blocks shown here.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_runner.py -q -m "slow or not slow"`
Expected: PASS (the slow runner tests start Jupyter kernels and take about half a minute).

- [ ] **Step 7: Commit**

```bash
git rm -q --cached notebooks/05_evaluate.ipynb
git add scripts/make_notebooks.py scripts/run_notebooks.py tests/test_runner.py README.md notebooks/*.ipynb
git commit -m "Run the six experiment notebooks with one command; README for the revised experiment

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- scripts/make_notebooks.py scripts/run_notebooks.py tests/test_runner.py README.md notebooks/
```

---

### Task 13: End-to-end test

**Files:**
- Modify: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: one slow test that runs the whole pipeline at toy size in a temporary directory.

- [ ] **Step 1: Replace `tests/test_end_to_end.py`**

```python
import json

import pytest

from racing import es, evaluate, expert, ga, imitation, paths


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    runs, champions, data_path = tmp_path / "runs", tmp_path / "champions", tmp_path / "dataset.npz"
    data, _ = expert.collect_dataset(data_path, splits={"train": [(0, 1), (0, 2)], "val": [(80, 4)],
                                                        "test": [(1000, 4)]}, n_jobs=2)
    for seed in (0, 1):
        imitation.train_baseline(data, out_dir=paths.run_dir("baseline", seed, runs), champion_dir=champions,
                                 epochs=2, seed=seed)
    imitation.run_tuner("kt_random", data, out_dir=paths.search_dir("kt_random", runs), max_trials=2, max_epochs=2)
    imitation.retrain("kt_random", data, seed=0, search=paths.search_dir("kt_random", runs),
                      out_dir=paths.run_dir("kt_random", 0, runs), champion_dir=champions, epochs=2)
    ga.run_ga(pop_size=20, kept=5, budget=40, seed=0, out_dir=paths.run_dir("ga", 0, runs), champion_dir=champions)
    es.run_es(mu=4, lam=16, budget=40, seed=0, out_dir=paths.run_dir("es", 0, runs), champion_dir=champions)
    es.run_es(mu=4, lam=16, budget=40, seed=1, initial=es.trained_baseline(1, champion_dir=champions),
              approach="es_seeded", out_dir=paths.run_dir("es_seeded", 1, runs), champion_dir=champions)
    results = evaluate.main(champions_dir=champions, runs_dir=runs, data_path=data_path, out_dir=tmp_path / "out",
                            test_seeds=[1000], val_seeds=[80], n_jobs=2, latency_calls=20)
    assert list(results["comparison"]["Method"]) == ["Baseline", "KT Random", "ES", "GA", "Seeded ES"]
    assert list(results["driving_selected"]["Method"]) == ["KT Random"]
    for name in ("comparison", "driving", "search_cost", "driving_selected", "dataset"):
        assert (tmp_path / "out" / "tables" / f"{name}.md").exists()
    best = json.loads((champions / "best_seeds.json").read_text())
    assert best["baseline"] in (0, 1) and best["es_seeded"] == 1
    assert json.loads((paths.run_dir("ga", 0, runs) / "summary.json").read_text())["evals"] == 40
```

The notebook check moved to `tests/test_runner.py` (Task 12).

- [ ] **Step 2: Run the test**

Run: `$PY -m pytest tests/test_end_to_end.py -m slow -q`
Expected: PASS (about two minutes: it builds the 20 level-4 validation tracks used for champion selection).

- [ ] **Step 3: Commit**

```bash
git add tests/test_end_to_end.py
git commit -m "End-to-end test for the revised pipeline: seeds, retraining, seeded ES, aggregated evaluation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- tests/test_end_to_end.py
```

---

## Controller Steps After the Tasks

These steps run in the controller's session, in order. They are not delegated.

1. **Full test suite.** `$PY -m pytest -q` and `$PY -m pytest -m slow -q`. All must pass before continuing.
2. **Viewer check.** Open `$PY -m racing.app`; confirm both panels fit, Stop works in both views, the race preview
   shows the map before the start, the difficulty changes the map, and GA training at population 100 uses worker
   processes (several Python processes in `ps`). Close it and confirm no worker processes remain.
3. **Benchmarks** (idle machine: no notebook kernel or other heavy process running).
   `$PY -m racing.bench`. Set `N_JOBS` to the fastest worker count (prefer the smaller count when within 5%),
   and `STEPS_PER_EXECUTION` and `JIT_COMPILE` to the fastest Keras setting. Record the measured values in the
   config comments and commit.
4. **Pilots.** `$PY scripts/pilots.py` (about 1–1.5 hours). Choose `GA_S_MAX` by the rule of spec §9.2. Check
   that the ES and seeded ES advance through the levels with the pace factor 1.3, and inspect the strict-success
   and tie fractions (spec §9.3). **Present the pilot table to the project owner and wait for a decision** on
   `GA_S_MAX`, `PACE_FACTOR` and, if σ stalls, the step-size rule. Then set the constants, update spec §9.2, §13
   and the risk list, and commit.
5. **Reported run.** With the owner's go-ahead: power profile `performance`, no other heavy processes, notebooks
   closed in any editor. Start `nohup $PY scripts/run_notebooks.py > /dev/null 2>&1 &` and follow
   `results/run_log.txt`. On failure, fix the cause and resume with `--from NN`.
6. **Documents.** Regenerate the PDFs of the specification and of this plan; update the README if the pilots
   changed any documented value.
7. **Integration.** Run the full suite once more, then merge `build` into `master` (superpowers
   finishing-a-development-branch).
