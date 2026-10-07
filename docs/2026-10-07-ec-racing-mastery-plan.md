# EC Racing Revision 3 (Mastery Training) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train every EC Racing method until it masters freshly generated F1-inspired circuits (≥ 90% of the population succeeding on every map at every level), then push it faster until it cannot improve, with NEAT added as an eighth method, DAgger for the imitation methods, and multi-day runs that checkpoint, resume and report progress live.

**Architecture:** One shared evolution engine (`racing/evolution.py`) owns the run loop: fresh maps, time-trial evaluation, the shared survival rule, the mastery tracker, logging, checkpoints, the status file and final selection. GA, ES (also seeded ES and the imitation speed push) and NEAT are *method* objects that only initialize a population and produce children. Populations expose one small interface (`__len__`, `subset`, `policy`, `driver`), so evaluation, the viewer and the evaluation tables treat dense and NEAT controllers alike.

**Tech Stack:** Python 3.13 (`~/miniconda3/bin/python`), numpy, scipy, pandas, matplotlib, joblib (loky), TensorFlow 2.21 / Keras 3.13 on CPU, keras-tuner 1.4.8, pygame, nbclient/nbformat, pytest.

**Spec:** `docs/2026-10-07-ec-racing-mastery-design.md` (revision 3). Sections of `docs/2026-10-05-ec-racing-design.md` (version 2.0) not changed by revision 3 remain in force. Executors read both.

## Global Constraints

- CPU only; Keras is obtained through `racing.backend.keras_cpu()`.
- Interpreter: `~/miniconda3/bin/python`. No new packages. Do not install or upgrade TensorFlow, Keras or numpy. (The desktop environment is built later from `requirements.txt`, Task 17.)
- Tests must never signal real processes (no `os.kill`/`os.killpg` on real PIDs in tests; inject fakes as the existing tests do).
- No personal names in code, docs, comments, test data or commit messages.
- Commits are path-scoped (`git add <paths>`; `git commit ... -- <paths>`), and every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never use `git add -A`, `git add .`, `git stash`, `git reset` or `git checkout -- .`.
- Work on branch `build`. Push only when the task says so.
- Fast tests: `~/miniconda3/bin/python -m pytest -q` (the `slow` marker is excluded by `pyproject.toml`). Slow tests: `~/miniconda3/bin/python -m pytest -q -m slow`. Every task ends with the fast suite passing.
- Results never depend on `n_jobs`, and a run that is interrupted and resumed produces exactly the results of an uninterrupted run (revision 3 §8.2).
- Match the surrounding code: module docstrings, short docstrings that cite the spec section (`(revision 3 §4.2)` for new behavior, `(spec §x)` for unchanged behavior), the existing comment density, numpy-vectorized style, constants in `racing/config.py`.
- Every number in revision 3 Table 3 lives in `racing/config.py` under the names of the Interface Contract below; code never repeats the literal.
- `tests/test_end_to_end.py`: Task 6 updates its `run_ga`/`run_es` calls, Task 12 its table names.
- Existing tests that encode revision 2.0 behavior that revision 3 removes (fixed training seeds, the level-1 circle, the pace gate, the evaluation budget, `LevelTracker`, `draw_seeds`, `best_gen*.npy`, six notebooks) are updated or deleted in the task that removes the behavior, never skipped.

## Review Focus

The five failure modes revision 3 implies that the per-feature tests do not obviously exercise, most likely first. Each is pinned by a test in the owning task.

1. **Memory over multi-day runs.** Fresh maps every generation must not accumulate in the track cache of the main process or the workers (each built track holds a road grid of a few MB). Expect: memory stays flat; the cache holds at most `config.TRACK_CACHE_SIZE` tracks. *Test in Task 1.*
2. **Interruption between the log row and the checkpoint.** Ctrl+C or a reboot can land after `log.csv` got a row but before the checkpoint was renamed. Expect: resuming truncates `log.csv` to the checkpoint's generation, so no generation appears twice, and the resumed run equals an uninterrupted one. *Test in Task 6.*
3. **Restarting the runner after changing settings or code.** Expect: a run whose recorded settings differ is refused with a message naming the differing settings, and `--fresh` is required; nothing silently mixes old and new results. *Tests in Tasks 4 and 15.*
4. **Monitoring a run from another terminal or under `nohup`.** The status file may be missing, mid-replace or from a finished run; stdout may not be a terminal. Expect: the monitor never crashes on a missing or unreadable status file, and without a terminal it prints whole status lines at most once per period instead of carriage-return updates. *Tests in Task 5.*
5. **A method that never reaches 90%.** Expect: the safety stop ends the run, the summary records the level reached and `goal_reached: false`, a final controller is still selected and saved, and the evaluation tables show "not reached" instead of failing. *Tests in Tasks 3, 6 and 12.*

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `racing/config.py` | All parameters of revision 3 Table 3; `EC_RACING_JOBS`; approaches incl. `neat`; `RUN_ORDER` | 1–3, 6, 7, 11, 15 |
| `racing/track_gen.py` | Easy loops, circuit generator (polygon, main straight, hairpins, chicanes, esses, fillets), random direction | 1 |
| `racing/tracks.py` | Bounded track cache | 1 |
| `racing/sim.py` | `RunResult.laps_done`; `target_time` | 2 |
| `racing/nets.py` | `DensePopulation`; dense helpers; champion loading for dense and NEAT | 2, 8 |
| `racing/population.py` | `TrialResult`; time-trial evaluation per map; final selection; dense champion saving | 2, 6 |
| `racing/curriculum.py` | `SeedSource`, `survivors`, `MasteryTracker`, `mutation_strength`, `StepSize` | 3 |
| `racing/checkpoint.py` (new) | Settings record, atomic pickle checkpoints, completed-run check | 4 |
| `racing/progress.py` (new) | Status file, milestones, live line, `Monitor`, `python -m racing.progress` | 5 |
| `racing/evolution.py` (new) | Shared run loop for every evolutionary method | 6 |
| `racing/ga.py`, `racing/es.py` | GA and ES method objects and `run_ga` / `run_es` wrappers | 6 |
| `racing/neat.py` (new) | Genome, innovations, operators, batched forward pass, Keras export, `NeatNet`, `NeatPopulation`, speciation, `NEAT` method, `run_neat` | 7–9 |
| `racing/expert.py` | Dataset on freshly generated maps | 10 |
| `racing/imitation.py`, `racing/paths.py` | Imitation stage files, skip-when-complete | 10 |
| `racing/dagger.py` (new) | DAgger rounds and the imitation speed push | 11 |
| `racing/evaluate.py` | Revision 3 §7 tables and figures | 12 |
| `racing/netfig.py` (new) | Network figure | 13 |
| `racing/app.py` | NEAT, new reports and stats, seeded ES precondition | 14 |
| `racing/bench.py`, `scripts/run_notebooks.py`, `scripts/make_notebooks.py` | Job counts for the desktop; resume/`--fresh`/live progress; eight notebooks | 15 |
| `scripts/pilots.py` | Revision 3 pilots | 16 |
| `README.md`, `requirements.txt`, `docs/2026-10-05-ec-racing-design.md` (+ PDFs) | Desktop/WSL setup, pinned packages, design spec 3.0 | 17 |

## Interface Contract

Every task's implementer sees only their own task. These names, signatures, fields and semantics are fixed; a task may add private helpers but must not rename anything here.

### `racing/config.py`

```python
# --- Simulation ---
LAPS = 3                                   # unchanged
# --- Track geometry (revision 3 §3) ---
LEVELS = {
    1: {"kind": "loop", "points": (4, 6), "radius": (60.0, 110.0), "min_corner": 45.0, "width": 16.0},
    2: {"kind": "circuit", "points": (6, 8), "radius": (60.0, 170.0), "slow": (25.0, 45.0), "fast": (50.0, 80.0),
        "main": 0.0, "hairpins": 0, "chicanes": (0, 0), "esses": (0, 0), "min_corner": 24.0, "width": 14.0},
    3: {"kind": "circuit", "points": (9, 12), "radius": (70.0, 220.0), "slow": (16.0, 35.0), "fast": (45.0, 90.0),
        "main": 170.0, "hairpins": 1, "chicanes": (0, 1), "esses": (0, 1), "min_corner": 14.0, "width": 12.0},
    4: {"kind": "circuit", "points": (13, 17), "radius": (80.0, 260.0), "slow": (10.0, 30.0), "fast": (45.0, 100.0),
        "main": 210.0, "hairpins": 2, "chicanes": (1, 2), "esses": (1, 2), "min_corner": 9.0, "width": 10.0},
}
CIRCUIT_STRETCH = (1.0, 1.5)
DIRECTION_SALT = 7               # per-map direction draw: default_rng([seed, level, DIRECTION_SALT])
TRACK_CACHE_SIZE = 64            # built tracks kept per process
# --- Seeds (revision 3 §3.4) ---
VAL_SEEDS = range(80, 100)       # unchanged values
TEST_SEEDS = range(1000, 1020)   # unchanged values
DATASET_SEED_RANGE = (1_000_000, 2_000_000)
DATASET_TRACKS = {1: 1, 2: 20, 3: 20, 4: 80}
TRAIN_SEED_START = 2_000_000
RUN_SEEDS = (0, 1, 2); SEARCH_SEED = 0   # unchanged
# --- Time trial, survival and levels (revision 3 §4) ---
TRACKS_PER_GEN = 3
TARGET_FACTOR = 1.3              # starting target: TARGET_FACTOR × three expert laps
SPEED_STEP = 0.95
MASTERY_FRACTION = 0.9
MASTERY_STREAK = 3
SPEED_PATIENCE = 500
SAFETY_PATIENCE = 2000
KEEP_MAX_FRACTION = 0.9
ELITE_FLOOR = 50
# --- GA ---  GA_POP = 1000; GA_TOURNAMENT = 5; GA_CROSSOVER_P = 0.5; GA_MUTATION_P = 0.1; GA_S_MAX = 0.1; GA_S_MIN = GA_S_MAX / 100
# --- ES ---  ES_POP = 550; ES_MU = 50 (parent pool); ES_SIGMA0 = 0.1; ES_FACTOR = 0.82; ES_SIGMA_MIN = 1e-4; ES_SIGMA_MAX = 1.0
# --- NEAT (revision 3 §5.4) ---
NEAT_POP = 1000; NEAT_CROSSOVER = 0.75; NEAT_TOURNAMENT = 3; NEAT_DISABLE_INHERIT = 0.75
NEAT_PERTURB = 0.8; NEAT_REPLACE = 0.05; NEAT_ADD_CONN = 0.05; NEAT_ADD_NODE = 0.03; NEAT_TOGGLE = 0.01
NEAT_SIGMA0 = 0.5; NEAT_C1 = 1.0; NEAT_C2 = 1.0; NEAT_C3 = 0.4; NEAT_SMALL_GENOME = 20
NEAT_THRESHOLD0 = 3.0; NEAT_THRESHOLD_STEP = 0.1; NEAT_TARGET_SPECIES = 20; NEAT_STAGNATION = 20
# --- DAgger (revision 3 §6.3) ---
DAGGER_MAPS = 20; DAGGER_EPOCHS = 30; DAGGER_HOLDOUT = 0.1; DAGGER_GOAL = 18; DAGGER_STREAK = 3; DAGGER_PATIENCE = 10
DAGGER_SALT = 31                 # a refinement's random generator: default_rng([run seed, method index, DAGGER_SALT])
# --- Parallelism ---
N_JOBS = int(os.environ.get("EC_RACING_JOBS", "6"))
# --- Approaches ---
APPROACHES = ("baseline", "kt_random", "kt_hyperband", "kt_bayesian", "es", "ga", "es_seeded", "neat")
TUNER_APPROACHES = ("kt_random", "kt_hyperband", "kt_bayesian")
IMITATION_APPROACHES = ("baseline", *TUNER_APPROACHES)
EVOLUTION_APPROACHES = ("ga", "es", "es_seeded", "neat")
LABELS["neat"] = "NEAT"; COLORS["neat"] = "#F0E442"
RUN_ORDER = tuple((a, s) for a in (*IMITATION_APPROACHES, *EVOLUTION_APPROACHES) for s in RUN_SEEDS)   # 24 runs
# --- Paths ---
STATUS_PATH = RESULTS_DIR / "status.json"
```

Removed from `config.py` (and every use): `PACE_FACTOR`, `LEVEL_UP_FRACTION`, `LEVEL_UP_STREAK`, `GA_KEPT`, `GA_KEEP_MAX_FRACTION`, `EVAL_BUDGET`, `ES_LAMBDA`, `TRAIN_SEEDS`, `TRAIN_DATA_TRACKS`. Task 1 adds the track values, Task 2 the time-trial values, Task 3 the survival/level values, Task 6 GA/ES changes, Task 7 NEAT values and approaches, Task 11 DAgger values, Task 15 `N_JOBS`, `STATUS_PATH` and `RUN_ORDER` (Task 5 may add `STATUS_PATH` and `RUN_ORDER` earlier if it needs them; whichever task comes first adds them).

### `racing/track_gen.py` (Task 1)

```python
def loop_points(rng, n_points, radius_range, step=0.5) -> np.ndarray      # unchanged
def fillet_path(poly, radii, step=0.5) -> np.ndarray                      # (M, 2) closed path, first point not repeated
def circuit_points(rng, spec, step=0.5) -> np.ndarray                     # revision 3 §3.2 steps 1-6
def runs_clockwise(seed, level) -> bool                                   # default_rng([seed, level, DIRECTION_SALT]).random() < 0.5
def geometry_ok(track, min_corner) -> bool                                # unchanged
def candidate_track(seed, level, attempt=0) -> Track | None               # loop or circuit; reversed when runs_clockwise
def make_track(seed, level) -> Track                                      # unchanged
```
`circle_points` and `oval_points` are deleted. `racing/tracks.get_track` uses `functools.lru_cache(maxsize=config.TRACK_CACHE_SIZE)`.

### `racing/sim.py` (Task 2)

```python
RunResult.laps_done: np.ndarray            # (n,) int, laps completed when the run ended
def target_time(track, factor, laps=config.LAPS) -> float   # factor * track.expert_lap_time * laps
```

### `racing/nets.py` (Tasks 2 and 8)

```python
class DensePopulation:
    def __init__(self, weights, hidden=config.BASELINE_HIDDEN, activation="relu")   # weights (n, P) float64
    weights: np.ndarray; hidden: tuple; activation: str
    def __len__(self) -> int
    def subset(self, indices) -> "DensePopulation"
    def policy(self)                        # sim policy: car i driven by row i
    def driver(self, i) -> DenseNet
def dense_spec(model) -> tuple[tuple, str]  # (hidden sizes, hidden activation) of a dense Keras model
def load_champion(approach, seed, directory=None) -> tuple[keras.Model, driver]
    # seed<k>.json is a NEAT genome -> neat.NeatNet; dense evolved: .npy with nets.dense_spec(model); otherwise from_keras
def champion_seeds(directory=None) -> dict   # unchanged (finds .keras files)
```
Every population type (DensePopulation, `neat.NeatPopulation`) has exactly: `__len__`, `subset(indices)`, `policy()`, `driver(i)`. Every driver (`DenseNet`, `neat.NeatNet`) has `act(obs) -> np.ndarray[int]`, `policy()`, `n_params`.

### `racing/population.py` (Tasks 2 and 6)

```python
@dataclass(eq=False)
class TrialResult:
    fitness: np.ndarray        # (n,) mean fitness over the maps (time limit = target time)
    success: np.ndarray        # (maps, n) bool: three laps within the target time
    laps: np.ndarray           # (maps, n) int: laps completed within the target time
    finish_time: np.ndarray    # (maps, n) s, nan unless success
    all_success -> (n,) bool   # property: success on every map
    lap_share -> (n,) float    # property: laps.sum(axis=0) / (config.LAPS * maps)
    map_shares -> (maps,) float  # property: success.mean(axis=1)
    min_share -> float         # property: map_shares.min()
    lap_times -> (maps, n)     # property: finish_time / config.LAPS
def split_tasks(n, seeds, n_jobs)                                             # unchanged
def evaluate_population(population, seeds, level, factor, n_jobs=1, backend="loky") -> TrialResult
def fresh_workers(n_jobs)                                                     # unchanged
def select_final(population, candidates, factor, n_jobs=1, seeds=config.VAL_SEEDS) -> tuple[int, int, float]
    # (index into population, validation successes, mean validation fitness); most successes, then fitness
def save_dense_champion(approach, seed, population, index, champion_dir=config.CHAMPIONS_DIR) -> Path
    # .keras via nets.to_keras(row, population.hidden, population.activation) + float64 .npy (no .json: a .json
    # next to a controller always means a NEAT genome; the dense architecture is read back with nets.dense_spec)
def clear_champion(approach, seed, champion_dir=config.CHAMPIONS_DIR)       # removes .keras, .npy and .json
```
`RunLogger`, `make_report`, `finish_fractions`, `select_champion`, `save_champion` (old signatures) move into or are replaced by `racing/evolution.py` (Task 6).

### `racing/curriculum.py` (Task 3)

```python
class SeedSource:
    def __init__(self, rng, start=config.TRAIN_SEED_START, used=())
    def draw(self, count) -> list[int]          # distinct, never previously drawn, in [start, 2**31 - 1)
    def state(self) -> list[int]                # the used seeds (sorted)
def survivors(rng, fitness, all_success, lap_share, cap, floor) -> np.ndarray   # int64 indices, revision 3 §4.2
    # successes fittest first (≤ cap); then each non-success kept with probability lap_share (one rng.random(n) draw),
    # fittest first, until cap; then, if fewer than min(floor, cap), the fittest remaining until min(floor, cap).
    # Order: successes, chance, floor.
EVENTS = ("level_up", "goal", "tightened", "end_speed", "end_stalled")
class MasteryTracker:
    def __init__(self, level=1, factor=config.TARGET_FACTOR)
    level: int; factor: float; streak: int; goal_reached: bool; mastered_factor: float (nan until the goal)
    outcome: str | None        # None, "speed_limit" or "stalled"
    def update(self, min_share) -> str | None   # one of EVENTS or None
    done -> bool                                # property: outcome is not None
    target_reduction -> float                   # property: 1 - mastered_factor / TARGET_FACTOR (nan before the goal)
    def state(self) -> dict; @classmethod from_state(cls, state) -> MasteryTracker
def mutation_strength(child_success, s_max=config.GA_S_MAX, s_min=config.GA_S_MIN) -> float   # s_min + (s_max - s_min)(1 - c); s_max if c is nan
class StepSize:
    def __init__(self, sigma0)
    sigma: float
    def update(self, parent_success, child_success, success_rate) -> float   # revision 3 §5.2 two-phase rule
    def reset(self) -> None                                                  # back to sigma0
    def state(self) -> dict; @classmethod from_state(cls, state) -> StepSize
```
`MasteryTracker.update` semantics (revision 3 §4.3–4.5): a generation passes when `min_share >= MASTERY_FRACTION`. Before the goal: a pass increments `streak`, a failure resets it; at `MASTERY_STREAK` on level < `MAX_LEVEL` → `level += 1`, `streak = 0`, safety counters reset, return `"level_up"`; on `MAX_LEVEL` → `goal_reached = True`, `mastered_factor = factor`, `factor *= SPEED_STEP`, `streak = 0`, return `"goal"`. Safety: the mean of the last three `min_share` values (fewer at the start of a level) is compared with the level's best; `SAFETY_PATIENCE` consecutive generations without a strictly higher mean → `outcome = "stalled"`, return `"end_stalled"`. After the goal: a pass streak of `MASTERY_STREAK` → `mastered_factor = factor`, `factor *= SPEED_STEP`, return `"tightened"`; `SPEED_PATIENCE` consecutive generations since the last mastery (goal or tightening) → `outcome = "speed_limit"`, return `"end_speed"`. When an update would both master and end, mastery wins.

`StepSize.update`: if `parent_success >= MASTERY_FRACTION` (reliability phase) → `sigma *= ES_FACTOR` if `child_success < MASTERY_FRACTION` else `sigma /= ES_FACTOR`; otherwise the 1/5 rule (`success_rate > 0.2` → `sigma /= ES_FACTOR`; `< 0.2` → `sigma *= ES_FACTOR`; nan → unchanged). Clipped to `[ES_SIGMA_MIN, ES_SIGMA_MAX]`. `success_fraction(child_fitness, parent_fitness)` (tie = half) moves from `es.py` to `curriculum.py`.

`LevelTracker` and `draw_seeds` are deleted.

### `racing/checkpoint.py` (Task 4)

```python
CHECKPOINT_NAME = "checkpoint.pkl"
SETTINGS_KEYS: tuple[str, ...]    # config names that affect results and exist by Task 4 (LEVELS, LAPS, track, time-trial,
                                  # survival, level, GA and ES values, MAX_EPOCHS, RUN_SEEDS, ...); Task 7 appends the NEAT_*
                                  # names and Task 11 the DAGGER_* names. N_JOBS is never included: results do not depend on it
                                  # Every checkpoint state and every summary (summary.json, dagger.json) carries "settings".
class SettingsMismatch(RuntimeError)          # message names each differing key with recorded and current values
def settings_record() -> dict                 # {name: value} normalized by a JSON round trip, so a summary loaded
                                              # from JSON compares equal (int dict keys, ranges and tuples)
def check_settings(recorded) -> None          # raises SettingsMismatch
def save(path, state) -> None                 # pickle to <path>.tmp-<pid>, then os.replace
def load(path) -> dict | None                 # None if missing
def completed(run_dir, name="summary.json") -> dict | None
    # the summary if it exists and its "settings" equal settings_record(); SettingsMismatch if they differ
```

### `racing/progress.py` (Task 5)

```python
def write_status(status, path=None) -> None   # atomic JSON; adds "updated" (ISO time); path defaults to config.STATUS_PATH
def read_status(path=None) -> dict | None     # None if missing or unreadable
def milestone(message, path=None) -> str      # appends "<ISO time>  <message>" to config.RUN_LOG_PATH; returns the line
def run_number(approach, seed) -> tuple[int, int]   # 1-based position in config.RUN_ORDER and its length
def bar(fraction, width=20) -> str            # block characters with eighth-steps, e.g. "███████████▍        "
def format_status(status) -> str
    # "[run 9/24] GA seed 1 · Hard · gen 412 · worst map 84% / 90% <bar> · streak 1/3 · 2h 13m"
    # speed push: "... · Extra hard · speed push −14.3% · gen ..."; DAgger: "... · DAgger round 7 · checks 16/20 · streak 1/3 · ..."
class Monitor:
    def __init__(self, status_path=None, log_path=None, out=None, interval=1.0, live=None, period=60.0)
        # live=None: live updating if out.isatty(); otherwise one status line per `period` seconds
    def poll(self) -> None      # print new run-log lines as permanent lines, then refresh the status line
    def start(self) -> None     # background thread calling poll every interval
    def stop(self) -> None      # final poll, then a newline if a live line is showing
def main(argv=None) -> int      # python -m racing.progress: Monitor in the foreground until Ctrl+C
```
Status dict keys written by runs: `approach`, `seed`, `stage` (`"evolution"`, `"speed push"`, `"dagger"`), `level`, `generation` (or `round` for DAgger), `min_share`, `streak`, `target_reduction`, `checks` (DAgger), `elapsed` (s). A DAgger status has `round` and `checks` and no `generation` or `min_share`; `format_status` handles both shapes.

### `racing/evolution.py` (Task 6)

```python
class Method(Protocol):
    approach: str
    population: DensePopulation | NeatPopulation      # current population; kept cars first
    is_kept: np.ndarray                               # bool (n,), True for cars carried over unchanged
    step_size: float                                  # logged; GA strength, ES/NEAT sigma
    def initialize(self, rng) -> None
    def advance(self, rng, result: TrialResult, kept: np.ndarray, event: str | None) -> None
        # replace population: population.subset(kept) first, then children; adapt step size; on "level_up" reset it
    def state(self) -> dict
    def load_state(self, state) -> None
    def save_champion(self, index, seed, champion_dir) -> Path
    def log_fields(self) -> dict                      # extra log columns, e.g. {"species": 17}; {} if none
def run(method, seed, out_dir, champion_dir=config.CHAMPIONS_DIR, n_jobs=1, on_generation=None, start_level=1,
        max_generations=None, rng_seed=None) -> dict
    # resumable loop (revision 3 §4, §8.2); writes log.csv, checkpoint.pkl, summary.json in out_dir; status + milestones
    # returns the existing summary at once when checkpoint.completed(out_dir) finds a completed run with matching settings
LOG_FIELDS = ["generation", "level", "factor", "evals", "best", "mean", "worst", "map_shares", "min_share", "n_kept",
              "kept_success", "child_success", "median_lap", "step_size", "streak", "event", "seeds", "elapsed"]
```
Report passed to `on_generation` (and to the viewer through its queue): `approach, seed, generation, level, factor, target_reduction, streak, seeds, population, fitness, all_success, is_kept, map_shares, min_share, kept_success, child_success, median_lap, step_size, best, mean, worst, evals, elapsed, event`, plus `method.log_fields()`.

The engine calls `population.evaluate_population` and `population.select_final` through the module (`from racing import population`), so tests monkeypatch one place.

Summary (`summary.json`): `approach, seed, settings, pop_size` (len of the initial population), `generations, evals, wall_time` (active seconds, accumulated across resumptions), `level_reached, goal_reached, outcome, mastered_factor, target_reduction, levels` (list of `{"level", "generation", "evals", "elapsed"}` for each level cleared and the goal), `start_level, val_successes, val_fitness, champion_path` (null when `champion_dir` is None), `tightenings` (list of `{"factor", "generation", "evals", "elapsed"}`). In `levels`, an entry `{"level": L, ...}` is appended when level L (1–3) is cleared and `{"level": MAX_LEVEL, ...}` when the mastery goal is reached, so the goal entry is the one with `level == MAX_LEVEL`.

### `racing/ga.py`, `racing/es.py` (Task 6)

```python
class GA(Method):  __init__(self, pop_size=config.GA_POP, hidden=config.BASELINE_HIDDEN, activation="relu", s_max=config.GA_S_MAX)
def run_ga(seed=0, pop_size=config.GA_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1,
           on_generation=None, start_level=1, max_generations=None, s_max=config.GA_S_MAX) -> dict
class ES(Method):  __init__(self, approach="es", pop_size=config.ES_POP, parents=config.ES_MU, hidden=config.BASELINE_HIDDEN,
                            activation="relu", initial=None, sigma0=config.ES_SIGMA0)
def run_es(seed=0, pop_size=config.ES_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1, on_generation=None,
           start_level=1, max_generations=None, initial=None, approach="es", hidden=config.BASELINE_HIDDEN,
           activation="relu") -> dict
def dagger_baseline(seed, runs_dir=None) -> np.ndarray     # flat weights of runs/baseline/seed<k>/dagger.keras (es.py)
```

### `racing/neat.py` (Tasks 7–9)

```python
INPUT_IDS = range(0, 10); OUTPUT_IDS = range(10, 25); FIRST_HIDDEN = 25
@dataclass(eq=False)
class Genome:
    node_ids: np.ndarray   # int64, sorted: output and hidden node ids (inputs are implicit)
    bias: np.ndarray       # float64, aligned with node_ids
    innov: np.ndarray      # int64, sorted connection innovation numbers
    src: np.ndarray; dst: np.ndarray   # int64 node ids, aligned with innov
    weight: np.ndarray     # float64
    enabled: np.ndarray    # bool
    def copy(self) -> "Genome"
    n_params -> int        # property: enabled connections + len(bias)
    def to_dict(self) -> dict; @classmethod from_dict(cls, d) -> Genome
class Innovations:
    def __init__(self, next_innov=0, next_node=FIRST_HIDDEN, conns=None, splits=None)
    def conn(self, src, dst) -> int           # same (src, dst) -> same number for the whole run
    def split(self, innov) -> int             # same split connection -> same node id for the whole run
    def state(self) -> dict; @classmethod from_state(cls, state) -> Innovations
def initial_genome(rng, innovations) -> Genome
def creates_cycle(genome, src, dst) -> bool
def mutate(rng, genome, innovations, sigma) -> Genome      # new genome; revision 3 §5.4 mutations
def crossover(rng, fitter, other) -> Genome
def distance(a, b) -> float
def node_depths(genome) -> dict[int, int]                   # longest enabled path from an input; inputs 0
class NeatNet:   __init__(self, genome); genome; act(obs); policy(); n_params; @classmethod from_json(cls, path)
class NeatPopulation:  __init__(self, genomes); __len__; subset(indices); policy(); driver(i) -> NeatNet; genomes: list
def compile_population(genomes) -> Compiled              # flat arrays grouped by depth
def forward(compiled, obs) -> np.ndarray                 # (n, 15) output values
def to_keras(genome) -> keras.Model                      # revision 3 §5.4 saved controller
def save_neat_champion(approach, seed, genome, champion_dir) -> Path   # .json genome + .keras
class Species: id, representative (Genome), members (list[int]), best_fitness, stale
class SpeciesSet:
    def __init__(self, threshold=config.NEAT_THRESHOLD0)
    def speciate(self, rng, genomes, fitness) -> None     # assign, update stagnation, adjust threshold
    def allot(self, n_children, fitness) -> dict[int, int] # species id -> children (largest remainder), stagnant get 0 unless best
    def state(self) -> dict; @classmethod from_state(cls, state) -> SpeciesSet
class NEAT(Method): __init__(self, pop_size=config.NEAT_POP)
def run_neat(seed=0, pop_size=config.NEAT_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1,
             on_generation=None, start_level=1, max_generations=None) -> dict
```

### `racing/paths.py`, `racing/imitation.py`, `racing/expert.py` (Task 10)

```python
def stage_path(approach, seed, stage, runs_dir=None) -> Path   # <runs>/<approach>/seed<k>/<stage>.keras, stage in ("imitation", "dagger")
def speed_dir(approach, seed, runs_dir=None) -> Path           # <runs>/<approach>/seed<k>/speed
# imitation.train_baseline / retrain: save the trained model to stage_path(..., "imitation") (no champion);
#   new parameter resume=True: return the existing summary when checkpoint.completed(out_dir) finds a matching one.
# imitation.run_tuner: new parameter resume=True: return the existing search summary when complete with matching settings.
# The viewer passes resume=False and clears its own app_runs/<approach>/seed<k> directory before training.
def dataset_seeds() -> list[tuple[int, int]]                    # expert.py: (seed, level) of the 121 dataset maps
# expert.default_splits(): {"train": dataset_seeds(), "val": [(s, 4) for VAL_SEEDS], "test": [(s, 4) for TEST_SEEDS]}
```

### `racing/dagger.py` (Task 11)

```python
def drive_and_label(driver, seed, level=config.MAX_LEVEL) -> tuple[np.ndarray, np.ndarray, int]   # X, y, respawns
def collect(driver, seeds, level=config.MAX_LEVEL, n_jobs=1) -> tuple[np.ndarray, np.ndarray, int]
def check(driver, seeds, factor=config.TARGET_FACTOR, level=config.MAX_LEVEL, n_jobs=1) -> int   # successes
def refine(approach, seed, data, runs_dir=None, n_jobs=1, on_round=None, max_rounds=None) -> dict
    # resumable; writes <run>/dagger.keras, <run>/dagger_rows.npz, <run>/dagger_state.pkl, <run>/dagger.json
def speed_push(approach, seed, runs_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1, on_generation=None,
               max_generations=None) -> dict          # ES from dagger.keras, start_level=MAX_LEVEL, out_dir=speed_dir
def run(approach, seed, data, runs_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1) -> dict   # refine + speed push
    # returns {"approach", "seed", "dagger": <refine summary>, "speed": <speed-push summary>}
```

### `racing/evaluate.py` (Task 12), `racing/netfig.py` (Task 13)

```python
DRIVE_COLUMNS = ["approach", "run_seed", "track", "finished", "success", "crashed", "race_time", "lap_time", "progress", "fitness"]
def driving_results(drivers, track_seeds, level=config.MAX_LEVEL, factor=config.TARGET_FACTOR, n_jobs=1, backend="loky") -> DataFrame
def best_seeds(validation) -> dict            # most validation successes, then mean fitness, then lowest seed
def results_table(per_track, runs_dir) -> DataFrame
def head_to_head_table(per_track) -> DataFrame
def training_cost_table(runs_dir) -> DataFrame
def stages_table(stage_results) -> DataFrame
def comparison_table(metrics) -> DataFrame    # course comparison; NEAT params from the genome
def figures(runs_dir, out_dir) -> list[Path]  # success_over_training, lap_time_over_training, step_sizes, kt_trials
def main(champions_dir=None, runs_dir=None, data_path=None, out_dir=None, test_seeds=config.TEST_SEEDS,
         val_seeds=config.VAL_SEEDS, n_jobs=1, latency_calls=1000) -> dict
    # keys: "results", "head_to_head", "training_cost", "stages", "comparison", "search_cost", "driving_selected",
    # "dataset" (DataFrames, also tables/<name>.csv and .md), "per_track", "validation", "metrics", "best_seeds", "figures"
# netfig.py
def network_graph(driver) -> dict             # {"columns": [[node labels...], ...], "edges": [(col_a, i, col_b, j, w), ...]}
def draw_networks(drivers, path) -> Path      # drivers: {approach: driver}; one panel per approach
```

## Tasks

| # | Task | Depends on |
|---|---|---|
| 1 | Circuit track generator, random direction, bounded track cache | — |
| 2 | Time-trial evaluation (`laps_done`, `target_time`, `DensePopulation`, `TrialResult`) | 1 |
| 3 | Survival rule, fresh seeds, mastery tracker, step sizes | 2 |
| 4 | Checkpoints and settings record | — |
| 5 | Progress: status file, milestones, live line, monitor | 4 |
| 6 | Evolution engine with GA and ES | 2–5 |
| 7 | NEAT genome, innovations and operators | 4, 5 |
| 8 | NEAT batched forward pass, Keras export, drivers | 7, 2 |
| 9 | NEAT speciation, method and `run_neat` | 6, 8 |
| 10 | Imitation dataset on fresh maps, stage files, skip-when-complete | 1, 3, 4 |
| 11 | DAgger refinement and imitation speed push | 3, 4, 5, 6, 10 |
| 12 | Evaluation tables and figures | 6, 9, 11 |
| 13 | Network figure | 8 |
| 14 | Viewer | 6, 9 |
| 15 | Runner, notebooks, bench, worker count | 5, 9, 11, 12 |
| 16 | Pilots | 6, 9 |
| 17 | Documentation, requirements, design spec 3.0 | all |

---

### Task 1: Circuit track generator, random direction, bounded track cache

Implements revision 3 §3.1–3.3 and Review Focus 1. Easy becomes random gentle loops (no longer one fixed circle);
Medium, Hard and Extra hard use the circuit generator ported from the approved prototype (`f1_points` / `fillet_path`
in the brainstorming prototype); every map runs clockwise or counter-clockwise, drawn from its seed. The track cache is
bounded because training now builds three fresh maps every generation.

Calibration evidence (prototype of exactly this code, 50 seeds per level, expert-checked): every seed of every level
was accepted and completed by the expert; mean attempts per map 3.9 (Easy, max 14), 1.0 (Medium), 1.2 (Hard), 1.5
(Extra hard, max 6); lengths 458–632 m (Easy), 589–988 m (Medium), 798–1655 m (Hard), 1174–2115 m (Extra hard); 24–29
of 50 maps clockwise per level.

**Files:**
- Modify: `racing/config.py` (the `LEVELS` block under `# --- Track geometry (spec §4) ---`, plus three new constants)
- Modify: `racing/tracks.py` (the `lru_cache` decorator of `get_track` and its docstring)
- Replace: `racing/track_gen.py` (whole file)
- Replace: `tests/test_track_gen.py` (whole file)

**Interfaces:**
- Consumes: `racing.track.build_track(points, width, seed, level)` (unchanged), `config.OVERLAP_MARGIN`,
  `config.MAX_TRACK_ATTEMPTS` (unchanged).
- Produces: `config.LEVELS` (revision 3 Table 1, keys exactly as in the Interface Contract), `config.CIRCUIT_STRETCH`,
  `config.DIRECTION_SALT`, `config.TRACK_CACHE_SIZE`; `track_gen.loop_points`, `track_gen.fillet_path(poly, radii,
  step=0.5)`, `track_gen.circuit_points(rng, spec, step=0.5)`, `track_gen.runs_clockwise(seed, level)`,
  `track_gen.geometry_ok`, `track_gen.candidate_track(seed, level, attempt=0)`, `track_gen.make_track(seed, level)`;
  `tracks.get_track` cached with `maxsize=config.TRACK_CACHE_SIZE`. Helpers `edge_lengths` and `insert_on_edge` are
  module-private in spirit (no other task uses them). `circle_points` and `oval_points` are deleted.

Existing tests that encode removed behavior, all in `tests/test_track_gen.py` and replaced wholesale in Step 1:
`test_level1_is_the_specified_circle` (the level-1 circle), `test_level2_oval_dimensions` (the level-2 oval),
`test_every_seed_generates` (iterates `config.TRAIN_SEEDS`, which Task 10 removes). `tests/conftest.py`'s
`circle_track` fixture builds its circle directly with `build_track` and stays. `tests/test_expert.py` keeps
passing unchanged (its slow `test_expert_drives_every_seed` still uses `TRAIN_DATA_TRACKS`, which Task 10 replaces).
Do not remove `TRAIN_SEEDS` or `TRAIN_DATA_TRACKS` here.

- [ ] **Step 1: Write the failing tests**

Replace the whole of `tests/test_track_gen.py` with:

```python
import types

import numpy as np
import pytest
from scipy.spatial import cKDTree

from racing import config, track_gen, tracks
from racing.track_gen import candidate_track, circuit_points, fillet_path, make_track, runs_clockwise


def slow_corners(track, radius=40.0):
    """Number of separate stretches of the centerline tighter than `radius`."""
    tight = np.abs(track.curvature) > 1.0 / radius
    return int(np.sum(tight & ~np.roll(tight, 1)))


def test_same_seed_gives_same_track():
    assert np.array_equal(make_track(5, 4).centerline, make_track(5, 4).centerline)


def test_different_seeds_give_different_tracks():
    assert make_track(1, 4).length != make_track(2, 4).length


def test_easy_maps_are_gentle_loops_that_differ_by_seed():
    spec = config.LEVELS[1]
    maps = [make_track(seed, 1) for seed in range(4)]
    assert len({round(t.length, 3) for t in maps}) == 4                 # no longer one fixed circle
    for t in maps:
        assert t.width == spec["width"] == 16.0
        assert np.max(np.abs(t.curvature)) <= 1 / spec["min_corner"] + 1e-9


def test_direction_is_drawn_per_map_and_fixed_by_its_seed():
    for level in (1, 2, 3, 4):
        clockwise = [runs_clockwise(seed, level) for seed in range(200)]
        assert 0.35 < np.mean(clockwise) < 0.65
        assert clockwise == [runs_clockwise(seed, level) for seed in range(200)]


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_track_turns_the_way_its_direction_says(level):
    for seed in range(6):
        t = make_track(seed, level)
        turning = float(np.sum(t.curvature) * t.spacing)          # +2π counter-clockwise, -2π clockwise
        assert np.isclose(abs(turning), 2 * np.pi, atol=0.05)
        assert (turning < 0) == runs_clockwise(seed, level)


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_maps_respect_width_and_minimum_corner_radius(level):
    spec = config.LEVELS[level]
    for seed in range(8):
        t = make_track(seed, level)
        assert t.width == spec["width"]
        assert np.max(np.abs(t.curvature)) <= 1 / spec["min_corner"] + 1e-9


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_maps_never_overlap_themselves(level):
    for seed in range(8):
        t = make_track(seed, level)
        pairs = cKDTree(t.centerline).query_pairs(t.width + config.OVERLAP_MARGIN, output_type="ndarray")
        if len(pairs) == 0:
            continue
        gap = np.abs(pairs[:, 0] - pairs[:, 1]) * t.spacing
        along = np.minimum(gap, t.length - gap)
        assert np.all(along <= 3 * (t.width + config.OVERLAP_MARGIN))


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_map_lengths_are_reasonable(level):
    for seed in range(5):
        assert 300.0 < make_track(seed, level).length < 2500.0


def test_harder_circuits_have_more_slow_corners():
    def mean_slow(level):
        return np.mean([slow_corners(make_track(seed, level)) for seed in range(8)])
    assert mean_slow(2) < mean_slow(3) < mean_slow(4)
    assert mean_slow(4) >= 8


def test_fillet_path_rounds_a_square_with_the_given_radius():
    square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    path = fillet_path(square, [20.0] * 4)
    assert np.all(np.linalg.norm(np.diff(np.vstack([path, path[:1]]), axis=0), axis=1) < 0.6)   # about 0.5 m apart
    cut = np.min(np.linalg.norm(path - np.array([100.0, 0.0]), axis=1))
    assert np.isclose(cut, 20.0 * np.sqrt(2) - 20.0, atol=0.05)   # a 20 m arc cuts a right-angle corner by r(√2 - 1)
    assert path.min() >= -1e-9 and path.max() <= 100.0 + 1e-9


def test_fillet_radius_shrinks_on_short_edges():
    thin = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]])
    path = fillet_path(thin, [100.0] * 4)                        # cannot hold 100 m arcs: at most 0.48 × 30 m
    assert np.all(np.abs(path) <= 30.0 + 1e-9) and len(path) > 50


def test_circuit_points_follow_the_spec_and_the_generator():
    spec = config.LEVELS[4]
    a = circuit_points(np.random.default_rng([3, 4, 0]), spec)
    b = circuit_points(np.random.default_rng([3, 4, 0]), spec)
    assert np.array_equal(a, b) and a.shape[1] == 2 and len(a) > 1000


def test_some_candidates_are_valid():
    assert any(candidate_track(0, 4, a) is not None for a in range(20))


def test_track_cache_is_bounded(monkeypatch):
    """Review focus 1: fresh maps every generation must not accumulate in the cache."""
    monkeypatch.setattr(track_gen, "candidate_track",
                        lambda seed, level, attempt=0: types.SimpleNamespace(seed=seed, expert_lap_time=float("nan")))
    from racing import expert
    monkeypatch.setattr(expert, "expert_lap_time", lambda track: 10.0)
    tracks.get_track.cache_clear()
    try:
        for seed in range(10**9, 10**9 + config.TRACK_CACHE_SIZE + 10):
            tracks.get_track(seed, 1)
        info = tracks.get_track.cache_info()
        assert info.maxsize == config.TRACK_CACHE_SIZE and info.currsize == config.TRACK_CACHE_SIZE
    finally:
        tracks.get_track.cache_clear()                        # no fake tracks for later tests


@pytest.mark.slow
def test_every_level_generates_drivable_maps():
    """Calibration (revision 3 §3.1): 50 seeds per level are accepted and completed by the expert."""
    for level in (1, 2, 3, 4):
        for seed in range(50):
            assert tracks.get_track(seed, level).expert_lap_time > 0
    for seed in list(config.VAL_SEEDS) + list(config.TEST_SEEDS):
        assert tracks.get_track(seed, 4).expert_lap_time > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_track_gen.py -q`
Expected: collection error, `ImportError: cannot import name 'circuit_points' from 'racing.track_gen'`.

- [ ] **Step 3: Set the revision 3 level table and the new track constants**

In `racing/config.py`, replace

```python
LEVELS = {
    1: {"kind": "circle", "radius": 50.0, "width": 16.0},
    2: {"kind": "oval", "straight": (80.0, 160.0), "radius": (25.0, 40.0), "width": 14.0},
    3: {"kind": "loop", "points": (6, 8), "radius": (80.0, 160.0), "min_corner": 40.0, "width": 14.0},
    4: {"kind": "loop", "points": (10, 16), "radius": (60.0, 200.0), "min_corner": 15.0, "width": 12.0},
}
MAX_LEVEL = 4
```

with

```python
LEVELS = {                   # revision 3 Table 1: Easy loops, circuits with F1-style features above it
    1: {"kind": "loop", "points": (4, 6), "radius": (60.0, 110.0), "min_corner": 45.0, "width": 16.0},
    2: {"kind": "circuit", "points": (6, 8), "radius": (60.0, 170.0), "slow": (25.0, 45.0), "fast": (50.0, 80.0),
        "main": 0.0, "hairpins": 0, "chicanes": (0, 0), "esses": (0, 0), "min_corner": 24.0, "width": 14.0},
    3: {"kind": "circuit", "points": (9, 12), "radius": (70.0, 220.0), "slow": (16.0, 35.0), "fast": (45.0, 90.0),
        "main": 170.0, "hairpins": 1, "chicanes": (0, 1), "esses": (0, 1), "min_corner": 14.0, "width": 12.0},
    4: {"kind": "circuit", "points": (13, 17), "radius": (80.0, 260.0), "slow": (10.0, 30.0), "fast": (45.0, 100.0),
        "main": 210.0, "hairpins": 2, "chicanes": (1, 2), "esses": (1, 2), "min_corner": 9.0, "width": 10.0},
}
MAX_LEVEL = 4
CIRCUIT_STRETCH = (1.0, 1.5)  # horizontal stretch of a circuit's base polygon (revision 3 §3.2)
DIRECTION_SALT = 7           # per-map direction draw: default_rng([seed, level, DIRECTION_SALT]) (revision 3 §3.3)
TRACK_CACHE_SIZE = 64        # built tracks kept per process; training maps are fresh every generation
```

- [ ] **Step 4: Bound the track cache**

In `racing/tracks.py`, replace

```python
@functools.lru_cache(maxsize=None)
def get_track(seed, level):
    """The track for (seed, level). Cached; callers must not modify it."""
```

with

```python
@functools.lru_cache(maxsize=config.TRACK_CACHE_SIZE)
def get_track(seed, level):
    """The track for (seed, level). The most recent config.TRACK_CACHE_SIZE tracks are cached, so fresh maps every
    generation do not accumulate (revision 3 §3.4); callers must not modify a track."""
```

The rest of `racing/tracks.py` is unchanged. Worker processes import the same function, so their caches are bounded
too (Review Focus 1).

- [ ] **Step 5: Write the generator**

Replace the whole of `racing/track_gen.py` with:

```python
"""Procedural track generation for curriculum levels 1-4 (revision 3 §3): gentle loops on Easy, and circuits of
straights and rounded corners with hairpins, chicanes and esses on Medium, Hard and Extra hard. Every map runs
clockwise or counter-clockwise, drawn from its seed."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from racing import config
from racing.track import build_track


def loop_points(rng, n_points, radius_range, step=0.5):
    """Control points at jittered angles and random radii, joined by a periodic cubic spline."""
    angles = (np.arange(n_points) + rng.uniform(-0.3, 0.3, n_points)) * (2 * np.pi / n_points)
    radii = rng.uniform(radius_range[0], radius_range[1], n_points)
    control = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    closed = np.vstack([control, control[:1]])
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    spline = CubicSpline(t, closed, bc_type="periodic")
    return spline(np.arange(0.0, t[-1], step))


def edge_lengths(poly):
    """Length of each polygon edge poly[i] -> poly[i + 1]."""
    return np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)


def fillet_path(poly, radii, step=0.5):
    """The closed polygon with each corner rounded by a circular arc of the given radius (reduced where the adjacent
    edges are too short to hold it), joined by straights. Returns points about `step` m apart, first not repeated."""
    n = len(poly)
    arcs = []
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        u, v = (a - b) / np.linalg.norm(a - b), (c - b) / np.linalg.norm(c - b)
        half = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)) / 2
        if half > np.radians(89.5):                           # (almost) straight through: no corner
            arcs.append(b[None, :])
            continue
        d = min(radii[i] / np.tan(half), 0.48 * np.linalg.norm(a - b), 0.48 * np.linalg.norm(c - b))
        r = d * np.tan(half)
        p1, p2 = b + u * d, b + v * d
        center = b + (u + v) / np.linalg.norm(u + v) * (r / np.sin(half))
        t1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
        t2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
        sweep = np.angle(np.exp(1j * (t2 - t1)))
        ts = t1 + sweep * np.linspace(0.0, 1.0, max(int(abs(sweep) * r / step), 2))
        arcs.append(np.column_stack([center[0] + r * np.cos(ts), center[1] + r * np.sin(ts)]))
    path = []
    for k, arc in enumerate(arcs):
        path.append(arc)
        following = arcs[(k + 1) % n][0]
        m = max(int(np.linalg.norm(following - arc[-1]) / step), 1)
        path.append(arc[-1] + (following - arc[-1]) * np.linspace(0.0, 1.0, m, endpoint=False)[1:, None])
    return np.vstack(path)


def insert_on_edge(poly, kinds, edge, along, side, kind):
    """Insert vertices on edge `edge` at offsets `along` (m, from its midpoint along the edge) and `side` (m, to its
    left), all marked `kind`."""
    a, b = poly[edge], poly[(edge + 1) % len(poly)]
    t = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-t[1], t[0]])
    middle = (a + b) / 2
    new = [middle + t * s + normal * o for s, o in zip(along, side)]
    return (np.vstack([poly[:edge + 1], new, poly[edge + 1:]]),
            kinds[:edge + 1] + [kind] * len(new) + kinds[edge + 1:])


def circuit_points(rng, spec, step=0.5):
    """A circuit (revision 3 §3.2): a stretched star polygon with a main straight, hairpins, chicanes and esses, its
    corners rounded with slow, fast or tight arcs."""
    n = int(rng.integers(spec["points"][0], spec["points"][1] + 1))
    angles = (np.arange(n) + rng.uniform(-0.3, 0.3, n)) * (2 * np.pi / n)
    radii = rng.uniform(spec["radius"][0], spec["radius"][1], n)
    stretch = rng.uniform(*config.CIRCUIT_STRETCH)
    poly = np.column_stack([stretch * radii * np.cos(angles), radii * np.sin(angles)])
    while spec["main"] and edge_lengths(poly).max() < spec["main"] and len(poly) > 5:
        longest = int(np.argmax(edge_lengths(poly)))
        poly = np.delete(poly, (longest + 2) % len(poly), axis=0)    # joins the two edges after the longest
    kinds = ["corner"] * len(poly)
    main = int(np.argmax(edge_lengths(poly)))
    pins = [(main + 1) % len(poly)]
    others = [j for j in range(len(poly)) if j not in pins and j != main]
    pins += [int(j) for j in rng.choice(others, max(spec["hairpins"] - 1, 0), replace=False)]
    for j in pins[:spec["hairpins"]]:
        poly[j] *= rng.uniform(1.25, 1.5)                               # outwards: a sharper corner
        kinds[j] = "hairpin"
    width = spec["width"]
    order = [int(e) for e in np.argsort(-edge_lengths(poly), kind="stable") if e != main]
    plan = (["chicane"] * int(rng.integers(spec["chicanes"][0], spec["chicanes"][1] + 1))
            + ["esses"] * int(rng.integers(spec["esses"][0], spec["esses"][1] + 1)))
    rng.shuffle(plan)
    for feature, edge in sorted(zip(plan, order[:len(plan)]), key=lambda pair: -pair[1]):   # last edge first
        length = edge_lengths(poly)[edge]
        side = rng.choice([-1.0, 1.0])
        if feature == "chicane" and length > 110.0:
            s, o = rng.uniform(12.0, 18.0), side * rng.uniform(0.9, 1.3) * width
            poly, kinds = insert_on_edge(poly, kinds, edge, [-1.5 * s, -0.5 * s, 0.5 * s, 1.5 * s], [0.0, o, o, 0.0],
                                         "chicane")
        elif feature == "esses" and length > 160.0:
            k = int(rng.integers(3, 5))
            along = (np.arange(k) - (k - 1) / 2) * rng.uniform(30.0, 45.0)
            o = rng.uniform(1.2, 2.0) * width
            poly, kinds = insert_on_edge(poly, kinds, edge, list(along), [side * o * (-1) ** j for j in range(k)],
                                         "esses")
    corner_radii = []
    for kind in kinds:
        if kind in ("hairpin", "chicane"):
            corner_radii.append(rng.uniform(spec["min_corner"], spec["min_corner"] + 6.0))
        elif kind == "esses":
            corner_radii.append(rng.uniform(spec["slow"][0] + 6.0, spec["slow"][1] + 6.0))
        else:
            corner_radii.append(rng.uniform(*spec["slow"]) if rng.random() < 0.5 else rng.uniform(*spec["fast"]))
    return fillet_path(poly, corner_radii, step)


def runs_clockwise(seed, level):
    """Direction of the map for (seed, level), the same for every generation attempt (revision 3 §3.3)."""
    return bool(np.random.default_rng([seed, level, config.DIRECTION_SALT]).random() < 0.5)


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
    if spec["kind"] == "loop":
        n_points = int(rng.integers(spec["points"][0], spec["points"][1] + 1))
        points = loop_points(rng, n_points, spec["radius"])
    else:
        points = circuit_points(rng, spec)
    if runs_clockwise(seed, level):
        points = points[::-1]
    track = build_track(points, spec["width"], seed, level)
    if not geometry_ok(track, spec["min_corner"]):
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

Notes for the implementer: `circuit_points` is the approved prototype (`f1_points` + `fillet_path`) with three
deliberate differences that the tests rely on: it draws from the caller's generator (`default_rng([seed, level,
attempt])`), it reads `CIRCUIT_STRETCH` from config, and the direction is applied by `candidate_track` from
`runs_clockwise`, which uses its own generator so the direction of a seed does not depend on the attempt number.
`geometry_ok` is now applied to every level (Easy loops included).

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_track_gen.py -q`
Expected: `26 passed, 1 deselected`.

- [ ] **Step 7: Run the calibration test**

Run: `~/miniconda3/bin/python -m pytest tests/test_track_gen.py -q -m slow`
Expected: `1 passed` (about 2 minutes). If a level fails, calibrate only that level's entry in `config.LEVELS`
(Easy: widen `"radius"` or lower `"min_corner"` by 5 m; a circuit level: lower `"min_corner"` by 1 m), rerun Steps 6
and 7, and state the changed values in the commit message.

- [ ] **Step 8: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all pass. (Level-1 maps used by `tests/test_population.py`, `tests/test_bench.py` and
`tests/test_expert.py` are now gentle loops; those tests only need drivable maps.)

- [ ] **Step 9: Commit**

```bash
git add racing/config.py racing/tracks.py racing/track_gen.py tests/test_track_gen.py
git commit -m "Fresh F1-inspired circuits: gentle Easy loops, circuit generator with hairpins, chicanes and esses, random direction, bounded track cache

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/config.py racing/tracks.py racing/track_gen.py tests/test_track_gen.py
```

### Task 2: Time-trial evaluation (`laps_done`, `target_time`, `DensePopulation`, `TrialResult`)

Implements revision 3 §4.1 (time trial, laps in time, fitness with the target time as the limit), §4.6 (final
selection by successes, then fitness) and the dense half of the population interface. The revision 2.0 evaluation
(fixed pace gate, `(fitness, finished, at_pace)` tuples) is still used by `racing/ga.py` and `racing/es.py` until Task 6,
so it stays in `racing/population.py` under the name `evaluate_weights`, and those two modules import it under their
old local name `evaluate_population` (their tests monkeypatch `ga.evaluate_population` / `es.evaluate_population` and
keep working unchanged).

**Files:**
- Modify: `racing/config.py` (one constant), `racing/sim.py`, `racing/nets.py`, `racing/ga.py` (imports), `racing/es.py` (imports)
- Replace: `racing/population.py` (whole file)
- Test: `tests/test_sim.py`, `tests/test_nets.py`, `tests/test_population.py`, `tests/test_bench.py`

**Interfaces:**
- Consumes: Task 1's `tracks.get_track`; `sim.simulate(track, policy, n, time_limit=None, laps=1)`;
  `nets.population_policy`, `nets.from_flat`, `nets.to_keras` (unchanged); `paths.champion_path`, `paths.remove`.
- Produces: `config.TARGET_FACTOR = 1.3`; `sim.RunResult.laps_done` (new **required** last field, `(n,)` int);
  `sim.target_time(track, factor, laps=config.LAPS)`; `nets.DensePopulation(weights, hidden, activation)` with
  `weights`, `hidden` (tuple), `activation`, `__len__`, `subset(indices)`, `policy()`, `driver(i)`;
  `nets.dense_spec(model) -> (hidden tuple, activation)`; `nets.load_champion` reading the architecture of an
  evolved dense controller from its Keras file with `dense_spec` when `<seed>.npy` exists; `population.TrialResult` (fields `fitness`, `success`, `laps`,
  `finish_time`; properties `all_success`, `lap_share`, `map_shares`, `min_share`, `lap_times`);
  `population.evaluate_population(population, seeds, level, factor, n_jobs=1, backend="loky") -> TrialResult`;
  `population.select_final(population, candidates, factor, n_jobs=1, seeds=config.VAL_SEEDS) -> (index, successes,
  mean fitness)`; `population.save_dense_champion(approach, seed, population, index, champion_dir)`;
  `population.clear_champion` (now also removes `.json`).
- Kept for Task 6 to delete (revision 2.0, do not use in new code): `population.evaluate_weights`, `_evaluate_track`,
  `finish_fractions`, `make_report`, `RunLogger`, `select_champion`, `save_champion`; `config.PACE_FACTOR`.

Champion file convention (Interface Contract): a dense evolved controller is `<seed>.keras` + `<seed>.npy`, and its
architecture is read back from the Keras file with `dense_spec`; a `<seed>.json` next to a controller always means a
NEAT genome (Task 8 adds that branch to `load_champion`, checked before `.npy`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sim.py`, and change its import line `from racing import config, sim` to
`from racing import config, expert, sim`:

```python
def test_simulate_reports_laps_done(circle_track):
    result = sim.simulate(circle_track, expert.expert_policy(circle_track), 2, laps=2)
    assert result.finished.all() and list(result.laps_done) == [2, 2]
    assert sim.simulate(circle_track, constant_policy(0.0, 1.0), 1).laps_done[0] == 0


def test_target_time_is_a_factor_of_the_expert_laps(circle_track):
    assert np.isclose(sim.target_time(circle_track, 1.3), 1.3 * 20.0 * config.LAPS)
    assert sim.target_time(circle_track, 1.0, laps=1) == 20.0
```

Append to `tests/test_nets.py`:

```python
def test_dense_population_subset_policy_and_driver_agree():
    weights = nets.init_population(np.random.default_rng(7), 5, (6,))
    cars = nets.DensePopulation(weights, (6,), "tanh")
    part = cars.subset([4, 1])
    assert len(cars) == 5 and len(part) == 2 and part.hidden == (6,) and part.activation == "tanh"
    obs = np.random.default_rng(8).random((2, config.N_INPUTS))
    assert np.array_equal(part.policy()(obs, None), nets.act(weights[[4, 1]], obs, (6,), "tanh"))
    assert np.array_equal(cars.driver(4).act(obs[:1]), part.policy()(obs, None)[:1])


def test_dense_spec_reads_hidden_sizes_and_activation():
    model = nets.to_keras(nets.init_population(np.random.default_rng(9), 1, (8, 4))[0], (8, 4), "tanh")
    assert nets.dense_spec(model) == ((8, 4), "tanh")
```

In `tests/test_population.py`:
1. Change the import line `from racing import config, nets, population, sim` to
   `from racing import config, expert, nets, population, sim`.
2. Replace every occurrence of `population.evaluate_population(` with `population.evaluate_weights(` (these
   revision 2.0 tests keep testing the legacy evaluator until Task 6 deletes it and them).
3. In `test_select_champion_picks_highest_validation_fitness`, replace
   `monkeypatch.setattr(population, "evaluate_population",` with `monkeypatch.setattr(population, "evaluate_weights",`.
4. In `test_at_pace_requires_finishing_within_the_pace_factor`, the fake `sim.RunResult(...)` must pass the new
   required field: replace `finish_time=np.array([limit - 1.0, limit + 1.0, np.nan]), steps=10)` with
   `finish_time=np.array([limit - 1.0, limit + 1.0, np.nan]), steps=10,` followed by a new line
   `                             laps_done=np.array([3, 3, 1]))`.
5. Append:

```python
# ----- revision 3 time trials (revision 3 §4.1, §4.6) -----

def dense(n, seed=0):
    return nets.DensePopulation(nets.init_population(np.random.default_rng(seed), n))


class ExpertCars:
    """n cars all driven by the expert; any object with len(), subset() and policy() can be evaluated."""

    def __init__(self, track, n):
        self.track, self.n = track, n

    def __len__(self):
        return self.n

    def subset(self, indices):
        return ExpertCars(self.track, len(indices))

    def policy(self):
        return expert.expert_policy(self.track)


def test_time_trial_shapes(fake_tracks):
    result = population.evaluate_population(dense(6), [0, 1], 1, config.TARGET_FACTOR)
    assert result.fitness.shape == (6,)
    assert result.success.shape == result.laps.shape == result.finish_time.shape == (2, 6)
    assert result.success.dtype == bool and np.all((result.laps >= 0) & (result.laps <= config.LAPS))
    assert np.all(np.isnan(result.finish_time[~result.success]))
    assert np.all((result.fitness >= 0) & (result.fitness < 2))


def test_trial_result_properties():
    success = np.array([[True, True, False, False], [True, False, True, False]])
    laps = np.array([[3, 3, 2, 0], [3, 1, 3, 0]])
    r = population.TrialResult(fitness=np.zeros(4), success=success, laps=laps,
                               finish_time=np.where(success, 60.0, np.nan))
    assert list(r.all_success) == [True, False, False, False]
    assert np.allclose(r.lap_share, [1.0, 4 / 6, 5 / 6, 0.0])
    assert np.allclose(r.map_shares, [0.5, 0.5]) and r.min_share == 0.5
    assert np.allclose(r.lap_times[success], 60.0 / config.LAPS)


def test_time_trial_ends_at_the_target_time(monkeypatch, fake_tracks):
    target = config.TARGET_FACTOR * fake_tracks.expert_lap_time * config.LAPS
    seen = []

    def fake_simulate(track, policy, n, time_limit=None, laps=1):
        seen.append((time_limit, laps))
        return sim.RunResult(fitness=np.array([1.5, 0.6, 0.2]), finished=np.array([True, False, False]),
                             crashed=np.array([False, False, True]), progress=np.array([1.0, 0.67, 0.2]),
                             finish_time=np.array([target - 1.0, np.nan, np.nan]), steps=10,
                             laps_done=np.array([3, 2, 0]))

    monkeypatch.setattr(population.sim, "simulate", fake_simulate)
    result = population.evaluate_population(dense(3), [0], 1, config.TARGET_FACTOR)
    assert seen == [(pytest.approx(target), config.LAPS)]
    assert list(result.success[0]) == [True, False, False] and list(result.laps[0]) == [3, 2, 0]
    assert result.finish_time[0, 0] == target - 1.0 and np.isnan(result.finish_time[0, 1])


def test_expert_succeeds_at_the_target_and_counts_laps_when_too_slow(fake_tracks):
    cars = ExpertCars(fake_tracks, 2)
    easy = population.evaluate_population(cars, [0], 1, config.TARGET_FACTOR)
    assert easy.success.all() and np.all(easy.laps == config.LAPS) and np.all(easy.fitness > 1)
    tight = population.evaluate_population(cars, [0], 1, 0.4)                   # 24 s for three laps
    assert not tight.success.any() and np.all((tight.laps >= 1) & (tight.laps < config.LAPS))
    assert np.all(tight.fitness < 1)


def test_time_trials_do_not_depend_on_n_jobs(fake_tracks):
    cars = dense(9, seed=1)
    serial = population.evaluate_population(cars, [0, 1, 2], 1, config.TARGET_FACTOR)
    threaded = population.evaluate_population(cars, [0, 1, 2], 1, config.TARGET_FACTOR, n_jobs=6,
                                              backend="threading")
    for field in ("fitness", "success", "laps"):
        assert np.array_equal(getattr(serial, field), getattr(threaded, field))
    assert np.array_equal(serial.finish_time, threaded.finish_time, equal_nan=True)


def test_a_subset_reproduces_its_cars_results(fake_tracks):
    cars = dense(10, seed=2)
    full = population.evaluate_population(cars, [0], 1, config.TARGET_FACTOR)
    part = population.evaluate_population(cars.subset([1, 7]), [0], 1, config.TARGET_FACTOR)
    assert np.array_equal(full.fitness[[1, 7]], part.fitness) and np.array_equal(full.laps[:, [1, 7]], part.laps)


def test_select_final_prefers_successes_then_fitness(monkeypatch):
    def fake(cars, seeds, level, factor, n_jobs=1):
        assert len(cars) == 4 and seeds == list(config.VAL_SEEDS) and level == config.MAX_LEVEL and factor == 1.1
        success = np.zeros((len(seeds), 4), dtype=bool)
        success[0, :3] = True
        success[1, 1:3] = True
        return population.TrialResult(fitness=np.array([1.9, 1.2, 1.5, 0.3]), success=success,
                                      laps=np.zeros((len(seeds), 4), dtype=np.int64),
                                      finish_time=np.full((len(seeds), 4), np.nan))

    monkeypatch.setattr(population, "evaluate_population", fake)
    assert population.select_final(dense(6), [5, 0, 3, 2], 1.1) == (3, 2, 1.5)


def test_dense_champion_keeps_its_architecture(tmp_path):
    weights = nets.init_population(np.random.default_rng(3), 2, (8, 4))
    cars = nets.DensePopulation(weights, (8, 4), "tanh")
    path = population.save_dense_champion("baseline", 1, cars, 1, tmp_path)
    assert path == tmp_path / "baseline" / "seed1.keras" and path.exists()
    saved = np.load(tmp_path / "baseline" / "seed1.npy")
    assert saved.dtype == np.float64 and np.array_equal(saved, weights[1])
    _, driver = nets.load_champion("baseline", 1, tmp_path)
    obs = np.random.default_rng(4).random((50, config.N_INPUTS))
    assert np.array_equal(driver.act(obs), cars.driver(1).act(obs))


def test_clear_champion_removes_every_file(tmp_path):
    population.save_dense_champion("es", 0, dense(1, seed=4), 0, tmp_path)
    assert sorted(p.suffix for p in (tmp_path / "es").iterdir()) == [".keras", ".npy"]
    population.clear_champion("es", 0, tmp_path)
    assert list((tmp_path / "es").iterdir()) == []
```

In `tests/test_bench.py`, change the import line to `from racing import bench, config, nets, population` and replace
`test_evaluate_population_multiprocess_consistency` with:

```python
def test_evaluate_population_multiprocess_consistency():
    """Multi-process evaluation must match single-process on a real level-1 map."""
    cars = nets.DensePopulation(nets.init_population(np.random.default_rng(0), 8))
    single = population.evaluate_population(cars, [0, 1], 1, config.TARGET_FACTOR)
    multi = population.evaluate_population(cars, [0, 1], 1, config.TARGET_FACTOR, n_jobs=2)
    for field in ("fitness", "success", "laps"):
        assert np.array_equal(getattr(single, field), getattr(multi, field))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_sim.py tests/test_nets.py tests/test_population.py tests/test_bench.py -q`
Expected: failures with `AttributeError: module 'racing.sim' has no attribute 'target_time'`,
`AttributeError: ... 'RunResult' object has no attribute 'laps_done'`, `AttributeError: module 'racing.nets' has no
attribute 'DensePopulation'` and `AttributeError: module 'racing.population' has no attribute 'evaluate_weights'`.

- [ ] **Step 3: Add the starting target factor**

In `racing/config.py`, directly below the line `PACE_FACTOR = 1.3 ...`, add:

```python
TARGET_FACTOR = 1.3          # time trial: starting target = TARGET_FACTOR × LAPS expert laps (revision 3 §4.1)
```

- [ ] **Step 4: Report laps and the target time from the simulator**

In `racing/sim.py`:
1. In `class RunResult`, after the line `    steps: int`, add
   `    laps_done: np.ndarray           # laps completed when the run ended (revision 3 §4.1)`.
2. After the function `time_limit_for`, add:

```python
def target_time(track, factor, laps=config.LAPS):
    """Target time of a time trial (revision 3 §4.1): factor × the expert's time for `laps` laps."""
    if np.isnan(track.expert_lap_time):
        raise ValueError("track has no expert lap time")
    return factor * track.expert_lap_time * laps
```

3. In `simulate`, replace `finish_time=cars.finish_time.copy(), steps=steps)` with
   `finish_time=cars.finish_time.copy(), steps=steps, laps_done=cars.laps_done.copy())`.

- [ ] **Step 5: Add the dense population type and the architecture-aware champion loader**

In `racing/nets.py`:
1. (No new import: Task 8 adds `import json` when it loads NEAT genomes.)
2. Insert directly above `class DenseNet:`:

```python
class DensePopulation:
    """Dense networks of one architecture, one flat weight vector per car (revision 3 §5). Like every population
    type, it offers len(), subset(indices), policy() and driver(i)."""

    def __init__(self, weights, hidden=config.BASELINE_HIDDEN, activation="relu"):
        self.weights = np.asarray(weights, dtype=float)
        self.hidden = tuple(int(units) for units in hidden)
        self.activation = activation

    def __len__(self):
        return len(self.weights)

    def subset(self, indices):
        return DensePopulation(self.weights[np.asarray(indices, dtype=np.int64)], self.hidden, self.activation)

    def policy(self):
        return population_policy(self.weights, self.hidden, self.activation)

    def driver(self, i):
        return from_flat(self.weights[int(i)], self.hidden, self.activation)
```

3. Insert directly above `def to_keras(`:

```python
def dense_spec(model):
    """(hidden sizes, hidden activation) of a dense Keras model; dropout layers are ignored."""
    keras = keras_cpu()
    dense = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    hidden = tuple(int(layer.units) for layer in dense[:-1])
    activation = dense[0].get_config()["activation"] if hidden else "relu"
    return hidden, activation
```

4. Replace the whole function `load_champion` with:

```python
def load_champion(approach, seed, directory=None):
    """(Keras model, driver) of one final controller. Evolved dense controllers drive with the float64 weights saved
    next to the Keras file (spec §6), in the architecture of that Keras file (revision 3 §6.4)."""
    keras_path = paths.champion_path(approach, seed, champion_dir=directory)
    model = keras_cpu().models.load_model(keras_path)
    flat_path = paths.champion_path(approach, seed, ".npy", directory)
    if flat_path.exists():
        return model, from_flat(np.load(flat_path), *dense_spec(model))
    return model, from_keras(model)
```

- [ ] **Step 6: Write the time-trial evaluation**

Replace the whole of `racing/population.py` with:

```python
"""Shared helpers for the evolutionary approaches (spec §9.1, revision 3 §4): time-trial evaluation, final selection
and saved controllers, and the revision 2.0 logging helpers still used by ga.py and es.py until Task 6."""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from racing import config, nets, paths, sim, tracks
from racing.tracks import get_track


def fresh_workers(n_jobs):
    """Starts an evolutionary run without built tracks, so every run pays the same cost of generating its tracks
    (spec §9.1): empties this process's track cache and, for a parallel run, shuts down joblib's reusable worker
    processes (and their track caches); the run's first parallel evaluation starts new ones."""
    tracks.get_track.cache_clear()
    if n_jobs > 1:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor(reuse=True).shutdown(wait=True)     # creating an executor starts no workers


def split_tasks(n, seeds, n_jobs):
    """(track position, track seed, individual indices) tasks for one evaluation (spec §9.5). Each track's
    individuals are split into n_jobs // len(seeds) shares (at least one), so a task drives many cars on one track."""
    shares = min(max(1, n_jobs // len(seeds)), max(n, 1))
    parts = np.array_split(np.arange(n), shares)
    return [(k, seed, part) for k, seed in enumerate(seeds) for part in parts]


@dataclass(eq=False)
class TrialResult:
    """One generation's time trials (revision 3 §4.1): one row per map, one column per car."""
    fitness: np.ndarray        # (n,) mean fitness over the maps, with the target time as the time limit
    success: np.ndarray        # (maps, n) three laps within the target time
    laps: np.ndarray           # (maps, n) laps completed within the target time
    finish_time: np.ndarray    # (maps, n) s, nan unless success

    @property
    def all_success(self):
        return self.success.all(axis=0)

    @property
    def lap_share(self):
        return self.laps.sum(axis=0) / (config.LAPS * self.laps.shape[0])

    @property
    def map_shares(self):
        return self.success.mean(axis=1)

    @property
    def min_share(self):
        return float(self.map_shares.min())

    @property
    def lap_times(self):
        return self.finish_time / config.LAPS


def _time_trial(population, seed, level, factor):
    """Fitness, success, laps in time and finish time of every car over config.LAPS laps of one map, ending at the
    target time (revision 3 §4.1)."""
    track = get_track(seed, level)
    result = sim.simulate(track, population.policy(), len(population), time_limit=sim.target_time(track, factor),
                          laps=config.LAPS)
    return (result.fitness, result.finished, np.minimum(result.laps_done, config.LAPS),
            np.where(result.finished, result.finish_time, np.nan))


def evaluate_population(population, seeds, level, factor, n_jobs=1, backend="loky"):
    """Time trials of every car on every map (revision 3 §4.1). The per-map results are combined in map order, so
    the result does not depend on n_jobs."""
    n = len(population)
    tasks = split_tasks(n, seeds, n_jobs)
    if n_jobs == 1 or len(tasks) == 1:
        parts = [_time_trial(population.subset(idx), seed, level, factor) for _, seed, idx in tasks]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_time_trial)(population.subset(idx), seed, level, factor) for _, seed, idx in tasks)
    maps = len(seeds)
    per_map = np.zeros((maps, n))
    success = np.zeros((maps, n), dtype=bool)
    laps = np.zeros((maps, n), dtype=np.int64)
    finish_time = np.full((maps, n), np.nan)
    for (k, _, idx), (fitness, done, laps_done, times) in zip(tasks, parts):
        per_map[k, idx] = fitness
        success[k, idx] = done
        laps[k, idx] = laps_done
        finish_time[k, idx] = times
    total = np.zeros(n)
    for k in range(maps):
        total += per_map[k]
    return TrialResult(fitness=total / maps, success=success, laps=laps, finish_time=finish_time)


def select_final(population, candidates, factor, n_jobs=1, seeds=config.VAL_SEEDS):
    """The final controller (revision 3 §4.6): of the candidate indices, the car with the most successes on the
    validation maps at target factor `factor`, then the highest mean fitness, then the earliest candidate. Returns
    (index into population, validation successes, mean validation fitness)."""
    candidates = np.asarray(candidates, dtype=np.int64)
    result = evaluate_population(population.subset(candidates), list(seeds), config.MAX_LEVEL, factor, n_jobs=n_jobs)
    successes = result.success.sum(axis=0)
    best = int(np.lexsort((-result.fitness, -successes))[0])
    return int(candidates[best]), int(successes[best]), float(result.fitness[best])


def save_dense_champion(approach, seed, population, index, champion_dir=config.CHAMPIONS_DIR):
    """Final dense controller (spec §6, revision 3 §6.4): a Keras file and the float64 weights (.npy); the
    architecture is the Keras file's. Returns the Keras file's path."""
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = population.weights[int(index)]
    nets.to_keras(flat, population.hidden, population.activation).save(path)
    np.save(paths.champion_path(approach, seed, ".npy", champion_dir), np.asarray(flat, dtype=float))
    return path


def clear_champion(approach, seed, champion_dir=config.CHAMPIONS_DIR):
    """Remove a final controller left by an earlier run (.keras, .npy and a NEAT genome's .json), so an interrupted
    run leaves none behind."""
    paths.remove(*(paths.champion_path(approach, seed, suffix, champion_dir) for suffix in (".keras", ".npy", ".json")))


# ----- revision 2.0 helpers, used by ga.py and es.py until the evolution engine replaces them (plan Task 6) -----

def _evaluate_track(weights, seed, level):
    """Fitness, finishing and finishing at pace (spec §9.1) of every network over config.LAPS laps of one track."""
    track = get_track(seed, level)
    result = sim.simulate(track, nets.population_policy(weights), len(weights), laps=config.LAPS)
    pace_limit = config.PACE_FACTOR * track.expert_lap_time * config.LAPS + config.TIME_EPS
    at_pace = result.finished & (np.nan_to_num(result.finish_time, nan=np.inf) <= pace_limit)
    return result.fitness, result.finished, at_pace


def evaluate_weights(weights, seeds, level, n_jobs=1, backend="loky"):
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


def finish_fractions(finished, is_kept):
    """Fraction of kept and of new individuals that finished; nan for an empty group."""
    kept = float(finished[is_kept].mean()) if is_kept.any() else float("nan")
    new = float(finished[~is_kept].mean()) if (~is_kept).any() else float("nan")
    return kept, new


def make_report(*, approach, generation, level, seeds, weights, fitness, finished, is_kept, kept_finish,
                new_finish, kept_pace, new_pace, step_size, evals, elapsed, leveled_up,
                whole_pace=float("nan"), strict_success=float("nan"), tie_fraction=float("nan")):
    """One generation's results, for logging and for the training view."""
    return {"approach": approach, "generation": int(generation), "level": int(level), "seeds": list(seeds),
            "weights": weights, "hidden": list(config.BASELINE_HIDDEN), "activation": "relu",
            "fitness": fitness, "finished": finished, "is_kept": is_kept,
            "kept_finish": kept_finish, "new_finish": new_finish, "kept_pace": kept_pace, "new_pace": new_pace,
            "whole_pace": float(whole_pace), "n_kept": int(np.sum(is_kept)),
            "step_size": float(step_size), "strict_success": float(strict_success),
            "tie_fraction": float(tie_fraction),
            "best": float(fitness.max()), "mean": float(fitness.mean()), "worst": float(fitness.min()),
            "evals": int(evals), "elapsed": float(elapsed), "leveled_up": bool(leveled_up)}


class RunLogger:
    """Writes one CSV row and the best network after every generation, and a summary at the end."""
    FIELDS = ["generation", "level", "evals", "best", "mean", "worst", "whole_pace", "n_kept", "kept_finish",
              "new_finish", "kept_pace", "new_pace", "step_size", "strict_success", "tie_fraction", "seeds", "elapsed", "leveled_up"]

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for stale in self.dir.glob("best_gen*.npy"):    # left over from an earlier run in the same directory
            stale.unlink()
        paths.remove(self.dir / "summary.json")
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
    fitness, _, _ = evaluate_weights(candidates, list(seeds), config.MAX_LEVEL, n_jobs=n_jobs)
    best = int(np.argmax(fitness))
    return candidates[best], float(fitness[best])


def save_champion(approach, seed, flat, champion_dir=config.CHAMPIONS_DIR):
    """Final controller as a Keras file and as float64 weights (spec §6). Returns the Keras file's path."""
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    nets.to_keras(flat).save(path)
    np.save(paths.champion_path(approach, seed, ".npy", champion_dir), np.asarray(flat, dtype=float))
    return path
```

- [ ] **Step 7: Keep ga.py and es.py on the revision 2.0 evaluator until Task 6**

In both `racing/ga.py` and `racing/es.py`, replace

```python
from racing.population import (RunLogger, clear_champion, evaluate_population, finish_fractions, fresh_workers,
                               make_report, save_champion, select_champion)
```

with

```python
from racing.population import (RunLogger, clear_champion, finish_fractions, fresh_workers, make_report,
                               save_champion, select_champion)
from racing.population import evaluate_weights as evaluate_population   # revision 2.0 evaluation until plan Task 6
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_sim.py tests/test_nets.py tests/test_population.py tests/test_bench.py -q`
Expected: all pass (62 tests).

- [ ] **Step 9: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add racing/config.py racing/sim.py racing/nets.py racing/population.py racing/ga.py racing/es.py tests/test_sim.py tests/test_nets.py tests/test_population.py tests/test_bench.py
git commit -m "Time trials: laps completed in time, target time, dense population type, per-map trial results, final selection by successes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/config.py racing/sim.py racing/nets.py racing/population.py racing/ga.py racing/es.py tests/test_sim.py tests/test_nets.py tests/test_population.py tests/test_bench.py
```

### Task 3: Survival rule, fresh seeds, mastery tracker, step sizes

Implements revision 3 §3.4 (fresh training seeds, never repeated), §4.2 (shared survival rule), §4.3–4.5 (levels at
90% on every map, the mastery goal, the speed push, the safety stop), §5.1 (GA strength from the children's success)
and §5.2 (two-phase σ), plus Review Focus 5 at the tracker level. `success_fraction` moves from `racing/es.py` into
`racing/curriculum.py` (es.py imports it, so `es.success_fraction` still resolves for `tests/test_es.py`).

`LevelTracker` and `draw_seeds` stay in `racing/curriculum.py` under a "revision 2.0" heading because `racing/ga.py`
and `racing/es.py` still use them; Task 6 deletes them together with their tests in `tests/test_curriculum.py`
(`test_level_up_on_third_consecutive_qualifying_generation`, `test_streak_resets_below_the_threshold`,
`test_below_threshold_never_qualifies`, `test_level_stops_at_max`, `test_draw_seeds_are_distinct_training_seeds`) and
with `config.LEVEL_UP_FRACTION`, `LEVEL_UP_STREAK`, `GA_KEPT`, `GA_KEEP_MAX_FRACTION`, `EVAL_BUDGET`, `ES_LAMBDA` and
`PACE_FACTOR`. `config.TRAIN_SEEDS` and `TRAIN_DATA_TRACKS` stay until Task 10 (the dataset still uses them).

Two semantics the tests pin beyond the contract text: (a) `survivors` uses an effective floor of `min(floor, cap)`, so
populations smaller than about 56 (viewer presets, pilots, tests) never keep everyone; (b) `MasteryTracker.update`
returns `None` and changes nothing once the run is `done`.

**Files:**
- Modify: `racing/config.py` (eight constants), `racing/es.py` (move `success_fraction`)
- Replace: `racing/curriculum.py` (whole file)
- Test: `tests/test_curriculum.py`

**Interfaces:**
- Consumes: `config.TARGET_FACTOR` (Task 2), `config.MAX_LEVEL`, `config.ES_FACTOR`, `config.ES_SIGMA_MIN`,
  `config.ES_SIGMA_MAX`, `config.GA_S_MAX`, `config.GA_S_MIN`.
- Produces: `config.TRAIN_SEED_START`, `MASTERY_FRACTION`, `MASTERY_STREAK`, `SPEED_STEP`, `SPEED_PATIENCE`,
  `SAFETY_PATIENCE`, `KEEP_MAX_FRACTION`, `ELITE_FLOOR`; `curriculum.EVENTS`; `curriculum.SeedSource(rng, start, used)`
  with `draw(count)` and `state()`; `curriculum.survivors(rng, fitness, all_success, lap_share, cap, floor)`;
  `curriculum.MasteryTracker(level, factor)` with `level`, `factor`, `streak`, `goal_reached`, `mastered_factor`,
  `outcome`, `update(min_share)`, `done`, `target_reduction`, `state()`, `from_state(state)`;
  `curriculum.mutation_strength(child_success, s_max, s_min)` (nan → `s_max`); `curriculum.success_fraction`;
  `curriculum.StepSize(sigma0)` with `sigma`, `sigma0`, `update(parent_success, child_success, success_rate)`,
  `reset()`, `state()`, `from_state(state)`. The engine (Task 6) calls
  `survivors(rng, result.fitness, result.all_success, result.lap_share, int(config.KEEP_MAX_FRACTION * n),
  config.ELITE_FLOOR)` once per generation.

- [ ] **Step 1: Write the failing tests**

In `tests/test_curriculum.py`, replace the header (everything above `def test_mutation_strength_endpoints_and_midpoint`)
with:

```python
import math
import pickle

import numpy as np

from racing import config
from racing.curriculum import (EVENTS, LevelTracker, MasteryTracker, SeedSource, StepSize, draw_seeds,
                               mutation_strength, success_fraction, survivors)

M = config.MASTERY_FRACTION
```

and append:

```python
# ----- revision 3 (§3.4, §4.2-4.5, §5.1-5.2) -----

def test_mutation_strength_without_children_is_s_max():
    assert mutation_strength(float("nan")) == config.GA_S_MAX
    assert mutation_strength(float("nan"), 0.2, 0.002) == 0.2


def test_seed_source_draws_fresh_distinct_seeds():
    source = SeedSource(np.random.default_rng(0))
    seeds = source.draw(3) + source.draw(3)
    assert len(set(seeds)) == 6 and all(config.TRAIN_SEED_START <= s < 2 ** 31 - 1 for s in seeds)
    assert source.state() == sorted(seeds)


class Scripted:
    """A generator stand-in that returns the given integers in order."""

    def __init__(self, values):
        self.values = list(values)

    def integers(self, low, high):
        return self.values.pop(0)


def test_seed_source_never_repeats_a_seed():
    source = SeedSource(Scripted([5, 5, 7, 9]), start=0, used=[7])
    assert source.draw(2) == [5, 9]
    assert source.state() == [5, 7, 9]


def test_seed_source_resumes_from_its_state():
    rng = np.random.default_rng(2)
    source = SeedSource(rng)
    source.draw(3)
    saved, used = rng.bit_generator.state, source.state()
    expected = source.draw(3)
    restored = np.random.default_rng()
    restored.bit_generator.state = saved
    assert SeedSource(restored, used=used).draw(3) == expected


def keep(fitness, all_success, lap_share, cap, floor, seed=0):
    return list(survivors(np.random.default_rng(seed), np.array(fitness, dtype=float),
                          np.array(all_success, dtype=bool), np.array(lap_share, dtype=float), cap, floor))


def test_successes_are_kept_fittest_first_up_to_the_cap():
    fitness, success = [1.2, 1.8, 0.5, 1.5, 1.1], [True, True, False, True, True]
    assert keep(fitness, success, [1, 1, 0, 1, 1], cap=3, floor=0) == [1, 3, 0]


def test_cars_without_laps_in_time_are_never_kept_by_chance():
    assert keep([0.9, 0.8, 0.7], [False] * 3, [0.0] * 3, cap=3, floor=0) == []


def test_order_is_successes_then_chance_then_floor():
    fitness = [0.2, 1.4, 0.9, 0.95, 0.1]
    success = [False, True, False, False, False]
    share = [0.0, 1.0, 1.0, 0.0, 0.0]                    # car 2 is kept by chance for certain, 0, 3 and 4 never
    assert keep(fitness, success, share, cap=4, floor=4) == [1, 2, 3, 0]


def test_chance_survival_matches_the_share_of_laps_in_time():
    n = 20000
    share = np.where(np.arange(n) % 2 == 0, 0.25, 0.75)
    kept = survivors(np.random.default_rng(3), np.linspace(0.9, 0.1, n), np.zeros(n, dtype=bool), share, n, 0)
    mask = np.zeros(n, dtype=bool)
    mask[kept] = True
    assert abs(mask[share == 0.25].mean() - 0.25) < 0.02 and abs(mask[share == 0.75].mean() - 0.75) < 0.02


def test_elite_floor_keeps_the_fittest_when_nobody_succeeds():
    kept = survivors(np.random.default_rng(4), np.arange(100.0), np.zeros(100, dtype=bool), np.zeros(100), 90, 50)
    assert sorted(kept) == list(range(50, 100))


def test_floor_never_exceeds_the_cap_in_small_populations():
    kept = survivors(np.random.default_rng(4), np.arange(20.0), np.zeros(20, dtype=bool), np.zeros(20), 18, 50)
    assert len(kept) == 18 and sorted(kept) == list(range(2, 20))


def test_survivors_take_one_draw_per_car():
    rng = np.random.default_rng(5)
    survivors(rng, np.ones(7), np.zeros(7, dtype=bool), np.full(7, 0.5), 6, 0)
    reference = np.random.default_rng(5)
    reference.random(7)
    assert rng.random() == reference.random()


def test_three_passing_generations_advance_a_level():
    t = MasteryTracker()
    assert [t.update(M), t.update(1.0), t.update(0.95)] == [None, None, "level_up"]
    assert t.level == 2 and t.streak == 0 and not t.goal_reached


def test_a_failing_generation_resets_the_streak():
    t = MasteryTracker()
    t.update(1.0)
    t.update(1.0)
    assert t.update(M - 0.01) is None and t.streak == 0
    assert [t.update(1.0) for _ in range(3)] == [None, None, "level_up"]


def test_goal_on_the_last_level_starts_the_speed_push():
    t = MasteryTracker(level=config.MAX_LEVEL)
    assert [t.update(1.0) for _ in range(3)] == [None, None, "goal"]
    assert t.goal_reached and t.level == config.MAX_LEVEL and t.target_reduction == 0.0
    assert t.mastered_factor == config.TARGET_FACTOR
    assert math.isclose(t.factor, config.TARGET_FACTOR * config.SPEED_STEP)


def test_each_mastered_target_is_tightened_by_the_speed_step():
    t = MasteryTracker(level=config.MAX_LEVEL)
    events = [t.update(1.0) for _ in range(9)]
    assert events.count("goal") == 1 and events.count("tightened") == 2 and set(events) <= set(EVENTS) | {None}
    assert math.isclose(t.mastered_factor, config.TARGET_FACTOR * config.SPEED_STEP ** 2)
    assert math.isclose(t.target_reduction, 1 - config.SPEED_STEP ** 2)


def test_speed_push_ends_after_patience_without_mastery(monkeypatch):
    monkeypatch.setattr(config, "SPEED_PATIENCE", 4)
    t = MasteryTracker(level=config.MAX_LEVEL)
    for _ in range(3):
        t.update(1.0)
    assert [t.update(0.5) for _ in range(4)] == [None, None, None, "end_speed"]
    assert t.done and t.outcome == "speed_limit" and t.update(1.0) is None


def test_mastery_wins_over_the_end_of_the_speed_push(monkeypatch):
    monkeypatch.setattr(config, "SPEED_PATIENCE", 3)
    t = MasteryTracker(level=config.MAX_LEVEL)
    for _ in range(3):
        t.update(1.0)
    assert [t.update(1.0) for _ in range(3)] == [None, None, "tightened"] and not t.done


def test_safety_stop_ends_a_run_that_stops_improving(monkeypatch):
    """Review focus 5: a population that never reaches 90% ends with the outcome 'stalled'."""
    monkeypatch.setattr(config, "SAFETY_PATIENCE", 5)
    t = MasteryTracker()
    assert [t.update(0.5) for _ in range(6)] == [None] * 5 + ["end_stalled"]
    assert t.done and t.outcome == "stalled" and t.level == 1 and not t.goal_reached
    assert math.isnan(t.target_reduction) and t.update(1.0) is None


def test_improvement_postpones_the_safety_stop(monkeypatch):
    monkeypatch.setattr(config, "SAFETY_PATIENCE", 3)
    t = MasteryTracker()
    assert [t.update(s) for s in (0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2)] == [None] * 7
    assert [t.update(0.2), t.update(0.2)] == [None, "end_stalled"]


def test_level_up_restarts_the_safety_counters(monkeypatch):
    monkeypatch.setattr(config, "SAFETY_PATIENCE", 3)
    t = MasteryTracker()
    assert [t.update(0.95) for _ in range(3)] == [None, None, "level_up"]
    assert [t.update(0.5) for _ in range(4)] == [None, None, None, "end_stalled"]


def test_tracker_state_round_trip():
    t = MasteryTracker()
    for share in (0.95, 0.4, 0.95):
        t.update(share)
    copy = MasteryTracker.from_state(pickle.loads(pickle.dumps(t.state())))
    for share in (0.95, 0.95, 0.95, 0.3):
        assert copy.update(share) == t.update(share)
    assert str(copy.state()) == str(t.state())


def test_step_size_follows_the_one_fifth_rule_while_learning():
    s = StepSize(0.1)
    assert math.isclose(s.update(0.5, 0.0, 0.3), 0.1 / config.ES_FACTOR)
    s.reset()
    assert math.isclose(s.update(0.5, 0.0, 0.1), 0.1 * config.ES_FACTOR)
    s.reset()
    assert s.update(0.5, 0.0, 0.2) == 0.1
    assert s.update(float("nan"), float("nan"), float("nan")) == 0.1


def test_step_size_follows_the_children_once_the_parents_are_reliable():
    s = StepSize(0.1)
    assert math.isclose(s.update(0.95, 0.5, 0.9), 0.1 * config.ES_FACTOR)      # the 1/5 rule would grow it
    s.reset()
    assert math.isclose(s.update(M, 0.92, 0.0), 0.1 / config.ES_FACTOR)       # the 1/5 rule would shrink it
    s.reset()
    assert s.update(1.0, float("nan"), 0.5) == 0.1


def test_step_size_is_clipped_and_round_trips():
    assert StepSize(config.ES_SIGMA_MAX).update(0.0, 0.0, 1.0) == config.ES_SIGMA_MAX
    assert StepSize(config.ES_SIGMA_MIN).update(0.0, 0.0, 0.0) == config.ES_SIGMA_MIN
    s = StepSize(0.3)
    s.update(0.0, 0.0, 0.0)
    copy = StepSize.from_state(s.state())
    assert copy.sigma == s.sigma and copy.sigma0 == 0.3
    copy.reset()
    assert copy.sigma == 0.3


def test_success_fraction_counts_ties_as_half():
    child, parent = np.array([2.0, 1.0, 3.0, 0.0]), np.array([1.0, 1.0, 1.0, 1.0])
    assert success_fraction(child, parent) == (1.0 + 0.5 + 1.0 + 0.0) / 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_curriculum.py -q`
Expected: collection error, `ImportError: cannot import name 'EVENTS' from 'racing.curriculum'`.

- [ ] **Step 3: Add the survival and level constants**

In `racing/config.py`, directly below `SEARCH_SEED = 0 ...` add

```python
TRAIN_SEED_START = 2_000_000  # fresh training maps: seeds drawn from here upwards, never repeated in a run (revision 3 §3.4)
```

and directly below the `TARGET_FACTOR` line added in Task 2 add

```python
MASTERY_FRACTION = 0.9       # lowest map success share for a passing generation (revision 3 §4.3)
MASTERY_STREAK = 3           # consecutive passing generations to advance a level or master a target
SPEED_STEP = 0.95            # target factor multiplier per mastered target in the speed push (revision 3 §4.4)
SPEED_PATIENCE = 500         # generations without mastering the current target that end the speed push
SAFETY_PATIENCE = 2000       # generations without a new best 3-generation mean lowest share that end a run (§4.5)
KEEP_MAX_FRACTION = 0.9      # kept cars fill at most this share of the population (revision 3 §4.2)
ELITE_FLOOR = 50             # at least this many of the fittest cars are kept
```

- [ ] **Step 4: Write the curriculum rules**

Replace the whole of `racing/curriculum.py` with:

```python
"""Curriculum rules (revision 3 §3.4, §4.2-4.5, §5.1-5.2): fresh training seeds, survival, levels with the mastery
goal, the speed push and the safety stop, GA mutation strength and the ES/NEAT step size."""
import numpy as np

from racing import config

EVENTS = ("level_up", "goal", "tightened", "end_speed", "end_stalled")


class SeedSource:
    """Seeds of freshly generated training maps (revision 3 §3.4): drawn by the run's random generator from `start`
    upwards and never repeated within the run. `used` restores the seeds of a resumed run."""

    def __init__(self, rng, start=config.TRAIN_SEED_START, used=()):
        self.rng = rng
        self.start = int(start)
        self.used = {int(seed) for seed in used}

    def draw(self, count):
        seeds = []
        while len(seeds) < count:
            seed = int(self.rng.integers(self.start, 2 ** 31 - 1))
            if seed not in self.used:
                self.used.add(seed)
                seeds.append(seed)
        return seeds

    def state(self):
        return sorted(self.used)


def survivors(rng, fitness, all_success, lap_share, cap, floor):
    """Indices of the cars kept unchanged (revision 3 §4.2): every car that succeeded on every map, fittest first;
    then each other car with probability equal to its share of laps in time, fittest first, while fewer than `cap`
    are kept; then, if fewer than min(floor, cap) are kept, the fittest remaining cars. One rng.random(n) draw per
    call, so the result depends only on the generator's state."""
    n = len(fitness)
    order = np.argsort(-np.asarray(fitness, dtype=float), kind="stable")
    lucky = rng.random(n) < np.asarray(lap_share, dtype=float)
    chosen = [int(i) for i in order if all_success[i]][:cap]
    taken = set(chosen)
    for i in order:
        if len(chosen) >= cap:
            break
        if int(i) not in taken and lucky[i]:
            chosen.append(int(i))
            taken.add(int(i))
    for i in order:
        if len(chosen) >= min(floor, cap):
            break
        if int(i) not in taken:
            chosen.append(int(i))
            taken.add(int(i))
    return np.array(chosen, dtype=np.int64)


class MasteryTracker:
    """Levels, the mastery goal, the speed push and the safety stop (revision 3 §4.3-4.5). update() takes the lowest
    map success share of one generation and returns one of EVENTS or None."""

    def __init__(self, level=1, factor=config.TARGET_FACTOR):
        self.level = int(level)
        self.factor = float(factor)
        self.streak = 0
        self.goal_reached = False
        self.mastered_factor = float("nan")
        self.outcome = None             # None, "speed_limit" or "stalled"
        self.recent = []                # the level's latest lowest shares (at most MASTERY_STREAK, i.e. the three
                                        # generations of revision 3 §4.5), for the safety stop
        self.best_mean = float("-inf")
        self.since_best = 0
        self.since_mastered = 0

    @property
    def done(self):
        return self.outcome is not None

    @property
    def target_reduction(self):
        """How much shorter the tightest mastered target is than the starting target; nan before the goal."""
        return 1.0 - self.mastered_factor / config.TARGET_FACTOR

    def _master(self):
        self.mastered_factor = self.factor
        self.factor *= config.SPEED_STEP
        self.streak = 0
        self.since_mastered = 0

    def update(self, min_share):
        if self.done:
            return None
        self.streak = self.streak + 1 if min_share >= config.MASTERY_FRACTION else 0
        if self.goal_reached:
            if self.streak >= config.MASTERY_STREAK:
                self._master()
                return "tightened"
            self.since_mastered += 1
            if self.since_mastered >= config.SPEED_PATIENCE:
                self.outcome = "speed_limit"
                return "end_speed"
            return None
        if self.streak >= config.MASTERY_STREAK:
            if self.level < config.MAX_LEVEL:
                self.level += 1
                self.streak = 0
                self.recent, self.best_mean, self.since_best = [], float("-inf"), 0
                return "level_up"
            self.goal_reached = True
            self._master()
            return "goal"
        self.recent = (self.recent + [float(min_share)])[-config.MASTERY_STREAK:]
        mean = float(np.mean(self.recent))
        if mean > self.best_mean:
            self.best_mean, self.since_best = mean, 0
        else:
            self.since_best += 1
            if self.since_best >= config.SAFETY_PATIENCE:
                self.outcome = "stalled"
                return "end_stalled"
        return None

    def state(self):
        return dict(vars(self))

    @classmethod
    def from_state(cls, state):
        tracker = cls()
        tracker.__dict__.update(state)
        tracker.recent = list(state["recent"])
        return tracker


def mutation_strength(child_success, s_max=config.GA_S_MAX, s_min=config.GA_S_MIN):
    """GA mutation strength (revision 3 §5.1): high when few children succeeded, s_min when all did; s_max when there
    were no children to measure (nan)."""
    if np.isnan(child_success):
        return s_max
    return s_min + (s_max - s_min) * (1.0 - child_success)


def success_fraction(child_fitness, parent_fitness):
    """Fraction of offspring that beat their own parent; a tie with the parent counts as half a success."""
    return float(np.mean(np.where(child_fitness > parent_fitness, 1.0,
                                  np.where(child_fitness == parent_fitness, 0.5, 0.0))))


class StepSize:
    """ES and NEAT step size σ (revision 3 §5.2): the 1/5 success rule while the population learns; once at least
    MASTERY_FRACTION of the parents succeed, σ shrinks while fewer than MASTERY_FRACTION of the children succeed and
    grows otherwise. Kept within [ES_SIGMA_MIN, ES_SIGMA_MAX]; a nan input leaves σ unchanged."""

    def __init__(self, sigma0):
        self.sigma0 = float(sigma0)
        self.sigma = float(sigma0)

    def update(self, parent_success, child_success, success_rate):
        if parent_success >= config.MASTERY_FRACTION:
            if child_success < config.MASTERY_FRACTION:
                self.sigma *= config.ES_FACTOR
            elif child_success >= config.MASTERY_FRACTION:
                self.sigma /= config.ES_FACTOR
        elif success_rate > 0.2:
            self.sigma /= config.ES_FACTOR
        elif success_rate < 0.2:
            self.sigma *= config.ES_FACTOR
        self.sigma = float(np.clip(self.sigma, config.ES_SIGMA_MIN, config.ES_SIGMA_MAX))
        return self.sigma

    def reset(self):
        self.sigma = self.sigma0

    def state(self):
        return {"sigma0": self.sigma0, "sigma": self.sigma}

    @classmethod
    def from_state(cls, state):
        step = cls(state["sigma0"])
        step.sigma = float(state["sigma"])
        return step


# ----- revision 2.0 rules, used by ga.py and es.py until the evolution engine replaces them (plan Task 6) -----

class LevelTracker:
    """Advances one level after `streak_needed` consecutive generations in which at least `threshold` of the whole
    generation finished at pace (spec §9.1)."""

    def __init__(self, level=1, streak_needed=config.LEVEL_UP_STREAK, threshold=config.LEVEL_UP_FRACTION,
                 max_level=config.MAX_LEVEL):
        self.level = level
        self.streak = 0
        self.streak_needed = streak_needed
        self.threshold = threshold
        self.max_level = max_level

    def update(self, pace_fraction):
        """Record the share of one generation that finished at pace. Returns True if the level advanced."""
        if self.level >= self.max_level:
            return False
        if pace_fraction >= self.threshold:
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

- [ ] **Step 5: Move `success_fraction` out of es.py**

In `racing/es.py`, delete the function `success_fraction` (its four lines and the two blank lines after it) and
replace the import line `from racing.curriculum import LevelTracker, draw_seeds` with
`from racing.curriculum import LevelTracker, draw_seeds, success_fraction`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_curriculum.py tests/test_es.py -q`
Expected: all pass (31 curriculum tests).

- [ ] **Step 7: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add racing/config.py racing/curriculum.py racing/es.py tests/test_curriculum.py
git commit -m "Survival rule, fresh training seeds, mastery tracker with speed push and safety stop, two-phase step size

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/config.py racing/curriculum.py racing/es.py tests/test_curriculum.py
```

---

### Task 4: Checkpoints and settings record

**Files:**
- Create: `racing/checkpoint.py`
- Create: `tests/test_checkpoint.py`
- Modify: `scripts/run_notebooks.py` (the `SETTINGS` tuple and `settings()`)
- Modify: `tests/test_runner.py` (`test_run_record_holds_the_settings_and_code_version`)

**Interfaces:**
- Consumes: `racing.config` names listed in `SETTINGS_KEYS` below. All of them exist after Tasks 1–3 (`LEVELS`, `CIRCUIT_STRETCH`, `DIRECTION_SALT` from Task 1; `TARGET_FACTOR` from Task 2; `TRACKS_PER_GEN`, `SPEED_STEP`, `MASTERY_FRACTION`, `MASTERY_STREAK`, `SPEED_PATIENCE`, `SAFETY_PATIENCE`, `KEEP_MAX_FRACTION`, `ELITE_FLOOR`, `TRAIN_SEED_START` from Task 3).
- Produces (used by Tasks 5, 6, 9, 10, 11, 15):
  - `CHECKPOINT_NAME = "checkpoint.pkl"`
  - `SETTINGS_KEYS: tuple[str, ...]` — a plain tuple of config names that affect results. Later tasks append to it: Task 6 adds `"ES_POP"`, Task 7 the `NEAT_*` names, Task 10 `DATASET_SEED_RANGE` and `DATASET_TRACKS` (if it adds them), Task 11 the `DAGGER_*` names. `N_JOBS` and `TRACK_CACHE_SIZE` are never in it: they change speed and memory, not results, and differ between machines.
  - `class SettingsMismatch(RuntimeError)`
  - `settings_record() -> dict` — `{name: value}` normalized by a JSON round trip (tuples, ranges and arrays become lists, dict keys strings), so a record read back from a JSON file compares equal.
  - `check_settings(recorded) -> None` — raises `SettingsMismatch` naming each differing key as `NAME: recorded X, now Y` and telling the user to start over with `--fresh`.
  - `save(path, state) -> None` — pickle to a temporary file in the same folder, then `os.replace`; a failed save leaves the previous checkpoint and no temporary file.
  - `load(path) -> dict | None`
  - `completed(run_dir, name="summary.json") -> dict | None` — the summary if it exists and its `"settings"` equal `settings_record()`; `SettingsMismatch` if they differ or the summary has no settings record.

- [ ] **Step 1: Confirm the configuration names exist**

Run:
```bash
~/miniconda3/bin/python -c "
from racing import config
names = ['LEVELS', 'CIRCUIT_STRETCH', 'DIRECTION_SALT', 'TARGET_FACTOR', 'TRACKS_PER_GEN', 'SPEED_STEP',
         'MASTERY_FRACTION', 'MASTERY_STREAK', 'SPEED_PATIENCE', 'SAFETY_PATIENCE', 'KEEP_MAX_FRACTION',
         'ELITE_FLOOR', 'TRAIN_SEED_START']
print([n for n in names if not hasattr(config, n)])"
```
Expected: `[]`. If a name is missing, Tasks 1–3 are incomplete: finish them first; do not invent the value here.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_checkpoint.py`:

```python
import json
import pickle

import numpy as np
import pytest

from racing import checkpoint, config


def test_settings_record_survives_a_json_round_trip_unchanged():
    record = checkpoint.settings_record()
    assert json.loads(json.dumps(record)) == record
    assert record["VAL_SEEDS"] == list(config.VAL_SEEDS) and record["RUN_SEEDS"] == list(config.RUN_SEEDS)
    assert set(record["LEVELS"]) == {"1", "2", "3", "4"}                  # int keys read back as strings
    assert record["STEER_LEVELS"] == config.STEER_LEVELS.tolist()


def test_settings_keys_hold_the_result_settings_only():
    keys = checkpoint.SETTINGS_KEYS
    assert isinstance(keys, tuple) and len(set(keys)) == len(keys)
    assert {"LAPS", "LEVELS", "TARGET_FACTOR", "MASTERY_FRACTION", "GA_MUTATION_P", "MAX_EPOCHS", "RUN_SEEDS"} <= set(keys)
    assert not {"N_JOBS", "TRACK_CACHE_SIZE", "WINDOW_SIZE", "COLORS", "RESULTS_DIR"} & set(keys)
    assert all(hasattr(config, name) for name in keys)
    assert set(checkpoint.settings_record()) == set(keys) and "N_JOBS" not in checkpoint.settings_record()


def test_own_record_passes_the_check_after_a_json_round_trip():
    checkpoint.check_settings(json.loads(json.dumps(checkpoint.settings_record())))


def test_check_names_each_changed_setting(monkeypatch):
    recorded = checkpoint.settings_record()
    monkeypatch.setattr(config, "LAPS", 5)
    monkeypatch.setattr(config, "GA_MUTATION_P", 0.3)
    with pytest.raises(checkpoint.SettingsMismatch) as error:
        checkpoint.check_settings(recorded)
    message = str(error.value)
    assert "LAPS: recorded 3, now 5" in message and "GA_MUTATION_P: recorded 0.1, now 0.3" in message
    assert "--fresh" in message


def test_a_setting_missing_from_the_record_is_a_difference():
    recorded = checkpoint.settings_record()
    recorded.pop("LAPS")
    with pytest.raises(checkpoint.SettingsMismatch, match="LAPS: recorded None, now 3"):
        checkpoint.check_settings(recorded)


def test_save_and_load_round_trip_with_the_random_generator(tmp_path):
    rng = np.random.default_rng(0)
    state = {"weights": rng.normal(size=(4, 3)), "rng": rng.bit_generator.state, "used": [2000001, 2000002]}
    path = tmp_path / "run" / checkpoint.CHECKPOINT_NAME
    checkpoint.save(path, state)
    loaded = checkpoint.load(path)
    assert np.array_equal(loaded["weights"], state["weights"]) and loaded["used"] == [2000001, 2000002]
    restored = np.random.default_rng()
    restored.bit_generator.state = loaded["rng"]
    assert restored.random() == rng.random()


def test_load_missing_is_none(tmp_path):
    assert checkpoint.load(tmp_path / "missing.pkl") is None


class Unpicklable:
    def __reduce__(self):
        raise RuntimeError("cannot pickle")


def test_a_failed_save_keeps_the_previous_checkpoint(tmp_path):
    path = tmp_path / checkpoint.CHECKPOINT_NAME
    checkpoint.save(path, {"generation": 4})
    with pytest.raises(RuntimeError, match="cannot pickle"):
        checkpoint.save(path, {"generation": 5, "bad": Unpicklable()})
    assert checkpoint.load(path) == {"generation": 4}
    assert [p.name for p in tmp_path.iterdir()] == [checkpoint.CHECKPOINT_NAME]


def write_summary(folder, settings):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps({"generations": 7, "settings": settings}))


def test_completed_returns_a_matching_summary(tmp_path):
    write_summary(tmp_path, checkpoint.settings_record())
    assert checkpoint.completed(tmp_path)["generations"] == 7


def test_completed_without_a_summary_is_none(tmp_path):
    assert checkpoint.completed(tmp_path / "never_ran") is None


def test_completed_with_other_settings_is_refused(tmp_path, monkeypatch):
    write_summary(tmp_path, checkpoint.settings_record())
    monkeypatch.setattr(config, "TRACKS_PER_GEN", 5)
    with pytest.raises(checkpoint.SettingsMismatch, match="TRACKS_PER_GEN"):
        checkpoint.completed(tmp_path)


def test_a_summary_without_settings_is_refused(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"generations": 7}))
    with pytest.raises(checkpoint.SettingsMismatch, match="no settings record"):
        checkpoint.completed(tmp_path)


def test_completed_reads_another_summary_name(tmp_path):
    (tmp_path / "dagger.json").write_text(json.dumps({"rounds": 3, "settings": checkpoint.settings_record()}))
    assert checkpoint.completed(tmp_path, "dagger.json")["rounds"] == 3


def test_checkpoints_are_plain_pickles(tmp_path):
    path = tmp_path / checkpoint.CHECKPOINT_NAME
    checkpoint.save(path, {"generation": 1})
    with open(path, "rb") as f:
        assert pickle.load(f) == {"generation": 1}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_checkpoint.py -q`
Expected: FAIL — `ImportError: cannot import name 'checkpoint' from 'racing'`.

- [ ] **Step 4: Write the module**

Create `racing/checkpoint.py`:

```python
"""Checkpoints of long runs and the record of the settings a run was made with (revision 3 §8.2). A run saves its
state after every generation or round and continues from it after an interruption; a run made with other settings
is never resumed and its results are never reused."""
import json
import os
import pickle
from pathlib import Path

import numpy as np

from racing import config

CHECKPOINT_NAME = "checkpoint.pkl"

# Configuration values that affect results. The worker count and the track cache only change speed and memory, and
# the viewer's values change nothing a run computes, so they are left out. Tasks that add such values append them.
SETTINGS_KEYS = (
    "DT", "GRIP", "YAW_MIN_SPEED", "SPEED_SCALE", "STALL_TIME", "TIME_LIMIT_FACTOR", "LAPS", "TIME_EPS",
    "N_RAYS", "RAY_ANGLES", "RAY_RANGE", "RAY_MIN_STEP", "RAY_MAX_ITERATIONS",
    "CENTERLINE_SPACING", "GRID_RES", "GRID_MARGIN", "OVERLAP_MARGIN", "MAX_TRACK_ATTEMPTS", "PROGRESS_BACK",
    "PROGRESS_AHEAD", "LEVELS", "MAX_LEVEL", "CIRCUIT_STRETCH", "DIRECTION_SALT",
    "N_INPUTS", "STEER_LEVELS", "THROTTLE_LEVELS", "N_ACTIONS", "BASELINE_HIDDEN",
    "EXPERT_LOOKAHEAD_BASE", "EXPERT_LOOKAHEAD_TIME", "EXPERT_GRIP_MARGIN", "EXPERT_SPEED_BAND", "EXPERT_SIGHT",
    "EXPERT_VALIDATION_TIME", "PERTURB_INTERVAL", "PERTURB_DURATION", "DATASET_SALT",
    "VAL_SEEDS", "TEST_SEEDS", "RUN_SEEDS", "SEARCH_SEED", "TRAIN_SEED_START",
    "TRACKS_PER_GEN", "TARGET_FACTOR", "SPEED_STEP", "MASTERY_FRACTION", "MASTERY_STREAK", "SPEED_PATIENCE",
    "SAFETY_PATIENCE", "KEEP_MAX_FRACTION", "ELITE_FLOOR",
    "GA_POP", "GA_TOURNAMENT", "GA_CROSSOVER_P", "GA_MUTATION_P", "GA_S_MAX", "GA_S_MIN",
    "ES_MU", "ES_SIGMA0", "ES_FACTOR", "ES_SIGMA_MIN", "ES_SIGMA_MAX",
    "LEARNING_RATE", "BATCH_SIZE", "MAX_EPOCHS", "PATIENCE", "KT_TRIALS", "KT_BAYES_INITIAL", "HYPERBAND_FACTOR",
    "HYPERBAND_ITERATIONS", "SIZE_RULE_TOLERANCE", "STEPS_PER_EXECUTION", "JIT_COMPILE",
)


class SettingsMismatch(RuntimeError):
    """The settings recorded with a run differ from the current configuration."""


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, range):
        return list(value)
    raise TypeError(f"a setting of type {type(value).__name__} cannot be recorded")


def plain(value):
    """`value` as it reads back from JSON: lists for tuples, ranges and arrays, strings for dict keys."""
    return json.loads(json.dumps(value, default=_json_value))


def settings_record():
    """{name: value} of the settings that affect results, in JSON form."""
    return {name: plain(getattr(config, name)) for name in SETTINGS_KEYS}


def check_settings(recorded):
    """Raise SettingsMismatch if `recorded` differs from the current settings, naming every difference."""
    current = settings_record()
    recorded = plain(recorded)
    differences = [f"{name}: recorded {recorded.get(name)!r}, now {current.get(name)!r}"
                   for name in sorted(set(recorded) | set(current)) if recorded.get(name) != current.get(name)]
    if differences:
        raise SettingsMismatch("this run was made with other settings (" + "; ".join(differences)
                               + "); start over with scripts/run_notebooks.py --fresh")


def save(path, state):
    """Pickle `state` to `path` through a temporary file in the same folder, so an interruption while writing
    leaves the previous checkpoint intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with open(temporary, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load(path):
    """The saved state, or None if there is no checkpoint."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def completed(run_dir, name="summary.json"):
    """The summary of a completed run in `run_dir`, or None if the run has not completed. A summary made with other
    settings, or without a settings record (an earlier version), raises SettingsMismatch."""
    path = Path(run_dir) / name
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    if "settings" not in summary:
        raise SettingsMismatch(f"{path} has no settings record (made by an earlier version); "
                               "start over with scripts/run_notebooks.py --fresh")
    check_settings(summary["settings"])
    return summary
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_checkpoint.py -q`
Expected: PASS (14 tests).

- [ ] **Step 6: Record the same settings in the run record**

The runner's hand-written `SETTINGS` tuple names values that revision 3 removes (`PACE_FACTOR`, `LEVEL_UP_FRACTION`, `GA_KEPT`, `GA_KEEP_MAX_FRACTION`, `EVAL_BUDGET`). The run record now holds the checkpoint settings record, so every place that records settings agrees.

In `tests/test_runner.py`, replace `test_run_record_holds_the_settings_and_code_version` with:

```python
def test_run_record_holds_the_settings_and_code_version(runner, tmp_path, monkeypatch):
    from racing import checkpoint
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir: None)
    assert run_main(runner, tmp_path, []) == 0
    info = json.loads((tmp_path / "info.json").read_text())
    assert info["settings"] == checkpoint.settings_record()
    assert info["settings"]["RUN_SEEDS"] == list(runner.config.RUN_SEEDS) and "N_JOBS" not in info["settings"]
    assert info["n_jobs"] == runner.config.N_JOBS
    assert len(info["git_commit"]) == 40 and int(info["git_commit"], 16) >= 0
    assert isinstance(info["git_dirty"], bool)
```

Run: `~/miniconda3/bin/python -m pytest tests/test_runner.py::test_run_record_holds_the_settings_and_code_version -q`
Expected: FAIL (the record still holds the old tuple, or `AttributeError` for a removed name).

In `scripts/run_notebooks.py`, replace the import line and the `SETTINGS` tuple:

```python
from racing import config  # noqa: E402

NOTEBOOK_DIR = PROJECT / "notebooks"
KERNEL = "ec-racing"
SETTINGS = ("N_JOBS", "STEPS_PER_EXECUTION", "JIT_COMPILE", "GA_S_MAX", "GA_S_MIN", "PACE_FACTOR",
            "LEVEL_UP_FRACTION", "GA_KEPT", "GA_KEEP_MAX_FRACTION",
            "MAX_EPOCHS", "RUN_SEEDS", "EVAL_BUDGET")
```

with

```python
from racing import checkpoint, config  # noqa: E402

NOTEBOOK_DIR = PROJECT / "notebooks"
KERNEL = "ec-racing"
```

and replace `settings()` with

```python
def settings():
    """The configuration values that shape the reported run: the checkpoint settings record (revision 3 §8.2)."""
    return checkpoint.settings_record()
```

Run: `~/miniconda3/bin/python -m pytest tests/test_runner.py -q`
Expected: PASS.

- [ ] **Step 7: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add racing/checkpoint.py tests/test_checkpoint.py scripts/run_notebooks.py tests/test_runner.py
git commit -m "Checkpoints and settings record for resumable runs (revision 3 §8.2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/checkpoint.py tests/test_checkpoint.py scripts/run_notebooks.py tests/test_runner.py
```

---

### Task 5: Progress: status file, milestones, live line, monitor

**Files:**
- Create: `racing/progress.py`
- Create: `tests/test_progress.py`
- Modify: `racing/config.py` (add `IMITATION_APPROACHES`, `RUN_ORDER`, `STATUS_PATH`, `DAGGER_MAPS`, `DAGGER_STREAK` if absent)
- Modify: `tests/conftest.py` (autouse fixture: every test gets its own status file and run log)

**Interfaces:**
- Consumes: `config.LABELS`, `config.DIFFICULTIES`, `config.MASTERY_FRACTION`, `config.MASTERY_STREAK` (Task 3), `config.RUN_LOG_PATH`.
- Produces (used by Tasks 6, 11, 15):
  - `write_status(status, path=None)`, `read_status(path=None) -> dict | None`, `milestone(message, path=None) -> str`
  - `run_number(approach, seed) -> (k, total)` — 1-based position in `config.RUN_ORDER`; `(0, total)` for a pair that is not part of the experiment (pilots, viewer).
  - `bar(fraction, width=20) -> str`, `duration(seconds) -> str` (`"0h 42m"`, `"2h 13m"`, `"1d 4h 03m"`), `format_status(status) -> str`
  - `class Monitor(status_path=None, log_path=None, out=None, interval=1.0, live=None, period=60.0)` with `poll()`, `start()`, `stop()`, `finish_line()`; attributes `live`, `thread`.
  - `main(argv=None) -> int` — `python -m racing.progress [--status PATH] [--log PATH] [--interval S]`.
  - Status dicts: evolution `{approach, seed, stage: "evolution" | "speed push", level, generation, min_share, streak, target_reduction, elapsed}`; DAgger `{approach, seed, stage: "dagger", level, round, checks, streak, elapsed}`. `format_status` never raises on a missing key.
  - `tests/conftest.py` fixture `private_progress_files` (autouse): `config.STATUS_PATH` and `config.RUN_LOG_PATH` point into a temporary folder in every test (Review Focus 4: tests never touch `results/`).

- [ ] **Step 1: Add the configuration values (if absent)**

Whichever of Tasks 5, 7, 11 and 15 comes first adds these; check with `grep -n "RUN_ORDER\|STATUS_PATH\|IMITATION_APPROACHES\|DAGGER_MAPS\|DAGGER_STREAK" racing/config.py` and add only the missing ones.

In `racing/config.py`, after `EVOLUTION_APPROACHES = ...`:

```python
IMITATION_APPROACHES = ("baseline", *TUNER_APPROACHES)
RUN_ORDER = tuple((a, s) for a in (*IMITATION_APPROACHES, *EVOLUTION_APPROACHES) for s in RUN_SEEDS)  # revision 3 §8.3
```

Before `# --- Approaches and colors` (Task 11 adds the remaining DAgger values to this block):

```python
# --- DAgger (revision 3 §6.3) ---
DAGGER_MAPS = 20             # fresh Extra hard maps driven per round, and checked after it
DAGGER_STREAK = 3            # consecutive rounds that must reach the goal
```

At the end of `# --- Paths ---`:

```python
STATUS_PATH = RESULTS_DIR / "status.json"
```

- [ ] **Step 2: Keep every test's progress files private**

Append to `tests/conftest.py`:

```python
@pytest.fixture(autouse=True)
def private_progress_files(tmp_path_factory, monkeypatch):
    """The status file and the run log of every test live in a temporary folder, never in results/
    (revision 3 §8.3)."""
    from racing import config
    folder = tmp_path_factory.mktemp("progress")
    monkeypatch.setattr(config, "STATUS_PATH", folder / "status.json")
    monkeypatch.setattr(config, "RUN_LOG_PATH", folder / "run_log.txt")
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_progress.py`:

```python
import io
import math
import time

import numpy as np
import pytest

from racing import config, progress


def status(**changes):
    base = {"approach": "ga", "seed": 1, "stage": "evolution", "level": 3, "generation": 412, "min_share": 0.84,
            "streak": 1, "target_reduction": float("nan"), "elapsed": 2 * 3600 + 13 * 60}
    return {**base, **changes}


@pytest.fixture
def order(monkeypatch):
    monkeypatch.setattr(config, "RUN_ORDER", tuple((a, s) for a in ("baseline", "ga") for s in (0, 1, 2)))


def test_status_round_trip_adds_the_time(tmp_path):
    path = tmp_path / "status.json"
    progress.write_status(status(), path)
    read = progress.read_status(path)
    assert read["generation"] == 412 and math.isnan(read["target_reduction"]) and "updated" in read
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_status_defaults_to_the_configured_file():
    progress.write_status(status())
    assert config.STATUS_PATH.exists() and progress.read_status()["approach"] == "ga"


def test_status_accepts_numpy_values(tmp_path):
    progress.write_status(status(generation=np.int64(5), min_share=np.float64(0.5), streak=np.int64(2)),
                          tmp_path / "s.json")
    assert progress.read_status(tmp_path / "s.json")["generation"] == 5


def test_missing_or_unreadable_status_is_none(tmp_path):
    assert progress.read_status(tmp_path / "missing.json") is None
    (tmp_path / "broken.json").write_text('{"approach": "ga", "gen')
    assert progress.read_status(tmp_path / "broken.json") is None
    (tmp_path / "list.json").write_text("[1, 2]")
    assert progress.read_status(tmp_path / "list.json") is None


def test_milestones_are_appended_with_a_time(tmp_path):
    path = tmp_path / "log.txt"
    first = progress.milestone("GA seed 0: started", path)
    progress.milestone("GA seed 0: Easy mastered at generation 41, now Medium", path)
    lines = path.read_text().splitlines()
    assert lines[0] == first and lines[0].endswith("  GA seed 0: started") and len(lines) == 2
    assert lines[0][4] == "-" and lines[0][10] == "T"                     # ISO date and time


def test_milestones_default_to_the_run_log():
    progress.milestone("NEAT seed 2: started")
    assert config.RUN_LOG_PATH.read_text().endswith("NEAT seed 2: started\n")


def test_run_number_follows_the_run_order(order):
    assert progress.run_number("baseline", 0) == (1, 6)
    assert progress.run_number("ga", 1) == (5, 6)
    assert progress.run_number("neat", 0) == (0, 6)                        # not part of this experiment


def test_bar_has_a_fixed_width_and_eighth_steps():
    assert progress.bar(1.0, 4) == "████" and progress.bar(0.0, 4) == "    "
    assert progress.bar(0.5, 4) == "██  " and progress.bar(0.0625, 4) == "▎   "
    assert progress.bar(2.0, 4) == "████" and progress.bar(float("nan"), 4) == "    "
    assert all(len(progress.bar(k / 37, 20)) == 20 for k in range(38))


def test_duration():
    assert progress.duration(2470) == "0h 41m" and progress.duration(7980) == "2h 13m"
    assert progress.duration(90061) == "1d 1h 01m" and progress.duration(None) == "–"


def test_evolution_line(order):
    line = progress.format_status(status())
    assert line.startswith("[run 5/6] GA seed 1 · Hard · gen 412 · worst map 84% / 90% ")
    assert progress.bar(0.84 / config.MASTERY_FRACTION) in line
    assert line.endswith(" · streak 1/3 · 2h 13m")


def test_speed_push_line(order):
    line = progress.format_status(status(stage="speed push", level=4, target_reduction=0.143, elapsed=90061))
    assert "GA seed 1 · Extra hard · speed push −14.3% · gen 412 · worst map" in line
    assert line.endswith("1d 1h 01m")


def test_dagger_line(order):
    line = progress.format_status({"approach": "baseline", "seed": 1, "stage": "dagger", "level": 4, "round": 7,
                                   "checks": 16, "streak": 1, "elapsed": 2520})
    assert line == "[run 2/6] Baseline seed 1 · DAgger round 7 · checks 16/20 · streak 1/3 · 0h 42m"


def test_incomplete_status_still_formats(order):
    assert progress.format_status({"approach": "ga"}).startswith("[run –/6] GA seed ?")
    assert progress.format_status({"approach": "ga", "seed": 0, "min_share": None, "level": None})
    assert progress.format_status({})


def make_monitor(tmp_path, live, period=60.0, interval=1.0):
    out = io.StringIO()
    monitor = progress.Monitor(tmp_path / "status.json", tmp_path / "log.txt", out=out, live=live, period=period,
                               interval=interval)
    return monitor, out


def test_monitor_prints_new_log_lines_then_the_live_line(tmp_path, order):
    (tmp_path / "log.txt").write_text("old line\n")
    monitor, out = make_monitor(tmp_path, live=True)
    progress.milestone("GA seed 1: started", tmp_path / "log.txt")
    progress.write_status(status(), tmp_path / "status.json")
    monitor.poll()
    text = out.getvalue()
    assert "old line" not in text and "GA seed 1: started\n" in text
    assert text.endswith("\r" + progress.format_status(status()))
    progress.milestone("GA seed 1: Hard mastered at generation 500, now Extra hard", tmp_path / "log.txt")
    monitor.poll()
    later = out.getvalue()[len(text):]
    assert later.startswith("\r") and "Hard mastered" in later and "started" not in later


def test_without_a_terminal_the_status_is_printed_once_per_period(tmp_path, order):
    progress.write_status(status(), tmp_path / "status.json")
    monitor, out = make_monitor(tmp_path, live=False, period=60.0)
    monitor.poll()
    monitor.poll()
    assert out.getvalue() == progress.format_status(status()) + "\n"
    every, out_every = make_monitor(tmp_path, live=False, period=0.0)
    every.poll()
    every.poll()
    assert out_every.getvalue().count("\n") == 2 and "\r" not in out_every.getvalue()


def test_live_mode_follows_the_terminal():
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    assert progress.Monitor(out=Terminal()).live and not progress.Monitor(out=io.StringIO()).live


def test_monitor_survives_missing_and_broken_status(tmp_path):
    monitor, out = make_monitor(tmp_path, live=False, period=0.0)
    monitor.poll()                                                         # no status file and no run log yet
    (tmp_path / "status.json").write_text("{not json")
    monitor.poll()
    assert out.getvalue().count("waiting for the first status") == 2


def test_monitor_follows_a_replaced_log(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("one\ntwo\n")
    monitor, out = make_monitor(tmp_path, live=False, period=3600.0)
    log.write_text("fresh\n")                                              # --fresh removed the old run log
    monitor.poll()
    assert "fresh\n" in out.getvalue() and "two" not in out.getvalue()


def test_a_partly_written_log_line_waits_for_its_end(tmp_path):
    monitor, out = make_monitor(tmp_path, live=False, period=3600.0)
    with open(tmp_path / "log.txt", "a") as f:
        f.write("half")
    monitor.poll()
    with open(tmp_path / "log.txt", "a") as f:
        f.write(" line\n")
    monitor.poll()
    assert "half line\n" in out.getvalue() and out.getvalue().count("half") == 1


def test_start_and_stop_end_the_live_line(tmp_path, order):
    progress.write_status(status(), tmp_path / "status.json")
    monitor, out = make_monitor(tmp_path, live=True, interval=0.01)
    monitor.start()
    time.sleep(0.05)
    monitor.stop()
    assert out.getvalue().endswith("\n") and progress.format_status(status()) in out.getvalue()
    assert not monitor.thread.is_alive()


def test_main_shows_recent_milestones_then_the_status_until_interrupted(tmp_path, monkeypatch, capsys, order):
    log = tmp_path / "log.txt"
    log.write_text("".join(f"line {k}\n" for k in range(15)))
    progress.write_status(status(), tmp_path / "status.json")

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(progress.time, "sleep", interrupt)
    assert progress.main(["--status", str(tmp_path / "status.json"), "--log", str(log)]) == 0
    printed = capsys.readouterr().out
    assert "line 4\n" not in printed and "line 5\n" in printed and "line 14\n" in printed
    assert progress.format_status(status()) in printed
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_progress.py -q`
Expected: FAIL — `ImportError: cannot import name 'progress' from 'racing'`.

- [ ] **Step 5: Write the module**

Create `racing/progress.py`:

```python
"""Progress of a long experiment (revision 3 §8.3): the running run writes its state to a status file and its
milestones to the run log; the runner, or a monitor in another terminal, shows the new milestones and a one-line
status.

Watch a run started in the background with:  ~/miniconda3/bin/python -m racing.progress
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from racing import config

EIGHTHS = " ▏▎▍▌▋▊▉"
FULL = "█"
RECENT = 10                  # run-log lines the separate monitor shows when it starts
WAITING = "waiting for the first status…"


def _json_value(value):
    if hasattr(value, "item"):                     # numpy scalars
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_status(status, path=None):
    """Replace the status file with `status` plus the time it was written."""
    path = Path(config.STATUS_PATH if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")    # same folder, so the replace is atomic
    temporary.write_text(json.dumps({**status, "updated": datetime.now().isoformat(timespec="seconds")},
                                    default=_json_value))
    os.replace(temporary, path)


def read_status(path=None):
    """The current status, or None if there is none or it cannot be read."""
    path = Path(config.STATUS_PATH if path is None else path)
    try:
        status = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return status if isinstance(status, dict) else None


def milestone(message, path=None):
    """Append a timestamped line to the run log and return it."""
    path = Path(config.RUN_LOG_PATH if path is None else path)
    line = f"{datetime.now().isoformat(timespec='seconds')}  {message}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line + "\n")
    return line


def run_number(approach, seed):
    """(position, total) of a run in config.RUN_ORDER; position 0 for a run outside the experiment."""
    order = list(config.RUN_ORDER)
    try:
        return order.index((approach, seed)) + 1, len(order)
    except ValueError:
        return 0, len(order)


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def bar(fraction, width=20):
    """A bar of `width` characters filled to `fraction` (0 to 1) in steps of one eighth of a character."""
    fraction = _number(fraction)
    fraction = 0.0 if math.isnan(fraction) else min(max(fraction, 0.0), 1.0)
    eighths = int(fraction * width * 8)
    text = FULL * (eighths // 8) + (EIGHTHS[eighths % 8] if eighths % 8 else "")
    return text.ljust(width)


def percent(value):
    value = _number(value)
    return "–" if math.isnan(value) else f"{100 * value:.0f}%"


def duration(seconds):
    """'0h 42m', '2h 13m' or '1d 4h 03m'; '–' when unknown."""
    seconds = _number(seconds)
    if math.isnan(seconds):
        return "–"
    minutes = int(max(seconds, 0.0)) // 60
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    return f"{days}d {hours}h {minutes:02d}m" if days else f"{hours}h {minutes:02d}m"


def format_status(status):
    """The one-line status, for example
    '[run 14/24] GA seed 1 · Hard · gen 412 · worst map 84% / 90% ███████████████▊     · streak 1/3 · 2h 13m'."""
    status = status or {}
    approach, seed = status.get("approach"), status.get("seed")
    position, total = run_number(approach, seed)
    name = config.LABELS.get(approach, approach or "?")
    parts = [f"[run {position or '–'}/{total}] {name} seed {'?' if seed is None else seed}"]
    stage = status.get("stage", "evolution")
    streak = status.get("streak", 0)
    if stage == "dagger":
        parts += [f"DAgger round {status.get('round', '?')}",
                  f"checks {status.get('checks', '?')}/{config.DAGGER_MAPS}",
                  f"streak {streak}/{config.DAGGER_STREAK}"]
    else:
        level = status.get("level")
        parts.append(config.DIFFICULTIES.get(level, f"level {'?' if level is None else level}"))
        if stage == "speed push":
            reduction = _number(status.get("target_reduction"))
            parts.append("speed push" if math.isnan(reduction) else f"speed push −{100 * reduction:.1f}%")
        share = _number(status.get("min_share"))
        fill = 0.0 if math.isnan(share) else share / config.MASTERY_FRACTION
        parts += [f"gen {status.get('generation', '?')}",
                  f"worst map {percent(share)} / {percent(config.MASTERY_FRACTION)} {bar(fill)}",
                  f"streak {streak}/{config.MASTERY_STREAK}"]
    parts.append(duration(status.get("elapsed")))
    return " · ".join(parts)


class Monitor:
    """Shows the run log's new lines and the status line (revision 3 §8.3). In a terminal the status line is
    updated in place; otherwise, as under nohup, it is printed as a whole line at most once per `period` seconds."""

    def __init__(self, status_path=None, log_path=None, out=None, interval=1.0, live=None, period=60.0):
        self.status_path = status_path
        self.log_path = Path(config.RUN_LOG_PATH if log_path is None else log_path)
        self.out = sys.stdout if out is None else out
        self.interval = interval
        self.live = bool(getattr(self.out, "isatty", lambda: False)()) if live is None else bool(live)
        self.period = period
        self.offset = self.log_path.stat().st_size if self.log_path.exists() else 0   # only lines written from now
        self.partial = ""
        self.shown = 0                     # length of the live line on screen
        self.last_print = None
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.thread = None

    def new_log_lines(self):
        try:
            size = self.log_path.stat().st_size
        except OSError:
            return []
        if size < self.offset:             # the log was replaced (a fresh run)
            self.offset, self.partial = 0, ""
        if size == self.offset:
            return []
        with open(self.log_path, "rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        self.offset = size
        *lines, self.partial = (self.partial + data.decode("utf-8", errors="replace")).split("\n")
        return lines

    def poll(self):
        """Print the run log's new lines as permanent lines, then refresh the status line."""
        with self.lock:
            lines = self.new_log_lines()
            status = read_status(self.status_path)
            text = WAITING if status is None else format_status(status)
            if self.live:
                if lines and self.shown:
                    self.out.write("\r" + " " * self.shown + "\r")
                    self.shown = 0
                for line in lines:
                    self.out.write(line + "\n")
                self.out.write("\r" + text + " " * max(self.shown - len(text), 0))
                self.shown = len(text)
            else:
                for line in lines:
                    self.out.write(line + "\n")
                now = time.monotonic()
                if self.last_print is None or now - self.last_print >= self.period:
                    self.out.write(text + "\n")
                    self.last_print = now
            self.out.flush()

    def _follow(self):
        while not self.stopping.wait(self.interval):
            self.poll()

    def start(self):
        """Poll now, then every `interval` seconds in a background thread."""
        self.poll()
        self.thread = threading.Thread(target=self._follow, daemon=True)
        self.thread.start()

    def finish_line(self):
        """End a live status line, so later output starts on a new line."""
        with self.lock:
            if self.live and self.shown:
                self.out.write("\n")
                self.out.flush()
                self.shown = 0

    def stop(self):
        """Stop the background thread, poll a last time and end the live line."""
        self.stopping.set()
        if self.thread is not None:
            self.thread.join()
        self.poll()
        self.finish_line()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Show the progress of the running EC Racing experiment.")
    parser.add_argument("--status", default=None, help="status file (default: results/status.json)")
    parser.add_argument("--log", default=None, help="run log (default: results/run_log.txt)")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between updates")
    args = parser.parse_args(argv)
    log_path = Path(config.RUN_LOG_PATH if args.log is None else args.log)
    if log_path.exists():
        for line in log_path.read_text().splitlines()[-RECENT:]:
            print(line)
    monitor = Monitor(args.status, log_path, interval=args.interval)
    try:
        while True:
            monitor.poll()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.finish_line()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_progress.py -q`
Expected: PASS (21 tests).

- [ ] **Step 7: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all pass (the autouse fixture changes no existing test's outcome: the runner tests pass their own log paths).

- [ ] **Step 8: Commit**

```bash
git add racing/progress.py tests/test_progress.py racing/config.py tests/conftest.py
git commit -m "Progress: status file, milestones, live status line and monitor (revision 3 §8.3)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/progress.py tests/test_progress.py racing/config.py tests/conftest.py
```

---

### Task 6: Evolution engine with GA and ES

**Files:**
- Create: `racing/evolution.py`, `tests/test_evolution.py`
- Rewrite: `racing/ga.py`, `racing/es.py`, `tests/test_ga.py`, `tests/test_es.py`
- Modify: `racing/population.py` (remove `finish_fractions`, `make_report`, `RunLogger`, `select_champion`, `save_champion` and the revision 2.0 evaluator `evaluate_weights` / `_evaluate_track`; add `select_final` and `save_dense_champion` if Task 2 did not; `clear_champion` also removes `.json`)
- Modify: `racing/curriculum.py` (remove the revision 2.0 `LevelTracker` and `draw_seeds`)
- Modify: `racing/nets.py` (`load_champion` drives dense `.npy` weights in the Keras model's architecture; `dense_spec` if Task 2 did not add it)
- Modify: `racing/config.py` (GA/ES blocks, budget; remove `PACE_FACTOR`, `LEVEL_UP_FRACTION`, `LEVEL_UP_STREAK`), `racing/checkpoint.py` (`SETTINGS_KEYS` gains `"ES_POP"`)
- Modify (minimal adaptation so the fast suite stays green; Tasks 12, 14, 15 and 16 rewrite these files): `racing/app.py`, `racing/evaluate.py`, `scripts/pilots.py`, `scripts/make_notebooks.py`
- Modify tests: `tests/test_population.py`, `tests/test_curriculum.py`, `tests/test_nets.py`, `tests/test_evaluate.py`, `tests/test_app.py`, `tests/test_pilots.py`, `tests/test_end_to_end.py`

**Interfaces:**
- Consumes:
  - Task 2: `population.TrialResult(fitness, success, laps, finish_time)` with properties `all_success`, `lap_share`, `map_shares`, `min_share`, `lap_times`; `population.evaluate_population(population, seeds, level, factor, n_jobs=1, backend="loky") -> TrialResult`; `population.fresh_workers(n_jobs)`; `population.clear_champion`; `nets.DensePopulation(weights, hidden, activation)` with `weights`, `hidden`, `activation`, `__len__`, `subset`, `policy`, `driver`; `nets.init_population`, `nets.to_keras`, `nets.to_flat`, `nets.from_flat`, `nets.from_keras`. `population.select_final`, `population.save_dense_champion` and `nets.dense_spec` are listed in the contract under Tasks 2/6: Step 1 checks whether Task 2 already added them.
  - Task 3: `curriculum.SeedSource(rng, start=..., used=())` with `draw(count)`, `state()`; `curriculum.survivors(rng, fitness, all_success, lap_share, cap, floor)`; `curriculum.MasteryTracker(level=1, factor=...)` with `level`, `factor`, `streak`, `goal_reached`, `mastered_factor`, `outcome`, `update(min_share)`, `done`, `target_reduction`, `state()`, `from_state()`; `curriculum.mutation_strength(child_success, s_max, s_min)`; `curriculum.StepSize(sigma0)` with `sigma`, `update(parent_success, child_success, success_rate)`, `reset()`, `state()`, `from_state()`; `curriculum.success_fraction(child_fitness, parent_fitness)`.
  - Task 4: `checkpoint.CHECKPOINT_NAME`, `checkpoint.save`, `checkpoint.load`, `checkpoint.check_settings`, `checkpoint.completed`, `checkpoint.settings_record`, `checkpoint.SettingsMismatch`.
  - Task 5: `progress.write_status`, `progress.milestone`, `progress.read_status`; the autouse `private_progress_files` fixture.
- Produces (used by Tasks 9, 11, 12, 14, 15, 16):
  - `evolution.run(method, seed, out_dir, champion_dir=config.CHAMPIONS_DIR, n_jobs=1, on_generation=None, start_level=1, max_generations=None, rng_seed=None) -> dict` — returns the existing summary at once when `checkpoint.completed(out_dir)` finds a completed run with matching settings; otherwise trains or resumes. With `champion_dir=None` no final controller is selected or saved and no status or milestone is written (viewer, pilots).
  - `evolution.LOG_NAME = "log.csv"`, `evolution.SUMMARY_NAME = "summary.json"`, `evolution.LOG_FIELDS` (contract list plus nothing else), `evolution.Method` (typing.Protocol), `evolution.start_log(path, fields=LOG_FIELDS)`, `evolution.append_log(path, report, fields=LOG_FIELDS)`, `evolution.log_header(path)`, `evolution.truncate_log(path, generation, fields=LOG_FIELDS)`. The columns of `log.csv` are `LOG_FIELDS` followed by the keys of `method.log_fields()` (NEAT: `species`, `mean_nodes`, `mean_conns`), fixed when the run starts; a resumed run keeps the header of its existing log.
  - Report keys (contract) plus `n_kept` (survivors chosen this generation). `kept_success` and `child_success` are the success shares of the cars carried over unchanged and of this generation's new children.
  - Summary keys (contract) plus `pop_size` (taken after `initialize()`) and `tightenings` (list of `{"factor", "generation", "evals", "elapsed"}`, one per `"tightened"` event). `levels` holds `{"level", "generation", "evals", "elapsed"}`: level L (1–3) when it is cleared by a `"level_up"` event, and level `MAX_LEVEL` when the `"goal"` event fires, so a speed push run (`start_level=MAX_LEVEL`) has at most the goal entry.
  - The engine calls `population.evaluate_population`, `population.select_final`, `population.fresh_workers` and `population.clear_champion` through the module (`from racing import population`), so tests patch one place: `monkeypatch.setattr(population, "evaluate_population", fake)`.
  - `ga.GA(pop_size=config.GA_POP, hidden=config.BASELINE_HIDDEN, activation="relu", s_max=config.GA_S_MAX)`, `ga.tournament`, `ga.run_ga(...)` (contract signature).
  - `es.ES(approach="es", pop_size=config.ES_POP, parents=config.ES_MU, hidden=..., activation="relu", initial=None, sigma0=config.ES_SIGMA0)` with public `ids`, `parent_ids`; `es.seeded_population(rng, flat, n, sigma)`; `es.run_es(...)` (contract signature); `es.dagger_baseline(seed, runs_dir=None)`; `es.trained_baseline(seed, champion_dir=...)` stays unchanged until Tasks 14–16 move its callers to `dagger_baseline`.
  - `population.select_final(population, candidates, factor, n_jobs=1, seeds=config.VAL_SEEDS) -> (index, successes, fitness)`, `population.save_dense_champion(approach, seed, population, index, champion_dir=config.CHAMPIONS_DIR) -> Path` (writes `.keras` and float64 `.npy` only; the architecture is read back from the Keras file by `nets.dense_spec`, so a dense champion never has a `.json`, which marks a NEAT champion).
  - `config.ES_POP = 550`; `config.GA_KEPT`, `GA_KEEP_MAX_FRACTION`, `ES_LAMBDA`, `EVAL_BUDGET`, `PACE_FACTOR`, `LEVEL_UP_FRACTION`, `LEVEL_UP_STREAK` removed; `curriculum.LevelTracker`, `curriculum.draw_seeds`, `population.evaluate_weights` and `population._evaluate_track` removed with their tests (Task 10 removes `config.TRAIN_SEEDS`, which `draw_seeds` reads).

- [ ] **Step 1: Check what Tasks 2–5 provide**

Run:
```bash
grep -n "def select_final\|def save_dense_champion\|class TrialResult\|def evaluate_population" racing/population.py
grep -n "def dense_spec\|class DensePopulation" racing/nets.py
grep -n "class SeedSource\|def survivors\|class MasteryTracker\|class StepSize\|def success_fraction\|def mutation_strength" racing/curriculum.py
ls racing/checkpoint.py racing/progress.py
```
Expected: `TrialResult`, `evaluate_population`, `DensePopulation` and every curriculum name are present, and both modules exist. If `select_final`, `save_dense_champion` or `dense_spec` is missing, Steps 8–9 add it with the code given there; if present, keep Task 2's version, check that its signature matches the contract, and skip that code block.

- [ ] **Step 2: Write the failing engine tests**

Create `tests/test_evolution.py`:

```python
import json
import math

import numpy as np
import pandas as pd
import pytest

from racing import checkpoint, config, curriculum, evolution, ga, paths, population, progress
from racing.population import TrialResult


def fake_evaluate(success_rate):
    """Evaluation without simulation: fitness falls with the size of a car's weights, and the fittest
    `success_rate` share of the population succeeds on every map."""
    def evaluate(population, seeds, level, factor, n_jobs=1, backend="loky"):
        weights = population.weights
        n = len(weights)
        fitness = 1.0 / (1.0 + np.sum(weights ** 2, axis=1))
        rank = np.argsort(np.argsort(-fitness, kind="stable"), kind="stable")
        success = np.tile(rank < success_rate * n, (len(seeds), 1))
        return TrialResult(fitness=fitness + success[0], success=success,
                           laps=np.where(success, config.LAPS, 1), finish_time=np.where(success, 90.0, np.nan))
    return evaluate


@pytest.fixture
def evaluation(monkeypatch):
    def use(success_rate):
        monkeypatch.setattr(population, "evaluate_population", fake_evaluate(success_rate))
    return use


def train(tmp_path, max_generations, name="run", **kwargs):
    reports = []
    kwargs.setdefault("champion_dir", None)
    summary = evolution.run(ga.GA(pop_size=20), 0, tmp_path / name, max_generations=max_generations,
                            on_generation=reports.append, **kwargs)
    return summary, reports


class Interrupt(Exception):
    pass


def interrupt_at(generation):
    def callback(report):
        if report["generation"] == generation:
            raise Interrupt
    return callback


def comparable(log_path):
    return pd.read_csv(log_path).drop(columns=["elapsed"])


def test_small_run_logs_every_generation(evaluation, tmp_path):
    evaluation(0.0)
    summary, reports = train(tmp_path, 3)
    assert summary["generations"] == len(reports) == 3 and summary["evals"] == 60 and summary["pop_size"] == 20
    assert [r["generation"] for r in reports] == [0, 1, 2] and math.isnan(reports[0]["step_size"])
    log = pd.read_csv(tmp_path / "run" / "log.csv")
    assert list(log.columns) == evolution.LOG_FIELDS and list(log["generation"]) == [0, 1, 2]
    maps = [s for r in reports for s in r["seeds"]]
    assert len(maps) == len(set(maps)) == 9 and min(maps) >= config.TRAIN_SEED_START   # fresh maps, never reused
    assert (tmp_path / "run" / checkpoint.CHECKPOINT_NAME).exists()
    saved = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert saved["settings"] == checkpoint.settings_record() and saved["goal_reached"] is False
    assert saved["champion_path"] is None and saved["level_reached"] == 1


def test_survivors_come_first_in_the_next_generation(evaluation, tmp_path):
    evaluation(0.5)
    _, reports = train(tmp_path, 2)
    kept = reports[1]["is_kept"]
    assert kept.sum() == reports[0]["n_kept"] and np.all(kept[:kept.sum()]) and not kept[kept.sum():].any()
    assert math.isnan(reports[0]["kept_success"]) and reports[0]["child_success"] == 0.5     # generation 0: all children
    assert reports[1]["kept_success"] == reports[1]["all_success"][kept].mean()    # children may outrank kept cars
    assert reports[1]["child_success"] == reports[1]["all_success"][~kept].mean()


def test_levels_advance_after_three_passing_generations(evaluation, tmp_path):
    evaluation(1.0)
    summary, reports = train(tmp_path, 7)
    assert [r["level"] for r in reports] == [1, 1, 1, 2, 2, 2, 3]
    assert [r["event"] for r in reports[:4]] == [None, None, "level_up", None]
    assert [(e["level"], e["generation"]) for e in summary["levels"]] == [(1, 2), (2, 5)]


def test_failing_generations_never_advance(evaluation, tmp_path):
    evaluation(0.85)                                         # below the 90% requirement on every map
    _, reports = train(tmp_path, 5)
    assert [r["level"] for r in reports] == [1] * 5
    assert all(r["min_share"] < config.MASTERY_FRACTION for r in reports)


def test_goal_then_the_speed_push_tightens_the_target(evaluation, tmp_path):
    evaluation(1.0)
    summary, reports = train(tmp_path, 6, start_level=config.MAX_LEVEL)
    assert reports[2]["event"] == "goal" and reports[5]["event"] == "tightened"
    assert math.isclose(reports[3]["factor"], config.TARGET_FACTOR * config.SPEED_STEP)
    assert summary["goal_reached"] and math.isclose(summary["target_reduction"], 1 - config.SPEED_STEP)
    assert summary["start_level"] == config.MAX_LEVEL and len(summary["tightenings"]) == 1
    assert [(e["level"], e["generation"]) for e in summary["levels"]] == [(config.MAX_LEVEL, 2)]   # the goal entry


class StallsAtOnce(curriculum.MasteryTracker):
    def update(self, min_share):
        super().update(min_share)
        self.outcome = "stalled"
        return "end_stalled"


def test_a_stalled_run_still_saves_a_final_controller(evaluation, monkeypatch, tmp_path):
    evaluation(0.0)
    monkeypatch.setattr(evolution, "MasteryTracker", StallsAtOnce)
    chosen = []

    def select(population, candidates, factor, n_jobs=1):
        chosen.append(factor)
        return int(candidates[0]), 3, 0.7

    monkeypatch.setattr(population, "select_final", select)
    summary, reports = train(tmp_path, None, champion_dir=tmp_path / "champions")
    assert len(reports) == 1 and reports[0]["event"] == "end_stalled"
    assert summary["goal_reached"] is False and summary["outcome"] == "stalled" and summary["level_reached"] == 1
    assert chosen == [config.TARGET_FACTOR] and summary["val_successes"] == 3
    assert summary["champion_path"] == str(paths.champion_path("ga", 0, champion_dir=tmp_path / "champions"))
    assert paths.champion_path("ga", 0, ".npy", tmp_path / "champions").exists()
    assert "GA seed 0: stopped at generation 0" in config.RUN_LOG_PATH.read_text()


def test_the_final_controller_comes_from_the_kept_cars_at_the_tightest_mastered_target(evaluation, monkeypatch,
                                                                                         tmp_path):
    evaluation(1.0)
    chosen = []

    def select(population, candidates, factor, n_jobs=1):
        chosen.append((np.asarray(candidates), factor, population))
        return int(candidates[-1]), 20, 1.9

    monkeypatch.setattr(population, "select_final", select)
    method = ga.GA(pop_size=20)
    summary = evolution.run(method, 4, tmp_path / "run", champion_dir=tmp_path / "champions",
                            start_level=config.MAX_LEVEL, max_generations=6)
    candidates, factor, cars = chosen[0]                  # not `population`: that would shadow the module above
    assert np.array_equal(candidates, np.flatnonzero(method.is_kept)) and cars is method.population
    assert math.isclose(factor, config.TARGET_FACTOR * config.SPEED_STEP)
    saved = np.load(paths.champion_path("ga", 4, ".npy", tmp_path / "champions"))
    assert np.array_equal(saved, method.population.weights[candidates[-1]])
    assert summary["val_successes"] == 20 and summary["wall_time"] > 0


def test_a_resumed_run_matches_an_uninterrupted_one(evaluation, tmp_path):
    evaluation(0.5)
    evolution.run(ga.GA(pop_size=20), 0, tmp_path / "whole", champion_dir=None, max_generations=5)
    with pytest.raises(Interrupt):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "split", champion_dir=None, max_generations=5,
                      on_generation=interrupt_at(3))
    # generation 3 was logged, but the run stopped before its checkpoint (Review Focus 2)
    assert list(pd.read_csv(tmp_path / "split" / "log.csv")["generation"]) == [0, 1, 2, 3]
    summary = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "split", champion_dir=None, max_generations=5)
    whole, split = comparable(tmp_path / "whole" / "log.csv"), comparable(tmp_path / "split" / "log.csv")
    assert list(split["generation"]) == [0, 1, 2, 3, 4]
    pd.testing.assert_frame_equal(whole, split)
    ends = [checkpoint.load(tmp_path / name / checkpoint.CHECKPOINT_NAME) for name in ("whole", "split")]
    assert np.array_equal(ends[0]["method"]["weights"], ends[1]["method"]["weights"])
    assert ends[0]["rng"] == ends[1]["rng"] and summary["generations"] == 5


def test_resuming_with_changed_settings_is_refused(evaluation, monkeypatch, tmp_path):
    evaluation(0.5)
    with pytest.raises(Interrupt):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=5,
                      on_generation=interrupt_at(1))
    monkeypatch.setattr(config, "GA_MUTATION_P", 0.3)
    with pytest.raises(checkpoint.SettingsMismatch, match="GA_MUTATION_P"):
        evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=5)


def test_a_completed_run_is_returned_without_training(evaluation, tmp_path):
    evaluation(0.5)
    first = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=2)
    again = evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=2,
                          on_generation=lambda report: pytest.fail("trained again"))
    assert again == json.loads(json.dumps(first))
    assert len(pd.read_csv(tmp_path / "run" / "log.csv")) == 2


def test_experiment_runs_report_status_and_milestones(evaluation, monkeypatch, tmp_path):
    evaluation(1.0)
    monkeypatch.setattr(population, "select_final",
                        lambda population, candidates, factor, n_jobs=1: (int(candidates[0]), 1, 1.0))
    evolution.run(ga.GA(pop_size=20), 2, tmp_path / "run", champion_dir=tmp_path / "champions", max_generations=4)
    status = progress.read_status()
    assert status["approach"] == "ga" and status["seed"] == 2 and status["stage"] == "evolution"
    assert status["level"] == 2 and status["generation"] == 3 and status["min_share"] == 1.0
    log = config.RUN_LOG_PATH.read_text()
    assert "GA seed 2: started" in log and "Easy mastered at generation 2, now Medium" in log
    assert "GA seed 2: finished" in log


def test_viewer_and_pilot_runs_report_nothing(evaluation, tmp_path):
    evaluation(1.0)
    train(tmp_path, 4)
    assert not config.STATUS_PATH.exists() and not config.RUN_LOG_PATH.exists()


def test_each_run_starts_with_fresh_workers_before_its_timer(evaluation, monkeypatch, tmp_path):
    evaluation(0.0)
    events = []

    class Clock:
        @staticmethod
        def time():
            events.append("clock")
            return 0.0

    monkeypatch.setattr(population, "fresh_workers", lambda n_jobs: events.append(("fresh", n_jobs)))
    monkeypatch.setattr(evolution, "time", Clock)
    evolution.run(ga.GA(pop_size=20), 0, tmp_path / "run", champion_dir=None, n_jobs=3, max_generations=1)
    assert events[:2] == [("fresh", 3), "clock"] and events.count(("fresh", 3)) == 1


def test_truncate_log_keeps_the_rows_before_a_generation(tmp_path):
    path = tmp_path / "log.csv"
    evolution.start_log(path)
    for g in range(4):
        evolution.append_log(path, {key: g if key == "generation" else ([1] if key in ("seeds", "map_shares")
                                                                      else 0) for key in evolution.LOG_FIELDS})
    evolution.truncate_log(path, 2)
    assert list(pd.read_csv(path)["generation"]) == [0, 1]


class WithSpecies(ga.GA):
    """A GA that reports one extra log column, as NEAT reports its species."""

    def log_fields(self):
        return {"species": 3}


def test_a_methods_extra_fields_are_log_columns_also_after_resuming(evaluation, tmp_path):
    evaluation(0.5)
    with pytest.raises(Interrupt):
        evolution.run(WithSpecies(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=3,
                      on_generation=interrupt_at(1))
    reports = []
    evolution.run(WithSpecies(pop_size=20), 0, tmp_path / "run", champion_dir=None, max_generations=3,
                  on_generation=reports.append)
    log = pd.read_csv(tmp_path / "run" / "log.csv")
    assert list(log.columns) == evolution.LOG_FIELDS + ["species"]
    assert list(log["generation"]) == [0, 1, 2] and list(log["species"]) == [3, 3, 3]
    assert [r["species"] for r in reports] == [3, 3]


@pytest.mark.slow
def test_results_do_not_depend_on_the_number_of_workers(tmp_path):
    for jobs in (1, 2):
        evolution.run(ga.GA(pop_size=12), 0, tmp_path / f"jobs{jobs}", champion_dir=None, n_jobs=jobs,
                      max_generations=2)
    pd.testing.assert_frame_equal(comparable(tmp_path / "jobs1" / "log.csv"),
                                  comparable(tmp_path / "jobs2" / "log.csv"))
```

- [ ] **Step 3: Run the engine tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_evolution.py -q`
Expected: FAIL — `ImportError: cannot import name 'evolution' from 'racing'`.

- [ ] **Step 4: Rewrite the GA and ES tests**

Replace `tests/test_ga.py` with:

```python
import math

import numpy as np
import pytest

from racing import config, ga, paths, population
from racing.curriculum import mutation_strength
from racing.population import TrialResult


def result(fitness, success):
    fitness = np.asarray(fitness, dtype=float)
    success = np.tile(np.asarray(success, dtype=bool), (config.TRACKS_PER_GEN, 1))
    return TrialResult(fitness=fitness, success=success, laps=np.where(success, config.LAPS, 0),
                       finish_time=np.where(success, 90.0, np.nan))


def no_success(population, seeds, level, factor, n_jobs=1, backend="loky"):
    n = len(population)
    return result(np.arange(n, dtype=float), np.zeros(n))


def started(pop_size=30, **kwargs):
    method = ga.GA(pop_size=pop_size, **kwargs)
    method.initialize(np.random.default_rng(0))
    return method


def test_tournament_prefers_fitter_individuals():
    winners = ga.tournament(np.random.default_rng(0), np.arange(100.0), 1000)
    assert winners.mean() > 75


def test_initial_population_has_no_kept_cars_and_no_strength():
    method = started()
    assert method.population.weights.shape == (30, 431) and not method.is_kept.any()
    assert math.isnan(method.step_size) and method.approach == "ga" and method.log_fields() == {}


def test_advance_keeps_the_survivors_first_and_unchanged():
    method = started()
    before = method.population.weights.copy()
    kept = np.array([5, 2, 9])
    method.advance(np.random.default_rng(1), result(np.arange(30.0), np.zeros(30)), kept, None)
    assert method.population.weights.shape == before.shape
    assert np.array_equal(method.population.weights[:3], before[kept])
    assert list(np.flatnonzero(method.is_kept)) == [0, 1, 2]


def test_zero_strength_children_only_recombine_parent_genes():
    method = started(pop_size=20, hidden=(2,), s_max=0.0)
    before = method.population.weights.copy()
    fitness = np.random.default_rng(3).random(20)
    method.advance(np.random.default_rng(2), result(fitness, np.zeros(20)), np.array([0, 1]), None)
    children = method.population.weights[2:]
    for j in range(before.shape[1]):
        assert set(children[:, j]).issubset(set(before[:, j]))


def test_strength_follows_the_previous_childrens_success():
    method = started(pop_size=20, s_max=0.2)                 # generation 0: every car is a child
    method.advance(np.random.default_rng(4), result(np.arange(20.0), np.arange(20) % 2 == 0), np.arange(10), None)
    assert math.isclose(method.step_size, mutation_strength(0.5, 0.2, 0.002))
    only_kept = np.arange(20) < 10                           # the kept cars succeed, no child does
    method.advance(np.random.default_rng(5), result(np.arange(20.0), only_kept), np.arange(10), None)
    assert math.isclose(method.step_size, 0.2)


def test_operators_solve_the_sphere_function():
    method = started(pop_size=100, hidden=(1,), s_max=0.05)
    rng = np.random.default_rng(3)
    start = np.min(np.sum(method.population.weights ** 2, axis=1))
    for _ in range(80):
        fitness = -np.sum(method.population.weights ** 2, axis=1)
        method.advance(rng, result(fitness, np.zeros(100)), np.argsort(-fitness)[:5], None)
    assert np.min(np.sum(method.population.weights ** 2, axis=1)) < start / 10


def test_state_round_trip_continues_identically():
    first = started(pop_size=20)
    second = ga.GA(pop_size=20)
    second.load_state(first.state())
    trial = result(np.arange(20.0), np.arange(20) < 4)
    first.advance(np.random.default_rng(7), trial, np.arange(6), None)
    second.advance(np.random.default_rng(7), trial, np.arange(6), None)
    assert np.array_equal(first.population.weights, second.population.weights)
    assert first.step_size == second.step_size and np.array_equal(first.is_kept, second.is_kept)


def test_population_must_hold_two_cars():
    with pytest.raises(ValueError, match="at least 2"):
        ga.GA(pop_size=1)


def test_run_ga_uses_the_seed_folder_and_saves_the_champion(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(population, "evaluate_population", no_success)
    monkeypatch.setattr(population, "select_final",
                        lambda population, candidates, factor, n_jobs=1: (int(candidates[0]), 0, 0.5))
    summary = ga.run_ga(seed=3, pop_size=20, max_generations=2, champion_dir=tmp_path / "champions")
    assert (tmp_path / "runs" / "ga" / "seed3" / "log.csv").exists() and summary["approach"] == "ga"
    assert summary["champion_path"] == str(paths.champion_path("ga", 3, champion_dir=tmp_path / "champions"))
    assert np.load(paths.champion_path("ga", 3, ".npy", tmp_path / "champions")).shape == (431,)


def test_run_ga_returns_a_completed_run(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "evaluate_population", no_success)
    first = ga.run_ga(pop_size=20, max_generations=1, out_dir=tmp_path, champion_dir=None)
    again = ga.run_ga(pop_size=20, max_generations=1, out_dir=tmp_path, champion_dir=None,
                      on_generation=lambda report: pytest.fail("trained again"))
    assert again["generations"] == first["generations"] == 1
```

Replace `tests/test_es.py` with:

```python
import math

import numpy as np
import pytest

from racing import config, es, nets, paths, population
from racing.population import TrialResult


def result(fitness, success):
    fitness = np.asarray(fitness, dtype=float)
    success = np.tile(np.asarray(success, dtype=bool), (config.TRACKS_PER_GEN, 1))
    return TrialResult(fitness=fitness, success=success, laps=np.where(success, config.LAPS, 0),
                       finish_time=np.where(success, 90.0, np.nan))


def no_success(population, seeds, level, factor, n_jobs=1, backend="loky"):
    n = len(population)
    return result(np.arange(n, dtype=float), np.zeros(n))


def first_candidate(population, candidates, factor, n_jobs=1):
    return int(candidates[0]), 0, 0.5


def started(pop_size=20, **kwargs):
    method = es.ES(pop_size=pop_size, **kwargs)
    method.initialize(np.random.default_rng(0))
    return method


def test_initial_population_and_step_size():
    method = started(parents=4)
    assert method.population.weights.shape == (20, 431) and not method.is_kept.any()
    assert method.step_size == config.ES_SIGMA0 and method.approach == "es" and method.log_fields() == {}


def test_seeded_population_starts_from_the_trained_network():
    flat = np.random.default_rng(1).normal(size=431)
    method = started(pop_size=50, initial=flat, approach="es_seeded")
    weights = method.population.weights
    assert np.array_equal(weights[0], flat) and not np.any(np.all(weights[1:] == flat, axis=1))
    assert 0.05 < np.std(weights[1:] - flat) < 0.2 and method.approach == "es_seeded"


def test_children_are_noisy_copies_of_the_fittest_cars():
    method = started(parents=4, sigma0=config.ES_SIGMA_MIN)          # σ is never below ES_SIGMA_MIN
    before = method.population.weights.copy()
    method.advance(np.random.default_rng(2), result(-np.arange(20.0), np.zeros(20)), np.arange(10), None)
    for child in method.population.weights[10:]:                  # cars 0-3 were the fittest
        assert np.any(np.all(np.isclose(child, before[:4], atol=10 * config.ES_SIGMA_MIN), axis=1))


def test_children_remember_their_parents():
    method = started(parents=4)
    ids = method.ids.copy()
    method.advance(np.random.default_rng(8), result(-np.arange(20.0), np.zeros(20)), np.arange(10), None)
    assert np.array_equal(method.ids[:10], ids[:10]) and set(method.ids[10:]).isdisjoint(ids)
    assert set(method.parent_ids[10:]) <= set(ids[:4])


def test_learning_phase_follows_the_one_fifth_rule():
    method = started(parents=4)
    method.advance(np.random.default_rng(3), result(-np.arange(20.0), np.zeros(20)), np.arange(10), None)
    assert method.step_size == config.ES_SIGMA0                    # no child has a parent yet
    fitness = np.zeros(20)
    fitness[10:] = 1.0                                             # every child beats its own parent
    method.advance(np.random.default_rng(4), result(fitness, np.zeros(20)), np.arange(10), None)
    assert math.isclose(method.step_size, config.ES_SIGMA0 / config.ES_FACTOR)


def test_reliability_phase_follows_the_childrens_success():
    method = started(parents=4)
    success = np.arange(20) < 15                                    # every parent, 75% of the children
    method.advance(np.random.default_rng(5), result(-np.arange(20.0), success), np.arange(10), None)
    assert math.isclose(method.step_size, config.ES_SIGMA0 * config.ES_FACTOR)
    method.advance(np.random.default_rng(6), result(-np.arange(20.0), np.ones(20)), np.arange(10), None)
    assert math.isclose(method.step_size, config.ES_SIGMA0)


def test_step_size_resets_on_a_level_up():
    method = started(parents=4)
    method.advance(np.random.default_rng(7), result(-np.arange(20.0), np.ones(20)), np.arange(10), "level_up")
    assert method.step_size == config.ES_SIGMA0


def test_state_round_trip_continues_identically():
    first = started(parents=4)
    second = es.ES(pop_size=20, parents=4)
    second.load_state(first.state())
    trial = result(-np.arange(20.0), np.arange(20) < 3)
    first.advance(np.random.default_rng(9), trial, np.arange(10), None)
    second.advance(np.random.default_rng(9), trial, np.arange(10), None)
    assert np.array_equal(first.population.weights, second.population.weights)
    assert np.array_equal(first.parent_ids, second.parent_ids) and first.step_size == second.step_size


def test_run_es_seeded_starts_from_the_initial_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "evaluate_population", no_success)
    flat = nets.init_population(np.random.default_rng(1), 1)[0]
    reports = []
    summary = es.run_es(pop_size=20, max_generations=1, initial=flat, approach="es_seeded", out_dir=tmp_path,
                        on_generation=reports.append, champion_dir=None)
    assert np.array_equal(reports[0]["population"].weights[0], flat) and reports[0]["approach"] == "es_seeded"
    assert summary["approach"] == "es_seeded"


def test_run_es_returns_a_completed_run(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "evaluate_population", no_success)
    es.run_es(pop_size=20, max_generations=1, out_dir=tmp_path, champion_dir=None)
    again = es.run_es(pop_size=20, max_generations=1, out_dir=tmp_path, champion_dir=None,
                      on_generation=lambda report: pytest.fail("trained again"))
    assert again["generations"] == 1


def test_run_es_any_architecture_round_trips_its_champion(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "evaluate_population", no_success)
    monkeypatch.setattr(population, "select_final", first_candidate)
    flat = nets.init_population(np.random.default_rng(1), 1, (8, 4))[0]
    summary = es.run_es(seed=1, pop_size=20, max_generations=1, initial=flat, approach="kt_random", hidden=(8, 4),
                        activation="tanh", out_dir=tmp_path / "run", champion_dir=tmp_path / "champions")
    model, driver = nets.load_champion("kt_random", 1, tmp_path / "champions")
    saved = np.load(paths.champion_path("kt_random", 1, ".npy", tmp_path / "champions"))
    obs = np.random.default_rng(2).random((5, config.N_INPUTS))
    assert np.array_equal(driver.act(obs), nets.from_flat(saved, (8, 4), "tanh").act(obs))
    assert nets.dense_spec(model) == ((8, 4), "tanh") and summary["approach"] == "kt_random"


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


def test_dagger_baseline_reads_the_refined_baseline(tmp_path):
    from racing import imitation
    model = imitation.build_baseline()
    path = paths.run_dir("baseline", 1, tmp_path) / "dagger.keras"
    path.parent.mkdir(parents=True)
    model.save(path)
    flat = es.dagger_baseline(1, runs_dir=tmp_path)
    assert flat.shape == (431,) and np.array_equal(flat, nets.to_flat(model))


def test_dagger_baseline_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="notebook 03"):
        es.dagger_baseline(0, runs_dir=tmp_path)
```

- [ ] **Step 5: Update the tests of the code this task replaces**

`tests/test_population.py`: delete the helper `report()` and the tests `test_logger_records_pace_and_es_fractions`, `test_logger_removes_a_stale_summary`, `test_select_champion_picks_highest_validation_fitness`, `test_save_champion_writes_keras_and_float64_weights`, `test_clear_champion_removes_both_files`, `test_finish_fractions`, `test_make_report_fields`, `test_logger_stays_readable_after_each_generation` and `test_logger_removes_stale_best_networks`. Also delete the revision 2.0 evaluator tests that Task 2 renamed to `evaluate_weights` (Task 2's time-trial tests replace them): `test_evaluate_population_shapes`, `test_parallel_evaluation_matches_serial`, `test_resimulating_a_subset_reproduces_logged_fitness` and `test_at_pace_requires_finishing_within_the_pace_factor`. In `test_fresh_workers_start_new_processes_and_clear_the_track_cache`, replace `weights = nets.init_population(np.random.default_rng(5), 4)` with `cars = nets.DensePopulation(nets.init_population(np.random.default_rng(5), 4))` and each of its three `population.evaluate_weights(weights, [0], 1, n_jobs=2)` with `population.evaluate_population(cars, [0], 1, config.TARGET_FACTOR, n_jobs=2)`. Remove imports that become unused (`json`, `math`, `pandas`). Append (skip a test whose name already exists from Task 2):

```python
def test_select_final_prefers_successes_then_fitness(monkeypatch):
    cars = nets.DensePopulation(nets.init_population(np.random.default_rng(0), 6))
    seen = []

    def fake(subset, seeds, level, factor, n_jobs=1, backend="loky"):     # `population` would shadow the module
        seen.append((len(subset), level, factor, list(seeds)))
        success = np.array([[True, True, False, True], [True, False, False, True]])
        return population.TrialResult(fitness=np.array([1.2, 1.9, 0.4, 1.5]), success=success,
                                      laps=np.where(success, config.LAPS, 1),
                                      finish_time=np.where(success, 90.0, np.nan))

    monkeypatch.setattr(population, "evaluate_population", fake)
    assert population.select_final(cars, np.array([5, 1, 3, 0]), 1.2, seeds=[80, 81]) == (0, 2, 1.5)
    assert seen == [(4, config.MAX_LEVEL, 1.2, [80, 81])]


def test_save_dense_champion_keeps_the_architecture(tmp_path):
    weights = nets.init_population(np.random.default_rng(3), 3, (8, 4))
    path = population.save_dense_champion("es", 2, nets.DensePopulation(weights, (8, 4), "tanh"), 1, tmp_path)
    assert path == tmp_path / "es" / "seed2.keras" and path.exists()
    assert not (tmp_path / "es" / "seed2.json").exists()                # a .json marks a NEAT champion
    saved = np.load(tmp_path / "es" / "seed2.npy")
    assert saved.dtype == np.float64 and np.array_equal(saved, weights[1])
    model, driver = nets.load_champion("es", 2, tmp_path)
    assert nets.dense_spec(model) == ((8, 4), "tanh") and driver.n_params == weights.shape[1]


def test_clear_champion_removes_every_file(tmp_path):
    cars = nets.DensePopulation(nets.init_population(np.random.default_rng(4), 1))
    population.save_dense_champion("es", 0, cars, 0, tmp_path)
    (tmp_path / "es" / "seed0.json").write_text("{}")
    population.clear_champion("es", 0, tmp_path)
    assert list((tmp_path / "es").iterdir()) == []
```

`tests/test_curriculum.py`: delete the revision 2.0 tests `test_level_up_on_third_consecutive_qualifying_generation`,
`test_streak_resets_below_the_threshold`, `test_below_threshold_never_qualifies`, `test_level_stops_at_max` and
`test_draw_seeds_are_distinct_training_seeds`, and replace the import of `racing.curriculum` with

```python
from racing.curriculum import (EVENTS, MasteryTracker, SeedSource, StepSize, mutation_strength, success_fraction,
                               survivors)
```

`tests/test_nets.py`: in `test_load_champion_drives_with_float64_weights`, replace
`population.save_champion("es", 0, flat, tmp_path)` with
`population.save_dense_champion("es", 0, nets.DensePopulation(flat[None, :]), 0, tmp_path)`.

`tests/test_evaluate.py`: in `test_main_with_partial_results`, replace
`population.save_champion(approach, seed, nets.init_population(rng, 1)[0], champions)` with
`population.save_dense_champion(approach, seed, nets.DensePopulation(nets.init_population(rng, 1)), 0, champions)`, and append:

```python
def test_figures_read_revision_3_logs(tmp_path):
    folder = tmp_path / "runs" / "ga" / "seed0"
    folder.mkdir(parents=True)
    pd.DataFrame({"generation": [0, 1, 2], "evals": [20, 40, 60], "best": [0.5, 1.2, 1.4],
                  "step_size": [np.nan, 0.1, 0.09], "event": ["", "level_up", ""]}).to_csv(folder / "log.csv",
                                                                                          index=False)
    written = evaluate.figures(tmp_path / "runs", tmp_path / "figures")
    assert tmp_path / "figures" / "fitness_curves.png" in written
```

`tests/test_app.py`:
- in `save_race_champions`, replace `population.save_champion(approach, seed, nets.init_population(rng, 1)[0], directory)` with `population.save_dense_champion(approach, seed, nets.DensePopulation(nets.init_population(rng, 1)), 0, directory)`;
- in `test_trainer_main_uses_app_runs_dir_and_never_saves_champions`, replace `calls["es"]["lam"] == 12` with `calls["es"]["pop_size"] == 12`;
- append:

```python
def test_a_report_with_a_population_plays_back(window):
    report = ga_report()
    report["population"] = nets.DensePopulation(report.pop("weights"))
    window.receive(report)
    window.update(1 / 60)
    assert window.playback is not None and window.playback.cars.laps == config.LAPS


def test_trainer_main_starts_each_viewer_run_afresh(monkeypatch, tmp_path):
    from racing import ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    stale = tmp_path / "ga" / "seed0" / "checkpoint.pkl"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"an earlier viewer run")
    seen = []
    monkeypatch.setattr(ga, "run_ga", lambda **kwargs: seen.append(stale.exists()))
    running = threading.Event()
    running.set()
    viewer.trainer_main("ga", 30, 0, 1, queue.Queue(), running)
    assert seen == [False]
```

`tests/test_pilots.py`: replace the first two tests and the call in the third with:

```python
def test_ga_pilot_summarizes_a_short_run(pilots, tmp_path):
    row = pilots.pilot_ga(0.2, tmp_path, generations=2, pop_size=20, n_jobs=1)
    assert row["pilot"] == "GA s_max=0.2" and row["level_reached"] >= 1 and row["generations"] == 2
    assert 0.0 <= row["min_share_last20"] <= 1.0 and (tmp_path / "ga_smax0.2" / "log.csv").exists()


def test_a_pilot_always_starts_afresh(pilots, tmp_path):
    pilots.pilot_ga(0.2, tmp_path, generations=1, pop_size=20, n_jobs=1)
    assert pilots.pilot_ga(0.2, tmp_path, generations=2, pop_size=20, n_jobs=1)["generations"] == 2


def test_es_pilot_reports_the_step_size(pilots, tmp_path):
    row = pilots.pilot_es(tmp_path, generations=2, pop_size=20, n_jobs=1)
    assert row["pilot"] == "ES" and row["sigma_last"] > 0 and 0.0 <= row["min_share_last20"] <= 1.0
```

and in `test_seeded_pilot_trains_its_own_baseline` replace `pilots.pilot_seeded(tmp_path, generations=1, mu=4, lam=16, n_jobs=1)` with `pilots.pilot_seeded(tmp_path, generations=1, pop_size=20, n_jobs=1)`.

`tests/test_end_to_end.py` (slow): replace the three evolutionary calls and the final assertion:

```python
    ga.run_ga(pop_size=20, max_generations=2, seed=0, out_dir=paths.run_dir("ga", 0, runs), champion_dir=champions)
    es.run_es(pop_size=20, max_generations=2, seed=0, out_dir=paths.run_dir("es", 0, runs), champion_dir=champions)
    es.run_es(pop_size=20, max_generations=2, seed=1, initial=es.trained_baseline(1, champion_dir=champions),
              approach="es_seeded", out_dir=paths.run_dir("es_seeded", 1, runs), champion_dir=champions)
```
```python
    assert json.loads((paths.run_dir("ga", 0, runs) / "summary.json").read_text())["evals"] == 40
```
(the last line is unchanged; two generations of 20 cars still make 40 evaluations).

- [ ] **Step 6: Run the updated tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_evolution.py tests/test_ga.py tests/test_es.py tests/test_population.py tests/test_curriculum.py tests/test_nets.py tests/test_app.py tests/test_pilots.py tests/test_evaluate.py -q`
Expected: FAIL (`racing.evolution` missing; `ga.GA`, `es.ES`, `population.save_dense_champion` or the new `run_*` keywords missing).

- [ ] **Step 7: Configuration**

In `racing/config.py`, replace the GA, ES and budget blocks:

```python
# --- Genetic algorithm (spec §9.2) ---
GA_POP = 1000
GA_KEPT = 50                 # the minimum number of survivors
GA_KEEP_MAX_FRACTION = 0.5   # survivors are at most this share of the population
GA_TOURNAMENT = 5
```
→
```python
# --- Genetic algorithm (revision 3 §5.1) ---
GA_POP = 1000
GA_TOURNAMENT = 5
```

```python
# --- Evolution strategy (spec §9.3) ---
ES_MU = 50
ES_LAMBDA = 500
```
→
```python
# --- Evolution strategy (revision 3 §5.2) ---
ES_POP = 550
ES_MU = 50                   # children are copies of one of the ES_MU fittest cars
```

```python
# --- Budget and parallelism ---
EVAL_BUDGET = 200_000
N_JOBS = 6
```
→
```python
# --- Parallelism ---
N_JOBS = 6
```
(keep the existing comment on the `N_JOBS` line; Task 15 changes the line itself).

In the block above the GA block, delete the three revision 2.0 lines `LEVEL_UP_FRACTION = 0.5 ...`,
`LEVEL_UP_STREAK = 3` and `PACE_FACTOR = 1.3 ...`, and rename its heading `# --- Curriculum (spec §9.1) ---` to
`# --- Time trial, survival and levels (revision 3 §4) ---` (it keeps `TRACKS_PER_GEN` and the values Tasks 2–3 added).

In `racing/checkpoint.py`, add `"ES_POP"` to `SETTINGS_KEYS` before `"ES_MU"`.

In `racing/curriculum.py`, delete the section `# ----- revision 2.0 rules, used by ga.py and es.py ...` to the end of the file
(`LevelTracker` and `draw_seeds`; the rewritten ga.py and es.py no longer use them).

- [ ] **Step 8: Population: final selection and dense champions**

In `racing/population.py`, delete the whole section `# ----- revision 2.0 helpers, used by ga.py and es.py ...` (`_evaluate_track`, `evaluate_weights`, `finish_fractions`, `make_report`, `RunLogger`, `select_champion` and `save_champion`), and remove imports that are then unused (`csv`, `json`, `Path`: check with `grep -n "csv\.\|json\.\|Path(" racing/population.py`). End the module docstring after "final selection and saved controllers." (drop the sentence about the revision 2.0 logging helpers).

If Step 1 found them missing, add:

```python
def select_final(population, candidates, factor, n_jobs=1, seeds=config.VAL_SEEDS):
    """The final controller among `candidates` (indices into `population`): the most successes on the validation
    maps as time trials at target factor `factor`, then the highest mean fitness, then the earliest candidate
    (revision 3 §4.6). Returns (index into population, validation successes, mean validation fitness)."""
    candidates = np.asarray(candidates, dtype=np.int64)
    result = evaluate_population(population.subset(candidates), list(seeds), config.MAX_LEVEL, factor, n_jobs=n_jobs)
    successes = result.success.sum(axis=0)
    best = int(np.lexsort((-result.fitness, -successes))[0])      # stable: ties keep the candidates' order
    return int(candidates[best]), int(successes[best]), float(result.fitness[best])


def save_dense_champion(approach, seed, population, index, champion_dir=config.CHAMPIONS_DIR):
    """Final controller `index` of a dense population as a Keras file and as float64 weights (spec §6). The Keras
    file records the architecture (nets.dense_spec). Returns the Keras file's path."""
    flat = np.asarray(population.weights[index], dtype=float)
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    nets.to_keras(flat, population.hidden, population.activation).save(path)
    np.save(paths.champion_path(approach, seed, ".npy", champion_dir), flat)
    return path
```

Replace `clear_champion` with:

```python
def clear_champion(approach, seed, champion_dir=config.CHAMPIONS_DIR):
    """Remove a final controller left by an earlier run, so an interrupted run leaves none behind."""
    paths.remove(*(paths.champion_path(approach, seed, suffix, champion_dir) for suffix in (".keras", ".npy", ".json")))
```

- [ ] **Step 9: Dense champions drive in their own architecture**

If Step 1 found `dense_spec` missing, add to `racing/nets.py` after `to_flat`:

```python
def dense_spec(model):
    """(hidden-layer sizes, hidden activation) of a dense Keras model; dropout layers do not count."""
    keras = keras_cpu()
    dense = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    hidden = tuple(int(layer.units) for layer in dense[:-1])
    return hidden, (dense[0].get_config()["activation"] if len(dense) > 1 else "relu")
```

In `load_champion`, replace the dense branch

```python
    flat_path = paths.champion_path(approach, seed, ".npy", directory)
    driver = from_flat(np.load(flat_path)) if flat_path.exists() else from_keras(model)
    return model, driver
```

with

```python
    flat_path = paths.champion_path(approach, seed, ".npy", directory)
    if flat_path.exists():                     # an evolved controller: its float64 weights, its own architecture
        return model, from_flat(np.load(flat_path), *dense_spec(model))
    return model, from_keras(model)
```

(keep any NEAT branch Task 2 placed before it).

- [ ] **Step 10: The engine**

Create `racing/evolution.py`:

```python
"""Shared run loop of every evolutionary method (revision 3 §4, §8.2, §8.3). A method object holds the population
and makes the children (racing/ga.py, racing/es.py, racing/neat.py); this loop drives fresh maps as time trials,
applies the survival rule, follows the mastery levels and the speed push, logs every generation, saves a checkpoint
after each one, reports progress and selects the final controller."""
import csv
import json
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from racing import checkpoint, config, population, progress
from racing.curriculum import MasteryTracker, SeedSource, survivors

LOG_NAME = "log.csv"
SUMMARY_NAME = "summary.json"
LOG_FIELDS = ["generation", "level", "factor", "evals", "best", "mean", "worst", "map_shares", "min_share", "n_kept",
              "kept_success", "child_success", "median_lap", "step_size", "streak", "event", "seeds", "elapsed"]


class Method(Protocol):
    """What the loop needs from a method. `population` holds the cars carried over unchanged first; `is_kept`
    marks them. `advance` must assign a new population object (reports keep the evaluated one)."""
    approach: str
    population: object
    is_kept: np.ndarray
    step_size: float

    def initialize(self, rng) -> None: ...
    def advance(self, rng, result, kept, event) -> None: ...
    def state(self) -> dict: ...
    def load_state(self, state) -> None: ...
    def save_champion(self, index, seed, champion_dir) -> Path: ...
    def log_fields(self) -> dict: ...


def share(mask):
    """Fraction of True values; nan for an empty selection."""
    return float(np.mean(mask)) if len(mask) else float("nan")


def median_lap(result):
    """Median lap time of the successful time trials; nan if none succeeded."""
    times = result.lap_times[result.success]
    return float(np.median(times)) if times.size else float("nan")


def plain(value):
    """A float for JSON: None instead of nan."""
    value = float(value)
    return None if np.isnan(value) else value


def start_log(path, fields=LOG_FIELDS):
    """A new log whose columns are `fields`: LOG_FIELDS, then the method's extra columns (Method.log_fields)."""
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(fields)


def log_header(path):
    """The columns of an existing log."""
    with open(path, newline="") as f:
        return next(csv.reader(f))


def append_log(path, report, fields=LOG_FIELDS):
    row = []
    for key in fields:
        value = report[key]
        if key in ("seeds", "map_shares"):
            value = " ".join(str(v) for v in value)
        elif key == "event":
            value = value or ""
        row.append(value)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def truncate_log(path, generation, fields=LOG_FIELDS):
    """Keep the header and the rows of the generations before `generation`. A row written after the last checkpoint
    belongs to a generation that runs again when the run resumes (revision 3 §8.2). A missing log is started with the
    columns `fields`."""
    path = Path(path)
    if not path.exists():
        start_log(path, fields)
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    kept = (rows[:1] or [list(fields)]) + [row for row in rows[1:] if row and int(row[0]) < generation]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", newline="") as f:
        csv.writer(f).writerows(kept)
    os.replace(temporary, path)


def write_json(path, data):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2))
    os.replace(temporary, path)


def label(method, seed):
    return f"{config.LABELS.get(method.approach, method.approach)} seed {seed}"


def milestone_text(event, method, seed, generation, level, tracker):
    name = label(method, seed)
    if event == "level_up":
        return (f"{name}: {config.DIFFICULTIES[level]} mastered at generation {generation}, "
                f"now {config.DIFFICULTIES[tracker.level]}")
    if event == "goal":
        return f"{name}: mastery goal reached at generation {generation}, speed push starts"
    if event == "tightened":
        return f"{name}: target {100 * tracker.target_reduction:.1f}% shorter mastered at generation {generation}"
    if event == "end_speed":
        return (f"{name}: speed push ended at generation {generation}, "
                f"target {100 * tracker.target_reduction:.1f}% shorter")
    return f"{name}: stopped at generation {generation}, {config.DIFFICULTIES[level]} not mastered"


def run(method, seed, out_dir, champion_dir=config.CHAMPIONS_DIR, n_jobs=1, on_generation=None, start_level=1,
        max_generations=None, rng_seed=None):
    """Train `method` with run seed `seed` until the speed push ends or the safety stop (revision 3 §4.4-4.5), or
    after `max_generations` (viewer, pilots, tests). A completed run in `out_dir` with the current settings is
    returned at once; an interrupted one resumes from its checkpoint. With champion_dir None no final controller is
    selected or saved and nothing is written to the status file or the run log: only the experiment's runs report
    progress. Returns the summary."""
    out_dir = Path(out_dir)
    done = checkpoint.completed(out_dir, SUMMARY_NAME)
    if done is not None:
        return done
    reporting = champion_dir is not None
    population.fresh_workers(n_jobs)                   # every run builds its own tracks (spec §9.1)
    began = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / LOG_NAME
    state = checkpoint.load(out_dir / checkpoint.CHECKPOINT_NAME)
    if state is None:
        rng = np.random.default_rng(seed if rng_seed is None else rng_seed)
        method.initialize(rng)
        seeds = SeedSource(rng)
        tracker = MasteryTracker(level=start_level)
        generation, evals, elapsed, levels, tightenings = 0, 0, 0.0, [], []
        pop_size = len(method.population)
        if champion_dir is not None:
            population.clear_champion(method.approach, seed, champion_dir)
        fields = LOG_FIELDS + list(method.log_fields())   # the method's extra columns, fixed for the whole run
        start_log(log_path, fields)
        if reporting:
            progress.milestone(f"{label(method, seed)}: started")
    else:
        checkpoint.check_settings(state["settings"])
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        method.load_state(state["method"])
        seeds = SeedSource(rng, used=state["seeds"])
        tracker = MasteryTracker.from_state(state["tracker"])
        generation, evals, elapsed = state["generation"], state["evals"], state["elapsed"]
        levels, tightenings, start_level = state["levels"], state["tightenings"], state["start_level"]
        pop_size = len(method.population)
        truncate_log(log_path, generation, LOG_FIELDS + list(method.log_fields()))
        fields = log_header(log_path)                     # a resumed run keeps the columns its log started with
        if reporting:
            progress.milestone(f"{label(method, seed)}: resumed at generation {generation}")

    def active():
        return elapsed + time.time() - began        # run time, accumulated across resumptions

    while not tracker.done and (max_generations is None or generation < max_generations):
        level, factor = tracker.level, tracker.factor
        maps = seeds.draw(config.TRACKS_PER_GEN)
        result = population.evaluate_population(method.population, maps, level, factor, n_jobs=n_jobs)
        n = len(method.population)
        evals += n
        cap = int(config.KEEP_MAX_FRACTION * n)
        kept = survivors(rng, result.fitness, result.all_success, result.lap_share, cap, config.ELITE_FLOOR)
        event = tracker.update(result.min_share)
        now = active()
        report = {"approach": method.approach, "seed": seed, "generation": generation, "level": level,
                  "factor": factor, "target_reduction": tracker.target_reduction, "streak": tracker.streak,
                  "seeds": maps, "population": method.population, "fitness": result.fitness,
                  "all_success": result.all_success, "is_kept": method.is_kept, "map_shares": result.map_shares,
                  "min_share": result.min_share, "n_kept": len(kept),
                  "kept_success": share(result.all_success[method.is_kept]),
                  "child_success": share(result.all_success[~method.is_kept]), "median_lap": median_lap(result),
                  "step_size": float(method.step_size), "best": float(result.fitness.max()),
                  "mean": float(result.fitness.mean()), "worst": float(result.fitness.min()), "evals": evals,
                  "elapsed": now, "event": event, **method.log_fields()}
        append_log(log_path, report, fields)
        if event in ("level_up", "goal"):
            levels.append({"level": int(level), "generation": generation, "evals": evals, "elapsed": now})
        elif event == "tightened":
            tightenings.append({"factor": float(tracker.mastered_factor), "generation": generation, "evals": evals,
                                "elapsed": now})
        if reporting:
            if event is not None:
                progress.milestone(milestone_text(event, method, seed, generation, level, tracker))
            progress.write_status({"approach": method.approach, "seed": seed,
                                   "stage": "speed push" if tracker.goal_reached else "evolution",
                                   "level": int(level), "generation": generation, "min_share": result.min_share,
                                   "streak": tracker.streak, "target_reduction": tracker.target_reduction,
                                   "elapsed": now})
        if on_generation is not None:
            on_generation(report)
        method.advance(rng, result, kept, event)
        generation += 1
        checkpoint.save(out_dir / checkpoint.CHECKPOINT_NAME, {
            "settings": checkpoint.settings_record(), "generation": generation, "evals": evals,
            "elapsed": active(), "rng": rng.bit_generator.state, "seeds": seeds.state(),
            "tracker": tracker.state(), "method": method.state(), "levels": levels, "tightenings": tightenings,
            "start_level": start_level})

    summary = {"approach": method.approach, "seed": seed, "settings": checkpoint.settings_record(),
               "generations": generation, "evals": evals, "pop_size": pop_size,
               "start_level": start_level, "level_reached": int(tracker.level),
               "goal_reached": bool(tracker.goal_reached), "outcome": tracker.outcome,
               "mastered_factor": plain(tracker.mastered_factor), "target_reduction": plain(tracker.target_reduction),
               "levels": levels, "tightenings": tightenings, "val_successes": None, "val_fitness": None,
               "champion_path": None}
    if champion_dir is not None:
        candidates = np.flatnonzero(method.is_kept)                 # the kept cars of the last generation
        if candidates.size == 0:
            candidates = np.arange(len(method.population))
        factor = tracker.mastered_factor if tracker.goal_reached else config.TARGET_FACTOR
        index, successes, fitness = population.select_final(method.population, candidates, factor, n_jobs=n_jobs)
        summary.update(val_successes=int(successes), val_fitness=float(fitness),
                       champion_path=str(method.save_champion(index, seed, champion_dir)))
    summary["wall_time"] = active()                                # includes the final selection (spec §9.1)
    write_json(out_dir / SUMMARY_NAME, summary)
    if reporting:
        found = "" if summary["val_successes"] is None else (
            f", final controller succeeds on {summary['val_successes']} of {len(config.VAL_SEEDS)} validation maps")
        progress.milestone(f"{label(method, seed)}: finished after {generation} generations{found}")
    return summary
```

- [ ] **Step 11: GA as a method**

Replace `racing/ga.py` with:

```python
"""Genetic algorithm over dense network weights (revision 3 §5.1): tournament selection, uniform crossover and
Gaussian mutation whose strength follows the share of the previous generation's children that succeeded."""
import numpy as np

from racing import config, evolution, nets, paths, population
from racing.curriculum import mutation_strength
from racing.nets import DensePopulation


def tournament(rng, fitness, count, size=config.GA_TOURNAMENT):
    """Indices of `count` winners, each the fittest of `size` individuals drawn from the whole population."""
    entrants = rng.integers(0, len(fitness), (count, size))
    return entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]


class GA:
    """The GA as a method of the evolution loop (racing/evolution.py)."""
    approach = "ga"

    def __init__(self, pop_size=config.GA_POP, hidden=config.BASELINE_HIDDEN, activation="relu", s_max=config.GA_S_MAX):
        if pop_size < 2:
            raise ValueError(f"population size ({pop_size}) must be at least 2")
        self.pop_size = pop_size
        self.hidden = tuple(hidden)
        self.activation = activation
        self.s_max = s_max
        self.population = None
        self.is_kept = None
        self.step_size = float("nan")

    def initialize(self, rng):
        self.population = DensePopulation(nets.init_population(rng, self.pop_size, self.hidden), self.hidden,
                                          self.activation)
        self.is_kept = np.zeros(self.pop_size, dtype=bool)
        self.step_size = float("nan")                  # generation 0 is not produced by mutation

    def advance(self, rng, result, kept, event):
        """Survivors first and unchanged, then children by tournament, uniform crossover and Gaussian mutation."""
        weights = self.population.weights
        n, p = weights.shape
        children_before = ~self.is_kept
        child_success = float(result.all_success[children_before].mean()) if children_before.any() else float("nan")
        self.step_size = mutation_strength(child_success, self.s_max, self.s_max / 100)
        n_new = n - len(kept)
        mother = weights[tournament(rng, result.fitness, n_new)]
        father = weights[tournament(rng, result.fitness, n_new)]
        children = np.where(rng.random((n_new, p)) < config.GA_CROSSOVER_P, mother, father)
        mutate = rng.random((n_new, p)) < config.GA_MUTATION_P
        children = children + mutate * rng.normal(0.0, self.step_size, (n_new, p))
        self.population = DensePopulation(np.concatenate([weights[kept], children]), self.hidden, self.activation)
        self.is_kept = np.arange(n) < len(kept)

    def state(self):
        return {"weights": self.population.weights, "is_kept": self.is_kept, "step_size": self.step_size,
                "hidden": self.hidden, "activation": self.activation}

    def load_state(self, state):
        self.hidden, self.activation = tuple(state["hidden"]), state["activation"]
        self.population = DensePopulation(state["weights"], self.hidden, self.activation)
        self.is_kept = state["is_kept"]
        self.step_size = state["step_size"]

    def save_champion(self, index, seed, champion_dir):
        return population.save_dense_champion(self.approach, seed, self.population, index, champion_dir)

    def log_fields(self):
        return {}


def run_ga(seed=0, pop_size=config.GA_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1,
           on_generation=None, start_level=1, max_generations=None, s_max=config.GA_S_MAX):
    """The GA with run seed `seed` (revision 3 §5.1). Returns the run's summary (racing/evolution.py)."""
    return evolution.run(GA(pop_size=pop_size, s_max=s_max), seed, out_dir or paths.run_dir("ga", seed),
                         champion_dir, n_jobs, on_generation, start_level, max_generations)
```

- [ ] **Step 12: ES as a method**

Replace `racing/es.py` with:

```python
"""Evolution strategy over dense network weights (revision 3 §5.2): each child is one of the fittest cars plus
Gaussian noise of size σ, with the two-phase step-size rule. Started from a trained network it is the seeded ES
(revision 3 §5.3) or the speed push of an imitation method (revision 3 §6.4)."""
import numpy as np

from racing import config, evolution, nets, paths, population
from racing.curriculum import StepSize, success_fraction
from racing.nets import DensePopulation


def seeded_population(rng, flat, n, sigma):
    """The trained network itself, then n − 1 copies with N(0, σ²) added to every weight."""
    flat = np.asarray(flat, dtype=float)
    return np.vstack([flat, flat + rng.normal(0.0, sigma, (n - 1, len(flat)))])


class ES:
    """The ES as a method of the evolution loop (racing/evolution.py). Every car has an id; a child records its
    parent's id, so the 1/5 rule compares each child with its own parent."""

    def __init__(self, approach="es", pop_size=config.ES_POP, parents=config.ES_MU, hidden=config.BASELINE_HIDDEN,
                 activation="relu", initial=None, sigma0=config.ES_SIGMA0):
        if pop_size < 2:
            raise ValueError(f"population size ({pop_size}) must be at least 2")
        self.approach = approach
        self.pop_size = pop_size
        self.parents = parents
        self.hidden = tuple(hidden)
        self.activation = activation
        self.initial = None if initial is None else np.asarray(initial, dtype=float)
        self.sigma0 = sigma0
        self.step = StepSize(sigma0)
        self.population = None
        self.is_kept = None
        self.ids = None
        self.parent_ids = None
        self.next_id = 0

    @property
    def step_size(self):
        return self.step.sigma

    def initialize(self, rng):
        if self.initial is None:
            weights = nets.init_population(rng, self.pop_size, self.hidden)
        else:
            weights = seeded_population(rng, self.initial, self.pop_size, self.sigma0)
        self.population = DensePopulation(weights, self.hidden, self.activation)
        self.is_kept = np.zeros(self.pop_size, dtype=bool)
        self.ids = np.arange(self.pop_size)
        self.parent_ids = np.full(self.pop_size, -1)
        self.next_id = self.pop_size
        self.step = StepSize(self.sigma0)

    def success_rate(self, fitness):
        """Share of this generation's children that beat their own parent, a tie counting half, over the children
        whose parent is still in the population (revision 3 §5.2); nan if there are none."""
        position = {int(i): k for k, i in enumerate(self.ids)}
        children = [k for k in np.flatnonzero(~self.is_kept) if int(self.parent_ids[k]) in position]
        if not children:
            return float("nan")
        parents = [position[int(self.parent_ids[k])] for k in children]
        return success_fraction(fitness[children], fitness[parents])

    def advance(self, rng, result, kept, event):
        """Adapt σ, then keep the survivors first and unchanged and add children of the ES_MU fittest cars."""
        weights = self.population.weights
        n, p = weights.shape
        top = np.argsort(-result.fitness, kind="stable")[:min(self.parents, n)]
        children_before = ~self.is_kept
        child_success = float(result.all_success[children_before].mean()) if children_before.any() else float("nan")
        self.step.update(float(result.all_success[top].mean()), child_success, self.success_rate(result.fitness))
        if event == "level_up":
            self.step.reset()                           # restart the step size on the harder maps
        n_new = n - len(kept)
        chosen = top[rng.integers(0, len(top), n_new)]
        children = weights[chosen] + rng.normal(0.0, self.step.sigma, (n_new, p))
        self.population = DensePopulation(np.concatenate([weights[kept], children]), self.hidden, self.activation)
        self.parent_ids = np.concatenate([self.parent_ids[kept], self.ids[chosen]])
        self.ids = np.concatenate([self.ids[kept], np.arange(self.next_id, self.next_id + n_new)])
        self.next_id += n_new
        self.is_kept = np.arange(n) < len(kept)

    def state(self):
        return {"weights": self.population.weights, "is_kept": self.is_kept, "ids": self.ids,
                "parent_ids": self.parent_ids, "next_id": self.next_id, "step": self.step.state(),
                "hidden": self.hidden, "activation": self.activation}

    def load_state(self, state):
        self.hidden, self.activation = tuple(state["hidden"]), state["activation"]
        self.population = DensePopulation(state["weights"], self.hidden, self.activation)
        self.is_kept, self.ids, self.parent_ids = state["is_kept"], state["ids"], state["parent_ids"]
        self.next_id = state["next_id"]
        self.step = StepSize.from_state(state["step"])

    def save_champion(self, index, seed, champion_dir):
        return population.save_dense_champion(self.approach, seed, self.population, index, champion_dir)

    def log_fields(self):
        return {}


def trained_baseline(seed, champion_dir=config.CHAMPIONS_DIR):
    """Flat float64 weights of the baseline controller trained with `seed` (spec §9.4)."""
    from racing.backend import keras_cpu
    path = paths.champion_path("baseline", seed, champion_dir=champion_dir)
    if not path.exists():
        raise FileNotFoundError(f"no baseline controller for seed {seed} at {path}; run notebook 02 first")
    return nets.to_flat(keras_cpu().models.load_model(path))


def dagger_baseline(seed, runs_dir=None):
    """Flat float64 weights of the baseline after its DAgger refinement, the start of the seeded ES (revision 3
    §5.3)."""
    from racing.backend import keras_cpu
    path = paths.run_dir("baseline", seed, runs_dir) / "dagger.keras"     # paths.stage_path(..., "dagger"), Task 10
    if not path.exists():
        raise FileNotFoundError(f"no refined baseline for seed {seed} at {path}; run notebook 03 first")
    return nets.to_flat(keras_cpu().models.load_model(path))


def run_es(seed=0, pop_size=config.ES_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1,
           on_generation=None, start_level=1, max_generations=None, initial=None, approach="es",
           hidden=config.BASELINE_HIDDEN, activation="relu"):
    """The ES with run seed `seed`; with `initial`, the flat weights of a trained network of architecture
    (hidden, activation), it starts from that network. Returns the run's summary (racing/evolution.py)."""
    method = ES(approach=approach, pop_size=pop_size, hidden=hidden, activation=activation, initial=initial)
    return evolution.run(method, seed, out_dir or paths.run_dir(approach, seed), champion_dir, n_jobs,
                         on_generation, start_level, max_generations)
```

- [ ] **Step 13: Keep the viewer, evaluation, pilots and notebook script working**

These are minimal changes; Tasks 12, 14, 15 and 16 rewrite the files.

`racing/app.py`:
- `self.population = {"ga": config.GA_POP, "es": config.ES_LAMBDA, "es_seeded": config.ES_LAMBDA}` → `self.population = {"ga": config.GA_POP, "es": config.ES_POP, "es_seeded": config.ES_POP}`;
- in `trainer_main`, after `out_dir = config.APP_RUNS_DIR / approach / f"seed{seed}"` add

```python
    if approach in config.EVOLUTION_APPROACHES:
        paths.remove(out_dir)              # a viewer run always starts afresh, never from an old checkpoint
```

  and change `es.run_es(lam=size, ...)` to `es.run_es(pop_size=size, ...)` (other arguments unchanged);
- replace `report_policy` with

```python
def report_policy(report, indices):
    """Policy for the chosen members of a training report."""
    if "population" in report:
        return report["population"].subset(indices).policy()
    if "weights" in report:
        return nets.population_policy(report["weights"][indices], tuple(report["hidden"]), report["activation"])
    return nets.mixed_policy([report["nets"][i] for i in indices])
```

- in `App.receive`, replace `report["leveled_up"]` with `report.get("leveled_up", report.get("event") in ("level_up", "goal"))`
  (engine reports carry `event` instead; Task 14 rewrites `receive`);
- in `stats_lines`, replace `report['kept_finish']` with `report.get('kept_finish', report.get('kept_success', nan))` and
  `report['new_finish']` with `report.get('new_finish', report.get('child_success', nan))`, so the viewer can show an
  engine report until Task 14 replaces the stats.

`racing/evaluate.py`: after `evolution_logs` add

```python
def level_events(log):
    """Rows at which a run advanced a level: the revision 3 `event` column, or revision 2.0's `leveled_up`."""
    if "event" in log:
        return log["event"].fillna("").isin(["level_up", "goal"])
    return log["leveled_up"].astype(bool)
```

and in `figures` replace `for evals in log.loc[log["leveled_up"].astype(bool), "evals"]:` with `for evals in log.loc[level_events(log), "evals"]:`.

`scripts/pilots.py`: replace `level_ups`, `pilot_ga`, `pilot_es` and `pilot_seeded` (and `paths` joins the imports: `from racing import config, es, ga, imitation, paths  # noqa: E402`):

```python
def level_ups(log):
    """Generations at which the level advanced or the mastery goal was reached, as text."""
    events = log["event"].fillna("")
    return " ".join(str(g) for g in log.loc[events.isin(["level_up", "goal"]), "generation"]) or "none"


def pilot_ga(s_max, out_root, generations=100, pop_size=config.GA_POP, n_jobs=config.N_JOBS):
    out = Path(out_root) / f"ga_smax{s_max}"
    paths.remove(out)                                  # a pilot always starts afresh
    result = ga.run_ga(pop_size=pop_size, seed=0, out_dir=out, champion_dir=None, n_jobs=n_jobs, s_max=s_max,
                       max_generations=generations)
    log = pd.read_csv(out / "log.csv")
    return {"pilot": f"GA s_max={s_max}", "generations": result["generations"],
            "level_reached": result["level_reached"], "level_ups": level_ups(log),
            "min_share_last20": round(float(log["min_share"].tail(20).mean()), 2),
            "child_success_last": round(float(log["child_success"].iloc[-1]), 2),
            "best_last": round(float(log["best"].iloc[-1]), 3), "minutes": round(result["wall_time"] / 60, 1)}


def pilot_es(out_root, generations=100, pop_size=config.ES_POP, n_jobs=config.N_JOBS, initial=None, name="ES",
             folder="es"):
    out = Path(out_root) / folder
    paths.remove(out)
    approach = "es" if initial is None else "es_seeded"
    result = es.run_es(pop_size=pop_size, seed=0, out_dir=out, champion_dir=None, n_jobs=n_jobs, initial=initial,
                       approach=approach, max_generations=generations)
    log = pd.read_csv(out / "log.csv")
    return {"pilot": name, "generations": result["generations"], "level_reached": result["level_reached"],
            "level_ups": level_ups(log), "min_share_last20": round(float(log["min_share"].tail(20).mean()), 2),
            "sigma_min": float(log["step_size"].min()), "sigma_last": float(log["step_size"].iloc[-1]),
            "child_success_last": round(float(log["child_success"].iloc[-1]), 2),
            "best_last": round(float(log["best"].iloc[-1]), 3), "minutes": round(result["wall_time"] / 60, 1)}


def pilot_seeded(out_root, generations=100, pop_size=config.ES_POP, n_jobs=config.N_JOBS):
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
    return pilot_es(folder, generations, pop_size, n_jobs, initial=initial, name="Seeded ES", folder="es_seeded")
```

In the module docstring, replace "with a shortened budget" with "for a fixed number of generations".

`scripts/make_notebooks.py`: replace `EVOLUTION_PROGRESS` with

```python
EVOLUTION_PROGRESS = (
    "def progress(report):\n"
    "    if report['generation'] % 50 == 0 or report['event']:\n"
    "        print(f\"gen {report['generation']:5d}  level {report['level']}  best {report['best']:.3f}  \"\n"
    "              f\"worst map {report['min_share']:.2f}  step {report['step_size']:.4f}  {report['event'] or ''}\")\n")
```

Do not run `scripts/make_notebooks.py` in this task: regenerating would overwrite the notebooks' saved outputs, which Task 15 archives first.

- [ ] **Step 14: Run the updated tests**

Run: `~/miniconda3/bin/python -m pytest tests/test_evolution.py tests/test_ga.py tests/test_es.py tests/test_population.py tests/test_curriculum.py tests/test_nets.py tests/test_app.py tests/test_pilots.py tests/test_evaluate.py tests/test_runner.py -q`
Expected: PASS.

- [ ] **Step 15: Run the fast suite and the slow engine tests**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all pass.

Run: `~/miniconda3/bin/python -m pytest -q -m slow tests/test_evolution.py tests/test_end_to_end.py`
Expected: PASS (real Easy maps are generated for the fresh seeds; a few minutes).

Run: `grep -rn "ES_LAMBDA\|GA_KEPT\|GA_KEEP_MAX_FRACTION\|EVAL_BUDGET\|PACE_FACTOR\|LEVEL_UP_\|LevelTracker\|draw_seeds\|evaluate_weights\|RunLogger\|make_report\|select_champion\|population\.save_champion\|best_gen" racing scripts tests`
Expected: no output.

- [ ] **Step 16: Commit**

```bash
git add racing/evolution.py racing/ga.py racing/es.py racing/population.py racing/curriculum.py racing/nets.py \
        racing/config.py racing/checkpoint.py racing/app.py racing/evaluate.py scripts/pilots.py scripts/make_notebooks.py \
        tests/test_evolution.py tests/test_ga.py tests/test_es.py tests/test_population.py tests/test_curriculum.py \
        tests/test_nets.py tests/test_evaluate.py tests/test_app.py tests/test_pilots.py tests/test_end_to_end.py
git commit -m "Evolution engine: fresh maps, time trials, survival, mastery levels, speed push, checkpoints; GA and ES as methods (revision 3 §4, §5.1-5.3, §8.2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/evolution.py racing/ga.py racing/es.py racing/population.py racing/curriculum.py racing/nets.py racing/config.py racing/checkpoint.py racing/app.py racing/evaluate.py scripts/pilots.py scripts/make_notebooks.py tests/test_evolution.py tests/test_ga.py tests/test_es.py tests/test_population.py tests/test_curriculum.py tests/test_nets.py tests/test_evaluate.py tests/test_app.py tests/test_pilots.py tests/test_end_to_end.py
```

---

### Task 7: NEAT genomes, innovation record and operators

**Files:**
- Create: `racing/neat.py`
- Modify: `racing/config.py` (NEAT block after the ES block; `APPROACHES`, `EVOLUTION_APPROACHES`, `LABELS`, `COLORS`)
- Modify: `racing/checkpoint.py` (`SETTINGS_KEYS` gains every `NEAT_*` name)
- Modify: `tests/test_paths.py:29-32` (the approach list now ends with `neat`)
- Test: `tests/test_neat.py` (new)

**Interfaces:**
- Consumes: `config.N_INPUTS` (10), `config.N_ACTIONS` (15); Task 4: `checkpoint.SETTINGS_KEYS` (a tuple of config names) and `checkpoint.settings_record()`.
- Produces (used by Tasks 8, 9, 12–14): the `NEAT_*` names in `checkpoint.SETTINGS_KEYS`, so a checkpoint or summary recorded with other NEAT settings is refused on resumption; `neat.INPUT_IDS`, `neat.OUTPUT_IDS`, `neat.FIRST_HIDDEN`, `neat.N_IN`, `neat.N_OUT`; `Genome(node_ids, bias, innov, src, dst, weight, enabled)` with `copy()`, `n_params`, `to_dict()`, `Genome.from_dict(d)`; `Innovations(next_innov=0, next_node=FIRST_HIDDEN, conns=None, splits=None)` with `conn(src, dst) -> int`, `split(innov) -> int`, `fresh_node() -> int`, `state() -> dict`, `Innovations.from_state(state)`; `initial_genome(rng, innovations) -> Genome`; `creates_cycle(genome, src, dst) -> bool`; `add_connection(rng, genome, innovations) -> Genome`; `add_node(rng, genome, innovations) -> Genome`; `mutate(rng, genome, innovations, sigma) -> Genome`; `crossover(rng, fitter, other) -> Genome`; `distance(a, b) -> float`; `node_depths(genome) -> dict[int, int]`. Config: every `NEAT_*` constant of the Interface Contract; `"neat"` in `APPROACHES` and `EVOLUTION_APPROACHES`; `LABELS["neat"] = "NEAT"`; `COLORS["neat"] = "#F0E442"`.

Design notes for the implementer:
- Node ids: inputs `0..9` are implicit (never stored), outputs `10..24`, hidden nodes from `25`. Connections go **from an input or hidden node to a hidden or output node**; outputs are sinks.
- The innovation record is kept for the whole run (revision 3 §5.4): the same `(src, dst)` pair always gets the same innovation number, and splitting the same connection always gets the same node id. If a genome already contains that node (it re-enabled the connection and splits it again), `add_node` uses `innovations.fresh_node()`, an id that is not recorded.
- `creates_cycle` follows **all** connections, enabled or not. The graph of all connections is then always acyclic, so toggling a connection on can never create a cycle. A crossover child only carries genes of its fitter parent (matching genes exist in both), so it is acyclic too.
- `mutate` reads the `config.NEAT_*` probabilities at call time (not as default arguments), so tests can monkeypatch them.
- Every random draw happens in a fixed order, so a run is reproducible from its random generator state.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_neat.py`:

```python
import numpy as np
import pytest

from racing import config, neat


def grown(rng, innovations, rounds=25):
    """A genome grown by many structural mutations, with some connections disabled and random biases."""
    genome = neat.initial_genome(rng, innovations)
    for _ in range(rounds):
        genome = neat.add_node(rng, genome, innovations)
        genome = neat.add_connection(rng, genome, innovations)
        if rng.random() < 0.3:
            k = int(rng.integers(len(genome.innov)))
            genome.enabled[k] = not genome.enabled[k]
    genome.bias = rng.normal(0.0, 0.5, len(genome.bias))
    return genome


def acyclic(genome):
    """True if every node can be ordered so that all connections (enabled or not) point forwards."""
    succ, indegree = {}, {int(n): 0 for n in (*neat.INPUT_IDS, *genome.node_ids.tolist())}
    for s, d in zip(genome.src.tolist(), genome.dst.tolist()):
        succ.setdefault(s, []).append(d)
        indegree[d] += 1
    ready = [n for n, k in indegree.items() if k == 0]
    seen = 0
    while ready:
        node = ready.pop()
        seen += 1
        for d in succ.get(node, ()):
            indegree[d] -= 1
            if indegree[d] == 0:
                ready.append(d)
    return seen == len(indegree)


NEAT_NAMES = ("NEAT_POP", "NEAT_CROSSOVER", "NEAT_TOURNAMENT", "NEAT_DISABLE_INHERIT", "NEAT_PERTURB",
              "NEAT_REPLACE", "NEAT_ADD_CONN", "NEAT_ADD_NODE", "NEAT_TOGGLE", "NEAT_SIGMA0", "NEAT_C1", "NEAT_C2",
              "NEAT_C3", "NEAT_SMALL_GENOME", "NEAT_THRESHOLD0", "NEAT_THRESHOLD_STEP", "NEAT_TARGET_SPECIES",
              "NEAT_STAGNATION")


def test_neat_is_an_evolutionary_approach_with_its_own_color():
    assert "neat" in config.APPROACHES and "neat" in config.EVOLUTION_APPROACHES
    assert config.LABELS["neat"] == "NEAT" and config.COLORS["neat"] == "#F0E442"


def test_neat_settings_are_recorded_for_resumption():
    from racing import checkpoint
    record = checkpoint.settings_record()
    for name in NEAT_NAMES:
        assert name in checkpoint.SETTINGS_KEYS
        assert record[name] == getattr(config, name)


def test_initial_genome_connects_every_input_to_every_output():
    genome = neat.initial_genome(np.random.default_rng(0), neat.Innovations())
    assert genome.node_ids.tolist() == list(neat.OUTPUT_IDS)
    assert np.all(genome.bias == 0.0)
    assert len(genome.innov) == neat.N_IN * neat.N_OUT == 150
    assert set(zip(genome.src.tolist(), genome.dst.tolist())) == {(i, o) for i in neat.INPUT_IDS for o in neat.OUTPUT_IDS}
    assert np.all(np.diff(genome.innov) > 0) and genome.enabled.all()
    assert genome.n_params == 165
    assert 0.8 < np.std(genome.weight) < 1.2                     # standard normal weights


def test_innovation_record_is_shared_by_all_genomes():
    rng = np.random.default_rng(1)
    innovations = neat.Innovations()
    a, b = neat.initial_genome(rng, innovations), neat.initial_genome(rng, innovations)
    assert np.array_equal(a.innov, b.innov) and not np.array_equal(a.weight, b.weight)
    assert innovations.conn(0, 10) == innovations.conn(0, 10) == int(a.innov[0])
    node = innovations.split(5)
    assert innovations.split(5) == node == neat.FIRST_HIDDEN
    assert innovations.fresh_node() == node + 1 and innovations.split(5) == node
    copy = neat.Innovations.from_state(innovations.state())
    assert copy.state() == innovations.state()
    assert copy.conn(3, 12) == innovations.conn(3, 12)


def test_creates_cycle_follows_all_connections():
    genome = neat.Genome.from_dict({
        "node_ids": [10, 25, 26], "bias": [0.0] * 3,
        "innov": [0, 1, 2], "src": [0, 25, 26], "dst": [25, 26, 10],
        "weight": [1.0] * 3, "enabled": [True, False, True]})   # 0 -> 25 -> 26 -> 10, the middle one disabled
    assert neat.creates_cycle(genome, 25, 25)                    # self-loop
    assert neat.creates_cycle(genome, 26, 25)                    # closes 25 -> 26 even though it is disabled
    assert not neat.creates_cycle(genome, 0, 26)
    assert not neat.creates_cycle(genome, 25, 10)


def test_add_node_splits_a_connection():
    rng = np.random.default_rng(3)
    innovations = neat.Innovations()
    genome = neat.initial_genome(rng, innovations)
    child = neat.add_node(rng, genome, innovations)
    assert len(child.node_ids) == len(genome.node_ids) + 1
    node = int(child.node_ids[-1])
    assert node == neat.FIRST_HIDDEN and child.bias[-1] == 0.0
    split = np.flatnonzero(genome.enabled & ~child.enabled[np.searchsorted(child.innov, genome.innov)])
    assert len(split) == 1
    s, d, w = int(genome.src[split[0]]), int(genome.dst[split[0]]), float(genome.weight[split[0]])
    new = {(int(a), int(b)): float(v) for a, b, v in zip(child.src, child.dst, child.weight)}
    assert new[(s, node)] == 1.0 and new[(node, d)] == w
    assert np.all(np.diff(child.innov) > 0)
    assert genome.enabled.all()                                  # the parent is unchanged


def test_the_same_split_gets_the_same_node_unless_the_genome_has_it():
    rng = np.random.default_rng(4)
    innovations = neat.Innovations()
    a = neat.initial_genome(rng, innovations)
    b = a.copy()
    a.enabled[:] = False
    a.enabled[7] = True
    b.enabled[:] = False
    b.enabled[7] = True
    split_a = neat.add_node(rng, a, innovations)
    split_b = neat.add_node(rng, b, innovations)
    assert split_a.node_ids[-1] == split_b.node_ids[-1] == neat.FIRST_HIDDEN
    assert innovations.state() == neat.Innovations.from_state(innovations.state()).state()
    again = split_a.copy()
    again.enabled[:] = False
    again.enabled[np.searchsorted(again.innov, a.innov[7])] = True          # re-enable the split connection
    twice = neat.add_node(rng, again, innovations)
    assert len(set(twice.node_ids.tolist())) == len(twice.node_ids)          # no duplicate node
    assert int(twice.node_ids[-1]) == neat.FIRST_HIDDEN + 1
    assert innovations.split(int(a.innov[7])) == neat.FIRST_HIDDEN           # the record keeps the first node


def test_structural_mutations_keep_genomes_valid():
    rng = np.random.default_rng(5)
    innovations = neat.Innovations()
    for _ in range(5):
        genome = grown(rng, innovations, rounds=40)
        pairs = list(zip(genome.src.tolist(), genome.dst.tolist()))
        assert len(pairs) == len(set(pairs))                                  # no duplicate connection
        assert all(s < neat.N_IN or s >= neat.FIRST_HIDDEN for s, _ in pairs)  # sources: inputs or hidden
        assert all(d >= neat.N_IN for _, d in pairs)                           # targets: outputs or hidden
        assert set(genome.dst.tolist()) <= set(genome.node_ids.tolist())
        assert np.all(np.diff(genome.innov) > 0) and np.all(np.diff(genome.node_ids) > 0)
        assert len(genome.bias) == len(genome.node_ids)
        assert acyclic(genome)


def test_weight_mutation_rates(monkeypatch):
    monkeypatch.setattr(config, "NEAT_ADD_CONN", 0.0)
    monkeypatch.setattr(config, "NEAT_ADD_NODE", 0.0)
    monkeypatch.setattr(config, "NEAT_TOGGLE", 0.0)
    rng = np.random.default_rng(6)
    innovations = neat.Innovations()
    genomes = [neat.initial_genome(rng, innovations) for _ in range(40)]
    before = np.concatenate([g.weight for g in genomes])
    still = np.concatenate([neat.mutate(rng, g, innovations, 0.0).weight for g in genomes])
    replaced = np.mean(still != before)
    assert abs(replaced - config.NEAT_REPLACE) < 0.02                 # σ = 0: only replacements change weights
    moved = np.concatenate([neat.mutate(rng, g, innovations, 1.0).weight for g in genomes])
    assert abs(np.mean(moved != before) - (config.NEAT_REPLACE + config.NEAT_PERTURB)) < 0.03
    assert all(g.n_params == 165 for g in genomes)                     # the parents are unchanged


def test_mutate_adds_structure_with_its_probabilities(monkeypatch):
    monkeypatch.setattr(config, "NEAT_ADD_NODE", 1.0)
    monkeypatch.setattr(config, "NEAT_ADD_CONN", 0.0)
    rng = np.random.default_rng(7)
    innovations = neat.Innovations()
    genome = neat.mutate(rng, neat.initial_genome(rng, innovations), innovations, 0.1)
    assert len(genome.node_ids) == neat.N_OUT + 1 and len(genome.innov) == 152


def test_crossover_aligns_genes_by_innovation():
    rng = np.random.default_rng(8)
    innovations = neat.Innovations()
    other = neat.initial_genome(rng, innovations)
    fitter = neat.add_node(rng, other.copy(), innovations)
    fitter.weight = fitter.weight + 10.0
    child = neat.crossover(rng, fitter, other)
    assert np.array_equal(child.innov, fitter.innov) and np.array_equal(child.node_ids, fitter.node_ids)
    common = np.isin(fitter.innov, other.innov)
    from_other = np.isin(child.weight[common], other.weight)
    assert 0.3 < from_other.mean() < 0.7                              # matching genes: either parent
    assert np.array_equal(child.weight[~common], fitter.weight[~common])   # disjoint and excess: the fitter parent


def test_a_gene_disabled_in_either_parent_is_usually_disabled():
    rng = np.random.default_rng(9)
    innovations = neat.Innovations()
    fitter = neat.initial_genome(rng, innovations)
    other = fitter.copy()
    other.enabled[:] = False
    disabled = np.mean([~neat.crossover(rng, fitter, other).enabled for _ in range(40)])
    assert abs(disabled - config.NEAT_DISABLE_INHERIT) < 0.03


def test_distance():
    rng = np.random.default_rng(10)
    innovations = neat.Innovations()
    a = neat.initial_genome(rng, innovations)
    assert neat.distance(a, a) == 0.0
    b = a.copy()
    b.weight = b.weight + 1.0
    assert np.isclose(neat.distance(a, b), config.NEAT_C3 * 1.0)
    c = neat.add_node(rng, a, innovations)                             # two excess genes, N = 152
    assert np.isclose(neat.distance(a, c), config.NEAT_C1 * 2 / 152)
    small = neat.Genome.from_dict({"node_ids": [10], "bias": [0.0], "innov": [0, 3], "src": [0, 1], "dst": [10, 10],
                                   "weight": [1.0, 1.0], "enabled": [True, True]})
    other = neat.Genome.from_dict({"node_ids": [10], "bias": [0.0], "innov": [0, 1, 5], "src": [0, 2, 3],
                                   "dst": [10, 10, 10], "weight": [2.0, 1.0, 1.0], "enabled": [True, True, True]})
    # matching 0 (|Δw| = 1); disjoint 1 and 3 (below the smaller maximum 3); excess 5; N = 1 below 20 genes
    assert np.isclose(neat.distance(small, other), config.NEAT_C1 * 1 + config.NEAT_C2 * 2 + config.NEAT_C3 * 1.0)


def test_node_depths_are_longest_enabled_paths():
    genome = neat.Genome.from_dict({
        "node_ids": [10, 11, 25, 26], "bias": [0.0] * 4,
        "innov": [0, 1, 2, 3, 4, 5], "src": [0, 25, 0, 26, 1, 2], "dst": [25, 26, 26, 10, 11, 10],
        "weight": [1.0] * 6, "enabled": [True, True, True, True, False, True]})
    depth = neat.node_depths(genome)
    assert depth[0] == 0 and depth[25] == 1 and depth[26] == 2 and depth[10] == 3
    assert depth[11] == 1                                              # no enabled input: bias only
    assert all(depth[o] >= 1 for o in neat.OUTPUT_IDS if o in depth)


def test_genome_round_trips_through_a_dict():
    rng = np.random.default_rng(11)
    genome = grown(rng, neat.Innovations())
    copy = neat.Genome.from_dict(genome.to_dict())
    for name in neat.Genome.FIELDS:
        assert np.array_equal(getattr(copy, name), getattr(genome, name))
        assert getattr(copy, name).dtype == getattr(genome, name).dtype
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_neat.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'racing.neat'`.

- [ ] **Step 3: Add the configuration**

In `racing/config.py`, after the `# --- Evolution strategy` block, add:

```python
# --- NEAT (revision 3 §5.4) ---
NEAT_POP = 1000
NEAT_CROSSOVER = 0.75        # share of children produced by crossover; the rest are mutated copies
NEAT_TOURNAMENT = 3          # parents are the winners of tournaments within their species
NEAT_DISABLE_INHERIT = 0.75  # a gene disabled in either parent is disabled in the child with this probability
NEAT_PERTURB = 0.8           # per weight and bias: perturbation by N(0, σ²)
NEAT_REPLACE = 0.05          # per weight and bias: replacement by a standard normal value
NEAT_ADD_CONN = 0.05         # per child: one new connection
NEAT_ADD_NODE = 0.03         # per child: one connection split by a new hidden node
NEAT_TOGGLE = 0.01           # per child: one connection's enabled flag toggled
NEAT_SIGMA0 = 0.5            # weights start standard normal, so the step size starts larger than the ES's
NEAT_C1 = 1.0                # compatibility distance δ = C1·E/N + C2·D/N + C3·W̄
NEAT_C2 = 1.0
NEAT_C3 = 0.4
NEAT_SMALL_GENOME = 20       # N = 1 when both genomes have fewer genes than this
NEAT_THRESHOLD0 = 3.0        # initial compatibility threshold
NEAT_THRESHOLD_STEP = 0.1    # threshold change per generation, towards NEAT_TARGET_SPECIES species
NEAT_TARGET_SPECIES = 20
NEAT_STAGNATION = 20         # generations without improvement after which a species receives no children
```

In the approaches block, change:

```python
APPROACHES = ("baseline", "kt_random", "kt_hyperband", "kt_bayesian", "es", "ga", "es_seeded", "neat")
TUNER_APPROACHES = ("kt_random", "kt_hyperband", "kt_bayesian")
EVOLUTION_APPROACHES = ("ga", "es", "es_seeded", "neat")
LABELS = {"baseline": "Baseline", "kt_random": "KT Random", "kt_hyperband": "KT Hyperband",
          "kt_bayesian": "KT Bayesian", "es": "ES", "ga": "GA", "es_seeded": "Seeded ES", "neat": "NEAT"}
COLORS = {"baseline": "#7F7F7F", "kt_random": "#E69F00", "kt_hyperband": "#56B4E9",
          "kt_bayesian": "#CC79A7", "es": "#009E73", "ga": "#D55E00", "es_seeded": "#0072B2",
          "neat": "#F0E442"}         # the remaining Okabe–Ito color (revision 3 §9)
```

(Keep `IMITATION_APPROACHES` and `RUN_ORDER`, which Task 5 placed directly after `EVOLUTION_APPROACHES`, where they are; `RUN_ORDER` then counts 24 runs.)

In `racing/checkpoint.py`, add the NEAT names at the end of the `SETTINGS_KEYS` tuple (keep every existing name):

```python
                 "NEAT_POP", "NEAT_CROSSOVER", "NEAT_TOURNAMENT", "NEAT_DISABLE_INHERIT", "NEAT_PERTURB",
                 "NEAT_REPLACE", "NEAT_ADD_CONN", "NEAT_ADD_NODE", "NEAT_TOGGLE", "NEAT_SIGMA0", "NEAT_C1",
                 "NEAT_C2", "NEAT_C3", "NEAT_SMALL_GENOME", "NEAT_THRESHOLD0", "NEAT_THRESHOLD_STEP",
                 "NEAT_TARGET_SPECIES", "NEAT_STAGNATION",
```

All NEAT values are plain ints and floats, so `settings_record()` stores them unchanged.

In `tests/test_paths.py`, update the approach-list test:

```python
def test_every_approach_has_a_label_and_color():
    assert config.APPROACHES[-1] == "neat"
    assert config.LABELS["es_seeded"] == "Seeded ES" and config.COLORS["es_seeded"] == "#0072B2"
    assert set(config.TUNER_APPROACHES) | set(config.EVOLUTION_APPROACHES) | {"baseline"} == set(config.APPROACHES)
    assert all(a in config.LABELS and a in config.COLORS for a in config.APPROACHES)
```

- [ ] **Step 4: Write the implementation**

Create `racing/neat.py`:

```python
"""NEAT, neuroevolution of augmenting topologies (revision 3 §5.4): feed-forward networks whose structure grows by
mutation. A genome lists node genes (output and hidden nodes, each with a bias; the inputs are implicit) and
connection genes. Every structural mutation is recorded for the whole run, so the same connection or split has the
same innovation number or node id in every genome, which aligns genomes for crossover and speciation."""
from dataclasses import dataclass

import numpy as np

from racing import config

N_IN = config.N_INPUTS                         # 10
N_OUT = config.N_ACTIONS                       # 15
INPUT_IDS = range(0, N_IN)
OUTPUT_IDS = range(N_IN, N_IN + N_OUT)
FIRST_HIDDEN = N_IN + N_OUT                    # 25
CONNECTION_ATTEMPTS = 20                       # random node pairs one add-connection mutation tries


@dataclass(eq=False)
class Genome:
    """Node genes (output and hidden ids, sorted, with their biases) and connection genes sorted by innovation."""
    node_ids: np.ndarray        # int64
    bias: np.ndarray            # float64, aligned with node_ids
    innov: np.ndarray           # int64, sorted
    src: np.ndarray             # int64 node ids, aligned with innov
    dst: np.ndarray
    weight: np.ndarray          # float64
    enabled: np.ndarray         # bool

    FIELDS = ("node_ids", "bias", "innov", "src", "dst", "weight", "enabled")
    DTYPES = (np.int64, float, np.int64, np.int64, np.int64, float, bool)

    def copy(self):
        return Genome(*(getattr(self, name).copy() for name in self.FIELDS))

    @property
    def n_params(self):
        """Enabled connections plus biases: the parameters the network uses (revision 3 §5.4)."""
        return int(np.sum(self.enabled)) + len(self.bias)

    def to_dict(self):
        return {name: getattr(self, name).tolist() for name in self.FIELDS}

    @classmethod
    def from_dict(cls, d):
        return cls(*(np.asarray(d[name], dtype=dtype) for name, dtype in zip(cls.FIELDS, cls.DTYPES)))


class Innovations:
    """The run's record of structural mutations: one innovation number per (source, target) pair and one node id per
    split connection."""

    def __init__(self, next_innov=0, next_node=FIRST_HIDDEN, conns=None, splits=None):
        self.next_innov = int(next_innov)
        self.next_node = int(next_node)
        self.conns = dict(conns or {})          # (src, dst) -> innovation number
        self.splits = dict(splits or {})        # innovation number of the split connection -> node id

    def conn(self, src, dst):
        key = (int(src), int(dst))
        if key not in self.conns:
            self.conns[key] = self.next_innov
            self.next_innov += 1
        return self.conns[key]

    def split(self, innov):
        innov = int(innov)
        if innov not in self.splits:
            self.splits[innov] = self.fresh_node()
        return self.splits[innov]

    def fresh_node(self):
        """A node id never used before; not recorded as a split."""
        node = self.next_node
        self.next_node += 1
        return node

    def state(self):
        return {"next_innov": self.next_innov, "next_node": self.next_node,
                "conns": sorted([s, d, i] for (s, d), i in self.conns.items()),
                "splits": sorted([i, n] for i, n in self.splits.items())}

    @classmethod
    def from_state(cls, state):
        return cls(state["next_innov"], state["next_node"], {(s, d): i for s, d, i in state["conns"]},
                   {i: n for i, n in state["splits"]})


def initial_genome(rng, innovations):
    """Every input connected to every output (150 connections), standard normal weights, zero biases."""
    src = np.repeat(np.arange(N_IN, dtype=np.int64), N_OUT)
    dst = np.tile(np.arange(N_IN, N_IN + N_OUT, dtype=np.int64), N_IN)
    innov = np.array([innovations.conn(s, d) for s, d in zip(src.tolist(), dst.tolist())], dtype=np.int64)
    weight = rng.standard_normal(len(src))
    order = np.argsort(innov, kind="stable")
    return Genome(node_ids=np.arange(N_IN, N_IN + N_OUT, dtype=np.int64), bias=np.zeros(N_OUT), innov=innov[order],
                  src=src[order], dst=dst[order], weight=weight[order], enabled=np.ones(len(src), dtype=bool))


def _with_connections(genome, innov, src, dst, weight):
    """A copy with new enabled connection genes, kept sorted by innovation number."""
    g = genome.copy()
    order = np.argsort(np.concatenate([g.innov, np.asarray(innov, dtype=np.int64)]), kind="stable")
    g.innov = np.concatenate([g.innov, np.asarray(innov, dtype=np.int64)])[order]
    g.src = np.concatenate([g.src, np.asarray(src, dtype=np.int64)])[order]
    g.dst = np.concatenate([g.dst, np.asarray(dst, dtype=np.int64)])[order]
    g.weight = np.concatenate([g.weight, np.asarray(weight, dtype=float)])[order]
    g.enabled = np.concatenate([g.enabled, np.ones(len(innov), dtype=bool)])[order]
    return g


def creates_cycle(genome, src, dst):
    """True if a connection src -> dst would close a cycle. Disabled connections count too, so enabling any
    connection later can never create one."""
    src, dst = int(src), int(dst)
    if src == dst:
        return True
    succ = {}
    for s, d in zip(genome.src.tolist(), genome.dst.tolist()):
        succ.setdefault(s, []).append(d)
    seen, stack = {dst}, [dst]
    while stack:
        node = stack.pop()
        if node == src:
            return True
        for nxt in succ.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def add_connection(rng, genome, innovations):
    """Connect a random pair of unconnected nodes, from an input or hidden node to a hidden or output node, with a
    standard normal weight, unless it would create a cycle. Gives up after CONNECTION_ATTEMPTS random pairs."""
    sources = np.concatenate([np.arange(N_IN, dtype=np.int64), genome.node_ids[genome.node_ids >= FIRST_HIDDEN]])
    targets = genome.node_ids
    existing = set(zip(genome.src.tolist(), genome.dst.tolist()))
    for _ in range(CONNECTION_ATTEMPTS):
        s = int(sources[rng.integers(len(sources))])
        d = int(targets[rng.integers(len(targets))])
        if (s, d) in existing or creates_cycle(genome, s, d):
            continue
        return _with_connections(genome, [innovations.conn(s, d)], [s], [d], [rng.standard_normal()])
    return genome


def add_node(rng, genome, innovations):
    """Split a random enabled connection with a hidden node: source -> node with weight 1, node -> target with the
    old weight, old connection disabled. The recorded node for that split is used unless this genome already has it
    (the connection was re-enabled and is split again); then a fresh, unrecorded node id is used."""
    on = np.flatnonzero(genome.enabled)
    if on.size == 0:
        return genome
    k = int(on[rng.integers(on.size)])
    s, d, w = int(genome.src[k]), int(genome.dst[k]), float(genome.weight[k])
    node = innovations.split(int(genome.innov[k]))
    if node in set(genome.node_ids.tolist()):
        node = innovations.fresh_node()
    g = genome.copy()
    g.enabled[k] = False
    position = int(np.searchsorted(g.node_ids, node))
    g.node_ids = np.insert(g.node_ids, position, node)
    g.bias = np.insert(g.bias, position, 0.0)
    return _with_connections(g, [innovations.conn(s, node), innovations.conn(node, d)], [s, node], [node, d],
                             [1.0, w])


def _mutate_values(rng, values, sigma):
    """Each value replaced by a standard normal value with probability NEAT_REPLACE, otherwise perturbed by
    N(0, σ²) with probability NEAT_PERTURB."""
    u = rng.random(len(values))
    noise = rng.normal(0.0, sigma, len(values))
    fresh = rng.standard_normal(len(values))
    replace = u < config.NEAT_REPLACE
    perturb = ~replace & (u < config.NEAT_REPLACE + config.NEAT_PERTURB)
    return np.where(replace, fresh, np.where(perturb, values + noise, values))


def mutate(rng, genome, innovations, sigma):
    """A mutated copy (revision 3 §5.4): weights and biases, then with their own probabilities a new connection, a
    new node and one toggled connection."""
    g = genome.copy()
    g.weight = _mutate_values(rng, g.weight, sigma)
    g.bias = _mutate_values(rng, g.bias, sigma)
    if rng.random() < config.NEAT_ADD_CONN:
        g = add_connection(rng, g, innovations)
    if rng.random() < config.NEAT_ADD_NODE:
        g = add_node(rng, g, innovations)
    if rng.random() < config.NEAT_TOGGLE and len(g.innov):
        k = int(rng.integers(len(g.innov)))
        g.enabled[k] = not g.enabled[k]
    return g


def crossover(rng, fitter, other):
    """Child of two genomes aligned by innovation number: matching genes from a random parent, disjoint and excess
    genes from the fitter parent; a matching gene disabled in either parent is disabled with probability
    NEAT_DISABLE_INHERIT. The child has the fitter parent's nodes; a node both parents have takes a random parent's
    bias."""
    child = fitter.copy()
    _, i_f, i_o = np.intersect1d(fitter.innov, other.innov, assume_unique=True, return_indices=True)
    take = rng.random(len(i_f)) < 0.5
    child.weight[i_f[take]] = other.weight[i_o[take]]
    disabled = ~fitter.enabled[i_f] | ~other.enabled[i_o]
    keep_off = rng.random(len(i_f)) < config.NEAT_DISABLE_INHERIT
    child.enabled[i_f] = ~(disabled & keep_off)
    _, n_f, n_o = np.intersect1d(fitter.node_ids, other.node_ids, assume_unique=True, return_indices=True)
    take = rng.random(len(n_f)) < 0.5
    child.bias[n_f[take]] = other.bias[n_o[take]]
    return child


def distance(a, b):
    """Compatibility distance δ = C1·E/N + C2·D/N + C3·W̄ (revision 3 §5.4): E excess genes (beyond the smaller
    genome's largest innovation number), D disjoint genes, W̄ the mean weight difference of matching genes, N the
    larger genome's gene count (1 if both have fewer than NEAT_SMALL_GENOME)."""
    if len(a.innov) == 0 and len(b.innov) == 0:
        return 0.0
    _, i_a, i_b = np.intersect1d(a.innov, b.innov, assume_unique=True, return_indices=True)
    unmatched = np.concatenate([np.delete(a.innov, i_a), np.delete(b.innov, i_b)])
    cutoff = min(a.innov.max() if len(a.innov) else -1, b.innov.max() if len(b.innov) else -1)
    excess = int(np.sum(unmatched > cutoff))
    disjoint = len(unmatched) - excess
    n = max(len(a.innov), len(b.innov))
    n = 1 if n < config.NEAT_SMALL_GENOME else n
    w = float(np.mean(np.abs(a.weight[i_a] - b.weight[i_b]))) if len(i_a) else 0.0
    return config.NEAT_C1 * excess / n + config.NEAT_C2 * disjoint / n + config.NEAT_C3 * w


def node_depths(genome):
    """Longest enabled path from an input to each node (Kahn's order). Inputs have depth 0; an output or hidden
    node without enabled inputs has depth 1 (its value is its bias)."""
    nodes = genome.node_ids.tolist()
    depth = {i: 0 for i in INPUT_IDS}
    depth.update({n: 1 for n in nodes})
    succ = {n: [] for n in nodes}
    indegree = {n: 0 for n in nodes}
    on = genome.enabled
    for s, d in zip(genome.src[on].tolist(), genome.dst[on].tolist()):
        if s >= N_IN:                         # hidden source; inputs are already final
            succ[s].append(d)
            indegree[d] += 1
    ready = [n for n in nodes if indegree[n] == 0]
    while ready:
        node = ready.pop()
        for d in succ[node]:
            depth[d] = max(depth[d], depth[node] + 1)
            indegree[d] -= 1
            if indegree[d] == 0:
                ready.append(d)
    return depth
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_neat.py tests/test_paths.py tests/test_checkpoint.py -q`
Expected: all pass.

- [ ] **Step 6: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all pass. `neat` now appears in the viewer's approach buttons before Task 14 gives it a trainer; `tests/test_app.py::test_training_panel_fits_the_window` only draws the panel, so it passes with eight approaches (three rows of three). If any other test fails only because it iterates over `config.APPROACHES` and starts training for each, restrict that loop to `[a for a in config.APPROACHES if a != "neat"]` with the comment `# NEAT gets a trainer in the viewer in Task 14`.

- [ ] **Step 7: Commit**

```bash
git add racing/neat.py racing/config.py racing/checkpoint.py tests/test_neat.py tests/test_paths.py
git commit -q -m "NEAT genomes, run-wide innovation record and genetic operators (revision 3 §5.4)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/neat.py racing/config.py racing/checkpoint.py tests/test_neat.py tests/test_paths.py
```

---

### Task 8: NEAT batched forward pass, Keras export and drivers

**Files:**
- Modify: `racing/neat.py` (append; add imports)
- Modify: `racing/nets.py` (`load_champion`; add `import json`)
- Test: `tests/test_neat.py` (append)

**Interfaces:**
- Consumes: Task 7 (`Genome`, `node_depths`, `add_node`, `add_connection`, `initial_genome`, `Innovations`, ids); Task 2 contract: every population type has `__len__`, `subset(indices)`, `policy()`, `driver(i)`; every driver has `act(obs)`, `policy()`, `n_params`; `population.save_dense_champion` writes `<seed>.keras` and `<seed>.npy` only (no `.json`: a `.json` next to a controller is always a NEAT genome); `nets.dense_spec(model)` and Task 2's `load_champion` (dense `.npy` read in the architecture `dense_spec` gives).
- Produces (used by Tasks 9, 12–14): `neat.Compiled`; `compile_population(genomes) -> Compiled`; `forward(compiled, obs) -> np.ndarray (n, 15)`; `NeatNet(genome)` with `genome`, `outputs(obs)`, `act(obs)`, `policy()`, `n_params`, `NeatNet.from_json(path)`; `NeatPopulation(genomes)` with `genomes`, `__len__`, `subset(indices)`, `policy()`, `driver(i)`; `to_keras(genome) -> keras.Model`; `save_neat_champion(approach, seed, genome, champion_dir=config.CHAMPIONS_DIR) -> Path`; `nets.load_champion` returning a `NeatNet` driver when `<seed>.json` exists (always a NEAT genome), a `DenseNet` in the Keras file's architecture (`dense_spec`) when `<seed>.npy` exists, else `from_keras(model)`.

Design notes for the implementer:
- **Batched forward pass (revision 3 §5.4).** `compile_population` lays every genome's values out in one flat vector: genome *g* occupies a block of `10 + len(node_ids)` slots (its inputs, then its nodes in `node_ids` order). All enabled connections of the whole population are grouped by the depth of their target node. `forward` then runs, per depth, one gather (`v[src] * w`), one `numpy.bincount` onto the depth's nodes and one activation (tanh for hidden nodes, identity for outputs). There is **no Python loop over cars per time step**; the loop is over depths only (typically 1–6).
- **Compile once per task.** `NeatPopulation.policy()` compiles once and returns a closure, so a worker compiles its share of the population once per generation and then only calls `forward` at every step.
- Per-genome sums are accumulated in connection order regardless of which other genomes share the compiled block, so results do not depend on how `split_tasks` divides the population (the "results never depend on n_jobs" constraint).
- `NeatNet` evaluates one genome on a batch of rows with one masked dense matrix per depth (zeros where there is no connection). It drives a car in the evaluation and the viewer exactly like `nets.DenseNet`.
- `to_keras` builds a functional model: per depth, one `Dense` layer for the hidden nodes (tanh) and one for the outputs (linear), each applied to the concatenation of the inputs and all shallower nodes, with zero weights where the genome has no connection; the new nodes are concatenated on; a final non-trained selection `Dense` (one-hot kernel, softmax) puts the outputs in output order. The parameter count reported for NEAT is `genome.n_params`, never `model.count_params()` (Task 12).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_neat.py`:

```python
from racing import nets, sim  # noqa: E402  (appended imports)


def reference_outputs(genome, row):
    """Direct per-genome computation: each node from its enabled incoming connections, recursively."""
    bias = dict(zip(genome.node_ids.tolist(), genome.bias.tolist()))
    incoming = {}
    for s, d, w, on in zip(genome.src.tolist(), genome.dst.tolist(), genome.weight.tolist(), genome.enabled.tolist()):
        if on:
            incoming.setdefault(d, []).append((s, w))
    memo = {}

    def value(node):
        if node < neat.N_IN:
            return row[node]
        if node not in memo:
            total = bias[node] + sum(w * value(s) for s, w in incoming.get(node, []))
            memo[node] = np.tanh(total) if node >= neat.FIRST_HIDDEN else total
        return memo[node]

    return np.array([value(o) for o in neat.OUTPUT_IDS])


def softmax(values):
    e = np.exp(values - values.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.fixture
def genomes():
    rng = np.random.default_rng(20)
    innovations = neat.Innovations()
    return [grown(rng, innovations, rounds=int(r)) for r in rng.integers(0, 30, 12)]


def test_batched_forward_matches_direct_computation(genomes):
    rng = np.random.default_rng(21)
    obs = rng.random((len(genomes), config.N_INPUTS))
    out = neat.forward(neat.compile_population(genomes), obs)
    assert out.shape == (len(genomes), config.N_ACTIONS)
    for i, genome in enumerate(genomes):
        assert np.allclose(out[i], reference_outputs(genome, obs[i]))


def test_batched_forward_does_not_depend_on_the_other_genomes(genomes):
    rng = np.random.default_rng(22)
    obs = rng.random((len(genomes), config.N_INPUTS))
    together = neat.forward(neat.compile_population(genomes), obs)
    for i in (0, 5, 11):
        alone = neat.forward(neat.compile_population([genomes[i]]), obs[i:i + 1])
        assert np.array_equal(alone[0], together[i])


def test_neat_net_matches_the_batched_forward_pass(genomes):
    rng = np.random.default_rng(23)
    rows = rng.random((50, config.N_INPUTS))
    for genome in genomes[:4]:
        net = neat.NeatNet(genome)
        batched = neat.forward(neat.compile_population([genome] * 50), rows)
        assert np.allclose(net.outputs(rows), batched)
        assert np.array_equal(net.act(rows), np.argmax(batched, axis=1))
        assert net.n_params == genome.n_params


def test_population_interface(genomes):
    population = neat.NeatPopulation(genomes)
    assert len(population) == 12
    part = population.subset([3, 1])
    assert len(part) == 2 and part.genomes[0] is genomes[3] and part.genomes[1] is genomes[1]
    obs = np.random.default_rng(24).random((2, config.N_INPUTS))
    classes = part.policy()(obs, None)
    assert classes.shape == (2,) and classes.min() >= 0 and classes.max() < config.N_ACTIONS
    assert classes[0] == population.driver(3).act(obs[:1])[0]


def test_a_neat_population_drives_in_the_simulator(genomes, circle_track):
    result = sim.simulate(circle_track, neat.NeatPopulation(genomes).policy(), len(genomes), laps=1)
    assert len(result.fitness) == len(genomes) and np.all(result.fitness >= 0.0)


def test_keras_export_chooses_the_same_actions(genomes):
    rng = np.random.default_rng(25)
    obs = rng.random((300, config.N_INPUTS)).astype(np.float32)
    for genome in (genomes[0], max(genomes, key=lambda g: len(g.node_ids))):
        model = neat.to_keras(genome)
        probabilities = model.predict(obs, verbose=0)
        expected = softmax(neat.NeatNet(genome).outputs(obs.astype(float)))
        assert np.allclose(probabilities, expected, atol=1e-4)
        assert np.mean(np.argmax(probabilities, axis=1) == neat.NeatNet(genome).act(obs)) >= 0.99


def test_saved_neat_champion_loads_as_its_genome(genomes, tmp_path):
    genome = max(genomes, key=lambda g: len(g.node_ids))
    path = neat.save_neat_champion("neat", 1, genome, tmp_path)
    assert path == tmp_path / "neat" / "seed1.keras" and path.exists()
    assert (tmp_path / "neat" / "seed1.json").exists()
    assert nets.champion_seeds(tmp_path) == {"neat": [1]}
    model, driver = nets.load_champion("neat", 1, tmp_path)
    assert isinstance(driver, neat.NeatNet) and driver.n_params == genome.n_params
    obs = np.random.default_rng(26).random((40, config.N_INPUTS))
    assert np.array_equal(driver.act(obs), neat.NeatNet(genome).act(obs))
    assert np.array_equal(neat.NeatNet.from_json(tmp_path / "neat" / "seed1.json").act(obs), driver.act(obs))


def test_dense_champion_architecture_is_read_from_the_keras_model(tmp_path):
    rng = np.random.default_rng(27)
    flat = nets.init_population(rng, 1, hidden=(8,))[0] + rng.normal(0, 0.2, nets.n_params((8,)))
    folder = tmp_path / "es"
    folder.mkdir()
    nets.to_keras(flat, (8,), "tanh").save(folder / "seed0.keras")
    np.save(folder / "seed0.npy", flat)                     # dense champions write no .json (contract)
    _, driver = nets.load_champion("es", 0, tmp_path)
    obs = rng.random((30, config.N_INPUTS))
    assert isinstance(driver, nets.DenseNet)
    assert np.array_equal(driver.act(obs), nets.from_flat(flat, (8,), "tanh").act(obs))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_neat.py -q`
Expected: the new tests fail with `AttributeError: module 'racing.neat' has no attribute 'forward'` (and `compile_population`, `NeatNet`, `NeatPopulation`, `to_keras`, `save_neat_champion`); `test_dense_champion_architecture_is_read_from_the_keras_model` already passes (Task 2 reads the architecture with `nets.dense_spec`); it pins that behavior through this task's rewrite.

- [ ] **Step 3: Write the implementation**

At the top of `racing/neat.py`, replace the imports with:

```python
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from racing import config, paths
from racing.backend import keras_cpu
```

Append to `racing/neat.py`:

```python
@dataclass(eq=False)
class Compiled:
    """A population of genomes as flat arrays (revision 3 §5.4). Values live in one vector: each genome has a block
    of slots, its 10 inputs and then its nodes. Each level holds one depth: the slots it computes, their biases,
    whether each is hidden (tanh) or an output (identity), and its enabled connections (source slot, position of the
    target within the level, weight)."""
    n: int
    size: int
    inputs: np.ndarray          # (n, 10) input slots
    outputs: np.ndarray         # (n, 15) output slots, in output order
    levels: list                # [(nodes, bias, hidden, src, target, weight)] for depths 1, 2, ...


def compile_population(genomes):
    """Flat arrays for the batched forward pass; compiled once per generation and task."""
    n = len(genomes)
    inputs = np.zeros((n, N_IN), dtype=np.int64)
    outputs = np.zeros((n, N_OUT), dtype=np.int64)
    node_slot, node_depth, node_bias, node_hidden = [], [], [], []
    conn_src, conn_dst, conn_w, conn_depth = [], [], [], []
    offset = 0
    for g, genome in enumerate(genomes):
        ids = genome.node_ids
        depth = node_depths(genome)
        depths = np.array([depth[node] for node in ids.tolist()], dtype=np.int64)
        slots = offset + N_IN + np.arange(len(ids))
        inputs[g] = offset + np.arange(N_IN)
        outputs[g] = slots[np.searchsorted(ids, np.arange(N_IN, N_IN + N_OUT))]
        node_slot.append(slots)
        node_depth.append(depths)
        node_bias.append(genome.bias)
        node_hidden.append(ids >= FIRST_HIDDEN)
        on = genome.enabled
        src, dst = genome.src[on], genome.dst[on]
        target = np.searchsorted(ids, dst)
        conn_src.append(np.where(src < N_IN, offset + src, offset + N_IN + np.searchsorted(ids, src)))
        conn_dst.append(slots[target])
        conn_w.append(genome.weight[on])
        conn_depth.append(depths[target])
        offset += N_IN + len(ids)
    if n == 0:
        return Compiled(0, 0, inputs, outputs, [])
    node_slot, node_depth = np.concatenate(node_slot), np.concatenate(node_depth)
    node_bias, node_hidden = np.concatenate(node_bias), np.concatenate(node_hidden)
    conn_src, conn_dst = np.concatenate(conn_src), np.concatenate(conn_dst)
    conn_w, conn_depth = np.concatenate(conn_w), np.concatenate(conn_depth)
    position = np.zeros(offset, dtype=np.int64)
    levels = []
    for d in range(1, int(node_depth.max()) + 1):
        nodes = node_slot[node_depth == d]
        position[nodes] = np.arange(len(nodes))
        here = conn_depth == d                                  # boolean masks keep each genome's order
        levels.append((nodes, node_bias[node_depth == d], node_hidden[node_depth == d], conn_src[here],
                       position[conn_dst[here]], conn_w[here]))
    return Compiled(n, offset, inputs, outputs, levels)


def forward(compiled, obs):
    """Output values (n, 15) of every genome for its own observation row: one gather, one bincount and one
    activation per depth, with no loop over cars."""
    v = np.zeros(compiled.size)
    v[compiled.inputs] = obs
    for nodes, bias, hidden, src, target, weight in compiled.levels:
        total = np.bincount(target, weights=v[src] * weight, minlength=len(nodes)) + bias
        v[nodes] = np.where(hidden, np.tanh(total), total)
    return v[compiled.outputs]


class NeatNet:
    """One NEAT controller as one masked dense matrix per depth; drives a car like nets.DenseNet."""

    def __init__(self, genome):
        self.genome = genome
        ids = genome.node_ids.tolist()
        column = {i: i for i in INPUT_IDS}
        column.update({node: N_IN + k for k, node in enumerate(ids)})
        depth = node_depths(genome)
        bias = dict(zip(ids, genome.bias.tolist()))
        on = genome.enabled
        conns = list(zip(genome.src[on].tolist(), genome.dst[on].tolist(), genome.weight[on].tolist()))
        self.width = N_IN + len(ids)
        self.output_columns = np.array([column[o] for o in OUTPUT_IDS])
        self.levels = []
        for d in range(1, max(depth[node] for node in ids) + 1):
            nodes = [node for node in ids if depth[node] == d]
            position = {node: j for j, node in enumerate(nodes)}
            matrix = np.zeros((self.width, len(nodes)))
            for s, t, w in conns:
                if t in position:
                    matrix[column[s], position[t]] += w
            self.levels.append((np.array([column[node] for node in nodes]), matrix,
                                np.array([bias[node] for node in nodes]),
                                np.array([node >= FIRST_HIDDEN for node in nodes])))

    @property
    def n_params(self):
        return self.genome.n_params

    def outputs(self, obs):
        obs = np.asarray(obs, dtype=float)
        values = np.zeros((len(obs), self.width))
        values[:, :N_IN] = obs
        for columns, matrix, bias, hidden in self.levels:
            total = values @ matrix + bias
            values[:, columns] = np.where(hidden, np.tanh(total), total)
        return values[:, self.output_columns]

    def act(self, obs):
        return np.argmax(self.outputs(obs), axis=1)

    def policy(self):
        return lambda obs, cars: self.act(obs)

    @classmethod
    def from_json(cls, path):
        return cls(Genome.from_dict(json.loads(Path(path).read_text())))


class NeatPopulation:
    """A NEAT population with the population interface of nets.DensePopulation."""

    def __init__(self, genomes):
        self.genomes = list(genomes)

    def __len__(self):
        return len(self.genomes)

    def subset(self, indices):
        return NeatPopulation([self.genomes[int(i)] for i in np.asarray(indices).ravel()])

    def policy(self):
        """Policy for sim.simulate: car i is driven by genome i. Compiled once, here."""
        compiled = compile_population(self.genomes)
        return lambda obs, cars: np.argmax(forward(compiled, obs), axis=1)

    def driver(self, i):
        return NeatNet(self.genomes[int(i)])


def to_keras(genome):
    """Equivalent Keras model (revision 3 §5.4): per depth, a tanh Dense layer for the hidden nodes and a linear
    Dense layer for the outputs, each applied to the concatenation of the inputs and all shallower nodes, with zero
    weights where the genome has no connection; then a fixed selection layer that puts the outputs in output order,
    with a softmax. Used for the accuracy and timing columns only."""
    keras = keras_cpu()
    depth = node_depths(genome)
    ids = genome.node_ids.tolist()
    bias = dict(zip(ids, genome.bias.tolist()))
    on = genome.enabled
    conns = list(zip(genome.src[on].tolist(), genome.dst[on].tolist(), genome.weight[on].tolist()))
    x = keras.Input((N_IN,))
    tensor, available = x, list(INPUT_IDS)                       # node id of each column of `tensor`
    for d in range(1, max(depth[node] for node in ids) + 1):
        row = {node: i for i, node in enumerate(available)}
        new_tensors, new_ids = [], []
        for hidden in (True, False):
            group = [node for node in ids if depth[node] == d and (node >= FIRST_HIDDEN) == hidden]
            if not group:
                continue
            col = {node: j for j, node in enumerate(group)}
            kernel = np.zeros((len(available), len(group)))
            for s, t, w in conns:
                if t in col:
                    kernel[row[s], col[t]] += w
            layer = keras.layers.Dense(len(group), activation="tanh" if hidden else "linear")
            new_tensors.append(layer(tensor))
            layer.set_weights([kernel, np.array([bias[node] for node in group])])
            new_ids += group
        tensor = keras.layers.Concatenate()([tensor, *new_tensors])
        available += new_ids
    select = np.zeros((len(available), N_OUT))
    index = {node: i for i, node in enumerate(available)}
    for j, node in enumerate(OUTPUT_IDS):
        select[index[node], j] = 1.0
    layer = keras.layers.Dense(N_OUT, activation="softmax", trainable=False)
    probabilities = layer(tensor)
    layer.set_weights([select, np.zeros(N_OUT)])
    return keras.Model(x, probabilities)


def save_neat_champion(approach, seed, genome, champion_dir=config.CHAMPIONS_DIR):
    """Final NEAT controller (revision 3 §5.4): its genome as <seed>.json, which drives, and the equivalent Keras
    model as <seed>.keras. Returns the Keras file's path."""
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_keras(genome).save(path)
    paths.champion_path(approach, seed, ".json", champion_dir).write_text(json.dumps(genome.to_dict()))
    return path
```

In `racing/nets.py`, add `import json` to the imports (if not already there) and replace `load_champion` with:

```python
def load_champion(approach, seed, directory=None):
    """(Keras model, driver) of one final controller. A NEAT controller drives with its genome (<seed>.json,
    revision 3 §5.4); an evolved dense controller with its float64 weights (.npy) and the architecture of its Keras
    file (dense_spec); an imitation controller with the Keras weights (spec §6)."""
    keras_path = paths.champion_path(approach, seed, champion_dir=directory)
    model = keras_cpu().models.load_model(keras_path)
    genome_path = paths.champion_path(approach, seed, ".json", directory)
    flat_path = paths.champion_path(approach, seed, ".npy", directory)
    if genome_path.exists():
        from racing import neat                    # neat imports evolution lazily; nets stays import-light
        driver = neat.NeatNet(neat.Genome.from_dict(json.loads(genome_path.read_text())))
    elif flat_path.exists():
        driver = from_flat(np.load(flat_path), *dense_spec(model))
    else:
        driver = from_keras(model)
    return model, driver
```

(Task 2 already gave `load_champion` its dense behavior (`.npy` with `dense_spec`); this replacement keeps it and adds the genome branch in front; `test_dense_champion_architecture_is_read_from_the_keras_model` pins the dense behavior.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_neat.py tests/test_nets.py -q`
Expected: all pass.

- [ ] **Step 5: Check the forward pass cost**

Run (not a test; a one-off check that the batched pass costs about as much as the physics step):

```bash
~/miniconda3/bin/python - <<'EOF'
import time
import numpy as np
from racing import config, neat
rng = np.random.default_rng(0)
innovations = neat.Innovations()
genomes = []
for _ in range(1000):
    g = neat.initial_genome(rng, innovations)
    for _ in range(int(rng.integers(0, 8))):
        g = neat.add_connection(rng, neat.add_node(rng, g, innovations), innovations)
    genomes.append(g)
t = time.perf_counter(); compiled = neat.compile_population(genomes); c = time.perf_counter() - t
obs = rng.random((1000, config.N_INPUTS))
t = time.perf_counter()
for _ in range(100):
    neat.forward(compiled, obs)
print(f"compile {c:.2f} s, forward {1000 * (time.perf_counter() - t) / 100:.2f} ms per step for 1000 cars")
EOF
```

Expected: compile well under 1 s and forward a few milliseconds per step. Record both numbers in the task report. If forward exceeds 10 ms per step, report it rather than optimizing (revision 3 §12 names a hidden-node cap as the remedy).

- [ ] **Step 6: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add racing/neat.py racing/nets.py tests/test_neat.py
git commit -q -m "NEAT batched forward pass, drivers, Keras export and saved controllers (revision 3 §5.4)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/neat.py racing/nets.py tests/test_neat.py
```

---

### Task 9: NEAT speciation, method and `run_neat`

**Files:**
- Modify: `racing/neat.py` (append; extend imports)
- Test: `tests/test_neat.py` (append)

**Interfaces:**
- Consumes: Tasks 7–8 (`Genome`, `Innovations`, `initial_genome`, `mutate`, `crossover`, `distance`, `NeatPopulation`, `save_neat_champion`); Task 3: `curriculum.StepSize(sigma0)` with `sigma`, `update(parent_success, child_success, success_rate)`, `reset()`, `state()`, `StepSize.from_state(state)`, and `curriculum.success_fraction(child_fitness, parent_fitness)`; Task 2: `population.TrialResult(fitness, success, laps, finish_time)` with `all_success`; Task 6: the `Method` protocol and `evolution.run(method, seed, out_dir, champion_dir=..., n_jobs=1, on_generation=None, start_level=1, max_generations=None, rng_seed=None) -> dict`, which calls `method.initialize(rng)` on a fresh start or `method.load_state(state)` on resumption, evaluates `method.population` with `population.evaluate_population(population, seeds, level, factor, n_jobs=...)`, calls `on_generation(report)` before `method.advance(rng, result, kept, event)`, and checkpoints after `advance`.
- Produces (used by Tasks 12, 14–16): `neat.Species`; `neat.SpeciesSet(threshold=config.NEAT_THRESHOLD0)` with `species`, `threshold`, `speciate(rng, genomes, fitness)`, `allot(n_children, fitness) -> dict[int, int]` (every species id, stagnant ones with 0 unless they hold the best car), `state()`, `SpeciesSet.from_state(state)`; `neat.NEAT(pop_size=config.NEAT_POP)` implementing the `Method` protocol (`approach = "neat"`, `pop_size`, `population`, `is_kept`, `step_size`, `initialize`, `advance`, `state`, `load_state`, `save_champion`, `log_fields` → `{"species", "mean_nodes", "mean_conns"}`) plus `child_success_rate(fitness) -> float`; `neat.run_neat(seed=0, pop_size=config.NEAT_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1, on_generation=None, start_level=1, max_generations=None) -> dict`.

Design notes for the implementer:
- **Order inside `advance`** (revision 3 §5.2, §5.4): (1) step size: `parent_success` = share of the `config.ES_MU` fittest cars that succeeded on every map; `child_success` = share of the previous children (`~is_kept`) that did; `success_rate` = children against their parent (`child_success_rate`); `steps.update(...)`, then `steps.reset()` if `event == "level_up"` (not on `"goal"` or `"tightened"`). (2) `species.speciate(rng, genomes, fitness)`. (3) `allot(pop_size - len(kept), fitness)`. (4) per species in list order, per allotted child: tournament parent `a`; draw `rng.random() < NEAT_CROSSOVER`; with a second tournament parent `b` (only if the species has more than one member) cross `fitter, other` (fitter = higher fitness, `a` on ties), else copy `a`; then `mutate` with the updated σ. (5) new population = kept genomes (in `kept` order) followed by the children; `is_kept`, `ids`, `parent_ids` follow.
- **Unique ids** identify cars across generations: kept cars keep their id, children get new ids, each child records the id of its fitter (or only) parent. `child_success_rate` compares each child's fitness with its parent's fitness in the same generation if the parent is still in the population (kept cars are re-evaluated on the same maps), using `success_fraction` (a tie counts half); children whose parent was dropped are left out; `nan` if none remain (the step size then stays unchanged, Task 3).
- **Speciation** (revision 3 §5.4): a genome joins the first species whose representative is within the threshold (`distance < threshold`), otherwise founds a species (representative: itself until the end of this call). Empty species are removed. A species' `stale` counter resets when its best fitness strictly improves, otherwise grows by one. The threshold moves by `NEAT_THRESHOLD_STEP` towards `NEAT_TARGET_SPECIES` species, never below one step. Finally each species' representative becomes a random member of this generation (used by the next call).
- **Allotment**: species with `stale >= NEAT_STAGNATION` get 0 children unless they contain the index of the population's best fitness; the others share `n_children` in proportion to their members' mean fitness (equal shares if all means are 0), rounded by the largest-remainder method (ties to the earlier species), so the counts always sum to `n_children`.
- **Skip when complete**: `evolution.run` itself returns the existing summary when `checkpoint.completed(out_dir)` matches the current settings, so `run_neat` needs nothing beyond calling `evolution.run`.
- **Exact resumption**: `state()` holds the genomes (as dicts), `is_kept`, `ids`, `parent_ids`, `next_id`, the innovation record, the species set (threshold, next id, each species' id, representative, members, best fitness, stale) and the step size. After `load_state(state)` and with the same random generator state, the next `advance` produces bit-identical genomes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_neat.py`:

```python
import pickle  # noqa: E402

from racing import evolution, population  # noqa: E402


def result_for(fitness, success, maps=1):
    """A TrialResult with the same fitness and success on every map."""
    success = np.repeat(np.asarray(success, dtype=bool)[None, :], maps, axis=0)
    return population.TrialResult(fitness=np.asarray(fitness, dtype=float), success=success,
                                  laps=np.where(success, config.LAPS, 1),
                                  finish_time=np.where(success, 60.0, np.nan))


def clusters(rng):
    innovations = neat.Innovations()
    near = [neat.initial_genome(rng, innovations) for _ in range(6)]     # pairwise δ ≈ 0.4 · 1.13
    far = []
    for genome in near[:4]:
        shifted = genome.copy()
        shifted.weight = shifted.weight + 20.0                          # δ to the near ones ≈ 8
        far.append(shifted)
    return near + far


def test_speciation_groups_genomes_by_distance():
    rng = np.random.default_rng(30)
    genomes = clusters(rng)
    species = neat.SpeciesSet(threshold=3.0)
    species.speciate(rng, genomes, np.ones(len(genomes)))
    assert sorted(sorted(s.members) for s in species.species) == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]]
    assert np.isclose(species.threshold, 3.0 - config.NEAT_THRESHOLD_STEP)          # 2 species < 20: tighten
    for s in species.species:
        assert any(np.array_equal(s.representative.weight, genomes[m].weight) for m in s.members)


def test_threshold_never_drops_below_one_step():
    rng = np.random.default_rng(31)
    species = neat.SpeciesSet(threshold=0.15)
    genomes = clusters(rng)
    for _ in range(5):
        species.speciate(rng, genomes, np.ones(len(genomes)))
    assert species.threshold >= config.NEAT_THRESHOLD_STEP - 1e-12


def test_allotment_is_proportional_with_largest_remainder():
    rng = np.random.default_rng(32)
    genomes = clusters(rng)
    fitness = np.array([3.0] * 6 + [1.0] * 4)
    species = neat.SpeciesSet(threshold=3.0)
    species.speciate(rng, genomes, fitness)
    near = next(s.id for s in species.species if 0 in s.members)
    far = next(s.id for s in species.species if 6 in s.members)
    assert species.allot(7, fitness) == {near: 5, far: 2}           # quotas 5.25 and 1.75
    assert sum(species.allot(13, fitness).values()) == 13


def test_stagnant_species_get_no_children_unless_they_hold_the_best():
    rng = np.random.default_rng(33)
    genomes = clusters(rng)
    fitness = np.array([1.0] * 6 + [0.5] * 4)
    species = neat.SpeciesSet(threshold=3.0)
    for _ in range(config.NEAT_STAGNATION + 1):                     # constant fitness: no improvement after the first
        species.speciate(rng, genomes, fitness)
    near = next(s for s in species.species if 0 in s.members)
    far = next(s for s in species.species if 6 in s.members)
    assert near.stale >= config.NEAT_STAGNATION and far.stale >= config.NEAT_STAGNATION
    assert species.allot(10, fitness) == {near.id: 10, far.id: 0}   # near holds the best car (index 0)


def test_species_set_round_trips_through_its_state():
    rng = np.random.default_rng(34)
    species = neat.SpeciesSet()
    species.speciate(rng, clusters(rng), np.arange(10.0))
    copy = neat.SpeciesSet.from_state(pickle.loads(pickle.dumps(species.state())))
    assert copy.state()["threshold"] == species.threshold
    for a, b in zip(species.species, copy.species):
        assert (a.id, a.members, a.best_fitness, a.stale) == (b.id, b.members, b.best_fitness, b.stale)
        assert np.array_equal(a.representative.innov, b.representative.innov)
        assert np.array_equal(a.representative.weight, b.representative.weight)


def test_advance_keeps_survivors_first_and_fills_the_population():
    rng = np.random.default_rng(35)
    method = neat.NEAT(pop_size=30)
    method.initialize(rng)
    assert method.approach == "neat" and len(method.population) == 30 and not method.is_kept.any()
    assert method.step_size == config.NEAT_SIGMA0
    fitness = rng.random(30)
    kept = np.array([3, 7, 11])
    before = [method.population.genomes[i] for i in kept]
    method.advance(rng, result_for(fitness, fitness > 0.5), kept, None)
    assert len(method.population) == 30
    assert method.is_kept.tolist() == [True] * 3 + [False] * 27
    for k, genome in enumerate(before):
        assert np.array_equal(method.population.genomes[k].weight, genome.weight)
    assert method.log_fields()["species"] >= 1
    assert set(method.log_fields()) == {"species", "mean_nodes", "mean_conns"}


def test_child_success_rate_compares_children_with_their_parents():
    method = neat.NEAT(pop_size=4)
    method.ids = np.array([10, 11, 12, 13])
    method.parent_ids = np.array([-1, -1, 10, 99])                  # car 3's parent is gone
    method.is_kept = np.array([True, True, False, False])
    assert method.child_success_rate(np.array([1.0, 2.0, 3.0, 0.0])) == 1.0       # 3 > 1; car 3 left out
    assert method.child_success_rate(np.array([3.0, 2.0, 3.0, 0.0])) == 0.5       # a tie counts half
    method.parent_ids = np.array([-1, -1, 98, 99])
    assert np.isnan(method.child_success_rate(np.ones(4)))


def test_step_size_resets_on_level_up_only():
    rng = np.random.default_rng(36)
    method = neat.NEAT(pop_size=20)
    method.initialize(rng)
    method.steps.sigma = 0.01
    fitness = rng.random(20)
    method.advance(rng, result_for(fitness, np.zeros(20)), np.array([0]), "tightened")
    assert method.step_size != config.NEAT_SIGMA0
    method.advance(rng, result_for(fitness, np.zeros(20)), np.array([0]), "level_up")
    assert method.step_size == config.NEAT_SIGMA0


def test_state_round_trip_continues_identically():
    rng_a = np.random.default_rng(37)
    a = neat.NEAT(pop_size=30)
    a.initialize(rng_a)
    scores = np.random.default_rng(99)
    first = scores.random(30)
    a.advance(rng_a, result_for(first, first > 0.4), np.array([0, 1, 2]), None)
    b = neat.NEAT(pop_size=30)
    b.load_state(pickle.loads(pickle.dumps(a.state())))
    rng_b = np.random.default_rng()
    rng_b.bit_generator.state = rng_a.bit_generator.state
    second = scores.random(30)
    for method, rng in ((a, rng_a), (b, rng_b)):
        method.advance(rng, result_for(second, second > 0.3), np.array([5, 2]), None)
    for ga, gb in zip(a.population.genomes, b.population.genomes):
        for name in neat.Genome.FIELDS:
            assert np.array_equal(getattr(ga, name), getattr(gb, name))
    assert a.innovations.state() == b.innovations.state()
    assert a.step_size == b.step_size and np.array_equal(a.ids, b.ids)


def fake_evaluation(pop, seeds, level, factor, n_jobs=1, backend="loky"):
    """Deterministic stand-in for the simulator: a genome's score follows its weights and biases; each map (seed)
    has its own success threshold."""
    score = np.array([np.tanh(g.weight[g.enabled].sum() / 10.0 + g.bias.sum()) for g in pop.genomes])
    thresholds = np.array([((s % 5) - 2) / 10.0 for s in seeds])
    success = score[None, :] > thresholds[:, None]
    fitness = np.where(success, 1.5 + 0.4 * score[None, :], 0.5 + 0.4 * score[None, :]).mean(axis=0)
    return population.TrialResult(fitness=fitness, success=success, laps=np.where(success, config.LAPS, 1),
                                  finish_time=np.where(success, 60.0, np.nan))


class Stop(Exception):
    pass


def test_run_neat_reports_and_resumes_exactly(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "evaluate_population", fake_evaluation)
    monkeypatch.setattr(evolution, "evaluate_population", fake_evaluation, raising=False)

    def record(into, stop_at=None):
        def on_generation(report):
            into.append((report["generation"], report["best"], report["mean"], report["worst"],
                         report["step_size"], report["min_share"], report["species"]))
            if report["generation"] == stop_at:
                raise Stop()
        return on_generation

    full = []
    summary = neat.run_neat(seed=3, pop_size=40, out_dir=tmp_path / "full", champion_dir=None,
                            on_generation=record(full), max_generations=6)
    assert summary["approach"] == "neat" and summary["generations"] == 6 and summary["pop_size"] == 40
    assert [row[0] for row in full] == list(range(6))
    first = []
    with pytest.raises(Stop):
        neat.run_neat(seed=3, pop_size=40, out_dir=tmp_path / "cut", champion_dir=None,
                      on_generation=record(first, stop_at=3), max_generations=6)
    resumed = []
    neat.run_neat(seed=3, pop_size=40, out_dir=tmp_path / "cut", champion_dir=None,
                  on_generation=record(resumed), max_generations=6)
    assert first[:3] == full[:3]
    assert resumed == full[3:]                                  # generation 3 re-run from the checkpoint


@pytest.mark.slow
def test_run_neat_drives_easy_maps_and_saves_its_controller(tmp_path):
    summary = neat.run_neat(seed=0, pop_size=24, out_dir=tmp_path / "run", champion_dir=tmp_path / "champions",
                            max_generations=2)
    assert summary["generations"] == 2 and summary["level_reached"] == 1
    model, driver = nets.load_champion("neat", 0, tmp_path / "champions")
    assert isinstance(driver, neat.NeatNet)
    obs = np.random.default_rng(38).random((20, config.N_INPUTS)).astype(np.float32)
    assert np.mean(np.argmax(model.predict(obs, verbose=0), axis=1) == driver.act(obs)) >= 0.99
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_neat.py -q`
Expected: the new tests fail with `AttributeError: module 'racing.neat' has no attribute 'SpeciesSet'` (and `NEAT`, `run_neat`).

- [ ] **Step 3: Write the implementation**

In `racing/neat.py`, extend the imports:

```python
from dataclasses import dataclass, field
```

and below `from racing.backend import keras_cpu` add:

```python
from racing.curriculum import StepSize, success_fraction
```

Append to `racing/neat.py`:

```python
@dataclass(eq=False)
class Species:
    """Genomes of one niche: its representative (from the previous generation), members (indices into the current
    population), best fitness so far and generations since that last improved."""
    id: int
    representative: Genome
    members: list = field(default_factory=list)
    best_fitness: float = float("-inf")
    stale: int = 0


class SpeciesSet:
    """Speciation by compatibility distance with a threshold steered towards NEAT_TARGET_SPECIES species, and the
    allotment of children to species (revision 3 §5.4)."""

    def __init__(self, threshold=config.NEAT_THRESHOLD0):
        self.threshold = float(threshold)
        self.species = []
        self.next_id = 0

    def speciate(self, rng, genomes, fitness):
        for s in self.species:
            s.members = []
        for i, genome in enumerate(genomes):
            for s in self.species:
                if distance(genome, s.representative) < self.threshold:
                    s.members.append(i)
                    break
            else:
                self.species.append(Species(self.next_id, genome, [i]))
                self.next_id += 1
        self.species = [s for s in self.species if s.members]
        for s in self.species:
            best = float(np.max(fitness[s.members]))
            if best > s.best_fitness:
                s.best_fitness, s.stale = best, 0
            else:
                s.stale += 1
        if len(self.species) > config.NEAT_TARGET_SPECIES:
            self.threshold += config.NEAT_THRESHOLD_STEP
        elif len(self.species) < config.NEAT_TARGET_SPECIES:
            self.threshold = max(self.threshold - config.NEAT_THRESHOLD_STEP, config.NEAT_THRESHOLD_STEP)
        for s in self.species:
            s.representative = genomes[s.members[int(rng.integers(len(s.members)))]]

    def allot(self, n_children, fitness):
        """Children per species id: proportional to mean member fitness, largest remainder; a stagnant species gets
        none unless it holds the population's best car."""
        best = int(np.argmax(fitness))
        eligible = [s.stale < config.NEAT_STAGNATION or best in s.members for s in self.species]
        means = np.array([max(float(np.mean(fitness[s.members])), 0.0) if ok else 0.0
                          for s, ok in zip(self.species, eligible)])
        if means.sum() <= 0.0:
            means = np.array(eligible, dtype=float)
        quotas = n_children * means / means.sum()
        counts = np.floor(quotas).astype(np.int64)
        extra = np.argsort(-(quotas - counts), kind="stable")[:n_children - int(counts.sum())]
        counts[extra] += 1
        return {s.id: int(c) for s, c in zip(self.species, counts)}

    def state(self):
        return {"threshold": self.threshold, "next_id": self.next_id,
                "species": [{"id": s.id, "representative": s.representative.to_dict(), "members": list(s.members),
                             "best_fitness": s.best_fitness, "stale": s.stale} for s in self.species]}

    @classmethod
    def from_state(cls, state):
        species = cls(state["threshold"])
        species.next_id = state["next_id"]
        species.species = [Species(s["id"], Genome.from_dict(s["representative"]), list(s["members"]),
                                   s["best_fitness"], s["stale"]) for s in state["species"]]
        return species


class NEAT:
    """NEAT as a method of the evolution engine (revision 3 §5.4): kept cars first, then children produced within
    species, with the two-phase step size of the ES."""
    approach = "neat"

    def __init__(self, pop_size=config.NEAT_POP):
        self.pop_size = int(pop_size)
        self.innovations = Innovations()
        self.species = SpeciesSet()
        self.steps = StepSize(config.NEAT_SIGMA0)
        self.population = None
        self.is_kept = None
        self.ids = None
        self.parent_ids = None
        self.next_id = 0

    @property
    def step_size(self):
        return self.steps.sigma

    def initialize(self, rng):
        self.population = NeatPopulation([initial_genome(rng, self.innovations) for _ in range(self.pop_size)])
        self.is_kept = np.zeros(self.pop_size, dtype=bool)
        self.ids = np.arange(self.pop_size, dtype=np.int64)
        self.parent_ids = np.full(self.pop_size, -1, dtype=np.int64)
        self.next_id = self.pop_size

    def child_success_rate(self, fitness):
        """Share of the previous children that beat their parent in this generation (a tie counts half); children
        whose parent is no longer in the population are left out; nan if none remain."""
        position = {int(car): k for k, car in enumerate(self.ids)}
        child, parent = [], []
        for k in np.flatnonzero(~self.is_kept):
            p = int(self.parent_ids[k])
            if p in position:
                child.append(k)
                parent.append(position[p])
        if not child:
            return float("nan")
        return success_fraction(fitness[np.array(child)], fitness[np.array(parent)])

    def _tournament(self, rng, members, fitness):
        entrants = members[rng.integers(0, len(members), config.NEAT_TOURNAMENT)]
        return int(entrants[np.argmax(fitness[entrants])])

    def advance(self, rng, result, kept, event):
        fitness, success = result.fitness, result.all_success
        genomes = self.population.genomes
        top = np.argsort(-fitness, kind="stable")[:config.ES_MU]
        children = ~self.is_kept
        child_success = float(np.mean(success[children])) if children.any() else float("nan")
        self.steps.update(float(np.mean(success[top])), child_success, self.child_success_rate(fitness))
        if event == "level_up":
            self.steps.reset()
        self.species.speciate(rng, genomes, fitness)
        allotment = self.species.allot(self.pop_size - len(kept), fitness)
        new_genomes, new_parents = [], []
        for s in self.species.species:
            members = np.asarray(s.members)
            for _ in range(allotment[s.id]):
                a = self._tournament(rng, members, fitness)
                cross = rng.random() < config.NEAT_CROSSOVER
                if cross and len(members) > 1:
                    b = self._tournament(rng, members, fitness)
                    fitter, other = (a, b) if fitness[a] >= fitness[b] else (b, a)
                    child = crossover(rng, genomes[fitter], genomes[other])
                else:
                    fitter, child = a, genomes[a]
                new_genomes.append(mutate(rng, child, self.innovations, self.steps.sigma))
                new_parents.append(int(self.ids[fitter]))
        kept = np.asarray(kept, dtype=np.int64)
        self.population = NeatPopulation([genomes[i] for i in kept] + new_genomes)
        self.is_kept = np.arange(len(self.population)) < len(kept)
        new_ids = np.arange(self.next_id, self.next_id + len(new_genomes), dtype=np.int64)
        self.next_id += len(new_genomes)
        self.ids = np.concatenate([self.ids[kept], new_ids])
        self.parent_ids = np.concatenate([np.full(len(kept), -1, dtype=np.int64),
                                          np.asarray(new_parents, dtype=np.int64)])

    def state(self):
        return {"pop_size": self.pop_size, "genomes": [g.to_dict() for g in self.population.genomes],
                "is_kept": self.is_kept.copy(), "ids": self.ids.copy(), "parent_ids": self.parent_ids.copy(),
                "next_id": self.next_id, "innovations": self.innovations.state(), "species": self.species.state(),
                "steps": self.steps.state()}

    def load_state(self, state):
        self.pop_size = int(state["pop_size"])
        self.population = NeatPopulation([Genome.from_dict(g) for g in state["genomes"]])
        self.is_kept = np.asarray(state["is_kept"], dtype=bool).copy()
        self.ids = np.asarray(state["ids"], dtype=np.int64).copy()
        self.parent_ids = np.asarray(state["parent_ids"], dtype=np.int64).copy()
        self.next_id = int(state["next_id"])
        self.innovations = Innovations.from_state(state["innovations"])
        self.species = SpeciesSet.from_state(state["species"])
        self.steps = StepSize.from_state(state["steps"])

    def save_champion(self, index, seed, champion_dir):
        return save_neat_champion(self.approach, seed, self.population.genomes[int(index)], champion_dir)

    def log_fields(self):
        genomes = self.population.genomes
        return {"species": len(self.species.species),
                "mean_nodes": float(np.mean([np.sum(g.node_ids >= FIRST_HIDDEN) for g in genomes])),
                "mean_conns": float(np.mean([np.sum(g.enabled) for g in genomes]))}


def run_neat(seed=0, pop_size=config.NEAT_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1,
             on_generation=None, start_level=1, max_generations=None):
    """One NEAT run with the evolution engine (revision 3 §4, §5.4): resumable, written to runs/neat/seed<k>."""
    from racing import evolution                      # evolution imports nets, which imports neat lazily
    out_dir = Path(out_dir or paths.run_dir("neat", seed))
    return evolution.run(NEAT(pop_size), seed, out_dir, champion_dir=champion_dir, n_jobs=n_jobs,
                         on_generation=on_generation, start_level=start_level, max_generations=max_generations)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_neat.py -q`
Expected: all pass (`summary["pop_size"]` is the length of the initial population, which Task 6's `evolution.run` records).

- [ ] **Step 5: Run the slow NEAT test**

Run: `~/miniconda3/bin/python -m pytest tests/test_neat.py -q -m slow`
Expected: 1 passed (two generations of 24 NEAT cars on fresh Easy maps, champion saved and reloaded).

- [ ] **Step 6: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add racing/neat.py tests/test_neat.py
git commit -q -m "NEAT speciation, method for the evolution engine and run_neat (revision 3 §5.4)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/neat.py tests/test_neat.py
```

---

### Task 10: Imitation dataset on fresh maps, stage files, skip-when-complete

Revision 3 §6.1–6.2, §6.5, §8.2. The imitation dataset is collected on 121 freshly generated maps (seeds drawn once from `config.DATASET_SEED_RANGE`) instead of the fixed revision 2.0 tracks. Trained imitation networks are saved as their *imitation stage* (`<run>/imitation.keras`), never as final controllers: the final controller of an imitation method comes from its speed push (Task 11). A complete dataset, training run or Keras Tuner search recorded with the current settings is returned instead of being repeated, so the runner can resume (revision 3 §8.2).

**Files:**
- Modify: `racing/config.py` (seed section)
- Modify: `racing/expert.py` (`dataset_seeds`, `default_splits`, `collect_dataset`)
- Modify: `racing/paths.py` (`STAGES`, `stage_path`, `speed_dir`)
- Modify: `racing/checkpoint.py` (`SETTINGS_KEYS`: the dataset names)
- Modify: `racing/imitation.py` (`STAGE_FILE`, `previous_run`, `finish_run`, `start_run`, `train_baseline`, `retrain`, `run_tuner`)
- Modify: `racing/app.py` (`trainer_main`: the two imitation calls)
- Modify: `tests/test_paths.py`, `tests/test_expert.py`, `tests/test_imitation.py`, `tests/test_app.py`
- Modify if they still pass `champion_dir` to imitation functions: `scripts/pilots.py`, `tests/test_pilots.py`, `tests/test_end_to_end.py`

**Interfaces:**
- Consumes (Task 4): `checkpoint.settings_record() -> dict`, `checkpoint.completed(run_dir, name="summary.json") -> dict | None` (raises `checkpoint.SettingsMismatch` when the recorded settings differ), `checkpoint.SettingsMismatch`.
- Consumes (Tasks 1 and 3): generated maps for every `(seed, level)` through `tracks.get_track` (Task 1); `config.TRAIN_SEED_START` (Task 3).
- Produces:
  - `config.DATASET_SEED_RANGE = (1_000_000, 2_000_000)`, `config.DATASET_TRACKS = {1: 1, 2: 20, 3: 20, 4: 80}`; `config.TRAIN_DATA_TRACKS` and `config.TRAIN_SEEDS` no longer exist.
  - `expert.dataset_seeds() -> list[tuple[int, int]]`, `expert.default_splits() -> dict`, `expert.collect_dataset(path=None, splits=None, n_jobs=1, backend="loky", resume=True)`; `<stem>_info.json` gains `"items"` and `"settings"`.
  - `paths.STAGES = ("imitation", "dagger")`, `paths.stage_path(approach, seed, stage, runs_dir=None) -> Path`, `paths.speed_dir(approach, seed, runs_dir=None) -> Path`.
  - `imitation.STAGE_FILE = "imitation.keras"`; `imitation.train_baseline(data, out_dir=None, on_epoch=None, epochs=config.MAX_EPOCHS, seed=0, resume=True) -> dict`; `imitation.retrain(approach, data, seed, search=None, out_dir=None, epochs=config.MAX_EPOCHS, resume=True) -> dict`; `imitation.run_tuner(..., export_models=True, resume=True) -> dict`. The `champion_dir` parameter of `train_baseline`, `retrain`, `finish_run` and `start_run` is removed.
  - Run summaries (`<run>/summary.json`) gain `"stage_path"`, `"learning_rate"` (the rate the network was trained with: `config.LEARNING_RATE` for the baseline, the chosen `learning_rate` for a retrained search) and `"settings"`. Search summaries gain `"settings"`.

- [ ] **Step 1: Write the failing tests for the new paths**

Add to `tests/test_paths.py` (add `import pytest` to its imports):

```python
def test_stage_paths_and_speed_directory(tmp_path):
    assert paths.stage_path("baseline", 1, "imitation", tmp_path) == tmp_path / "baseline" / "seed1" / "imitation.keras"
    assert paths.stage_path("kt_random", 0, "dagger", tmp_path) == tmp_path / "kt_random" / "seed0" / "dagger.keras"
    assert paths.speed_dir("kt_bayesian", 2, tmp_path) == tmp_path / "kt_bayesian" / "seed2" / "speed"


def test_stage_paths_follow_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    assert paths.stage_path("baseline", 0, "dagger") == tmp_path / "runs" / "baseline" / "seed0" / "dagger.keras"


def test_unknown_stage_is_refused(tmp_path):
    with pytest.raises(ValueError, match="stage"):
        paths.stage_path("baseline", 0, "final", tmp_path)
```

- [ ] **Step 2: Write the failing tests for the dataset**

In `tests/test_expert.py`, add `checkpoint` to the import line (`from racing import checkpoint, config, expert, sim, tracks`) and add:

```python
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
    monkeypatch.setattr(checkpoint, "settings_record", lambda: {"TARGET_FACTOR": 9.9})
    with pytest.raises(checkpoint.SettingsMismatch):
        expert.collect_dataset(tmp_path / "d.npz", splits=SMALL_SPLITS)
```

Replace the slow test `test_expert_drives_every_seed` (whatever its current body) with:

```python
@pytest.mark.slow
def test_expert_drives_every_dataset_and_held_out_map():
    for seed, level in expert.dataset_seeds():
        tracks.get_track(seed, level)
    for seed in list(config.VAL_SEEDS) + list(config.TEST_SEEDS):
        tracks.get_track(seed, config.MAX_LEVEL)
```

- [ ] **Step 3: Write the failing tests for the imitation stage and skipping**

In `tests/test_imitation.py`, change the import line to `from racing import checkpoint, config, imitation, nets, paths`. Replace `test_train_baseline_learns_reports_and_saves`, `test_train_baseline_default_directory_is_per_seed`, `test_interrupted_training_leaves_no_stale_results`, `test_train_baseline_is_reproducible_with_a_seed` and `test_retrain_builds_the_chosen_architecture` with the following, and add the remaining new tests:

```python
def test_train_baseline_learns_reports_and_saves_its_stage(toy_data, tmp_path):
    seen = []
    summary = imitation.train_baseline(toy_data, out_dir=tmp_path / "run",
                                       on_epoch=lambda epoch, model, acc: seen.append((epoch, acc)), epochs=5,
                                       seed=2)
    stage = tmp_path / "run" / "imitation.keras"
    assert stage.exists() and summary["stage_path"] == str(stage)
    assert len(seen) == summary["epochs"] and summary["val_accuracy"] > 0.2 and summary["seed"] == 2
    assert summary["learning_rate"] == config.LEARNING_RATE
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["params"] == 431
    assert checkpoint.completed(tmp_path / "run")["val_accuracy"] == summary["val_accuracy"]


def test_train_baseline_saves_its_stage_per_seed_and_no_final_controller(monkeypatch, toy_data, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "CHAMPIONS_DIR", tmp_path / "champions")
    imitation.train_baseline(toy_data, epochs=1, seed=1)
    assert paths.stage_path("baseline", 1, "imitation").exists()
    assert (tmp_path / "runs" / "baseline" / "seed1" / "summary.json").exists()
    assert not (tmp_path / "champions").exists()


def test_interrupted_training_leaves_no_stale_results(monkeypatch, toy_data, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text("{}")
    (run / "imitation.keras").write_bytes(b"stale")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(imitation, "fit", interrupted)
    with pytest.raises(KeyboardInterrupt):
        imitation.train_baseline(toy_data, out_dir=run, epochs=1, resume=False)
    assert not (run / "summary.json").exists() and not (run / "imitation.keras").exists()


def test_train_baseline_is_reproducible_with_a_seed(toy_data, tmp_path):
    runs = [imitation.train_baseline(toy_data, out_dir=tmp_path / f"run{i}", epochs=3, seed=3) for i in range(2)]
    assert runs[0]["val_accuracy"] == runs[1]["val_accuracy"]


def test_complete_training_is_not_repeated(monkeypatch, toy_data, tmp_path):
    first = imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0)
    monkeypatch.setattr(imitation, "fit", lambda *args, **kwargs: pytest.fail("trained again"))
    again = imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0)
    assert again == json.loads(json.dumps(first))


def test_training_without_resume_or_without_its_stage_file_trains_again(monkeypatch, toy_data, tmp_path):
    imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0)
    real_fit, calls = imitation.fit, []
    monkeypatch.setattr(imitation, "fit", lambda *args, **kwargs: calls.append(1) or real_fit(*args, **kwargs))
    imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0, resume=False)
    (tmp_path / "run" / "imitation.keras").unlink()
    imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0)
    assert len(calls) == 2


def test_training_recorded_with_other_settings_is_refused(monkeypatch, toy_data, tmp_path):
    imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0)
    monkeypatch.setattr(checkpoint, "settings_record", lambda: {"MAX_EPOCHS": 1})
    with pytest.raises(checkpoint.SettingsMismatch):
        imitation.train_baseline(toy_data, out_dir=tmp_path / "run", epochs=1, seed=0)


def test_retrain_builds_the_chosen_architecture(toy_data, tmp_path):
    search = tmp_path / "search"
    search.mkdir()
    chosen = {"layers": 2, "units_0": 8, "units_1": 4, "activation": "tanh", "dropout": 0.0, "learning_rate": 0.003}
    (search / "summary.json").write_text(json.dumps({"chosen_hyperparameters": chosen}))
    summary = imitation.retrain("kt_random", toy_data, seed=1, search=search, out_dir=tmp_path / "run", epochs=2)
    assert summary["params"] == nets.n_params((8, 4)) and summary["seed"] == 1 and summary["wall_time"] > 0
    assert summary["learning_rate"] == 0.003 and (tmp_path / "run" / "imitation.keras").exists()


def test_a_complete_search_is_not_run_again(monkeypatch, toy_data, tmp_path):
    out = tmp_path / "kt_random"
    out.mkdir()
    done = {"approach": "kt_random", "wall_time": 12.5, "settings": checkpoint.settings_record()}
    (out / "summary.json").write_text(json.dumps(done))
    monkeypatch.setattr(imitation, "make_tuner", lambda *args, **kwargs: pytest.fail("searched again"))
    assert imitation.run_tuner("kt_random", toy_data, out_dir=out)["wall_time"] == 12.5
```

In the slow test `test_trial_epochs_are_the_epochs_trained`, add as its last line:

```python
    assert checkpoint.completed(tmp_path / approach)["trials"] == summary["trials"]
```

In `tests/test_app.py`, add after `test_trainer_main_uses_the_reported_dataset`:

```python
def test_trainer_main_trains_imitation_afresh_and_saves_no_final_controller(monkeypatch, tmp_path):
    from racing import imitation
    fake_imitation(monkeypatch, tmp_path)
    options = []
    monkeypatch.setattr(imitation, "train_baseline", lambda data, **kwargs: options.append(kwargs))
    run_baseline_trainer()
    assert options[0]["resume"] is False and "champion_dir" not in options[0]
    assert options[0]["out_dir"] == tmp_path / "app_runs" / "baseline" / "seed0"
```

- [ ] **Step 4: Run the new tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_paths.py tests/test_expert.py tests/test_imitation.py tests/test_app.py -k "stage or dataset_maps or default_splits or complete or collect_again or other_settings or retrain or search_is_not or afresh or reproducible or interrupted or learns_reports"`
Expected: FAIL — `AttributeError: module 'racing.paths' has no attribute 'stage_path'`, `AttributeError: module 'racing.expert' has no attribute 'dataset_seeds'`, `TypeError: train_baseline() got an unexpected keyword argument 'resume'`, and `KeyError: 'resume'` in the app test.

- [ ] **Step 5: Update the seed section of `racing/config.py`**

Replace the lines defining `TRAIN_SEEDS` and `TRAIN_DATA_TRACKS` (delete `TRAIN_SEEDS` too if an earlier task left it) so that the seed section reads:

```python
# --- Seeds (revision 3 §3.4) ---
VAL_SEEDS = range(80, 100)
TEST_SEEDS = range(1000, 1020)
DATASET_SEED_RANGE = (1_000_000, 2_000_000)   # the imitation dataset's maps are drawn once from this range
DATASET_TRACKS = {1: 1, 2: 20, 3: 20, 4: 80}  # dataset maps per level (revision 3 §6.1)
TRAIN_SEED_START = 2_000_000                  # evolutionary and DAgger maps are drawn from here on (revision 3 §3.4)
RUN_SEEDS = (0, 1, 2)        # every method is run once per seed (spec §11)
SEARCH_SEED = 0              # each Keras Tuner search runs once (spec §8.2)
```

(If Task 3 already defined `TRAIN_SEED_START` elsewhere in the file, keep a single definition here.) Keep `DATASET_SALT = 99` in the expert section.

The dataset settings change results, so add `"DATASET_SEED_RANGE"`, `"DATASET_TRACKS"` and `"DATASET_SALT"` (each only
if missing) to the `SETTINGS_KEYS` tuple in `racing/checkpoint.py`, and add to `tests/test_expert.py`:

```python
def test_dataset_settings_are_recorded():
    assert {"DATASET_SEED_RANGE", "DATASET_TRACKS", "DATASET_SALT"} <= set(checkpoint.settings_record())
```

Run: `grep -rn "TRAIN_DATA_TRACKS\|TRAIN_SEEDS" racing scripts tests`
Expected: no output.

- [ ] **Step 6: Add the stage paths to `racing/paths.py`**

Change the module docstring's first line to `"""Where each run, stage and final controller is stored (spec §3.3, §9.1; revision 3 §6.5)."""` and add after `search_dir`:

```python
STAGES = ("imitation", "dagger")


def stage_path(approach, seed, stage, runs_dir=None):
    """An imitation network after one stage of revision 3 §6.5: <runs>/<approach>/seed<k>/<stage>.keras."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of: {', '.join(STAGES)}")
    return run_dir(approach, seed, runs_dir) / f"{stage}.keras"


def speed_dir(approach, seed, runs_dir=None):
    """The speed push of one imitation network (revision 3 §6.4): <runs>/<approach>/seed<k>/speed."""
    return run_dir(approach, seed, runs_dir) / "speed"
```

- [ ] **Step 7: Collect the dataset on fresh maps in `racing/expert.py`**

Change the import line to `from racing import checkpoint, config, sim`. Replace `default_splits` with:

```python
def dataset_seeds():
    """(seed, level) of the imitation dataset's maps (revision 3 §6.1): config.DATASET_TRACKS maps per level, with
    distinct seeds drawn once from config.DATASET_SEED_RANGE, so every collection uses the same maps."""
    low, high = config.DATASET_SEED_RANGE
    total = sum(config.DATASET_TRACKS.values())
    seeds = low + np.random.default_rng(config.DATASET_SALT).choice(high - low, total, replace=False)
    items, position = [], 0
    for level, count in sorted(config.DATASET_TRACKS.items()):
        items += [(int(seed), level) for seed in seeds[position:position + count]]
        position += count
    return items


def default_splits():
    return {"train": dataset_seeds(),
            "val": [(seed, config.MAX_LEVEL) for seed in config.VAL_SEEDS],
            "test": [(seed, config.MAX_LEVEL) for seed in config.TEST_SEEDS]}
```

Replace `collect_dataset` with:

```python
def _write_json(path, data):
    """Write JSON so that an interruption cannot leave a partial file behind."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2))
    os.replace(temporary, path)


def collect_dataset(path=None, splits=None, n_jobs=1, backend="loky", resume=True):
    """Collect the train/val/test rows (spec §7.3) and save them as a compressed .npz file. Tracks are distributed
    across worker processes; each has its own random number generator, so the rows do not depend on n_jobs.
    Returns the arrays and a summary per split (tracks, rows, respawns after crashes) and records the collection
    time, the maps and the settings in <stem>_info.json next to the dataset. With resume, a complete dataset of the
    same maps recorded with the current settings is loaded instead of collected again (revision 3 §8.2)."""
    path = Path(config.DATASET_PATH if path is None else path)
    splits = default_splits() if splits is None else splits
    items = {name: [[int(seed), int(level)] for seed, level in entries] for name, entries in splits.items()}
    info_path = path.with_name(f"{path.stem}_info.json")
    if resume and path.exists():
        info = checkpoint.completed(path.parent, name=info_path.name)
        if info is not None and info.get("items") == items:
            with np.load(path) as stored:
                return {key: stored[key] for key in stored.files}, info["splits"]
    start = time.time()
    data, summary = {}, {}
    for name, entries in splits.items():
        if n_jobs == 1:
            parts = [_collect_track(seed, level) for seed, level in entries]
        else:
            from joblib import Parallel, delayed
            parts = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_collect_track)(seed, level) for seed, level in entries)
        data[f"X_{name}"] = np.concatenate([p[0] for p in parts]).astype(np.float32)
        data[f"y_{name}"] = np.concatenate([p[1] for p in parts]).astype(np.int64)
        summary[name] = {"tracks": len(entries), "rows": len(data[f"y_{name}"]),
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
    _write_json(info_path, {"wall_time": wall_time, "n_jobs": n_jobs, "splits": summary, "items": items,
                            "settings": checkpoint.settings_record()})
    return data, summary
```

Update the module docstring's first line to `"""Scripted expert driver (spec §7.1) and imitation dataset collection on freshly generated maps (spec §7.2-7.3, revision 3 §6.1)."""`.

- [ ] **Step 8: Save imitation stages and skip complete runs in `racing/imitation.py`**

Change the import line to `from racing import checkpoint, config, nets, paths`. Replace `finish_run`, `start_run` and `train_baseline` with:

```python
STAGE_FILE = "imitation.keras"     # the trained network in its run directory (revision 3 §6.5)


def finish_run(approach, seed, model, history, wall_time, out_dir, extra=None):
    """Save the trained network as the imitation stage, then write the run's summary. The summary records the
    settings and is written last, so its presence marks a complete run (revision 3 §8.2). The final controller of an
    imitation method comes from its speed push (revision 3 §6.4), not from here."""
    stage = Path(out_dir) / STAGE_FILE
    model.save(stage)
    summary = {"approach": approach, "seed": seed, "epochs": len(history.history["val_accuracy"]),
               "val_accuracy": float(max(history.history["val_accuracy"])), "params": int(model.count_params()),
               "wall_time": wall_time, "stage_path": str(stage), **(extra or {}),
               "settings": checkpoint.settings_record()}
    (Path(out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def start_run(out_dir):
    """Remove this run's earlier summary and stage file, so an interrupted run leaves neither behind."""
    out_dir = Path(out_dir)
    paths.remove(out_dir / "summary.json", out_dir / STAGE_FILE)
    out_dir.mkdir(parents=True, exist_ok=True)


def previous_run(out_dir, resume):
    """The summary of a complete earlier run in out_dir with the current settings, or None. A summary recorded with
    other settings raises checkpoint.SettingsMismatch (revision 3 §8.2)."""
    if not resume:
        return None
    summary = checkpoint.completed(out_dir)
    if summary is not None and (Path(out_dir) / STAGE_FILE).exists():
        return summary
    return None


def train_baseline(data, out_dir=None, on_epoch=None, epochs=config.MAX_EPOCHS, seed=0, resume=True):
    out_dir = Path(out_dir or paths.run_dir("baseline", seed))
    done = previous_run(out_dir, resume)
    if done is not None:
        return done
    start_run(out_dir)
    keras_cpu().utils.set_random_seed(seed)
    model = build_baseline()
    history, wall_time = fit(model, data, epochs, on_epoch)
    return finish_run("baseline", seed, model, history, wall_time, out_dir,
                      extra={"learning_rate": config.LEARNING_RATE})
```

In `run_tuner`, add the parameter `resume=True` after `export_models=True`, extend the docstring with the sentence `With resume, a complete search recorded with the current settings is returned instead of run again (revision 3 §8.2).`, insert directly after `out_dir = Path(out_dir or paths.search_dir(approach))`:

```python
    if resume:
        done = checkpoint.completed(out_dir)
        if done is not None:
            return done
```

and add `"settings": checkpoint.settings_record()` as the last entry of its `summary` dict.

Replace `retrain` with:

```python
def retrain(approach, data, seed, search=None, out_dir=None, epochs=config.MAX_EPOCHS, resume=True):
    """Train the architecture selected by a search from scratch with `seed` (spec §8.3)."""
    search = Path(search or paths.search_dir(approach))
    out_dir = Path(out_dir or paths.run_dir(approach, seed))
    done = previous_run(out_dir, resume)
    if done is not None:
        return done
    chosen = json.loads((search / "summary.json").read_text())["chosen_hyperparameters"]
    start_run(out_dir)
    keras_cpu().utils.set_random_seed(seed)
    model = build_network(architecture(chosen), chosen["activation"], float(chosen["dropout"]),
                          float(chosen["learning_rate"]))
    history, wall_time = fit(model, data, epochs)
    return finish_run(approach, seed, model, history, wall_time, out_dir,
                      extra={"hyperparameters": chosen, "learning_rate": float(chosen["learning_rate"])})
```

Change the module docstring to `"""Imitation-learning controllers (spec §8): a baseline network, three Keras Tuner searches, and retraining of the
selected architectures with each run seed. Each trained network is saved as its imitation stage (revision 3 §6.5)."""`.

- [ ] **Step 9: Train imitation afresh in the viewer (`racing/app.py`)**

In `trainer_main`, replace the two imitation calls with:

```python
    if approach == "baseline":
        imitation.train_baseline(data, out_dir=out_dir, seed=seed, resume=False,
                                 on_epoch=lambda epoch, model, acc: add(epoch, model, acc))
    else:
        imitation.run_tuner(approach, data, out_dir=out_dir, on_trial=add, seed=seed, export_models=False,
                            resume=False)
```

(The viewer always shows training as it happens, so it never returns an earlier run.)

- [ ] **Step 10: Remove `champion_dir` from the remaining imitation calls**

Run: `grep -n "train_baseline\|imitation.retrain" scripts/pilots.py tests/test_pilots.py tests/test_end_to_end.py`
For every call of `imitation.train_baseline(...)` or `imitation.retrain(...)` listed that passes `champion_dir=...`, delete that keyword argument (in `scripts/pilots.py` the call becomes `imitation.train_baseline(imitation.load_dataset(dataset), out_dir=folder / "baseline", seed=0)`). If `tests/test_pilots.py` asserts `trained[0]["champion_dir"] == tmp_path / "seeded" / "champions"`, change that part of the assertion to `trained[0]["out_dir"] == tmp_path / "seeded" / "baseline"`.
Imitation training no longer saves a final controller, so the slow `tests/test_end_to_end.py` now stops at `es.trained_baseline(1, champion_dir=champions)`; Task 11 replaces that call with the DAgger stage and the speed push. Do not run the slow end-to-end test in this task.
Run: `grep -rn "champion_dir" racing/imitation.py`
Expected: no output.

- [ ] **Step 11: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_paths.py tests/test_expert.py tests/test_imitation.py tests/test_app.py tests/test_pilots.py`
Expected: PASS.

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: PASS (fast suite).

- [ ] **Step 12: Commit**

```bash
git add racing/config.py racing/checkpoint.py racing/expert.py racing/paths.py racing/imitation.py racing/app.py tests/test_paths.py tests/test_expert.py tests/test_imitation.py tests/test_app.py
git add scripts/pilots.py tests/test_pilots.py tests/test_end_to_end.py   # only the files Step 10 changed
git commit -m "Imitation dataset on fresh maps; imitation stage files; complete runs are not repeated

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/config.py racing/checkpoint.py racing/expert.py racing/paths.py racing/imitation.py racing/app.py tests/test_paths.py tests/test_expert.py tests/test_imitation.py tests/test_app.py scripts/pilots.py tests/test_pilots.py tests/test_end_to_end.py
```

(Drop from both lists any of the last three paths that Step 10 left unchanged.)

---

### Task 11: DAgger refinement and the imitation speed push

Revision 3 §6.3–6.5, §8.2–8.3. `racing/dagger.py` refines each trained imitation network in rounds: the car drives `DAGGER_MAPS` fresh Extra hard maps (respawned after a crash), the expert labels every state the car reached, the rows join the network's dataset (the original training rows plus every round's corrections), training continues from the current weights, and a time trial on `DAGGER_MAPS` further fresh maps checks the result. The goal is `DAGGER_GOAL` successes in `DAGGER_STREAK` consecutive rounds; after `DAGGER_PATIENCE` rounds without a new best check result, the best network so far is kept. The refined network becomes the `dagger` stage, from which an ES speed push (Task 6's `es.run_es`) starts at Extra hard and produces the method's final controller.

Every round is checkpointed. The checkpoint holds the weights, the random generator state, the used seeds and the number of correction rows; each round starts from a freshly loaded network after seeding Keras for that round, so a refinement interrupted at any point and resumed ends exactly like an uninterrupted one.

**Files:**
- Modify: `racing/config.py` (DAgger section)
- Modify: `racing/checkpoint.py` (`SETTINGS_KEYS`: every `DAGGER_*` name not listed yet, `DAGGER_SALT` included)
- Create: `racing/dagger.py`
- Test: `tests/test_dagger.py`
- Modify: `tests/test_end_to_end.py` (slow; the imitation champions now come from DAgger and the speed push, and the seeded ES starts from the DAgger stage)

**Interfaces:**
- Consumes:
  - Task 2: `sim.target_time(track, factor, laps=config.LAPS)`, `sim.time_limit_for(track, laps)`; `nets.from_keras(model) -> DenseNet` (`.act(obs)`, `.policy()`, `.layers`), `nets.to_flat(model)`, `nets.dense_spec(model) -> (hidden tuple, activation)`.
  - Task 3: `curriculum.SeedSource(rng, start=config.TRAIN_SEED_START, used=())` with `.draw(count) -> list[int]` and `.state() -> list[int]`.
  - Task 4: `checkpoint.settings_record()`, `checkpoint.check_settings(recorded)`, `checkpoint.save(path, state)`, `checkpoint.load(path)`, `checkpoint.completed(run_dir, name)`, `checkpoint.SettingsMismatch`, `checkpoint.SETTINGS_KEYS`.
  - Task 5: `progress.write_status(status)`, `progress.milestone(message)` (default paths `config.STATUS_PATH`, `config.RUN_LOG_PATH`, resolved when called).
  - Task 6: `es.run_es(seed, pop_size, out_dir, champion_dir, n_jobs, on_generation, start_level, max_generations, initial, approach, hidden, activation) -> dict` (resumable; returns the existing summary of a complete run).
  - Task 10: `paths.stage_path`, `paths.speed_dir`, `paths.run_dir`; the run summary's `"learning_rate"`; `imitation.compile_model(model, learning_rate)`, `imitation.early_stopping()`.
  - Existing: `expert.expert_controls`, `expert.respawn`, `actions.class_to_controls`, `actions.controls_to_class`, `tracks.get_track`.
- Produces:
  - `config.DAGGER_MAPS = 20`, `DAGGER_EPOCHS = 30`, `DAGGER_HOLDOUT = 0.1`, `DAGGER_GOAL = 18`, `DAGGER_STREAK = 3`, `DAGGER_PATIENCE = 10`, `DAGGER_SALT = 31`.
  - `dagger.drive_and_label(driver, seed, level=config.MAX_LEVEL) -> (X float32 (m, 10), y int64 (m,), respawns int)`
  - `dagger.collect(driver, seeds, level=config.MAX_LEVEL, n_jobs=1) -> (X, y, respawns)` (rows in seed order)
  - `dagger.check(driver, seeds, factor=config.TARGET_FACTOR, level=config.MAX_LEVEL, n_jobs=1) -> int`
  - `dagger.refine(approach, seed, data, runs_dir=None, n_jobs=1, on_round=None, max_rounds=None) -> dict`; files in `paths.run_dir(approach, seed, runs_dir)`: `dagger_state.pkl` (removed when complete), `dagger_rows.npz` (`X`, `y`: correction rows), `dagger.keras` (= `paths.stage_path(..., "dagger")`), `dagger.json` (summary: `approach, seed, settings, rounds, goal_reached, outcome` (`"goal"`, `"stalled"` or `"max_rounds"`), `best_checks, final_checks, correction_rows, respawns, wall_time, stage_path, history` (one record per round: `round, rows, respawns, epochs, checks, streak, elapsed`)).
  - `dagger.speed_push(approach, seed, runs_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1, on_generation=None, max_generations=None) -> dict`
  - `dagger.run(approach, seed, data, runs_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1) -> {"approach": str, "seed": int, "dagger": dict, "speed": dict}`
  - Status written after each round: `{"approach", "seed", "stage": "dagger", "level": config.MAX_LEVEL, "round", "checks", "streak", "elapsed"}`; milestones when a refinement ends.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dagger.py`:

```python
import json

import numpy as np
import pytest

from racing import checkpoint, config, dagger, es, expert, imitation, nets, paths, progress, sim
from racing.actions import controls_to_class
from racing.backend import keras_cpu

FULL_RIGHT = int(controls_to_class(-1.0, 1.0)[0])     # full right lock at full throttle: leaves the circle


class Constant:
    """A driver that always takes the same action."""

    def __init__(self, action):
        self.action = action

    def act(self, obs):
        return np.full(len(obs), self.action, dtype=np.int64)

    def policy(self):
        return lambda obs, cars: self.act(obs)


class Expert:
    """The expert as a driver of one map."""

    def __init__(self, track):
        self.track = track

    def policy(self):
        return expert.expert_policy(self.track)


@pytest.fixture(autouse=True)
def private_results(monkeypatch, tmp_path):
    """Status and milestones go to the test's own folder, never to results/."""
    monkeypatch.setattr(config, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(config, "RUN_LOG_PATH", tmp_path / "run_log.txt")


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(10, 15))

    def make(n):
        x = rng.random((n, 10)).astype(np.float32)
        return x, np.argmax(x @ w, axis=1).astype(np.int64)

    (x_tr, y_tr), (x_va, y_va) = make(2000), make(300)
    return {"X_train": x_tr, "y_train": y_tr, "X_val": x_va, "y_val": y_va}


@pytest.fixture
def trained_runs(tmp_path, toy_data):
    """A runs folder with a baseline imitation stage for seed 0."""
    runs = tmp_path / "runs"
    imitation.train_baseline(toy_data, out_dir=paths.run_dir("baseline", 0, runs), epochs=1, seed=0)
    return runs


def scripted(monkeypatch, checks):
    """collect returns seven rows that depend only on the maps; check returns `checks` in order and records the
    drivers it was given. Returns (maps collected per round, drivers checked)."""
    collected, drivers, queue = [], [], list(checks)

    def collect(driver, seeds, level=config.MAX_LEVEL, n_jobs=1):
        collected.append(list(seeds))
        rng = np.random.default_rng(seeds[0])
        return rng.random((7, config.N_INPUTS)).astype(np.float32), rng.integers(0, config.N_ACTIONS, 7), 1

    def check(driver, seeds, **kwargs):
        drivers.append(driver)
        return queue.pop(0)

    monkeypatch.setattr(dagger, "collect", collect)
    monkeypatch.setattr(dagger, "check", check)
    monkeypatch.setattr(config, "DAGGER_EPOCHS", 1)
    return collected, drivers


def test_dagger_parameters_are_recorded_settings():
    assert (config.DAGGER_MAPS, config.DAGGER_EPOCHS, config.DAGGER_HOLDOUT, config.DAGGER_GOAL,
            config.DAGGER_STREAK, config.DAGGER_PATIENCE) == (20, 30, 0.1, 18, 3, 10)
    assert {"DAGGER_MAPS", "DAGGER_EPOCHS", "DAGGER_HOLDOUT", "DAGGER_GOAL", "DAGGER_STREAK",
            "DAGGER_PATIENCE", "DAGGER_SALT"} <= set(checkpoint.SETTINGS_KEYS)


def test_labels_are_the_expert_action_in_the_states_the_car_reaches(monkeypatch, circle_track):
    monkeypatch.setattr(dagger, "get_track", lambda seed, level: circle_track)
    real_step, expected, driven = sim.step, [], []

    def step(track, cars, steer, throttle, time_limit):
        expected.append(int(controls_to_class(*expert.expert_controls(track, cars))[0]))
        driven.append((float(steer[0]), float(throttle[0])))
        real_step(track, cars, steer, throttle, time_limit)

    monkeypatch.setattr(sim, "step", step)
    x, y, respawns = dagger.drive_and_label(Constant(FULL_RIGHT), seed=5)
    assert list(y) == expected and len(x) == len(y) > 0
    assert x.dtype == np.float32 and x.shape[1] == config.N_INPUTS and y.dtype == np.int64
    assert set(driven) == {(-1.0, 1.0)}                 # the car drives itself; the expert only labels
    assert respawns >= 1                               # after a crash the car is put back and continues


def test_collect_keeps_the_rows_of_every_map_in_order(monkeypatch, circle_track):
    monkeypatch.setattr(dagger, "get_track", lambda seed, level: circle_track)
    one = dagger.drive_and_label(Constant(FULL_RIGHT), seed=1)
    x, y, respawns = dagger.collect(Constant(FULL_RIGHT), [1, 2])
    assert len(y) == 2 * len(one[1]) and np.array_equal(x[:len(one[0])], one[0]) and respawns == 2 * one[2]


def test_check_counts_time_trials_completed_within_the_target(monkeypatch, circle_track):
    monkeypatch.setattr(dagger, "get_track", lambda seed, level: circle_track)
    assert dagger.check(Constant(FULL_RIGHT), [1, 2]) == 0
    assert dagger.check(Expert(circle_track), [1, 2]) == 2                # 3 laps well within 1.3 × 3 × 20 s
    assert dagger.check(Expert(circle_track), [1], factor=0.3) == 0       # 18 s for three laps: impossible


def test_rounds_reach_the_goal_and_rows_accumulate(monkeypatch, toy_data, trained_runs):
    collected, _ = scripted(monkeypatch, [10, 18, 19, 20])
    sizes, real_train = [], dagger._train_round
    monkeypatch.setattr(dagger, "_train_round", lambda model, x, y, rng: sizes.append(len(y)) or real_train(model, x, y, rng))
    summary = dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs)
    assert summary["outcome"] == "goal" and summary["goal_reached"] and summary["rounds"] == 4
    assert [r["streak"] for r in summary["history"]] == [0, 1, 2, 3]
    assert summary["correction_rows"] == 28 and summary["final_checks"] == 20 and summary["best_checks"] == 20
    assert sizes == [len(toy_data["y_train"]) + 7 * k for k in (1, 2, 3, 4)]   # original rows plus every round
    run = paths.run_dir("baseline", 0, trained_runs)
    with np.load(run / "dagger_rows.npz") as rows:
        assert len(rows["y"]) == 28
    assert paths.stage_path("baseline", 0, "dagger", trained_runs).exists()
    assert not (run / "dagger_state.pkl").exists()
    maps = [seed for seeds in collected for seed in seeds]
    assert len(set(maps)) == len(maps) == 4 * config.DAGGER_MAPS and min(maps) >= config.TRAIN_SEED_START
    status = progress.read_status()
    assert status["stage"] == "dagger" and status["round"] == 4 and status["checks"] == 20
    assert "DAgger goal" in config.RUN_LOG_PATH.read_text()


def test_a_stalled_refinement_keeps_the_best_network(monkeypatch, toy_data, trained_runs):
    monkeypatch.setattr(config, "DAGGER_PATIENCE", 3)
    _, drivers = scripted(monkeypatch, [5, 12, 9, 11, 10])
    summary = dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs)
    assert summary["outcome"] == "stalled" and not summary["goal_reached"] and summary["rounds"] == 5
    assert summary["best_checks"] == 12 and summary["final_checks"] == 10
    model = keras_cpu().models.load_model(paths.stage_path("baseline", 0, "dagger", trained_runs), compile=False)
    for (w, b), (best_w, best_b) in zip(nets.from_keras(model).layers, drivers[1].layers):
        assert np.allclose(w, best_w) and np.allclose(b, best_b)


def test_a_complete_refinement_is_not_repeated(monkeypatch, toy_data, trained_runs):
    scripted(monkeypatch, [18, 18, 18])
    first = dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs)
    monkeypatch.setattr(dagger, "collect", lambda *args, **kwargs: pytest.fail("refined again"))
    assert dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs) == json.loads(json.dumps(first))


def test_a_round_limit_ends_the_refinement_with_the_best_network(monkeypatch, toy_data, trained_runs):
    scripted(monkeypatch, [4, 6])
    summary = dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs, max_rounds=2)
    assert summary["outcome"] == "max_rounds" and summary["rounds"] == 2 and summary["best_checks"] == 6


def test_a_refinement_checkpoint_with_other_settings_is_refused(toy_data, trained_runs):
    run = paths.run_dir("baseline", 0, trained_runs)
    checkpoint.save(run / "dagger_state.pkl", {"settings": {"TARGET_FACTOR": 9.9}})
    with pytest.raises(checkpoint.SettingsMismatch):
        dagger.refine("baseline", 0, toy_data, runs_dir=trained_runs)


def test_refinement_needs_the_imitation_stage(toy_data, tmp_path):
    with pytest.raises(FileNotFoundError, match="notebook 02"):
        dagger.refine("kt_random", 1, toy_data, runs_dir=tmp_path / "runs")


@pytest.mark.slow
def test_an_interrupted_refinement_resumes_to_the_same_result(monkeypatch, toy_data, tmp_path):
    """Rows and checks depend only on the maps, so both refinements see the same data; the interrupted one stops
    inside its second round, after that round's rows were saved but before its checkpoint."""
    monkeypatch.setattr(config, "DAGGER_EPOCHS", 2)

    def collect(driver, seeds, level=config.MAX_LEVEL, n_jobs=1):
        rng = np.random.default_rng(seeds[0])
        return rng.random((40, config.N_INPUTS)).astype(np.float32), rng.integers(0, config.N_ACTIONS, 40), 0

    monkeypatch.setattr(dagger, "collect", collect)
    monkeypatch.setattr(dagger, "check", lambda driver, seeds, **kwargs: 10 + seeds[0] % 7)
    results, real_train = {}, dagger._train_round
    for name in ("straight", "interrupted"):
        runs = tmp_path / name
        imitation.train_baseline(toy_data, out_dir=paths.run_dir("baseline", 0, runs), epochs=1, seed=0)
        if name == "interrupted":
            calls = []

            def flaky(*args):
                calls.append(1)
                if len(calls) == 2:
                    raise KeyboardInterrupt
                return real_train(*args)

            monkeypatch.setattr(dagger, "_train_round", flaky)
            with pytest.raises(KeyboardInterrupt):
                dagger.refine("baseline", 0, toy_data, runs_dir=runs, max_rounds=3)
            monkeypatch.setattr(dagger, "_train_round", real_train)
        results[name] = dagger.refine("baseline", 0, toy_data, runs_dir=runs, max_rounds=3)
    straight, resumed = results["straight"], results["interrupted"]
    assert [r["checks"] for r in straight["history"]] == [r["checks"] for r in resumed["history"]]
    assert straight["correction_rows"] == resumed["correction_rows"] == 120
    load = keras_cpu().models.load_model
    a = load(paths.stage_path("baseline", 0, "dagger", tmp_path / "straight"), compile=False).get_weights()
    b = load(paths.stage_path("baseline", 0, "dagger", tmp_path / "interrupted"), compile=False).get_weights()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_speed_push_grows_an_es_population_from_the_refined_network(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    model = imitation.build_network([8, 4], "tanh", 0.1, 0.003)
    stage = paths.stage_path("kt_random", 2, "dagger", runs)
    stage.parent.mkdir(parents=True)
    model.save(stage)
    calls = {}
    monkeypatch.setattr(es, "run_es", lambda **kwargs: calls.update(kwargs) or {"approach": kwargs["approach"]})
    result = dagger.speed_push("kt_random", 2, runs_dir=runs, champion_dir=tmp_path / "champions", max_generations=3)
    assert result == {"approach": "kt_random"}
    assert calls["approach"] == "kt_random" and calls["seed"] == 2 and calls["start_level"] == config.MAX_LEVEL
    assert tuple(calls["hidden"]) == (8, 4) and calls["activation"] == "tanh" and calls["max_generations"] == 3
    assert calls["out_dir"] == paths.speed_dir("kt_random", 2, runs)
    assert calls["champion_dir"] == tmp_path / "champions"
    assert np.array_equal(calls["initial"], nets.to_flat(model))


def test_run_refines_then_pushes(monkeypatch):
    order = []
    monkeypatch.setattr(dagger, "refine", lambda approach, seed, data, **kwargs: order.append("refine") or {"r": 1})
    monkeypatch.setattr(dagger, "speed_push", lambda approach, seed, **kwargs: order.append("speed") or {"s": 2})
    assert dagger.run("baseline", 1, {}) == {"approach": "baseline", "seed": 1, "dagger": {"r": 1}, "speed": {"s": 2}}
    assert order == ["refine", "speed"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_dagger.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'racing.dagger'` (collection error).

- [ ] **Step 3: Add the DAgger parameters to `racing/config.py`**

Task 5 created the `# --- DAgger (revision 3 §6.3) ---` block with `DAGGER_MAPS` and `DAGGER_STREAK`. Do not
define those two again; replace the whole block so that it reads:

```python
# --- DAgger (revision 3 §6.3) ---
DAGGER_MAPS = 20             # fresh Extra hard maps driven per round, and as many more for the round's check
DAGGER_EPOCHS = 30           # at most, per round
DAGGER_HOLDOUT = 0.1         # share of the aggregated rows held out for early stopping
DAGGER_GOAL = 18             # successful checks out of DAGGER_MAPS ...
DAGGER_STREAK = 3            # ... in this many consecutive rounds
DAGGER_PATIENCE = 10         # rounds without a new best check result before the refinement stops
DAGGER_SALT = 31             # a refinement's random generator: default_rng([run seed, method index, DAGGER_SALT])
```

Then check that each name appears exactly once: `grep -c "^DAGGER_" racing/config.py` prints `7`.

Run: `grep -n "DAGGER" racing/checkpoint.py`
If the output does not list all seven names `DAGGER_MAPS`, `DAGGER_EPOCHS`, `DAGGER_HOLDOUT`, `DAGGER_GOAL`, `DAGGER_STREAK`, `DAGGER_PATIENCE`, `DAGGER_SALT` (the salt chooses the refinement's maps, so it changes results), add the missing ones as strings to the `SETTINGS_KEYS` tuple in `racing/checkpoint.py`, after the NEAT names.

- [ ] **Step 4: Create `racing/dagger.py`**

```python
"""DAgger refinement of the imitation controllers and their speed push (revision 3 §6.3-6.5).

A refinement runs in rounds. The car drives fresh Extra hard maps; at every step the expert labels the car's exact
state; the rows join the network's dataset and training continues; a time trial on further fresh maps checks the
result. Every round is checkpointed, and every round starts from a freshly loaded network after seeding Keras for
that round, so an interrupted refinement resumes where it stopped and ends exactly like an uninterrupted one."""
import json
import os
import time
from pathlib import Path

import numpy as np

from racing import checkpoint, config, es, expert, imitation, nets, paths, progress, sim
from racing.actions import class_to_controls, controls_to_class
from racing.backend import keras_cpu
from racing.curriculum import SeedSource
from racing.tracks import get_track

STATE_NAME = "dagger_state.pkl"
ROWS_NAME = "dagger_rows.npz"
SUMMARY_NAME = "dagger.json"


def drive_and_label(driver, seed, level=config.MAX_LEVEL):
    """One car driven by `driver` for config.LAPS laps of one map, with the expert's action recorded in every state
    the car reaches (revision 3 §6.3 steps 1-2). A crash puts the car back on the centerline, as during dataset
    collection, so the whole run is covered. Returns inputs, labels and the number of respawns."""
    track = get_track(seed, level)
    cars = sim.start_cars(track, 1, config.LAPS)
    time_limit = sim.time_limit_for(track, config.LAPS)
    inputs, labels, respawns = [], [], 0
    while not cars.done[0]:
        obs = sim.sense(track, cars)
        inputs.append(obs[0])
        labels.append(int(controls_to_class(*expert.expert_controls(track, cars))[0]))
        steer, throttle = class_to_controls(driver.act(obs))
        sim.step(track, cars, steer, throttle, time_limit)
        if cars.crashed[0] and cars.time < time_limit - config.TIME_EPS:
            respawns += expert.respawn(track, cars)
    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.int64), respawns


def _per_map(function, items, n_jobs):
    """function(*item) for every item, in order; maps are distributed across worker processes."""
    if n_jobs == 1 or len(items) <= 1:
        return [function(*item) for item in items]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(function)(*item) for item in items)


def collect(driver, seeds, level=config.MAX_LEVEL, n_jobs=1):
    """Labelled rows of every map in `seeds`, in seed order, so the rows do not depend on n_jobs."""
    parts = _per_map(drive_and_label, [(driver, seed, level) for seed in seeds], n_jobs)
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            int(sum(p[2] for p in parts)))


def _time_trial(driver, seed, level, factor):
    track = get_track(seed, level)
    result = sim.simulate(track, driver.policy(), 1, time_limit=sim.target_time(track, factor), laps=config.LAPS)
    return bool(result.finished[0])


def check(driver, seeds, factor=config.TARGET_FACTOR, level=config.MAX_LEVEL, n_jobs=1):
    """Number of maps on which the car completes the time trial: config.LAPS laps within the target time
    (revision 3 §4.1, §6.3 step 4)."""
    return int(sum(_per_map(_time_trial, [(driver, seed, level, factor) for seed in seeds], n_jobs)))


def _rng_seed(approach, seed):
    """Every method and run seed draws its own maps."""
    return [seed, config.IMITATION_APPROACHES.index(approach), config.DAGGER_SALT]


def _round_model(stage_file, weights, rate, seed, number):
    """The network with `weights`, freshly loaded and compiled after seeding Keras for round `number`: a round starts
    from the same state whether or not the refinement was interrupted before it."""
    keras = keras_cpu()
    keras.utils.set_random_seed(seed * 1000 + number)
    model = keras.models.load_model(stage_file, compile=False)
    model.set_weights(weights)
    return imitation.compile_model(model, rate)


def _train_round(model, x, y, rng):
    """Continue training on the aggregated rows (revision 3 §6.3 step 3): at most config.DAGGER_EPOCHS epochs with
    early stopping on a config.DAGGER_HOLDOUT hold-out. Returns the epochs trained."""
    order = rng.permutation(len(y))
    held = int(round(config.DAGGER_HOLDOUT * len(y)))
    val, train = order[:held], order[held:]
    history = model.fit(x[train], y[train], validation_data=(x[val], y[val]), epochs=config.DAGGER_EPOCHS,
                        batch_size=config.BATCH_SIZE, callbacks=[imitation.early_stopping()], verbose=0)
    return len(history.history["val_accuracy"])


def _load_rows(path, count):
    """The first `count` correction rows in `path`. Rows of a round interrupted before its checkpoint are dropped."""
    if count == 0 or not path.exists():
        return np.zeros((0, config.N_INPUTS), dtype=np.float32), np.zeros(0, dtype=np.int64)
    with np.load(path) as stored:
        return stored["X"][:count], stored["y"][:count]


def _save_rows(path, x, y):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as f:
        np.savez(f, X=x, y=y)
    os.replace(temporary, path)


def _save_model(model, path):
    """Save a Keras model so that an interruption cannot leave a partial file behind."""
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.keras")
    model.save(temporary)
    os.replace(temporary, path)


def _write_json(path, data):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2))
    os.replace(temporary, path)


def _finished_message(label, seed, state):
    rounds, best = state["round"], state["best_checks"]
    if state["outcome"] == "goal":
        return (f"{label} seed {seed}: DAgger goal reached in round {rounds} "
                f"({state['history'][-1]['checks']}/{config.DAGGER_MAPS} checks)")
    if state["outcome"] == "stalled":
        return (f"{label} seed {seed}: DAgger stopped after {rounds} rounds without a new best "
                f"(best {best}/{config.DAGGER_MAPS} checks); the speed push starts from the best network")
    return f"{label} seed {seed}: DAgger stopped at the round limit ({rounds} rounds, best {best}/{config.DAGGER_MAPS})"


def refine(approach, seed, data, runs_dir=None, n_jobs=1, on_round=None, max_rounds=None):
    """DAgger refinement of one trained imitation network (revision 3 §6.3). Returns the summary of a complete
    refinement recorded with the current settings, resumes an interrupted one, and otherwise starts from the
    imitation stage. Writes the refined network as the 'dagger' stage: the network of the final round when the goal
    is reached, otherwise the network with the best check result."""
    run = paths.run_dir(approach, seed, runs_dir)
    done = checkpoint.completed(run, name=SUMMARY_NAME)
    if done is not None:
        return done
    label = config.LABELS[approach]
    imitation_stage = paths.stage_path(approach, seed, "imitation", runs_dir)
    if not imitation_stage.exists():
        raise FileNotFoundError(f"{imitation_stage} not found: train the {label} network first (notebook 02)")
    state_path, rows_path = run / STATE_NAME, run / ROWS_NAME
    dagger_stage = paths.stage_path(approach, seed, "dagger", runs_dir)
    rate = float(json.loads((run / "summary.json").read_text()).get("learning_rate", config.LEARNING_RATE))
    state = checkpoint.load(state_path)
    if state is None:
        paths.remove(rows_path, dagger_stage)
        initial = keras_cpu().models.load_model(imitation_stage, compile=False).get_weights()
        state = {"settings": checkpoint.settings_record(), "round": 0, "streak": 0, "best_checks": -1,
                 "since_best": 0, "n_rows": 0, "respawns": 0,
                 "rng": np.random.default_rng(_rng_seed(approach, seed)).bit_generator.state, "used": [],
                 "weights": initial, "best_weights": initial, "history": [], "elapsed": 0.0, "outcome": None}
    else:
        checkpoint.check_settings(state["settings"])
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]
    source = SeedSource(rng, used=state["used"])
    x_rows, y_rows = _load_rows(rows_path, state["n_rows"])
    while state["outcome"] is None:
        if max_rounds is not None and state["round"] >= max_rounds:
            state["outcome"] = "max_rounds"
            break
        began, number = time.time(), state["round"] + 1
        model = _round_model(imitation_stage, state["weights"], rate, seed, number)
        x_new, y_new, respawns = collect(nets.from_keras(model), source.draw(config.DAGGER_MAPS), n_jobs=n_jobs)
        x_rows, y_rows = np.concatenate([x_rows, x_new]), np.concatenate([y_rows, y_new])
        _save_rows(rows_path, x_rows, y_rows)
        epochs = _train_round(model, np.concatenate([data["X_train"], x_rows]),
                              np.concatenate([data["y_train"], y_rows]), rng)
        checks = check(nets.from_keras(model), source.draw(config.DAGGER_MAPS), n_jobs=n_jobs)
        weights = model.get_weights()
        streak = state["streak"] + 1 if checks >= config.DAGGER_GOAL else 0
        if checks > state["best_checks"]:
            state.update(best_checks=checks, since_best=0, best_weights=weights)
        else:
            state["since_best"] += 1
        if streak >= config.DAGGER_STREAK:
            state["outcome"] = "goal"
        elif state["since_best"] >= config.DAGGER_PATIENCE:
            state["outcome"] = "stalled"
        elapsed = state["elapsed"] + time.time() - began
        record = {"round": number, "rows": int(len(y_new)), "respawns": int(respawns), "epochs": epochs,
                  "checks": checks, "streak": streak, "elapsed": elapsed}
        state.update(round=number, streak=streak, n_rows=int(len(y_rows)), respawns=state["respawns"] + respawns,
                     rng=rng.bit_generator.state, used=source.state(), weights=weights,
                     history=state["history"] + [record], elapsed=elapsed)
        checkpoint.save(state_path, state)
        progress.write_status({"approach": approach, "seed": seed, "stage": "dagger", "level": config.MAX_LEVEL,
                               "round": number, "checks": checks, "streak": streak, "elapsed": elapsed})
        if on_round is not None:
            on_round(record)
    final = state["weights"] if state["outcome"] == "goal" else state["best_weights"]
    model = keras_cpu().models.load_model(imitation_stage, compile=False)
    model.set_weights(final)
    _save_model(model, dagger_stage)
    history = state["history"]
    summary = {"approach": approach, "seed": seed, "settings": checkpoint.settings_record(),
               "rounds": state["round"], "goal_reached": state["outcome"] == "goal", "outcome": state["outcome"],
               "best_checks": state["best_checks"], "final_checks": history[-1]["checks"] if history else None,
               "correction_rows": state["n_rows"], "respawns": state["respawns"], "wall_time": state["elapsed"],
               "stage_path": str(dagger_stage), "history": history}
    _write_json(run / SUMMARY_NAME, summary)
    paths.remove(state_path)
    progress.milestone(_finished_message(label, seed, state))
    return summary


def speed_push(approach, seed, runs_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1, on_generation=None,
               max_generations=None):
    """Evolution strategy grown from the refined network, from Extra hard on (revision 3 §6.4). Its final controller
    is the method's final controller; the run itself resumes and skips like every evolutionary run."""
    model = keras_cpu().models.load_model(paths.stage_path(approach, seed, "dagger", runs_dir), compile=False)
    hidden, activation = nets.dense_spec(model)
    return es.run_es(seed=seed, out_dir=paths.speed_dir(approach, seed, runs_dir), champion_dir=champion_dir,
                     n_jobs=n_jobs, on_generation=on_generation, start_level=config.MAX_LEVEL,
                     max_generations=max_generations, initial=nets.to_flat(model), approach=approach,
                     hidden=hidden, activation=activation)


def run(approach, seed, data, runs_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1):
    """Refinement, then speed push: one counted run of revision 3 §8.3."""
    refined = refine(approach, seed, data, runs_dir=runs_dir, n_jobs=n_jobs)
    pushed = speed_push(approach, seed, runs_dir=runs_dir, champion_dir=champion_dir, n_jobs=n_jobs)
    return {"approach": approach, "seed": seed, "dagger": refined, "speed": pushed}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_dagger.py`
Expected: PASS (the slow resume test is deselected).

Run: `~/miniconda3/bin/python -m pytest -q -m slow tests/test_dagger.py`
Expected: PASS — `test_an_interrupted_refinement_resumes_to_the_same_result`.

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: PASS (fast suite).

- [ ] **Step 6: Update the end-to-end pipeline test**

Since Task 10, imitation training saves no final controller, so the slow `tests/test_end_to_end.py` must produce the imitation champions the way the experiment does: DAgger, then the speed push. In `tests/test_end_to_end.py`:

1. Change the import line to `from racing import config, dagger, es, evaluate, expert, ga, imitation, nets, paths` and add `from racing.backend import keras_cpu` below it.
2. Change the test's signature to `def test_pipeline_end_to_end(tmp_path, monkeypatch):` and make these its first two lines (small rounds; the real values are pinned in `tests/test_dagger.py`):

```python
    monkeypatch.setattr(config, "DAGGER_MAPS", 2)
    monkeypatch.setattr(config, "DAGGER_EPOCHS", 2)
```

3. Replace the seeded ES call (the `es.run_es(...)` call that passes `initial=es.trained_baseline(1, champion_dir=champions)`) with:

```python
    for approach, seed in (("baseline", 0), ("baseline", 1), ("kt_random", 0)):
        dagger.refine(approach, seed, data, runs_dir=runs, max_rounds=1)
        refined = keras_cpu().models.load_model(paths.stage_path(approach, seed, "dagger", runs), compile=False)
        hidden, activation = nets.dense_spec(refined)
        # what dagger.speed_push does, with 20 cars instead of config.ES_POP
        es.run_es(pop_size=20, max_generations=1, seed=seed, start_level=config.MAX_LEVEL,
                  initial=nets.to_flat(refined), approach=approach, hidden=hidden, activation=activation,
                  out_dir=paths.speed_dir(approach, seed, runs), champion_dir=champions)
    es.run_es(pop_size=20, max_generations=2, seed=1, initial=es.dagger_baseline(1, runs_dir=runs),
              approach="es_seeded", out_dir=paths.run_dir("es_seeded", 1, runs), champion_dir=champions)
```

Leave every other line as it is (Task 12 changes only its table names).

Run: `~/miniconda3/bin/python -m pytest -q -m slow tests/test_end_to_end.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add racing/config.py racing/checkpoint.py racing/dagger.py tests/test_dagger.py tests/test_end_to_end.py
git commit -m "DAgger refinement of the imitation networks with checkpointed rounds, and their ES speed push

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/config.py racing/checkpoint.py racing/dagger.py tests/test_dagger.py tests/test_end_to_end.py
```

(Leave `racing/checkpoint.py` out of both lists if Step 3 did not change it.)

---

### Task 12: Evaluation tables and figures

Rewrites `racing/evaluate.py` for revision 3 §7: every final controller drives the held-out maps as a three-lap time trial and is recorded as **finished** (within the simulator's limit of twice the reference lap time per lap) and **succeeded** (within the starting target, `TARGET_FACTOR` × three reference laps). The tables lead with finishes and lap times and never compare with the expert. The network figure is added in Task 13.

**Files:**
- Modify (replace whole file): `racing/evaluate.py`
- Test (replace whole file): `tests/test_evaluate.py`
- Test: `tests/test_end_to_end.py` (table-name assertions only; its training calls are updated by Tasks 6, 10 and 11)

**Interfaces:**
- Consumes:
  - `sim.simulate(track, policy, n, laps=...)` → `RunResult` (with `laps_done`, Task 2); `sim.target_time(track, factor, laps=config.LAPS)` (Task 2).
  - `config.TARGET_FACTOR`, `config.MASTERY_FRACTION`, `config.IMITATION_APPROACHES`, `config.APPROACHES` (incl. `"neat"`), `config.LABELS`, `config.COLORS`, `config.DIFFICULTIES`, `config.MAX_LEVEL`, `config.RUN_SEEDS`.
  - `nets.champion_seeds`, `nets.load_champion(approach, seed, directory)` → `(keras model, driver)`; every driver has `act`, `policy`, `n_params` (Tasks 2, 8).
  - `nets.DensePopulation`, `population.save_dense_champion(approach, seed, population, index, champion_dir)` (tests only, Tasks 2 and 6).
  - `paths.stage_path(approach, seed, stage, runs_dir)`, `paths.speed_dir(approach, seed, runs_dir)`, `paths.run_dir`, `paths.search_dir` (Task 10).
  - Evolution `summary.json` fields (Task 6): `wall_time, evals, generations, level_reached, goal_reached, outcome, target_reduction, start_level, levels[{level, generation, evals, elapsed}]`; `log.csv` columns `generation, min_share, median_lap, step_size, event` (`LOG_FIELDS`).
  - DAgger record `<runs>/<approach>/seed<k>/dagger.json` (Task 11): fields `rounds`, `goal_reached`, `wall_time`. Missing fields show "–".
  - Imitation training `summary.json` in `<runs>/<approach>/seed<k>/` (Task 10): field `wall_time`.
- Produces (used by Tasks 13 and 15):
  - `DRIVE_COLUMNS`, `driving_results(drivers, track_seeds, level=config.MAX_LEVEL, factor=config.TARGET_FACTOR, n_jobs=1, backend="loky")`, `best_seeds(validation)`, `results_table(per_track, runs_dir)`, `head_to_head_table(per_track)`, `training_cost_table(runs_dir)`, `stage_results(runs_dir, per_track, test_seeds=config.TEST_SEEDS, n_jobs=1)`, `stages_table(stage_results)`, `metrics_per_seed(models, data, runs_dir, latency_calls=1000, drivers=None)`, `comparison_table(metrics)`, `evolution_logs(runs_dir)`, `figures(runs_dir, out_dir)`.
  - `main(...)` returns a dict with exactly the keys `results, head_to_head, training_cost, stages, comparison, search_cost, driving_selected, dataset` (DataFrames; notebook 08 in Task 15 displays them by these names) and `per_track, validation, metrics, best_seeds, figures` (a list of Paths; Task 13 appends `networks.png`), and writes `tables/<name>.csv|.md` for the eight tables plus `driving_per_track.csv`, `validation_per_track.csv`, `stages_per_track.csv`, `metrics_per_seed.csv`.
  - `driving_table` and the `level_reached` metric column are removed (the results and training-cost tables replace them).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_evaluate.py` with:

```python
import json

import numpy as np
import pandas as pd
import pytest

from racing import config, evaluate, nets, paths, population, sim


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(evaluate, "get_track", lambda seed, level: circle_track)
    return circle_track


def drivers(n=3, seed=0):
    flats = nets.init_population(np.random.default_rng(seed), n)
    return {("ga", k): nets.from_flat(flats[k]) for k in range(n)}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def drive_rows(rows):
    """A per-track frame from (approach, seed, track, finished, succeeded, crashed, lap time) tuples."""
    return pd.DataFrame([{"approach": a, "run_seed": s, "track": t, "finished": f, "success": ok, "crashed": c,
                          "race_time": lap * config.LAPS, "lap_time": lap, "progress": 1.0 if f else 0.5,
                          "fitness": 1.5 if f else 0.5} for a, s, t, f, ok, c, lap in rows],
                        columns=evaluate.DRIVE_COLUMNS)


def test_controllers_drive_all_laps(monkeypatch, fake_tracks):
    seen = []
    real = evaluate.sim.simulate

    def recording(track, policy, n, laps=1, **kwargs):
        seen.append(laps)
        return real(track, policy, n, laps=laps, **kwargs)

    monkeypatch.setattr(evaluate.sim, "simulate", recording)
    frame = evaluate.driving_results(drivers(), [1000])
    assert seen == [config.LAPS]
    done = frame[frame["finished"]]
    assert np.allclose(done["lap_time"], done["race_time"] / config.LAPS)


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


def test_success_means_three_laps_within_the_target(monkeypatch, fake_tracks):
    result = sim.RunResult(fitness=np.array([1.2, 1.1, 0.4]), finished=np.array([True, True, False]),
                           crashed=np.array([False, False, True]), progress=np.array([1.0, 1.0, 0.3]),
                           finish_time=np.array([70.0, 90.0, np.nan]), steps=1, laps_done=np.array([3, 3, 0]))
    monkeypatch.setattr(evaluate.sim, "simulate", lambda track, policy, n, **kwargs: result)
    frame = evaluate.driving_results(drivers(), [1000])          # target: 1.3 × 20 s × 3 laps = 78 s
    assert list(frame["finished"]) == [True, True, False] and list(frame["success"]) == [True, False, False]
    assert frame["lap_time"].iloc[1] == pytest.approx(30.0)


def test_best_seeds_most_successes_then_fitness_then_lowest_seed():
    rows = [("ga", 0, True, 1.5), ("ga", 0, True, 1.5), ("ga", 1, True, 1.9), ("ga", 1, False, 1.9),
            ("es", 0, True, 1.2), ("es", 2, True, 1.2), ("neat", 1, False, 0.4), ("neat", 2, False, 0.6)]
    frame = pd.DataFrame([{"approach": a, "run_seed": s, "success": ok, "fitness": f} for a, s, ok, f in rows])
    assert evaluate.best_seeds(frame) == {"es": 0, "ga": 0, "neat": 2}


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


def test_every_controller_is_timed_with_the_same_prediction_settings():
    flat = nets.init_population(np.random.default_rng(9), 1)[0]
    trained = nets.to_keras(flat)
    trained.compile(optimizer="adam", loss="sparse_categorical_crossentropy", jit_compile=True,
                    steps_per_execution=4)
    x = np.random.default_rng(10).random((64, config.N_INPUTS)).astype(np.float32)
    trained.predict(x, verbose=0)                       # builds a predict function with the training settings
    for model in (trained, nets.to_keras(flat)):
        evaluate.same_prediction_settings(model)
        assert model.jit_compile is False and model.steps_per_execution == 1 and model.predict_function is None
        assert model.predict(x, verbose=0).shape == (64, config.N_ACTIONS)


def stub_timing(monkeypatch, order=None):
    record = order.append if order is not None else (lambda name: None)
    monkeypatch.setattr(evaluate, "same_prediction_settings", lambda model: record("settings"))
    monkeypatch.setattr(evaluate, "accuracy", lambda model, x, y: record("accuracy") or 0.5)
    monkeypatch.setattr(evaluate, "inference_time", lambda model, x: record("batch") or 1e-6)
    monkeypatch.setattr(evaluate, "single_sample_latency", lambda model, x, calls: record("single") or 1e-3)


def test_metrics_apply_the_prediction_settings_before_any_prediction(monkeypatch, tmp_path):
    order = []
    stub_timing(monkeypatch, order)
    model = nets.to_keras(nets.init_population(np.random.default_rng(11), 1)[0])
    data = {"X_test": np.zeros((4, config.N_INPUTS), np.float32), "y_test": np.zeros(4, int)}
    evaluate.metrics_per_seed({("ga", 0): model}, data, tmp_path)
    assert order == ["settings", "accuracy", "batch", "single"]


def test_metrics_count_driver_parameters_and_all_imitation_training(monkeypatch, tmp_path):
    stub_timing(monkeypatch)

    class NeatLike:
        n_params = 19

    model = nets.to_keras(nets.init_population(np.random.default_rng(13), 1)[0])
    write_json(paths.run_dir("baseline", 0, tmp_path) / "summary.json", {"wall_time": 10.0})
    write_json(paths.run_dir("baseline", 0, tmp_path) / "dagger.json", {"wall_time": 20.0, "rounds": 3})
    write_json(paths.speed_dir("baseline", 0, tmp_path) / "summary.json", {"wall_time": 30.0})
    data = {"X_test": np.zeros((4, config.N_INPUTS), np.float32), "y_test": np.zeros(4, int)}
    metrics = evaluate.metrics_per_seed({("neat", 0): model, ("baseline", 0): model}, data, tmp_path,
                                        drivers={("neat", 0): NeatLike()}).set_index("approach")
    assert metrics.loc["neat", "params"] == 19 and metrics.loc["baseline", "params"] == model.count_params()
    assert metrics.loc["baseline", "training_s"] == 60.0 and np.isnan(metrics.loc["neat", "training_s"])


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
                     "single_ms": 0.5, "training_s": 30.0 + seed, "search_s": 900.0})
    for seed in (0, 1):
        rows.append({"approach": "ga", "run_seed": seed, "accuracy": 0.2, "params": 431, "batch_us": 2.0,
                     "single_ms": 0.5, "training_s": 600.0, "search_s": np.nan})
    return pd.DataFrame(rows)


def test_comparison_table():
    table = evaluate.comparison_table(metrics_frame())
    assert list(table["Method"]) == ["KT Random", "GA"] and list(table["Seeds"]) == [3, 2]
    assert table.loc[0, "Test accuracy"] == "0.6200 ± 0.0200" and table.loc[0, "Parameters"] == "447"
    assert table.loc[0, "Search time (s)"] == "900.0" and table.loc[1, "Search time (s)"] == "–"


def test_results_table_counts_finishes_successes_and_target_reduction(tmp_path):
    nan = np.nan
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0), ("ga", 0, 1001, False, False, True, nan),
                            ("ga", 1, 1000, True, False, False, 54.0), ("ga", 1, 1001, True, True, False, 60.0),
                            ("es", 0, 1000, True, True, False, 52.0), ("es", 0, 1001, False, False, True, nan)])
    write_json(paths.run_dir("ga", 0, tmp_path) / "summary.json", {"goal_reached": True, "target_reduction": 0.143})
    write_json(paths.run_dir("ga", 1, tmp_path) / "summary.json", {"goal_reached": False})
    table = evaluate.results_table(per_track, tmp_path)
    assert list(table["Method"]) == ["ES", "GA"]
    ga = table.set_index("Method").loc["GA"]
    assert ga["Seeds"] == 2 and ga["Finished"] == "3 / 4" and ga["Finished per seed"] == "1, 2"
    assert ga["Succeeded"] == "2 / 4" and ga["Succeeded per seed"] == "1, 1"
    assert ga["Mean lap time (s)"] == "53.50 ± 4.95" and ga["Best lap time (s)"] == "50.00"
    assert ga["Mean race time (s)"] == "160.5 ± 14.8" and ga["Crashes"] == "0.5 ± 0.7"
    assert ga["Target reduction (per seed)"] == "14.3%, not reached"
    assert table.set_index("Method").loc["ES", "Target reduction (per seed)"] == "–"


def test_head_to_head_compares_the_maps_every_controller_finished():
    nan = np.nan
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0), ("ga", 1, 1000, True, True, False, 54.0),
                            ("es", 0, 1000, True, True, False, 51.0), ("ga", 0, 1001, True, True, False, 60.0),
                            ("ga", 1, 1001, True, True, False, 58.0), ("es", 0, 1001, True, True, False, 62.0),
                            ("ga", 0, 1002, True, True, False, 55.0), ("ga", 1, 1002, True, True, False, 55.0),
                            ("es", 0, 1002, False, False, True, nan)])
    table = evaluate.head_to_head_table(per_track).set_index("Method")
    column = "Mean lap time on the 2 shared maps (s)"
    assert table.loc["GA", column] == "55.50 ± 0.71" and table.loc["ES", column] == "56.50"
    assert table.loc["GA", "Fastest on (maps)"] == 1 and table.loc["ES", "Fastest on (maps)"] == 1


def test_head_to_head_without_shared_maps():
    per_track = drive_rows([("ga", 0, 1000, True, True, False, 50.0), ("es", 0, 1000, False, False, True, np.nan)])
    table = evaluate.head_to_head_table(per_track)
    column = "Mean lap time on the 0 shared maps (s)"
    assert list(table[column]) == ["–", "–"] and list(table["Fastest on (maps)"]) == [0, 0]


def test_training_cost_table(tmp_path):
    level = lambda n, generation, elapsed: {"level": n, "generation": generation, "evals": generation * 1000,
                                            "elapsed": elapsed}
    write_json(paths.run_dir("ga", 0, tmp_path) / "summary.json", {
        "start_level": 1, "goal_reached": True, "outcome": "speed_limit", "generations": 1200, "evals": 1_200_000,
        "wall_time": 36000.0, "level_reached": 4,
        "levels": [level(1, 25, 60.0), level(2, 80, 600.0), level(3, 120, 1800.0), level(4, 300, 7200.0)]})
    write_json(paths.run_dir("ga", 1, tmp_path) / "summary.json", {
        "start_level": 1, "goal_reached": False, "outcome": "stalled", "generations": 2100, "evals": 2_100_000,
        "wall_time": 5000.0, "level_reached": 2, "levels": [level(1, 30, 90.0)]})
    write_json(paths.run_dir("baseline", 0, tmp_path) / "summary.json", {"wall_time": 15.0})
    write_json(paths.run_dir("baseline", 0, tmp_path) / "dagger.json",
               {"rounds": 7, "goal_reached": True, "wall_time": 3600.0})
    write_json(paths.speed_dir("baseline", 0, tmp_path) / "summary.json", {
        "start_level": 4, "goal_reached": True, "outcome": "speed_limit", "generations": 600, "evals": 330_000,
        "wall_time": 7200.0, "level_reached": 4, "levels": [level(4, 40, 900.0)]})
    write_json(paths.search_dir("kt_random", tmp_path) / "summary.json", {"wall_time": 914.2})
    write_json(paths.run_dir("kt_random", 0, tmp_path) / "summary.json", {"wall_time": 12.0})
    table = evaluate.training_cost_table(tmp_path)
    assert list(table["Method"]) == ["Baseline", "KT Random", "GA", "GA"] and list(table["Seed"]) == [0, 0, 0, 1]
    base, kt, ga0, ga1 = (table.iloc[k] for k in range(4))
    assert ga0["Easy cleared"] == "gen 25 · 25,000 evals · 0.02 h"
    assert ga0["Mastery goal"] == "gen 300 · 300,000 evals · 2.00 h"
    assert ga0["Run end"] == "speed limit · gen 1,200" and ga0["Evaluations"] == "1,200,000"
    assert ga0["Evolution (h)"] == "10.00" and ga0["Total (h)"] == "10.00" and ga0["Search (h)"] == "–"
    assert ga1["Medium cleared"] == "not reached" and ga1["Mastery goal"] == "not reached"
    assert ga1["Run end"] == "stalled on level 2 · gen 2,100"
    assert base["Easy cleared"] == "–" and base["Mastery goal"] == "gen 40 · 40,000 evals · 0.25 h"
    assert base["DAgger rounds"] == "7" and base["DAgger goal"] == "reached" and base["DAgger (h)"] == "1.00"
    assert base["Imitation training (h)"] == "0.00" and base["Total (h)"] == "3.00"
    assert kt["Search (h)"] == "0.25" and kt["Run end"] == "–" and kt["Mastery goal"] == "–"
    assert kt["Evaluations"] == "–" and kt["DAgger rounds"] == "–" and kt["Total (h)"] == "0.00"


def test_training_cost_table_without_records_is_empty(tmp_path):
    assert evaluate.training_cost_table(tmp_path).empty


def test_stage_results_and_table(fake_tracks, tmp_path):
    flats = nets.init_population(np.random.default_rng(12), 2)
    for stage, flat in zip(("imitation", "dagger"), flats):
        path = paths.stage_path("baseline", 0, stage, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nets.to_keras(flat).save(path)
    final = drive_rows([("baseline", 0, 1000, True, True, False, 50.0), ("ga", 0, 1000, True, True, False, 40.0)])
    results = evaluate.stage_results(tmp_path, final, test_seeds=[1000])
    assert sorted(results["stage"].unique()) == ["dagger", "imitation", "speed push"]
    assert set(results["approach"]) == {"baseline"}
    table = evaluate.stages_table(results)
    assert list(table["Stage"]) == ["After imitation", "After DAgger", "After speed push"]
    assert set(table["Method"]) == {"Baseline"} and table.loc[2, "Finished"] == "1 / 1"


def test_stage_results_without_stage_files(tmp_path):
    results = evaluate.stage_results(tmp_path, drive_rows([("ga", 0, 1000, True, True, False, 40.0)]), [1000])
    assert results.empty and "stage" in results.columns and evaluate.stages_table(results).empty


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


def write_log(folder, events, shares, laps):
    folder.mkdir(parents=True, exist_ok=True)
    n = len(shares)
    pd.DataFrame({"generation": range(n), "level": 1, "evals": np.arange(1, n + 1) * 10, "min_share": shares,
                  "median_lap": laps, "step_size": np.linspace(0.1, 0.05, n),
                  "event": events}).to_csv(folder / "log.csv", index=False)


def test_figures_per_method(tmp_path):
    runs = tmp_path / "runs"
    for seed in (0, 1):
        write_log(paths.run_dir("ga", seed, runs), [None, "level_up", None, "goal"], [0.1, 0.92, 0.5, 0.95],
                  [np.nan, np.nan, 40.0, 38.5])
    write_log(paths.speed_dir("baseline", 0, runs), [None, "goal", "tightened"], [0.91, 0.93, 0.95],
              [45.0, 44.0, 42.0])
    logs = evaluate.evolution_logs(runs)
    assert sorted(logs) == ["baseline", "ga"] and sorted(logs["ga"]) == [0, 1] and sorted(logs["baseline"]) == [0]
    written = evaluate.figures(runs, tmp_path / "figures")
    assert [p.name for p in written] == ["success_over_training.png", "lap_time_over_training.png",
                                         "step_sizes.png"]


def test_light_series_colors_get_an_outline():
    assert evaluate._outline(config.COLORS["neat"]) is not None
    assert evaluate._outline(config.COLORS["ga"]) is None and evaluate._outline(config.COLORS["es"]) is None


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
        population.save_dense_champion(approach, seed, nets.DensePopulation(nets.init_population(rng, 1)), 0,
                                       champions)
    write_json(paths.run_dir("ga", 0, runs) / "summary.json", {
        "wall_time": 12.0, "level_reached": 2, "goal_reached": False, "outcome": "stalled", "generations": 10,
        "evals": 1000, "start_level": 1, "levels": [{"level": 1, "generation": 4, "evals": 400, "elapsed": 5.0}]})
    np.savez(tmp_path / "dataset.npz", X_test=rng.random((40, config.N_INPUTS)).astype(np.float32),
             y_test=rng.integers(0, 15, 40))
    results = evaluate.main(champions_dir=champions, runs_dir=runs, data_path=tmp_path / "dataset.npz",
                            out_dir=tmp_path / "out", test_seeds=[1000], val_seeds=[80], latency_calls=5)
    assert set(results) == {"results", "head_to_head", "training_cost", "stages", "comparison", "search_cost",
                            "driving_selected", "dataset", "per_track", "validation", "metrics", "best_seeds",
                            "figures"}
    assert list(results["results"]["Method"]) == ["ES", "GA"] and list(results["comparison"]["Method"]) == ["ES", "GA"]
    cost = results["training_cost"].set_index("Seed")
    assert cost.loc[0, "Mastery goal"] == "not reached" and cost.loc[0, "Run end"] == "stalled on level 2 · gen 10"
    assert json.loads((champions / "best_seeds.json").read_text()).keys() == {"ga", "es"}
    for name in ("results", "head_to_head", "training_cost", "stages", "comparison", "search_cost",
                 "driving_selected", "dataset"):
        assert (tmp_path / "out" / "tables" / f"{name}.md").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_evaluate.py -q`
Expected: FAIL — among others `test_driving_results_columns_and_rows` (no `success` column), `AttributeError: module 'racing.evaluate' has no attribute 'results_table'` / `head_to_head_table` / `training_cost_table` / `stage_results` / `_outline`, and `test_best_seeds_most_successes_then_fitness_then_lowest_seed` (the old rule ranks by mean fitness alone and picks GA seed 1).

- [ ] **Step 3: Write the implementation**

Replace `racing/evaluate.py` with:

```python
"""Final evaluation (spec §11, revision 3 §7): three-lap time trials on held-out maps, test accuracy, size, speed and
training cost, as tables and figures. Each method has one final controller per run seed; tables report the mean and
standard deviation across seeds and lead with completion and lap times."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from racing import config, nets, paths, sim
from racing.tracks import get_track

DRIVE_COLUMNS = ["approach", "run_seed", "track", "finished", "success", "crashed", "race_time", "lap_time",
                 "progress", "fitness"]
PER_SEED_COLUMNS = ["approach", "run_seed", "maps", "finished", "success", "lap_time", "race_time", "best_lap",
                    "crashes"]
STAGES = (("imitation", "After imitation"), ("dagger", "After DAgger"), ("speed push", "After speed push"))
LEVEL_NAMES = {**{level: f"{name} cleared" for level, name in config.DIFFICULTIES.items() if level < config.MAX_LEVEL},
               config.MAX_LEVEL: "Mastery goal"}
SEED_STYLES = ("-", "--", ":")
EVENT_STYLES = {"level_up": "-", "goal": "-", "tightened": ":"}


def _drive_track(drivers, track_seed, level, factor):
    """All config.LAPS laps for every controller on one track, simulated together (cars never interact). A car
    finishes within the simulator's limit of twice the reference lap time per lap, and succeeds if it finishes
    within the target time (revision 3 §7.1)."""
    track = get_track(track_seed, level)
    keys = list(drivers)
    result = sim.simulate(track, nets.mixed_policy([drivers[k] for k in keys]), len(keys), laps=config.LAPS)
    target = sim.target_time(track, factor)
    rows = []
    for i, (approach, run_seed) in enumerate(keys):
        race_time = float(result.finish_time[i])
        finished = bool(result.finished[i])
        rows.append({"approach": approach, "run_seed": run_seed, "track": track_seed, "finished": finished,
                     "success": finished and race_time <= target + config.TIME_EPS,
                     "crashed": bool(result.crashed[i]), "race_time": race_time,
                     "lap_time": race_time / config.LAPS, "progress": float(result.progress[i]),
                     "fitness": float(result.fitness[i])})
    return rows


def driving_results(drivers, track_seeds, level=config.MAX_LEVEL, factor=config.TARGET_FACTOR, n_jobs=1,
                    backend="loky"):
    """A three-lap time trial per controller per track. drivers: {(approach, run seed or trial id): driver}. The
    tracks are distributed across worker processes; the result does not depend on n_jobs."""
    track_seeds = list(track_seeds)
    if not drivers:
        return pd.DataFrame(columns=DRIVE_COLUMNS)
    if n_jobs == 1 or len(track_seeds) == 1:
        parts = [_drive_track(drivers, seed, level, factor) for seed in track_seeds]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_drive_track)(drivers, seed, level, factor) for seed in track_seeds)
    return pd.DataFrame([row for part in parts for row in part], columns=DRIVE_COLUMNS)


def best_seeds(validation):
    """For each approach, the run seed whose controller succeeded on the most validation maps; ties go to the higher
    mean validation fitness, then to the lowest seed (revision 3 §7.1)."""
    best = {}
    for approach in config.APPROACHES:
        sub = validation[validation["approach"] == approach]
        if sub.empty:
            continue
        per_seed = sub.groupby("run_seed").agg(successes=("success", "sum"), fitness=("fitness", "mean"))
        ranked = sorted(per_seed.itertuples(), key=lambda r: (-r.successes, -r.fitness, r.Index))
        best[approach] = int(ranked[0].Index)
    return best


def same_prediction_settings(model):
    """Every controller is timed with the same prediction settings: no XLA compilation and one batch per call
    (spec §11). A loaded imitation model would otherwise keep its training settings, an evolutionary one the
    Keras default, which depends on the installed devices."""
    model.jit_compile = False
    model.steps_per_execution = 1
    model.predict_function = None              # rebuilt with these settings by the next predict call
    return model


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


def run_summary(approach, seed, runs_dir):
    """Summary of the evolutionary run that produced a final controller: the run itself for the evolutionary methods,
    the speed push for the imitation methods (revision 3 §6.4). {} if it does not exist."""
    if approach in config.IMITATION_APPROACHES:
        return read_json(paths.speed_dir(approach, seed, runs_dir) / "summary.json")
    return read_json(paths.run_dir(approach, seed, runs_dir) / "summary.json")


def dagger_record(approach, seed, runs_dir):
    """The DAgger refinement record of an imitation run (revision 3 §6.3); {} if it does not exist."""
    return read_json(paths.run_dir(approach, seed, runs_dir) / "dagger.json")


def training_seconds(approach, seed, runs_dir):
    """Training time of one final controller: the evolutionary run, or for an imitation method its initial training,
    DAgger refinement and speed push together. The Keras Tuner search is reported separately. nan without records."""
    parts = [run_summary(approach, seed, runs_dir).get("wall_time")]
    if approach in config.IMITATION_APPROACHES:
        parts += [read_json(paths.run_dir(approach, seed, runs_dir) / "summary.json").get("wall_time"),
                  dagger_record(approach, seed, runs_dir).get("wall_time")]
    values = [float(v) for v in parts if v is not None]
    return float(sum(values)) if values else float("nan")


def metrics_per_seed(models, data, runs_dir, latency_calls=1000, drivers=None):
    """One row per final controller: test accuracy, size, speed and training cost. With drivers, the parameter count
    is the driver's, so a NEAT controller counts its genome's connections and biases rather than the zero-padded
    layers of its Keras copy (revision 3 §5.4)."""
    rows = []
    for (approach, run_seed), model in models.items():
        same_prediction_settings(model)
        search = float("nan")
        if approach in config.TUNER_APPROACHES:
            search = read_json(paths.search_dir(approach, runs_dir) / "summary.json").get("wall_time", float("nan"))
        key = (approach, run_seed)
        params = drivers[key].n_params if drivers is not None and key in drivers else model.count_params()
        rows.append({"approach": approach, "run_seed": run_seed,
                     "accuracy": accuracy(model, data["X_test"], data["y_test"]),
                     "params": int(params),
                     "batch_us": inference_time(model, data["X_test"]) * 1e6,
                     "single_ms": single_sample_latency(model, data["X_test"], latency_calls) * 1e3,
                     "training_s": training_seconds(approach, run_seed, runs_dir), "search_s": float(search)})
    return pd.DataFrame(rows, columns=["approach", "run_seed", "accuracy", "params", "batch_us", "single_ms",
                                       "training_s", "search_s"])


def mean_std(values, digits):
    """'mean ± std' across seeds (sample standard deviation); the value alone if there is one value or all are
    equal; '–' if there is none."""
    values = [float(v) for v in values if not np.isnan(float(v))]
    if not values:
        return "–"
    if len(values) == 1 or all(v == values[0] for v in values):
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def seeds_note(values, digits):
    """mean_std of per-seed values, naming how many seeds it covers when some seeds have no value."""
    values = [float(v) for v in values]
    cell = mean_std(values, digits)
    covered = sum(1 for v in values if not np.isnan(v))
    if 0 < covered < len(values):
        cell += f" ({covered} of {len(values)} seeds)"
    return cell


def hours(seconds):
    """Seconds as hours with two decimals; '–' when unknown."""
    if seconds is None or np.isnan(float(seconds)):
        return "–"
    return f"{float(seconds) / 3600:.2f}"


def per_seed(per_track):
    """One row per (approach, run seed): maps driven, finished and succeeded, mean lap and race time and best lap
    time over the finished maps, and crashes."""
    rows = []
    for (approach, seed), runs in per_track.groupby(["approach", "run_seed"]):
        done = runs[runs["finished"].astype(bool)]
        rows.append({"approach": approach, "run_seed": seed, "maps": len(runs),
                     "finished": int(runs["finished"].astype(bool).sum()),
                     "success": int(runs["success"].astype(bool).sum()),
                     "lap_time": float(done["lap_time"].mean()) if len(done) else float("nan"),
                     "race_time": float(done["race_time"].mean()) if len(done) else float("nan"),
                     "best_lap": float(done["lap_time"].min()) if len(done) else float("nan"),
                     "crashes": int(runs["crashed"].astype(bool).sum())})
    return pd.DataFrame(rows, columns=PER_SEED_COLUMNS)


def reduction_cell(summary):
    """The speed push's result for one run: the target reduction reached, 'not reached' if the run never reached the
    mastery goal, '–' without a summary (revision 3 §4.4)."""
    if not summary:
        return "–"
    if not summary.get("goal_reached"):
        return "not reached"
    return f"{100 * float(summary['target_reduction']):.1f}%"


def results_table(per_track, runs_dir):
    """Revision 3 §7.2 table 1: test maps finished and succeeded (in total and per seed), lap and race times,
    crashes, and the target reduction of each seed's speed push."""
    seeds = per_seed(per_track)
    rows = []
    for approach in config.APPROACHES:
        sub = seeds[seeds["approach"] == approach].sort_values("run_seed")
        if sub.empty:
            continue
        total = int(sub["maps"].sum())
        best_lap = "–" if sub["best_lap"].isna().all() else f"{sub['best_lap'].min():.2f}"
        rows.append({"Method": config.LABELS[approach], "Seeds": len(sub),
                     "Finished": f"{int(sub['finished'].sum())} / {total}",
                     "Finished per seed": ", ".join(str(int(v)) for v in sub["finished"]),
                     "Succeeded": f"{int(sub['success'].sum())} / {total}",
                     "Succeeded per seed": ", ".join(str(int(v)) for v in sub["success"]),
                     "Mean lap time (s)": seeds_note(sub["lap_time"], 2), "Best lap time (s)": best_lap,
                     "Mean race time (s)": seeds_note(sub["race_time"], 1),
                     "Crashes": mean_std(sub["crashes"], 1),
                     "Target reduction (per seed)": ", ".join(
                         reduction_cell(run_summary(approach, int(s), runs_dir)) for s in sub["run_seed"])})
    return pd.DataFrame(rows)


def head_to_head_table(per_track):
    """Revision 3 §7.2 table 2: mean lap time on the test maps that every final controller finished, and on how many
    of those maps each method had the lowest lap time (its mean over seeds)."""
    if per_track.empty:
        return pd.DataFrame()
    finished_all = per_track.groupby("track")["finished"].all()
    shared = sorted(finished_all[finished_all.astype(bool)].index)
    column = f"Mean lap time on the {len(shared)} shared maps (s)"
    on_shared = per_track[per_track["track"].isin(shared)]
    fastest = {}
    if shared:
        means = on_shared.groupby(["track", "approach"])["lap_time"].mean().unstack()
        for approach in means.idxmin(axis=1):
            fastest[approach] = fastest.get(approach, 0) + 1
    rows = []
    for approach in config.APPROACHES:
        sub = per_track[per_track["approach"] == approach]
        if sub.empty:
            continue
        mine = on_shared[on_shared["approach"] == approach]
        laps = [float(runs["lap_time"].mean()) for _, runs in mine.groupby("run_seed")]
        rows.append({"Method": config.LABELS[approach], "Seeds": int(sub["run_seed"].nunique()),
                     column: mean_std(laps, 2), "Fastest on (maps)": fastest.get(approach, 0)})
    return pd.DataFrame(rows)


def milestone_cell(summary, level):
    """When a run cleared `level` (on the last level: reached the mastery goal), as generation, evaluations and
    active time (revision 3 §1, §7.2); 'not reached' if it never did; '–' without a summary or for a level below the
    run's start level."""
    if not summary or summary.get("start_level", 1) > level:
        return "–"
    for entry in summary.get("levels", []):
        if entry.get("level") == level:
            return (f"gen {int(entry['generation']):,} · {int(entry['evals']):,} evals · "
                    f"{hours(entry['elapsed'])} h")
    return "not reached"


def end_cell(summary):
    """How a run ended: at its speed limit, stalled on a level, or stopped by a generation limit."""
    if not summary:
        return "–"
    outcome = summary.get("outcome")
    if outcome == "speed_limit":
        how = "speed limit"
    elif outcome == "stalled":
        how = f"stalled on level {summary.get('level_reached', '?')}"
    else:
        how = "stopped"
    return f"{how} · gen {int(summary.get('generations', 0)):,}"


def training_cost_table(runs_dir):
    """Revision 3 §7.2 table 3, one row per method and seed: when each level was cleared and the mastery goal
    reached, how the run ended, evaluations, and the time of each training stage. Seeds without any record are left
    out; missing records show '–' and goals never reached show 'not reached'."""
    rows = []
    for approach in config.APPROACHES:
        search = {}
        if approach in config.TUNER_APPROACHES:
            search = read_json(paths.search_dir(approach, runs_dir) / "summary.json")
        for seed in config.RUN_SEEDS:
            summary = run_summary(approach, seed, runs_dir)
            imitation, dagger = {}, {}
            if approach in config.IMITATION_APPROACHES:
                imitation = read_json(paths.run_dir(approach, seed, runs_dir) / "summary.json")
                dagger = dagger_record(approach, seed, runs_dir)
            if not (summary or imitation or dagger):
                continue
            row = {"Method": config.LABELS[approach], "Seed": seed}
            for level in range(1, config.MAX_LEVEL + 1):
                row[LEVEL_NAMES[level]] = milestone_cell(summary, level)
            row["Run end"] = end_cell(summary)
            row["Evaluations"] = f"{int(summary['evals']):,}" if "evals" in summary else "–"
            row["Search (h)"] = hours(search.get("wall_time"))
            row["Imitation training (h)"] = hours(imitation.get("wall_time"))
            row["DAgger rounds"] = str(dagger["rounds"]) if "rounds" in dagger else "–"
            row["DAgger goal"] = "–" if not dagger else ("reached" if dagger.get("goal_reached") else "not reached")
            row["DAgger (h)"] = hours(dagger.get("wall_time"))
            row["Evolution (h)"] = hours(summary.get("wall_time"))
            row["Total (h)"] = hours(training_seconds(approach, seed, runs_dir))
            rows.append(row)
    return pd.DataFrame(rows)


def stage_results(runs_dir, per_track, test_seeds=config.TEST_SEEDS, n_jobs=1):
    """Test-map results of the imitation methods after initial training and after DAgger (from their stage files)
    and after the speed push (their final controllers, taken from per_track), with a 'stage' column
    (revision 3 §6.5)."""
    from racing.backend import keras_cpu
    keras = keras_cpu()
    parts = []
    for stage in ("imitation", "dagger"):
        drivers = {}
        for approach in config.IMITATION_APPROACHES:
            for seed in config.RUN_SEEDS:
                path = paths.stage_path(approach, seed, stage, runs_dir)
                if path.exists():
                    drivers[(approach, seed)] = nets.from_keras(keras.models.load_model(path))
        if drivers:
            parts.append(driving_results(drivers, test_seeds, n_jobs=n_jobs).assign(stage=stage))
    final = per_track[per_track["approach"].isin(config.IMITATION_APPROACHES)]
    if len(final):
        parts.append(final.assign(stage="speed push"))
    if not parts:
        return pd.DataFrame(columns=[*DRIVE_COLUMNS, "stage"])
    return pd.concat(parts, ignore_index=True)


def stages_table(results):
    """Revision 3 §7.2 table 4: the imitation methods' test results after each stage."""
    rows = []
    for approach in config.IMITATION_APPROACHES:
        for stage, name in STAGES:
            sub = results[(results["approach"] == approach) & (results["stage"] == stage)]
            if sub.empty:
                continue
            seeds = per_seed(sub)
            total = int(seeds["maps"].sum())
            rows.append({"Method": config.LABELS[approach], "Stage": name, "Seeds": len(seeds),
                         "Finished": f"{int(seeds['finished'].sum())} / {total}",
                         "Succeeded": f"{int(seeds['success'].sum())} / {total}",
                         "Mean lap time (s)": seeds_note(seeds["lap_time"], 2),
                         "Crashes": mean_std(seeds["crashes"], 1)})
    return pd.DataFrame(rows)


def comparison_table(metrics):
    """The course comparison (revision 3 §7.2 table 5), measured as in spec §11."""
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
    """{approach: {run seed: log}} for every evolutionary run whose log exists: the runs of the evolutionary methods
    and the speed pushes of the imitation methods."""
    logs = {}
    for approach in config.APPROACHES:
        found = {}
        for seed in config.RUN_SEEDS:
            if approach in config.IMITATION_APPROACHES:
                path = paths.speed_dir(approach, seed, runs_dir) / "log.csv"
            else:
                path = paths.run_dir(approach, seed, runs_dir) / "log.csv"
            if path.exists():
                found[seed] = pd.read_csv(path)
        if found:
            logs[approach] = found
    return logs


def _outline(color):
    """Path effects that draw a thin dark outline around a line in a light color (NEAT's yellow), so it stays
    visible on a white background; None for darker colors."""
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as effects
    r, g, b = mcolors.to_rgb(color)
    if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.7:
        return None
    return [effects.Stroke(linewidth=2.6, foreground="0.35"), effects.Normal()]


def _line_kwargs(color):
    outline = _outline(color)
    return {"color": color} if outline is None else {"color": color, "path_effects": outline}


def _panels(plt, logs, column, ylabel, threshold=None):
    """Small multiples: one panel per method with data in `column`, one line per seed (told apart by line style),
    level advancements and the mastery goal as solid and target reductions as dotted grey vertical lines. None if no
    method has data."""
    from matplotlib.lines import Line2D
    with_data = {approach: runs for approach, runs in logs.items()
                 if any(column in log and log[column].notna().any() for log in runs.values())}
    if not with_data:
        return None
    n = len(with_data)
    ncols = min(4, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False, sharey=True)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, (approach, runs) in zip(axes.flat, with_data.items()):
        for k, (seed, log) in enumerate(sorted(runs.items())):
            if column not in log:
                continue
            ax.plot(log["generation"], log[column], linewidth=1.2, linestyle=SEED_STYLES[k % len(SEED_STYLES)],
                    label=f"seed {seed}", **_line_kwargs(config.COLORS[approach]))
            events = log["event"].fillna("").astype(str) if "event" in log else []
            for generation, event in zip(log["generation"], events):
                if event in EVENT_STYLES:
                    ax.axvline(generation, color="0.6", linewidth=0.6, linestyle=EVENT_STYLES[event], zorder=0)
        if threshold is not None:
            ax.axhline(threshold, color="0.3", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_title(config.LABELS[approach], fontsize=10)
        ax.set_xlabel("Generation", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
    for row in axes:
        row[0].set_ylabel(ylabel, fontsize=8)
    handles, _ = axes.flat[0].get_legend_handles_labels()
    handles += [Line2D([], [], color="0.6", linewidth=0.6, label="level cleared / goal"),
                Line2D([], [], color="0.6", linewidth=0.6, linestyle=":", label="target cut")]
    axes.flat[0].legend(handles=handles, frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def figures(runs_dir, out_dir):
    """Revision 3 §7.3: the lowest map success share and the median lap time over training (one panel per method,
    one line per seed), the step sizes, and the Keras Tuner trials. Only figures with data are written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    logs = evolution_logs(runs_dir)
    for name, column, ylabel, threshold in (
            ("success_over_training.png", "min_share", "Lowest map success share", config.MASTERY_FRACTION),
            ("lap_time_over_training.png", "median_lap", "Median lap time (s)", None)):
        fig = _panels(plt, logs, column, ylabel, threshold)
        if fig is None:
            continue
        if column == "min_share":
            fig.axes[0].set_ylim(0.0, 1.02)                 # the y-axis is shared, so this sets every panel
        written.append(out_dir / name)
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

    with_steps = {approach: runs for approach, runs in logs.items()
                  if any("step_size" in log for log in runs.values())}
    if with_steps:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for approach, runs in with_steps.items():
            if approach == "ga":
                name = "GA mutation strength"
            elif approach in config.IMITATION_APPROACHES:
                name = f"{config.LABELS[approach]} speed push σ"
            else:
                name = f"{config.LABELS[approach]} σ"
            for k, (seed, log) in enumerate(sorted(runs.items())):
                if "step_size" in log:
                    ax.plot(log["generation"], log["step_size"], linewidth=1.0, label=name if k == 0 else None,
                            **_line_kwargs(config.COLORS[approach]))
        ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.legend(frameon=False, fontsize=7)
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
    stages = stage_results(runs_dir, per_track, test_seeds, n_jobs)
    selected = driving_selected_table(runs_dir, val_seeds, test_seeds, n_jobs=n_jobs)
    with np.load(data_path) as d:
        data = {key: d[key] for key in d.files}
    metrics = metrics_per_seed(models, data, runs_dir, latency_calls, drivers)   # timed after all parallel work
    tables = {"results": results_table(per_track, runs_dir), "head_to_head": head_to_head_table(per_track),
              "training_cost": training_cost_table(runs_dir), "stages": stages_table(stages),
              "comparison": comparison_table(metrics), "search_cost": search_cost_table(runs_dir),
              "driving_selected": selected, "dataset": dataset_table(data_path)}
    folder = out_dir / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    per_track.to_csv(folder / "driving_per_track.csv", index=False)
    validation.to_csv(folder / "validation_per_track.csv", index=False)
    stages.to_csv(folder / "stages_per_track.csv", index=False)
    metrics.to_csv(folder / "metrics_per_seed.csv", index=False)
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
        (folder / f"{name}.md").write_text(to_markdown(frame))
    return {**tables, "per_track": per_track, "validation": validation, "metrics": metrics, "best_seeds": best,
            "figures": figures(runs_dir, out_dir / "figures")}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_evaluate.py -q`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: PASS. Any remaining caller of `evaluate.driving_table` or of the `level_reached` metric column (search with `grep -rn "driving_table\|level_reached" racing scripts tests`) is updated in this step; `scripts/make_notebooks.py` is rewritten in Task 15 and may still name `driving` until then — do not edit it here.

- [ ] **Step 6: Update the end-to-end test's table names and run it**

In `tests/test_end_to_end.py`, replace

```python
    for name in ("comparison", "driving", "search_cost", "driving_selected", "dataset"):
        assert (tmp_path / "out" / "tables" / f"{name}.md").exists()
```

with

```python
    for name in ("results", "head_to_head", "training_cost", "stages", "comparison", "search_cost",
                 "driving_selected", "dataset"):
        assert (tmp_path / "out" / "tables" / f"{name}.md").exists()
```

Leave every other line of the test as Tasks 6, 10 and 11 left it.

Run: `~/miniconda3/bin/python -m pytest tests/test_end_to_end.py -m slow -q`
Expected: PASS. If it fails on anything other than the evaluation (a training call or a `Method` list that Tasks 6, 10 or 11 own), stop and report it rather than editing their lines.

- [ ] **Step 7: Commit**

```bash
git add racing/evaluate.py tests/test_evaluate.py tests/test_end_to_end.py
git commit -m "Evaluation for revision 3: time trials (finished and succeeded), results, head-to-head, training cost and stage tables, success and lap-time figures" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/evaluate.py tests/test_evaluate.py tests/test_end_to_end.py
```

---

### Task 13: Network figure

Adds `racing/netfig.py`, which draws each method's best final controller as a graph (revision 3 §7.3), and wires it into `evaluate.main` as `figures/networks.png`.

**Files:**
- Create: `racing/netfig.py`
- Modify: `racing/evaluate.py` (imports and the end of `main`)
- Test: `tests/test_netfig.py` (new), `tests/test_evaluate.py` (`test_main_with_partial_results`)

**Interfaces:**
- Consumes: `nets.DenseNet` (`layers` = list of `(W (a, b), bias (b,))`, `n_params`); `neat.NeatNet` (attribute `genome`, `n_params`), `neat.Genome` fields `node_ids, bias, innov, src, dst, weight, enabled`, `neat.node_depths(genome)`, `neat.INPUT_IDS`, `neat.OUTPUT_IDS`, `neat.FIRST_HIDDEN` (Tasks 7–8); `config.RAY_ANGLES`, `config.STEER_LEVELS`, `config.THROTTLE_LEVELS`, `config.N_ACTIONS`, `config.APPROACHES`, `config.LABELS`; `evaluate.main` from Task 12.
- Produces: `netfig.input_labels() -> list[str]`, `netfig.output_labels() -> list[str]`, `netfig.network_graph(driver) -> {"columns": [[label, ...], ...], "edges": [(col_a, row_i, col_b, row_j, weight), ...]}`, `netfig.node_positions(columns)`, `netfig.draw_networks(drivers, path) -> Path` (drivers: `{approach: driver}`; raises `ValueError` when empty). `evaluate.main(...)["figures"]` ends with `<out_dir>/figures/networks.png` when a best seed exists.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_netfig.py`:

```python
import numpy as np
import pytest

from racing import neat, netfig, nets


def dense_driver():
    return nets.from_flat(nets.init_population(np.random.default_rng(0), 1)[0])


def neat_driver():
    """Input 0 -> hidden 25 -> output 10, input 1 -> output 11, and a disabled input 2 -> output 12."""
    genome = neat.Genome(node_ids=np.array([*neat.OUTPUT_IDS, 25], dtype=np.int64), bias=np.zeros(16),
                         innov=np.array([0, 1, 2, 3], dtype=np.int64), src=np.array([0, 25, 1, 2], dtype=np.int64),
                         dst=np.array([25, 10, 11, 12], dtype=np.int64), weight=np.array([1.0, -2.0, 0.5, 0.3]),
                         enabled=np.array([True, True, True, False]))
    return neat.NeatNet(genome)


def test_labels_name_the_rays_speed_and_actions():
    inputs = netfig.input_labels()
    assert len(inputs) == 10 and inputs[0] == "−90°" and inputs[4] == "0°" and inputs[8] == "+90°"
    assert inputs[9] == "speed"
    outputs = netfig.output_labels()
    assert len(outputs) == 15 and outputs[0] == "R1 brake" and outputs[8] == "S gas" and outputs[14] == "L1 gas"


def test_dense_graph_matches_the_network():
    driver = dense_driver()
    graph = netfig.network_graph(driver)
    assert [len(c) for c in graph["columns"]] == [10, 16, 15] and len(graph["edges"]) == 10 * 16 + 16 * 15
    weights = {(a, i, b, j): w for a, i, b, j, w in graph["edges"]}
    assert weights[(0, 3, 1, 7)] == driver.layers[0][0][3, 7] and weights[(1, 5, 2, 14)] == driver.layers[1][0][5, 14]


def test_neat_graph_places_hidden_nodes_by_depth_and_skips_disabled_connections():
    graph = netfig.network_graph(neat_driver())
    assert [len(c) for c in graph["columns"]] == [10, 1, 15]
    assert sorted(graph["edges"]) == sorted([(0, 0, 1, 0, 1.0), (1, 0, 2, 0, -2.0), (0, 1, 2, 1, 0.5)])


def test_node_positions_span_the_panel():
    xy = netfig.node_positions([["a", "b"], ["c"], ["d", "e", "f"]])
    assert [x for x, _ in xy[0]] == [0.0, 0.0] and xy[2][0][0] == 1.0 and xy[1][0] == (0.5, 0.5)
    assert all(0.0 < y < 1.0 for column in xy for _, y in column)


def test_draw_networks_writes_a_png(tmp_path):
    path = netfig.draw_networks({"neat": neat_driver(), "ga": dense_driver()}, tmp_path / "figures" / "networks.png")
    assert path == tmp_path / "figures" / "networks.png" and path.read_bytes()[:4] == b"\x89PNG"


def test_draw_networks_without_controllers_raises(tmp_path):
    with pytest.raises(ValueError):
        netfig.draw_networks({}, tmp_path / "networks.png")
```

In `tests/test_evaluate.py`, add one line at the end of `test_main_with_partial_results`:

```python
    assert results["figures"][-1] == tmp_path / "out" / "figures" / "networks.png" and results["figures"][-1].exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_netfig.py tests/test_evaluate.py::test_main_with_partial_results -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'racing.netfig'`, and the `networks.png` assertion in `test_main_with_partial_results`.

- [ ] **Step 3: Write the implementation**

Create `racing/netfig.py`:

```python
"""Network figure (revision 3 §7.3): each method's best final controller drawn as a graph. The nine distance rays
and the speed input are on the left, hidden nodes in columns by depth, the 15 actions on the right. Connections are
blue when positive and red when negative, wider and more opaque when stronger; biases are not drawn."""
from pathlib import Path

import numpy as np

from racing import config

POSITIVE = "#2166AC"
NEGATIVE = "#B2182B"
STEER_NAMES = {-1.0: "R1", -0.5: "R½", 0.0: "S", 0.5: "L½", 1.0: "L1"}      # positive steering turns left
THROTTLE_NAMES = {-1.0: "brake", 0.0: "coast", 1.0: "gas"}


def input_labels():
    """The nine distance rays by angle (−90° points to the right, +90° to the left) and the speed input."""
    labels = []
    for angle in np.degrees(config.RAY_ANGLES):
        degrees = int(round(float(angle)))
        labels.append("0°" if degrees == 0 else f"{degrees:+d}°".replace("-", "−"))
    return labels + ["speed"]


def output_labels():
    """The 15 actions as steering and throttle, in action-class order."""
    n_throttle = len(config.THROTTLE_LEVELS)
    return [f"{STEER_NAMES[float(config.STEER_LEVELS[k // n_throttle])]} "
            f"{THROTTLE_NAMES[float(config.THROTTLE_LEVELS[k % n_throttle])]}" for k in range(config.N_ACTIONS)]


def _neat_graph(genome):
    """Columns and edges of a NEAT genome: inputs, hidden nodes grouped by depth, outputs; enabled connections only.
    A hidden node that no input reaches is drawn at depth 1."""
    from racing import neat
    depth = neat.node_depths(genome)
    hidden = [int(n) for n in genome.node_ids if int(n) >= neat.FIRST_HIDDEN]
    hidden_depth = {n: max(int(depth.get(n, 1)), 1) for n in hidden}
    column_of = {n: 0 for n in neat.INPUT_IDS}
    row_of = {n: k for k, n in enumerate(neat.INPUT_IDS)}
    columns = [input_labels()]
    for level in sorted(set(hidden_depth.values())):
        members = sorted(n for n in hidden if hidden_depth[n] == level)
        for k, n in enumerate(members):
            column_of[n], row_of[n] = len(columns), k
        columns.append([""] * len(members))
    for k, n in enumerate(neat.OUTPUT_IDS):
        column_of[n], row_of[n] = len(columns), k
    columns.append(output_labels())
    edges = [(column_of[int(s)], row_of[int(s)], column_of[int(d)], row_of[int(d)], float(w))
             for s, d, w, on in zip(genome.src, genome.dst, genome.weight, genome.enabled) if on]
    return {"columns": columns, "edges": edges}


def network_graph(driver):
    """Columns of node labels (inputs first, actions last; hidden nodes unlabelled) and edges
    (column a, row i, column b, row j, weight) of a dense or NEAT controller."""
    from racing import neat
    if isinstance(driver, neat.NeatNet):
        return _neat_graph(driver.genome)
    columns = [input_labels()] + [[""] * w.shape[1] for w, _ in driver.layers[:-1]] + [output_labels()]
    edges = []
    for k, (weights, _) in enumerate(driver.layers):
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                edges.append((k, i, k + 1, j, float(weights[i, j])))
    return {"columns": columns, "edges": edges}


def node_positions(columns):
    """(x, y) of every node: columns evenly spaced from x = 0 to 1, nodes evenly spaced within their column."""
    last = max(len(columns) - 1, 1)
    return [[(c / last, 1.0 - (k + 0.5) / len(labels)) for k in range(len(labels))]
            for c, labels in enumerate(columns)]


def _draw_graph(ax, graph):
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    columns, edges = graph["columns"], graph["edges"]
    xy = node_positions(columns)
    if edges:
        weights = np.array([e[4] for e in edges])
        scale = np.abs(weights) / max(float(np.abs(weights).max()), 1e-12)     # per panel
        order = np.argsort(scale, kind="stable")                              # strongest drawn last, on top
        segments = [(xy[edges[k][0]][edges[k][1]], xy[edges[k][2]][edges[k][3]]) for k in order]
        colors = [(*mcolors.to_rgb(POSITIVE if weights[k] >= 0 else NEGATIVE), 0.12 + 0.78 * scale[k]) for k in order]
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.2 + 2.0 * scale[order]))
    for column in xy:
        xs, ys = zip(*column)
        ax.scatter(xs, ys, s=10 if len(column) <= 20 else 4, color="0.25", zorder=3)
    for (x, y), label in zip(xy[0], columns[0]):
        ax.text(x - 0.03, y, label, ha="right", va="center", fontsize=6, color="0.15")
    for (x, y), label in zip(xy[-1], columns[-1]):
        ax.text(x + 0.03, y, label, ha="left", va="center", fontsize=6, color="0.15")
    ax.set_xlim(-0.25, 1.3)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def draw_networks(drivers, path):
    """One panel per method in config.APPROACHES order, titled with the method and its parameter count, written as
    a PNG at `path`. drivers: {approach: driver}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    approaches = [a for a in config.APPROACHES if a in drivers]
    if not approaches:
        raise ValueError("no controllers to draw")
    ncols = min(4, len(approaches))
    nrows = -(-len(approaches) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.8 * nrows), squeeze=False)
    for ax in axes.flat[len(approaches):]:
        ax.set_visible(False)
    for ax, approach in zip(axes.flat, approaches):
        _draw_graph(ax, network_graph(drivers[approach]))
        ax.set_title(f"{config.LABELS[approach]} · {drivers[approach].n_params:,} parameters", fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
```

In `racing/evaluate.py`, change the import line

```python
from racing import config, nets, paths, sim
```

to

```python
from racing import config, netfig, nets, paths, sim
```

and replace the `return` statement at the end of `main` with:

```python
    drawn = figures(runs_dir, out_dir / "figures")
    if best:                                            # each method's best final controller (revision 3 §7.3)
        drawn.append(netfig.draw_networks({approach: drivers[(approach, seed)] for approach, seed in best.items()},
                                          out_dir / "figures" / "networks.png"))
    return {**tables, "per_track": per_track, "validation": validation, "metrics": metrics, "best_seeds": best,
            "figures": drawn}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_netfig.py tests/test_evaluate.py -q`
Expected: PASS.

- [ ] **Step 5: Look at the figure**

Run: `~/miniconda3/bin/python -c "from tests.test_netfig import dense_driver, neat_driver; from racing import netfig; print(netfig.draw_networks({'ga': dense_driver(), 'neat': neat_driver()}, '/tmp/networks-check.png'))"` and open the printed PNG (Read tool). Check: ray labels on the left and action labels on the right are legible and do not overlap the panel title; the NEAT panel shows one hidden node between the columns; blue and red lines are distinguishable. Fix sizes in `_draw_graph`/`draw_networks` if not, then rerun Step 4.

- [ ] **Step 6: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add racing/netfig.py racing/evaluate.py tests/test_netfig.py tests/test_evaluate.py
git commit -m "Network figure: each method's best controller as a graph, dense and NEAT, in the evaluation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/netfig.py racing/evaluate.py tests/test_netfig.py tests/test_evaluate.py
```

---

### Task 14: Viewer

Brings `racing/app.py` to revision 3 §9: NEAT in the training and race views, training reports from the evolution engine (any population type), stats for the 90% rule, a plot of the lowest map success share, the seeded ES started from the DAgger-refined baseline, and fresh app runs on every Play.

**Files:**
- Modify: `racing/app.py` (`report_policy`, `trainer_main`, `stats_lines` and a new `share_text`, `App.__init__`, `App.start_training`, `App.receive`, `App.draw_train_panel`, `App.draw_plot`)
- Modify: `racing/config.py` (`POPULATION_PRESETS`)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes:
  - Evolution reports (Task 6): `approach, seed, generation, level, factor, target_reduction, streak, seeds, population, fitness, all_success, is_kept, map_shares, min_share, kept_success, child_success, median_lap, step_size, best, mean, worst, evals, elapsed, event`, plus `species` for NEAT (`NEAT.log_fields()`, Task 9). `report["population"]` has `subset(indices)` and `policy()` (`nets.DensePopulation`, `neat.NeatPopulation`).
  - `ga.run_ga(seed=, pop_size=, out_dir=, champion_dir=, n_jobs=, on_generation=, start_level=, max_generations=)`, `es.run_es(..., initial=, approach=)`, `es.dagger_baseline(seed, runs_dir=None)`, `neat.run_neat(...)` (Tasks 6, 9); all resume from a checkpoint in `out_dir`, so the viewer removes `out_dir` first.
  - `paths.stage_path(approach, seed, stage, runs_dir)` (Task 10); `nets.load_champion` handles NEAT (Task 8); `population.save_dense_champion`, `neat.save_neat_champion(approach, seed, genome, champion_dir)`, `neat.initial_genome`, `neat.Innovations`, `neat.NeatPopulation` (tests).
  - `imitation.train_baseline(data, out_dir=, seed=, on_epoch=, resume=False)` and `imitation.run_tuner(approach, data, out_dir=, on_trial=, seed=, export_models=False, resume=False)` (Task 10): called without `champion_dir` and with `resume=False`, so a completed reported run is never skipped and training is always shown live.
  - `config.ES_POP`, `config.NEAT_POP`, `config.MASTERY_FRACTION`, `config.MASTERY_STREAK`, `config.TARGET_FACTOR`, `config.RUNS_DIR`.
- Produces: `app.report_policy(report, indices)`, `app.stats_lines(report, approach) -> tuple[str, ...]`, `app.share_text(shares) -> str`, `App.runs_dir`; `config.POPULATION_PRESETS = (100, 250, 550, 1000, 2000)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_app.py`:

1. Change the import line `from racing import config, nets, paths, population` to:

```python
from racing import config, neat, nets, paths, population
```

2. Replace the helper `ga_report` with:

```python
def ga_report(generation=0, n=30, approach="ga", population_=None, **extra):
    """A training report as the evolution engine sends it (revision 3 §9)."""
    rng = np.random.default_rng(generation)
    fitness = rng.random(n)
    if population_ is None:
        population_ = nets.DensePopulation(nets.init_population(rng, n))
    shares = np.array([0.84, 0.92, 0.95])
    report = {"approach": approach, "seed": 0, "generation": generation, "level": 1,
              "factor": config.TARGET_FACTOR, "target_reduction": float("nan"), "streak": 1, "seeds": [0, 1, 2],
              "population": population_, "fitness": fitness, "all_success": np.zeros(n, dtype=bool),
              "is_kept": np.zeros(n, dtype=bool), "map_shares": shares, "min_share": float(shares.min()),
              "kept_success": float("nan"), "child_success": 0.0, "median_lap": float("nan"), "step_size": 0.5,
              "best": float(fitness.max()), "mean": float(fitness.mean()), "worst": float(fitness.min()),
              "evals": n, "elapsed": 1.0, "event": None}
    report.update(extra)
    return report


def neat_report(generation=0, n=20):
    rng, innovations = np.random.default_rng(generation), neat.Innovations()
    genomes = [neat.initial_genome(rng, innovations) for _ in range(n)]
    return ga_report(generation, n, approach="neat", population_=neat.NeatPopulation(genomes), species=3)
```

3. Replace `test_training_stats_fit_the_panel` with:

```python
def test_training_stats_fit_the_panel(window):
    window.receive(ga_report())
    window.update(1 / 60)
    widest = dict(level=4, generation=123456, map_shares=np.ones(3), min_share=1.0, streak=3, target_reduction=0.185,
                  step_size=0.0505, best=1.5843, mean=1.2329, species=20, is_kept=np.ones(1000, dtype=bool))
    for approach in config.EVOLUTION_APPROACHES:
        report = ga_report(approach=approach, **widest)
        for line in viewer.stats_lines(report, approach):
            assert window.small.size(line)[0] <= window.panel.width - 32, line
    imitation = {"generation": 149, "best": 0.7046, "mean": 0.6123, "level": 4}
    for approach in config.IMITATION_APPROACHES:
        for line in viewer.stats_lines(imitation, approach):
            assert window.small.size(line)[0] <= window.panel.width - 32, line


def test_stats_lines_show_maps_streak_and_target():
    lines = viewer.stats_lines(ga_report(level=4, generation=412, streak=1, target_reduction=0.143,
                                         map_shares=np.array([0.84, 0.92, 0.95])), "ga")
    assert lines[0] == "Extra hard    Generation 412"
    assert lines[1] == "Maps succeeded: 84% 92% 95%  (need 90%)"
    assert lines[2] == "Streak 1/3    Target cut mastered: −14.3%"
    assert lines[3].startswith("Mutation strength: 0.5")
    assert viewer.stats_lines(ga_report(), "es")[2].endswith("mastered: –")
    assert viewer.stats_lines(neat_report(), "neat")[4].endswith("Species 3")
```

4. Add after `test_app_plays_a_report`:

```python
def test_neat_report_plays_in_the_training_view(window):
    window.approach = "neat"
    window.receive(neat_report())
    for _ in range(60):
        window.update(1 / 60)
        window.draw()
    assert window.playback is not None and window.playback.cars.laps == config.LAPS


def test_history_marks_level_ups_and_target_cuts(window):
    for generation, event in enumerate((None, "level_up", None, "goal", "tightened", "end_speed")):
        window.receive(ga_report(generation, event=event))
    assert [h[3] for h in window.history] == [False, True, False, True, True, False]
    assert [h[1] for h in window.history] == [0.84] * 6
    window.draw()                                        # the success-share plot draws with markers
```

5. Replace `test_trainer_main_uses_app_runs_dir_and_never_saves_champions` with:

```python
def test_trainer_main_uses_app_runs_dir_and_never_saves_champions(monkeypatch, tmp_path):
    from racing import es, ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    calls = {}

    def fake(name):
        def run(**kwargs):
            calls[name] = dict(kwargs, existed=(kwargs["out_dir"] / "checkpoint.pkl").exists())
            kwargs["on_generation"](ga_report(0))
        return run

    monkeypatch.setattr(ga, "run_ga", fake("ga"))
    monkeypatch.setattr(es, "run_es", fake("es"))
    monkeypatch.setattr(neat, "run_neat", fake("neat"))
    stale = tmp_path / "ga" / "seed0" / "checkpoint.pkl"          # left by an earlier Play
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    reports, running = queue.Queue(), threading.Event()
    running.set()
    viewer.trainer_main("ga", 30, 0, 1, reports, running)
    viewer.trainer_main("es", 12, 0, 1, reports, running)
    viewer.trainer_main("neat", 100, 0, 1, reports, running)
    for name, size in (("ga", 30), ("es", 12), ("neat", 100)):
        assert calls[name]["champion_dir"] is None and calls[name]["pop_size"] == size
        assert calls[name]["max_generations"] is None and calls[name]["out_dir"] == tmp_path / name / "seed0"
    assert not calls["ga"]["existed"]                              # every Play starts a new run
    assert [reports.get_nowait()["generation"] for _ in range(3)] == [0, 0, 0] and reports.empty()
```

6. In `test_trainer_main_passes_seed_level_and_seeded_start`, replace the line

```python
    monkeypatch.setattr(es, "trained_baseline", lambda seed: f"baseline seed {seed}")
```

with

```python
    monkeypatch.setattr(es, "dagger_baseline", lambda seed: f"baseline seed {seed}")
```

7. Replace `save_race_champions` with:

```python
def save_race_champions(directory, best=None, approaches=(("es", 0), ("ga", 0), ("ga", 1))):
    rng = np.random.default_rng(0)
    for approach, seed in approaches:
        if approach == "neat":
            neat.save_neat_champion(approach, seed, neat.initial_genome(rng, neat.Innovations()), directory)
        else:
            population.save_dense_champion(approach, seed, nets.DensePopulation(nets.init_population(rng, 1)), 0,
                                           directory)
    if best is not None:
        (directory / "best_seeds.json").write_text(json.dumps(best))
```

8. Add after `test_race_panel_text_fits_the_panel`:

```python
def test_a_race_of_every_method_fits_the_panel(window, tmp_path, monkeypatch):
    save_race_champions(tmp_path, approaches=[(approach, 0) for approach in config.APPROACHES])
    window.champions_dir = tmp_path
    window.set_mode("race")
    window.update(1 / 60)
    window.start_race()
    window.update(1 / 60)
    assert [approach for approach, _ in window.lineup] == list(config.APPROACHES)
    drawn, text = [], window.text

    def recording_text(string, x, y, font=None, color=viewer.INK):
        drawn.append((string, y, font or window.font))
        return text(string, x, y, font, color)

    monkeypatch.setattr(window, "text", recording_text)
    window.buttons = []
    window.draw_race_panel(window.panel.left + 16, 58, window.panel.width - 32)
    assert sum(string.startswith(f"{k}. ") for string, _, _ in drawn for k in range(1, 9)) == 8
    for string, y, font in drawn:
        assert y + font.get_linesize() <= window.panel.bottom - 30, string
    assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons)
```

9. Replace `test_seeded_es_without_baseline_shows_a_message` with:

```python
def test_seeded_es_without_the_dagger_baseline_shows_a_message(window, tmp_path):
    window.runs_dir = tmp_path
    window.champions_dir = tmp_path                      # the old check never finds a controller: no real process
    window.select_approach("es_seeded")
    window.start_training()
    assert window.process is None and "notebook 03" in window.status and len(window.status) <= 56


def test_seeded_es_starts_with_the_dagger_baseline(window, tmp_path, monkeypatch):
    created = {}

    class FakeProcess:
        def __init__(self, target, args, daemon):
            created.update(args=args)
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
    stage = paths.stage_path("baseline", config.RUN_SEEDS[0], "dagger", tmp_path)
    stage.parent.mkdir(parents=True)
    stage.write_bytes(b"")
    window.runs_dir = tmp_path
    window.select_approach("es_seeded")
    window.start_training()
    assert created["args"][:2] == ("es_seeded", config.ES_POP)
```

10. Add after `test_trainer_main_uses_the_reported_dataset`:

```python
def test_viewer_imitation_training_never_resumes(monkeypatch, tmp_path):
    from racing import imitation
    fake_imitation(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(imitation, "train_baseline",
                        lambda data, **kwargs: calls.append(("baseline", kwargs["resume"], kwargs.get("champion_dir"))))
    monkeypatch.setattr(imitation, "run_tuner",
                        lambda approach, data, **kwargs: calls.append((approach, kwargs["resume"], None)))
    running = threading.Event()
    running.set()
    viewer.trainer_main("baseline", 0, 0, 1, queue.Queue(), running)
    viewer.trainer_main("kt_random", 0, 0, 1, queue.Queue(), running)
    assert calls == [("baseline", False, None), ("kt_random", False, None)]
```

(Task 10 already made both imitation calls pass `resume=False`, so this regression test passes from the start.)

11. Delete `test_a_report_with_a_population_plays_back` (added in Task 6): it builds its report with `report.pop("weights")`, which the new `ga_report` no longer has; every report now carries a population, so `test_training_playback_drives_all_laps` covers it. Keep Task 6's `test_trainer_main_starts_each_viewer_run_afresh` and Task 10's `test_trainer_main_trains_imitation_afresh_and_saves_no_final_controller` unchanged.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/miniconda3/bin/python -m pytest tests/test_app.py -q`
Expected: FAIL — `stats_lines` raises `KeyError: 'kept_finish'` for the new reports, `trainer_main` has no NEAT branch and does not call `es.dagger_baseline`, `App` has no `runs_dir` (the seeded-ES tests), and `App.population` has no `"neat"` entry.

- [ ] **Step 3: Write the implementation**

In `racing/config.py`, replace

```python
POPULATION_PRESETS = (100, 250, 500, 1000, 2000)
```

with

```python
POPULATION_PRESETS = (100, 250, 550, 1000, 2000)    # 550 is the ES population (revision 3 §5.2)
```

In `racing/app.py`:

Replace `report_policy` with:

```python
def report_policy(report, indices):
    """Policy for the chosen members of a training report: an evolutionary population (dense or NEAT, revision 3
    §9) or the networks of an imitation report."""
    if "population" in report:
        return report["population"].subset(indices).policy()
    return nets.mixed_policy([report["nets"][i] for i in indices])
```

Replace `trainer_main` with:

```python
def trainer_main(approach, size, seed, level, reports, running):
    """Training process: trains one approach with run seed `seed`, starting at curriculum level `level`, and sends a
    report after every generation, trial or epoch. Every Play starts a new run: the approach's folder in
    config.APP_RUNS_DIR is removed first, since evolutionary runs resume from a checkpoint they find there. Writes
    only to config.APP_RUNS_DIR, never saves final controllers and trains until stopped (revision 3 §9)."""
    from racing import es, ga, imitation, neat
    from racing.expert import collect_dataset

    def send(report):
        reports.put(report)
        running.wait()                      # training holds here while the viewer is paused

    out_dir = config.APP_RUNS_DIR / approach / f"seed{seed}"
    paths.remove(out_dir)
    common = dict(seed=seed, pop_size=size, start_level=level, out_dir=out_dir, on_generation=send,
                  champion_dir=None, n_jobs=config.N_JOBS, max_generations=None)
    if approach == "ga":
        ga.run_ga(**common)
        return
    if approach in ("es", "es_seeded"):
        initial = es.dagger_baseline(seed) if approach == "es_seeded" else None
        es.run_es(**common, initial=initial, approach=approach)
        return
    if approach == "neat":
        neat.run_neat(**common)
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
```

Replace `stats_lines` (keep `percent` and `number` above it) with:

```python
def share_text(shares):
    """Success shares of the generation's maps, e.g. '84% 92% 95%'."""
    return " ".join(percent(float(s)) for s in shares)


def stats_lines(report, approach):
    """Training panel stats lines (revision 3 §9). Evolutionary reports: difficulty and generation, each map's
    success share against the 90% requirement, the streak and the target cut mastered, the step size and the kept
    cars, the fitness (and NEAT's species). Imitation reports: the epoch or trial and the validation accuracy."""
    if "min_share" not in report:
        unit = "Trial" if approach.startswith("kt") else "Epoch"
        return (f"{unit} {report['generation']}",
                f"Validation accuracy: best {report['best']:.3f}  mean {report['mean']:.3f}")
    level = config.DIFFICULTIES.get(report["level"], f"Level {report['level']}")
    reduction = float(report.get("target_reduction", float("nan")))
    mastered = "–" if np.isnan(reduction) else f"−{100 * reduction:.1f}%"
    size_name = "Mutation strength" if approach == "ga" else "σ"
    last = f"Best {report['best']:.3f}    Mean {report['mean']:.3f}"
    if "species" in report:
        last += f"    Species {report['species']}"
    return (f"{level}    Generation {report['generation']}",
            f"Maps succeeded: {share_text(report['map_shares'])}  (need {percent(config.MASTERY_FRACTION)})",
            f"Streak {report['streak']}/{config.MASTERY_STREAK}    Target cut mastered: {mastered}",
            f"{size_name}: {number(report['step_size'])}    Kept {int(np.sum(report['is_kept']))}",
            last)
```

In `App.__init__`, replace (as Task 6 left it)

```python
        self.population = {"ga": config.GA_POP, "es": config.ES_POP, "es_seeded": config.ES_POP}
```

with

```python
        self.population = {"ga": config.GA_POP, "es": config.ES_POP, "es_seeded": config.ES_POP,
                           "neat": config.NEAT_POP}
        self.runs_dir = config.RUNS_DIR     # where the seeded ES finds the DAgger-refined baseline
```

In `App.start_training`, replace the seeded-ES check

```python
        if self.approach == "es_seeded" and not paths.champion_path("baseline", self.run_seed,
                                                                   champion_dir=self.champions_dir).exists():
            self.status = "Seeded ES needs the baseline controller: run notebook 02 first."
            return
```

with

```python
        if self.approach == "es_seeded" and not paths.stage_path("baseline", self.run_seed, "dagger",
                                                                 self.runs_dir).exists():
            self.status = "Seeded ES needs the DAgger baseline: run notebook 03."
            return
```

Replace `App.receive` with:

```python
    def receive(self, report):
        """Show a report; the plot follows the lowest map success share (evolution, with level advancements, the
        mastery goal and target cuts marked) or the validation accuracy (imitation)."""
        self.latest = report
        if "min_share" in report:
            marked = report.get("event") in ("level_up", "goal", "tightened")
            self.history.append((report["generation"], float(report["min_share"]), float("nan"), marked))
        else:
            self.history.append((report["generation"], report["best"], report["mean"], report["leveled_up"]))
```

In `App.draw_train_panel`, replace

```python
            self.text("Population" if self.approach == "ga" else "Population (λ)", x, y + 5, self.small, MUTED)
```

with

```python
            self.text("Population", x, y + 5, self.small, MUTED)
```

Replace `App.draw_plot` with:

```python
    def draw_plot(self, rect):
        pygame.draw.rect(self.screen, (255, 255, 255), rect)
        pygame.draw.rect(self.screen, BUTTON, rect, 1)
        shares = self.latest is not None and "min_share" in self.latest
        title = (f"Lowest map success share, {len(self.history)} generations" if shares
                 else f"Best and mean validation accuracy, {len(self.history)} reports")
        self.text(title, rect.left + 6, rect.top + 4, self.small, MUTED)
        if len(self.history) < 2:
            return
        gens = np.array([h[0] for h in self.history], dtype=float)
        main = np.array([h[1] for h in self.history], dtype=float)
        second = np.array([h[2] for h in self.history], dtype=float)
        if shares:
            lo, hi = 0.0, 1.0
        else:
            values = np.concatenate([main, second[~np.isnan(second)]])
            lo, hi = float(values.min()), float(values.max())
        span = hi - lo if hi > lo else 1.0
        width = max(gens[-1] - gens[0], 1.0)

        def point(g, v):
            return (rect.left + 6 + (g - gens[0]) / width * (rect.width - 12),
                    rect.bottom - 6 - (v - lo) / span * (rect.height - 30))

        for g, _, _, marked in self.history:
            if marked:
                px = point(g, lo)[0]
                pygame.draw.line(self.screen, MUTED, (px, rect.top + 22), (px, rect.bottom - 2), 1)
        if shares:                                          # the 90% requirement
            y = point(gens[0], config.MASTERY_FRACTION)[1]
            pygame.draw.line(self.screen, BUTTON_ACTIVE, (rect.left + 6, y), (rect.right - 6, y), 1)
        if not np.isnan(second).all():
            pygame.draw.lines(self.screen, MUTED, False, [point(g, v) for g, v in zip(gens, second)], 1)
        pygame.draw.lines(self.screen, hex_to_rgb(config.COLORS[self.approach]), False,
                          [point(g, v) for g, v in zip(gens, main)], 2)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest tests/test_app.py -q`
Expected: PASS (all tests, including the unchanged leaderboard, panel-fit, pause, process-group and background-track tests).

- [ ] **Step 5: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: PASS. `grep -rn "ES_LAMBDA\|trained_baseline\|kept_finish\|whole_pace" racing tests scripts` finds nothing in `racing/app.py` or `tests/test_app.py`.

- [ ] **Step 6: Look at the viewer**

Run: `~/miniconda3/bin/python -m racing.app`, choose NEAT, Easy, population 100 and press Play; after a few generations check that the cars drive, the stats show the three map shares against 90% and "Species", the plot shows the lowest share with its 90% line, and Stop ends training. Switch to Race: every method with a saved controller appears in the controller list and the leaderboard fits. Close the window.

- [ ] **Step 7: Commit**

```bash
git add racing/app.py racing/config.py tests/test_app.py
git commit -m "Viewer for revision 3: NEAT, engine reports, 90% map stats and success-share plot, seeded ES from the DAgger baseline, fresh app runs" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/app.py racing/config.py tests/test_app.py
```

---

### Task 15: Runner, notebooks, bench and worker count

Resume-by-default experiment runner with `--fresh`, a settings check before anything runs, the live progress
monitor, eight notebooks (the revision 2.0 notebooks are archived first), a worker count read from
`EC_RACING_JOBS`, and a generation benchmark that covers a 24-thread desktop and prints its recommendation
(revision 3 §8.1–8.4).

**Files:**
- Modify: `racing/config.py` (worker count; `STATUS_PATH` and `RUN_ORDER` unless an earlier task added them)
- Modify: `racing/bench.py` (job counts, recommendation)
- Modify: `scripts/run_notebooks.py` (whole file)
- Modify: `scripts/make_notebooks.py` (whole file)
- Regenerate: `notebooks/01_dataset.ipynb` … `notebooks/08_evaluate.ipynb`; delete `notebooks/03_ga.ipynb`, `notebooks/04_es.ipynb`, `notebooks/05_es_seeded.ipynb`, `notebooks/06_evaluate.ipynb`
- Create: `tests/test_config.py`
- Modify: `tests/test_bench.py`, `tests/test_runner.py`

**Interfaces:**
- Consumes: `checkpoint.settings_record()`, `checkpoint.check_settings(recorded)`, `checkpoint.SettingsMismatch`,
  `checkpoint.load(path)`, `checkpoint.CHECKPOINT_NAME` (Task 4); `progress.milestone(message, path)`,
  `progress.Monitor(status_path, log_path, out=...)` with `start()`/`stop()` (Task 5); `paths.remove`;
  in the notebooks: `expert.collect_dataset`, `imitation.load_dataset/train_baseline/run_tuner/retrain` (Task 10),
  `dagger.run(approach, seed, data, n_jobs=...)` (Task 11), `ga.run_ga`, `es.run_es`, `es.dagger_baseline` (Task 6),
  `neat.run_neat` (Task 9), `evaluate.main(n_jobs=...)` returning the keys `results`, `head_to_head`,
  `training_cost`, `stages`, `comparison`, `search_cost`, `driving_selected`, `dataset`, `figures` (Task 12).
- Produces: `config.worker_count(environ=os.environ) -> int`; `config.N_JOBS`; `config.STATUS_PATH`;
  `config.RUN_ORDER`; `bench.job_counts(cpus=None) -> list[int]`; `bench.recommend(times) -> int`;
  `bench.generation_main() -> dict[int, float]`; `run_notebooks.fresh_start(results_dir)`;
  `run_notebooks.settings_problems(runs_dir) -> list[str]`; `run_notebooks.run(notebook_dir, start, log_path,
  info_path, status_path, monitor, out) -> int`; `run_notebooks.main(argv, notebook_dir, log_path, info_path,
  results_dir, status_path, monitor, out) -> int` with exit codes 0 finished, 1 notebook failed, 2 invalid options,
  3 settings differ; `make_notebooks.ARCHIVE`, `make_notebooks.archive_once() -> list[str]`, eight `NOTEBOOKS`.

- [ ] **Step 1: Write the failing configuration tests**

Create `tests/test_config.py`:

```python
import pytest

from racing import config


def test_worker_count_reads_the_environment():
    assert config.worker_count({}) == 6
    assert config.worker_count({"EC_RACING_JOBS": ""}) == 6
    assert config.worker_count({"EC_RACING_JOBS": "16"}) == 16
    assert config.worker_count({"EC_RACING_JOBS": " 12 "}) == 12


@pytest.mark.parametrize("value", ["0", "-2", "four", "1.5"])
def test_worker_count_rejects_values_that_are_not_a_positive_whole_number(value):
    with pytest.raises(ValueError, match="EC_RACING_JOBS"):
        config.worker_count({"EC_RACING_JOBS": value})


def test_run_order_lists_the_24_counted_runs():
    assert len(config.RUN_ORDER) == 24
    assert config.RUN_ORDER[0] == ("baseline", 0) and config.RUN_ORDER[-1] == ("neat", 2)
    assert config.RUN_ORDER.index(("ga", 1)) == 13
    assert config.STATUS_PATH == config.RESULTS_DIR / "status.json"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_config.py`
Expected: FAIL with `AttributeError: module 'racing.config' has no attribute 'worker_count'` (the `RUN_ORDER`
test passes already if Task 5 added it).

- [ ] **Step 3: Read the worker count from the environment**

In `racing/config.py`, add `import os` above `from pathlib import Path`, and replace the line
`N_JOBS = 6                   # worker processes; racing/bench.py, ...` with:

```python
def worker_count(environ=os.environ):
    """Worker processes for parallel evaluation (revision 3 §8.4): EC_RACING_JOBS, or 6 when it is not set. The best
    value for a machine is measured with `python -m racing.bench generation`."""
    value = environ.get("EC_RACING_JOBS", "").strip()
    if not value:
        return 6
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"EC_RACING_JOBS must be a whole number of at least 1, not {value!r}")
    return int(value)


N_JOBS = worker_count()      # results do not depend on it (spec §9.5); recorded in results/run_info.json
```

If `STATUS_PATH` and `RUN_ORDER` are not yet in `racing/config.py` (`grep -n "RUN_ORDER\|STATUS_PATH" racing/config.py`
prints nothing), add `RUN_ORDER` after `COLORS` and `STATUS_PATH` after `RUN_LOG_PATH`:

```python
RUN_ORDER = tuple((approach, seed) for approach in (*IMITATION_APPROACHES, *EVOLUTION_APPROACHES)
                  for seed in RUN_SEEDS)     # the 24 runs counted by the progress display (revision 3 §8.3)
```

```python
STATUS_PATH = RESULTS_DIR / "status.json"
```

- [ ] **Step 4: Run the configuration tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_config.py`
Expected: 6 passed.

- [ ] **Step 5: Write the failing benchmark tests**

Append to `tests/test_bench.py`:

```python
def test_job_counts_stop_at_the_number_of_logical_processors():
    assert bench.job_counts(24) == [1, 4, 6, 8, 12, 16, 20, 24]
    assert bench.job_counts(20) == [1, 4, 6, 8, 12, 16, 20]
    assert bench.job_counts(1) == [1]


def test_recommend_prefers_fewer_workers_on_a_near_tie():
    assert bench.recommend({1: 6.9, 6: 1.90, 12: 1.89, 16: 2.5}) == 6
    assert bench.recommend({1: 6.9, 6: 2.4, 12: 1.9}) == 12


def test_generation_main_prints_the_command_for_the_fastest_count(monkeypatch, capsys):
    monkeypatch.setattr(bench, "get_track", lambda seed, level: None)
    monkeypatch.setattr(bench, "job_counts", lambda cpus=None: [1, 4])
    monkeypatch.setattr(bench, "generation_time", lambda pop, jobs, repeats=3: {1: 5.0, 4: 2.0}[jobs])
    assert bench.generation_main() == {1: 5.0, 4: 2.0}
    assert "export EC_RACING_JOBS=4" in capsys.readouterr().out
```

- [ ] **Step 6: Run them to verify they fail**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_bench.py -k "job_counts or recommend or generation_main"`
Expected: FAIL with `AttributeError: module 'racing.bench' has no attribute 'job_counts'`.

- [ ] **Step 7: Extend the benchmark**

In `racing/bench.py`: add `import os` to the imports; replace `JOB_COUNTS = (1, 3, 6, 9, 12, 15)` with
`JOB_COUNTS = (1, 4, 6, 8, 12, 16, 20, 24)      # up to a 24-thread desktop (revision 3 §8.4)`; in the module
docstring replace the first `Run with:` line by `Run with:  python -m racing.bench              (both)` and keep
the other two lines with `python` instead of `~/miniconda3/bin/python`; replace `generation_main` with:

```python
def job_counts(cpus=None):
    """The worker counts to time: JOB_COUNTS up to the number of logical processors."""
    if cpus is None:
        cpus = os.cpu_count() or 1
    return [jobs for jobs in JOB_COUNTS if jobs <= cpus]


def recommend(times):
    """The worker count with the shortest generation time; on a near tie (within 2%) the smaller count, which leaves
    the machine more headroom."""
    best = min(times.values())
    return min(jobs for jobs, seconds in times.items() if seconds <= best * 1.02)


def generation_main():
    """Times one worst-case generation for each worker count and prints the command that selects the fastest."""
    for seed in SEEDS:
        get_track(seed, config.MAX_LEVEL)
    print(f"One generation of {config.GA_POP} cars, every car completing a lap on 3 level-4 tracks:")
    times = {}
    for jobs in job_counts():
        times[jobs] = generation_time(config.GA_POP, jobs)
        print(f"  {jobs:2d} process(es): {times[jobs]:6.2f} s", flush=True)
    best = recommend(times)
    print(f"Fastest: {best} worker processes (current setting {config.N_JOBS}). Select it with:")
    print(f"  export EC_RACING_JOBS={best}")
    return times
```

- [ ] **Step 8: Run the benchmark tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_bench.py`
Expected: all passed.

- [ ] **Step 9: Write the failing runner tests**

In `tests/test_runner.py`:

1. Add `import io` to the imports.
2. Replace `run_main` with a version that keeps every test away from the real `results/`:

```python
def run_main(runner, tmp_path, argv, monitor=False, out=None):
    return runner.main(argv, notebook_dir=tmp_path / "notebooks", log_path=tmp_path / "log.txt",
                       info_path=tmp_path / "info.json", results_dir=tmp_path / "results",
                       status_path=tmp_path / "status.json", monitor=monitor, out=out)
```

3. Replace `test_run_record_holds_the_settings_and_code_version` with:

```python
def test_run_record_holds_the_settings_and_code_version(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir: None)
    assert run_main(runner, tmp_path, []) == 0
    info = json.loads((tmp_path / "info.json").read_text())
    assert info["settings"] == json.loads(json.dumps(runner.checkpoint.settings_record()))
    assert "N_JOBS" not in info["settings"]            # results do not depend on it; recorded as n_jobs
    assert info["n_jobs"] == runner.config.N_JOBS
    assert len(info["git_commit"]) == 40 and int(info["git_commit"], 16) >= 0
    assert isinstance(info["git_dirty"], bool)
```

4. Replace `test_make_notebooks_removes_only_the_obsolete_notebook` and
   `test_experiment_notebooks_are_valid_and_compile` with:

```python
def load_make_notebooks():
    path = SCRIPT.parent / "make_notebooks.py"
    spec = importlib.util.spec_from_file_location("make_notebooks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_notebooks_archives_once_then_replaces_the_obsolete_notebooks(tmp_path, monkeypatch):
    module = load_make_notebooks()
    folder, archive = tmp_path / "notebooks", tmp_path / "archive"
    monkeypatch.setattr(module, "FOLDER", folder)
    monkeypatch.setattr(module, "ARCHIVE", archive)
    folder.mkdir()
    for name in ("01_dataset.ipynb", "03_ga.ipynb", "06_evaluate.ipynb", "09_extra.ipynb", "notes.ipynb"):
        write_notebook(folder / name, f"print({name!r})")
    module.main()
    assert sorted(p.name for p in archive.iterdir()) == ["01_dataset.ipynb", "03_ga.ipynb", "06_evaluate.ipynb",
                                                         "09_extra.ipynb"]
    assert "03_ga" in nbformat.read(archive / "03_ga.ipynb", as_version=4).cells[0].source
    assert sorted(p.name for p in folder.glob("*.ipynb")) == sorted([*module.NOTEBOOKS, "09_extra.ipynb",
                                                                     "notes.ipynb"])
    module.main()                                       # a second rewrite leaves the archive as it was
    assert "01_dataset" in nbformat.read(archive / "01_dataset.ipynb", as_version=4).cells[0].source


def test_a_fresh_clone_without_revision_2_notebooks_archives_nothing(tmp_path, monkeypatch):
    module = load_make_notebooks()
    folder, archive = tmp_path / "notebooks", tmp_path / "archive"
    monkeypatch.setattr(module, "FOLDER", folder)
    monkeypatch.setattr(module, "ARCHIVE", archive)
    folder.mkdir()
    write_notebook(folder / "01_dataset.ipynb", "print('revision 3')")
    module.main()
    assert not archive.exists()


def test_experiment_notebooks_are_valid_and_compile():
    import ast
    folder = Path(__file__).resolve().parent.parent / "notebooks"
    names = sorted(p.name for p in folder.glob("*.ipynb"))
    assert names == ["01_dataset.ipynb", "02_imitation.ipynb", "03_dagger.ipynb", "04_ga.ipynb", "05_es.ipynb",
                     "06_es_seeded.ipynb", "07_neat.ipynb", "08_evaluate.ipynb"]
    for name in names:
        notebook = nbformat.read(folder / name, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
```

5. Add these tests (Review Focus item 3 is the first two):

```python
def changed_settings(runner):
    return {**runner.checkpoint.settings_record(), "TARGET_FACTOR": 1.2}


def test_results_from_other_settings_are_refused_until_fresh(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    executed = []
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir: executed.append(path.name))
    record = tmp_path / "results" / "runs" / "ga" / "seed0" / "summary.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"approach": "ga", "settings": changed_settings(runner)}))
    assert run_main(runner, tmp_path, []) == 3
    err = capsys.readouterr().err
    assert "TARGET_FACTOR" in err and "--fresh" in err and "ga/seed0/summary.json" in err
    assert executed == [] and not (tmp_path / "info.json").exists()
    assert run_main(runner, tmp_path, ["--fresh"]) == 0
    assert executed == ["01_a.ipynb", "02_b.ipynb", "03_c.ipynb"] and not record.exists()


def test_a_checkpoint_from_other_settings_is_refused(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir: pytest.fail("a notebook was run"))
    path = tmp_path / "results" / "runs" / "neat" / "seed2" / runner.checkpoint.CHECKPOINT_NAME
    path.parent.mkdir(parents=True)
    runner.checkpoint.save(path, {"generation": 7, "settings": changed_settings(runner)})
    assert run_main(runner, tmp_path, []) == 3
    assert "neat/seed2" in capsys.readouterr().err


def test_records_without_settings_are_refused_until_fresh(runner, tmp_path, monkeypatch, capsys):
    """checkpoint.completed refuses a record without settings, so the runner refuses it before notebook 01."""
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir: None)
    record = tmp_path / "results" / "runs" / "ga" / "seed0" / "summary.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"approach": "ga", "evals": 200000}))       # a revision 2.0 record
    assert run_main(runner, tmp_path, []) == 3
    assert "ga/seed0/summary.json" in capsys.readouterr().err
    assert run_main(runner, tmp_path, ["--fresh"]) == 0


def test_fresh_removes_results_but_keeps_pilots_viewer_runs_and_archive(runner, tmp_path):
    results = tmp_path / "results"
    for name in ("runs/ga/seed0/log.csv", "champions/ga/seed0.keras", "tables/a.md", "figures/a.png",
                 "pilots/summary.md", "app_runs/ga/seed0/log.csv", "archive/revision2-notebooks/01_dataset.ipynb",
                 "dataset.npz", "dataset_info.json", "status.json", "run_log.txt", "run_info.json"):
        (results / name).parent.mkdir(parents=True, exist_ok=True)
        (results / name).write_text("x")
    runner.fresh_start(results)
    assert sorted(p.name for p in results.iterdir()) == ["app_runs", "archive", "pilots"]
    runner.fresh_start(tmp_path / "missing")            # nothing to remove is not an error


def test_fresh_cannot_be_combined_with_from(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir: pytest.fail("a notebook was run"))
    assert run_main(runner, tmp_path, ["--fresh", "--from", "02"]) == 2
    assert "--fresh" in capsys.readouterr().err


def test_a_settings_mismatch_inside_a_notebook_stops_with_code_3(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)

    def stopped(path, kernel_dir):
        raise RuntimeError("SettingsMismatch: TARGET_FACTOR: recorded 1.2, current 1.3")

    monkeypatch.setattr(runner, "execute", stopped)
    assert run_main(runner, tmp_path, []) == 3
    assert "--fresh" in (tmp_path / "log.txt").read_text()


def test_the_monitor_prints_the_runner_milestones(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir: None)
    out = io.StringIO()
    assert run_main(runner, tmp_path, [], monitor=True, out=out) == 0
    text = out.getvalue()
    assert "run started" in text and "done 03_c.ipynb" in text and "run finished" in text
    assert text.index("start 01_a.ipynb") < text.index("run finished")


def test_a_rerun_keeps_the_record_of_earlier_notebooks(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir: None)
    assert run_main(runner, tmp_path, []) == 0
    assert run_main(runner, tmp_path, ["--from", "03"]) == 0
    info = json.loads((tmp_path / "info.json").read_text())
    assert [e["notebook"] for e in info["notebooks"]] == ["01_a.ipynb", "02_b.ipynb", "03_c.ipynb"]
    assert [r["from"] for r in info["runs"]] == [None, "03"]
```

6. In the four `@pytest.mark.slow` tests, replace every call
   `runner.run(folder, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json")` (and the one with
   `start="02"`) by the same call with the extra arguments `status_path=tmp_path / "status.json", monitor=False`,
   so slow tests never touch the real status file.

- [ ] **Step 10: Run them to verify they fail**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_runner.py`
Expected: FAIL — `TypeError: main() got an unexpected keyword argument 'results_dir'`, and the notebook-name test
fails on the six old notebooks.

- [ ] **Step 11: Rewrite the runner**

Replace `scripts/run_notebooks.py` with:

```python
"""Runs the experiment notebooks in order and saves their outputs into the notebook files (revision 3 §8).

Run with:        python scripts/run_notebooks.py               (continues where an earlier run stopped)
Start over:      python scripts/run_notebooks.py --fresh       (removes the earlier results first)
From a notebook: python scripts/run_notebooks.py --from 04
In background:   mkdir -p results
                 nohup python scripts/run_notebooks.py > results/run_console.txt 2>&1 &
Progress:        python -m racing.progress                     (from another terminal)

Exit codes: 0 finished; 1 a notebook failed; 2 invalid options; 3 earlier results were produced with different
settings (run again with --fresh).
"""
import argparse
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from racing import checkpoint, config, paths, progress  # noqa: E402

NOTEBOOK_DIR = PROJECT / "notebooks"
KERNEL = "ec-racing"
FRESH_REMOVE = ("runs", "champions", "tables", "figures", "dataset.npz", "dataset_info.json", "status.json",
                "run_log.txt", "run_info.json")       # under results/; pilots, app_runs and archive are kept
RECORD_FILES = ("summary.json", "dagger.json")        # run records that carry the settings they were produced with
STATE_FILES = (checkpoint.CHECKPOINT_NAME, "dagger_state.pkl")


def now():
    return datetime.now().isoformat(timespec="seconds")


def log(message, path):
    """A milestone of the run, appended to the run log; the progress monitor prints it (revision 3 §8.3)."""
    progress.milestone(message, path)


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


def git(*args):
    """Output of a git command in the project, or None without git or outside a repository."""
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def code_version():
    """The commit the run starts from, and whether tracked files differ from it ("unknown" without git)."""
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit.strip() if commit else "unknown",
            "git_dirty": bool(status.strip()) if status is not None else "unknown"}


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
        if manager.is_alive():
            manager.shutdown_kernel(now=True)          # a finished notebook's kernel does not stay resident


def write_info(info, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(info, indent=2))


def fresh_start(results_dir):
    """--fresh: remove the results of earlier runs so the experiment starts over. Pilots, viewer runs and the
    archive are kept."""
    paths.remove(*(Path(results_dir) / name for name in FRESH_REMOVE))


def recorded_settings(path):
    """The settings recorded in a run record or checkpoint, or None if it has none (an earlier version) or cannot be
    read."""
    try:
        if path.name in RECORD_FILES:
            data = json.loads(path.read_text())
        else:
            data = checkpoint.load(path)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, AttributeError, ImportError):
        return None
    return data.get("settings") if isinstance(data, dict) else None


def settings_problems(runs_dir):
    """One message per run record or checkpoint under runs_dir whose recorded settings differ from the current ones
    (revision 3 §8.2). A record without readable settings, such as one of revision 2.0, counts as different, as it
    does for checkpoint.completed, so the run stops here rather than inside a notebook."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    problems = []
    for path in sorted(runs_dir.rglob("*")):
        if path.name not in RECORD_FILES + STATE_FILES:
            continue
        recorded = recorded_settings(path)
        if recorded is None:
            problems.append(f"{path.relative_to(runs_dir).as_posix()}: no readable settings record "
                            "(made by an earlier version)")
            continue
        try:
            checkpoint.check_settings(recorded)
        except checkpoint.SettingsMismatch as error:
            problems.append(f"{path.relative_to(runs_dir).as_posix()}: {error}")
    return problems


def run(notebook_dir=NOTEBOOK_DIR, start=None, log_path=config.RUN_LOG_PATH, info_path=config.RUN_INFO_PATH,
        status_path=config.STATUS_PATH, monitor=True, out=None):
    """Execute the notebooks in order while the progress monitor shows the live status line and the milestones.
    The record keeps the entries of notebooks run earlier; each notebook's entry is replaced when it runs again."""
    from ipykernel.kernelspec import write_kernel_spec

    notebook_dir = Path(notebook_dir)
    info = json.loads(Path(info_path).read_text()) if Path(info_path).exists() else {}
    info.update({"processor": processor(), "logical_processors": os.cpu_count(), "n_jobs": config.N_JOBS,
                 "power_profile": power_profile(), "versions": versions(),
                 "settings": checkpoint.settings_record(), **code_version()})
    info.setdefault("notebooks", [])
    info.setdefault("runs", []).append({"started": now(), "from": start})
    names = notebook_names(notebook_dir, start)
    paths.remove(status_path)                          # the live line shows only this run's status
    watcher = progress.Monitor(status_path, log_path, out=out) if monitor else None
    if watcher is not None:
        watcher.start()
    try:
        log(f"run started: {' '.join(names)} ({config.N_JOBS} workers, power profile {info['power_profile']})",
            log_path)
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
                except Exception as error:             # a failing cell; its output is saved in the notebook
                    text = str(error).strip()
                    entry["failed"] = text.splitlines()[-1] if text else type(error).__name__
                    entry["seconds"] = round(time.time() - began, 1)
                    write_info(info, info_path)
                    if "SettingsMismatch" in text:
                        log(f"STOPPED {name}: results on disk were produced with different settings; "
                            f"run again with --fresh to start over", log_path)
                        return 3
                    log(f"FAILED {name}: {entry['failed']}", log_path)
                    return 1
                entry["finished"] = now()
                entry["seconds"] = round(time.time() - began, 1)
                write_info(info, info_path)
                log(f"done {name} in {entry['seconds'] / 60:.1f} min", log_path)
        log("run finished", log_path)
        return 0
    finally:
        if watcher is not None:
            watcher.stop()


def main(argv=None, notebook_dir=NOTEBOOK_DIR, log_path=config.RUN_LOG_PATH, info_path=config.RUN_INFO_PATH,
         results_dir=config.RESULTS_DIR, status_path=config.STATUS_PATH, monitor=True, out=None):
    parser = argparse.ArgumentParser(description="Run the EC Racing experiment notebooks in order. An interrupted "
                                                 "run continues where it stopped.")
    parser.add_argument("--from", dest="start", help="number of the first notebook to run, for example 04")
    parser.add_argument("--fresh", action="store_true", help="remove the earlier results and start over")
    args = parser.parse_args(argv)
    if args.start is not None:
        prefixes = sorted({name[:2] for name in notebook_names(notebook_dir)})
        if args.start not in prefixes:
            print(f"error: --from must be the number of an existing notebook: {', '.join(prefixes)}",
                  file=sys.stderr)
            return 2
        if args.fresh:
            print("error: --fresh starts over from the first notebook and cannot be combined with --from",
                  file=sys.stderr)
            return 2
    if args.fresh:
        fresh_start(results_dir)
    problems = settings_problems(Path(results_dir) / "runs")
    if problems:
        print("error: earlier results were produced with different settings:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("Run again with --fresh to remove them and start over.", file=sys.stderr)
        return 3
    return run(notebook_dir, args.start, log_path, info_path, status_path, monitor, out)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 12: Rewrite the notebook generator**

Replace `scripts/make_notebooks.py` with:

```python
"""Writes the eight experiment notebooks as plain nbformat-4 JSON (no extra dependency).

Run with:  python scripts/make_notebooks.py

Before the notebooks are first rewritten, the existing experiment notebooks, which hold the outputs of the revision
2.0 run, are copied to results/archive/revision2-notebooks/. Every run function skips completed runs and continues
interrupted ones (revision 3 §8.2), so re-executing a notebook continues where it stopped.
"""
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
FOLDER = PROJECT / "notebooks"
ARCHIVE = PROJECT / "results" / "archive" / "revision2-notebooks"
SETUP = "import sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path.cwd().parent))"

BRIEF = (
    "BRIEF_KEYS = ('approach', 'seed', 'level_reached', 'goal_reached', 'outcome', 'target_reduction',\n"
    "              'generations', 'rounds', 'val_successes', 'val_accuracy', 'wall_time')\n\n\n"
    "def brief(summary):\n"
    "    \"\"\"One line per run; the live progress and milestones are shown by scripts/run_notebooks.py.\"\"\"\n"
    "    for value in summary.values():\n"
    "        if isinstance(value, dict) and 'approach' in value:\n"
    "            brief(value)                      # a stage of a combined run, e.g. DAgger and speed push\n"
    "    print({key: summary[key] for key in BRIEF_KEYS if key in summary}, flush=True)\n")

EVALUATION_TABLES = ("results", "head_to_head", "training_cost", "stages", "comparison", "search_cost",
                     "driving_selected", "dataset")

NOTEBOOKS = {
    "01_dataset.ipynb": [
        ("markdown", "# 1. Expert dataset\nCollects the imitation-learning dataset on freshly generated maps "
                     "(revision 3 §6.1)."),
        ("code", SETUP),
        ("code", "import json\n\nimport pandas as pd\n\nfrom racing import config\nfrom racing.expert import "
                 "collect_dataset\n\ndata, summary = collect_dataset(n_jobs=config.N_JOBS)\n"
                 "print('collection time (s):', json.loads(config.DATASET_INFO_PATH.read_text())['wall_time'])\n"
                 "pd.DataFrame(summary).T"),
    ],
    "02_imitation.ipynb": [
        ("markdown", "# 2. Imitation-learning controllers\nBaseline with each run seed, the three Keras Tuner "
                     "searches, and training of each selected architecture with each run seed (design "
                     "specification §8; revision 3 §6.2)."),
        ("code", SETUP),
        ("code", "from racing import config, imitation\n\n\n" + BRIEF + "\n\ndata = imitation.load_dataset()\n"
                 "for seed in config.RUN_SEEDS:\n    brief(imitation.train_baseline(data, seed=seed))\n"
                 "for approach in config.TUNER_APPROACHES:\n    brief(imitation.run_tuner(approach, data))\n"
                 "    for seed in config.RUN_SEEDS:\n        brief(imitation.retrain(approach, data, seed=seed))"),
    ],
    "03_dagger.ipynb": [
        ("markdown", "# 3. DAgger and speed push\nEach imitation controller is refined by DAgger rounds and then "
                     "pushed for speed by an evolution strategy grown from it, once per run seed (revision 3 "
                     "§6.3–6.4)."),
        ("code", SETUP),
        ("code", "from racing import config, dagger, imitation\n\n\n" + BRIEF + "\n\ndata = imitation.load_dataset()\n"
                 "for approach in config.IMITATION_APPROACHES:\n    for seed in config.RUN_SEEDS:\n"
                 "        brief(dagger.run(approach, seed, data, n_jobs=config.N_JOBS))"),
    ],
    "04_ga.ipynb": [
        ("markdown", "# 4. Genetic algorithm\nFrom Easy to the mastery goal and the speed push, once per run seed "
                     "(revision 3 §4, §5.1)."),
        ("code", SETUP),
        ("code", "from racing import config, ga\n\n\n" + BRIEF + "\n\n"
                 "for seed in config.RUN_SEEDS:\n    brief(ga.run_ga(seed=seed, n_jobs=config.N_JOBS))"),
    ],
    "05_es.ipynb": [
        ("markdown", "# 5. Evolution strategy\nThe ES with the two-phase step-size rule, once per run seed "
                     "(revision 3 §4, §5.2)."),
        ("code", SETUP),
        ("code", "from racing import config, es\n\n\n" + BRIEF + "\n\n"
                 "for seed in config.RUN_SEEDS:\n    brief(es.run_es(seed=seed, n_jobs=config.N_JOBS))"),
    ],
    "06_es_seeded.ipynb": [
        ("markdown", "# 6. Seeded evolution strategy\nThe ES started from the DAgger-refined baseline of the same "
                     "run seed (revision 3 §5.3)."),
        ("code", SETUP),
        ("code", "from racing import config, es\n\n\n" + BRIEF + "\n\n"
                 "for seed in config.RUN_SEEDS:\n"
                 "    brief(es.run_es(seed=seed, initial=es.dagger_baseline(seed), approach='es_seeded',\n"
                 "                    n_jobs=config.N_JOBS))"),
    ],
    "07_neat.ipynb": [
        ("markdown", "# 7. NEAT\nNeuroevolution of augmenting topologies, once per run seed (revision 3 §5.4)."),
        ("code", SETUP),
        ("code", "from racing import config, neat\n\n\n" + BRIEF + "\n\n"
                 "for seed in config.RUN_SEEDS:\n    brief(neat.run_neat(seed=seed, n_jobs=config.N_JOBS))"),
    ],
    "08_evaluate.ipynb": [
        ("markdown", "# 8. Evaluation\nTest maps finished and succeeded, lap times, training cost, imitation "
                     "stages and the course comparison, as mean ± standard deviation across run seeds "
                     "(revision 3 §7)."),
        ("code", SETUP),
        ("code", "from IPython.display import display\n\nfrom racing import config, evaluate\n\n"
                 "results = evaluate.main(n_jobs=config.N_JOBS)\n"
                 f"for name in {EVALUATION_TABLES!r}:\n    print(name)\n    display(results[name])"),
        ("code", "import json\n\n# run conditions recorded by scripts/run_notebooks.py (revision 3 §8)\n"
                 "if config.RUN_INFO_PATH.exists():\n"
                 "    print(json.dumps(json.loads(config.RUN_INFO_PATH.read_text()), indent=1))"),
        ("code", "from IPython.display import Image\n\nfor path in results['figures']:\n"
                 "    display(Image(filename=str(path)))"),
    ],
}

OBSOLETE = ("05_evaluate.ipynb", "03_ga.ipynb", "04_es.ipynb", "05_es_seeded.ipynb",
            "06_evaluate.ipynb")         # notebooks of earlier specifications; other files are kept


def cell(kind, source, index):
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "id": f"cell-{index}", "metadata": {}, "source": lines}
    return {"cell_type": "code", "id": f"cell-{index}", "metadata": {}, "execution_count": None, "outputs": [],
            "source": lines}


def archive_once():
    """Copy the existing experiment notebooks to ARCHIVE before they are first rewritten, while revision 2.0
    notebooks (any of OBSOLETE) are still present; they hold the outputs of the revision 2.0 run. Does nothing once
    the archive exists or on a fresh clone without revision 2.0 notebooks. Returns the copied names."""
    if ARCHIVE.exists() or not any((FOLDER / name).exists() for name in OBSOLETE):
        return []
    names = sorted(p.name for p in FOLDER.glob("[0-9][0-9]_*.ipynb"))
    if not names:
        return []
    ARCHIVE.mkdir(parents=True)
    for name in names:
        shutil.copy2(FOLDER / name, ARCHIVE / name)
    return names


def main():
    FOLDER.mkdir(exist_ok=True)
    archive_once()
    for name in OBSOLETE:
        (FOLDER / name).unlink(missing_ok=True)
    for name, cells in NOTEBOOKS.items():
        notebook = {"cells": [cell(kind, source, k) for k, (kind, source) in enumerate(cells)],
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python"}},
                    "nbformat": 4, "nbformat_minor": 5}
        (FOLDER / name).write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 13: Regenerate the notebooks**

Run: `~/miniconda3/bin/python scripts/make_notebooks.py && ls results/archive/revision2-notebooks notebooks`
Expected: the archive lists `01_dataset.ipynb 02_imitation.ipynb 03_ga.ipynb 04_es.ipynb 05_es_seeded.ipynb
06_evaluate.ipynb` (the revision 2.0 notebooks with their outputs; `results/` is not under version control), and
`notebooks/` lists exactly the eight new notebooks.

- [ ] **Step 14: Run the runner tests and the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_runner.py tests/test_config.py tests/test_bench.py`
Expected: all passed.
Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all passed.
Run: `~/miniconda3/bin/python -m pytest -q -m slow tests/test_runner.py`
Expected: 4 passed.

- [ ] **Step 15: Check the command line by hand**

Run: `~/miniconda3/bin/python scripts/run_notebooks.py --fresh --from 02; echo "exit $?"`
Expected: `error: --fresh starts over from the first notebook and cannot be combined with --from` and `exit 2`.
Run: `EC_RACING_JOBS=0 ~/miniconda3/bin/python -c "import racing.config"`
Expected: `ValueError: EC_RACING_JOBS must be a whole number of at least 1, not '0'`.

- [ ] **Step 16: Commit**

```bash
git add racing/config.py racing/bench.py scripts/run_notebooks.py scripts/make_notebooks.py \
        tests/test_config.py tests/test_bench.py tests/test_runner.py \
        notebooks/01_dataset.ipynb notebooks/02_imitation.ipynb notebooks/03_dagger.ipynb notebooks/04_ga.ipynb \
        notebooks/05_es.ipynb notebooks/06_es_seeded.ipynb notebooks/07_neat.ipynb notebooks/08_evaluate.ipynb \
        notebooks/03_ga.ipynb notebooks/04_es.ipynb notebooks/05_es_seeded.ipynb notebooks/06_evaluate.ipynb
git commit -m "Runner resumes by default with --fresh, settings check and live progress; eight notebooks; worker count from EC_RACING_JOBS

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- racing/config.py racing/bench.py scripts/run_notebooks.py \
        scripts/make_notebooks.py tests/test_config.py tests/test_bench.py tests/test_runner.py notebooks/
```

(`git add` of the four deleted notebook paths stages their removal.)

### Task 16: Pilots

Short runs of each evolutionary method from Easy with seed 0 that show, before the reported run, whether the 90%
requirement can be reached with the reported settings (revision 3 §10.2). The revision 2.0 pilots (GA mutation
range, ES tie behavior) are replaced.

**Files:**
- Modify: `scripts/pilots.py` (whole file)
- Modify: `tests/test_pilots.py` (whole file)
- Modify: `racing/es.py`, `tests/test_es.py` (remove `trained_baseline`, whose last caller this task replaces)

**Interfaces:**
- Consumes: `ga.run_ga`, `es.run_es`, `es.dagger_baseline(seed, runs_dir=None)` (Task 6); `neat.run_neat` (Task 9);
  all with the keywords `seed`, `out_dir`, `champion_dir`, `n_jobs`, `max_generations`, `pop_size`;
  `paths.stage_path(approach, seed, stage, runs_dir=None)` (Task 10); `paths.remove`; the log columns
  `generation`, `level`, `event`, `min_share` (`evolution.LOG_FIELDS`, Task 6) and the summary fields
  `generations`, `level_reached`, `wall_time` (Task 6).
- Produces: `pilots.PILOTS`, `pilots.OUT`, `pilots.level_ups(log) -> str`, `pilots.pilot(name, out_root=None,
  max_generations=300, n_jobs=None, pop_size=None) -> dict`, `pilots.main(argv=None) -> int`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_pilots.py` with:

```python
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from racing import nets, population, tracks

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "pilots.py"


@pytest.fixture
def pilots(monkeypatch, circle_track):
    """The pilots script with every map replaced by the small circle, so a pilot of two generations takes seconds."""
    def fake(seed, level):
        return circle_track

    fake.cache_clear = lambda: None          # population.fresh_workers empties tracks.get_track's cache
    monkeypatch.setattr(population, "get_track", fake, raising=False)
    monkeypatch.setattr(tracks, "get_track", fake)
    spec = importlib.util.spec_from_file_location("pilots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_level_ups_lists_each_level_reached_and_the_goal(pilots):
    log = pd.DataFrame({"generation": [0, 1, 2, 3], "level": [1, 1, 2, 4],
                        "event": [None, "level_up", "level_up", "goal"]})
    assert pilots.level_ups(log) == "2@1 3@2 goal@3"
    assert pilots.level_ups(log.assign(event=None)) == "none"


@pytest.mark.parametrize("name", ["ga", "es", "neat"])
def test_a_short_pilot_summarizes_its_run(pilots, tmp_path, name):
    row = pilots.pilot(name, tmp_path, max_generations=2, n_jobs=1, pop_size=60)
    assert row["generations"] == 2 and row["level_reached"] >= 1
    assert isinstance(row["level_ups"], str) and 0.0 <= row["min_share_last20"] <= 1.0
    assert (tmp_path / name / "log.csv").exists()


def test_a_pilot_always_starts_over(pilots, tmp_path):
    pilots.pilot("ga", tmp_path, max_generations=2, n_jobs=1, pop_size=60)
    row = pilots.pilot("ga", tmp_path, max_generations=3, n_jobs=1, pop_size=60)
    assert row["generations"] == 3 and len(pd.read_csv(tmp_path / "ga" / "log.csv")) == 3


def test_the_seeded_pilot_is_skipped_without_the_dagger_baseline(pilots, tmp_path, monkeypatch):
    monkeypatch.setattr(pilots.paths, "stage_path",
                        lambda approach, seed, stage, runs_dir=None: tmp_path / "missing.keras")
    row = pilots.pilot("seeded", tmp_path, max_generations=2, n_jobs=1, pop_size=60)
    assert row["pilot"] == "Seeded ES" and row["note"].startswith("skipped")
    assert not (tmp_path / "seeded").exists()


def test_the_seeded_pilot_starts_from_the_dagger_baseline(pilots, tmp_path, monkeypatch):
    baseline = tmp_path / "dagger.keras"
    baseline.write_bytes(b"")
    monkeypatch.setattr(pilots.paths, "stage_path", lambda approach, seed, stage, runs_dir=None: baseline)
    initial = nets.init_population(np.random.default_rng(0), 1)[0]
    monkeypatch.setattr(pilots.es, "dagger_baseline", lambda seed, runs_dir=None: initial)
    row = pilots.pilot("seeded", tmp_path, max_generations=2, n_jobs=1, pop_size=60)
    assert row["pilot"] == "Seeded ES" and row["generations"] == 2


def test_main_appends_a_summary_and_rejects_unknown_pilots(pilots, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pilots, "OUT", tmp_path)
    monkeypatch.setattr(pilots, "pilot", lambda name, out_root, generations: {"pilot": name, "generations": generations})
    assert pilots.main(["ga", "--generations", "5"]) == 0
    assert pilots.main(["neat"]) == 0
    text = (tmp_path / "summary.md").read_text()
    assert "| ga | 5 |" in text and "| neat | 300 |" in text and text.count("## ") == 2
    assert pilots.main(["bogus"]) == 2 and "bogus" in capsys.readouterr().err
```

- [ ] **Step 2: Run them to verify they fail**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_pilots.py`
Expected: FAIL — `AttributeError: module 'pilots' has no attribute 'pilot'`, and the old `level_ups` fails with
`KeyError: 'leveled_up'`.

- [ ] **Step 3: Rewrite the pilots**

Replace `scripts/pilots.py` with:

```python
"""Pilot runs of revision 3 (§10.2): short runs of each evolutionary method from Easy with seed 0, which show
whether the 90% requirement can be reached with the reported settings before the reported run starts. Each pilot
starts over in results/pilots/revision3/<name>/ and adds a row to results/pilots/revision3/summary.md. Pilots are
not part of the reported results.

Run with:  python scripts/pilots.py                       (ga, es, neat and seeded; 300 generations each)
           python scripts/pilots.py ga neat               (selected pilots)
           python scripts/pilots.py --generations 500 es
The seeded pilot needs the DAgger-refined baseline of seed 0 (notebook 03) and is skipped without it.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racing import config, es, ga, neat, paths  # noqa: E402

PILOTS = ("ga", "es", "neat", "seeded")
NAMES = {"ga": "GA", "es": "ES", "neat": "NEAT", "seeded": "Seeded ES"}
OUT = config.PILOTS_DIR / "revision3"


def level_ups(log):
    """When each level was reached and the mastery goal met, for example '2@25 3@74 4@94 goal@120'."""
    marks = []
    for row in log.itertuples(index=False):
        if row.event == "level_up":
            marks.append(f"{int(row.level) + 1}@{int(row.generation)}")
        elif row.event == "goal":
            marks.append(f"goal@{int(row.generation)}")
    return " ".join(marks) or "none"


def run_pilot(name, out, max_generations, n_jobs, pop_size):
    common = {"seed": 0, "out_dir": out, "champion_dir": None, "n_jobs": n_jobs, "max_generations": max_generations}
    if pop_size is not None:
        common["pop_size"] = pop_size
    if name == "ga":
        return ga.run_ga(**common)
    if name == "es":
        return es.run_es(**common)
    if name == "neat":
        return neat.run_neat(**common)
    return es.run_es(initial=es.dagger_baseline(0), approach="es_seeded", **common)


def pilot(name, out_root=None, max_generations=300, n_jobs=None, pop_size=None):
    """One pilot from Easy with seed 0, summarized as one table row."""
    out_root = Path(OUT if out_root is None else out_root)
    n_jobs = config.N_JOBS if n_jobs is None else n_jobs
    if name == "seeded" and not paths.stage_path("baseline", 0, "dagger").exists():
        return {"pilot": NAMES[name], "note": "skipped: the DAgger-refined baseline of seed 0 does not exist yet "
                                              "(run notebook 03 first)"}
    out = out_root / name
    paths.remove(out)                       # a pilot always starts over; evolutionary runs otherwise resume
    summary = run_pilot(name, out, max_generations, n_jobs, pop_size)
    log = pd.read_csv(out / "log.csv")
    return {"pilot": NAMES[name], "generations": int(summary["generations"]),
            "level_reached": int(summary["level_reached"]), "level_ups": level_ups(log),
            "min_share_last20": round(float(log["min_share"].tail(20).mean()), 3),
            "minutes": round(float(summary["wall_time"]) / 60, 1)}


def markdown(frame):
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Revision 3 pilot runs from Easy with seed 0.")
    parser.add_argument("pilots", nargs="*", help=f"any of {', '.join(PILOTS)} (default: all)")
    parser.add_argument("--generations", type=int, default=300, help="generations per pilot (default 300)")
    args = parser.parse_args(argv)
    selected = args.pilots or list(PILOTS)
    unknown = [name for name in selected if name not in PILOTS]
    if unknown:
        print(f"error: unknown pilot(s) {', '.join(unknown)}; choose from {', '.join(PILOTS)}", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in selected:
        rows.append(pilot(name, OUT, args.generations))
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    with open(OUT / "summary.md", "a") as f:
        f.write(f"## {datetime.now().isoformat(timespec='seconds')}, {args.generations} generations\n\n")
        f.write(markdown(table) + "\n\n")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the pilot tests to verify they pass**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_pilots.py`
Expected: 8 passed.

Then remove the revision 2.0 seeded start, which reads `champions/baseline/seed<k>.keras` (in revision 3 the speed-pushed
baseline): delete the function `trained_baseline` from `racing/es.py`, and delete `test_trained_baseline_reads_the_baseline_controller`
and `test_trained_baseline_missing` from `tests/test_es.py`.
Run: `grep -rn "trained_baseline" racing scripts tests`
Expected: no output.

- [ ] **Step 5: Run the fast suite**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all passed.

- [ ] **Step 6: Check the command line by hand**

Run: `~/miniconda3/bin/python scripts/pilots.py bogus; echo "exit $?"`
Expected: `error: unknown pilot(s) bogus; choose from ga, es, neat, seeded` and `exit 2`.

- [ ] **Step 7: Commit**

```bash
git add scripts/pilots.py tests/test_pilots.py racing/es.py tests/test_es.py
git commit -m "Pilots for revision 3: GA, ES, NEAT and seeded ES from Easy, level progression and lowest map share

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- scripts/pilots.py tests/test_pilots.py racing/es.py tests/test_es.py
```

- [ ] **Step 8: Run the pilots (calibration, before the reported run)**

Run (about one to three hours on the development machine; not part of the test suite):
`~/miniconda3/bin/python scripts/pilots.py ga es neat`
Expected: three rows in `results/pilots/revision3/summary.md`. A pilot whose `level_ups` is `none` after 300
generations has not cleared Easy. Report the table to the user before the reported run starts: revision 3 §10.2
requires the survival and step-size settings to be reviewed with the user if a level is not reached, and this plan
does not change them on its own.

### Task 17: Documentation, requirements and design specification 3.0

A pinned package list (with a test that the environment matches it), a README that takes a new Windows desktop
from nothing to a running experiment in WSL, the design specification updated to version 3.0 with revision 3
folded in, regenerated PDFs, and the branch pushed (revision 3 §8.4, §13).

**Files:**
- Create: `requirements.txt`, `tests/test_requirements.py`, `scripts/md2pdf.py`
- Modify: `README.md` (whole file)
- Modify: `docs/2026-10-05-ec-racing-design.md`; regenerate `docs/2026-10-05-ec-racing-design.pdf`

**Interfaces:**
- Consumes: everything built in Tasks 1–16, as described by its contract names; `scripts/md2pdf.py`
  (created in Step 0; usage `python scripts/md2pdf.py doc.md [doc.pdf]`; needs `markdown_it` and `google-chrome`,
  both present on the development machine).
- Produces: `requirements.txt`; `tests/test_requirements.py::pinned() -> dict[str, str]`.

- [ ] **Step 0: Add the PDF script**

Create `scripts/md2pdf.py` (it renders the specification PDFs; it is a documentation tool, not part of the experiment):

```python
"""Render a Markdown document to PDF with markdown-it and headless Chrome.

Usage: python md2pdf.py doc.md [doc.pdf]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

CSS = """
@page { size: Letter; margin: 2cm 2cm 2.2cm 2cm; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 10pt; line-height: 1.45; color: #1d2330; }
h1 { font-size: 20pt; margin: 0 0 4pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #c9ced8; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; }
p, li { orphans: 3; widows: 3; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #f2f4f7; padding: 0 2px; }
pre { background: #f6f7f9; border: 1px solid #e1e5ea; padding: 8pt; font-size: 8.3pt; line-height: 1.3;
      overflow-wrap: anywhere; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9ced8; padding: 3pt 6pt; vertical-align: top; text-align: left; }
th { background: #eef1f5; }
em { color: #3c4556; }
"""


def main():
    source = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else source.with_suffix(".pdf")
    body = MarkdownIt("commonmark").enable("table").render(source.read_text())
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "doc.html"
        page.write_text(html)
        subprocess.run(["google-chrome", "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={target}", page.as_uri()], check=True, capture_output=True)
    print(target)


if __name__ == "__main__":
    main()
```

- [ ] **Step 1: Print the installed versions**

Run:

```bash
~/miniconda3/bin/python - <<'EOF'
import platform
from importlib.metadata import version
print("# Python", platform.python_version())
for name in ["numpy", "scipy", "pandas", "matplotlib", "tensorflow", "keras", "keras-tuner", "joblib", "pygame",
             "nbclient", "nbformat", "ipykernel", "jupyter_client", "pytest"]:
    print(f"{name}=={version(name)}")
EOF
```

Expected (development machine, 7 October 2026; use the printed values if they differ):
`# Python 3.13.11`, `numpy==2.4.1`, `scipy==1.17.1`, `pandas==2.3.3`, `matplotlib==3.10.8`, `tensorflow==2.21.0`,
`keras==3.13.2`, `keras-tuner==1.4.8`, `joblib==1.5.3`, `pygame==2.6.1`, `nbclient==0.11.0`, `nbformat==5.11.1`,
`ipykernel==7.3.0`, `jupyter_client==8.9.1`, `pytest==9.1.1`.

- [ ] **Step 2: Write the failing requirements test**

Create `tests/test_requirements.py`:

```python
import platform
from importlib.metadata import version
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def pinned():
    """{package: version} of every pinned line of requirements.txt."""
    pins = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, value = line.partition("==")
            pins[name.strip()] = value.strip()
    return pins


def test_requirements_pin_every_package_exactly():
    pins = pinned()
    assert {"numpy", "scipy", "pandas", "matplotlib", "tensorflow", "keras", "keras-tuner", "joblib", "pygame",
            "nbclient", "nbformat", "ipykernel", "jupyter_client", "pytest"} == set(pins)
    assert all(pins.values())


def test_the_environment_matches_the_pins():
    """The experiment runs in the environment requirements.txt describes (revision 3 §8.4)."""
    assert platform.python_version().startswith("3.13.")
    assert {name: version(name) for name in pinned()} == pinned()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_requirements.py`
Expected: FAIL with `FileNotFoundError` for `requirements.txt`.

- [ ] **Step 4: Write `requirements.txt`**

Create `requirements.txt` with the versions printed in Step 1:

```
# EC Racing environment (revision 3 §8.4). Python 3.13.11.
# Install into a fresh Python 3.13 environment with:  pip install -r requirements.txt
numpy==2.4.1
scipy==1.17.1
pandas==2.3.3
matplotlib==3.10.8
tensorflow==2.21.0
keras==3.13.2
keras-tuner==1.4.8
joblib==1.5.3
pygame==2.6.1
nbclient==0.11.0
nbformat==5.11.1
ipykernel==7.3.0
jupyter_client==8.9.1
pytest==9.1.1
```

- [ ] **Step 5: Run the requirements test to verify it passes**

Run: `~/miniconda3/bin/python -m pytest -q tests/test_requirements.py`
Expected: 2 passed.

- [ ] **Step 6: Rewrite the README**

Replace `README.md` with:

````markdown
# EC Racing

EC Racing compares methods for finding small neural network driving controllers. Eight controllers race on freshly
generated circuits, whose harder levels borrow features of Formula 1 circuits, in a top-down 2D simulator:

- **Imitation learning:** a hand-designed baseline and three Keras Tuner searches (random search, Hyperband,
  Bayesian optimization) are trained to imitate a scripted expert driver, refined by DAgger and then pushed for speed
  by an evolution strategy grown from each of them.
- **Neuroevolution:** a genetic algorithm and an evolution strategy evolve the weights of a fixed network by driving;
  NEAT evolves the network structure together with its weights.
- **Hybrid:** a seeded evolution strategy starts from the DAgger-refined baseline and continues by driving.

Every method trains until it masters the curriculum: on every freshly generated map of a generation, at least 90% of
its population must complete three laps within the target time, three generations in a row, on each level from Easy
to Extra hard. The target time is then shortened by 5% at a time until the population can no longer keep up. Every
method runs with three seeds; results are reported as mean ± standard deviation. The design specification is
`docs/2026-10-05-ec-racing-design.md` (version 3.0); the revision 3 specification and plan are in `docs/`.

## Setup on a Windows desktop (WSL 2)

The experiment runs on Linux. On Windows it runs in the Windows Subsystem for Linux.

1. **Install WSL 2 with Ubuntu.** In PowerShell as administrator: `wsl --install -d Ubuntu`, restart Windows, then
   open *Ubuntu* from the Start menu and create a Linux user when asked.
2. **Install Miniconda** in the Ubuntu terminal:

   ```bash
   cd ~
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b
   ~/miniconda3/bin/conda init bash
   exec bash
   ```

3. **Give WSL access to GitHub over SSH:**

   ```bash
   ssh-keygen -t ed25519 -C "ec-racing desktop"      # press Enter at each question
   cat ~/.ssh/id_ed25519.pub
   ```

   Copy the printed line into GitHub → Settings → SSH and GPG keys → New SSH key, then check it with
   `ssh -T git@github.com`.
4. **Clone the project into the Linux home folder** (not under `/mnt/c`: files on the Windows drive are much slower
   from WSL):

   ```bash
   cd ~
   git clone <repository SSH URL> ec-racing      # the URL under Code → SSH on the repository's GitHub page
   cd ec-racing
   git checkout build
   ```

5. **Create the Python environment:**

   ```bash
   conda create -y -n ec-racing python=3.13
   conda activate ec-racing
   pip install -r requirements.txt
   python -m pytest -q                               # the fast tests should all pass
   ```

   If `conda` asks to accept the terms of service of its default channels, run the command it prints and repeat
   `conda create`. Activate the environment (`conda activate ec-racing`) in every new terminal.
6. **Keep Windows awake.** WSL pauses while Windows sleeps. Set Settings → System → Power → *Screen and sleep* →
   "When plugged in, put my device to sleep after" to *Never* (or run `powercfg /change standby-timeout-ac 0` in
   PowerShell) for the duration of the run.
7. **Choose the number of worker processes** for this machine:

   ```bash
   python -m racing.bench generation
   ```

   It times one generation for each worker count up to the number of logical processors and ends with a line such as
   `export EC_RACING_JOBS=16`. Make it permanent:

   ```bash
   echo 'export EC_RACING_JOBS=16' >> ~/.bashrc       # use the number the benchmark printed
   source ~/.bashrc
   ```

To update the code later: `git pull`.

## Setup on Linux

On a Linux machine follow steps 2–7 without WSL. On the development laptop the Conda base environment is used
instead of `ec-racing`, with `~/miniconda3/bin/python` in place of `python`.

## Running the experiment

One command runs the eight notebooks in order and saves their outputs into the notebook files:

```bash
python scripts/run_notebooks.py
```

The run takes days rather than hours: every method trains until it masters the curriculum and can no longer get
faster. A live status line shows the current run, level, generation, the worst map's success share against 90%, the
streak and the elapsed time; level-ups, mastery goals, target reductions and finished runs are printed as permanent
lines and kept in `results/run_log.txt`.

To run it in the background, so the terminal can be closed:

```bash
mkdir -p results
nohup python scripts/run_notebooks.py > results/run_console.txt 2>&1 &
python -m racing.progress                            # live status from any terminal; Ctrl+C stops watching only
```

- **Stopping and resuming:** stop with Ctrl+C (or `kill` the background process). Start the same command again to
  continue: finished runs are skipped and an interrupted run continues from its last checkpoint, losing at most one
  generation. A resumed run produces exactly the results of an uninterrupted one.
- **From a notebook:** `python scripts/run_notebooks.py --from 04` starts at notebook 04.
- **Starting over:** `python scripts/run_notebooks.py --fresh` removes the earlier results (runs, controllers,
  dataset, tables, figures and run record; pilots, viewer runs and the archive are kept) and starts from notebook 01.
- **Changed settings:** if the settings in `racing/config.py` differ from those recorded by earlier runs, the runner
  stops before running anything, names the differing settings and exits with code 3. Run again with `--fresh`.

| Notebook | Purpose |
|---|---|
| `01_dataset` | Expert dataset on freshly generated maps |
| `02_imitation` | Baseline, Keras Tuner searches, training with each seed |
| `03_dagger` | DAgger refinement and speed push of the four imitation methods, three seeds each |
| `04_ga` | Genetic algorithm, three seeds |
| `05_es` | Evolution strategy, three seeds |
| `06_es_seeded` | Seeded evolution strategy, three seeds |
| `07_neat` | NEAT, three seeds |
| `08_evaluate` | Tables and figures |

The processor, worker count, package versions, settings, code version (git commit) and notebook times are recorded
in `results/run_info.json`. All reported times come from one machine.

Results are written to `results/`, which is not under version control:

| Path | Contents |
|---|---|
| `results/dataset.npz`, `results/dataset_info.json` | Expert dataset and its collection time |
| `results/runs/<method>/seed<k>/` | Evolutionary runs: `log.csv` (one row per generation), `checkpoint.pkl`, `summary.json` |
| `results/runs/<method>/seed<k>/` (imitation methods) | `imitation.keras`, `dagger.keras`, the DAgger rows, state and summary, and `speed/` for the speed push |
| `results/runs/<method>/search/` | Keras Tuner search: trials, summary and exported models |
| `results/champions/<method>/seed<k>.keras` | Final controllers (evolved dense networks also as `.npy`; NEAT genomes as `.json`) |
| `results/champions/best_seeds.json` | Each method's best seed on the validation maps |
| `results/tables/`, `results/figures/` | Results tables (CSV and Markdown) and figures, including the network figure |
| `results/status.json`, `results/run_info.json`, `results/run_log.txt`, `results/run_console.txt` | Live status, record, log and console output of the run |
| `results/pilots/revision3/` | Pilot runs |
| `results/archive/revision2-notebooks/` | The notebooks of the revision 2.0 run, with their outputs |

To copy results to Windows, open `\\wsl$\Ubuntu\home\<Linux user>\ec-racing\results` in File Explorer.

## Pilots and benchmarks

```bash
python scripts/pilots.py ga es neat       # short runs from Easy: is the 90% requirement reachable?
python -m racing.bench generation         # generation time per worker count; prints the EC_RACING_JOBS to use
python -m racing.bench keras              # Keras epoch time per steps_per_execution and jit_compile
```

The pilots write to `results/pilots/revision3/` and are not part of the reported results.

## Viewer

```bash
python -m racing.app
```

On Windows 11 the window opens through WSLg.

- **Training view:** choose an approach, run seed, difficulty and population size, then press Play (or Space).
  Training follows the mastery rules and continues until Stop. The panel shows the level, each map's success share
  against 90%, the streak and the target reduction. Training started here writes only to `results/app_runs/`.
- **Race view:** pick a test map with the arrow keys and a difficulty; the map is shown with the cars on the start
  line. For each method, choose its best seed, a specific seed, or off, then press Start race; Stop returns to the
  start line.
- **Difficulty:** Easy (gentle loops on a wide road), Medium (circuits), Hard (circuits with a main straight into a
  hairpin, sometimes a chicane or esses), Extra hard (narrow circuits with two hairpins, chicanes, esses and fast
  sweepers). Maps run clockwise or counter-clockwise.
- **Playback speed:** 1× is twice real time.

## Tests

```bash
python -m pytest              # fast tests
python -m pytest -m slow      # slow tests: generator acceptance, Keras Tuner, runner, end-to-end
```
````

- [ ] **Step 7: Update the design specification to version 3.0**

Edit `docs/2026-10-05-ec-racing-design.md` as follows. "Copy revision 3 §X" means: copy the body text of that
section of `docs/2026-10-07-ec-racing-mastery-design.md` verbatim, then rewrite its cross-references with this
renumbering (revision 3 → version 3.0), and leave every other word unchanged. Text quoted below from the current
document may be wrapped across lines there and may carry Markdown backticks (for example `` `t_run` ``); search
for a few words of it:

| Revision 3 | Version 3.0 | Revision 3 | Version 3.0 |
|---|---|---|---|
| §3.1 | §4.2 | §5.4 | §9.9 |
| §3.2 | §4.3 | §6.1 | §7.3 |
| §3.3 | §4.4 | §6.2 | §8.1–8.3 |
| §3.4 | §4.6 | §6.3 | §8.4 |
| §4.1 | §9.2 | §6.4 | §8.5 |
| §4.2 | §9.3 | §6.5 | §8.6 |
| §4.3, §4.4, §4.5 | §9.4 | §7, §7.1, §7.2, §7.3 | §11, §11.1, §11.3, §11.4 |
| §4.6 | §9.5 | §8, §8.1–§8.4 | §3.3 |
| §5 | §9.6–9.9 | §9 | §10 |
| §5.1 | §9.6 | §10, §10.1, §10.2 | §12, §12.1, §12.2 |
| §5.2 | §9.7 | §11, Table 3 | §13, Table 6 |
| §5.3 | §9.8 | §12 | §14 |

Changes, in document order:

1. **Version line:** `ECDL Term Project · Fall 2026 · Version 3.0 · <date>`, with the date printed by
   `date '+%-d %B %Y'`.
2. **Summary:** replace the whole section body with:

   > This document specifies EC Racing, a top-down two-dimensional racing simulator. It is used to compare
   > methods for finding small neural network driving controllers. Eight controllers compete on freshly
   > generated circuits whose harder levels borrow features of Formula 1 circuits.
   >
   > - **Imitation learning (four controllers).** These are trained by supervised imitation of a scripted
   >   expert driver. One is a hand-designed baseline. In the other three, Keras Tuner selects the
   >   architecture using random search, Hyperband or Bayesian optimization. Each is then refined by dataset
   >   aggregation (DAgger) and pushed for speed by an evolution strategy grown from it.
   > - **Neuroevolution (three controllers).** A genetic algorithm (GA) and a (μ+λ) evolution strategy (ES)
   >   optimize the weights of a fixed network directly from driving performance. NEAT evolves the network
   >   structure together with its weights.
   > - **Hybrid (one controller).** A seeded ES starts from the DAgger-refined baseline network, which has
   >   learned from the expert, and then optimizes the weights from driving performance alone.
   >
   > Every method trains until it masters the curriculum: on each freshly generated map of a generation, at
   > least 90% of its population must complete three laps within a target time, in three consecutive
   > generations, on each of four levels. The target time is then shortened in steps of 5% until the
   > population can no longer meet it. Every method is run with three random seeds, and results are reported
   > as the mean and standard deviation across seeds. All final controllers are evaluated on held-out maps for
   > completion, lap time, classification accuracy, network size, inference time and training cost. A
   > visualization application shows training in progress and races between the final controllers. A single
   > command executes the complete experiment; it saves checkpoints, resumes after an interruption and reports
   > its progress live.
   >
   > Version 3.0 incorporates revision 3 (`docs/2026-10-07-ec-racing-mastery-design.md`): freshly generated
   > circuits, time trials, the mastery criterion and speed push in place of a fixed evaluation budget, NEAT,
   > DAgger and the execution of multi-day runs. Version 2.0 corrected the measurement of training epochs and
   > inference time and added replication across seeds, the seeded ES and the run record.

3. **§1.1:** append the paragraph:

   > Two further questions are whether evolving the network structure together with its weights (NEAT) finds
   > better drivers than evolving the weights of a fixed network, and how far dataset aggregation (DAgger) and
   > a subsequent evolutionary speed push close the gap between imitation-trained and evolved controllers.
   > Every method is trained to the same mastery criterion, and the cost of reaching it is compared.

4. **§1.2:** in the first bullet's sub-list, replace `an ES (μ+λ) with the 1/5 success rule` with
   `an ES (μ+λ) with the 1/5 success rule (the learning phase of its step-size rule, §9.7)`, and after the
   objectives bullet insert the bullet `- Driving is measured by the number of held-out maps completed within the
   target time and by lap time.`
5. **§2:** replace the bullets `Human-driven cars, …` and `A user interface for browsing …` with:

   > - Human-driven cars, menu systems and track editing. The circuit generator borrows features of Formula 1
   >   circuits, but no real circuit is reproduced.
   > - A user interface for browsing earlier generations.

6. **§3.1:** replace the diagram with:

   ```
   track_gen ──► tracks (fresh training maps; fixed validation / test maps; levels 1–4)
                    │
                    ▼
   sim (numpy, every car stepped at once: physics, rays, crash, progress, fitness)
      │                         │                                   │
      ▼                         ▼                                   ▼
   expert ──► dataset      evolution.py: time trials, survival,   app.py (pygame)
      │        ▲           levels, speed push, checkpoints         training view
      ▼        │           ├─ ga.py   ├─ es.py   ├─ neat.py         race view
   imitation.py│                ▲          │                          ▲
   (baseline + │                │          ▼                          │
    KT Random/ │   dagger.py ───┘   final controllers ────────────────┤
    Hyperband/ └── (labels the        (+ .npy / NEAT genome)          │
    Bayesian)       car's states)                                     │
      └──► imitation stage ──► DAgger stage ──► speed push (ES) ──────┘
                    │
                    ▼
   evaluate.py, netfig.py ──► tables + figures for the paper
   checkpoint.py, progress.py: resumable runs, live status line
   ```

7. **§3.2:** in the module tree, change and add these lines (keep the others):

   ```
   │   ├── track_gen.py    gentle loops, F1-inspired circuit generator, direction, geometric validation
   │   ├── tracks.py       validated tracks with the expert's lap time (bounded cache)
   │   ├── nets.py         dense populations and drivers, Keras conversion, champion loading
   │   ├── curriculum.py   fresh seeds, survival rule, mastery tracker, step-size rules
   │   ├── population.py   time-trial evaluation per map, final selection, dense champions
   │   ├── evolution.py    shared run loop: logging, checkpoints, status, final selection
   │   ├── ga.py           genetic algorithm
   │   ├── es.py           evolution strategy (also seeded ES and speed push)
   │   ├── neat.py         NEAT: genomes, operators, speciation, batched forward pass, Keras export
   │   ├── dagger.py       DAgger refinement and speed push of the imitation methods
   │   ├── checkpoint.py   settings record, atomic checkpoints, completed-run check
   │   ├── progress.py     status file, milestones, live status line, monitor
   │   ├── netfig.py       network figure
   ├── notebooks/          01_dataset, 02_imitation, 03_dagger, 04_ga, 05_es, 06_es_seeded, 07_neat, 08_evaluate
   ```

   and replace the bullet `For readability, the code uses plain loops …` with
   `- For readability, the code uses plain loops and explicit conditionals; the one shared abstraction is the
   evolutionary method interface (§9.1), which lets every evolutionary approach use the same run loop.`
8. **§3.3:** replace the whole section body with:

   > The complete experiment is executed by a single command, `scripts/run_notebooks.py`. It executes the
   > eight notebooks in order and saves their outputs into the notebook files, so the notebooks form the
   > record of the reported run.
   >
   > - **Order:** 01 dataset → 02 imitation → 03 DAgger and speed push of the imitation methods → 04 GA →
   >   05 ES → 06 seeded ES → 07 NEAT → 08 evaluation. The seeded ES depends on the DAgger-refined baseline
   >   from notebook 03. Notebooks run one at a time, so no two methods compete for the processor while
   >   their training time is measured.
   > - **Kernel:** each notebook runs in a kernel started with the same Python interpreter as the runner,
   >   with `notebooks/` as the working directory.

   followed by the three bullets of revision 3 §8.2 (Checkpoints, Resumption, Settings check) copied, with
   ` The option `--from NN` starts at a given notebook.` appended to the Resumption bullet and
   ` The runner checks every run record and checkpoint before it executes the first notebook, and exits with
   code 3.` appended to the Settings check bullet; then the four bullets of revision 3 §8.3 copied as one bullet
   list under the lead `- **Progress display:**` (indented one level); then:

   > - **Run record:** the runner writes `results/run_info.json`. It records the processor model, the number
   >   of logical processors, the number of worker processes, the operating-system power profile where
   >   available, the versions of Python, numpy, TensorFlow, Keras and keras-tuner, the settings that affect
   >   results (the values of §13), the code version (commit and whether the working tree had uncommitted
   >   changes), and the start and end time of each notebook.

   followed by the bullets of revision 3 §8.4 copied (Worker count, Platform, Environment, Repository) and its
   closing sentence; then replace the old calibration paragraph with:

   > The pilot runs of §12.2 (`scripts/pilots.py`) check before the reported run that the mastery requirement
   > can be reached. The pilots write to `results/pilots/revision3/` and are not part of the reported results.

9. **§4.1:** append to the section: `The centerline is stored in driving order, so a map that runs clockwise is the
   reversed point sequence of its counter-clockwise form (§4.4).`
10. **§4.2:** replace the section body with: the sentence `The evolutionary controllers train on a curriculum of
    four levels, from gentle loops on a wide road to narrow circuits with features of Formula 1 circuits.`; then
    revision 3 §3.1 copied, with its table caption changed to `*Table 1. Curriculum levels and generator settings.
    Lengths in metres.*`
11. **New §4.3 Circuit generator** and **new §4.4 Direction:** copy revision 3 §3.2 and §3.3 under these headings.
12. **§4.3 Validation → §4.5 Validation:** replace the body with:

    > A generated map is rejected if any of the following holds:
    >
    > - the road comes too close to another part of itself
    > - any corner is tighter than the level's tightest corner
    > - the expert driver cannot complete a lap
    >
    > A rejected seed is retried deterministically using the triple (seed, level, attempt), so a given seed
    > always yields the same map. The direction is drawn from the seed alone (§4.4), so it is the same for every
    > attempt.

13. **§4.4 Seed allocation → §4.6 Seed allocation:** replace the body with:

    > - **Training (evolution and DAgger):** drawn by each run's random number generator from 2,000,000
    >   upwards, at the current level, and never repeated within a run. No training map is used twice.
    > - **Imitation dataset:** 121 seeds drawn once from 1,000,000–1,999,999: one at Easy, 20 at Medium, 20 at
    >   Hard and 80 at Extra hard.
    > - **Validation:** seeds 80–99, Extra hard, fixed.
    > - **Test:** seeds 1000–1019, Extra hard, fixed. Test maps are never used for training or model selection.
    > - **Application races:** the race view can also build the test seeds at the lower levels (§10.3). These
    >   maps serve only for display and are never used for training, selection or the reported results.
    >
    > Validation and test maps remain fixed sets, so that every method is evaluated on the same maps.

14. **§5.4:** replace the termination list and the text after it with:

    > - the car completes three laps
    > - the car crashes
    > - the car makes no progress for 5 s
    > - the time limit is reached
    >
    > In training and in the time-trial checks of DAgger, the time limit is the target time (§9.2). In the
    > evaluation of the final controllers and in the races of the application, it is twice the expert's lap time
    > per lap (six expert lap times in total), so that slow completions are recorded as well.

    and in the paragraph before the list keep `A run consists of three laps, …` unchanged.
15. **§5.5:** replace `where t_run is the time for the three laps` with `where t_run is the time for the three laps
    and t_limit is the time limit of the run (§5.4)`, and replace the last paragraph with: `Under this definition,
    every completion outranks every failure, so completion separates controllers early in training. Among
    completions, the faster run ranks higher.`
16. **§7.3:** replace the bullet list and the sentence `The expected total …` with:

    > - **Training:** 121 freshly generated maps: one at Easy, 20 at Medium, 20 at Hard and 80 at Extra hard
    >   (§4.6).
    > - **Validation:** the 20 validation maps.
    > - **Test:** the 20 test maps.
    >
    > The number of rows of each split is reported in the dataset table (§11.3).

    Keep the final paragraph (distributed collection, shared cost), changing `the four imitation methods and the
    seeded ES` to `the four imitation methods, their DAgger refinement and the seeded ES`.
17. **§8.3:** replace `These three networks are the method's final controllers.` with `These three networks are
    the method's controllers after initial training. DAgger (§8.4) and the speed push (§8.5) produce its final
    controllers.`
18. **New §8.4 DAgger refinement, §8.5 Speed push, §8.6 Stages:** copy revision 3 §6.3, §6.4 and §6.5.
19. **§9:** replace the whole chapter with these sections:
    - **§9.1 Common procedure** — text:

      > All evolutionary approaches share one run loop: the GA, the ES, the seeded ES, NEAT and the speed push
      > of the imitation methods (§8.5). The GA and the ES use the baseline architecture (431 weights) with
      > Glorot-uniform initial weights. The seeded ES and the speed pushes start from trained networks. NEAT
      > starts from minimal networks and grows them. Each approach is run with seeds 0, 1 and 2. Runs have no
      > evaluation budget; they end through the speed push or the safety stop (§9.4).
      >
      > **Each generation proceeds as follows:**
      >
      > 1. Three maps are freshly generated at the current level from new seeds drawn by the run's random
      >    number generator (§4.6).
      > 2. Every car drives each map as a three-lap time trial (§9.2). Kept cars are re-evaluated, because the
      >    maps change every generation.
      > 3. A car's fitness is its mean over the three maps. The success share of a map is the fraction of the
      >    whole population that succeeded on it.
      > 4. The survival rule (§9.3) decides which cars are kept, and the method's operators (§9.6–9.9) produce
      >    the children that fill the rest of the population.
      > 5. The level tracker (§9.4) records the lowest map success share.
      >
      > **Logging:** each run writes to its own directory, `results/runs/<approach>/seed<k>/`, one CSV row per
      > generation with the generation and level, the target factor, the number of evaluations, the best, mean
      > and worst fitness, the success share of each map and the lowest of them, the number of kept cars, the
      > success shares of kept cars and children, the median lap time of the successful cars, the step size
      > (GA mutation strength or σ), the streak, the event of the generation (level advancement, mastery goal,
      > target reduction or end), the map seeds and the elapsed time. NEAT also logs its number of species. A
      > checkpoint is written after every generation (§3.3).
      >
      > **Run time:** the reported time is the active run time, accumulated across resumptions, including the
      > final controller selection. Each run starts with new worker processes, so every run includes the same
      > cost of generating its maps.

    - **§9.2 Time trial** — copy revision 3 §4.1.
    - **§9.3 Survival** — copy revision 3 §4.2.
    - **§9.4 Level advancement, speed push and safety stop** — copy revision 3 §4.3, §4.4 and §4.5 as three
      paragraphs led by `**Level advancement.**`, `**Speed push.**` and `**Safety stop.**`.
    - **§9.5 Final controller** — copy revision 3 §4.6, and append: `The final controller is exported as a `.keras`
      file at `results/champions/<approach>/seed<k>`, together with its float64 weights (`.npy`) for dense
      networks or its genome (`.json`) for NEAT.`
    - **§9.6 Genetic algorithm** — copy revision 3 §5.1, then append the old §9.2 bullets `Each remaining new
      individual is created in three steps` (with its three numbered steps) unchanged.
    - **§9.7 Evolution strategy** — copy revision 3 §5.2, then append the old §9.3 bullet **Ties** unchanged and the
      sentence `The factor 0.82 follows Schwefel.`
    - **§9.8 Seeded evolution strategy** — copy revision 3 §5.3, then append the old §9.4 bullet
      **Interpretation** with `the baseline` replaced by `the DAgger-refined baseline`.
    - **§9.9 NEAT** — copy revision 3 §5.4.
    - **§9.10 Computational performance** — the old §9.5 unchanged, with these two bullets inserted before
      **Further measures**:

      > - **Fresh maps:** each generation builds its three maps in the worker processes that drive them. A map
      >   takes 0.1–0.6 s to build, against several seconds of driving. Every process keeps at most 64 built
      >   tracks, so memory stays constant over runs of many thousands of generations.
      > - **NEAT:** all genomes of a generation are compiled into flat arrays and evaluated together, depth by
      >   depth (§9.9), so NEAT drives its whole population at once like the other methods.

20. **§10.1 Table 3:** add the row `| NEAT | Yellow `#F0E442` |` after Seeded ES.
21. **§10.2:** in *Controls*, replace `For the GA, ES and seeded ES, the user sets the population size before
    starting. For the ES variants, this setting is the number of offspring λ.` with `For the GA, ES, seeded ES and
    NEAT, the user sets the population size before starting.` In *Training process*, replace `it sends the weights,
    fitness values, level and track seeds` with `it sends the population, fitness values, map success shares,
    level, target factor and map seeds`, and replace the last bullet's two sentences from `The seeded ES in the
    application starts …` with `The seeded ES in the application starts from the DAgger-refined baseline of the
    selected run seed. If it does not exist, the seeded ES cannot be started and the view states that notebook 03
    must be run first.` In *Track display*, replace `The GA and ES variants draw new tracks every generation` with
    `The evolutionary approaches draw new maps every generation`. Replace the *Side panel* list with:

    > - approach, level and generation
    > - the success share of each of the generation's maps against the 90% requirement, and the streak
    > - the target reduction during the speed push
    > - mutation strength (GA) or σ (ES, NEAT), and the number of species (NEAT)
    > - best and mean fitness
    > - a live plot of the lowest map success share per generation against the 90% line, with level
    >   advancements, the mastery goal and target reductions marked (for the imitation approaches, the best and
    >   mean validation accuracy)

22. **§10.3:** replace `*Best* is the seed whose controller achieved the highest mean validation fitness` with
    `*Best* is the seed whose controller succeeded on the most validation maps, ties broken by mean validation
    fitness`.
23. **§10.4 Table 4:** replace the four rows with:

    | Difficulty | Level | Track |
    |---|---|---|
    | Easy | 1 | Gentle loop on a 16 m road |
    | Medium | 2 | Circuit on a 14 m road |
    | Hard | 3 | Circuit with a main straight into a hairpin, possibly a chicane or esses, on a 12 m road |
    | Extra hard | 4 | Circuit with two hairpins, chicanes, esses and fast sweepers, on a 10 m road |

    and in the paragraph below, replace `GA and ES training starts` with `evolutionary training starts`.
24. **§11:** replace the whole chapter with: revision 3 §7 copied (keeping its subsections as §11.1 Test
    procedure, then Tables and Figures), with this subsection inserted after §11.1:

    > ### 11.2 Measurements of the course comparison
    >
    > - **Accuracy evaluation:** each controller's predicted classes are compared with the expert labels in the
    >   test dataset. For the evolutionary controllers, this measures agreement with an expert they never
    >   observed, which is reported as a result in its own right.
    > - **Inference time:** *batch*, Keras `predict` on 10,000 test rows in a single batch, reported per sample
    >   and averaged over five runs, with a warm-up call identical to the timed calls; *single sample*, the
    >   median time of 1,000 calls of the Keras model on one row. Inference is timed after all parallel work has
    >   finished, with the same prediction settings for every controller (no XLA compilation, one batch per
    >   call).
    > - **Parameters:** the Keras parameter count; for NEAT, the number of enabled connections plus biases.
    > - **Training time:** baseline training per seed; Keras Tuner search time (once per method) and training
    >   per seed; DAgger and speed push per seed; evolutionary run time per seed; dataset collection once, as a
    >   shared cost. The run conditions recorded in `results/run_info.json` (§3.3) are reported with the
    >   results.

    and with the copied revision 3 §7.2 and §7.3 renumbered §11.3 and §11.4.
25. **§12:** replace the chapter with `## 12. Verification` and:
    - **§12.1 Tests** — `Automated tests use pytest.` followed by the old bullets **Simulation**, **Expert driver**,
      **Network interface**, **Work distribution**, **Keras Tuner accounting**, **Measurement**, **Experiment
      runner** and **Application** unchanged; the old **Sensors** bullet with `on the level-1 circle` replaced by
      `on a circular test track`; then the bullets of revision 3 §10.1 copied. Delete the old bullets **Track
      generation**, **GA and ES**, **Seeded ES**, **Curriculum**, **GA survivors**, **Clean start** and
      **Re-simulation fidelity**; add instead:

      > - **Seeded ES:** the first initial individual equals the DAgger-refined network exactly, and the others
      >   differ from it.
      > - **Re-simulation fidelity:** re-simulating a logged generation reproduces exactly the fitness values
      >   recorded during training.

    - **§12.2 Pilot runs** — copy revision 3 §10.2.
26. **§13 Table 6:** replace the table with:

    | Parameter | Value |
    |---|---|
    | Time step | 0.05 s |
    | Grip limit μg (friction circle) | 12 m/s² |
    | Speed input scaling | 40 m/s (normalization only) |
    | Sensors | 9 rays from −90° to +90°, 50 m range |
    | Sphere tracing | Minimum step 0.5 m; at most 48 iterations per ray |
    | Laps per run | 3 (training, evaluation and application races) |
    | Time limit | Training: the target time; evaluation and application: twice the expert lap time per lap |
    | No-progress timeout | 5 s |
    | Curriculum maps | Table 1; direction random per map; at most 64 built tracks per process |
    | Maps per generation | 3, freshly generated |
    | Starting target factor f | 1.3 × three expert laps |
    | Success | All three laps within the target time |
    | Survival | Successes kept; others kept with probability equal to their share of laps in time; at most 90% of the population; at least the 50 fittest |
    | Level advancement and mastery goal | Lowest map success share ≥ 90% in three consecutive generations |
    | Speed push | Target reduced by 5% per mastered target; ends after 500 generations without mastering the current target |
    | Safety stop | 2,000 generations without a new best three-generation mean of the lowest map success share |
    | GA | N = 1,000; tournament 5; crossover p = 0.5; mutation p = 0.1; s_max = 0.1; s_min = s_max / 100 |
    | ES, seeded ES and speed push | N = 550; children from the 50 fittest; σ₀ = 0.1; factor 0.82; σ in [1e-4, 1.0]; reliability phase at 90% |
    | NEAT | N = 1,000; crossover 0.75; weight perturbation 0.8 (σ₀ = 0.5, two-phase rule as ES); weight replacement 0.05; add connection 0.05; add node 0.03; toggle 0.01; δ = E/N + D/N + 0.4 W̄; threshold from 3.0 towards 20 species; stagnation 20 generations; tanh hidden nodes |
    | DAgger | 20 maps per round; at most 30 epochs per round, patience 5, 10% hold-out; goal 18 of 20 checks in three consecutive rounds; stall after 10 rounds |
    | Keras training | Adam 1e-3; batch 256; at most 150 epochs; patience 5; `steps_per_execution` and `jit_compile` set by benchmark |
    | Keras Tuner | 30 trials (random search, Bayesian optimization with 10 initial random trials); Hyperband with `max_epochs` 150, factor 3, one iteration |
    | Model selection | Smallest network within one percentage point of the best validation accuracy |
    | Replication | Seeds 0, 1 and 2 for every method; Keras Tuner searches run once (seed 0) |
    | Map seeds | Validation 80–99; test 1000–1019; dataset 1,000,000–1,999,999; training from 2,000,000 |
    | Worker processes | `EC_RACING_JOBS`, default 6, chosen by benchmark per machine |
    | Application playback | 1× equals twice real time |

    Keep the sentence below the table.
27. **§14:** keep risks 1 (Course alignment), 5 (Seeded ES near the expert), 7 (Hyperband budget), 8 (Action
    resolution) and 9 (Replication scope) unchanged; replace risks 2, 3, 4 and 6 with the five risks of revision 3
    §12 copied (Unreachable levels, Run time, NEAT cost, DAgger data growth, Generator acceptance); renumber the
    list 1–10 in this order: Course alignment, Unreachable levels, Run time, NEAT cost, DAgger data growth,
    Generator acceptance, Seeded ES near the expert, Hyperband budget, Action resolution, Replication scope.
28. **§15:** replace the section body with:

    > - **Environment:** Python 3.13 with the packages pinned in `requirements.txt` (numpy, scipy, pandas,
    >   matplotlib, TensorFlow 2.21, Keras 3.13, keras-tuner, joblib, pygame, nbclient, nbformat, ipykernel,
    >   jupyter_client and pytest). A test verifies that the installed versions match the pinned ones.
    > - **Platform:** Linux. On the Windows desktop used for the reported run, the Windows Subsystem for Linux
    >   (WSL 2) with the project in the Linux file system.
    > - **Execution:** CPU only. Worker processes call `tf.config.set_visible_devices([], 'GPU')`. Every
    >   reported run uses the same machine, which is recorded in `results/run_info.json`.
    > - **Version control:** a private GitHub repository; the `results/` directory is excluded.

- [ ] **Step 8: Check the specification for left-over revision 2.0 text**

Run: `grep -n "200,000\|0–79\|Circle\|Oval\|at pace\|LevelTracker\|best network of each generation" docs/2026-10-05-ec-racing-design.md`
Expected: only the version-history sentence of the Summary may match (`200,000` does not appear there either, so
normally no output). Every match elsewhere is revision 2.0 text that step 7 missed: fix it the same way.
Run: `grep -n "revision 3 §" docs/2026-10-05-ec-racing-design.md`
Expected: no output (no copied text still points at revision 3 section numbers).

- [ ] **Step 9: Regenerate the PDFs**

Run:

```bash
MD2PDF=scripts/md2pdf.py
~/miniconda3/bin/python $MD2PDF docs/2026-10-05-ec-racing-design.md
~/miniconda3/bin/python $MD2PDF docs/2026-10-07-ec-racing-mastery-design.md
```

Expected: each command prints the path of its PDF. Open `docs/2026-10-05-ec-racing-design.pdf` (Read tool, pages
1–3 and the page with Table 6) and confirm that the tables and the diagram render. 

- [ ] **Step 10: Run the full fast and slow suites**

Run: `~/miniconda3/bin/python -m pytest -q`
Expected: all passed.
Run: `~/miniconda3/bin/python -m pytest -q -m slow`
Expected: all passed.

- [ ] **Step 11: Commit**

```bash
git add requirements.txt tests/test_requirements.py scripts/md2pdf.py README.md docs/2026-10-05-ec-racing-design.md \
        docs/2026-10-05-ec-racing-design.pdf docs/2026-10-07-ec-racing-mastery-design.pdf
git commit -m "Design specification 3.0 with revision 3; README for the WSL desktop; pinned requirements

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- requirements.txt tests/test_requirements.py scripts/md2pdf.py README.md \
        docs/2026-10-05-ec-racing-design.md docs/2026-10-05-ec-racing-design.pdf \
        docs/2026-10-07-ec-racing-mastery-design.pdf
```

(If regenerating `docs/2026-10-07-ec-racing-mastery-design.pdf` produced no change, `git add` stages nothing for it
and the commit still succeeds.)

- [ ] **Step 12: Push the branch**

Run: `git push origin build`
Expected: `build -> build` and no error. The desktop then gets the code with `git pull` (or the clone of the
README).

