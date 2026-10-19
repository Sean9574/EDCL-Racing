"""Default parameters of EC Racing (design specification §13). Every module reads its constants from here."""
import os
from pathlib import Path

import numpy as np

# --- Simulation (spec §5) ---
DT = 0.05                    # s, fixed time step (20 Hz)
GRIP = 12.0                  # m/s², friction-circle limit μg; the only vehicle limit
YAW_MIN_SPEED = 1.0          # m/s, speed floor used only in the yaw-rate formula
SPEED_SCALE = 40.0           # m/s, input normalization only; not a speed limit
STALL_TIME = 5.0             # s without progress ends a run
TIME_LIMIT_FACTOR = 2.0      # time limit per lap = factor × expert lap time
LAPS = 3                     # laps per run in training and evaluation (spec §5.4)
TIME_EPS = 1e-9              # tolerance for comparisons of accumulated simulation time

# --- Sensors (spec §5.2) ---
N_RAYS = 9
RAY_ANGLES = np.radians(np.linspace(-90.0, 90.0, N_RAYS))   # index 0 = right, 4 = ahead, 8 = left
RAY_RANGE = 50.0             # m, revision 3's sensor length and the ray input scale (an input of 1.0 is 50 m)
RAY_MIN_STEP = 0.5           # m, smallest sphere-tracing step
RAY_MAX_ITERATIONS = 48      # sphere-tracing iterations of a RAY_RANGE ray; scaled with each ray's length (spec §5.2)

# --- Car setup (spec §9.11) ---
DEFAULT_SETUP = (RAY_RANGE, 1.0)   # (sensor length m, steering sensitivity) of every car in revision 3
SENSOR_LENGTH_LIMITS = (10.0, 150.0)   # m, the sensor lengths evolution may choose (spec §9.11)
STEER_GAIN_LIMITS = (1.0 / 3.0, 3.0)   # the steering sensitivities evolution may choose
SETUP_MUTATION_P = 0.2       # chance that each setup value of a child is mutated (spec §9.11)
SETUP_MUTATION_SD = 0.1      # standard deviation of that mutation on the natural-log scale (about ±10%)

# --- Track geometry (spec §4) ---
CENTERLINE_SPACING = 1.0     # m
GRID_RES = 0.5               # m per road-grid cell
GRID_MARGIN = 10.0           # m of grass around the road in the grid
OVERLAP_MARGIN = 6.0         # m of grass required between separate parts of the road
MAX_TRACK_ATTEMPTS = 200
PROGRESS_BACK = 5.0          # m behind the last matched centerline point
PROGRESS_AHEAD = 30.0        # m ahead of the last matched centerline point
LEVELS = {                   # spec Table 1: Easy loops, circuits with F1-style features above it
    1: {"kind": "loop", "points": (4, 6), "radius": (60.0, 110.0), "min_corner": 45.0},
    2: {"kind": "circuit", "points": (6, 8), "radius": (60.0, 170.0), "slow": (25.0, 45.0), "fast": (50.0, 80.0),
        "main": 0.0, "hairpins": 0, "chicanes": (0, 0), "esses": (0, 0), "min_corner": 24.0},
    3: {"kind": "circuit", "points": (9, 12), "radius": (70.0, 220.0), "slow": (16.0, 35.0), "fast": (45.0, 90.0),
        "main": 170.0, "hairpins": 1, "chicanes": (0, 1), "esses": (0, 1), "min_corner": 14.0},
    4: {"kind": "circuit", "points": (13, 17), "radius": (80.0, 260.0), "slow": (10.0, 30.0), "fast": (45.0, 100.0),
        "main": 210.0, "hairpins": 2, "chicanes": (1, 2), "esses": (1, 2), "min_corner": 9.0},
}
MAX_LEVEL = 4
CIRCUIT_STRETCH = (1.0, 1.5)  # horizontal stretch of a circuit's base polygon (spec §4.3)
DIRECTION_SALT = 7           # per-map direction draw: default_rng([seed, level, DIRECTION_SALT]) (spec §4.4)
WIDTH_RANGE = (10.0, 16.0)   # m: every map's road width is drawn from this range (spec §4.2)
WIDTH_SALT = 13              # a map's width: default_rng([seed, level, WIDTH_SALT]), the same for every attempt
TRACK_CACHE_SIZE = 8         # built tracks kept per process (an Extra hard track holds 10-14 MB); training maps are
                             # fresh every generation, so a larger cache only costs memory in every worker

# --- Network interface (spec §6) ---
WIDTH_SCALE = 20.0           # m, input scale of the road width (spec §5.2)
N_INPUTS = N_RAYS + 3        # rays, speed, position across the road, road width (spec §5.2)
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

# --- Seeds (spec §4.6) ---
VAL_SEEDS = range(80, 100)
TEST_SEEDS = range(1000, 1020)
DATASET_SEED_RANGE = (1_000_000, 2_000_000)   # the imitation dataset's maps are drawn once from this range
DATASET_TRACKS = {1: 1, 2: 20, 3: 20, 4: 80}  # dataset maps per level (spec §7.3)
TRAIN_SEED_START = 2_000_000                  # evolutionary and DAgger maps are drawn from here on (spec §4.6)
SEEDED_RUN = 0               # run 0 uses seed 0 and replays exactly; every other run sets no seed (spec §3.3)
RUN_SEEDS = (0, 1, 2, 3)  # run 0 uses seed 0, runs 1-3 set no seed (spec §3.3, §11)
SEARCH_SEED = 0              # each Keras Tuner search runs once (spec §8.2)

# --- Time trial, survival and levels (spec §9.2-9.4) ---
TRACKS_PER_GEN = 3
TARGET_FACTOR = 1.3          # time trial: starting target = TARGET_FACTOR × LAPS expert laps (spec §9.2)
MASTERY_FRACTION = 0.9       # lowest map success share for a passing generation (spec §9.4)
MASTERY_STREAK = 3           # consecutive passing generations to advance a level or master a target
SPEED_STEP = 0.98            # target factor multiplier per mastered target in the speed push (spec §9.4)
SPEED_MASTERY_FRACTION = 0.75  # lowest map success share of a passing speed-push generation (spec §9.4)
SPEED_PATIENCE = 600         # generations without mastering the current target that end the speed push (300 to 3.5)
STALL_GENERATIONS = 100      # generations of the speed push without a mastered target after which it is stalled
STALL_BOOST = 3.0            # in a stall, the decision layer's mutation is this many times as strong (spec §9.4)
STALL_STEER_P = 0.5          # in a stall, a child's steering sensitivity mutates with this probability ...
STALL_STEER_SD = 0.2         # ... and this standard deviation on the log scale (spec §9.4, §9.11)
SAFETY_PATIENCE = 2000       # generations without a new best 3-generation mean lowest share that end a run (spec §9.4)
KEEP_MAX_FRACTION = 0.9      # kept cars fill at most this share of the population (spec §9.3)
ELITE_FLOOR = 50             # at least this many of the fittest cars are kept

# --- Genetic algorithm (spec §9.6) ---
GA_POP = 1000
GA_TOURNAMENT = 5
GA_CROSSOVER_P = 0.5
GA_MUTATION_P = 0.1
GA_S_MAX = 0.1               # chosen by the pilot runs (scripts/pilots.py) from 0.5, 0.2 and 0.1, fastest level progression
GA_S_MIN = GA_S_MAX / 100

# --- Evolution strategy (spec §9.7) ---
ES_POP = 550
ES_MU = 50                   # children are copies of one of the ES_MU fittest cars
ES_SIGMA0 = 0.1
ES_FACTOR = 0.82
ES_SIGMA_MIN = 1e-4
ES_SIGMA_MAX = 1.0

# --- NEAT (spec §9.9) ---
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

# --- Parallelism ---
def worker_count(environ=os.environ):
    """Worker processes (spec §3.3): those of the numpy simulator, DAgger's driving, dataset collection and the final
    evaluation. EC_RACING_JOBS, or 6 when it is not set. Each holds its own built maps, so their number is fixed."""
    value = environ.get("EC_RACING_JOBS", "").strip()
    if not value:
        return 6
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"EC_RACING_JOBS must be a whole number of at least 1, not {value!r}")
    return int(value)


N_JOBS = worker_count()      # results do not depend on it (spec §9.10); recorded in results/run_info.json


def thread_setting(environ=os.environ):
    """EC_RACING_THREADS (spec §3.3): a fixed number of threads for the compiled simulator, or None when it is not set,
    for one thread per logical processor available to the process."""
    value = environ.get("EC_RACING_THREADS", "").strip()
    if not value:
        return None
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"EC_RACING_THREADS must be a whole number of at least 1, not {value!r}")
    return int(value)


THREADS = thread_setting()   # results do not depend on it (spec §9.10); recorded in results/run_info.json


def thread_count():
    """Threads of one call of the compiled simulator (spec §3.3, §5.7): THREADS when it is set, otherwise the logical
    processors available to this process now, so runs that share a machine share its processors through the operating
    system, and a run alone uses them all."""
    return THREADS if THREADS is not None else len(os.sched_getaffinity(0))


def thread_label():
    """THREADS as the run records show it: the number, or "all"."""
    return "all" if THREADS is None else THREADS


def simulator_choice(environ=os.environ):
    """The simulator (spec §5.7): EC_RACING_SIM, "auto" (the compiled simulator when it builds, otherwise numpy;
    the default) or "numpy"."""
    value = environ.get("EC_RACING_SIM", "").strip() or "auto"
    if value not in ("auto", "numpy"):
        raise ValueError(f"EC_RACING_SIM must be auto or numpy, not {value!r}")
    return value


SIMULATOR = simulator_choice()   # results do not depend on it (spec §5.7); recorded in results/run_info.json

# A Keras job and the compiled runs beside it share the processors (spec §3.3). Measured on the development machine, one
# training epoch of the baseline: alone 0.186 s with one TensorFlow thread per processor, 0.144 s with 4; beside three
# compiled runs of 20 threads 1.13 s, and 0.48 s with 4 threads and the runs at the lowest priority.
KERAS_THREADS = 4            # TensorFlow intra-op threads of every Keras job
KERAS_INTER_THREADS = 2      # TensorFlow inter-op threads of every Keras job
EVOLUTION_NICE = 19          # an evolution phase (and a speed push) runs at the lowest priority, so Keras jobs come first

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
STEPS_PER_EXECUTION = 64     # batches per training-function call; racing/bench.py, s/epoch with XLA: 1: 0.367, 4: 0.201, 16: 0.134, 64: 0.128
JIT_COMPILE = False          # XLA is ~11% faster per epoch (racing/bench.py, 64: 0.128 vs 0.144 s), but on the
                             # desktop's i9-12900K its JIT failed after ~130 Keras Tuner trials ("Failed to
                             # materialize symbols") and stayed broken in that process; reliability wins

# --- DAgger (spec §8.4) ---
DAGGER_MAPS = 20             # fresh Extra hard maps driven per round, and as many more for the round's check
DAGGER_EPOCHS = 30           # at most, per round
DAGGER_HOLDOUT = 0.1         # share of the aggregated rows held out for early stopping
DAGGER_GOAL = 18             # successful checks out of DAGGER_MAPS ...
DAGGER_STREAK = 3            # ... in this many consecutive rounds
DAGGER_PATIENCE = 10         # rounds without a new best check result before the refinement stops
DAGGER_SALT = 31             # a refinement's random generator: default_rng([run seed, method index, DAGGER_SALT])

# --- Approaches and colors (spec §10.1) ---
APPROACHES = ("baseline", "kt_random", "kt_hyperband", "kt_bayesian", "es", "ga", "es_seeded", "neat")
TUNER_APPROACHES = ("kt_random", "kt_hyperband", "kt_bayesian")
EVOLUTION_APPROACHES = ("ga", "es", "es_seeded", "neat")
IMITATION_APPROACHES = ("baseline", *TUNER_APPROACHES)
RUN_ORDER = tuple((a, s) for a in (*IMITATION_APPROACHES, *EVOLUTION_APPROACHES) for s in RUN_SEEDS)  # spec §3.3
LABELS = {"baseline": "Baseline", "kt_random": "KT Random", "kt_hyperband": "KT Hyperband",
          "kt_bayesian": "KT Bayesian", "es": "ES", "ga": "GA", "es_seeded": "Seeded ES", "neat": "NEAT"}
COLORS = {"baseline": "#7F7F7F", "kt_random": "#E69F00", "kt_hyperband": "#56B4E9",
          "kt_bayesian": "#CC79A7", "es": "#009E73", "ga": "#D55E00", "es_seeded": "#0072B2",
          "neat": "#F0E442"}         # the remaining Okabe-Ito color (spec §10.1)

# --- Application (spec §10) ---
WINDOW_SIZE = (1280, 800)
PANEL_WIDTH = 380
FPS = 60
POPULATION_PRESETS = (100, 250, 550, 1000, 2000)    # 550 is the ES population (spec §9.7)
SPEEDS = (1, 2, 4, 8)
PLAYBACK_BASE = 2.0          # simulated seconds per real second at 1× playback
MAX_STEPS_PER_FRAME = 10     # simulation steps per drawn frame; a stalled frame cannot trigger a burst
DEFAULT_SHOWN = 100
KT_VIEW_SEED = 80
RACE_LAPS = LAPS
DIFFICULTIES = {1: "Easy", 2: "Medium", 3: "Hard", 4: "Extra hard"}   # curriculum level → name (spec §10.4)
APP_GRIPS = {"Normal": GRIP, "Unlimited": 10.0 * GRIP}   # m/s², display grip of the viewer's own simulations only
                                                         # (spec §10.5); never a recorded setting
ROBUSTNESS_WIDTHS = (8.0, 10.0, 14.0, 18.0)   # m, the robustness table's widths (spec §11.1); evaluation only
REPLAY_POLL_SECONDS = 1.0    # the notebook-run view looks for a newer generation at most this often
CAMERA_ZOOMS = (1, 2, 4, 8)  # the camera button's steps; above 1 the camera follows the best car or the race leader

# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
FONT_PATH = PROJECT_DIR / "racing" / "fonts" / "DejaVuSansCondensed.ttf"   # the viewer's font, shipped with the project so
                             # every text has the same width on every machine (racing/fonts/LICENSE)
RESULTS_DIR = PROJECT_DIR / "results"
DATASET_PATH = RESULTS_DIR / "dataset.npz"
DATASET_INFO_PATH = RESULTS_DIR / "dataset_info.json"
RUNS_DIR = RESULTS_DIR / "runs"
APP_RUNS_DIR = RESULTS_DIR / "app_runs"
APP_CHAMPIONS_DIR = APP_RUNS_DIR / "champions"   # the best car of each viewer run (spec §10.6); never evaluated
CHAMPIONS_DIR = RESULTS_DIR / "champions"
BEST_SEEDS_NAME = "best_seeds.json"
PILOTS_DIR = RESULTS_DIR / "pilots"
RUN_INFO_PATH = RESULTS_DIR / "run_info.json"
RUN_LOG_PATH = RESULTS_DIR / "run_log.txt"
STATUS_PATH = Path(os.environ.get("EC_RACING_STATUS") or RESULTS_DIR / "status.json")   # a job trained side by side
                             # gets its own status file, also inside a notebook's kernel (spec §3.3)
STATUS_DIR = RESULTS_DIR / "status"          # one status file per run when runs train side by side (fleet)
TRAIN_LOGS_DIR = RESULTS_DIR / "train_logs"  # what each run trained side by side prints itself
STATUS_STALE_AFTER = 600.0   # s; the monitor marks an older status as not updated, as a stopped run leaves it
