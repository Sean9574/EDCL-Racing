"""Default parameters of EC Racing (design specification §13). Every module reads its constants from here."""
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

# --- Seeds (revision 3 §3.4) ---
VAL_SEEDS = range(80, 100)
TEST_SEEDS = range(1000, 1020)
DATASET_SEED_RANGE = (1_000_000, 2_000_000)   # the imitation dataset's maps are drawn once from this range
DATASET_TRACKS = {1: 1, 2: 20, 3: 20, 4: 80}  # dataset maps per level (revision 3 §6.1)
TRAIN_SEED_START = 2_000_000                  # evolutionary and DAgger maps are drawn from here on (revision 3 §3.4)
RUN_SEEDS = (0, 1, 2)        # every method is run once per seed (spec §11)
SEARCH_SEED = 0              # each Keras Tuner search runs once (spec §8.2)

# --- Time trial, survival and levels (revision 3 §4) ---
TRACKS_PER_GEN = 3
TARGET_FACTOR = 1.3          # time trial: starting target = TARGET_FACTOR × LAPS expert laps (revision 3 §4.1)
MASTERY_FRACTION = 0.9       # lowest map success share for a passing generation (revision 3 §4.3)
MASTERY_STREAK = 3           # consecutive passing generations to advance a level or master a target
SPEED_STEP = 0.95            # target factor multiplier per mastered target in the speed push (revision 3 §4.4)
SPEED_PATIENCE = 500         # generations without mastering the current target that end the speed push
SAFETY_PATIENCE = 2000       # generations without a new best 3-generation mean lowest share that end a run (§4.5)
KEEP_MAX_FRACTION = 0.9      # kept cars fill at most this share of the population (revision 3 §4.2)
ELITE_FLOOR = 50             # at least this many of the fittest cars are kept

# --- Genetic algorithm (revision 3 §5.1) ---
GA_POP = 1000
GA_TOURNAMENT = 5
GA_CROSSOVER_P = 0.5
GA_MUTATION_P = 0.1
GA_S_MAX = 0.1               # chosen by the pilot runs (scripts/pilots.py) from 0.5, 0.2 and 0.1, fastest level progression
GA_S_MIN = GA_S_MAX / 100

# --- Evolution strategy (revision 3 §5.2) ---
ES_POP = 550
ES_MU = 50                   # children are copies of one of the ES_MU fittest cars
ES_SIGMA0 = 0.1
ES_FACTOR = 0.82
ES_SIGMA_MIN = 1e-4
ES_SIGMA_MAX = 1.0

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

# --- Parallelism ---
N_JOBS = 6                   # worker processes; racing/bench.py, worst-case generation: 1: 6.87 s, 3: 2.61 s, 6: 1.88 s, 9: 3.12 s, 12: 2.94 s, 15: 2.74 s

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
JIT_COMPILE = True           # XLA compilation; racing/bench.py: faster than without at every steps_per_execution (64: 0.128 vs 0.144 s)

# --- DAgger (revision 3 §6.3) ---
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
RUN_ORDER = tuple((a, s) for a in (*IMITATION_APPROACHES, *EVOLUTION_APPROACHES) for s in RUN_SEEDS)  # revision 3 §8.3
LABELS = {"baseline": "Baseline", "kt_random": "KT Random", "kt_hyperband": "KT Hyperband",
          "kt_bayesian": "KT Bayesian", "es": "ES", "ga": "GA", "es_seeded": "Seeded ES", "neat": "NEAT"}
COLORS = {"baseline": "#7F7F7F", "kt_random": "#E69F00", "kt_hyperband": "#56B4E9",
          "kt_bayesian": "#CC79A7", "es": "#009E73", "ga": "#D55E00", "es_seeded": "#0072B2",
          "neat": "#F0E442"}         # the remaining Okabe-Ito color (revision 3 §9)

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
RACE_LAPS = LAPS
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
STATUS_PATH = RESULTS_DIR / "status.json"
