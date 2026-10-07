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
RUN_SEEDS = (0, 1, 2)        # every method is run once per seed (spec §11)
SEARCH_SEED = 0              # each Keras Tuner search runs once (spec §8.2)

# --- Curriculum (spec §9.1) ---
TRACKS_PER_GEN = 3
LEVEL_UP_FRACTION = 0.5     # share of the whole generation that must finish at pace to count toward a level-up
LEVEL_UP_STREAK = 3
PACE_FACTOR = 1.3            # finishing at pace: lap time ≤ PACE_FACTOR × expert lap time; confirmed by pilots

# --- Genetic algorithm (spec §9.2) ---
GA_POP = 1000
GA_KEPT = 50                 # the minimum number of survivors
GA_KEEP_MAX_FRACTION = 0.5   # survivors are at most this share of the population
GA_TOURNAMENT = 5
GA_CROSSOVER_P = 0.5
GA_MUTATION_P = 0.1
GA_S_MAX = 0.1               # chosen by the pilot runs (scripts/pilots.py) from 0.5, 0.2 and 0.1, fastest level progression
GA_S_MIN = GA_S_MAX / 100

# --- Evolution strategy (spec §9.3) ---
ES_MU = 50
ES_LAMBDA = 500
ES_SIGMA0 = 0.1
ES_FACTOR = 0.82
ES_SIGMA_MIN = 1e-4
ES_SIGMA_MAX = 1.0

# --- Budget and parallelism ---
EVAL_BUDGET = 200_000
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
