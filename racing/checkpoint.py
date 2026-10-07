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
