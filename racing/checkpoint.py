"""Checkpoints of long runs and the record of the settings a run was made with (spec §3.3). A run saves its
state after every generation or round and continues from it after an interruption; a run made with other settings
is never resumed and its results are never reused."""
import json
import os
import pickle
import platform
from pathlib import Path

import numpy as np

from racing import config, fastsim

CHECKPOINT_NAME = "checkpoint.pkl"

# Configuration values that affect results. The worker count and the track cache only change speed and memory, and
# the viewer's values change nothing a run computes, so they are left out. Tasks that add such values append them.
SETTINGS_KEYS = (
    "DT", "G", "GRIP_LOW", "GRIP_MAX", "GRIP_FULL_SPEED", "GRIP_RISE", "TOP_SPEED", "YAW_MIN_SPEED", "SPEED_SCALE",
    "STALL_TIME", "TIME_LIMIT_FACTOR", "LAPS", "TIME_EPS",
    "N_RAYS", "RAY_ANGLES", "RAY_SCALE", "RAY_TRACE_LENGTH", "RAY_MIN_STEP", "RAY_MAX_ITERATIONS",
    "CENTERLINE_SPACING", "GRID_RES", "GRID_MARGIN", "OVERLAP_MARGIN", "MAX_TRACK_ATTEMPTS", "PROGRESS_BACK",
    "PROGRESS_AHEAD", "LEVELS", "MAX_LEVEL", "CIRCUIT_STRETCH", "DIRECTION_SALT", "VALIDATION_LEVEL",
    "WIDTH_VARIATION", "CHICANE_MIN_EDGE", "CHICANE_STEP", "ESSES_MIN_EDGE", "ESSES_SPACING", "WIDTH_SALT",
    "N_INPUTS", "WIDTH_SCALE", "STEER_LEVELS", "THROTTLE_LEVELS", "N_ACTIONS", "BASELINE_HIDDEN",
    "SENSOR_START", "SENSOR_LENGTH_LIMITS", "STEER_GAIN_LIMITS", "DEFAULT_SETUP", "SETUP_MUTATION_P",
    "SETUP_MUTATION_SD",
    "EXPERT_LOOKAHEAD_BASE", "EXPERT_LOOKAHEAD_TIME", "EXPERT_GRIP_MARGIN", "EXPERT_SPEED_BAND", "EXPERT_SIGHT",
    "EXPERT_VALIDATION_TIME", "EXPERT_PREVIEW_TIME", "EXPERT_CROSS_TRACK_GAIN", "EXPERT_HEADING_GAIN",
    "PERTURB_INTERVAL", "PERTURB_DURATION", "DATASET_SALT",
    "DATASET_SEED_RANGE", "DATASET_TRACKS",
    "VAL_SEEDS", "RUN_SEEDS", "SEARCH_SEED", "TRAIN_SEED_START",
    "TEST_LEVEL", "TRAINING_CIRCUITS", "TEST_CIRCUITS", "CIRCUIT_DATA",
    "RACING_LINE_MARGIN", "RACING_LINE_STEP", "RACING_LINE_ITERATIONS", "RACING_LINE_ROUNDS",
    "RACING_LINE_TOLERANCE", "RACING_LINE_SMOOTHING", "LINE_AHEAD", "LINE_CURVATURE_SCALE", "LINE_CURVATURE_CLIP",
    "FINAL_MIN_SUCCESSES",
    "TRACKS_PER_GEN", "TARGET_FACTOR", "SPEED_STEP", "SPEED_MASTERY_FRACTION", "MASTERY_FRACTION", "MASTERY_STREAK", "SPEED_PATIENCE",
    "STALL_GENERATIONS", "STALL_BOOST", "STALL_STEER_P", "STALL_STEER_SD",
    "SAFETY_PATIENCE", "KEEP_MAX_FRACTION", "ELITE_FLOOR",
    "GA_POP", "GA_TOURNAMENT", "GA_CROSSOVER_P", "GA_MUTATION_P", "GA_S_MAX", "GA_S_MIN",
    "ES_POP", "ES_MU", "ES_SIGMA0", "ES_FACTOR", "ES_SIGMA_MIN", "ES_SIGMA_MAX",
    "DAGGER_MAPS", "DAGGER_EPOCHS", "DAGGER_HOLDOUT", "DAGGER_GOAL", "DAGGER_STREAK", "DAGGER_PATIENCE",
    "DAGGER_SALT",
    "LEARNING_RATE", "BATCH_SIZE", "MAX_EPOCHS", "PATIENCE", "KT_TRIALS", "KT_BAYES_INITIAL", "HYPERBAND_FACTOR",
    "HYPERBAND_ITERATIONS", "SIZE_RULE_TOLERANCE", "STEPS_PER_EXECUTION", "JIT_COMPILE",
)

# Settings that affect only one method's results (spec §3.3): recorded and checked only in that method's runs, so the
# runs of the other methods made before such a setting existed stay valid. NEAT's settings were shared until version
# 3.11: a record made before version 3.12 still lists them, and check_settings ignores them in another method's run.
METHOD_SETTINGS_KEYS = {
    "ga": ("GA_START_HIDDEN", "GA_ADD_UNIT", "GA_REMOVE_UNIT", "GA_ADD_LAYER", "GA_REMOVE_LAYER"),
    "neat": ("NEAT_POP", "NEAT_CROSSOVER", "NEAT_TOURNAMENT", "NEAT_DISABLE_INHERIT", "NEAT_PERTURB",
             "NEAT_REPLACE", "NEAT_ADD_CONN", "NEAT_ADD_NODE", "NEAT_TOGGLE", "NEAT_SIGMA0", "NEAT_C1",
             "NEAT_C2", "NEAT_C3", "NEAT_SMALL_GENOME", "NEAT_THRESHOLD0", "NEAT_THRESHOLD_STEP",
             "NEAT_TARGET_SPECIES", "NEAT_STAGNATION"),
}


# Settings used only when a run picks its final controller (spec §3.3, §9.5, version 3.15): recorded with every run but
# never compared, so a stricter final pick refuses no checkpoint and redoes no stored result.
END_SETTINGS_KEYS = ("FINAL_MIN_SUCCESSES",)


def settings_keys(approach=None):
    """The settings that affect the results of `approach`'s runs: the shared ones, then the method's own."""
    return SETTINGS_KEYS + METHOD_SETTINGS_KEYS.get(approach, ())


class SettingsMismatch(RuntimeError):
    """The settings recorded with a run differ from the current configuration."""


class UnreadableRecord(RuntimeError):
    """A run record that cannot be read, such as one cut short by a reboot; the message names the file."""


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


def settings_record(approach=None):
    """{name: value} of the settings that affect the results of `approach`'s runs (the shared settings when approach
    is None), in JSON form."""
    return {name: plain(getattr(config, name)) for name in settings_keys(approach)}


def processor():
    """The processor's model name, from /proc/cpuinfo on Linux."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def machine_record():
    """The machine a run finished on: runs of one experiment may be trained on several machines (scripts/fleet.py),
    so each run's summary says where, with which thread setting of the compiled simulator and how many worker
    processes (n_jobs), and which simulator ran its time trials (spec §3.3, §5.7)."""
    return {"host": platform.node() or "unknown", "processor": processor(), "logical_processors": os.cpu_count(),
            "threads": config.thread_label(), "n_jobs": config.N_JOBS, "simulator": fastsim.description()}


def check_settings(recorded, approach=None):
    """Raise SettingsMismatch if `recorded` differs from the current settings of `approach`'s runs, naming every
    difference. Another method's own settings in `recorded` (NEAT's, in a record made before version 3.12) are
    ignored, and the settings of END_SETTINGS_KEYS are never compared."""
    current = settings_record(approach)
    recorded = plain(recorded)
    ignored = {name for method, names in METHOD_SETTINGS_KEYS.items() if method != approach for name in names}
    ignored |= set(END_SETTINGS_KEYS)
    differences = [f"{name}: recorded {recorded.get(name)!r}, now {current.get(name)!r}"
                   for name in sorted((set(recorded) | set(current)) - ignored)
                   if recorded.get(name) != current.get(name)]
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


def write_json(path, data):
    """Write `data` as JSON through a temporary file in the same folder that is synced to disk before it replaces
    `path`, so neither an interruption nor a reboot leaves a partial or empty record (spec §3.3). Everything
    written before the record, such as the final controller or stage file it vouches for, reaches the disk first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        text = json.dumps(data, indent=2)
        os.sync()                                      # a record never reaches the disk before what it describes
        with open(temporary, "w") as f:
            f.write(text)
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


def read_record(path):
    """A run record (JSON object); UnreadableRecord if it is empty, cut short or not an object."""
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise UnreadableRecord(f"{path} cannot be read ({type(error).__name__}), probably cut short by a crash or "
                               "a reboot: delete only this file and run again; its step is then redone") from error
    if not isinstance(record, dict):
        raise UnreadableRecord(f"{path} is not a run record: delete only this file and run again; its step is "
                               "then redone")
    return record


def ended_before_hardening(summary):
    """Whether `summary` records a run of the experiment that ended under version 3.14 or earlier, at its speed limit
    or by an end request: it has no hardening record. With its checkpoint beside it, such a run is not finished; it
    resumes into hardening (spec §3.3, §9.4, version 3.15)."""
    return "hardening" not in summary and summary.get("outcome") in ("speed_limit", "time_limit")


def completed(run_dir, name="summary.json", state_name=CHECKPOINT_NAME, approach=None):
    """The summary of a completed run in `run_dir`, or None if the run has not completed. A summary made with other
    settings, or without a settings record (an earlier version), raises SettingsMismatch. A summary that cannot be
    read counts as not completed when the run's checkpoint `state_name` is next to it (the run finishes again from
    the checkpoint); otherwise it raises UnreadableRecord naming the file. `approach` names the method whose own
    settings are checked too."""
    path = Path(run_dir) / name
    if not path.exists():
        return None
    try:
        summary = read_record(path)
    except UnreadableRecord:
        if (path.parent / state_name).exists():
            return None
        raise
    if "settings" not in summary:
        raise SettingsMismatch(f"{path} has no settings record (made by an earlier version); "
                               "start over with scripts/run_notebooks.py --fresh")
    check_settings(summary["settings"], approach)
    return summary
