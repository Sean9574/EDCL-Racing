"""Scripted expert driver (spec §7.1), whose steering and throttle are fractions of the grip a(v), and imitation
dataset collection on generated maps and the real training layouts (spec §7.2-7.3)."""
import os
import time
from pathlib import Path

import numpy as np

from racing import checkpoint, config, fastsim, sim
from racing.actions import class_to_controls, controls_to_class
from racing.tracks import get_track


def expert_controls(track, cars):
    """Continuous steering and throttle chosen by the expert for every car, as fractions of the grip a(v) at the car's
    speed (spec §7.1). It coasts while its speed is within its band of the target speed: the larger of
    EXPERT_SPEED_BAND and one step's full speed change a(v) DT, so a step of gas or brake never carries it across the
    whole band, and in a steady corner it settles at a speed and keeps all of a(v) for turning. While its steering is
    saturated it does not accelerate, so a corner's exit taken at full lock keeps all of a(v) for turning."""
    m = len(track.centerline)
    here = np.mod(cars.index, m)
    look = config.EXPERT_LOOKAHEAD_BASE + config.EXPERT_LOOKAHEAD_TIME * cars.speed
    target = track.centerline[np.mod(here + np.round(look / track.spacing).astype(np.int64), m)]
    dx, dy = target[:, 0] - cars.x, target[:, 1] - cars.y
    alpha = np.angle(np.exp(1j * (np.arctan2(dy, dx) - cars.heading)))
    path_curvature = 2.0 * np.sin(alpha) / np.maximum(np.hypot(dx, dy), 1e-6)
    grip = sim.grip_at(cars.speed)
    steer = np.clip(cars.speed ** 2 * path_curvature / grip, -1.0, 1.0)
    target = target_speed(track, here)
    band = np.maximum(config.EXPERT_SPEED_BAND, grip * config.DT)
    full_lock = np.abs(steer) >= 1.0
    throttle = np.where((cars.speed < target - band) & ~full_lock, 1.0,
                        np.where(cars.speed > target + band, -1.0, 0.0))
    return steer, throttle


def corner_speed(kappa):
    """The highest speed v with v² κ ≤ EXPERT_GRIP_MARGIN × a(v) for curvatures κ in 1/m (spec §7.1): below the grip
    ceiling v² (κ − m c) ≤ m a0, at the ceiling v² κ ≤ m a_max; the lower of the two bounds."""
    margin = config.EXPERT_GRIP_MARGIN
    kappa = np.maximum(np.asarray(kappa, dtype=float), 1e-9)
    excess = kappa - margin * config.GRIP_RISE
    rising = np.where(excess > 0.0, np.sqrt(margin * config.GRIP_LOW / np.maximum(excess, 1e-12)), np.inf)
    return np.minimum(rising, np.sqrt(margin * config.GRIP_MAX / kappa))


def sight_speed(sight=None):
    """The highest speed from which the expert stops within `sight` m (EXPERT_SIGHT) braking at EXPERT_GRIP_MARGIN ×
    a(v) (spec §7.1): below GRIP_FULL_SPEED the braking distance from v is ln(a(v) / a0) / (2 m c), above it the
    ceiling adds (v² − v_full²) / (2 m a_max). About 98 m/s for 150 m."""
    sight = config.EXPERT_SIGHT if sight is None else float(sight)
    margin, low, rise = config.EXPERT_GRIP_MARGIN, config.GRIP_LOW, config.GRIP_RISE
    full = config.GRIP_FULL_SPEED
    rising_distance = np.log((low + rise * full * full) / low) / (2.0 * margin * rise)
    if sight <= rising_distance:
        return float(np.sqrt(low * np.expm1(2.0 * margin * rise * sight) / rise))
    return float(np.sqrt(full * full + 2.0 * margin * config.GRIP_MAX * (sight - rising_distance)))


def target_speed(track, here):
    """The expert's target speed at centerline points `here` (spec §7.1): the corner speed of the tightest curve within
    its sight, capped by the speed from which it can stop within its sight and by the top speed."""
    m = len(track.centerline)
    sight = int(round(config.EXPERT_SIGHT / track.spacing))
    window = np.mod(here[:, None] + np.arange(sight + 1)[None, :], m)
    kappa = np.max(np.abs(track.curvature[window]), axis=1)
    return np.minimum(np.minimum(corner_speed(kappa), sight_speed()), config.TOP_SPEED)


def expert_policy(track):
    """Policy for sim.simulate: the expert, quantized to the 15 action classes."""
    def policy(obs, cars):
        return controls_to_class(*expert_controls(track, cars))
    return policy


def expert_lap_time(track):
    """The expert's lap time on this track, or None if it crashes, stalls or runs out of time; on the compiled
    simulator when it is available (spec §5.7), with the same result."""
    if fastsim.available():
        return fastsim.expert_lap_time(track)
    result = sim.simulate(track, expert_policy(track), 1, time_limit=config.EXPERT_VALIDATION_TIME)
    return float(result.finish_time[0]) if result.finished[0] else None


def respawn(track, cars):
    """Put every crashed car back on the centerline at its matched point, heading along the track at the
    expert's target speed there, and let it continue. Returns the number of cars moved."""
    j = np.flatnonzero(cars.crashed)
    here = np.mod(cars.index[j], len(track.centerline))
    cars.x[j] = track.centerline[here, 0]
    cars.y[j] = track.centerline[here, 1]
    cars.heading[j] = np.arctan2(track.tangent[here, 1], track.tangent[here, 0])
    cars.speed[j] = target_speed(track, here)
    cars.crashed[j] = False
    cars.done[j] = False
    return len(j)


def collect_rows(track, rng):
    """One lap of expert driving with random steering disturbances. Labels are always the expert's own
    choice. A crash puts the car back on the centerline (spec §7.2). Returns inputs, labels and the
    number of respawns."""
    cars = sim.start_cars(track, 1)
    next_kick = rng.uniform(*config.PERTURB_INTERVAL)
    kick_end, kick_steer = -1.0, 0.0
    inputs, labels, respawns = [], [], 0
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
        if cars.crashed[0] and cars.time < config.EXPERT_VALIDATION_TIME - config.TIME_EPS:
            respawns += respawn(track, cars)
    return np.array(inputs), np.array(labels, dtype=np.int64), respawns


def dataset_seeds():
    """(seed, level) of the imitation dataset's training maps (spec §7.3): config.DATASET_TRACKS generated maps per
    level, with distinct seeds drawn once from config.DATASET_SEED_RANGE, so every collection uses the same maps, then
    the 38 real training layouts."""
    low, high = config.DATASET_SEED_RANGE
    total = sum(config.DATASET_TRACKS.values())
    seeds = low + np.random.default_rng(config.DATASET_SALT).choice(high - low, total, replace=False)
    items, position = [], 0
    for level, count in sorted(config.DATASET_TRACKS.items()):
        items += [(int(seed), level) for seed in seeds[position:position + count]]
        position += count
    from racing import circuits
    return items + [(k, config.MAX_LEVEL) for k in range(circuits.N_LAYOUTS)]


def default_splits():
    """The dataset's splits (spec §7.3): the training maps, the validation maps and the four test circuits."""
    return {"train": dataset_seeds(),
            "val": [(seed, config.VALIDATION_LEVEL) for seed in config.VAL_SEEDS],
            "test": [(k, config.TEST_LEVEL) for k in config.TEST_TRACKS]}


def _collect_track(seed, level):
    """Rows of one track, with that track's own random number generator (spec §7.3)."""
    return collect_rows(get_track(seed, level), np.random.default_rng([seed, level, config.DATASET_SALT]))


def collect_dataset(path=None, splits=None, n_jobs=1, backend="loky", resume=True):
    """Collect the train/val/test rows (spec §7.3) and save them as a compressed .npz file. Tracks are distributed
    across worker processes; each has its own random number generator, so the rows do not depend on n_jobs.
    Returns the arrays and a summary per split (tracks, rows, respawns after crashes) and records the collection
    time, the maps and the settings in <stem>_info.json next to the dataset. With resume, a complete dataset of the
    same maps recorded with the current settings is loaded instead of collected again (spec §3.3)."""
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
    checkpoint.write_json(info_path, {"wall_time": wall_time, "n_jobs": n_jobs, "splits": summary, "items": items,
                                      "settings": checkpoint.settings_record()})
    return data, summary
