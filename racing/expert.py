"""Scripted expert driver (spec §7.1) and imitation dataset collection (spec §7.2-7.3)."""
import os
from pathlib import Path

import numpy as np

from racing import config, sim
from racing.actions import class_to_controls, controls_to_class
from racing.tracks import get_track


def expert_controls(track, cars):
    """Continuous steering and throttle fractions chosen by the expert for every car."""
    m = len(track.centerline)
    here = np.mod(cars.index, m)
    look = config.EXPERT_LOOKAHEAD_BASE + config.EXPERT_LOOKAHEAD_TIME * cars.speed
    target = track.centerline[np.mod(here + np.round(look / track.spacing).astype(np.int64), m)]
    dx, dy = target[:, 0] - cars.x, target[:, 1] - cars.y
    alpha = np.angle(np.exp(1j * (np.arctan2(dy, dx) - cars.heading)))
    path_curvature = 2.0 * np.sin(alpha) / np.maximum(np.hypot(dx, dy), 1e-6)
    steer = np.clip(cars.speed ** 2 * path_curvature / config.GRIP, -1.0, 1.0)
    target = target_speed(track, here)
    band = config.EXPERT_SPEED_BAND
    throttle = np.where(cars.speed < target - band, 1.0, np.where(cars.speed > target + band, -1.0, 0.0))
    return steer, throttle


def target_speed(track, here):
    """The expert's target speed at centerline points `here`: the corner speed for the tightest curve in
    sight, capped by the speed from which it can stop within sight."""
    m = len(track.centerline)
    sight = int(round(config.EXPERT_SIGHT / track.spacing))
    window = np.mod(here[:, None] + np.arange(sight + 1)[None, :], m)
    kappa = np.max(np.abs(track.curvature[window]), axis=1)
    corner_speed = np.sqrt(config.EXPERT_GRIP_MARGIN * config.GRIP / np.maximum(kappa, 1e-9))
    sight_speed = np.sqrt(2.0 * config.EXPERT_GRIP_MARGIN * config.GRIP * config.EXPERT_SIGHT)
    return np.minimum(corner_speed, sight_speed)


def expert_policy(track):
    """Policy for sim.simulate: the expert, quantized to the 15 action classes."""
    def policy(obs, cars):
        return controls_to_class(*expert_controls(track, cars))
    return policy


def expert_lap_time(track):
    """The expert's lap time on this track, or None if it crashes, stalls or runs out of time."""
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


def default_splits():
    train = [(seed, level) for level, seeds in config.TRAIN_DATA_TRACKS.items() for seed in seeds]
    return {"train": train,
            "val": [(seed, config.MAX_LEVEL) for seed in config.VAL_SEEDS],
            "test": [(seed, config.MAX_LEVEL) for seed in config.TEST_SEEDS]}


def collect_dataset(path=None, splits=None):
    """Collect the train/val/test rows (spec §7.3) and save them as a compressed .npz file.
    Returns the arrays and a summary per split: tracks, rows and respawns after crashes."""
    path = Path(config.DATASET_PATH if path is None else path)
    splits = default_splits() if splits is None else splits
    data, summary = {}, {}
    for name, items in splits.items():
        xs, ys, respawns = [], [], 0
        for seed, level in items:
            x, y, r = collect_rows(get_track(seed, level), np.random.default_rng([seed, level, config.DATASET_SALT]))
            xs.append(x)
            ys.append(y)
            respawns += r
        data[f"X_{name}"] = np.concatenate(xs).astype(np.float32)
        data[f"y_{name}"] = np.concatenate(ys).astype(np.int64)
        summary[name] = {"tracks": len(items), "rows": len(data[f"y_{name}"]), "respawns": respawns}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")   # same directory, so the replace is atomic
    try:
        with open(temporary, "wb") as f:
            np.savez_compressed(f, **data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return data, summary
