"""Scripted expert driver (spec §7.1), whose steering and throttle are fractions of the grip a(v), and imitation
dataset collection on generated maps and the real training layouts (spec §7.2-7.3). From version 3.10 the expert
follows the racing line, its target speed the track's speed profile of its sight along the line at its grip share."""
import os
import time
from pathlib import Path

import numpy as np

from racing import checkpoint, config, fastsim, ideal, sim
from racing.actions import class_to_controls, controls_to_class
from racing.tracks import get_track


def expert_controls(track, cars):
    """Continuous steering and throttle chosen by the expert for every car, as fractions of the grip a(v) at the car's
    speed (spec §7.1, version 3.10). The steering follows the racing line: the path curvature it asks for is the
    line's curvature EXPERT_PREVIEW_TIME × v ahead of the matched point, less EXPERT_CROSS_TRACK_GAIN × the car's
    distance left of the line / L² and EXPERT_HEADING_GAIN × its heading relative to the line / L, where L = 6 m +
    0.2 s × v (EXPERT_LOOKAHEAD_BASE, EXPERT_LOOKAHEAD_TIME); the steering is that curvature × v² / a(v), clipped to
    [−1, 1]. The target speed is the track's profile at the matched point. The throttle is the speed error (target
    minus speed) divided by the band, the larger of EXPERT_SPEED_BAND and one step's full speed change a(v) DT,
    clipped to [−1, 1]: once quantized, full gas or full brake beyond the band, half gas or half brake within it and
    coast near the target, so a step of gas or brake never carries it across the whole band, and in a steady corner it
    settles at a speed and keeps all of a(v) for turning. While its steering is quantized to full lock the throttle is
    never above 0, so a corner's exit taken at full lock keeps all of a(v) for turning."""
    m = len(track.centerline)
    here = np.mod(cars.index, m)
    tangent = track.line_tangent[here]
    dx, dy = cars.x - track.line[here, 0], cars.y - track.line[here, 1]
    offset = -dx * tangent[:, 1] + dy * tangent[:, 0]                       # m left of the line
    heading_error = np.angle(np.exp(1j * (cars.heading - track.line_heading[here])))   # rad, left of the line
    preview = np.mod(here + np.round(cars.speed * config.EXPERT_PREVIEW_TIME / track.spacing).astype(np.int64), m)
    look = config.EXPERT_LOOKAHEAD_BASE + config.EXPERT_LOOKAHEAD_TIME * cars.speed
    path_curvature = (track.line_curvature[preview] - config.EXPERT_CROSS_TRACK_GAIN * offset / (look * look)
                      - config.EXPERT_HEADING_GAIN * heading_error / look)
    grip = sim.grip_at(cars.speed)
    steer = np.clip(cars.speed ** 2 * path_curvature / grip, -1.0, 1.0)
    target = target_speed(track, here)
    band = np.maximum(config.EXPERT_SPEED_BAND, grip * config.DT)
    throttle = np.clip((target - cars.speed) / band, -1.0, 1.0)
    level = class_to_controls(controls_to_class(steer, throttle))[0]           # the steering level it will drive
    throttle = np.where(np.abs(level) >= 1.0, np.minimum(throttle, 0.0), throttle)   # no gas at full lock
    return steer, throttle


def corner_speed(kappa):
    """The expert's corner speed: ideal.corner_speed at its grip share (spec §7.1)."""
    return ideal.corner_speed(kappa, config.EXPERT_GRIP_MARGIN)


def target_speed(track, here):
    """The expert's target speed at centerline points `here` (spec §7.1): the speed profile of its sight along the
    racing line, computed when the track was built."""
    return track.expert_speed[here]


def expert_policy(track):
    """Policy for sim.simulate: the expert, quantized to the 45 action classes."""
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
    """Put every crashed car back on the racing line at the line point of its matched centerline point, heading along
    the line at the expert's target speed there, and let it continue (spec §7.2; on the centerline up to version 3.9).
    Returns the number of cars moved."""
    j = np.flatnonzero(cars.crashed)
    here = np.mod(cars.index[j], len(track.centerline))
    cars.x[j] = track.line[here, 0]
    cars.y[j] = track.line[here, 1]
    cars.heading[j] = track.line_heading[here]
    cars.speed[j] = target_speed(track, here)
    cars.crashed[j] = False
    cars.done[j] = False
    return len(j)


def collect_rows(track, rng):
    """One lap of expert driving with random steering disturbances. Labels are always the expert's own
    choice. A crash puts the car back on the racing line (spec §7.2). Returns inputs, labels and the
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
