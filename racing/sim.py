"""Vectorized driving simulator (spec §5). Every per-car quantity is an array with one entry per car, including each
car's setup: its sensor length and steering sensitivity (spec §9.11)."""
from dataclasses import dataclass

import numpy as np

from racing import config
from racing.actions import class_to_controls
from racing.track import wall_distance


@dataclass(eq=False)
class Cars:
    """State of a batch of cars on one track. Cars never interact with each other."""
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    index: np.ndarray               # matched centerline point, counted without wrapping
    progress: np.ndarray            # m, furthest distance reached along the centerline
    last_progress_time: np.ndarray
    laps_done: np.ndarray
    lap_start_time: np.ndarray
    last_lap: np.ndarray            # s, nan until a lap is completed
    best_lap: np.ndarray            # s, nan until a lap is completed
    crashed: np.ndarray
    finished: np.ndarray            # completed `laps` laps
    done: np.ndarray                # crashed, finished, stalled or out of time
    finish_time: np.ndarray         # s, nan unless finished
    ray_length: np.ndarray          # m, each car's sensor length (spec §5.2)
    steer_gain: np.ndarray          # each car's steering sensitivity (spec §5.1)
    laps: int = 1
    time: float = 0.0


@dataclass(eq=False)
class RunResult:
    fitness: np.ndarray
    finished: np.ndarray
    crashed: np.ndarray
    progress: np.ndarray            # fraction of the run completed, 0 to 1
    finish_time: np.ndarray
    steps: int
    laps_done: np.ndarray           # laps completed when the run ended (spec §9.2)


def setup_columns(setup, n):
    """(sensor lengths, steering sensitivities) of n cars from an (n, 2) setup array, or None for the default setup
    (150 m, k = 1; config.DEFAULT_SETUP) for every car."""
    if setup is None:
        length, gain = config.DEFAULT_SETUP
        return np.full(n, float(length)), np.full(n, float(gain))
    setup = np.asarray(setup, dtype=float)
    if setup.shape != (n, 2):
        raise ValueError(f"setup must have shape ({n}, 2), one (sensor length, steering sensitivity) row per car; "
                         f"got {setup.shape}")
    return setup[:, 0].copy(), setup[:, 1].copy()


def start_cars(track, n, laps=1, setup=None):
    """n cars at rest on the start line, facing the direction of travel, each with its setup (spec §9.11): an
    (n, 2) array of sensor lengths and steering sensitivities, or None for the default setup (150 m, k = 1)."""
    x0, y0 = track.start_position
    nan = np.full(n, np.nan)
    ray_length, steer_gain = setup_columns(setup, n)
    return Cars(
        x=np.full(n, x0), y=np.full(n, y0), heading=np.full(n, track.start_heading), speed=np.zeros(n),
        index=np.zeros(n, dtype=np.int64), progress=np.zeros(n), last_progress_time=np.zeros(n),
        laps_done=np.zeros(n, dtype=np.int64), lap_start_time=np.zeros(n), last_lap=nan.copy(),
        best_lap=nan.copy(), crashed=np.zeros(n, dtype=bool), finished=np.zeros(n, dtype=bool),
        done=np.zeros(n, dtype=bool), finish_time=nan.copy(), ray_length=ray_length, steer_gain=steer_gain,
        laps=laps,
    )


def ray_caps(lengths):
    """Sphere-tracing iterations allowed for rays of the given lengths: ⌈48 × L / 50⌉ iterations for a ray of length L
    (spec §5.2)."""
    return np.ceil(config.RAY_MAX_ITERATIONS * np.asarray(lengths, dtype=float)
                   / config.RAY_TRACE_LENGTH).astype(np.int64)


def cast_rays(track, x, y, heading, lengths=None):
    """Distance to the track boundary along each sensor ray, by sphere tracing. Each car's rays reach at most its
    sensor length (lengths, default SENSOR_START, 150 m) and are traced for at most ray_caps(length) iterations.
    Returns (n, N_RAYS) in m."""
    n = len(x)
    lengths = np.full(n, config.SENSOR_START) if lengths is None else np.asarray(lengths, dtype=float)
    reach = np.repeat(lengths, config.N_RAYS)
    caps = np.repeat(ray_caps(lengths), config.N_RAYS)
    angles = (np.asarray(heading)[:, None] + config.RAY_ANGLES[None, :]).ravel()
    dx, dy = np.cos(angles), np.sin(angles)
    ox, oy = np.repeat(x, config.N_RAYS), np.repeat(y, config.N_RAYS)
    t = np.zeros(n * config.N_RAYS)
    active = np.arange(n * config.N_RAYS)
    for iteration in range(int(caps.max()) if caps.size else 0):
        active = active[caps[active] > iteration]          # a ray stops at its own cap
        if active.size == 0:
            break
        d = wall_distance(track, ox[active] + dx[active] * t[active], oy[active] + dy[active] * t[active])
        hit = d <= 0.0
        t[active] = np.where(hit, t[active], t[active] + np.maximum(d, config.RAY_MIN_STEP))
        active = active[~hit & (t[active] < reach[active])]
    return np.minimum(t, reach).reshape(n, config.N_RAYS)


def road_position(track, cars, rows):
    """Position across the road of the cars `rows`: −1 at the left edge, 0 on the centerline, +1 at the right edge,
    relative to the driving direction, from each car's matched centerline point (spec §5.2)."""
    i = np.asarray(cars.index[rows]) % len(track.centerline)
    tangent = track.tangent[i]
    dx = cars.x[rows] - track.centerline[i, 0]
    dy = cars.y[rows] - track.centerline[i, 1]
    left = -dx * tangent[:, 1] + dy * tangent[:, 0]          # component along the left normal
    return np.clip(-left / (track.width / 2), -1.0, 1.0)


def sense(track, cars):
    """Network inputs (spec §5.2): ray distances / 150 m (RAY_SCALE; each car's rays reach its own sensor length, so a
    ray input can exceed 1), speed / 105 m/s (SPEED_SCALE), position across the road (−1 at the left edge,
    +1 at the right edge) and road width / WIDTH_SCALE. Rows of cars that are done stay zero."""
    obs = np.zeros((len(cars.x), config.N_INPUTS))
    live = np.flatnonzero(~cars.done)
    if live.size:
        rays = cast_rays(track, cars.x[live], cars.y[live], cars.heading[live], cars.ray_length[live])
        obs[live, :config.N_RAYS] = rays / config.RAY_SCALE
        obs[live, config.N_RAYS] = cars.speed[live] / config.SPEED_SCALE
        obs[live, config.N_RAYS + 1] = road_position(track, cars, live)
        obs[live, config.N_RAYS + 2] = track.width / config.WIDTH_SCALE
    return obs


def grip_at(speed):
    """The grip limit a(v) in m/s² at speed v (spec §5.1): min(GRIP_LOW + GRIP_RISE v², GRIP_MAX), rising with the
    square of speed like a Formula 1 car's downforce, up to its ceiling."""
    speed = np.asarray(speed, dtype=float)
    return np.minimum(config.GRIP_LOW + config.GRIP_RISE * speed * speed, config.GRIP_MAX)


def step(track, cars, steer, throttle, time_limit, grip_scale=1.0):
    """Advance every car that is not done by one time step. steer and throttle are fractions in [-1, 1] of the grip
    a(v) at the car's speed; each car's steering is multiplied by its steering sensitivity before the grip limit, and
    speed is capped at the top speed (spec §5.1). `grip_scale` multiplies a(v): 1 in training, 10 when the application
    shows a race with unlimited grip (spec §10.5)."""
    live = ~cars.done
    grip = grip_scale * grip_at(cars.speed)
    a_lat = np.clip(steer * cars.steer_gain, -1.0, 1.0) * grip
    a_long = np.clip(throttle, -1.0, 1.0) * grip
    total = np.hypot(a_lat, a_long)
    scale = np.where(total > grip, grip / np.maximum(total, 1e-12), 1.0)
    a_lat, a_long = a_lat * scale, a_long * scale                       # friction circle

    yaw_rate = a_lat / np.maximum(cars.speed, config.YAW_MIN_SPEED)
    speed = np.minimum(np.maximum(cars.speed + a_long * config.DT, 0.0), config.TOP_SPEED)
    heading = cars.heading + yaw_rate * config.DT
    cars.x = np.where(live, cars.x + speed * np.cos(heading) * config.DT, cars.x)
    cars.y = np.where(live, cars.y + speed * np.sin(heading) * config.DT, cars.y)
    cars.speed = np.where(live, speed, cars.speed)
    cars.heading = np.where(live, heading, cars.heading)
    cars.time += config.DT

    crashed = live & (wall_distance(track, cars.x, cars.y) <= 0.0)
    cars.crashed |= crashed
    cars.speed[crashed] = 0.0
    moving = live & ~crashed
    _update_progress(track, cars, moving)
    stalled = moving & (cars.time - cars.last_progress_time >= config.STALL_TIME - config.TIME_EPS)
    out_of_time = live & (cars.time >= time_limit - config.TIME_EPS)
    cars.done |= crashed | stalled | out_of_time | cars.finished


def _update_progress(track, cars, mask):
    """Match each car to the nearest centerline point near its last match; update progress, laps and finishing."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    m = len(track.centerline)
    back = int(round(config.PROGRESS_BACK / track.spacing))
    ahead = int(round(config.PROGRESS_AHEAD / track.spacing))
    candidates = cars.index[idx, None] + np.arange(-back, ahead + 1)[None, :]
    points = track.centerline[np.mod(candidates, m)]
    d2 = (points[..., 0] - cars.x[idx, None]) ** 2 + (points[..., 1] - cars.y[idx, None]) ** 2
    cars.index[idx] = candidates[np.arange(idx.size), np.argmin(d2, axis=1)]

    reached = cars.index[idx] * track.spacing
    cars.last_progress_time[idx[reached > cars.progress[idx]]] = cars.time
    cars.progress[idx] = np.maximum(cars.progress[idx], reached)

    laps_now = np.round(cars.progress[idx] / track.spacing).astype(np.int64) // m
    new_lap = laps_now > cars.laps_done[idx]
    j = idx[new_lap]
    lap_time = cars.time - cars.lap_start_time[j]
    cars.last_lap[j] = lap_time
    cars.best_lap[j] = np.fmin(cars.best_lap[j], lap_time)
    cars.lap_start_time[j] = cars.time
    cars.laps_done[j] = laps_now[new_lap]
    complete = j[cars.laps_done[j] >= cars.laps]
    cars.finished[complete] = True
    cars.finish_time[complete] = cars.time


def fitness(cars, track, time_limit):
    """Spec §5.5: completed fraction if unfinished (below 1); 1 + (1 - finish_time / time_limit) if finished."""
    fraction = np.minimum(cars.progress / (track.length * cars.laps), np.nextafter(1.0, 0.0))
    return np.where(cars.finished, 1.0 + (1.0 - cars.finish_time / time_limit), fraction)


def time_limit_for(track, laps=1):
    """Twice the expert's lap time per lap."""
    if np.isnan(track.expert_lap_time):
        raise ValueError("track has no expert lap time; pass time_limit explicitly")
    return config.TIME_LIMIT_FACTOR * track.expert_lap_time * laps


def target_time(track, factor, laps=config.LAPS):
    """Target time of a time trial (spec §9.2): factor × the expert's time for `laps` laps."""
    if np.isnan(track.expert_lap_time):
        raise ValueError("track has no expert lap time")
    return factor * track.expert_lap_time * laps


def simulate(track, policy, n, time_limit=None, laps=1, setup=None, grip_scale=1.0):
    """Drive n cars from the start line until every car is done. policy(obs, cars) returns action classes. setup:
    the cars' (n, 2) setups, None for the default setup (150 m, k = 1); grip_scale: the multiple of a(v) the cars
    drive with, 1 except in the application (spec §9.11, §10.5)."""
    if time_limit is None:
        time_limit = time_limit_for(track, laps)
    cars = start_cars(track, n, laps, setup)
    steps = 0
    while not cars.done.all():
        steer, throttle = class_to_controls(policy(sense(track, cars), cars))
        step(track, cars, steer, throttle, time_limit, grip_scale=grip_scale)
        steps += 1
    return RunResult(fitness=fitness(cars, track, time_limit), finished=cars.finished.copy(),
                     crashed=cars.crashed.copy(), progress=np.minimum(cars.progress / (track.length * laps), 1.0),
                     finish_time=cars.finish_time.copy(), steps=steps, laps_done=cars.laps_done.copy())
