"""Vectorized driving simulator (spec §5). Every per-car quantity is an array with one entry per car."""
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


def start_cars(track, n, laps=1):
    """n cars at rest on the start line, facing the direction of travel."""
    x0, y0 = track.start_position
    nan = np.full(n, np.nan)
    return Cars(
        x=np.full(n, x0), y=np.full(n, y0), heading=np.full(n, track.start_heading), speed=np.zeros(n),
        index=np.zeros(n, dtype=np.int64), progress=np.zeros(n), last_progress_time=np.zeros(n),
        laps_done=np.zeros(n, dtype=np.int64), lap_start_time=np.zeros(n), last_lap=nan.copy(),
        best_lap=nan.copy(), crashed=np.zeros(n, dtype=bool), finished=np.zeros(n, dtype=bool),
        done=np.zeros(n, dtype=bool), finish_time=nan.copy(), laps=laps,
    )


def cast_rays(track, x, y, heading):
    """Distance to the track boundary along each sensor ray, by sphere tracing. Returns (n, N_RAYS) in m."""
    n = len(x)
    angles = (np.asarray(heading)[:, None] + config.RAY_ANGLES[None, :]).ravel()
    dx, dy = np.cos(angles), np.sin(angles)
    ox, oy = np.repeat(x, config.N_RAYS), np.repeat(y, config.N_RAYS)
    t = np.zeros(n * config.N_RAYS)
    active = np.arange(n * config.N_RAYS)
    for _ in range(config.RAY_MAX_ITERATIONS):
        if active.size == 0:
            break
        d = wall_distance(track, ox[active] + dx[active] * t[active], oy[active] + dy[active] * t[active])
        hit = d <= 0.0
        t[active] = np.where(hit, t[active], t[active] + np.maximum(d, config.RAY_MIN_STEP))
        active = active[~hit & (t[active] < config.RAY_RANGE)]
    return np.minimum(t, config.RAY_RANGE).reshape(n, config.N_RAYS)


def sense(track, cars):
    """Network inputs: ray distances / RAY_RANGE and speed / SPEED_SCALE. Rows of cars that are done stay zero."""
    obs = np.zeros((len(cars.x), config.N_INPUTS))
    live = np.flatnonzero(~cars.done)
    if live.size:
        obs[live, :config.N_RAYS] = cast_rays(track, cars.x[live], cars.y[live], cars.heading[live]) / config.RAY_RANGE
        obs[live, config.N_RAYS] = cars.speed[live] / config.SPEED_SCALE
    return obs


def step(track, cars, steer, throttle, time_limit):
    """Advance every car that is not done by one time step. steer and throttle are grip fractions in [-1, 1]."""
    live = ~cars.done
    a_lat = np.clip(steer, -1.0, 1.0) * config.GRIP
    a_long = np.clip(throttle, -1.0, 1.0) * config.GRIP
    total = np.hypot(a_lat, a_long)
    scale = np.where(total > config.GRIP, config.GRIP / np.maximum(total, 1e-12), 1.0)
    a_lat, a_long = a_lat * scale, a_long * scale                       # friction circle

    yaw_rate = a_lat / np.maximum(cars.speed, config.YAW_MIN_SPEED)
    speed = np.maximum(cars.speed + a_long * config.DT, 0.0)
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


def simulate(track, policy, n, time_limit=None, laps=1):
    """Drive n cars from the start line until every car is done. policy(obs, cars) returns action classes."""
    if time_limit is None:
        time_limit = time_limit_for(track, laps)
    cars = start_cars(track, n, laps)
    steps = 0
    while not cars.done.all():
        steer, throttle = class_to_controls(policy(sense(track, cars), cars))
        step(track, cars, steer, throttle, time_limit)
        steps += 1
    return RunResult(fitness=fitness(cars, track, time_limit), finished=cars.finished.copy(),
                     crashed=cars.crashed.copy(), progress=np.minimum(cars.progress / (track.length * laps), 1.0),
                     finish_time=cars.finish_time.copy(), steps=steps)
