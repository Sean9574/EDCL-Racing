"""The car's quasi-steady-state physics on a fixed path (spec §4.8): at every point the highest speed with
v²|κ| ≤ share × a(v), at most the top speed, then the speed changes that the grip left after cornering,
√((share a(v))² − (v²κ)²), allows forwards (acceleration) and backwards (braking). Smooth, continuous control is
assumed, so a lap is a bound the simulator's controllers approach. Used for the ideal laps of Table 0, the check of
the racing line and, from version 3.10, the expert's target speeds (spec §7.1)."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from racing import config
from racing.sim import grip_at


def corner_speed(kappa, share=1.0):
    """The highest speed v with v² κ ≤ share × a(v) for curvatures κ in 1/m: below the grip ceiling v² (κ − share c) ≤
    share a0, at the ceiling v² κ ≤ share a_max; the lower of the two bounds. Not capped by the top speed."""
    kappa = np.maximum(np.asarray(kappa, dtype=float), 1e-9)
    excess = kappa - share * config.GRIP_RISE
    rising = np.where(excess > 0.0, np.sqrt(share * config.GRIP_LOW / np.maximum(excess, 1e-12)), np.inf)
    return np.minimum(rising, np.sqrt(share * config.GRIP_MAX / kappa))


def stop_speed(distance, share):
    """The highest speed from which the car stops within `distance` m braking at share × a(v) (spec §7.1): below
    GRIP_FULL_SPEED the braking distance from v is ln(a(v) / a0) / (2 share c), above it the ceiling adds
    (v² − v_full²) / (2 share a_max)."""
    low, rise, full = config.GRIP_LOW, config.GRIP_RISE, config.GRIP_FULL_SPEED
    rising_distance = np.log((low + rise * full * full) / low) / (2.0 * share * rise)
    if distance <= rising_distance:
        return float(np.sqrt(low * np.expm1(2.0 * share * rise * distance) / rise))
    return float(np.sqrt(full * full + 2.0 * share * config.GRIP_MAX * (distance - rising_distance)))


def sight_speeds(points, curvature, spacing, share, sight):
    """The expert's target speed at every point of a closed line (spec §7.1): the start of a speed profile of the next
    round(sight / spacing) points, at every point the corner limit v²|κ| ≤ share × a(v) (and the top speed), then
    backwards from the window's end the speed from which the car slows to the next point's speed braking with the grip
    left after cornering over the step between them; capped by stop_speed(sight, share) and the top speed."""
    m = len(points)
    window = int(round(sight / spacing))
    step = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)        # from point j to j + 1
    bend = np.abs(np.asarray(curvature, dtype=float))
    limit = np.minimum(corner_speed(bend, share), config.TOP_SPEED)
    index = np.arange(m)
    v = limit[(index + window) % m]
    for k in range(window - 1, -1, -1):
        here, ahead = (index + k) % m, (index + k + 1) % m
        spare = np.sqrt(np.maximum((share * grip_at(v)) ** 2 - (v * v * bend[ahead]) ** 2, 0.0))
        v = np.minimum(limit[here], np.sqrt(v * v + 2.0 * step[here] * spare))
    return np.ascontiguousarray(np.minimum(np.minimum(v, stop_speed(sight, share)), config.TOP_SPEED))


def path_curvature(points, smooth):
    """(points every 1 m, curvature in 1/m) of the closed path through `points` (first point not repeated): a periodic
    cubic spline over the arc length, sampled every metre, its curvature from the spline's derivatives, smoothed by a
    Gaussian of `smooth` m standard deviation (none for 0)."""
    points = np.asarray(points, dtype=float)
    closed = np.vstack([points, points[:1]])
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    spline = CubicSpline(arc, closed, bc_type="periodic")
    t = np.arange(0.0, arc[-1], 1.0)
    d1, d2 = spline(t, 1), spline(t, 2)
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.hypot(d1[:, 0], d1[:, 1]) ** 3
    return spline(t), (gaussian_filter1d(kappa, smooth, mode="wrap") if smooth else kappa)


def lap_time(points, smooth=config.RACING_LINE_SMOOTHING, share=1.0):
    """The car's fastest flying lap in s on the closed path through `points` (spec §4.8): the corner limit at every
    metre, then three rounds of passes forwards and backwards around the loop, so the start joins up. Every step is
    weighted by its length: 1 m, and L − (n − 1) m for the last, which closes the loop."""
    points = np.asarray(points, dtype=float)
    closed = np.vstack([points, points[:1]])
    length = float(np.cumsum(np.hypot(*np.diff(closed, axis=0).T))[-1])   # as path_curvature measures it
    _, kappa = path_curvature(points, smooth)
    n = len(kappa)
    step = np.append(np.ones(n - 1), length - (n - 1))             # step[k]: from sample k to sample k + 1
    v = speed_profile(np.abs(kappa), step, share)
    return float(np.sum(step / ((v + np.roll(v, -1)) / 2.0)))


def speed_profile(bend, step, share=1.0):
    """The car's speed at every point of a closed path, as lap_time drives it: the corner limit for the curvatures
    `bend` (1/m, unsigned) and the top speed, then three rounds of passes forwards (acceleration) and backwards
    (braking) around the loop with the grip left after cornering, so the start joins up; step[k] is the distance in m
    from point k to point k + 1."""
    bend = np.asarray(bend, dtype=float)
    v = np.minimum(corner_speed(bend, share), config.TOP_SPEED)
    n = len(v)

    def spare(k):
        return np.sqrt(max((share * float(grip_at(v[k]))) ** 2 - (v[k] ** 2 * bend[k]) ** 2, 0.0))

    for _ in range(3):
        for i in range(2 * n):                                         # acceleration, forwards
            a, b = i % n, (i + 1) % n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2.0 * spare(a) * step[a]))
        for i in range(2 * n, 0, -1):                                  # braking, backwards
            a, b = i % n, (i - 1) % n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2.0 * spare(a) * step[b]))
    return v
