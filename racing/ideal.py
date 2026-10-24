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
    bend = np.abs(kappa)
    v = np.minimum(corner_speed(bend, share), config.TOP_SPEED)
    n = len(v)
    step = np.append(np.ones(n - 1), length - (n - 1))             # step[k]: from sample k to sample k + 1

    def spare(k):
        return np.sqrt(max((share * float(grip_at(v[k]))) ** 2 - (v[k] ** 2 * bend[k]) ** 2, 0.0))

    for _ in range(3):
        for i in range(2 * n):                                         # acceleration, forwards
            a, b = i % n, (i + 1) % n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2.0 * spare(a) * step[a]))
        for i in range(2 * n, 0, -1):                                  # braking, backwards
            a, b = i % n, (i - 1) % n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2.0 * spare(a) * step[b]))
    return float(np.sum(step / ((v + np.roll(v, -1)) / 2.0)))
