"""Procedural track generation for curriculum levels 1-4 (spec §4.2-4.3)."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from racing import config
from racing.track import build_track


def circle_points(radius, step=0.5):
    """Counter-clockwise circle starting at the bottom."""
    n = int(np.ceil(2 * np.pi * radius / step))
    a = np.linspace(-np.pi / 2, 1.5 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])


def oval_points(straight, radius, step=0.5):
    """Two straights joined by semicircles, counter-clockwise, starting mid-way along the bottom straight."""
    half = straight / 2
    points = [(x, -radius) for x in np.arange(-half, half, step)]
    points += [(half + radius * np.cos(a), radius * np.sin(a)) for a in np.arange(-np.pi / 2, np.pi / 2, step / radius)]
    points += [(x, radius) for x in np.arange(half, -half, -step)]
    points += [(-half + radius * np.cos(a), radius * np.sin(a)) for a in np.arange(np.pi / 2, 1.5 * np.pi, step / radius)]
    points = np.array(points)
    start = np.argmin(np.hypot(points[:, 0], points[:, 1] + radius))
    return np.roll(points, -start, axis=0)


def loop_points(rng, n_points, radius_range, step=0.5):
    """Control points at jittered angles and random radii, joined by a periodic cubic spline."""
    angles = (np.arange(n_points) + rng.uniform(-0.3, 0.3, n_points)) * (2 * np.pi / n_points)
    radii = rng.uniform(radius_range[0], radius_range[1], n_points)
    control = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    closed = np.vstack([control, control[:1]])
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    spline = CubicSpline(t, closed, bc_type="periodic")
    return spline(np.arange(0.0, t[-1], step))


def geometry_ok(track, min_corner):
    """True if no corner is tighter than min_corner and separate parts of the road stay apart."""
    if np.max(np.abs(track.curvature)) > 1.0 / min_corner:
        return False
    reach = track.width + config.OVERLAP_MARGIN
    pairs = cKDTree(track.centerline).query_pairs(reach, output_type="ndarray")
    if len(pairs) == 0:
        return True
    gap = np.abs(pairs[:, 0] - pairs[:, 1]) * track.spacing
    along = np.minimum(gap, track.length - gap)
    return not np.any(along > 3 * reach)


def candidate_track(seed, level, attempt=0):
    """One generation attempt for (seed, level). Returns None if the geometry is rejected."""
    spec = config.LEVELS[level]
    rng = np.random.default_rng([seed, level, attempt])
    if spec["kind"] == "circle":
        points = circle_points(spec["radius"])
    elif spec["kind"] == "oval":
        points = oval_points(rng.uniform(*spec["straight"]), rng.uniform(*spec["radius"]))
    else:
        n_points = int(rng.integers(spec["points"][0], spec["points"][1] + 1))
        points = loop_points(rng, n_points, spec["radius"])
    track = build_track(points, spec["width"], seed, level)
    if spec["kind"] == "loop" and not geometry_ok(track, spec["min_corner"]):
        return None
    return track


def make_track(seed, level):
    """The first geometrically valid track for (seed, level)."""
    for attempt in range(config.MAX_TRACK_ATTEMPTS):
        track = candidate_track(seed, level, attempt)
        if track is not None:
            return track
    raise RuntimeError(f"no valid track geometry for seed {seed}, level {level}")
