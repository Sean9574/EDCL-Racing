"""Procedural track generation for curriculum levels 1-4 (revision 3 §3): gentle loops on Easy, and circuits of
straights and rounded corners with hairpins, chicanes and esses on Medium, Hard and Extra hard. Every map runs
clockwise or counter-clockwise, drawn from its seed."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from racing import config
from racing.track import build_track


def loop_points(rng, n_points, radius_range, step=0.5):
    """Control points at jittered angles and random radii, joined by a periodic cubic spline."""
    angles = (np.arange(n_points) + rng.uniform(-0.3, 0.3, n_points)) * (2 * np.pi / n_points)
    radii = rng.uniform(radius_range[0], radius_range[1], n_points)
    control = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    closed = np.vstack([control, control[:1]])
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    spline = CubicSpline(t, closed, bc_type="periodic")
    return spline(np.arange(0.0, t[-1], step))


def edge_lengths(poly):
    """Length of each polygon edge poly[i] -> poly[i + 1]."""
    return np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)


def fillet_path(poly, radii, step=0.5):
    """The closed polygon with each corner rounded by a circular arc of the given radius (reduced where the adjacent
    edges are too short to hold it), joined by straights. Returns points about `step` m apart, first not repeated."""
    n = len(poly)
    arcs = []
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        u, v = (a - b) / np.linalg.norm(a - b), (c - b) / np.linalg.norm(c - b)
        half = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)) / 2
        if half > np.radians(89.5):                           # (almost) straight through: no corner
            arcs.append(b[None, :])
            continue
        d = min(radii[i] / np.tan(half), 0.48 * np.linalg.norm(a - b), 0.48 * np.linalg.norm(c - b))
        r = d * np.tan(half)
        p1, p2 = b + u * d, b + v * d
        center = b + (u + v) / np.linalg.norm(u + v) * (r / np.sin(half))
        t1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
        t2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
        sweep = np.angle(np.exp(1j * (t2 - t1)))
        ts = t1 + sweep * np.linspace(0.0, 1.0, max(int(abs(sweep) * r / step), 2))
        arcs.append(np.column_stack([center[0] + r * np.cos(ts), center[1] + r * np.sin(ts)]))
    path = []
    for k, arc in enumerate(arcs):
        path.append(arc)
        following = arcs[(k + 1) % n][0]
        m = max(int(np.linalg.norm(following - arc[-1]) / step), 1)
        path.append(arc[-1] + (following - arc[-1]) * np.linspace(0.0, 1.0, m, endpoint=False)[1:, None])
    return np.vstack(path)


def insert_on_edge(poly, kinds, edge, along, side, kind):
    """Insert vertices on edge `edge` at offsets `along` (m, from its midpoint along the edge) and `side` (m, to its
    left), all marked `kind`."""
    a, b = poly[edge], poly[(edge + 1) % len(poly)]
    t = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-t[1], t[0]])
    middle = (a + b) / 2
    new = [middle + t * s + normal * o for s, o in zip(along, side)]
    return (np.vstack([poly[:edge + 1], new, poly[edge + 1:]]),
            kinds[:edge + 1] + [kind] * len(new) + kinds[edge + 1:])


def circuit_points(rng, spec, width, step=0.5):
    """A circuit (revision 3 §3.2): a stretched star polygon with a main straight, hairpins, chicanes and esses, its
    corners rounded with slow, fast or tight arcs."""
    n = int(rng.integers(spec["points"][0], spec["points"][1] + 1))
    angles = (np.arange(n) + rng.uniform(-0.3, 0.3, n)) * (2 * np.pi / n)
    radii = rng.uniform(spec["radius"][0], spec["radius"][1], n)
    stretch = rng.uniform(*config.CIRCUIT_STRETCH)
    poly = np.column_stack([stretch * radii * np.cos(angles), radii * np.sin(angles)])
    while spec["main"] and edge_lengths(poly).max() < spec["main"] and len(poly) > 5:
        longest = int(np.argmax(edge_lengths(poly)))
        poly = np.delete(poly, (longest + 2) % len(poly), axis=0)    # joins the two edges after the longest
    kinds = ["corner"] * len(poly)
    main = int(np.argmax(edge_lengths(poly)))
    pins = [(main + 1) % len(poly)]
    others = [j for j in range(len(poly)) if j not in pins and j != main]
    pins += [int(j) for j in rng.choice(others, max(spec["hairpins"] - 1, 0), replace=False)]
    for j in pins[:spec["hairpins"]]:
        poly[j] *= rng.uniform(1.25, 1.5)                               # outwards: a sharper corner
        kinds[j] = "hairpin"
    order = [int(e) for e in np.argsort(-edge_lengths(poly), kind="stable") if e != main]
    plan = (["chicane"] * int(rng.integers(spec["chicanes"][0], spec["chicanes"][1] + 1))
            + ["esses"] * int(rng.integers(spec["esses"][0], spec["esses"][1] + 1)))
    rng.shuffle(plan)
    for feature, edge in sorted(zip(plan, order[:len(plan)]), key=lambda pair: -pair[1]):   # last edge first
        length = edge_lengths(poly)[edge]
        side = rng.choice([-1.0, 1.0])
        if feature == "chicane" and length > 110.0:
            s, o = rng.uniform(12.0, 18.0), side * rng.uniform(0.9, 1.3) * width
            poly, kinds = insert_on_edge(poly, kinds, edge, [-1.5 * s, -0.5 * s, 0.5 * s, 1.5 * s], [0.0, o, o, 0.0],
                                         "chicane")
        elif feature == "esses" and length > 160.0:
            k = int(rng.integers(3, 5))
            along = (np.arange(k) - (k - 1) / 2) * rng.uniform(30.0, 45.0)
            o = rng.uniform(1.2, 2.0) * width
            poly, kinds = insert_on_edge(poly, kinds, edge, list(along), [side * o * (-1) ** j for j in range(k)],
                                         "esses")
    corner_radii = []
    for kind in kinds:
        if kind in ("hairpin", "chicane"):
            corner_radii.append(rng.uniform(spec["min_corner"], spec["min_corner"] + 6.0))
        elif kind == "esses":
            corner_radii.append(rng.uniform(spec["slow"][0] + 6.0, spec["slow"][1] + 6.0))
        else:
            corner_radii.append(rng.uniform(*spec["slow"]) if rng.random() < 0.5 else rng.uniform(*spec["fast"]))
    return fillet_path(poly, corner_radii, step)


def runs_clockwise(seed, level):
    """Direction of the map for (seed, level), the same for every generation attempt (revision 3 §3.3)."""
    return bool(np.random.default_rng([seed, level, config.DIRECTION_SALT]).random() < 0.5)


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


def map_width(seed, level):
    """Road width of map (seed, level) in m, drawn from WIDTH_RANGE by the seed and level alone, so a map keeps it on
    every generation attempt (revision 3.2 §2.1)."""
    low, high = config.WIDTH_RANGE
    return float(np.random.default_rng([seed, level, config.WIDTH_SALT]).uniform(low, high))


def candidate_track(seed, level, attempt=0):
    """One generation attempt for (seed, level). Returns None if the geometry is rejected."""
    spec = config.LEVELS[level]
    width = map_width(seed, level)
    rng = np.random.default_rng([seed, level, attempt])
    if spec["kind"] == "loop":
        n_points = int(rng.integers(spec["points"][0], spec["points"][1] + 1))
        points = loop_points(rng, n_points, spec["radius"])
    else:
        points = circuit_points(rng, spec, width)
    if runs_clockwise(seed, level):
        points = points[::-1]
    track = build_track(points, width, seed, level)
    if not geometry_ok(track, spec["min_corner"]):
        return None
    return track


def make_track(seed, level):
    """The first geometrically valid track for (seed, level)."""
    for attempt in range(config.MAX_TRACK_ATTEMPTS):
        track = candidate_track(seed, level, attempt)
        if track is not None:
            return track
    raise RuntimeError(f"no valid track geometry for seed {seed}, level {level}")
