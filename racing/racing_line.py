"""Racing lines (spec §4.8, version 3.10): the minimum-curvature line within a track's road. Nodes every
RACING_LINE_STEP m move along the road's normal within RACING_LINE_MARGIN of each edge; their offsets minimize the
squared curvature at the nodes, linearized and solved as bounded least squares; a chord-length cubic spline through the
nodes is sampled at every centerline point; node bounds are tightened where the spline leaves the road's bounds."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits

from racing import config

PUBLISHED_DIR = config.PROJECT_DIR / "data" / "racetracks" / "racelines"   # the database's own lines, for the check


def _node_offsets(center, normal, lower, upper, iterations, reference=None, weights=None):
    """Offsets α of the nodes along their normals that minimize Σ_k (w_k (D p)_k · r_k)², p = center + α normal, D the
    cyclic second difference and r_k the previous solution's normals (the nodes' own at first). Each solve
    re-linearizes on the new line's normals and reweights each row by (node spacing / the line's local spacing)², so
    the rows measure curvature. Returns (α, normals, weights) to continue from."""
    n = len(center)
    index = np.arange(n)
    reference = normal.copy() if reference is None else reference
    weights = np.ones(n) if weights is None else weights
    spacing = float(np.mean(np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)))
    upper = np.maximum(upper, lower + 1e-9)                      # lsq_linear needs lower < upper
    alpha = np.zeros(n)
    for _ in range(iterations):
        matrix = np.zeros((n, n))
        for shift, coefficient in ((-1, 1.0), (0, -2.0), (1, 1.0)):
            column = (index + shift) % n
            matrix[index, column] += coefficient * np.einsum("ij,ij->i", normal[column], reference)
        bend = np.roll(center, 1, axis=0) - 2.0 * center + np.roll(center, -1, axis=0)
        target = -np.einsum("ij,ij->i", bend, reference)
        with threadpool_limits(limits=1):                        # the same line in every process (spec §4.8)
            alpha = lsq_linear(weights[:, None] * matrix, weights * target, bounds=(lower, upper), method="trf",
                               tol=1e-10).x
        line = center + alpha[:, None] * normal
        chord = np.roll(line, -1, axis=0) - np.roll(line, 1, axis=0)
        direction = chord / np.linalg.norm(chord, axis=1, keepdims=True)
        reference = np.column_stack([-direction[:, 1], direction[:, 0]])
        step = np.linalg.norm(np.roll(line, -1, axis=0) - line, axis=1)
        weights = (spacing / ((step + np.roll(step, 1)) / 2.0)) ** 2
    return alpha, reference, weights


def solve(center, tangent, half_left, half_right, spacing, margin=config.RACING_LINE_MARGIN):
    """The racing line of a road (spec §4.8): (points (M, 2), unit directions (M, 2), curvature (M,) in 1/m, positive
    for left turns, smoothed over RACING_LINE_SMOOTHING m), one row per centerline point, the line's point abeam of it.
    `margin` is the clearance kept from each edge (the published lines' 0.65 m for the check)."""
    m = len(center)
    k = max(1, min(int(round(config.RACING_LINE_STEP / spacing)), m // 8))
    nodes = np.arange(0, m, k)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    upper_all = np.maximum(np.asarray(half_left, dtype=float) - margin, 0.0)
    lower_all = -np.maximum(np.asarray(half_right, dtype=float) - margin, 0.0)
    lower, upper = lower_all[nodes].copy(), upper_all[nodes].copy()
    owner = np.minimum(np.arange(m) // k, len(nodes) - 1)       # the node that starts each point's interval
    node_index = np.append(nodes, m)
    reference = weights = None
    for round_ in range(config.RACING_LINE_ROUNDS):
        iterations = config.RACING_LINE_ITERATIONS if round_ == 0 else 1
        alpha, reference, weights = _node_offsets(center[nodes], normal[nodes], lower, upper, iterations, reference,
                                                  weights)
        knots = center[nodes] + alpha[:, None] * normal[nodes]
        closed = np.vstack([knots, knots[:1]])
        chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
        spline = CubicSpline(chord, closed, bc_type="periodic")
        at = np.interp(np.arange(m), node_index, chord)
        points = spline(at)
        offset = np.einsum("ij,ij->i", points - center, normal)
        over, under = np.maximum(offset - upper_all, 0.0), np.maximum(lower_all - offset, 0.0)
        if max(over.max(), under.max()) <= config.RACING_LINE_TOLERANCE:
            break
        for excess, side in ((over, "upper"), (under, "lower")):
            worst = np.zeros(len(nodes))
            np.maximum.at(worst, owner, excess)
            shrink = np.maximum(worst, np.roll(worst, 1))          # both nodes around each interval
            if side == "upper":
                upper = np.maximum(upper - shrink, lower)
            else:
                lower = np.minimum(lower + shrink, upper)
    d1, d2 = spline(at, 1), spline(at, 2)
    speed = np.hypot(d1[:, 0], d1[:, 1])
    curvature = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed ** 3
    curvature = gaussian_filter1d(curvature, config.RACING_LINE_SMOOTHING / spacing, mode="wrap")
    return points, d1 / speed[:, None], curvature


def fields(center, tangent, half_left, half_right, spacing):
    """The Track fields of the racing line (spec §4.1, §4.8), contiguous float64 arrays, read in place by the compiled
    simulator (spec §5.7)."""
    points, direction, curvature = solve(center, tangent, half_left, half_right, spacing)
    return {"line": np.ascontiguousarray(points, dtype=np.float64),
            "line_tangent": np.ascontiguousarray(direction, dtype=np.float64),
            "line_heading": np.ascontiguousarray(np.arctan2(direction[:, 1], direction[:, 0]), dtype=np.float64),
            "line_curvature": np.ascontiguousarray(curvature, dtype=np.float64)}
