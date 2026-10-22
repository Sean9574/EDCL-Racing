"""Track data structure and geometry shared by the generator, the simulator and the viewer (spec §4.1)."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from racing import config


@dataclass(eq=False)
class Track:
    """A closed-loop track. Cars drive in the direction of increasing arc length (spec §4.1)."""
    seed: int
    level: int
    centerline: np.ndarray      # (M, 2), point 0 is the start line
    tangent: np.ndarray         # (M, 2) unit direction of travel
    curvature: np.ndarray       # (M,) signed curvature in 1/m, positive for left turns
    half_left: np.ndarray       # (M,) m from each centerline point to the left edge
    half_right: np.ndarray      # (M,) m from each centerline point to the right edge
    left: np.ndarray            # (M, 2) left road edge
    right: np.ndarray           # (M, 2) right road edge
    spacing: float              # m between centerline points
    length: float               # m, equal to M * spacing
    origin: np.ndarray          # (2,) world position of the road grid's lower-left corner
    dist: np.ndarray            # (rows, cols) float32 m from each road cell to the nearest grass cell; 0 on grass
    expert_lap_time: float = float("nan")
    name: str = ""              # a real circuit's database name (spec §4.7); empty for a generated map

    @property
    def widths(self):
        """(M,) the road's width at every centerline point."""
        return self.half_left + self.half_right

    @property
    def max_width(self):
        return float(np.max(self.widths))

    @property
    def start_position(self):
        return self.centerline[0].copy()

    @property
    def start_heading(self):
        return float(np.arctan2(self.tangent[0, 1], self.tangent[0, 0]))


def resample_loop(points, spacing, values=None):
    """Resample a closed polyline (first point not repeated) to equal spacing. Returns (points, actual spacing), and
    with `values` ((N, k), one row per input point) also the values interpolated linearly along the arc."""
    closed = np.vstack([points, points[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    m = max(int(round(s[-1] / spacing)), 8)
    s_new = np.arange(m) * (s[-1] / m)
    out = np.column_stack([np.interp(s_new, s, closed[:, 0]), np.interp(s_new, s, closed[:, 1])])
    if values is None:
        return out, s[-1] / m
    closed_values = np.vstack([values, values[:1]])
    return out, s[-1] / m, np.column_stack([np.interp(s_new, s, closed_values[:, j])
                                             for j in range(closed_values.shape[1])])


def track_shape(points, half_left, half_right):
    """The geometry of a closed loop of points in driving order (spec §4.1): the centerline resampled every
    CENTERLINE_SPACING m with its spacing, the half widths interpolated along it, unit tangents, curvature (lightly
    smoothed over ±2 points) and the road edges. half_left and half_right are scalars or one value per point."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    halves = np.column_stack([np.broadcast_to(np.asarray(half_left, dtype=float), (n,)),
                              np.broadcast_to(np.asarray(half_right, dtype=float), (n,))])
    center, spacing, halves = resample_loop(points, config.CENTERLINE_SPACING, halves)
    segment = np.roll(center, -1, axis=0) - center
    seg_heading = np.arctan2(segment[:, 1], segment[:, 0])
    turn = np.angle(np.exp(1j * (seg_heading - np.roll(seg_heading, 1))))
    curvature = turn / spacing
    curvature = sum(np.roll(curvature, k) for k in range(-2, 3)) / 5.0   # light smoothing over ±2 points
    chord = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent = chord / np.linalg.norm(chord, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])             # points left of travel
    return {"center": center, "spacing": spacing, "half_left": halves[:, 0], "half_right": halves[:, 1],
            "tangent": tangent, "curvature": curvature, "left": center + normal * halves[:, :1],
            "right": center - normal * halves[:, 1:]}


def road_grid(center, tangent, half_left, half_right):
    """(origin, dist) of the road grid (spec §4.1): a cell is road when its distance to the nearest centerline sample is
    at most the half width on its side of that sample's centerline point; dist is each road cell's distance to the
    nearest grass cell (float32, 0 on grass). With equal half widths this is the band around the centerline of up to
    version 3.7."""
    m = len(center)
    margin = float(max(np.max(half_left), np.max(half_right))) + config.GRID_MARGIN
    origin = center.min(axis=0) - margin
    top = center.max(axis=0) + margin
    cols, rows = np.ceil((top - origin) / config.GRID_RES).astype(int)
    dense, dense_spacing = resample_loop(center, config.GRID_RES / 2)
    c = np.floor((dense[:, 0] - origin[0]) / config.GRID_RES).astype(int)
    r = np.floor((dense[:, 1] - origin[1]) / config.GRID_RES).astype(int)
    on_line = np.zeros((rows, cols), dtype=bool)
    on_line[r, c] = True
    owner = np.zeros((rows, cols), dtype=np.int64)
    spacing = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1).mean()
    owner[r, c] = np.mod(np.round(np.arange(len(dense)) * dense_spacing / spacing).astype(np.int64), m)
    to_line, (near_r, near_c) = distance_transform_edt(~on_line, return_indices=True)
    k = owner[near_r, near_c]                                           # the centerline point nearest each cell
    cell_x = origin[0] + (np.arange(cols) + 0.5) * config.GRID_RES
    cell_y = origin[1] + (np.arange(rows) + 0.5) * config.GRID_RES
    leftward = (-(cell_x[None, :] - center[k, 0]) * tangent[k, 1] + (cell_y[:, None] - center[k, 1]) * tangent[k, 0])
    half = np.where(leftward >= 0.0, np.asarray(half_left)[k], np.asarray(half_right)[k])
    road = to_line * config.GRID_RES <= half
    dist = np.where(road, distance_transform_edt(road) * config.GRID_RES - config.GRID_RES / 2, 0.0)
    return origin, dist.astype(np.float32)


def build_track(points, half_left, half_right, seed, level, name=""):
    """A Track from a closed loop of points in driving order (any spacing, first point not repeated) and the road's
    half widths to the left and right, scalars or one per point (spec §4.1)."""
    shape = track_shape(points, half_left, half_right)
    origin, dist = road_grid(shape["center"], shape["tangent"], shape["half_left"], shape["half_right"])
    m = len(shape["center"])
    return Track(seed=seed, level=level, centerline=shape["center"], tangent=shape["tangent"],
                 curvature=shape["curvature"], half_left=shape["half_left"], half_right=shape["half_right"],
                 left=shape["left"], right=shape["right"], spacing=shape["spacing"], length=m * shape["spacing"],
                 origin=origin, dist=dist, name=name)


def wall_distance(track, x, y):
    """Distance in m from each point to the nearest grass cell; 0 on grass or outside the grid."""
    rows, cols = track.dist.shape
    c = np.floor((np.asarray(x) - track.origin[0]) / config.GRID_RES).astype(np.int64)
    r = np.floor((np.asarray(y) - track.origin[1]) / config.GRID_RES).astype(np.int64)
    inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    return np.where(inside, track.dist[np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)], 0.0)
