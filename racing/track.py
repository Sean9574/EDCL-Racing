"""Track data structure and geometry shared by the generator, the simulator and the viewer (spec §4.1)."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from racing import config


@dataclass(eq=False)
class Track:
    """A closed-loop track. Cars drive in the direction of increasing arc length."""
    seed: int
    level: int
    width: float
    centerline: np.ndarray      # (M, 2), point 0 is the start line
    tangent: np.ndarray         # (M, 2) unit direction of travel
    curvature: np.ndarray       # (M,) signed curvature in 1/m, positive for left turns
    left: np.ndarray            # (M, 2) left road edge
    right: np.ndarray           # (M, 2) right road edge
    spacing: float              # m between centerline points
    length: float               # m, equal to M * spacing
    origin: np.ndarray          # (2,) world position of the road grid's lower-left corner
    dist: np.ndarray            # (rows, cols) m from each road cell to the nearest grass cell; 0 on grass
    expert_lap_time: float = float("nan")

    @property
    def start_position(self):
        return self.centerline[0].copy()

    @property
    def start_heading(self):
        return float(np.arctan2(self.tangent[0, 1], self.tangent[0, 0]))


def resample_loop(points, spacing):
    """Resample a closed polyline (first point not repeated) to equal spacing. Returns (points, actual spacing)."""
    closed = np.vstack([points, points[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    m = max(int(round(s[-1] / spacing)), 8)
    s_new = np.arange(m) * (s[-1] / m)
    return np.column_stack([np.interp(s_new, s, closed[:, 0]), np.interp(s_new, s, closed[:, 1])]), s[-1] / m


def build_track(points, width, seed, level):
    """Build a Track from a closed loop of points in driving order (any spacing, first point not repeated)."""
    center, spacing = resample_loop(np.asarray(points, dtype=float), config.CENTERLINE_SPACING)
    m = len(center)

    segment = np.roll(center, -1, axis=0) - center
    seg_heading = np.arctan2(segment[:, 1], segment[:, 0])
    turn = np.angle(np.exp(1j * (seg_heading - np.roll(seg_heading, 1))))
    curvature = turn / spacing
    curvature = sum(np.roll(curvature, k) for k in range(-2, 3)) / 5.0   # light smoothing over ±2 points

    chord = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent = chord / np.linalg.norm(chord, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])             # points left of travel
    left = center + normal * width / 2
    right = center - normal * width / 2

    origin = center.min(axis=0) - (width / 2 + config.GRID_MARGIN)
    top = center.max(axis=0) + (width / 2 + config.GRID_MARGIN)
    cols, rows = np.ceil((top - origin) / config.GRID_RES).astype(int)
    dense, _ = resample_loop(center, config.GRID_RES / 2)
    c = np.floor((dense[:, 0] - origin[0]) / config.GRID_RES).astype(int)
    r = np.floor((dense[:, 1] - origin[1]) / config.GRID_RES).astype(int)
    on_line = np.zeros((rows, cols), dtype=bool)
    on_line[r, c] = True
    road = distance_transform_edt(~on_line) * config.GRID_RES <= width / 2
    dist = np.where(road, distance_transform_edt(road) * config.GRID_RES - config.GRID_RES / 2, 0.0)

    return Track(seed=seed, level=level, width=float(width), centerline=center, tangent=tangent,
                 curvature=curvature, left=left, right=right, spacing=spacing, length=m * spacing,
                 origin=origin, dist=dist)


def wall_distance(track, x, y):
    """Distance in m from each point to the nearest grass cell; 0 on grass or outside the grid."""
    rows, cols = track.dist.shape
    c = np.floor((np.asarray(x) - track.origin[0]) / config.GRID_RES).astype(np.int64)
    r = np.floor((np.asarray(y) - track.origin[1]) / config.GRID_RES).astype(np.int64)
    inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    return np.where(inside, track.dist[np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)], 0.0)
