"""Real circuits (spec §4.7): the racetrack database of the Technical University of Munich, vendored in
data/racetracks/ (LGPL-3.0), read at real scale with the road's width to the left and to the right of every point;
the training and test circuits, their racing direction and official length, and the pole laps the test circuits are
compared with (spec §11.1)."""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from racing import config

DATA_DIR = config.PROJECT_DIR / "data" / "racetracks" / "tracks"
N_LAYOUTS = 2 * len(config.TRAINING_CIRCUITS)        # every training circuit in both directions (spec §4.4)

NAMES = {"Austin": "Austin", "Budapest": "Budapest", "Catalunya": "Catalunya", "Hockenheim": "Hockenheim",
         "Melbourne": "Melbourne", "MexicoCity": "Mexico City", "Montreal": "Montreal", "Monza": "Monza",
         "Sakhir": "Sakhir", "SaoPaulo": "São Paulo", "Sepang": "Sepang", "Shanghai": "Shanghai",
         "Silverstone": "Silverstone", "Sochi": "Sochi", "Spa": "Spa", "Spielberg": "Spielberg", "Suzuka": "Suzuka",
         "YasMarina": "Yas Marina", "BrandsHatch": "Brands Hatch", "IMS": "Indianapolis",
         "MoscowRaceway": "Moscow Raceway", "Norisring": "Norisring", "Nuerburgring": "Nürburgring",
         "Oschersleben": "Oschersleben", "Zandvoort": "Zandvoort"}

# The racing direction is the database's order of points; its sense, checked against the official circuit maps for
# the Formula 1 circuits, guards against a file read backwards. The other circuits are as the database orders them.
ROTATION = {name: "clockwise" for name in NAMES}
ROTATION.update({name: "counter-clockwise" for name in ("Austin", "SaoPaulo", "YasMarina", "IMS", "MoscowRaceway",
                                                         "Norisring")})

OFFICIAL_LENGTH = {          # km, the layouts of about 2019 (spec §4.7)
    "Austin": 5.513, "Budapest": 4.381, "Catalunya": 4.655, "Hockenheim": 4.574, "Melbourne": 5.303,
    "MexicoCity": 4.304, "Montreal": 4.361, "Monza": 5.793, "Sakhir": 5.412, "SaoPaulo": 4.309, "Sepang": 5.543,
    "Shanghai": 5.451, "Silverstone": 5.891, "Sochi": 5.848, "Spa": 7.004, "Spielberg": 4.318, "Suzuka": 5.807,
    "YasMarina": 5.554, "BrandsHatch": 3.908, "IMS": 4.023, "MoscowRaceway": 4.070, "Norisring": 2.300,
    "Nuerburgring": 5.148, "Oschersleben": 3.696, "Zandvoort": 4.307}


@dataclass(frozen=True)
class PoleLap:
    seconds: float
    driver: str
    season: int
    source: str


POLE_LAPS = {                # spec §11.1, Table 5: the last season run on the database's layout
    "Melbourne": PoleLap(80.486, "Lewis Hamilton", 2019, "Formula1.com qualifying report"),
    "Sakhir": PoleLap(87.866, "Charles Leclerc", 2019, "FIA; Sky Sports timesheet"),
    "Sepang": PoleLap(90.076, "Lewis Hamilton", 2017, "FIA pole position document"),
    "Silverstone": PoleLap(85.093, "Valtteri Bottas", 2019, "Formula1.com qualifying report"),
}


def read(name):
    """(x, y, right, left) of the database file `name`: the centre line in m and the road's width to the right and to
    the left of every point (columns x_m, y_m, w_tr_right_m, w_tr_left_m)."""
    data = np.loadtxt(DATA_DIR / f"{name}.csv", delimiter=",", comments="#")
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def outline(name, spacing=config.CENTERLINE_SPACING):
    """Circuit `name` in its racing direction from the database's first point (spec §4.7): (points (n, 2), half_left
    (n,), half_right (n,)), where the half widths are the distances from the centre line to the left and right edges.
    The centre line and both widths are periodic cubic splines over the arc length of the data, sampled at n evenly
    spaced positions about `spacing` m apart."""
    x, y, right, left = read(name)
    values = np.column_stack([x, y, left, right])
    closed = np.vstack([values, values[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1])))])
    spline = CubicSpline(s, closed, bc_type="periodic")
    n = max(int(round(s[-1] / spacing)), 8)
    sampled = spline(np.arange(n) * (s[-1] / n))
    return sampled[:, :2], sampled[:, 2], sampled[:, 3]


def reverse_outline(points, half_left, half_right):
    """The same circuit driven the other way from the same start (spec §4.4): point 0 stays, the others run in reverse
    order, and the left and right widths change sides."""
    order = np.concatenate([[0], np.arange(len(points) - 1, 0, -1)])
    return points[order], half_right[order], half_left[order]


def perimeter(points):
    """Length in m of the closed outline through `points`."""
    closed = np.vstack([points, points[:1]])
    return float(np.sum(np.hypot(*np.diff(closed, axis=0).T)))


def rotation(points):
    """'clockwise' or 'counter-clockwise': the sense of the closed outline, from the sign of its area."""
    x, y = points[:, 0], points[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return "counter-clockwise" if area > 0 else "clockwise"


def complexity(points, step=2.0):
    """The complexity measures of Table 1e (spec §4.7) of a closed outline: its length in km, its corners (stretches
    whose radius, smoothed over 20 m, stays below 300 m and that turn by more than 20°), corners per km and degrees
    of turning per km."""
    closed = np.vstack([points, points[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    t = np.arange(0.0, s[-1], step)
    x, y = np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])
    k = 5                                                         # heading over ±10 m
    heading = np.unwrap(np.arctan2(np.roll(y, -k) - np.roll(y, k), np.roll(x, -k) - np.roll(x, k)))
    wrap = heading[:1] + 2 * np.pi * np.sign(heading[-1] - heading[0])
    turn = np.diff(np.concatenate([heading, wrap]))
    curvature = np.convolve(turn / step, np.ones(10) / 10, mode="same")
    tight = 1.0 / np.maximum(np.abs(curvature), 1e-9) < 300.0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], tight.astype(int), [0]])))
    corners = sum(1 for a, b in zip(edges[::2], edges[1::2]) if np.degrees(abs(np.sum(turn[a:b]))) > 20.0)
    km = s[-1] / 1000.0
    return {"km": km, "corners": corners, "corners_per_km": corners / km,
            "turning_per_km": np.degrees(np.sum(np.abs(turn))) / km}


def layout(index):
    """(circuit, reversed) of training layout `index` (spec §4.6): TRAINING_CIRCUITS[index // 2], in its racing
    direction for an even index and the other way for an odd one."""
    index = int(index)
    if not 0 <= index < N_LAYOUTS:
        raise ValueError(f"training layouts are numbered 0 to {N_LAYOUTS - 1}, not {index}")
    return config.TRAINING_CIRCUITS[index // 2], bool(index % 2)
