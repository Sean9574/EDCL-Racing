"""Real circuits (spec §4.7): the racetrack database of the Technical University of Munich, vendored in
data/racetracks/ (LGPL-3.0), read at real scale with the road's width to the left and to the right of every point;
the training and test circuits, their racing direction and official length, and the pole laps the test circuits are
compared with (spec §11.1). Their tracks are Extra hard's 38 training layouts and the test level's four circuits
(spec §4.6), whose road grids are built once per machine and memory-mapped by every process (spec §9.10)."""
import hashlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from racing import config, ideal
from racing.track import Track, road_grid, track_shape

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


_grid_lock = threading.Lock()
OLD_GRID_AGE = 24 * 3600     # s: a grid of another identity is removed once it is this old (as fastsim libraries)


def grid_id(name):
    """The identity of circuit `name`'s road grid: the first 16 hexadecimal digits of the SHA-256 of its data file, the
    grid's settings and the code that builds it. A grid of another identity is never used."""
    digest = hashlib.sha256((DATA_DIR / f"{name}.csv").read_bytes())
    for value in (config.GRID_RES, config.GRID_MARGIN, config.CENTERLINE_SPACING):
        digest.update(repr(float(value)).encode())
    for source in ("track.py", "circuits.py"):
        digest.update(Path(__file__).with_name(source).read_bytes())
    return digest.hexdigest()[:16]


def _save(path, array):
    """np.save to a temporary file in the same folder, then renamed, so no process reads a partly written file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(temporary, "wb") as f:
        np.save(f, array)
    os.replace(temporary, path)


def _remove_old_grids(folder, name, stem, age=OLD_GRID_AGE):
    """Removes circuit `name`'s grid files (and leftover temporary files) of other identities than `stem` once they are
    `age` old, as fastsim.remove_old_libraries does; a younger one may still be in use by a job that started before
    the source changed. Safe on Linux even while another process has the old file mapped."""
    cutoff = time.time() - age
    for old in [*folder.glob(f"{name}-[0-9a-f]*.npy"), *folder.glob(f".{name}-[0-9a-f]*.tmp")]:
        if old.name.startswith(f"{stem}.") or old.name.startswith(f".{stem}."):
            continue
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except FileNotFoundError:                    # removed by another job at the same moment
            pass


def _grid(name, shape):
    """(origin, dist) of circuit `name` in its racing direction (spec §4.7, §9.10): built once per machine in
    CIRCUIT_GRIDS_DIR and memory-mapped read-only, so every process and both directions share one copy."""
    folder = Path(config.CIRCUIT_GRIDS_DIR)
    stem = f"{name}-{grid_id(name)}"
    dist_path, origin_path = folder / f"{stem}.npy", folder / f"{stem}.origin.npy"
    with _grid_lock:
        if not (dist_path.exists() and origin_path.exists()):
            folder.mkdir(parents=True, exist_ok=True)
            origin, dist = road_grid(shape["center"], shape["tangent"], shape["half_left"], shape["half_right"])
            _save(origin_path, origin)
            _save(dist_path, dist)                   # written last: its presence means the grid is complete
            _remove_old_grids(folder, name, stem)
    return np.load(origin_path), np.load(dist_path, mmap_mode="r")


OLD_LINE_AGE = OLD_GRID_AGE  # s: a line file of another identity is removed once it is this old


def line_id(name):
    """The identity of circuit `name`'s racing lines: the first 16 hexadecimal digits of the SHA-256 of its data file,
    the line's settings and the code that computes it. A line of another identity is never used."""
    digest = hashlib.sha256((DATA_DIR / f"{name}.csv").read_bytes())
    for value in (config.CENTERLINE_SPACING, config.RACING_LINE_MARGIN, config.RACING_LINE_STEP,
                  config.RACING_LINE_ITERATIONS, config.RACING_LINE_ROUNDS, config.RACING_LINE_TOLERANCE,
                  config.RACING_LINE_SMOOTHING):
        digest.update(repr(float(value)).encode())
    for source in ("track.py", "circuits.py", "racing_line.py"):
        digest.update(Path(__file__).with_name(source).read_bytes())
    return digest.hexdigest()[:16]


def _line(name, reverse, shape):
    """The racing-line fields of circuit `name` in one direction (spec §4.8, §9.10): computed once per machine into
    RACING_LINES_DIR as an (M, 6) array [x, y, tx, ty, heading, curvature] and read by every process."""
    from racing import racing_line
    folder = Path(config.RACING_LINES_DIR)
    stem = f"{name}{'-reversed' if reverse else ''}-{line_id(name)}"
    path = folder / f"{stem}.npy"
    with _grid_lock:
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            line = racing_line.fields(shape["center"], shape["tangent"], shape["half_left"], shape["half_right"],
                                      shape["spacing"])
            _save(path, np.column_stack([line["line"], line["line_tangent"], line["line_heading"],
                                         line["line_curvature"]]))
            _remove_old_grids(folder, f"{name}{'-reversed' if reverse else ''}", stem, OLD_LINE_AGE)
    table = np.load(path)
    return {"line": np.ascontiguousarray(table[:, 0:2]), "line_tangent": np.ascontiguousarray(table[:, 2:4]),
            "line_heading": np.ascontiguousarray(table[:, 4]), "line_curvature": np.ascontiguousarray(table[:, 5])}


def circuit_track(name, reverse=False, seed=0, level=config.MAX_LEVEL):
    """Circuit `name` as a Track at real scale with its widths (spec §4.7), in its racing direction or reversed from
    the same start; the road grid is that of the racing direction, which is the same road. The expert's target speeds
    (spec §7.1) are computed from the cached line, never cached: they follow EXPERT_GRIP_MARGIN."""
    points, left, right = outline(name)
    shape = track_shape(points, left, right)
    origin, dist = _grid(name, shape)
    if reverse:
        shape = track_shape(*reverse_outline(shape["center"], shape["half_left"], shape["half_right"]))
    line = _line(name, reverse, shape)
    expert_speed = ideal.sight_speeds(line["line"], line["line_curvature"], shape["spacing"],
                                      config.EXPERT_GRIP_MARGIN, config.EXPERT_SIGHT)
    m = len(shape["center"])
    return Track(seed=int(seed), level=int(level), centerline=shape["center"], tangent=shape["tangent"],
                 curvature=shape["curvature"], half_left=shape["half_left"], half_right=shape["half_right"],
                 left=shape["left"], right=shape["right"], spacing=shape["spacing"], length=m * shape["spacing"],
                 origin=origin, dist=dist, **line, expert_speed=expert_speed, name=name)


def level_track(seed, level):
    """The real track of (seed, level) (spec §4.6): at MAX_LEVEL training layout `seed` (0-37, layout()), at TEST_LEVEL
    test circuit TEST_CIRCUITS[seed] in its racing direction."""
    if level == config.MAX_LEVEL:
        name, backwards = layout(seed)
        return circuit_track(name, backwards, seed, level)
    if level == config.TEST_LEVEL:
        if not 0 <= int(seed) < len(config.TEST_CIRCUITS):
            raise ValueError(f"test circuits are numbered 0 to {len(config.TEST_CIRCUITS) - 1}, not {seed}")
        return circuit_track(config.TEST_CIRCUITS[int(seed)], False, seed, level)
    raise ValueError(f"level {level} has no real circuits")
