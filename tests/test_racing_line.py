import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import stadium
from racing import circuits, config, ideal, racing_line, track_gen
from racing.track import track_shape

PUBLISHED_CLEARANCE = 0.65   # m: the published lines' clearance from the edges (first percentile 0.56–0.70 m, §4.8)


def outline_shape(name, reverse=False):
    """circuits.circuit_track's shape of circuit `name`: reversed, its forward centerline driven the other way."""
    shape = track_shape(*circuits.outline(name))
    if reverse:
        shape = track_shape(*circuits.reverse_outline(shape["center"], shape["half_left"], shape["half_right"]))
    return shape


def check_within_bounds(center, tangent, half_left, half_right, line):
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    offset = np.einsum("ij,ij->i", line - center, normal)
    upper = np.maximum(half_left - config.RACING_LINE_MARGIN, 0.0) + config.RACING_LINE_TOLERANCE + 1e-9
    lower = -np.maximum(half_right - config.RACING_LINE_MARGIN, 0.0) - config.RACING_LINE_TOLERANCE - 1e-9
    assert np.all(offset <= upper) and np.all(offset >= lower)


def test_a_track_has_one_line_point_direction_and_curvature_per_centerline_point(circle_track):
    m = len(circle_track.centerline)
    assert circle_track.line.shape == (m, 2) and circle_track.line_tangent.shape == (m, 2)
    assert circle_track.line_heading.shape == (m,) and circle_track.line_curvature.shape == (m,)
    assert np.allclose(np.hypot(*circle_track.line_tangent.T), 1.0)
    assert np.array_equal(circle_track.line_heading,
                          np.arctan2(circle_track.line_tangent[:, 1], circle_track.line_tangent[:, 0]))
    for name in ("line", "line_tangent", "line_heading", "line_curvature"):
        array = getattr(circle_track, name)
        assert array.dtype == np.float64 and array.flags["C_CONTIGUOUS"], name


def test_on_a_stadium_the_line_runs_wide_on_the_straights_and_inside_at_the_apex():
    """Spec §4.8: on a counter-clockwise stadium (12 m road) the line keeps to the outside (right) along the middle of
    each straight and to the inside (left) at each bend's apex, at the margin from the edge."""
    track = stadium()
    normal = np.column_stack([-track.tangent[:, 1], track.tangent[:, 0]])
    offset = np.einsum("ij,ij->i", track.line - track.centerline, normal)
    reach = 6.0 - config.RACING_LINE_MARGIN
    apex = np.argmax(track.centerline[:, 0])                     # the far point of the right-hand bend
    middle = np.argmin(np.abs(track.centerline[:, 0]) + 1e3 * (track.centerline[:, 1] > 0))   # bottom straight
    assert offset[apex] > reach - 0.2 and offset[middle] < -reach + 0.2
    check_within_bounds(track.centerline, track.tangent, track.half_left, track.half_right, track.line)


@pytest.mark.parametrize("level", [1, 2, 3, config.VALIDATION_LEVEL])
def test_every_point_of_a_generated_maps_line_lies_within_its_bounds(level):
    built = [t for t in (track_gen.candidate_track(seed, level, 0) for seed in (0, 1, 2)) if t is not None]
    assert built
    for track in built:
        check_within_bounds(track.centerline, track.tangent, track.half_left, track.half_right, track.line)


@pytest.mark.parametrize("name, reverse", [("Monza", False), ("Monza", True), ("MexicoCity", False),
                                           ("YasMarina", False)])
def test_every_point_of_a_real_layouts_line_lies_within_its_bounds(name, reverse):
    track = circuits.circuit_track(name, reverse)
    check_within_bounds(track.centerline, track.tangent, track.half_left, track.half_right, track.line)


def test_the_line_is_the_same_when_computed_again():
    shape = outline_shape("Spielberg")
    first = racing_line.solve(shape["center"], shape["tangent"], shape["half_left"], shape["half_right"],
                              shape["spacing"])
    again = racing_line.solve(shape["center"], shape["tangent"], shape["half_left"], shape["half_right"],
                              shape["spacing"])
    for a, b in zip(first, again):
        assert np.array_equal(a, b)


def test_the_line_is_the_same_in_a_process_with_more_blas_threads(tmp_path):
    """Spec §4.8: the solve holds BLAS to one thread, so a joblib worker, which runs with fewer threads than the main
    process, and a resumed run compute the same line to the last bit."""
    shape = outline_shape("Spielberg")
    here = racing_line.solve(shape["center"], shape["tangent"], shape["half_left"], shape["half_right"],
                             shape["spacing"])[0]
    code = ("import sys, numpy as np\n"
            "from racing import circuits, racing_line\n"
            "from racing.track import track_shape\n"
            "s = track_shape(*circuits.outline('Spielberg'))\n"
            "np.save(sys.argv[1], racing_line.solve(s['center'], s['tangent'], s['half_left'], s['half_right'],"
            " s['spacing'])[0])\n")
    out = tmp_path / "line.npy"
    environ = {**os.environ, "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    subprocess.run([sys.executable, "-c", code, str(out)], check=True, env=environ, cwd=config.PROJECT_DIR)
    assert np.array_equal(np.load(out), here)


def test_a_real_circuits_line_is_cached_once_and_equals_a_fresh_one(monkeypatch):
    shape = outline_shape("Spielberg", reverse=True)
    fresh = racing_line.fields(shape["center"], shape["tangent"], shape["half_left"], shape["half_right"],
                               shape["spacing"])
    first = circuits.circuit_track("Spielberg", True)
    files = list(config.RACING_LINES_DIR.glob(f"Spielberg-reversed-{circuits.line_id('Spielberg')}*.npy"))
    assert len(files) == 1
    monkeypatch.setattr(racing_line, "solve", lambda *args, **kwargs: pytest.fail("the cache was not used"))
    second = circuits.circuit_track("Spielberg", True)
    for name, array in fresh.items():
        assert np.array_equal(getattr(first, name), array) and np.array_equal(getattr(second, name), array), name


def test_the_published_lines_cover_the_23_circuits():
    names = set(config.TRAINING_CIRCUITS) | set(config.TEST_CIRCUITS)
    assert {path.stem for path in racing_line.PUBLISHED_DIR.glob("*.csv")} == names


@pytest.mark.slow
@pytest.mark.parametrize("name", config.TRAINING_CIRCUITS + config.TEST_CIRCUITS)
def test_the_line_is_within_2_percent_of_the_published_line(name):
    """Spec §4.8, §12.1: with the published lines' clearance of 0.65 m, the ideal lap on this line is within 2% of the
    ideal lap on the published line (both with the curvature smoothed over 10 m)."""
    shape = outline_shape(name)
    line, _, _ = racing_line.solve(shape["center"], shape["tangent"], shape["half_left"], shape["half_right"],
                                   shape["spacing"], margin=PUBLISHED_CLEARANCE)
    published = np.loadtxt(racing_line.PUBLISHED_DIR / f"{name}.csv", delimiter=",", comments="#")[:, :2]
    ours, theirs = ideal.lap_time(line), ideal.lap_time(published)
    assert abs(ours / theirs - 1.0) <= 0.02, (name, ours, theirs)
