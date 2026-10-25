import numpy as np
from scipy.ndimage import distance_transform_edt

from racing import config
from racing.track import build_track, resample_loop, track_shape, wall_distance


def test_resample_loop_is_evenly_spaced():
    a = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    points, spacing = resample_loop(np.column_stack([10 * np.cos(a), 10 * np.sin(a)]), 1.0)
    gaps = np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)
    assert abs(spacing - 1.0) < 0.05
    assert np.allclose(gaps, gaps.mean(), rtol=0.05)


def test_circle_geometry(circle_track):
    t = circle_track
    assert abs(t.length - 2 * np.pi * 50) < 1.0
    assert np.allclose(t.curvature, 1 / 50, atol=0.002)                  # counter-clockwise: left turn
    assert np.allclose(np.linalg.norm(t.left, axis=1), 42, atol=0.05)     # left of travel is the infield
    assert np.allclose(np.linalg.norm(t.right, axis=1), 58, atol=0.05)
    assert np.isclose(t.length, len(t.centerline) * t.spacing)


def test_a_built_track_has_its_racing_line(circle_track):
    """Spec §4.1, §4.8: build_track gives every track a line point, direction, heading and curvature per centerline
    point."""
    m = len(circle_track.centerline)
    assert circle_track.line.shape == (m, 2) and circle_track.line_tangent.shape == (m, 2)
    assert circle_track.line_heading.shape == (m,) and circle_track.line_curvature.shape == (m,)


def test_a_built_track_has_the_experts_target_speeds(monkeypatch, circle_track):
    """Spec §7.1: the expert's target speed at every point is the speed profile of its sight along the racing line at
    its grip share, computed when a track is built; a real circuit's follows the grip share, not its cached line."""
    from racing import circuits, ideal

    def profile(track):
        return ideal.sight_speeds(track.line, track.line_curvature, track.spacing, config.EXPERT_GRIP_MARGIN,
                                  config.EXPERT_SIGHT)

    for track in (circle_track, circuits.circuit_track("Spielberg", True)):
        assert np.array_equal(track.expert_speed, profile(track))
        assert track.expert_speed.dtype == np.float64 and track.expert_speed.flags["C_CONTIGUOUS"]
    before = circuits.circuit_track("Spielberg", True).expert_speed
    monkeypatch.setattr(config, "EXPERT_GRIP_MARGIN", config.EXPERT_GRIP_MARGIN - 0.1)   # a share other than 0.80
    slower = circuits.circuit_track("Spielberg", True)
    assert np.array_equal(slower.expert_speed, profile(slower)) and np.all(slower.expert_speed <= before)
    assert np.any(slower.expert_speed < before)          # the speeds follow the share, not the cached racing line


def test_start_pose(circle_track):
    assert np.allclose(circle_track.start_position, [0, -50], atol=1e-9)
    assert abs(circle_track.start_heading) < 0.01                         # driving +x at the bottom


def test_wall_distance(circle_track):
    d = wall_distance(circle_track, np.zeros(4), np.array([-50.0, -45.0, 0.0, -70.0]))
    assert abs(d[0] - 8.0) < 0.6      # on the centerline: half the width
    assert abs(d[1] - 3.0) < 0.6      # 5 m toward the infield
    assert d[2] == 0.0                # infield grass
    assert d[3] == 0.0                # outside the grid


# ----- per-point road widths (spec §4.1) -----

def test_a_symmetric_road_is_the_old_band_around_the_centerline(circle_track):
    """Equal widths reproduce the road of up to version 3.7: every cell within half the width of the centerline."""
    a = np.linspace(-np.pi / 2, 1.5 * np.pi, 3200, endpoint=False)
    points = np.column_stack([50 * np.cos(a), 50 * np.sin(a)])
    track = build_track(points, 8.0, 8.0, seed=0, level=1)
    shape = track_shape(points, 8.0, 8.0)
    dense, _ = resample_loop(shape["center"], config.GRID_RES / 2)
    rows, cols = track.dist.shape
    on_line = np.zeros((rows, cols), dtype=bool)
    on_line[np.floor((dense[:, 1] - track.origin[1]) / config.GRID_RES).astype(int),
            np.floor((dense[:, 0] - track.origin[0]) / config.GRID_RES).astype(int)] = True
    old_road = distance_transform_edt(~on_line) * config.GRID_RES <= 8.0
    assert np.array_equal(track.dist > 0, old_road)
    assert track.dist.dtype == np.float32 and track.max_width == 16.0 and np.all(track.widths == 16.0)


def test_each_side_of_the_road_has_its_own_width():
    points = np.column_stack([np.linspace(-300.0, 300.0, 1201), np.zeros(1201)])       # a long straight, closed
    points = np.vstack([points, np.column_stack([np.linspace(300.0, -300.0, 1201), np.full(1201, 400.0)])[1:-1]])
    track = build_track(points, 3.0, 6.0, seed=0, level=1)
    assert wall_distance(track, 0.0, 2.5) > 0.0 and wall_distance(track, 0.0, 3.6) == 0.0     # left of travel: 3 m
    assert wall_distance(track, 0.0, -5.5) > 0.0 and wall_distance(track, 0.0, -6.6) == 0.0  # right of travel: 6 m


def test_widths_change_along_the_lap():
    a = np.linspace(-np.pi / 2, 1.5 * np.pi, 3200, endpoint=False)
    points = np.column_stack([200 * np.cos(a), 200 * np.sin(a)])
    half = 4.0 + 2.0 * (np.cos(a + np.pi / 2) > 0)                    # 8 m wide on half of the lap, 12 m on the other
    track = build_track(points, half, half, seed=0, level=1)
    widths = np.round(track.widths, 6)
    steps = (widths != 8.0) & (widths != 12.0)                       # interpolated where a step falls between points
    assert set(widths[~steps]) == {8.0, 12.0} and np.count_nonzero(steps) <= 2
    assert np.all((widths[steps] > 8.0) & (widths[steps] < 12.0))
