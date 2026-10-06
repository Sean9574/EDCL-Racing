import numpy as np

from racing.track import resample_loop, wall_distance


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


def test_start_pose(circle_track):
    assert np.allclose(circle_track.start_position, [0, -50], atol=1e-9)
    assert abs(circle_track.start_heading) < 0.01                         # driving +x at the bottom


def test_wall_distance(circle_track):
    d = wall_distance(circle_track, np.zeros(4), np.array([-50.0, -45.0, 0.0, -70.0]))
    assert abs(d[0] - 8.0) < 0.6      # on the centerline: half the width
    assert abs(d[1] - 3.0) < 0.6      # 5 m toward the infield
    assert d[2] == 0.0                # infield grass
    assert d[3] == 0.0                # outside the grid
