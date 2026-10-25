import numpy as np

from racing import config, ideal, sim


def circle_points(radius, step=0.5):
    a = np.arange(0.0, 2 * np.pi, step / radius)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])


def test_the_corner_speed_uses_the_grip_at_that_speed():
    """Spec §4.8: the highest v with v² κ ≤ share × a(v); not capped by the top speed."""
    kappa = np.array([1 / 15.0, 1 / 60.0, 1 / 150.0, 1 / 400.0, 1 / 5000.0])
    for share in (1.0, 0.85):
        v = ideal.corner_speed(kappa, share)
        bound = kappa < 1 / 300.0
        assert np.allclose((v * v * kappa)[~bound], share * sim.grip_at(v)[~bound], rtol=1e-12)
        assert np.all(v * v * kappa <= share * sim.grip_at(v) * (1 + 1e-12))
    assert ideal.corner_speed(1 / 5000.0) > config.TOP_SPEED


def test_the_ideal_lap_of_a_circle_is_its_length_over_its_corner_speed():
    """Spec §4.8: on a circle the quasi-steady-state lap is a steady corner at the corner speed."""
    for radius in (50.0, 100.0, 400.0):
        speed = min(float(ideal.corner_speed(1 / radius)), config.TOP_SPEED)
        assert np.isclose(ideal.lap_time(circle_points(radius), smooth=0.0), 2 * np.pi * radius / speed, rtol=2e-3)


def test_a_straight_is_driven_at_the_top_speed_between_two_corners():
    """A stadium of two 1 km straights: the ideal lap is faster than driving it all at the corner speed and slower
    than driving it all at the top speed."""
    s = np.arange(0.0, 1000.0, 1.0)
    a = np.arange(-np.pi / 2, np.pi / 2, 1.0 / 80.0)
    points = np.vstack([np.column_stack([s, np.full(len(s), -80.0)]),
                        np.column_stack([1000.0 + 80.0 * np.cos(a), 80.0 * np.sin(a)]),
                        np.column_stack([1000.0 - s, np.full(len(s), 80.0)]),
                        np.column_stack([-80.0 * np.cos(a), -80.0 * np.sin(a)])])
    length = 2000.0 + 2 * np.pi * 80.0
    lap = ideal.lap_time(points)
    assert length / config.TOP_SPEED < lap < length / float(ideal.corner_speed(1 / 80.0))


def test_the_stop_speed_stops_within_its_distance():
    for distance, share in ((150.0, 0.85), (200.0, 1.0), (60.0, 1.0)):
        v = ideal.stop_speed(distance, share)
        speeds = np.linspace(v, 0.0, 200001)
        travelled = np.trapezoid(speeds / (share * sim.grip_at(speeds)), -speeds)
        assert np.isclose(travelled, distance, rtol=1e-4)
    assert 151.5 < ideal.stop_speed(150.0, 0.85) < 152.5         # version 3.11's grip (97.5-98.5 m/s to 3.10)
    assert ideal.stop_speed(200.0, 1.0) > config.TOP_SPEED      # 200 m at full grip stops from the top speed


def test_on_a_circle_the_target_speed_is_the_corner_speed_everywhere():
    radius = 100.0
    points = circle_points(radius, step=1.0)
    curvature = np.full(len(points), 1.0 / radius)
    target = ideal.sight_speeds(points, curvature, 1.0, 1.0, 200.0)
    assert np.allclose(target, min(float(ideal.corner_speed(1 / radius)), config.TOP_SPEED))


def test_the_target_slows_for_a_corner_within_the_sight_only():
    """A straight ending in a 30 m radius bend: far from the bend the target is the top speed, within braking distance
    it falls, and the drop starts no earlier than 200 m before the bend."""
    straight = np.column_stack([np.arange(0.0, 1000.0, 1.0), np.zeros(1000)])
    curvature = np.zeros(1000)
    curvature[600:700] = 1.0 / 30.0
    target = ideal.sight_speeds(straight, curvature, 1.0, 1.0, 200.0)
    assert target[300] == config.TOP_SPEED and target[650] == min(float(ideal.corner_speed(1 / 30.0)),
                                                                   config.TOP_SPEED)
    assert np.all(target[:400] == config.TOP_SPEED) and target[580] < config.TOP_SPEED   # braking from 552 m
