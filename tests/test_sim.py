import numpy as np
import pytest

from racing import config, expert, sim
from racing.actions import class_to_controls, controls_to_class
from racing.track import build_track

from conftest import stadium


def constant_policy(steer, throttle):
    action = int(controls_to_class([steer], [throttle])[0])
    return lambda obs, cars: np.full(len(cars.x), action)


def test_zero_steering_drives_straight(circle_track):
    cars = sim.start_cars(circle_track, 1)
    x0, y0, h0 = cars.x[0], cars.y[0], cars.heading[0]
    for _ in range(20):
        sim.step(circle_track, cars, np.array([0.0]), np.array([1.0]), time_limit=100.0)
    assert cars.heading[0] == h0
    assert np.isclose(np.arctan2(cars.y[0] - y0, cars.x[0] - x0), h0)
    assert cars.speed[0] > 0


def test_grip_limit_is_never_exceeded(circle_track):
    cars = sim.start_cars(circle_track, 1)
    cars.speed[:] = 10.0
    h0 = cars.heading[0]
    sim.step(circle_track, cars, np.array([1.0]), np.array([1.0]), time_limit=100.0)
    a_long = (cars.speed[0] - 10.0) / config.DT
    a_lat = (cars.heading[0] - h0) / config.DT * 10.0
    assert np.hypot(a_long, a_lat) <= sim.grip_at(10.0) + 1e-9
    assert np.isclose(a_long, a_lat)                  # proportional scaling keeps the requested direction


def test_leaving_the_road_is_a_crash(circle_track):
    result = sim.simulate(circle_track, constant_policy(0.0, 1.0), 1)
    assert result.crashed[0] and not result.finished[0]
    assert 0.0 < result.progress[0] < 0.5


def test_car_that_never_moves_stops_at_the_stall_timeout(circle_track):
    result = sim.simulate(circle_track, constant_policy(0.0, 0.0), 1)
    assert not result.finished[0] and not result.crashed[0]
    assert result.steps * config.DT <= config.STALL_TIME + 2 * config.DT
    assert result.fitness[0] == 0.0


def test_backward_driving_earns_nothing(circle_track):
    cars = sim.start_cars(circle_track, 1)
    cars.heading[:] += np.pi
    cars.speed[:] = 10.0
    steer = np.array([-(10.0 ** 2 / 50.0) / sim.grip_at(10.0)])   # clockwise turn that follows the circle backwards
    while not cars.done[0]:
        sim.step(circle_track, cars, steer, np.array([0.0]), time_limit=100.0)
    assert cars.progress[0] == 0.0 and cars.laps_done[0] == 0
    assert not cars.crashed[0]                               # stopped by the stall timeout, still on the road


def test_full_lap_is_detected(circle_track):
    cars = sim.start_cars(circle_track, 1)
    cars.speed[:] = 15.0
    steer = np.array([(15.0 ** 2 / 50.0) / sim.grip_at(15.0)])
    while not cars.done[0]:
        sim.step(circle_track, cars, steer, np.array([0.0]), time_limit=100.0)
    assert cars.finished[0] and not cars.crashed[0]
    assert abs(cars.finish_time[0] - circle_track.length / 15.0) < 0.5
    assert np.isclose(cars.last_lap[0], cars.finish_time[0])


def test_multi_lap_race_counts_laps(circle_track):
    cars = sim.start_cars(circle_track, 1, laps=2)
    cars.speed[:] = 15.0
    steer = np.array([(15.0 ** 2 / 50.0) / sim.grip_at(15.0)])
    while not cars.done[0]:
        sim.step(circle_track, cars, steer, np.array([0.0]), time_limit=100.0)
    assert cars.finished[0] and cars.laps_done[0] == 2
    assert np.isfinite(cars.best_lap[0])


def test_fitness_formula(circle_track):
    cars = sim.start_cars(circle_track, 3)
    cars.progress[:] = [0.0, circle_track.length * 0.5, circle_track.length]
    cars.finished[:] = [False, False, True]
    cars.finish_time[:] = [np.nan, np.nan, 10.0]
    f = sim.fitness(cars, circle_track, time_limit=40.0)
    assert f[0] == 0.0
    assert abs(f[1] - 0.5) < 1e-12
    assert abs(f[2] - 1.75) < 1e-12


def test_rays_on_circle(circle_track):
    cars = sim.start_cars(circle_track, 1)
    rays = sim.cast_rays(circle_track, cars.x, cars.y, cars.heading)[0]
    assert abs(rays[0] - 8.0) < 0.75                        # right: outer edge
    assert abs(rays[-1] - 8.0) < 1.0                       # left: inner edge
    assert abs(rays[4] - np.sqrt(58 ** 2 - 50 ** 2)) < 0.75  # ahead: chord to the outer edge


def test_sense_is_zero_for_finished_cars(circle_track):
    cars = sim.start_cars(circle_track, 2)
    cars.done[1] = True
    obs = sim.sense(circle_track, cars)
    assert obs.shape == (2, config.N_INPUTS)
    assert np.all(obs[1] == 0.0) and np.all(obs[0, :config.N_RAYS] > 0)


def test_cars_are_independent_of_the_batch(circle_track):
    def run(speeds):
        cars = sim.start_cars(circle_track, len(speeds))
        cars.speed[:] = speeds
        for _ in range(200):
            obs = sim.sense(circle_track, cars)
            classes = np.floor(obs[:, 4] * 13 + obs[:, 9] * 7).astype(np.int64) % config.N_ACTIONS
            steer, throttle = class_to_controls(classes)
            sim.step(circle_track, cars, steer, throttle, time_limit=100.0)
        return cars

    speeds = 5.0 + np.arange(8.0)
    full, part = run(speeds), run(speeds[[2, 5]])
    for name in ("x", "y", "heading", "speed", "progress"):
        assert np.array_equal(getattr(full, name)[[2, 5]], getattr(part, name))


def test_time_limit_requires_expert_time(circle_track):
    circle_track.expert_lap_time = float("nan")
    try:
        sim.time_limit_for(circle_track)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_simulate_reports_laps_done(circle_track):
    result = sim.simulate(circle_track, expert.expert_policy(circle_track), 2, laps=2)
    assert result.finished.all() and list(result.laps_done) == [2, 2]
    assert sim.simulate(circle_track, constant_policy(0.0, 1.0), 1).laps_done[0] == 0


def test_target_time_is_a_factor_of_the_expert_laps(circle_track):
    assert np.isclose(sim.target_time(circle_track, 1.3), 1.3 * 20.0 * config.LAPS)
    assert sim.target_time(circle_track, 1.0, laps=1) == 20.0


# ----- revision 3.1: per-car sensor length and steering sensitivity, the grip scale as a parameter -----

def road_poses(track, n, seed):
    """n random positions on the road with random headings."""
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(track.dist > 0)
    pick = rng.choice(len(rows), n, replace=False)
    x = track.origin[0] + (cols[pick] + 0.5) * config.GRID_RES
    y = track.origin[1] + (rows[pick] + 0.5) * config.GRID_RES
    return x, y, rng.uniform(-np.pi, np.pi, n)


def default_setups(n):
    return np.tile(np.asarray(config.DEFAULT_SETUP, dtype=float), (n, 1))


def test_simulate_with_the_default_setup_equals_simulate_without_one(circle_track):
    policy = expert.expert_policy(circle_track)
    plain = sim.simulate(circle_track, policy, 3, laps=2)
    explicit = sim.simulate(circle_track, policy, 3, laps=2, setup=default_setups(3))
    for name in ("fitness", "finished", "crashed", "progress", "laps_done"):
        assert np.array_equal(getattr(plain, name), getattr(explicit, name))
    assert np.array_equal(plain.finish_time, explicit.finish_time, equal_nan=True)


def test_rays_reach_their_cars_sensor_length():
    track = stadium(straight=600.0)
    x, y, heading = np.full(3, -300.0), np.full(3, -60.0), np.zeros(3)     # start of the bottom straight, centered
    rays = sim.cast_rays(track, x, y, heading, np.array([30.0, 150.0, 300.0]))
    assert list(rays[:, 4]) == [30.0, 150.0, 300.0]                          # straight ahead: 600 m of road
    assert np.all(rays[0] <= 30.0) and np.all(rays <= np.array([30.0, 150.0, 300.0])[:, None])


def test_steering_sensitivity_scales_the_steering_levels_up_to_the_grip(circle_track):
    gains = np.array([1.0 / 3.0, 1.0, 3.0, 3.0])
    cars = sim.start_cars(circle_track, 4, setup=np.column_stack([np.full(4, config.SENSOR_START), gains]))
    cars.speed[:] = 10.0
    h0 = cars.heading.copy()
    sim.step(circle_track, cars, np.array([1.0, 1.0, 0.5, 1.0]), np.zeros(4), time_limit=100.0)
    a_lat = (cars.heading - h0) / config.DT * 10.0
    assert np.allclose(a_lat, sim.grip_at(10.0) * np.array([1.0 / 3.0, 1.0, 1.0, 1.0]))   # k × s, clipped at the grip


def test_simulate_passes_the_grip_and_the_setup(monkeypatch, circle_track):
    seen = []
    real = sim.step

    def recording(track, cars, steer, throttle, time_limit, grip_scale=1.0):
        seen.append((grip_scale, cars.ray_length.copy(), cars.steer_gain.copy()))
        real(track, cars, steer, throttle, time_limit, grip_scale=grip_scale)

    monkeypatch.setattr(sim, "step", recording)
    setup = np.array([[20.0, 2.0], [120.0, 0.5]])
    sim.simulate(circle_track, constant_policy(0.0, 1.0), 2, grip_scale=10.0, setup=setup)
    assert seen and all(g == 10.0 for g, _, _ in seen)
    assert np.array_equal(seen[0][1], [20.0, 120.0]) and np.array_equal(seen[0][2], [2.0, 0.5])


def test_start_cars_rejects_a_setup_of_the_wrong_shape(circle_track):
    with pytest.raises(ValueError, match="setup"):
        sim.start_cars(circle_track, 3, setup=default_setups(2))


# ----- the Formula 1 car and its sensors (spec §5.1, §5.2, version 3.8) -----

def test_the_grip_rises_with_speed_to_its_ceiling():
    """Spec §5.1: a(v) = min(a0 + c v², a_max), 1.8 g at rest, 5 g from 250 km/h, never more."""
    assert config.GRIP_LOW == 1.8 * 9.81 and config.GRIP_MAX == 5.0 * 9.81 and config.TOP_SPEED == 105.0
    assert np.isclose(config.GRIP_RISE, 0.00651, atol=5e-6)
    a = sim.grip_at(np.array([0.0, 250.0 / 3.6, 105.0, 30.0]))
    assert a[0] == config.GRIP_LOW and np.isclose(a[1], config.GRIP_MAX) and a[2] == config.GRIP_MAX
    assert a[3] == config.GRIP_LOW + config.GRIP_RISE * 30.0 * 30.0
    assert np.all(sim.grip_at(np.linspace(0.0, 200.0, 2001)) <= config.GRIP_MAX)


def test_a_standing_start_accelerates_at_the_low_speed_grip(circle_track):
    cars = sim.start_cars(circle_track, 1)
    sim.step(circle_track, cars, np.zeros(1), np.ones(1), time_limit=100.0)
    assert cars.speed[0] == config.GRIP_LOW * config.DT


def test_no_car_exceeds_the_top_speed():
    track = stadium(straight=1500.0, radius=300.0, width=14.0)
    cars = sim.start_cars(track, 2)
    cars.speed[:] = [104.9, 105.0]
    for _ in range(5):
        sim.step(track, cars, np.zeros(2), np.ones(2), time_limit=100.0)
        assert np.all(cars.speed <= config.TOP_SPEED)
    assert list(cars.speed) == [config.TOP_SPEED, config.TOP_SPEED]


def test_the_friction_circle_uses_the_grip_at_the_cars_speed(circle_track):
    cars = sim.start_cars(circle_track, 1)
    cars.speed[:] = 40.0
    h0 = cars.heading[0]
    sim.step(circle_track, cars, np.ones(1), np.ones(1), time_limit=100.0)
    a_long = (cars.speed[0] - 40.0) / config.DT
    a_lat = (cars.heading[0] - h0) / config.DT * 40.0
    assert np.isclose(np.hypot(a_long, a_lat), sim.grip_at(40.0))
    assert np.isclose(a_long, a_lat)                  # proportional scaling keeps the requested direction


def test_the_grip_scale_multiplies_the_grip(circle_track):
    normal, high = sim.start_cars(circle_track, 1), sim.start_cars(circle_track, 1)
    sim.step(circle_track, normal, np.zeros(1), np.ones(1), time_limit=100.0)
    sim.step(circle_track, high, np.zeros(1), np.ones(1), time_limit=100.0, grip_scale=10.0)
    assert normal.speed[0] == config.GRIP_LOW * config.DT and high.speed[0] == 10.0 * config.GRIP_LOW * config.DT


def test_every_car_starts_with_150_m_sensors_and_inputs_scaled_to_them():
    assert config.DEFAULT_SETUP == (config.SENSOR_START, 1.0) == (150.0, 1.0)
    assert config.SENSOR_LENGTH_LIMITS == (30.0, 1000.0) and config.SPEED_SCALE == config.TOP_SPEED
    assert list(sim.ray_caps([50.0, 150.0, 1000.0])) == [48, 144, 960]
    track = stadium()
    cars = sim.start_cars(track, 1)
    cars.speed[:] = 52.5
    obs = sim.sense(track, cars)
    assert obs[0, 4] == 1.0                           # 300 m of road ahead, the ray stops at 150 m
    assert np.all(obs[0, :config.N_RAYS] <= 1.0) and obs[0, config.N_RAYS] == 0.5


def test_a_1000_m_ray_reaches_its_length_on_a_long_straight():
    track = stadium(straight=2400.0, radius=200.0, width=14.0)
    x, y, heading = np.array([-1200.0]), np.array([-200.0]), np.zeros(1)   # start of the bottom straight, centered
    rays = sim.cast_rays(track, x, y, heading, np.array([1000.0]))
    assert rays[0, 4] == 1000.0 and np.all(rays <= 1000.0)


# ----- position across the road and road width (spec §5.2) -----

def placed(track, offset_left, index=0):
    """One car at centerline point `index` (lap-counting), `offset_left` m to the left of the driving direction."""
    cars = sim.start_cars(track, 1)
    i = index % len(track.centerline)
    tx, ty = track.tangent[i]
    cars.x[:] = track.centerline[i, 0] - ty * offset_left
    cars.y[:] = track.centerline[i, 1] + tx * offset_left
    cars.index[:] = index
    return cars


@pytest.mark.parametrize("reverse", [False, True])
def test_position_runs_from_the_left_edge_to_the_right_edge(circle_track, reverse):
    track = circle_track if not reverse else build_track(circle_track.centerline[::-1], circle_track.width, 0, 1)
    half = track.width / 2
    for offset, expected in ((0.0, 0.0), (half / 2, -0.5), (-half / 2, 0.5), (half * 0.999, -0.999)):
        cars = placed(track, offset)
        assert sim.road_position(track, cars, np.array([0])) == pytest.approx([expected], abs=1e-3)


def test_position_wraps_the_lap_counting_index(circle_track):
    m = len(circle_track.centerline)
    a = sim.road_position(circle_track, placed(circle_track, 2.0, index=7), np.array([0]))
    b = sim.road_position(circle_track, placed(circle_track, 2.0, index=7 + 2 * m), np.array([0]))
    assert a == pytest.approx(b) and a[0] == pytest.approx(-2.0 / (circle_track.width / 2), abs=1e-3)


def test_sense_adds_position_and_width_and_keeps_done_rows_zero(circle_track):
    cars = sim.start_cars(circle_track, 2)
    cars.done[1] = True
    obs = sim.sense(circle_track, cars)
    assert obs.shape == (2, 12) == (2, config.N_INPUTS)
    assert obs[0, config.N_RAYS + 1] == pytest.approx(0.0, abs=1e-6)
    assert obs[0, config.N_RAYS + 2] == pytest.approx(circle_track.width / config.WIDTH_SCALE)
    assert not obs[1].any()
