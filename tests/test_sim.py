import numpy as np
import pytest

from racing import config, expert, sim
from racing.actions import class_to_controls, controls_to_class
from racing.track import build_track, wall_distance


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
    assert np.hypot(a_long, a_lat) <= config.GRIP + 1e-9
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
    steer = np.array([-(10.0 ** 2 / 50.0) / config.GRIP])     # clockwise turn that follows the circle backwards
    while not cars.done[0]:
        sim.step(circle_track, cars, steer, np.array([0.0]), time_limit=100.0)
    assert cars.progress[0] == 0.0 and cars.laps_done[0] == 0
    assert not cars.crashed[0]                               # stopped by the stall timeout, still on the road


def test_full_lap_is_detected(circle_track):
    cars = sim.start_cars(circle_track, 1)
    cars.speed[:] = 15.0
    steer = np.array([(15.0 ** 2 / 50.0) / config.GRIP])
    while not cars.done[0]:
        sim.step(circle_track, cars, steer, np.array([0.0]), time_limit=100.0)
    assert cars.finished[0] and not cars.crashed[0]
    assert abs(cars.finish_time[0] - circle_track.length / 15.0) < 0.5
    assert np.isclose(cars.last_lap[0], cars.finish_time[0])


def test_multi_lap_race_counts_laps(circle_track):
    cars = sim.start_cars(circle_track, 1, laps=2)
    cars.speed[:] = 15.0
    steer = np.array([(15.0 ** 2 / 50.0) / config.GRIP])
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


# ----- revision 3.1: per-car sensor length and steering sensitivity, grip as a parameter -----

def stadium(straight=300.0, radius=60.0, width=12.0):
    """Two straights joined by semicircles, counter-clockwise from the start of the bottom straight."""
    s = np.arange(0.0, straight, 0.5)
    a = np.arange(-np.pi / 2, np.pi / 2, 0.5 / radius)
    points = np.vstack([np.column_stack([s - straight / 2, np.full(len(s), -radius)]),
                        np.column_stack([straight / 2 + radius * np.cos(a), radius * np.sin(a)]),
                        np.column_stack([straight / 2 - s, np.full(len(s), radius)]),
                        np.column_stack([-straight / 2 - radius * np.cos(a), -radius * np.sin(a)])])
    track = build_track(points, width, seed=0, level=4)
    track.expert_lap_time = 60.0
    return track


def reference_cast_rays(track, x, y, heading):
    """sim.cast_rays of revision 3, verbatim (every ray RAY_RANGE long, at most RAY_MAX_ITERATIONS iterations), plus
    which rays stopped at the iteration cap."""
    n = len(x)
    angles = (np.asarray(heading)[:, None] + config.RAY_ANGLES[None, :]).ravel()
    dx, dy = np.cos(angles), np.sin(angles)
    ox, oy = np.repeat(x, config.N_RAYS), np.repeat(y, config.N_RAYS)
    t = np.zeros(n * config.N_RAYS)
    active = np.arange(n * config.N_RAYS)
    for _ in range(config.RAY_MAX_ITERATIONS):
        if active.size == 0:
            break
        d = wall_distance(track, ox[active] + dx[active] * t[active], oy[active] + dy[active] * t[active])
        hit = d <= 0.0
        t[active] = np.where(hit, t[active], t[active] + np.maximum(d, config.RAY_MIN_STEP))
        active = active[~hit & (t[active] < config.RAY_RANGE)]
    capped = np.zeros(n * config.N_RAYS, dtype=bool)
    capped[active] = True
    return np.minimum(t, config.RAY_RANGE).reshape(n, config.N_RAYS), capped.reshape(n, config.N_RAYS)


def reference_sense(track, cars):
    """sim.sense of revision 3, verbatim."""
    obs = np.zeros((len(cars.x), config.N_INPUTS))
    live = np.flatnonzero(~cars.done)
    if live.size:
        rays, _ = reference_cast_rays(track, cars.x[live], cars.y[live], cars.heading[live])
        obs[live, :config.N_RAYS] = rays / config.RAY_RANGE
        obs[live, config.N_RAYS] = cars.speed[live] / config.SPEED_SCALE
    return obs


def reference_step(track, cars, steer, throttle, time_limit):
    """sim.step of revision 3, verbatim."""
    live = ~cars.done
    a_lat = np.clip(steer, -1.0, 1.0) * config.GRIP
    a_long = np.clip(throttle, -1.0, 1.0) * config.GRIP
    total = np.hypot(a_lat, a_long)
    scale = np.where(total > config.GRIP, config.GRIP / np.maximum(total, 1e-12), 1.0)
    a_lat, a_long = a_lat * scale, a_long * scale
    yaw_rate = a_lat / np.maximum(cars.speed, config.YAW_MIN_SPEED)
    speed = np.maximum(cars.speed + a_long * config.DT, 0.0)
    heading = cars.heading + yaw_rate * config.DT
    cars.x = np.where(live, cars.x + speed * np.cos(heading) * config.DT, cars.x)
    cars.y = np.where(live, cars.y + speed * np.sin(heading) * config.DT, cars.y)
    cars.speed = np.where(live, speed, cars.speed)
    cars.heading = np.where(live, heading, cars.heading)
    cars.time += config.DT
    crashed = live & (wall_distance(track, cars.x, cars.y) <= 0.0)
    cars.crashed |= crashed
    cars.speed[crashed] = 0.0
    moving = live & ~crashed
    sim._update_progress(track, cars, moving)
    stalled = moving & (cars.time - cars.last_progress_time >= config.STALL_TIME - config.TIME_EPS)
    out_of_time = live & (cars.time >= time_limit - config.TIME_EPS)
    cars.done |= crashed | stalled | out_of_time | cars.finished


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


def test_the_default_setup_is_revision_3s():
    assert config.DEFAULT_SETUP == (config.RAY_RANGE, 1.0) == (50.0, 1.0)


def test_a_ray_of_revision_3s_length_keeps_revision_3s_iteration_cap():
    assert list(sim.ray_caps([10.0, 50.0, 150.0])) == [10, config.RAY_MAX_ITERATIONS, 3 * config.RAY_MAX_ITERATIONS]


def test_default_rays_are_traced_exactly_as_in_revision_3(circle_track):
    track = stadium()
    x, y, heading = road_poses(track, 2000, seed=3)
    x = np.append(x, -100.0)                   # 0.3 m from the inner edge of the bottom straight, looking along it
    y = np.append(y, -54.3)
    heading = np.append(heading, 0.0)
    expected, capped = reference_cast_rays(track, x, y, heading)
    assert capped.any()                        # the iteration cap binds for some rays (it does for the last pose)
    assert np.array_equal(sim.cast_rays(track, x, y, heading), expected)
    assert np.array_equal(sim.cast_rays(track, x, y, heading, np.full(len(x), config.RAY_RANGE)), expected)
    cx, cy, ch = road_poses(circle_track, 500, seed=4)
    assert np.array_equal(sim.cast_rays(circle_track, cx, cy, ch), reference_cast_rays(circle_track, cx, cy, ch)[0])


def test_a_default_setup_reproduces_a_revision_3_drive():
    track = stadium()
    n = 40
    new = sim.start_cars(track, n, laps=3, setup=default_setups(n))
    old = sim.start_cars(track, n, laps=3)
    rng_new, rng_old = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(900):
        obs_new, obs_old = sim.sense(track, new), reference_sense(track, old)
        assert np.array_equal(obs_new[:, :config.N_RAYS + 1], obs_old[:, :config.N_RAYS + 1])   # revision 3's columns
        actions = []
        for cars, rng in ((new, rng_new), (old, rng_old)):
            classes = controls_to_class(*expert.expert_controls(track, cars))
            flip = rng.random(n) < 0.1                       # disturbances, so cars also crash and graze walls
            actions.append(np.where(flip, rng.integers(0, config.N_ACTIONS, n), classes))
        assert np.array_equal(actions[0], actions[1])
        steer, throttle = class_to_controls(actions[0])
        sim.step(track, new, steer, throttle, time_limit=200.0)
        reference_step(track, old, steer, throttle, time_limit=200.0)
    for name in ("x", "y", "heading", "speed", "progress", "laps_done", "crashed", "finished", "done"):
        assert np.array_equal(getattr(new, name), getattr(old, name)), name
    assert np.array_equal(new.finish_time, old.finish_time, equal_nan=True)
    assert old.crashed.any() and (old.progress > 300.0).any()            # crashes and long drives both covered


def test_simulate_with_the_default_setup_equals_simulate_without_one(circle_track):
    policy = expert.expert_policy(circle_track)
    plain = sim.simulate(circle_track, policy, 3, laps=2)
    explicit = sim.simulate(circle_track, policy, 3, laps=2, setup=default_setups(3))
    for name in ("fitness", "finished", "crashed", "progress", "laps_done"):
        assert np.array_equal(getattr(plain, name), getattr(explicit, name))
    assert np.array_equal(plain.finish_time, explicit.finish_time, equal_nan=True)


def test_rays_reach_their_cars_sensor_length():
    track = stadium()
    x, y, heading = np.full(3, -150.0), np.full(3, -60.0), np.zeros(3)     # start of the bottom straight, centered
    rays = sim.cast_rays(track, x, y, heading, np.array([10.0, 50.0, 150.0]))
    assert list(rays[:, 4]) == [10.0, 50.0, 150.0]                           # straight ahead: 300 m of road
    assert np.all(rays[0] <= 10.0) and np.all(rays <= np.array([10.0, 50.0, 150.0])[:, None])


def test_long_rays_trace_past_revision_3s_cap():
    track = stadium()
    x, y, heading = np.array([-100.0, -100.0]), np.array([-54.3, -54.3]), np.zeros(2)   # grazing the inner edge
    rays = sim.cast_rays(track, x, y, heading, np.array([50.0, 150.0]))
    _, capped = reference_cast_rays(track, x[:1], y[:1], heading[:1])
    assert capped[0, 4]                                         # revision 3 stops this grazing ray at its cap
    assert rays[1, 4] > rays[0, 4] + 40.0                       # a 150 m sensor gets three times the iterations


def test_sense_scales_rays_by_revision_3s_length():
    track = stadium()
    cars = sim.start_cars(track, 2, setup=np.array([[150.0, 1.0], [10.0, 1.0]]))
    obs = sim.sense(track, cars)
    assert obs[0, 4] == 150.0 / config.RAY_RANGE == 3.0 and obs[1, 4] == 0.2
    assert np.all(obs[:, :config.N_RAYS] <= np.array([[3.0], [0.2]]))


def test_steering_sensitivity_scales_the_steering_levels_up_to_the_grip(circle_track):
    gains = np.array([1.0 / 3.0, 1.0, 3.0, 3.0])
    cars = sim.start_cars(circle_track, 4, setup=np.column_stack([np.full(4, 50.0), gains]))
    cars.speed[:] = 10.0
    h0 = cars.heading.copy()
    sim.step(circle_track, cars, np.array([1.0, 1.0, 0.5, 1.0]), np.zeros(4), time_limit=100.0)
    a_lat = (cars.heading - h0) / config.DT * 10.0
    assert np.allclose(a_lat, config.GRIP * np.array([1.0 / 3.0, 1.0, 1.0, 1.0]))   # k × s, clipped at the grip


def test_grip_is_a_parameter_of_the_step(circle_track):
    normal, high = sim.start_cars(circle_track, 1), sim.start_cars(circle_track, 1)
    sim.step(circle_track, normal, np.zeros(1), np.ones(1), time_limit=100.0)
    sim.step(circle_track, high, np.zeros(1), np.ones(1), time_limit=100.0, grip=10 * config.GRIP)
    assert np.isclose(normal.speed[0], config.GRIP * config.DT)
    assert np.isclose(high.speed[0], 10 * config.GRIP * config.DT)
    turning = sim.start_cars(circle_track, 1)
    turning.speed[:] = 10.0
    h0 = turning.heading[0]
    sim.step(circle_track, turning, np.ones(1), np.ones(1), time_limit=100.0, grip=10 * config.GRIP)
    a_long = (turning.speed[0] - 10.0) / config.DT
    a_lat = (turning.heading[0] - h0) / config.DT * 10.0
    assert np.isclose(np.hypot(a_long, a_lat), 10 * config.GRIP)          # the friction circle at the given grip


def test_simulate_passes_the_grip_and_the_setup(monkeypatch, circle_track):
    seen = []
    real = sim.step

    def recording(track, cars, steer, throttle, time_limit, grip=config.GRIP):
        seen.append((grip, cars.ray_length.copy(), cars.steer_gain.copy()))
        real(track, cars, steer, throttle, time_limit, grip=grip)

    monkeypatch.setattr(sim, "step", recording)
    setup = np.array([[20.0, 2.0], [120.0, 0.5]])
    sim.simulate(circle_track, constant_policy(0.0, 1.0), 2, grip=7.0, setup=setup)
    assert seen and all(g == 7.0 for g, _, _ in seen)
    assert np.array_equal(seen[0][1], [20.0, 120.0]) and np.array_equal(seen[0][2], [2.0, 0.5])


def test_start_cars_rejects_a_setup_of_the_wrong_shape(circle_track):
    with pytest.raises(ValueError, match="setup"):
        sim.start_cars(circle_track, 3, setup=default_setups(2))


# ----- position across the road and road width (revision 3.2 §3) -----

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
