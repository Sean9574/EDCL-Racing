import numpy as np

from racing import config, sim
from racing.actions import class_to_controls, controls_to_class


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
