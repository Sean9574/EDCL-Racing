import json

import numpy as np
import pytest

from racing import config, setup


def test_setup_parameters():
    assert config.SENSOR_LENGTH_LIMITS == (30.0, 1000.0) and config.STEER_GAIN_LIMITS == (1.0 / 3.0, 3.0)
    assert (config.SETUP_MUTATION_P, config.SETUP_MUTATION_SD) == (0.2, 0.1)
    assert setup.COLUMNS == ("sensor_length", "steer_gain")


def test_default_rows_are_the_default_setup():
    rows = setup.default(4)
    assert rows.shape == (4, 2) and rows.dtype == np.float64
    assert np.all(rows == np.asarray(config.DEFAULT_SETUP)) and np.all(rows == [config.SENSOR_START, 1.0])
    assert np.array_equal(setup.as_row(None), config.DEFAULT_SETUP)


def test_as_rows_copies_and_checks_the_shape():
    values = np.array([[20.0, 2.0], [30.0, 0.5]])
    rows = setup.as_rows(values, 2)
    values[0, 0] = 99.0
    assert rows[0, 0] == 20.0                                    # a copy, not a view
    assert np.array_equal(setup.as_rows(None, 3), setup.default(3))
    with pytest.raises(ValueError, match="setup"):
        setup.as_rows(values, 3)


def test_random_setups_start_at_the_starting_sensor_length_with_log_uniform_steering():
    """Spec §9.11 (version 3.8): every car of a random-start population starts with 150 m sensors; its steering
    sensitivity is log-uniform within its limits."""
    rows = setup.random(np.random.default_rng(0), 20000)
    assert rows.shape == (20000, 2) and np.all(rows[:, 0] == config.SENSOR_START)
    low, high = config.STEER_GAIN_LIMITS
    values = rows[:, 1]
    assert values.min() >= low and values.max() <= high
    assert abs(np.mean(values < np.sqrt(low * high)) - 0.5) < 0.02          # median at the geometric mean
    assert abs(np.mean(values < low * (high / low) ** 0.25) - 0.25) < 0.02  # a quarter in the first log quarter


def test_random_makes_one_uniform_draw():
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    setup.random(a, 5)
    b.uniform(size=5)
    assert a.bit_generator.state == b.bit_generator.state


def test_clip_keeps_values_within_the_limits():
    clipped = setup.clip(np.array([[1.0, 0.1], [5000.0, 9.0], [60.0, 1.5]]))
    low = [config.SENSOR_LENGTH_LIMITS[0], config.STEER_GAIN_LIMITS[0]]
    high = [config.SENSOR_LENGTH_LIMITS[1], config.STEER_GAIN_LIMITS[1]]
    assert np.array_equal(clipped, [low, high, [60.0, 1.5]])


def test_crossover_takes_each_value_from_either_parent():
    a, b = np.full((5000, 2), 20.0), np.full((5000, 2), 80.0)
    child = setup.crossover(np.random.default_rng(2), a, b)
    assert set(np.unique(child)) == {20.0, 80.0}
    assert abs(np.mean(child == 20.0) - 0.5) < 0.02
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    setup.crossover(r1, a[:4], b[:4])
    r2.random((4, 2))
    assert r1.bit_generator.state == r2.bit_generator.state                     # one draw


def test_mutation_changes_a_fifth_of_the_values_by_about_ten_percent():
    rows = np.tile([50.0, 1.0], (20000, 1))
    mutated = setup.mutate(np.random.default_rng(4), rows)
    changed = mutated != rows
    assert abs(changed.mean() - config.SETUP_MUTATION_P) < 0.01
    ratios = np.log(mutated[changed] / rows[changed])
    assert abs(ratios.std() - config.SETUP_MUTATION_SD) < 0.005 and abs(ratios.mean()) < 0.005


def test_mutation_stays_within_the_limits_and_always_draws_two_arrays():
    edge = np.tile([config.SENSOR_LENGTH_LIMITS[1], config.STEER_GAIN_LIMITS[0]], (1000, 1))
    mutated = setup.mutate(np.random.default_rng(5), edge)
    assert np.all(mutated[:, 0] <= config.SENSOR_LENGTH_LIMITS[1])
    assert np.all(mutated[:, 1] >= config.STEER_GAIN_LIMITS[0])
    r1, r2 = np.random.default_rng(6), np.random.default_rng(6)
    setup.mutate(r1, edge[:3])
    r2.random((3, 2))
    r2.normal(0.0, config.SETUP_MUTATION_SD, (3, 2))
    assert r1.bit_generator.state == r2.bit_generator.state


def test_without_a_boost_the_mutation_is_bit_identical_to_the_old_draw():
    """Review focus 1: the old code drew rng.normal(0, SETUP_MUTATION_SD); the new draws standard normals and scales
    them per column. numpy computes normal(0, sd) as 0 + sd * z, so both give the same floats and the same draws."""
    for seed in range(20):
        z_old = np.random.default_rng(seed).normal(0.0, config.SETUP_MUTATION_SD, 100_000)
        z_new = np.random.default_rng(seed).normal(0.0, 1.0, 100_000) * config.SETUP_MUTATION_SD
        assert np.array_equal(z_old, z_new)
        rows = setup.random(np.random.default_rng(seed + 100), 5000)
        old_rng, new_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        mask = old_rng.random(rows.shape) < config.SETUP_MUTATION_P               # revision 3.1's mutation
        old = setup.clip(np.where(mask, rows * np.exp(old_rng.normal(0.0, config.SETUP_MUTATION_SD, rows.shape)),
                                  rows))
        assert np.array_equal(setup.mutate(new_rng, rows), old)
        assert new_rng.bit_generator.state == old_rng.bit_generator.state


def test_a_boost_mutates_only_the_steering_sensitivity_harder_with_the_same_draws():
    """Spec §9.4: in a stall the steering sensitivity mutates with probability 0.5 and standard deviation 0.2 on the
    log scale; the sensor length as usual, from the same draws."""
    values = np.tile([50.0, 1.0], (20000, 1))
    plain_rng, boosted_rng = np.random.default_rng(3), np.random.default_rng(3)
    plain = setup.mutate(plain_rng, values)
    boosted = setup.mutate(boosted_rng, values, boost=True)
    assert plain_rng.bit_generator.state == boosted_rng.bit_generator.state     # the same draws
    assert np.array_equal(plain[:, 0], boosted[:, 0])                           # the sensor length as usual
    changed = boosted[:, 1] != 1.0
    assert abs(changed.mean() - config.STALL_STEER_P) < 0.01
    assert abs(np.std(np.log(boosted[changed, 1])) - config.STALL_STEER_SD) < 0.005
    assert np.all((boosted[:, 1] >= config.STEER_GAIN_LIMITS[0]) & (boosted[:, 1] <= config.STEER_GAIN_LIMITS[1]))


def test_a_setup_round_trips_through_its_json_form():
    row = np.array([73.5, 0.62])
    d = setup.to_dict(row)
    assert d == {"sensor_length": 73.5, "steer_gain": 0.62} and json.loads(json.dumps(d)) == d
    assert np.array_equal(setup.from_dict(d), row)


def test_summary_reports_means_and_spreads():
    assert setup.summary(np.array([[40.0, 1.0], [60.0, 2.0]])) == {
        "sensor_mean": 50.0, "sensor_std": 10.0, "steer_mean": 1.5, "steer_std": 0.5}


# ----- setups travel with the cars (spec §6, §9.11, §12.1) -----

from racing import dagger, evaluate, nets, neat, population, sim  # noqa: E402

ROWS = np.array([[20.0, 2.0], [120.0, 0.5], [50.0, 1.0], [80.0, 3.0]])


def dense_with_setup():
    return nets.DensePopulation(nets.init_population(np.random.default_rng(0), 4), setup=ROWS)


def neat_genomes(n, seed=1):
    rng, innovations = np.random.default_rng(seed), neat.Innovations()
    return [neat.initial_genome(rng, innovations) for _ in range(n)]


def recording_simulate(monkeypatch, seen):
    """Wrap sim.simulate (one module object, seen by every caller) to record each call's setup."""
    real = sim.simulate

    def recording(track, policy, n, time_limit=None, laps=1, setup=None, grip_scale=1.0):
        seen.append(None if setup is None else np.array(setup))
        return real(track, policy, n, time_limit=time_limit, laps=laps, setup=setup, grip_scale=grip_scale)

    monkeypatch.setattr(sim, "simulate", recording)


def test_a_dense_population_carries_its_setup_through_subset_and_driver():
    rows = ROWS.copy()
    cars = nets.DensePopulation(nets.init_population(np.random.default_rng(0), 4), setup=rows)
    rows[0, 0] = 99.0                                            # the population keeps its own copy
    assert np.array_equal(cars.setup, ROWS)
    assert np.array_equal(cars.subset([3, 1]).setup, ROWS[[3, 1]])
    assert np.array_equal(cars.driver(1).setup, ROWS[1])
    assert np.array_equal(nets.DensePopulation(cars.weights).setup, setup.default(4))
    assert np.array_equal(nets.from_flat(cars.weights[0]).setup, config.DEFAULT_SETUP)


def test_a_neat_population_carries_its_setup_through_subset_and_driver():
    genomes = neat_genomes(4)
    cars = neat.NeatPopulation(genomes, ROWS)
    assert np.array_equal(cars.subset([2, 0]).setup, ROWS[[2, 0]])
    assert np.array_equal(cars.driver(3).setup, ROWS[3])
    assert np.array_equal(neat.NeatPopulation(genomes).setup, setup.default(4))
    assert np.array_equal(neat.NeatNet(genomes[0]).setup, config.DEFAULT_SETUP)


def test_drivers_setup_lists_each_drivers_setup_in_order():
    cars = dense_with_setup()
    drivers = [cars.driver(i) for i in (2, 0, 3)] + [neat.NeatPopulation(neat_genomes(1), ROWS[1:2]).driver(0)]
    assert np.array_equal(nets.drivers_setup(drivers), ROWS[[2, 0, 3, 1]])


def test_parallel_evaluation_gives_every_task_its_cars_setups(monkeypatch, circle_track, numpy_simulator):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    rows = setup.random(np.random.default_rng(3), 9)
    cars = nets.DensePopulation(nets.init_population(np.random.default_rng(2), 9), setup=rows)
    seen = []
    recording_simulate(monkeypatch, seen)
    serial = population.evaluate_population(cars, [0, 1, 2], 1, config.TARGET_FACTOR)
    threaded = population.evaluate_population(cars, [0, 1, 2], 1, config.TARGET_FACTOR, n_jobs=6,
                                              backend="threading")
    for field in ("fitness", "success", "laps"):
        assert np.array_equal(getattr(serial, field), getattr(threaded, field))
    assert len(seen) > 6 and all(s is not None for s in seen)    # the threaded run splits the population
    stacked = np.vstack(seen)
    for row in rows:                                              # each car's row once per map and evaluation
        assert np.sum(np.all(stacked == row, axis=1)) == 2 * 3


def test_final_selection_drives_the_candidates_with_their_setups(monkeypatch, circle_track, numpy_simulator):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    cars = dense_with_setup()
    seen = []
    recording_simulate(monkeypatch, seen)
    population.select_final(cars, [3, 1], config.TARGET_FACTOR, seeds=[80])
    assert len(seen) == 1 and np.array_equal(seen[0], ROWS[[3, 1]])


def test_a_dense_champion_round_trips_its_setup(tmp_path):
    population.save_dense_champion("ga", 0, dense_with_setup(), 3, tmp_path)
    assert json.loads((tmp_path / "ga" / "seed0.setup.json").read_text()) == setup.to_dict(ROWS[3])
    _, driver = nets.load_champion("ga", 0, tmp_path)
    assert np.array_equal(driver.setup, ROWS[3])


def test_a_neat_champion_round_trips_its_setup(tmp_path):
    genome = neat_genomes(1)[0]
    neat.save_neat_champion("neat", 1, genome, tmp_path, setup=ROWS[1])
    _, driver = nets.load_champion("neat", 1, tmp_path)
    assert isinstance(driver, neat.NeatNet) and np.array_equal(driver.setup, ROWS[1])
    neat.save_neat_champion("neat", 2, genome, tmp_path)                       # no setup given: revision 3's
    assert json.loads((tmp_path / "neat" / "seed2.setup.json").read_text()) == setup.to_dict(config.DEFAULT_SETUP)


def test_a_controller_without_a_setup_file_drives_with_revision_3s_setup(tmp_path):
    (tmp_path / "baseline").mkdir()
    nets.to_keras(nets.init_population(np.random.default_rng(5), 1)[0]).save(tmp_path / "baseline" / "seed0.keras")
    _, driver = nets.load_champion("baseline", 0, tmp_path)
    assert np.array_equal(driver.setup, config.DEFAULT_SETUP)


def test_evaluation_drives_each_controller_with_its_setup(monkeypatch, circle_track):
    monkeypatch.setattr(evaluate, "get_track", lambda seed, level: circle_track)
    cars = dense_with_setup()
    drivers = {("ga", 0): cars.driver(3), ("es", 1): cars.driver(0)}
    seen = []
    recording_simulate(monkeypatch, seen)
    evaluate.driving_results(drivers, [0], level=1)
    assert len(seen) == 1 and np.array_equal(seen[0], ROWS[[3, 0]])


def test_dagger_drives_and_checks_with_the_drivers_setup(monkeypatch, circle_track):
    monkeypatch.setattr(dagger, "get_track", lambda seed, level: circle_track)
    driver = dense_with_setup().driver(1)
    started = []
    real_start = sim.start_cars

    def recording_start(track, n, laps=1, setup=None):
        started.append(np.array(setup))
        return real_start(track, n, laps, setup)

    monkeypatch.setattr(sim, "start_cars", recording_start)
    dagger.drive_and_label(driver, 0, level=1)
    assert np.array_equal(started[0], ROWS[1:2])
    seen = []
    recording_simulate(monkeypatch, seen)
    dagger.check(driver, [0], level=1)
    assert len(seen) == 1 and np.array_equal(seen[0], ROWS[1:2])
