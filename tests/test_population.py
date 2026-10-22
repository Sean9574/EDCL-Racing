import numpy as np
import pytest

from racing import config, expert, nets, paths, population, sim
from racing import setup as car_setup


@pytest.fixture
def fake_tracks(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    return circle_track


def test_split_tasks_covers_every_individual_on_every_track():
    tasks = population.split_tasks(10, [5, 6, 7], 6)
    assert [(k, seed) for k, seed, _ in tasks] == [(0, 5), (0, 5), (1, 6), (1, 6), (2, 7), (2, 7)]
    for k in range(3):
        idx = np.concatenate([i for position, _, i in tasks if position == k])
        assert np.array_equal(np.sort(idx), np.arange(10))


def test_split_tasks_with_one_job_gives_one_task_per_track():
    tasks = population.split_tasks(10, [5, 6, 7], 1)
    assert [(k, seed, len(i)) for k, seed, i in tasks] == [(0, 5, 10), (1, 6, 10), (2, 7, 10)]


def test_split_tasks_never_makes_empty_chunks():
    assert all(len(i) > 0 for _, _, i in population.split_tasks(2, [0], 8))


def worker_pids(n_jobs=2):
    """Process ids of joblib's worker processes. Each task pauses, so every worker takes one."""
    from joblib import Parallel, delayed

    def pid_after_pause():
        import os
        import time
        time.sleep(0.3)
        return os.getpid()

    return set(Parallel(n_jobs=n_jobs, batch_size=1)(delayed(pid_after_pause)() for _ in range(n_jobs)))


def test_fresh_workers_start_new_processes_and_clear_the_track_cache():
    from racing import tracks
    cars = nets.DensePopulation(nets.init_population(np.random.default_rng(5), 4))
    population.evaluate_population(cars, [0], 1, config.TARGET_FACTOR, n_jobs=2)
    first = worker_pids()
    population.evaluate_population(cars, [0], 1, config.TARGET_FACTOR, n_jobs=2)
    assert len(first) == 2 and worker_pids() == first          # without the helper, the workers are reused
    tracks.get_track(0, 1)
    population.fresh_workers(2)
    assert tracks.get_track.cache_info().currsize == 0
    population.evaluate_population(cars, [0], 1, config.TARGET_FACTOR, n_jobs=2)
    second = worker_pids()
    assert len(second) == 2 and second.isdisjoint(first)


def test_fresh_workers_with_one_job_only_clears_the_track_cache(monkeypatch):
    from joblib.externals import loky
    from racing import tracks
    monkeypatch.setattr(loky, "get_reusable_executor", lambda **kwargs: pytest.fail("executor touched"))
    tracks.get_track(0, 1)
    population.fresh_workers(1)
    assert tracks.get_track.cache_info().currsize == 0


# ----- revision 3 time trials (spec §9.2, §9.5) -----

def dense(n, seed=0):
    return nets.DensePopulation(nets.init_population(np.random.default_rng(seed), n))


class ExpertCars:
    """n cars all driven by the expert; any object with len(), subset() and policy() can be evaluated."""

    def __init__(self, track, n):
        self.track, self.n = track, n
        self.setup = car_setup.default(n)                # revision 3's sensors and steering (spec §9.11)

    def __len__(self):
        return self.n

    def subset(self, indices):
        return ExpertCars(self.track, len(indices))

    def policy(self):
        return expert.expert_policy(self.track)


def test_time_trial_shapes(fake_tracks):
    result = population.evaluate_population(dense(6), [0, 1], 1, config.TARGET_FACTOR)
    assert result.fitness.shape == (6,)
    assert result.success.shape == result.laps.shape == result.finish_time.shape == (2, 6)
    assert result.success.dtype == bool and np.all((result.laps >= 0) & (result.laps <= config.LAPS))
    assert np.all(np.isnan(result.finish_time[~result.success]))
    assert np.all((result.fitness >= 0) & (result.fitness < 2))


def test_trial_result_properties():
    success = np.array([[True, True, False, False], [True, False, True, False]])
    laps = np.array([[3, 3, 2, 0], [3, 1, 3, 0]])
    r = population.TrialResult(fitness=np.zeros(4), success=success, laps=laps,
                               finish_time=np.where(success, 60.0, np.nan))
    assert list(r.all_success) == [True, False, False, False]
    assert np.allclose(r.lap_share, [1.0, 4 / 6, 5 / 6, 0.0])
    assert np.allclose(r.map_shares, [0.5, 0.5]) and r.min_share == 0.5
    assert np.allclose(r.lap_times[success], 60.0 / config.LAPS)


def test_time_trial_ends_at_the_target_time(monkeypatch, fake_tracks, numpy_simulator):
    target = config.TARGET_FACTOR * fake_tracks.expert_lap_time * config.LAPS
    seen = []

    def fake_simulate(track, policy, n, time_limit=None, laps=1, setup=None):
        seen.append((time_limit, laps))
        return sim.RunResult(fitness=np.array([1.5, 0.6, 0.2]), finished=np.array([True, False, False]),
                             crashed=np.array([False, False, True]), progress=np.array([1.0, 0.67, 0.2]),
                             finish_time=np.array([target - 1.0, np.nan, np.nan]), steps=10,
                             laps_done=np.array([3, 2, 0]), lap_times=np.full((3, 3), np.nan))

    monkeypatch.setattr(population.sim, "simulate", fake_simulate)
    result = population.evaluate_population(dense(3), [0], 1, config.TARGET_FACTOR)
    assert seen == [(pytest.approx(target), config.LAPS)]
    assert list(result.success[0]) == [True, False, False] and list(result.laps[0]) == [3, 2, 0]
    assert result.finish_time[0, 0] == target - 1.0 and np.isnan(result.finish_time[0, 1])


def test_expert_succeeds_at_the_target_and_counts_laps_when_too_slow(fake_tracks):
    cars = ExpertCars(fake_tracks, 2)
    easy = population.evaluate_population(cars, [0], 1, config.TARGET_FACTOR)
    assert easy.success.all() and np.all(easy.laps == config.LAPS) and np.all(easy.fitness > 1)
    tight = population.evaluate_population(cars, [0], 1, 0.4)                   # 24 s for three laps
    assert not tight.success.any() and np.all((tight.laps >= 1) & (tight.laps < config.LAPS))
    assert np.all(tight.fitness < 1)


def test_time_trials_do_not_depend_on_n_jobs(fake_tracks):
    cars = dense(9, seed=1)
    serial = population.evaluate_population(cars, [0, 1, 2], 1, config.TARGET_FACTOR)
    threaded = population.evaluate_population(cars, [0, 1, 2], 1, config.TARGET_FACTOR, n_jobs=6,
                                              backend="threading")
    for field in ("fitness", "success", "laps"):
        assert np.array_equal(getattr(serial, field), getattr(threaded, field))
    assert np.array_equal(serial.finish_time, threaded.finish_time, equal_nan=True)


def test_a_subset_reproduces_its_cars_results(fake_tracks):
    cars = dense(10, seed=2)
    full = population.evaluate_population(cars, [0], 1, config.TARGET_FACTOR)
    part = population.evaluate_population(cars.subset([1, 7]), [0], 1, config.TARGET_FACTOR)
    assert np.array_equal(full.fitness[[1, 7]], part.fitness) and np.array_equal(full.laps[:, [1, 7]], part.laps)


def test_select_final_prefers_successes_then_fitness(monkeypatch):
    def fake(cars, seeds, level, factor, n_jobs=1):
        assert len(cars) == 4 and seeds == list(config.VAL_SEEDS) and level == config.VALIDATION_LEVEL and factor == 1.1
        success = np.zeros((len(seeds), 4), dtype=bool)
        success[0, :3] = True
        success[1, 1:3] = True
        return population.TrialResult(fitness=np.array([1.9, 1.2, 1.5, 0.3]), success=success,
                                      laps=np.zeros((len(seeds), 4), dtype=np.int64),
                                      finish_time=np.full((len(seeds), 4), np.nan))

    monkeypatch.setattr(population, "evaluate_population", fake)
    assert population.select_final(dense(6), [5, 0, 3, 2], 1.1) == (3, 2, 1.5)


def test_dense_champion_keeps_its_architecture(tmp_path):
    weights = nets.init_population(np.random.default_rng(3), 2, (8, 4))
    cars = nets.DensePopulation(weights, (8, 4), "tanh")
    path = population.save_dense_champion("baseline", 1, cars, 1, tmp_path)
    assert path == tmp_path / "baseline" / "seed1.keras" and path.exists()
    saved = np.load(tmp_path / "baseline" / "seed1.npy")
    assert saved.dtype == np.float64 and np.array_equal(saved, weights[1])
    _, driver = nets.load_champion("baseline", 1, tmp_path)
    obs = np.random.default_rng(4).random((50, config.N_INPUTS))
    assert np.array_equal(driver.act(obs), cars.driver(1).act(obs))


def test_clear_champion_removes_every_file(tmp_path):
    population.save_dense_champion("es", 0, dense(1, seed=4), 0, tmp_path)
    assert sorted(p.name for p in (tmp_path / "es").iterdir()) == ["seed0.keras", "seed0.npy", "seed0.setup.json"]
    (tmp_path / "es" / "seed0.json").write_text("{}")
    population.clear_champion("es", 0, tmp_path)
    assert list((tmp_path / "es").iterdir()) == []


def test_save_dense_champion_keeps_the_architecture(tmp_path):
    weights = nets.init_population(np.random.default_rng(3), 3, (8, 4))
    path = population.save_dense_champion("es", 2, nets.DensePopulation(weights, (8, 4), "tanh"), 1, tmp_path)
    assert path == tmp_path / "es" / "seed2.keras" and path.exists()
    assert not (tmp_path / "es" / "seed2.json").exists()                # a .json marks a NEAT champion
    saved = np.load(tmp_path / "es" / "seed2.npy")
    assert saved.dtype == np.float64 and np.array_equal(saved, weights[1])
    model, driver = nets.load_champion("es", 2, tmp_path)
    assert nets.dense_spec(model) == ((8, 4), "tanh") and driver.n_params == weights.shape[1]


def test_compiled_time_trials_use_the_thread_count_not_the_worker_count(monkeypatch, fake_tracks):
    from racing import fastsim
    if not fastsim.available():
        pytest.skip("the compiled simulator does not run here")
    seen = []
    real = fastsim.simulate
    monkeypatch.setattr(fastsim, "simulate", lambda *a, **k: seen.append(k["threads"]) or real(*a, **k))
    monkeypatch.setattr(config, "thread_count", lambda: 3)
    population.evaluate_population(dense(4), [0, 1], 1, config.TARGET_FACTOR, n_jobs=7)
    assert seen == [3]


def test_a_shaped_champion_saves_and_drives_like_its_car(tmp_path, shaped_cars):
    cars = shaped_cars(6)
    obs = np.random.default_rng(8).random((4, config.N_INPUTS))
    for index in (0, 2):                                  # architectures () and (8, 3)
        population.save_dense_champion("ga", index, cars, index, tmp_path)
        model, driver = nets.load_champion("ga", index, tmp_path)
        assert np.array_equal(np.load(paths.champion_path("ga", index, ".npy", tmp_path)), cars.weights[index])
        assert np.array_equal(driver.act(obs), cars.driver(index).act(obs))
        assert np.array_equal(driver.setup, cars.setup[index])
        assert model.count_params() == nets.n_params(cars.shapes[index])
