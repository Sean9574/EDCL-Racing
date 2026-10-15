"""The compiled simulator (spec §5.7, §12.1): its build and fallback, its network programs, and exact equality with the
numpy simulator."""
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from racing import config, fastsim

needs_compiler = pytest.mark.skipif(shutil.which(fastsim.COMPILER) is None, reason="no g++ on this machine")


@pytest.fixture(autouse=True)
def compiled_simulator(monkeypatch):
    """Every test here compares the compiled simulator with numpy, whatever EC_RACING_SIM says."""
    monkeypatch.setattr(config, "SIMULATOR", "auto")


@pytest.fixture
def private_library(monkeypatch, tmp_path):
    """The source and library in a temporary folder, with nothing loaded or failed yet in this process."""
    source = tmp_path / "fastsim.cpp"
    shutil.copy(fastsim.SOURCE, source)
    monkeypatch.setattr(fastsim, "SOURCE", source)
    monkeypatch.setattr(fastsim, "LIBRARY", tmp_path / "_fastsim.so")
    monkeypatch.setattr(fastsim, "_library", None)
    monkeypatch.setattr(fastsim, "_failure", None)
    return tmp_path


def test_numpy_choice_disables_the_compiled_simulator(monkeypatch):
    monkeypatch.setattr(config, "SIMULATOR", "numpy")
    assert not fastsim.available()
    assert fastsim.description() == "numpy"


@needs_compiler
def test_the_library_builds_and_describes_its_compiler(private_library):
    assert fastsim.available()
    assert fastsim.LIBRARY.exists()
    assert fastsim.description().startswith("compiled (g++ ")


def test_a_failed_build_warns_once_and_falls_back(private_library, monkeypatch, capsys):
    monkeypatch.setattr(fastsim, "COMPILER", "no-such-compiler-ec-racing")
    assert not fastsim.available()
    assert not fastsim.available()
    err = capsys.readouterr().err
    assert err.count("using the numpy simulator") == 1
    assert fastsim.description() == "numpy"


@needs_compiler
def test_a_stale_library_is_rebuilt(private_library):
    fastsim.build()
    source_time = fastsim.SOURCE.stat().st_mtime
    os.utime(fastsim.LIBRARY, (source_time - 60, source_time - 60))   # older than the source, as after a git pull
    fastsim.build()
    rebuilt = fastsim.LIBRARY.stat().st_mtime_ns
    assert rebuilt >= fastsim.SOURCE.stat().st_mtime_ns
    fastsim.build()                                             # up to date now: no rebuild
    assert fastsim.LIBRARY.stat().st_mtime_ns == rebuilt


@needs_compiler
def test_an_unloadable_library_is_rebuilt(private_library):
    fastsim.LIBRARY.write_bytes(b"not a library")
    later = time.time() + 5
    os.utime(fastsim.LIBRARY, (later, later))                   # newer than the source, so only loading reveals it
    assert fastsim.available()


@needs_compiler
def test_concurrent_builds_all_load(private_library):
    """Four processes that start at once each load a complete library (spec §5.7)."""
    code = (f"import sys; sys.path.insert(0, {str(config.PROJECT_DIR)!r}); from pathlib import Path; "
            f"from racing import fastsim; fastsim.SOURCE = Path({str(fastsim.SOURCE)!r}); "
            f"fastsim.LIBRARY = Path({str(fastsim.LIBRARY)!r}); fastsim.library(); print('ok')")
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(4)]
    outputs = [p.communicate(timeout=300) for p in procs]
    assert all(p.returncode == 0 and out.strip() == "ok" for p, (out, _) in zip(procs, outputs)), outputs
    assert not list(private_library.glob("*.tmp"))


from racing import nets, sim
from racing import setup as car_setup
from racing.tracks import get_track


def same_results(a, b):
    """Every RunResult field equal; NaN finish times in the same places (spec §5.7)."""
    for field in ("fitness", "finished", "crashed", "progress", "laps_done"):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field
    assert np.array_equal(a.finish_time, b.finish_time, equal_nan=True)
    assert a.steps == b.steps


def numpy_results(tracks, population, limits, laps, grip=config.GRIP):
    return [sim.simulate(t, population.policy(), len(population), time_limit=limit, laps=laps,
                         setup=population.setup, grip=grip) for t, limit in zip(tracks, limits)]


def dense_cars(n, hidden=config.BASELINE_HIDDEN, activation="relu", n_inputs=config.N_INPUTS, seed=0):
    """Random dense networks with random biases and random setups, so cars crash at many different points."""
    rng = np.random.default_rng(seed)
    weights = nets.init_population(rng, n, hidden, n_inputs)
    weights = weights + rng.normal(0.0, 0.3, weights.shape)
    return nets.DensePopulation(weights, hidden, activation, car_setup.random(rng, n))


ARCHITECTURES = [((16,), "relu", config.N_INPUTS), ((32, 16), "tanh", config.N_INPUTS),
                 ((8,), "linear", config.N_INPUTS), ((16,), "relu", 10)]


@needs_compiler
@pytest.mark.parametrize("hidden, activation, n_inputs", ARCHITECTURES)
def test_dense_programs_compute_numpys_outputs(hidden, activation, n_inputs):
    cars = dense_cars(20, hidden, activation, n_inputs)
    obs = np.random.default_rng(1).random((20, config.N_INPUTS)) * 2 - 0.5
    assert np.array_equal(fastsim.forward(cars, obs), nets.forward(cars.weights, obs, cars.hidden, cars.activation))


@needs_compiler
@pytest.mark.parametrize("hidden, activation, n_inputs", ARCHITECTURES)
def test_dense_time_trials_equal_numpys(hidden, activation, n_inputs, circle_track):
    tracks = [circle_track, get_track(0, 1), get_track(1, 2)]
    cars = dense_cars(60, hidden, activation, n_inputs, seed=3)
    limits = [sim.target_time(t, config.TARGET_FACTOR) for t in tracks]
    for compiled, reference in zip(fastsim.simulate(tracks, cars, limits, laps=config.LAPS),
                                   numpy_results(tracks, cars, limits, config.LAPS)):
        same_results(compiled, reference)


@needs_compiler
def test_results_do_not_depend_on_the_number_of_threads(circle_track):
    tracks = [circle_track, get_track(0, 1), get_track(2, 1)]
    cars = dense_cars(40, seed=4)
    limits = [sim.target_time(t, config.TARGET_FACTOR) for t in tracks]
    one = fastsim.simulate(tracks, cars, limits, laps=config.LAPS, threads=1)
    many = fastsim.simulate(tracks, cars, limits, laps=config.LAPS, threads=7)
    for a, b in zip(one, many):
        same_results(a, b)


@needs_compiler
def test_the_grip_setting_is_passed_through(circle_track):
    cars = dense_cars(30, seed=5)
    limit = sim.time_limit_for(circle_track, 1)
    same_results(fastsim.simulate([circle_track], cars, [limit], grip=7.0)[0],
                 numpy_results([circle_track], cars, [limit], 1, grip=7.0)[0])


@needs_compiler
def test_tiny_and_duplicate_populations_match(circle_track):
    cars = dense_cars(5, seed=6)
    limit = sim.target_time(circle_track, config.TARGET_FACTOR)
    for subset in ([], [2], [1, 1, 3, 1]):
        part = cars.subset(subset)
        compiled = fastsim.simulate([circle_track], part, [limit], laps=config.LAPS)[0]
        if not subset:
            assert compiled.steps == 0 and len(compiled.fitness) == 0
            continue
        same_results(compiled, numpy_results([circle_track], part, [limit], config.LAPS)[0])


@needs_compiler
def test_mismatched_arguments_are_refused(circle_track):
    cars = dense_cars(3)
    with pytest.raises(ValueError, match="time limit"):
        fastsim.simulate([circle_track, circle_track], cars, [10.0])
