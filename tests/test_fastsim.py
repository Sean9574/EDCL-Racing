"""The compiled simulator (spec §5.7, §12.1): its build and fallback, its network programs, and exact equality with the
numpy simulator."""
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from racing import config, fastsim
from racing.actions import class_to_controls

from conftest import spread_setups, stadium

needs_compiler = pytest.mark.skipif(shutil.which(fastsim.COMPILER) is None, reason="no g++ on this machine")


@pytest.fixture(autouse=True)
def compiled_simulator(monkeypatch):
    """Every test here compares the compiled simulator with numpy, whatever EC_RACING_SIM says."""
    monkeypatch.setattr(config, "SIMULATOR", "auto")


@pytest.fixture
def private_library(monkeypatch, tmp_path):
    """The source in a temporary folder, where its library is built, with nothing loaded, checked or failed yet in
    this process."""
    source = tmp_path / "fastsim.cpp"
    shutil.copy(fastsim.SOURCE, source)
    monkeypatch.setattr(fastsim, "SOURCE", source)
    monkeypatch.setattr(fastsim, "_library", None)
    monkeypatch.setattr(fastsim, "_failure", None)
    monkeypatch.setattr(fastsim, "_checked", False)
    return tmp_path


def test_numpy_choice_disables_the_compiled_simulator(monkeypatch):
    monkeypatch.setattr(config, "SIMULATOR", "numpy")
    assert not fastsim.available()
    assert fastsim.description() == "numpy"


@needs_compiler
def test_the_library_builds_and_describes_its_compiler(private_library):
    assert fastsim.available()
    assert fastsim.library_path().exists()
    assert fastsim.library_path().parent == private_library
    assert fastsim.description().startswith("compiled (g++ ")


@needs_compiler
def test_the_description_names_the_compiler_that_built_the_library_without_running_it(private_library,
                                                                                       monkeypatch):
    installed = subprocess.run([fastsim.COMPILER, "-dumpfullversion"], capture_output=True, text=True).stdout.strip()
    assert fastsim.available()
    monkeypatch.setattr(fastsim, "COMPILER", "no-such-compiler-ec-racing")     # g++ removed after the build
    built = fastsim.library().compiler_version().decode()
    assert built.split()[0] == installed
    assert fastsim.description() == f"compiled (g++ {built})"


def test_a_failed_build_warns_once_and_falls_back(private_library, monkeypatch, capsys):
    monkeypatch.setattr(fastsim, "COMPILER", "no-such-compiler-ec-racing")
    assert not fastsim.available()
    assert not fastsim.available()
    err = capsys.readouterr().err
    assert err.count("using the numpy simulator") == 1
    assert fastsim.description() == "numpy"


def test_the_build_identity_covers_the_flags_and_the_processor(private_library, monkeypatch):
    first = fastsim.build_id()
    assert len(first) == 16 and int(first, 16) >= 0
    assert fastsim.library_path().name == f"_fastsim-{first}.so"
    monkeypatch.setattr(fastsim, "cpu_flags", lambda: "flags\t\t: fpu sse2")     # another processor
    other_processor = fastsim.build_id()
    monkeypatch.setattr(fastsim, "FLAGS", [*fastsim.FLAGS, "-g"])                # other flags
    assert len({first, other_processor, fastsim.build_id()}) == 3


@needs_compiler
def test_a_changed_source_is_built_under_a_new_name(private_library):
    """After a git pull that changes fastsim.cpp, the library of the new source is built; the earlier one is younger
    than a day and stays for jobs that may still use it (spec §5.7)."""
    earlier = fastsim.library_path()
    fastsim.build(earlier)
    fastsim.SOURCE.write_text(fastsim.SOURCE.read_text() + "\n// changed by a git pull\n")
    current = fastsim.library_path()
    assert current != earlier and not current.exists()
    assert fastsim.available()
    assert current.exists() and earlier.exists()


@needs_compiler
def test_a_library_of_the_current_identity_is_used_as_it_is(private_library):
    path = fastsim.library_path()
    fastsim.build(path)
    built = path.stat().st_mtime_ns
    assert fastsim.available()
    assert path.stat().st_mtime_ns == built


@needs_compiler
def test_an_unloadable_library_is_rebuilt(private_library):
    fastsim.library_path().write_bytes(b"not a library")             # a damaged file under the current name
    assert fastsim.available()


@needs_compiler
def test_libraries_of_other_identities_are_removed_once_a_day_old(private_library):
    day_old = time.time() - fastsim.OLD_LIBRARY_AGE - 60
    old = private_library / "_fastsim-0123456789abcdef.so"
    young = private_library / "_fastsim-fedcba9876543210.so"
    unnamed = private_library / "_fastsim.so"                         # the name before build identities
    for path in (old, young, unnamed):
        path.write_bytes(b"an earlier library")
    os.utime(old, (day_old, day_old))
    os.utime(unnamed, (day_old, day_old))
    assert fastsim.available()
    assert not old.exists() and not unnamed.exists()
    assert young.exists() and fastsim.library_path().exists()


@needs_compiler
def test_concurrent_builds_all_load(private_library):
    """Four processes that start at once each load a complete library (spec §5.7)."""
    code = (f"import sys; sys.path.insert(0, {str(config.PROJECT_DIR)!r}); from pathlib import Path; "
            f"from racing import fastsim; fastsim.SOURCE = Path({str(fastsim.SOURCE)!r}); "
            f"fastsim.library(); print('ok')")
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(4)]
    outputs = [p.communicate(timeout=300) for p in procs]
    assert all(p.returncode == 0 and out.strip() == "ok" for p, (out, _) in zip(procs, outputs)), outputs
    assert not list(private_library.glob("*.tmp"))
    assert [path.name for path in private_library.glob("_fastsim*")] == [fastsim.library_path().name]


@needs_compiler
def test_threads_that_start_at_once_share_one_library(private_library, capsys):
    """Four threads of one process (the viewer builds maps on threads) call available() at the same moment on a
    library that is not built yet: each gets True, without leftover files or warnings (spec §5.7)."""
    start = threading.Barrier(4)
    answers = []

    def first_use():
        start.wait()
        answers.append(fastsim.available())

    threads = [threading.Thread(target=first_use) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert answers == [True] * 4
    assert not list(private_library.glob("*.tmp"))
    assert capsys.readouterr().err == ""


from racing import nets, sim
from racing.tracks import get_track


def same_results(a, b):
    """Every RunResult field equal; NaN finish times in the same places (spec §5.7)."""
    for field in ("fitness", "finished", "crashed", "progress", "laps_done"):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field
    assert np.array_equal(a.finish_time, b.finish_time, equal_nan=True)
    assert np.array_equal(a.lap_times, b.lap_times, equal_nan=True)
    assert a.steps == b.steps


def numpy_results(tracks, population, limits, laps, grip_scale=1.0):
    return [sim.simulate(t, population.policy(), len(population), time_limit=limit, laps=laps,
                         setup=population.setup, grip_scale=grip_scale) for t, limit in zip(tracks, limits)]


def dense_cars(n, hidden=config.BASELINE_HIDDEN, activation="relu", n_inputs=config.N_INPUTS, seed=0):
    """Random dense networks with random biases and random setups of spread sensor lengths (30-1,000 m) and steering
    sensitivities, so cars crash at many different points and every ray-iteration cap is compared."""
    rng = np.random.default_rng(seed)
    weights = nets.init_population(rng, n, hidden, n_inputs)
    weights = weights + rng.normal(0.0, 0.3, weights.shape)
    return nets.DensePopulation(weights, hidden, activation, spread_setups(rng, n))


ARCHITECTURES = [((16,), "relu", config.N_INPUTS), ((32, 16), "tanh", config.N_INPUTS),
                 ((8,), "linear", config.N_INPUTS), ((16,), "relu", 10), ((1,), "relu", config.N_INPUTS)]


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
@pytest.mark.parametrize("n", [1, 6, 40])
def test_shaped_programs_compute_numpys_outputs(n, shaped_cars):
    cars = shaped_cars(n, seed=n)
    obs = np.random.default_rng(5).random((n, config.N_INPUTS)) * 2 - 0.5
    assert np.array_equal(fastsim.forward(cars, obs), cars.logits(obs))


@needs_compiler
def test_shaped_time_trials_equal_numpys(circle_track, shaped_cars):
    tracks = [circle_track, get_track(0, 1), get_track(1, 2)]
    cars = shaped_cars(48, seed=7)
    limits = [sim.target_time(t, config.TARGET_FACTOR) for t in tracks]
    assert fastsim.supports(cars)
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
def test_a_thread_that_cannot_start_does_not_end_the_process():
    """With the process limit reached no thread can start: the calling thread drives every car instead, with the
    same results, and the process does not abort (spec §5.7)."""
    code = f"""
import resource, sys, threading
sys.path.insert(0, {str(config.PROJECT_DIR)!r})
import numpy as np
from racing import config, fastsim, nets, sim
from racing.track import build_track
config.SIMULATOR = "auto"                                          # whatever EC_RACING_SIM says, as in this module
assert fastsim.available()
a = np.linspace(-np.pi / 2, 1.5 * np.pi, 3200, endpoint=False)
track = build_track(np.column_stack([50 * np.cos(a), 50 * np.sin(a)]), 8.0, 8.0, seed=0, level=1)
track.expert_lap_time = 20.0
cars = nets.DensePopulation(nets.init_population(np.random.default_rng(0), 12))
limit = sim.target_time(track, config.TARGET_FACTOR)
one = fastsim.simulate([track], cars, [limit], laps=config.LAPS, threads=1)[0]
resource.setrlimit(resource.RLIMIT_NPROC, (1, resource.getrlimit(resource.RLIMIT_NPROC)[1]))
try:
    threading.Thread(target=lambda: None).start()
    print("threads still start")
    sys.exit(0)
except RuntimeError:
    pass
many = fastsim.simulate([track], cars, [limit], laps=config.LAPS, threads=8)[0]
same = (np.array_equal(one.fitness, many.fitness) and np.array_equal(one.progress, many.progress)
        and np.array_equal(one.finish_time, many.finish_time, equal_nan=True) and one.steps == many.steps)
print("same" if same else "different")
"""
    fastsim.library()                                               # built here, before the limit
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    if done.stdout.strip() == "threads still start":
        pytest.skip("the process limit does not stop threads here")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "same"


@needs_compiler
def test_nan_outputs_choose_actions_as_np_argmax_does(circle_track):
    """np.argmax counts NaN as the largest value, so the first NaN output is the action; the library chooses the
    same, whether the NaN comes first, after a number or among several (spec §5.7)."""
    cars = dense_cars(6, seed=10)
    for car, outputs in enumerate([[0], [8], [5, 8], [14], [3, 11, 2], list(range(config.N_ACTIONS))]):
        cars.weights[car, -config.N_ACTIONS + np.array(outputs)] = np.nan      # biases of the output layer
    limit = sim.target_time(circle_track, config.TARGET_FACTOR)
    same_results(fastsim.simulate([circle_track], cars, [limit], laps=config.LAPS)[0],
                 numpy_results([circle_track], cars, [limit], config.LAPS)[0])


@needs_compiler
def test_the_grip_setting_is_passed_through(circle_track):
    cars = dense_cars(30, seed=5)
    limit = sim.time_limit_for(circle_track, 1)
    same_results(fastsim.simulate([circle_track], cars, [limit], grip_scale=7.0)[0],
                 numpy_results([circle_track], cars, [limit], 1, grip_scale=7.0)[0])


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


def library_math(kind, x, y=None):
    """The library's check_math: function `kind` (0 numpy's tanh, 1 sin, 2 cos, 3 atan2, 4 hypot) at x, or at the
    pairs (x, y). Kind 5 (sincos) writes 2n values and is refused here."""
    if kind not in (0, 1, 2, 3, 4):
        raise ValueError(f"library_math computes kinds 0 to 4, not {kind}")
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.zeros_like(x) if y is None else np.ascontiguousarray(y, dtype=np.float64)
    out = np.empty_like(x)
    fastsim.library().check_math(kind, fastsim._pointer(x), fastsim._pointer(y), fastsim._pointer(out), len(x))
    return out


TANH_SPECIALS = np.array([0.0, 5e-324, 1e-300, 0.5, 19.06, 20.0, 1e300, np.inf])


@needs_compiler
def test_numpy_tanh_port_matches_numpy():
    """numpy's own float64 tanh, bit for bit (spec §5.7): normal and uniform values, tiny and huge magnitudes, arbitrary
    bit patterns (NaNs among them), the ulps around every edge of numpy's coefficient table and the special values."""
    rng = np.random.default_rng(7)
    edges = np.array([s * 2.0 ** e for e in range(-3, 5) for s in (1.0, 1.5)] + [24.0])   # where the table row changes
    near_edges = (edges.view(np.int64)[:, None] + np.arange(-8, 9)).ravel().view(np.float64)
    x = np.concatenate([rng.normal(0.0, 3.0, 500_000), rng.uniform(-25.0, 25.0, 500_000),
                        rng.normal(0.0, 0.2, 400_000),
                        rng.choice([-1.0, 1.0], 300_000) * 10.0 ** rng.uniform(-320.0, 308.0, 300_000),
                        rng.integers(0, 2 ** 64, 300_000, dtype=np.uint64, endpoint=False).view(np.float64),
                        near_edges, -near_edges, TANH_SPECIALS, -TANH_SPECIALS, [np.nan]])
    assert np.array_equal(library_math(0, x).view(np.uint64), np.tanh(x).view(np.uint64))


@needs_compiler
def test_math_functions_match_numpy_here():
    assert fastsim._math_mismatch(fastsim.library()) is None


@needs_compiler
@pytest.mark.parametrize("check, name", [("_math_mismatch", "tanh"), ("_network_mismatch", "dense forward"),
                                         ("_network_mismatch", "shaped forward"),
                                         ("_network_mismatch", "neat forward")])
def test_a_load_check_mismatch_falls_back(private_library, monkeypatch, capsys, check, name):
    monkeypatch.setattr(fastsim, check, lambda lib: name)
    assert not fastsim.available()
    assert not fastsim.available()
    err = capsys.readouterr().err
    assert err.count("using the numpy simulator") == 1
    assert f"the compiled simulator's {name} differs from numpy's on this machine" in err


@needs_compiler
def test_network_outputs_match_numpy_here():
    assert fastsim._network_mismatch(fastsim.library()) is None


@needs_compiler
def test_the_network_check_sees_a_last_bit_difference(monkeypatch):
    """The load-time check compares bits: one ulp in one output of numpy's dense or NEAT forward is a mismatch."""
    from racing import neat
    lib = fastsim.library()
    dense_forward, neat_forward = nets.forward, neat.forward
    monkeypatch.setattr(nets, "forward", lambda *args: np.nextafter(dense_forward(*args), np.inf))
    assert fastsim._network_mismatch(lib) == "dense forward"
    monkeypatch.setattr(nets, "forward", dense_forward)
    shaped = nets.ShapedPopulation.logits
    monkeypatch.setattr(nets.ShapedPopulation, "logits", lambda self, obs: np.nextafter(shaped(self, obs), np.inf))
    assert fastsim._network_mismatch(lib) == "shaped forward"
    monkeypatch.setattr(nets.ShapedPopulation, "logits", shaped)
    monkeypatch.setattr(nets, "forward", dense_forward)
    monkeypatch.setattr(neat, "forward", lambda *args: np.nextafter(neat_forward(*args), np.inf))
    assert fastsim._network_mismatch(lib) == "neat forward"


from racing import neat


def grown_genomes(n, seed=20):
    """Genomes grown by node and connection mutations, with some connections disabled and random biases."""
    rng = np.random.default_rng(seed)
    innovations = neat.Innovations()
    genomes = []
    for rounds in rng.integers(0, 30, n):
        genome = neat.initial_genome(rng, innovations)
        for _ in range(int(rounds)):
            genome = neat.add_node(rng, genome, innovations)
            genome = neat.add_connection(rng, genome, innovations)
            if rng.random() < 0.3:
                k = int(rng.integers(len(genome.innov)))
                genome.enabled[k] = not genome.enabled[k]
        genome.bias = rng.normal(0.0, 0.5, len(genome.bias))
        genomes.append(genome)
    return genomes


@needs_compiler
def test_neat_programs_compute_numpys_outputs():
    genomes = grown_genomes(12)
    cars = neat.NeatPopulation(genomes)
    obs = np.random.default_rng(21).random((12, config.N_INPUTS)) * 2 - 0.5
    assert np.array_equal(fastsim.forward(cars, obs), neat.forward(neat.compile_population(genomes), obs))


@needs_compiler
def test_neat_time_trials_equal_numpys(circle_track):
    tracks = [circle_track, get_track(0, 1), get_track(1, 2)]
    rng = np.random.default_rng(22)
    cars = neat.NeatPopulation(grown_genomes(40, seed=23), spread_setups(rng, 40))
    limits = [sim.target_time(t, config.TARGET_FACTOR) for t in tracks]
    for compiled, reference in zip(fastsim.simulate(tracks, cars, limits, laps=config.LAPS),
                                   numpy_results(tracks, cars, limits, config.LAPS)):
        same_results(compiled, reference)


@needs_compiler
def test_the_sincos_check_is_exercised():
    assert [name for name, *_ in fastsim._math_inputs()][-1] == "sincos"
    assert fastsim._math_mismatch(fastsim.library()) is None


@needs_compiler
@pytest.mark.skipif(shutil.which("objdump") is None, reason="no objdump on this machine")
def test_the_library_calls_sin_and_cos_not_sincos():
    """g++ fuses the sin and cos of one angle into the C library's sincos unless the build forbids it, and glibc
    2.35's sincos differs from its own sin and cos, which numpy calls, in about one angle in 1,500 (spec §5.7)."""
    fastsim.library()
    symbols = subprocess.run(["objdump", "-T", str(fastsim.library_path())], capture_output=True, text=True,
                             check=True).stdout.split()
    assert "sin" in symbols and "cos" in symbols and "sincos" not in symbols


def _bound_versions(path, names=("sin", "cos", "atan2", "hypot")):
    """The C library version each of `names` is bound to in the shared object at `path` (objdump -T)."""
    found = {}
    listing = subprocess.run(["objdump", "-T", str(path)], capture_output=True, text=True, check=True).stdout
    for line in listing.splitlines():
        match = re.search(r"\*UND\*\s+\S+\s+\(?(GLIBC_[\d.]+)\)?\s+(\w+)$", line)
        if match and match.group(2) in names:
            found[match.group(2)] = match.group(1)
    return found


@needs_compiler
@pytest.mark.skipif(shutil.which("objdump") is None, reason="no objdump on this machine")
def test_the_library_binds_the_math_functions_numpy_binds():
    """The C library may hold several versions of a function: glibc 2.35 added a rewritten hypot. The library must
    call the version numpy calls (spec §5.7)."""
    fastsim.library()
    umath = next(Path(np.__file__).parent.glob("*core/_multiarray_umath*.so"))
    ours, numpys = _bound_versions(fastsim.library_path()), _bound_versions(umath)
    assert set(ours) == {"sin", "cos", "atan2", "hypot"}
    assert ours == {name: numpys[name] for name in ours}


@needs_compiler
def test_a_library_without_an_entry_point_falls_back(private_library, capsys):
    """A library under the current name that loads but lacks an entry point (not one this source built) is
    unavailable: one warning, then numpy."""
    stub = private_library / "stub.cpp"
    stub.write_text('extern "C" void simulate() {}\n')
    subprocess.run([fastsim.COMPILER, *fastsim.FLAGS, str(stub), "-o", str(fastsim.library_path())], check=True)
    assert not fastsim.available()
    assert not fastsim.available()
    assert capsys.readouterr().err.count("using the numpy simulator") == 1


from racing import expert, track_gen


def numpy_expert(track, n=1, time_limit=config.EXPERT_VALIDATION_TIME, laps=1):
    return sim.simulate(track, expert.expert_policy(track), n, time_limit=time_limit, laps=laps)


@needs_compiler
def test_the_expert_drives_exactly_as_in_numpy(circle_track):
    maps = [(seed, level) for level in range(1, config.MAX_LEVEL) for seed in (3, 4)]     # the generated levels
    maps.append((4, config.VALIDATION_LEVEL))
    tracks = [circle_track] + [t for t in (track_gen.candidate_track(seed, level, 0) for seed, level in maps)
                               if t is not None]
    assert tracks[-1].level == config.VALIDATION_LEVEL
    limits = [config.EXPERT_VALIDATION_TIME] * len(tracks)
    for compiled, track in zip(fastsim.simulate_expert(tracks, 1, limits), tracks):
        same_results(compiled, numpy_expert(track))


@needs_compiler
def test_expert_lap_time_uses_the_compiled_simulator(circle_track, monkeypatch):
    lap = expert.expert_lap_time(circle_track)
    reference = numpy_expert(circle_track)
    assert lap == float(reference.finish_time[0])
    monkeypatch.setattr(fastsim.sim, "simulate", lambda *a, **k: pytest.fail("numpy simulator used"))
    assert expert.expert_lap_time(circle_track) == lap


@needs_compiler
def test_several_expert_cars_over_three_laps(circle_track):
    limit = sim.target_time(circle_track, 0.4)          # too tight: every car runs out of time partway
    same_results(fastsim.simulate_expert([circle_track], 3, [limit], laps=config.LAPS)[0],
                 numpy_expert(circle_track, 3, limit, config.LAPS))


@needs_compiler
def test_the_compiled_car_matches_numpy_at_formula_1_speeds():
    track = stadium(straight=1500.0, radius=300.0, width=14.0)
    cars = dense_cars(40, seed=11)
    cars.setup[:, 0] = np.linspace(30.0, 1000.0, 40)                    # every sensor length up to 1,000 m
    limits = [track.expert_lap_time * 3]
    for grip_scale in (1.0, 10.0):
        compiled = fastsim.simulate([track], cars, limits, laps=config.LAPS, grip_scale=grip_scale)[0]
        reference = sim.simulate(track, cars.policy(), len(cars), time_limit=limits[0], laps=config.LAPS,
                                 setup=cars.setup, grip_scale=grip_scale)
        same_results(compiled, reference)


@needs_compiler
def test_the_compiled_expert_matches_numpy_on_a_fast_stadium():
    track = stadium(straight=1500.0, radius=300.0, width=14.0)
    compiled = fastsim.simulate_expert([track], 3, [600.0], laps=2)[0]
    reference = sim.simulate(track, expert.expert_policy(track), 3, time_limit=600.0, laps=2)
    same_results(compiled, reference)
    assert reference.finished.all()


@needs_compiler
def test_the_compiled_expert_uses_the_half_levels_as_numpy_does(circle_track):
    """Spec §5.7, §7.1: the expert's proportional throttle and its quantization to the 45 classes are identical in
    both simulators, and its laps use the half levels."""
    tracks_ = [circle_track, get_track(2_000_000, 2), get_track(2_000_001, 3)]
    limits = [config.EXPERT_VALIDATION_TIME] * 3
    compiled = fastsim.simulate_expert(tracks_, 1, limits, laps=config.LAPS)
    for track, limit, fast in zip(tracks_, limits, compiled):
        seen = []
        policy = expert.expert_policy(track)

        def recording(obs, cars):
            classes = policy(obs, cars)
            seen.extend(classes.tolist())
            return classes

        reference = sim.simulate(track, recording, 1, time_limit=limit, laps=config.LAPS)
        same_results(fast, reference)
        throttles = {float(t) for t in class_to_controls(np.array(seen))[1]}
        assert throttles & {-0.5, 0.5}                     # the half levels are used


from racing.track import build_track


def uneven_ellipse():
    """A 420 × 300 m ellipse (semi-axes; ends of radius 214 m, taken near 5 g) whose left and right widths differ and
    change along the lap."""
    a = np.linspace(-np.pi / 2, 1.5 * np.pi, 4000, endpoint=False)
    points = np.column_stack([300 * np.cos(a) * 1.4, 300 * np.sin(a)])
    left = 5.0 + 2.0 * np.sin(3 * a)
    right = 6.0 + 1.5 * np.cos(2 * a)
    return build_track(points, left, right, seed=0, level=1)


@needs_compiler
def test_the_expert_completes_three_laps_of_a_420_by_300_m_ellipse_in_both_simulators():
    """Spec §7.1: the expert coasts within one step's speed change of its target, so it keeps the grip for turning in
    long corners near 5 g; with a 0.5 m/s band alone it crashed here on lap 2."""
    track = uneven_ellipse()
    compiled = fastsim.simulate_expert([track], 1, [config.EXPERT_VALIDATION_TIME], laps=config.LAPS)[0]
    reference = sim.simulate(track, expert.expert_policy(track), 1, time_limit=config.EXPERT_VALIDATION_TIME,
                             laps=config.LAPS)
    same_results(compiled, reference)
    assert reference.finished[0] and np.isfinite(reference.lap_times).all()


@needs_compiler
def test_the_compiled_simulator_matches_numpy_on_uneven_widths_including_lap_times():
    track = uneven_ellipse()                                              # an ellipse the expert drives 3 laps on
    track.expert_lap_time = expert.expert_lap_time(track)
    cars = dense_cars(60, seed=5)
    limits = [3 * track.expert_lap_time * 1.5]
    compiled = fastsim.simulate([track], cars, limits, laps=config.LAPS)[0]
    reference = sim.simulate(track, cars.policy(), len(cars), time_limit=limits[0], laps=config.LAPS,
                             setup=cars.setup)
    same_results(compiled, reference)
    assert np.array_equal(compiled.lap_times, reference.lap_times, equal_nan=True)
    expert_compiled = fastsim.simulate_expert([track], 2, limits, laps=config.LAPS)[0]
    expert_reference = sim.simulate(track, expert.expert_policy(track), 2, time_limit=limits[0], laps=config.LAPS)
    assert np.array_equal(expert_compiled.lap_times, expert_reference.lap_times, equal_nan=True)
    assert np.isfinite(expert_reference.lap_times).all()


@needs_compiler
def test_the_compiled_simulator_matches_numpy_on_real_circuits():
    tracks_ = [get_track(0, config.MAX_LEVEL), get_track(1, config.MAX_LEVEL), get_track(3, config.TEST_LEVEL)]
    cars = dense_cars(24, seed=9)
    limits = [sim.target_time(t, config.TARGET_FACTOR) for t in tracks_]
    for compiled, reference in zip(fastsim.simulate(tracks_, cars, limits, laps=config.LAPS),
                                   numpy_results(tracks_, cars, limits, config.LAPS)):
        same_results(compiled, reference)


@needs_compiler
@pytest.mark.slow
def test_the_expert_drives_three_laps_of_every_real_track_alike_in_both_simulators():
    """Spec §4.5, §7.1, §12.1: the expert completes the 38 training layouts and the 4 test circuits for config.LAPS
    laps, with the same result in both simulators (spec §5.7)."""
    from racing import circuits
    real = ([(k, config.MAX_LEVEL) for k in range(circuits.N_LAYOUTS)]
            + [(k, config.TEST_LEVEL) for k in config.TEST_TRACKS])
    limit = config.LAPS * config.EXPERT_VALIDATION_TIME
    for seed, level in real:
        track = circuits.level_track(seed, level)
        reference = numpy_expert(track, time_limit=limit, laps=config.LAPS)
        assert reference.finished[0] and np.isfinite(reference.lap_times).all(), (track.name, seed, level)
        same_results(fastsim.simulate_expert([track], 1, [limit], laps=config.LAPS)[0], reference)


def test_the_simulator_reads_a_tracks_arrays_without_copying_them(circle_track):
    """The arrays the compiled simulator reads are stored contiguous once when a track is built, so a call never copies
    them; a real circuit's memory-mapped road grid is read in place (spec §5.7, §9.10)."""
    from racing import circuits
    for track in (circle_track, circuits.circuit_track("Spielberg"), circuits.circuit_track("Spielberg", True)):
        keep = []
        fastsim._c_track(track, 10.0, keep)
        dist, *arrays = keep[0]
        assert np.shares_memory(dist, track.dist)
        for array, name in zip(arrays, ("centerline", "tangent", "curvature", "half_left", "half_right")):
            assert np.shares_memory(array, getattr(track, name)), name


from racing import population


@needs_compiler
def test_training_time_trials_use_the_compiled_simulator_with_the_same_result(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    cars = dense_cars(30, seed=8)
    compiled = population.evaluate_population(cars, [0, 1, 2], 1, config.TARGET_FACTOR, n_jobs=4)
    monkeypatch.setattr(config, "SIMULATOR", "numpy")
    reference = population.evaluate_population(cars, [0, 1, 2], 1, config.TARGET_FACTOR, n_jobs=1)
    for field in ("fitness", "success", "laps"):
        assert np.array_equal(getattr(compiled, field), getattr(reference, field)), field
    assert np.array_equal(compiled.finish_time, reference.finish_time, equal_nan=True)


@needs_compiler
def test_unknown_population_types_use_numpy(monkeypatch, circle_track):
    from test_population import ExpertCars          # pytest puts tests/ on the path; "tests." can hit a site-packages "tests"
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    result = population.evaluate_population(ExpertCars(circle_track, 2), [0], 1, config.TARGET_FACTOR)
    assert result.success.all()


@needs_compiler
@pytest.mark.slow
def test_a_trained_population_matches_numpy_on_real_layouts():
    """Trained cars drive full laps, so every part of the simulator runs for thousands of steps (spec §12.1); they
    drive three real training layouts, as at Extra hard (a checkpoint records no layouts)."""
    from racing import checkpoint
    state = cars = None
    for path in sorted(config.RESULTS_DIR.glob("runs/*/seed*/checkpoint.pkl")):
        candidate = checkpoint.load(path)
        method = candidate["method"]
        if not all(key in method for key in ("weights", "hidden", "activation", "setup")):
            continue
        population_ = nets.DensePopulation(method["weights"], method["hidden"], method["activation"], method["setup"])
        if population_.n_inputs == config.N_INPUTS:
            state, cars, found = candidate, population_, path
            break
    if state is None:
        pytest.skip("no dense checkpoint in results/")
    tracks = [get_track(k, config.MAX_LEVEL) for k in (0, 1, 2)]
    limits = [sim.target_time(t, config.TARGET_FACTOR) for t in tracks]
    finishers = []
    for compiled, reference in zip(fastsim.simulate(tracks, cars, limits, laps=config.LAPS, threads=6),
                                   numpy_results(tracks, cars, limits, config.LAPS)):
        same_results(compiled, reference)                            # a real difference always fails
        finishers.append(int(compiled.finished.sum()))
    if max(finishers) < len(cars) // 2:
        pytest.skip(f"checkpoint {found} has too few finishers for a long-trajectory check")


@needs_compiler
def test_final_selection_with_setups_matches_numpy(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    cars = dense_cars(6, seed=9)
    compiled = population.select_final(cars, [3, 1], config.TARGET_FACTOR, seeds=[80])
    monkeypatch.setattr(config, "SIMULATOR", "numpy")
    reference = population.select_final(cars, [3, 1], config.TARGET_FACTOR, seeds=[80])
    assert compiled == reference


@needs_compiler
def test_neat_training_time_trials_match_numpy(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    cars = neat.NeatPopulation(grown_genomes(30, seed=24), spread_setups(np.random.default_rng(25), 30))
    compiled = population.evaluate_population(cars, [0, 1, 2], 1, config.TARGET_FACTOR, n_jobs=4)
    monkeypatch.setattr(config, "SIMULATOR", "numpy")
    reference = population.evaluate_population(cars, [0, 1, 2], 1, config.TARGET_FACTOR, n_jobs=1)
    for field in ("fitness", "success", "laps"):
        assert np.array_equal(getattr(compiled, field), getattr(reference, field)), field
    assert np.array_equal(compiled.finish_time, reference.finish_time, equal_nan=True)


def test_the_compiler_is_ec_racing_cxx_then_the_g_plus_plus_beside_python_then_the_path(tmp_path):
    python = tmp_path / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    assert fastsim.compiler({"EC_RACING_CXX": "/opt/gcc/bin/g++-14"}, python) == "/opt/gcc/bin/g++-14"
    assert fastsim.compiler({}, python) == "g++"                                 # none beside it: the search path
    (python.parent / "g++").write_text("")
    assert fastsim.compiler({}, python) == str(python.parent / "g++")             # a conda environment's own
    assert fastsim.compiler({"EC_RACING_CXX": "  "}, python) == str(python.parent / "g++")


def test_a_named_compiler_that_does_not_exist_warns_once_and_falls_back(private_library, monkeypatch, capsys):
    monkeypatch.setattr(fastsim, "COMPILER", fastsim.compiler({"EC_RACING_CXX": "/nonexistent/g++"}))
    assert not fastsim.available() and not fastsim.available()
    assert capsys.readouterr().err.count("using the numpy simulator") == 1
