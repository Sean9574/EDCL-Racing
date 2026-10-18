"""Compiled simulator (spec §5.7): time trials of dense and NEAT populations and the expert's lap, computed by
racing/fastsim.cpp exactly as racing/sim.py computes them. The library is built with g++ the first time it is needed,
under a name given by its build identity (the source, the compiler flags and the processor); without g++ the numpy
simulator runs instead, with identical results."""
import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from racing import config, nets, sim
from racing import setup as car_setup

SOURCE = Path(__file__).with_name("fastsim.cpp")
COMPILER = "g++"
# -ffp-contract=off: no fused multiply-add, so every product and sum rounds as it does in numpy; -fno-builtin-sin and
# -fno-builtin-cos: the sin and cos of one angle stay two calls, as numpy makes them, instead of the C library's sincos,
# which in glibc 2.35 differs from its own sin and cos (spec §5.7)
FLAGS = ["-O3", "-march=native", "-ffp-contract=off", "-fno-builtin-sin", "-fno-builtin-cos", "-std=c++17", "-shared",
         "-fPIC", "-pthread"]
OLD_LIBRARY_AGE = 24 * 3600     # s: a library of another build identity is removed once it is this old

# The first use builds, loads and checks the library once per process. The viewer builds maps on several threads at
# once, so _lock guards that sequence and the three globals below.
_lock = threading.Lock()
_library = None          # the loaded library
_failure = None          # the error of a failed build, load or check in this process; the numpy simulator runs
_checked = False         # the library's math functions and network outputs equal numpy's in this process


def cpu_flags():
    """The flags line of /proc/cpuinfo (the instructions -march=native may use), or "" where it cannot be read."""
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("flags"):
                    return line.strip()
    except OSError:
        pass
    return ""


def build_id():
    """The build identity (spec §5.7): the first 16 hex digits of the SHA-256 of fastsim.cpp, the compiler flags and
    the processor's flags. A library built from other source, with other flags or for another processor has
    another identity, and so another file name."""
    digest = hashlib.sha256(SOURCE.read_bytes())
    digest.update(b"\0" + " ".join(FLAGS).encode())
    digest.update(b"\0" + cpu_flags().encode())
    return digest.hexdigest()[:16]


def library_path():
    """The library of the current build identity: _fastsim-<build_id>.so next to fastsim.cpp."""
    return SOURCE.with_name(f"_fastsim-{build_id()}.so")


def build(path):
    """Compiles fastsim.cpp into `path`. The compiler writes a temporary file unique to this call, which is then
    renamed into place, so threads and processes that build at once never load a half-written library (spec §5.7)."""
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{path.stem}.", suffix=".tmp")
    os.close(descriptor)
    try:
        subprocess.run([COMPILER, *FLAGS, str(SOURCE), "-o", temporary], check=True, capture_output=True,
                       text=True)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def remove_old_libraries(current):
    """Removes the libraries of other build identities next to `current` once they are OLD_LIBRARY_AGE old; a
    younger one may still be in use by a job that started before the source changed."""
    cutoff = time.time() - OLD_LIBRARY_AGE
    for old in current.parent.glob("_fastsim*.so"):
        if old == current:
            continue
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except FileNotFoundError:                    # removed by another job at the same moment
            pass


def _load():
    """The library of the current build identity, loaded: a file under that name is used as it is, and built first
    if there is none. A file under that name that does not load (damaged) is built again."""
    path = library_path()
    if not path.exists():
        build(path)
    try:
        lib = ctypes.CDLL(str(path))
    except OSError:
        build(path)
        lib = ctypes.CDLL(str(path))
    remove_old_libraries(path)
    return lib


def _loaded():
    """library() for a caller that holds _lock."""
    global _library
    if _library is None:
        lib = _load()
        _declare(lib)
        _library = lib
    return _library


def library():
    """The loaded library, built first when needed. Raises OSError, CalledProcessError or AttributeError (an entry
    point missing) if it cannot be built, loaded or declared. Safe to call from several threads at once."""
    with _lock:
        return _loaded()


def available():
    """True when the compiled simulator runs: EC_RACING_SIM is not "numpy", the library builds and loads, and its
    tanh, sin, cos, sincos, atan2 and hypot and its dense and NEAT network outputs equal numpy's on this machine
    (checked once per process). A failed build or check prints one warning, and the numpy simulator runs instead
    (spec §5.7). Safe to call from several threads at once: the first call builds, loads and checks the library while
    the others wait."""
    global _failure, _checked
    with _lock:
        if config.SIMULATOR == "numpy" or _failure is not None:
            return False
        try:
            lib = _loaded()
        except (OSError, subprocess.CalledProcessError, AttributeError) as error:
            _failure = error
            detail = (getattr(error, "stderr", None) or str(error)).strip()
            print(f"EC Racing: the compiled simulator is unavailable ({detail}); using the numpy simulator",
                  file=sys.stderr)
            return False
        if not _checked:
            name = _math_mismatch(lib)
            if name is None:
                name = _network_mismatch(lib)
            if name is not None:
                _failure = RuntimeError(f"the compiled simulator's {name} differs from numpy's")
                print(f"EC Racing: the compiled simulator's {name} differs from numpy's on this machine; using the "
                      f"numpy simulator", file=sys.stderr)
                return False
            _checked = True
        return True


def _math_inputs():
    """Fixed inputs, about 200,000 each, of the functions check_math computes, in the order of its kinds: (name,
    numpy's function, x, y or None); sincos, kind 5, has no single numpy function (see _math_mismatch). Headings
    accumulate over laps, so sin and cos also see large angles."""
    rng = np.random.default_rng(0)
    n = 100_000
    specials = np.array([0.0, 5e-324, 1e-300, 0.5, 19.06, 20.0, 1e300, np.inf])
    tanh_x = np.concatenate([rng.normal(0.0, 3.0, n), rng.uniform(-25.0, 25.0, n), specials, -specials, [np.nan]])
    angles = np.concatenate([rng.uniform(-4.0, 4.0, n), rng.uniform(-200.0, 200.0, n)])
    px, py = rng.uniform(-150.0, 150.0, (2, 2 * n))
    return [("tanh", np.tanh, tanh_x, None), ("sin", np.sin, angles, None), ("cos", np.cos, angles, None),
            ("atan2", np.arctan2, px, py), ("hypot", np.hypot, px, py), ("sincos", None, angles, None)]


def _bits(values):
    """values as unsigned integers, every NaN as the one quiet NaN."""
    return np.where(np.isnan(values), np.nan, values).view(np.uint64)


def _math_mismatch(lib):
    """The name of the first of tanh, sin, cos, atan2, hypot and sincos whose values in the library differ from
    numpy's on fixed inputs, or None. sincos is the library's sin and cos of one angle together, as the engine computes
    them, which a compiler that fused them into the C library's sincos would change (FLAGS forbid that); its first half
    is compared with np.sin and its second with np.cos. These are the
    functions the engine calls that are not exactly rounded, so equal values here make the compiled simulator equal
    to numpy on this machine (spec §5.7)."""
    for kind, (name, function, x, y) in enumerate(_math_inputs()):
        n = len(x)
        out = np.empty(2 * n if name == "sincos" else n)
        lib.check_math(kind, _pointer(x), _pointer(x if y is None else y), _pointer(out), n)
        if name == "sincos":
            same = (np.array_equal(_bits(out[:n]), _bits(np.sin(x)))
                    and np.array_equal(_bits(out[n:]), _bits(np.cos(x))))
        else:
            expected = function(x) if y is None else function(x, y)
            same = np.array_equal(_bits(out), _bits(expected))
        if not same:
            return name
    return None


def _network_mismatch(lib):
    """The name "dense forward" or "neat forward" if the library's network outputs differ from numpy's on small
    fixed populations, else None (spec §5.7). Dense networks of two hidden layers (ReLU and tanh; 1 and 7 cars; all
    config.N_INPUTS inputs and the first 10, as an earlier revision's networks read) are summed in np.einsum's
    order, and grown NEAT genomes in np.bincount's; equal outputs show that the library sums as numpy does on this
    machine. Takes a few milliseconds."""
    from racing import neat
    rng = np.random.default_rng(0)
    for activation in ("relu", "tanh"):
        for n in (1, 7):
            for n_inputs in (config.N_INPUTS, 10):
                weights = rng.normal(0.0, 0.5, (n, nets.n_params((16, 8), n_inputs)))
                cars = nets.DensePopulation(weights, (16, 8), activation)
                obs = rng.uniform(-0.5, 1.5, (n, config.N_INPUTS))
                expected = nets.forward(cars.weights, obs, cars.hidden, cars.activation)
                if not np.array_equal(_bits(_forward(lib, cars, obs)), _bits(expected)):
                    return "dense forward"
    rng = np.random.default_rng(0)
    innovations = neat.Innovations()
    genomes = []
    for rounds in (0, 2, 5, 9, 14, 20):
        genome = neat.initial_genome(rng, innovations)
        for _ in range(rounds):
            genome = neat.add_node(rng, genome, innovations)
            genome = neat.add_connection(rng, genome, innovations)
        genome.bias = rng.normal(0.0, 0.5, len(genome.bias))
        genomes.append(genome)
    obs = rng.uniform(-0.5, 1.5, (len(genomes), config.N_INPUTS))
    expected = neat.forward(neat.compile_population(genomes), obs)
    if not np.array_equal(_bits(_forward(lib, neat.NeatPopulation(genomes), obs)), _bits(expected)):
        return "neat forward"
    return None


def description():
    """The simulator in use, for results/run_info.json and the run summaries: "compiled (g++ <version>)", with the
    version of the compiler that built the library (recorded in it), or "numpy"."""
    if not available():
        return "numpy"
    return f"compiled (g++ {library().compiler_version().decode()})"


P_DOUBLE = ctypes.POINTER(ctypes.c_double)
P_INT64 = ctypes.POINTER(ctypes.c_int64)
P_UINT8 = ctypes.POINTER(ctypes.c_uint8)
ACTIVATIONS = {"relu": 0, "tanh": 1, "linear": 2}
IDENTITY = ACTIVATIONS["linear"]


class Track(ctypes.Structure):
    _fields_ = [("dist", P_DOUBLE), ("rows", ctypes.c_int64), ("cols", ctypes.c_int64),
                ("origin_x", ctypes.c_double), ("origin_y", ctypes.c_double), ("width", ctypes.c_double),
                ("centerline", P_DOUBLE), ("tangent", P_DOUBLE), ("curvature", P_DOUBLE), ("m", ctypes.c_int64),
                ("spacing", ctypes.c_double), ("start_x", ctypes.c_double), ("start_y", ctypes.c_double),
                ("start_heading", ctypes.c_double), ("back", ctypes.c_int64), ("ahead", ctypes.c_int64),
                ("sight", ctypes.c_int64), ("time_limit", ctypes.c_double)]


class Settings(ctypes.Structure):
    _fields_ = ([(name, ctypes.c_double) for name in (
                    "dt", "grip", "yaw_min_speed", "speed_scale", "ray_range", "width_scale", "ray_min_step",
                    "grid_res", "stall_time", "time_eps", "expert_grip", "lookahead_base", "lookahead_time",
                    "grip_margin", "speed_band", "expert_sight")]
                + [(name, ctypes.c_int64) for name in ("ray_max_iterations", "laps", "n_inputs")]
                + [("ray_angles", P_DOUBLE), ("n_rays", ctypes.c_int64), ("steer_levels", P_DOUBLE),
                   ("n_steer", ctypes.c_int64), ("throttle_levels", P_DOUBLE), ("n_throttle", ctypes.c_int64)])


class CPrograms(ctypes.Structure):
    _fields_ = [("node_offset", P_INT64), ("bias", P_DOUBLE), ("activation", P_INT64), ("conn_offset", P_INT64),
                ("source", P_INT64), ("weight", P_DOUBLE), ("outputs", P_INT64), ("n_outputs", ctypes.c_int64)]


class Results(ctypes.Structure):
    _fields_ = [("finished", P_UINT8), ("crashed", P_UINT8), ("progress", P_DOUBLE), ("finish_time", P_DOUBLE),
                ("laps_done", P_INT64), ("steps", P_INT64)]


def _declare(lib):
    """Argument types of the library's entry points."""
    lib.simulate.restype = None
    lib.simulate.argtypes = [ctypes.POINTER(Track), ctypes.c_int64, ctypes.POINTER(Settings),
                             ctypes.POINTER(CPrograms), P_DOUBLE, ctypes.c_int64, ctypes.c_int64,
                             ctypes.POINTER(Results)]
    lib.forward.restype = None
    lib.forward.argtypes = [ctypes.POINTER(CPrograms), P_DOUBLE, ctypes.c_int64, ctypes.c_int64, P_DOUBLE]
    lib.check_math.restype = None
    lib.check_math.argtypes = [ctypes.c_int64, P_DOUBLE, P_DOUBLE, P_DOUBLE, ctypes.c_int64]
    lib.compiler_version.restype = ctypes.c_char_p
    lib.compiler_version.argtypes = []


@dataclass(eq=False)
class Programs:
    """Network programs of n cars (spec §5.7): car c's nodes are node_offset[c] .. node_offset[c + 1] - 1, node k's
    connections conn_offset[k] .. conn_offset[k + 1] - 1. A connection's source is a slot of its own car: input i is
    slot i, the car's node j is slot N_INPUTS + j."""
    node_offset: np.ndarray     # int64 (n + 1,)
    bias: np.ndarray            # float64, one per node
    activation: np.ndarray      # int64, one per node: ACTIVATIONS
    conn_offset: np.ndarray     # int64 (n_nodes + 1,)
    source: np.ndarray          # int64, one per connection
    weight: np.ndarray          # float64, one per connection
    outputs: np.ndarray         # int64 (n, N_ACTIONS): output slots in action order


def dense_program(population):
    """Programs of a nets.DensePopulation: layer by layer, each unit is a node whose connections come from every unit
    of the previous layer (or every input the network reads) in index order, so it sums as np.einsum does."""
    n = len(population)
    sizes = nets.layer_sizes(population.hidden, population.n_inputs)
    layers = nets.unflatten(population.weights, population.hidden)
    act = ACTIVATIONS[population.activation]
    bias, activation, counts, source, weight = [], [], [], [], []
    previous = np.arange(sizes[0])                      # slots of the inputs the network reads
    next_slot = config.N_INPUTS
    for k, (w, b) in enumerate(layers):
        a, units = sizes[k], sizes[k + 1]
        bias.append(b)                                                   # (n, units)
        activation.append(np.full(units, act if k < len(layers) - 1 else IDENTITY))
        counts.append(np.full(units, a))
        source.append(np.tile(previous, units))                         # unit j's sources, j = 0, 1, ...
        weight.append(np.transpose(w, (0, 2, 1)).reshape(n, units * a))  # unit j's weights, in source order
        previous = next_slot + np.arange(units)
        next_slot += units
    per_car = sum(sizes[1:])
    node_counts = np.tile(np.concatenate(counts), n)
    return Programs(node_offset=np.arange(n + 1, dtype=np.int64) * per_car,
                    bias=np.ascontiguousarray(np.concatenate(bias, axis=1).ravel(), dtype=np.float64),
                    activation=np.tile(np.concatenate(activation), n).astype(np.int64),
                    conn_offset=np.concatenate([[0], np.cumsum(node_counts)]).astype(np.int64),
                    source=np.tile(np.concatenate(source), n).astype(np.int64),
                    weight=np.ascontiguousarray(np.concatenate(weight, axis=1).ravel(), dtype=np.float64),
                    outputs=np.tile(previous, (n, 1)).astype(np.int64))


def neat_program(population):
    """Programs of a neat.NeatPopulation: nodes by depth (neat.node_depths), each with its enabled connections in
    genome order, so it sums as np.bincount does in neat.forward; hidden nodes use tanh, outputs the identity."""
    from racing import neat
    node_offset, bias, activation, counts, source, weight, outputs = [0], [], [], [], [], [], []
    for genome in population.genomes:
        depth = neat.node_depths(genome)
        ids = genome.node_ids.tolist()
        order = sorted(range(len(ids)), key=lambda k: depth[ids[k]])            # stable: ties keep id order
        slot = {i: i for i in range(neat.N_IN)}
        slot.update({ids[k]: config.N_INPUTS + j for j, k in enumerate(order)})
        on = np.flatnonzero(genome.enabled)
        for k in order:
            node = ids[k]
            mine = on[genome.dst[on] == node]                                    # genome order
            bias.append(float(genome.bias[k]))
            activation.append(ACTIVATIONS["tanh"] if node >= neat.FIRST_HIDDEN else IDENTITY)
            counts.append(len(mine))
            source.extend(slot[int(s)] for s in genome.src[mine])
            weight.extend(genome.weight[mine].tolist())
        node_offset.append(node_offset[-1] + len(ids))
        outputs.append([slot[o] for o in neat.OUTPUT_IDS])
    return Programs(node_offset=np.array(node_offset, dtype=np.int64), bias=np.array(bias, dtype=np.float64),
                    activation=np.array(activation, dtype=np.int64),
                    conn_offset=np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64),
                    source=np.array(source, dtype=np.int64), weight=np.array(weight, dtype=np.float64),
                    outputs=np.array(outputs, dtype=np.int64).reshape(len(population.genomes), config.N_ACTIONS))


def supports(population):
    """True for the populations the compiled simulator drives: dense and NEAT (spec §5.7)."""
    from racing import neat
    return isinstance(population, (nets.DensePopulation, neat.NeatPopulation))


def program(population):
    if isinstance(population, nets.DensePopulation):
        return dense_program(population)
    from racing import neat
    if isinstance(population, neat.NeatPopulation):
        return neat_program(population)
    raise TypeError(f"the compiled simulator drives dense and NEAT populations, not {type(population).__name__}")


def _pointer(array, kind=P_DOUBLE):
    return array.ctypes.data_as(kind)


def _c_programs(programs, keep):
    arrays = {name: np.ascontiguousarray(getattr(programs, name),
                                         dtype=np.float64 if name in ("bias", "weight") else np.int64) for name in
              ("node_offset", "bias", "activation", "conn_offset", "source", "weight", "outputs")}
    keep.append(arrays)
    kinds = {"bias": P_DOUBLE, "weight": P_DOUBLE}
    return CPrograms(n_outputs=arrays["outputs"].shape[1],
                     **{name: _pointer(array, kinds.get(name, P_INT64)) for name, array in arrays.items()})


def forward(population, obs):
    """Output values (n, N_ACTIONS) of every car's network for its own row of obs, computed by the library; the
    load-time check and the tests compare them with numpy's."""
    return _forward(library(), population, obs)


def _forward(lib, population, obs):
    """forward() with a given library, for the check in available(), which holds _lock."""
    programs = program(population)
    obs = np.ascontiguousarray(obs, dtype=np.float64)
    n = len(population)
    if obs.shape != (n, config.N_INPUTS):
        raise ValueError(f"obs must have shape ({n}, {config.N_INPUTS}), got {obs.shape}")
    out = np.zeros((n, config.N_ACTIONS))
    keep = []
    lib.forward(ctypes.byref(_c_programs(programs, keep)), _pointer(obs), n, config.N_INPUTS, _pointer(out))
    return out


def _settings(grip, laps, keep):
    def doubles(values):
        array = np.ascontiguousarray(values, dtype=np.float64)
        keep.append(array)
        return _pointer(array)

    return Settings(dt=config.DT, grip=grip, yaw_min_speed=config.YAW_MIN_SPEED, speed_scale=config.SPEED_SCALE,
                    ray_range=config.RAY_RANGE, width_scale=config.WIDTH_SCALE, ray_min_step=config.RAY_MIN_STEP,
                    grid_res=config.GRID_RES, stall_time=config.STALL_TIME, time_eps=config.TIME_EPS,
                    expert_grip=config.GRIP, lookahead_base=config.EXPERT_LOOKAHEAD_BASE,
                    lookahead_time=config.EXPERT_LOOKAHEAD_TIME, grip_margin=config.EXPERT_GRIP_MARGIN,
                    speed_band=config.EXPERT_SPEED_BAND, expert_sight=config.EXPERT_SIGHT,
                    ray_max_iterations=config.RAY_MAX_ITERATIONS, laps=laps, n_inputs=config.N_INPUTS,
                    ray_angles=doubles(config.RAY_ANGLES), n_rays=config.N_RAYS,
                    steer_levels=doubles(config.STEER_LEVELS), n_steer=len(config.STEER_LEVELS),
                    throttle_levels=doubles(config.THROTTLE_LEVELS), n_throttle=len(config.THROTTLE_LEVELS))


def _c_track(track, time_limit, keep):
    arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in (track.dist, track.centerline, track.tangent,
                                                                   track.curvature)]
    keep.append(arrays)
    dist, centerline, tangent, curvature = arrays
    x0, y0 = track.start_position
    return Track(dist=_pointer(dist), rows=dist.shape[0], cols=dist.shape[1], origin_x=track.origin[0],
                 origin_y=track.origin[1], width=track.width, centerline=_pointer(centerline),
                 tangent=_pointer(tangent), curvature=_pointer(curvature), m=len(centerline), spacing=track.spacing,
                 start_x=x0, start_y=y0, start_heading=track.start_heading,
                 back=int(round(config.PROGRESS_BACK / track.spacing)),
                 ahead=int(round(config.PROGRESS_AHEAD / track.spacing)),
                 sight=int(round(config.EXPERT_SIGHT / track.spacing)), time_limit=float(time_limit))


def _run(tracks, programs, setup, time_limits, laps, grip, threads):
    """One library call for every track: programs None drives the expert. A list of sim.RunResult, one per track."""
    tracks, time_limits = list(tracks), list(time_limits)
    if len(time_limits) != len(tracks):
        raise ValueError(f"one time limit per track: {len(tracks)} tracks, {len(time_limits)} time limits")
    setup = np.ascontiguousarray(setup, dtype=np.float64)
    n = len(setup)
    if setup.shape != (n, 2):
        raise ValueError(f"setup must have shape (n, 2), got {setup.shape}")
    if programs is not None and len(programs.node_offset) != n + 1:
        raise ValueError(f"{len(programs.node_offset) - 1} network programs for {n} cars")
    keep = []
    c_tracks = (Track * max(len(tracks), 1))(*[_c_track(t, limit, keep) for t, limit in zip(tracks, time_limits)])
    c_settings = _settings(grip, laps, keep)
    c_programs = None if programs is None else ctypes.byref(_c_programs(programs, keep))
    shape = (len(tracks), n)
    finished, crashed = np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8)
    progress, finish_time = np.zeros(shape), np.full(shape, np.nan)
    laps_done, steps = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    c_results = Results(finished=_pointer(finished, P_UINT8), crashed=_pointer(crashed, P_UINT8),
                        progress=_pointer(progress), finish_time=_pointer(finish_time),
                        laps_done=_pointer(laps_done, P_INT64), steps=_pointer(steps, P_INT64))
    library().simulate(c_tracks, len(tracks), ctypes.byref(c_settings), c_programs, _pointer(setup), n, int(threads),
                       ctypes.byref(c_results))
    results = []
    for k, (track, limit) in enumerate(zip(tracks, time_limits)):
        cars = SimpleNamespace(progress=progress[k], laps=laps, finished=finished[k].astype(bool),
                               finish_time=finish_time[k])
        results.append(sim.RunResult(fitness=sim.fitness(cars, track, limit), finished=finished[k].astype(bool),
                                     crashed=crashed[k].astype(bool),
                                     progress=np.minimum(progress[k] / (track.length * laps), 1.0),
                                     finish_time=finish_time[k].copy(), steps=int(steps[k].max(initial=0)),
                                     laps_done=laps_done[k].copy()))
    return results


def simulate(tracks, population, time_limits, laps=1, grip=config.GRIP, threads=1):
    """sim.simulate(track, population.policy(), len(population), time_limit, laps, population.setup, grip) for every
    track, in one call with `threads` threads (spec §5.7). Returns one sim.RunResult per track."""
    return _run(tracks, program(population), population.setup, time_limits, laps, grip, threads)


def simulate_expert(tracks, n, time_limits, laps=1, threads=1):
    """sim.simulate(track, expert.expert_policy(track), n, time_limit, laps) for every track: n expert cars with the
    default setup, in one call (spec §5.7). Returns one sim.RunResult per track."""
    return _run(tracks, None, car_setup.default(n), time_limits, laps, config.GRIP, threads)


def expert_lap_time(track):
    """expert.expert_lap_time on the compiled simulator: the expert's lap time, or None if it does not finish."""
    result = simulate_expert([track], 1, [config.EXPERT_VALIDATION_TIME])[0]
    return float(result.finish_time[0]) if result.finished[0] else None
