"""Compiled simulator (spec §5.7): time trials of dense and NEAT populations and the expert's lap, computed by
racing/fastsim.cpp exactly as racing/sim.py computes them. The library is built with g++ the first time it is needed
and again whenever fastsim.cpp is newer; without g++ the numpy simulator runs instead, with identical results."""
import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from racing import config, nets, sim
from racing import setup as car_setup

SOURCE = Path(__file__).with_name("fastsim.cpp")
LIBRARY = Path(__file__).with_name("_fastsim.so")
COMPILER = "g++"
# -ffp-contract=off: no fused multiply-add, so every product and sum rounds as it does in numpy (spec §5.7)
FLAGS = ["-O3", "-march=native", "-ffp-contract=off", "-std=c++17", "-shared", "-fPIC", "-pthread"]

_library = None          # the loaded library
_failure = None          # the error of a failed build or load in this process; the numpy simulator runs instead


def build(force=False):
    """Compiles fastsim.cpp into LIBRARY unless the library is newer than the source. The compiler writes a file of
    this process's own, which is then renamed into place, so runs that start at once never load a half-written
    library (spec §5.7)."""
    if not force and LIBRARY.exists() and LIBRARY.stat().st_mtime_ns >= SOURCE.stat().st_mtime_ns:
        return
    temporary = LIBRARY.with_name(f"{LIBRARY.name}.{os.getpid()}.tmp")
    try:
        subprocess.run([COMPILER, *FLAGS, str(SOURCE), "-o", str(temporary)], check=True, capture_output=True,
                       text=True)
        os.replace(temporary, LIBRARY)
    finally:
        temporary.unlink(missing_ok=True)


def _load():
    build()
    try:
        return ctypes.CDLL(str(LIBRARY))
    except OSError:                                  # a damaged or foreign library: build it again
        build(force=True)
        return ctypes.CDLL(str(LIBRARY))


def library():
    """The loaded library, built first when needed. Raises OSError or CalledProcessError if it cannot be built."""
    global _library
    if _library is None:
        lib = _load()
        _declare(lib)
        _library = lib
    return _library


def available():
    """True when the compiled simulator runs: EC_RACING_SIM is not "numpy" and the library builds and loads. A failed
    build prints one warning with the compiler's message, and the numpy simulator runs instead (spec §5.7)."""
    global _failure
    if config.SIMULATOR == "numpy" or _failure is not None:
        return False
    try:
        library()
    except (OSError, subprocess.CalledProcessError) as error:
        _failure = error
        detail = (getattr(error, "stderr", None) or str(error)).strip()
        print(f"EC Racing: the compiled simulator is unavailable ({detail}); using the numpy simulator",
              file=sys.stderr)
        return False
    return True


def description():
    """The simulator in use, for results/run_info.json: "compiled (g++ <version>)" or "numpy"."""
    if not available():
        return "numpy"
    version = subprocess.run([COMPILER, "-dumpfullversion"], capture_output=True, text=True).stdout.strip()
    return f"compiled (g++ {version})"


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


def supports(population):
    """True for the populations the compiled simulator drives: dense and NEAT (spec §5.7)."""
    from racing import neat
    return isinstance(population, (nets.DensePopulation, neat.NeatPopulation))


def program(population):
    if isinstance(population, nets.DensePopulation):
        return dense_program(population)
    raise TypeError(f"the compiled simulator drives dense and NEAT populations, not {type(population).__name__}")


def _pointer(array, kind=P_DOUBLE):
    return array.ctypes.data_as(kind)


def _c_programs(programs, keep):
    arrays = {name: np.ascontiguousarray(getattr(programs, name)) for name in
              ("node_offset", "bias", "activation", "conn_offset", "source", "weight", "outputs")}
    keep.append(arrays)
    kinds = {"bias": P_DOUBLE, "weight": P_DOUBLE}
    return CPrograms(n_outputs=arrays["outputs"].shape[1],
                     **{name: _pointer(array, kinds.get(name, P_INT64)) for name, array in arrays.items()})


def forward(population, obs):
    """Output values (n, N_ACTIONS) of every car's network for its own row of obs, computed by the library; the
    tests compare them with numpy's."""
    programs = program(population)
    obs = np.ascontiguousarray(obs, dtype=np.float64)
    n = len(population)
    if obs.shape != (n, config.N_INPUTS):
        raise ValueError(f"obs must have shape ({n}, {config.N_INPUTS}), got {obs.shape}")
    out = np.zeros((n, config.N_ACTIONS))
    keep = []
    library().forward(ctypes.byref(_c_programs(programs, keep)), _pointer(obs), n, config.N_INPUTS, _pointer(out))
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
