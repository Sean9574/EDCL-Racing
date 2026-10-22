"""Car setups (spec §9.11): each car's sensor length and steering sensitivity, as one row of an (n, 2)
array next to its network. Every method draws, inherits and mutates setups with these rules, so setup evolution
favours no method. Values change on the natural-log scale, so halving and doubling are equally large steps."""
import numpy as np

from racing import config

COLUMNS = ("sensor_length", "steer_gain")


def _limits():
    """(low, high) arrays of the two columns, read from config at call time."""
    low = np.array([config.SENSOR_LENGTH_LIMITS[0], config.STEER_GAIN_LIMITS[0]], dtype=float)
    high = np.array([config.SENSOR_LENGTH_LIMITS[1], config.STEER_GAIN_LIMITS[1]], dtype=float)
    return low, high


def default(n):
    """n rows of the default setup (config.DEFAULT_SETUP: 150 m sensors, k = 1)."""
    return np.tile(np.asarray(config.DEFAULT_SETUP, dtype=float), (int(n), 1))


def as_rows(values, n):
    """The (n, 2) float64 setup array of n cars: a copy of `values`, or default(n) for None."""
    if values is None:
        return default(n)
    values = np.array(values, dtype=float)
    if values.shape != (int(n), 2):
        raise ValueError(f"setup must have shape ({n}, 2), one (sensor length, steering sensitivity) row per car; "
                         f"got {values.shape}")
    return values


def as_row(value):
    """The (2,) float64 setup of one car: a copy of `value`, or the default setup for None."""
    return as_rows(None if value is None else np.asarray(value, dtype=float)[None, :], 1)[0]


def random(rng, n):
    """n setups of a random-start population (spec §9.11): every car's sensor length is SENSOR_START, and its steering
    sensitivity is drawn uniformly on the log scale within STEER_GAIN_LIMITS (one rng.uniform call)."""
    low, high = config.STEER_GAIN_LIMITS
    gain = np.exp(rng.uniform(np.log(low), np.log(high), int(n)))
    return np.column_stack([np.full(int(n), config.SENSOR_START), gain])


def clip(values):
    """Setups moved into the limits of spec §9.11."""
    low, high = _limits()
    return np.clip(np.asarray(values, dtype=float), low, high)


def crossover(rng, a, b):
    """Each value from setup a or setup b with equal probability (spec §9.11): one rng.random call."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.where(rng.random(a.shape) < 0.5, a, b)


def mutate(rng, values, boost=False):
    """Each value, with probability SETUP_MUTATION_P, multiplied by exp(N(0, SETUP_MUTATION_SD)), then clipped
    (spec §9.11). With boost (a stall, spec §9.4) the steering sensitivity mutates with probability STALL_STEER_P and
    standard deviation STALL_STEER_SD; the sensor length as usual. Always draws the same two arrays of values.shape,
    the uniforms and then the standard normals, so a boost changes no other draw."""
    values = np.asarray(values, dtype=float)
    p = np.full(values.shape[-1], config.SETUP_MUTATION_P)
    sd = np.full(values.shape[-1], config.SETUP_MUTATION_SD)
    if boost:
        p[1], sd[1] = config.STALL_STEER_P, config.STALL_STEER_SD
    mask = rng.random(values.shape) < p
    noise = rng.normal(0.0, 1.0, values.shape) * sd
    return clip(np.where(mask, values * np.exp(noise), values))


def to_dict(row):
    """One setup as JSON-ready values: {"sensor_length": m, "steer_gain": k}."""
    return {name: float(value) for name, value in zip(COLUMNS, np.asarray(row, dtype=float))}


def from_dict(d):
    """The (2,) setup row of to_dict's form."""
    return np.array([float(d[name]) for name in COLUMNS])


def summary(values):
    """Mean and standard deviation of both setup values of a population, for logs and the application."""
    values = np.asarray(values, dtype=float)
    return {"sensor_mean": float(values[:, 0].mean()), "sensor_std": float(values[:, 0].std()),
            "steer_mean": float(values[:, 1].mean()), "steer_std": float(values[:, 1].std())}
