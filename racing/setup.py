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
    """n rows of revision 3's setup (config.DEFAULT_SETUP)."""
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
    """The (2,) float64 setup of one car: a copy of `value`, or revision 3's for None."""
    return as_rows(None if value is None else np.asarray(value, dtype=float)[None, :], 1)[0]


def random(rng, n):
    """n setups drawn uniformly on the log scale within the limits (spec §9.11): one rng.uniform call."""
    low, high = _limits()
    return np.exp(rng.uniform(np.log(low), np.log(high), (int(n), 2)))


def clip(values):
    """Setups moved into the limits of spec §9.11."""
    low, high = _limits()
    return np.clip(np.asarray(values, dtype=float), low, high)


def crossover(rng, a, b):
    """Each value from setup a or setup b with equal probability (spec §9.11): one rng.random call."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.where(rng.random(a.shape) < 0.5, a, b)


def mutate(rng, values):
    """Each value, with probability SETUP_MUTATION_P, multiplied by exp(N(0, SETUP_MUTATION_SD)), then clipped
    (spec §9.11). Always draws two arrays of values.shape, the mask and then the noise."""
    values = np.asarray(values, dtype=float)
    mask = rng.random(values.shape) < config.SETUP_MUTATION_P
    noise = rng.normal(0.0, config.SETUP_MUTATION_SD, values.shape)
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
