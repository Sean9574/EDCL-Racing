"""Curriculum rules (spec §4.6, §9.3-9.4, §9.6-9.7): fresh training seeds and the real training layouts, survival,
levels with the mastery goal, the speed push and the safety stop, GA mutation strength and the ES/NEAT step size."""
import numpy as np

from racing import config

EVENTS = ("level_up", "goal", "tightened", "end_speed", "end_stalled", "hardened")


class SeedSource:
    """Training maps (spec §4.6): seeds of freshly generated maps, drawn by the run's random generator from `start`
    upwards and never repeated within the run, and at Extra hard the real training layouts. `used` restores the seeds
    of a resumed run."""

    def __init__(self, rng, start=config.TRAIN_SEED_START, used=()):
        self.rng = rng
        self.start = int(start)
        self.used = {int(seed) for seed in used}

    def draw(self, count, level=None):
        """`count` maps for a generation at `level` (spec §4.6): at Extra hard, distinct layouts of the 38 real
        training layouts (they repeat between generations); otherwise fresh seeds never used before in the run."""
        if level == config.MAX_LEVEL:
            from racing import circuits
            return [int(k) for k in self.rng.choice(circuits.N_LAYOUTS, count, replace=False)]
        seeds = []
        while len(seeds) < count:
            seed = int(self.rng.integers(self.start, 2 ** 31 - 1))
            if seed not in self.used:
                self.used.add(seed)
                seeds.append(seed)
        return seeds

    def state(self):
        return sorted(self.used)


def survivors(rng, fitness, all_success, lap_share, cap, floor):
    """Indices of the cars kept unchanged (spec §9.3): every car that succeeded on every map, fittest first;
    then each other car with probability equal to its share of laps in time, fittest first, while fewer than `cap`
    are kept; then, if fewer than min(floor, cap) are kept, the fittest remaining cars. One rng.random(n) draw per
    call, so the result depends only on the generator's state."""
    n = len(fitness)
    order = np.argsort(-np.asarray(fitness, dtype=float), kind="stable")
    lucky = rng.random(n) < np.asarray(lap_share, dtype=float)
    chosen = [int(i) for i in order if all_success[i]][:cap]
    taken = set(chosen)
    for i in order:
        if len(chosen) >= cap:
            break
        if int(i) not in taken and lucky[i]:
            chosen.append(int(i))
            taken.add(int(i))
    for i in order:
        if len(chosen) >= min(floor, cap):
            break
        if int(i) not in taken:
            chosen.append(int(i))
            taken.add(int(i))
    return np.array(chosen, dtype=np.int64)


def mastery_fraction(speed_push):
    """Lowest map success share of a passing generation: MASTERY_FRACTION on the levels, SPEED_MASTERY_FRACTION in
    the speed push (spec §9.4)."""
    return config.SPEED_MASTERY_FRACTION if speed_push else config.MASTERY_FRACTION


class MasteryTracker:
    """Levels, the mastery goal, the speed push, hardening and the safety stop (spec §9.4). update() takes the lowest
    map success share of one generation and returns one of EVENTS or None."""

    def __init__(self, level=1, factor=config.TARGET_FACTOR):
        self.level = int(level)
        self.factor = float(factor)
        self.streak = 0
        self.goal_reached = False
        self.hardening = False          # the run hardens (spec §9.4, version 3.15)
        self.mastered_factor = float("nan")
        self.outcome = None             # None, "hardened", "stalled" or "time_limit" (an end request, set by
                                        # evolution.run); "speed_limit" to version 3.14
        self.recent = []                # the level's latest lowest shares (at most MASTERY_STREAK, i.e. the three
                                        # generations of spec §9.4), for the safety stop
        self.best_mean = float("-inf")
        self.since_best = 0
        self.since_mastered = 0

    @property
    def done(self):
        return self.outcome is not None

    @property
    def speed_push(self):
        """Whether a generation is judged by the speed push's rule (spec §9.4): after the goal, before hardening."""
        return self.goal_reached and not self.hardening

    @property
    def stalled(self):
        """The speed push has gone STALL_GENERATIONS generations without mastering its target (spec §9.4): the
        stall boost is on until the next mastered target. Unrelated to outcome == "stalled" (the safety stop before the
        goal)."""
        return self.speed_push and not self.done and self.since_mastered >= config.STALL_GENERATIONS

    @property
    def target_reduction(self):
        """How much shorter the tightest mastered target is than the starting target; nan before the goal."""
        return 1.0 - self.mastered_factor / config.TARGET_FACTOR

    def _master(self):
        self.mastered_factor = self.factor
        self.factor *= config.SPEED_STEP
        self.streak = 0
        self.since_mastered = 0

    def harden(self):
        """Start hardening (spec §9.4, version 3.15): every later generation is judged by the level rule at the last
        mastered target, which no longer tightens; before the goal the levels go on and the goal ends hardening. Clears
        the outcome of a run that ended before version 3.15."""
        self.hardening = True
        self.outcome = None
        if self.goal_reached:
            self.factor = self.mastered_factor
        self.streak = 0
        self.since_mastered = 0

    def update(self, min_share):
        if self.done:
            return None
        self.streak = self.streak + 1 if min_share >= mastery_fraction(self.speed_push) else 0
        if self.hardening and self.goal_reached:
            if self.streak >= config.MASTERY_STREAK:
                self.outcome = "hardened"
                return "hardened"
            return None
        if self.goal_reached:
            if self.streak >= config.MASTERY_STREAK:
                self._master()
                return "tightened"
            self.since_mastered += 1
            if self.since_mastered >= config.SPEED_PATIENCE:
                self.harden()
                return "end_speed"
            return None
        if self.streak >= config.MASTERY_STREAK:
            if self.level < config.MAX_LEVEL:
                self.level += 1
                self.streak = 0
                self.recent, self.best_mean, self.since_best = [], float("-inf"), 0
                return "level_up"
            self.goal_reached = True
            if self.hardening:                  # a run hardening before its goal ends at it
                self.mastered_factor = self.factor
                self.outcome = "hardened"
                return "hardened"
            self._master()
            return "goal"
        self.recent = (self.recent + [float(min_share)])[-config.MASTERY_STREAK:]
        mean = float(np.mean(self.recent))
        if mean > self.best_mean:
            self.best_mean, self.since_best = mean, 0
        else:
            self.since_best += 1
            if config.SAFETY_PATIENCE is not None and self.since_best >= config.SAFETY_PATIENCE:
                self.outcome = "stalled"
                return "end_stalled"
        return None

    def state(self):
        return dict(vars(self))

    @classmethod
    def from_state(cls, state):
        tracker = cls()
        tracker.__dict__.update(state)
        tracker.recent = list(state["recent"])
        return tracker


def mutation_strength(child_success, s_max=config.GA_S_MAX, s_min=config.GA_S_MIN):
    """GA mutation strength (spec §9.6): high when few children succeeded, s_min when all did; s_max when there
    were no children to measure (nan)."""
    if np.isnan(child_success):
        return s_max
    return s_min + (s_max - s_min) * (1.0 - child_success)


def success_fraction(child_fitness, parent_fitness):
    """Fraction of offspring that beat their own parent; a tie with the parent counts as half a success."""
    return float(np.mean(np.where(child_fitness > parent_fitness, 1.0,
                                  np.where(child_fitness == parent_fitness, 0.5, 0.0))))


class StepSize:
    """ES and NEAT step size σ (spec §9.7): the 1/5 success rule while the population learns; once at least
    MASTERY_FRACTION of the parents succeed, σ shrinks while fewer than MASTERY_FRACTION of the children succeed and
    grows otherwise. Kept within [ES_SIGMA_MIN, ES_SIGMA_MAX]; a nan input leaves σ unchanged."""

    def __init__(self, sigma0):
        self.sigma0 = float(sigma0)
        self.sigma = float(sigma0)

    def update(self, parent_success, child_success, success_rate):
        if parent_success >= config.MASTERY_FRACTION:
            if child_success < config.MASTERY_FRACTION:
                self.sigma *= config.ES_FACTOR
            elif child_success >= config.MASTERY_FRACTION:
                self.sigma /= config.ES_FACTOR
        elif success_rate > 0.2:
            self.sigma /= config.ES_FACTOR
        elif success_rate < 0.2:
            self.sigma *= config.ES_FACTOR
        self.sigma = float(np.clip(self.sigma, config.ES_SIGMA_MIN, config.ES_SIGMA_MAX))
        return self.sigma

    def reset(self):
        self.sigma = self.sigma0

    def state(self):
        return {"sigma0": self.sigma0, "sigma": self.sigma}

    @classmethod
    def from_state(cls, state):
        step = cls(state["sigma0"])
        step.sigma = float(state["sigma"])
        return step
