"""Curriculum rules (spec §9.1-9.2): level advancement, GA mutation strength, per-generation tracks."""
import math

import numpy as np

from racing import config


def mutation_strength(new_finish, s_max=config.GA_S_MAX, s_min=config.GA_S_MIN):
    """High when few new individuals finish; s_min when all of them do."""
    return s_min + (s_max - s_min) * (1.0 - new_finish)


class LevelTracker:
    """Advances one level after `streak_needed` consecutive qualifying generations."""

    def __init__(self, level=1, streak_needed=config.LEVEL_UP_STREAK,
                 new_threshold=config.LEVEL_UP_NEW_FRACTION, max_level=config.MAX_LEVEL):
        self.level = level
        self.streak = 0
        self.streak_needed = streak_needed
        self.new_threshold = new_threshold
        self.max_level = max_level

    def update(self, kept_finish, new_finish):
        """Record one generation's finishing fractions. Returns True if the level advanced."""
        if self.level >= self.max_level:
            return False
        qualifies = (not math.isnan(kept_finish)) and kept_finish >= 1.0 and new_finish >= self.new_threshold
        if qualifies:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.streak_needed:
            self.level += 1
            self.streak = 0
            return True
        return False


def draw_seeds(rng):
    """Training seeds for one generation's tracks, drawn without replacement."""
    seeds = rng.choice(np.array(list(config.TRAIN_SEEDS)), config.TRACKS_PER_GEN, replace=False)
    return [int(s) for s in seeds]
