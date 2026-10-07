import math
import pickle

import numpy as np

from racing import config
from racing.curriculum import (EVENTS, LevelTracker, MasteryTracker, SeedSource, StepSize, draw_seeds,
                               mutation_strength, success_fraction, survivors)

M = config.MASTERY_FRACTION


def test_mutation_strength_endpoints_and_midpoint():
    assert math.isclose(mutation_strength(0.0), config.GA_S_MAX)
    assert math.isclose(mutation_strength(1.0), config.GA_S_MIN)
    assert math.isclose(mutation_strength(0.5), (config.GA_S_MAX + config.GA_S_MIN) / 2)


def test_level_up_on_third_consecutive_qualifying_generation():
    t = LevelTracker()
    f = config.LEVEL_UP_FRACTION
    assert not t.update(f)
    assert not t.update(min(1.0, f + 0.2))
    assert t.update(1.0)
    assert t.level == 2


def test_streak_resets_below_the_threshold():
    t = LevelTracker()
    f = config.LEVEL_UP_FRACTION
    t.update(1.0)
    t.update(1.0)
    assert not t.update(f - 0.01)
    assert not t.update(1.0)
    assert not t.update(1.0)
    assert t.update(1.0)


def test_below_threshold_never_qualifies():
    t = LevelTracker()
    for _ in range(5):
        assert not t.update(config.LEVEL_UP_FRACTION - 0.01)
    assert t.level == 1


def test_level_stops_at_max():
    t = LevelTracker(level=config.MAX_LEVEL)
    for _ in range(5):
        assert not t.update(1.0)
    assert t.level == config.MAX_LEVEL


def test_draw_seeds_are_distinct_training_seeds():
    seeds = draw_seeds(np.random.default_rng(0))
    assert len(seeds) == config.TRACKS_PER_GEN == len(set(seeds))
    assert all(s in config.TRAIN_SEEDS for s in seeds)


# ----- revision 3 (§3.4, §4.2-4.5, §5.1-5.2) -----

def test_mutation_strength_without_children_is_s_max():
    assert mutation_strength(float("nan")) == config.GA_S_MAX
    assert mutation_strength(float("nan"), 0.2, 0.002) == 0.2


def test_seed_source_draws_fresh_distinct_seeds():
    source = SeedSource(np.random.default_rng(0))
    seeds = source.draw(3) + source.draw(3)
    assert len(set(seeds)) == 6 and all(config.TRAIN_SEED_START <= s < 2 ** 31 - 1 for s in seeds)
    assert source.state() == sorted(seeds)


class Scripted:
    """A generator stand-in that returns the given integers in order."""

    def __init__(self, values):
        self.values = list(values)

    def integers(self, low, high):
        return self.values.pop(0)


def test_seed_source_never_repeats_a_seed():
    source = SeedSource(Scripted([5, 5, 7, 9]), start=0, used=[7])
    assert source.draw(2) == [5, 9]
    assert source.state() == [5, 7, 9]


def test_seed_source_resumes_from_its_state():
    rng = np.random.default_rng(2)
    source = SeedSource(rng)
    source.draw(3)
    saved, used = rng.bit_generator.state, source.state()
    expected = source.draw(3)
    restored = np.random.default_rng()
    restored.bit_generator.state = saved
    assert SeedSource(restored, used=used).draw(3) == expected


def keep(fitness, all_success, lap_share, cap, floor, seed=0):
    return list(survivors(np.random.default_rng(seed), np.array(fitness, dtype=float),
                          np.array(all_success, dtype=bool), np.array(lap_share, dtype=float), cap, floor))


def test_successes_are_kept_fittest_first_up_to_the_cap():
    fitness, success = [1.2, 1.8, 0.5, 1.5, 1.1], [True, True, False, True, True]
    assert keep(fitness, success, [1, 1, 0, 1, 1], cap=3, floor=0) == [1, 3, 0]


def test_cars_without_laps_in_time_are_never_kept_by_chance():
    assert keep([0.9, 0.8, 0.7], [False] * 3, [0.0] * 3, cap=3, floor=0) == []


def test_order_is_successes_then_chance_then_floor():
    fitness = [0.2, 1.4, 0.9, 0.95, 0.1]
    success = [False, True, False, False, False]
    share = [0.0, 1.0, 1.0, 0.0, 0.0]                    # car 2 is kept by chance for certain, 0, 3 and 4 never
    assert keep(fitness, success, share, cap=4, floor=4) == [1, 2, 3, 0]


def test_chance_survival_matches_the_share_of_laps_in_time():
    n = 20000
    share = np.where(np.arange(n) % 2 == 0, 0.25, 0.75)
    kept = survivors(np.random.default_rng(3), np.linspace(0.9, 0.1, n), np.zeros(n, dtype=bool), share, n, 0)
    mask = np.zeros(n, dtype=bool)
    mask[kept] = True
    assert abs(mask[share == 0.25].mean() - 0.25) < 0.02 and abs(mask[share == 0.75].mean() - 0.75) < 0.02


def test_elite_floor_keeps_the_fittest_when_nobody_succeeds():
    kept = survivors(np.random.default_rng(4), np.arange(100.0), np.zeros(100, dtype=bool), np.zeros(100), 90, 50)
    assert sorted(kept) == list(range(50, 100))


def test_floor_never_exceeds_the_cap_in_small_populations():
    kept = survivors(np.random.default_rng(4), np.arange(20.0), np.zeros(20, dtype=bool), np.zeros(20), 18, 50)
    assert len(kept) == 18 and sorted(kept) == list(range(2, 20))


def test_survivors_take_one_draw_per_car():
    rng = np.random.default_rng(5)
    survivors(rng, np.ones(7), np.zeros(7, dtype=bool), np.full(7, 0.5), 6, 0)
    reference = np.random.default_rng(5)
    reference.random(7)
    assert rng.random() == reference.random()


def test_three_passing_generations_advance_a_level():
    t = MasteryTracker()
    assert [t.update(M), t.update(1.0), t.update(0.95)] == [None, None, "level_up"]
    assert t.level == 2 and t.streak == 0 and not t.goal_reached


def test_a_failing_generation_resets_the_streak():
    t = MasteryTracker()
    t.update(1.0)
    t.update(1.0)
    assert t.update(M - 0.01) is None and t.streak == 0
    assert [t.update(1.0) for _ in range(3)] == [None, None, "level_up"]


def test_goal_on_the_last_level_starts_the_speed_push():
    t = MasteryTracker(level=config.MAX_LEVEL)
    assert [t.update(1.0) for _ in range(3)] == [None, None, "goal"]
    assert t.goal_reached and t.level == config.MAX_LEVEL and t.target_reduction == 0.0
    assert t.mastered_factor == config.TARGET_FACTOR
    assert math.isclose(t.factor, config.TARGET_FACTOR * config.SPEED_STEP)


def test_each_mastered_target_is_tightened_by_the_speed_step():
    t = MasteryTracker(level=config.MAX_LEVEL)
    events = [t.update(1.0) for _ in range(9)]
    assert events.count("goal") == 1 and events.count("tightened") == 2 and set(events) <= set(EVENTS) | {None}
    assert math.isclose(t.mastered_factor, config.TARGET_FACTOR * config.SPEED_STEP ** 2)
    assert math.isclose(t.target_reduction, 1 - config.SPEED_STEP ** 2)


def test_speed_push_ends_after_patience_without_mastery(monkeypatch):
    monkeypatch.setattr(config, "SPEED_PATIENCE", 4)
    t = MasteryTracker(level=config.MAX_LEVEL)
    for _ in range(3):
        t.update(1.0)
    assert [t.update(0.5) for _ in range(4)] == [None, None, None, "end_speed"]
    assert t.done and t.outcome == "speed_limit" and t.update(1.0) is None


def test_mastery_wins_over_the_end_of_the_speed_push(monkeypatch):
    monkeypatch.setattr(config, "SPEED_PATIENCE", 3)
    t = MasteryTracker(level=config.MAX_LEVEL)
    for _ in range(3):
        t.update(1.0)
    assert [t.update(1.0) for _ in range(3)] == [None, None, "tightened"] and not t.done


def test_safety_stop_ends_a_run_that_stops_improving(monkeypatch):
    """Review focus 5: a population that never reaches 90% ends with the outcome 'stalled'."""
    monkeypatch.setattr(config, "SAFETY_PATIENCE", 5)
    t = MasteryTracker()
    assert [t.update(0.5) for _ in range(6)] == [None] * 5 + ["end_stalled"]
    assert t.done and t.outcome == "stalled" and t.level == 1 and not t.goal_reached
    assert math.isnan(t.target_reduction) and t.update(1.0) is None


def test_improvement_postpones_the_safety_stop(monkeypatch):
    monkeypatch.setattr(config, "SAFETY_PATIENCE", 3)
    t = MasteryTracker()
    assert [t.update(s) for s in (0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2)] == [None] * 7
    assert [t.update(0.2), t.update(0.2)] == [None, "end_stalled"]


def test_level_up_restarts_the_safety_counters(monkeypatch):
    monkeypatch.setattr(config, "SAFETY_PATIENCE", 3)
    t = MasteryTracker()
    assert [t.update(0.95) for _ in range(3)] == [None, None, "level_up"]
    assert [t.update(0.5) for _ in range(4)] == [None, None, None, "end_stalled"]


def test_tracker_state_round_trip():
    t = MasteryTracker()
    for share in (0.95, 0.4, 0.95):
        t.update(share)
    copy = MasteryTracker.from_state(pickle.loads(pickle.dumps(t.state())))
    for share in (0.95, 0.95, 0.95, 0.3):
        assert copy.update(share) == t.update(share)
    assert str(copy.state()) == str(t.state())


def test_step_size_follows_the_one_fifth_rule_while_learning():
    s = StepSize(0.1)
    assert math.isclose(s.update(0.5, 0.0, 0.3), 0.1 / config.ES_FACTOR)
    s.reset()
    assert math.isclose(s.update(0.5, 0.0, 0.1), 0.1 * config.ES_FACTOR)
    s.reset()
    assert s.update(0.5, 0.0, 0.2) == 0.1
    assert s.update(float("nan"), float("nan"), float("nan")) == 0.1


def test_step_size_follows_the_children_once_the_parents_are_reliable():
    s = StepSize(0.1)
    assert math.isclose(s.update(0.95, 0.5, 0.9), 0.1 * config.ES_FACTOR)      # the 1/5 rule would grow it
    s.reset()
    assert math.isclose(s.update(M, 0.92, 0.0), 0.1 / config.ES_FACTOR)       # the 1/5 rule would shrink it
    s.reset()
    assert s.update(1.0, float("nan"), 0.5) == 0.1


def test_step_size_is_clipped_and_round_trips():
    assert StepSize(config.ES_SIGMA_MAX).update(0.0, 0.0, 1.0) == config.ES_SIGMA_MAX
    assert StepSize(config.ES_SIGMA_MIN).update(0.0, 0.0, 0.0) == config.ES_SIGMA_MIN
    s = StepSize(0.3)
    s.update(0.0, 0.0, 0.0)
    copy = StepSize.from_state(s.state())
    assert copy.sigma == s.sigma and copy.sigma0 == 0.3
    copy.reset()
    assert copy.sigma == 0.3


def test_success_fraction_counts_ties_as_half():
    child, parent = np.array([2.0, 1.0, 3.0, 0.0]), np.array([1.0, 1.0, 1.0, 1.0])
    assert success_fraction(child, parent) == (1.0 + 0.5 + 1.0 + 0.0) / 4
