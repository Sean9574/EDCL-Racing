"""Shared run loop of every evolutionary method (revision 3 §4, §8.2, §8.3). A method object holds the population
and makes the children (racing/ga.py, racing/es.py, racing/neat.py); this loop drives fresh maps as time trials,
applies the survival rule, follows the mastery levels and the speed push, logs every generation, saves a checkpoint
after each one, reports progress and selects the final controller."""
import csv
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from racing import checkpoint, config, population, progress
from racing.curriculum import MasteryTracker, SeedSource, survivors
from racing.nets import DensePopulation

LOG_NAME = "log.csv"
SUMMARY_NAME = "summary.json"
LOG_FIELDS = ["generation", "level", "factor", "evals", "best", "mean", "worst", "map_shares", "min_share", "n_kept",
              "kept_success", "child_success", "median_lap", "step_size", "streak", "event", "seeds", "elapsed"]


class Method(Protocol):
    """What the loop needs from a method. `population` holds the cars carried over unchanged first; `is_kept`
    marks them. `advance` must assign a new population object (reports keep the evaluated one)."""
    approach: str
    population: object
    is_kept: np.ndarray
    step_size: float

    def initialize(self, rng) -> None: ...
    def advance(self, rng, result, kept, event) -> None: ...
    def state(self) -> dict: ...
    def load_state(self, state) -> None: ...
    def save_champion(self, index, seed, champion_dir) -> Path: ...
    def log_fields(self) -> dict: ...


def share(mask):
    """Fraction of True values; nan for an empty selection."""
    return float(np.mean(mask)) if len(mask) else float("nan")


def children_success(result, is_kept):
    """Share of this generation's new children (the cars not carried over) that succeeded on every map; nan if
    there are none."""
    return share(result.all_success[~is_kept])


class DenseMethod:
    """What GA and ES share: a dense population of at least two cars, survivors first and unchanged, saving the
    final controller through population.save_dense_champion."""
    approach = None

    def __init__(self, pop_size, hidden, activation):
        if pop_size < 2:
            raise ValueError(f"population size ({pop_size}) must be at least 2")
        self.pop_size = pop_size
        self.hidden = tuple(hidden)
        self.activation = activation
        self.population = None
        self.is_kept = None

    def assemble(self, kept_weights, children):
        """The next population: the kept cars first, then the children; marks the kept cars."""
        self.population = DensePopulation(np.concatenate([kept_weights, children]), self.hidden, self.activation)
        self.is_kept = np.arange(len(self.population)) < len(kept_weights)

    def save_champion(self, index, seed, champion_dir):
        return population.save_dense_champion(self.approach, seed, self.population, index, champion_dir)

    def log_fields(self):
        return {}


def median_lap(result):
    """Median lap time of the successful time trials; nan if none succeeded."""
    times = result.lap_times[result.success]
    return float(np.median(times)) if times.size else float("nan")


def plain(value):
    """A float for JSON: None instead of nan."""
    value = float(value)
    return None if np.isnan(value) else value


def start_log(path, fields=LOG_FIELDS):
    """A new log whose columns are `fields`: LOG_FIELDS, then the method's extra columns (Method.log_fields)."""
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(fields)


def log_header(path):
    """The columns of an existing log."""
    with open(path, newline="") as f:
        return next(csv.reader(f))


def append_log(path, report, fields=LOG_FIELDS):
    row = []
    for key in fields:
        value = report[key]
        if key in ("seeds", "map_shares"):
            value = " ".join(str(v) for v in value)
        elif key == "event":
            value = value or ""
        row.append(value)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)
        f.flush()
        os.fsync(f.fileno())           # on disk before the generation's checkpoint, which covers it (revision 3 §8.2)


def complete_row(row, header, generation):
    """Whether a log row is whole and belongs to a generation before `generation`: a crash can cut the last row
    short or, after a reboot, leave zero bytes in its place."""
    return len(row) == len(header) and row[0].isascii() and row[0].isdigit() and int(row[0]) < generation


def truncate_log(path, generation, fields=LOG_FIELDS):
    """Keep the header and the whole rows of the generations before `generation`. A row written after the last
    checkpoint belongs to a generation that runs again when the run resumes, and a damaged row is dropped
    (revision 3 §8.2). A missing log is started with the columns `fields`."""
    path = Path(path)
    if not path.exists():
        start_log(path, fields)
        return
    with open(path, newline="", errors="replace") as f:
        rows = list(csv.reader(f))
    header = rows[0] if rows and rows[0][:1] == fields[:1] else list(fields)
    kept = [header] + [row for row in rows[1:] if complete_row(row, header, generation)]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", newline="") as f:
        csv.writer(f).writerows(kept)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def label(method, seed):
    return f"{config.LABELS.get(method.approach, method.approach)} seed {seed}"


def milestone_text(event, method, seed, generation, level, tracker):
    name = label(method, seed)
    if event == "level_up":
        return (f"{name}: {config.DIFFICULTIES[level]} mastered at generation {generation}, "
                f"now {config.DIFFICULTIES[tracker.level]}")
    if event == "goal":
        return f"{name}: mastery goal reached at generation {generation}, speed push starts"
    if event == "tightened":
        return f"{name}: target {100 * tracker.target_reduction:.1f}% shorter mastered at generation {generation}"
    if event == "end_speed":
        return (f"{name}: speed push ended at generation {generation}, "
                f"target {100 * tracker.target_reduction:.1f}% shorter")
    return f"{name}: stopped at generation {generation}, {config.DIFFICULTIES[level]} not mastered"


def run(method, seed, out_dir, champion_dir=config.CHAMPIONS_DIR, n_jobs=1, on_generation=None, start_level=1,
        max_generations=None, rng_seed=None):
    """Train `method` with run seed `seed` until the speed push ends or the safety stop (revision 3 §4.4-4.5), or
    after `max_generations` (viewer, pilots, tests). A completed run in `out_dir` with the current settings is
    returned at once; an interrupted one resumes from its checkpoint. With champion_dir None no final controller is
    selected or saved and nothing is written to the status file or the run log: only the experiment's runs report
    progress. Returns the summary."""
    out_dir = Path(out_dir)
    done = checkpoint.completed(out_dir, SUMMARY_NAME)
    if done is not None:
        return done
    reporting = champion_dir is not None
    population.fresh_workers(n_jobs)                   # every run builds its own tracks (spec §9.1)
    began = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / LOG_NAME
    state = checkpoint.load(out_dir / checkpoint.CHECKPOINT_NAME)
    if state is None:
        rng = np.random.default_rng(seed if rng_seed is None else rng_seed)
        method.initialize(rng)
        seeds = SeedSource(rng)
        tracker = MasteryTracker(level=start_level)
        generation, evals, elapsed, levels, tightenings = 0, 0, 0.0, [], []
        pop_size = len(method.population)
        if champion_dir is not None:
            population.clear_champion(method.approach, seed, champion_dir)
        fields = LOG_FIELDS + list(method.log_fields())   # the method's extra columns, fixed for the whole run
        start_log(log_path, fields)
        if reporting:
            progress.milestone(f"{label(method, seed)}: started")
    else:
        checkpoint.check_settings(state["settings"])
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        method.load_state(state["method"])
        seeds = SeedSource(rng, used=state["seeds"])
        tracker = MasteryTracker.from_state(state["tracker"])
        generation, evals, elapsed = state["generation"], state["evals"], state["elapsed"]
        levels, tightenings, start_level = state["levels"], state["tightenings"], state["start_level"]
        pop_size = len(method.population)
        truncate_log(log_path, generation, LOG_FIELDS + list(method.log_fields()))
        fields = log_header(log_path)                     # a resumed run keeps the columns its log started with
        if reporting:
            progress.milestone(f"{label(method, seed)}: resumed at generation {generation}")

    def active():
        return elapsed + time.time() - began        # run time, accumulated across resumptions

    while not tracker.done and (max_generations is None or generation < max_generations):
        level, factor = tracker.level, tracker.factor
        maps = seeds.draw(config.TRACKS_PER_GEN)
        result = population.evaluate_population(method.population, maps, level, factor, n_jobs=n_jobs)
        n = len(method.population)
        evals += n
        cap = int(config.KEEP_MAX_FRACTION * n)
        kept = survivors(rng, result.fitness, result.all_success, result.lap_share, cap, config.ELITE_FLOOR)
        event = tracker.update(result.min_share)
        now = active()
        report = {"approach": method.approach, "seed": seed, "generation": generation, "level": level,
                  "factor": factor, "target_reduction": tracker.target_reduction, "streak": tracker.streak,
                  "seeds": maps, "population": method.population, "fitness": result.fitness,
                  "all_success": result.all_success, "is_kept": method.is_kept, "map_shares": result.map_shares,
                  "min_share": result.min_share, "n_kept": len(kept),
                  "kept_success": share(result.all_success[method.is_kept]),
                  "child_success": children_success(result, method.is_kept), "median_lap": median_lap(result),
                  "step_size": float(method.step_size), "best": float(result.fitness.max()),
                  "mean": float(result.fitness.mean()), "worst": float(result.fitness.min()), "evals": evals,
                  "elapsed": now, "event": event, **method.log_fields()}
        append_log(log_path, report, fields)
        if event in ("level_up", "goal"):
            levels.append({"level": int(level), "generation": generation, "evals": evals, "elapsed": now})
        elif event == "tightened":
            tightenings.append({"factor": float(tracker.mastered_factor), "generation": generation, "evals": evals,
                                "elapsed": now})
        if reporting:
            if event is not None:
                progress.milestone(milestone_text(event, method, seed, generation, level, tracker))
            progress.write_status({"approach": method.approach, "seed": seed,
                                   "stage": "speed push" if tracker.goal_reached else "evolution",
                                   "level": int(level), "generation": generation, "min_share": result.min_share,
                                   "streak": tracker.streak, "target_reduction": tracker.target_reduction,
                                   "elapsed": now})
        if on_generation is not None:
            on_generation(report)
        method.advance(rng, result, kept, event)
        generation += 1
        checkpoint.save(out_dir / checkpoint.CHECKPOINT_NAME, {
            "settings": checkpoint.settings_record(), "generation": generation, "evals": evals,
            "elapsed": active(), "rng": rng.bit_generator.state, "seeds": seeds.state(),
            "tracker": tracker.state(), "method": method.state(), "levels": levels, "tightenings": tightenings,
            "start_level": start_level})

    summary = {"approach": method.approach, "seed": seed, "settings": checkpoint.settings_record(),
               "generations": generation, "evals": evals, "pop_size": pop_size,
               "start_level": start_level, "level_reached": int(tracker.level),
               "goal_reached": bool(tracker.goal_reached), "outcome": tracker.outcome,
               "mastered_factor": plain(tracker.mastered_factor), "target_reduction": plain(tracker.target_reduction),
               "levels": levels, "tightenings": tightenings, "val_successes": None, "val_fitness": None,
               "champion_path": None}
    if champion_dir is not None:
        candidates = np.flatnonzero(method.is_kept)                 # the kept cars of the last generation
        if candidates.size == 0:
            candidates = np.arange(len(method.population))
        factor = tracker.mastered_factor if tracker.goal_reached else config.TARGET_FACTOR
        index, successes, fitness = population.select_final(method.population, candidates, factor, n_jobs=n_jobs)
        summary.update(val_successes=int(successes), val_fitness=float(fitness),
                       champion_path=str(method.save_champion(index, seed, champion_dir)))
    summary["wall_time"] = active()                                # includes the final selection (spec §9.1)
    checkpoint.write_json(out_dir / SUMMARY_NAME, summary)
    if reporting:
        found = "" if summary["val_successes"] is None else (
            f", final controller succeeds on {summary['val_successes']} of {len(config.VAL_SEEDS)} validation maps")
        progress.milestone(f"{label(method, seed)}: finished after {generation} generations{found}")
    return summary
