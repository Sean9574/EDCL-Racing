"""Shared run loop of every evolutionary method (spec §3.2, §9.1-9.5). A method object holds the population
and makes the children (racing/ga.py, racing/es.py, racing/neat.py); this loop drives fresh maps as time trials,
applies the survival rule, follows the mastery levels and the speed push, logs every generation, saves a checkpoint
after each one, reports progress and selects the final controller."""
import csv
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from racing import checkpoint, config, paths, population, progress, runs, setup
from racing.curriculum import MasteryTracker, SeedSource, survivors
from racing.nets import DensePopulation

LOG_NAME = "log.csv"
SUMMARY_NAME = "summary.json"
EARLIER_SUMMARY_NAME = "summary_before_hardening.json"   # the summary of a run that ended before version 3.15
# Row g's `boost` is the boost used for the children bred after generation g (evaluate.stalls_broken depends on it).
LOG_FIELDS = ["generation", "level", "factor", "evals", "best", "mean", "worst", "map_shares", "min_share", "n_kept",
              "kept_success", "child_success", "median_lap", "sensor_mean", "sensor_std", "steer_mean", "steer_std",
              "step_size", "boost", "streak", "event", "seeds", "elapsed"]


class Method(Protocol):
    """What the loop needs from a method. `population` holds the cars carried over unchanged first; `is_kept`
    marks them; `population.setup` holds every car's setup (spec §9.11). `advance` must assign a new population
    object (reports keep the evaluated one); with `boost` (a stall of the speed push, spec §9.4) it mutates the
    decision layer and the steering sensitivity harder without changing its draws."""
    approach: str
    population: object
    is_kept: np.ndarray
    step_size: float

    def initialize(self, rng) -> None: ...
    def advance(self, rng, result, kept, event, boost=False) -> None: ...
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
    """What the ES uses (the GA has networks of its own shapes, racing/ga.py): a dense population of at least two
    cars, survivors first and unchanged, saving the final controller through population.save_dense_champion."""
    approach = None

    def __init__(self, pop_size, hidden, activation):
        if pop_size < 2:
            raise ValueError(f"population size ({pop_size}) must be at least 2")
        self.pop_size = pop_size
        self.hidden = tuple(hidden)
        self.activation = activation
        self.population = None
        self.is_kept = None

    @property
    def setup(self):
        """Every car's setup, aligned with the population (spec §9.11)."""
        return self.population.setup

    def assemble(self, kept_weights, children, kept_setups, children_setups):
        """The next population: the kept cars first, then the children, each with its setup (spec §9.11);
        marks the kept cars."""
        self.population = DensePopulation(np.concatenate([kept_weights, children]), self.hidden, self.activation,
                                          setup=np.concatenate([kept_setups, children_setups]))
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
        os.fsync(f.fileno())           # on disk before the generation's checkpoint, which covers it (spec §3.2)


def complete_row(row, header, generation):
    """Whether a log row is whole and belongs to a generation before `generation`: a crash can cut the last row
    short or, after a reboot, leave zero bytes in its place."""
    return len(row) == len(header) and row[0].isascii() and row[0].isdigit() and int(row[0]) < generation


def truncate_log(path, generation, fields=LOG_FIELDS):
    """Keep the header and the whole rows of the generations before `generation`. A row written after the last
    checkpoint belongs to a generation that runs again when the run resumes, and a damaged row is dropped
    (spec §3.2). A missing log is started with the columns `fields`."""
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
    return f"{config.LABELS.get(method.approach, method.approach)} run {seed}"


def milestone_text(event, method, seed, generation, level, tracker, goal_before=True):
    name = label(method, seed)
    if event == "level_up":
        return (f"{name}: {config.DIFFICULTIES[level]} mastered at generation {generation}, "
                f"now {config.DIFFICULTIES[tracker.level]}")
    if event == "goal":
        return f"{name}: mastery goal reached at generation {generation}, speed push starts"
    if event == "tightened":
        return f"{name}: target {100 * tracker.target_reduction:.1f}% shorter mastered at generation {generation}"
    if event == "end_speed":
        return (f"{name}: speed push ended at generation {generation}, hardening at the target "
                f"{100 * tracker.target_reduction:.1f}% shorter")
    if event == "hardened":
        if not goal_before:                                     # hardening before the goal ends at the goal
            return f"{name}: mastery goal reached at generation {generation}, hardened"
        return f"{name}: hardened at generation {generation}"
    return f"{name}: stopped at generation {generation}, {config.DIFFICULTIES[level]} not mastered"


def hardening_text(method, seed, generation, tracker):
    """The run log's line for a run that hardens from the generation it resumes at (spec §9.4, version 3.15)."""
    name = label(method, seed)
    if tracker.goal_reached:
        return (f"{name}: hardening at the target {100 * tracker.target_reduction:.1f}% shorter, "
                f"from generation {generation}")
    return f"{name}: hardening from generation {generation}, on {config.DIFFICULTIES[tracker.level]}"


def lower_priority(nice=config.EVOLUTION_NICE):
    """Lower the priority of every thread of this process to `nice` (spec §3.2); Linux keeps one per thread, and
    numpy's math library starts its threads at import. The compiled simulator's threads and the worker processes
    started afterwards inherit it, so an evolution phase leaves the processors to the Keras jobs that share the machine
    and uses whatever they leave. Raising a priority needs privileges, so it is only ever lowered."""
    tasks = Path("/proc/self/task")
    for thread in [int(task.name) for task in tasks.iterdir()] if tasks.is_dir() else [0]:
        try:
            if os.getpriority(os.PRIO_PROCESS, thread) < nice:
                os.setpriority(os.PRIO_PROCESS, thread, nice)
        except OSError:
            pass                                        # a thread that ended meanwhile, or no permission


def run(method, seed, out_dir, champion_dir=config.CHAMPIONS_DIR, n_jobs=1, on_generation=None, start_level=1,
        max_generations=None, rng_seed=None):
    """Train `method` with run seed `seed` until it is hardened after its speed push, the safety stop or an end
    request (spec §9.4), or after `max_generations` (viewer, pilots, tests). A run of the experiment that ended before
    version 3.15 hardens when its checkpoint is beside its summary (or its checkpoint ended at the speed limit); that
    summary is kept as EARLIER_SUMMARY_NAME once the checkpoint has been read. An end request (config.END_FILE,
    version 3.13) ends a run of the experiment before its next generation, once it has trained at least one; it then
    picks its final controller as at any other end. A completed run in `out_dir` with the current settings is
    returned at once; an interrupted one resumes from its checkpoint. With champion_dir None no final controller is
    selected or saved and nothing is written to the status file or the run log: only the experiment's runs report
    progress. Returns the summary."""
    out_dir = Path(out_dir)
    reporting = champion_dir is not None
    done = checkpoint.completed(out_dir, SUMMARY_NAME, approach=method.approach)
    if done is not None and not (reporting and checkpoint.ended_before_hardening(done)
                                 and (out_dir / checkpoint.CHECKPOINT_NAME).exists()):
        return done
    lower_priority()                                   # Keras jobs beside it come first (spec §3.2)
    population.fresh_workers(n_jobs)                   # every run builds its own tracks (spec §9.1)
    began = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / LOG_NAME
    state = checkpoint.load(out_dir / checkpoint.CHECKPOINT_NAME)
    if state is None:
        rng = runs.generator(seed, seed) if rng_seed is None else np.random.default_rng(rng_seed)
        method.initialize(rng)
        seeds = SeedSource(rng)
        tracker = MasteryTracker(level=start_level)
        generation, evals, elapsed, levels, tightenings, hardening = 0, 0, 0.0, [], [], None
        pop_size = len(method.population)
        if champion_dir is not None:
            population.clear_champion(method.approach, seed, champion_dir)
        fields = LOG_FIELDS + list(method.log_fields())   # the method's extra columns, fixed for the whole run
        start_log(log_path, fields)
        if reporting:
            progress.milestone(f"{label(method, seed)}: started")
    else:
        checkpoint.check_settings(state["settings"], method.approach)
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
        hardening = state.get("hardening")
        ended = done is not None or (out_dir / EARLIER_SUMMARY_NAME).exists() or tracker.outcome == "speed_limit"
        if reporting and "hardening" not in state and ended:   # a run ended before version 3.15 (spec §9.4)
            if (out_dir / SUMMARY_NAME).exists():
                os.replace(out_dir / SUMMARY_NAME, out_dir / EARLIER_SUMMARY_NAME)   # kept beside the run
            tracker.harden()
            hardening = {"factor": float(tracker.factor), "from_generation": generation}
            progress.milestone(hardening_text(method, seed, generation, tracker))

    def active():
        return elapsed + time.time() - began        # run time, accumulated across resumptions

    while not tracker.done and (max_generations is None or generation < max_generations):
        if reporting and generation > 0 and config.END_FILE.exists():      # an end request (spec §9.4)
            tracker.outcome = "time_limit"
            progress.milestone(f"{label(method, seed)}: ended by request at generation {generation}")
            break
        level, factor = tracker.level, tracker.factor
        maps = seeds.draw(config.TRACKS_PER_GEN, level)
        result = population.evaluate_population(method.population, maps, level, factor, n_jobs=n_jobs)
        n = len(method.population)
        evals += n
        cap = int(config.KEEP_MAX_FRACTION * n)
        kept = survivors(rng, result.fitness, result.all_success, result.lap_share, cap, config.ELITE_FLOOR)
        goal_before = tracker.goal_reached
        speed_push = tracker.speed_push                     # the rule this generation is judged by (spec §9.4)
        event = tracker.update(result.min_share)
        boost = tracker.stalled                             # the stall boost for the children (spec §9.4)
        now = active()
        report = {"approach": method.approach, "seed": seed, "generation": generation, "level": level,
                  "factor": factor, "target_reduction": tracker.target_reduction, "streak": tracker.streak,
                  "seeds": maps, "population": method.population, "fitness": result.fitness,
                  "all_success": result.all_success, "is_kept": method.is_kept, "map_shares": result.map_shares,
                  "min_share": result.min_share, "speed_push": speed_push, "n_kept": len(kept),
                  "kept_success": share(result.all_success[method.is_kept]),
                  "child_success": children_success(result, method.is_kept), "median_lap": median_lap(result),
                  "step_size": float(method.step_size), "boost": int(boost), "best": float(result.fitness.max()),
                  "mean": float(result.fitness.mean()), "worst": float(result.fitness.min()), "evals": evals,
                  "elapsed": now, "event": event, **setup.summary(method.population.setup), **method.log_fields()}
        append_log(log_path, report, fields)
        if event == "level_up" or (event in ("goal", "hardened") and not goal_before):
            levels.append({"level": int(level), "generation": generation, "evals": evals, "elapsed": now})
        elif event == "tightened":
            tightenings.append({"factor": float(tracker.mastered_factor), "generation": generation, "evals": evals,
                                "elapsed": now})
        if event == "end_speed":
            hardening = {"factor": float(tracker.factor), "from_generation": generation + 1}
        elif event == "hardened":
            hardening.update(generation=generation, evals=evals, elapsed=now)
        if reporting:
            if event is not None:
                progress.milestone(milestone_text(event, method, seed, generation, level, tracker, goal_before))
            progress.write_status({"approach": method.approach, "seed": seed,
                                   "stage": ("hardening" if tracker.hardening
                                             else "speed push" if tracker.goal_reached else "evolution"),
                                   "level": int(level), "generation": generation, "min_share": result.min_share,
                                   "streak": tracker.streak, "target_reduction": tracker.target_reduction,
                                   "elapsed": now})
        if reporting:
            checkpoint.save(out_dir / paths.REPLAY_NAME, report)     # the viewer replays it (never read by training)
        if on_generation is not None:
            on_generation(report)
        method.advance(rng, result, kept, event, boost=boost)
        generation += 1
        checkpoint.save(out_dir / checkpoint.CHECKPOINT_NAME, {
            "settings": checkpoint.settings_record(method.approach), "generation": generation, "evals": evals,
            "elapsed": active(), "rng": rng.bit_generator.state, "seeds": seeds.state(),
            "tracker": tracker.state(), "method": method.state(), "levels": levels, "tightenings": tightenings,
            "hardening": hardening, "start_level": start_level})

    summary = {"approach": method.approach, "seed": seed, "settings": checkpoint.settings_record(method.approach),
               "generations": generation, "evals": evals, "pop_size": pop_size,
               "start_level": start_level, "level_reached": int(tracker.level),
               "goal_reached": bool(tracker.goal_reached), "outcome": tracker.outcome,
               "mastered_factor": plain(tracker.mastered_factor), "target_reduction": plain(tracker.target_reduction),
               "levels": levels, "tightenings": tightenings, "hardening": hardening, "val_successes": None,
               "val_fitness": None, "champion_path": None, "final_setup": None}
    if champion_dir is not None:
        candidates = np.flatnonzero(method.is_kept)                 # the kept cars of the last generation
        if candidates.size == 0:
            candidates = np.arange(len(method.population))
        index, successes, fitness = population.select_final(method.population, candidates, n_jobs=n_jobs)
        summary.update(val_successes=int(successes), val_fitness=float(fitness),
                       champion_path=str(method.save_champion(index, seed, champion_dir)),
                       final_setup=setup.to_dict(method.population.setup[index]))
    summary["wall_time"] = active()                                # includes the final selection (spec §9.1)
    summary["machine"] = checkpoint.machine_record()               # runs may be trained on several machines
    checkpoint.write_json(out_dir / SUMMARY_NAME, summary)
    if reporting:
        found = "" if summary["val_successes"] is None else (
            f", final controller finishes {summary['val_successes']} of {len(config.VAL_SEEDS)} validation maps")
        progress.milestone(f"{label(method, seed)}: finished after {generation} generations{found}")
    return summary
