"""EC Racing viewer logic (spec §10): the training view and the race view. The App holds all state and has no drawing;
the browser page shows it."""
import csv
import json
import multiprocessing as mp
import os
import pickle
import queue
import secrets
import signal
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

from racing import checkpoint, circuits, config, curriculum, ideal, netwindow, nets, paths, population, runs, sim
from racing.actions import class_to_controls
from racing.tracks import get_track

UNUSABLE_RUN = 3                    # trainer exit code: the saved viewer run cannot be resumed (spec §10.6)
UNUSABLE_ERRORS = (checkpoint.SettingsMismatch, checkpoint.UnreadableRecord, EOFError, pickle.UnpicklingError)
SEEDED_RUN = config.SEEDED_RUN               # run 0 is the seeded run (spec §3.2)
RACE_RUNS = ("best", "all", "seeded", "unseeded")       # the race control's choices, in cycling order (spec §10.3)


def is_app(car):
    """Whether a race car is an app car (approach, seed, "app") rather than a reported controller (approach, seed)."""
    return tuple(car[2:]) == ("app",)


def car_method(car):
    """The method label of a race car: 'GA' for a reported controller, 'GA (app)' for an app car."""
    return config.LABELS[car[0]] + (" (app)" if is_app(car) else "")


def select_indices(fitness, mode, count, sample_seed=0):
    """Cars to display: a fixed random sample ('all'), the fittest ('best') or the least fit ('worst')."""
    n = len(fitness)
    count = int(min(max(count, 1), n))
    if mode == "best":
        return np.argsort(-fitness, kind="stable")[:count]
    if mode == "worst":
        return np.argsort(fitness, kind="stable")[:count]
    return np.sort(np.random.default_rng(sample_seed).choice(n, count, replace=False))







class Playback:
    """Re-simulates a set of controllers on one track, a few steps at a time, for display. Each car drives with its
    own setup and every car with the viewer's display grip, grip_scale × a(v) (spec §10.5); training never uses this
    class."""

    def __init__(self, track, policy, n, laps=1, time_limit=None, setup=None, grip_scale=1.0):
        self.track = track
        self.policy = policy
        self.grip_scale = float(grip_scale)
        self.cars = sim.start_cars(track, n, laps, setup)
        self.time_limit = sim.time_limit_for(track, laps) if time_limit is None else time_limit
        self.elapsed = 0.0              # s of race time simulated
        self.actions = np.full(len(self.cars.x), -1, dtype=int)     # each car's last action class, -1 before the first step
        self.inputs = sim.sense(track, self.cars)   # every car's inputs at the last step, on the start line until the
                                                    # first (the network window shows them)

    @property
    def over(self):
        return bool(self.cars.done.all())

    def advance(self, steps):
        for _ in range(steps):
            if self.over:
                return
            self.inputs = sim.sense(self.track, self.cars)
            actions = np.asarray(self.policy(self.inputs, self.cars))
            self.actions = actions.astype(int)
            steer, throttle = class_to_controls(actions)
            sim.step(self.track, self.cars, steer, throttle, self.time_limit, grip_scale=self.grip_scale)
            self.elapsed += config.DT


def report_policy(report, indices):
    """Policy for the chosen members of a training report: an evolutionary population (dense or NEAT, spec §10.2)
    or the networks of an imitation report."""
    if "population" in report:
        return report["population"].subset(indices).policy()
    return nets.mixed_policy([report["nets"][i] for i in indices])


def report_inputs(report):
    """Inputs the networks of a training report take; None for NEAT genomes, whose inputs are implicit. A report of
    an earlier revision may take fewer than config.N_INPUTS (spec §10.2)."""
    if "population" not in report:
        return report["nets"][0].n_inputs
    population = report["population"]
    return population.n_inputs if isinstance(population, (nets.DensePopulation, nets.ShapedPopulation)) else None


def run_folders(results_dir=config.RESULTS_DIR):
    """The results folders the viewer shows runs of: ("current", the results folder), then each archive in
    results/archive that holds runs or controllers, by name."""
    results_dir = Path(results_dir)
    archive = results_dir / "archive"
    folders = [("current", results_dir)]
    if archive.is_dir():
        folders += [(folder.name, folder) for folder in sorted(archive.iterdir())
                    if (folder / "runs").is_dir() or (folder / "champions").is_dir()]
    return folders


GRIP_LABELS = {"Normal": "Normal", "Unlimited": "Unlimited (10×)"}      # config.APP_GRIPS on the grip button


def folder_record(folder):
    """The settings an archive recorded in its run_info.json, or None without a readable record."""
    try:
        settings = json.loads((Path(folder) / config.RUN_INFO_PATH.name).read_text())["settings"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return settings if isinstance(settings, dict) else None


def folder_profile(name, folder):
    """(inputs, road widths) of a results folder's networks and maps: this revision's (config.N_INPUTS, None: each
    map draws its own width) for the current results; an archive's from its own run_info.json settings: its input
    count, and the road width per level where an earlier revision names one number per level (revision 3.1). An
    archive without a readable record is taken as current."""
    settings = None if name == "current" else folder_record(folder)
    if settings is None:
        return config.N_INPUTS, None
    try:
        inputs = int(settings["N_INPUTS"])
    except (KeyError, TypeError, ValueError):
        return config.N_INPUTS, None
    try:
        widths = {int(level): float(spec["width"]) for level, spec in settings["LEVELS"].items()}
    except (KeyError, TypeError, ValueError, AttributeError):       # no widths, or ranges: each map draws its own
        widths = None
    return inputs, widths


def before_real_circuits(name, folder):
    """Whether an archive was made before version 3.8 (spec §4.7), when Extra hard was a generated map and the car
    had other grip: its record has no test circuits. Such runs cannot be replayed on the real circuits."""
    settings = None if name == "current" else folder_record(folder)
    return settings is not None and "TEST_CIRCUITS" not in settings


def other_interface(name, folder):
    """The note for a results folder, the current one included, made for another network interface (spec §10.3), or
    None: 15 actions up to version 3.8, or this action set with 12 inputs in version 3.9. This version would misread
    such a folder's controllers and runs, so they are neither raced nor replayed. A record without an action count (an
    archive of revision 3.1, whose networks read their first inputs, spec §10.2) or without a record is not flagged;
    nets.load_champion still refuses every controller of another interface."""
    settings = folder_record(folder)
    if settings is None:
        return None
    try:
        actions = int(settings["N_ACTIONS"])
    except (KeyError, TypeError, ValueError):
        return None
    if actions != config.N_ACTIONS:
        return f"made with {actions} actions; this version has {config.N_ACTIONS}"
    try:
        inputs = int(settings["N_INPUTS"])
    except (KeyError, TypeError, ValueError):
        return None
    if inputs != config.N_INPUTS:
        return f"made with {inputs} inputs; this version has {config.N_INPUTS}"
    return None


def view_seed(level):
    """The map the imitation approaches are driven on in the training view (spec §10.2): the first training layout
    at Extra hard, validation map 80 at the generated levels."""
    return 0 if level == config.MAX_LEVEL else config.KT_VIEW_SEED


def report_setup(report, indices):
    """Setups of the chosen members of a training report: the population's rows, or each imitation network's own
    (spec §9.11)."""
    if "population" in report:
        return report["population"].subset(indices).setup
    return nets.drivers_setup([report["nets"][i] for i in indices])


def pole_gap(flying_lap, pole):
    """A flying lap's difference to the pole lap as in the pole table ('-0.486', '+1.234'); '–' without either."""
    return "–" if pole is None or np.isnan(flying_lap) else f"{flying_lap - pole:+.2f}"


MPH_PER_MPS = 3600 / 1609.344   # miles per hour in one m/s
MARKED_EVENTS = {"level_up": "tall", "goal": "tall", "tightened": "short", "hardened": "short"}   # training plot marks (§10.2)
NET_FRAME_INTERVAL = 0.1     # s: the network window gets the shown car's node values at most this often


def race_leader(cars):
    """The race leader's index: the car furthest along that is still racing, else the car furthest along."""
    racing = np.flatnonzero(~cars.done)
    pool = racing if racing.size else np.arange(len(cars.progress))
    return int(pool[np.argmax(cars.progress[pool])])


def speed_text(speed):
    """A speed in m/s as whole miles per hour, '194 mph'."""
    return f"{speed * MPH_PER_MPS:.0f} mph"


GHOST = -1                      # the pole ghost's car index in leaderboard rows
GhostState = namedtuple("GhostState", "x y heading progress laps finished")


class PoleGhost:
    """A test circuit's pole lap driven as a car of the race view (display only; cars never sense it): it starts on
    the start line at speed and laps the racing line in exactly the pole time, its speed along the lap the car's
    speed profile (ideal.speed_profile) scaled to that time. Only the lap time is real; where the driver was within a
    lap is an estimate."""

    def __init__(self, track, pole, laps):
        self.track, self.pole, self.laps = track, pole, laps
        self.name = f"{pole.driver} {pole.season} (pole)"
        step = np.linalg.norm(np.roll(track.line, -1, axis=0) - track.line, axis=1)
        v = ideal.speed_profile(np.abs(track.line_curvature), step)
        times = np.concatenate([[0.0], np.cumsum(step / ((v + np.roll(v, -1)) / 2.0))])
        self.times = times * (pole.seconds / times[-1])       # s at each line point; the last closes the lap

    def state(self, elapsed):
        """The ghost after `elapsed` s of the race: position, heading, distance along the centerline in m (as the
        cars' progress), laps completed and whether it has finished."""
        lap, line = self.pole.seconds, self.track.line
        done = min(int(max(elapsed, 0.0) // lap), self.laps)
        if done == self.laps:
            return GhostState(float(line[0, 0]), float(line[0, 1]), float(self.track.line_heading[0]),
                              self.laps * self.track.length, done, True)
        into = elapsed - done * lap
        k = min(int(np.searchsorted(self.times, into, side="right")) - 1, len(line) - 1)
        f = (into - self.times[k]) / (self.times[k + 1] - self.times[k])
        x, y = line[k] + f * (line[(k + 1) % len(line)] - line[k])
        return GhostState(float(x), float(y), float(self.track.line_heading[k]),
                          done * self.track.length + (k + f) * self.track.spacing, done, False)


def leaderboard(playback, count, pole=None, ghost=None):
    """Rows (position, car index, laps completed, last lap, best lap, status, gap to pole), ordered by distance
    covered. On a test circuit (`pole` in seconds) the gap is the car's flying lap minus the pole lap (spec §10.3).
    With the pole `ghost` (PoleGhost), its row, car index GHOST, takes its place by distance, behind cars level with
    it."""
    nan = float("nan")
    if playback is None:
        return [(k + 1, k, 0, nan, nan, "ready", "–") for k in range(count)]
    cars = playback.cars
    flying = sim.flying_laps(cars.lap_times)
    rows = []
    for i in np.argsort(-cars.progress, kind="stable"):
        if cars.crashed[i]:
            status = "crashed"
        elif cars.finished[i]:
            status = "finished"
        elif cars.done[i]:
            status = "stopped"
        else:
            status = "racing"
        rows.append((int(i), int(cars.laps_done[i]), float(cars.last_lap[i]), float(cars.best_lap[i]), status,
                     pole_gap(float(flying[i]), pole)))
    if ghost is not None:
        g = ghost.state(playback.elapsed)
        lap = ghost.pole.seconds if g.laps else nan
        rows.insert(int(np.sum(cars.progress >= g.progress)),
                    (GHOST, g.laps, lap, lap, "finished" if g.finished else "racing", pole_gap(lap, pole)))
    return [(position + 1, *row) for position, row in enumerate(rows)]


def save_app_car(report, champion_dir):
    """The best car of an evolutionary report, saved with its setup as the run's app car (spec §10.6): the
    files of a final controller, in `champion_dir`. They are written in a staging folder and each moved into place in
    one step, so the race view, which may load the car while training continues, never reads a half-written file."""
    from racing import neat
    cars = report["population"]
    best = int(np.argmax(report["fitness"]))
    approach, seed = report["approach"], report["seed"]
    staging = Path(champion_dir) / ".staging"        # not an approach folder, so the race view never lists it
    paths.remove(staging)
    if isinstance(cars, neat.NeatPopulation):
        neat.save_neat_champion(approach, seed, cars.genomes[best], staging, setup=cars.setup[best])
    else:
        population.save_dense_champion(approach, seed, cars, best, staging)
    target = Path(champion_dir) / approach
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted((staging / approach).iterdir()):
        os.replace(path, target / path.name)
    paths.remove(staging)


def save_app_model(approach, seed, model, champion_dir):
    """An imitation network saved as the run's app car (spec §10.6). It has no setup file, so it drives with
    the default setup. The file is replaced in one step, so the race view never reads half a model."""
    population.clear_champion(approach, seed, champion_dir)
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.keras")
    model.save(temporary)
    os.replace(temporary, path)


def trainer_main(approach, size, seed, level, reports, running, app_runs_dir=None, app_champions_dir=None,
                 acks=None):
    """Training process: trains one approach with run seed `seed` and sends a report after every generation, trial
    or epoch (spec §10.2). With `acks`, training runs in step with the viewer: after each report it waits until
    the viewer has replayed it (or is not watching), so the generation on screen is the one just trained. An evolutionary run continues the saved run of the same approach and seed in the app
    runs folder (App.new_run removes it to start over); imitation training always starts afresh. The run's best car
    so far is saved to the app champions folder so it can race (spec §10.6); the reported results are never
    written. Returns 0, or UNUSABLE_RUN when the saved run cannot be resumed (other settings or a damaged file)."""
    from racing import es, ga, imitation, neat
    from racing.expert import collect_dataset

    app_runs_dir = Path(config.APP_RUNS_DIR if app_runs_dir is None else app_runs_dir)
    app_champions_dir = Path(config.APP_CHAMPIONS_DIR if app_champions_dir is None else app_champions_dir)
    out_dir = app_runs_dir / approach / f"seed{seed}"

    def send(report):
        if "population" in report:
            save_app_car(report, app_champions_dir)
        reports.put(report)
        if acks is not None:
            acks.get()                      # in step with the viewer: wait until it has replayed this report
        running.wait()                      # training holds here while the viewer is paused

    common = dict(seed=seed, pop_size=size, start_level=level, out_dir=out_dir, on_generation=send,
                  champion_dir=None, n_jobs=config.N_JOBS, max_generations=None)
    if approach in config.EVOLUTION_APPROACHES:
        try:
            if approach == "ga":
                ga.run_ga(**common)
            elif approach == "neat":
                neat.run_neat(**common)
            else:
                initial = None
                if approach == "es_seeded":
                    try:
                        initial = es.dagger_baseline(seed)
                    except ValueError as error:         # a refined baseline made for other inputs or actions
                        print(f"The refined baseline of run {seed} cannot start the seeded ES: {error}",
                              file=sys.stderr, flush=True)
                        return UNUSABLE_RUN
                es.run_es(**common, initial=initial, approach=approach)
        except UNUSABLE_ERRORS as error:
            print(f"The saved viewer run in {out_dir} cannot be resumed: {error}", file=sys.stderr, flush=True)
            return UNUSABLE_RUN
        return 0
    paths.remove(out_dir)                   # imitation training in the viewer is never resumed
    path = config.DATASET_PATH
    if not (path.exists() and imitation.dataset_matches(path)):    # a dataset of other settings is never trained on
        path = app_runs_dir / "dataset.npz"     # never write the reported dataset
        if path.exists() and not imitation.dataset_matches(path):
            paths.remove(path, path.with_name(imitation.dataset_info_name(path)))
        if not path.exists():
            collect_dataset(path)
    data = imitation.load_dataset(path)
    members, scores = [], []

    def add(number, model, val_accuracy):
        if not scores or val_accuracy > max(scores):
            save_app_model(approach, seed, model, app_champions_dir)    # the best network so far can race
        members.append(nets.from_keras(model))
        scores.append(val_accuracy)
        fitness = np.array(scores)
        send({"approach": approach, "generation": int(number), "level": level,
              "seeds": [view_seed(level)], "nets": list(members), "fitness": fitness,
              "best": float(fitness.max()), "mean": float(fitness.mean()), "worst": float(fitness.min()),
              "leveled_up": False})

    if approach == "baseline":                         # resume=False: always train live, never skip (spec §10.6)
        imitation.train_baseline(data, out_dir=out_dir, seed=seed, resume=False, report=False,
                                 on_epoch=lambda epoch, model, acc: add(epoch, model, acc))
    else:                       # the search of the unseeded run takes a number from the operating system (spec §3.2)
        imitation.run_tuner(approach, data, out_dir=out_dir, on_trial=add,
                            seed=seed if runs.seeded(seed) else secrets.randbits(31), export_models=False,
                            resume=False, report=False)
    return 0


def watch_parent(parent, kill, interval=1.0):
    """Waits until the process that started the trainer is gone (the parent process changes), then ends the
    trainer's process group with kill(0, SIGTERM)."""
    while os.getppid() == parent:
        time.sleep(interval)
    kill(0, signal.SIGTERM)


def trainer_process(approach, size, seed, level, reports, running, app_runs_dir=None, app_champions_dir=None,
                    acks=None):
    """Entry point of the training process. It leads its own process group, so stopping it also stops the worker
    processes it starts (spec §10.2). A watcher thread ends the group if the viewer exits without stopping it. The
    process exits with the trainer's code, so the viewer can tell an unusable saved run (spec §10.6)."""
    os.setpgrp()
    threading.Thread(target=watch_parent, args=(os.getppid(), os.killpg), daemon=True).start()
    code = trainer_main(approach, size, seed, level, reports, running, app_runs_dir, app_champions_dir, acks)
    if code:
        sys.exit(code)


def car_name(approach, seed, app=False, archived=False):
    """A race car's name in the leaderboard, the network selector and the window title: run 0 'GA (seeded)', runs 1
    to 3 'GA r1' to 'GA r3' (spec §10.3). Every run of an archive of an earlier revision was seeded, so there
    each is named by its seed, 'GA s0', 'GA s1'; an app car keeps its seed, 'GA (app) s1' (spec §10.6)."""
    label = config.LABELS[approach]
    if app:
        return f"{label} (app) s{seed}"
    if archived:
        return f"{label} s{seed}"
    if seed == SEEDED_RUN:
        return f"{label} (seeded)"
    return f"{label} r{seed}"


def leaderboard_line(position, approach, seed, laps, last, best, status, app=False, archived=False, gap=None,
                     name=None):
    """One leaderboard row: position, car name (`name`, or the car's name from its approach and seed), laps completed,
    last/best lap time (s), status and, on a test circuit (`gap` given), the flying lap's difference to the pole
    lap."""
    times = "/".join("–" if np.isnan(value) else f"{value:.1f}" for value in (last, best))
    line = f"{position}. {name or car_name(approach, seed, app, archived)} lap {laps} {times} {status}"
    return line if gap is None else f"{line} {gap}"


def percent(value):
    return "–" if np.isnan(value) else f"{100 * value:.0f}%"


def number(value):
    return "–" if np.isnan(value) else f"{value:.4g}"


def share_text(shares):
    """Success shares of the generation's maps, e.g. '84% 92% 95%'."""
    return " ".join(percent(float(s)) for s in shares)


def setup_line(report):
    """The population's mean and spread of sensor length and steering sensitivity (spec §10.2)."""
    return (f"Sensors {report['sensor_mean']:.0f} m ± {report['sensor_std']:.0f}  "
            f"Steering ×{report['steer_mean']:.2f} ± {report['steer_std']:.2f}")


def log_history(log):
    """The plot history of a run from its log.csv: (generation, lowest map share, nan, event name or None) per
    generation; empty for a missing or unreadable log."""
    if not Path(log).exists():
        return []
    history = []
    try:
        with open(log, newline="") as f:
            for row in csv.DictReader(f):
                history.append((int(row["generation"]), float(row["min_share"]), float("nan"),
                                row.get("event") or None))
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return history


REPLAY_ERRORS = (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError, ValueError, KeyError,
                 TypeError, IndexError)
REPLAY_KEYS = {"approach", "generation", "level", "seeds", "population", "fitness"}


def stats_lines(report, approach):
    """Training panel stats lines (spec §10.2). Evolutionary reports: difficulty and generation, each map's
    success share against the 90% requirement, the streak and the target cut mastered, the step size and the kept
    cars, the fitness (and NEAT's species). With setups, the population's mean setup (spec §10.2). Imitation
    reports: the epoch or trial and the validation accuracy."""
    if "min_share" not in report:
        unit = "Trial" if approach.startswith("kt") else "Epoch"
        return (f"{unit} {report['generation']}",
                f"Validation accuracy: best {report['best']:.3f}  mean {report['mean']:.3f}")
    level = config.DIFFICULTIES.get(report["level"], f"Level {report['level']}")
    reduction = float(report.get("target_reduction", float("nan")))
    mastered = "–" if np.isnan(reduction) else f"−{100 * reduction:.1f}%"
    size_name = "Mutation strength" if approach == "ga" else "σ"
    last = f"Best {report['best']:.3f}    Mean {report['mean']:.3f}"
    if "species" in report:
        last += f"    Species {report['species']}"
    lines = (f"{level}    Generation {report['generation']}",
             f"Maps succeeded: {share_text(report['map_shares'])}  "
             f"(need {percent(curriculum.mastery_fraction(report.get('speed_push', False)))})",
             f"Streak {report['streak']}/{config.MASTERY_STREAK}    Target cut mastered: {mastered}",
             f"{size_name}: {number(report['step_size'])}    Kept {int(np.sum(report['is_kept']))}",
             last)
    return lines + (setup_line(report),) if "sensor_mean" in report else lines


class App:
    """The viewer window: training view and race view."""

    def __init__(self, background_tracks=True):
        self.alive = True
        self.watched = True                 # False: nobody looks at the page, so nothing is replayed (reports still flow)
        self.mode = "train"
        self.background_tracks = background_tracks
        self.tracks = {}
        self.building = set()
        self.speed = 1
        self.step_budget = 0.0
        self.status = "Press Play to start training."
        # training view
        self.approach = "ga"
        self.run_seed = config.RUN_SEEDS[0]
        self.difficulty = 1
        self.population = {"ga": config.GA_POP, "es": config.ES_POP, "es_seeded": config.ES_POP,
                           "neat": config.NEAT_POP}
        self.results_dir = config.RESULTS_DIR           # the current results; archives are in its archive folder
        self.folder_name = "current"                    # the results folder whose runs and controllers are shown
        self.folder_inputs, self.folder_widths = config.N_INPUTS, None      # folder_profile of that folder
        self.folder_old = False                         # an archive made before the real circuits (spec §4.7)
        self.folder_note = other_interface(self.folder_name, self.results_dir)    # why a folder is not shown (§10.3)
        self.runs_dir = config.RUNS_DIR                 # the shown folder's notebook runs
        self.train_runs_dir = config.RUNS_DIR           # where the seeded ES trained here finds the DAgger baseline
        self.app_runs_dir = config.APP_RUNS_DIR             # saved viewer runs (spec §10.6)
        self.app_champions_dir = config.APP_CHAMPIONS_DIR   # their best cars, raceable in the race view
        self.show = "best"
        self.shown = config.DEFAULT_SHOWN
        self.grip_name = "Normal"           # display grip of the viewer's own simulations (spec §10.5)
        self.process = None
        self.reports = None
        self.running = None
        self.acks = None                    # tells the trainer a report has been replayed (training in step)
        self.unacked = None                 # the report the trainer waits on until its replay ends
        self.halted = False                 # Stop freezes the training replay until the next Play
        self.source = "app"                 # "app": training started here; "notebook": a reported run, read-only
        self.replay_paused = False          # Pause of the notebook-run replay (the run itself never waits)
        self.replay_mtime = None            # modification time of the loaded replay file
        self.replay_polled = None           # clock time of the last look at the replay file
        self.clock = time.monotonic
        self.latest = None
        self.history = []
        self.playback = None
        self.playback_report = None
        self.playback_generation = None
        self.indices = None
        self.best_slot = 0
        # network window
        self.net_process = None
        self.net_queue = None
        self.follow = None                  # the race car the page follows and the network window shows; None: the leader
        self.net_sent = None                # identity of the network the window shows
        self.net_driver = None              # its driver, for the live node values
        self.net_member = None              # in training: the shown network's index in the report
        self.net_values_at = float("-inf")  # time.monotonic() of the last node values sent
        self.net_values_step = None         # the replay's elapsed time they were sent at
        self.net_history = []               # the best network of every generation seen of the run (the timeline)
        self.net_history_sent = 0           # how many of them the open window has
        # race view
        self.champions_dir = config.CHAMPIONS_DIR
        self.race_index = 0                 # into race_tracks(): the test circuits or the display seeds
        self.race_level = config.MAX_LEVEL
        self.race_available = {}            # approach -> run seeds with a saved controller
        self.race_runs = "best"             # the runs that race: "best", "all", "seeded" or "unseeded" (spec §10.3)
        self.best = {}                      # approach -> best run seed recorded by the evaluation (spec §11.1)
        self.race_on = {}                   # approach -> whether its runs race
        self.drivers = {}                   # (approach, seed) -> DenseNet
        self.lineup = []                    # (approach, seed) of the cars in the race
        self.race = None                    # Playback: the preview until the race starts
        self.ghost = None                   # PoleGhost of a test circuit's race, else None
        self.ghosts = {}                    # (circuit, track length, pole lap) -> PoleGhost, built once
        self.racing = False
        self.failed = set()                 # (approach, seed) whose controller file could not be loaded
        self.app_available = {}             # approach -> run seeds with a saved app car (spec §10.6)
        self.app_choice = {}                # approach -> "off" or a run seed
        self.app_lineup = []                # (approach, seed, "app") of the app cars in the race
        self.race_group = "final"           # the controller list shown: "final" (reported) or "app"

    # ----- tracks -----
    def request_track(self, seed, level):
        """The track if it is built; otherwise start building it and return None (spec §10.2). Without background
        building, the track is built at once."""
        key = (int(seed), int(level), self.map_width(level))
        if self.replays_old_archive() and int(level) == config.MAX_LEVEL:
            self.status = (f"{self.folder_name}: made with the earlier car; its Extra hard maps were generated and "
                           f"cannot be replayed on the real circuits.")
            return None
        if key in self.tracks:
            return self.tracks[key]
        if not self.background_tracks:
            self.build_track(key)
            return self.tracks.get(key)
        if key not in self.building:
            self.building.add(key)
            threading.Thread(target=self.build_track, args=(key,), daemon=True).start()
        return None

    def replays_old_archive(self):
        """Whether the training view shows a run of an archive made before the real circuits (spec §4.7)."""
        return self.folder_old and self.mode == "train" and self.source == "notebook"

    def map_width(self, level):
        """Road width of the maps shown for the selected results folder: an earlier revision's width for the level, or
        None, each map's own; training started here always uses this revision's maps."""
        if self.folder_widths is None or (self.mode == "train" and self.source == "app"):
            return None
        return self.folder_widths.get(int(level))

    def build_track(self, key):
        """Build one track; a failure is shown in the status line and the next request tries again."""
        seed, level, width = key
        try:
            self.tracks[key] = get_track(seed, level) if width is None else get_track(seed, level, width)
        except Exception as error:
            self.status = f"Track seed {key[0]}, level {key[1]} could not be built: {error}"
        finally:
            self.building.discard(key)

    # ----- training process -----
    def start_training(self):
        if self.approach == "es_seeded" and not paths.stage_path("baseline", self.run_seed, "dagger",
                                                                 self.train_runs_dir).exists():
            self.status = "Seeded ES needs the DAgger baseline: run notebook 03."
            return
        context = mp.get_context("spawn")
        self.reports = context.Queue()
        self.running = context.Event()
        self.running.set()
        self.acks, self.unacked = context.Queue(), None
        size = self.population.get(self.approach, 0)
        self.process = context.Process(target=trainer_process, daemon=False,
                                       args=(self.approach, size, self.run_seed, self.difficulty, self.reports,
                                             self.running, self.app_runs_dir, self.app_champions_dir, self.acks))
        self.process.start()
        self.halted = False
        self.latest, self.history, self.playback, self.playback_generation = None, self.saved_history(), None, None
        self.net_sent = None                                 # a new run's first report goes to the window
        self.status = f"Training {config.LABELS[self.approach]}, run {self.run_seed}."

    def toggle_play(self):
        if self.source == "notebook":                       # pauses only the replay of the notebook run
            self.replay_paused = not self.replay_paused
            self.status = "Replay paused." if self.replay_paused else "Replaying the notebook run."
            return
        if self.process is None:
            self.start_training()
        elif self.running.is_set():
            self.running.clear()
            self.status = "Paused."
        else:
            self.running.set()
            self.status = f"Training {config.LABELS[self.approach]}."

    def new_run(self):
        """Delete the saved run of the selected approach and run seed and its app car, then start training afresh
        with the current settings (spec §10.6). Ignored while training runs."""
        if self.process is not None or self.source == "notebook":
            return
        paths.remove(self.saved_run_dir())
        population.clear_champion(self.approach, self.run_seed, self.app_champions_dir)
        self.start_training()

    def saved_run_dir(self):
        return self.app_runs_dir / self.approach / f"seed{self.run_seed}"

    def has_saved_run(self):
        """Whether Play continues a saved run of the selected evolutionary approach and run seed."""
        return (self.approach in config.EVOLUTION_APPROACHES
                and (self.saved_run_dir() / checkpoint.CHECKPOINT_NAME).exists())

    def saved_history(self):
        """The plot history of the saved run from its log (lowest map share per generation, level advancements,
        goal and target cuts marked), so a resumed run's plot continues where it stopped; empty for imitation, for
        a new run or for an unreadable log."""
        if self.approach not in config.EVOLUTION_APPROACHES:
            return []
        return log_history(self.saved_run_dir() / "log.csv")

    def stop_training(self):
        """End the training process and the worker processes it started, and drop its report queue."""
        if self.process is not None and self.process.is_alive():
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self.process.terminate()             # the process has not yet become a group leader
            self.process.join(timeout=2)
        self.process = None
        self.discard_reports()

    @property
    def paused(self):
        """Pause holds the training and freezes its replay."""
        return self.process is not None and self.running is not None and not self.running.is_set()

    def acknowledge(self):
        """Lets the trainer continue after the report it waits on (training in step with the viewer)."""
        if self.unacked is None:
            return
        if self.acks is not None:
            self.acks.put(True)
            if self.mode == "train":
                unit = ("generation" if "min_share" in self.unacked
                        else "trial" if self.approach.startswith("kt") else "epoch")
                self.status = f"Training {unit} {self.unacked['generation'] + 1}…"
        self.unacked = None

    # ----- notebook runs (read-only) -----
    def toggle_source(self):
        """Switch the training view between training started here and a notebook run, whose latest generation is
        replayed without ever writing to the run or slowing it down. Only while no training runs here."""
        if self.process is not None:
            return
        self.source = "notebook" if self.source == "app" else "app"
        if self.run_seed not in self.run_choices():         # an archive's other run: training here has runs 0 to 3
            self.run_seed = config.RUN_SEEDS[0]
        self.halted = False
        self.clear_view()
        self.status = "Looking for the notebook run." if self.source == "notebook" else "Press Play to start training."

    def clear_view(self):
        """Forget the shown reports, so the next ones come only from the selected source, method and seed."""
        self.latest, self.history, self.playback, self.playback_report = None, [], None, None
        self.playback_generation = None
        self.replay_mtime, self.replay_polled, self.replay_paused = None, None, False
        self.net_sent = None

    def folder_label(self):
        return f"Runs: {self.folder_name} ▸"

    def archived(self):
        """Whether the shown runs and controllers are an archive of an earlier revision, whose runs were all
        seeded."""
        return self.folder_name != "current"

    def cycle_folder(self):
        """Show the next results folder's notebook runs and controllers: the current results, then each archive of an
        earlier revision, whose runs drive on that revision's maps (spec §10.2). Training started here keeps
        this revision's folders. Not during a race."""
        if self.racing:
            return
        folders = run_folders(self.results_dir)
        names = [name for name, _ in folders]
        k = (names.index(self.folder_name) + 1) % len(folders) if self.folder_name in names else 0
        self.folder_name, folder = folders[k]
        self.folder_inputs, self.folder_widths = folder_profile(self.folder_name, folder)
        self.folder_old = before_real_circuits(self.folder_name, folder)
        self.folder_note = other_interface(self.folder_name, folder)
        self.runs_dir, self.champions_dir = folder / "runs", folder / "champions"
        if self.mode == "race":
            self.load_race()
        elif self.source == "notebook":
            self.clear_view()
            self.status = f"Showing the {self.folder_name} runs."

    def replay_file(self):
        return paths.replay_path(self.approach, self.run_seed, self.runs_dir)

    def poll_replay(self):
        """Load the selected notebook run's latest generation if its replay file changed, looking at most once per
        REPLAY_POLL_SECONDS. A missing, older or unreadable file is named in the status line; reading never
        writes."""
        now = self.clock()
        if self.replay_polled is not None and now - self.replay_polled < config.REPLAY_POLL_SECONDS:
            return
        self.replay_polled = now
        path = self.replay_file()
        name = f"{config.LABELS[self.approach]} run {self.run_seed}"
        if self.folder_note:
            self.status = f"{self.folder_name}: {self.folder_note}, so its runs are not replayed."
            return
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            older = any((path.parent / f).exists() for f in (checkpoint.CHECKPOINT_NAME, "summary.json"))
            self.status = ("Run predates replays: see its cars in Race." if older
                           else f"Nothing to replay yet for {name}.")
            return
        if stamp == self.replay_mtime:
            return
        try:
            report = checkpoint.load(path)
            readable = isinstance(report, dict) and REPLAY_KEYS <= report.keys()
            inputs = report_inputs(report) if readable else None
        except REPLAY_ERRORS:
            readable = False
        if not readable:
            self.status = "Replay file unreadable; retrying."
            return
        self.replay_mtime = stamp
        if (inputs or config.N_INPUTS) != self.folder_inputs:          # NEAT genomes read this revision's inputs
            self.status = f"{name}: older revision, {inputs or self.folder_inputs} inputs."
            return                                                      # read again only when the file changes
        self.receive(report)
        self.history = log_history(path.parent / "log.csv")
        self.status = f"Replaying the notebook run {name}."

    def stop(self):
        """Stop control: ends training in the training view and the race in the race view."""
        if self.mode == "race":
            self.stop_race()
            return
        if self.source == "notebook":                       # nothing to stop: the notebook run is not ours
            return
        self.stop_training()
        self.halted = True                                   # the replay freezes where it is (spec §10.2)
        self.status = "Stopped. Change the settings or press Play."

    def discard_reports(self):
        """Drop the report and acknowledgement queues so a stopped trainer never reaches a new view."""
        for channel in (self.reports, self.acks):
            if channel is not None and hasattr(channel, "cancel_join_thread"):
                channel.cancel_join_thread()
                channel.close()
        self.reports, self.acks, self.unacked = None, None, None

    def drain_reports(self):
        if self.reports is None:
            return
        while True:
            try:
                self.receive(self.reports.get_nowait())
            except queue.Empty:
                break

    def poll_reports(self):
        self.drain_reports()
        if self.process is not None and not self.process.is_alive():
            self.drain_reports()                     # a report sent just before the process ended
            code = self.process.exitcode
            if code == 0:
                self.status = "Training finished. New run starts over."
            elif code == UNUSABLE_RUN:
                self.status = "Saved run cannot resume: press New run to start over."
            else:
                self.status = f"Training stopped (exit code {code})."
            self.process = None
            self.discard_reports()

    def receive(self, report):
        """Show a report; the plot follows the lowest map success share (evolution, with level advancements, the
        mastery goal and target cuts marked) or the validation accuracy (imitation). The trainer waits until the
        report has been replayed; in the race view nobody watches the training, so it continues at once."""
        self.latest = report
        if self.acks is not None:
            self.unacked = report
            if self.mode == "race":
                self.acknowledge()
        if "min_share" in report:
            self.history.append((report["generation"], float(report["min_share"]), float("nan"),
                                 report.get("event")))
        else:
            self.history.append((report["generation"], report["best"], report["mean"], report.get("event")))

    # ----- network window -----
    def toggle_network_window(self):
        """Open the separate network window, or close it if it is open."""
        if self.net_process is not None:
            self.close_network_window()
            return
        context = mp.get_context("spawn")
        self.net_queue = context.Queue()
        self.net_process = context.Process(target=netwindow.window_process, daemon=False, args=(self.net_queue,))
        self.net_process.start()
        self.net_sent = None

    def close_network_window(self):
        """Ask the window to quit, wait briefly, terminate it if it stays, and forget it."""
        if self.net_process is not None and self.net_process.is_alive():
            self.net_queue.put(None)
            self.net_process.join(timeout=2)
            if self.net_process.is_alive():
                self.net_process.terminate()
                self.net_process.join(timeout=1)
        self.reset_network_window()

    def reset_network_window(self):
        if self.net_queue is not None and hasattr(self.net_queue, "cancel_join_thread"):
            self.net_queue.cancel_join_thread()
            self.net_queue.close()
        self.net_process, self.net_queue, self.net_sent = None, None, None
        self.net_driver, self.net_member, self.net_values_at, self.net_values_step = None, None, float("-inf"), None
        self.net_history_sent = 0

    def followed_car(self):
        """The race car the page follows and the network window shows: the chosen car while it still races, else the
        race leader (the first car before a race exists); None for an empty race."""
        cars = self.race_cars()
        if not cars:
            return None
        if self.follow in cars:
            return self.follow
        if self.race is not None and len(self.race.cars.x) == len(cars):
            return cars[race_leader(self.race.cars)]
        return cars[0]

    def selected_car(self):
        """The race car whose network the window shows."""
        return self.followed_car()

    def set_follow(self, index):
        """Follow the race car at `index` of race_cars(); None (or an index outside the race) follows the leader."""
        cars = self.race_cars()
        self.follow = cars[index] if index is not None and 0 <= index < len(cars) else None

    def set_shown(self, count):
        """Show `count` cars of the training population (at least 1, at most the population size)."""
        self.shown = int(min(max(count, 1), self.population_size()))
        self.playback = None

    def shutdown(self):
        """Stop the training, close the network window and end the viewer."""
        self.stop_training()
        self.close_network_window()
        self.alive = False

    def record_generation(self):
        """Keep the best network of the newest report's generation for the window's timeline (collected while the
        viewer watches, whether or not the window is open); a report of another run, or of an earlier generation (a
        new run), starts the timeline again."""
        report = self.latest
        if report is None:
            return
        approach, generation = report["approach"], int(report["generation"])
        seed = report.get("seed", self.run_seed)
        run = (self.source, approach, seed)
        last = self.net_history[-1] if self.net_history else None
        if last is not None and last["run"] == run and last["generation"] == generation:
            return
        if last is not None and (last["run"] != run or generation < last["generation"]):
            self.net_history, self.net_history_sent = [], 0
        if "population" in report:
            member = int(np.argmax(report["fitness"]))
            driver, unit = report["population"].driver(member), "generation"
        else:
            member = len(report["nets"]) - 1
            driver, unit = report["nets"][-1], "trial" if approach.startswith("kt") else "epoch"
        level = int(report["level"])
        share = "min_share" in report
        self.net_history.append({
            "generation": generation, "member": member, "driver": driver,
            "title": (f"{config.LABELS[approach]} · run {seed} · {unit} {generation} · "
                      f"{driver.n_params:,} parameters{netwindow.netfig.setup_text(driver)}"),
            "curve": float(report["min_share"] if share else report["best"]), "kind": "share" if share else "accuracy",
            "event": report.get("event"), "level": level,
            "map": (int(report["seeds"][0]), level, self.map_width(level)), "grip": self.grip_scale,
            "where": config.DIFFICULTIES.get(level, ""), "run": run})

    def shown_network(self):
        """(identity, driver, title) of the race car whose network the window should show now, or None."""
        car = self.selected_car()
        if car is None:
            return None
        identity = ("race", *car)
        if identity == self.net_sent:
            return identity, None, None
        try:
            driver = self.driver(*car)
        except Exception:
            self.drop_car(car)
            self.status = f"Controller {car_method(car)} run {car[1]} could not be loaded; skipped."
            return None
        return identity, driver, (f"{car_name(car[0], car[1], is_app(car), self.archived())} · "
                                  f"{driver.n_params:,} parameters{netwindow.netfig.setup_text(driver)}")

    def poll_network_window(self):
        """Forget a window the user closed; send what it shows when that changed (a race car's network, or the
        training timeline's new generations), then the shown car's live node values."""
        if self.net_process is None:
            return
        if not self.net_process.is_alive():
            self.reset_network_window()
            return
        if self.mode == "train":
            self.send_generations()
        else:
            shown = self.shown_network()
            if shown is not None and shown[0] != self.net_sent:
                identity, driver, title = shown
                self.net_queue.put(netwindow.network_message(driver, title))
                self.net_sent, self.net_driver = identity, driver
                self.net_values_at, self.net_values_step = float("-inf"), None
        self.send_network_values()

    def send_generations(self):
        """Send the window the timeline's generations it does not have yet (back from a race, the newest again, which
        brings its timeline back); the newest is the one whose live values follow."""
        if not self.net_history:
            return
        if self.net_sent is not None and self.net_sent[0] == "race":
            self.net_history_sent = min(self.net_history_sent, len(self.net_history) - 1)
        for entry in self.net_history[self.net_history_sent:]:
            self.net_queue.put(netwindow.generation_message(entry))
        self.net_history_sent = len(self.net_history)
        newest = self.net_history[-1]
        identity = ("train", newest["run"][1], newest["run"][2], newest["generation"])
        if identity != self.net_sent:
            self.net_sent, self.net_driver, self.net_member = identity, newest["driver"], newest["member"]
            self.net_values_at, self.net_values_step = float("-inf"), None

    def network_inputs(self):
        """(replay, row, where) of the shown car while it can drive: the selected race car, or in training the shown
        network's car if the replay of its generation drives it; None otherwise."""
        if self.net_driver is None or self.net_sent is None:
            return None
        if self.net_sent[0] == "race":
            car, cars = tuple(self.net_sent[1:]), self.race_cars()
            if self.race is None or car not in cars:
                return None
            return self.race, cars.index(car), self.race_name().split(" · pole")[0]
        members = [] if self.playback is None else [int(i) for i in self.indices]
        if self.playback_generation != self.net_sent[3] or self.net_member not in members:
            return None
        level = config.DIFFICULTIES.get(self.playback_report["level"], "")
        return self.playback, members.index(self.net_member), self.playback.track.name or level

    def send_network_values(self):
        """Send the shown car's node values (netwindow.values_message) at most every NET_FRAME_INTERVAL s, and only
        when it has driven since the last; never for a car that is done."""
        found = self.network_inputs()
        if found is None:
            return
        playback, row, where = found
        if playback.inputs is None or playback.cars.done[row] or playback.elapsed == self.net_values_step:
            return
        now = time.monotonic()
        if now - self.net_values_at < NET_FRAME_INTERVAL:
            return
        self.net_queue.put(netwindow.values_message(self.net_driver, playback.inputs[row], where))
        self.net_values_at, self.net_values_step = now, playback.elapsed

    # ----- settings (changeable only while no training runs) -----
    def select_approach(self, approach):
        if self.process is not None or approach == self.approach:
            return
        self.stop_training()
        self.approach = approach
        self.clear_view()
        self.status = "Press Play to start training." if self.source == "app" else "Looking for the notebook run."

    def change_population(self, direction):
        if self.process is not None:
            return
        presets = list(config.POPULATION_PRESETS)
        current = self.population[self.approach]
        position = presets.index(current) if current in presets else 0
        self.population[self.approach] = presets[min(max(position + direction, 0), len(presets) - 1)]

    def run_choices(self):
        """The runs the run control cycles through: runs 0 to 3 for training here; for a notebook run also any other
        run in the shown folder, such as an archive's (spec §10.2)."""
        if self.source != "notebook":
            return list(config.RUN_SEEDS)
        return sorted({*config.RUN_SEEDS, *paths.run_seeds(self.approach, self.runs_dir)})

    def change_run_seed(self, direction):
        if self.process is not None:
            return
        seeds = self.run_choices()
        if self.run_seed in seeds:
            here = seeds.index(self.run_seed)
        else:                                       # a run the shown folder does not hold: start at either end
            here = -1 if direction > 0 else 0
        self.run_seed = seeds[(here + direction) % len(seeds)]
        if self.source == "notebook":
            self.clear_view()

    def set_difficulty(self, level):
        if self.process is not None or self.source == "notebook":
            return
        self.difficulty = level
        if self.approach not in config.EVOLUTION_APPROACHES:
            self.playback = None                     # the imitation views drive the map of this level

    def population_size(self):
        if self.latest is not None:
            return len(self.latest["fitness"])
        return self.population.get(self.approach, 1)

    def set_show(self, mode):
        self.show = mode
        self.playback = None

    def set_speed(self, speed):
        self.speed = speed

    # ----- window -----
    @property
    def grip_scale(self):
        """The display grip as a multiple of a(v) (config.APP_GRIPS)."""
        return config.APP_GRIPS[self.grip_name]

    def set_grip(self, name):
        """Display grip for the viewer's races and training replays (spec §10.5). Training, evaluation and the
        expert always use a(v). Locked while a race runs; a running replay restarts with the new grip, a
        stopped one stays frozen and the next replay uses it."""
        if name not in config.APP_GRIPS or name == self.grip_name or self.racing:
            return
        self.grip_name = name
        if self.mode == "race":
            self.reset_race()
        elif not self.halted:
            self.playback = None

    def toggle_grip(self):
        names = list(config.APP_GRIPS)
        self.set_grip(names[(names.index(self.grip_name) + 1) % len(names)])

    # ----- playback -----
    def start_playback(self, report, track):
        fitness = np.asarray(report["fitness"], dtype=float)
        self.indices = select_indices(fitness, self.show, self.shown, sample_seed=report["generation"])
        self.playback = Playback(track, report_policy(report, self.indices), len(self.indices), laps=config.LAPS,
                                 setup=report_setup(report, self.indices), grip_scale=self.grip_scale)
        self.best_slot = int(np.argmax(fitness[self.indices]))
        if report is not self.playback_report and self.process is not None:
            self.status = f"Training {config.LABELS[self.approach]}, run {self.run_seed}."
        self.playback_report = report
        self.playback_generation = report["generation"]
        self.step_budget = 0.0

    def shown_track(self):
        """The track on screen in the training view: the playback's, or for imitation approaches the map they are
        driven on, which is known before training starts."""
        if self.playback is not None:
            return self.playback.track
        if self.approach not in config.EVOLUTION_APPROACHES and self.source == "app":
            return self.request_track(view_seed(self.difficulty), self.difficulty)
        return None

    def map_text(self):
        track = self.shown_track()
        if track is not None:
            if track.level in (config.MAX_LEVEL, config.TEST_LEVEL) and track.name:
                reversed_ = " (reversed)" if track.level == config.MAX_LEVEL and track.seed % 2 else ""
                return f"Map: {circuits.NAMES.get(track.name, track.name)}{reversed_}"
            return (f"Map: seed {track.seed}, level {track.level}, "
                    f"road {track.widths.min():.1f}-{track.widths.max():.1f} m")
        if self.source == "notebook" and self.latest is None:
            return "Map: shown with the run's latest generation"
        if self.approach in config.EVOLUTION_APPROACHES and self.latest is None:
            return "Map: shown with the first generation"
        return "Map: building the track"

    def set_mode(self, mode):
        entering = mode != self.mode
        self.mode = mode
        self.step_budget = 0.0
        if mode == "race":
            self.acknowledge()                       # nobody watches the training: it continues at full speed
        if mode == "race" and (entering or not self.race_available):
            self.load_race()
        elif mode == "train" and self.status.startswith("No trained controllers"):
            self.status = "Training view."

    # ----- race view -----
    def load_race(self):
        """Find the saved final controllers, each method's best run and the app cars (spec §10.3, §10.6). The runs
        control and the methods' switches are kept while the view is open; a method seen for the first time is
        switched on. Without a readable best-run record, best races run 0."""
        self.race_available = nets.champion_seeds(self.champions_dir)
        self.app_available = nets.champion_seeds(self.app_champions_dir)
        self.drivers = {}
        self.failed = set()
        self.best = {}
        note = ""
        best_path = paths.best_seeds_path(self.champions_dir)
        if not best_path.exists():
            note = f"no best-run record yet (the evaluation, notebook 08, writes it); best races run {SEEDED_RUN}."
        else:
            try:
                for approach, seed in json.loads(best_path.read_text()).items():
                    if isinstance(seed, int) and not isinstance(seed, bool):
                        self.best[approach] = seed
            except (ValueError, TypeError, AttributeError, OSError):
                self.best = {}
                note = f"{best_path.name} could not be read; best races run {SEEDED_RUN}."
        for approach in self.race_available:
            self.race_on.setdefault(approach, True)
        kept_app = self.app_choice
        self.app_choice = {approach: kept_app.get(approach) if kept_app.get(approach) in seeds else "off"
                           for approach, seeds in self.app_available.items()}     # app cars start switched off
        if not self.app_available:
            self.race_group = "final"
        if not self.race_available and not self.app_available:
            self.status = f"No trained controllers found in {self.champions_dir}."
        else:
            self.status = note or "Race view."
        self.reset_race()

    def race_seeds(self, approach):
        """The saved runs of `approach` that race (spec §10.3): its best run for 'best' (run 0 without a record),
        run 0 for 'seeded', runs 1 to 3 for 'unseeded', every saved run for 'all'; none when it is switched off."""
        if not self.race_on.get(approach, True):
            return []
        saved = self.race_available[approach]
        if self.race_runs == "best":
            choice = self.best.get(approach, SEEDED_RUN)
            return [choice] if choice in saved else saved[:1]
        if self.race_runs == "seeded":
            return [seed for seed in saved if seed == SEEDED_RUN]
        if self.race_runs == "unseeded":
            return [seed for seed in saved if seed != SEEDED_RUN]
        return list(saved)

    def race_runs_label(self):
        return f"Race: {self.race_runs} runs ▸"

    def cycle_race_runs(self):
        """Race the best runs, all runs, the seeded runs or the unseeded runs (spec §10.3). Not during a race."""
        if self.racing:
            return
        self.race_runs = RACE_RUNS[(RACE_RUNS.index(self.race_runs) + 1) % len(RACE_RUNS)]
        self.reset_race()

    def toggle_race(self, approach):
        """Switch a method's runs on or off (spec §10.3). Not during a race."""
        if self.racing:
            return
        self.race_on[approach] = not self.race_on.get(approach, True)
        self.reset_race()

    def app_choice_label(self, approach):
        choice = self.app_choice.get(approach, "off")
        return "off" if choice == "off" else f"seed {choice}"

    def cycle_app_choice(self, approach):
        if self.racing:
            return
        options = ["off", *self.app_available[approach]]
        current = self.app_choice.get(approach, "off")
        self.app_choice[approach] = options[(options.index(current) + 1) % len(options)]
        self.reset_race()

    def race_group_label(self):
        return f"Controllers: {'app runs' if self.race_group == 'app' else 'reported'} ▸"

    def cycle_race_group(self):
        """Switch the controller list between the reported controllers and the app cars (spec §10.6)."""
        if self.app_available:
            self.race_group = "app" if self.race_group == "final" else "final"

    def race_cars(self):
        """The cars of the race: the chosen reported controllers (approach, seed), then the chosen app cars
        (approach, seed, "app")."""
        return self.lineup + self.app_lineup

    def drop_car(self, car):
        """Leave a car whose controller cannot be loaded out of the race."""
        self.failed.add(car)
        for cars in (self.lineup, self.app_lineup):
            if car in cars:
                cars.remove(car)

    def reset_race(self):
        """Back to the map preview: the selected controllers and app cars wait on the start line."""
        self.racing = False
        self.race = None
        self.ghost = None
        self.step_budget = 0.0
        self.lineup = [(approach, seed) for approach in self.race_available for seed in self.race_seeds(approach)
                       if (approach, seed) not in self.failed]
        self.app_lineup = [(approach, choice, "app") for approach, choice in self.app_choice.items()
                           if choice != "off" and (approach, choice, "app") not in self.failed]
        if self.follow not in self.race_cars():
            self.follow = None

    def driver(self, approach, seed, source="final"):
        """The driver of a reported controller, or of an app car (source "app"), loaded once."""
        key = (approach, seed) if source == "final" else (approach, seed, source)
        if key not in self.drivers:
            directory = self.champions_dir if source == "final" else self.app_champions_dir
            model, driver = nets.load_champion(approach, seed, directory)
            inputs = model.input_shape[-1]
            expected = self.folder_inputs if source == "final" else config.N_INPUTS     # app cars are this revision's
            if inputs != expected:                           # a controller of another revision (spec §10.3)
                raise ValueError(f"made for {inputs} inputs, these runs have {expected}")
            self.drivers[key] = driver
        return self.drivers[key]

    def race_tracks(self):
        """(seeds, level) of the race view's tracks (spec §4.6, §10.3): the test circuits at Extra hard, otherwise
        the display seeds at the chosen level."""
        if self.race_level == config.MAX_LEVEL:
            return config.TEST_TRACKS, config.TEST_LEVEL
        return config.DISPLAY_SEEDS, self.race_level

    def race_track(self):
        seeds, level = self.race_tracks()
        return self.request_track(seeds[self.race_index % len(seeds)], level)

    def pole_record(self):
        """The selected test circuit's pole lap (circuits.PoleLap), or None at the generated levels."""
        seeds, level = self.race_tracks()
        if level != config.TEST_LEVEL:
            return None
        return circuits.POLE_LAPS[config.TEST_CIRCUITS[seeds[self.race_index % len(seeds)]]]

    def pole_lap(self):
        """The pole lap in seconds of the selected test circuit, or None at the generated levels."""
        record = self.pole_record()
        return None if record is None else record.seconds

    def pole_ghost(self, track):
        """The selected test circuit's pole ghost on `track`, or None at the generated levels."""
        record = self.pole_record()
        if record is None:
            return None
        key = (track.name, track.length, record)
        if key not in self.ghosts:
            self.ghosts[key] = PoleGhost(track, record, config.RACE_LAPS)
        return self.ghosts[key]

    def race_name(self):
        """The selected race track as shown in the panel (spec §10.3): a test circuit with its pole lap, otherwise
        the display seed and the difficulty."""
        seeds, level = self.race_tracks()
        seed = seeds[self.race_index % len(seeds)]
        if level == config.TEST_LEVEL:
            name = config.TEST_CIRCUITS[seed]
            pole = circuits.POLE_LAPS[name]
            return f"{circuits.NAMES[name]} · pole {pole.seconds:.3f} s ({pole.driver}, {pole.season})"
        return f"seed {seed} · {config.APP_DIFFICULTIES[level]}"

    def prepare_race(self):
        """Build the preview once the track is ready: every selected car on the start line."""
        if self.race is None and self.race_cars():
            track = self.race_track()
            if track is not None:
                drivers = []
                for car in list(self.race_cars()):
                    try:
                        drivers.append(self.driver(*car))
                    except Exception as error:
                        self.drop_car(car)
                        self.status = f"{car_method(car)} run {car[1]} left out: {error}."
                if drivers:
                    self.race = Playback(track, nets.mixed_policy(drivers), len(drivers), laps=config.RACE_LAPS,
                                         setup=nets.drivers_setup(drivers), grip_scale=self.grip_scale)
                    self.ghost = self.pole_ghost(track)

    def start_race(self):
        self.prepare_race()
        if self.race is not None:
            self.racing = True
            self.status = "Racing."

    def stop_race(self):
        self.reset_race()
        self.status = "Race stopped. Change the controllers, track or difficulty, or start again."

    def change_race_track(self, direction):
        if self.racing:
            return
        self.race_index = (self.race_index + direction) % len(self.race_tracks()[0])
        self.reset_race()

    def set_race_level(self, level):
        if self.racing:
            return
        self.race_level = level
        self.reset_race()

    def update(self, dt):
        self.poll_reports()
        self.record_generation()
        self.poll_network_window()
        if self.mode == "train" and self.source == "notebook":
            if not self.replay_paused and (self.playback is None or self.playback.over):
                self.poll_replay()
                if self.latest is not None:              # the newest generation, or the same one again
                    track = self.request_track(self.latest["seeds"][0], self.latest["level"])
                    if track is not None:
                        self.start_playback(self.latest, track)
            active = None if self.replay_paused else self.playback
        elif self.mode == "train":
            frozen = self.halted or self.paused
            if not frozen and self.latest is not None:
                if self.playback is not None and self.playback.over and self.playback_report is self.latest:
                    self.acknowledge()               # replayed once: the next generation trains (in step)
                elif self.playback is None or self.playback.over:
                    track = self.request_track(self.latest["seeds"][0], self.latest["level"])
                    if track is not None:
                        self.start_playback(self.latest, track)
            active = None if frozen else self.playback
        else:
            self.prepare_race()
            active = self.race if self.racing else None
        if active is not None and self.watched:
            self.step_budget += self.speed * config.PLAYBACK_BASE * dt / config.DT
            steps = int(self.step_budget)
            self.step_budget = 0.0 if steps > config.MAX_STEPS_PER_FRAME else self.step_budget - steps
            steps = min(steps, config.MAX_STEPS_PER_FRAME)
            active.advance(steps)

    # ----- commands -----
    def command(self, name, arg=None):
        """Run one of the page's buttons: `name` from COMMANDS with its argument. ValueError for an unknown name or a
        bad argument; nothing changes then."""
        if name not in COMMANDS:
            raise ValueError(f"unknown command {name!r}")
        COMMANDS[name](self, arg)

    # ----- drawing -----
    def leader_speed_line(self):
        """The race leader's name and speed, '1st  KT Random r1  194 mph'; None before a race."""
        if self.race is None or len(self.race.cars.x) == 0:
            return None
        i = race_leader(self.race.cars)
        car = self.race_cars()[i]
        name = car_name(car[0], car[1], app=is_app(car), archived=self.archived())
        return f"1st  {name}  {speed_text(float(self.race.cars.speed[i]))}"


def _one_of(arg, options, what):
    try:
        found = not isinstance(arg, bool) and arg in options
    except TypeError:                   # an unhashable argument
        found = False
    if not found:
        raise ValueError(f"{what}: {arg!r} is not one of {list(options)}")
    return arg


def _direction(arg):
    return _one_of(arg, (-1, 1), "direction")


def _whole(arg, what):
    if isinstance(arg, bool) or not isinstance(arg, (int, np.integer)):
        raise ValueError(f"{what}: {arg!r} is not a whole number")
    return int(arg)


def _follow(app, arg):
    if arg is not None and not 0 <= _whole(arg, "follow") < len(app.race_cars()):
        raise ValueError(f"follow: {arg!r} is not a car of the race")
    app.set_follow(arg)


def _app_choice(app, arg):
    if arg not in app.app_available:
        raise ValueError(f"app_choice: {arg!r} has no app car")
    app.cycle_app_choice(arg)


COMMANDS = {        # the page's commands (name -> function of the app and its argument)
    "mode": lambda app, arg: app.set_mode(_one_of(arg, ("train", "race"), "mode")),
    "approach": lambda app, arg: app.select_approach(_one_of(arg, config.APPROACHES, "approach")),
    "source": lambda app, arg: app.toggle_source(),
    "folder": lambda app, arg: app.cycle_folder(),
    "run": lambda app, arg: app.change_run_seed(_direction(arg)),
    "difficulty": lambda app, arg: app.set_difficulty(_one_of(arg, config.DIFFICULTIES, "difficulty")),
    "population": lambda app, arg: app.change_population(_direction(arg)),
    "play": lambda app, arg: app.toggle_play(),
    "stop": lambda app, arg: app.stop(),
    "new_run": lambda app, arg: app.new_run(),
    "show": lambda app, arg: app.set_show(_one_of(arg, ("all", "best", "worst"), "show")),
    "shown": lambda app, arg: app.set_shown(_whole(arg, "shown")),
    "speed": lambda app, arg: app.set_speed(_one_of(arg, config.SPEEDS, "speed")),
    "grip": lambda app, arg: app.set_grip(_one_of(arg, config.APP_GRIPS, "grip")),
    "network": lambda app, arg: app.toggle_network_window(),
    "track": lambda app, arg: app.change_race_track(_direction(arg)),
    "race_level": lambda app, arg: app.set_race_level(_one_of(arg, config.APP_DIFFICULTIES, "race_level")),
    "start": lambda app, arg: app.start_race(),
    "race_runs": lambda app, arg: app.cycle_race_runs(),
    "method": lambda app, arg: app.toggle_race(_one_of(arg, config.APPROACHES, "method")),
    "group": lambda app, arg: app.cycle_race_group(),
    "app_choice": _app_choice,
    "follow": _follow,
    "quit": lambda app, arg: app.shutdown(),
}


def main(argv=None):
    """Start the viewer: serve the page, open the browser and run until the Quit button or Ctrl+C."""
    import argparse
    import logging
    import subprocess
    import webbrowser
    from racing import server as web

    parser = argparse.ArgumentParser(prog="python -m racing.app", description="The EC Racing viewer.")
    parser.add_argument("--port", type=int, default=config.VIEWER_PORT, help="first port to try (default %(default)s)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    options = parser.parse_args(argv)

    config.VIEWER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(config.VIEWER_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log = logging.getLogger("racing.viewer")
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    viewer = App()
    viewer_server = web.ViewerServer(viewer, port=options.port)
    try:
        log.info("viewer starting")
        url = viewer_server.start()
        print("EC Racing viewer")
        print(f"  open:     {url}")
        print(f"  results:  {config.RESULTS_DIR}")
        print(f"  log:      {config.VIEWER_LOG_PATH}")
        print("  quit:     Ctrl+C here, or the Quit button", flush=True)
        if not options.no_browser:
            try:
                command = web.browser_command(url, Path("/proc/version").read_text())
            except OSError:
                command = None
            try:
                if command is not None:
                    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    webbrowser.open(url)
            except OSError as error:
                log.warning("could not open the browser: %s", error)
                print(f"  (could not open the browser: {error})")
        viewer_server.run_forever()
    except KeyboardInterrupt:
        log.info("Ctrl+C")
    except Exception:
        log.exception("viewer failed")
        raise
    finally:
        viewer_server.stop()
        viewer.shutdown()
        log.removeHandler(handler)
        handler.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
