"""EC Racing viewer (spec §10): live training view and race view through one top-down camera.

Run with:  ~/miniconda3/bin/python -m racing.app
"""
import csv
import json
import multiprocessing as mp
import os
import pickle
import queue
import signal
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from racing import checkpoint, config, curriculum, netwindow, nets, paths, population, sim
from racing.actions import class_to_controls
from racing.tracks import get_track

GRASS = (214, 230, 205)
ROAD = (92, 96, 104)
EDGE = (245, 245, 245)
PANEL_BG = (248, 249, 251)
INK = (29, 35, 48)
MUTED = (91, 100, 116)
BUTTON = (228, 232, 238)
BUTTON_ACTIVE = (55, 62, 78)     # dark neutral: never mistaken for an approach (spec §10.1)
DISABLED = (238, 240, 243)       # a locked button's fill
LOCKED_TINT = 0.35               # a locked button that is on keeps this much of its color
RAY = (255, 196, 0)
WHITE = (255, 255, 255)
MIN_LINE_CONTRAST = 3.0          # a plot line with less contrast against white gets a dark outline (WCAG 2)


def hex_to_rgb(code):
    code = code.lstrip("#")
    return tuple(int(code[i:i + 2], 16) for i in (0, 2, 4))


def luminance(rgb):
    """Relative luminance of an sRGB color (WCAG 2): 0 for black, 1 for white."""
    c = np.asarray(rgb[:3], dtype=float) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(c @ np.array([0.2126, 0.7152, 0.0722]))


def contrast(a, b):
    """Contrast ratio of two colors (WCAG 2), from 1 to 21."""
    light, dark = sorted((luminance(a), luminance(b)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def ink_for(fill):
    """Text color on `fill`: dark ink on light colors such as NEAT's yellow, white on dark ones, whichever
    contrasts more."""
    return INK if contrast(fill, INK) >= contrast(fill, WHITE) else WHITE


def needs_outline(color, background=WHITE):
    """Whether a line or swatch in `color` needs a dark outline to stand out from `background`."""
    return contrast(color, background) < MIN_LINE_CONTRAST


UNUSABLE_RUN = 3                    # trainer exit code: the saved viewer run cannot be resumed (spec §10.6)
UNUSABLE_ERRORS = (checkpoint.SettingsMismatch, checkpoint.UnreadableRecord, EOFError, pickle.UnpicklingError)
SEEDED_RUN = getattr(config, "SEEDED_RUN", 0)   # run 0 is the seeded run; the fallback goes once config defines it
RACE_RUNS = ("all", "seeded", "unseeded")       # the race control's choices, in cycling order (spec §10.3)


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


class Camera:
    """Maps world coordinates (m) to pixels so that a track fills the view rectangle; with `zoom` above 1 the scale
    is that many times larger and the view is centred on `center` (a point in m, such as a followed car)."""

    def __init__(self, track, rect, zoom=1, center=None):
        edges = np.vstack([track.left, track.right])
        lo, hi = edges.min(axis=0), edges.max(axis=0)
        self.scale = 0.94 * min(rect.width / (hi[0] - lo[0]), rect.height / (hi[1] - lo[1])) * zoom
        self.center = (lo + hi) / 2 if center is None else np.asarray(center, dtype=float)
        self.rect = rect

    def to_screen(self, x, y):
        sx = self.rect.centerx + (np.asarray(x) - self.center[0]) * self.scale
        sy = self.rect.centery - (np.asarray(y) - self.center[1]) * self.scale
        return sx, sy


def render_track(track, camera, size):
    """Grass, road surface, edges and start line, drawn once per track."""
    surface = pygame.Surface(size)
    surface.fill(GRASS)
    lx, ly = camera.to_screen(track.left[:, 0], track.left[:, 1])
    rx, ry = camera.to_screen(track.right[:, 0], track.right[:, 1])
    m = len(lx)
    for i in range(m):
        j = (i + 1) % m
        pygame.draw.polygon(surface, ROAD, [(lx[i], ly[i]), (lx[j], ly[j]), (rx[j], ry[j]), (rx[i], ry[i])])
    pygame.draw.lines(surface, EDGE, True, list(zip(lx, ly)), 2)
    pygame.draw.lines(surface, EDGE, True, list(zip(rx, ry)), 2)
    pygame.draw.line(surface, INK, (lx[0], ly[0]), (rx[0], ry[0]), 3)
    return surface


def visible_segments(track, camera):
    """Indices i of the road segments from centerline point i to i + 1 that reach into the camera's view."""
    lx, ly = camera.to_screen(track.left[:, 0], track.left[:, 1])
    rx, ry = camera.to_screen(track.right[:, 0], track.right[:, 1])
    nxt = np.roll(np.arange(len(lx)), -1)
    xs = np.stack([lx, lx[nxt], rx[nxt], rx])
    ys = np.stack([ly, ly[nxt], ry[nxt], ry])
    rect = camera.rect
    inside = ((xs.max(axis=0) >= rect.left) & (xs.min(axis=0) <= rect.right)
              & (ys.max(axis=0) >= rect.top) & (ys.min(axis=0) <= rect.bottom))
    return np.flatnonzero(inside)


def draw_track_part(surface, track, camera):
    """Road, edges and start line of the part of the track in the camera's view, drawn directly: a zoomed view draws
    this every frame instead of keeping a picture of the whole track at its scale."""
    lx, ly = camera.to_screen(track.left[:, 0], track.left[:, 1])
    rx, ry = camera.to_screen(track.right[:, 0], track.right[:, 1])
    m = len(lx)
    segments = visible_segments(track, camera)
    for i in segments:
        j = (i + 1) % m
        pygame.draw.polygon(surface, ROAD, [(lx[i], ly[i]), (lx[j], ly[j]), (rx[j], ry[j]), (rx[i], ry[i])])
    for i in segments:
        j = (i + 1) % m
        pygame.draw.line(surface, EDGE, (lx[i], ly[i]), (lx[j], ly[j]), 2)
        pygame.draw.line(surface, EDGE, (rx[i], ry[i]), (rx[j], ry[j]), 2)
    if 0 in segments:
        pygame.draw.line(surface, INK, (lx[0], ly[0]), (rx[0], ry[0]), 3)


def car_polygon(sx, sy, heading, size):
    """Triangle pointing along the heading. Screen y points down, so world rotations flip sign."""
    c, s = np.cos(heading), np.sin(heading)
    shape = [(size, 0.0), (-0.6 * size, 0.5 * size), (-0.6 * size, -0.5 * size)]
    return [(sx + px * c - py * s, sy - (px * s + py * c)) for px, py in shape]


class Playback:
    """Re-simulates a set of controllers on one track, a few steps at a time, for display. Each car drives with its
    own setup and every car with the viewer's display grip (spec §10.5); training never uses this class."""

    def __init__(self, track, policy, n, laps=1, time_limit=None, setup=None, grip=config.GRIP):
        self.track = track
        self.policy = policy
        self.grip = float(grip)
        self.cars = sim.start_cars(track, n, laps, setup)
        self.time_limit = sim.time_limit_for(track, laps) if time_limit is None else time_limit

    @property
    def over(self):
        return bool(self.cars.done.all())

    def advance(self, steps):
        for _ in range(steps):
            if self.over:
                return
            steer, throttle = class_to_controls(self.policy(sim.sense(self.track, self.cars), self.cars))
            sim.step(self.track, self.cars, steer, throttle, self.time_limit, grip=self.grip)


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
    return population.n_inputs if isinstance(population, nets.DensePopulation) else None


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


def folder_profile(name, folder):
    """(inputs, road widths) of a results folder's networks and maps: this revision's (config.N_INPUTS, None: each
    map draws its own width) for the current results; an archive's from its run_info.json settings, where an earlier
    revision names one road width per level (revision 3.1). An archive without a readable record is taken as current."""
    if name == "current":
        return config.N_INPUTS, None
    try:
        settings = json.loads((Path(folder) / config.RUN_INFO_PATH.name).read_text())["settings"]
        inputs = int(settings["N_INPUTS"])
        levels = settings["LEVELS"]
        widths = ({int(level): float(spec["width"]) for level, spec in levels.items()}
                  if all("width" in spec for spec in levels.values()) else None)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return config.N_INPUTS, None
    return inputs, widths


def report_setup(report, indices):
    """Setups of the chosen members of a training report: the population's rows, or each imitation network's own
    (spec §9.11)."""
    if "population" in report:
        return report["population"].subset(indices).setup
    return nets.drivers_setup([report["nets"][i] for i in indices])


def leaderboard(playback, count):
    """Rows (position, car index, laps completed, last lap, best lap, status), ordered by distance covered."""
    nan = float("nan")
    if playback is None:
        return [(k + 1, k, 0, nan, nan, "ready") for k in range(count)]
    cars = playback.cars
    rows = []
    for position, i in enumerate(np.argsort(-cars.progress, kind="stable")):
        if cars.crashed[i]:
            status = "crashed"
        elif cars.finished[i]:
            status = "finished"
        elif cars.done[i]:
            status = "stopped"
        else:
            status = "racing"
        rows.append((position + 1, int(i), int(cars.laps_done[i]), float(cars.last_lap[i]), float(cars.best_lap[i]),
                     status))
    return rows


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
    the revision 3 setup. The file is replaced in one step, so the race view never reads half a model."""
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
                initial = es.dagger_baseline(seed) if approach == "es_seeded" else None
                es.run_es(**common, initial=initial, approach=approach)
        except UNUSABLE_ERRORS as error:
            print(f"The saved viewer run in {out_dir} cannot be resumed: {error}", file=sys.stderr, flush=True)
            return UNUSABLE_RUN
        return 0
    paths.remove(out_dir)                   # imitation training in the viewer is never resumed
    path = config.DATASET_PATH
    if not path.exists():
        path = app_runs_dir / "dataset.npz"     # never write the reported dataset
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
              "seeds": [config.KT_VIEW_SEED], "nets": list(members), "fitness": fitness,
              "best": float(fitness.max()), "mean": float(fitness.mean()), "worst": float(fitness.min()),
              "leveled_up": False})

    if approach == "baseline":                         # resume=False: always train live, never skip (spec §10.6)
        imitation.train_baseline(data, out_dir=out_dir, seed=seed, resume=False, report=False,
                                 on_epoch=lambda epoch, model, acc: add(epoch, model, acc))
    else:
        imitation.run_tuner(approach, data, out_dir=out_dir, on_trial=add, seed=seed, export_models=False,
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


def car_name(approach, seed, app=False):
    """A race car's name in the leaderboard, the network selector and the window title: run 0 'GA (seeded)', run 1
    'GA', any other run (of an earlier revision) 'GA s2' (spec §10.3); an app car keeps its seed, 'GA (app) s1'
    (spec §10.6)."""
    label = config.LABELS[approach]
    if app:
        return f"{label} (app) s{seed}"
    if seed == SEEDED_RUN:
        return f"{label} (seeded)"
    return label if seed == 1 else f"{label} s{seed}"


def leaderboard_line(position, approach, seed, laps, last, best, status, app=False):
    """One leaderboard row: position, car name, laps completed, last/best lap time (s), status."""
    times = "/".join("–" if np.isnan(value) else f"{value:.1f}" for value in (last, best))
    return f"{position}. {car_name(approach, seed, app)}  lap {laps}  {times}  {status}"


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
    """The plot history of a run from its log.csv: (generation, lowest map share, nan, marked) per generation, level
    advancements, the goal and target cuts marked; empty for a missing or unreadable log."""
    if not Path(log).exists():
        return []
    history = []
    try:
        with open(log, newline="") as f:
            for row in csv.DictReader(f):
                history.append((int(row["generation"]), float(row["min_share"]), float("nan"),
                                row.get("event") in ("level_up", "goal", "tightened")))
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

    def __init__(self, screen, background_tracks=True):
        self.screen = screen
        self.font = pygame.font.SysFont("dejavusans", 15)
        self.small = pygame.font.SysFont("dejavusans", 13)
        self.fullscreen = False
        self.layout()
        self.alive = True
        self.mode = "train"
        self.buttons = []
        self.slider = None
        self.dragging = False
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
        self.runs_dir = config.RUNS_DIR                 # the shown folder's notebook runs
        self.train_runs_dir = config.RUNS_DIR           # where the seeded ES trained here finds the DAgger baseline
        self.app_runs_dir = config.APP_RUNS_DIR             # saved viewer runs (spec §10.6)
        self.app_champions_dir = config.APP_CHAMPIONS_DIR   # their best cars, raceable in the race view
        self.show = "best"
        self.shown = config.DEFAULT_SHOWN
        self.rays = False
        self.zoom_index = 0                 # into config.CAMERA_ZOOMS; above 1× the camera follows a car
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
        self.net_choice = 0                 # index into the race lineup
        self.net_sent = None                # identity of the network the window shows
        # race view
        self.champions_dir = config.CHAMPIONS_DIR
        self.race_index = 0                 # into config.TEST_SEEDS
        self.race_level = config.MAX_LEVEL
        self.race_available = {}            # approach -> run seeds with a saved controller
        self.race_runs = "all"              # the runs that race: "seeded", "unseeded" or "all" (spec §10.3)
        self.race_on = {}                   # approach -> whether its runs race
        self.drivers = {}                   # (approach, seed) -> DenseNet
        self.lineup = []                    # (approach, seed) of the cars in the race
        self.race = None                    # Playback: the preview until the race starts
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
        if key in self.tracks:
            return self.tracks[key]
        if not self.background_tracks:
            self.build_track(key)
            return self.tracks.get(key)
        if key not in self.building:
            self.building.add(key)
            threading.Thread(target=self.build_track, args=(key,), daemon=True).start()
        return None

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
        self.status = f"Training {config.LABELS[self.approach]}, seed {self.run_seed}."

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
        name = f"{config.LABELS[self.approach]} seed {self.run_seed}"
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
            marked = report.get("event") in ("level_up", "goal", "tightened")
            self.history.append((report["generation"], float(report["min_share"]), float("nan"), marked))
        else:
            self.history.append((report["generation"], report["best"], report["mean"], report["leveled_up"]))

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

    def selected_car(self):
        """The race car whose network the window shows: the first car if the choice is outside the race; None for
        an empty race."""
        cars = self.race_cars()
        if not cars:
            return None
        return cars[self.net_choice if 0 <= self.net_choice < len(cars) else 0]

    def cycle_network_car(self):
        cars = self.race_cars()
        if cars:
            self.net_choice = (cars.index(self.selected_car()) + 1) % len(cars)

    def network_car_label(self):
        car = self.selected_car()
        return "Network of: –" if car is None else f"Network of: {car_name(car[0], car[1], is_app(car))} ▸"

    def shown_network(self):
        """(identity, driver, title) of the network the window should show now, or None."""
        if self.mode == "train":
            report = self.latest
            if report is None:
                return None
            approach, generation = report["approach"], report["generation"]
            seed = report.get("seed", self.run_seed)
            identity = ("train", approach, seed, generation)
            if identity == self.net_sent:
                return identity, None, None            # unchanged: build nothing
            if "population" in report:
                driver, unit = report["population"].driver(int(np.argmax(report["fitness"]))), "generation"
            else:
                driver, unit = report["nets"][-1], "trial" if approach.startswith("kt") else "epoch"
            title = (f"{config.LABELS[approach]} · seed {seed} · {unit} {generation} · "
                     f"{driver.n_params:,} parameters{netwindow.netfig.setup_text(driver)}")
            return identity, driver, title
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
        return identity, driver, (f"{car_name(car[0], car[1], is_app(car))} · {driver.n_params:,} parameters"
                                  f"{netwindow.netfig.setup_text(driver)}")

    def poll_network_window(self):
        """Forget a window the user closed; send the shown network when it changed."""
        if self.net_process is None:
            return
        if not self.net_process.is_alive():
            self.reset_network_window()
            return
        shown = self.shown_network()
        if shown is not None and shown[0] != self.net_sent:
            identity, driver, title = shown
            self.net_queue.put(netwindow.network_message(driver, title))
            self.net_sent = identity

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

    def change_run_seed(self, direction):
        if self.process is not None:
            return
        seeds = list(config.RUN_SEEDS)
        self.run_seed = seeds[(seeds.index(self.run_seed) + direction) % len(seeds)]
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
    def layout(self):
        """The map view and the panel for the window's current size; the pictures of whole tracks are redrawn at
        the new size."""
        width, height = self.screen.get_size()
        self.view = pygame.Rect(0, 0, max(width - config.PANEL_WIDTH, 1), height)
        self.panel = pygame.Rect(self.view.right, 0, config.PANEL_WIDTH, height)
        self.surfaces = {}

    def toggle_fullscreen(self):
        """Full screen at the display's resolution, or back to a resizable window of config.WINDOW_SIZE."""
        if self.fullscreen:
            self.screen = pygame.display.set_mode(config.WINDOW_SIZE, pygame.RESIZABLE)
            self.fullscreen = False
        else:
            try:
                screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            except pygame.error as error:
                self.status = f"Full screen is not available here ({error})."
                return
            if screen.get_width() <= config.PANEL_WIDTH:                    # no usable display size reported
                self.screen = pygame.display.set_mode(config.WINDOW_SIZE, pygame.RESIZABLE)
                self.status = "Full screen is not available here."
                self.layout()
                return
            self.screen = screen
            self.fullscreen = True
        self.layout()

    def resized(self):
        """The window was resized by dragging its border."""
        if not self.fullscreen:
            self.screen = pygame.display.get_surface() or self.screen
            self.layout()

    def camera_zoom(self):
        return config.CAMERA_ZOOMS[self.zoom_index]

    def camera_label(self):
        zoom = self.camera_zoom()
        return f"Camera: {'Whole map' if zoom == 1 else f'Follow {zoom}×'} ▸"

    def cycle_camera(self):
        self.zoom_index = (self.zoom_index + 1) % len(config.CAMERA_ZOOMS)

    def zoom_by(self, steps):
        """Zoom in (steps > 0) or out by whole steps of config.CAMERA_ZOOMS, without wrapping (the mouse wheel)."""
        self.zoom_index = min(max(self.zoom_index + int(steps), 0), len(config.CAMERA_ZOOMS) - 1)

    def follow_point(self, playback, track):
        """Where the zoomed camera looks (m): the best car of the shown generation in training, the race leader (the
        car furthest along that is still racing, else furthest along) in a race; the start line without cars."""
        if playback is None or len(playback.cars.x) == 0:
            return (track.left[0] + track.right[0]) / 2
        cars = playback.cars
        if self.mode == "train":
            i = self.best_slot if self.best_slot < len(cars.x) else 0
        else:
            racing = np.flatnonzero(~cars.done)
            pool = racing if racing.size else np.arange(len(cars.x))
            i = int(pool[np.argmax(cars.progress[pool])])
        return np.array([cars.x[i], cars.y[i]])

    def view_camera(self, track, playback):
        """The camera of this frame: the whole map, or zoomed onto the followed car."""
        zoom = self.camera_zoom()
        if zoom == 1:
            return self.track_surface(track)[0]
        return Camera(track, self.view, zoom, self.follow_point(playback, track))

    def toggle_rays(self):
        self.rays = not self.rays

    @property
    def grip(self):
        return config.APP_GRIPS[self.grip_name]

    def set_grip(self, name):
        """Display grip for the viewer's races and training replays (spec §10.5). Training, evaluation and the
        expert always use config.GRIP. Locked while a race runs; a running replay restarts with the new grip, a
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

    def set_shown_from(self, x):
        fraction = min(max((x - self.slider.left) / self.slider.width, 0.0), 1.0)
        self.shown = 1 + int(round(fraction * (self.population_size() - 1)))
        self.playback = None

    # ----- playback -----
    def start_playback(self, report, track):
        fitness = np.asarray(report["fitness"], dtype=float)
        self.indices = select_indices(fitness, self.show, self.shown, sample_seed=report["generation"])
        self.playback = Playback(track, report_policy(report, self.indices), len(self.indices), laps=config.LAPS,
                                 setup=report_setup(report, self.indices), grip=self.grip)
        self.best_slot = int(np.argmax(fitness[self.indices]))
        if report is not self.playback_report and self.process is not None:
            self.status = f"Training {config.LABELS[self.approach]}, seed {self.run_seed}."
        self.playback_report = report
        self.playback_generation = report["generation"]
        self.step_budget = 0.0

    def shown_track(self):
        """The track on screen in the training view: the playback's, or for imitation approaches the map they are
        driven on, which is known before training starts."""
        if self.playback is not None:
            return self.playback.track
        if self.approach not in config.EVOLUTION_APPROACHES and self.source == "app":
            return self.request_track(config.KT_VIEW_SEED, self.difficulty)
        return None

    def map_text(self):
        track = self.shown_track()
        if track is not None:
            return f"Map: seed {track.seed}, level {track.level}, road {track.width:.1f} m"
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
        """Find the saved final controllers and app cars (spec §10.3, §10.6). The runs control and the methods'
        switches are kept while the view is open; a method seen for the first time is switched on."""
        self.race_available = nets.champion_seeds(self.champions_dir)
        self.app_available = nets.champion_seeds(self.app_champions_dir)
        self.drivers = {}
        self.failed = set()
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
            self.status = "Race view."
        self.reset_race()

    def race_seeds(self, approach):
        """The saved runs of `approach` that race (spec §10.3): run 0 for 'seeded', run 1 for 'unseeded', every
        saved run for 'all'; none when the approach is switched off."""
        if not self.race_on.get(approach, True):
            return []
        saved = self.race_available[approach]
        if self.race_runs == "seeded":
            return [seed for seed in saved if seed == SEEDED_RUN]
        if self.race_runs == "unseeded":
            return [seed for seed in saved if seed == 1]
        return list(saved)

    def race_runs_label(self):
        return f"Race: {self.race_runs} runs ▸"

    def cycle_race_runs(self):
        """Race all runs, the seeded runs or the unseeded runs (spec §10.3). Not during a race."""
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
        self.step_budget = 0.0
        selected = self.selected_car()
        self.lineup = [(approach, seed) for approach in self.race_available for seed in self.race_seeds(approach)
                       if (approach, seed) not in self.failed]
        self.app_lineup = [(approach, choice, "app") for approach, choice in self.app_choice.items()
                           if choice != "off" and (approach, choice, "app") not in self.failed]
        cars = self.race_cars()
        self.net_choice = cars.index(selected) if selected in cars else 0

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
            if not isinstance(driver, nets.DenseNet) and inputs != config.N_INPUTS:
                raise ValueError("an earlier revision's NEAT controllers cannot be shown")
            self.drivers[key] = driver
        return self.drivers[key]

    def race_track(self):
        return self.request_track(config.TEST_SEEDS[self.race_index], self.race_level)

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
                                         setup=nets.drivers_setup(drivers), grip=self.grip)

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
        self.race_index = (self.race_index + direction) % len(config.TEST_SEEDS)
        self.reset_race()

    def set_race_level(self, level):
        if self.racing:
            return
        self.race_level = level
        self.reset_race()

    def update(self, dt):
        self.poll_reports()
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
        if active is not None:
            self.step_budget += self.speed * config.PLAYBACK_BASE * dt / config.DT
            steps = int(self.step_budget)
            self.step_budget = 0.0 if steps > config.MAX_STEPS_PER_FRAME else self.step_budget - steps
            steps = min(steps, config.MAX_STEPS_PER_FRAME)
            active.advance(steps)

    # ----- input -----
    def handle(self, event):
        if event.type == pygame.QUIT:
            self.alive = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.slider is not None and self.slider.collidepoint(event.pos):
                self.dragging = True
                self.set_shown_from(event.pos[0])
                return
            for rect, action in self.buttons:
                if rect.collidepoint(event.pos):
                    action()
                    return
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_shown_from(event.pos[0])
        elif event.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
            self.resized()
        elif event.type == pygame.MOUSEWHEEL:
            if self.view.collidepoint(pygame.mouse.get_pos()):
                self.zoom_by(event.y)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                self.cycle_camera()
            elif event.key == pygame.K_F11 or (event.key == pygame.K_ESCAPE and self.fullscreen):
                self.toggle_fullscreen()
            elif event.key == pygame.K_SPACE and self.mode == "train":
                self.toggle_play()
            elif event.key == pygame.K_LEFT and self.mode == "race":
                self.change_race_track(-1)
            elif event.key == pygame.K_RIGHT and self.mode == "race":
                self.change_race_track(1)

    # ----- drawing -----
    def track_surface(self, track):
        key = (track.seed, track.level, track.width)
        if key not in self.surfaces:
            camera = Camera(track, self.view)
            self.surfaces[key] = (camera, render_track(track, camera, self.view.size))
        return self.surfaces[key]

    def draw(self):
        self.screen.fill(GRASS)
        self.buttons = []
        self.slider = None
        if self.mode == "train":
            track, playback = self.shown_track(), self.playback
        else:
            playback = self.race
            available = self.race_available or self.app_available
            track = playback.track if playback is not None else (self.race_track() if available else None)
        if track is not None:
            camera = self.view_camera(track, playback)
            self.screen.set_clip(self.view)                 # a zoomed track and its cars stay out of the panel
            if self.camera_zoom() == 1:
                self.screen.blit(self.track_surface(track)[1], self.view.topleft)
            else:
                draw_track_part(self.screen, track, camera)
            if playback is not None and self.mode == "train":
                self.draw_population(playback, camera)
            elif playback is not None:
                self.draw_racers(playback, camera)
            self.screen.set_clip(None)
        self.draw_panel()

    def draw_population(self, playback, camera):
        cars = playback.cars
        sx, sy = camera.to_screen(cars.x, cars.y)
        size = max(6.0, 4.5 * camera.scale)
        color = hex_to_rgb(config.COLORS[self.approach])
        overlay = pygame.Surface(self.view.size, pygame.SRCALPHA)
        for i in range(len(sx)):
            alpha = 90 if cars.done[i] else 170
            pygame.draw.polygon(overlay, color + (alpha,), car_polygon(sx[i], sy[i], cars.heading[i], size))
        self.screen.blit(overlay, self.view.topleft)
        b = self.best_slot
        if b < len(sx):
            pygame.draw.polygon(self.screen, INK, car_polygon(sx[b], sy[b], cars.heading[b], size * 1.3), 2)
            if self.rays and not cars.done[b]:
                lengths = sim.cast_rays(playback.track, cars.x[b:b + 1], cars.y[b:b + 1], cars.heading[b:b + 1],
                                        lengths=cars.ray_length[b:b + 1])[0]   # this car's own sensor length
                angles = cars.heading[b] + config.RAY_ANGLES
                ex, ey = camera.to_screen(cars.x[b] + np.cos(angles) * lengths, cars.y[b] + np.sin(angles) * lengths)
                for k in range(config.N_RAYS):
                    pygame.draw.line(self.screen, RAY, (sx[b], sy[b]), (ex[k], ey[k]), 1)

    def draw_racers(self, playback, camera):
        cars = playback.cars
        sx, sy = camera.to_screen(cars.x, cars.y)
        size = max(8.0, 6.0 * camera.scale)
        if self.rays:                                   # every racing car's rays, each at its own sensor length
            live = np.flatnonzero(~cars.done)
            if live.size:
                lengths = sim.cast_rays(playback.track, cars.x[live], cars.y[live], cars.heading[live],
                                        lengths=cars.ray_length[live])
                for row, i in enumerate(live):
                    angles = cars.heading[i] + config.RAY_ANGLES
                    ex, ey = camera.to_screen(cars.x[i] + np.cos(angles) * lengths[row],
                                              cars.y[i] + np.sin(angles) * lengths[row])
                    for k in range(config.N_RAYS):
                        pygame.draw.line(self.screen, RAY, (sx[i], sy[i]), (ex[k], ey[k]), 1)
        for i, car in enumerate(self.race_cars()):
            color = hex_to_rgb(config.COLORS[car[0]])
            if cars.crashed[i]:
                color = tuple(int(0.5 * c + 127) for c in color)          # faded
            shape = car_polygon(sx[i], sy[i], cars.heading[i], size)
            pygame.draw.polygon(self.screen, color, shape)
            if is_app(car):                                                # app cars: a white border
                pygame.draw.polygon(self.screen, WHITE, shape, 3)
            elif car[1] != SEEDED_RUN:                 # run 1 and up: a white stripe, rear to nose (spec §10.3)
                (nose_x, nose_y), (left_x, left_y), (right_x, right_y) = shape
                pygame.draw.line(self.screen, WHITE, ((left_x + right_x) / 2, (left_y + right_y) / 2),
                                 (nose_x, nose_y), 2)
            pygame.draw.polygon(self.screen, INK, shape, 1)

    def text(self, string, x, y, font=None, color=INK):
        surface = (font or self.font).render(string, True, color)
        self.screen.blit(surface, (x, y))
        return y + surface.get_height() + 4

    def swatch(self, approach, rect):
        """An approach's color square; a light color gets a dark outline on the light panel."""
        color = hex_to_rgb(config.COLORS[approach])
        pygame.draw.rect(self.screen, color, rect)
        if needs_outline(color, PANEL_BG):
            pygame.draw.rect(self.screen, INK, rect, 1)

    def button(self, rect, label, action, active=False, enabled=True, color=None):
        fill = hex_to_rgb(color) if (color and active) else (BUTTON_ACTIVE if active else BUTTON)
        if not enabled:                 # locked; a colored button that is on keeps a light tint of its color
            fill = (tuple(round(LOCKED_TINT * c + (1 - LOCKED_TINT) * d) for c, d in zip(fill, DISABLED))
                    if color and active else DISABLED)
        pygame.draw.rect(self.screen, fill, rect, border_radius=6)
        if color and not active:
            pygame.draw.rect(self.screen, hex_to_rgb(color), pygame.Rect(rect.left, rect.top, 5, rect.height),
                             border_top_left_radius=6, border_bottom_left_radius=6)
        ink = ink_for(fill) if active and enabled else (INK if enabled else MUTED)     # disabled: the light fill
        label_surface = self.small.render(label, True, ink)
        self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
        if enabled:
            self.buttons.append((rect, action))

    def draw_speed_buttons(self, x, y, w):
        """The playback speeds in one row, named on the left like the track and run seed rows. Returns the next
        free y."""
        self.text("Playback speed", x, y + 5, self.small, MUTED)
        quarter = (w - 110 - 24) // 4
        for k, speed in enumerate(config.SPEEDS):
            self.button(pygame.Rect(x + 110 + k * (quarter + 8), y, quarter, 28), f"{speed}×",
                        lambda s=speed: self.set_speed(s), self.speed == speed)
        return y + 36

    def draw_panel(self):
        pygame.draw.rect(self.screen, PANEL_BG, self.panel)
        x, y, w = self.panel.left + 16, 14, self.panel.width - 32
        third = (w - 16) // 3
        self.button(pygame.Rect(x, y, third, 30), "Training", lambda: self.set_mode("train"), self.mode == "train")
        self.button(pygame.Rect(x + third + 8, y, third, 30), "Race", lambda: self.set_mode("race"),
                    self.mode == "race")
        self.button(pygame.Rect(x + 2 * (third + 8), y, third, 30), "Windowed" if self.fullscreen else "Full screen",
                    self.toggle_fullscreen, self.fullscreen)
        y += 44
        if self.mode == "train":
            self.draw_train_panel(x, y, w)
        else:
            self.draw_race_panel(x, y, w)
        self.text(self.status[:56], x, self.panel.bottom - 26, self.small, MUTED)

    def draw_difficulty(self, x, y, w, level, action, enabled):
        """Easy / Medium / Hard / Extra hard buttons (spec §10.4). Returns the next free y."""
        y = self.text("Difficulty", x, y, self.small, MUTED)
        quarter = (w - 24) // 4
        for k, (value, name) in enumerate(config.DIFFICULTIES.items()):
            self.button(pygame.Rect(x + k * (quarter + 8), y, quarter, 26), name, lambda v=value: action(v),
                        level == value, enabled=enabled)
        return y + 34

    def draw_train_panel(self, x, y, w):
        idle = self.process is None
        third = (w - 16) // 3
        y = self.text("Approach", x, y, self.small, MUTED)
        for k, approach in enumerate(config.APPROACHES):
            rect = pygame.Rect(x + (k % 3) * (third + 8), y + (k // 3) * 34, third, 28)
            self.button(rect, config.LABELS[approach], lambda a=approach: self.select_approach(a),
                        approach == self.approach, enabled=idle, color=config.COLORS[approach])
        y += 3 * 34 + 4
        notebook = self.source == "notebook"
        half = (w - 8) // 2
        self.button(pygame.Rect(x, y, half, 26), f"Source: {'Notebook run' if notebook else 'App training'} ▸",
                    self.toggle_source, notebook, enabled=idle)
        self.button(pygame.Rect(x + half + 8, y, half, 26), self.folder_label(), self.cycle_folder,
                    self.folder_name != "current", enabled=idle and notebook)
        y += 34
        self.text("Run seed", x, y + 5, self.small, MUTED)
        self.button(pygame.Rect(x + 110, y, 30, 28), "<", lambda: self.change_run_seed(-1), enabled=idle)
        self.text(str(self.run_seed), x + 152, y + 5)
        self.button(pygame.Rect(x + 180, y, 30, 28), ">", lambda: self.change_run_seed(1), enabled=idle)
        y += 36
        y = self.draw_difficulty(x, y, w, self.difficulty, self.set_difficulty, idle and not notebook)
        if self.approach in self.population and not notebook:
            self.text("Population", x, y + 5, self.small, MUTED)
            self.button(pygame.Rect(x + 110, y, 30, 28), "−", lambda: self.change_population(-1), enabled=idle)
            self.text(str(self.population[self.approach]), x + 152, y + 5)
            self.button(pygame.Rect(x + 210, y, 30, 28), "+", lambda: self.change_population(1), enabled=idle)
            y += 36
        if notebook:
            playing = not self.replay_paused
        else:
            playing = self.process is not None and self.running is not None and self.running.is_set()
        self.button(pygame.Rect(x, y, third, 32), "Pause" if playing else "Play", self.toggle_play, playing)
        self.button(pygame.Rect(x + third + 8, y, third, 32), "Stop", self.stop, enabled=not idle and not notebook)
        self.button(pygame.Rect(x + 2 * (third + 8), y, third, 32), "New run", self.new_run,
                    enabled=idle and not notebook)
        y += 42
        y = self.text("Show", x, y, self.small, MUTED)
        for k, mode in enumerate(("all", "best", "worst")):
            self.button(pygame.Rect(x + k * (third + 8), y, third, 26), mode.capitalize(),
                        lambda m=mode: self.set_show(m), self.show == mode)
        y += 34
        size = self.population_size()
        shown = min(self.shown, size)
        y = self.text(f"Cars shown: {shown}", x, y, self.small, MUTED)
        self.slider = pygame.Rect(x, y + 4, w, 10)
        pygame.draw.rect(self.screen, BUTTON, self.slider, border_radius=5)
        knob = x + (shown - 1) / max(size - 1, 1) * w
        pygame.draw.circle(self.screen, BUTTON_ACTIVE, (int(knob), y + 9), 8)
        y += 24
        y = self.draw_speed_buttons(x, y, w)
        y = self.draw_view_toggles(x, y, w)
        y = self.draw_network_button(x, y, w)
        y = self.draw_stats(x, y)
        bottom = self.panel.bottom - 34
        if bottom - y > 50:
            self.draw_plot(pygame.Rect(x, y + 4, w, bottom - y - 4))

    def draw_view_toggles(self, x, y, w):
        """Sensor rays and the display grip side by side (spec §10.2, §10.5). Returns the next free y."""
        half = (w - 8) // 2
        self.button(pygame.Rect(x, y, half, 26), f"Sensor rays: {'on' if self.rays else 'off'}", self.toggle_rays,
                    self.rays)
        self.button(pygame.Rect(x + half + 8, y, half, 26), f"Grip: {self.grip_name}", self.toggle_grip,
                    self.grip_name != "Normal", enabled=not self.racing)
        return y + 34

    def draw_network_button(self, x, y, w):
        """The network window button and the camera button side by side. Returns the next free y."""
        half = (w - 8) // 2
        self.button(pygame.Rect(x, y, half, 26), "Network window", self.toggle_network_window,
                    self.net_process is not None)
        self.button(pygame.Rect(x + half + 8, y, half, 26), self.camera_label(), self.cycle_camera,
                    self.camera_zoom() != 1)
        return y + 34

    def draw_stats(self, x, y):
        y = self.text(self.map_text(), x, y, self.small)
        r = self.latest
        if r is None:
            if self.source == "app" and self.process is None and self.has_saved_run():
                return self.text("Saved run found: Play resumes, New run restarts.", x, y, self.small, MUTED)
            return self.text("No generation completed yet.", x, y, self.small, MUTED)
        for line in stats_lines(r, r["approach"]):
            y = self.text(line, x, y, self.small)
        return y

    def draw_plot(self, rect):
        pygame.draw.rect(self.screen, (255, 255, 255), rect)
        pygame.draw.rect(self.screen, BUTTON, rect, 1)
        shares = self.latest is not None and "min_share" in self.latest
        title = (f"Lowest map success share, {len(self.history)} generations" if shares
                 else f"Best and mean validation accuracy, {len(self.history)} reports")
        self.text(title, rect.left + 6, rect.top + 4, self.small, MUTED)
        if len(self.history) < 2:
            return
        gens = np.array([h[0] for h in self.history], dtype=float)
        main = np.array([h[1] for h in self.history], dtype=float)
        second = np.array([h[2] for h in self.history], dtype=float)
        if shares:
            lo, hi = 0.0, 1.0
        else:
            values = np.concatenate([main, second[~np.isnan(second)]])
            lo, hi = float(values.min()), float(values.max())
        span = hi - lo if hi > lo else 1.0
        width = max(gens[-1] - gens[0], 1.0)

        def point(g, v):
            return (rect.left + 6 + (g - gens[0]) / width * (rect.width - 12),
                    rect.bottom - 6 - (v - lo) / span * (rect.height - 30))

        for g, _, _, marked in self.history:
            if marked:
                px = point(g, lo)[0]
                pygame.draw.line(self.screen, MUTED, (px, rect.top + 22), (px, rect.bottom - 2), 1)
        if shares:                                          # the requirement of the latest generation
            y = point(gens[0], curriculum.mastery_fraction(self.latest.get("speed_push", False)))[1]
            pygame.draw.line(self.screen, BUTTON_ACTIVE, (rect.left + 6, y), (rect.right - 6, y), 1)
        if not np.isnan(second).all():
            pygame.draw.lines(self.screen, MUTED, False, [point(g, v) for g, v in zip(gens, second)], 1)
        color, points = hex_to_rgb(config.COLORS[self.approach]), [point(g, v) for g, v in zip(gens, main)]
        if needs_outline(color):                           # NEAT's yellow on the white plot
            pygame.draw.lines(self.screen, INK, False, points, 4)
        pygame.draw.lines(self.screen, color, False, points, 2)

    def draw_race_panel(self, x, y, w):
        idle = not self.racing
        half = (w - 8) // 2
        if not self.race_available and not self.app_available:     # another folder may have some
            self.button(pygame.Rect(x, y, half, 22), self.folder_label(), self.cycle_folder,
                        self.folder_name != "current", enabled=idle)
            self.text("No trained controllers found.", x, y + 30, self.small, MUTED)
            return
        self.text("Test track", x, y + 5, self.small, MUTED)
        self.button(pygame.Rect(x + 110, y, 30, 28), "<", lambda: self.change_race_track(-1), enabled=idle)
        self.text(f"seed {config.TEST_SEEDS[self.race_index]}", x + 150, y + 5)
        self.button(pygame.Rect(x + 240, y, 30, 28), ">", lambda: self.change_race_track(1), enabled=idle)
        y += 36
        y = self.draw_difficulty(x, y, w, self.race_level, self.set_race_level, idle)
        half = (w - 8) // 2
        self.button(pygame.Rect(x, y, half, 32), "Start race", self.start_race, self.racing,
                    enabled=idle and bool(self.race_cars()))
        self.button(pygame.Rect(x + half + 8, y, half, 32), "Stop", self.stop_race, enabled=self.racing)
        y += 42
        y = self.draw_view_toggles(x, y, w)
        y = self.draw_network_button(x, y, w)
        self.button(pygame.Rect(x, y - 6, w, 26), self.network_car_label(), self.cycle_network_car,
                    enabled=bool(self.race_cars()))
        y += 28
        self.button(pygame.Rect(x, y, half, 22), self.race_runs_label(), self.cycle_race_runs,
                    self.race_runs != "all", enabled=idle)
        self.button(pygame.Rect(x + half + 8, y, half, 22), self.folder_label(), self.cycle_folder,
                    self.folder_name != "current", enabled=idle)
        y += 26
        if self.app_available:
            self.button(pygame.Rect(x, y, half, 22), self.race_group_label(), self.cycle_race_group)
            y += 26
        if self.race_group == "app" and self.app_available:
            for approach in self.app_available:
                self.swatch(approach, pygame.Rect(x, y + 6, 10, 10))
                self.text(f"{config.LABELS[approach]} (app)", x + 16, y + 3, self.small)
                self.button(pygame.Rect(x + 140, y, w - 140, 22), self.app_choice_label(approach),
                            lambda a=approach: self.cycle_app_choice(a), enabled=idle)
                y += 24
        else:           # each method's switch, filled when on, in a grid of three: a race of 16 runs keeps every row
            third = (w - 16) // 3
            for k, approach in enumerate(self.race_available):
                self.button(pygame.Rect(x + (k % 3) * (third + 8), y + (k // 3) * 26, third, 22),
                            config.LABELS[approach], lambda a=approach: self.toggle_race(a),
                            self.race_on.get(approach, True), enabled=idle, color=config.COLORS[approach])
            y += 26 * -(-len(self.race_available) // 3)
        y += 4
        y = self.draw_speed_buttons(x, y, w)
        y = self.text(f"Leaderboard ({config.RACE_LAPS} laps): lap, last/best lap time (s)", x, y, self.small, MUTED)
        cars = self.race_cars()
        rows = leaderboard(self.race, len(cars))
        room = max((self.panel.bottom - 30 - y) // (self.small.get_linesize() + 4), 1)
        shown = rows if len(rows) <= room else rows[:room - 1]
        for position, i, laps, last, best, status in shown:
            car = cars[i]
            self.swatch(car[0], pygame.Rect(x, y + 3, 10, 10))
            y = self.text(leaderboard_line(position, car[0], car[1], laps, last, best, status, app=is_app(car)),
                          x + 16, y, self.small)
        if len(shown) < len(rows):
            self.text(f"… {len(rows) - len(shown)} more", x + 16, y, self.small, MUTED)


def main():
    pygame.init()
    pygame.display.set_caption("EC Racing")
    screen = pygame.display.set_mode(config.WINDOW_SIZE, pygame.RESIZABLE)
    viewer = App(screen)
    clock = pygame.time.Clock()
    try:
        while viewer.alive:
            dt = clock.tick(config.FPS) / 1000.0
            for event in pygame.event.get():
                viewer.handle(event)
            viewer.update(dt)
            viewer.draw()
            pygame.display.flip()
    finally:
        viewer.stop_training()
        viewer.close_network_window()
        pygame.quit()


if __name__ == "__main__":
    main()
