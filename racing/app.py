"""EC Racing viewer (spec §10): live training view and race view through one top-down camera.

Run with:  ~/miniconda3/bin/python -m racing.app
"""
import json
import multiprocessing as mp
import os
import queue
import signal
import threading
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from racing import config, nets, paths, sim
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
RAY = (255, 196, 0)


def hex_to_rgb(code):
    code = code.lstrip("#")
    return tuple(int(code[i:i + 2], 16) for i in (0, 2, 4))


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
    """Maps world coordinates (m) to pixels so that a track fills the view rectangle."""

    def __init__(self, track, rect):
        edges = np.vstack([track.left, track.right])
        lo, hi = edges.min(axis=0), edges.max(axis=0)
        self.scale = 0.94 * min(rect.width / (hi[0] - lo[0]), rect.height / (hi[1] - lo[1]))
        self.center = (lo + hi) / 2
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


def car_polygon(sx, sy, heading, size):
    """Triangle pointing along the heading. Screen y points down, so world rotations flip sign."""
    c, s = np.cos(heading), np.sin(heading)
    shape = [(size, 0.0), (-0.6 * size, 0.5 * size), (-0.6 * size, -0.5 * size)]
    return [(sx + px * c - py * s, sy - (px * s + py * c)) for px, py in shape]


class Playback:
    """Re-simulates a set of controllers on one track, a few steps at a time, for display."""

    def __init__(self, track, policy, n, laps=1, time_limit=None):
        self.track = track
        self.policy = policy
        self.cars = sim.start_cars(track, n, laps)
        self.time_limit = sim.time_limit_for(track, laps) if time_limit is None else time_limit

    @property
    def over(self):
        return bool(self.cars.done.all())

    def advance(self, steps):
        for _ in range(steps):
            if self.over:
                return
            steer, throttle = class_to_controls(self.policy(sim.sense(self.track, self.cars), self.cars))
            sim.step(self.track, self.cars, steer, throttle, self.time_limit)


def report_policy(report, indices):
    """Policy for the chosen members of a training report: an evolutionary population (dense or NEAT, revision 3
    §9) or the networks of an imitation report."""
    if "population" in report:
        return report["population"].subset(indices).policy()
    return nets.mixed_policy([report["nets"][i] for i in indices])


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


def trainer_main(approach, size, seed, level, reports, running):
    """Training process: trains one approach with run seed `seed`, starting at curriculum level `level`, and sends a
    report after every generation, trial or epoch. Every Play starts a new run: the approach's folder in
    config.APP_RUNS_DIR is removed first, since evolutionary runs resume from a checkpoint they find there. Writes
    only to config.APP_RUNS_DIR, never saves final controllers and trains until stopped (revision 3 §9)."""
    from racing import es, ga, imitation, neat
    from racing.expert import collect_dataset

    def send(report):
        reports.put(report)
        running.wait()                      # training holds here while the viewer is paused

    out_dir = config.APP_RUNS_DIR / approach / f"seed{seed}"
    paths.remove(out_dir)
    common = dict(seed=seed, pop_size=size, start_level=level, out_dir=out_dir, on_generation=send,
                  champion_dir=None, n_jobs=config.N_JOBS, max_generations=None)
    if approach == "ga":
        ga.run_ga(**common)
        return
    if approach in ("es", "es_seeded"):
        initial = es.dagger_baseline(seed) if approach == "es_seeded" else None
        es.run_es(**common, initial=initial, approach=approach)
        return
    if approach == "neat":
        neat.run_neat(**common)
        return
    path = config.DATASET_PATH
    if not path.exists():
        path = config.APP_RUNS_DIR / "dataset.npz"     # never write the reported dataset
        if not path.exists():
            collect_dataset(path)
    data = imitation.load_dataset(path)
    members, scores = [], []

    def add(number, model, val_accuracy):
        members.append(nets.from_keras(model))
        scores.append(val_accuracy)
        fitness = np.array(scores)
        send({"approach": approach, "generation": int(number), "level": level,
              "seeds": [config.KT_VIEW_SEED], "nets": list(members), "fitness": fitness,
              "best": float(fitness.max()), "mean": float(fitness.mean()), "worst": float(fitness.min()),
              "leveled_up": False})

    if approach == "baseline":                         # resume=False: always train live, never skip (revision 3 §9)
        imitation.train_baseline(data, out_dir=out_dir, seed=seed, resume=False,
                                 on_epoch=lambda epoch, model, acc: add(epoch, model, acc))
    else:
        imitation.run_tuner(approach, data, out_dir=out_dir, on_trial=add, seed=seed, export_models=False,
                            resume=False)


def watch_parent(parent, kill, interval=1.0):
    """Waits until the process that started the trainer is gone (the parent process changes), then ends the
    trainer's process group with kill(0, SIGTERM)."""
    while os.getppid() == parent:
        time.sleep(interval)
    kill(0, signal.SIGTERM)


def trainer_process(approach, size, seed, level, reports, running):
    """Entry point of the training process. It leads its own process group, so stopping it also stops the worker
    processes it starts (spec §10.2). A watcher thread ends the group if the viewer exits without stopping it."""
    os.setpgrp()
    threading.Thread(target=watch_parent, args=(os.getppid(), os.killpg), daemon=True).start()
    trainer_main(approach, size, seed, level, reports, running)


def leaderboard_line(position, approach, seed, laps, last, best, status):
    """One leaderboard row: position, method and seed, laps completed, last/best lap time (s), status."""
    times = "/".join("–" if np.isnan(value) else f"{value:.1f}" for value in (last, best))
    return f"{position}. {config.LABELS[approach]} s{seed}  lap {laps}  {times}  {status}"


def percent(value):
    return "–" if np.isnan(value) else f"{100 * value:.0f}%"


def number(value):
    return "–" if np.isnan(value) else f"{value:.4g}"


def share_text(shares):
    """Success shares of the generation's maps, e.g. '84% 92% 95%'."""
    return " ".join(percent(float(s)) for s in shares)


def stats_lines(report, approach):
    """Training panel stats lines (revision 3 §9). Evolutionary reports: difficulty and generation, each map's
    success share against the 90% requirement, the streak and the target cut mastered, the step size and the kept
    cars, the fitness (and NEAT's species). Imitation reports: the epoch or trial and the validation accuracy."""
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
    return (f"{level}    Generation {report['generation']}",
            f"Maps succeeded: {share_text(report['map_shares'])}  (need {percent(config.MASTERY_FRACTION)})",
            f"Streak {report['streak']}/{config.MASTERY_STREAK}    Target cut mastered: {mastered}",
            f"{size_name}: {number(report['step_size'])}    Kept {int(np.sum(report['is_kept']))}",
            last)


class App:
    """The viewer window: training view and race view."""

    def __init__(self, screen, background_tracks=True):
        self.screen = screen
        self.font = pygame.font.SysFont("dejavusans", 15)
        self.small = pygame.font.SysFont("dejavusans", 13)
        width, height = screen.get_size()
        self.view = pygame.Rect(0, 0, width - config.PANEL_WIDTH, height)
        self.panel = pygame.Rect(self.view.right, 0, config.PANEL_WIDTH, height)
        self.alive = True
        self.mode = "train"
        self.buttons = []
        self.slider = None
        self.dragging = False
        self.surfaces = {}
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
        self.runs_dir = config.RUNS_DIR     # where the seeded ES finds the DAgger-refined baseline
        self.show = "best"
        self.shown = config.DEFAULT_SHOWN
        self.rays = False
        self.process = None
        self.reports = None
        self.running = None
        self.latest = None
        self.history = []
        self.playback = None
        self.playback_generation = None
        self.indices = None
        self.best_slot = 0
        # race view
        self.champions_dir = config.CHAMPIONS_DIR
        self.race_index = 0                 # into config.TEST_SEEDS
        self.race_level = config.MAX_LEVEL
        self.race_available = {}            # approach -> run seeds with a saved controller
        self.race_choice = {}               # approach -> "best", a run seed, or "off"
        self.best = {}                      # approach -> best run seed recorded by the evaluation
        self.drivers = {}                   # (approach, seed) -> DenseNet
        self.lineup = []                    # (approach, seed) of the cars in the race
        self.race = None                    # Playback: the preview until the race starts
        self.racing = False
        self.failed = set()                 # (approach, seed) whose controller file could not be loaded

    # ----- tracks -----
    def request_track(self, seed, level):
        """The track if it is built; otherwise start building it and return None (spec §10.2). Without background
        building, the track is built at once."""
        key = (int(seed), int(level))
        if key in self.tracks:
            return self.tracks[key]
        if not self.background_tracks:
            self.build_track(key)
            return self.tracks.get(key)
        if key not in self.building:
            self.building.add(key)
            threading.Thread(target=self.build_track, args=(key,), daemon=True).start()
        return None

    def build_track(self, key):
        """Build one track; a failure is shown in the status line and the next request tries again."""
        try:
            self.tracks[key] = get_track(*key)
        except Exception as error:
            self.status = f"Track seed {key[0]}, level {key[1]} could not be built: {error}"
        finally:
            self.building.discard(key)

    # ----- training process -----
    def start_training(self):
        if self.approach == "es_seeded" and not paths.stage_path("baseline", self.run_seed, "dagger",
                                                                 self.runs_dir).exists():
            self.status = "Seeded ES needs the DAgger baseline: run notebook 03."
            return
        context = mp.get_context("spawn")
        self.reports = context.Queue()
        self.running = context.Event()
        self.running.set()
        size = self.population.get(self.approach, 0)
        self.process = context.Process(target=trainer_process, daemon=False,
                                       args=(self.approach, size, self.run_seed, self.difficulty, self.reports,
                                             self.running))
        self.process.start()
        self.latest, self.history, self.playback, self.playback_generation = None, [], None, None
        self.status = f"Training {config.LABELS[self.approach]}, seed {self.run_seed}."

    def toggle_play(self):
        if self.process is None:
            self.start_training()
        elif self.running.is_set():
            self.running.clear()
            self.status = "Paused at the next generation boundary."
        else:
            self.running.set()
            self.status = f"Training {config.LABELS[self.approach]}."

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

    def stop(self):
        """Stop control: ends training in the training view and the race in the race view."""
        if self.mode == "race":
            self.stop_race()
            return
        self.stop_training()
        self.status = "Stopped. Change the settings or press Play."

    def discard_reports(self):
        """Drop the report queue so reports from a stopped trainer never reach a new view."""
        if self.reports is not None and hasattr(self.reports, "cancel_join_thread"):
            self.reports.cancel_join_thread()
            self.reports.close()
        self.reports = None

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
            self.status = "Training finished." if code == 0 else f"Training stopped (exit code {code})."
            self.process = None
            self.discard_reports()

    def receive(self, report):
        """Show a report; the plot follows the lowest map success share (evolution, with level advancements, the
        mastery goal and target cuts marked) or the validation accuracy (imitation)."""
        self.latest = report
        if "min_share" in report:
            marked = report.get("event") in ("level_up", "goal", "tightened")
            self.history.append((report["generation"], float(report["min_share"]), float("nan"), marked))
        else:
            self.history.append((report["generation"], report["best"], report["mean"], report["leveled_up"]))

    # ----- settings (changeable only while no training runs) -----
    def select_approach(self, approach):
        if self.process is not None or approach == self.approach:
            return
        self.stop_training()
        self.approach = approach
        self.latest, self.history, self.playback, self.playback_generation = None, [], None, None
        self.status = "Press Play to start training."

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

    def set_difficulty(self, level):
        if self.process is not None:
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

    def toggle_rays(self):
        self.rays = not self.rays

    def set_shown_from(self, x):
        fraction = min(max((x - self.slider.left) / self.slider.width, 0.0), 1.0)
        self.shown = 1 + int(round(fraction * (self.population_size() - 1)))
        self.playback = None

    # ----- playback -----
    def start_playback(self, report, track):
        fitness = np.asarray(report["fitness"], dtype=float)
        self.indices = select_indices(fitness, self.show, self.shown, sample_seed=report["generation"])
        self.playback = Playback(track, report_policy(report, self.indices), len(self.indices),
                                 laps=config.LAPS)
        self.best_slot = int(np.argmax(fitness[self.indices]))
        self.playback_generation = report["generation"]
        self.step_budget = 0.0

    def shown_track(self):
        """The track on screen in the training view: the playback's, or for imitation approaches the map they are
        driven on, which is known before training starts."""
        if self.playback is not None:
            return self.playback.track
        if self.approach not in config.EVOLUTION_APPROACHES:
            return self.request_track(config.KT_VIEW_SEED, self.difficulty)
        return None

    def map_text(self):
        track = self.shown_track()
        if track is not None:
            return f"Map: seed {track.seed}, level {track.level}"
        if self.approach in config.EVOLUTION_APPROACHES and self.latest is None:
            return "Map: shown with the first generation"
        return "Map: building the track"

    def set_mode(self, mode):
        entering = mode != self.mode
        self.mode = mode
        self.step_budget = 0.0
        if mode == "race" and (entering or not self.race_available):
            self.load_race()
        elif mode == "train" and self.status.startswith("No trained controllers"):
            self.status = "Training view."

    # ----- race view -----
    def load_race(self):
        """Find the saved final controllers and each approach's best run seed (spec §10.3)."""
        self.race_available = nets.champion_seeds(self.champions_dir)
        self.drivers = {}
        self.failed = set()
        self.best = {}
        note = ""
        best_path = paths.best_seeds_path(self.champions_dir)
        if best_path.exists():
            try:
                for approach, seed in json.loads(best_path.read_text()).items():
                    if isinstance(seed, int) and not isinstance(seed, bool):
                        self.best[approach] = seed
            except (ValueError, TypeError, AttributeError, OSError):
                self.best = {}
                note = f"{best_path.name} could not be read; using the lowest seeds."
        kept = self.race_choice
        self.race_choice = {}
        for approach, seeds in self.race_available.items():
            choice = kept.get(approach, "best")
            self.race_choice[approach] = choice if choice in ("best", "off", *seeds) else "best"
        if not self.race_available:
            self.status = f"No trained controllers found in {self.champions_dir}."
        else:
            self.status = note or "Race view."
        self.reset_race()

    def chosen_seed(self, approach):
        """The run seed whose controller races for `approach`, or None if the approach is off. Without an
        evaluation record, 'best' falls back to the lowest saved seed."""
        choice = self.race_choice.get(approach, "best")
        if choice == "off":
            return None
        seeds = self.race_available[approach]
        if choice == "best":
            best = self.best.get(approach)
            return best if best in seeds else seeds[0]
        return choice

    def choice_label(self, approach):
        choice = self.race_choice.get(approach, "best")
        if choice == "best":
            return f"best (seed {self.chosen_seed(approach)})"
        if choice == "off":
            return "off"
        return f"seed {choice}"

    def cycle_choice(self, approach):
        if self.racing:
            return
        options = ["best", *self.race_available[approach], "off"]
        current = self.race_choice.get(approach, "best")
        self.race_choice[approach] = options[(options.index(current) + 1) % len(options)]
        self.reset_race()

    def reset_race(self):
        """Back to the map preview: the selected controllers wait on the start line."""
        self.racing = False
        self.race = None
        self.step_budget = 0.0
        self.lineup = []
        for approach in self.race_available:
            seed = self.chosen_seed(approach)
            if seed is not None and (approach, seed) not in self.failed:
                self.lineup.append((approach, seed))

    def driver(self, approach, seed):
        if (approach, seed) not in self.drivers:
            self.drivers[(approach, seed)] = nets.load_champion(approach, seed, self.champions_dir)[1]
        return self.drivers[(approach, seed)]

    def race_track(self):
        return self.request_track(config.TEST_SEEDS[self.race_index], self.race_level)

    def prepare_race(self):
        """Build the preview once the track is ready: every selected car on the start line."""
        if self.race is None and self.lineup:
            track = self.race_track()
            if track is not None:
                drivers = []
                for approach, seed in list(self.lineup):
                    try:
                        drivers.append(self.driver(approach, seed))
                    except Exception:
                        self.failed.add((approach, seed))
                        self.lineup.remove((approach, seed))
                        self.status = f"Controller {config.LABELS[approach]} seed {seed} could not be loaded; skipped."
                if drivers:
                    self.race = Playback(track, nets.mixed_policy(drivers), len(drivers), laps=config.RACE_LAPS)

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
        if self.mode == "train":
            if self.latest is not None and (self.playback is None or self.playback.over):
                track = self.request_track(self.latest["seeds"][0], self.latest["level"])
                if track is not None:
                    self.start_playback(self.latest, track)
            active = self.playback
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
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and self.mode == "train":
                self.toggle_play()
            elif event.key == pygame.K_LEFT and self.mode == "race":
                self.change_race_track(-1)
            elif event.key == pygame.K_RIGHT and self.mode == "race":
                self.change_race_track(1)

    # ----- drawing -----
    def track_surface(self, track):
        key = (track.seed, track.level)
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
            track = playback.track if playback is not None else (self.race_track() if self.race_available else None)
        if track is not None:
            camera, surface = self.track_surface(track)
            self.screen.blit(surface, self.view.topleft)
            if playback is not None and self.mode == "train":
                self.draw_population(playback, camera)
            elif playback is not None:
                self.draw_racers(playback, camera)
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
                lengths = sim.cast_rays(playback.track, cars.x[b:b + 1], cars.y[b:b + 1], cars.heading[b:b + 1])[0]
                angles = cars.heading[b] + config.RAY_ANGLES
                ex, ey = camera.to_screen(cars.x[b] + np.cos(angles) * lengths, cars.y[b] + np.sin(angles) * lengths)
                for k in range(config.N_RAYS):
                    pygame.draw.line(self.screen, RAY, (sx[b], sy[b]), (ex[k], ey[k]), 1)

    def draw_racers(self, playback, camera):
        cars = playback.cars
        sx, sy = camera.to_screen(cars.x, cars.y)
        size = max(8.0, 6.0 * camera.scale)
        for i, (approach, _) in enumerate(self.lineup):
            color = hex_to_rgb(config.COLORS[approach])
            if cars.crashed[i]:
                color = tuple(int(0.5 * c + 127) for c in color)          # faded
            shape = car_polygon(sx[i], sy[i], cars.heading[i], size)
            pygame.draw.polygon(self.screen, color, shape)
            pygame.draw.polygon(self.screen, INK, shape, 1)

    def text(self, string, x, y, font=None, color=INK):
        surface = (font or self.font).render(string, True, color)
        self.screen.blit(surface, (x, y))
        return y + surface.get_height() + 4

    def button(self, rect, label, action, active=False, enabled=True, color=None):
        fill = hex_to_rgb(color) if (color and active) else (BUTTON_ACTIVE if active else BUTTON)
        pygame.draw.rect(self.screen, fill if enabled else (238, 240, 243), rect, border_radius=6)
        if color and not active:
            pygame.draw.rect(self.screen, hex_to_rgb(color), pygame.Rect(rect.left, rect.top, 5, rect.height),
                             border_top_left_radius=6, border_bottom_left_radius=6)
        ink = (255, 255, 255) if active else (INK if enabled else MUTED)
        label_surface = self.small.render(label, True, ink)
        self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
        if enabled:
            self.buttons.append((rect, action))

    def draw_speed_buttons(self, x, y, w):
        y = self.text("Playback speed", x, y, self.small, MUTED)
        quarter = (w - 24) // 4
        for k, speed in enumerate(config.SPEEDS):
            self.button(pygame.Rect(x + k * (quarter + 8), y, quarter, 28), f"{speed}×",
                        lambda s=speed: self.set_speed(s), self.speed == speed)
        return y + 36

    def draw_panel(self):
        pygame.draw.rect(self.screen, PANEL_BG, self.panel)
        x, y, w = self.panel.left + 16, 14, self.panel.width - 32
        half = (w - 8) // 2
        self.button(pygame.Rect(x, y, half, 30), "Training", lambda: self.set_mode("train"), self.mode == "train")
        self.button(pygame.Rect(x + half + 8, y, half, 30), "Race", lambda: self.set_mode("race"), self.mode == "race")
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
        self.text("Run seed", x, y + 5, self.small, MUTED)
        self.button(pygame.Rect(x + 110, y, 30, 28), "<", lambda: self.change_run_seed(-1), enabled=idle)
        self.text(str(self.run_seed), x + 152, y + 5)
        self.button(pygame.Rect(x + 180, y, 30, 28), ">", lambda: self.change_run_seed(1), enabled=idle)
        y += 36
        y = self.draw_difficulty(x, y, w, self.difficulty, self.set_difficulty, idle)
        if self.approach in self.population:
            self.text("Population", x, y + 5, self.small, MUTED)
            self.button(pygame.Rect(x + 110, y, 30, 28), "−", lambda: self.change_population(-1), enabled=idle)
            self.text(str(self.population[self.approach]), x + 152, y + 5)
            self.button(pygame.Rect(x + 210, y, 30, 28), "+", lambda: self.change_population(1), enabled=idle)
            y += 36
        playing = self.process is not None and self.running is not None and self.running.is_set()
        half = (w - 8) // 2
        self.button(pygame.Rect(x, y, half, 32), "Pause" if playing else "Play", self.toggle_play, playing)
        self.button(pygame.Rect(x + half + 8, y, half, 32), "Stop", self.stop, enabled=not idle)
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
        self.button(pygame.Rect(x, y, w, 26), f"Sensor rays: {'on' if self.rays else 'off'}", self.toggle_rays,
                    self.rays)
        y += 34
        y = self.draw_stats(x, y)
        bottom = self.panel.bottom - 34
        if bottom - y > 50:
            self.draw_plot(pygame.Rect(x, y + 4, w, bottom - y - 4))

    def draw_stats(self, x, y):
        y = self.text(self.map_text(), x, y, self.small)
        r = self.latest
        if r is None:
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
        if shares:                                          # the 90% requirement
            y = point(gens[0], config.MASTERY_FRACTION)[1]
            pygame.draw.line(self.screen, BUTTON_ACTIVE, (rect.left + 6, y), (rect.right - 6, y), 1)
        if not np.isnan(second).all():
            pygame.draw.lines(self.screen, MUTED, False, [point(g, v) for g, v in zip(gens, second)], 1)
        pygame.draw.lines(self.screen, hex_to_rgb(config.COLORS[self.approach]), False,
                          [point(g, v) for g, v in zip(gens, main)], 2)

    def draw_race_panel(self, x, y, w):
        if not self.race_available:
            self.text("No trained controllers found.", x, y, self.small, MUTED)
            return
        idle = not self.racing
        self.text("Test track", x, y + 5, self.small, MUTED)
        self.button(pygame.Rect(x + 110, y, 30, 28), "<", lambda: self.change_race_track(-1), enabled=idle)
        self.text(f"seed {config.TEST_SEEDS[self.race_index]}", x + 150, y + 5)
        self.button(pygame.Rect(x + 240, y, 30, 28), ">", lambda: self.change_race_track(1), enabled=idle)
        y += 36
        y = self.draw_difficulty(x, y, w, self.race_level, self.set_race_level, idle)
        half = (w - 8) // 2
        self.button(pygame.Rect(x, y, half, 32), "Start race", self.start_race, self.racing,
                    enabled=idle and bool(self.lineup))
        self.button(pygame.Rect(x + half + 8, y, half, 32), "Stop", self.stop_race, enabled=self.racing)
        y += 42
        y = self.text("Controllers", x, y, self.small, MUTED)
        for approach in self.race_available:
            pygame.draw.rect(self.screen, hex_to_rgb(config.COLORS[approach]), pygame.Rect(x, y + 8, 10, 10))
            self.text(config.LABELS[approach], x + 16, y + 5, self.small)
            self.button(pygame.Rect(x + 140, y, w - 140, 24), self.choice_label(approach),
                        lambda a=approach: self.cycle_choice(a), enabled=idle)
            y += 28
        y += 4
        y = self.draw_speed_buttons(x, y, w)
        y = self.text(f"Leaderboard ({config.RACE_LAPS} laps): lap, last/best lap time (s)", x, y, self.small, MUTED)
        for position, i, laps, last, best, status in leaderboard(self.race, len(self.lineup)):
            approach, seed = self.lineup[i]
            pygame.draw.rect(self.screen, hex_to_rgb(config.COLORS[approach]), pygame.Rect(x, y + 3, 10, 10))
            y = self.text(leaderboard_line(position, approach, seed, laps, last, best, status), x + 16, y,
                          self.small)


def main():
    pygame.init()
    pygame.display.set_caption("EC Racing")
    screen = pygame.display.set_mode(config.WINDOW_SIZE)
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
        pygame.quit()


if __name__ == "__main__":
    main()
