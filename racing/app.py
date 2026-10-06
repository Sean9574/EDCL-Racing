"""EC Racing viewer (spec §10): live training view and race view through one top-down camera.

Run with:  ~/miniconda3/bin/python -m racing.app
"""
import multiprocessing as mp
import os
import queue

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from racing import config, nets, sim
from racing.actions import class_to_controls
from racing.tracks import get_track

GRASS = (214, 230, 205)
ROAD = (92, 96, 104)
EDGE = (245, 245, 245)
PANEL_BG = (248, 249, 251)
INK = (29, 35, 48)
MUTED = (91, 100, 116)
BUTTON = (228, 232, 238)
BUTTON_ACTIVE = (0, 114, 178)
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


def mixed_policy(dense_nets):
    """One car per network; each network may have its own architecture."""
    def policy(obs, cars):
        return np.array([net.act(obs[i:i + 1])[0] for i, net in enumerate(dense_nets)], dtype=np.int64)
    return policy


def report_policy(report, indices):
    """Policy for the chosen members of a training report."""
    if "weights" in report:
        return nets.population_policy(report["weights"][indices], tuple(report["hidden"]), report["activation"])
    return mixed_policy([report["nets"][i] for i in indices])


def leaderboard(playback, names):
    """Rows (name, laps completed, last lap, best lap, status), ordered by distance covered."""
    if playback is None:
        return [(name, 0, float("nan"), float("nan"), "ready") for name in names]
    cars = playback.cars
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
        rows.append((names[i], int(cars.laps_done[i]), float(cars.last_lap[i]), float(cars.best_lap[i]), status))
    return rows


def trainer_main(approach, size, reports, running):
    """Training process: trains one approach and sends a report after every generation, trial or epoch.
    Writes to config.APP_RUNS_DIR and never saves champions."""
    from racing import es, ga, imitation
    from racing.expert import collect_dataset

    def send(report):
        running.wait()                      # training holds here while the viewer is paused
        reports.put(report)

    out_dir = config.APP_RUNS_DIR / approach
    if approach == "ga":
        ga.run_ga(pop_size=size, out_dir=out_dir, on_generation=send, champion_dir=None, n_jobs=config.N_JOBS)
        return
    if approach == "es":
        es.run_es(lam=size, out_dir=out_dir, on_generation=send, champion_dir=None, n_jobs=config.N_JOBS)
        return
    if not config.DATASET_PATH.exists():
        collect_dataset()
    data = imitation.load_dataset()
    members, scores = [], []

    def add(number, model, val_accuracy):
        members.append(nets.from_keras(model))
        scores.append(val_accuracy)
        fitness = np.array(scores)
        nan = float("nan")
        send({"approach": approach, "generation": int(number), "level": config.MAX_LEVEL,
              "seeds": [config.KT_VIEW_SEED], "nets": list(members), "fitness": fitness,
              "kept_finish": nan, "new_finish": nan, "step_size": nan, "best": float(fitness.max()),
              "mean": float(fitness.mean()), "worst": float(fitness.min()), "leveled_up": False})

    if approach == "baseline":
        imitation.train_baseline(data, out_dir=out_dir, champion_dir=None,
                                 on_epoch=lambda epoch, model, acc: add(epoch, model, acc))
    else:
        imitation.run_tuner(approach, data, out_dir=out_dir, champion_dir=None, on_trial=add)


def percent(value):
    return "–" if np.isnan(value) else f"{100 * value:.0f}%"


def number(value):
    return "–" if np.isnan(value) else f"{value:.4g}"


class App:
    """The viewer window: training view and race view."""

    def __init__(self, screen):
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
        self.track_cache = {}
        self.speed = 1
        self.step_budget = 0.0
        self.status = "Press Play to start training."
        # training view
        self.approach = "ga"
        self.population = {"ga": config.GA_POP, "es": config.ES_LAMBDA}
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
        self.race_seed = 0
        self.race_names = []
        self.race_nets = None
        self.race = None

    # ----- training process -----
    def start_training(self):
        context = mp.get_context("spawn")
        self.reports = context.Queue()
        self.running = context.Event()
        self.running.set()
        size = self.population.get(self.approach, 0)
        self.process = context.Process(target=trainer_main, args=(self.approach, size, self.reports, self.running),
                                       daemon=True)
        self.process.start()
        self.latest, self.history, self.playback, self.playback_generation = None, [], None, None
        self.status = f"Training {config.LABELS[self.approach]}."

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
        if self.process is not None and hasattr(self.process, "terminate") and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.process = None

    def poll_reports(self):
        if self.reports is not None:
            while True:
                try:
                    self.receive(self.reports.get_nowait())
                except queue.Empty:
                    break
        if self.process is not None and not self.process.is_alive():
            code = self.process.exitcode
            self.status = "Training finished." if code == 0 else f"Training stopped (exit code {code})."
            self.process = None

    def receive(self, report):
        self.latest = report
        self.history.append((report["generation"], report["best"], report["mean"], report["leveled_up"]))

    # ----- settings -----
    def select_approach(self, approach):
        if approach == self.approach:
            return
        self.stop_training()
        self.approach = approach
        self.latest, self.history, self.playback, self.playback_generation = None, [], None, None
        self.status = "Press Play to start training."

    def change_population(self, direction):
        presets = list(config.POPULATION_PRESETS)
        current = self.population[self.approach]
        position = presets.index(current) if current in presets else 0
        self.population[self.approach] = presets[min(max(position + direction, 0), len(presets) - 1)]

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

    def set_mode(self, mode):
        self.mode = mode
        self.step_budget = 0.0
        if mode == "race" and self.race_nets is None:
            self.load_race()

    # ----- playback -----
    def start_playback(self, report):
        fitness = np.asarray(report["fitness"], dtype=float)
        self.indices = select_indices(fitness, self.show, self.shown, sample_seed=report["generation"])
        track = get_track(report["seeds"][0], report["level"])
        self.playback = Playback(track, report_policy(report, self.indices), len(self.indices))
        self.best_slot = int(np.argmax(fitness[self.indices]))
        self.playback_generation = report["generation"]
        self.step_budget = 0.0

    def load_race(self):
        models = nets.load_champions(self.champions_dir)
        self.race_names = list(models)
        self.race_nets = [nets.from_keras(model) for model in models.values()]
        if not self.race_names:
            self.status = f"No trained controllers found in {self.champions_dir}."

    def start_race(self):
        if not self.race_names:
            return
        track = get_track(config.TEST_SEEDS[self.race_seed], config.MAX_LEVEL)
        self.race = Playback(track, mixed_policy(self.race_nets), len(self.race_nets), laps=config.RACE_LAPS)
        self.step_budget = 0.0

    def change_race_track(self, direction):
        self.race_seed = (self.race_seed + direction) % len(config.TEST_SEEDS)
        self.race = None

    def update(self, dt):
        if self.mode == "train":
            self.poll_reports()
            if self.latest is not None and (self.playback is None or self.playback.over):
                self.start_playback(self.latest)
            active = self.playback
        else:
            active = self.race
        if active is not None:
            self.step_budget += self.speed * dt / config.DT
            steps = int(self.step_budget)
            self.step_budget -= steps
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
        if key not in self.track_cache:
            camera = Camera(track, self.view)
            self.track_cache[key] = (camera, render_track(track, camera, self.view.size))
        return self.track_cache[key]

    def draw(self):
        self.screen.fill(GRASS)
        self.buttons = []
        self.slider = None
        playback = self.playback if self.mode == "train" else self.race
        if playback is not None:
            camera, surface = self.track_surface(playback.track)
            self.screen.blit(surface, self.view.topleft)
            if self.mode == "train":
                self.draw_population(playback, camera)
            else:
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
        for i, name in enumerate(self.race_names):
            color = hex_to_rgb(config.COLORS[name])
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
        return y + 40

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

    def draw_train_panel(self, x, y, w):
        third = (w - 16) // 3
        y = self.text("Approach", x, y, self.small, MUTED)
        for k, approach in enumerate(config.APPROACHES):
            rect = pygame.Rect(x + (k % 3) * (third + 8), y + (k // 3) * 36, third, 30)
            self.button(rect, config.LABELS[approach], lambda a=approach: self.select_approach(a),
                        approach == self.approach, color=config.COLORS[approach])
        y += 80
        if self.approach in self.population:
            y = self.text("Population (λ)" if self.approach == "es" else "Population", x, y, self.small, MUTED)
            idle = self.process is None
            self.button(pygame.Rect(x, y, 30, 28), "−", lambda: self.change_population(-1), enabled=idle)
            self.text(str(self.population[self.approach]), x + 44, y + 5)
            self.button(pygame.Rect(x + 110, y, 30, 28), "+", lambda: self.change_population(1), enabled=idle)
            y += 40
        playing = self.process is not None and self.running is not None and self.running.is_set()
        self.button(pygame.Rect(x, y, w, 34), "Pause" if playing else "Play", self.toggle_play, playing)
        y += 46
        y = self.text("Show", x, y, self.small, MUTED)
        for k, mode in enumerate(("all", "best", "worst")):
            self.button(pygame.Rect(x + k * (third + 8), y, third, 28), mode.capitalize(),
                        lambda m=mode: self.set_show(m), self.show == mode)
        y += 38
        size = self.population_size()
        shown = min(self.shown, size)
        y = self.text(f"Cars shown: {shown}", x, y, self.small, MUTED)
        self.slider = pygame.Rect(x, y + 4, w, 10)
        pygame.draw.rect(self.screen, BUTTON, self.slider, border_radius=5)
        knob = x + (shown - 1) / max(size - 1, 1) * w
        pygame.draw.circle(self.screen, BUTTON_ACTIVE, (int(knob), y + 9), 8)
        y += 26
        y = self.draw_speed_buttons(x, y, w)
        self.button(pygame.Rect(x, y, w, 28), f"Sensor rays: {'on' if self.rays else 'off'}", self.toggle_rays, self.rays)
        y += 40
        y = self.draw_stats(x, y)
        self.draw_plot(pygame.Rect(x, y + 6, w, 150))

    def draw_stats(self, x, y):
        r = self.latest
        if r is None:
            return self.text("No generation completed yet.", x, y, self.small, MUTED)
        unit = "Trial" if r["approach"].startswith("kt") else ("Epoch" if r["approach"] == "baseline" else "Generation")
        size_name = "σ" if r["approach"] == "es" else "Mutation strength"
        for line in (f"Level {r['level']}    {unit} {r['generation']}",
                     f"Finished: kept {percent(r['kept_finish'])}   new {percent(r['new_finish'])}",
                     f"{size_name}: {number(r['step_size'])}",
                     f"Best {r['best']:.3f}    Mean {r['mean']:.3f}"):
            y = self.text(line, x, y)
        return y

    def draw_plot(self, rect):
        pygame.draw.rect(self.screen, (255, 255, 255), rect)
        pygame.draw.rect(self.screen, BUTTON, rect, 1)
        self.text(f"Best and mean fitness, {len(self.history)} reports", rect.left + 6, rect.top + 4, self.small, MUTED)
        if len(self.history) < 2:
            return
        gens = np.array([h[0] for h in self.history], dtype=float)
        best = np.array([h[1] for h in self.history])
        mean = np.array([h[2] for h in self.history])
        lo, hi = min(best.min(), mean.min()), max(best.max(), mean.max())
        span = hi - lo if hi > lo else 1.0
        width = max(gens[-1] - gens[0], 1.0)

        def point(g, v):
            return (rect.left + 6 + (g - gens[0]) / width * (rect.width - 12),
                    rect.bottom - 6 - (v - lo) / span * (rect.height - 30))

        for g, _, _, leveled_up in self.history:
            if leveled_up:
                px = point(g, lo)[0]
                pygame.draw.line(self.screen, MUTED, (px, rect.top + 22), (px, rect.bottom - 2), 1)
        pygame.draw.lines(self.screen, MUTED, False, [point(g, v) for g, v in zip(gens, mean)], 1)
        pygame.draw.lines(self.screen, hex_to_rgb(config.COLORS[self.approach]), False,
                          [point(g, v) for g, v in zip(gens, best)], 2)

    def draw_race_panel(self, x, y, w):
        if not self.race_names:
            self.text("No trained controllers found.", x, y, self.small, MUTED)
            return
        y = self.text("Test track", x, y, self.small, MUTED)
        self.button(pygame.Rect(x, y, 30, 28), "<", lambda: self.change_race_track(-1))
        self.text(f"seed {config.TEST_SEEDS[self.race_seed]}", x + 44, y + 5)
        self.button(pygame.Rect(x + 140, y, 30, 28), ">", lambda: self.change_race_track(1))
        y += 40
        self.button(pygame.Rect(x, y, w, 34), "Start race", self.start_race)
        y += 46
        y = self.draw_speed_buttons(x, y, w)
        y = self.text(f"Leaderboard ({config.RACE_LAPS} laps)", x, y, self.small, MUTED)
        for name, laps, last, best, status in leaderboard(self.race, self.race_names):
            pygame.draw.rect(self.screen, hex_to_rgb(config.COLORS[name]), pygame.Rect(x, y + 3, 10, 10))
            y = self.text(f"{config.LABELS[name]:<13} lap {laps}  last {number(last)}  best {number(best)}  {status}",
                          x + 16, y, self.small)


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
