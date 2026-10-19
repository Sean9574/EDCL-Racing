import json
import os
import queue
import signal
import threading
import time

import numpy as np
import pygame
import pytest

from racing import app as viewer
from racing import checkpoint, config, neat, nets, paths, population, setup, sim


@pytest.fixture
def window(monkeypatch, circle_track):
    monkeypatch.setattr(viewer, "get_track", lambda seed, level: circle_track)
    monkeypatch.setattr(viewer.os, "killpg", lambda pgid, sig: None)     # tests never signal real processes
    pygame.init()
    screen = pygame.display.set_mode(config.WINDOW_SIZE)
    instance = viewer.App(screen, background_tracks=False)
    yield instance
    instance.process = None
    instance.stop_training()
    pygame.quit()


def ga_report(generation=0, n=30, approach="ga", population_=None, **extra):
    """A training report as the evolution engine sends it (spec §10.2)."""
    rng = np.random.default_rng(generation)
    fitness = rng.random(n)
    if population_ is None:
        population_ = nets.DensePopulation(nets.init_population(rng, n))
    shares = np.array([0.84, 0.92, 0.95])
    report = {"approach": approach, "seed": 0, "generation": generation, "level": 1,
              "factor": config.TARGET_FACTOR, "target_reduction": float("nan"), "streak": 1, "seeds": [0, 1, 2],
              "population": population_, "fitness": fitness, "all_success": np.zeros(n, dtype=bool),
              "is_kept": np.zeros(n, dtype=bool), "map_shares": shares, "min_share": float(shares.min()),
              "kept_success": float("nan"), "child_success": 0.0, "median_lap": float("nan"), "step_size": 0.5,
              "best": float(fitness.max()), "mean": float(fitness.mean()), "worst": float(fitness.min()),
              "evals": n, "elapsed": 1.0, "event": None, "sensor_mean": 50.0, "sensor_std": 0.0, "steer_mean": 1.0,
              "steer_std": 0.0}
    report.update(extra)
    return report


def neat_report(generation=0, n=20):
    rng, innovations = np.random.default_rng(generation), neat.Innovations()
    genomes = [neat.initial_genome(rng, innovations) for _ in range(n)]
    return ga_report(generation, n, approach="neat", population_=neat.NeatPopulation(genomes), species=3)


def test_button_text_is_dark_on_light_colors_and_white_on_dark_ones():
    """NEAT's yellow is too light for white text (spec §10.1)."""
    rgb = viewer.hex_to_rgb
    assert viewer.ink_for(rgb(config.COLORS["neat"])) == viewer.INK
    assert viewer.ink_for(rgb(config.COLORS["es_seeded"])) == viewer.WHITE         # dark blue
    assert viewer.ink_for(viewer.BUTTON_ACTIVE) == viewer.WHITE
    assert viewer.ink_for(viewer.WHITE) == viewer.INK
    for color in config.COLORS.values():                                       # the better of the two inks
        fill = rgb(color)
        assert viewer.contrast(fill, viewer.ink_for(fill)) == max(viewer.contrast(fill, viewer.INK),
                                                                   viewer.contrast(fill, viewer.WHITE))


def test_light_plot_lines_get_a_dark_outline():
    rgb = viewer.hex_to_rgb
    assert viewer.needs_outline(rgb(config.COLORS["neat"]))
    assert not viewer.needs_outline(rgb(config.COLORS["es_seeded"])) and not viewer.needs_outline(viewer.INK)


def test_the_selected_neat_button_has_dark_text(window):
    rect = pygame.Rect(10, 10, 160, 28)
    window.screen.fill((0, 0, 0))
    window.button(rect, "NEAT", lambda: None, active=True, color=config.COLORS["neat"])
    pixels = {tuple(window.screen.get_at((x, y)))[:3] for x in range(rect.left, rect.right)
              for y in range(rect.top, rect.bottom)}
    assert viewer.INK in pixels and (255, 255, 255) not in pixels


def test_a_disabled_selected_button_keeps_legible_text(window):
    """A method switched on is drawn disabled during a race: muted text, never white."""
    rect = pygame.Rect(10, 10, 160, 28)
    window.screen.fill(viewer.PANEL_BG)
    window.button(rect, "Baseline", lambda: None, active=True, enabled=False, color=config.COLORS["baseline"])
    pixels = {tuple(window.screen.get_at((x, y)))[:3] for x in range(rect.left, rect.right)
              for y in range(rect.top, rect.bottom)}
    darkest = min(pixels, key=viewer.luminance)                             # the text's ink
    assert (255, 255, 255) not in pixels and viewer.contrast(darkest, viewer.MUTED) < 1.2


def test_a_locked_method_switch_still_shows_whether_it_is_on(window):
    """During a race the switches are locked: one switched on keeps a light tint of its color, one switched off the
    plain disabled fill, and the muted text stays legible on both."""
    for color in config.COLORS.values():
        fills = []
        for active in (True, False):
            rect = pygame.Rect(10, 10, 110, 22)
            window.screen.fill(viewer.PANEL_BG)
            window.button(rect, "", lambda: None, active=active, enabled=False, color=color)
            fills.append(tuple(window.screen.get_at((rect.left + 12, rect.top + 4)))[:3])
        on, off = fills
        assert off == viewer.DISABLED and on not in (viewer.DISABLED, viewer.hex_to_rgb(color)), color
        assert viewer.contrast(on, viewer.MUTED) >= 3.0, color


def test_select_indices_modes():
    fitness = np.array([0.3, 0.9, 0.1, 0.5])
    assert list(viewer.select_indices(fitness, "best", 2)) == [1, 3]
    assert list(viewer.select_indices(fitness, "worst", 2)) == [2, 0]
    sample = viewer.select_indices(fitness, "all", 3, sample_seed=7)
    assert len(set(sample)) == 3 and np.array_equal(sample, viewer.select_indices(fitness, "all", 3, sample_seed=7))


def test_select_indices_clamps_the_count():
    fitness = np.arange(5.0)
    assert len(viewer.select_indices(fitness, "best", 0)) == 1
    assert len(viewer.select_indices(fitness, "all", 1000)) == 5


def test_camera_keeps_the_track_inside_the_view(circle_track):
    rect = pygame.Rect(0, 0, 900, 800)
    camera = viewer.Camera(circle_track, rect)
    for edge in (circle_track.left, circle_track.right):
        sx, sy = camera.to_screen(edge[:, 0], edge[:, 1])
        assert sx.min() >= 0 and sx.max() <= 900 and sy.min() >= 0 and sy.max() <= 800


def test_training_playback_drives_all_laps(window):
    window.receive(ga_report())
    window.update(1 / 60)
    assert window.playback.cars.laps == config.LAPS


def test_training_stats_fit_the_panel(window):
    window.receive(ga_report())
    window.update(1 / 60)
    widest = dict(level=4, generation=123456, map_shares=np.ones(3), min_share=1.0, streak=3, target_reduction=0.185,
                  step_size=0.0505, best=1.5843, mean=1.2329, species=20, is_kept=np.ones(1000, dtype=bool),
                  sensor_mean=150.0, sensor_std=70.0, steer_mean=3.0, steer_std=1.33)
    for approach in config.EVOLUTION_APPROACHES:
        report = ga_report(approach=approach, **widest)
        for line in viewer.stats_lines(report, approach):
            assert window.small.size(line)[0] <= window.panel.width - 32, line
    imitation = {"generation": 149, "best": 0.7046, "mean": 0.6123, "level": 4}
    for approach in config.IMITATION_APPROACHES:
        for line in viewer.stats_lines(imitation, approach):
            assert window.small.size(line)[0] <= window.panel.width - 32, line


def test_stats_lines_show_maps_streak_and_target():
    lines = viewer.stats_lines(ga_report(level=4, generation=412, streak=1, target_reduction=0.143,
                                         map_shares=np.array([0.84, 0.92, 0.95])), "ga")
    assert lines[0] == "Extra hard    Generation 412"
    assert lines[1] == "Maps succeeded: 84% 92% 95%  (need 90%)"
    assert lines[2] == "Streak 1/3    Target cut mastered: −14.3%"
    assert lines[3].startswith("Mutation strength: 0.5")
    assert viewer.stats_lines(ga_report(), "es")[2].endswith("mastered: –")
    assert viewer.stats_lines(neat_report(), "neat")[4].endswith("Species 3")


def test_stats_lines_show_the_population_setup():
    lines = viewer.stats_lines(ga_report(sensor_mean=62.4, sensor_std=18.2, steer_mean=1.4, steer_std=0.3), "ga")
    assert lines[5] == "Sensors 62 m ± 18  Steering ×1.40 ± 0.30"
    report = ga_report()
    for key in ("sensor_mean", "sensor_std", "steer_mean", "steer_std"):
        del report[key]
    assert len(viewer.stats_lines(report, "ga")) == 5        # a report without setups shows no setup line


def test_playback_drives_each_car_with_its_setup_and_the_display_grip(circle_track):
    full_left_gas = lambda obs, cars: np.full(len(cars.x), config.N_ACTIONS - 1)
    cars_setup = np.array([[30.0, 1.0], [150.0, 0.5]])
    normal = viewer.Playback(circle_track, full_left_gas, 2, setup=cars_setup)
    unlimited = viewer.Playback(circle_track, full_left_gas, 2, setup=cars_setup, grip=config.APP_GRIPS["Unlimited"])
    assert normal.grip == config.GRIP and unlimited.grip == config.APP_GRIPS["Unlimited"]
    assert np.array_equal(normal.cars.ray_length, [30.0, 150.0]) and np.array_equal(normal.cars.steer_gain, [1.0, 0.5])
    normal.advance(5)
    unlimited.advance(5)
    assert np.all(unlimited.cars.speed > 5 * normal.cars.speed)    # ten times the grip accelerates far harder


def test_the_training_replay_uses_the_report_setups_and_the_display_grip(window):
    rng = np.random.default_rng(4)
    cars = nets.DensePopulation(nets.init_population(rng, 30), setup=setup.random(rng, 30))
    window.set_grip("Unlimited")
    window.receive(ga_report(population_=cars))
    window.update(1 / 60)
    assert window.playback.grip == config.APP_GRIPS["Unlimited"]
    assert np.array_equal(window.playback.cars.ray_length, cars.subset(window.indices).setup[:, 0])
    assert np.array_equal(window.playback.cars.steer_gain, cars.subset(window.indices).setup[:, 1])
    window.set_grip("Normal")                                    # the replay restarts with the new grip
    window.update(1 / 60)
    assert window.playback.grip == config.GRIP


def test_a_stopped_replay_keeps_its_grip(window):
    window.receive(ga_report())
    window.update(1 / 60)
    window.stop()
    frozen = window.playback
    window.set_grip("Unlimited")
    window.update(1 / 60)
    assert window.playback is frozen and frozen.grip == config.GRIP and window.grip_name == "Unlimited"


def test_the_race_uses_driver_setups_and_the_display_grip(window, tmp_path):
    race_window(window, tmp_path)
    window.set_grip("Unlimited")
    window.update(1 / 60)
    drivers = [window.driver(approach, seed) for approach, seed in window.lineup]
    assert window.race.grip == config.APP_GRIPS["Unlimited"]
    assert np.array_equal(window.race.cars.ray_length, nets.drivers_setup(drivers)[:, 0])
    window.start_race()
    window.set_grip("Normal")                                    # locked while racing
    assert window.grip_name == "Unlimited" and window.race.grip == config.APP_GRIPS["Unlimited"]


def test_toggle_grip_cycles_the_display_grips(window):
    names = list(config.APP_GRIPS)
    assert window.grip_name == names[0] == "Normal" and window.grip == config.GRIP
    window.toggle_grip()
    assert window.grip_name == names[1] and window.grip == config.APP_GRIPS[names[1]]
    window.toggle_grip()
    assert window.grip_name == names[0]


def test_the_display_grip_never_reaches_training(window, monkeypatch, tmp_path):
    """Spec §10.5: the trainer process gets the same arguments whatever the display grip, the training functions
    get no grip, and the training grip itself never changes."""
    from racing import ga
    created = []

    class FakeProcess:
        def __init__(self, target, args, daemon):
            created.append(args)
            self.pid = 0

        def start(self):
            pass

        def is_alive(self):
            return False

    class FakeContext:
        Process = FakeProcess

        def Queue(self):
            return FakeQueue()

        def Event(self):
            return threading.Event()

    monkeypatch.setattr(viewer.mp, "get_context", lambda method: FakeContext())
    window.start_training()
    window.process = None
    window.set_grip("Unlimited")
    window.start_training()
    assert len(created) == 2 and created[0][:4] == created[1][:4] and len(created[1]) == len(created[0])
    assert not any(isinstance(a, float) and a == config.APP_GRIPS["Unlimited"] for a in created[1])
    calls = {}
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)       # never the real results/

    def fake_run_ga(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(ga, "run_ga", fake_run_ga)
    reports, running = queue.Queue(), threading.Event()
    running.set()
    viewer.trainer_main("ga", 30, 0, 1, reports, running)
    assert "grip" not in calls and calls["pop_size"] == 30
    assert config.GRIP == 12.0 and config.APP_GRIPS["Normal"] == config.GRIP


def test_rays_follow_each_cars_sensor_length(window, monkeypatch, tmp_path):
    seen = []
    cast = sim.cast_rays

    def recording(track, x, y, heading, lengths=None):
        seen.append(None if lengths is None else np.array(lengths))
        return cast(track, x, y, heading, lengths)

    monkeypatch.setattr(viewer.sim, "cast_rays", recording)
    rng = np.random.default_rng(6)
    cars = nets.DensePopulation(nets.init_population(rng, 30), setup=setup.random(rng, 30))
    window.rays = True
    window.receive(ga_report(population_=cars))
    window.update(1 / 60)
    seen.clear()
    window.draw()
    best = window.best_slot
    assert len(seen) == 1 and np.array_equal(seen[0], window.playback.cars.ray_length[best:best + 1])
    race_window(window, tmp_path)
    window.update(1 / 60)
    window.start_race()
    window.update(1 / 60)
    seen.clear()
    window.draw()
    live = np.flatnonzero(~window.race.cars.done)
    assert len(seen) == 1 and np.array_equal(seen[0], window.race.cars.ray_length[live])


def test_the_camera_cycles_from_the_whole_map_through_follow_zooms(window):
    labels = []
    for _ in config.CAMERA_ZOOMS:
        labels.append(window.camera_label())
        window.cycle_camera()
    assert labels == ["Camera: Whole map ▸", "Camera: Follow 2× ▸", "Camera: Follow 4× ▸", "Camera: Follow 8× ▸"]
    assert window.camera_zoom() == 1
    window.zoom_by(5)
    assert window.camera_zoom() == config.CAMERA_ZOOMS[-1]               # the wheel stops at the ends
    window.zoom_by(-1)
    assert window.camera_zoom() == config.CAMERA_ZOOMS[-2]
    window.zoom_by(-9)
    assert window.camera_zoom() == 1


def test_the_c_key_and_the_wheel_over_the_map_change_the_camera(window, monkeypatch):
    window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
    assert window.camera_zoom() == 2
    monkeypatch.setattr(viewer.pygame.mouse, "get_pos", lambda: window.view.center)
    window.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))
    assert window.camera_zoom() == 4
    monkeypatch.setattr(viewer.pygame.mouse, "get_pos", lambda: window.panel.center)
    window.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))       # over the panel: no zoom
    assert window.camera_zoom() == 4


def test_the_camera_button_sits_next_to_the_network_window_button(window):
    window.draw()
    (rect, _), = [(r, a) for r, a in window.buttons if a == window.cycle_camera]
    (network, _), = [(r, a) for r, a in window.buttons if a == window.toggle_network_window]
    assert rect.top == network.top and rect.left > network.right
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center))
    assert window.camera_zoom() == 2


def test_a_follow_camera_centres_on_the_best_car_in_training(window):
    window.receive(ga_report())
    window.update(1 / 60)
    for _ in range(30):
        window.update(1 / 60)
    cars, best = window.playback.cars, window.best_slot
    window.cycle_camera()
    camera = window.view_camera(window.playback.track, window.playback)
    assert camera.center == pytest.approx([cars.x[best], cars.y[best]])
    sx, sy = camera.to_screen(cars.x[best], cars.y[best])
    assert (sx, sy) == pytest.approx(window.view.center)
    whole = viewer.Camera(window.playback.track, window.view)
    assert camera.scale == pytest.approx(2 * whole.scale)


def test_a_follow_camera_centres_on_the_race_leader(window, tmp_path):
    race_window(window, tmp_path)
    window.update(1 / 60)
    window.start_race()
    for _ in range(60):
        window.update(1 / 60)
    cars = window.race.cars
    window.zoom_by(2)
    leader = window.follow_point(window.race, window.race.track)
    racing = np.flatnonzero(~cars.done)
    best = racing[np.argmax(cars.progress[racing])]
    assert leader == pytest.approx([cars.x[best], cars.y[best]])
    cars.done[best] = True                                               # the leader crashed or finished
    following = window.follow_point(window.race, window.race.track)
    still = np.flatnonzero(~cars.done)
    if still.size:
        assert following == pytest.approx([cars.x[still[np.argmax(cars.progress[still])]],
                                           cars.y[still[np.argmax(cars.progress[still])]]])


def test_without_cars_a_follow_camera_looks_at_the_start_line(window, circle_track):
    window.cycle_camera()
    start = (circle_track.left[0] + circle_track.right[0]) / 2
    assert window.follow_point(None, circle_track) == pytest.approx(start)


def test_zoomed_views_draw_only_the_visible_track(window, circle_track):
    whole = viewer.Camera(circle_track, window.view)
    assert len(viewer.visible_segments(circle_track, whole)) == len(circle_track.left)
    start = (circle_track.left[0] + circle_track.right[0]) / 2
    close = viewer.Camera(circle_track, window.view, 8, start)
    part = viewer.visible_segments(circle_track, close)
    assert 0 in part and 0 < len(part) < len(circle_track.left) / 2


@pytest.mark.parametrize("zoom_steps", [1, 2, 3])
def test_every_zoom_draws_in_both_views(window, tmp_path, zoom_steps):
    window.zoom_by(zoom_steps)
    window.draw()                                                        # no cars yet: the start line
    window.receive(ga_report())
    for _ in range(5):
        window.update(1 / 60)
        window.draw()
    window.rays = True
    window.draw()
    race_window(window, tmp_path)
    window.update(1 / 60)
    window.draw()
    window.start_race()
    for _ in range(5):
        window.update(1 / 60)
        window.draw()
    assert window.screen.get_clip() == window.screen.get_rect()          # the view's clip is lifted again


def fake_display(monkeypatch, fullscreen_size=(1920, 1080)):
    """pygame.display.set_mode answering with surfaces of the requested size, or `fullscreen_size` for (0, 0)."""
    calls = []

    def set_mode(size, flags=0):
        calls.append((tuple(size), flags))
        return pygame.Surface(fullscreen_size if tuple(size) == (0, 0) else size)

    monkeypatch.setattr(viewer.pygame.display, "set_mode", set_mode)
    return calls


def test_full_screen_fills_the_display_and_windowed_mode_returns(window, monkeypatch):
    calls = fake_display(monkeypatch)
    window.toggle_fullscreen()
    assert window.fullscreen and calls[-1] == ((0, 0), pygame.FULLSCREEN)
    assert window.view.size == (1920 - config.PANEL_WIDTH, 1080) and window.panel.right == 1920
    window.draw()
    assert any(a == window.toggle_fullscreen for _, a in window.buttons)
    window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert not window.fullscreen and calls[-1] == (config.WINDOW_SIZE, pygame.RESIZABLE)
    assert window.view.size == (config.WINDOW_SIZE[0] - config.PANEL_WIDTH, config.WINDOW_SIZE[1])
    window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F11))
    assert window.fullscreen


def test_track_pictures_are_redrawn_for_the_new_size(window, monkeypatch, circle_track):
    fake_display(monkeypatch)
    small = window.track_surface(circle_track)[1].get_size()
    window.toggle_fullscreen()
    big = window.track_surface(circle_track)[1].get_size()
    assert small == (config.WINDOW_SIZE[0] - config.PANEL_WIDTH, config.WINDOW_SIZE[1])
    assert big == (1920 - config.PANEL_WIDTH, 1080)


def test_without_a_usable_display_full_screen_says_so(window, monkeypatch):
    fake_display(monkeypatch, fullscreen_size=(0, 0))
    window.toggle_fullscreen()
    assert not window.fullscreen and "not available" in window.status
    assert window.view.size == (config.WINDOW_SIZE[0] - config.PANEL_WIDTH, config.WINDOW_SIZE[1])

    def failing(size, flags=0):
        raise pygame.error("no display mode")

    monkeypatch.setattr(viewer.pygame.display, "set_mode", failing)
    window.toggle_fullscreen()
    assert not window.fullscreen and "no display mode" in window.status


def test_a_resized_window_lays_out_the_view_and_panel_again(window, monkeypatch):
    monkeypatch.setattr(viewer.pygame.display, "get_surface", lambda: pygame.Surface((1600, 900)))
    window.handle(pygame.event.Event(pygame.VIDEORESIZE, size=(1600, 900), w=1600, h=900))
    assert window.view.size == (1600 - config.PANEL_WIDTH, 900) and window.panel.left == 1600 - config.PANEL_WIDTH
    window.draw()


def test_app_plays_a_report(window):
    window.receive(ga_report())
    for _ in range(120):
        window.update(1 / 60)
        window.draw()
    assert window.playback is not None and window.playback_generation == 0


def test_stop_freezes_the_training_replay(window):
    window.receive(ga_report())
    for _ in range(30):
        window.update(1 / 60)
    playback = window.playback
    assert playback is not None and playback.cars.time > 0
    window.stop()
    frozen_time, frozen_x = playback.cars.time, playback.cars.x.copy()
    for _ in range(300):                                 # long enough for a replay to end and start again
        window.update(1 / 60)
        window.draw()
    assert window.playback is playback                   # not restarted
    assert playback.cars.time == frozen_time and np.array_equal(playback.cars.x, frozen_x)
    assert window.latest is not None                     # the last generation's stats stay on the panel


def test_play_after_stop_moves_the_replay_again(window, monkeypatch):
    window.receive(ga_report())
    for _ in range(10):
        window.update(1 / 60)
    window.stop()
    monkeypatch.setattr(window, "start_training", lambda: setattr(window, "halted", False))
    window.toggle_play()
    time_before = window.playback.cars.time
    for _ in range(10):
        window.update(1 / 60)
    assert window.playback.cars.time > time_before


def test_training_waits_until_the_viewer_has_replayed_the_generation(monkeypatch, tmp_path):
    from racing import ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)

    def run(**kwargs):
        for generation in (0, 1):
            kwargs["on_generation"](ga_report(generation))

    monkeypatch.setattr(ga, "run_ga", run)
    reports, acks, running = queue.Queue(), queue.Queue(), threading.Event()
    running.set()
    trainer = threading.Thread(target=viewer.trainer_main, args=("ga", 30, 0, 1, reports, running),
                               kwargs={"acks": acks}, daemon=True)
    trainer.start()
    try:
        assert reports.get(timeout=5)["generation"] == 0
        with pytest.raises(queue.Empty):
            reports.get(timeout=0.5)                         # generation 1 waits for the replay of generation 0
        acks.put(True)
        assert reports.get(timeout=5)["generation"] == 1
    finally:
        acks.put(True)
        acks.put(True)
    trainer.join(timeout=5)
    assert not trainer.is_alive()


def test_each_generation_is_replayed_once_then_training_continues(window):
    window.acks = queue.Queue()
    window.receive(ga_report(0))
    window.update(1 / 60)
    playback = window.playback
    assert playback is not None and window.acks.empty()             # still replaying: training waits
    playback.cars.done[:] = True                                     # the replay ends
    for _ in range(30):
        window.update(1 / 60)
    assert window.acks.qsize() == 1                                  # released exactly once
    assert window.playback is playback                               # not replayed again while the next one trains
    assert "generation 1" in window.status
    window.receive(ga_report(1))
    window.update(1 / 60)
    assert window.playback is not playback and window.playback_generation == 1
    assert window.acks.qsize() == 1                                  # generation 1 is not released before its replay


def test_pause_freezes_the_replay(window):
    class Alive:
        pid = 0

        def is_alive(self):
            return True

    window.process, window.running = Alive(), threading.Event()
    window.running.set()
    window.receive(ga_report())
    for _ in range(10):
        window.update(1 / 60)
    window.toggle_play()                                             # Pause
    frozen = window.playback.cars.time
    for _ in range(30):
        window.update(1 / 60)
    assert window.playback.cars.time == frozen
    window.toggle_play()                                             # Play
    for _ in range(10):
        window.update(1 / 60)
    assert window.playback.cars.time > frozen


def test_the_race_view_lets_training_run_free(window, tmp_path):
    window.champions_dir = tmp_path
    window.acks = queue.Queue()
    window.receive(ga_report(0))
    window.update(1 / 60)
    assert window.acks.empty()                                       # waits for its replay in the training view
    window.set_mode("race")
    assert window.acks.qsize() == 1                                  # released when nobody watches the training
    window.receive(ga_report(1))
    assert window.acks.qsize() == 2                                  # and every report while the race view is shown


def test_neat_report_plays_in_the_training_view(window):
    window.approach = "neat"
    window.receive(neat_report())
    for _ in range(60):
        window.update(1 / 60)
        window.draw()
    assert window.playback is not None and window.playback.cars.laps == config.LAPS


def test_history_marks_level_ups_and_target_cuts(window):
    for generation, event in enumerate((None, "level_up", None, "goal", "tightened", "end_speed")):
        window.receive(ga_report(generation, event=event))
    assert [h[3] for h in window.history] == [False, True, False, True, True, False]
    assert [h[1] for h in window.history] == [0.84] * 6
    window.draw()                                        # the success-share plot draws with markers


def test_playback_advances_the_simulation(circle_track):
    playback = viewer.Playback(circle_track, lambda obs, cars: np.full(len(cars.x), 8), 3)   # class 8 = straight, gas
    playback.advance(10)
    assert abs(playback.cars.time - 10 * config.DT) < 1e-9 and np.all(playback.cars.speed > 0)


def test_cars_shown_never_exceeds_population(window):
    window.shown = 10_000
    window.receive(ga_report(n=30))
    window.update(1 / 60)
    assert len(window.indices) == 30


def test_app_moves_to_the_newer_generation(window):
    window.receive(ga_report(0))
    window.update(1 / 60)
    window.playback.cars.done[:] = True
    window.receive(ga_report(1))
    window.update(1 / 60)
    assert window.playback_generation == 1


def test_race_without_champions_shows_a_message(window, tmp_path):
    window.champions_dir = tmp_path
    window.set_mode("race")
    window.update(1 / 60)
    window.draw()
    assert "No trained controllers" in window.status


def test_dead_training_process_is_reported(window):
    class Dead:
        exitcode = 1

        def is_alive(self):
            return False

    window.process = Dead()
    window.reports = queue.Queue()
    window.poll_reports()
    assert "stopped" in window.status.lower() and window.process is None


class FakeQueue:
    """Stands in for a multiprocessing queue."""

    def __init__(self, items=()):
        self.items = list(items)
        self.closed = False

    def get_nowait(self):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)

    def cancel_join_thread(self):
        pass

    def close(self):
        self.closed = True


def test_switching_approach_discards_stale_reports(window):
    old = FakeQueue([ga_report(0)])
    window.reports = old
    window.select_approach("es")
    window.update(1 / 60)
    assert old.closed and window.reports is None and window.latest is None and window.history == []


def test_finished_training_discards_the_queue(window):
    class Done:
        exitcode = 0

        def is_alive(self):
            return False

    window.process, window.reports = Done(), FakeQueue()
    window.poll_reports()
    assert window.reports is None and "finished" in window.status.lower()


def test_reports_are_drained_in_race_mode(window, tmp_path):
    window.champions_dir = tmp_path
    window.set_mode("race")
    window.reports = FakeQueue([ga_report(0), ga_report(1)])
    window.update(1 / 60)
    assert window.reports.items == [] and window.latest["generation"] == 1


def test_trainer_main_uses_app_runs_dir_and_resumes_saved_runs(monkeypatch, tmp_path):
    from racing import es, ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    calls = {}

    def fake(name):
        def run(**kwargs):
            calls[name] = dict(kwargs, existed=(kwargs["out_dir"] / "checkpoint.pkl").exists())
            kwargs["on_generation"](ga_report(0))
        return run

    monkeypatch.setattr(ga, "run_ga", fake("ga"))
    monkeypatch.setattr(es, "run_es", fake("es"))
    monkeypatch.setattr(neat, "run_neat", fake("neat"))
    stale = tmp_path / "ga" / "seed0" / "checkpoint.pkl"          # left by an earlier Play
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    reports, running = queue.Queue(), threading.Event()
    running.set()
    viewer.trainer_main("ga", 30, 0, 1, reports, running)
    viewer.trainer_main("es", 12, 0, 1, reports, running)
    viewer.trainer_main("neat", 100, 0, 1, reports, running)
    for name, size in (("ga", 30), ("es", 12), ("neat", 100)):
        assert calls[name]["champion_dir"] is None and calls[name]["pop_size"] == size
        assert calls[name]["max_generations"] is None and calls[name]["out_dir"] == tmp_path / name / "seed0"
    assert calls["ga"]["existed"]                                  # Play continues the saved run (spec §10.6)
    assert [reports.get_nowait()["generation"] for _ in range(3)] == [0, 0, 0] and reports.empty()


def test_pause_and_resume_toggle_the_running_event(window):
    class Alive:
        pid = 0

        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    window.process, window.running = Alive(), threading.Event()
    window.running.set()
    window.toggle_play()
    assert not window.running.is_set()
    window.toggle_play()
    assert window.running.is_set()


def save_race_champions(directory, approaches=(("es", 0), ("ga", 0), ("ga", 1))):
    rng = np.random.default_rng(0)
    for approach, seed in approaches:
        if approach == "neat":
            neat.save_neat_champion(approach, seed, neat.initial_genome(rng, neat.Innovations()), directory)
        else:
            population.save_dense_champion(approach, seed, nets.DensePopulation(nets.init_population(rng, 1)), 0,
                                           directory)


def race_window(window, tmp_path):
    save_race_champions(tmp_path)                                           # es run 0, ga runs 0 and 1
    window.champions_dir = tmp_path
    window.race_runs = "all"                                                # the default best races one run each
    window.set_mode("race")
    return window


def champions_0_to_3(window, tmp_path, best=None):
    """Controllers: ga runs 0-3 and es runs 0-1, and optionally the evaluation's best-run record."""
    save_race_champions(tmp_path, approaches=[("es", 0), ("es", 1)] + [("ga", run) for run in range(4)])
    if best is not None:
        paths.best_seeds_path(tmp_path).write_text(best)
    window.champions_dir = tmp_path
    window.set_mode("race")


def test_best_races_each_methods_best_run_and_is_the_default(window, tmp_path):
    """spec §10.3: best races the run the evaluation recorded as each method's best; best is the default."""
    champions_0_to_3(window, tmp_path, best='{"ga": 2, "es": 0}')
    assert window.race_runs == "best" and window.race_runs_label() == "Race: best runs ▸"
    assert window.best == {"ga": 2, "es": 0}
    assert window.lineup == [("es", 0), ("ga", 2)]
    assert window.status == "Race view."


@pytest.mark.parametrize("record", [None, '{"ga": ', '[1, 2]', '{"ga": "2", "es": true, "neat": 3}'])
def test_best_falls_back_to_run_0_without_a_readable_record(window, tmp_path, record):
    """No record, a corrupt one, non-integer values, or a run that was not saved: run 0 of each method."""
    champions_0_to_3(window, tmp_path, best=record)
    assert window.lineup == [("es", 0), ("ga", 0)]
    if record in ('{"ga": ', '[1, 2]'):
        assert window.best == {} and "best_seeds.json could not be read" in window.status
    else:
        assert "could not be read" not in window.status


def test_best_without_the_recorded_run_saved_races_the_lowest_saved_run(window, tmp_path):
    save_race_champions(tmp_path, approaches=[("ga", 1), ("ga", 2), ("es", 0)])
    paths.best_seeds_path(tmp_path).write_text('{"ga": 3}')
    window.champions_dir = tmp_path
    window.set_mode("race")
    assert window.lineup == [("es", 0), ("ga", 1)]


def test_unseeded_races_runs_1_to_3_and_all_races_every_run(window, tmp_path):
    """spec §10.3: the control cycles best, all, seeded, unseeded."""
    champions_0_to_3(window, tmp_path, best='{"ga": 2, "es": 0}')
    assert window.lineup == [("es", 0), ("ga", 2)]
    window.cycle_race_runs()
    assert window.race_runs == "all" and window.race_runs_label() == "Race: all runs ▸"
    assert window.lineup == [("es", 0), ("es", 1)] + [("ga", run) for run in range(4)]
    window.cycle_race_runs()
    assert window.race_runs == "seeded" and window.lineup == [("es", 0), ("ga", 0)]
    assert window.race_runs_label() == "Race: seeded runs ▸"
    window.cycle_race_runs()
    assert window.race_runs == "unseeded" and window.lineup == [("es", 1), ("ga", 1), ("ga", 2), ("ga", 3)]
    assert window.race_runs_label() == "Race: unseeded runs ▸"
    window.cycle_race_runs()
    assert window.race_runs == "best" and window.lineup == [("es", 0), ("ga", 2)]


def test_a_method_can_be_switched_off(window, tmp_path):
    race_window(window, tmp_path)
    assert window.race_on == {"es": True, "ga": True}
    window.toggle_race("ga")
    assert all(car[0] != "ga" for car in window.lineup) and window.lineup == [("es", 0)]
    window.toggle_race("ga")
    assert ("ga", 0) in window.lineup and ("ga", 1) in window.lineup


def test_the_runs_control_and_the_switches_are_kept_while_the_view_is_open(window, tmp_path):
    race_window(window, tmp_path)
    window.cycle_race_runs()
    window.toggle_race("es")
    window.set_mode("train")
    window.set_mode("race")                                                 # loads the controllers again
    assert window.race_runs == "seeded" and window.race_on == {"es": False, "ga": True}
    assert window.lineup == [("ga", 0)]


def recording_buttons(window, monkeypatch):
    """The buttons the window draws, as (label, action, active, color, rect)."""
    drawn, real = [], window.button

    def recording_button(rect, label, action, active=False, enabled=True, color=None):
        drawn.append((label, action, active, color, rect))
        return real(rect, label, action, active, enabled, color)

    monkeypatch.setattr(window, "button", recording_button)
    return drawn


def method_switches(drawn):
    """The race view's method switches among drawn buttons, as (label, on, rect): a method's name in its color."""
    colors = {config.LABELS[approach]: config.COLORS[approach] for approach in config.APPROACHES}
    return [(label, active, rect) for label, _, active, color, rect in drawn
            if color is not None and colors.get(label) == color]


def test_the_race_panel_has_the_race_control_and_a_switch_per_method(window, tmp_path, monkeypatch):
    race_window(window, tmp_path)
    drawn = recording_buttons(window, monkeypatch)
    window.draw()
    assert ("Race: all runs ▸", window.cycle_race_runs) in [(label, action) for label, action, *_ in drawn]
    assert [(label, on) for label, on, _ in method_switches(drawn)] == [("ES", True), ("GA", True)]
    next(action for label, action, *_ in drawn if label == "GA")()          # GA's switch
    assert window.race_on["ga"] is False and window.lineup == [("es", 0)]
    drawn.clear()
    window.draw()
    assert [(label, on) for label, on, _ in method_switches(drawn)] == [("ES", True), ("GA", False)]


def test_all_races_every_saved_run_of_an_earlier_revision(window, tmp_path, monkeypatch, circle_track):
    """An archive of an earlier revision may hold runs 0, 1 and 2; 'all' races every one. Every run of an archive
    was seeded, so the leaderboard names each by its seed, 'GA s0' to 'GA s2' (spec §10.3)."""
    monkeypatch.setattr(viewer, "get_track", lambda seed, level, width=None: circle_track)
    folder = archive_folder(tmp_path, inputs=config.N_INPUTS)
    save_race_champions(folder / "champions", approaches=[("ga", 0), ("ga", 1), ("ga", 2)])
    window.results_dir = tmp_path
    window.race_runs = "all"
    window.set_mode("race")
    window.cycle_folder()
    assert window.race_available == {"ga": [0, 1, 2]}
    assert {("ga", 0), ("ga", 1), ("ga", 2)} <= set(window.lineup)
    window.update(1 / 60)
    window.start_race()
    window.update(1 / 60)
    assert len(window.race.cars.x) == 3
    drawn, text = [], window.text

    def recording_text(string, x, y, font=None, color=viewer.INK):
        drawn.append(string)
        return text(string, x, y, font, color)

    monkeypatch.setattr(window, "text", recording_text)
    window.draw()
    rows = [string for string in drawn if string[:1].isdigit() and ". " in string]
    assert sorted(row.split(" lap")[0].split(". ", 1)[1] for row in rows) == ["GA s0", "GA s1", "GA s2"]
    window.stop_race()
    window.cycle_race_runs()
    assert window.lineup == [("ga", 0)]
    window.cycle_race_runs()
    assert window.lineup == [("ga", 1), ("ga", 2)]                         # unseeded: every run but run 0


def test_the_leaderboard_names_the_seeded_run():
    assert viewer.leaderboard_line(1, "ga", 0, 2, 50.0, 49.0, "racing").startswith("1. GA (seeded) lap 2")
    assert viewer.leaderboard_line(2, "ga", 1, 2, 50.0, 49.0, "racing").startswith("2. GA r1 lap 2")
    assert viewer.leaderboard_line(3, "ga", 2, 2, 50.0, 49.0, "racing").startswith("3. GA r2 lap 2")
    assert viewer.leaderboard_line(4, "ga", 0, 2, 50.0, 49.0, "racing", app=True).startswith("4. GA (app) s0 lap 2")


def test_run_1_cars_carry_a_white_stripe(window, tmp_path, monkeypatch):
    """spec §10.3: the run 1 car carries a white stripe from its rear to its nose; run 0 cars and app cars do not."""
    app_race_window(window, tmp_path)                       # reported: es 0, ga 0, ga 1; app cars: ga 2, neat 0
    window.cycle_app_choice("ga")
    window.update(1 / 60)
    assert window.race_cars() == [("es", 0), ("ga", 0), ("ga", 1), ("ga", 2, "app")]
    camera = window.view_camera(window.race.track, window.race)            # draws the track picture once
    calls = []
    monkeypatch.setattr(viewer.pygame.draw, "polygon",
                        lambda surface, color, points, width=0: calls.append(("polygon", tuple(color), points, width)))
    monkeypatch.setattr(viewer.pygame.draw, "line",
                        lambda surface, color, start, end, width=1: calls.append(("line", tuple(color), start, end,
                                                                                  width)))
    window.draw_racers(window.race, camera)
    fills = [k for k, call in enumerate(calls) if call[0] == "polygon" and call[3] == 0]
    stripes = [k for k, call in enumerate(calls) if call[0] == "line"]
    assert len(fills) == 4 and len(stripes) == 1 and fills[2] < stripes[0] < fills[3]      # only ga run 1's
    nose, left, right = calls[fills[2]][2]
    _, color, start, end, width = calls[stripes[0]]
    assert color == viewer.WHITE and width == 2 and end == nose
    assert start == pytest.approx(((left[0] + right[0]) / 2, (left[1] + right[1]) / 2))


def test_an_archives_runs_are_named_by_their_seeds_and_carry_no_stripe(window, tmp_path, monkeypatch, circle_track):
    """Every run of an archive of an earlier revision was seeded: its cars are named 'GA s0' and 'GA s1' and none
    carries the run 1 stripe; the race control still races run 0, run 1 or all (spec §10.3)."""
    monkeypatch.setattr(viewer, "get_track", lambda seed, level, width=None: circle_track)
    assert [viewer.car_name("ga", 0, archived=True), viewer.car_name("ga", 1, archived=True),
            viewer.car_name("ga", 1, app=True, archived=True)] == ["GA s0", "GA s1", "GA (app) s1"]
    folder = archive_folder(tmp_path, inputs=config.N_INPUTS)
    save_race_champions(folder / "champions", approaches=[("ga", 0), ("ga", 1)])
    save_race_champions(tmp_path / "champions", approaches=[("ga", 1)])
    window.results_dir = tmp_path
    window.race_runs = "all"
    window.set_mode("race")
    window.cycle_folder()
    window.update(1 / 60)
    assert window.race_cars() == [("ga", 0), ("ga", 1)] and window.network_car_label() == "Network of: GA s0 ▸"
    camera = window.view_camera(window.race.track, window.race)
    lines = []
    monkeypatch.setattr(viewer.pygame.draw, "line",
                        lambda surface, color, start, end, width=1: lines.append((start, end)))
    window.draw_racers(window.race, camera)
    assert lines == []
    window.cycle_race_runs()
    assert window.lineup == [("ga", 0)]
    window.cycle_race_runs()
    assert window.lineup == [("ga", 1)]
    window.cycle_folder()                                   # the current results: run 1 carries its stripe again
    window.update(1 / 60)
    assert window.race_cars() == [("ga", 1)]
    window.draw_racers(window.race, window.view_camera(window.race.track, window.race))
    assert len(lines) == 1 and window.network_car_label() == "Network of: GA r1 ▸"


def test_preview_waits_on_the_start_line_until_started(window, tmp_path):
    race_window(window, tmp_path)
    for _ in range(5):
        window.update(1 / 60)
        window.draw()
    assert window.race is not None and window.race.cars.time == 0.0 and not window.racing
    window.start_race()
    window.update(0.1)
    assert window.racing and window.race.cars.time > 0
    window.stop()
    window.update(1 / 60)
    assert not window.racing and window.race.cars.time == 0.0


def test_race_settings_are_locked_while_racing(window, tmp_path):
    race_window(window, tmp_path)
    window.update(1 / 60)
    window.start_race()
    lineup = list(window.lineup)
    window.change_race_track(1)
    window.set_race_level(2)
    window.cycle_race_runs()
    window.toggle_race("ga")
    assert window.race_index == 0 and window.race_level == config.MAX_LEVEL and window.lineup == lineup
    assert window.race_runs == "all" and window.race_on["ga"]
    window.draw()
    assert window.cycle_race_runs not in [action for _, action in window.buttons]      # drawn disabled


def test_race_difficulty_sets_the_track_level(monkeypatch, window, tmp_path, circle_track):
    calls = []
    monkeypatch.setattr(viewer, "get_track", lambda seed, level: calls.append((seed, level)) or circle_track)
    race_window(window, tmp_path)
    window.set_race_level(2)
    window.update(1 / 60)
    assert (config.TEST_SEEDS[0], 2) in calls


def test_race_draws_a_leaderboard_and_fits_the_window(window, tmp_path):
    race_window(window, tmp_path)
    window.update(1 / 60)
    window.start_race()
    for _ in range(30):
        window.update(1 / 60)
        window.draw()
    rows = viewer.leaderboard(window.race, len(window.lineup))
    assert sorted(r[0] for r in rows) == [1, 2, 3] and sorted(r[1] for r in rows) == [0, 1, 2]
    assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons)


def test_unreadable_controller_is_skipped(window, tmp_path):
    race_window(window, tmp_path)
    paths.champion_path("ga", 0, champion_dir=tmp_path).write_bytes(b"garbage")
    paths.champion_path("ga", 0, ".npy", tmp_path).unlink()
    window.update(1 / 60)
    window.update(1 / 60)
    assert ("ga", 0) not in window.lineup and window.race is not None
    assert window.lineup == [("es", 0), ("ga", 1)] and "GA run 0 left out" in window.status


def test_leaderboard_orders_by_distance(circle_track):
    playback = viewer.Playback(circle_track, lambda obs, cars: np.zeros(len(cars.x), dtype=int), 2, laps=3)
    playback.cars.progress[:] = [10.0, 50.0]
    playback.cars.crashed[0] = True
    rows = viewer.leaderboard(playback, 2)
    assert [r[1] for r in rows] == [1, 0] and rows[1][5] == "crashed" and [r[0] for r in rows] == [1, 2]
    ready = viewer.leaderboard(None, 2)
    assert [(r[0], r[1], r[2], r[5]) for r in ready] == [(1, 0, 0, "ready"), (2, 1, 0, "ready")]


def test_champions_saved_later_are_picked_up(window, tmp_path):
    window.champions_dir = tmp_path
    window.set_mode("race")
    assert "No trained controllers" in window.status
    window.set_mode("train")
    save_race_champions(tmp_path)
    window.race_runs = "all"
    window.set_mode("race")
    assert window.race_available == {"es": [0], "ga": [0, 1]}
    assert window.lineup == [("es", 0), ("ga", 0), ("ga", 1)]
    save_race_champions(tmp_path, approaches=[("es", 1)])
    window.set_mode("train")
    window.set_mode("race")
    assert window.lineup == [("es", 0), ("es", 1), ("ga", 0), ("ga", 1)]


def test_step_burst_is_capped_and_budget_dropped(window):
    window.receive(ga_report())
    window.update(0.0)
    window.update(10.0)
    assert abs(window.playback.cars.time - config.MAX_STEPS_PER_FRAME * config.DT) < 1e-9
    assert window.step_budget == 0.0


def click(pos):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)


def test_handle_clicks_a_button(window):
    window.draw()
    rays = window.rays
    target = next(r for r, a in window.buttons if a == window.toggle_rays)
    window.handle(click(target.center))
    assert window.rays != rays


def test_handle_slider_sets_cars_shown(window):
    window.receive(ga_report(n=30))
    window.draw()
    window.handle(click((window.slider.left, window.slider.centery)))
    assert window.shown == 1 and window.dragging
    window.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(window.slider.right, window.slider.centery), rel=(0, 0), buttons=(1, 0, 0)))
    assert window.shown == 30
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(0, 0)))
    assert not window.dragging


def test_pausing_still_delivers_the_finished_generation(monkeypatch, tmp_path):
    from racing import ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    monkeypatch.setattr(ga, "run_ga", lambda **kwargs: kwargs["on_generation"](ga_report(0)))
    reports, running = queue.Queue(), threading.Event()           # paused: running is not set
    trainer = threading.Thread(target=viewer.trainer_main, args=("ga", 30, 0, 1, reports, running), daemon=True)
    trainer.start()
    try:
        assert reports.get(timeout=5)["generation"] == 0          # the finished generation is shown while paused
        assert trainer.is_alive()                                  # and training holds until resumed
    finally:
        running.set()
    trainer.join(timeout=5)
    assert not trainer.is_alive()


def fake_imitation(monkeypatch, tmp_path):
    """Stubs dataset collection and baseline training; returns the lists they record into."""
    from racing import expert, imitation
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path / "app_runs")
    monkeypatch.setattr(config, "DATASET_PATH", tmp_path / "results" / "dataset.npz")
    collected, trained = [], []

    def collect(path=None, splits=None):
        collected.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, y_train=np.array([1]))
        return {}, {}

    monkeypatch.setattr(expert, "collect_dataset", collect)
    monkeypatch.setattr(imitation, "train_baseline", lambda data, **kwargs: trained.append(data))
    return collected, trained


def run_baseline_trainer():
    running = threading.Event()
    running.set()
    viewer.trainer_main("baseline", 0, 0, 1, queue.Queue(), running)


def test_trainer_main_collects_its_own_dataset(monkeypatch, tmp_path):
    collected, trained = fake_imitation(monkeypatch, tmp_path)
    run_baseline_trainer()
    assert collected == [tmp_path / "app_runs" / "dataset.npz"] and not config.DATASET_PATH.exists()
    run_baseline_trainer()                                          # reuses the dataset it collected
    assert len(collected) == 1 and [list(d["y_train"]) for d in trained] == [[1], [1]]


def test_trainer_main_uses_the_reported_dataset(monkeypatch, tmp_path):
    collected, trained = fake_imitation(monkeypatch, tmp_path)
    config.DATASET_PATH.parent.mkdir(parents=True)
    np.savez(config.DATASET_PATH, y_train=np.array([7]))
    run_baseline_trainer()
    assert collected == [] and list(trained[0]["y_train"]) == [7]


def test_trainer_main_trains_imitation_afresh_and_saves_no_final_controller(monkeypatch, tmp_path):
    from racing import imitation
    fake_imitation(monkeypatch, tmp_path)
    options = []
    monkeypatch.setattr(imitation, "train_baseline", lambda data, **kwargs: options.append(kwargs))
    run_baseline_trainer()
    assert options[0]["resume"] is False and "champion_dir" not in options[0]
    assert options[0]["out_dir"] == tmp_path / "app_runs" / "baseline" / "seed0"


def test_viewer_imitation_training_never_resumes(monkeypatch, tmp_path):
    from racing import imitation
    fake_imitation(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(imitation, "train_baseline",
                        lambda data, **kwargs: calls.append(("baseline", kwargs["resume"], kwargs["report"])))
    monkeypatch.setattr(imitation, "run_tuner",
                        lambda approach, data, **kwargs: calls.append((approach, kwargs["resume"], kwargs["report"])))
    running = threading.Event()
    running.set()
    viewer.trainer_main("baseline", 0, 0, 1, queue.Queue(), running)
    viewer.trainer_main("kt_random", 0, 0, 1, queue.Queue(), running)
    assert calls == [("baseline", False, False), ("kt_random", False, False)]     # never the experiment's status


def test_trainer_main_passes_seed_level_and_seeded_start(monkeypatch, tmp_path):
    from racing import es, ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    calls = {}
    monkeypatch.setattr(ga, "run_ga", lambda **kwargs: calls.setdefault("ga", kwargs))
    monkeypatch.setattr(es, "run_es", lambda **kwargs: calls.setdefault("es", kwargs))
    monkeypatch.setattr(es, "dagger_baseline", lambda seed: f"baseline seed {seed}")
    reports, running = queue.Queue(), threading.Event()
    running.set()
    viewer.trainer_main("ga", 30, 2, 3, reports, running)
    viewer.trainer_main("es_seeded", 12, 1, 2, reports, running)
    assert calls["ga"]["seed"] == 2 and calls["ga"]["start_level"] == 3
    assert calls["es"]["approach"] == "es_seeded" and calls["es"]["initial"] == "baseline seed 1"
    assert calls["es"]["out_dir"] == tmp_path / "es_seeded" / "seed1" and calls["es"]["start_level"] == 2


def test_the_viewers_search_of_run_0_is_seeded_and_of_run_1_is_not(monkeypatch, tmp_path):
    """spec §3.3, §10.2: the viewer's own Keras Tuner search of run 0 is seeded with 0; run 1's takes a number from
    the operating system each time, so it never replays."""
    from racing import imitation
    fake_imitation(monkeypatch, tmp_path)
    seeds, drawn, real = [], [], viewer.secrets.randbits
    monkeypatch.setattr(imitation, "run_tuner", lambda approach, data, **kwargs: seeds.append(kwargs["seed"]))
    monkeypatch.setattr(viewer.secrets, "randbits", lambda bits: drawn.append((bits, real(bits))) or drawn[-1][1])
    running = threading.Event()
    running.set()
    for run in (0, 1, 1):
        viewer.trainer_main("kt_random", 0, run, 1, queue.Queue(), running)
    assert seeds[0] == 0 and [bits for bits, _ in drawn] == [31, 31]
    assert seeds[1:] == [value for _, value in drawn] and seeds[1] != seeds[2]


def test_training_process_is_not_a_daemon(window, monkeypatch):
    created = {}

    class FakeProcess:
        def __init__(self, target, args, daemon):
            created.update(target=target, args=args, daemon=daemon)
            self.pid = 0

        def start(self):
            pass

        def is_alive(self):
            return False

    class FakeContext:
        Process = FakeProcess

        def Queue(self):
            return FakeQueue()

        def Event(self):
            return threading.Event()

    monkeypatch.setattr(viewer.mp, "get_context", lambda method: FakeContext())
    window.change_run_seed(1)
    window.set_difficulty(2)
    window.start_training()
    assert created["daemon"] is False and created["target"] is viewer.trainer_process
    assert created["args"][:4] == ("ga", config.GA_POP, config.RUN_SEEDS[1], 2)
    assert window.acks is not None and created["args"][-1] is window.acks     # the trainer waits for the replays
    assert window.status == "Training GA, run 1."


def test_stop_ends_the_trainer_process_group(window, monkeypatch):
    killed = []
    monkeypatch.setattr(viewer.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))

    class Alive:
        pid = 4321

        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    window.process, window.reports = Alive(), FakeQueue()
    window.stop()
    assert killed == [(4321, signal.SIGTERM)] and window.process is None and "Stopped" in window.status


def test_trainer_ends_its_process_group_once_the_viewer_is_gone(monkeypatch):
    parents = iter([100, 100, 1])                          # the viewer, the viewer, then the adopting process
    monkeypatch.setattr(viewer.os, "getppid", lambda: next(parents))
    killed = []
    viewer.watch_parent(100, lambda pgid, sig: killed.append((pgid, sig)), interval=0)
    assert killed == [(0, signal.SIGTERM)]


def test_trainer_process_watches_the_viewer(monkeypatch):
    events = []

    class FakeThread:
        def __init__(self, target, args, daemon):
            self.call = (target, args, daemon)

        def start(self):
            events.append(self.call)

    def fake_killpg(pgid, sig):                            # tests never signal real processes
        pass

    monkeypatch.setattr(viewer.os, "killpg", fake_killpg)
    monkeypatch.setattr(viewer.os, "setpgrp", lambda: events.append("setpgrp"))
    monkeypatch.setattr(viewer.os, "getppid", lambda: 4321)
    monkeypatch.setattr(viewer.threading, "Thread", FakeThread)
    monkeypatch.setattr(viewer, "trainer_main", lambda *args: events.append("train"))
    viewer.trainer_process("ga", 10, 0, 1, None, None)
    assert events == ["setpgrp", (viewer.watch_parent, (4321, fake_killpg), True), "train"]


def test_leaderboard_lines_fit_the_panel(window):
    width = window.panel.width - 16 - 16 - 16           # panel margins on both sides and the color swatch
    for approach in config.APPROACHES:
        for seed, app in ((0, False), (1, False), (2, False), (2, True)):      # 'GA (seeded)' is the widest run
            for status in ("ready", "racing", "finished", "crashed", "stopped"):
                line = viewer.leaderboard_line(2 * len(config.APPROACHES), approach, seed, config.RACE_LAPS, 888.88,
                                               999.99, status, app=app)
                assert window.small.size(line)[0] <= width - 4, line      # 4 px to spare: fonts differ by machine
    line = viewer.leaderboard_line(1, "kt_hyperband", 0, 1, 52.31, 51.94, "racing")
    assert line == "1. KT Hyperband (seeded) lap 1 52.3/51.9 racing"
    assert viewer.leaderboard_line(2, "ga", 1, 0, float("nan"), float("nan"), "ready") == "2. GA r1 lap 0 –/– ready"


def test_race_panel_text_fits_the_panel(window, tmp_path, monkeypatch):
    race_window(window, tmp_path)
    window.update(1 / 60)
    window.start_race()
    window.update(1 / 60)
    drawn, text = [], window.text

    def recording_text(string, x, y, font=None, color=viewer.INK):
        drawn.append((string, x, font or window.font))
        return text(string, x, y, font, color)

    monkeypatch.setattr(window, "text", recording_text)
    window.draw_race_panel(window.panel.left + 16, 58, window.panel.width - 32)
    assert any(string.startswith("1. ") for string, _, _ in drawn)
    for string, x, font in drawn:
        assert x + font.size(string)[0] <= window.panel.right - 16, string


def test_a_race_of_every_method_fits_the_panel(window, tmp_path, monkeypatch):
    save_race_champions(tmp_path, approaches=[(approach, 0) for approach in config.APPROACHES])
    window.champions_dir = tmp_path
    window.set_mode("race")
    window.update(1 / 60)
    window.start_race()
    window.update(1 / 60)
    assert [approach for approach, _ in window.lineup] == list(config.APPROACHES)
    drawn, text = [], window.text

    def recording_text(string, x, y, font=None, color=viewer.INK):
        drawn.append((string, y, font or window.font))
        return text(string, x, y, font, color)

    monkeypatch.setattr(window, "text", recording_text)
    window.buttons = []
    window.draw_race_panel(window.panel.left + 16, 58, window.panel.width - 32)
    assert sum(string.startswith(f"{k}. ") for string, _, _ in drawn for k in range(1, 9)) == 8
    for string, y, font in drawn:
        assert y + font.get_linesize() <= window.panel.bottom - 30, string
    assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons)


@pytest.mark.parametrize("size", [config.WINDOW_SIZE, (1920, 1080)])
def test_a_race_of_both_runs_of_every_method_fits_the_panel(window, tmp_path, monkeypatch, size):
    """16 cars, the race control on all runs (spec §10.3): the panel fits and, from the default window up, the
    leaderboard shows every car; the method switches form a grid of three columns."""
    window.screen = pygame.display.set_mode(size)
    window.layout()
    save_race_champions(tmp_path, approaches=[(approach, run) for approach in config.APPROACHES for run in (0, 1)])
    window.champions_dir = tmp_path
    window.race_runs = "all"
    window.set_mode("race")
    window.update(1 / 60)
    window.start_race()
    window.update(1 / 60)
    assert len(window.lineup) == 16 and len(window.race.cars.x) == 16
    drawn, text = [], window.text

    def recording_text(string, x, y, font=None, color=viewer.INK):
        drawn.append((string, x, y, font or window.font))
        return text(string, x, y, font, color)

    monkeypatch.setattr(window, "text", recording_text)
    buttons = recording_buttons(window, monkeypatch)
    window.buttons = []
    window.draw_race_panel(window.panel.left + 16, 58, window.panel.width - 32)
    for string, x, y, font in drawn:
        assert y + font.get_linesize() <= window.panel.bottom - 30, string
        assert x + font.size(string)[0] <= window.panel.right - 16, string
    assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons)
    for label, *_, rect in buttons:
        assert window.small.size(label)[0] <= rect.width - 10, label             # each label fits its button
    rows = [string for string, _, _, _ in drawn if string[:1].isdigit() and ". " in string]
    assert len(rows) == 16 and not any(string.startswith("… ") for string, _, _, _ in drawn)
    switches = method_switches(buttons)
    assert [label for label, _, _ in switches] == [config.LABELS[approach] for approach in config.APPROACHES]
    assert len({rect.left for _, _, rect in switches}) == 3 and len({rect.top for _, _, rect in switches}) == 3


def test_settings_are_locked_while_training(window):
    class Alive:
        pid = 0

        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    window.process = Alive()
    window.select_approach("es")
    window.change_run_seed(1)
    window.set_difficulty(3)
    window.change_population(1)
    assert window.approach == "ga" and window.run_seed == config.RUN_SEEDS[0] and window.difficulty == 1
    assert window.population["ga"] == config.GA_POP


def test_the_run_control_names_the_seeded_run(window, monkeypatch):
    """spec §10.2: the training view's control reads 'Run', its value '0 (seeded)' for run 0 and '1' for run 1."""
    drawn, text = [], window.text

    def recording_text(string, x, y, font=None, color=viewer.INK):
        drawn.append((string, x, font or window.font))
        return text(string, x, y, font, color)

    monkeypatch.setattr(window, "text", recording_text)
    buttons = recording_buttons(window, monkeypatch)
    window.draw()
    strings = [string for string, _, _ in drawn]
    assert "Run" in strings and "Run seed" not in strings
    value, x, font = next(item for item in drawn if item[0] == "0 (seeded)")
    later = next(rect for label, *_, rect in buttons if label == ">")
    assert x + font.size(value)[0] < later.left                         # the value fits before the > button
    drawn.clear()
    window.change_run_seed(1)
    window.draw()
    assert "1" in [string for string, _, _ in drawn] and "0 (seeded)" not in [string for string, _, _ in drawn]


def test_the_training_status_names_the_run(window, circle_track):
    window.run_seed = 1
    window.process = object()                           # a training process, never started
    window.start_playback(ga_report(0), circle_track)
    assert window.status == "Training GA, run 1."


def test_seeded_es_without_the_dagger_baseline_shows_a_message(window, tmp_path):
    window.train_runs_dir = tmp_path
    window.champions_dir = tmp_path                      # the old check never finds a controller: no real process
    window.select_approach("es_seeded")
    window.start_training()
    assert window.process is None and "notebook 03" in window.status and len(window.status) <= 56


def test_seeded_es_starts_with_the_dagger_baseline(window, tmp_path, monkeypatch):
    created = {}

    class FakeProcess:
        def __init__(self, target, args, daemon):
            created.update(args=args)
            self.pid = 0

        def start(self):
            pass

        def is_alive(self):
            return False

    class FakeContext:
        Process = FakeProcess

        def Queue(self):
            return FakeQueue()

        def Event(self):
            return threading.Event()

    monkeypatch.setattr(viewer.mp, "get_context", lambda method: FakeContext())
    stage = paths.stage_path("baseline", config.RUN_SEEDS[0], "dagger", tmp_path)
    stage.parent.mkdir(parents=True)
    stage.write_bytes(b"")
    window.train_runs_dir = tmp_path
    window.select_approach("es_seeded")
    window.start_training()
    assert created["args"][:2] == ("es_seeded", config.ES_POP)


def test_final_report_of_a_finished_trainer_is_shown(window):
    class Done:
        exitcode = 0

        def is_alive(self):
            return False

    class LateQueue(FakeQueue):
        """The trainer's last report becomes readable only after its process has ended."""

        def __init__(self):
            super().__init__()
            self.calls = 0

        def get_nowait(self):
            self.calls += 1
            if self.calls == 2:
                self.items.append(ga_report(5))
            return super().get_nowait()

    window.process, window.reports = Done(), LateQueue()
    window.poll_reports()
    assert window.latest["generation"] == 5 and "finished" in window.status.lower()


def test_one_x_runs_at_twice_real_time(window):
    window.receive(ga_report())
    window.update(0.0)
    window.update(0.1)
    assert abs(window.playback.cars.time - config.PLAYBACK_BASE * 0.1) < 1e-9


def test_tracks_are_built_in_the_background(monkeypatch, circle_track):
    release = threading.Event()

    def slow_track(seed, level):
        release.wait(5)
        return circle_track

    monkeypatch.setattr(viewer, "get_track", slow_track)
    pygame.init()
    app = viewer.App(pygame.display.set_mode(config.WINDOW_SIZE))
    try:
        app.receive(ga_report())
        app.update(1 / 60)
        app.draw()
        assert app.playback is None and "Map:" in app.map_text()     # the window keeps drawing meanwhile
        release.set()
        for _ in range(200):
            app.update(1 / 60)
            if app.playback is not None:
                break
            time.sleep(0.01)
        assert app.playback is not None
    finally:
        release.set()
        pygame.quit()


def test_map_text_names_the_track(window, circle_track):
    assert window.map_text() == "Map: shown with the first generation"      # GA before training
    window.receive(ga_report())
    window.update(1 / 60)
    assert window.map_text() == (f"Map: seed {circle_track.seed}, level {circle_track.level}, "
                                 f"road {circle_track.width:.1f} m")


def test_imitation_preview_uses_the_difficulty(monkeypatch, window, circle_track):
    calls = []
    monkeypatch.setattr(viewer, "get_track", lambda seed, level: calls.append((seed, level)) or circle_track)
    window.select_approach("baseline")
    window.set_difficulty(3)
    window.draw()
    assert (config.KT_VIEW_SEED, 3) in calls


def test_training_panel_fits_the_window(window):
    for approach in config.APPROACHES:
        window.approach = approach
        window.draw()
        assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons), approach
        assert window.slider.bottom <= window.panel.bottom - 30
    window.receive(ga_report())
    window.update(1 / 60)
    for approach in config.APPROACHES:
        window.approach = approach
        window.draw()
        assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons), approach
        assert window.slider.bottom <= window.panel.bottom - 30, approach


def test_a_failed_track_build_is_reported_and_retried(window, monkeypatch, circle_track):
    calls = []

    def flaky(seed, level):
        calls.append(seed)
        if len(calls) == 1:
            raise RuntimeError("no drivable track")
        return circle_track

    monkeypatch.setattr(viewer, "get_track", flaky)
    assert window.request_track(0, 1) is None and "seed 0" in window.status
    assert window.request_track(0, 1) is circle_track


def test_a_failed_background_build_is_reported_and_retried(monkeypatch, circle_track):
    calls = []

    def flaky(seed, level):
        calls.append(seed)
        if len(calls) == 1:
            raise RuntimeError("no drivable track")
        return circle_track

    monkeypatch.setattr(viewer, "get_track", flaky)
    pygame.init()
    app = viewer.App(pygame.display.set_mode(config.WINDOW_SIZE))
    try:
        track = app.request_track(0, 1)
        for _ in range(200):
            if not app.building:
                break
            time.sleep(0.01)
        assert track is None and "seed 0" in app.status and not app.building
        track = app.request_track(0, 1)
        for _ in range(200):
            if app.request_track(0, 1) is not None:
                break
            time.sleep(0.01)
        assert app.request_track(0, 1) is circle_track
    finally:
        pygame.quit()


def test_a_light_color_swatch_is_outlined(window):
    rect = pygame.Rect(10, 10, 10, 10)
    for approach, outlined in (("neat", True), ("es_seeded", False)):
        window.screen.fill(viewer.PANEL_BG)
        window.swatch(approach, rect)
        assert (tuple(window.screen.get_at(rect.topleft))[:3] == viewer.INK) == outlined
        assert tuple(window.screen.get_at(rect.center))[:3] == viewer.hex_to_rgb(config.COLORS[approach])


# ----- network window -----
class NetProcess:
    """Stands in for the network window's process."""
    instances = []

    def __init__(self, target, args, daemon):
        self.target, self.args, self.daemon = target, args, daemon
        self.alive, self.started, self.joined, self.terminated = False, False, False, False
        NetProcess.instances.append(self)

    def start(self):
        self.started = self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.joined = True
        if self.args[0].items and self.args[0].items[-1] is None:      # the window obeys the quit message
            self.alive = False

    def terminate(self):
        self.terminated, self.alive = True, False


class NetQueue(FakeQueue):
    def put(self, item):
        self.items.append(item)


@pytest.fixture
def net(window, monkeypatch):
    NetProcess.instances = []

    class Context:
        Process = NetProcess

        def Queue(self):
            return NetQueue()

    monkeypatch.setattr(viewer.mp, "get_context", lambda method: Context())
    return window


def sent(window):
    return [m for m in window.net_queue.items if m is not None]


def test_the_network_button_opens_and_closes_one_window_process(net):
    net.toggle_network_window()
    process = NetProcess.instances[0]
    assert process.started and process.daemon is False and process.target is viewer.netwindow.window_process
    net.toggle_network_window()
    assert len(NetProcess.instances) == 1 and process.joined and not process.terminated
    assert net.net_process is None and net.net_queue is None


def test_a_window_that_ignores_quit_is_terminated(net):
    net.toggle_network_window()
    process = NetProcess.instances[0]
    process.join = lambda timeout=None: None
    net.toggle_network_window()
    assert process.terminated and net.net_process is None


def test_a_closed_window_resets_the_state(net):
    net.toggle_network_window()
    NetProcess.instances[0].alive = False                       # the user closed it
    net.update(1 / 60)
    assert net.net_process is None and net.net_queue is None and net.net_sent is None
    net.toggle_network_window()
    assert len(NetProcess.instances) == 2                       # the button opens it again


def test_a_new_report_sends_exactly_one_message(net):
    net.toggle_network_window()
    net.update(1 / 60)
    assert sent(net) == []                                      # nothing to show yet
    report = ga_report(0)
    net.receive(report)
    net.update(1 / 60)
    net.update(1 / 60)
    assert len(sent(net)) == 1
    best = int(np.argmax(report["fitness"]))
    assert sent(net)[0]["graph"] == viewer.netwindow.netfig.network_graph(report["population"].driver(best))
    assert sent(net)[0]["title"] == (f"GA · run {net.run_seed} · generation 0 · "
                                     f"{report['population'].driver(best).n_params:,} parameters · 50 m · ×1.00")
    net.receive(ga_report(1))
    net.update(1 / 60)
    assert len(sent(net)) == 2


def test_the_neat_window_shows_the_evolved_genome(net):
    net.approach = "neat"
    net.toggle_network_window()
    net.receive(neat_report(3))
    net.update(1 / 60)
    assert "generation 3" in sent(net)[0]["title"] and sent(net)[0]["title"].startswith(config.LABELS["neat"])


def test_imitation_shows_the_latest_member(net):
    net.approach = "baseline"
    members = [nets.from_flat(row) for row in nets.init_population(np.random.default_rng(1), 3)]
    net.toggle_network_window()
    net.receive({"approach": "baseline", "generation": 2, "level": 1, "seeds": [0], "nets": members,
                 "fitness": np.array([0.1, 0.2, 0.3]), "best": 0.3, "mean": 0.2, "worst": 0.1, "leveled_up": False})
    net.update(1 / 60)
    message = sent(net)[0]
    assert message["graph"] == viewer.netwindow.netfig.network_graph(members[-1]) and " epoch 2 " in message["title"]


def test_an_unchanged_report_builds_no_driver_and_the_title_uses_the_report_seed(net, monkeypatch):
    report = ga_report(0, seed=7)
    calls = []
    original = report["population"].driver
    monkeypatch.setattr(report["population"], "driver", lambda i: calls.append(i) or original(i))
    net.toggle_network_window()
    net.receive(report)
    net.run_seed = 99                                           # the selector changed after the run started
    for _ in range(5):
        net.update(1 / 60)
    assert len(calls) == 1 and len(sent(net)) == 1 and " · run 7 · " in sent(net)[0]["title"]


def test_network_window_titles_name_the_setup(net, tmp_path):
    rng = np.random.default_rng(3)
    cars = nets.DensePopulation(nets.init_population(rng, 4), setup=np.array([[120.0, 0.5]] * 4))
    net.toggle_network_window()
    net.receive(ga_report(0, n=4, population_=cars))
    net.update(1 / 60)
    assert sent(net)[-1]["title"].endswith(" parameters · 120 m · ×0.50")
    save_race_champions(tmp_path)
    net.champions_dir = tmp_path
    net.set_mode("race")
    net.update(1 / 60)
    assert sent(net)[-1]["title"].endswith(" parameters · 50 m · ×1.00")


def test_a_reopened_window_gets_the_current_network(net):
    net.receive(ga_report(0))
    net.toggle_network_window()
    net.update(1 / 60)
    net.toggle_network_window()
    net.toggle_network_window()
    net.update(1 / 60)
    assert len(sent(net)) == 1


def three_car_race(window, tmp_path):
    save_race_champions(tmp_path, approaches=(("es", 0), ("ga", 0), ("neat", 0)))
    window.champions_dir = tmp_path
    window.set_mode("race")
    assert window.lineup == [("es", 0), ("ga", 0), ("neat", 0)]


def test_the_race_cycle_button_walks_the_lineup_and_wraps(net, tmp_path):
    three_car_race(net, tmp_path)
    net.toggle_network_window()
    net.update(1 / 60)
    assert len(sent(net)) == 1 and sent(net)[0]["title"].startswith("ES (seeded) · ")
    assert "generation" not in sent(net)[0]["title"] and "seed 0" not in sent(net)[0]["title"]
    net.update(1 / 60)
    assert len(sent(net)) == 1                                  # unchanged: nothing sent
    seen = []
    for _ in range(4):
        net.cycle_network_car()
        seen.append(net.net_choice)
        net.update(1 / 60)
    assert seen == [1, 2, 0, 1] and len(sent(net)) == 5
    assert sent(net)[-1]["title"].startswith("GA (seeded) · ")


def test_the_selection_falls_back_to_the_first_car(net, tmp_path):
    three_car_race(net, tmp_path)
    net.net_choice = 2                                          # neat
    net.toggle_race("neat")
    assert net.lineup == [("es", 0), ("ga", 0)] and net.net_choice == 0
    net.net_choice = 5
    assert net.selected_car() == ("es", 0)


def test_the_selection_stays_on_its_car_when_others_leave(net, tmp_path):
    three_car_race(net, tmp_path)
    net.net_choice = 1                                          # ga
    net.toggle_race("es")
    assert net.lineup == [("ga", 0), ("neat", 0)] and net.selected_car() == ("ga", 0) and net.net_choice == 0


def test_the_network_buttons_fit_both_panels(net, tmp_path):
    for approach in config.APPROACHES:
        net.approach = approach
        net.draw()
        assert any(rect.height and action == net.toggle_network_window for rect, action in net.buttons)
        assert all(rect.bottom <= net.panel.bottom - 30 for rect, _ in net.buttons), approach
    save_race_champions(tmp_path, approaches=[(approach, 0) for approach in config.APPROACHES])
    net.champions_dir = tmp_path
    net.set_mode("race")
    net.update(1 / 60)
    net.start_race()
    net.update(1 / 60)
    net.buttons = []
    net.draw()
    actions = [action for _, action in net.buttons]
    assert net.toggle_network_window in actions and net.cycle_network_car in actions
    assert all(rect.bottom <= net.panel.bottom - 30 for rect, _ in net.buttons)
    assert net.lineup[0] == ("baseline", 0) and net.network_car_label() == "Network of: Baseline (seeded) ▸"


def test_the_network_window_names_race_cars_as_the_leaderboard_does(net, tmp_path):
    """spec §10.3: one name per car in the leaderboard, the network selector and the window title."""
    assert [viewer.car_name("ga", 0), viewer.car_name("ga", 1), viewer.car_name("ga", 2),
            viewer.car_name("ga", 1, app=True)] == ["GA (seeded)", "GA r1", "GA r2", "GA (app) s1"]
    assert viewer.leaderboard_line(5, "ga", 1, 0, float("nan"), float("nan"), "ready").startswith("5. GA r1 lap 0")
    app_race_window(net, tmp_path)                          # reported: es 0, ga 0, ga 1; app cars: ga 2, neat 0
    net.cycle_app_choice("ga")
    net.toggle_network_window()
    names = []
    for _ in net.race_cars():
        net.update(1 / 60)
        names.append((net.network_car_label(), sent(net)[-1]["title"].split(" · ")[0]))
        net.cycle_network_car()
    assert names == [("Network of: ES (seeded) ▸", "ES (seeded)"), ("Network of: GA (seeded) ▸", "GA (seeded)"),
                     ("Network of: GA r1 ▸", "GA r1"), ("Network of: GA (app) s2 ▸", "GA (app) s2")]


# ----- saved application runs (spec §10.6) -----
def fake_trainer_context(monkeypatch, created):
    """Training processes that record their arguments and never start (tests never start real processes)."""
    class FakeProcess:
        def __init__(self, target, args, daemon):
            created.append(args)
            self.pid = 0

        def start(self):
            pass

        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    class FakeContext:
        Process = FakeProcess

        def Queue(self):
            return FakeQueue()

        def Event(self):
            return threading.Event()

    monkeypatch.setattr(viewer.mp, "get_context", lambda method: FakeContext())


def save_app_run(window, approach="ga", seed=0):
    """A saved viewer run (checkpoint and a two-generation log) and its app car."""
    run = window.app_runs_dir / approach / f"seed{seed}"
    run.mkdir(parents=True)
    (run / checkpoint.CHECKPOINT_NAME).write_bytes(b"saved")
    (run / "log.csv").write_text("generation,min_share,event\n0,0.5,\n1,0.9,level_up\n")
    save_race_champions(window.app_champions_dir, approaches=[(approach, seed)])
    return run


def test_app_runs_and_cars_of_tests_live_in_a_temporary_folder(window):
    real = config.RESULTS_DIR / "app_runs"
    assert config.APP_RUNS_DIR != real and window.app_runs_dir == config.APP_RUNS_DIR
    assert window.app_champions_dir == config.APP_CHAMPIONS_DIR == config.APP_RUNS_DIR / "champions"


def test_trainer_main_resumes_a_saved_evolutionary_run(monkeypatch, tmp_path):
    from racing import ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    saved = tmp_path / "ga" / "seed0" / "checkpoint.pkl"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"an earlier viewer run")
    seen = []
    monkeypatch.setattr(ga, "run_ga", lambda **kwargs: seen.append((saved.exists(), kwargs["out_dir"])))
    running = threading.Event()
    running.set()
    assert viewer.trainer_main("ga", 30, 0, 1, queue.Queue(), running) == 0
    assert seen == [(True, tmp_path / "ga" / "seed0")]


def test_trainer_main_still_starts_imitation_afresh(monkeypatch, tmp_path):
    fake_imitation(monkeypatch, tmp_path)
    old = tmp_path / "app_runs" / "baseline" / "seed0" / "imitation.keras"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"an earlier viewer run")
    run_baseline_trainer()
    assert not old.exists()


def test_the_best_car_of_each_generation_is_saved_as_an_app_car(monkeypatch, tmp_path):
    from racing import ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    sent = []

    def run(**kwargs):
        for generation in (0, 1):
            sent.append(ga_report(generation))
            kwargs["on_generation"](sent[-1])

    monkeypatch.setattr(ga, "run_ga", run)
    running = threading.Event()
    running.set()
    viewer.trainer_main("ga", 30, 0, 1, queue.Queue(), running)
    last = sent[-1]
    best = int(np.argmax(last["fitness"]))
    _, driver = nets.load_champion("ga", 0, config.APP_CHAMPIONS_DIR)
    obs = np.random.default_rng(5).random((40, config.N_INPUTS))
    assert np.array_equal(driver.act(obs), last["population"].driver(best).act(obs))
    assert np.array_equal(driver.setup, last["population"].setup[best])
    assert not (config.APP_CHAMPIONS_DIR / ".staging").exists()      # files moved into place, staging removed


def test_a_neat_app_car_keeps_its_genome(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    report = neat_report(0)
    monkeypatch.setattr(neat, "run_neat", lambda **kwargs: kwargs["on_generation"](report))
    running = threading.Event()
    running.set()
    viewer.trainer_main("neat", 20, 0, 1, queue.Queue(), running)
    best = int(np.argmax(report["fitness"]))
    _, driver = nets.load_champion("neat", 0, config.APP_CHAMPIONS_DIR)
    assert isinstance(driver, neat.NeatNet)
    obs = np.random.default_rng(6).random((40, config.N_INPUTS))
    assert np.array_equal(driver.act(obs), report["population"].driver(best).act(obs))


def test_the_best_imitation_network_so_far_is_the_app_car(monkeypatch, tmp_path):
    from racing import imitation
    fake_imitation(monkeypatch, tmp_path)
    rng = np.random.default_rng(7)
    models = [nets.to_keras(nets.init_population(rng, 1)[0]) for _ in range(3)]

    def train(data, on_epoch=None, **kwargs):
        for epoch, (model, accuracy) in enumerate(zip(models, (0.4, 0.7, 0.6))):
            on_epoch(epoch, model, accuracy)

    monkeypatch.setattr(imitation, "train_baseline", train)
    run_baseline_trainer()
    _, driver = nets.load_champion("baseline", 0, config.APP_CHAMPIONS_DIR)
    obs = rng.random((40, config.N_INPUTS))
    assert np.array_equal(driver.act(obs), nets.from_keras(models[1]).act(obs))     # epoch 1 had the best accuracy


@pytest.mark.parametrize("error", [checkpoint.SettingsMismatch("GA_POP: recorded 1000, now 900"),
                                   checkpoint.UnreadableRecord("summary.json cannot be read"),
                                   EOFError("Ran out of input")])
def test_an_unusable_saved_run_ends_the_trainer_with_code_3(monkeypatch, tmp_path, capsys, error):
    from racing import ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)

    def run(**kwargs):
        raise error

    monkeypatch.setattr(ga, "run_ga", run)
    running = threading.Event()
    running.set()
    reports = queue.Queue()
    assert viewer.trainer_main("ga", 30, 0, 1, reports, running) == viewer.UNUSABLE_RUN == 3
    assert reports.empty() and "cannot be resumed" in capsys.readouterr().err


def test_the_trainer_process_exits_with_the_trainer_code(monkeypatch):
    monkeypatch.setattr(viewer.os, "setpgrp", lambda: None)                 # never changes the test's process group
    monkeypatch.setattr(viewer, "watch_parent", lambda parent, kill: None)  # the watcher signals nothing
    monkeypatch.setattr(viewer, "trainer_main", lambda *args: viewer.UNUSABLE_RUN)
    with pytest.raises(SystemExit) as stop:
        viewer.trainer_process("ga", 30, 0, 1, queue.Queue(), threading.Event())
    assert stop.value.code == 3
    monkeypatch.setattr(viewer, "trainer_main", lambda *args: 0)
    viewer.trainer_process("ga", 30, 0, 1, queue.Queue(), threading.Event())          # returns normally


def test_the_panel_asks_for_new_run_when_a_saved_run_cannot_resume(window):
    class Unusable:
        exitcode = 3

        def is_alive(self):
            return False

    window.process, window.reports = Unusable(), FakeQueue()
    window.poll_reports()
    window.draw()
    assert "New run" in window.status and len(window.status) <= 56 and window.process is None


def test_a_finished_run_says_new_run_starts_over(window):
    class Done:
        exitcode = 0

        def is_alive(self):
            return False

    window.process, window.reports = Done(), FakeQueue()
    window.poll_reports()
    assert "finished" in window.status.lower() and "New run" in window.status


def test_new_run_deletes_the_saved_run_and_its_app_car_then_starts(window, monkeypatch):
    created = []
    fake_trainer_context(monkeypatch, created)
    run = save_app_run(window)
    window.new_run()
    assert not run.exists()
    assert not paths.champion_path("ga", 0, champion_dir=window.app_champions_dir).exists()
    assert len(created) == 1 and created[0][:3] == ("ga", config.GA_POP, 0)
    assert created[0][6:] == (window.app_runs_dir, window.app_champions_dir, window.acks)
    assert window.history == []
    window.new_run()                                    # ignored while training runs
    assert len(created) == 1


def test_play_resumes_a_saved_run_and_continues_its_plot(window, monkeypatch):
    created = []
    fake_trainer_context(monkeypatch, created)
    run = save_app_run(window)
    assert window.has_saved_run()
    window.toggle_play()
    assert run.exists() and len(created) == 1
    assert [(g, s, marked) for g, s, _, marked in window.history] == [(0, 0.5, False), (1, 0.9, True)]


def test_imitation_approaches_have_no_saved_run(window):
    save_app_run(window, approach="baseline")
    window.select_approach("baseline")
    assert not window.has_saved_run() and window.saved_history() == []


def test_an_idle_panel_mentions_a_saved_run(window, monkeypatch):
    save_app_run(window)
    drawn, text = [], window.text

    def recording_text(string, x, y, font=None, color=viewer.INK):
        drawn.append(string)
        return text(string, x, y, font, color)

    monkeypatch.setattr(window, "text", recording_text)
    window.draw()
    assert any(string.startswith("Saved run found") for string in drawn)


def test_the_new_run_button_is_offered_only_while_idle(window, monkeypatch):
    window.draw()
    assert any(action == window.new_run for _, action in window.buttons)
    fake_trainer_context(monkeypatch, [])
    window.toggle_play()
    window.draw()
    assert not any(action == window.new_run for _, action in window.buttons)


def app_race_window(window, tmp_path):
    save_race_champions(tmp_path)                                         # reported: es 0, ga 0, ga 1
    save_race_champions(window.app_champions_dir, approaches=[("ga", 2), ("neat", 0)])
    window.champions_dir = tmp_path
    window.race_runs = "all"
    window.set_mode("race")
    return window


def test_app_cars_form_their_own_group_switched_off_by_default(window, tmp_path):
    app_race_window(window, tmp_path)
    assert window.race_available == {"es": [0], "ga": [0, 1]}
    assert window.app_available == {"ga": [2], "neat": [0]}
    assert window.app_lineup == [] and window.app_choice_label("ga") == "off"
    window.cycle_app_choice("ga")
    assert window.app_choice_label("ga") == "seed 2"
    assert window.race_cars() == window.lineup + [("ga", 2, "app")]
    window.update(1 / 60)
    window.start_race()
    window.update(1 / 60)
    assert window.race is not None and len(window.race.cars.x) == len(window.race_cars())
    assert nets.champion_seeds(window.champions_dir) == {"es": [0], "ga": [0, 1]}     # reported set untouched


def test_the_race_group_button_switches_the_controller_list(window, tmp_path, monkeypatch):
    app_race_window(window, tmp_path)
    drawn, text = [], window.text

    def recording_text(string, x, y, font=None, color=viewer.INK):
        drawn.append(string)
        return text(string, x, y, font, color)

    monkeypatch.setattr(window, "text", recording_text)
    buttons = recording_buttons(window, monkeypatch)
    window.draw()
    assert "Controllers: reported ▸" in window.race_group_label() and ("ES", True) in [
        (label, on) for label, on, _ in method_switches(buttons)]
    window.cycle_race_group()
    drawn.clear()
    buttons.clear()
    window.draw()
    assert window.race_group == "app" and "NEAT (app)" in drawn and not method_switches(buttons)
    window.cycle_race_group()
    assert window.race_group == "final"


def test_without_app_cars_the_race_panel_is_unchanged(window, tmp_path):
    race_window(window, tmp_path)
    window.cycle_race_group()                                   # nothing to switch to
    assert window.app_available == {} and window.race_group == "final"


def test_the_evaluation_never_reads_app_cars():
    import inspect
    from racing import evaluate
    source = inspect.getsource(evaluate)
    assert "APP_CHAMPIONS_DIR" not in source and "APP_RUNS_DIR" not in source and "app_runs" not in source


def test_app_cars_are_labelled_in_the_network_selector_and_leaderboard(net, tmp_path):
    app_race_window(net, tmp_path)
    net.cycle_app_choice("neat")
    assert net.race_cars()[-1] == ("neat", 0, "app")
    while net.selected_car() != ("neat", 0, "app"):
        net.cycle_network_car()
    assert net.network_car_label() == "Network of: NEAT (app) s0 ▸"
    assert viewer.leaderboard_line(3, "neat", 0, 1, 31.0, 30.5, "racing", app=True).startswith("3. NEAT (app) s0 lap 1")
    assert viewer.leaderboard_line(2, "ga", 1, 0, float("nan"), float("nan"), "ready") == "2. GA r1 lap 0 –/– ready"


def test_a_failed_app_car_is_dropped_with_a_message(window, tmp_path):
    app_race_window(window, tmp_path)
    window.cycle_app_choice("ga")
    paths.champion_path("ga", 2, champion_dir=window.app_champions_dir).write_bytes(b"not a model")
    window.drivers = {}
    window.update(1 / 60)
    assert ("ga", 2, "app") not in window.race_cars() and "GA (app) run 2" in window.status


@pytest.mark.parametrize("group", ["final", "app"])
def test_a_race_of_every_method_and_every_app_car_fits_the_panel(window, tmp_path, monkeypatch, group):
    every = [(approach, 0) for approach in config.APPROACHES]
    save_race_champions(tmp_path, approaches=every)
    save_race_champions(window.app_champions_dir, approaches=every)
    window.champions_dir = tmp_path
    window.set_mode("race")
    for approach in config.APPROACHES:
        window.cycle_app_choice(approach)
    if group == "app":
        window.cycle_race_group()
    window.update(1 / 60)
    window.start_race()
    window.update(1 / 60)
    assert len(window.race_cars()) == 2 * len(config.APPROACHES)
    drawn, text = [], window.text

    def recording_text(string, x, y, font=None, color=viewer.INK):
        drawn.append((string, y, font or window.font))
        return text(string, x, y, font, color)

    monkeypatch.setattr(window, "text", recording_text)
    window.buttons = []
    window.draw_race_panel(window.panel.left + 16, 58, window.panel.width - 32)
    for string, y, font in drawn:
        assert y + font.get_linesize() <= window.panel.bottom - 30, string
    assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons)
    rows = [string for string, _, _ in drawn if string[:1].isdigit() and ". " in string]
    assert rows and (len(rows) == 16 or any(string.startswith("… ") for string, _, _ in drawn))


# ----- notebook runs in the training view (read-only replay of a reported run) -----
def notebook_view(window, tmp_path):
    window.runs_dir = tmp_path
    window.toggle_source()
    assert window.source == "notebook"
    return window


def save_replay(window, report, approach="ga", seed=0):
    path = paths.replay_path(approach, seed, window.runs_dir)
    checkpoint.save(path, report)
    return path


def bump(path, seconds):
    """Moves a file's modification time forward, so the view sees it as rewritten."""
    stamp = path.stat().st_mtime_ns + int(seconds * 1e9)
    os.utime(path, ns=(stamp, stamp))


def test_a_notebook_run_replays_its_latest_generation(window, tmp_path, monkeypatch):
    notebook_view(window, tmp_path)
    monkeypatch.setattr(window, "start_training", lambda: pytest.fail("the notebook view never trains"))
    save_replay(window, ga_report(7))
    window.update(1 / 60)
    assert window.latest["generation"] == 7 and window.playback is not None and window.playback_generation == 7
    for _ in range(30):
        window.update(1 / 60)
        window.draw()
    assert window.playback.cars.time > 0


def test_a_notebook_run_can_show_just_its_best_car(window, tmp_path):
    notebook_view(window, tmp_path)
    report = ga_report(3)
    save_replay(window, report)
    window.set_show("best")
    window.shown = 1
    window.update(1 / 60)
    assert list(window.indices) == [int(np.argmax(report["fitness"]))] and len(window.playback.cars.x) == 1


def test_a_notebook_replay_follows_the_run(window, tmp_path):
    notebook_view(window, tmp_path)
    now = [0.0]
    window.clock = lambda: now[0]
    path = save_replay(window, ga_report(1))
    window.update(1 / 60)
    first = window.playback
    first.cars.done[:] = True
    window.update(1 / 60)                                            # nothing newer: the same generation again
    assert window.playback is not first and window.playback_generation == 1
    save_replay(window, ga_report(2))
    bump(path, 1)
    window.playback.cars.done[:] = True
    window.update(1 / 60)                                            # looked less than a poll interval ago
    assert window.playback_generation == 1
    now[0] += config.REPLAY_POLL_SECONDS
    window.playback.cars.done[:] = True
    window.update(1 / 60)
    assert window.playback_generation == 2 and window.latest["generation"] == 2


def test_the_notebook_plot_follows_the_runs_log(window, tmp_path):
    notebook_view(window, tmp_path)
    path = save_replay(window, ga_report(2))
    (path.parent / "log.csv").write_text("generation,min_share,event\n0,0.5,\n1,0.7,level_up\n2,0.8,\n")
    window.update(1 / 60)
    assert [h[0] for h in window.history] == [0, 1, 2] and [h[3] for h in window.history] == [False, True, False]
    window.draw()


def test_the_notebook_view_names_what_it_cannot_replay(window, tmp_path):
    notebook_view(window, tmp_path)
    now = [0.0]
    window.clock = lambda: now[0]
    messages = []

    def look():
        now[0] += config.REPLAY_POLL_SECONDS
        window.update(1 / 60)
        window.draw()
        messages.append(window.status)

    look()
    assert "Nothing to replay" in window.status and window.latest is None
    run = paths.run_dir("ga", 0, tmp_path)
    run.mkdir(parents=True)
    (run / checkpoint.CHECKPOINT_NAME).write_bytes(b"older run")
    look()
    assert "predates replays" in window.status
    path = paths.replay_path("ga", 0, tmp_path)
    path.write_bytes(b"cut short")
    look()
    assert "unreadable" in window.status and window.latest is None
    save_replay(window, ga_report(4))
    look()
    assert window.latest["generation"] == 4
    path.write_bytes(b"cut short again")
    bump(path, 5)
    window.playback.cars.done[:] = True
    look()
    assert "unreadable" in window.status and window.latest["generation"] == 4      # the current replay is kept
    assert window.playback is not None
    assert all(len(message) <= 56 for message in messages)


def test_the_notebook_view_names_a_run_of_an_earlier_revision(window, tmp_path):
    """A replay saved by revision 3.1 holds networks of 10 inputs: it is named, never driven (spec §10.2)."""
    notebook_view(window, tmp_path)
    for approach, hidden in (("ga", config.BASELINE_HIDDEN), ("kt_random", (16, 8))):
        window.select_approach(approach)
        weights = np.zeros((30, nets.n_params(hidden) - 2 * hidden[0]))       # two inputs fewer
        save_replay(window, ga_report(5, approach=approach, population_=nets.DensePopulation(weights, hidden)),
                    approach=approach)
        window.update(1 / 60)
        window.draw()
        assert "older revision, 10 inputs" in window.status and len(window.status) <= 56
        assert window.latest is None and window.playback is None


# ----- archived runs of an earlier revision (results/archive/<name>) -----
def archive_folder(results, name="revision3.1", inputs=10, widths=(16.0, 14.0, 12.0, 10.0)):
    """An archive folder as the runner leaves it: runs, controllers and the settings record of its revision."""
    folder = results / "archive" / name
    (folder / "runs").mkdir(parents=True)
    (folder / "champions").mkdir()
    levels = {str(level): {**spec, "width": width} for (level, spec), width in zip(config.LEVELS.items(), widths)}
    (folder / "run_info.json").write_text(json.dumps({"settings": {"N_INPUTS": inputs, "LEVELS": levels}}))
    return folder


def old_population(n=30, hidden=config.BASELINE_HIDDEN):
    return nets.DensePopulation(nets.init_population(np.random.default_rng(1), n, hidden, n_inputs=10), hidden)


def test_the_run_folders_are_the_current_results_and_each_archive_with_runs(tmp_path):
    folder = archive_folder(tmp_path)
    (tmp_path / "archive" / "revision2-notebooks").mkdir()                  # notebooks only: nothing to show
    assert viewer.run_folders(tmp_path) == [("current", tmp_path), ("revision3.1", folder)]


def test_an_archive_names_its_inputs_and_road_widths(tmp_path):
    folder = archive_folder(tmp_path)
    assert viewer.folder_profile("revision3.1", folder) == (10, {1: 16.0, 2: 14.0, 3: 12.0, 4: 10.0})
    assert viewer.folder_profile("current", tmp_path) == (config.N_INPUTS, None)
    (folder / "run_info.json").write_text("cut short")
    assert viewer.folder_profile("revision3.1", folder) == (config.N_INPUTS, None)


def test_the_notebook_view_replays_an_archived_run_on_its_revisions_maps(window, tmp_path, monkeypatch,
                                                                          circle_track):
    built = []
    monkeypatch.setattr(viewer, "get_track", lambda seed, level, width=None: built.append(width) or circle_track)
    folder = archive_folder(tmp_path)
    window.results_dir = tmp_path
    notebook_view(window, tmp_path)
    window.cycle_folder()
    assert window.runs_dir == folder / "runs" and window.folder_label() == "Runs: revision3.1 ▸"
    save_replay(window, ga_report(5, level=4, population_=old_population()))
    for _ in range(10):
        window.update(1 / 60)
        window.draw()
    assert window.latest["generation"] == 5 and built == [10.0] and window.playback.cars.time > 0
    window.cycle_folder()                                                   # back to the current results
    assert window.runs_dir == tmp_path / "runs" and window.latest is None


def test_the_notebook_view_replays_every_run_of_an_archive(window, tmp_path, monkeypatch, circle_track):
    """An archive may hold a run outside runs 0 to 3 (here run 4): the run control of the notebook view cycles
    through every run in the selected folder, while training started here keeps runs 0 to 3 (spec §10.2)."""
    monkeypatch.setattr(viewer, "get_track", lambda seed, level, width=None: circle_track)
    archive_folder(tmp_path, inputs=config.N_INPUTS)
    window.results_dir = tmp_path
    notebook_view(window, tmp_path)
    window.cycle_folder()
    for run in (0, 1, 4):
        save_replay(window, ga_report(4 + run), seed=run)
    window.update(1 / 60)
    assert window.latest["generation"] == 4
    drawn, text = [], window.text
    monkeypatch.setattr(window, "text", lambda string, *args: drawn.append(string) or text(string, *args))
    window.draw()
    assert "0" in drawn and "0 (seeded)" not in drawn                  # every run of an archive was seeded
    for _ in range(4):
        window.change_run_seed(1)
    window.update(1 / 60)
    assert window.run_seed == 4 and window.latest["generation"] == 8
    assert window.status == "Replaying the notebook run GA run 4."
    window.change_run_seed(1)
    assert window.run_seed == 0
    window.change_run_seed(-1)
    assert window.run_seed == 4
    window.cycle_folder()                                   # the current results hold no run 4
    window.update(1 / 60)
    assert window.status == "Nothing to replay yet for GA run 4."
    window.change_run_seed(-1)
    assert window.run_seed == 3
    window.cycle_folder()                                   # the archive again
    window.change_run_seed(1)
    assert window.run_seed == 4
    window.toggle_source()                                  # training here: runs 0 to 3 only
    assert window.source == "app" and window.run_seed == config.RUN_SEEDS[0]
    for _ in range(4):
        window.change_run_seed(1)
    assert window.run_seed == config.RUN_SEEDS[0]


def test_an_archived_neat_run_of_other_inputs_is_named(window, tmp_path):
    archive_folder(tmp_path)
    window.results_dir = tmp_path
    notebook_view(window, tmp_path)
    window.cycle_folder()
    window.select_approach("neat")
    save_replay(window, neat_report(3), approach="neat")
    window.update(1 / 60)
    assert "older revision, 10 inputs" in window.status and window.latest is None


def test_the_race_view_races_an_archives_controllers_on_its_maps(window, tmp_path, monkeypatch, circle_track):
    built = []
    monkeypatch.setattr(viewer, "get_track",
                        lambda seed, level, width=None: built.append((level, width)) or circle_track)
    folder = archive_folder(tmp_path)
    population.save_dense_champion("ga", 0, old_population(1), 0, folder / "champions")
    save_race_champions(folder / "champions", approaches=[("neat", 0)])     # NEAT of other inputs: left out
    window.results_dir = tmp_path
    window.set_mode("race")
    window.cycle_folder()
    assert window.champions_dir == folder / "champions" and window.race_available == {"ga": [0], "neat": [0]}
    window.start_race()
    for _ in range(10):
        window.update(1 / 60)
        window.draw()
    assert window.race_cars() == [("ga", 0)] and window.race.cars.time > 0
    assert (config.MAX_LEVEL, 10.0) in built


def test_app_training_ignores_the_shown_folder(window, tmp_path):
    archive_folder(tmp_path)
    window.results_dir = tmp_path
    window.set_mode("race")
    window.cycle_folder()
    window.set_mode("train")
    assert window.source == "app" and window.train_runs_dir == config.RUNS_DIR


def test_notebook_play_and_pause_move_only_the_replay(window, tmp_path, monkeypatch):
    notebook_view(window, tmp_path)
    monkeypatch.setattr(window, "start_training", lambda: pytest.fail("the notebook view never trains"))
    save_replay(window, ga_report(0))
    for _ in range(10):
        window.update(1 / 60)
    window.toggle_play()                                             # Pause
    frozen = window.playback.cars.time
    for _ in range(30):
        window.update(1 / 60)
    assert window.playback.cars.time == frozen
    window.toggle_play()                                             # Play
    for _ in range(10):
        window.update(1 / 60)
    assert window.playback.cars.time > frozen
    window.stop()
    window.new_run()
    assert window.process is None and window.latest is not None
    assert paths.replay_path("ga", 0, tmp_path).exists()


def test_the_notebook_view_writes_nothing(window, tmp_path, monkeypatch):
    notebook_view(window, tmp_path)
    monkeypatch.setattr(window, "start_training", lambda: pytest.fail("the notebook view never trains"))
    path = save_replay(window, ga_report(0))
    (path.parent / "log.csv").write_text("generation,min_share,event\n0,0.5,\n")
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    for _ in range(60):
        window.update(1 / 60)
        window.draw()
    window.toggle_play()
    window.toggle_play()
    window.stop()
    window.new_run()
    window.change_run_seed(1)
    window.update(1 / 60)
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")} == before


def test_the_source_cannot_change_while_training_runs(window):
    class Alive:
        pid = 0

        def is_alive(self):
            return True

    window.process = Alive()
    window.toggle_source()
    assert window.source == "app"


def test_changing_method_or_seed_loads_that_notebook_run(window, tmp_path):
    notebook_view(window, tmp_path)
    save_replay(window, ga_report(1), seed=config.RUN_SEEDS[0])
    save_replay(window, ga_report(5), seed=config.RUN_SEEDS[1])
    save_replay(window, neat_report(9), approach="neat", seed=config.RUN_SEEDS[1])
    window.update(1 / 60)
    assert window.latest["generation"] == 1
    window.change_run_seed(1)
    window.update(1 / 60)
    assert window.latest["generation"] == 5 and window.playback_generation == 5
    window.select_approach("neat")
    window.update(1 / 60)
    assert window.latest["approach"] == "neat" and window.playback_generation == 9
    window.draw()


def test_an_imitation_method_replays_its_speed_push(window, tmp_path):
    notebook_view(window, tmp_path)
    window.select_approach("baseline")
    save_replay(window, ga_report(4, approach="baseline"), approach="baseline")
    assert paths.replay_path("baseline", 0, tmp_path).parent == paths.speed_dir("baseline", 0, tmp_path)
    window.update(1 / 60)
    assert window.latest["generation"] == 4 and window.playback is not None
    window.draw()


def test_switching_back_to_app_training_clears_the_notebook_replay(window, tmp_path):
    notebook_view(window, tmp_path)
    save_replay(window, ga_report(3))
    window.update(1 / 60)
    window.toggle_source()
    assert window.source == "app" and window.latest is None and window.playback is None
    window.update(1 / 60)
    assert window.latest is None                                     # app training shows only its own reports


def test_the_notebook_panel_fits_the_window(window, tmp_path):
    notebook_view(window, tmp_path)
    for approach in config.APPROACHES:
        window.approach = approach
        window.draw()
        assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons), approach
    save_replay(window, neat_report(2), approach="neat")
    window.select_approach("neat")
    window.update(1 / 60)
    window.draw()
    assert window.latest is not None
    assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons)
    assert window.slider.bottom <= window.panel.bottom - 30


def test_a_controller_of_another_input_count_is_left_out_of_the_race(window, tmp_path, monkeypatch):
    race_window(window, tmp_path)

    class EarlierModel:
        input_shape = (None, 10)

    real = viewer.nets.load_champion
    monkeypatch.setattr(viewer.nets, "load_champion",
                        lambda a, s, d=None: (EarlierModel(), real(a, s, d)[1]) if a == "ga" else real(a, s, d))
    window.update(1 / 60)
    window.prepare_race()
    assert window.race_cars() and all(car[0] != "ga" for car in window.race_cars())
    assert "10 inputs" in window.status and window.race is not None


def test_the_viewer_ships_its_font():
    """The viewer's texts are measured against this font, so it travels with the project: a machine without DejaVu
    Sans Condensed would otherwise draw every text about 12% wider (spec §10)."""
    assert config.FONT_PATH.is_file() and config.FONT_PATH.with_name("LICENSE").is_file()
    import pygame
    pygame.font.init()
    font = pygame.font.Font(str(config.FONT_PATH), 13)
    assert font.size("16. KT Hyperband (seeded) lap 3 888.9/1000.0 finished")[0] <= 332 - 4


def test_runs_1_to_3_are_named_r1_to_r3():
    assert viewer.car_name("ga", 0) == "GA (seeded)" and viewer.car_name("ga", 1) == "GA r1"
    assert viewer.car_name("ga", 3) == "GA r3" and viewer.car_name("ga", 2, archived=True) == "GA s2"
    assert viewer.car_name("ga", 1, app=True) == "GA (app) s1"


def test_a_race_of_all_32_runs_shows_the_rows_that_fit_then_a_count(window, tmp_path, monkeypatch):
    """spec §10.3: 8 methods x 4 runs at 1280x800 show as many rows as fit, then '… N more', nothing overlapping;
    the default best (8 cars) shows every row."""
    window.screen = pygame.display.set_mode((1280, 800))
    window.layout()
    save_race_champions(tmp_path, approaches=[(approach, run) for approach in config.APPROACHES for run in range(4)])
    window.champions_dir = tmp_path
    window.set_mode("race")

    def panel(cars):
        window.update(1 / 60)
        window.start_race()
        window.update(1 / 60)
        assert len(window.race.cars.x) == cars
        drawn, text = [], window.text

        def recording_text(string, x, y, font=None, color=viewer.INK):
            drawn.append((string, x, y, font or window.font))
            return text(string, x, y, font, color)

        monkeypatch.setattr(window, "text", recording_text)
        window.buttons = []
        window.draw_race_panel(window.panel.left + 16, 58, window.panel.width - 32)
        for string, x, y, font in drawn:
            assert y + font.get_linesize() <= window.panel.bottom - 30 - 4, string
            assert x + font.size(string)[0] <= window.panel.right - 16, string
        assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons)
        window.stop_race()
        return ([string for string, *_ in drawn if string[:1].isdigit() and ". " in string],
                [string for string, *_ in drawn if string.startswith("… ")])

    rows, more = panel(8)                                                   # best: one run per method
    assert len(rows) == 8 and more == []
    window.cycle_race_runs()                                                # all
    rows, more = panel(32)
    assert 0 < len(rows) < 32 and more == [f"… {32 - len(rows)} more"]
