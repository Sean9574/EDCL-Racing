import os
import queue
import signal
import threading
import time

import numpy as np
import pygame
import pytest

from racing import app as viewer
from racing import config, nets, paths


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


def ga_report(generation=0, n=30):
    rng = np.random.default_rng(generation)
    fitness = rng.random(n)
    return {"approach": "ga", "generation": generation, "level": 1, "seeds": [0, 1, 2],
            "weights": nets.init_population(rng, n), "hidden": [16], "activation": "relu",
            "fitness": fitness, "finished": np.zeros(n, dtype=bool), "is_kept": np.zeros(n, dtype=bool),
            "kept_finish": float("nan"), "new_finish": 0.0, "kept_pace": float("nan"), "new_pace": 0.0, "step_size": 0.5, "best": float(fitness.max()),
            "mean": float(fitness.mean()), "worst": float(fitness.min()), "evals": n, "elapsed": 1.0,
            "leveled_up": False}


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


def test_app_plays_a_report(window):
    window.receive(ga_report())
    for _ in range(120):
        window.update(1 / 60)
        window.draw()
    assert window.playback is not None and window.playback_generation == 0


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


def test_leaderboard_orders_by_distance(circle_track):
    playback = viewer.Playback(circle_track, lambda obs, cars: np.zeros(len(cars.x), dtype=int), 2, laps=3)
    playback.cars.progress[:] = [10.0, 50.0]
    playback.cars.crashed[0] = True
    rows = viewer.leaderboard(playback, ["baseline", "ga"])
    assert [r[1] for r in rows] == ["ga", "baseline"] and rows[1][5] == "crashed"
    assert [r[0] for r in rows] == [1, 2]


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


def test_step_burst_is_capped_and_budget_dropped(window):
    window.mode = "race"
    window.race = viewer.Playback(viewer.get_track(0, 1), lambda o, c: np.full(len(c.x), 8), 1)
    window.update(10.0)
    assert abs(window.race.cars.time - config.MAX_STEPS_PER_FRAME * config.DT) < 1e-9 and window.step_budget == 0.0


def test_trainer_main_uses_app_runs_dir_and_never_saves_champions(monkeypatch, tmp_path):
    from racing import es, ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    calls = {}

    def fake_ga(**kwargs):
        calls["ga"] = kwargs
        kwargs["on_generation"](ga_report(0))

    def fake_es(**kwargs):
        calls["es"] = kwargs

    monkeypatch.setattr(ga, "run_ga", fake_ga)
    monkeypatch.setattr(es, "run_es", fake_es)
    reports, running = queue.Queue(), threading.Event()
    running.set()
    viewer.trainer_main("ga", 30, 0, 1, reports, running)
    viewer.trainer_main("es", 12, 0, 1, reports, running)
    assert calls["ga"]["champion_dir"] is None and calls["ga"]["pop_size"] == 30
    assert calls["es"]["champion_dir"] is None and calls["es"]["lam"] == 12
    assert calls["ga"]["out_dir"] == tmp_path / "ga" / "seed0" and calls["es"]["out_dir"] == tmp_path / "es" / "seed0"
    assert reports.get_nowait()["generation"] == 0 and reports.empty()


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


def stub_champions(window):
    rng = np.random.default_rng(0)
    flat = nets.init_population(rng, 2)
    window.race_names = ["ga", "es"]
    window.race_nets = [nets.from_flat(flat[0]), nets.from_flat(flat[1])]


def test_race_runs_and_draws_a_leaderboard(window):
    stub_champions(window)
    window.set_mode("race")
    window.start_race()
    for _ in range(30):
        window.update(1 / 60)
        window.draw()
    rows = viewer.leaderboard(window.race, window.race_names)
    assert sorted(r[0] for r in rows) == [1, 2] and window.race.cars.time > 0


def test_champions_saved_later_are_picked_up(window, tmp_path, monkeypatch):
    window.champions_dir = tmp_path
    window.set_mode("race")
    assert "No trained controllers" in window.status
    window.set_mode("train")
    assert "No trained controllers" not in window.status
    stub = {"ga": object()}
    monkeypatch.setattr(nets, "load_champions", lambda directory: stub)
    monkeypatch.setattr(nets, "from_keras", lambda model: "net")
    window.set_mode("race")
    assert window.race_names == ["ga"] and window.race_nets == ["net"]


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


def test_trainer_main_passes_seed_level_and_seeded_start(monkeypatch, tmp_path):
    from racing import es, ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    calls = {}
    monkeypatch.setattr(ga, "run_ga", lambda **kwargs: calls.setdefault("ga", kwargs))
    monkeypatch.setattr(es, "run_es", lambda **kwargs: calls.setdefault("es", kwargs))
    monkeypatch.setattr(es, "trained_baseline", lambda seed: f"baseline seed {seed}")
    reports, running = queue.Queue(), threading.Event()
    running.set()
    viewer.trainer_main("ga", 30, 2, 3, reports, running)
    viewer.trainer_main("es_seeded", 12, 1, 2, reports, running)
    assert calls["ga"]["seed"] == 2 and calls["ga"]["start_level"] == 3
    assert calls["es"]["approach"] == "es_seeded" and calls["es"]["initial"] == "baseline seed 1"
    assert calls["es"]["out_dir"] == tmp_path / "es_seeded" / "seed1" and calls["es"]["start_level"] == 2


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


def test_seeded_es_without_baseline_shows_a_message(window, tmp_path):
    window.champions_dir = tmp_path
    window.select_approach("es_seeded")
    window.start_training()
    assert window.process is None and "notebook 02" in window.status


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
    assert window.map_text() == f"Map: seed {circle_track.seed}, level {circle_track.level}"


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
