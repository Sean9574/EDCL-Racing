import queue
import threading

import numpy as np
import pygame
import pytest

from racing import app as viewer
from racing import config, nets


@pytest.fixture
def window(monkeypatch, circle_track):
    monkeypatch.setattr(viewer, "get_track", lambda seed, level: circle_track)
    pygame.init()
    screen = pygame.display.set_mode(config.WINDOW_SIZE)
    instance = viewer.App(screen)
    yield instance
    instance.stop_training()
    pygame.quit()


def ga_report(generation=0, n=30):
    rng = np.random.default_rng(generation)
    fitness = rng.random(n)
    return {"approach": "ga", "generation": generation, "level": 1, "seeds": [0, 1, 2],
            "weights": nets.init_population(rng, n), "hidden": [16], "activation": "relu",
            "fitness": fitness, "finished": np.zeros(n, dtype=bool), "is_kept": np.zeros(n, dtype=bool),
            "kept_finish": float("nan"), "new_finish": 0.0, "step_size": 0.5, "best": float(fitness.max()),
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
    viewer.trainer_main("ga", 30, reports, running)
    viewer.trainer_main("es", 12, reports, running)
    assert calls["ga"]["champion_dir"] is None and calls["ga"]["pop_size"] == 30
    assert calls["es"]["champion_dir"] is None and calls["es"]["lam"] == 12
    assert calls["ga"]["out_dir"] == tmp_path / "ga" and calls["es"]["out_dir"] == tmp_path / "es"
    assert reports.get_nowait()["generation"] == 0 and reports.empty()


def test_pause_and_resume_toggle_the_running_event(window):
    class Alive:
        def is_alive(self):
            return True

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
    trainer = threading.Thread(target=viewer.trainer_main, args=("ga", 30, reports, running), daemon=True)
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
    viewer.trainer_main("baseline", 0, queue.Queue(), running)


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
