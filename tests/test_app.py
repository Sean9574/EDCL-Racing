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
from racing import config, neat, nets, paths, population


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
    """A training report as the evolution engine sends it (revision 3 §9)."""
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
              "evals": n, "elapsed": 1.0, "event": None}
    report.update(extra)
    return report


def neat_report(generation=0, n=20):
    rng, innovations = np.random.default_rng(generation), neat.Innovations()
    genomes = [neat.initial_genome(rng, innovations) for _ in range(n)]
    return ga_report(generation, n, approach="neat", population_=neat.NeatPopulation(genomes), species=3)


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
                  step_size=0.0505, best=1.5843, mean=1.2329, species=20, is_kept=np.ones(1000, dtype=bool))
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


def test_app_plays_a_report(window):
    window.receive(ga_report())
    for _ in range(120):
        window.update(1 / 60)
        window.draw()
    assert window.playback is not None and window.playback_generation == 0


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


def test_trainer_main_uses_app_runs_dir_and_never_saves_champions(monkeypatch, tmp_path):
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
    assert not calls["ga"]["existed"]                              # every Play starts a new run
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


def save_race_champions(directory, best=None, approaches=(("es", 0), ("ga", 0), ("ga", 1))):
    rng = np.random.default_rng(0)
    for approach, seed in approaches:
        if approach == "neat":
            neat.save_neat_champion(approach, seed, neat.initial_genome(rng, neat.Innovations()), directory)
        else:
            population.save_dense_champion(approach, seed, nets.DensePopulation(nets.init_population(rng, 1)), 0,
                                           directory)
    if best is not None:
        (directory / "best_seeds.json").write_text(json.dumps(best))


def race_window(window, tmp_path, best=None):
    save_race_champions(tmp_path, best)
    window.champions_dir = tmp_path
    window.set_mode("race")
    return window


def test_race_lineup_uses_the_best_seeds(window, tmp_path):
    race_window(window, tmp_path, {"ga": 1, "es": 0})
    assert window.lineup == [("es", 0), ("ga", 1)]
    assert window.choice_label("ga") == "best (seed 1)"


def test_race_lineup_falls_back_without_best_seeds(window, tmp_path):
    race_window(window, tmp_path)
    assert window.lineup == [("es", 0), ("ga", 0)]


def test_cycling_a_choice_picks_seeds_and_off(window, tmp_path):
    race_window(window, tmp_path, {"ga": 1})
    window.cycle_choice("ga")
    assert ("ga", 0) in window.lineup and window.choice_label("ga") == "seed 0"
    window.cycle_choice("ga")
    assert ("ga", 1) in window.lineup
    window.cycle_choice("ga")
    assert window.lineup == [("es", 0)] and window.choice_label("ga") == "off"
    window.cycle_choice("ga")
    assert window.lineup == [("es", 0), ("ga", 1)]


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
    window.cycle_choice("ga")
    assert window.race_index == 0 and window.race_level == config.MAX_LEVEL and window.lineup == lineup


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
    assert sorted(r[0] for r in rows) == [1, 2] and sorted(r[1] for r in rows) == [0, 1]
    assert all(rect.bottom <= window.panel.bottom - 30 for rect, _ in window.buttons)


def test_corrupt_best_seeds_falls_back_to_the_lowest_seed(window, tmp_path):
    save_race_champions(tmp_path)
    (tmp_path / "best_seeds.json").write_text('{"ga": ')
    window.champions_dir = tmp_path
    window.set_mode("race")
    assert ("ga", 0) in window.lineup and "best_seeds.json" in window.status


def test_best_seed_without_a_controller_falls_back(window, tmp_path):
    race_window(window, tmp_path, {"ga": 5})
    assert ("ga", 0) in window.lineup


def test_unreadable_controller_is_skipped(window, tmp_path):
    save_race_champions(tmp_path)
    window.champions_dir = tmp_path
    window.set_mode("race")
    paths.champion_path("ga", 0, champion_dir=tmp_path).write_bytes(b"garbage")
    paths.champion_path("ga", 0, ".npy", tmp_path).unlink()
    window.update(1 / 60)
    window.update(1 / 60)
    assert ("ga", 0) not in window.lineup and window.race is not None
    assert window.lineup == [("es", 0)] and "seed 0" in window.status and "GA" in window.status


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
    window.set_mode("race")
    assert window.race_available == {"es": [0], "ga": [0, 1]}
    assert window.lineup == [("es", 0), ("ga", 0)]
    (tmp_path / "best_seeds.json").write_text(json.dumps({"ga": 1}))
    window.set_mode("train")
    window.set_mode("race")
    assert window.lineup == [("es", 0), ("ga", 1)]


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
                        lambda data, **kwargs: calls.append(("baseline", kwargs["resume"], kwargs.get("champion_dir"))))
    monkeypatch.setattr(imitation, "run_tuner",
                        lambda approach, data, **kwargs: calls.append((approach, kwargs["resume"], None)))
    running = threading.Event()
    running.set()
    viewer.trainer_main("baseline", 0, 0, 1, queue.Queue(), running)
    viewer.trainer_main("kt_random", 0, 0, 1, queue.Queue(), running)
    assert calls == [("baseline", False, None), ("kt_random", False, None)]


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
        for status in ("ready", "racing", "finished", "crashed", "stopped"):
            line = viewer.leaderboard_line(len(config.APPROACHES), approach, max(config.RUN_SEEDS),
                                           config.RACE_LAPS, 888.88, 999.99, status)
            assert window.small.size(line)[0] <= width, line
    line = viewer.leaderboard_line(1, "kt_hyperband", 0, 1, 52.31, 51.94, "racing")
    assert line == "1. KT Hyperband s0  lap 1  52.3/51.9  racing"
    assert viewer.leaderboard_line(2, "ga", 1, 0, float("nan"), float("nan"), "ready") == "2. GA s1  lap 0  –/–  ready"


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


def test_seeded_es_without_the_dagger_baseline_shows_a_message(window, tmp_path):
    window.runs_dir = tmp_path
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
    window.runs_dir = tmp_path
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


def test_trainer_main_starts_each_viewer_run_afresh(monkeypatch, tmp_path):
    from racing import ga
    monkeypatch.setattr(config, "APP_RUNS_DIR", tmp_path)
    stale = tmp_path / "ga" / "seed0" / "checkpoint.pkl"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"an earlier viewer run")
    seen = []
    monkeypatch.setattr(ga, "run_ga", lambda **kwargs: seen.append(stale.exists()))
    running = threading.Event()
    running.set()
    viewer.trainer_main("ga", 30, 0, 1, queue.Queue(), running)
    assert seen == [False]
