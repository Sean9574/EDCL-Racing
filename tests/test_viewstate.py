import json
import time

import numpy as np
import pytest

from racing import app as viewer
from racing import config
from racing.viewstate import frame_state, panel_state, scene_state
from test_app import (app_race_window, fake_trainer_context, ga_report, race_window,  # noqa: F401
                      save_app_run)


def training_window(window, n=30):
    window.receive(ga_report(n=n))
    window.update(1 / 60)
    return window


def test_scene_is_downsampled_to_about_3_m_and_json(circle_track):
    s = scene_state(circle_track)
    step = max(1, round(config.SCENE_SPACING / circle_track.spacing))
    assert len(s["left"]) == len(circle_track.left[::step])
    assert len(s["right"]) == len(s["line"]) == len(s["left"])
    assert s["key"] == f"0-1--{circle_track.length:.1f}" and len(s["start"]) == 2
    json.dumps(s)


def test_frame_names_its_scene(window):
    assert frame_state(window) is None                          # a GA run with no generation yet: no track
    training_window(window)
    frame = frame_state(window)
    assert frame["scene"] == scene_state(window.shown_track())["key"]
    assert frame["mode"] == "train" and len(frame["x"]) == len(window.indices)


def test_frame_of_2000_cars_is_small_and_fast(window):
    window.population["ga"] = 2000
    window.shown = 2000
    window.receive(ga_report(n=2000))
    window.update(1 / 60)
    window.update(1 / 60)
    assert len(window.playback.cars.x) == 2000
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        frame = frame_state(window)
        best = min(best, time.perf_counter() - start)
    assert len(json.dumps(frame)) < 150_000 and best < 0.02


def test_race_frame_sends_rays_of_every_live_car_and_training_only_the_focus(window, tmp_path):
    training_window(window)
    window.update(1 / 60)
    frame = frame_state(window)
    assert list(frame["rays"]) == [str(frame["focus"])] and len(frame["rays"][str(frame["focus"])]) == config.N_RAYS
    assert frame["action"] in nets_labels() and frame["ghost"] is None
    focus = frame["focus"]
    assert frame["sensor"] == {str(focus): round(float(window.playback.cars.ray_length[focus]), 1)}
    race_window(window, tmp_path)
    window.update(1 / 60)
    assert frame_state(window)["action"] == ""                     # before the first step
    window.command("start")
    for _ in range(5):
        window.update(1 / 60)
    frame = frame_state(window)
    live = [str(i) for i, done in enumerate(frame["done"]) if not done]
    assert frame["mode"] == "race" and sorted(frame["rays"]) == sorted(live) and len(live) == 3
    assert frame["action"] in nets_labels() and frame["ghost"] is not None and len(frame["ghost"]) == 3
    assert frame["sensor"] == {k: round(float(window.race.cars.ray_length[int(k)]), 1) for k in live}
    assert all(max(frame["rays"][k]) <= frame["sensor"][k] for k in live)        # no ray is longer than its sensor
    window.race.cars.crashed[1] = True
    flags = frame_state(window)
    assert len(flags["crashed"]) == len(flags["done"]) == len(flags["x"])
    assert flags["crashed"] == [0, 1, 0]
    window.command("follow", 2)
    assert frame_state(window)["focus"] == 2
    json.dumps(frame)


def nets_labels():
    from racing import netfig
    return netfig.output_labels()


def test_panel_leaderboard_keeps_lap_times_crash_status_and_the_pole_row(window, tmp_path):
    race_window(window, tmp_path)
    window.update(1 / 60)
    window.command("start")
    window.update(1 / 60)
    window.race.cars.crashed[1] = True
    window.race.cars.done[1] = True
    window.race.cars.last_lap[0] = window.race.cars.best_lap[0] = 51.94
    state = panel_state(window)
    rows = state["leaderboard"]
    assert state["pole"] is not None and len(rows) == 4
    pole = [r for r in rows if r["car"] == "pole"]
    assert len(pole) == 1 and pole[0]["name"] == window.ghost.name
    cars = [r for r in rows if r["car"] != "pole"]
    assert {"position", "car", "name", "color", "laps", "last", "best", "status", "gap"} <= set(cars[0])
    assert [r["status"] for r in cars if r["car"] == 1] == ["crashed"]
    assert [(r["last"], r["best"]) for r in cars if r["car"] == 0] == [(51.94, 51.94)]
    assert [r["last"] for r in cars if r["car"] == 2] == [None]
    assert [r["name"] for r in cars if r["car"] == 0] == ["ES (seeded)"]
    json.dumps(state)


def test_panel_controls_lock_like_the_old_panel(window, tmp_path, monkeypatch):
    state = panel_state(window)
    assert all(a["enabled"] for a in state["approaches"]) and not state["stop"]["enabled"]
    assert state["new_run"]["enabled"] and state["difficulty"]["enabled"] and state["population"]["enabled"]
    fake_trainer_context(monkeypatch, [])
    window.command("play")
    state = panel_state(window)
    assert not any(a["enabled"] for a in state["approaches"]) and state["stop"]["enabled"]
    assert not state["new_run"]["enabled"] and not state["difficulty"]["enabled"]
    assert state["play"] == {"label": "Pause", "active": True}
    window.command("play")
    assert panel_state(window)["play"] == {"label": "Play", "active": False}
    window.process = None
    window.command("source")                                    # a notebook run
    state = panel_state(window)
    assert state["source"] == "notebook" and not state["difficulty"]["enabled"] and not state["stop"]["enabled"]
    assert state["population"] is None and not state["new_run"]["enabled"] and state["folder"]["enabled"]
    race_window(window, tmp_path)
    state = panel_state(window)
    assert state["grip"]["enabled"] and state["start"]["enabled"] and not state["stop"]["enabled"]
    window.command("start")
    state = panel_state(window)
    assert state["racing"] and not state["grip"]["enabled"] and not state["start"]["enabled"]
    assert state["stop"]["enabled"] and not state["track"]["enabled"] and not state["race_level"]["enabled"]
    assert not any(m["enabled"] for m in state["methods"])
    window.command("stop")
    assert not panel_state(window)["racing"]


def test_panel_history_marks_tall_and_short_events(window):
    for generation, event in enumerate(["level_up", "goal", "tightened", "hardened", "other", None]):
        window.receive(ga_report(generation, event=event, min_share=0.5 + 0.05 * generation))
    history = panel_state(window)["history"]
    assert history["kind"] == "share"
    assert [p[3] for p in history["points"]] == ["tall", "tall", "short", "short", None, None]
    assert [p[0] for p in history["points"]] == list(range(6)) and history["points"][0][2] is None
    assert history["need"] == pytest.approx(0.9)
    window.clear_view()
    window.receive(ga_report(approach="kt_random", population_=None) | {"best": 0.7, "mean": 0.6})
    json.dumps(panel_state(window))


def test_stats_rows_name_every_training_statistic(window):
    state = panel_state(window)
    assert state["stats"][0][0] == "Map" and state["note"] == "No generation completed yet."
    window.receive(ga_report(level=4, generation=412, target_reduction=0.143))
    state = panel_state(window)
    labels = [label for label, _ in state["stats"]]
    for wanted in ("Map", "Difficulty", "Generation", "Maps succeeded", "Streak", "Target cut", "Kept", "Best",
                   "Mean", "Sensors", "Steering"):
        assert wanted in labels
    rows = dict(state["stats"])
    assert rows["Difficulty"] == "Extra hard" and rows["Maps succeeded"] == "84% 92% 95%  (need 90%)"
    assert rows["Target cut"] == "−14.3%" and state["note"] is None
    lines = viewer.stats_lines(window.latest, "ga")                # the same values as the text lines
    assert rows["Generation"] in lines[0] and rows["Maps succeeded"] in lines[1]
    json.dumps(state)


def test_panel_of_the_race_view_lists_cars_methods_and_app_choices(window, tmp_path):
    app_race_window(window, tmp_path)
    window.command("app_choice", "ga")
    window.command("follow", 1)
    state = panel_state(window)
    assert [c["name"] for c in state["cars"]] == ["ES (seeded)", "GA (seeded)", "GA r1", "GA (app) s2"]
    assert [c["stripe"] for c in state["cars"]] == [False, False, True, False]
    assert [c["app"] for c in state["cars"]] == [False, False, False, True]
    assert [m["id"] for m in state["methods"]] == ["es", "ga"]
    assert [(a["id"], a["choice"]) for a in state["app_choices"]] == [("ga", "seed 2"), ("neat", "off")]
    assert state["follow"] == 1 and state["race_runs"]["value"] == "all" and state["track"]["name"]
    assert [c[0] for c in state["race_level"]["choices"]] == list(config.APP_DIFFICULTIES)
    json.dumps(state)


def test_every_command_name_is_accepted_and_unknown_ones_raise(window, tmp_path, monkeypatch):
    app_race_window(window, tmp_path)
    window.set_mode("train")
    fake_trainer_context(monkeypatch, [])
    calls = {"mode": "race", "approach": "es", "source": None, "folder": None, "run": 1, "difficulty": 2,
             "population": 1, "play": None, "stop": None, "new_run": None, "show": "worst", "shown": 5, "speed": 4,
             "grip": "Unlimited", "network": None, "track": 1, "race_level": 3, "start": None, "race_runs": None,
             "method": "ga", "group": None, "app_choice": "ga", "follow": None, "quit": None}
    assert set(calls) == set(viewer.COMMANDS)
    for name, arg in calls.items():
        if name == "network":
            monkeypatch.setattr(window, "toggle_network_window", lambda: None)
        window.process = None                                   # a command may have started a fake training
        window.command(name, arg)
    assert window.speed == 4 and window.grip_name == "Unlimited" and not window.alive
    with pytest.raises(ValueError):
        window.command("explode")


def test_commands_act_on_the_app(window, tmp_path):
    training_window(window)
    window.command("shown", 7)
    assert window.shown == 7 and window.playback is None
    window.command("show", "all")
    window.command("speed", 8)
    window.command("difficulty", 3)
    window.command("population", -1)
    assert (window.show, window.speed, window.difficulty) == ("all", 8, 3)
    presets = list(config.POPULATION_PRESETS)
    assert panel_state(window)["population"]["value"] == presets[presets.index(config.GA_POP) - 1]
    window.command("approach", "neat")
    assert window.approach == "neat" and panel_state(window)["color"] == config.COLORS["neat"]
    window.command("mode", "race")
    assert window.mode == "race"
    window.command("follow", None)
    assert window.follow is None


def test_bad_command_arguments_raise(window, tmp_path):
    race_window(window, tmp_path)
    window.set_mode("train")
    before = (window.approach, window.speed, window.difficulty, window.follow)
    for name, arg in (("approach", "x"), ("speed", 3), ("difficulty", 9), ("follow", 99), ("follow", "a"),
                      ("follow", True), ("run", 2), ("population", 0), ("show", "some"), ("shown", "9"),
                      ("shown", 2.5), ("grip", "Slippery"), ("race_level", 9), ("mode", "pit"), ("method", "zz"),
                      ("app_choice", "ga"), ("track", 0), ("speed", [1]), ("difficulty", True)):
        with pytest.raises(ValueError):
            window.command(name, arg)
    assert before == (window.approach, window.speed, window.difficulty, window.follow)


def test_panel_state_is_plain_json_in_every_view(window, tmp_path):
    json.dumps(panel_state(window))
    training_window(window)
    state = panel_state(window)
    assert json.loads(json.dumps(state)) == state
    app_race_window(window, tmp_path)
    window.update(1 / 60)
    state = panel_state(window)
    assert json.loads(json.dumps(state)) == state
