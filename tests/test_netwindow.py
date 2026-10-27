import pickle
import queue

import numpy as np
import pygame
import pytest

from racing import config, netfig, netwindow
from test_netfig import dense_driver, neat_driver


class ScriptedQueue:
    """A message queue that replays a script: a message, "empty", "resize" (posts a resize event, then empty) or
    None (quit). Anything after the script is empty."""

    def __init__(self, script, size=(500, 400)):
        self.script, self.size = list(script), size

    def get_nowait(self):
        if not self.script:
            raise queue.Empty
        item = self.script.pop(0)
        if item == "empty":
            raise queue.Empty
        if item == "resize":
            pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, size=self.size, w=self.size[0],
                                                 h=self.size[1]))
            raise queue.Empty
        return item


@pytest.fixture
def dummy_video(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    yield
    pygame.quit()


@pytest.mark.parametrize("make", [dense_driver, neat_driver])
def test_network_message_is_picklable_and_holds_the_graph(make):
    driver = make()
    message = netwindow.network_message(driver, "GA · seed 0")
    assert message["title"] == "GA · seed 0" and message["graph"] == netfig.network_graph(driver)
    assert pickle.loads(pickle.dumps(message)) == message


def test_the_window_draws_one_message_and_quits(dummy_video):
    drawn = []
    message = netwindow.network_message(dense_driver(), "t")
    netwindow.window_main(ScriptedQueue([message, "empty", "empty", None]), on_draw=drawn.append)
    assert len(drawn) == 1 and drawn[0] == netwindow.START_SIZE


def test_only_the_latest_message_is_drawn(dummy_video):
    drawn = []
    first, last = (netwindow.network_message(dense_driver(), t) for t in ("a", "b"))
    netwindow.window_main(ScriptedQueue([first, last, "empty", None]), on_draw=drawn.append)
    assert len(drawn) == 1


def test_a_resize_redraws_at_the_new_size(dummy_video):
    drawn = []
    message = netwindow.network_message(neat_driver(), "t")
    netwindow.window_main(ScriptedQueue([message, "resize", "empty", None]), on_draw=drawn.append)
    assert drawn == [netwindow.START_SIZE, (500, 400)]


def test_nothing_is_drawn_without_a_network(dummy_video):
    drawn = []
    netwindow.window_main(ScriptedQueue(["empty", None]), on_draw=drawn.append)
    assert drawn == []


def test_the_window_process_ends_when_the_viewer_is_gone(monkeypatch):
    events = []

    class FakeThread:
        def __init__(self, target, args, daemon):
            events.append((target, args, daemon))

        def start(self):
            events.append("watch")

    monkeypatch.setattr(netwindow.threading, "Thread", FakeThread)
    monkeypatch.setattr(netwindow.os, "getppid", lambda: 77)
    monkeypatch.setattr(netwindow, "window_main", lambda messages, parent=None: events.append("main"))
    netwindow.window_process("queue")
    target, args, daemon = events[0]
    assert target is netwindow.watch_parent and args[0] == 77 and daemon and events[1:] == ["watch", "main"]
    ended = []
    monkeypatch.setattr(netwindow.os, "getppid", lambda: 1)
    netwindow.watch_parent(77, lambda: ended.append(True), interval=0)
    assert ended == [True]


# ----- live values (version 3.15 viewer) -----

def frame(driver, seed=0, where="Silverstone"):
    return netwindow.values_message(driver, np.random.default_rng(seed).uniform(-1.0, 1.0, config.N_INPUTS), where)


@pytest.fixture
def drawn(monkeypatch):
    """What the window draws: (values, where, title) per redraw."""
    calls = []

    def render(graph, values, title, where, size):
        calls.append((values, where, title))
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    monkeypatch.setattr(netwindow.netfig, "render_live", render)
    return calls


@pytest.mark.parametrize("make", [dense_driver, neat_driver])
def test_a_values_message_is_picklable_and_holds_every_nodes_value(make):
    driver = make()
    message = frame(driver)
    assert message["where"] == "Silverstone" and pickle.loads(pickle.dumps(message)) == message
    assert [len(v) for v in message["values"]] == [len(c) for c in netfig.network_graph(driver)["columns"]]


def test_live_values_redraw_the_network(dummy_video, drawn, monkeypatch):
    monkeypatch.setattr(netwindow, "FRAME_INTERVAL", 0.0)
    driver = dense_driver()
    first, last = frame(driver, 1), frame(driver, 2, "Sepang")
    netwindow.window_main(ScriptedQueue([netwindow.network_message(driver, "t"), "empty", first, last, "empty", None]))
    assert [(where, title) for _, where, title in drawn] == [("", "t"), ("Sepang", "t")]
    assert drawn[0][0] is None and drawn[1][0] == last["values"]           # only the latest values are drawn


def test_frames_are_drawn_at_most_once_per_frame_interval(dummy_video, drawn, monkeypatch):
    monkeypatch.setattr(netwindow, "FRAME_INTERVAL", 3600.0)
    driver = dense_driver()
    netwindow.window_main(ScriptedQueue([netwindow.network_message(driver, "t"), "empty", frame(driver, 1), "empty",
                                         frame(driver, 2), "empty", None]))
    assert len(drawn) == 1                                    # the network at once, no frame within the interval


def test_a_new_network_is_drawn_at_once_and_without_the_old_values(dummy_video, drawn, monkeypatch):
    monkeypatch.setattr(netwindow, "FRAME_INTERVAL", 3600.0)
    old, new = dense_driver(), neat_driver()
    netwindow.window_main(ScriptedQueue([netwindow.network_message(old, "a"), frame(old), "empty",
                                         netwindow.network_message(new, "b"), "empty", None]))
    assert [(values is None, title) for values, _, title in drawn] == [(False, "a"), (True, "b")]


def test_values_before_any_network_draw_nothing(dummy_video, drawn):
    netwindow.window_main(ScriptedQueue([frame(dense_driver()), "empty", None]))
    assert drawn == []


# ----- the training timeline (version 3.15 viewer) -----

MAP = (config.TRAIN_SEED_START + 11, 1, None)              # an Easy map: (seed, level, width)


def entry(generation, driver=None, run=("app", "ga", 0), share=0.5, event=None, level=1):
    """What the viewer sends for one generation (netwindow.generation_message's input)."""
    driver = driver or dense_driver()
    return {"generation": generation, "title": f"GA · run 0 · generation {generation}", "driver": driver,
            "curve": share, "kind": "share", "event": event, "level": level, "map": MAP, "grip": 1.0,
            "where": "Easy", "run": run}


def generations(n, **kwargs):
    return [netwindow.generation_message(entry(g, **kwargs)) for g in range(n)]


def test_a_generation_message_is_picklable_and_holds_the_network():
    driver = dense_driver()
    message = netwindow.generation_message(entry(7, driver))
    assert message["graph"] == netfig.network_graph(driver) and message["generation"] == 7
    copy = pickle.loads(pickle.dumps(message))
    assert copy["title"] == message["title"] and copy["driver"].n_params == driver.n_params


def test_the_timeline_follows_the_newest_generation_until_you_step_back():
    timeline = netwindow.Timeline()
    for message in generations(5):
        timeline.add(message)
    assert timeline.live and timeline.shown()["generation"] == 4
    timeline.step(-1)
    assert not timeline.live and timeline.shown()["generation"] == 3
    timeline.add(generations(6)[5])                          # a new generation does not move a chosen one
    assert timeline.shown()["generation"] == 3
    timeline.go(0)
    timeline.step(-1)
    assert timeline.shown()["generation"] == 0               # clamped
    timeline.go(99)
    assert timeline.live and timeline.shown()["generation"] == 5


def test_play_steps_one_generation_every_few_seconds_and_ends_live():
    timeline = netwindow.Timeline()
    for message in generations(3):
        timeline.add(message)
    timeline.toggle_play(now=0.0)                            # from the newest: plays from the first
    assert timeline.playing and timeline.shown()["generation"] == 0
    timeline.tick(now=netwindow.PLAY_SECONDS - 0.1)
    assert timeline.shown()["generation"] == 0
    timeline.tick(now=netwindow.PLAY_SECONDS)
    assert timeline.shown()["generation"] == 1
    timeline.tick(now=2 * netwindow.PLAY_SECONDS)
    assert timeline.live and not timeline.playing            # it reached the newest


def test_a_new_run_starts_a_new_timeline():
    timeline = netwindow.Timeline()
    for message in generations(4):
        timeline.add(message)
    timeline.add(netwindow.generation_message(entry(0, run=("app", "es", 1))))
    assert len(timeline.entries) == 1 and timeline.shown()["run"] == ("app", "es", 1)
    timeline.add(netwindow.generation_message(entry(0, run=("app", "es", 1), share=0.9)))
    assert len(timeline.entries) == 1 and timeline.shown()["curve"] == 0.9     # the same generation again: replaced


def test_a_past_generation_drives_the_map_on_its_own():
    driver = dense_driver()
    drive = netwindow.Drive(driver, MAP, grip=1.0)
    first = drive.inputs(now=0.0)
    later = drive.inputs(now=1.0)                            # 20 steps of 0.05 s later
    assert first.shape == (config.N_INPUTS,) and not np.array_equal(first, later) and drive.steps == 20
    for t in range(2, 200):
        drive.inputs(now=float(t))                          # crashes and finishes start again on the line
    assert not drive.cars.done[0]


def click(name, size=netwindow.START_SIZE):
    rect = netwindow.strip_layout(size)[name]
    return ("click", rect.center)


class ClickingQueue(ScriptedQueue):
    """A ScriptedQueue that also understands ("click", (x, y)): a left click there."""

    def get_nowait(self):
        if self.script and isinstance(self.script[0], tuple) and self.script[0][0] == "click":
            _, pos = self.script.pop(0)
            for kind in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                pygame.event.post(pygame.event.Event(kind, pos=pos, button=1))
            raise queue.Empty
        return super().get_nowait()


@pytest.fixture
def frames(monkeypatch):
    """Every diagram the window draws: (title, values, size)."""
    calls = []

    def render(graph, values, title, where, size):
        calls.append((title, values, size))
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    monkeypatch.setattr(netwindow.netfig, "render_live", render)
    monkeypatch.setattr(netwindow, "FRAME_INTERVAL", 0.0)
    return calls


def test_training_leaves_room_for_the_timeline_and_a_race_does_not(dummy_video, frames):
    width, height = netwindow.START_SIZE
    netwindow.window_main(ClickingQueue([*generations(2), "empty", netwindow.network_message(dense_driver(), "race"),
                                         "empty", None]))
    sizes = {title: size for title, _, size in frames}
    assert sizes["GA · run 0 · generation 1"] == (width, height - netwindow.STRIP)
    assert sizes["race"] == (width, height)


def test_stepping_back_shows_the_earlier_network_driving_on_its_own(dummy_video, frames):
    driver = dense_driver()
    script = [netwindow.generation_message(entry(g, driver)) for g in range(3)]
    netwindow.window_main(ClickingQueue([*script, "empty", click("back"), "empty", "empty", "empty", None]))
    shown = [(title, values) for title, values, _ in frames if title.endswith("generation 1")]
    assert shown and shown[-1][1] is not None                # the window drove generation 1's car itself


def test_viewer_values_are_drawn_only_for_the_newest_generation(dummy_video, frames):
    driver = dense_driver()
    live = netwindow.values_message(driver, np.full(config.N_INPUTS, 0.3), "Easy")
    script = [netwindow.generation_message(entry(g, driver)) for g in range(3)]
    netwindow.window_main(ClickingQueue([*script, live, "empty", click("back"), live, "empty", None]))
    newest = [values for title, values, _ in frames if title.endswith("generation 2")]
    assert live["values"] in newest
    earlier = [values for title, values, _ in frames if title.endswith("generation 1")]
    assert earlier and all(values != live["values"] for values in earlier)


def test_clicking_the_timeline_picks_a_generation_and_live_returns(dummy_video, frames):
    layout = netwindow.strip_layout(netwindow.START_SIZE)["timeline"]
    first = ("click", (layout.left + 1, layout.centery))
    netwindow.window_main(ClickingQueue([*generations(5), "empty", first, "empty", click("live"), "empty", None]))
    titles = [title for title, _, _ in frames]
    assert "GA · run 0 · generation 0" in titles and titles[-1] == "GA · run 0 · generation 4"
