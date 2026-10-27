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
