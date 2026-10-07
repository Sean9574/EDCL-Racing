import pickle
import queue

import pygame
import pytest

from racing import netfig, netwindow
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
