import http.client
import json
import threading
import time

import pytest

from racing import app as viewer
from racing import config
from racing.server import ViewerServer, browser_command


def wait_for(condition, seconds=3.0):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@pytest.fixture
def server(window, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "VIEWER_LOG_PATH", tmp_path / "viewer.log")
    window.command("mode", "race")                      # a track on show, so the page has a scene
    instance = ViewerServer(window, port=0)
    instance.url = instance.start()
    instance.done = threading.Thread(target=instance.run_forever, daemon=True)
    instance.done.start()
    yield instance
    instance.stop()


def connection(server):
    host, port = server.url.removeprefix("http://").rstrip("/").split(":")
    return http.client.HTTPConnection(host, int(port), timeout=5)


def get(server, path):
    conn = connection(server)
    conn.request("GET", path)
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response.status, body


def post_command(server, payload):
    conn = connection(server)
    conn.request("POST", "/command", json.dumps(payload), {"Content-Type": "application/json"})
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response.status, body


def read_events(server, wanted, seconds=5.0):
    """Open /events and read until every name in `wanted` arrived; the (name, data) list so far."""
    conn = connection(server)
    conn.request("GET", "/events")
    response = conn.getresponse()
    assert response.status == 200 and response.getheader("Content-Type").startswith("text/event-stream")
    seen, name, end = [], None, time.time() + seconds
    while time.time() < end and not set(wanted) <= {n for n, _ in seen}:
        line = response.readline().decode().rstrip("\n")
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            seen.append((name, json.loads(line[6:])))
    conn.close()
    return seen


def test_index_page_is_served(server):
    status, body = get(server, "/")
    assert status == 200 and b'id="sim"' in body


def test_static_paths_outside_the_web_folder_are_refused(server):
    assert get(server, "/static/index.html")[0] == 200
    assert get(server, "/static/missing.js")[0] == 404
    assert get(server, "/static/../app.py")[0] == 404
    assert get(server, "/static/%2e%2e/app.py")[0] == 404
    assert get(server, "/nothing")[0] == 404


def test_command_reaches_the_app(server, window):
    assert post_command(server, {"name": "speed", "arg": 4})[0] == 204
    assert wait_for(lambda: window.speed == 4, 1.0)


def test_bad_command_is_400(server):
    status, body = post_command(server, {"name": "nope"})
    assert status == 400 and b"nope" in body
    assert post_command(server, {"name": "speed", "arg": 123})[0] == 400
    conn = connection(server)
    conn.request("POST", "/command", "not json")
    assert conn.getresponse().status == 400


def test_events_send_scene_panel_and_frame(server):
    seen = read_events(server, {"scene", "panel", "frame"})
    assert {"scene", "panel", "frame"} <= {n for n, _ in seen}
    scene = next(d for n, d in seen if n == "scene")
    frame = next(d for n, d in seen if n == "frame")
    assert frame["scene"] == scene["key"]


def test_every_page_gets_the_scene_on_connect(server):
    for _ in range(2):
        assert read_events(server, {"scene"})[0][0] == "scene"


def test_watched_follows_connected_pages(server, window):
    assert wait_for(lambda: window.watched is False, 1.0)
    conn = connection(server)
    conn.request("GET", "/events")
    conn.getresponse().readline()
    assert wait_for(lambda: window.watched is True, 1.0) and server.clients == 1
    conn.close()
    assert wait_for(lambda: window.watched is False, 1.0) and server.clients == 0


def test_busy_port_falls_back(window):
    first = ViewerServer(window, port=0)
    port = int(first.start().rstrip("/").rsplit(":", 1)[1])
    second = ViewerServer(window, port=port)
    try:
        assert second.start().rstrip("/").endswith(f":{port + 1}")
    finally:
        second.stop()
        first.stop()


def test_quit_stops_the_server_and_the_app(server, window):
    assert post_command(server, {"name": "quit"})[0] == 204
    server.done.join(3.0)
    assert not server.done.is_alive() and window.alive is False


def test_stop_does_not_wait_for_open_pages(server):
    conn = connection(server)
    conn.request("GET", "/events")
    conn.getresponse().readline()
    started = time.time()
    server.stop()
    assert time.time() - started < 2.0
    conn.close()


def test_main_prints_the_banner_and_quits(window, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "VIEWER_LOG_PATH", tmp_path / "logs" / "viewer.log")
    monkeypatch.setattr(viewer, "App", lambda **kwargs: window)
    result = []
    thread = threading.Thread(target=lambda: result.append(viewer.main(["--port", "0", "--no-browser"])), daemon=True)
    thread.start()
    out = ""

    def banner():
        nonlocal out
        out += capsys.readouterr().out
        return "open:" in out
    assert wait_for(banner, 5.0)
    url = next(line.split()[1] for line in out.splitlines() if "open:" in line)
    assert out.startswith("EC Racing viewer") and "log:" in out and "quit:" in out
    conn = http.client.HTTPConnection(*url.removeprefix("http://").rstrip("/").split(":"))
    conn.request("POST", "/command", json.dumps({"name": "quit"}))
    assert conn.getresponse().status == 204
    thread.join(5.0)
    assert result == [0] and "command" in (tmp_path / "logs" / "viewer.log").read_text()


def test_browser_command_uses_windows_on_wsl():
    assert browser_command("http://127.0.0.1:8765/", "Linux ... microsoft-standard-WSL2") == \
        ["cmd.exe", "/c", "start", "", "http://127.0.0.1:8765/"]
    assert browser_command("http://127.0.0.1:8765/", "Linux version 6.8.0-142-generic") is None
