"""The viewer's local web server: serves the page, streams the scene, panel and frames as server-sent events and takes
the page's commands. One loop thread owns the App (commands, updates, payloads); request threads only read the
payloads it publishes and queue commands for it."""
import json
import logging
import mimetypes
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from racing import config, viewstate

log = logging.getLogger("racing.viewer")

LOOP_HZ = 60                    # App updates per second
COMMAND_WAIT = 5.0              # s a request waits for the loop to run its command
KEEPALIVE = 1.0                 # s without an event before an event stream sends a comment (and notices a closed page)
PORT_TRIES = 20                 # the configured port, then the next 19
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")


def browser_command(url, proc_version):
    """The command that opens `url` in the Windows browser when running under WSL; None elsewhere."""
    if "microsoft" in proc_version.lower():
        return ["cmd.exe", "/c", "start", "", url]
    return None


def _json(value):
    return json.dumps(value, separators=(",", ":"))


class _Command:
    """A command waiting for the loop thread; the request thread waits on `done`."""

    def __init__(self, name, arg):
        self.name, self.arg = name, arg
        self.done = threading.Event()
        self.error = None


class ViewerServer:
    def __init__(self, app, host="127.0.0.1", port=config.VIEWER_PORT, web_dir=config.WEB_DIR):
        self.app = app
        self.host, self.port = host, port
        self.web_dir = Path(web_dir)
        self.httpd = None
        self.commands = queue.Queue()
        self.state = threading.Condition()      # guards everything below; notified on every new payload
        self.clients = 0
        self.scene_key = self.scene_json = None
        self.panel_json, self.panel_version = None, 0
        self.frame_json, self.frame_version = None, 0
        self.stopping = threading.Event()
        self.finished = threading.Event()       # set once the viewer has been told to quit
        self.threads = []

    # ----- lifecycle -----
    def start(self):
        """Bind the port (the next free ones when busy), start serving and running the App; returns the page's URL."""
        last = None
        for offset in range(PORT_TRIES):
            try:
                self.httpd = _Server((self.host, self.port + offset if self.port else 0), self._handler())
                break
            except OSError as error:
                last = error
        if self.httpd is None:
            raise last
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.threads = [threading.Thread(target=self.httpd.serve_forever, name="viewer-http", daemon=True),
                        threading.Thread(target=self._loop, name="viewer-loop", daemon=True)]
        for thread in self.threads:
            thread.start()
        url = f"http://{self.host}:{self.port}/"
        log.info("serving %s", url)
        return url

    def run_forever(self):
        """Block until the page's Quit button (or stop()), then shut everything down."""
        self.finished.wait()
        self.stop()

    def stop(self):
        if self.stopping.is_set():
            return
        self.stopping.set()
        self.finished.set()
        with self.state:
            self.state.notify_all()             # event streams end
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(2.0)
        log.info("server stopped")

    # ----- the loop thread: the only one that touches the App -----
    def _loop(self):
        app, last = self.app, time.perf_counter()
        next_frame = next_panel = 0.0
        status = app.status
        problem = None
        while not self.stopping.is_set():
            now = time.perf_counter()
            dt, last = now - last, now
            try:
                self._run_commands()
                if not app.alive:
                    break
                app.watched = self.clients > 0
                app.update(min(dt, 0.25))
                if app.status != status:
                    status = app.status
                    log.info("status: %s", status)
                if self.clients > 0:
                    next_frame, next_panel = self._publish(now, next_frame, next_panel)
                problem = None
            except Exception as error:                  # keep the viewer up; log each new problem once
                if str(error) != problem:
                    log.exception("viewer loop error")
                problem = str(error)
            self.stopping.wait(max(0.0, 1.0 / LOOP_HZ - (time.perf_counter() - now)))
        if not app.alive:
            log.info("viewer quit")
            time.sleep(0.2)                             # let the quit request's answer go out first
            self.finished.set()

    def _run_commands(self):
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                self.app.command(command.name, command.arg)
                log.info("command %s %r", command.name, command.arg)
            except ValueError as error:
                command.error = str(error)
                log.warning("command %s %r refused: %s", command.name, command.arg, error)
            except Exception as error:
                command.error = f"{type(error).__name__}: {error}"
                log.exception("command %s %r failed", command.name, command.arg)
            command.done.set()
            if not self.app.alive:
                return

    def _publish(self, now, next_frame, next_panel):
        app = self.app
        track = viewstate.shown_track(app)
        with_scene = None
        if track is not None and viewstate.scene_key(track) != self.scene_key:
            with_scene = viewstate.scene_state(track)
        panel = None
        if now >= next_panel:
            panel = _json(viewstate.panel_state(app))
            next_panel = now + 1.0 / config.PANEL_HZ
        frame = None
        if now >= next_frame:
            state = viewstate.frame_state(app)
            frame = None if state is None else _json(state)
            next_frame = now + 1.0 / config.FRAME_HZ
        if with_scene is None and frame is None and (panel is None or panel == self.panel_json):
            return next_frame, next_panel
        with self.state:
            if with_scene is not None:
                self.scene_key, self.scene_json = with_scene["key"], _json(with_scene)
            if panel is not None and panel != self.panel_json:
                self.panel_json, self.panel_version = panel, self.panel_version + 1
            if frame is not None:
                self.frame_json, self.frame_version = frame, self.frame_version + 1
            self.state.notify_all()
        return next_frame, next_panel

    # ----- request threads -----
    def submit(self, name, arg):
        """Queue a command for the loop thread and wait for it; None when it ran, else the error text."""
        command = _Command(name, arg)
        self.commands.put(command)
        if not command.done.wait(COMMAND_WAIT):
            return "the viewer did not answer"
        return command.error

    def _client(self, delta):
        with self.state:
            self.clients += delta
            count = self.clients
        return count

    def stream(self, write):
        """Send events through `write(text)` until the page leaves or the server stops. A page gets the scene first."""
        count = self._client(+1)
        log.info("page connected (%d open)", count)
        sent_scene, sent_panel, sent_frame = None, 0, 0
        try:
            while not self.stopping.is_set():
                with self.state:
                    if not self.state.wait_for(lambda: self.stopping.is_set() or self._news(sent_scene, sent_panel,
                                                                                          sent_frame), KEEPALIVE):
                        out = ": keepalive\n\n"
                    else:
                        out, sent_scene, sent_panel, sent_frame = self._events(sent_scene, sent_panel, sent_frame)
                if out:
                    write(out)
        except (OSError, ValueError):
            pass                                        # the page went away
        finally:
            count = self._client(-1)
            log.info("page disconnected (%d open)", count)

    def _news(self, scene, panel, frame):
        return ((self.scene_json is not None and self.scene_key != scene) or self.panel_version != panel
                or self.frame_version != frame)

    def _events(self, scene, panel, frame):
        out = []
        if self.scene_json is not None and self.scene_key != scene:
            out.append(f"event: scene\ndata: {self.scene_json}\n\n")
            scene = self.scene_key
        if self.panel_version != panel and self.panel_json is not None:
            out.append(f"event: panel\ndata: {self.panel_json}\n\n")
            panel = self.panel_version
        if self.frame_version != frame and self.frame_json is not None:
            out.append(f"event: frame\ndata: {self.frame_json}\n\n")
            frame = self.frame_version
        return "".join(out), scene, panel, frame

    def file_for(self, name):
        """The file under the web folder that `name` (a URL path) names; None when missing or outside it."""
        root = self.web_dir.resolve()
        path = (root / unquote(name).lstrip("/")).resolve()
        if root != path and root not in path.parents or not path.is_file():
            return None
        return path

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                log.debug("http %s", format % args)

            def reply(self, status, body=b"", content_type="text/plain; charset=utf-8"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def send_file(self, path):
                kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if kind.startswith("text/") or kind in ("application/json", "image/svg+xml"):
                    kind += "; charset=utf-8"
                self.reply(200, path.read_bytes(), kind)

            def do_GET(self):
                route = urlsplit(self.path).path
                if route == "/":
                    path = server.file_for("index.html")
                elif route.startswith("/static/"):
                    path = server.file_for(route[len("/static/"):])
                elif route == "/events":
                    return self.events()
                else:
                    path = None
                if path is None:
                    return self.reply(404, b"not found")
                self.send_file(path)

            def events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def write(text):
                    self.wfile.write(text.encode())
                    self.wfile.flush()
                server.stream(write)

            def do_POST(self):
                if urlsplit(self.path).path != "/command":
                    return self.reply(404, b"not found")
                try:
                    body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                    message = json.loads(body)
                    name = message["name"]
                    if not isinstance(name, str):
                        raise TypeError("name must be a string")
                except (ValueError, KeyError, TypeError) as error:
                    return self.reply(400, f"bad command message: {error}".encode())
                error = server.submit(name, message.get("arg"))
                if error is None:
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    self.reply(400, error.encode())

        return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False         # a busy port must fail, so the next one is tried
