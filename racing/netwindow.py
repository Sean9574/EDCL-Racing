"""Network window (viewer feature): a separate pygame window that shows one car's neural network live, as
netfig.render_live draws it. In a race the viewer sends the shown car's network when it changes; in training it sends
the best network of every generation it trains or replays, and the window keeps them on a timeline under the
diagram. While the shown car drives, the viewer sends the value of every node about ten times a second. A past
generation chosen on the timeline drives the newest generation's map on its own in this window, so it is shown live
too. A new network or size is drawn at once and new values at most once per FRAME_INTERVAL, so racing in the viewer
stays smooth. The window ends when the viewer ends."""
import os
import queue
import threading
import time

import numpy as np
import pygame

from racing import config, netfig

START_SIZE = (900, 640)
BACKGROUND = (255, 255, 255)
IDLE_FPS = 30
FRAME_INTERVAL = 0.1        # s: new values are drawn at most this often
STRIP = 118                 # px under the diagram for the training timeline and its buttons
PLAY_SECONDS = 3.0          # s each generation is shown when the timeline plays
MAX_DRIVE_STEPS = 40        # simulation steps a past generation's car may catch up in one frame (2 s)
INK, MUTED, FAINT, RIM = (29, 35, 48), (95, 102, 115), (154, 161, 173), (201, 206, 216)
BLUE, DEEP, PANEL = (37, 99, 168), (30, 58, 110), (248, 249, 251)


def network_message(driver, title):
    """What the viewer sends when the shown race car changes: the diagram's title and graph (lists and floats, so it
    pickles cheaply)."""
    return {"title": title, "graph": netfig.network_graph(driver)}


def values_message(driver, obs, where):
    """What the viewer sends while the car drives: the value of every node for the car's inputs `obs`, and where it
    drives (a circuit's name, or the level)."""
    return {"values": netfig.node_values(driver, obs), "where": where}


def generation_message(entry):
    """What the viewer sends for each generation in training: `entry` (the generation, its best network's driver and
    title, the run's progress as "curve" of "kind" "share" (the lowest map success share) or "accuracy", its event and
    level, the map (seed, level, width) it replays with the display grip and where that is, and the run it belongs to)
    with the network's graph."""
    return {**entry, "graph": netfig.network_graph(entry["driver"])}


class Timeline:
    """The best network of each generation the window has been sent, which one it shows (the newest while live), and
    whether it plays through them one generation every PLAY_SECONDS."""

    def __init__(self):
        self.entries, self.at, self.playing, self.turned = [], None, False, 0.0

    @property
    def live(self):
        return self.at is None

    def index(self):
        return len(self.entries) - 1 if self.at is None else self.at

    def shown(self):
        return self.entries[self.index()] if self.entries else None

    def add(self, message):
        """Append a generation; a generation of another run, or an earlier one (a new run), starts the timeline
        again, and the same generation again replaces it."""
        last = self.entries[-1] if self.entries else None
        if last is not None and (message["run"] != last["run"] or message["generation"] < last["generation"]):
            self.__init__()
            last = None
        if last is not None and message["generation"] == last["generation"]:
            self.entries[-1] = message
        else:
            self.entries.append(message)

    def go(self, index):
        """Show entry `index` (clamped); the newest one is live."""
        if self.entries:
            index = max(0, min(int(index), len(self.entries) - 1))
            self.at = None if index == len(self.entries) - 1 else index

    def step(self, delta):
        self.go(self.index() + delta)

    def toggle_play(self, now):
        """Play from the shown generation (from the first when live), or pause."""
        if self.playing:
            self.playing = False
            return
        if self.live:
            self.go(0)
        self.playing, self.turned = True, now

    def tick(self, now):
        """Move a playing timeline on; True when the shown generation changed. Reaching the newest ends play."""
        if not self.playing or now - self.turned < PLAY_SECONDS:
            return False
        self.step(1)
        self.turned = now
        if self.live:
            self.playing = False
        return True


class Drive:
    """A past generation's best car driving a map on its own in real time, so the timeline shows its network live
    (the viewer drives only the newest generation's). Crashed or finished, it starts again on the line."""

    def __init__(self, driver, map_key, grip=1.0, laps=1):
        from racing import sim, tracks
        self.track = tracks.get_track(*map_key)
        self.driver, self.grip, self.laps = driver, float(grip), laps
        self.time_limit = sim.time_limit_for(self.track, laps)
        self.clock, self.steps = None, 0
        self.restart()

    def restart(self):
        from racing import nets, sim
        self.cars = sim.start_cars(self.track, 1, self.laps, nets.drivers_setup([self.driver]))

    def inputs(self, now):
        """The car's inputs at clock time `now`, after driving the time since the last call (at most
        MAX_DRIVE_STEPS steps)."""
        from racing import sim
        from racing.actions import class_to_controls
        if self.clock is None:
            self.clock = now
        steps = int((now - self.clock) / config.DT)
        self.clock = now if steps > MAX_DRIVE_STEPS else self.clock + steps * config.DT
        for _ in range(min(steps, MAX_DRIVE_STEPS)):
            if self.cars.done[0]:
                self.restart()
            steer, throttle = class_to_controls(self.driver.act(sim.sense(self.track, self.cars)))
            sim.step(self.track, self.cars, steer, throttle, self.time_limit, grip_scale=self.grip)
            self.steps += 1
        if self.cars.done[0]:
            self.restart()
        return sim.sense(self.track, self.cars)[0]


def strip_layout(size):
    """The training strip's rectangles at window size `size`: the timeline and the buttons back, play, forward and
    live."""
    width, height = size
    top = height - STRIP
    rects = {"timeline": pygame.Rect(60, top + 24, max(width - 140, 40), 50)}
    x = 60
    for name, w in (("back", 40), ("play", 104), ("forward", 40), ("live", 56)):
        rects[name] = pygame.Rect(x, top + 86, w, 26)
        x += w + 8
    return rects


def index_at(x, rect, n):
    """The timeline entry under x."""
    if n <= 1:
        return 0
    return int(round(min(max((x - rect.left) / max(rect.width, 1), 0.0), 1.0) * (n - 1)))


def draw_strip(screen, timeline, size, fonts):
    """The timeline (the run's progress per collected generation, level-ups marked, the shown one as a playhead) and
    the buttons, under the diagram."""
    small, tiny = fonts
    width, height = size
    top = height - STRIP
    rects = strip_layout(size)
    screen.fill(BACKGROUND, pygame.Rect(0, top, width, STRIP))
    screen.blit(tiny.render("Best network of each generation, collected while you watch", True, FAINT), (24, top + 4))
    box = rects["timeline"]
    pygame.draw.rect(screen, PANEL, box, border_radius=4)
    entries, n = timeline.entries, len(timeline.entries)

    def point(i):
        x = box.left + (box.width * i / (n - 1) if n > 1 else box.width / 2)
        return x, box.bottom - box.height * min(max(float(entries[i]["curve"]), 0.0), 1.0)

    if entries and entries[-1]["kind"] == "share":
        y = box.bottom - box.height * config.MASTERY_FRACTION
        for x in range(box.left, box.right, 8):
            pygame.draw.line(screen, RIM, (x, y), (min(x + 4, box.right), y))
        screen.blit(tiny.render(f"{100 * config.MASTERY_FRACTION:.0f}%", True, FAINT), (box.right + 6, y - 8))
    for i, e in enumerate(entries):
        if e["event"] in ("level_up", "goal"):
            x = point(i)[0]
            pygame.draw.line(screen, MUTED, (x, box.top), (x, box.bottom))
            name = "goal" if e["event"] == "goal" else config.DIFFICULTIES.get(e["level"] + 1, "")
            screen.blit(tiny.render(name, True, MUTED), (x + 4, box.top + 2))
    if n > 1:
        pygame.draw.lines(screen, BLUE, False, [point(i) for i in range(n)], 2)
    if entries:
        x, y = point(timeline.index())
        pygame.draw.line(screen, INK, (x, box.top - 4), (x, box.bottom + 4), 2)
        pygame.draw.circle(screen, DEEP, (int(x), int(y)), 5)
        shown = timeline.shown()
        label = small.render(f"generation {shown['generation']} · {timeline.index() + 1} of {n} collected", True,
                             MUTED)
        screen.blit(label, label.get_rect(right=width - 24, centery=rects["live"].centery))
    play = "❚❚  pause" if timeline.playing else "▶  play"
    labels = {"back": "◀", "play": play, "forward": "▶", "live": "Live"}
    for name, text in labels.items():
        on = (name == "play" and timeline.playing) or (name == "live" and timeline.live)
        rect = rects[name]
        pygame.draw.rect(screen, DEEP if on else BACKGROUND, rect, border_radius=13)
        if not on:
            pygame.draw.rect(screen, RIM, rect, 1, border_radius=13)
        image = small.render(text, True, BACKGROUND if on else INK)
        screen.blit(image, image.get_rect(center=rect.center))


def draw_image(screen, image):
    """Blit an (height, width, 3) uint8 array onto the display."""
    height, width = image.shape[:2]
    screen.blit(pygame.image.frombuffer(np.ascontiguousarray(image).tobytes(), (width, height), "RGB"), (0, 0))


def window_main(messages, parent=None, on_draw=None):
    """Open the window and show what `messages` (a queue; None means quit) sends: a race car's network, or the
    training timeline. `on_draw(size)` is called after each redraw of a diagram (a hook for tests). Returns when the
    window is closed or None arrives."""
    pygame.init()
    pygame.display.set_caption("EC Racing – network")
    size = START_SIZE
    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    font = pygame.font.Font(str(config.FONT_PATH), 16)                  # shipped: the same widths on every machine
    fonts = (pygame.font.Font(str(config.FONT_PATH), 13), pygame.font.Font(str(config.FONT_PATH), 12))
    clock = pygame.time.Clock()
    race, frame, timeline, mode, running = None, None, Timeline(), None, True
    drive, drive_key, dragging = None, None, False
    urgent, fresh, drawn_at = True, False, float("-inf")    # urgent: something new to show; fresh: new values

    def control(name, now):
        if name == "back":
            timeline.step(-1)
        elif name == "forward":
            timeline.step(1)
        elif name == "play":
            timeline.toggle_play(now)
        elif name == "live":
            timeline.go(len(timeline.entries))

    try:
        while running:
            now = time.monotonic()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    size, urgent = (max(int(event.w), 1), max(int(event.h), 1)), True
                elif mode == "train" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    rects = strip_layout(size)
                    if rects["timeline"].inflate(0, 16).collidepoint(event.pos):
                        dragging = True
                        timeline.go(index_at(event.pos[0], rects["timeline"], len(timeline.entries)))
                    for name in ("back", "play", "forward", "live"):
                        if rects[name].collidepoint(event.pos):
                            control(name, now)
                    urgent = True
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging = False
                elif mode == "train" and event.type == pygame.MOUSEMOTION and dragging:
                    timeline.go(index_at(event.pos[0], strip_layout(size)["timeline"], len(timeline.entries)))
                    urgent = True
                elif mode == "train" and event.type == pygame.KEYDOWN:
                    keys = {pygame.K_LEFT: "back", pygame.K_RIGHT: "forward", pygame.K_SPACE: "play",
                            pygame.K_l: "live", pygame.K_END: "live"}
                    if event.key in keys:
                        control(keys[event.key], now)
                        urgent = True
            while running:
                try:
                    item = messages.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    running = False
                elif "generation" in item:                   # training: the newest generation's best network
                    timeline.add(item)
                    mode, frame, urgent = "train", None, True
                elif "graph" in item:                        # a race car's network
                    race, mode, frame, urgent = item, "race", None, True
                elif mode is not None:                       # values of the race car or the newest generation
                    frame, fresh = item, True
            if mode == "train" and timeline.tick(now):
                urgent = True
            due = now - drawn_at >= FRAME_INTERVAL
            past = mode == "train" and not timeline.live
            if running and (urgent or (fresh and due) or (past and due)):
                screen = pygame.display.get_surface() or screen
                screen.fill(BACKGROUND)
                if mode is None:
                    text = font.render("Waiting for a network…", True, (90, 90, 90))
                    screen.blit(text, text.get_rect(center=(size[0] // 2, size[1] // 2)))
                elif mode == "race":
                    draw_image(screen, netfig.render_live(race["graph"], frame and frame["values"], race["title"],
                                                          frame["where"] if frame else "", size))
                else:
                    shown, newest = timeline.shown(), timeline.entries[-1]
                    if past:
                        key = (shown["run"], shown["generation"], newest["map"])
                        if key != drive_key:
                            drive, drive_key = Drive(shown["driver"], newest["map"], newest["grip"]), key
                        values = netfig.node_values(shown["driver"], drive.inputs(now))
                        where = f"{newest['where']} · driven here"
                    else:
                        values = frame and frame["values"]
                        where = frame["where"] if frame else newest["where"]
                    diagram = (size[0], max(size[1] - STRIP, 120))
                    draw_image(screen, netfig.render_live(shown["graph"], values, shown["title"], where, diagram))
                    draw_strip(screen, timeline, size, fonts)
                if mode is not None and on_draw is not None:
                    on_draw(size)
                pygame.display.flip()
                urgent, fresh, drawn_at = False, False, now
            clock.tick(IDLE_FPS)
    finally:
        pygame.quit()


def watch_parent(parent, leave, interval=1.0):
    """Waits until the viewer that started the window is gone (the parent process changes), then calls leave()."""
    while os.getppid() == parent:
        time.sleep(interval)
    leave()


def window_process(messages):
    """Entry point of the window's process. A watcher thread ends it if the viewer exits without closing it."""
    threading.Thread(target=watch_parent, args=(os.getppid(), lambda: os._exit(0)), daemon=True).start()
    window_main(messages)
