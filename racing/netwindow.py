"""Network window (viewer feature): a separate pygame window that shows one car's neural network live, as
netfig.render_live draws it. The viewer sends the network when the shown network changes and then, while the car
drives, the value of every node about ten times a second; the window draws a new network or size at once and new
values at most once per FRAME_INTERVAL, so racing in the viewer stays smooth. It ends when the viewer ends."""
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


def network_message(driver, title):
    """What the viewer sends when the shown network changes: the diagram's title and graph (lists and floats, so it
    pickles cheaply)."""
    return {"title": title, "graph": netfig.network_graph(driver)}


def values_message(driver, obs, where):
    """What the viewer sends while the car drives: the value of every node for the car's inputs `obs`, and where it
    drives (a circuit's name, or the level)."""
    return {"values": netfig.node_values(driver, obs), "where": where}


def draw_image(screen, image):
    """Blit an (height, width, 3) uint8 array onto the display."""
    height, width = image.shape[:2]
    screen.blit(pygame.image.frombuffer(np.ascontiguousarray(image).tobytes(), (width, height), "RGB"), (0, 0))


def window_main(messages, parent=None, on_draw=None):
    """Open the window and show the latest network in `messages` (a queue; None means quit). `on_draw(size)` is
    called after each redraw (a hook for tests). Returns when the window is closed or None arrives."""
    pygame.init()
    pygame.display.set_caption("EC Racing – network")
    size = START_SIZE
    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    font = pygame.font.Font(str(config.FONT_PATH), 16)                  # shipped: the same widths on every machine
    clock = pygame.time.Clock()
    message, frame, running = None, None, True
    urgent, fresh, drawn_at = True, False, float("-inf")    # urgent: a new network or size; fresh: new values
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    size, urgent = (max(int(event.w), 1), max(int(event.h), 1)), True
            while running:
                try:
                    item = messages.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    running = False
                elif "graph" in item:
                    message, frame, urgent = item, None, True
                elif message is not None:                # values of the shown network
                    frame, fresh = item, True
            due = fresh and time.monotonic() - drawn_at >= FRAME_INTERVAL
            if running and (urgent or due):
                screen = pygame.display.get_surface() or screen
                screen.fill(BACKGROUND)
                if message is None:
                    text = font.render("Waiting for a network…", True, (90, 90, 90))
                    screen.blit(text, text.get_rect(center=(size[0] // 2, size[1] // 2)))
                else:
                    draw_image(screen, netfig.render_live(message["graph"], frame and frame["values"],
                                                          message["title"], frame["where"] if frame else "", size))
                    if on_draw is not None:
                        on_draw(size)
                pygame.display.flip()
                urgent, fresh, drawn_at = False, False, time.monotonic()
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
