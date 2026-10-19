"""Network window (viewer feature): a separate pygame window that shows one car's neural network as the diagram of
racing/netfig.py. The viewer sends a message only when the shown network changes; the window redraws only when it
receives one or is resized, so racing in the viewer stays smooth. It ends when the viewer ends."""
import os
import queue
import threading
import time

import numpy as np
import pygame

from racing import config, netfig

START_SIZE = (900, 640)
BACKGROUND = (255, 255, 255)
IDLE_FPS = 20


def network_message(driver, title):
    """What the viewer sends: the diagram's title and graph (lists and floats, so it pickles cheaply)."""
    return {"title": title, "graph": netfig.network_graph(driver)}


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
    message, dirty, running = None, True, True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    size, dirty = (max(int(event.w), 1), max(int(event.h), 1)), True
            while running:
                try:
                    item = messages.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    running = False
                else:
                    message, dirty = item, True
            if dirty and running:
                screen = pygame.display.get_surface() or screen
                screen.fill(BACKGROUND)
                if message is None:
                    text = font.render("Waiting for a network…", True, (90, 90, 90))
                    screen.blit(text, text.get_rect(center=(size[0] // 2, size[1] // 2)))
                else:
                    draw_image(screen, netfig.render_graph(message["graph"], message["title"], size))
                    if on_draw is not None:
                        on_draw(size)
                pygame.display.flip()
                dirty = False
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
