"""Progress of a long experiment (spec §3.3): the running run writes its state to a status file and its
milestones to the run log; the runner, or a monitor in another terminal, shows the new milestones and a one-line
status.

Watch a run started in the background with:  ~/miniconda3/bin/python -m racing.progress

Runs trained side by side (scripts/run_notebooks.py --runs) each write their own status file in results/status/; the
monitor then shows one line per run.
"""
import argparse
import json
import math
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from racing import config

EIGHTHS = " ▏▎▍▌▋▊▉"
FULL = "█"
RECENT = 10                  # run-log lines the separate monitor shows when it starts
WAITING = "waiting for the first status…"


def _json_value(value):
    if hasattr(value, "item"):                     # numpy scalars
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_status(status, path=None):
    """Replace the status file with `status` plus the time it was written."""
    path = Path(config.STATUS_PATH if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")    # same folder, so the replace is atomic
    temporary.write_text(json.dumps({**status, "updated": datetime.now().isoformat(timespec="seconds")},
                                    default=_json_value))
    os.replace(temporary, path)


def read_status(path=None):
    """The current status, or None if there is none or it cannot be read."""
    path = Path(config.STATUS_PATH if path is None else path)
    try:
        status = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return status if isinstance(status, dict) else None


def read_statuses(folder=None):
    """The status of every run in `folder` (one file per run trained side by side), in the experiment's run
    order; unreadable files are left out."""
    folder = Path(config.STATUS_DIR if folder is None else folder)
    statuses = [status for status in (read_status(path) for path in sorted(folder.glob("*.json")))
                if status is not None]
    return sorted(statuses, key=lambda s: (run_number(s.get("approach"), s.get("seed"))[0], str(s.get("approach"))))


def milestone(message, path=None):
    """Append a timestamped line to the run log and return it."""
    path = Path(config.RUN_LOG_PATH if path is None else path)
    line = f"{datetime.now().isoformat(timespec='seconds')}  {message}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line + "\n")
    return line


def run_number(approach, seed):
    """(position, total) of a run in config.RUN_ORDER; position 0 for a run outside the experiment."""
    order = list(config.RUN_ORDER)
    try:
        return order.index((approach, seed)) + 1, len(order)
    except ValueError:
        return 0, len(order)


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def bar(fraction, width=20):
    """A bar of `width` characters filled to `fraction` (0 to 1) in steps of one eighth of a character."""
    fraction = _number(fraction)
    fraction = 0.0 if math.isnan(fraction) else min(max(fraction, 0.0), 1.0)
    eighths = int(fraction * width * 8)
    text = FULL * (eighths // 8) + (EIGHTHS[eighths % 8] if eighths % 8 else "")
    return text.ljust(width)


def percent(value):
    value = _number(value)
    return "–" if math.isnan(value) else f"{100 * value:.0f}%"


def duration(seconds):
    """'0h 42m', '2h 13m' or '1d 4h 03m'; '–' when unknown."""
    seconds = _number(seconds)
    if math.isnan(seconds):
        return "–"
    minutes = int(max(seconds, 0.0)) // 60
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    return f"{days}d {hours}h {minutes:02d}m" if days else f"{hours}h {minutes:02d}m"


def accuracy(value):
    value = _number(value)
    return "–" if math.isnan(value) else f"{100 * value:.1f}%"


def format_status(status):
    """The one-line status, for example
    '[run 10/16] GA run 1 · Hard · gen 412 · worst map 84% / 90% ███████████████▊     · streak 1/3 · 2h 13m'."""
    status = status or {}
    approach, seed = status.get("approach"), status.get("seed")
    name = config.LABELS.get(approach, approach or "?")
    stage = status.get("stage", "evolution")
    if stage == "search":                          # a Keras Tuner search belongs to no run seed
        total = status.get("trials")
        trials = f"trial {status.get('trial', 0)}" + (f"/{total}" if total else "")
        return " · ".join([f"[search] {name}", trials, f"best val acc {accuracy(status.get('best'))}",
                           duration(status.get("elapsed"))])
    position, total = run_number(approach, seed)
    parts = [f"[run {position or '–'}/{total}] {name} run {'?' if seed is None else seed}"]
    streak = status.get("streak", 0)
    if stage == "imitation":
        parts += [f"imitation epoch {status.get('epoch', '?')}",
                  f"val acc {accuracy(status.get('val_accuracy'))} (best {accuracy(status.get('best'))})"]
    elif stage == "dagger":
        parts += [f"DAgger round {status.get('round', '?')}",
                  f"checks {status.get('checks', '?')}/{config.DAGGER_MAPS}",
                  f"streak {streak}/{config.DAGGER_STREAK}"]
    else:
        level = status.get("level")
        parts.append(config.DIFFICULTIES.get(level, f"level {'?' if level is None else level}"))
        if stage == "speed push":
            reduction = _number(status.get("target_reduction"))
            parts.append("speed push" if math.isnan(reduction) else f"speed push −{100 * reduction:.1f}%")
        share = _number(status.get("min_share"))
        need = config.SPEED_MASTERY_FRACTION if stage == "speed push" else config.MASTERY_FRACTION   # spec §9.4
        fill = 0.0 if math.isnan(share) else share / need
        parts += [f"gen {status.get('generation', '?')}",
                  f"worst map {percent(share)} / {percent(need)} {bar(fill)}",
                  f"streak {streak}/{config.MASTERY_STREAK}"]
    parts.append(duration(status.get("elapsed")))
    return " · ".join(parts)


def stale_note(status, now=None):
    """' (no update for N min)' when the status was written more than config.STATUS_STALE_AFTER seconds ago, as a
    run that stopped without finishing leaves it; '' otherwise or when the time it was written is unknown."""
    try:
        age = ((now or datetime.now()) - datetime.fromisoformat(status["updated"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return ""
    return f" (no update for {int(age // 60)} min)" if age > config.STATUS_STALE_AFTER else ""


class Monitor:
    """Shows the run log's new lines and the status line (spec §3.3). In a terminal the status line is
    updated in place; otherwise, as under nohup, it is printed as a whole line at most once per `period` seconds. A
    status that has not been updated for config.STATUS_STALE_AFTER seconds says so. With `status_dir`, the folder
    of runs trained side by side, it shows one status line per run instead."""

    def __init__(self, status_path=None, log_path=None, out=None, interval=1.0, live=None, period=60.0,
                 status_dir=None):
        self.status_path = status_path
        self.status_dir = status_dir
        self.log_path = Path(config.RUN_LOG_PATH if log_path is None else log_path)
        self.out = sys.stdout if out is None else out
        self.interval = interval
        self.live = bool(getattr(self.out, "isatty", lambda: False)()) if live is None else bool(live)
        self.period = period
        self.offset = self.log_path.stat().st_size if self.log_path.exists() else 0   # only lines written from now
        self.partial = ""
        self.shown = 0                     # length of the live line on screen
        self.block = 0                     # status lines on screen, with status_dir
        self.last_print = None
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.thread = None

    def new_log_lines(self):
        try:
            size = self.log_path.stat().st_size
        except OSError:
            return []
        if size < self.offset:             # the log was replaced (a fresh run)
            self.offset, self.partial = 0, ""
        if size == self.offset:
            return []
        with open(self.log_path, "rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        self.offset = size
        *lines, self.partial = (self.partial + data.decode("utf-8", errors="replace")).split("\n")
        return lines

    def status_texts(self):
        if self.status_dir is None:
            status = read_status(self.status_path)
            return [WAITING if status is None else format_status(status) + stale_note(status)]
        return [format_status(status) + stale_note(status) for status in read_statuses(self.status_dir)] or [WAITING]

    def draw_block(self, lines, texts):
        """Replace the status lines on screen with `texts`, printing the new log lines above them. Each line is cut
        to the terminal's width, so that a wrapped line cannot leave the cursor below the block."""
        width = max(shutil.get_terminal_size((200, 24)).columns - 1, 20)
        if self.block:
            self.out.write("\r" + "\x1b[1A" * (self.block - 1) + "\x1b[J")   # back to the block's first line
        for line in lines:
            self.out.write(line + "\n")
        self.out.write("\n".join(text[:width] for text in texts))
        self.block = len(texts)

    def poll(self):
        """Print the run log's new lines as permanent lines, then refresh the status line."""
        with self.lock:
            lines = self.new_log_lines()
            texts = self.status_texts()
            text = texts[0]
            if self.live and self.status_dir is not None:
                self.draw_block(lines, texts)
            elif self.live:
                if lines and self.shown:
                    self.out.write("\r" + " " * self.shown + "\r")
                    self.shown = 0
                for line in lines:
                    self.out.write(line + "\n")
                self.out.write("\r" + text + " " * max(self.shown - len(text), 0))
                self.shown = len(text)
            else:
                for line in lines:
                    self.out.write(line + "\n")
                now = time.monotonic()
                if self.last_print is None or now - self.last_print >= self.period:
                    for text in texts:
                        self.out.write(text + "\n")
                    self.last_print = now
            self.out.flush()

    def _follow(self):
        while not self.stopping.wait(self.interval):
            self.poll()

    def start(self):
        """Poll now, then every `interval` seconds in a background thread."""
        self.poll()
        self.thread = threading.Thread(target=self._follow, daemon=True)
        self.thread.start()

    def finish_line(self):
        """End a live status line, so later output starts on a new line."""
        with self.lock:
            if self.live and (self.shown or self.block):
                self.out.write("\n")
                self.out.flush()
                self.shown = self.block = 0

    def stop(self):
        """Stop the background thread, poll a last time and end the live line."""
        self.stopping.set()
        if self.thread is not None:
            self.thread.join()
        self.poll()
        self.finish_line()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Show the progress of the running EC Racing experiment.")
    parser.add_argument("--status", default=None, help="status file (default: results/status.json)")
    parser.add_argument("--status-dir", default=None,
                        help="folder of runs trained side by side (default: results/status/ when it holds any)")
    parser.add_argument("--log", default=None, help="run log (default: results/run_log.txt)")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between updates")
    args = parser.parse_args(argv)
    log_path = Path(config.RUN_LOG_PATH if args.log is None else args.log)
    if log_path.exists():
        for line in log_path.read_text().splitlines()[-RECENT:]:
            print(line)
    status_dir = args.status_dir
    if status_dir is None and args.status is None and any(Path(config.STATUS_DIR).glob("*.json")):
        status_dir = config.STATUS_DIR
    monitor = Monitor(args.status, log_path, interval=args.interval, status_dir=status_dir)
    try:
        while True:
            monitor.poll()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.finish_line()
    return 0


if __name__ == "__main__":
    sys.exit(main())
