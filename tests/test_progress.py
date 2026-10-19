import io
import json
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from racing import config, progress


def status(**changes):
    base = {"approach": "ga", "seed": 1, "stage": "evolution", "level": 3, "generation": 412, "min_share": 0.84,
            "streak": 1, "target_reduction": float("nan"), "elapsed": 2 * 3600 + 13 * 60}
    return {**base, **changes}


@pytest.fixture
def order(monkeypatch):
    monkeypatch.setattr(config, "RUN_ORDER", tuple((a, s) for a in ("baseline", "ga") for s in (0, 1, 2)))


def test_status_round_trip_adds_the_time(tmp_path):
    path = tmp_path / "status.json"
    progress.write_status(status(), path)
    read = progress.read_status(path)
    assert read["generation"] == 412 and math.isnan(read["target_reduction"]) and "updated" in read
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_status_defaults_to_the_configured_file():
    progress.write_status(status())
    assert config.STATUS_PATH.exists() and progress.read_status()["approach"] == "ga"


def test_status_accepts_numpy_values(tmp_path):
    progress.write_status(status(generation=np.int64(5), min_share=np.float64(0.5), streak=np.int64(2)),
                          tmp_path / "s.json")
    assert progress.read_status(tmp_path / "s.json")["generation"] == 5


def test_missing_or_unreadable_status_is_none(tmp_path):
    assert progress.read_status(tmp_path / "missing.json") is None
    (tmp_path / "broken.json").write_text('{"approach": "ga", "gen')
    assert progress.read_status(tmp_path / "broken.json") is None
    (tmp_path / "list.json").write_text("[1, 2]")
    assert progress.read_status(tmp_path / "list.json") is None


def test_milestones_are_appended_with_a_time(tmp_path):
    path = tmp_path / "log.txt"
    first = progress.milestone("GA seed 0: started", path)
    progress.milestone("GA seed 0: Easy mastered at generation 41, now Medium", path)
    lines = path.read_text().splitlines()
    assert lines[0] == first and lines[0].endswith("  GA seed 0: started") and len(lines) == 2
    assert lines[0][4] == "-" and lines[0][10] == "T"                     # ISO date and time


def test_milestones_default_to_the_run_log():
    progress.milestone("NEAT seed 2: started")
    assert config.RUN_LOG_PATH.read_text().endswith("NEAT seed 2: started\n")


def test_run_number_follows_the_run_order(order):
    assert progress.run_number("baseline", 0) == (1, 6)
    assert progress.run_number("ga", 1) == (5, 6)
    assert progress.run_number("neat", 0) == (0, 6)                        # not part of this experiment


def test_bar_has_a_fixed_width_and_eighth_steps():
    assert progress.bar(1.0, 4) == "████" and progress.bar(0.0, 4) == "    "
    assert progress.bar(0.5, 4) == "██  " and progress.bar(0.0625, 4) == "▎   "
    assert progress.bar(2.0, 4) == "████" and progress.bar(float("nan"), 4) == "    "
    assert all(len(progress.bar(k / 37, 20)) == 20 for k in range(38))


def test_duration():
    assert progress.duration(2470) == "0h 41m" and progress.duration(7980) == "2h 13m"
    assert progress.duration(90061) == "1d 1h 01m" and progress.duration(None) == "–"


def test_evolution_line(order):
    line = progress.format_status(status())
    assert line.startswith("[run 5/6] GA run 1 · Hard · gen 412 · worst map 84% / 90% ")
    assert progress.bar(0.84 / config.MASTERY_FRACTION) in line
    assert line.endswith(" · streak 1/3 · 2h 13m")


def test_speed_push_line(order):
    line = progress.format_status(status(stage="speed push", level=4, target_reduction=0.143, elapsed=90061))
    assert "GA run 1 · Extra hard · speed push −14.3% · gen 412 · worst map" in line
    assert line.endswith("1d 1h 01m")


def test_dagger_line(order):
    line = progress.format_status({"approach": "baseline", "seed": 1, "stage": "dagger", "level": 4, "round": 7,
                                   "checks": 16, "streak": 1, "elapsed": 2520})
    assert line == "[run 2/6] Baseline run 1 · DAgger round 7 · checks 16/20 · streak 1/3 · 0h 42m"


def test_imitation_line(order):
    line = progress.format_status({"approach": "baseline", "seed": 1, "stage": "imitation", "epoch": 12,
                                   "val_accuracy": 0.9123, "best": 0.9156, "elapsed": 190})
    assert line == "[run 2/6] Baseline run 1 · imitation epoch 12 · val acc 91.2% (best 91.6%) · 0h 03m"


def test_search_line(order):
    line = progress.format_status({"approach": "kt_random", "stage": "search", "trial": 7, "trials": 30,
                                   "best": 0.9214, "elapsed": 720})
    assert line == "[search] KT Random · trial 7/30 · best val acc 92.1% · 0h 12m"
    hyperband = progress.format_status({"approach": "kt_hyperband", "stage": "search", "trial": 0, "elapsed": 5})
    assert hyperband == "[search] KT Hyperband · trial 0 · best val acc – · 0h 00m"


def test_incomplete_status_still_formats(order):
    assert progress.format_status({"approach": "ga"}).startswith("[run –/6] GA run ?")
    assert progress.format_status({"approach": "ga", "seed": 0, "min_share": None, "level": None})
    assert progress.format_status({})


def make_monitor(tmp_path, live, period=60.0, interval=1.0):
    out = io.StringIO()
    monitor = progress.Monitor(tmp_path / "status.json", tmp_path / "log.txt", out=out, live=live, period=period,
                               interval=interval)
    return monitor, out


def test_monitor_prints_new_log_lines_then_the_live_line(tmp_path, order):
    (tmp_path / "log.txt").write_text("old line\n")
    monitor, out = make_monitor(tmp_path, live=True)
    progress.milestone("GA seed 1: started", tmp_path / "log.txt")
    progress.write_status(status(), tmp_path / "status.json")
    monitor.poll()
    text = out.getvalue()
    assert "old line" not in text and "GA seed 1: started\n" in text
    assert text.endswith("\r" + progress.format_status(status()))
    progress.milestone("GA seed 1: Hard mastered at generation 500, now Extra hard", tmp_path / "log.txt")
    monitor.poll()
    later = out.getvalue()[len(text):]
    assert later.startswith("\r") and "Hard mastered" in later and "started" not in later


def test_without_a_terminal_the_status_is_printed_once_per_period(tmp_path, order):
    progress.write_status(status(), tmp_path / "status.json")
    monitor, out = make_monitor(tmp_path, live=False, period=60.0)
    monitor.poll()
    monitor.poll()
    assert out.getvalue() == progress.format_status(status()) + "\n"
    every, out_every = make_monitor(tmp_path, live=False, period=0.0)
    every.poll()
    every.poll()
    assert out_every.getvalue().count("\n") == 2 and "\r" not in out_every.getvalue()


def test_live_mode_follows_the_terminal():
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    assert progress.Monitor(out=Terminal()).live and not progress.Monitor(out=io.StringIO()).live


def test_monitor_survives_missing_and_broken_status(tmp_path):
    monitor, out = make_monitor(tmp_path, live=False, period=0.0)
    monitor.poll()                                                         # no status file and no run log yet
    (tmp_path / "status.json").write_text("{not json")
    monitor.poll()
    assert out.getvalue().count("waiting for the first status") == 2


def test_monitor_follows_a_replaced_log(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("one\ntwo\n")
    monitor, out = make_monitor(tmp_path, live=False, period=3600.0)
    log.write_text("fresh\n")                                              # --fresh removed the old run log
    monitor.poll()
    assert "fresh\n" in out.getvalue() and "two" not in out.getvalue()


def test_a_partly_written_log_line_waits_for_its_end(tmp_path):
    monitor, out = make_monitor(tmp_path, live=False, period=3600.0)
    with open(tmp_path / "log.txt", "a") as f:
        f.write("half")
    monitor.poll()
    with open(tmp_path / "log.txt", "a") as f:
        f.write(" line\n")
    monitor.poll()
    assert "half line\n" in out.getvalue() and out.getvalue().count("half") == 1


def test_start_and_stop_end_the_live_line(tmp_path, order):
    progress.write_status(status(), tmp_path / "status.json")
    monitor, out = make_monitor(tmp_path, live=True, interval=0.01)
    monitor.start()
    time.sleep(0.05)
    monitor.stop()
    assert out.getvalue().endswith("\n") and progress.format_status(status()) in out.getvalue()
    assert not monitor.thread.is_alive()


def test_main_shows_recent_milestones_then_the_status_until_interrupted(tmp_path, monkeypatch, capsys, order):
    log = tmp_path / "log.txt"
    log.write_text("".join(f"line {k}\n" for k in range(15)))
    progress.write_status(status(), tmp_path / "status.json")

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(progress.time, "sleep", interrupt)
    assert progress.main(["--status", str(tmp_path / "status.json"), "--log", str(log)]) == 0
    printed = capsys.readouterr().out
    assert "line 4\n" not in printed and "line 5\n" in printed and "line 14\n" in printed
    assert progress.format_status(status()) in printed


def written_ago(tmp_path, minutes):
    """A status file last written `minutes` ago, as a run that stopped leaves it."""
    written = status(updated=(datetime.now() - timedelta(minutes=minutes)).isoformat(timespec="seconds"))
    (tmp_path / "status.json").write_text(json.dumps(written))
    return written


@pytest.mark.parametrize("live", [True, False])
def test_a_status_without_updates_for_a_while_says_so(tmp_path, order, live):
    """A run that died leaves its last status behind; the monitor must not show it as running."""
    written = written_ago(tmp_path, 25)
    monitor, out = make_monitor(tmp_path, live=live, period=0.0)
    monitor.poll()
    assert progress.format_status(written) + " (no update for 25 min)" in out.getvalue()


def test_a_recent_status_has_no_note(tmp_path, order):
    written = written_ago(tmp_path, config.STATUS_STALE_AFTER / 60 - 2)
    monitor, out = make_monitor(tmp_path, live=False, period=0.0)
    monitor.poll()
    assert out.getvalue() == progress.format_status(written) + "\n"
    assert progress.stale_note({}) == progress.stale_note({"updated": "not a time"}) == ""



# ----- runs trained side by side: one status file each -----

def side_by_side_statuses(folder):
    folder.mkdir(exist_ok=True)
    progress.write_status(status(seed=2, generation=7), folder / "ga-seed2.json")
    progress.write_status(status(approach="baseline", seed=0, generation=3), folder / "baseline-seed0.json")
    (folder / "broken.json").write_text("{half")
    return [status(approach="baseline", seed=0, generation=3), status(seed=2, generation=7)]


def test_statuses_of_runs_side_by_side_are_read_in_run_order(tmp_path, order):
    expected = side_by_side_statuses(tmp_path / "status")
    read = progress.read_statuses(tmp_path / "status")
    assert [(s["approach"], s["seed"]) for s in read] == [(s["approach"], s["seed"]) for s in expected]
    assert progress.read_statuses(tmp_path / "none") == []


def test_without_a_terminal_each_run_side_by_side_gets_a_line(tmp_path, order):
    expected = side_by_side_statuses(tmp_path / "status")
    out = io.StringIO()
    monitor = progress.Monitor(None, tmp_path / "log.txt", out=out, live=False, period=0.0,
                               status_dir=tmp_path / "status")
    monitor.poll()
    assert out.getvalue().splitlines() == [progress.format_status(s) for s in expected]


def test_live_lines_of_runs_side_by_side_are_redrawn_in_place(tmp_path, order):
    expected = side_by_side_statuses(tmp_path / "status")
    out = io.StringIO()
    monitor = progress.Monitor(None, tmp_path / "log.txt", out=out, live=True, status_dir=tmp_path / "status")
    monitor.poll()
    first = out.getvalue()
    assert first == "\n".join(progress.format_status(s) for s in expected)
    progress.milestone("GA seed 2: Hard mastered at generation 7, now Extra hard", tmp_path / "log.txt")
    monitor.poll()
    later = out.getvalue()[len(first):]
    assert later.startswith("\r\x1b[1A\x1b[J") and "Hard mastered at generation 7, now Extra hard\n" in later
    assert later.endswith(progress.format_status(expected[1]))
    monitor.finish_line()
    assert out.getvalue().endswith("\n") and monitor.block == 0


def test_runs_side_by_side_wait_for_their_first_status(tmp_path):
    out = io.StringIO()
    monitor = progress.Monitor(None, tmp_path / "log.txt", out=out, live=False, period=0.0,
                               status_dir=tmp_path / "status")
    monitor.poll()
    assert out.getvalue() == progress.WAITING + "\n"


def test_main_follows_runs_side_by_side_when_there_are_any(tmp_path, monkeypatch, capsys, order):
    expected = side_by_side_statuses(config.STATUS_DIR)

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(progress.time, "sleep", interrupt)
    assert progress.main(["--log", str(tmp_path / "log.txt")]) == 0
    printed = capsys.readouterr().out
    assert all(progress.format_status(s) in printed for s in expected)


def test_the_speed_push_line_shows_its_75_percent_requirement(order):
    push = progress.format_status(status(stage="speed push", target_reduction=0.1, level=4))
    levels = progress.format_status(status())
    assert "/ 75%" in push and "/ 90%" not in push and "/ 90%" in levels
