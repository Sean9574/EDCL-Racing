"""What the browser page shows of the viewer (spec §10): the scene geometry, the panel and each frame, as plain JSON
types. The page draws; the App decides."""
import numpy as np

from racing import app as viewer
from racing import circuits, config, curriculum, netfig

GHOST_COLOR = "#222630"         # the pole ghost: a dark neutral car, never a method's color


def _round(values, digits):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def _clean(value, digits=2):
    """A float rounded for the page, None for nan."""
    return None if value is None or np.isnan(value) else round(float(value), digits)


def scene_key(track):
    """Names a track's geometry: seed, level, circuit, length and road width range (an archive's constant-width maps
    share the centreline of this revision's maps)."""
    return (f"{track.seed}-{track.level}-{track.name}-{track.length:.1f}"
            f"-{track.widths.min():.1f}-{track.widths.max():.1f}")


def scene_state(track):
    """A track's geometry about SCENE_SPACING m apart: the road edges, the racing line and the start line."""
    step = max(1, round(config.SCENE_SPACING / track.spacing))
    name = circuits.NAMES.get(track.name, track.name) if track.name else f"Seed {track.seed}, level {track.level}"
    return {"key": scene_key(track), "name": name,
            "left": _round(track.left[::step], 2), "right": _round(track.right[::step], 2),
            "line": _round(track.line[::step], 2),
            "start": _round([track.left[0], track.right[0]], 2)}


def shown_track(app):
    """The track on the page: the race's, or the training view's."""
    if app.mode == "race":
        return app.race.track if app.race is not None else app.race_track()
    return app.shown_track()


def _focus(app, playback):
    """The index of the car the page follows."""
    count = len(playback.cars.x)
    if app.mode == "train":
        return min(app.best_slot, max(count - 1, 0))
    cars = app.race_cars()
    car = app.followed_car()
    return cars.index(car) if car in cars and cars.index(car) < count else 0


def _rays(app, playback, focus):
    cars = playback.cars
    if app.mode == "train":
        wanted = np.array([focus]) if len(cars.x) and not cars.done[focus] else np.array([], dtype=int)
    else:
        wanted = np.flatnonzero(~cars.done)
    if not wanted.size:
        return {}
    lengths = viewer.sim.cast_rays(playback.track, cars.x[wanted], cars.y[wanted], cars.heading[wanted],
                                   lengths=cars.ray_length[wanted])
    return {str(int(i)): _round(row, 1) for i, row in zip(wanted, lengths)}


def _sensors(playback, rays, focus):
    """The sensor lengths (m) of the cars whose rays are sent and of the focus car, keyed like `rays`: a ray this long
    met no road edge."""
    cars = playback.cars
    shown = {int(i) for i in rays} | ({focus} if len(cars.x) else set())
    return {str(i): round(float(cars.ray_length[i]), 1) for i in sorted(shown)}


def frame_state(app):
    """The cars now: positions, speeds, headings, who is done or crashed, the followed car's action, rays and sensor
    length, and the pole ghost. None without a track on show; with a track but no cars yet the car lists are empty."""
    track = shown_track(app)
    if track is None:
        return None
    playback = app.race if app.mode == "race" else app.playback
    frame = {"scene": scene_key(track), "mode": app.mode, "t": 0.0, "x": [], "y": [], "v": [], "h": [],
             "done": [], "crashed": [], "focus": 0, "rays": {}, "sensor": {}, "action": "", "ghost": None}
    if playback is None or playback.track is not track:
        return frame
    cars = playback.cars
    focus = _focus(app, playback)
    action = int(playback.actions[focus]) if len(cars.x) else -1
    rays = _rays(app, playback, focus)
    frame.update(t=round(float(playback.elapsed), 2), x=_round(cars.x, 2), y=_round(cars.y, 2),
                 v=_round(cars.speed, 2), h=_round(cars.heading, 3), done=cars.done.astype(int).tolist(),
                 crashed=cars.crashed.astype(int).tolist(), focus=focus, rays=rays,
                 sensor=_sensors(playback, rays, focus),
                 action=netfig.output_labels()[action] if action >= 0 else "")
    if app.mode == "race" and app.ghost is not None:
        g = app.ghost.state(playback.elapsed)
        frame["ghost"] = [round(g.x, 2), round(g.y, 2), round(g.heading, 3)]
    return frame


def stats_rows(report, approach):
    """The training statistics as [label, value] rows, the same values as the App's stats_lines."""
    if "min_share" not in report:
        unit = "Trial" if approach.startswith("kt") else "Epoch"
        return [[unit, str(report["generation"])], ["Best", f"{report['best']:.3f}"],
                ["Mean", f"{report['mean']:.3f}"]]
    reduction = float(report.get("target_reduction", float("nan")))
    need = viewer.percent(curriculum.mastery_fraction(report.get("speed_push", False)))
    rows = [["Difficulty", config.DIFFICULTIES.get(report["level"], f"Level {report['level']}")],
            ["Generation", str(report["generation"])],
            ["Maps succeeded", f"{viewer.share_text(report['map_shares'])}  (need {need})"],
            ["Streak", f"{report['streak']}/{config.MASTERY_STREAK}"],
            ["Target cut", "–" if np.isnan(reduction) else f"−{100 * reduction:.1f}%"],
            ["Mutation strength" if approach == "ga" else "σ", viewer.number(report["step_size"])],
            ["Kept", str(int(np.sum(report["is_kept"])))],
            ["Best", f"{report['best']:.3f}"], ["Mean", f"{report['mean']:.3f}"]]
    if "species" in report:
        rows.append(["Species", str(report["species"])])
    if "sensor_mean" in report:
        rows += [["Sensors", f"{report['sensor_mean']:.0f} m ± {report['sensor_std']:.0f}"],
                 ["Steering", f"×{report['steer_mean']:.2f} ± {report['steer_std']:.2f}"]]
    return rows


def _choices(names):
    return [[int(level), name] for level, name in names.items()]


def _history(app):
    shares = app.latest is not None and "min_share" in app.latest
    points = [[int(g), _clean(value, 4), _clean(second, 4), viewer.MARKED_EVENTS.get(event)]
              for g, value, second, event in app.history]
    need = float(curriculum.mastery_fraction(app.latest.get("speed_push", False))) if shares else None
    return {"kind": "share" if shares else "accuracy", "points": points, "need": need}


def _leaderboard(app, pole):
    cars = app.race_cars()
    rows = []
    for position, i, laps, last, best, status, gap in viewer.leaderboard(app.race, len(cars), pole, ghost=app.ghost):
        if i == viewer.GHOST:
            name, color, car = app.ghost.name, GHOST_COLOR, "pole"
        else:
            name = viewer.car_name(cars[i][0], cars[i][1], viewer.is_app(cars[i]), app.archived())
            color, car = config.COLORS[cars[i][0]], int(i)
        rows.append({"position": int(position), "car": car, "name": name, "color": color, "laps": int(laps),
                     "last": _clean(last), "best": _clean(best), "status": status,
                     "gap": gap if pole is not None else None})
    return rows


def panel_state(app):
    """Everything the panel shows and which of its controls are usable, as plain JSON types, by the rules of the old
    panel. Every key is always present; the page reads the training keys in the training view and the race keys in
    the race view."""
    idle = app.process is None
    notebook = app.source == "notebook"
    racing = bool(app.racing)
    seeded = app.run_seed == viewer.SEEDED_RUN and not (notebook and app.archived())
    playing = (not app.replay_paused) if notebook else (
        app.process is not None and app.running is not None and app.running.is_set())
    size = app.population_size()
    cars = app.race_cars()
    pole = app.pole_lap()
    followed = cars.index(app.follow) if app.follow in cars else None
    note = None
    if app.latest is None:
        note = ("Saved run found: Play resumes, New run restarts."
                if app.source == "app" and app.process is None and app.has_saved_run()
                else "No generation completed yet.")
    stats = stats_rows(app.latest, app.latest["approach"]) if app.latest is not None else []
    return {
        "mode": app.mode, "status": app.status, "source": app.source, "approach": app.approach,
        "color": config.COLORS[app.approach],
        "approaches": [{"id": a, "label": config.LABELS[a], "color": config.COLORS[a], "active": a == app.approach,
                        "enabled": idle} for a in config.APPROACHES],
        "folder": {"label": app.folder_label(),
                   "enabled": (not racing) if app.mode == "race" else idle and notebook},
        "run": {"label": f"{app.run_seed} (seeded)" if seeded else str(app.run_seed), "enabled": idle},
        "difficulty": {"choices": _choices(config.DIFFICULTIES), "active": int(app.difficulty),
                       "enabled": idle and not notebook},
        "population": (None if app.approach not in app.population or notebook
                       else {"value": int(app.population[app.approach]), "enabled": idle}),
        "play": {"label": "Pause" if playing else "Play", "active": bool(playing)},
        "stop": {"enabled": racing if app.mode == "race" else (not idle and not notebook)},
        "new_run": {"enabled": idle and not notebook},
        "show": app.show, "shown": int(min(app.shown, size)), "size": int(size),
        "speed": int(app.speed), "speeds": [int(s) for s in config.SPEEDS],
        "grip": {"name": app.grip_name, "label": viewer.GRIP_LABELS[app.grip_name], "enabled": not racing},
        "network_open": app.net_process is not None,
        "stats": [["Map", app.map_text().removeprefix("Map: ")], *stats],
        "note": note,
        "history": _history(app),
        "track": {"name": app.race_name(), "enabled": not racing},
        "race_level": {"choices": _choices(config.APP_DIFFICULTIES), "active": int(app.race_level),
                       "enabled": not racing},
        "racing": racing,
        "paused": bool(app.race_paused),
        "start": {"enabled": not racing and bool(cars)},
        "race_runs": {"value": app.race_runs, "enabled": not racing},
        "group": {"value": app.race_group, "available": bool(app.app_available)},
        "methods": [{"id": a, "label": config.LABELS[a], "color": config.COLORS[a], "on": bool(app.race_on.get(a, True)),
                     "enabled": not racing} for a in app.race_available],
        "app_choices": [{"id": a, "label": f"{config.LABELS[a]} (app)", "color": config.COLORS[a],
                         "choice": app.app_choice_label(a), "enabled": not racing} for a in app.app_available],
        "cars": [{"name": viewer.car_name(c[0], c[1], viewer.is_app(c), app.archived()), "color": config.COLORS[c[0]],
                  "stripe": not viewer.is_app(c) and c[1] != viewer.SEEDED_RUN and not app.archived(),
                  "app": viewer.is_app(c)} for c in cars],
        "leaderboard": _leaderboard(app, pole),
        "pole": None if pole is None else float(pole),
        "follow": followed,
    }
