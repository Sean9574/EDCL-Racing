"""Validated tracks (spec §4.5, §4.6): generated geometry that the expert can drive, and the real circuits, with the
expert's lap time."""
import functools

from racing import config


@functools.lru_cache(maxsize=config.TRACK_CACHE_SIZE)
def get_track(seed, level, width=None):
    """The track for (seed, level) with its expert lap time (spec §4.5, §4.6): a generated map on levels 1-3 and the
    validation setting (with its own widths unless `width` gives an earlier revision's constant width), a real training
    layout at Extra hard and a test circuit at TEST_LEVEL. The most recent TRACK_CACHE_SIZE tracks are cached; callers
    must not modify a track."""
    from racing import circuits, expert, track_gen
    if level in (config.MAX_LEVEL, config.TEST_LEVEL):
        track = circuits.level_track(seed, level)
        lap_time = expert.expert_lap_time(track)
        if lap_time is None:
            raise RuntimeError(f"the expert cannot drive {track.name} (seed {seed}, level {level})")
        track.expert_lap_time = lap_time
        return track
    for attempt in range(config.MAX_TRACK_ATTEMPTS):
        track = track_gen.candidate_track(seed, level, attempt, width=width)
        if track is None:
            continue
        lap_time = expert.expert_lap_time(track)
        if lap_time is not None:
            track.expert_lap_time = lap_time
            return track
    raise RuntimeError(f"no drivable track for seed {seed}, level {level}")
