"""Validated tracks (spec §4.5): generated geometry that the expert can drive, with its lap time."""
import functools

from racing import config


@functools.lru_cache(maxsize=config.TRACK_CACHE_SIZE)
def get_track(seed, level, width=None):
    """The track for (seed, level), with its own road width unless `width` gives another (an earlier revision's maps,
    spec §4.2). The most recent config.TRACK_CACHE_SIZE tracks are cached, so fresh maps every generation do not
    accumulate (spec §9.10); callers must not modify a track."""
    from racing import expert, track_gen
    for attempt in range(config.MAX_TRACK_ATTEMPTS):
        track = track_gen.candidate_track(seed, level, attempt, width=width)
        if track is None:
            continue
        lap_time = expert.expert_lap_time(track)
        if lap_time is not None:
            track.expert_lap_time = lap_time
            return track
    raise RuntimeError(f"no drivable track for seed {seed}, level {level}")
