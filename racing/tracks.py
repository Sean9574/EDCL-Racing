"""Validated tracks (spec §4.3): generated geometry that the expert can drive, with its lap time."""
import functools

from racing import config


@functools.lru_cache(maxsize=config.TRACK_CACHE_SIZE)
def get_track(seed, level):
    """The track for (seed, level). The most recent config.TRACK_CACHE_SIZE tracks are cached, so fresh maps every
    generation do not accumulate (revision 3 §3.4); callers must not modify a track."""
    from racing import expert, track_gen
    for attempt in range(config.MAX_TRACK_ATTEMPTS):
        track = track_gen.candidate_track(seed, level, attempt)
        if track is None:
            continue
        lap_time = expert.expert_lap_time(track)
        if lap_time is not None:
            track.expert_lap_time = lap_time
            return track
    raise RuntimeError(f"no drivable track for seed {seed}, level {level}")
