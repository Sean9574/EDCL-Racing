"""Measures the expert's grip share (spec §7.1): the highest share, from 1.0 down in steps of 0.01, at which the expert
finishes every real layout (the 38 training layouts and the 4 test circuits) for config.LAPS laps, as the slow test of
the real tracks requires (spec §12.1), and every validation map and generated dataset map builds. The generated maps
are built only at a share where every real layout is finished: a map the expert cannot drive takes all
MAX_TRACK_ATTEMPTS candidates, and the share fails anyway. Prints each share tried with its failures, then the value
for config.EXPERT_GRIP_MARGIN.
Usage: python scripts/expert_grip.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racing import circuits, config, expert, fastsim, sim, tracks   # noqa: E402


def drives(track):
    """Whether the expert completes config.LAPS laps of `track`, on the compiled simulator when it is available."""
    limit = config.LAPS * config.EXPERT_VALIDATION_TIME
    if fastsim.available():
        return bool(fastsim.simulate_expert([track], 1, [limit], laps=config.LAPS)[0].finished[0])
    return bool(sim.simulate(track, expert.expert_policy(track), 1, time_limit=limit, laps=config.LAPS).finished[0])


def failures(share):
    """The maps the expert cannot drive at grip share `share`: the real layouts it does not finish, else the generated
    maps that do not build."""
    config.EXPERT_GRIP_MARGIN = share
    tracks.get_track.cache_clear()
    failed = []
    real = ([(k, config.MAX_LEVEL) for k in range(circuits.N_LAYOUTS)]
            + [(k, config.TEST_LEVEL) for k in config.TEST_TRACKS])
    for seed, level in real:
        track = circuits.level_track(seed, level)
        if not drives(track):
            failed.append(track.name + ("" if level == config.TEST_LEVEL else f" {seed}"))
    if failed:
        return failed
    generated = [(seed, config.VALIDATION_LEVEL) for seed in config.VAL_SEEDS]
    generated += [(seed, level) for seed, level in expert.dataset_seeds() if level < config.MAX_LEVEL]
    for seed, level in generated:
        try:
            tracks.get_track(seed, level)
        except RuntimeError:
            failed.append(f"level {level} seed {seed}")
    return failed


def main():
    for hundredths in range(100, 49, -1):
        share = hundredths / 100
        failed = failures(share)
        print(f"grip share {share:.2f}: {len(failed)} maps not driven {failed[:6]}", flush=True)
        if not failed:
            print(f"EXPERT_GRIP_MARGIN = {share:.2f}")
            return 0
    print("no share from 1.00 to 0.50 drives every map")
    return 1


if __name__ == "__main__":
    sys.exit(main())
