"""Where each run, stage and final controller is stored (spec §3.3, §8.6, §9.1)."""
import shutil
from pathlib import Path

from racing import config


def run_dir(approach, seed, runs_dir=None):
    """Logs and summary of one training run: <runs>/<approach>/seed<k>."""
    return Path(config.RUNS_DIR if runs_dir is None else runs_dir) / approach / f"seed{seed}"


def run_seeds(approach, runs_dir=None):
    """The run seeds k of the folders <runs>/<approach>/seed<k> that exist, in order. An archive of an earlier
    revision may hold runs other than config.RUN_SEEDS (spec §3.3)."""
    folder = Path(config.RUNS_DIR if runs_dir is None else runs_dir) / approach
    if not folder.is_dir():
        return []
    return sorted(int(path.name[4:]) for path in folder.glob("seed*") if path.is_dir() and path.name[4:].isdigit())


def search_dir(approach, runs_dir=None):
    """The single Keras Tuner search of one method: <runs>/<approach>/search."""
    return Path(config.RUNS_DIR if runs_dir is None else runs_dir) / approach / "search"


STAGES = ("imitation", "dagger")


def stage_path(approach, seed, stage, runs_dir=None):
    """An imitation network after one stage of spec §8.6: <runs>/<approach>/seed<k>/<stage>.keras."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of: {', '.join(STAGES)}")
    return run_dir(approach, seed, runs_dir) / f"{stage}.keras"


def speed_dir(approach, seed, runs_dir=None):
    """The speed push of one imitation network (spec §8.5): <runs>/<approach>/seed<k>/speed."""
    return run_dir(approach, seed, runs_dir) / "speed"


def final_record(approach, seed, runs_dir=None):
    """The record a run writes when it is complete: an evolutionary run's summary, an imitation method's speed-push
    summary (spec §3.3)."""
    if approach in config.EVOLUTION_APPROACHES:
        return run_dir(approach, seed, runs_dir) / "summary.json"
    return speed_dir(approach, seed, runs_dir) / "summary.json"


REPLAY_NAME = "replay.pkl"


def replay_path(approach, seed, runs_dir=None):
    """The latest evaluated generation of a reported run, kept for the viewer: in the run folder of an evolutionary
    approach, in the speed push folder of an imitation approach (its evolutionary stage)."""
    if approach in config.EVOLUTION_APPROACHES:
        return run_dir(approach, seed, runs_dir) / REPLAY_NAME
    return speed_dir(approach, seed, runs_dir) / REPLAY_NAME


def champion_path(approach, seed, suffix=".keras", champion_dir=None):
    """One final controller: <champions>/<approach>/seed<k>.keras, or the .npy weights next to it."""
    return Path(config.CHAMPIONS_DIR if champion_dir is None else champion_dir) / approach / f"seed{seed}{suffix}"


def best_seeds_path(champion_dir=None):
    """The record of each approach's best run seed, written by the evaluation (spec §11)."""
    return Path(config.CHAMPIONS_DIR if champion_dir is None else champion_dir) / config.BEST_SEEDS_NAME


def remove(*targets):
    """Delete the files and directories that exist; missing ones are ignored."""
    for target in targets:
        target = Path(target)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
