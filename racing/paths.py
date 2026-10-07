"""Where each run, stage and final controller is stored (spec §3.3, §9.1; revision 3 §6.5)."""
import shutil
from pathlib import Path

from racing import config


def run_dir(approach, seed, runs_dir=None):
    """Logs and summary of one training run: <runs>/<approach>/seed<k>."""
    return Path(config.RUNS_DIR if runs_dir is None else runs_dir) / approach / f"seed{seed}"


def search_dir(approach, runs_dir=None):
    """The single Keras Tuner search of one method: <runs>/<approach>/search."""
    return Path(config.RUNS_DIR if runs_dir is None else runs_dir) / approach / "search"


STAGES = ("imitation", "dagger")


def stage_path(approach, seed, stage, runs_dir=None):
    """An imitation network after one stage of revision 3 §6.5: <runs>/<approach>/seed<k>/<stage>.keras."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of: {', '.join(STAGES)}")
    return run_dir(approach, seed, runs_dir) / f"{stage}.keras"


def speed_dir(approach, seed, runs_dir=None):
    """The speed push of one imitation network (revision 3 §6.4): <runs>/<approach>/seed<k>/speed."""
    return run_dir(approach, seed, runs_dir) / "speed"


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
