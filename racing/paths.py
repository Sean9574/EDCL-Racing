"""Where each run and final controller is stored (spec §3.3, §9.1): one directory or file per approach and seed."""
import shutil
from pathlib import Path

from racing import config


def run_dir(approach, seed, runs_dir=None):
    """Logs and summary of one training run: <runs>/<approach>/seed<k>."""
    return Path(config.RUNS_DIR if runs_dir is None else runs_dir) / approach / f"seed{seed}"


def search_dir(approach, runs_dir=None):
    """The single Keras Tuner search of one method: <runs>/<approach>/search."""
    return Path(config.RUNS_DIR if runs_dir is None else runs_dir) / approach / "search"


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
