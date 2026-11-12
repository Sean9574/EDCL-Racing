"""Redraws the results figures (spec §11.4) from the saved training logs and tables, without new runs: the figures
of the evaluation except the networks figure, which needs the trained controllers. Prints the folder it reads and
each file it writes.
Usage: python scripts/redraw_figures.py [results folder, default results/]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racing import config, evaluate   # noqa: E402


def main(results_dir):
    results_dir = Path(results_dir)
    print("EC Racing: redraw the results figures")
    print(f"Reading {results_dir / 'runs'} and {results_dir / 'tables'}")
    written = evaluate.redraw_figures(results_dir)
    for path in written:
        print(f"Wrote {path}")
    print(f"{len(written)} figures written" if written else "No logs or tables found; no figure written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else config.RESULTS_DIR)
