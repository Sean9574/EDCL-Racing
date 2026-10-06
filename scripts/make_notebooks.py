"""Writes the five experiment notebooks as plain nbformat-4 JSON (no extra dependency).

Run with:  ~/miniconda3/bin/python scripts/make_notebooks.py
"""
import json
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent / "notebooks"
SETUP = "import sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path.cwd().parent))"

NOTEBOOKS = {
    "01_dataset.ipynb": [
        ("markdown", "# 1. Expert dataset\nCollects the imitation-learning dataset (design specification §7)."),
        ("code", SETUP),
        ("code", "from racing.expert import collect_dataset\n\ndata = collect_dataset()\n"
                 "{key: value.shape for key, value in data.items()}"),
    ],
    "02_imitation.ipynb": [
        ("markdown", "# 2. Imitation-learning controllers\nBaseline and Keras Tuner searches (design specification §8)."),
        ("code", SETUP),
        ("code", "from racing import imitation\n\ndata = imitation.load_dataset()\nimitation.train_baseline(data)"),
        ("code", "for approach in ['kt_random', 'kt_hyperband', 'kt_bayesian']:\n"
                 "    print(imitation.run_tuner(approach, data))"),
    ],
    "03_ga.ipynb": [
        ("markdown", "# 3. Genetic algorithm\nNeuroevolution with the curriculum (design specification §9.2)."),
        ("code", SETUP),
        ("code", "from racing import config, ga\n\n\ndef progress(report):\n"
                 "    if report['generation'] % 10 == 0 or report['leveled_up']:\n"
                 "        print(f\"gen {report['generation']:4d}  level {report['level']}  best {report['best']:.3f}  \"\n"
                 "              f\"mean {report['mean']:.3f}  new finish {report['new_finish']:.2f}\")\n\n\n"
                 "ga.run_ga(on_generation=progress, n_jobs=config.N_JOBS)"),
    ],
    "04_es.ipynb": [
        ("markdown", "# 4. Evolution strategy\n(μ+λ)-ES with the 1/5 success rule (design specification §9.3)."),
        ("code", SETUP),
        ("code", "from racing import config, es\n\n\ndef progress(report):\n"
                 "    if report['generation'] % 10 == 0 or report['leveled_up']:\n"
                 "        print(f\"gen {report['generation']:4d}  level {report['level']}  best {report['best']:.3f}  \"\n"
                 "              f\"mean {report['mean']:.3f}  sigma {report['step_size']:.4f}\")\n\n\n"
                 "es.run_es(on_generation=progress, n_jobs=config.N_JOBS)"),
    ],
    "05_evaluate.ipynb": [
        ("markdown", "# 5. Evaluation\nTest tracks, test accuracy, size and inference time (design specification §11)."),
        ("code", SETUP),
        ("code", "from racing import evaluate\n\nresults = evaluate.main()\nresults['comparison']"),
        ("code", "results['driving']"),
        ("code", "from IPython.display import Image, display\n\nfor path in results['figures']:\n"
                 "    display(Image(filename=str(path)))"),
    ],
}


def cell(kind, source):
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def main():
    FOLDER.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        notebook = {"cells": [cell(kind, source) for kind, source in cells],
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python"}},
                    "nbformat": 4, "nbformat_minor": 5}
        (FOLDER / name).write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    main()
