"""Writes the six experiment notebooks as plain nbformat-4 JSON (no extra dependency).

Run with:  ~/miniconda3/bin/python scripts/make_notebooks.py
"""
import json
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent / "notebooks"
SETUP = "import sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path.cwd().parent))"

EVOLUTION_PROGRESS = (
    "def progress(report):\n"
    "    if report['generation'] % 50 == 0 or report['event']:\n"
    "        print(f\"gen {report['generation']:5d}  level {report['level']}  best {report['best']:.3f}  \"\n"
    "              f\"worst map {report['min_share']:.2f}  step {report['step_size']:.4f}  {report['event'] or ''}\")\n")

NOTEBOOKS = {
    "01_dataset.ipynb": [
        ("markdown", "# 1. Expert dataset\nCollects the imitation-learning dataset (design specification §7)."),
        ("code", SETUP),
        ("code", "import json\n\nimport pandas as pd\n\nfrom racing import config\nfrom racing.expert import "
                 "collect_dataset\n\ndata, summary = collect_dataset(n_jobs=config.N_JOBS)\n"
                 "print('collection time (s):', json.loads(config.DATASET_INFO_PATH.read_text())['wall_time'])\n"
                 "pd.DataFrame(summary).T"),
    ],
    "02_imitation.ipynb": [
        ("markdown", "# 2. Imitation-learning controllers\nBaseline with each run seed, the three Keras Tuner "
                     "searches, and retraining of each selected architecture with each run seed (design "
                     "specification §8)."),
        ("code", SETUP),
        ("code", "from racing import config, imitation\n\ndata = imitation.load_dataset()\n"
                 "for seed in config.RUN_SEEDS:\n    print(imitation.train_baseline(data, seed=seed))"),
        ("code", "for approach in config.TUNER_APPROACHES:\n    summary = imitation.run_tuner(approach, data)\n"
                 "    print({k: v for k, v in summary.items() if k != 'chosen_hyperparameters'})\n"
                 "    for seed in config.RUN_SEEDS:\n        print(imitation.retrain(approach, data, seed=seed))"),
    ],
    "03_ga.ipynb": [
        ("markdown", "# 3. Genetic algorithm\nNeuroevolution with the curriculum, once per run seed (design "
                     "specification §9.2)."),
        ("code", SETUP),
        ("code", "from racing import config, ga\n\n\n" + EVOLUTION_PROGRESS + "\n\n"
                 "for seed in config.RUN_SEEDS:\n    print(f'--- seed {seed}')\n"
                 "    print(ga.run_ga(seed=seed, on_generation=progress, n_jobs=config.N_JOBS))"),
    ],
    "04_es.ipynb": [
        ("markdown", "# 4. Evolution strategy\n(μ+λ)-ES with the 1/5 success rule, once per run seed (design "
                     "specification §9.3)."),
        ("code", SETUP),
        ("code", "from racing import config, es\n\n\n" + EVOLUTION_PROGRESS + "\n\n"
                 "for seed in config.RUN_SEEDS:\n    print(f'--- seed {seed}')\n"
                 "    print(es.run_es(seed=seed, on_generation=progress, n_jobs=config.N_JOBS))"),
    ],
    "05_es_seeded.ipynb": [
        ("markdown", "# 5. Seeded evolution strategy\nThe ES started from the baseline trained with the same run "
                     "seed (design specification §9.4)."),
        ("code", SETUP),
        ("code", "from racing import config, es\n\n\n" + EVOLUTION_PROGRESS + "\n\n"
                 "for seed in config.RUN_SEEDS:\n    print(f'--- seed {seed}')\n"
                 "    print(es.run_es(seed=seed, initial=es.trained_baseline(seed), approach='es_seeded',\n"
                 "                    on_generation=progress, n_jobs=config.N_JOBS))"),
    ],
    "06_evaluate.ipynb": [
        ("markdown", "# 6. Evaluation\nTest tracks, test accuracy, size, inference time and training cost, as "
                     "mean ± standard deviation across run seeds (design specification §11)."),
        ("code", SETUP),
        ("code", "from racing import config, evaluate\n\nresults = evaluate.main(n_jobs=config.N_JOBS)\n"
                 "results['comparison']"),
        ("code", "results['driving']"),
        ("code", "results['search_cost']"),
        ("code", "results['driving_selected']"),
        ("code", "results['dataset']"),
        ("code", "import json\n\n# run conditions recorded by scripts/run_notebooks.py (spec §3.3)\n"
                 "if config.RUN_INFO_PATH.exists():\n"
                 "    print(json.dumps(json.loads(config.RUN_INFO_PATH.read_text()), indent=1))"),
        ("code", "from IPython.display import Image, display\n\nfor path in results['figures']:\n"
                 "    display(Image(filename=str(path)))"),
    ],
}


OBSOLETE = ("05_evaluate.ipynb",)      # the evaluation notebook before specification 2.0; other files are kept


def cell(kind, source, index):
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "id": f"cell-{index}", "metadata": {}, "source": lines}
    return {"cell_type": "code", "id": f"cell-{index}", "metadata": {}, "execution_count": None, "outputs": [],
            "source": lines}


def main():
    FOLDER.mkdir(exist_ok=True)
    for name in OBSOLETE:
        (FOLDER / name).unlink(missing_ok=True)
    for name, cells in NOTEBOOKS.items():
        notebook = {"cells": [cell(kind, source, k) for k, (kind, source) in enumerate(cells)],
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python"}},
                    "nbformat": 4, "nbformat_minor": 5}
        (FOLDER / name).write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    main()
