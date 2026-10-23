"""Writes the eight experiment notebooks as plain nbformat-4 JSON (no extra dependency).

Run with:  python scripts/make_notebooks.py

Before the notebooks are first rewritten, the existing experiment notebooks, which hold the outputs of the revision
2.0 run, are copied to results/archive/revision2-notebooks/. Every run function skips completed runs and continues
interrupted ones (spec §3.3), so re-executing a notebook continues where it stopped.
"""
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
FOLDER = PROJECT / "notebooks"
ARCHIVE = PROJECT / "results" / "archive" / "revision2-notebooks"
SETUP = "import sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path.cwd().parent))"

BRIEF = (
    "BRIEF_KEYS = ('approach', 'seed', 'level_reached', 'goal_reached', 'outcome', 'target_reduction',\n"
    "              'generations', 'rounds', 'val_successes', 'val_accuracy', 'wall_time')\n\n\n"
    "def brief(summary):\n"
    "    \"\"\"One line per run; the live progress and milestones are shown by scripts/run_notebooks.py.\"\"\"\n"
    "    for value in summary.values():\n"
    "        if isinstance(value, dict) and 'approach' in value:\n"
    "            brief(value)                      # a stage of a combined run, e.g. DAgger and speed push\n"
    "    print({key: summary[key] for key in BRIEF_KEYS if key in summary}, flush=True)\n")

EVALUATION_TABLES = ("results", "pole_laps", "head_to_head", "training_cost", "stages", "comparison", "search_cost",
                     "driving_selected", "dataset", "robustness", "evolved_networks")

NOTEBOOKS = {
    "01_dataset.ipynb": [
        ("markdown", "# 1. Expert dataset\nCollects the imitation-learning dataset on freshly generated maps "
                     "(spec §7.3)."),
        ("code", SETUP),
        ("code", "import json\n\nimport pandas as pd\n\nfrom racing import config\nfrom racing.expert import "
                 "collect_dataset\n\ndata, summary = collect_dataset(n_jobs=config.N_JOBS)\n"
                 "print('collection time (s):', json.loads(config.DATASET_INFO_PATH.read_text())['wall_time'])\n"
                 "pd.DataFrame(summary).T"),
    ],
    "02_imitation.ipynb": [
        ("markdown", "# 2. Imitation-learning controllers\nBaseline with each run seed, the three Keras Tuner "
                     "searches, and training of each selected architecture with each run seed (design "
                     "specification §8)."),
        ("code", SETUP),
        ("code", "from racing import config, imitation\n\n\n" + BRIEF + "\n\ndata = imitation.load_dataset()\n"
                 "for seed in config.RUN_SEEDS:\n    brief(imitation.train_baseline(data, seed=seed))\n"
                 "for approach in config.TUNER_APPROACHES:\n    brief(imitation.run_tuner(approach, data))\n"
                 "    for seed in config.RUN_SEEDS:\n        brief(imitation.retrain(approach, data, seed=seed))"),
    ],
    "03_dagger.ipynb": [
        ("markdown", "# 3. DAgger and speed push\nEach imitation controller is refined by DAgger rounds and then "
                     "pushed for speed by an evolution strategy grown from it, once per run seed (spec §8.4–8.5)."),
        ("code", SETUP),
        ("code", "from racing import config, dagger, imitation\n\n\n" + BRIEF + "\n\ndata = imitation.load_dataset()\n"
                 "for approach in config.IMITATION_APPROACHES:\n    for seed in config.RUN_SEEDS:\n"
                 "        brief(dagger.run(approach, seed, data, n_jobs=config.N_JOBS))"),
    ],
    "04_ga.ipynb": [
        ("markdown", "# 4. Genetic algorithm\nFrom Easy to the mastery goal and the speed push, once per run seed "
                     "(spec §9.2–9.5, §9.6)."),
        ("code", SETUP),
        ("code", "from racing import config, ga\n\n\n" + BRIEF + "\n\n"
                 "for seed in config.RUN_SEEDS:\n    brief(ga.run_ga(seed=seed, n_jobs=config.N_JOBS))"),
    ],
    "05_es.ipynb": [
        ("markdown", "# 5. Evolution strategy\nThe ES with the two-phase step-size rule, once per run seed "
                     "(spec §9.2–9.5, §9.7)."),
        ("code", SETUP),
        ("code", "from racing import config, es\n\n\n" + BRIEF + "\n\n"
                 "for seed in config.RUN_SEEDS:\n    brief(es.run_es(seed=seed, n_jobs=config.N_JOBS))"),
    ],
    "06_es_seeded.ipynb": [
        ("markdown", "# 6. Seeded evolution strategy\nThe ES started from the DAgger-refined baseline of the same "
                     "run seed (spec §9.8)."),
        ("code", SETUP),
        ("code", "from racing import config, es\n\n\n" + BRIEF + "\n\n"
                 "for seed in config.RUN_SEEDS:\n"
                 "    brief(es.run_es(seed=seed, initial=es.dagger_baseline(seed), approach='es_seeded',\n"
                 "                    n_jobs=config.N_JOBS))"),
    ],
    "07_neat.ipynb": [
        ("markdown", "# 7. NEAT\nNeuroevolution of augmenting topologies, once per run seed (spec §9.9)."),
        ("code", SETUP),
        ("code", "from racing import config, neat\n\n\n" + BRIEF + "\n\n"
                 "for seed in config.RUN_SEEDS:\n    brief(neat.run_neat(seed=seed, n_jobs=config.N_JOBS))"),
    ],
    "08_evaluate.ipynb": [
        ("markdown", "# 8. Evaluation\nTest maps finished and succeeded, lap times, training cost, imitation "
                     "stages and the course comparison, reported for every run of each method and as their "
                     "mean and standard deviation, written `r0 · r1 · r2 · r3 (mean ± sd)`, with each method's best "
                     "run (spec §11.3), and each method's flying laps against the pole laps of the four test circuits "
                     "(spec §11.1)."),
        ("code", SETUP),
        ("code", "from IPython.display import display\n\nfrom racing import config, evaluate\n\n"
                 "results = evaluate.main(n_jobs=config.N_JOBS)\n"
                 f"for name in {EVALUATION_TABLES!r}:\n    print(name)\n    display(results[name])"),
        ("code", "import json\n\n# run conditions recorded by scripts/run_notebooks.py (spec §3.3)\n"
                 "if config.RUN_INFO_PATH.exists():\n"
                 "    print(json.dumps(json.loads(config.RUN_INFO_PATH.read_text()), indent=1))"),
        ("code", "from IPython.display import Image\n\nfor path in results['figures']:\n"
                 "    display(Image(filename=str(path)))"),
    ],
}

OBSOLETE = ("05_evaluate.ipynb", "03_ga.ipynb", "04_es.ipynb", "05_es_seeded.ipynb",
            "06_evaluate.ipynb")         # notebooks of earlier specifications; other files are kept


def cell(kind, source, index):
    lines = source.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "id": f"cell-{index}", "metadata": {}, "source": lines}
    return {"cell_type": "code", "id": f"cell-{index}", "metadata": {}, "execution_count": None, "outputs": [],
            "source": lines}


def archive_once():
    """Copy the existing experiment notebooks to ARCHIVE before they are first rewritten, while revision 2.0
    notebooks (any of OBSOLETE) are still present; they hold the outputs of the revision 2.0 run. Does nothing once
    the archive exists or on a fresh clone without revision 2.0 notebooks. Returns the copied names."""
    if ARCHIVE.exists() or not any((FOLDER / name).exists() for name in OBSOLETE):
        return []
    names = sorted(p.name for p in FOLDER.glob("[0-9][0-9]_*.ipynb"))
    if not names:
        return []
    ARCHIVE.mkdir(parents=True)
    for name in names:
        shutil.copy2(FOLDER / name, ARCHIVE / name)
    return names


def main():
    FOLDER.mkdir(exist_ok=True)
    archive_once()
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
