import json

import pytest

from racing import es, evaluate, expert, ga, imitation


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    runs, champions = tmp_path / "runs", tmp_path / "champions"
    data = expert.collect_dataset(tmp_path / "dataset.npz",
                                  splits={"train": [(0, 1), (0, 2)], "val": [(80, 4)], "test": [(1000, 4)]})
    imitation.train_baseline(data, out_dir=runs / "baseline", champion_dir=champions, epochs=2)
    ga.run_ga(pop_size=20, kept=5, budget=40, out_dir=runs / "ga", champion_dir=champions)
    es.run_es(mu=4, lam=16, budget=40, out_dir=runs / "es", champion_dir=champions)
    results = evaluate.main(champions_dir=champions, runs_dir=runs, data_path=tmp_path / "dataset.npz",
                            out_dir=tmp_path / "out", test_seeds=[1000])
    assert set(results["comparison"]["Method"]) == {"Baseline", "ES", "GA"}
    assert (tmp_path / "out" / "tables" / "comparison.md").exists()
    assert json.loads((runs / "ga" / "summary.json").read_text())["evals"] == 40


def test_notebooks_are_valid_json():
    from pathlib import Path
    folder = Path(__file__).resolve().parent.parent / "notebooks"
    names = sorted(p.name for p in folder.glob("*.ipynb"))
    assert names == ["01_dataset.ipynb", "02_imitation.ipynb", "03_ga.ipynb", "04_es.ipynb", "05_evaluate.ipynb"]
    for path in folder.glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4 and notebook["cells"]
