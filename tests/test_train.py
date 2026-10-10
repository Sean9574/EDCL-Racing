import json
from pathlib import Path

import pytest

from racing import config, train

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def loop_cell(name):
    """The notebook's code cell that trains its runs."""
    cells = json.loads((NOTEBOOK_DIR / name).read_text())["cells"]
    sources = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    return next(source for source in sources if "for seed in config.RUN_SEEDS" in source)


@pytest.fixture
def calls(monkeypatch):
    """Every training call, with its arguments, instead of the training."""
    from racing import es, ga, neat
    made = []

    def record(name):
        def call(**kwargs):
            made.append((name, kwargs))
            return {"approach": kwargs.get("approach", name), "seed": kwargs["seed"]}
        return call

    monkeypatch.setattr(ga, "run_ga", record("run_ga"))
    monkeypatch.setattr(es, "run_es", record("run_es"))
    monkeypatch.setattr(neat, "run_neat", record("run_neat"))
    monkeypatch.setattr(es, "dagger_baseline", lambda seed: f"refined baseline {seed}")
    return made


@pytest.mark.parametrize("approach", list(train.NOTEBOOKS))
def test_a_run_trained_outside_its_notebook_makes_the_notebooks_call(approach, calls, capsys):
    exec(compile(loop_cell(train.NOTEBOOKS[approach]), train.NOTEBOOKS[approach], "exec"), {})
    from_notebook = list(calls)
    calls.clear()
    for seed in config.RUN_SEEDS:
        train.train(approach, seed)
    assert calls == from_notebook and len(calls) == len(config.RUN_SEEDS)


def test_every_evolutionary_approach_has_its_notebook():
    assert set(train.NOTEBOOKS) == set(config.EVOLUTION_APPROACHES)
    assert all((NOTEBOOK_DIR / name).exists() for name in train.NOTEBOOKS.values())


def test_runs_are_read_as_approach_and_seed():
    assert train.parse_runs("ga:1,neat:0") == [("ga", 1), ("neat", 0)]
    assert train.parse_runs(" es_seeded:2 , es:0 ,") == [("es_seeded", 2), ("es", 0)]


@pytest.mark.parametrize("text", ["baseline:0", "ga:3", "ga", "ga:x", "ga:1,ga:1", "", " , "])
def test_runs_outside_the_experiment_are_refused(text):
    with pytest.raises(ValueError):
        train.parse_runs(text)


def test_main_trains_one_run_with_its_own_status_file(monkeypatch, tmp_path, capsys):
    seen = []
    monkeypatch.setattr(train, "train", lambda approach, seed: seen.append((approach, seed, config.STATUS_PATH)) or {
        "approach": approach, "seed": seed, "generations": 5})
    assert train.main(["neat", "2", "--status", str(tmp_path / "neat-seed2.json")]) == 0
    assert seen == [("neat", 2, tmp_path / "neat-seed2.json")]
    assert "'generations': 5" in capsys.readouterr().out
