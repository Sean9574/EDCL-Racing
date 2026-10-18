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
    """Every training call, as (name, positional arguments, keyword arguments), instead of the training."""
    from racing import dagger, es, ga, imitation, neat
    made = []

    def record(module, name):
        def call(*args, **kwargs):
            made.append((name, args, kwargs))
            return {"approach": approach_of((name, args, kwargs)), "seed": seed_of((name, args, kwargs))}
        monkeypatch.setattr(module, name, call)

    for module, name in ((ga, "run_ga"), (es, "run_es"), (neat, "run_neat"), (dagger, "run"),
                         (imitation, "train_baseline"), (imitation, "retrain"), (imitation, "run_tuner")):
        record(module, name)
    monkeypatch.setattr(es, "dagger_baseline", lambda seed: f"refined baseline {seed}")
    monkeypatch.setattr(imitation, "load_dataset", lambda: "the expert dataset")
    return made


def approach_of(call):
    name, args, kwargs = call
    if name == "train_baseline":
        return "baseline"
    if name in ("retrain", "run_tuner", "run"):
        return args[0]
    return kwargs.get("approach", {"run_ga": "ga", "run_es": "es", "run_neat": "neat"}[name])


def seed_of(call):
    name, args, kwargs = call
    return args[1] if name == "run" else kwargs.get("seed")


def run_cells(*notebooks):
    for name in notebooks:
        exec(compile(loop_cell(name), name, "exec"), {})


@pytest.mark.parametrize("approach", config.EVOLUTION_APPROACHES)
def test_an_evolution_run_trained_outside_its_notebook_makes_the_notebooks_call(approach, calls):
    run_cells(train.NOTEBOOKS[approach])
    from_notebook = [call for call in calls if approach_of(call) == approach]
    calls.clear()
    for seed in config.RUN_SEEDS:
        train.train(approach, seed)
    assert calls == from_notebook and len(calls) == len(config.RUN_SEEDS)


@pytest.mark.parametrize("approach", config.IMITATION_APPROACHES)
def test_an_imitation_run_makes_notebook_02s_call_for_its_seed_then_notebook_03s(approach, calls):
    run_cells("02_imitation.ipynb", "03_dagger.ipynb")
    from_notebooks = [call for call in calls if approach_of(call) == approach and call[0] != "run_tuner"]
    calls.clear()
    for seed in config.RUN_SEEDS:
        train.train(approach, seed)
    expected = [call for seed in config.RUN_SEEDS for call in from_notebooks if seed_of(call) == seed]
    assert calls == expected and len(calls) == 2 * len(config.RUN_SEEDS)


@pytest.mark.parametrize("approach", config.TUNER_APPROACHES)
def test_a_search_job_makes_notebook_02s_search_call(approach, calls):
    run_cells("02_imitation.ipynb")
    from_notebook = [call for call in calls if call[0] == "run_tuner" and call[1][0] == approach]
    calls.clear()
    train.search(approach)
    assert calls == from_notebook and len(calls) == 1


def test_only_the_tuner_methods_have_a_search():
    with pytest.raises(ValueError):
        train.search("baseline")


def test_main_runs_one_search_with_its_own_status_file(monkeypatch, tmp_path, capsys):
    seen = []
    monkeypatch.setattr(train, "search", lambda approach: seen.append((approach, config.STATUS_PATH)) or {
        "approach": approach, "trials": 30, "chosen_params": 463})
    assert train.main(["search", "kt_bayesian", "--status", str(tmp_path / "search-kt_bayesian.json")]) == 0
    assert seen == [("kt_bayesian", tmp_path / "search-kt_bayesian.json")]
    assert "'trials': 30" in capsys.readouterr().out


def test_every_approach_has_its_notebook():
    assert set(train.NOTEBOOKS) == set(config.APPROACHES)
    assert all((NOTEBOOK_DIR / name).exists() for name in train.NOTEBOOKS.values())


def test_runs_are_read_as_approach_and_seed():
    assert train.parse_runs("ga:1,neat:0") == [("ga", 1), ("neat", 0)]
    assert train.parse_runs(" es_seeded:2 , es:0 ,") == [("es_seeded", 2), ("es", 0)]


def test_imitation_runs_are_accepted():
    assert train.parse_runs("baseline:0,kt_bayesian:2") == [("baseline", 0), ("kt_bayesian", 2)]


@pytest.mark.parametrize("text", ["nb:01", "ga:3", "ga", "ga:x", "ga:1,ga:1", "", " , "])
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
