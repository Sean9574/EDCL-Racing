import importlib.util
import json
from pathlib import Path

import nbformat
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_notebooks.py"


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("run_notebooks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_notebook(path, source):
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]), path)


def test_notebook_order_and_start(runner, tmp_path):
    for name in ("02_b.ipynb", "01_a.ipynb", "10_c.ipynb", "notes.ipynb"):
        write_notebook(tmp_path / name, "1")
    assert runner.notebook_names(tmp_path) == ["01_a.ipynb", "02_b.ipynb", "10_c.ipynb"]
    assert runner.notebook_names(tmp_path, "02") == ["02_b.ipynb", "10_c.ipynb"]


def three_notebooks(tmp_path):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    for name in ("01_a.ipynb", "02_b.ipynb", "03_c.ipynb"):
        write_notebook(folder / name, "1")
    return folder


def run_main(runner, tmp_path, argv):
    return runner.main(argv, notebook_dir=tmp_path / "notebooks", log_path=tmp_path / "log.txt",
                       info_path=tmp_path / "info.json")


@pytest.mark.parametrize("start", ["3", "07", "03_c", "03_c.ipynb"])
def test_from_must_name_an_existing_notebook(runner, tmp_path, monkeypatch, capsys, start):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir: pytest.fail("a notebook was run"))
    assert run_main(runner, tmp_path, ["--from", start]) == 2
    assert "01, 02, 03" in capsys.readouterr().err
    assert not (tmp_path / "info.json").exists() and not (tmp_path / "log.txt").exists()


def test_from_an_existing_notebook_runs_it_and_the_later_ones(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    executed = []
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir: executed.append(path.name))
    assert run_main(runner, tmp_path, ["--from", "02"]) == 0
    assert executed == ["02_b.ipynb", "03_c.ipynb"]
    info = json.loads((tmp_path / "info.json").read_text())
    assert [e["notebook"] for e in info["notebooks"]] == ["02_b.ipynb", "03_c.ipynb"]
    assert info["runs"][-1]["from"] == "02"


def test_run_record_holds_the_settings_and_code_version(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir: None)
    assert run_main(runner, tmp_path, []) == 0
    info = json.loads((tmp_path / "info.json").read_text())
    assert list(info["settings"]) == ["N_JOBS", "STEPS_PER_EXECUTION", "JIT_COMPILE", "GA_S_MAX", "GA_S_MIN",
                                      "PACE_FACTOR", "LEVEL_UP_NEW_FRACTION", "MAX_EPOCHS", "RUN_SEEDS",
                                      "EVAL_BUDGET"]
    assert info["settings"]["RUN_SEEDS"] == list(runner.config.RUN_SEEDS)
    assert info["settings"]["N_JOBS"] == runner.config.N_JOBS
    assert len(info["git_commit"]) == 40 and int(info["git_commit"], 16) >= 0
    assert isinstance(info["git_dirty"], bool)


def test_code_version_without_git_is_unknown(runner, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(runner.subprocess, "run", missing)
    assert runner.code_version() == {"git_commit": "unknown", "git_dirty": "unknown"}


def test_make_notebooks_removes_only_the_obsolete_notebook(tmp_path, monkeypatch):
    path = SCRIPT.parent / "make_notebooks.py"
    spec = importlib.util.spec_from_file_location("make_notebooks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "FOLDER", tmp_path)
    for name in ("05_evaluate.ipynb", "07_extra.ipynb", "notes.ipynb"):
        write_notebook(tmp_path / name, "1")
    module.main()
    assert sorted(p.name for p in tmp_path.glob("*.ipynb")) == sorted([*module.NOTEBOOKS, "07_extra.ipynb",
                                                                       "notes.ipynb"])


@pytest.mark.slow
def test_runs_notebooks_in_order_and_records_the_run(runner, tmp_path):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    write_notebook(folder / "01_a.ipynb", "import os, sys\nprint(os.getcwd())\nprint(sys.executable)")
    write_notebook(folder / "02_b.ipynb", "print('second')")
    code = runner.run(folder, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json")
    assert code == 0
    first = nbformat.read(folder / "01_a.ipynb", as_version=4).cells[0].outputs[0]["text"].split()
    assert first == [str(folder), runner.sys.executable]
    info = json.loads((tmp_path / "info.json").read_text())
    assert [e["notebook"] for e in info["notebooks"]] == ["01_a.ipynb", "02_b.ipynb"]
    assert all(e["seconds"] >= 0 for e in info["notebooks"]) and info["n_jobs"] >= 1
    assert "done 02_b.ipynb" in (tmp_path / "log.txt").read_text()


@pytest.mark.slow
def test_resume_keeps_earlier_record_entries(runner, tmp_path):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    write_notebook(folder / "01_a.ipynb", "print(1)")
    write_notebook(folder / "02_b.ipynb", "print(2)")
    runner.run(folder, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json")
    runner.run(folder, start="02", log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json")
    info = json.loads((tmp_path / "info.json").read_text())
    assert [e["notebook"] for e in info["notebooks"]] == ["01_a.ipynb", "02_b.ipynb"]


@pytest.mark.slow
def test_each_notebook_kernel_is_shut_down(runner, tmp_path):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    write_notebook(folder / "01_a.ipynb", "import os\nprint(os.getpid())")
    assert runner.run(folder, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json") == 0
    pid = int(nbformat.read(folder / "01_a.ipynb", as_version=4).cells[0].outputs[0]["text"])
    assert not Path(f"/proc/{pid}").exists()


@pytest.mark.slow
def test_a_failing_notebook_stops_the_run(runner, tmp_path):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    write_notebook(folder / "01_a.ipynb", "raise ValueError('broken')")
    write_notebook(folder / "02_b.ipynb", "print(2)")
    code = runner.run(folder, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json")
    info = json.loads((tmp_path / "info.json").read_text())
    assert code == 1 and [e["notebook"] for e in info["notebooks"]] == ["01_a.ipynb"]
    assert "failed" in info["notebooks"][0]
    assert nbformat.read(folder / "01_a.ipynb", as_version=4).cells[0].outputs     # the error is saved


def test_experiment_notebooks_are_valid_and_compile():
    import ast
    folder = Path(__file__).resolve().parent.parent / "notebooks"
    names = sorted(p.name for p in folder.glob("*.ipynb"))
    assert names == ["01_dataset.ipynb", "02_imitation.ipynb", "03_ga.ipynb", "04_es.ipynb", "05_es_seeded.ipynb",
                     "06_evaluate.ipynb"]
    for name in names:
        notebook = nbformat.read(folder / name, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
