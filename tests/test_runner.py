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
