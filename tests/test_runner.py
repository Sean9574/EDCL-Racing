import fcntl
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import nbformat
import pytest

from racing import checkpoint, jobs

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


def run_main(runner, tmp_path, argv, monitor=False, out=None):
    return runner.main(argv, notebook_dir=tmp_path / "notebooks", log_path=tmp_path / "log.txt",
                       info_path=tmp_path / "info.json", results_dir=tmp_path / "results",
                       status_path=tmp_path / "status.json", monitor=monitor, out=out)


@pytest.mark.parametrize("start", ["3", "07", "03_c", "03_c.ipynb"])
def test_from_must_name_an_existing_notebook(runner, tmp_path, monkeypatch, capsys, start):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: pytest.fail("a notebook was run"))
    assert run_main(runner, tmp_path, ["--from", start]) == 2
    assert "01, 02, 03" in capsys.readouterr().err
    assert not (tmp_path / "info.json").exists() and not (tmp_path / "log.txt").exists()


def test_from_an_existing_notebook_runs_it_and_the_later_ones(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    executed = []
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: executed.append(path.name))
    assert run_main(runner, tmp_path, ["--from", "02"]) == 0
    assert executed == ["02_b.ipynb", "03_c.ipynb"]
    info = json.loads((tmp_path / "info.json").read_text())
    assert [e["notebook"] for e in info["notebooks"]] == ["02_b.ipynb", "03_c.ipynb"]
    assert info["runs"][-1]["from"] == "02"


def test_run_record_holds_the_settings_and_code_version(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: None)
    assert run_main(runner, tmp_path, []) == 0
    info = json.loads((tmp_path / "info.json").read_text())
    assert info["settings"] == json.loads(json.dumps(runner.checkpoint.settings_record()))
    assert "N_JOBS" not in info["settings"]            # results do not depend on it; recorded as n_jobs
    assert info["n_jobs"] == runner.config.N_JOBS
    assert len(info["git_commit"]) == 40 and int(info["git_commit"], 16) >= 0
    assert isinstance(info["git_dirty"], bool)


def test_code_version_without_git_is_unknown(runner, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(runner.subprocess, "run", missing)
    assert runner.code_version() == {"git_commit": "unknown", "git_dirty": "unknown"}


def load_make_notebooks():
    path = SCRIPT.parent / "make_notebooks.py"
    spec = importlib.util.spec_from_file_location("make_notebooks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_notebooks_archives_once_then_replaces_the_obsolete_notebooks(tmp_path, monkeypatch):
    module = load_make_notebooks()
    folder, archive = tmp_path / "notebooks", tmp_path / "archive"
    monkeypatch.setattr(module, "FOLDER", folder)
    monkeypatch.setattr(module, "ARCHIVE", archive)
    folder.mkdir()
    for name in ("01_dataset.ipynb", "03_ga.ipynb", "06_evaluate.ipynb", "09_extra.ipynb", "notes.ipynb"):
        write_notebook(folder / name, f"print({name!r})")
    module.main()
    assert sorted(p.name for p in archive.iterdir()) == ["01_dataset.ipynb", "03_ga.ipynb", "06_evaluate.ipynb",
                                                         "09_extra.ipynb"]
    assert "03_ga" in nbformat.read(archive / "03_ga.ipynb", as_version=4).cells[0].source
    assert sorted(p.name for p in folder.glob("*.ipynb")) == sorted([*module.NOTEBOOKS, "09_extra.ipynb",
                                                                     "notes.ipynb"])
    module.main()                                       # a second rewrite leaves the archive as it was
    assert "01_dataset" in nbformat.read(archive / "01_dataset.ipynb", as_version=4).cells[0].source


def test_a_fresh_clone_without_revision_2_notebooks_archives_nothing(tmp_path, monkeypatch):
    module = load_make_notebooks()
    folder, archive = tmp_path / "notebooks", tmp_path / "archive"
    monkeypatch.setattr(module, "FOLDER", folder)
    monkeypatch.setattr(module, "ARCHIVE", archive)
    folder.mkdir()
    write_notebook(folder / "01_dataset.ipynb", "print('revision 3')")
    module.main()
    assert not archive.exists()


@pytest.mark.slow
def test_runs_notebooks_in_order_and_records_the_run(runner, tmp_path):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    write_notebook(folder / "01_a.ipynb", "import os, sys\nprint(os.getcwd())\nprint(sys.executable)")
    write_notebook(folder / "02_b.ipynb", "print('second')")
    code = runner.run(folder, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json",
                     status_path=tmp_path / "status.json", monitor=False)
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
    runner.run(folder, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json",
                     status_path=tmp_path / "status.json", monitor=False)
    runner.run(folder, start="02", log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json",
                     status_path=tmp_path / "status.json", monitor=False)
    info = json.loads((tmp_path / "info.json").read_text())
    assert [e["notebook"] for e in info["notebooks"]] == ["01_a.ipynb", "02_b.ipynb"]


@pytest.mark.slow
def test_each_notebook_kernel_is_shut_down(runner, tmp_path):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    write_notebook(folder / "01_a.ipynb", "import os\nprint(os.getpid())")
    assert runner.run(folder, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json",
                     status_path=tmp_path / "status.json", monitor=False) == 0
    pid = int(nbformat.read(folder / "01_a.ipynb", as_version=4).cells[0].outputs[0]["text"])
    assert not Path(f"/proc/{pid}").exists()


@pytest.mark.slow
def test_a_failing_notebook_stops_the_run(runner, tmp_path):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    write_notebook(folder / "01_a.ipynb", "raise ValueError('broken')")
    write_notebook(folder / "02_b.ipynb", "print(2)")
    code = runner.run(folder, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json",
                     status_path=tmp_path / "status.json", monitor=False)
    info = json.loads((tmp_path / "info.json").read_text())
    assert code == 1 and [e["notebook"] for e in info["notebooks"]] == ["01_a.ipynb"]
    assert "failed" in info["notebooks"][0]
    assert nbformat.read(folder / "01_a.ipynb", as_version=4).cells[0].outputs     # the error is saved


def test_experiment_notebooks_are_valid_and_compile():
    import ast
    folder = Path(__file__).resolve().parent.parent / "notebooks"
    names = sorted(p.name for p in folder.glob("*.ipynb"))
    assert names == ["01_dataset.ipynb", "02_imitation.ipynb", "03_dagger.ipynb", "04_ga.ipynb", "05_es.ipynb",
                     "06_es_seeded.ipynb", "07_neat.ipynb", "08_evaluate.ipynb"]
    for name in names:
        notebook = nbformat.read(folder / name, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)


def changed_settings(runner):
    return {**runner.checkpoint.settings_record(), "TARGET_FACTOR": 1.2}


def test_results_from_other_settings_are_refused_until_fresh(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    executed = []
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: executed.append(path.name))
    record = tmp_path / "results" / "runs" / "ga" / "seed0" / "summary.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"approach": "ga", "settings": changed_settings(runner)}))
    assert run_main(runner, tmp_path, []) == 3
    err = capsys.readouterr().err
    assert "TARGET_FACTOR" in err and "--fresh" in err and "ga/seed0/summary.json" in err
    assert executed == [] and not (tmp_path / "info.json").exists()
    assert run_main(runner, tmp_path, ["--fresh"]) == 0
    assert executed == ["01_a.ipynb", "02_b.ipynb", "03_c.ipynb"] and not record.exists()


def test_a_checkpoint_from_other_settings_is_refused(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: pytest.fail("a notebook was run"))
    path = tmp_path / "results" / "runs" / "neat" / "seed2" / runner.checkpoint.CHECKPOINT_NAME
    path.parent.mkdir(parents=True)
    runner.checkpoint.save(path, {"generation": 7, "settings": changed_settings(runner)})
    assert run_main(runner, tmp_path, []) == 3
    assert "neat/seed2" in capsys.readouterr().err


def test_records_without_settings_are_refused_until_fresh(runner, tmp_path, monkeypatch, capsys):
    """checkpoint.completed refuses a record without settings, so the runner refuses it before notebook 01."""
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: None)
    record = tmp_path / "results" / "runs" / "ga" / "seed0" / "summary.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"approach": "ga", "evals": 200000}))       # a revision 2.0 record
    assert run_main(runner, tmp_path, []) == 3
    assert "ga/seed0/summary.json" in capsys.readouterr().err
    assert run_main(runner, tmp_path, ["--fresh"]) == 0


def test_fresh_removes_results_but_keeps_pilots_viewer_runs_and_archive(runner, tmp_path):
    results = tmp_path / "results"
    for name in ("runs/ga/seed0/log.csv", "champions/ga/seed0.keras", "tables/a.md", "figures/a.png",
                 "pilots/summary.md", "app_runs/ga/seed0/log.csv", "archive/revision2-notebooks/01_dataset.ipynb",
                 "dataset.npz", "dataset_info.json", "status.json", "run_log.txt", "run_info.json"):
        (results / name).parent.mkdir(parents=True, exist_ok=True)
        (results / name).write_text("x")
    runner.fresh_start(results)
    assert sorted(p.name for p in results.iterdir()) == ["app_runs", "archive", "pilots"]
    runner.fresh_start(tmp_path / "missing")            # nothing to remove is not an error


def test_fresh_cannot_be_combined_with_from(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: pytest.fail("a notebook was run"))
    assert run_main(runner, tmp_path, ["--fresh", "--from", "02"]) == 2
    assert "--fresh" in capsys.readouterr().err


def test_a_settings_mismatch_inside_a_notebook_stops_with_code_3(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)

    def stopped(path, kernel_dir, output=None):
        raise RuntimeError("SettingsMismatch: TARGET_FACTOR: recorded 1.2, current 1.3")

    monkeypatch.setattr(runner, "execute", stopped)
    assert run_main(runner, tmp_path, []) == 3
    assert "--fresh" in (tmp_path / "log.txt").read_text()


def test_the_monitor_prints_the_runner_milestones(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: None)
    out = io.StringIO()
    assert run_main(runner, tmp_path, [], monitor=True, out=out) == 0
    text = out.getvalue()
    assert "run started" in text and "done 03_c.ipynb" in text and "run finished" in text
    assert text.index("start 01_a.ipynb") < text.index("run finished")


def test_a_rerun_keeps_the_record_of_earlier_notebooks(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: None)
    assert run_main(runner, tmp_path, []) == 0
    assert run_main(runner, tmp_path, ["--from", "03"]) == 0
    info = json.loads((tmp_path / "info.json").read_text())
    assert [e["notebook"] for e in info["notebooks"]] == ["01_a.ipynb", "02_b.ipynb", "03_c.ipynb"]
    assert [r["from"] for r in info["runs"]] == [None, "03"]


def test_an_unreadable_record_is_named_and_only_that_file_is_to_be_deleted(runner, tmp_path, monkeypatch, capsys):
    """A record cut short by a reboot is not a settings change: --fresh would remove every result."""
    three_notebooks(tmp_path)
    executed = []
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: executed.append(path.name))
    record = tmp_path / "results" / "runs" / "baseline" / "seed0" / "summary.json"
    record.parent.mkdir(parents=True)
    record.write_text("")
    broken = tmp_path / "results" / "runs" / "neat" / "seed1" / runner.checkpoint.CHECKPOINT_NAME
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"\0" * 64)
    assert run_main(runner, tmp_path, []) == 3
    err = capsys.readouterr().err
    assert "baseline/seed0/summary.json" in err and "neat/seed1/checkpoint.pkl" in err
    assert "delete" in err.lower() and "--fresh" not in err and "earlier version" not in err
    assert executed == [] and record.exists() and broken.exists()
    record.unlink()
    broken.unlink()
    assert run_main(runner, tmp_path, []) == 0 and len(executed) == 3


def test_an_unreadable_summary_next_to_its_checkpoint_finishes_by_itself(runner, tmp_path, monkeypatch):
    """The run redoes its final selection from the checkpoint (checkpoint.completed), so the runner starts."""
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: None)
    run = tmp_path / "results" / "runs" / "ga" / "seed0"
    run.mkdir(parents=True)
    (run / "summary.json").write_text("")
    runner.checkpoint.save(run / runner.checkpoint.CHECKPOINT_NAME,
                           {"generation": 9, "settings": runner.checkpoint.settings_record()})
    assert run_main(runner, tmp_path, []) == 0


def test_the_run_record_is_synced_to_disk(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: None)
    written, real = [], runner.checkpoint.write_json
    monkeypatch.setattr(runner.checkpoint, "write_json",
                        lambda path, data: written.append(Path(path).name) or real(path, data))
    assert run_main(runner, tmp_path, []) == 0
    assert written and set(written) == {"info.json"}


def hold_lock(results):
    """The runner lock of `results`, held as a second runner would hold it."""
    results.mkdir(parents=True, exist_ok=True)
    handle = open(results / ".runner.lock", "a")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def test_a_second_runner_on_the_same_results_stops_with_code_4(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: pytest.fail("a notebook was run"))
    (tmp_path / "results" / "runs").mkdir(parents=True)                 # what --fresh would remove
    held = hold_lock(tmp_path / "results")
    held.write("4242\n")                                                # the holder records its process
    held.flush()
    try:
        assert run_main(runner, tmp_path, []) == 4
        assert run_main(runner, tmp_path, ["--fresh"]) == 4
    finally:
        held.close()
    err = capsys.readouterr().err
    assert "another runner (process 4242)" in err and str(tmp_path / "results") in err
    assert (tmp_path / "results" / "runs").exists() and not (tmp_path / "info.json").exists()
    assert not (tmp_path / "log.txt").exists()


def test_the_runner_holds_the_lock_for_the_whole_run_and_releases_it(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    seen = []

    def execute(path, kernel_dir, output=None):
        assert (tmp_path / "results" / ".runner.lock").read_text() == f"{runner.os.getpid()}\n"
        try:
            hold_lock(tmp_path / "results").close()
            seen.append("free")
        except BlockingIOError:
            seen.append("held")
        if path.name == "02_b.ipynb":
            raise RuntimeError("broken")

    monkeypatch.setattr(runner, "execute", execute)
    assert run_main(runner, tmp_path, []) == 1                           # a failing notebook ends the run
    assert seen == ["held", "held"]
    hold_lock(tmp_path / "results").close()                             # released
    assert run_main(runner, tmp_path, ["--from", "03"]) == 0 and seen[-1] == "held"
    hold_lock(tmp_path / "results").close()


def test_the_lock_creates_the_results_folder(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: None)
    assert not (tmp_path / "results").exists()
    assert run_main(runner, tmp_path, []) == 0
    assert (tmp_path / "results" / ".runner.lock").exists()



def test_each_start_records_its_code_version_and_a_changed_commit_is_reported(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: None)
    versions = iter([{"git_commit": "a" * 40, "git_dirty": False}, {"git_commit": "a" * 40, "git_dirty": True},
                     {"git_commit": "b" * 40, "git_dirty": False}])
    monkeypatch.setattr(runner, "code_version", lambda: next(versions))
    assert run_main(runner, tmp_path, []) == 0
    assert run_main(runner, tmp_path, ["--from", "03"]) == 0
    assert "warning" not in capsys.readouterr().err                     # the same commit
    assert run_main(runner, tmp_path, ["--from", "03"]) == 0               # resumed after a git pull
    err = capsys.readouterr().err
    assert "warning" in err and "aaaaaaa" in err and "bbbbbbb" in err
    info = json.loads((tmp_path / "info.json").read_text())
    assert [(r["git_commit"], r["git_dirty"]) for r in info["runs"]] == [("a" * 40, False), ("a" * 40, True),
                                                                         ("b" * 40, False)]
    assert info["git_commit"] == "b" * 40
    assert "aaaaaaa" in (tmp_path / "log.txt").read_text()


def test_a_record_from_before_per_start_versions_is_compared_with_its_last_commit(runner, tmp_path, monkeypatch,
                                                                                  capsys):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: None)
    monkeypatch.setattr(runner, "code_version", lambda: {"git_commit": "b" * 40, "git_dirty": False})
    (tmp_path / "info.json").write_text(json.dumps({"git_commit": "c" * 40, "runs": [{"started": "x", "from": None}],
                                                    "notebooks": []}))
    assert run_main(runner, tmp_path, ["--from", "02"]) == 0
    assert "ccccccc" in capsys.readouterr().err


def test_an_unknown_code_version_is_not_reported_as_a_change(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: None)
    versions = iter([{"git_commit": "unknown", "git_dirty": "unknown"}, {"git_commit": "b" * 40, "git_dirty": False}])
    monkeypatch.setattr(runner, "code_version", lambda: next(versions))
    assert run_main(runner, tmp_path, []) == 0 and run_main(runner, tmp_path, []) == 0
    assert "warning" not in capsys.readouterr().err


def test_notebook_kernels_use_the_cpu_without_tensorflow_logs(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    specs = []
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: specs.append(
        json.loads((Path(kernel_dir) / runner.KERNEL / "kernel.json").read_text())))
    assert run_main(runner, tmp_path, []) == 0
    assert len(specs) == 3 and specs[0]["env"] == {"CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3"}


STOP_SECONDS = 30            # a stopped runner ends its kernel and exits within this time


def quiet_runner(tmp_path):
    """Starts the runner in its own process on one notebook that prints nothing for ten minutes, as a long training
    cell does. Returns the process and the file the notebook writes its kernel's process id to."""
    folder = tmp_path / "notebooks"
    folder.mkdir()
    pid_file = tmp_path / "kernel.pid"
    write_notebook(folder / "01_quiet.ipynb",
                   f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(600)")
    script = (f"import importlib.util, sys\n"
              f"spec = importlib.util.spec_from_file_location('run_notebooks', {str(SCRIPT)!r})\n"
              f"runner = importlib.util.module_from_spec(spec)\n"
              f"spec.loader.exec_module(runner)\n"
              f"sys.exit(runner.main([], notebook_dir={str(folder)!r}, log_path={str(tmp_path / 'log.txt')!r}, "
              f"info_path={str(tmp_path / 'info.json')!r}, results_dir={str(tmp_path / 'results')!r}, "
              f"status_path={str(tmp_path / 'status.json')!r}, monitor=False))\n")
    process = subprocess.Popen([sys.executable, "-c", script], cwd=SCRIPT.parent.parent)
    deadline = time.time() + 120
    while not (pid_file.exists() and pid_file.read_text()):
        assert process.poll() is None and time.time() < deadline, "the notebook never started"
        time.sleep(0.2)
    return process, int(pid_file.read_text())


@pytest.mark.slow
@pytest.mark.parametrize("stop", [signal.SIGINT, signal.SIGTERM])
def test_ctrl_c_or_kill_stops_a_quiet_notebook_and_its_kernel(runner, tmp_path, stop):
    """The test signals only the runner and the kernel it started itself."""
    process, kernel = quiet_runner(tmp_path)
    try:
        process.send_signal(stop)
        assert process.wait(timeout=STOP_SECONDS) == runner.STOPPED
        deadline = time.time() + 10
        while Path(f"/proc/{kernel}").exists() and time.time() < deadline:
            time.sleep(0.2)
        assert not Path(f"/proc/{kernel}").exists()
        assert "STOPPED 01_quiet.ipynb" in (tmp_path / "log.txt").read_text()
    finally:
        if process.poll() is None:
            process.kill()
        if Path(f"/proc/{kernel}").exists():
            os.kill(kernel, signal.SIGKILL)


def test_kernel_output_goes_to_a_log_next_to_the_run_log(runner, tmp_path, monkeypatch):
    three_notebooks(tmp_path)
    outputs = []
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: outputs.append(output.name))
    assert run_main(runner, tmp_path, []) == 0
    assert outputs == [str(tmp_path / runner.KERNEL_LOG)] * 3
    assert runner.KERNEL_LOG in runner.FRESH_REMOVE


@pytest.mark.slow
def test_a_kernels_own_messages_stay_out_of_the_terminal(runner, tmp_path, capfd):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    write_notebook(folder / "01_a.ipynb", "import os\nos.write(2, b'native message from the kernel\\n')\nprint('cell')")
    kernels = tmp_path / "kernels"
    runner.write_kernel(kernels)
    with open(tmp_path / "kernel_log.txt", "a") as output:
        runner.execute(folder / "01_a.ipynb", kernels, output)
    assert "native message" not in capfd.readouterr().err
    assert "native message from the kernel" in (tmp_path / "kernel_log.txt").read_text()


# ----- jobs side by side (--runs, spec §3.3) -----

def fake_jobs(tmp_path, seconds=0.3, fail=()):
    """A job_command stand-in: each job writes its start and end times and, for a run, a status; it makes the
    records jobs.makes names under tmp_path/results unless it fails, and exits with 3 (printing 'boom') if its label
    is in `fail`, else 0 (printing 'ok')."""
    results = tmp_path / "results"

    def command(job, status_file):
        name = job.label.replace(":", "-")
        made = [str(p) for p in jobs.makes(job, results)]
        code = 3 if job.label in fail else 0
        status = {"approach": job.approach, "seed": job.seed} if not job.is_notebook else {"stage": "search"}
        script = (f"import json, pathlib, sys, time\n"
                  f"open({str(tmp_path / (name + '.start'))!r}, 'w').write(repr(time.time()))\n"
                  f"json.dump({status!r}, open({str(status_file)!r}, 'w'))\n"
                  f"time.sleep({seconds})\n"
                  f"for p in ({made!r} if {code} == 0 else []):\n"
                  f"    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)\n"
                  f"    pathlib.Path(p).write_text('{{}}')\n"
                  f"open({str(tmp_path / (name + '.end'))!r}, 'w').write(repr(time.time()))\n"
                  f"print({'boom' if code else 'ok'!r})\nsys.exit({code})\n")
        return [sys.executable, "-c", script]
    return command


def side_by_side(runner, tmp_path, job_list, at_once, command, monitor=False, out=None):
    return runner.run_side_by_side(job_list, at_once, log_path=tmp_path / "log.txt", info_path=tmp_path / "info.json",
                                   status_dir=tmp_path / "status", logs_dir=tmp_path / "train_logs",
                                   monitor=monitor, out=out, command=command, results_dir=tmp_path / "results")


def started(tmp_path, name):
    return float((tmp_path / f"{name}.start").read_text())


def ended(tmp_path, name):
    return float((tmp_path / f"{name}.end").read_text())


def most_at_once(tmp_path, names):
    spans = [(started(tmp_path, n), ended(tmp_path, n)) for n in names]
    return max(sum(start <= moment < end for start, end in spans) for moment, _ in spans)


def test_jobs_side_by_side_train_at_most_at_once_and_are_recorded(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.05)
    assert side_by_side(runner, tmp_path, jobs.parse_jobs("ga:0,ga:1,es:0"), 2, fake_jobs(tmp_path)) == 0
    assert most_at_once(tmp_path, ["ga-0", "ga-1", "es-0"]) == 2
    log = (tmp_path / "log.txt").read_text()
    for name in ("ga:0", "ga:1", "es:0"):
        assert f"start {name}" in log and f"done {name}" in log
    assert "runs finished" in log and "2 at a time" in log
    info = json.loads((tmp_path / "info.json").read_text())
    assert [e["run"] for e in info["trained"]] == ["ga:0", "ga:1", "es:0"]
    assert all("finished" in e for e in info["trained"])
    assert info["runs"][-1]["side_by_side"] == ["ga:0", "ga:1", "es:0"] and info["runs"][-1]["at_once"] == 2
    assert list((tmp_path / "status").iterdir()) == []                  # each job's status goes when it ends
    assert (tmp_path / "train_logs" / "ga-seed1.log").read_text().strip() == "ok"


def test_a_job_starts_only_when_its_input_exists_and_holds_no_slot_while_waiting(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.05)
    listed = jobs.parse_jobs("nb:01,nb:02,ga:0")
    assert side_by_side(runner, tmp_path, listed, 2, fake_jobs(tmp_path)) == 0
    assert started(tmp_path, "nb-02") >= ended(tmp_path, "nb-01")         # notebook 02 waited for the dataset
    assert started(tmp_path, "ga-0") < ended(tmp_path, "nb-01")           # and ga:0 used the slot meanwhile
    assert (tmp_path / "train_logs" / "nb-02.log").exists()


def test_dependants_of_a_failed_job_are_reported_and_the_rest_go_on(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.05)
    listed = jobs.parse_jobs("nb:01,nb:02,baseline:0,ga:0")
    assert side_by_side(runner, tmp_path, listed, 3, fake_jobs(tmp_path, fail={"nb:01"})) == 1
    log = (tmp_path / "log.txt").read_text()
    assert "FAILED nb:01: boom" in log and "done ga:0" in log
    assert "SKIPPED nb:02: waits for dataset_info.json, which nb:01 did not make" in log
    assert "SKIPPED baseline:0" in log and not (tmp_path / "nb-02.start").exists()
    info = json.loads((tmp_path / "info.json").read_text())
    assert {e["run"] for e in info["trained"] if "skipped" in e} == {"nb:02", "baseline:0"}


def test_the_monitor_shows_a_line_per_run_side_by_side(runner, tmp_path, monkeypatch, order):
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.05)
    out = io.StringIO()
    monkeypatch.setattr(runner.progress.Monitor, "__init__", wrap_monitor(runner.progress.Monitor.__init__))
    assert side_by_side(runner, tmp_path, jobs.parse_jobs("ga:0,es:0"), 2, fake_jobs(tmp_path, 1.5),
                        monitor=True, out=out) == 0
    text = out.getvalue()
    assert "GA seed 0" in text and "ES seed 0" in text and "start ga:0" in text


def wrap_monitor(init):
    """Monitor.__init__ that polls quickly and prints every status, so a short test sees the lines."""
    def wrapped(self, *args, **kwargs):
        init(self, *args, **{**kwargs, "interval": 0.05, "period": 0.0, "live": False})
    return wrapped


@pytest.fixture
def order(monkeypatch):
    from racing import config
    monkeypatch.setattr(config, "RUN_ORDER", (("ga", 0), ("es", 0)))


def test_stop_processes_ends_a_training_and_its_workers(runner):
    process = subprocess.Popen([sys.executable, "-c", "import subprocess, sys, time\n"
                                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
                                "time.sleep(600)"], start_new_session=True)
    time.sleep(0.5)
    runner.stop_processes([process], grace=5)
    assert process.returncode is not None
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)                                         # nothing of the group is left


@pytest.mark.parametrize("argv", [["--runs", "nb:03"], ["--runs", "ga:3"], ["--runs", "ga:1,ga:1"],
                                  ["--runs", "ga:1", "--from", "01"], ["--runs", "ga:1", "--fresh"],
                                  ["--runs", "ga:1", "--at-once", "0"]])
def test_invalid_jobs_are_refused_before_anything_starts(runner, tmp_path, monkeypatch, capsys, argv):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "run_side_by_side", lambda *a, **k: pytest.fail("jobs were started"))
    assert run_main(runner, tmp_path, argv) == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "info.json").exists()


def test_a_job_whose_input_nobody_makes_is_refused_before_anything_starts(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "run_side_by_side", lambda *a, **k: pytest.fail("jobs were started"))
    assert run_main(runner, tmp_path, ["--runs", "es_seeded:1,ga:0"]) == 2
    error = capsys.readouterr().err
    assert "es_seeded:1" in error and "baseline/seed1/dagger.json" in error and "ga:0" not in error
    refined = tmp_path / "results" / "runs" / "baseline" / "seed1" / "dagger.json"
    refined.parent.mkdir(parents=True)
    checkpoint.write_json(refined, {"settings": checkpoint.settings_record()})     # a refinement of these settings
    refined.with_name("dagger.keras").write_bytes(b"network")
    started_with = []
    monkeypatch.setattr(runner, "run_side_by_side",
                        lambda job_list, at_once, *a, **k: started_with.append(([j.label for j in job_list], at_once))
                        or 0)
    assert run_main(runner, tmp_path, ["--runs", "es_seeded:1,ga:0", "--at-once", "2"]) == 0
    assert started_with == [(["es_seeded:1", "ga:0"], 2)]
    assert run_main(runner, tmp_path, ["--runs", "baseline:0,es_seeded:0,nb:01,nb:02"]) == 0   # made in the list


def test_jobs_side_by_side_take_the_results_lock(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "run_side_by_side", lambda *a, **k: pytest.fail("jobs were started"))
    with hold_lock(tmp_path / "results"):
        assert run_main(runner, tmp_path, ["--runs", "ga:0"]) == 4


def test_fresh_also_removes_the_side_by_side_files(runner):
    assert {"status", "train_logs", "machines"} <= set(runner.FRESH_REMOVE)


def test_a_notebook_job_runs_this_script_and_writes_its_status_to_its_own_file(runner, tmp_path):
    status = tmp_path / "status" / "nb-02.json"
    command = runner.job_command(jobs.parse_job("nb:02"), status)
    assert command[-2:] == ["--notebook-job", "02"] and command[1].endswith("run_notebooks.py")
    env = runner.job_env(status)
    assert env["EC_RACING_STATUS"] == str(status) and env["CUDA_VISIBLE_DEVICES"] == "-1"
    seen = subprocess.run([sys.executable, "-c", "from racing import config; print(config.STATUS_PATH)"],
                          env=env, capture_output=True, text=True, cwd=SCRIPT.parent.parent).stdout.strip()
    assert seen == str(status)                                          # also inside the notebook's kernel
    assert runner.job_command(jobs.parse_job("ga:1"), status)[1:4] == ["-m", "racing.train", "ga"]


def test_a_notebook_job_executes_one_notebook_and_saves_it(runner, tmp_path, monkeypatch):
    folder = three_notebooks(tmp_path)
    ran = []
    monkeypatch.setattr(runner, "execute", lambda path, kernel_dir, output=None: ran.append(path.name))
    assert runner.main(["--notebook-job", "02"], notebook_dir=folder, log_path=tmp_path / "log.txt") == 0
    assert ran == ["02_b.ipynb"]


def test_archive_moves_what_fresh_would_delete(runner, tmp_path):
    results = tmp_path / "results"
    for name in ("runs/ga/seed0", "champions/ga", "pilots/x", "archive/older", "app_runs/ga"):
        (results / name).mkdir(parents=True)
    (results / "run_log.txt").write_text("old")
    target = runner.archive_results(results, "revision3.1")
    assert (target / "runs" / "ga" / "seed0").is_dir() and (target / "run_log.txt").read_text() == "old"
    assert not (results / "runs").exists() and not (results / "run_log.txt").exists()
    assert (results / "pilots" / "x").is_dir() and (results / "app_runs" / "ga").is_dir()
    assert (results / "archive" / "older").is_dir()
    with pytest.raises(FileExistsError):
        runner.archive_results(results, "revision3.1")


@pytest.mark.parametrize("extra", [["--fresh"], ["--from", "02"], ["--runs", "ga:0"]])
def test_archive_runs_nothing_else(runner, tmp_path, capsys, extra):
    three_notebooks(tmp_path)
    assert run_main(runner, tmp_path, ["--archive", "x", *extra]) == 2


def test_archive_alone_moves_the_results_and_exits(runner, tmp_path, monkeypatch, capsys):
    three_notebooks(tmp_path)
    (tmp_path / "results" / "runs" / "ga").mkdir(parents=True)
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("a notebook was run"))
    assert run_main(runner, tmp_path, ["--archive", "revision3.1"]) == 0
    assert (tmp_path / "results" / "archive" / "revision3.1" / "runs" / "ga").is_dir()
    assert not (tmp_path / "results" / "runs").exists()
    assert "revision3.1" in capsys.readouterr().out
    assert run_main(runner, tmp_path, ["--archive", "revision3.1"]) == 2   # never overwrites an archive


def stopping_runner(tmp_path):
    """The runner in its own process with two quiet runs side by side and a seeded ES run waiting for a baseline
    that only the (never finishing) first run would make. Returns the process and the run processes' ids."""
    pids = [tmp_path / "a.pid", tmp_path / "b.pid"]
    script = (f"import importlib.util, signal, sys\n"
              f"from racing import jobs\n"
              f"spec = importlib.util.spec_from_file_location('run_notebooks', {str(SCRIPT)!r})\n"
              f"runner = importlib.util.module_from_spec(spec)\n"
              f"spec.loader.exec_module(runner)\n"
              f"signal.signal(signal.SIGTERM, runner.stop_on_kill)\n"
              f"pids = {[str(p) for p in pids]!r}\n"
              f"def command(job, status_file):\n"
              f"    return [sys.executable, '-c', f'import os, time; open({{pids[job.seed]!r}}, \"w\").write(str(os.getpid())); "
              f"time.sleep(600)']\n"
              f"sys.exit(runner.run_side_by_side(jobs.parse_jobs('baseline:0,ga:1,es_seeded:0'), 2, "
              f"{str(tmp_path / 'log.txt')!r}, {str(tmp_path / 'info.json')!r}, {str(tmp_path / 'status')!r}, "
              f"{str(tmp_path / 'logs')!r}, monitor=False, command=command, results_dir={str(tmp_path / 'r')!r}))\n")
    for path in jobs.needs(jobs.parse_job("baseline:0"), tmp_path / "r"):      # baseline:0's own inputs exist
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    process = subprocess.Popen([sys.executable, "-c", script], cwd=SCRIPT.parent.parent)
    deadline = time.time() + 60
    while not all(p.exists() and p.read_text() for p in pids):
        assert process.poll() is None and time.time() < deadline, "the runs never started"
        time.sleep(0.1)
    return process, [int(p.read_text()) for p in pids]


@pytest.mark.slow
@pytest.mark.parametrize("stop", [signal.SIGINT, signal.SIGTERM])
def test_ctrl_c_or_kill_stops_every_running_job_and_lists_the_waiting_ones(runner, tmp_path, stop):
    process, children = stopping_runner(tmp_path)
    try:
        process.send_signal(stop)
        assert process.wait(timeout=STOP_SECONDS) == runner.STOPPED
        time.sleep(0.5)
        assert not any(Path(f"/proc/{pid}").exists() for pid in children)
        log = (tmp_path / "log.txt").read_text()
        assert "STOPPED baseline:0 ga:1" in log and "not started: es_seeded:0" in log
        info = json.loads((tmp_path / "info.json").read_text())
        assert all("stopped" in e for e in info["trained"])
    finally:
        if process.poll() is None:
            process.kill()
        for pid in children:
            if Path(f"/proc/{pid}").exists():
                os.kill(pid, signal.SIGKILL)


def test_check_reports_earlier_results_of_other_settings_without_running_anything(runner, tmp_path, monkeypatch,
                                                                                  capsys):
    three_notebooks(tmp_path)
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("a notebook was run"))
    assert run_main(runner, tmp_path, ["--check"]) == 0
    old = tmp_path / "results" / "runs" / "ga" / "seed0" / "summary.json"
    old.parent.mkdir(parents=True)
    checkpoint.write_json(old, {"settings": {**checkpoint.settings_record(), "GRIP": 99.0}})
    assert run_main(runner, tmp_path, ["--check"]) == 3
    assert "GRIP" in capsys.readouterr().err
    with hold_lock(tmp_path / "results"):
        assert run_main(runner, tmp_path, ["--check"]) == 3              # read-only: no lock needed
