"""Runs the experiment notebooks in order and saves their outputs into the notebook files (revision 3 §8).

Run with:        python scripts/run_notebooks.py               (continues where an earlier run stopped)
Start over:      python scripts/run_notebooks.py --fresh       (removes the earlier results first)
From a notebook: python scripts/run_notebooks.py --from 04
In background:   mkdir -p results
                 nohup python scripts/run_notebooks.py > results/run_console.txt 2>&1 &
Progress:        python -m racing.progress                     (from another terminal)

Exit codes: 0 finished; 1 a notebook failed; 2 invalid options; 3 earlier results were produced with different
settings (run again with --fresh).
"""
import argparse
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from racing import checkpoint, config, paths, progress  # noqa: E402

NOTEBOOK_DIR = PROJECT / "notebooks"
KERNEL = "ec-racing"
FRESH_REMOVE = ("runs", "champions", "tables", "figures", "dataset.npz", "dataset_info.json", "status.json",
                "run_log.txt", "run_info.json")       # under results/; pilots, app_runs and archive are kept
RECORD_FILES = ("summary.json", "dagger.json")        # run records that carry the settings they were produced with
STATE_FILES = (checkpoint.CHECKPOINT_NAME, "dagger_state.pkl")


def now():
    return datetime.now().isoformat(timespec="seconds")


def log(message, path):
    """A milestone of the run, appended to the run log; the progress monitor prints it (revision 3 §8.3)."""
    progress.milestone(message, path)


def processor():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def power_profile():
    try:
        result = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def versions():
    from importlib.metadata import PackageNotFoundError, version
    found = {"python": platform.python_version()}
    for package in ("numpy", "tensorflow", "keras", "keras-tuner"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "missing"
    return found


def git(*args):
    """Output of a git command in the project, or None without git or outside a repository."""
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def code_version():
    """The commit the run starts from, and whether tracked files differ from it ("unknown" without git)."""
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit.strip() if commit else "unknown",
            "git_dirty": bool(status.strip()) if status is not None else "unknown"}


def notebook_names(folder, start=None):
    """Experiment notebooks (named NN_*.ipynb) in order, from notebook `start` on."""
    names = sorted(p.name for p in Path(folder).glob("[0-9][0-9]_*.ipynb"))
    return [name for name in names if start is None or name[:2] >= start]


def execute(path, kernel_dir):
    """Run every cell of one notebook in a new kernel on this interpreter, with the notebook's folder as working
    directory, and save the outputs into the file, also when a cell fails."""
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client.manager import KernelManager
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name=KERNEL, kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir)]))
    client = NotebookClient(notebook, km=manager, timeout=None, resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        if manager.is_alive():
            manager.shutdown_kernel(now=True)          # a finished notebook's kernel does not stay resident


def write_info(info, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(info, indent=2))


def fresh_start(results_dir):
    """--fresh: remove the results of earlier runs so the experiment starts over. Pilots, viewer runs and the
    archive are kept."""
    paths.remove(*(Path(results_dir) / name for name in FRESH_REMOVE))


def recorded_settings(path):
    """The settings recorded in a run record or checkpoint, or None if it has none (an earlier version) or cannot be
    read."""
    try:
        if path.name in RECORD_FILES:
            data = json.loads(path.read_text())
        else:
            data = checkpoint.load(path)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, AttributeError, ImportError):
        return None
    return data.get("settings") if isinstance(data, dict) else None


def settings_problems(runs_dir):
    """One message per run record or checkpoint under runs_dir whose recorded settings differ from the current ones
    (revision 3 §8.2). A record without readable settings, such as one of revision 2.0, counts as different, as it
    does for checkpoint.completed, so the run stops here rather than inside a notebook."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    problems = []
    for path in sorted(runs_dir.rglob("*")):
        if path.name not in RECORD_FILES + STATE_FILES:
            continue
        recorded = recorded_settings(path)
        if recorded is None:
            problems.append(f"{path.relative_to(runs_dir).as_posix()}: no readable settings record "
                            "(made by an earlier version)")
            continue
        try:
            checkpoint.check_settings(recorded)
        except checkpoint.SettingsMismatch as error:
            problems.append(f"{path.relative_to(runs_dir).as_posix()}: {error}")
    return problems


def run(notebook_dir=NOTEBOOK_DIR, start=None, log_path=config.RUN_LOG_PATH, info_path=config.RUN_INFO_PATH,
        status_path=config.STATUS_PATH, monitor=True, out=None):
    """Execute the notebooks in order while the progress monitor shows the live status line and the milestones.
    The record keeps the entries of notebooks run earlier; each notebook's entry is replaced when it runs again."""
    from ipykernel.kernelspec import write_kernel_spec

    notebook_dir = Path(notebook_dir)
    info = json.loads(Path(info_path).read_text()) if Path(info_path).exists() else {}
    info.update({"processor": processor(), "logical_processors": os.cpu_count(), "n_jobs": config.N_JOBS,
                 "power_profile": power_profile(), "versions": versions(),
                 "settings": checkpoint.settings_record(), **code_version()})
    info.setdefault("notebooks", [])
    info.setdefault("runs", []).append({"started": now(), "from": start})
    names = notebook_names(notebook_dir, start)
    paths.remove(status_path)                          # the live line shows only this run's status
    watcher = progress.Monitor(status_path, log_path, out=out) if monitor else None
    if watcher is not None:
        watcher.start()
    try:
        log(f"run started: {' '.join(names)} ({config.N_JOBS} workers, power profile {info['power_profile']})",
            log_path)
        with tempfile.TemporaryDirectory() as kernels:
            write_kernel_spec(Path(kernels) / KERNEL)
            for name in names:
                entry = {"notebook": name, "started": now()}
                info["notebooks"] = [e for e in info["notebooks"] if e["notebook"] != name] + [entry]
                write_info(info, info_path)
                log(f"start {name}", log_path)
                began = time.time()
                try:
                    execute(notebook_dir / name, kernels)
                except Exception as error:             # a failing cell; its output is saved in the notebook
                    text = str(error).strip()
                    entry["failed"] = text.splitlines()[-1] if text else type(error).__name__
                    entry["seconds"] = round(time.time() - began, 1)
                    write_info(info, info_path)
                    if "SettingsMismatch" in text:
                        log(f"STOPPED {name}: results on disk were produced with different settings; "
                            f"run again with --fresh to start over", log_path)
                        return 3
                    log(f"FAILED {name}: {entry['failed']}", log_path)
                    return 1
                entry["finished"] = now()
                entry["seconds"] = round(time.time() - began, 1)
                write_info(info, info_path)
                log(f"done {name} in {entry['seconds'] / 60:.1f} min", log_path)
        log("run finished", log_path)
        return 0
    finally:
        if watcher is not None:
            watcher.stop()


def main(argv=None, notebook_dir=NOTEBOOK_DIR, log_path=config.RUN_LOG_PATH, info_path=config.RUN_INFO_PATH,
         results_dir=config.RESULTS_DIR, status_path=config.STATUS_PATH, monitor=True, out=None):
    parser = argparse.ArgumentParser(description="Run the EC Racing experiment notebooks in order. An interrupted "
                                                 "run continues where it stopped.")
    parser.add_argument("--from", dest="start", help="number of the first notebook to run, for example 04")
    parser.add_argument("--fresh", action="store_true", help="remove the earlier results and start over")
    args = parser.parse_args(argv)
    if args.start is not None:
        prefixes = sorted({name[:2] for name in notebook_names(notebook_dir)})
        if args.start not in prefixes:
            print(f"error: --from must be the number of an existing notebook: {', '.join(prefixes)}",
                  file=sys.stderr)
            return 2
        if args.fresh:
            print("error: --fresh starts over from the first notebook and cannot be combined with --from",
                  file=sys.stderr)
            return 2
    if args.fresh:
        fresh_start(results_dir)
    problems = settings_problems(Path(results_dir) / "runs")
    if problems:
        print("error: earlier results were produced with different settings:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("Run again with --fresh to remove them and start over.", file=sys.stderr)
        return 3
    return run(notebook_dir, args.start, log_path, info_path, status_path, monitor, out)


if __name__ == "__main__":
    sys.exit(main())
