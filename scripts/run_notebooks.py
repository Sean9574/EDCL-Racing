"""Runs the experiment notebooks in order and saves their outputs into the notebook files (revision 3 §8).

Run with:        python scripts/run_notebooks.py               (continues where an earlier run stopped)
Start over:      python scripts/run_notebooks.py --fresh       (removes the earlier results first)
From a notebook: python scripts/run_notebooks.py --from 04
In background:   mkdir -p results
                 nohup python scripts/run_notebooks.py > results/run_console.txt 2>&1 &
Progress:        python -m racing.progress                     (from another terminal)

Exit codes: 0 finished; 1 a notebook failed; 2 invalid options; 3 earlier results were produced with different
settings (run again with --fresh) or cannot be read (delete the files named); 4 another runner is using the same
results.
"""
import argparse
import fcntl
import os
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
KERNEL_ENV = {"CUDA_VISIBLE_DEVICES": "-1",      # CPU only (spec §9): TensorFlow does not look for a GPU,
              "TF_CPP_MIN_LOG_LEVEL": "3"}      # and its C++ log stays out of the progress display
FRESH_REMOVE = ("runs", "champions", "tables", "figures", "dataset.npz", "dataset_info.json", "status.json",
                "run_log.txt", "run_info.json")       # under results/; pilots, app_runs and archive are kept
RECORD_STATES = {"summary.json": checkpoint.CHECKPOINT_NAME,      # run records that carry their settings, and the
                 "dagger.json": "dagger_state.pkl"}                # checkpoint a run finishes again from
RECORD_FILES = tuple(RECORD_STATES)
STATE_FILES = tuple(RECORD_STATES.values())
LOCK_NAME = ".runner.lock"                            # under results/; held by the running runner


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


def code_change(info, version):
    """A warning if this start's commit differs from the previous start's in the run record `info`, else None. The
    results of notebooks run earlier were produced with the earlier code."""
    runs = info.get("runs") or [{}]
    previous, current = runs[-1].get("git_commit", info.get("git_commit")), version["git_commit"]
    if previous in (None, "unknown") or current == "unknown" or previous == current:
        return None
    return (f"warning: the code changed since the previous start (commit {previous[:7]}, now {current[:7]}); "
            "notebooks run earlier used the earlier code. Do not git pull during a run.")


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
    checkpoint.write_json(path, info)


def lock_results(results_dir):
    """An exclusive lock on results_dir/.runner.lock, so two runners never work on the same results. Returns the
    open lock file, which records this process and releases the lock when closed, or None if another runner holds
    it. The lock file is not inherited by the notebook kernels."""
    path = Path(results_dir) / LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    handle.truncate(0)
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def lock_holder(results_dir):
    """The process id recorded by the runner holding the lock, or None if unknown."""
    try:
        text = (Path(results_dir) / LOCK_NAME).read_text().strip()
    except OSError:
        return None
    return text if text.isdigit() else None


def fresh_start(results_dir):
    """--fresh: remove the results of earlier runs so the experiment starts over. Pilots, viewer runs and the
    archive are kept."""
    paths.remove(*(Path(results_dir) / name for name in FRESH_REMOVE))


def recorded_settings(path):
    """The settings recorded in a run record or checkpoint, or None if it has none (an earlier version). Raises
    checkpoint.UnreadableRecord if the file cannot be read."""
    if path.name in RECORD_FILES:
        data = checkpoint.read_record(path)
    else:
        try:
            data = checkpoint.load(path)
        except Exception as error:                     # any damage to a pickle: truncated, zeroed, foreign
            raise checkpoint.UnreadableRecord(f"{path} cannot be read ({type(error).__name__})") from error
    return data.get("settings") if isinstance(data, dict) else None


def finishes_from_checkpoint(path):
    """Whether the run of an unreadable record finishes again by itself: its checkpoint is next to it and readable
    (checkpoint.completed)."""
    state = path.parent / RECORD_STATES[path.name]
    if not state.exists():
        return False
    try:
        recorded_settings(state)
    except checkpoint.UnreadableRecord:
        return False
    return True


def check_results(runs_dir):
    """(settings problems, unreadable files) of the run records and checkpoints under runs_dir (revision 3 §8.2),
    so the run stops here rather than inside a notebook. A record whose settings differ, or that has none (such as
    one of revision 2.0, which checkpoint.completed refuses), is a settings problem, one message each. A file that
    cannot be read, such as one cut short by a reboot, is listed by its path; a record whose run finishes again from
    its checkpoint is not."""
    runs_dir = Path(runs_dir)
    problems, unreadable = [], []
    if not runs_dir.exists():
        return problems, unreadable
    for path in sorted(runs_dir.rglob("*")):
        if path.name not in RECORD_FILES + STATE_FILES:
            continue
        name = path.relative_to(runs_dir).as_posix()
        try:
            recorded = recorded_settings(path)
        except checkpoint.UnreadableRecord:
            if path.name in STATE_FILES or not finishes_from_checkpoint(path):
                unreadable.append(path)
            continue
        if recorded is None:
            problems.append(f"{name}: no settings record (made by an earlier version)")
            continue
        try:
            checkpoint.check_settings(recorded)
        except checkpoint.SettingsMismatch as error:
            problems.append(f"{name}: {error}")
    return problems, unreadable


def run(notebook_dir=NOTEBOOK_DIR, start=None, log_path=config.RUN_LOG_PATH, info_path=config.RUN_INFO_PATH,
        status_path=config.STATUS_PATH, monitor=True, out=None):
    """Execute the notebooks in order while the progress monitor shows the live status line and the milestones.
    The record keeps the entries of notebooks run earlier; each notebook's entry is replaced when it runs again.
    Every start is recorded with its code version, and a commit that differs from the previous start's is reported."""
    from ipykernel.kernelspec import write_kernel_spec

    notebook_dir = Path(notebook_dir)
    info = checkpoint.read_record(info_path) if Path(info_path).exists() else {}
    version = code_version()
    warning = code_change(info, version)
    info.update({"processor": processor(), "logical_processors": os.cpu_count(), "n_jobs": config.N_JOBS,
                 "power_profile": power_profile(), "versions": versions(),
                 "settings": checkpoint.settings_record(), **version})
    info.setdefault("notebooks", [])
    info.setdefault("runs", []).append({"started": now(), "from": start, **version})   # each start's code version
    if warning is not None:
        print(warning, file=sys.stderr)
        log(warning, log_path)
    names = notebook_names(notebook_dir, start)
    paths.remove(status_path)                          # the live line shows only this run's status
    watcher = progress.Monitor(status_path, log_path, out=out) if monitor else None
    if watcher is not None:
        watcher.start()
    try:
        log(f"run started: {' '.join(names)} ({config.N_JOBS} workers, power profile {info['power_profile']})",
            log_path)
        with tempfile.TemporaryDirectory() as kernels:
            write_kernel_spec(Path(kernels) / KERNEL, overrides={"env": KERNEL_ENV})
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
    lock = lock_results(results_dir)
    if lock is None:
        holder = lock_holder(results_dir)
        who = f"another runner (process {holder})" if holder else "another runner"
        print(f"error: {who} is using {results_dir}; wait for it to finish or stop it, then run again",
              file=sys.stderr)
        return 4
    try:
        if args.fresh:
            fresh_start(results_dir)
        problems, unreadable = check_results(Path(results_dir) / "runs")
        if unreadable:
            print("error: these result files cannot be read, probably because a crash or reboot cut them short:",
                  file=sys.stderr)
            for path in unreadable:
                print(f"  {path}", file=sys.stderr)
            print("Delete only these files and run again: the steps they belong to are redone, every other result "
                  "is kept.", file=sys.stderr)
        if problems:
            print("error: earlier results were produced with different settings:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print("Run again with --fresh to remove them and start over.", file=sys.stderr)
        if problems or unreadable:
            return 3
        return run(notebook_dir, args.start, log_path, info_path, status_path, monitor, out)
    finally:
        lock.close()                                   # releases the lock


if __name__ == "__main__":
    sys.exit(main())
