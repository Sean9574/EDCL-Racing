"""Runs the experiment notebooks in order and saves their outputs into the notebook files (spec §3.3).

Run with:        python scripts/run_notebooks.py               (continues where an earlier run stopped)
Start over:      python scripts/run_notebooks.py --fresh       (removes the earlier results first)
From a notebook: python scripts/run_notebooks.py --from 04
Side by side:    python scripts/run_notebooks.py --runs nb:01,search:kt_random,baseline:0,ga:1 --at-once 8
                 (trains these jobs, up to eight at a time: runs of any method without their notebooks, the Keras
                 Tuner searches and notebook 01; each starts as soon as its inputs exist. The notebooks find the
                 runs and searches complete later. scripts/fleet.py starts this on several machines.)
Archive:         python scripts/run_notebooks.py --archive revision3.1
                 (moves the earlier results to results/archive/revision3.1/ and exits; the next start begins afresh)
In background:   mkdir -p results
                 nohup python scripts/run_notebooks.py > results/run_console.txt 2>&1 &
Progress:        python -m racing.progress                     (from another terminal)

Stop:            Ctrl+C, or from another terminal: pkill -f scripts/run_notebooks.py
                 (the notebook's kernel is ended too; run again to continue where it stopped)

Exit codes: 0 finished; 1 a notebook or run failed; 2 invalid options, or a run's input is missing; 3 earlier
results were produced with different settings (run again with --fresh) or cannot be read (delete the files named);
4 another runner is using the same results; 130 stopped with Ctrl+C or kill.
"""
import argparse
import fcntl
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from racing import checkpoint, config, fastsim, jobs, paths, progress  # noqa: E402

NOTEBOOK_DIR = PROJECT / "notebooks"
KERNEL = "ec-racing"
KERNEL_ENV = {"CUDA_VISIBLE_DEVICES": "-1",      # CPU only (spec §3.3): TensorFlow does not look for a GPU,
              "TF_CPP_MIN_LOG_LEVEL": "3"}      # and its C++ log stays out of the progress display
KERNEL_LOG = "kernel_log.txt"                         # next to the run log: what the notebook kernels print themselves
FRESH_REMOVE = ("runs", "champions", "tables", "figures", "dataset.npz", "dataset_info.json", "status.json",
                "run_log.txt", "run_info.json", KERNEL_LOG, "status", "train_logs", "machines")
                # under results/; pilots, app_runs and archive are kept
RECORD_STATES = {"summary.json": checkpoint.CHECKPOINT_NAME,      # run records that carry their settings, and the
                 "dagger.json": "dagger_state.pkl"}                # checkpoint a run finishes again from
RECORD_FILES = tuple(RECORD_STATES)
STATE_FILES = tuple(RECORD_STATES.values())
LOCK_NAME = ".runner.lock"                            # under results/; held by the running runner
STOPPED = 130                                         # exit code after Ctrl+C or kill (128 + SIGINT)
POLL_SECONDS = 0.5                                    # how often runs trained side by side are checked
STOP_GRACE = 20.0                                     # s a stopped training process may take before it is killed


def now():
    return datetime.now().isoformat(timespec="seconds")


def log(message, path):
    """A milestone of the run, appended to the run log; the progress monitor prints it (spec §3.3)."""
    progress.milestone(message, path)


def processor():
    return checkpoint.processor()


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


def write_kernel(kernel_dir):
    """The kernel spec the notebooks run with: this interpreter, GPUs hidden and TensorFlow's C++ log off."""
    from ipykernel.kernelspec import write_kernel_spec
    write_kernel_spec(Path(kernel_dir) / KERNEL, overrides={"env": KERNEL_ENV})


def execute(path, kernel_dir, output=None):
    """Run every cell of one notebook in a new kernel on this interpreter, with the notebook's folder as working
    directory, and save the outputs into the file, also when a cell fails. What the kernel process prints itself
    (library messages, not cell output) goes to the open file `output`, so it stays out of the progress display."""
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client.manager import KernelManager
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name=KERNEL, kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir)]))
    client = NotebookClient(notebook, km=manager, timeout=None, resources={"metadata": {"path": str(path.parent)}})
    errors = []

    def work():
        try:
            client.execute(**({} if output is None else {"stdout": output, "stderr": output}))
        except BaseException as error:                # raised again below, in the main thread
            errors.append(error)

    # The client waits for the kernel's next message without a time limit, so a notebook that prints nothing for
    # hours would hold off Ctrl+C and kill. It runs in a thread; the main thread only waits and can always stop.
    worker = threading.Thread(target=work, name=f"notebook {path.name}", daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.2)
    finally:
        if manager.is_alive():
            manager.shutdown_kernel(now=True)          # a finished or stopped notebook's kernel does not stay resident
        nbformat.write(notebook, path)
    if errors:
        raise errors[0]


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
    """(settings problems, unreadable files) of the run records and checkpoints under runs_dir (spec §3.3),
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
    notebook_dir = Path(notebook_dir)
    info = checkpoint.read_record(info_path) if Path(info_path).exists() else {}
    version = code_version()
    warning = code_change(info, version)
    info.update({"processor": processor(), "logical_processors": os.cpu_count(),
                 "threads": config.thread_label(), "n_jobs": config.N_JOBS,
                 "simulator": fastsim.description(),
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
        log(f"run started: {' '.join(names)} ({config.N_JOBS} workers, threads {config.thread_label()}, "
            f"simulator {fastsim.description()}, power profile {info['power_profile']})", log_path)
        with tempfile.TemporaryDirectory() as kernels, open(Path(log_path).with_name(KERNEL_LOG), "a") as output:
            write_kernel(kernels)
            for name in names:
                entry = {"notebook": name, "started": now()}
                info["notebooks"] = [e for e in info["notebooks"] if e["notebook"] != name] + [entry]
                write_info(info, info_path)
                log(f"start {name}", log_path)
                began = time.time()
                try:
                    execute(notebook_dir / name, kernels, output)
                except KeyboardInterrupt:              # Ctrl+C or kill: the kernel has been ended
                    entry["stopped"] = now()
                    entry["seconds"] = round(time.time() - began, 1)
                    write_info(info, info_path)
                    log(f"STOPPED {name}: the run was stopped; run again to continue where it stopped", log_path)
                    return STOPPED
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


def train_command(approach, seed, status_file):
    """The process that trains one run side by side: racing/train.py, with its own status file."""
    return [sys.executable, "-m", "racing.train", approach, str(seed), "--status", str(status_file)]


def job_command(job, status_file):
    """The process of one job: notebook 01 (this script with --notebook-job), a Keras Tuner search or a training run
    (racing/train.py)."""
    if job.is_notebook:
        return [sys.executable, str(Path(__file__).resolve()), "--notebook-job", job.notebook[:2]]
    if job.is_search:
        return [sys.executable, "-m", "racing.train", "search", job.search, "--status", str(status_file)]
    return train_command(job.approach, job.seed, status_file)


def job_env(status_file):
    """A job's environment: the kernels' (CPU only, quiet TensorFlow) and its own status file, which a notebook's
    kernel inherits (racing/config.py reads EC_RACING_STATUS)."""
    return {**os.environ, **KERNEL_ENV, "PYTHONUNBUFFERED": "1", "EC_RACING_STATUS": str(status_file)}


def job_stem(job):
    """File name stem of a job's status file and output log: ga-seed1, nb-01, search-kt_random."""
    if job.is_notebook:
        return f"nb-{job.notebook[:2]}"
    if job.is_search:
        return f"search-{job.search}"
    return f"{job.approach}-seed{job.seed}"


def last_line(path):
    try:
        lines = [line for line in Path(path).read_text(errors="replace").splitlines() if line.strip()]
    except OSError:
        return ""
    return lines[-1].strip() if lines else ""


def stop_processes(processes, grace=STOP_GRACE):
    """End training processes and their worker processes: each runs in its own process group, which gets SIGTERM,
    then SIGKILL after `grace` seconds. A run saves its checkpoint after every generation, so it continues from the
    last finished generation; a notebook job ends its kernel."""
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.time() + grace
    for process in processes:
        try:
            process.wait(timeout=max(deadline - time.time(), 0.1))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()


def archive_results(results_dir, name):
    """--archive NAME: move what --fresh would delete to results/archive/NAME/, so the next start begins afresh
    (spec §3.3). Pilots, viewer runs and earlier archives stay. Refuses an existing archive."""
    target = Path(results_dir) / "archive" / name
    if target.exists():
        raise FileExistsError(f"{target} already exists; choose another archive name")
    target.mkdir(parents=True)
    for item in FRESH_REMOVE:
        source = Path(results_dir) / item
        if source.exists():
            shutil.move(str(source), str(target / item))
    return target


def notebook_job(number, notebook_dir=NOTEBOOK_DIR, log_path=config.RUN_LOG_PATH):
    """--notebook-job NN: execute one notebook and save its outputs, for the side-by-side runner, which holds the
    results lock and logs the job. kill ends the kernel too. Returns 0, 1 if a cell failed, or STOPPED."""
    names = [name for name in notebook_names(notebook_dir) if name[:2] == number]
    if not names:
        print(f"error: no notebook {number} in {notebook_dir}", file=sys.stderr)
        return 2
    previous = signal.signal(signal.SIGTERM, stop_on_kill)
    try:
        with tempfile.TemporaryDirectory() as kernels, open(Path(log_path).with_name(KERNEL_LOG), "a") as output:
            write_kernel(kernels)
            execute(Path(notebook_dir) / names[0], kernels, output)
    except KeyboardInterrupt:
        return STOPPED
    except Exception as error:                         # a failing cell; its output is saved in the notebook
        text = str(error).strip()
        print(text.splitlines()[-1] if text else type(error).__name__, file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


def run_side_by_side(job_list, at_once, log_path=config.RUN_LOG_PATH, info_path=config.RUN_INFO_PATH,
                     status_dir=config.STATUS_DIR, logs_dir=config.TRAIN_LOGS_DIR, monitor=True, out=None,
                     command=job_command, results_dir=None):
    """Run the jobs `job_list` (racing/jobs.py) side by side, at most `at_once` at a time, each in its own process
    with its own status file and output log, while the monitor shows one line per job (spec §3.3). A job
    starts, in the order of the list, as soon as a slot is free and its inputs exist; a waiting job holds no slot. A
    job that fails is reported, and so are the jobs left waiting for what it would have made; the others go on. A
    finished run is never trained again; an interrupted one continues from its checkpoint. Returns 0, 1 if a job
    failed or was skipped, or STOPPED after Ctrl+C or kill, which stops every running job."""
    results_dir = Path(log_path).parent if results_dir is None else Path(results_dir)
    info = checkpoint.read_record(info_path) if Path(info_path).exists() else {}
    version = code_version()
    warning = code_change(info, version)
    info.update({"processor": processor(), "logical_processors": os.cpu_count(),
                 "threads": config.thread_label(), "n_jobs": config.N_JOBS,
                 "simulator": fastsim.description(),
                 "power_profile": power_profile(), "versions": versions(),
                 "settings": checkpoint.settings_record(), **version})
    names = [job.label for job in job_list]
    info.setdefault("runs", []).append({"started": now(), "side_by_side": names, "at_once": at_once,
                                        "host": platform.node(), **version})
    info.setdefault("trained", [])
    if warning is not None:
        print(warning, file=sys.stderr)
        log(warning, log_path)
    status_dir, logs_dir = Path(status_dir), Path(logs_dir)
    paths.remove(status_dir)                           # the monitor shows only this start's jobs
    status_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    pending, active, failed, ended = list(job_list), {}, [], set()

    def record(entry):
        info["trained"] = [e for e in info["trained"] if e["run"] != entry["run"]] + [entry]
        write_info(info, info_path)

    def blocked(job):
        """Why `job` can never start (an input is missing and every listed job that makes it has ended), or None."""
        for path in jobs.needs(job, results_dir):
            if path.exists():
                continue
            makers = [other.label for other in job_list if path in jobs.makes(other, results_dir)]
            if all(label in ended for label in makers):
                return f"waits for {path.name}, which {' '.join(makers) or 'no listed job'} did not make"
        return None

    watcher = progress.Monitor(None, log_path, out=out, status_dir=status_dir) if monitor else None
    if watcher is not None:
        watcher.start()
    try:
        log(f"run started: {' '.join(names)}, {at_once} at a time ({config.N_JOBS} workers each, threads "
            f"{config.thread_label()}, simulator {fastsim.description()}, {platform.node()})", log_path)
        while pending or active:
            for job in list(pending):
                reason = blocked(job)
                if reason is not None:
                    pending.remove(job)
                    ended.add(job.label)
                    failed.append(job.label)
                    record({"run": job.label, "skipped": reason})
                    log(f"SKIPPED {job.label}: {reason}", log_path)
            for job in list(pending):
                if len(active) >= at_once:
                    break
                if not jobs.ready(job, results_dir):
                    continue
                pending.remove(job)
                stem = job_stem(job)
                status_file = status_dir / f"{stem}.json"
                output = open(logs_dir / f"{stem}.log", "a")
                process = subprocess.Popen(command(job, status_file), cwd=PROJECT, env=job_env(status_file),
                                           stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           start_new_session=True)   # Ctrl+C reaches the runner, which stops it
                entry = {"run": job.label, "started": now()}
                record(entry)
                active[process] = (job.label, status_file, output, entry, time.time())
                log(f"start {job.label}", log_path)
            for process in list(active):
                code = process.poll()
                if code is None:
                    continue
                name, status_file, output, entry, began = active.pop(process)
                output.close()
                paths.remove(status_file)
                ended.add(name)
                entry["seconds"] = round(time.time() - began, 1)
                if code == 0:
                    entry["finished"] = now()
                    log(f"done {name} in {entry['seconds'] / 60:.1f} min", log_path)
                else:
                    entry["failed"] = last_line(output.name) or f"exit code {code}"
                    failed.append(name)
                    log(f"FAILED {name}: {entry['failed']} (output: {output.name})", log_path)
                write_info(info, info_path)
            if pending or active:
                time.sleep(POLL_SECONDS)
        log("runs finished" if not failed else f"runs finished; failed or skipped: {' '.join(failed)}", log_path)
        return 1 if failed else 0
    except KeyboardInterrupt:                          # Ctrl+C or kill: every running job stops at its checkpoint
        stop_processes(list(active))
        for name, status_file, output, entry, began in active.values():
            output.close()
            paths.remove(status_file)
            entry["stopped"] = now()
            entry["seconds"] = round(time.time() - began, 1)
        write_info(info, info_path)
        stopped = [name for name, *_ in active.values()]
        waiting = f"; not started: {' '.join(job.label for job in pending)}" if pending else ""
        log(f"STOPPED {' '.join(stopped) or 'the jobs'}: run again to continue where they stopped{waiting}", log_path)
        return STOPPED
    finally:
        if watcher is not None:
            watcher.stop()


def report_results(results_dir):
    """Check the earlier results (check_results) and print what stops a run; True if anything does."""
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
        print("Run again with --fresh to remove them and start over, or keep them with --archive NAME.",
              file=sys.stderr)
    return bool(problems or unreadable)


def stop_on_kill(signum, frame):
    """kill (SIGTERM) stops the run like Ctrl+C: the notebook's kernel is ended and the run log says so."""
    raise KeyboardInterrupt


def main(argv=None, notebook_dir=NOTEBOOK_DIR, log_path=config.RUN_LOG_PATH, info_path=config.RUN_INFO_PATH,
         results_dir=config.RESULTS_DIR, status_path=config.STATUS_PATH, monitor=True, out=None,
         status_dir=config.STATUS_DIR, logs_dir=config.TRAIN_LOGS_DIR):
    parser = argparse.ArgumentParser(description="Run the EC Racing experiment notebooks in order. An interrupted "
                                                 "run continues where it stopped.")
    parser.add_argument("--from", dest="start", help="number of the first notebook to run, for example 04")
    parser.add_argument("--fresh", action="store_true", help="remove the earlier results and start over")
    parser.add_argument("--runs", help="run these jobs side by side instead of the notebooks in order: runs of any "
                                       "method, the Keras Tuner searches search:<method> and notebook 01, for example "
                                       "nb:01,search:kt_random,baseline:0,ga:1")
    parser.add_argument("--at-once", type=int, default=1, help="with --runs: how many run at the same time")
    parser.add_argument("--archive", metavar="NAME", help="move the earlier results to results/archive/NAME/ and exit")
    parser.add_argument("--check", action="store_true",
                        help="only check whether the earlier results can be continued (exit code 0, or 3)")
    parser.add_argument("--notebook-job", help=argparse.SUPPRESS)        # one notebook, for --runs
    args = parser.parse_args(argv)
    if args.notebook_job is not None:
        return notebook_job(args.notebook_job, notebook_dir, log_path)
    if args.check:                                     # read-only, so no lock (scripts/fleet.py start asks this)
        return 3 if report_results(results_dir) else 0
    if args.archive is not None and (args.fresh or args.start is not None or args.runs is not None):
        print("error: --archive moves the earlier results and runs nothing; use it on its own", file=sys.stderr)
        return 2
    if args.runs is not None:
        try:
            args.runs = jobs.parse_jobs(args.runs)
        except ValueError as error:
            print(f"error: --runs {error}", file=sys.stderr)
            return 2
        if args.start is not None or args.fresh:
            print("error: --runs cannot be combined with --from or --fresh", file=sys.stderr)
            return 2
        if args.at_once < 1:
            print("error: --at-once must be at least 1", file=sys.stderr)
            return 2
        unmet = jobs.unmet(args.runs, results_dir)
        for job, path in unmet:
            print(f"error: {job.label} needs {path}, which neither exists nor is made by a listed job",
                  file=sys.stderr)
        if unmet:
            return 2
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
    previous = signal.signal(signal.SIGTERM, stop_on_kill)
    try:
        return locked_main(args, notebook_dir, log_path, info_path, results_dir, status_path, monitor, out,
                           status_dir, logs_dir)
    except KeyboardInterrupt:                          # stopped before or after a notebook ran
        return STOPPED
    finally:
        signal.signal(signal.SIGTERM, previous)


def locked_main(args, notebook_dir, log_path, info_path, results_dir, status_path, monitor, out,
                status_dir=config.STATUS_DIR, logs_dir=config.TRAIN_LOGS_DIR):
    """main() once the options are valid: takes the results lock, checks earlier results and runs."""
    lock = lock_results(results_dir)
    if lock is None:
        holder = lock_holder(results_dir)
        who = f"another runner (process {holder})" if holder else "another runner"
        print(f"error: {who} is using {results_dir}; wait for it to finish or stop it, then run again",
              file=sys.stderr)
        return 4
    try:
        if getattr(args, "archive", None) is not None:
            try:
                target = archive_results(results_dir, args.archive)
            except FileExistsError as error:
                print(f"error: {error}", file=sys.stderr)
                return 2
            print(f"The earlier results are in {target}; the next start begins afresh.")
            return 0
        if args.fresh:
            fresh_start(results_dir)
        if report_results(results_dir):
            return 3
        job_list = getattr(args, "runs", None)
        if job_list is not None:
            return run_side_by_side(job_list, args.at_once, log_path, info_path, status_dir, logs_dir, monitor, out,
                                    results_dir=results_dir)
        return run(notebook_dir, args.start, log_path, info_path, status_path, monitor, out)
    finally:
        lock.close()                                   # releases the lock


if __name__ == "__main__":
    sys.exit(main())
