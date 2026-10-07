"""Runs the experiment notebooks in order and saves their outputs into the notebook files (spec §3.3).

Run with:      ~/miniconda3/bin/python scripts/run_notebooks.py
Resume:        ~/miniconda3/bin/python scripts/run_notebooks.py --from 03
In background: nohup ~/miniconda3/bin/python scripts/run_notebooks.py > /dev/null 2>&1 &
Progress:      tail -f results/run_log.txt
"""
import argparse
import json
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

from racing import config  # noqa: E402

NOTEBOOK_DIR = PROJECT / "notebooks"
KERNEL = "ec-racing"


def now():
    return datetime.now().isoformat(timespec="seconds")


def log(message, path):
    line = f"{now()}  {message}"
    print(line, flush=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line + "\n")


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


def write_info(info, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(info, indent=2))


def run(notebook_dir=NOTEBOOK_DIR, start=None, log_path=config.RUN_LOG_PATH, info_path=config.RUN_INFO_PATH):
    """Execute the notebooks in order. A resumed run (start given) keeps the earlier notebooks' record entries."""
    from ipykernel.kernelspec import write_kernel_spec

    notebook_dir = Path(notebook_dir)
    info = {}
    if start is not None and Path(info_path).exists():
        info = json.loads(Path(info_path).read_text())
    info.update({"processor": processor(), "logical_processors": os.cpu_count(), "n_jobs": config.N_JOBS,
                 "power_profile": power_profile(), "versions": versions()})
    info.setdefault("notebooks", [])
    info.setdefault("runs", []).append({"started": now(), "from": start})
    names = notebook_names(notebook_dir, start)
    log(f"run started: {' '.join(names)} (power profile {info['power_profile']}, {config.N_JOBS} workers)", log_path)
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
            except Exception as error:                 # a failing cell; its output is saved in the notebook
                entry["failed"] = str(error).strip().splitlines()[-1] if str(error).strip() else type(error).__name__
                entry["seconds"] = round(time.time() - began, 1)
                write_info(info, info_path)
                log(f"FAILED {name}: {entry['failed']}", log_path)
                return 1
            entry["finished"] = now()
            entry["seconds"] = round(time.time() - began, 1)
            write_info(info, info_path)
            log(f"done {name} in {entry['seconds'] / 60:.1f} min", log_path)
    log("run finished", log_path)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the EC Racing experiment notebooks in order.")
    parser.add_argument("--from", dest="start", help="number of the first notebook to run, for example 03")
    args = parser.parse_args(argv)
    return run(start=args.start)


if __name__ == "__main__":
    sys.exit(main())
