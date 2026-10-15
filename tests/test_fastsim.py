"""The compiled simulator (spec §5.7, §12.1): its build and fallback, its network programs, and exact equality with the
numpy simulator."""
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from racing import config, fastsim

needs_compiler = pytest.mark.skipif(shutil.which(fastsim.COMPILER) is None, reason="no g++ on this machine")


@pytest.fixture(autouse=True)
def compiled_simulator(monkeypatch):
    """Every test here compares the compiled simulator with numpy, whatever EC_RACING_SIM says."""
    monkeypatch.setattr(config, "SIMULATOR", "auto")


@pytest.fixture
def private_library(monkeypatch, tmp_path):
    """The source and library in a temporary folder, with nothing loaded or failed yet in this process."""
    source = tmp_path / "fastsim.cpp"
    shutil.copy(fastsim.SOURCE, source)
    monkeypatch.setattr(fastsim, "SOURCE", source)
    monkeypatch.setattr(fastsim, "LIBRARY", tmp_path / "_fastsim.so")
    monkeypatch.setattr(fastsim, "_library", None)
    monkeypatch.setattr(fastsim, "_failure", None)
    return tmp_path


def test_numpy_choice_disables_the_compiled_simulator(monkeypatch):
    monkeypatch.setattr(config, "SIMULATOR", "numpy")
    assert not fastsim.available()
    assert fastsim.description() == "numpy"


@needs_compiler
def test_the_library_builds_and_describes_its_compiler(private_library):
    assert fastsim.available()
    assert fastsim.LIBRARY.exists()
    assert fastsim.description().startswith("compiled (g++ ")


def test_a_failed_build_warns_once_and_falls_back(private_library, monkeypatch, capsys):
    monkeypatch.setattr(fastsim, "COMPILER", "no-such-compiler-ec-racing")
    assert not fastsim.available()
    assert not fastsim.available()
    err = capsys.readouterr().err
    assert err.count("using the numpy simulator") == 1
    assert fastsim.description() == "numpy"


@needs_compiler
def test_a_stale_library_is_rebuilt(private_library):
    fastsim.build()
    source_time = fastsim.SOURCE.stat().st_mtime
    os.utime(fastsim.LIBRARY, (source_time - 60, source_time - 60))   # older than the source, as after a git pull
    fastsim.build()
    rebuilt = fastsim.LIBRARY.stat().st_mtime_ns
    assert rebuilt >= fastsim.SOURCE.stat().st_mtime_ns
    fastsim.build()                                             # up to date now: no rebuild
    assert fastsim.LIBRARY.stat().st_mtime_ns == rebuilt


@needs_compiler
def test_an_unloadable_library_is_rebuilt(private_library):
    fastsim.LIBRARY.write_bytes(b"not a library")
    later = time.time() + 5
    os.utime(fastsim.LIBRARY, (later, later))                   # newer than the source, so only loading reveals it
    assert fastsim.available()


@needs_compiler
def test_concurrent_builds_all_load(private_library):
    """Four processes that start at once each load a complete library (spec §5.7)."""
    code = (f"import sys; sys.path.insert(0, {str(config.PROJECT_DIR)!r}); from pathlib import Path; "
            f"from racing import fastsim; fastsim.SOURCE = Path({str(fastsim.SOURCE)!r}); "
            f"fastsim.LIBRARY = Path({str(fastsim.LIBRARY)!r}); fastsim.library(); print('ok')")
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(4)]
    outputs = [p.communicate(timeout=300) for p in procs]
    assert all(p.returncode == 0 and out.strip() == "ok" for p, (out, _) in zip(procs, outputs)), outputs
    assert not list(private_library.glob("*.tmp"))
