"""Compiled simulator (spec §5.7): time trials of dense and NEAT populations and the expert's lap, computed by
racing/fastsim.cpp exactly as racing/sim.py computes them. The library is built with g++ the first time it is needed
and again whenever fastsim.cpp is newer; without g++ the numpy simulator runs instead, with identical results."""
import ctypes
import os
import subprocess
import sys
from pathlib import Path

from racing import config

SOURCE = Path(__file__).with_name("fastsim.cpp")
LIBRARY = Path(__file__).with_name("_fastsim.so")
COMPILER = "g++"
# -ffp-contract=off: no fused multiply-add, so every product and sum rounds as it does in numpy (spec §5.7)
FLAGS = ["-O3", "-march=native", "-ffp-contract=off", "-std=c++17", "-shared", "-fPIC", "-pthread"]

_library = None          # the loaded library
_failure = None          # the error of a failed build or load in this process; the numpy simulator runs instead


def build(force=False):
    """Compiles fastsim.cpp into LIBRARY unless the library is newer than the source. The compiler writes a file of
    this process's own, which is then renamed into place, so runs that start at once never load a half-written
    library (spec §5.7)."""
    if not force and LIBRARY.exists() and LIBRARY.stat().st_mtime_ns >= SOURCE.stat().st_mtime_ns:
        return
    temporary = LIBRARY.with_name(f"{LIBRARY.name}.{os.getpid()}.tmp")
    try:
        subprocess.run([COMPILER, *FLAGS, str(SOURCE), "-o", str(temporary)], check=True, capture_output=True,
                       text=True)
        os.replace(temporary, LIBRARY)
    finally:
        temporary.unlink(missing_ok=True)


def _load():
    build()
    try:
        return ctypes.CDLL(str(LIBRARY))
    except OSError:                                  # a damaged or foreign library: build it again
        build(force=True)
        return ctypes.CDLL(str(LIBRARY))


def library():
    """The loaded library, built first when needed. Raises OSError or CalledProcessError if it cannot be built."""
    global _library
    if _library is None:
        lib = _load()
        _declare(lib)
        _library = lib
    return _library


def available():
    """True when the compiled simulator runs: EC_RACING_SIM is not "numpy" and the library builds and loads. A failed
    build prints one warning with the compiler's message, and the numpy simulator runs instead (spec §5.7)."""
    global _failure
    if config.SIMULATOR == "numpy" or _failure is not None:
        return False
    try:
        library()
    except (OSError, subprocess.CalledProcessError) as error:
        _failure = error
        detail = (getattr(error, "stderr", None) or str(error)).strip()
        print(f"EC Racing: the compiled simulator is unavailable ({detail}); using the numpy simulator",
              file=sys.stderr)
        return False
    return True


def description():
    """The simulator in use, for results/run_info.json: "compiled (g++ <version>)" or "numpy"."""
    if not available():
        return "numpy"
    version = subprocess.run([COMPILER, "-dumpfullversion"], capture_output=True, text=True).stdout.strip()
    return f"compiled (g++ {version})"


def _declare(lib):
    """Argument types of the library's entry points (filled in with the structures of Task 2)."""
