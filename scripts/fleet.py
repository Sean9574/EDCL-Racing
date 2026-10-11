"""Trains the experiment's evolutionary runs on several machines at once (README, "Several machines").

Run it on the desktop, in its Ubuntu shell with the ec-racing environment active. The servers are reached from there
with the ssh command of scripts/fleet.toml (Windows' ssh.exe, which goes through the school VPN):

  python scripts/fleet.py setup     a server gets this commit's code, Python 3.13 with the pinned packages, and the
                                    inputs its jobs need that it does not make itself; again after every code change
  python scripts/fleet.py start     every machine starts its runs in a detached tmux session named "fleet"
  python scripts/fleet.py status    one line per run in training on every machine, and the latest milestones
  python scripts/fleet.py stop      Ctrl+C in every session: the runs stop and continue where they stopped at the
                                    next start
  python scripts/fleet.py collect   the servers' finished runs are copied into this machine's results
  python scripts/fleet.py archive NAME   every machine moves its earlier results to results/archive/NAME/
  python scripts/fleet.py finish    once all 24 runs have finished: collect, then notebooks 03 to 08 (they find
                                    the runs complete and report them) and the evaluation, in the "fleet" session here

Which machine trains which runs: scripts/fleet.toml. Watch a session with  tmux attach -t fleet  (detach with
Ctrl+B, then D); on a server, log in first with  ssh.exe <host>.

Exit codes: 0 done; 1 a check failed or nothing could be done (the message says why); 2 invalid plan or options.
"""
import argparse
import base64
import collections
import contextlib
import hashlib
import importlib.util
import io
import json
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

if sys.version_info < (3, 11):
    sys.exit("error: run fleet.py with the ec-racing environment's Python (conda activate ec-racing)")

import tomllib  # noqa: E402

from racing import checkpoint, config, jobs, paths, progress  # noqa: E402

PLAN = Path(__file__).resolve().with_name("fleet.toml")
SESSION = "fleet"
REMOTE_PYTHON = '"$HOME"/miniforge3/envs/ec-racing/bin/python'
MINIFORGE = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
CODE_PATHS = ("racing", "scripts", "requirements.txt")    # what a server runs; it gets the last commit of them
BUNDLE = ".fleet/code.bundle"                             # under a server's project folder
INCOMING = ".incoming"                                    # under results/: runs being collected
STOP_WAIT = 120.0                                         # s `stop` waits for the runners to end
LOG_LINES = 6                                             # milestones `status` shows per machine
MACHINE_KEYS = {"host", "ssh", "dir", "at_once", "workers", "runs"}
CONNECT_HINT = ("Is GlobalProtect connected on this desktop (tray icon), and is the SSH key installed on the server "
                "(README, \"Several machines\")?")
END_NOTE = ('code=$?; echo; echo "fleet: the runner ended with exit code $code '
            '(0 finished, 1 a run failed, 130 stopped). Ctrl+D closes this window."; exec bash')


class FleetError(RuntimeError):
    """A check failed; the message says what to do."""


class PlanError(ValueError):
    """scripts/fleet.toml cannot be used."""


@dataclass
class Machine:
    name: str
    runs: list                 # [(approach, seed)]
    at_once: int = 1
    workers: int = 6
    host: str = None           # None: the machine fleet.py runs on
    ssh: str = "ssh"
    dir: str = "ec-racing"     # a server's project folder, relative to its home folder

    @property
    def local(self):
        return self.host is None

    @property
    def root(self):
        """The project folder, as a bash word."""
        return shlex.quote(str(PROJECT)) if self.local else '"$HOME"/' + shlex.quote(self.dir)

    @property
    def python(self):
        return shlex.quote(sys.executable) if self.local else REMOTE_PYTHON

    @property
    def where(self):
        return self.name if self.local else f"{self.name} ({self.host})"


# ----- plan -----

def positive(entry, key, default, name):
    value = entry.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PlanError(f"machine {name}: {key} must be a whole number of at least 1, not {value!r}")
    return value


def load_plan(path=PLAN):
    """The machines of the plan file, each with its runs. Each run belongs to one machine, and exactly one machine
    has no host: the one fleet.py runs on, which collects the results."""
    try:
        data = tomllib.loads(Path(path).read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PlanError(f"{path} cannot be read: {error}") from error
    machines, owner = [], {}
    for name, entry in (data.get("machines") or {}).items():
        if not isinstance(entry, dict):
            raise PlanError(f"machine {name}: expected a [machines.{name}] table")
        unknown = sorted(set(entry) - MACHINE_KEYS)
        if unknown:
            raise PlanError(f"machine {name}: unknown keys {', '.join(unknown)}")
        try:
            runs = [jobs.parse_job(run) for run in entry.get("runs", [])]
        except ValueError as error:
            raise PlanError(f"machine {name}: {error}") from error
        for run in runs:
            if run in owner:
                raise PlanError(f"run {train_label(run)} is on both {owner[run]} and {name}")
            owner[run] = name
        machines.append(Machine(name=name, runs=runs, at_once=positive(entry, "at_once", 1, name),
                                workers=positive(entry, "workers", 6, name), host=entry.get("host"),
                                ssh=entry.get("ssh", "ssh"), dir=entry.get("dir", "ec-racing")))
    if sum(machine.local for machine in machines) != 1:
        raise PlanError("exactly one machine must have no host: the one fleet.py runs on")
    return machines


def train_label(run):
    """A job's label, or approach:seed of a run given as a tuple."""
    return run.label if isinstance(run, jobs.Job) else f"{run[0]}:{run[1]}"


# ----- shell on a machine -----
#
# A server is reached with its ssh command, and files travel with the matching scp. Windows' clients (ssh.exe, which
# reaches the school network through GlobalProtect) are started through a batch file, with Windows' own redirection
# from NUL and to files: started from WSL with their input or output passed through WSL, ssh.exe finishes the
# command on the server but never exits, and it stalls on more than a few kilobytes of input even from a file. So no
# script or data ever passes ssh's standard input: a script travels in the command line, base64-encoded, and files go
# with scp.

SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
MAX_COMMAND = 6000                      # characters of an encoded script; a line of a Windows batch file holds 8191
WINDOWS_WORK = Path("/mnt/c/Users/Public/ec-racing-fleet")   # batch files and copies in transit; no spaces in it
OUTGOING = ".fleet/outgoing.tar"        # under a server's project folder: the archive `download` copies here
COMMAND_TIMEOUT = 300.0                 # s for a script on a server
COPY_TIMEOUT = 1800.0                   # s for copying one file
CONNECTION_WINDOW = 30.0                # s; a precaution against servers that block an address connecting more
CONNECTIONS_PER_WINDOW = 4              # often than this many times per window
recent_connections = collections.defaultdict(collections.deque)    # host -> times of the latest connections


def remote_command(script):
    """`script` as one command line for the server's shell."""
    encoded = base64.b64encode(script.encode()).decode()
    if len(encoded) > MAX_COMMAND:
        raise FleetError(f"a script for a server is too long ({len(encoded)} characters encoded)")
    return f"echo {encoded} | base64 -d | bash"


def windows_client(machine):
    return machine.ssh.lower().endswith(".exe")


def scp_program(machine):
    """The scp next to the machine's ssh: scp.exe for ssh.exe."""
    folder, slash, name = machine.ssh.rpartition("/")
    return folder + slash + name.replace("ssh", "scp", 1)


def text_of(data):
    return (data or b"").decode("utf-8", errors="replace").replace("\r\n", "\n")


def windows_path(path):
    """/mnt/c/Users/... as Windows writes it: C:\\Users\\..."""
    parts = Path(path).parts
    if len(parts) < 3 or parts[:2] != ("/", "mnt") or len(parts[2]) != 1:
        raise FleetError(f"{path} is not on a Windows drive")
    return parts[2].upper() + ":\\" + "\\".join(parts[3:])


def batch_text(arguments, folder):
    """A batch file that runs `arguments` with nothing on their input and their output in out.txt and err.txt of
    the Windows folder `folder`, and exits with their exit code."""
    words = " ".join(f'"{word}"' if any(c in word for c in ' |&<>^()') else word for word in arguments)
    return f'@{words} < NUL > "{folder}\\out.txt" 2> "{folder}\\err.txt"\r\n@exit /b %ERRORLEVEL%\r\n'


@contextlib.contextmanager
def windows_folder():
    """A fresh folder under WINDOWS_WORK, removed afterwards."""
    WINDOWS_WORK.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="call-", dir=WINDOWS_WORK))
    try:
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def run_windows(arguments, folder, stream=False, timeout=None):
    """Run a Windows client's command line through a batch file in `folder`; (exit code, output, errors). With
    `stream`, the output is printed while it grows. Raises subprocess.TimeoutExpired after `timeout` seconds."""
    win = windows_path(folder)
    (folder / "run.bat").write_bytes(batch_text(arguments, win).encode())
    process = subprocess.Popen(["cmd.exe", "/c", f"{win}\\run.bat"], cwd="/mnt/c", stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = None if timeout is None else time.monotonic() + timeout
    shown = 0
    while True:
        code = process.poll()
        if stream:
            shown = show_new(folder / "out.txt", shown)
        if code is not None:
            break
        if deadline is not None and time.monotonic() > deadline:
            process.kill()
            raise subprocess.TimeoutExpired(arguments, timeout)
        time.sleep(0.2)
    read = lambda name: text_of((folder / name).read_bytes()) if (folder / name).exists() else ""
    out, err = read("out.txt"), read("err.txt")
    if stream and err:
        print(err, end="", file=sys.stderr, flush=True)
    return code, out, err


def show_new(path, shown):
    """Print what was added to `path` since `shown` bytes; returns the new count."""
    try:
        data = path.read_bytes()
    except OSError:
        return shown
    if len(data) > shown:
        sys.stdout.write(text_of(data[shown:]))
        sys.stdout.flush()
    return len(data)


def pace(machine, out=sys.stdout):
    """Wait until one more connection to the machine's server keeps within CONNECTIONS_PER_WINDOW."""
    times = recent_connections[machine.host]
    while True:
        now = time.monotonic()
        while times and now - times[0] > CONNECTION_WINDOW:
            times.popleft()
        if len(times) < CONNECTIONS_PER_WINDOW:
            times.append(now)
            return
        wait = CONNECTION_WINDOW - (now - times[0]) + 0.5
        if wait > 3:
            print(f"(waiting {wait:.0f} s before the next connection to the server)", file=out, flush=True)
        time.sleep(wait)


def run_client(machine, arguments, stream=False, timeout=None, folder=None):
    """Run an ssh or scp command line of `machine` with nothing on its input; (exit code, output, errors). A Windows
    client runs through a batch file in `folder`, or in a fresh folder under WINDOWS_WORK. Connections are paced."""
    pace(machine)
    try:
        if windows_client(machine):
            if folder is not None:
                return run_windows(arguments, folder, stream, timeout)
            with windows_folder() as fresh:
                return run_windows(arguments, fresh, stream, timeout)
        if stream:
            return run_streamed(arguments, timeout)
        result = subprocess.run(arguments, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
        return result.returncode, text_of(result.stdout), text_of(result.stderr)
    except subprocess.TimeoutExpired:
        raise FleetError(f"{machine.where}: no answer within {timeout:.0f} s. {CONNECT_HINT}") from None


def run_streamed(arguments, timeout=None):
    """Run `arguments`, printing their output as it arrives and keeping it; errors are part of the output."""
    process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    kept = []
    for line in process.stdout:
        sys.stdout.write(text_of(line))
        sys.stdout.flush()
        kept.append(line)
    return process.wait(timeout=timeout), text_of(b"".join(kept)), ""


def run_shell(machine, script, stream=False, timeout=None):
    """Run a bash script on `machine` and return (exit code, output, errors) as text; with `stream` the output also
    goes to this terminal as it arrives."""
    if machine.local:
        pipe = None if stream else subprocess.PIPE
        result = subprocess.run(["bash", "-s"], input=script.encode(), stdout=pipe, stderr=pipe, timeout=timeout)
        return result.returncode, text_of(result.stdout), text_of(result.stderr)
    if timeout is None and not stream:
        timeout = COMMAND_TIMEOUT
    return run_client(machine, [machine.ssh, *SSH_OPTIONS, machine.host, remote_command(script)], stream, timeout)


def shell(machine, script, stream=False, timeout=None):
    """run_shell, raising FleetError when the script fails or the machine cannot be reached."""
    code, out, err = run_shell(machine, script, stream=stream, timeout=timeout)
    if code != 0:
        reason = (err.strip() or out.strip()).splitlines()[-1:] or [f"exit code {code}"]
        hint = f" {CONNECT_HINT}" if code == 255 and not machine.local else ""
        raise FleetError(f"{machine.where}: {reason[0]}.{hint}")
    return out


def scp_check(machine, code, err, what):
    if code != 0:
        reason = err.strip().splitlines()[-1:] or [f"exit code {code}"]
        hint = f" {CONNECT_HINT}" if code == 255 else ""
        raise FleetError(f"{machine.where}: copying {what} failed: {reason[0]}.{hint}")


def copy_to(machine, local, target):
    """Copy the file `local` here to `target`, relative to the server's home folder (its folder must exist)."""
    destination = f"{machine.host}:{target}"
    if windows_client(machine):
        with windows_folder() as folder:
            shutil.copyfile(local, folder / "file")
            code, out, err = run_client(machine, [scp_program(machine), "-q", *SSH_OPTIONS,
                                                  windows_path(folder / "file"), destination],
                                        timeout=COPY_TIMEOUT, folder=folder)
    else:
        code, out, err = run_client(machine, [scp_program(machine), "-q", *SSH_OPTIONS, str(local), destination],
                                    timeout=COPY_TIMEOUT)
    scp_check(machine, code, err, target)


def copy_from(machine, source, local):
    """Copy `source`, relative to the server's home folder, to the file `local` here."""
    origin = f"{machine.host}:{source}"
    if windows_client(machine):
        with windows_folder() as folder:
            code, out, err = run_client(machine, [scp_program(machine), "-q", *SSH_OPTIONS, origin,
                                                  windows_path(folder / "file")],
                                        timeout=COPY_TIMEOUT, folder=folder)
            scp_check(machine, code, err, source)
            shutil.copyfile(folder / "file", local)
    else:
        code, out, err = run_client(machine, [scp_program(machine), "-q", *SSH_OPTIONS, origin, str(local)],
                                    timeout=COPY_TIMEOUT)
        scp_check(machine, code, err, source)


def upload(machine, data, target):
    """Write `data` to `target` under the machine's project folder, checking its SHA-256 there. It is copied into
    the home folder first, which always exists, then moved into place: two connections."""
    staging = f".ec-racing-upload-{secrets.token_hex(4)}.part"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "file"
        path.write_bytes(data)
        copy_to(machine, path, staging)
    digest, quoted = hashlib.sha256(data).hexdigest(), shlex.quote(target)
    answer = shell(machine, f"set -e\nmkdir -p {machine.root}\ncd {machine.root}\n"
                            f"mkdir -p \"$(dirname {quoted})\"\n"
                            f"if [ \"$(sha256sum \"$HOME\"/{staging} | cut -d' ' -f1)\" = {digest} ]; then "
                            f"mv \"$HOME\"/{staging} {quoted}; echo uploaded; "
                            f"else rm -f \"$HOME\"/{staging}; echo damaged; fi\n")
    if "uploaded" not in answer:
        raise FleetError(f"{machine.where}: {target} arrived damaged (checksum mismatch); try again")


def pack_lines(members):
    """Script lines that pack `members` (paths or globs under the project folder, made by this module from approach
    names and seeds; those that match nothing are left out) into OUTGOING and print its SHA-256."""
    return (f"shopt -s nullglob\nfiles=()\n"
            f"for f in {' '.join(members)}; do [ -e \"$f\" ] && files+=(\"$f\"); done\n"
            f"mkdir -p \"$(dirname {OUTGOING})\"\nrm -f {OUTGOING}\n"
            f"if [ ${{#files[@]}} -gt 0 ]; then tar -cf {OUTGOING} \"${{files[@]}}\"; "
            f"else tar -cf {OUTGOING} -T /dev/null; fi\n"
            f"echo \"SHA256 $(sha256sum {OUTGOING} | cut -d' ' -f1)\"\n")


def pack_script(machine, members):
    return f"set -e\ncd {machine.root}\n" + pack_lines(members)


def fetch_packed(machine, answer):
    """The archive a pack script made, copied here and checked against the SHA-256 it printed in `answer`."""
    lines = [line for line in answer.splitlines() if line.startswith("SHA256 ")]
    if not lines:
        raise FleetError(f"{machine.where}: the archive was not made")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "archive.tar"
        copy_from(machine, f"{machine.dir}/{OUTGOING}", path)
        data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != lines[-1].split()[1]:
        raise FleetError(f"{machine.where}: the download arrived damaged (checksum mismatch); try again")
    return data


def download(machine, members):
    """A tar archive of `members` under the machine's project folder: two connections. The archive stays on the
    server until the next download replaces it."""
    return fetch_packed(machine, shell(machine, pack_script(machine, members)))


# ----- code version -----

def git(*args):
    result = subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True)
    if result.returncode != 0:
        raise FleetError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def local_commit():
    return git("rev-parse", "HEAD").strip()


def code_changes():
    """Tracked code that differs from the last commit; a server would not get it."""
    return git("status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS).strip()


def require_committed_code():
    changes = code_changes()
    if changes:
        raise FleetError("these code files differ from the last commit, which is what a server gets; commit or "
                         f"undo the changes first:\n{changes}")


# ----- setup -----

SETUP_ARCHIVE = ".fleet/setup.tar"     # under a server's project folder: the code bundle and the baselines


def code_lines():
    """Script lines that unpack the setup archive and check out its commit. Tracked files there are replaced; the
    server's results are not tracked and stay."""
    return (f"tar -xf {SETUP_ARCHIVE}\n"
            "git rev-parse --git-dir >/dev/null 2>&1 || git init -q\n"
            f"git fetch -q {BUNDLE} HEAD\n"
            "git -c advice.detachedHead=false checkout -q --force --detach FETCH_HEAD\n"
            'echo "code: commit $(git rev-parse HEAD)"\n')


def environment_lines():
    """Script lines that install Miniforge and the environment if missing, then the pinned packages, and check the
    imports."""
    return f"""if [ ! -x "$HOME"/miniforge3/bin/conda ]; then
  echo "installing Miniforge (conda with conda-forge) into ~/miniforge3 ..."
  curl -fsSL {MINIFORGE} -o "$HOME"/miniforge-installer.sh
  bash "$HOME"/miniforge-installer.sh -b -p "$HOME"/miniforge3
  rm -f "$HOME"/miniforge-installer.sh
fi
if [ ! -x {REMOTE_PYTHON} ]; then
  echo "creating the ec-racing environment (Python 3.13) ..."
  "$HOME"/miniforge3/bin/conda create -y -q -n ec-racing python=3.13
fi
echo "installing the pinned packages (requirements.txt) ..."
{REMOTE_PYTHON} -m pip install -q --disable-pip-version-check -r requirements.txt
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 {REMOTE_PYTHON} -c "import numpy, tensorflow, racing.train; print('environment ready: Python', __import__('platform').python_version())"
"""


def setup_script(machine):
    return f"set -e\ncd {machine.root}\n" + code_lines() + environment_lines()


def external_inputs(machine):
    """(local file, path relative to the project) of every input the machine's jobs need that none of its own jobs
    make (revision 3.2 §6.3): a server must be given them before its jobs can start."""
    results = PROJECT / "results"
    made = {path for job in machine.runs for path in jobs.makes(job, results)}
    found = []
    for job in machine.runs:
        for path in jobs.needs(job, results):
            item = (path, path.relative_to(PROJECT).as_posix())
            if path not in made and item not in found:
                found.append(item)
    return found


def maker_of(path):
    """Labels of the experiment's jobs that make `path`."""
    every = [*(f"nb:{n}" for n in jobs.NOTEBOOK_JOBS),
             *(f"{a}:{s}" for a in config.APPROACHES for s in config.RUN_SEEDS)]
    return [label for label in every if path in jobs.makes(jobs.parse_job(label), PROJECT / "results")]


def setup_archive(machine, folder):
    """A tar archive of this commit as a git bundle and of the inputs the machine's jobs need from elsewhere,
    written in `folder`; returns its path."""
    bundle = Path(folder) / "code.bundle"
    git("bundle", "create", "-q", str(bundle), "HEAD")
    archive = Path(folder) / "setup.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(bundle, arcname=BUNDLE)
        for local, target in external_inputs(machine):
            if not local.exists():
                raise FleetError(f"{local} is missing here; {' or '.join(maker_of(local)) or 'no job'} makes it")
            tar.add(local, arcname=target)
    return archive


def setup(machines, out=sys.stdout):
    """Give every server this commit's code, its baselines and the environment: three connections each."""
    require_committed_code()
    commit = local_commit()
    for machine in machines:
        if machine.local:
            continue
        print(f"== {machine.where}", file=out, flush=True)
        with tempfile.TemporaryDirectory() as folder:
            archive = setup_archive(machine, folder)
            print(f"sending the code (commit {commit[:7]}) and {len(external_inputs(machine))} input files, "
                  f"{archive.stat().st_size / 1e6:.1f} MB ...", file=out, flush=True)
            upload(machine, archive.read_bytes(), SETUP_ARCHIVE)
        answer = shell(machine, setup_script(machine), stream=True)
        if f"commit {commit}" not in answer:
            raise FleetError(f"{machine.where}: the code was not checked out")
        print(f"{machine.name} is ready.", file=out, flush=True)
    return 0


# ----- start, status, stop -----

def probe_script(machine):
    """Facts `start` checks before anything starts: the commit, the results lock, the session and missing inputs."""
    lines = [f"cd {machine.root} 2>/dev/null || {{ echo 'project missing'; exit 0; }}",
             'echo "commit $(git rev-parse HEAD 2>/dev/null || echo none)"',
             "mkdir -p results",
             'if flock -n results/.runner.lock true; then echo "lock free"; '
             'else echo "lock held $(head -c 20 results/.runner.lock 2>/dev/null)"; fi',
             f"tmux has-session -t {SESSION} 2>/dev/null && echo 'session yes' || echo 'session no'",
             f"[ -x {machine.python} ] || echo 'python missing'",
             f"if [ -x {machine.python} ] && [ -f scripts/run_notebooks.py ]; then "
             f"{machine.python} scripts/run_notebooks.py --check >/dev/null 2>&1 || echo 'results differ'; fi"]
    for local, target in external_inputs(machine):
        lines.append(f"[ -f {shlex.quote(target)} ] || echo 'missing {target}'")
    return "\n".join(lines) + "\n"


def probe(machine):
    facts = {"missing": []}
    for line in shell(machine, probe_script(machine), timeout=120).splitlines():
        key, _, value = line.partition(" ")
        if key == "missing":
            facts["missing"].append(value)
        elif key in ("commit", "lock", "session", "python", "project", "results"):
            facts[key] = value
    return facts


def environment_here():
    """The machine fleet.py runs on trains with fleet.py's own Python, so that must be the ec-racing environment."""
    return importlib.util.find_spec("tensorflow") is not None


def readiness(machine, facts, commit):
    """Why `machine` cannot start its runs now, as messages; none when it can."""
    problems = []
    if machine.local and not environment_here():
        problems.append(f"{machine.where}: this Python ({sys.executable}) has no TensorFlow; activate the "
                        "environment first (conda activate ec-racing)")
    if "project" in facts:
        return [f"{machine.where}: the project is not there; run: python scripts/fleet.py setup"]
    if not machine.local and facts.get("commit") != commit:
        problems.append(f"{machine.where}: has code {facts.get('commit', 'none')[:7]}, this desktop "
                        f"{commit[:7]}; run: python scripts/fleet.py setup")
    if facts.get("lock", "").startswith("held"):
        holder = facts["lock"].split()[1:] or ["?"]
        problems.append(f"{machine.where}: a runner (process {holder[0]}) is still training there; stop it first "
                        "(python scripts/fleet.py stop, or Ctrl+C in its tmux session)")
    if "python" in facts:
        problems.append(f"{machine.where}: no ec-racing Python; run: python scripts/fleet.py setup")
    if "results" in facts:
        problems.append(f"{machine.where}: its earlier results were made with other settings or cannot be read, so "
                        "its runner would stop at once; keep them aside first: python scripts/fleet.py archive NAME")
    for target in facts["missing"]:
        problems.append(f"{machine.where}: {target} is missing; run: python scripts/fleet.py setup")
    return problems


def session_script(machine, command):
    """Start `command` in a new detached tmux session; an earlier fleet session (its runner has ended, the lock
    showed that) is closed first. The window stays open after the runner ends, showing how it ended."""
    inner = f"cd {machine.root} && {command}; {END_NOTE}"
    return (f"set -e\ncd {machine.root}\n"
            f"if tmux has-session -t {SESSION} 2>/dev/null; then tmux kill-session -t {SESSION}; fi\n"
            f"tmux new-session -d -s {SESSION} -x 200 -y 50 {shlex.quote(inner)}\necho started\n")


def train_command(machine):
    runs = ",".join(train_label(run) for run in machine.runs)
    return (f"EC_RACING_JOBS={machine.workers} {machine.python} scripts/run_notebooks.py --runs {runs} "
            f"--at-once {machine.at_once}")


def start(machines, out=sys.stdout):
    """Check every machine first; start the runs only if every machine can, so the fleet starts whole or not."""
    require_committed_code()
    commit = local_commit()
    busy = [machine for machine in machines if machine.runs]
    problems = []
    for machine in busy:
        try:
            problems += readiness(machine, probe(machine), commit)
        except FleetError as error:
            problems.append(str(error))
    if problems:
        raise FleetError("nothing was started:\n  " + "\n  ".join(problems))
    for machine in busy:
        if "started" not in shell(machine, session_script(machine, train_command(machine))):
            raise FleetError(f"{machine.where}: the tmux session did not start")
        print(f"{machine.where}: training {', '.join(map(train_label, machine.runs))}, {machine.at_once} at a time, "
              f"in tmux session '{SESSION}'", file=out)
    print("Follow them with: python scripts/fleet.py status    (or tmux attach -t fleet on a machine)", file=out)
    return 0


def status_script(machine):
    return (f"cd {machine.root} 2>/dev/null || {{ echo '@@project missing'; exit 0; }}\n"
            "mkdir -p results\n"
            "if flock -n results/.runner.lock true; then echo '@@runner no'; else echo '@@runner yes'; fi\n"
            "for f in results/status/*.json; do [ -f \"$f\" ] && echo \"@@status $(cat \"$f\")\"; done\n"
            "echo '@@log'\n"
            f"tail -n {LOG_LINES} results/run_log.txt 2>/dev/null || true\n")


def parse_status(text):
    """(runner running or None if the project is missing, [status], [log line]) from status_script's output."""
    running, statuses, log, in_log = None, [], [], False
    for line in text.splitlines():
        if in_log:
            log.append(line)
        elif line.startswith("@@runner "):
            running = line.split()[1] == "yes"
        elif line.startswith("@@status "):
            try:
                statuses.append(json.loads(line[len("@@status "):]))
            except ValueError:
                pass
        elif line == "@@log":
            in_log = True
    return running, statuses, log


def status(machines, out=sys.stdout):
    for machine in machines:
        print(f"== {machine.where}", file=out)
        try:
            running, statuses, log = parse_status(shell(machine, status_script(machine), timeout=120))
        except FleetError as error:
            print(f"  cannot check: {error}", file=out)
            continue
        if running is None:
            print("  the project is not there yet (python scripts/fleet.py setup)", file=out)
            continue
        print(f"  runner: {'training' if running else 'not running'}", file=out)
        order = lambda s: progress.run_number(s.get("approach"), s.get("seed"))[0]
        for item in sorted(statuses, key=order):
            print("  " + progress.format_status(item) + progress.stale_note(item), file=out)
        for line in log:
            print("  " + line, file=out)
    return 0


def stop(machines, out=sys.stdout, wait=STOP_WAIT):
    """Ctrl+C in every machine's fleet session, then wait until their runners have ended."""
    script = (f"if tmux has-session -t {SESSION} 2>/dev/null; then tmux send-keys -t {SESSION} C-c; "
              "echo sent; else echo none; fi\n")
    asked = []
    for machine in machines:
        try:
            answer = shell(machine, script, timeout=120)
        except FleetError as error:
            print(f"{machine.where}: {error}", file=out)
            continue
        print(f"{machine.where}: {'stopping' if 'sent' in answer else f'no {SESSION} session'}", file=out)
        if "sent" in answer:
            asked.append(machine)
    deadline = time.time() + wait
    while asked and time.time() < deadline:
        time.sleep(10)
        for machine in list(asked):
            if parse_status(shell(machine, status_script(machine), timeout=120))[0] is False:
                print(f"{machine.where}: stopped; the runs continue where they stopped at the next start", file=out)
                asked.remove(machine)
    for machine in asked:
        print(f"{machine.where}: still ending after {wait:.0f} s; check with: python scripts/fleet.py status",
              file=out)
    return 1 if asked else 0


# ----- collect and finish -----

def collect_script(machine):
    """Name the machine's finished runs and pack them, with the run log and record, into OUTGOING."""
    lines = [f"cd {machine.root} 2>/dev/null || exit 0", "done=()"]
    for job in machine.runs:
        if job.is_notebook:
            continue
        final = paths.final_record(job.approach, job.seed, Path("results") / "runs").as_posix()
        lines.append(f"[ -f {final} ] && "
                     f"{{ echo 'done {job.approach} {job.seed}'; done+=(results/runs/{job.approach}/seed{job.seed} "
                     f"results/champions/{job.approach}/seed{job.seed}.*); }}")
    return "\n".join(lines) + "\n" + pack_lines(['"${done[@]}"', "results/run_log.txt", "results/run_info.json"])


def run_members(run):
    approach, seed = (run.approach, run.seed) if isinstance(run, jobs.Job) else run
    return [f"results/runs/{approach}/seed{seed}", f"results/champions/{approach}/seed{seed}.*"]


def same_file(a, b):
    return a.exists() and b.exists() and a.read_bytes() == b.read_bytes()


def place_run(staging, results, run, source):
    """Move one collected run from `staging` into `results`: its folder and its final controller's files. Returns
    'copied' or 'already here'; raises FleetError, leaving `results` unchanged, if the run is not complete, was made
    with other settings, or a different copy of it is already here."""
    approach, seed = run
    name = train_label(run)
    incoming = staging / "runs" / approach / f"seed{seed}"
    record = paths.final_record(approach, seed, staging / "runs")
    try:
        summary = checkpoint.read_record(record)
    except (OSError, checkpoint.UnreadableRecord) as error:
        raise FleetError(f"{name} from {source}: its summary cannot be read ({error})") from error
    try:
        checkpoint.check_settings(summary.get("settings") if isinstance(summary.get("settings"), dict) else {})
    except checkpoint.SettingsMismatch as error:
        raise FleetError(f"{name} from {source}: {error}") from error
    target = results / "runs" / approach / f"seed{seed}"
    champions = sorted((staging / "champions" / approach).glob(f"seed{seed}.*"))
    here = paths.final_record(approach, seed, results / "runs")
    if here.exists():
        if same_file(here, record):
            return "already here"
        raise FleetError(f"{name}: a different finished run is already in {target}; nothing was copied")
    if target.exists() and not joins(incoming, target, approach):
        raise FleetError(f"{name}: an unfinished copy of this run is in {target}; delete that folder if the run "
                         f"from {source} should replace it")
    for champion in champions:
        existing = results / "champions" / approach / champion.name
        if existing.exists() and not same_file(existing, champion):
            raise FleetError(f"{name}: {existing} belongs to a different run; nothing was copied")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.copytree(incoming, target, dirs_exist_ok=True)                # adds the run to notebook 02's files
    else:
        shutil.move(str(incoming), str(target))
    for champion in champions:
        destination = results / "champions" / approach / champion.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(champion), str(destination))
    return "copied"


def joins(incoming, target, approach):
    """Whether a run collected from a server may be added to the folder `target` here: an imitation method's folder
    holds the files notebook 02 made here (the server got them from here), so it joins when every file in both is
    identical. An evolutionary run's folder here is an unfinished copy of the same run, which is never mixed in."""
    if approach not in config.IMITATION_APPROACHES:
        return False
    for path in incoming.rglob("*"):
        twin = target / path.relative_to(incoming)
        if path.is_file() and twin.exists() and not same_file(path, twin):
            return False
    return True


def collect(machines, out=sys.stdout, results=None):
    """Copy every server's finished runs into this machine's results, and its run log and run record into
    results/machines/<name>/. Unfinished runs stay where they are. Returns the number of problems."""
    results = Path(config.RESULTS_DIR if results is None else results)
    problems = 0
    for machine in machines:
        if machine.local or not machine.runs:
            continue
        try:
            answer = shell(machine, collect_script(machine))
            if "SHA256 " not in answer:
                print(f"{machine.where}: the project is not there yet", file=out)
                continue
            done = [(line.split()[1], int(line.split()[2])) for line in answer.splitlines()
                    if line.startswith("done ")]
            archive = fetch_packed(machine, answer)
        except FleetError as error:
            print(f"{machine.where}: {error}", file=out)
            problems += 1
            continue
        staging = results / INCOMING / machine.name
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(staging, filter="data")
            staging = staging / "results"
            record = results / "machines" / machine.name
            record.mkdir(parents=True, exist_ok=True)
            for name in ("run_log.txt", "run_info.json"):
                if (staging / name).exists():
                    shutil.copyfile(staging / name, record / name)
            for run in done:
                try:
                    print(f"{machine.where}: {train_label(run)} {place_run(staging, results, run, machine.name)}",
                          file=out)
                except FleetError as error:
                    print(f"{machine.where}: {error}", file=out)
                    problems += 1
            waiting = [job.label for job in machine.runs if not job.is_notebook and (job.approach, job.seed) not in done]
            if waiting:
                print(f"{machine.where}: not finished yet: {', '.join(waiting)}", file=out)
        finally:
            shutil.rmtree(results / INCOMING / machine.name, ignore_errors=True)
    return problems


def unfinished(results=None):
    """The experiment's runs (all 24) without their final record here."""
    results = Path(config.RESULTS_DIR if results is None else results)
    return [(approach, seed) for approach in config.APPROACHES for seed in config.RUN_SEEDS
            if not paths.final_record(approach, seed, results / "runs").exists()]


def finish(machines, out=sys.stdout):
    """Collect, then, once all 24 runs are here, run notebooks 03 to 08 (they find the runs complete and report them,
    revision 3.2 §6.3) and the evaluation in this machine's fleet session."""
    if collect(machines, out):
        raise FleetError("some runs could not be collected; see above")
    missing = unfinished()
    if missing:
        raise FleetError(f"not every run has finished: {', '.join(map(train_label, missing))}")
    here = next(machine for machine in machines if machine.local)
    readiness_problems = readiness(here, probe(here), local_commit())
    if readiness_problems:
        raise FleetError("\n".join(readiness_problems))
    command = f"{here.python} scripts/run_notebooks.py --from 03"
    if "started" not in shell(here, session_script(here, command)):
        raise FleetError(f"{here.where}: the tmux session did not start")
    print(f"{here.where}: notebooks 03 to 08 run in tmux session '{SESSION}' (tmux attach -t {SESSION})", file=out)
    return 0


def archive_script(machine, name):
    return (f"cd {machine.root} 2>/dev/null || {{ echo 'no project here, nothing to archive'; exit 0; }}\n"
            f"{machine.python} scripts/run_notebooks.py --archive {shlex.quote(name)}\n")


def archive(machines, name, out=sys.stdout):
    """Move every machine's earlier results to results/archive/NAME/ (run_notebooks.py --archive), so the next start
    begins afresh everywhere (revision 3.2 §6.1)."""
    problems = 0
    for machine in machines:
        code, answer, err = run_shell(machine, archive_script(machine, name))
        print(f"{machine.where}: {(answer.strip() or err.strip() or f'exit code {code}').splitlines()[-1]}", file=out)
        problems += code != 0
    return 1 if problems else 0


COMMANDS = {"setup": setup, "start": start, "status": status, "stop": stop, "finish": finish}


def main(argv=None, plan=PLAN, out=sys.stdout):
    parser = argparse.ArgumentParser(description="Train the experiment's runs on several machines at once.")
    parser.add_argument("command", choices=[*COMMANDS, "collect", "archive"])
    parser.add_argument("name", nargs="?", help="with archive: the archive's name, for example revision3.1")
    args = parser.parse_args(argv)
    if (args.command == "archive") != (args.name is not None):
        print("error: archive takes a name (fleet.py archive revision3.1), and only archive does", file=sys.stderr)
        return 2
    try:
        machines = load_plan(plan)
    except PlanError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        if args.command == "collect":
            return 1 if collect(machines, out) else 0
        if args.command == "archive":
            return archive(machines, args.name, out)
        return COMMANDS[args.command](machines, out=out)
    except FleetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
