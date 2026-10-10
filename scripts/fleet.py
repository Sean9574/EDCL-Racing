"""Trains the experiment's evolutionary runs on several machines at once (README, "Several machines").

Run it on the desktop, in its Ubuntu shell with the ec-racing environment active. The servers are reached from there
with the ssh command of scripts/fleet.toml (Windows' ssh.exe, which goes through the school VPN):

  python scripts/fleet.py setup     a server gets this commit's code, Python 3.13 with the pinned packages, and the
                                    refined baselines its seeded ES runs start from; again after every code change
  python scripts/fleet.py start     every machine starts its runs in a detached tmux session named "fleet"
  python scripts/fleet.py status    one line per run in training on every machine, and the latest milestones
  python scripts/fleet.py stop      Ctrl+C in every session: the runs stop and continue where they stopped at the
                                    next start
  python scripts/fleet.py collect   the servers' finished runs are copied into this machine's results
  python scripts/fleet.py finish    once every run has finished: collect, then the notebooks from 04 and the
                                    evaluation, in the "fleet" session here

Which machine trains which runs: scripts/fleet.toml. Watch a session with  tmux attach -t fleet  (detach with
Ctrl+B, then D); on a server, log in first with  ssh.exe <host>.

Exit codes: 0 done; 1 a check failed or nothing could be done (the message says why); 2 invalid plan or options.
"""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
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

from racing import checkpoint, config, progress, train  # noqa: E402

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
            runs = [train.parse_run(run) for run in entry.get("runs", [])]
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
    return f"{run[0]}:{run[1]}"


# ----- shell on a machine -----

def run_shell(machine, script, stream=False, timeout=None):
    """Run a bash script on `machine` and return (exit code, output, errors) as text; with `stream` the output goes
    to this terminal instead. A server is reached with its ssh command, and the script goes over standard input, so
    nothing in it is quoted for the Windows client."""
    if machine.local:
        command = ["bash", "-s"]
    else:
        command = [machine.ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", machine.host, "bash -s"]
    pipe = None if stream else subprocess.PIPE
    result = subprocess.run(command, input=script.encode(), stdout=pipe, stderr=pipe, timeout=timeout)
    text = lambda data: (data or b"").decode("utf-8", errors="replace").replace("\r\n", "\n")
    return result.returncode, text(result.stdout), text(result.stderr)


def shell(machine, script, stream=False, timeout=None):
    """run_shell, raising FleetError when the script fails or the machine cannot be reached."""
    code, out, err = run_shell(machine, script, stream=stream, timeout=timeout)
    if code != 0:
        reason = (err.strip() or out.strip()).splitlines()[-1:] or [f"exit code {code}"]
        hint = f" {CONNECT_HINT}" if code == 255 and not machine.local else ""
        raise FleetError(f"{machine.where}: {reason[0]}.{hint}")
    return out


def upload_script(machine, data, target):
    """A script that writes `data` to `target` (relative to the project folder) and checks its SHA-256. The data
    travels as base64 inside the script, so no binary stream passes the Windows client."""
    digest = hashlib.sha256(data).hexdigest()
    quoted = shlex.quote(target)
    return (f"set -e\nmkdir -p {machine.root}\ncd {machine.root}\nmkdir -p \"$(dirname {quoted})\"\n"
            f"base64 -d > {quoted}.part <<'EC_RACING_DATA'\n{base64.encodebytes(data).decode()}EC_RACING_DATA\n"
            f"test \"$(sha256sum {quoted}.part | cut -d' ' -f1)\" = {digest}\n"
            f"mv {quoted}.part {quoted}\necho uploaded\n")


def upload(machine, data, target):
    if "uploaded" not in shell(machine, upload_script(machine, data, target)):
        raise FleetError(f"{machine.where}: {target} was not written")


def download_script(machine, members):
    """A script that prints the SHA-256 of a tar archive of `members` (paths or globs under the project folder, made
    by this module from approach names and seeds; those that match nothing are left out), then the archive in
    base64."""
    return (f"set -e\ncd {machine.root}\nshopt -s nullglob\nfiles=()\n"
            f"for f in {' '.join(members)}; do [ -e \"$f\" ] && files+=(\"$f\"); done\n"
            "archive=$(mktemp)\n"
            "if [ ${#files[@]} -gt 0 ]; then tar -cf \"$archive\" \"${files[@]}\"; "
            "else tar -cf \"$archive\" -T /dev/null; fi\n"
            "echo \"SHA256 $(sha256sum \"$archive\" | cut -d' ' -f1)\"\n"
            "base64 \"$archive\"\nrm -f \"$archive\"\n")


def unpack_download(text):
    """The archive bytes of download_script's output, after checking its SHA-256."""
    first, _, rest = text.partition("\n")
    if not first.startswith("SHA256 "):
        raise FleetError("the download did not start with its checksum")
    data = base64.b64decode("".join(rest.split()))
    if hashlib.sha256(data).hexdigest() != first.split()[1]:
        raise FleetError("the download arrived damaged (checksum mismatch); try again")
    return data


def download(machine, members):
    return unpack_download(shell(machine, download_script(machine, members)))


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

def code_script(machine):
    """Check out the uploaded commit in the server's project folder. Tracked files there are replaced; its results
    are not tracked and stay."""
    return f"""set -e
cd {machine.root}
git rev-parse --git-dir >/dev/null 2>&1 || git init -q
git fetch -q {BUNDLE} HEAD
git -c advice.detachedHead=false checkout -q --force --detach FETCH_HEAD
echo "code: commit $(git rev-parse HEAD)"
"""


def environment_script(machine):
    """Install Miniforge and the environment if missing, then the pinned packages, and check the imports."""
    return f"""set -e
cd {machine.root}
if [ ! -x "$HOME"/miniforge3/bin/conda ]; then
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


def baseline_inputs(machine):
    """(local file, target) of the refined baselines the machine's seeded ES runs start from (notebook 03)."""
    inputs = []
    for approach, seed in machine.runs:
        if approach == "es_seeded":
            relative = f"results/runs/baseline/seed{seed}/dagger.keras"
            inputs.append((PROJECT / relative, relative))
    return inputs


def send_code(machine, out=sys.stdout):
    """The server's project folder gets this machine's last commit, as a git bundle."""
    commit = local_commit()
    with tempfile.TemporaryDirectory() as folder:
        bundle = Path(folder) / "code.bundle"
        git("bundle", "create", "-q", str(bundle), "HEAD")
        print(f"sending the code (commit {commit[:7]}, {bundle.stat().st_size / 1e6:.1f} MB) ...", file=out,
              flush=True)
        upload(machine, bundle.read_bytes(), BUNDLE)
    if f"commit {commit}" not in shell(machine, code_script(machine)):
        raise FleetError(f"{machine.where}: the code was not checked out")


def setup(machines, out=sys.stdout):
    require_committed_code()
    for machine in machines:
        if machine.local:
            continue
        print(f"== {machine.where}", file=out, flush=True)
        shell(machine, "echo ok", timeout=120)
        send_code(machine, out)
        shell(machine, environment_script(machine), stream=True)
        for local, target in baseline_inputs(machine):
            if not local.exists():
                raise FleetError(f"{local} is missing here; notebook 03 makes it")
            print(f"sending {target} ...", file=out, flush=True)
            upload(machine, local.read_bytes(), target)
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
             f"[ -x {machine.python} ] || echo 'python missing'"]
    for local, target in baseline_inputs(machine):
        lines.append(f"[ -f {shlex.quote(target)} ] || echo 'missing {target}'")
    return "\n".join(lines) + "\n"


def probe(machine):
    facts = {"missing": []}
    for line in shell(machine, probe_script(machine), timeout=120).splitlines():
        key, _, value = line.partition(" ")
        if key == "missing":
            facts["missing"].append(value)
        elif key in ("commit", "lock", "session", "python", "project"):
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
        time.sleep(3)
        for machine in list(asked):
            if parse_status(shell(machine, status_script(machine), timeout=120))[0] is False:
                print(f"{machine.where}: stopped; the runs continue where they stopped at the next start", file=out)
                asked.remove(machine)
    for machine in asked:
        print(f"{machine.where}: still ending after {wait:.0f} s; check with: python scripts/fleet.py status",
              file=out)
    return 1 if asked else 0


# ----- collect and finish -----

def finished_script(machine):
    checks = [f"[ -f results/runs/{a}/seed{s}/summary.json ] && echo 'done {a} {s}'" for a, s in machine.runs]
    return f"cd {machine.root} 2>/dev/null || exit 0\n" + "\n".join(checks) + "\ntrue\n"


def run_members(run):
    approach, seed = run
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
    record = incoming / "summary.json"
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
    if (target / "summary.json").exists():
        if same_file(target / "summary.json", record):
            return "already here"
        raise FleetError(f"{name}: a different finished run is already in {target}; nothing was copied")
    if target.exists():
        raise FleetError(f"{name}: an unfinished copy of this run is in {target}; delete that folder if the run "
                         f"from {source} should replace it")
    for champion in champions:
        existing = results / "champions" / approach / champion.name
        if existing.exists() and not same_file(existing, champion):
            raise FleetError(f"{name}: {existing} belongs to a different run; nothing was copied")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(incoming), str(target))
    for champion in champions:
        destination = results / "champions" / approach / champion.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(champion), str(destination))
    return "copied"


def collect(machines, out=sys.stdout, results=None):
    """Copy every server's finished runs into this machine's results, and its run log and run record into
    results/machines/<name>/. Unfinished runs stay where they are. Returns the number of problems."""
    results = Path(config.RESULTS_DIR if results is None else results)
    problems = 0
    for machine in machines:
        if machine.local or not machine.runs:
            continue
        try:
            done = [(line.split()[1], int(line.split()[2])) for line in shell(machine, finished_script(machine))
                    .splitlines() if line.startswith("done ")]
            members = [member for run in done for member in run_members(run)]
            archive = download(machine, members + ["results/run_log.txt", "results/run_info.json"])
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
            waiting = [train_label(run) for run in machine.runs if run not in done]
            if waiting:
                print(f"{machine.where}: not finished yet: {', '.join(waiting)}", file=out)
        finally:
            shutil.rmtree(results / INCOMING / machine.name, ignore_errors=True)
    return problems


def unfinished(results=None):
    """The experiment's evolutionary runs without a summary here."""
    results = Path(config.RESULTS_DIR if results is None else results)
    return [(approach, seed) for approach in config.EVOLUTION_APPROACHES for seed in config.RUN_SEEDS
            if not (results / "runs" / approach / f"seed{seed}" / "summary.json").exists()]


def finish(machines, out=sys.stdout):
    """Collect, then, once every evolutionary run is here, run the notebooks from 04 (they find the runs complete
    and report them) and the evaluation in this machine's fleet session."""
    if collect(machines, out):
        raise FleetError("some runs could not be collected; see above")
    missing = unfinished()
    if missing:
        raise FleetError(f"not every run has finished: {', '.join(map(train_label, missing))}")
    here = next(machine for machine in machines if machine.local)
    readiness_problems = readiness(here, probe(here), local_commit())
    if readiness_problems:
        raise FleetError("\n".join(readiness_problems))
    command = f"{here.python} scripts/run_notebooks.py --from 04"
    if "started" not in shell(here, session_script(here, command)):
        raise FleetError(f"{here.where}: the tmux session did not start")
    print(f"{here.where}: notebooks 04 to 08 run in tmux session '{SESSION}' (tmux attach -t {SESSION})", file=out)
    return 0


COMMANDS = {"setup": setup, "start": start, "status": status, "stop": stop, "finish": finish}


def main(argv=None, plan=PLAN, out=sys.stdout):
    parser = argparse.ArgumentParser(description="Train the experiment's runs on several machines at once.")
    parser.add_argument("command", choices=[*COMMANDS, "collect"])
    args = parser.parse_args(argv)
    try:
        machines = load_plan(plan)
    except PlanError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        if args.command == "collect":
            return 1 if collect(machines, out) else 0
        return COMMANDS[args.command](machines, out=out)
    except FleetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
