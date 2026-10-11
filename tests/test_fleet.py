"""scripts/fleet.py, with every machine faked: its scripts run in a local bash whose home folder is the machine's own
temporary folder, and tmux is a stub that records what it is asked. Nothing is started for real."""
import fcntl
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from racing import checkpoint, config, jobs, paths, progress

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fleet.py"
TMUX_STUB = """#!/bin/bash
mkdir -p "$TMUX_STATE"
echo "$*" >> "$TMUX_STATE/calls"
case "$1" in
  has-session) [ -f "$TMUX_STATE/session" ];;
  new-session) command="${@: -1}"; printf '%s' "$command" > "$TMUX_STATE/session"
               if [ -n "$TMUX_RUN" ]; then sh -c "$command" </dev/null >/dev/null 2>&1; fi;;
  kill-session) rm -f "$TMUX_STATE/session";;
  send-keys) echo "$*" >> "$TMUX_STATE/keys";;
esac
"""


class Fake:
    """The fleet module with fake machines: `home(name)` is a machine's home folder, `desktop` the project folder
    of the machine fleet.py runs on."""

    def __init__(self, tmp_path, monkeypatch):
        spec = importlib.util.spec_from_file_location("fleet", SCRIPT)
        self.fleet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fleet)
        self.tmp = tmp_path
        self.desktop = tmp_path / "desktop"
        (self.desktop / "results").mkdir(parents=True)
        self.env = {}
        stubs = tmp_path / "bin"
        stubs.mkdir()
        (stubs / "tmux").write_text(TMUX_STUB)
        (stubs / "tmux").chmod(0o755)
        monkeypatch.setattr(self.fleet, "PROJECT", self.desktop)
        monkeypatch.setattr(config, "RESULTS_DIR", self.desktop / "results")
        monkeypatch.setattr(self.fleet, "local_commit", lambda: self.commit)
        monkeypatch.setattr(self.fleet, "code_changes", lambda: "")
        monkeypatch.setattr(self.fleet, "environment_here", lambda: True)
        self.commit = "0" * 40

        def run_shell(machine, script, stream=False, timeout=None):
            home = self.home(machine.name)
            env = {**os.environ, "HOME": str(home), "PATH": f"{stubs}:{os.environ['PATH']}",
                   "TMUX_STATE": str(home / ".tmux"), **self.env}
            result = subprocess.run(["bash", "-s"], input=script.encode(), capture_output=True, env=env, timeout=60)
            return result.returncode, result.stdout.decode(), result.stderr.decode()

        def copy_to(machine, local, target):                  # scp: paths relative to the home folder
            destination = self.home(machine.name) / target
            if not destination.parent.is_dir():
                raise self.fleet.FleetError(f"scp: {destination.parent}: No such file or directory")
            shutil.copyfile(local, destination)

        def copy_from(machine, source, local):
            shutil.copyfile(self.home(machine.name) / source, local)

        monkeypatch.setattr(self.fleet, "run_shell", run_shell)
        monkeypatch.setattr(self.fleet, "copy_to", copy_to)
        monkeypatch.setattr(self.fleet, "copy_from", copy_from)

    def home(self, name):
        home = self.tmp / "homes" / name
        home.mkdir(parents=True, exist_ok=True)
        return home

    def machines(self):
        return [self.fleet.Machine("desktop", jobs.parse_jobs("ga:1,neat:0"), at_once=2, workers=6),
                self.fleet.Machine("server", jobs.parse_jobs("ga:2,es_seeded:0"), at_once=3, workers=6, host="me@server",
                                   ssh="ssh")]

    def server_project(self):
        return self.home("server") / "ec-racing"

    def ready_server(self):
        """A server after setup: a checkout at self.commit, the environment's Python and the refined baseline."""
        project = self.server_project()
        project.mkdir(parents=True, exist_ok=True)
        git = ["git", "-C", str(project), "-c", "user.name=test", "-c", "user.email=test@example.com"]
        if not (project / ".git").exists():
            subprocess.run(git + ["init", "-q"], check=True)
            (project / "README.md").write_text("server copy\n")
            subprocess.run(git + ["add", "README.md"], check=True)
            subprocess.run(git + ["commit", "-q", "-m", "code"], check=True)
        self.commit = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        python = self.home("server") / "miniforge3" / "envs" / "ec-racing" / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("#!/bin/sh\n")
        python.chmod(0o755)
        refined = project / "results" / "runs" / "baseline" / "seed0"
        refined.mkdir(parents=True, exist_ok=True)
        (refined / "dagger.keras").write_bytes(b"refined baseline")
        checkpoint.write_json(refined / "dagger.json", {"settings": checkpoint.settings_record()})

    def session(self, name):
        """The command the machine's fleet session was started with, or None."""
        path = self.home(name) / ".tmux" / "session"
        return path.read_text() if path.exists() else None


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return Fake(tmp_path, monkeypatch)


# ----- plan -----

def write_plan(tmp_path, text):
    path = tmp_path / "fleet.toml"
    path.write_text(text)
    return path


def test_the_plan_names_each_machines_runs(fake):
    plan = write_plan(fake.tmp, '[machines.here]\nat_once = 2\nruns = ["ga:1"]\n'
                                '[machines.far]\nhost = "me@far"\nssh = "ssh.exe"\nworkers = 3\nruns = ["es:0", "neat:2"]\n')
    here, far = fake.fleet.load_plan(plan)
    assert (here.local, here.runs, here.at_once, here.workers) == (True, jobs.parse_jobs("ga:1"), 2, 6)
    assert (far.local, far.host, far.ssh, far.dir, far.runs, far.at_once) == (False, "me@far", "ssh.exe", "ec-racing",
                                                                              jobs.parse_jobs("es:0,neat:2"), 1)


@pytest.mark.parametrize("text, message", [
    ('[machines.a]\nruns = ["ga:1"]\n[machines.b]\nhost = "x"\nruns = ["ga:1"]\n', "on both a and b"),
    ('[machines.a]\nruns = ["nb:03"]\n', "notebook jobs are"),
    ('[machines.a]\nruns = ["ga:3"]\n', "seed must be one of"),
    ('[machines.a]\nhost = "x"\nruns = ["ga:1"]\n', "exactly one machine"),
    ('[machines.a]\nruns = []\n[machines.b]\nruns = []\n', "exactly one machine"),
    ('[machines.a]\nrun = ["ga:1"]\n', "unknown keys run"),
    ('[machines.a]\nat_once = 0\n', "at_once must be"),
    ('[machines.a\n', "cannot be read"),
])
def test_a_plan_that_cannot_work_is_refused(fake, text, message):
    with pytest.raises(fake.fleet.PlanError, match=message):
        fake.fleet.load_plan(write_plan(fake.tmp, text))


def test_the_repository_plan_runs_the_whole_experiment_once_with_no_inputs_from_elsewhere(fake, monkeypatch):
    monkeypatch.setattr(fake.fleet, "PROJECT", SCRIPT.parent.parent)
    machines = fake.fleet.load_plan()
    labels = [job.label for machine in machines for job in machine.runs]
    expected = {"nb:01", "nb:02"} | {f"{a}:{s}" for a in config.APPROACHES for s in config.RUN_SEEDS}
    assert len(labels) == len(set(labels)) and set(labels) == expected
    assert sum(machine.local for machine in machines) == 1
    assert all(machine.workers % config.TRACKS_PER_GEN == 0 for machine in machines)
    assert all(fake.fleet.external_inputs(machine) == [] for machine in machines)

def test_an_invalid_plan_exits_with_code_2(fake, capsys):
    assert fake.fleet.main(["status"], plan=write_plan(fake.tmp, "[machines.a]\nhost = 'x'\n")) == 2
    assert "exactly one machine" in capsys.readouterr().err


# ----- transfers -----

def test_files_travel_to_a_server_and_back_intact(fake):
    server = fake.machines()[1]
    data = os.urandom(200_000)
    fake.fleet.upload(server, data, "results/in/data.bin")
    assert (fake.server_project() / "results" / "in" / "data.bin").read_bytes() == data
    assert not list(fake.home("server").glob(".ec-racing-upload-*"))       # nothing is left in the home folder
    archive = fake.fleet.download(server, ["results/in/data.bin", "results/in/none.*", "results/missing.txt"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        assert tar.getnames() == ["results/in/data.bin"] and tar.extractfile("results/in/data.bin").read() == data


def test_a_download_of_nothing_is_an_empty_archive(fake):
    fake.server_project().mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(fake.fleet.download(fake.machines()[1], ["results/none.*"]))) as tar:
        assert tar.getnames() == []


def test_a_damaged_download_is_refused(fake, monkeypatch):
    fake.server_project().mkdir(parents=True)
    monkeypatch.setattr(fake.fleet, "copy_from", lambda machine, source, local: Path(local).write_bytes(b"garbage"))
    with pytest.raises(fake.fleet.FleetError, match="damaged"):
        fake.fleet.download(fake.machines()[1], ["results/none.*"])


def test_a_damaged_upload_is_refused_and_left_out(fake, monkeypatch):
    def damage(machine, local, target):
        (fake.home(machine.name) / target).write_bytes(Path(local).read_bytes()[:-1])
    monkeypatch.setattr(fake.fleet, "copy_to", damage)
    with pytest.raises(fake.fleet.FleetError, match="arrived damaged"):
        fake.fleet.upload(fake.machines()[1], b"model weights", "results/runs/baseline/seed0/dagger.keras")
    folder = fake.server_project() / "results" / "runs" / "baseline" / "seed0"
    assert folder.is_dir() and list(folder.iterdir()) == []
    assert not list(fake.home("server").glob(".ec-racing-upload-*"))


def test_a_script_for_a_server_travels_in_the_command_line(fake):
    script = "set -e\nx='it''s'\nprintf '%s|%s\\n' \"$x\" \"$HOME\"\n"
    command = fake.fleet.remote_command(script)
    assert "\n" not in command and "'" not in command and "%" not in command
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                            env={**os.environ, "HOME": "/home/x"})
    assert result.stdout == "its|/home/x\n"
    with pytest.raises(fake.fleet.FleetError, match="too long"):
        fake.fleet.remote_command("echo x\n" * 2000)


def test_the_largest_scripts_fit_in_a_command_line(fake):
    server = fake.machines()[1]
    every_run = [jobs.parse_job(f"{a}:{s}") for a in config.APPROACHES for s in config.RUN_SEEDS]
    members = [m for run in every_run for m in fake.fleet.run_members(run)]
    for script in (fake.fleet.setup_script(server), fake.fleet.pack_script(server, members),
                   fake.fleet.collect_script(fake.fleet.Machine("s", every_run, host="h", ssh="ssh.exe")),
                   fake.fleet.probe_script(fake.fleet.Machine("s", every_run, host="h", ssh="ssh.exe")),
                   fake.fleet.session_script(server, fake.fleet.train_command(
                       fake.fleet.Machine("s", every_run, host="h", ssh="ssh.exe")))):
        fake.fleet.remote_command(script)


def test_windows_clients_run_from_a_batch_file_with_files_for_input_and_output(fake):
    text = fake.fleet.batch_text(["ssh.exe", "-o", "BatchMode=yes", "me@h", "echo QQ== | base64 -d | bash"],
                                 "C:\\Users\\Public\\f")
    assert text == ('@ssh.exe -o BatchMode=yes me@h "echo QQ== | base64 -d | bash" < NUL '
                    '> "C:\\Users\\Public\\f\\out.txt" 2> "C:\\Users\\Public\\f\\err.txt"\r\n@exit /b %ERRORLEVEL%\r\n')
    assert fake.fleet.windows_path("/mnt/c/Users/Public/f/file") == "C:\\Users\\Public\\f\\file"
    with pytest.raises(fake.fleet.FleetError):
        fake.fleet.windows_path("/home/sean/file")
    assert fake.fleet.scp_program(fake.fleet.Machine("s", [], host="h", ssh="ssh.exe")) == "scp.exe"
    assert fake.fleet.scp_program(fake.fleet.Machine("s", [], host="h", ssh="/usr/bin/ssh")) == "/usr/bin/scp"


def test_a_server_that_cannot_be_reached_says_what_to_check(fake, monkeypatch):
    monkeypatch.setattr(fake.fleet, "run_shell", lambda *a, **k: (255, "", "ssh: connect to host x: timed out"))
    with pytest.raises(fake.fleet.FleetError, match="GlobalProtect"):
        fake.fleet.shell(fake.machines()[1], "echo ok")


def test_setup_checks_out_this_commit_and_brings_the_baselines(fake, monkeypatch):
    git = ["git", "-C", str(fake.desktop), "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run(git + ["init", "-q"], check=True)
    (fake.desktop / "racing.py").write_text("print('code')\n")
    subprocess.run(git + ["add", "racing.py"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "code"], check=True)
    monkeypatch.setattr(fake.fleet, "local_commit", lambda: fake.fleet.git("rev-parse", "HEAD").strip())
    monkeypatch.setattr(fake.fleet, "environment_lines", lambda: "echo environment ready\n")   # no installs here
    refined = fake.desktop / "results" / "runs" / "baseline" / "seed0"
    refined.mkdir(parents=True)
    (refined / "dagger.keras").write_bytes(b"refined baseline 0")
    checkpoint.write_json(refined / "dagger.json", {"settings": checkpoint.settings_record()})
    kept = fake.server_project() / "results" / "kept.txt"
    kept.parent.mkdir(parents=True)
    kept.write_text("a result")
    assert fake.fleet.setup(fake.machines(), out=io.StringIO()) == 0
    assert (fake.server_project() / "racing.py").read_text() == "print('code')\n"
    assert (fake.server_project() / "results" / "runs" / "baseline" / "seed0" / "dagger.keras").read_bytes() == \
        b"refined baseline 0"
    assert kept.read_text() == "a result"                                   # the server's results stay
    commit = subprocess.run(["git", "-C", str(fake.server_project()), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    assert commit == fake.fleet.local_commit()


def test_setup_needs_the_baselines_here(fake, monkeypatch):
    monkeypatch.setattr(fake.fleet, "git", lambda *args: Path(args[-2]).write_bytes(b"bundle") if "bundle" in args
                        else "")
    with pytest.raises(fake.fleet.FleetError, match="is missing here; baseline:0 makes it"):
        fake.fleet.setup(fake.machines(), out=io.StringIO())


def test_connections_to_a_server_are_paced(fake, monkeypatch):
    monkeypatch.setattr(fake.fleet, "CONNECTIONS_PER_WINDOW", 2)
    monkeypatch.setattr(fake.fleet, "CONNECTION_WINDOW", 0.4)
    server = fake.machines()[1]
    began = time.monotonic()
    for _ in range(3):
        fake.fleet.pace(server, out=io.StringIO())
    assert time.monotonic() - began >= 0.4                                 # the third waited for the first to age
    fake.fleet.pace(fake.fleet.Machine("other", [], host="other@h"), out=io.StringIO())
    assert time.monotonic() - began < 1.5                                   # another server is not held up


def test_the_environment_is_python_3_13_with_the_pinned_packages(fake):
    script = fake.fleet.environment_lines()
    assert "python=3.13" in script and "-r requirements.txt" in script and "Miniforge3-Linux-x86_64.sh" in script


# ----- start -----

def test_nothing_starts_until_every_machine_is_ready(fake):
    with pytest.raises(fake.fleet.FleetError, match="project is not there") as error:
        fake.fleet.start(fake.machines(), out=io.StringIO())
    assert "nothing was started" in str(error.value)
    assert fake.session("desktop") is None and fake.session("server") is None


def test_a_missing_baseline_or_other_code_on_the_server_stops_the_start(fake):
    fake.ready_server()
    (fake.server_project() / "results" / "runs" / "baseline" / "seed0" / "dagger.keras").unlink()
    with pytest.raises(fake.fleet.FleetError, match="baseline/seed0/dagger.keras is missing"):
        fake.fleet.start(fake.machines(), out=io.StringIO())
    fake.ready_server()
    fake.commit = "1" * 40
    with pytest.raises(fake.fleet.FleetError, match="fleet.py setup"):
        fake.fleet.start(fake.machines(), out=io.StringIO())
    assert fake.session("desktop") is None and fake.session("server") is None


def test_a_runner_still_training_stops_the_start(fake):
    fake.ready_server()
    lock = fake.desktop / "results" / ".runner.lock"
    with open(lock, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.write("4242\n")
        handle.flush()
        with pytest.raises(fake.fleet.FleetError, match=r"process 4242\) is still training"):
            fake.fleet.start(fake.machines(), out=io.StringIO())
    assert fake.session("server") is None


def test_the_desktop_must_train_with_the_ec_racing_environment(fake, monkeypatch):
    fake.ready_server()
    monkeypatch.setattr(fake.fleet, "environment_here", lambda: False)
    with pytest.raises(fake.fleet.FleetError, match="conda activate ec-racing"):
        fake.fleet.start(fake.machines(), out=io.StringIO())
    assert fake.session("desktop") is None and fake.session("server") is None


def test_uncommitted_code_stops_the_start(fake, monkeypatch):
    monkeypatch.setattr(fake.fleet, "code_changes", lambda: " M racing/sim.py")
    with pytest.raises(fake.fleet.FleetError, match="racing/sim.py"):
        fake.fleet.start(fake.machines(), out=io.StringIO())


def test_start_opens_a_detached_session_on_every_machine(fake):
    fake.ready_server()
    out = io.StringIO()
    assert fake.fleet.start(fake.machines(), out=out) == 0
    desktop, server = fake.session("desktop"), fake.session("server")
    assert f"EC_RACING_JOBS=6 {sys.executable} scripts/run_notebooks.py --runs ga:1,neat:0 --at-once 2" in desktop
    assert ('"$HOME"/miniforge3/envs/ec-racing/bin/python scripts/run_notebooks.py --runs ga:2,es_seeded:0 '
            "--at-once 3") in server
    assert "-d -s fleet" in (fake.home("server") / ".tmux" / "calls").read_text()
    assert "ga:2, es_seeded:0" in out.getvalue()


def test_a_session_runs_in_the_project_folder_and_stays_open(fake):
    fake.ready_server()
    fake.env["TMUX_RUN"] = "1"
    server = fake.machines()[1]
    fake.fleet.shell(server, fake.fleet.session_script(server, "pwd > where.txt"))
    assert (fake.server_project() / "where.txt").read_text().strip() == str(fake.server_project())
    assert fake.session("server").endswith("exec bash")


def test_an_ended_session_is_replaced_at_the_next_start(fake):
    fake.ready_server()
    fake.fleet.start(fake.machines(), out=io.StringIO())
    fake.fleet.start(fake.machines(), out=io.StringIO())                  # the runners ended: the lock is free
    calls = (fake.home("server") / ".tmux" / "calls").read_text().splitlines()
    assert [call.split()[0] for call in calls].count("kill-session") == 1


# ----- status and stop -----

def test_status_shows_every_run_in_training_and_the_latest_milestones(fake, monkeypatch):
    monkeypatch.setattr(config, "RUN_ORDER", (("ga", 1), ("ga", 2), ("neat", 0), ("es_seeded", 0)))
    fake.ready_server()
    results = fake.server_project() / "results"
    progress.write_status({"approach": "ga", "seed": 2, "level": 2, "generation": 40, "min_share": 0.5},
                          results / "status" / "ga-seed2.json")
    progress.write_status({"approach": "es_seeded", "seed": 0, "level": 1, "generation": 9, "min_share": 0.1},
                          results / "status" / "es_seeded-seed0.json")
    (results / "run_log.txt").write_text("".join(f"milestone {k}\n" for k in range(10)))
    out = io.StringIO()
    assert fake.fleet.status(fake.machines(), out=out) == 0
    text = out.getvalue()
    assert "== desktop" in text and "== server (me@server)" in text and "runner: not running" in text
    assert text.index("GA seed 2") < text.index("Seeded ES seed 0")
    assert "milestone 9" in text and "milestone 3" not in text


def test_status_of_a_server_without_the_project(fake):
    out = io.StringIO()
    fake.fleet.status(fake.machines(), out=out)
    assert "not there yet" in out.getvalue()


def test_stop_presses_ctrl_c_in_every_session_and_waits(fake):
    fake.ready_server()
    fake.fleet.start(fake.machines(), out=io.StringIO())
    out = io.StringIO()
    assert fake.fleet.stop(fake.machines(), out=out, wait=10) == 0
    for name in ("desktop", "server"):
        assert "send-keys -t fleet C-c" in (fake.home(name) / ".tmux" / "keys").read_text()
    assert out.getvalue().count("stopped; the runs continue") == 2


def test_stop_without_sessions_does_nothing(fake):
    out = io.StringIO()
    assert fake.fleet.stop(fake.machines(), out=out, wait=1) == 0
    assert out.getvalue().count("no fleet session") == 2


# ----- collect and finish -----

def finished_run(results, approach, seed, settings=None, champion=b"weights"):
    run = results / "runs" / approach / f"seed{seed}"
    record = paths.final_record(approach, seed, results / "runs")
    record.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_json(record, {"approach": approach, "seed": seed,
                                   "settings": settings or checkpoint.settings_record()})
    (run / "log.csv").write_text("generation\n0\n")
    champions = results / "champions" / approach
    champions.mkdir(parents=True, exist_ok=True)
    (champions / f"seed{seed}.npy").write_bytes(champion)
    (champions / f"seed{seed}.setup.json").write_text("{}")
    return run


def test_collect_copies_the_servers_finished_runs_only(fake):
    fake.ready_server()
    server = fake.server_project() / "results"
    finished_run(server, "ga", 2)
    unfinished = server / "runs" / "es_seeded" / "seed0"
    unfinished.mkdir(parents=True)
    (unfinished / "checkpoint.pkl").write_bytes(b"state")
    (server / "run_log.txt").write_text("server milestones\n")
    out = io.StringIO()
    assert fake.fleet.collect(fake.machines(), out=out) == 0
    here = fake.desktop / "results"
    assert (here / "runs" / "ga" / "seed2" / "log.csv").exists()
    assert (here / "champions" / "ga" / "seed2.npy").read_bytes() == b"weights"
    assert not (here / "runs" / "es_seeded").exists()
    assert (here / "machines" / "server" / "run_log.txt").read_text() == "server milestones\n"
    assert "ga:2 copied" in out.getvalue() and "not finished yet: es_seeded:0" in out.getvalue()
    assert not (here / fake.fleet.INCOMING).exists() or not any((here / fake.fleet.INCOMING).iterdir())
    again = io.StringIO()
    assert fake.fleet.collect(fake.machines(), out=again) == 0
    assert "ga:2 already here" in again.getvalue()


def test_collect_never_replaces_a_different_run(fake):
    fake.ready_server()
    finished_run(fake.server_project() / "results", "ga", 2)
    here = fake.desktop / "results"
    finished_run(here, "ga", 2, champion=b"other weights")
    (here / "runs" / "ga" / "seed2" / "summary.json").write_text('{"settings": {}, "other": 1}')
    out = io.StringIO()
    assert fake.fleet.collect(fake.machines(), out=out) == 1
    assert "a different finished run is already" in out.getvalue()
    assert (here / "champions" / "ga" / "seed2.npy").read_bytes() == b"other weights"


def test_collect_never_replaces_an_unfinished_copy(fake):
    fake.ready_server()
    finished_run(fake.server_project() / "results", "ga", 2)
    partial = fake.desktop / "results" / "runs" / "ga" / "seed2"
    partial.mkdir(parents=True)
    (partial / "checkpoint.pkl").write_bytes(b"mine")
    out = io.StringIO()
    assert fake.fleet.collect(fake.machines(), out=out) == 1
    assert "unfinished copy" in out.getvalue() and (partial / "checkpoint.pkl").read_bytes() == b"mine"
    assert not (fake.desktop / "results" / "champions" / "ga").exists()


def test_collect_refuses_a_run_made_with_other_settings(fake):
    fake.ready_server()
    finished_run(fake.server_project() / "results", "ga", 2, settings={**checkpoint.settings_record(), "GRIP": 99.0})
    out = io.StringIO()
    assert fake.fleet.collect(fake.machines(), out=out) == 1
    assert "GRIP" in out.getvalue() and not (fake.desktop / "results" / "runs" / "ga").exists()


def test_finish_waits_for_every_run(fake):
    fake.ready_server()
    with pytest.raises(fake.fleet.FleetError, match="not every run has finished"):
        fake.fleet.finish(fake.machines(), out=io.StringIO())
    assert fake.session("desktop") is None


def test_finish_needs_all_24_runs_and_runs_the_notebooks_from_03(fake):
    fake.ready_server()
    on_server = {job.label for job in fake.machines()[1].runs}
    for approach in config.APPROACHES:
        for seed in config.RUN_SEEDS:
            target = fake.server_project() if f"{approach}:{seed}" in on_server else fake.desktop
            if (approach, seed) != ("kt_bayesian", 2):
                finished_run(target / "results", approach, seed)
    with pytest.raises(fake.fleet.FleetError, match="kt_bayesian:2"):
        fake.fleet.finish(fake.machines(), out=io.StringIO())
    finished_run(fake.desktop / "results", "kt_bayesian", 2)
    out = io.StringIO()
    assert fake.fleet.finish(fake.machines(), out=out) == 0
    assert f"{sys.executable} scripts/run_notebooks.py --from 03" in fake.session("desktop")
    assert fake.session("server") is None
    assert (fake.desktop / "results" / "runs" / "es_seeded" / "seed0" / "summary.json").exists()


def test_main_reports_a_failed_check_with_code_1(fake, capsys, monkeypatch):
    monkeypatch.setattr(fake.fleet, "load_plan", lambda plan: fake.machines())
    assert fake.fleet.main(["start"]) == 1
    assert "nothing was started" in capsys.readouterr().err


def test_runs_record_the_machine_they_finished_on():
    record = checkpoint.machine_record()
    assert set(record) == {"host", "processor", "logical_processors", "n_jobs"}
    assert record["n_jobs"] == config.N_JOBS and json.dumps(record)


def test_jobs_whose_inputs_the_machine_makes_itself_need_nothing_sent(fake):
    desktop = fake.fleet.Machine("desktop", jobs.parse_jobs("nb:01,nb:02,baseline:0,es_seeded:0"), at_once=4)
    assert fake.fleet.external_inputs(desktop) == []
    server = fake.fleet.Machine("server", jobs.parse_jobs("es_seeded:1,ga:0"), host="me@h")
    assert [target for _, target in fake.fleet.external_inputs(server)] == [
        "results/runs/baseline/seed1/dagger.json", "results/runs/baseline/seed1/dagger.keras"]


def test_a_dagger_run_on_a_server_counts_as_finished_only_after_its_speed_push(fake):
    script = fake.fleet.collect_script(fake.fleet.Machine("s", jobs.parse_jobs("baseline:1,nb:01"), host="h"))
    assert "results/runs/baseline/seed1/speed/summary.json" in script and "nb" not in script.split("done=()")[1]


def test_the_plan_reads_jobs(fake):
    plan = write_plan(fake.tmp, '[machines.here]\nruns = ["nb:01", "nb:02", "baseline:0", "es_seeded:0"]\n')
    (here,) = fake.fleet.load_plan(plan)
    assert [job.label for job in here.runs] == ["nb:01", "nb:02", "baseline:0", "es_seeded:0"]
    assert fake.fleet.train_command(here).endswith("--runs nb:01,nb:02,baseline:0,es_seeded:0 --at-once 1")
