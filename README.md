# EC Racing

EC Racing compares methods for finding small neural network driving controllers. Eight controllers race in a top-down
2D simulator. The cars have the grip and top speed of a Formula 1 car (1.8 g at low speed, rising to 5 g at 250 km/h;
at most 378 km/h; no engine power or drag model). Easy to Hard are generated circuits at Formula 1 size; Extra hard is
19 real training circuits, driven in both directions. Four held-out circuits (Melbourne, Sakhir, Sepang and
Silverstone) form the test, where each final controller's best flying lap is compared with the circuit's pole lap (lap times are whole steps of 0.05 s, so a lap within 0.05 s of pole is not resolvable):

- **Imitation learning:** a hand-designed baseline and three Keras Tuner searches (random search, Hyperband,
  Bayesian optimization) are trained to imitate a scripted expert driver, refined by DAgger and then pushed for speed
  by an evolution strategy grown from each of them.
- **Neuroevolution:** a genetic algorithm grows its own networks from the inputs wired to the outputs (hidden
  units and layers, no size limit), an evolution strategy evolves the weights of a fixed network, and NEAT evolves nodes
  and connections.
- **Hybrid:** a seeded evolution strategy starts from the DAgger-refined baseline and continues by driving.

Every method trains until it masters the curriculum: on every map of a generation (generated, or real circuits drawn
from the 38 training layouts), at least 90% of its population must complete three laps within the target time, three
generations in a row, on each level from Easy to Extra hard. In the speed push that follows, the target time is
shortened by 2% at a time whenever half of the population beats it three generations in a row, until 1,000 generations
pass without that. Every method is trained four times: run 0 is seeded (seed 0) and replays exactly, runs 1 to 3 set
no seed, so their random choices differ every time; results are reported for each run and their mean. Every evolved
car also evolves its own setup next to its network: the length of its distance sensors (30–1,000 m, 150 m at the
start) and its steering sensitivity (⅓–3), judged only by success and lap time.

Every car chooses one of 45 actions: nine steering levels in steps of 0.25 (full lock left to full lock right) and five
throttle levels (full and half brake, coast, half and full gas), so a network has 45 outputs (spec §6).

**Stall boost:** after 100 speed-push generations without a mastered target, the decision layer (the output layer's
weights and biases; NEAT's connections into output nodes) mutates three times as hard, and the steering sensitivity
mutates with probability 0.5 and standard deviation 0.2, until the next target is mastered; the speed push ends after
1,000 such generations (spec §9.4). The boost only scales random draws that are made anyway, so run 0 still replays
exactly. `python scripts/boost_pilot.py boost` and `python scripts/boost_pilot.py plain` compare the speed push with
and without it (spec §12.2); the results are in `results/pilots/revision3.6/`.

Generated maps have a road width that changes along the lap, and the networks see their position across the road and
the road's width next to the nine distance rays and the speed (12 inputs), so that they drive any width rather than
the one they trained on. The evaluation adds a robustness table: the four test circuits rebuilt at constant road
widths of 6, 10, 14 and 18 m, the first narrower than any road met in training. The design specification is
`docs/ec-racing-design.pdf` (version 3.9, which incorporates every revision).

The evaluation also reports the pole-lap table and figure of the four test circuits (notebook 08).

**Data and licences:** the real circuits come from the racetrack database of the Technical University of Munich
(https://github.com/TUMFTM/racetrack-database), LGPL-3.0, vendored in `data/racetracks/` (see
`data/racetracks/SOURCE.md`).

## Setup on a Windows desktop (WSL 2)

The experiment runs on Linux. On Windows it runs in the Windows Subsystem for Linux.

1. **Install WSL 2 with Ubuntu.** In PowerShell as administrator: `wsl --install -d Ubuntu`, restart Windows, then
   open *Ubuntu* from the Start menu and create a Linux user when asked.
2. **Install Miniconda** in the Ubuntu terminal:

   ```bash
   cd ~
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b
   ~/miniconda3/bin/conda init bash
   exec bash
   ```

3. **Give WSL access to GitHub over SSH:**

   ```bash
   ssh-keygen -t ed25519 -C "ec-racing desktop"      # press Enter at each question
   cat ~/.ssh/id_ed25519.pub
   ```

   Copy the printed line into GitHub → Settings → SSH and GPG keys → New SSH key, then check it with
   `ssh -T git@github.com`.
4. **Clone the project into the Linux home folder** (not under `/mnt/c`: files on the Windows drive are much slower
   from WSL):

   ```bash
   cd ~
   git clone <repository SSH URL> ec-racing      # the URL under Code → SSH on the repository's GitHub page
   cd ec-racing
   git checkout build
   ```

5. **Create the Python environment:**

   ```bash
   conda create -y -n ec-racing python=3.13
   conda activate ec-racing
   pip install -r requirements.txt
   python -m pytest -q                               # the fast tests should all pass
   ```

   If `conda` asks to accept the terms of service of its default channels, run the command it prints and repeat
   `conda create`. Activate the environment (`conda activate ec-racing`) in every new terminal.
### Compiled simulator

Training runs its time trials on a compiled simulator (`racing/fastsim.cpp`, specification §5.7), which gives
exactly the numpy simulator's results at about a tenth of the processor time. It needs g++:

    sudo apt install build-essential

Without administrator rights, conda-forge's compiler works as well: `conda install -c conda-forge gxx` into the
environment. The `g++` beside the environment's Python is used first, then `g++` on the path; `EC_RACING_CXX` names
another compiler. `scripts/fleet.py setup` installs it on a server that has no g++.

The library is built into `racing/_fastsim-<id>.so` the first time a run needs it, where `<id>` identifies the
source, the compiler flags and the processor, so a `git pull` that changes `fastsim.cpp` leads to a new build. When
the library loads, it is checked against numpy on this machine (math functions and network outputs). Without g++, or
if the check finds a difference, a run prints one warning and uses the numpy simulator, with the same results.
`EC_RACING_SIM=numpy` forces numpy. The compiled simulator uses every logical processor available to the run;
runs side by side share them through the operating system at no cost (specification §3.3). `EC_RACING_THREADS=8`
fixes the number, to keep processors free for other work.

6. **Keep Windows awake and without restarts.** WSL pauses while Windows sleeps. Set Settings → System → Power →
   *Screen and sleep* → "When plugged in, put my device to sleep after" to *Never* (or run `powercfg /change
   standby-timeout-ac 0` in PowerShell) for the duration of the run. Pause Windows Update for the duration of the run
   as well (Settings → Windows Update → *Pause updates*): an update restart stops the run, which then has to be
   resumed by hand.
7. **Give WSL enough memory.** WSL 2 may use only about half of the machine's memory by default, and every worker
   process holds its own Python, NumPy and track cache (up to 8 built tracks, about 0.8 GB; a real circuit's road grid
   takes up to about 105 MB and is loaded from the cache, its distance map memory-mapped; see Several machines). To
   raise the limit, create `%UserProfile%\.wslconfig` in Windows with, for example,

   ```ini
   [wsl2]
   memory=24GB
   ```

   sized for the machine (leave a few GB for Windows), then run `wsl --shutdown` in PowerShell and open Ubuntu again
   to apply it. Check the memory with `free -h` during the first hour at Extra hard (notebook 03 reaches it first);
   if *available* falls below about 2 GB, stop the run, lower `EC_RACING_JOBS` and resume.
8. **Worker processes.** The compiled simulator needs no setting: it uses every processor. `EC_RACING_JOBS` (default
   6) is the number of worker processes of the numpy simulator, DAgger's driving, dataset collection and the final
   evaluation; each holds its own maps (step 7). `python -m racing.bench generation` times only the simulator in
   use; `EC_RACING_SIM=numpy python -m racing.bench generation` times numpy and ends with a line such as
   `export EC_RACING_JOBS=16`; add it to `~/.bashrc` to keep it.

To update the code later: `git pull`, but never while a run is in progress (see below).

## Setup on Linux

On a Linux machine follow steps 2–8 without WSL, and install g++ as described in the "Compiled simulator" subsection
after step 5. On the development laptop the Conda base environment is used instead of `ec-racing`, with
`~/miniconda3/bin/python` in place of `python`.

## Running the experiment

One command runs the eight notebooks in order and saves their outputs into the notebook files:

```bash
python scripts/run_notebooks.py
```

The run takes days rather than hours: every method trains until it masters the curriculum and can no longer get
faster. A live status line shows the current run, level, generation, the worst map's success share against 90%, the
streak and the elapsed time (during the imitation training of notebook 02: the epoch and validation accuracy, or the
Keras Tuner trial); level-ups, mastery goals, target reductions and finished runs are printed as permanent lines and
kept in `results/run_log.txt`. The notebooks run on the CPU, and what their kernels print themselves (TensorFlow and
library messages) goes to `results/kernel_log.txt` instead of the terminal.

Under WSL 2, Windows can stop the Linux distribution when no WSL terminal is open, which would end the run. Keep a
WSL terminal open for the whole run (it may be minimized). The recommended way is `tmux`, which also lets a terminal
window be replaced without stopping the run:

```bash
sudo apt install tmux                                # once
tmux new -s ec-racing                                # starts a session
conda activate ec-racing
python scripts/run_notebooks.py                      # runs in the foreground inside the session
```

Detach with Ctrl+b then d (the run continues), and reattach with `tmux attach -t ec-racing`. Alternatively, run it with
`nohup`, writing the console output to a log file; a WSL window must still stay open:

```bash
mkdir -p results
nohup python scripts/run_notebooks.py > results/run_console.txt 2>&1 &
```

From another terminal, watch the run with:

```bash
python -m racing.progress                            # live status; Ctrl+C stops watching only
```

- **Stopping and resuming:** stop with Ctrl+C, or from another terminal with `pkill -f scripts/run_notebooks.py`. The
  runner ends the notebook's kernel (and its worker processes), notes the stop in the run log and exits with code 130
  within a few seconds. Start the same command again to continue: finished runs are skipped and an interrupted run
  continues from its last checkpoint, losing at most one generation. A resumed run 0 produces exactly the same results
  as an uninterrupted one; a resumed run (1 to 3) continues from the state of its random generator saved in the
  checkpoint.
- **One runner at a time:** a second `run_notebooks.py` on the same `results/` stops at once with exit code 4 and
  names the process that holds `results/.runner.lock`.
- **After a crash or power loss:** if a result file was cut short, the runner lists it and exits with code 3. Delete
  only the files it lists and start again; the steps they belong to are redone. Do not use `--fresh` for this.
- **A stopped run:** when the status has not changed for ten minutes, the status line ends with
  "(no update for N min)": the run may have stopped, or be in a step without status updates such as the evaluation.
  Check whether the runner is still running (`tmux attach -t ec-racing`, or `ps`).
- **During the run:** do not `git pull` or edit the code (the runner records the commit of every start and warns
  when it changed), and do not use the viewer's training view, which competes for the processors and inflates the
  reported times.
- **From a notebook:** `python scripts/run_notebooks.py --from 04` starts at notebook 04.
- **Starting over:** `python scripts/run_notebooks.py --fresh` removes the earlier results (runs, controllers,
  dataset, tables, figures and run record; pilots, viewer runs and the archive are kept) and starts from notebook 01.
- **Keeping the earlier results:** `python scripts/run_notebooks.py --archive NAME` moves what `--fresh` would delete
  to `results/archive/NAME/` and exits without running anything; the next start begins afresh. It never overwrites an
  archive. The revision 3.1 results are archived as `revision3.1`.
- **Changed settings:** if the settings in `racing/config.py` differ from those recorded by earlier runs, the runner
  stops before running anything, names the differing settings and exits with code 3. Run again with `--fresh`, or
  archive the earlier results first with `--archive`.

| Notebook | Purpose |
|---|---|
| `01_dataset` | Expert dataset on the generated levels and the real layouts |
| `02_imitation` | Baseline, Keras Tuner searches, training of runs 0 to 3 |
| `03_dagger` | DAgger refinement and speed push of the four imitation methods, runs 0 to 3 each |
| `04_ga` | Genetic algorithm, runs 0 to 3 |
| `05_es` | Evolution strategy, runs 0 to 3 |
| `06_es_seeded` | Seeded evolution strategy, runs 0 to 3 |
| `07_neat` | NEAT, runs 0 to 3 |
| `08_evaluate` | Tables and figures |

The processor, worker count and thread setting, package versions, settings, code version (git commit) and notebook times are recorded
in `results/run_info.json`. Each evolutionary run's `summary.json` also records the machine it finished on (host,
processor, worker count, thread setting), because runs may be trained side by side or on several machines (below). Their training
times then depend on the machine and on what else ran on it, so compare the methods' cost by evaluations
(`evals`, exact on any machine) and give training times as context.

Results are written to `results/`, which is not under version control:

| Path | Contents |
|---|---|
| `results/dataset.npz`, `results/dataset_info.json` | Expert dataset and its collection time |
| `results/runs/<method>/seed<k>/` | Evolutionary runs: `log.csv` (one row per generation), `checkpoint.pkl`, `summary.json` |
| `results/runs/<method>/seed<k>/` (imitation methods) | `imitation.keras`, `dagger.keras`, the DAgger rows, state and summary, and `speed/` for the speed push |
| `results/runs/<method>/search/` | Keras Tuner search: trials, summary and exported models |
| `results/champions/<method>/seed<k>.keras` | Final controllers (evolved dense networks also as `.npy`; NEAT genomes as `.json`) |
| `results/champions/best_seeds.json` | Each method's better run on the validation maps, named in the report's tables |
| `results/tables/`, `results/figures/` | Results tables (CSV and Markdown) and figures, including the network figure |
| `results/status.json`, `results/run_info.json`, `results/run_log.txt`, `results/run_console.txt` | Live status, record, log and console output of the run |
| `results/status/`, `results/train_logs/` | Status file and output of each run trained side by side (`--runs`) |
| `results/machines/<name>/` | Run log and record of a server, copied by `scripts/fleet.py collect` |
| `results/pilots/revision3/`, `results/pilots/revision3.1/`, `results/pilots/revision3.2/`, `results/pilots/revision3.8/` | Pilot runs of revisions 3, 3.1 (with setup evolution), 3.2 (with road widths) and 3.8 (Formula 1 car, real circuits) |
| `results/circuit_grids/` | The real circuits' road grids, built once per machine (23 files, about 570 MB), kept between runs |
| `results/app_runs/<method>/seed<k>/`, `results/app_runs/champions/` | Training started in the viewer, and the best car of each such run; never part of the reported results |
| `results/archive/revision2-notebooks/` | The notebooks of the revision 2.0 run, with their outputs |
| `results/archive/revision3.1/` | The revision 3.1 results (runs, controllers, dataset, run record), shown by the viewer's **Runs** button |

To copy results to Windows, open `\\wsl$\Ubuntu\home\<Linux user>\ec-racing\results` in File Explorer.

### Several machines

A compiled run uses every processor and runs side by side share them, so several jobs side by side keep a machine busy
through each run's map building and Keras phases, and several machines can share the experiment. The runs are
independent: run 0's results depend only on its seed and the settings, so where it trains changes only its training
time, while runs 1 to 3 also depend on chance. Keras jobs come first: they train with 4 TensorFlow threads, and an
evolution phase runs at the lowest priority (nice 19), using whatever processors the searches and DAgger refinements
leave (specification §3.3).

**Side by side on one machine:**
`python scripts/run_notebooks.py --runs nb:01,search:kt_random,baseline:0,ga:1 --at-once 8` runs these jobs: runs of
any method, trained directly (`python -m racing.train <approach> <seed>`, the same calls as their notebooks), the
Keras Tuner searches (`search:kt_random`, `search:kt_hyperband`, `search:kt_bayesian`;
`python -m racing.train search <method>`) and notebook 01 (`nb:01`), executed with its outputs saved. Each job has its
own status line and output log in `results/train_logs/`, and starts, in the order listed, as soon as a slot is free and
its inputs exist:

| Job | Waits for |
|---|---|
| `nb:01` (dataset), `ga:k`, `es:k`, `neat:k` | nothing |
| `search:kt_random`, `search:kt_hyperband`, `search:kt_bayesian` | the dataset (notebook 01) |
| `baseline:k` (imitation network, DAgger, speed push) | the dataset |
| `kt_random:k`, `kt_hyperband:k`, `kt_bayesian:k` (retraining, DAgger, speed push) | the method's search |
| `es_seeded:k` | the refined Baseline of seed k (its DAgger run) |

A waiting job holds no slot. A job whose input nothing in the list makes is refused before anything starts; if a job
fails, the jobs waiting for it are reported and the others go on. Ctrl+C stops the running jobs and lists the ones not
started; the same command continues them. Afterwards, `python scripts/run_notebooks.py --from 02` finds the networks,
searches and runs complete and only reports them.

**On several machines:** `scripts/fleet.py` starts, watches, stops and collects runs on the desktop and on servers
reached from it, following `scripts/fleet.toml` (which machine runs which jobs, how many at a time, with how many
workers each). The plan follows measured generation times and memory (its comments show the arithmetic) and runs
every job at once: the desktop the dataset, the three searches, the 16 imitation runs and the 4 seeded ES runs, about
7.8 h; the server every GA, ES and NEAT run (12 runs), about 7 h (the speed push's patience of 1,000 generations adds
about 20% to the times of the plan's first estimate, and each stall the boost breaks restarts it, so a run can take
longer). It runs on the desktop, in its Ubuntu shell with the
`ec-racing` environment active. A server is reached with Windows' `ssh.exe`, which goes through the school VPN
(GlobalProtect); WSL's own `ssh` does not.

Once, connect GlobalProtect and give the server Windows' SSH key, so that fleet.py can log in without a password:

```bash
ssh-keygen.exe -t ed25519                            # press Enter three times
cat "/mnt/c/Users/<Windows user>/.ssh/id_ed25519.pub" | ssh.exe <user>@<server> "mkdir -p ~/.ssh; chmod 700 ~/.ssh; cat >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"
ssh.exe -o BatchMode=yes <user>@<server> hostname    # prints the server's name, without a password
```

Then, with GlobalProtect connected:

```bash
python scripts/fleet.py setup      # the server gets this commit's code, Python 3.13 with the pinned packages, and the
                                   # inputs its jobs need from the desktop, if any (again after every code change)
python scripts/fleet.py start      # every machine starts its runs in a detached tmux session named "fleet"
python scripts/fleet.py status     # one line per run in training on every machine, and the latest milestones
python scripts/fleet.py stop       # Ctrl+C in every session; the next start continues where the runs stopped
python scripts/fleet.py collect    # copies the servers' finished runs here (any time; finished runs only)
python scripts/fleet.py archive revision3.1   # every machine moves its earlier results to results/archive/revision3.1/
python scripts/fleet.py finish     # once all 32 runs have finished: collect, then notebooks 02 to 08 here
```

**Version 3.8** changes every setting that affects results, so every result of version 3.7 and earlier is refused by
`run_notebooks.py --check` and is set aside before the new run:

```bash
python scripts/fleet.py stop
python scripts/fleet.py setup
python scripts/fleet.py archive revision3.6
# on each machine, before start: measure the peak memory of one evolution job and one Keras job, e.g. the 3.8 pilots
# (scripts/pilots.py) plus one final selection (the largest peak), each under `/usr/bin/time -v` ("Maximum resident
# set size"), then set that machine's at_once in scripts/fleet.toml from them
python scripts/fleet.py start
```

Each machine builds the real circuits' road grids once, in `results/circuit_grids/` (46 files for 23 grids, about
570 MB, kept between runs; grids of older code are removed after a day). Memory measured in version 3.8, peak of one
process: building one real circuit's grid peaks near 1 GB (Monza 976 MB, about 1 s); one generation's three maps
0.34 GB at Easy, 0.39 GB at Medium and 0.69 GB at Hard; the final selection, which holds all 20 validation maps at
once, about 2.4 GB; building one validation map or a robustness-width map peaks at 1.2–1.4 GB.
The `at_once` bounds in `scripts/fleet.toml` were set for version 3.8 from the pilots (desktop 20 jobs at once, server
12); version 3.9 changes only the output layers, so they hold.

**Version 3.9** changes the actions (45 instead of 15) and the speed push (50%, patience 1,000), so every result of
version 3.8 is refused by `run_notebooks.py --check` and is set aside before the new run; the pole check of the 3.8
controllers stays with it:

```bash
python scripts/fleet.py stop
python scripts/fleet.py setup
python scripts/fleet.py archive revision3.8
python scripts/fleet.py start
```

Version 3.7 trained only the GA again; the runs of the other methods made with 3.6 stay valid:

```bash
python scripts/fleet.py stop
python scripts/fleet.py setup
python scripts/fleet.py archive revision3.6-ga --runs ga:0,ga:1,ga:2,ga:3
python scripts/fleet.py start
```

The runners stop first, so that no job of 3.6 starts with the code of 3.7. The unfinished ES, NEAT and imitation runs
continue from their checkpoints at the next start, while the GA runs of 3.6 are set aside and trained again. The race
view's "best" run per method (`results/champions/best_seeds.json`) is renewed when notebook 08 runs after the GA runs
have finished (`fleet.py finish`).

- `start` checks every machine first and starts nothing unless all are ready: the server has this commit (commit
  code changes first; a server gets the last commit), its environment and inputs, no machine holds earlier results
  made with other settings (keep those aside with `fleet.py archive NAME`, or only some runs with
  `fleet.py archive NAME --runs ga:0,ga:1`), and no runner is still training on any
  machine. Stop a runner started by hand first (Ctrl+C in its tmux session).
- The runs keep going when GlobalProtect or SSH disconnects; only fleet.py's commands need the connection. On the
  desktop, keep WSL running as for any run (see above).
- Run fleet.py in a terminal that stays open until the command ends, never with `nohup` or `setsid`: Windows programs
  that WSL starts (`ssh.exe`) end with the WSL session that started them. `setup` installs the server's environment
  detached on the server, so a dropped connection cannot stop the install; it follows the install's log
  (`~/ec-racing/.fleet/setup.log` on the server) and can simply be run again, which continues a running install
  rather than starting a second one. SSH notices a dead connection within about a minute.
- fleet.py opens few SSH connections per command (`setup` three, `start` two, `status` one, `collect` two) and, as
  a precaution against servers that block addresses connecting too often, at most four per 30 seconds.
- If a command says the server did not answer, check that GlobalProtect is still connected: its tunnel can stop
  passing traffic while the tray icon still shows it as connected; reconnect it and run the command again.
- `ssh.exe` misbehaves when WSL passes it input or output (it does not exit, and it stalls on larger input), so
  fleet.py runs it from a batch file in `C:\Users\Public\ec-racing-fleet` with Windows' own redirection, passes
  scripts in the command line and copies files with `scp.exe`.
- `collect` never replaces a run: a run that is already here in another version, an unfinished copy of it, or a run
  made with other settings is reported and left alone.
- Watch a session directly with `tmux attach -t fleet` (on a server, after `ssh.exe <user>@<server>`); detach with
  Ctrl+b then d. After the runner ends, the window stays open and shows its exit code.

## Pilots and benchmarks

```bash
python scripts/pilots.py ga es neat       # short runs from Easy: is the 90% requirement reachable?
python -m racing.bench generation         # generation time per thread or worker count
python -m racing.bench keras              # Keras epoch time per steps_per_execution and jit_compile (XLA stays
                                          # off: on the desktop it failed after ~130 Keras Tuner trials)
```

The pilots write to `results/pilots/revision3.8/` and are not part of the reported results; their summary includes
the populations' final mean sensor length and steering sensitivity. Before notebook 03 has run, leave out the seeded
pilot: it needs a refined baseline of this revision.

## Viewer

```bash
python -m racing.app
```

On Windows 11 the window opens through WSLg.

- **Training view:** choose an approach, run (*0 (seeded)* or *1*), difficulty and population size, then press Play
  (or Space). Training follows the mastery rules and continues until Stop, which freezes the cars where they are.
  Training runs in step with the replay: each generation is replayed once, then the next one trains, so the generation
  on screen is the one just trained. The playback speed sets the pace; Pause freezes both. In the race view training
  runs at full speed. The panel shows the level, each map's success share against 90%, the streak, the target
  reduction and the population's mean sensor length and steering sensitivity.
- **Saved viewer training:** training started here is kept in `results/app_runs/<method>/seed<k>/`, separate from the
  reported results. For GA, ES, seeded ES and NEAT, Play continues the saved run of the selected method and run where
  it stopped (difficulty and population size apply only to a new run); **New run** deletes it and starts over. If a
  saved run cannot be resumed (other settings after a code update, or a damaged file), the panel says so: press New
  run. Baseline and Keras Tuner training always start afresh.
- **Watching a notebook run:** **Source: Notebook run** replays the latest generation of the reported run of the
  selected method and run (`scripts/run_notebooks.py`), with the same panel, plot and Show selector, so you can watch
  the whole population or just the best car (Best, 1 car). When a replay ends, the newest generation is loaded, so a
  run in progress is followed live; the view only reads the run, never slows it down or writes to it. For Baseline and
  Keras Tuner it shows the speed push. Runs started before this feature have no replay file; race their cars instead.
- **Earlier revisions' runs:** the **Runs** button (beside Source, and in the race view beside the Race control)
  switches between the current results and each archive in `results/archive/` (for example `revision3.1`). An archived
  run is replayed, and its controllers race, on its own revision's maps (revision 3.1: one road width per level, 16,
  14, 12 and 10 m, read from the archive's `run_info.json`), and its networks read only the inputs they were trained
  with (revision 3.1: the nine rays and the speed). Every run of an archive was seeded: the Run control reaches each
  of its runs (an earlier revision may also hold run 2), and its cars are named by their seeds (*GA s0*, *GA s1*, …)
  and carry no stripe. A run whose networks do not match the selected folder is named in the status line instead.
  Training started in the viewer always uses the current results.
- **Race view:** pick a test map with the arrow keys and a difficulty; the map is shown with the cars on the start
  line. The **Race** control chooses the runs that race: *best runs* (each method's best run; the default), *all runs*
  (every saved run of every method, up to 32 cars), *seeded runs* (run 0 of every method) or *unseeded runs* (runs 1
  to 3), and each method's button below it switches that method on or off; both are kept while the viewer is open. All
  runs of a method drive in its color; runs 1 to 3 cars carry a white stripe, and the leaderboard names run 0 *GA
  (seeded)* and runs 1 to 3 *GA r1* to *GA r3*. Press Start race; Stop returns to the start line. The **Controllers**
  button switches the list to the **app runs**: the best car of each viewer training run (saved after every
  generation, or whenever an imitation network improves), labelled for example *GA (app) s1*. App cars start switched
  off, race next to the reported controllers when switched on (drawn with a white border), and are never part of the
  reported results. When many cars race, the leaderboard lists as many as fit.
- **Grip:** *Normal* (a(v), 1.8 g rising to 5 g, the training grip) or *Unlimited* (10 × a(v)) sets the grip of the
  cars you watch in both views. It never affects training or results.
- **Sensor rays:** shows the outlined car's nine rays, each drawn to that car's own sensor length.
- **Camera:** *Whole map*, or *Follow 2×*, *4×* or *8×*: zoomed in and following the best car (the outlined one) in
  the training view, or the race leader (the car furthest along that is still racing) in the race view. The mouse
  wheel over the map zooms in and out, and the C key cycles the camera as the button does.
- **Full screen:** the Full screen button or F11 fills the display; Windowed, F11 or Esc returns to a window, which
  can also be resized by dragging its border. The map and panel follow the window's size.
- **Network window:** the Network window button opens a separate window (movable to another monitor, resizable, closed
  with the viewer) that draws a car's neural network. In the training view it shows the best car of the latest
  generation (the latest epoch or trial for imitation) and redraws with every report, so NEAT's growth is visible; in
  the race view the Network of button cycles through the cars in the race.
- **Difficulty:** Easy (gentle loops), Medium (circuits), Hard (circuits with a main straight into a hairpin,
  sometimes a chicane or esses), Extra hard (the 19 real training circuits, in both directions, with their real
  widths; the test circuits are the four held-out ones). Generated maps run clockwise or counter-clockwise and the
  panel's map line names the road width.
- **Controllers of earlier revisions** have 10 inputs instead of 12; the race view leaves them out and says so.
- **Playback speed:** 1× is twice real time.

## Tests

```bash
python -m pytest              # fast tests
python -m pytest -m slow      # slow tests: generator acceptance, Keras Tuner, runner, end-to-end
```
