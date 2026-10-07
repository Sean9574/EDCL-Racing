# EC Racing

EC Racing compares methods for finding small neural network driving controllers. Eight controllers race on freshly
generated circuits, whose harder levels borrow features of Formula 1 circuits, in a top-down 2D simulator:

- **Imitation learning:** a hand-designed baseline and three Keras Tuner searches (random search, Hyperband,
  Bayesian optimization) are trained to imitate a scripted expert driver, refined by DAgger and then pushed for speed
  by an evolution strategy grown from each of them.
- **Neuroevolution:** a genetic algorithm and an evolution strategy evolve the weights of a fixed network by driving;
  NEAT evolves the network structure together with its weights.
- **Hybrid:** a seeded evolution strategy starts from the DAgger-refined baseline and continues by driving.

Every method trains until it masters the curriculum: on every freshly generated map of a generation, at least 90% of
its population must complete three laps within the target time, three generations in a row, on each level from Easy
to Extra hard. The target time is then shortened by 5% at a time until the population can no longer keep up. Every
method runs with three seeds; results are reported as mean ± standard deviation. The design specification is
`docs/2026-10-05-ec-racing-design.md` (version 3.0); the revision 3 specification and plan are in `docs/`.

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
6. **Keep Windows awake.** WSL pauses while Windows sleeps. Set Settings → System → Power → *Screen and sleep* →
   "When plugged in, put my device to sleep after" to *Never* (or run `powercfg /change standby-timeout-ac 0` in
   PowerShell) for the duration of the run.
7. **Choose the number of worker processes** for this machine:

   ```bash
   python -m racing.bench generation
   ```

   It times one generation for each worker count up to the number of logical processors and ends with a line such as
   `export EC_RACING_JOBS=16`. Make it permanent:

   ```bash
   echo 'export EC_RACING_JOBS=16' >> ~/.bashrc       # use the number the benchmark printed
   source ~/.bashrc
   ```

To update the code later: `git pull`.

## Setup on Linux

On a Linux machine follow steps 2–7 without WSL. On the development laptop the Conda base environment is used
instead of `ec-racing`, with `~/miniconda3/bin/python` in place of `python`.

## Running the experiment

One command runs the eight notebooks in order and saves their outputs into the notebook files:

```bash
python scripts/run_notebooks.py
```

The run takes days rather than hours: every method trains until it masters the curriculum and can no longer get
faster. A live status line shows the current run, level, generation, the worst map's success share against 90%, the
streak and the elapsed time; level-ups, mastery goals, target reductions and finished runs are printed as permanent
lines and kept in `results/run_log.txt`.

To run it in the background, so the terminal can be closed:

```bash
mkdir -p results
nohup python scripts/run_notebooks.py > results/run_console.txt 2>&1 &
python -m racing.progress                            # live status from any terminal; Ctrl+C stops watching only
```

- **Stopping and resuming:** stop with Ctrl+C (or `kill` the background process). Start the same command again to
  continue: finished runs are skipped and an interrupted run continues from its last checkpoint, losing at most one
  generation. A resumed run produces exactly the results of an uninterrupted one.
- **From a notebook:** `python scripts/run_notebooks.py --from 04` starts at notebook 04.
- **Starting over:** `python scripts/run_notebooks.py --fresh` removes the earlier results (runs, controllers,
  dataset, tables, figures and run record; pilots, viewer runs and the archive are kept) and starts from notebook 01.
- **Changed settings:** if the settings in `racing/config.py` differ from those recorded by earlier runs, the runner
  stops before running anything, names the differing settings and exits with code 3. Run again with `--fresh`.

| Notebook | Purpose |
|---|---|
| `01_dataset` | Expert dataset on freshly generated maps |
| `02_imitation` | Baseline, Keras Tuner searches, training with each seed |
| `03_dagger` | DAgger refinement and speed push of the four imitation methods, three seeds each |
| `04_ga` | Genetic algorithm, three seeds |
| `05_es` | Evolution strategy, three seeds |
| `06_es_seeded` | Seeded evolution strategy, three seeds |
| `07_neat` | NEAT, three seeds |
| `08_evaluate` | Tables and figures |

The processor, worker count, package versions, settings, code version (git commit) and notebook times are recorded
in `results/run_info.json`. All reported times come from one machine.

Results are written to `results/`, which is not under version control:

| Path | Contents |
|---|---|
| `results/dataset.npz`, `results/dataset_info.json` | Expert dataset and its collection time |
| `results/runs/<method>/seed<k>/` | Evolutionary runs: `log.csv` (one row per generation), `checkpoint.pkl`, `summary.json` |
| `results/runs/<method>/seed<k>/` (imitation methods) | `imitation.keras`, `dagger.keras`, the DAgger rows, state and summary, and `speed/` for the speed push |
| `results/runs/<method>/search/` | Keras Tuner search: trials, summary and exported models |
| `results/champions/<method>/seed<k>.keras` | Final controllers (evolved dense networks also as `.npy`; NEAT genomes as `.json`) |
| `results/champions/best_seeds.json` | Each method's best seed on the validation maps |
| `results/tables/`, `results/figures/` | Results tables (CSV and Markdown) and figures, including the network figure |
| `results/status.json`, `results/run_info.json`, `results/run_log.txt`, `results/run_console.txt` | Live status, record, log and console output of the run |
| `results/pilots/revision3/` | Pilot runs |
| `results/archive/revision2-notebooks/` | The notebooks of the revision 2.0 run, with their outputs |

To copy results to Windows, open `\\wsl$\Ubuntu\home\<Linux user>\ec-racing\results` in File Explorer.

## Pilots and benchmarks

```bash
python scripts/pilots.py ga es neat       # short runs from Easy: is the 90% requirement reachable?
python -m racing.bench generation         # generation time per worker count; prints the EC_RACING_JOBS to use
python -m racing.bench keras              # Keras epoch time per steps_per_execution and jit_compile
```

The pilots write to `results/pilots/revision3/` and are not part of the reported results.

## Viewer

```bash
python -m racing.app
```

On Windows 11 the window opens through WSLg.

- **Training view:** choose an approach, run seed, difficulty and population size, then press Play (or Space).
  Training follows the mastery rules and continues until Stop. The panel shows the level, each map's success share
  against 90%, the streak and the target reduction. Training started here writes only to `results/app_runs/`.
- **Race view:** pick a test map with the arrow keys and a difficulty; the map is shown with the cars on the start
  line. For each method, choose its best seed, a specific seed, or off, then press Start race; Stop returns to the
  start line.
- **Difficulty:** Easy (gentle loops on a wide road), Medium (circuits), Hard (circuits with a main straight into a
  hairpin, sometimes a chicane or esses), Extra hard (narrow circuits with two hairpins, chicanes, esses and fast
  sweepers). Maps run clockwise or counter-clockwise.
- **Playback speed:** 1× is twice real time.

## Tests

```bash
python -m pytest              # fast tests
python -m pytest -m slow      # slow tests: generator acceptance, Keras Tuner, runner, end-to-end
```
