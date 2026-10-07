# EC Racing

EC Racing compares methods for finding small neural network driving controllers. Seven controllers race on
procedurally generated tracks in a top-down 2D simulator:

- **Imitation learning:** a hand-designed baseline and three Keras Tuner searches (random search, Hyperband,
  Bayesian optimization) are trained to imitate a scripted expert driver.
- **Neuroevolution:** a genetic algorithm and a (μ+λ) evolution strategy evolve network weights by driving,
  following a curriculum from a circle to fully varied tracks. A level is passed only when the cars finish at pace.
- **Hybrid:** a seeded ES starts from the trained baseline and continues by driving.

Every method runs with three seeds; results are reported as mean ± standard deviation. The design specification
is `docs/2026-10-05-ec-racing-design.md`; the implementation plans are in `docs/`.

## Setup

The project uses the Conda base environment (Python 3.13, TensorFlow 2.21, Keras 3.13):

```bash
~/miniconda3/bin/python -m pip install keras-tuner pygame pytest nbclient nbformat
```

numpy, scipy, matplotlib, pandas, joblib and IPython come from the Conda base environment.

## Running the experiment

One command runs the six notebooks in order and saves their outputs into the notebook files:

```bash
~/miniconda3/bin/python scripts/run_notebooks.py
```

The run takes several hours. To run it in the background and follow its progress:

```bash
mkdir -p results
nohup ~/miniconda3/bin/python scripts/run_notebooks.py > results/run_console.txt 2>&1 &
tail -f results/run_log.txt
```

The console output, including any error, is kept in `results/run_console.txt`. The per-generation progress of
each evolutionary run is in `results/runs/<method>/seed<k>/log.csv`.

After an interruption, resume from a notebook, for example the GA:

```bash
~/miniconda3/bin/python scripts/run_notebooks.py --from 03
```

Run the experiment on an otherwise idle machine with a fixed power profile (`powerprofilesctl set performance`),
because training times are reported. The processor, power profile, package versions, main settings, code version
(git commit) and notebook times are recorded in `results/run_info.json`.

| Notebook | Purpose |
|---|---|
| `01_dataset` | Expert dataset |
| `02_imitation` | Baseline, Keras Tuner searches, retraining with each seed |
| `03_ga` | Genetic algorithm, three seeds |
| `04_es` | Evolution strategy, three seeds |
| `05_es_seeded` | Seeded evolution strategy, three seeds |
| `06_evaluate` | Tables and figures |

The notebooks can also be run one at a time in Jupyter or VS Code with the Conda base kernel; the working directory
must be `notebooks/`.

Results are written to `results/`, which is not under version control:

| Path | Contents |
|---|---|
| `results/dataset.npz`, `results/dataset_info.json` | Expert dataset and its collection time |
| `results/runs/<method>/seed<k>/` | Logs, summaries and per-generation best networks of each run |
| `results/runs/<method>/search/` | Keras Tuner search: trials, summary and exported models |
| `results/champions/<method>/seed<k>.keras` | Final controllers (evolutionary ones also as `.npy`) |
| `results/champions/best_seeds.json` | Each method's best seed on the validation tracks |
| `results/tables/`, `results/figures/` | Results tables (CSV and Markdown) and figures |
| `results/run_info.json`, `results/run_log.txt`, `results/run_console.txt` | Record, log and console output of the experiment run |

## Calibration and benchmarks

```bash
~/miniconda3/bin/python scripts/pilots.py          # GA mutation range, ES step size, seeded ES
~/miniconda3/bin/python -m racing.bench            # generation time per worker count; Keras epoch time
```

The pilots write to `results/pilots/`. The benchmark results set `N_JOBS`, `STEPS_PER_EXECUTION` and
`JIT_COMPILE` in `racing/config.py`.

## Viewer

```bash
~/miniconda3/bin/python -m racing.app
```

- **Training view:** choose an approach, run seed, difficulty and population size, then press Play (or Space).
  Stop ends training so the settings can be changed. The panel names the map being shown. Training started here
  writes only to `results/app_runs/`.
- **Race view:** pick a test track with the arrow keys and a difficulty; the map is shown with the cars on the
  start line. For each method, choose its best seed, a specific seed, or off, then press Start race; Stop returns
  to the start line.
- **Difficulty:** Easy (circle), Medium (oval), Hard and Extra hard (seeded loops).
- **Playback speed:** 1× is twice real time.

## Tests

```bash
~/miniconda3/bin/python -m pytest              # fast tests
~/miniconda3/bin/python -m pytest -m slow      # slow tests: every track seed, Keras Tuner, runner, end-to-end
```
