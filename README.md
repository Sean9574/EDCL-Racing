# EC Racing

EC Racing compares methods for finding small neural network driving controllers. Six controllers race on
procedurally generated tracks in a top-down 2D simulator:

- **Imitation learning:** a hand-designed baseline and three Keras Tuner searches (random search, Hyperband,
  Bayesian optimization) are trained to imitate a scripted expert driver.
- **Neuroevolution:** a genetic algorithm and a (μ+λ) evolution strategy evolve network weights by driving,
  following a curriculum from a circle to fully varied tracks.

The design specification is in `docs/2026-10-05-ec-racing-design.md`, and the implementation plan is in
`docs/2026-10-05-ec-racing-plan.md`.

## Setup

The project uses the Conda base environment (Python 3.13, TensorFlow 2.21, Keras 3.13):

```bash
~/miniconda3/bin/python -m pip install keras-tuner pygame pytest
```

## Running the experiments

Run the notebooks in `notebooks/` in order:

| Notebook | Purpose |
|---|---|
| `01_dataset` | Expert dataset |
| `02_imitation` | Baseline and Keras Tuner searches |
| `03_ga` | Genetic algorithm |
| `04_es` | Evolution strategy |
| `05_evaluate` | Tables and figures |

Results are written to `results/`, which is not under version control:

| Path | Contents |
|---|---|
| `results/dataset.npz` | Expert dataset |
| `results/runs/<approach>/` | Logs, summaries, per-generation best networks |
| `results/champions/<approach>.keras` | Final controllers |
| `results/tables/` | Results tables (CSV and Markdown) |
| `results/figures/` | Figures |

## Viewer

```bash
~/miniconda3/bin/python -m racing.app
```

- **Training view:** select an approach, set the population size, and press Play (or Space). The view shows the
  latest completed generation. Training started from the viewer writes to `results/app_runs/` and never replaces
  the final controllers.
- **Race view:** the final controllers race three laps on a held-out test track. Select the track with the arrow
  keys.

## Tests

```bash
~/miniconda3/bin/python -m pytest              # fast tests
~/miniconda3/bin/python -m pytest -m slow      # slow tests: every track seed, Keras Tuner searches, end-to-end
```

## Generation-time benchmark

```bash
~/miniconda3/bin/python -m racing.bench
```

The number of worker processes used for population evaluation is `N_JOBS` in `racing/config.py`.
