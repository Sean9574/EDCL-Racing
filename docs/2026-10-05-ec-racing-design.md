# EC Racing: Design Specification

**Evolutionary and Gradient-Based Search for Neural Network Driving Controllers**\
ECDL Term Project · Fall 2026 · Version 3.3 · 14 October 2026

## Summary

This document specifies EC Racing, a top-down two-dimensional racing simulator. It is used to compare
methods for finding small neural network driving controllers. Eight controllers compete on freshly
generated circuits whose harder levels borrow features of Formula 1 circuits.

- **Imitation learning (four controllers).** These are trained by supervised imitation of a scripted
  expert driver. One is a hand-designed baseline. In the other three, Keras Tuner selects the
  architecture using random search, Hyperband or Bayesian optimization. Each is then refined by dataset
  aggregation (DAgger) and pushed for speed by an evolution strategy grown from it.
- **Neuroevolution (three controllers).** A genetic algorithm (GA) and a (μ+λ) evolution strategy (ES)
  optimize the weights of a fixed network directly from driving performance. NEAT evolves the network
  structure together with its weights.
- **Hybrid (one controller).** A seeded ES starts from the DAgger-refined baseline network, which has
  learned from the expert, and then optimizes the weights from driving performance alone.

Every method trains until it masters the curriculum: on each freshly generated map of a generation, at
least 90% of its population must complete three laps within a target time, in three consecutive
generations, on each of four levels. In the speed push that follows, the target time is shortened in steps of 2%
whenever 75% of the population meets it, until the population can no longer keep up. Every map has its own road width,
10 to 16 m, and the networks see their position across the road and the road's width, so that they drive any width. Every method is run with three random seeds, and results are reported
as the mean and standard deviation across seeds. All final controllers are evaluated on held-out maps for
completion, lap time, classification accuracy, network size, inference time and training cost. A
visualization application shows training in progress and races between the final controllers. A single
command executes the complete experiment; it saves checkpoints, resumes after an interruption and reports
its progress live.

Version 3.3 adds a compiled simulator (§5.7): the time trials of training and the expert's validation lap run in a C++
library that reproduces the numpy simulator exactly, at about a tenth of the processor time per generation.
Version 3.2 incorporates revision 3.2: a road width of 10 to 16 m
per map, two inputs for the position across the road and the road width, a stronger speed push, a robustness table at
fixed widths, and jobs that run side by side on several machines as soon as their inputs exist.
Version 3.1 incorporates revision 3.1: every evolved car also evolves its sensor length and steering sensitivity, the application can display races under a much higher grip, and training started in the application is kept. Version 3.0 incorporates revision 3: freshly generated
circuits, time trials, the mastery criterion and speed push in place of a fixed evaluation budget, NEAT,
DAgger and the execution of multi-day runs. Version 2.0 corrected the measurement of training epochs and
inference time and added replication across seeds, the seeded ES and the run record.

## 1. Introduction

### 1.1 Research question

On tracks not seen during training, do controllers evolved through simulated driving outperform
controllers that are trained to imitate an expert and whose architectures are tuned by conventional
hyperparameter optimization?

A secondary question is whether evolution that starts from an imitation-trained controller combines
the strengths of both approaches: the expert's driving competence at the start, and the freedom to
exceed the expert's pace afterwards.

Two further questions are whether evolving the network structure together with its weights (NEAT) finds
better drivers than evolving the weights of a fixed network, and how far dataset aggregation (DAgger) and
a subsequent evolutionary speed push close the gap between imitation-trained and evolved controllers.
Every method is trained to the same mastery criterion, and the cost of reaching it is compared.

### 1.2 Alignment with course requirements

The project addresses the course project guidelines (*Suggested Term Projects*, version 1.0,
5 October 2026):

- All required models are included:
  - a baseline model
  - Keras Tuner with random search, Hyperband and Bayesian optimization
  - an ES (μ+λ) with the 1/5 success rule (the learning phase of its step-size rule, §9.7)
  - a GA
- The objectives are to maximize test accuracy and to minimize network size and inference time.
- Driving is measured by the number of held-out maps completed within the target time and by lap time.
- All networks are small, so training time stays short.
- The deliverable is a research paper. The project falls under Project 7 ("any research-oriented
  project using DL, EC, and HPO").

## 2. Scope

The following are outside the scope of this project:

- Camera-based perception, including semantic segmentation.
- Human-driven cars, menu systems and track editing. The circuit generator borrows features of Formula 1
  circuits, but no real circuit is reproduced.
- Collisions between cars.
- Recording, replay or video export of training runs.
- A user interface for browsing earlier generations.
- GPU acceleration. Preliminary measurements on the target hardware showed CPU execution to be faster
  for networks of this size.

## 3. System Architecture

### 3.1 Overview

<div>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 391" width="100%" role="img" aria-label="Figure 1: components and data flow" style="display:block;margin:10pt 0 6pt">
<defs><marker id="fig1-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker></defs>
<text x="0" y="12" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Maps</text>
<rect x="0" y="31" width="88" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="44" y="45.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">track_gen.py</text>
<rect x="106" y="31" width="70" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="141" y="45.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">tracks.py</text>
<path d="M88,42 L106,42" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<text x="0" y="65.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673">fresh training maps; fixed validation</text>
<text x="0" y="76.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673">and test maps; levels 1–4</text>
<text x="220.5" y="12" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Simulator</text>
<rect x="220.5" y="19" width="224" height="46" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<rect x="227.5" y="26" width="100" height="32" rx="2.5" fill="#ffffff" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="277.5" y="40.5" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">sim.py</text>
<text x="277.5" y="52.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">numpy, reference</text>
<rect x="337.5" y="26" width="100" height="32" rx="2.5" fill="#ffffff" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="387.5" y="40.5" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">fastsim</text>
<text x="387.5" y="52.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">compiled C++</text>
<path d="M176,42 L220.5,42" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<text x="456.5" y="34.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">fastsim gives identical results and</text>
<text x="456.5" y="45.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">runs the time trials and the expert's</text>
<text x="456.5" y="56.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">map-validation lap</text>
<rect x="0" y="89" width="320.5" height="208" rx="3" fill="#f5f6f8"/>
<rect x="344.5" y="89" width="320.5" height="208" rx="3" fill="#f5f6f8"/>
<text x="202" y="105" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="end" font-weight="bold">Imitation learning</text>
<text x="463" y="105" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Evolution</text>
<path d="M260,65 L260,89" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<path d="M405,65 L405,89" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<rect x="212" y="119" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="260" y="133.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">expert.py</text>
<text x="202" y="133.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">scripted expert driver</text>
<rect x="212" y="155" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="260" y="169.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Dataset</text>
<text x="202" y="169.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">inputs and the expert's actions</text>
<path d="M260,141 L260,155" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<rect x="212" y="191" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="260" y="205.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">imitation.py</text>
<text x="202" y="199.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">baseline; Keras Tuner random,</text>
<text x="202" y="210.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">Hyperband and Bayesian</text>
<path d="M260,177 L260,191" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<rect x="212" y="227" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="260" y="241.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">dagger.py</text>
<text x="202" y="235.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">DAgger: the expert labels</text>
<text x="202" y="246.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">the car's states</text>
<path d="M260,213 L260,227" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<rect x="212" y="263" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="260" y="277.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Speed push</text>
<text x="202" y="277.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">ES grown from the refined network</text>
<path d="M260,249 L260,263" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<rect x="357" y="115" width="298" height="30" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="133.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">evolution.py</text>
<text x="463" y="127.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673">shared run loop: time trials, survival,</text>
<text x="463" y="138.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673">levels, speed push, checkpoints</text>
<rect x="357" y="155" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="169.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">ga.py</text>
<text x="463" y="169.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">genetic algorithm</text>
<rect x="357" y="191" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="205.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">es.py</text>
<text x="463" y="205.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">(μ+λ) evolution strategy</text>
<rect x="357" y="227" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="241.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Seeded ES</text>
<text x="463" y="241.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">ES from the DAgger-refined baseline</text>
<rect x="357" y="263" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="277.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">neat.py</text>
<text x="463" y="277.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">NEAT: structure and weights</text>
<path d="M308,238 L357,238" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<rect x="234.5" y="321" width="196" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="332.5" y="335.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Final controllers</text>
<text x="332.5" y="347.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle"><tspan font-family="DejaVu Sans Mono" font-size="9">.keras</tspan>  ·  <tspan font-family="DejaVu Sans Mono" font-size="9">.npy</tspan>  ·  NEAT genome</text>
<path d="M260,285 L260,321" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<path d="M405,297 L405,321" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<rect x="42.5" y="321" width="168" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="126.5" y="335.5" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">evaluate.py, netfig.py</text>
<text x="126.5" y="347.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">tables and figures</text>
<rect x="454.5" y="321" width="168" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="538.5" y="335.5" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">app.py</text>
<text x="538.5" y="347.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">training view and race view</text>
<path d="M234.5,337 L210.5,337" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<path d="M430.5,337 L454.5,337" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#fig1-head)"/>
<line x1="0" y1="371" x2="665" y2="371" stroke="#5f6673" stroke-opacity="0.35" stroke-width="1"/>
<text x="0" y="386" font-family="DejaVu Sans" font-size="9" fill="#5f6673"><tspan font-family="DejaVu Sans Mono" font-size="9.5">checkpoint.py</tspan> and <tspan font-family="DejaVu Sans Mono" font-size="9.5">progress.py</tspan> serve every run: resumable runs and a live status line</text>
</svg>
</div>

*Figure 1. Components and data flow.*

A single simulator is shared by data collection, evolutionary training, the training view and the race
view. Its compiled implementation, which runs the time trials and the expert's validation lap, gives
identical results (§5.7). A controller therefore behaves identically in every context.

### 3.2 Module structure

```
Project/
├── README.md
├── requirements.txt    pinned package versions (§15)
├── racing/
│   ├── config.py       default parameters (§13) and color assignments
│   ├── actions.py      action table: class ↔ steering and throttle fractions
│   ├── backend.py      Keras access with GPUs hidden
│   ├── track.py        track data structure, road grid, wall-distance lookup
│   ├── track_gen.py    gentle loops, F1-inspired circuit generator, direction, geometric validation
│   ├── tracks.py       validated tracks with the expert's lap time (bounded cache)
│   ├── sim.py          vehicle dynamics, sensors, collisions, progress, fitness (reference implementation)
│   ├── fastsim.cpp     compiled simulator: time trials and the expert's lap (§5.7)
│   ├── fastsim.py      builds and calls the compiled simulator; network programs
│   ├── setup.py        evolved car setup: defaults, random draws, crossover, mutation
│   ├── nets.py         dense populations and drivers, Keras conversion, champion loading
│   ├── expert.py       scripted expert driver and dataset collection
│   ├── imitation.py    baseline, three Keras Tuner searches, model selection
│   ├── curriculum.py   fresh seeds, survival rule, mastery tracker, step-size rules
│   ├── population.py   time-trial evaluation per map, final selection, dense champions
│   ├── evolution.py    shared run loop: logging, checkpoints, status, final selection
│   ├── ga.py           genetic algorithm
│   ├── es.py           evolution strategy (also seeded ES and speed push)
│   ├── neat.py         NEAT: genomes, operators, speciation, batched forward pass, Keras export
│   ├── dagger.py       DAgger refinement and speed push of the imitation methods
│   ├── checkpoint.py   settings record, atomic checkpoints, completed-run check
│   ├── progress.py     status file, milestones, live status line, monitor
│   ├── evaluate.py     test-map evaluation, tables, figures
│   ├── netfig.py       network figure
│   ├── netwindow.py    the application's network window (separate process)
│   ├── bench.py        generation-time and Keras training-time measurement
│   └── app.py          pygame application: training view and race view
├── notebooks/          01_dataset, 02_imitation, 03_dagger, 04_ga, 05_es, 06_es_seeded, 07_neat, 08_evaluate
├── scripts/            notebook generation, experiment runner, pilot runs, PDF rendering
├── tests/
├── results/            generated data (not under version control)
└── docs/               this specification (PDF)
```

*Figure 2. Repository layout.*

- Each module has a single responsibility.
- For readability, the code uses plain loops and explicit conditionals; the one shared abstraction is the
  evolutionary method interface (§9.1), which lets every evolutionary approach use the same run loop.
- Notebooks run the reported experiments with fixed settings by calling into the `racing` package.

### 3.3 Experiment execution

The complete experiment is executed by a single command, `scripts/run_notebooks.py`. It executes the
eight notebooks in order and saves their outputs into the notebook files, so the notebooks form the
record of the reported run.

- **Order:** 01 dataset → 02 imitation → 03 DAgger and speed push of the imitation methods → 04 GA →
  05 ES → 06 seeded ES → 07 NEAT → 08 evaluation. The seeded ES depends on the DAgger-refined baseline
  from notebook 03. Notebooks run one at a time, so no two methods compete for the processor while
  their training time is measured.
- **Jobs side by side and on several machines.** One run gains little beyond 6 to 8 worker processes
  (`racing/bench.py`), so the experiment may instead run as jobs side by side,
  `scripts/run_notebooks.py --runs nb:01,nb:02,baseline:0,ga:1 --at-once 4`, and on several machines with
  `scripts/fleet.py` (plan in `scripts/fleet.toml`). A job is a run of any method, trained directly by
  `racing/train.py`, which makes exactly the call of its notebook (checked by a test), or notebook 01 or 02. Each
  job has its own status file and output log, and starts as soon as a slot is free and its inputs exist: notebook 02
  waits for the dataset, a DAgger run for its imitation network, a seeded ES run for its refined baseline; a waiting
  job holds no slot (`racing/jobs.py`, version 3.2). The notebooks later find the runs complete and only report
  them. A run's results depend only on its seed and the settings, so where it trains changes only its training time.
  Each run's summary records the machine it finished on (host, processor, worker count), and the training cost is
  compared by evaluations, which are exact on any machine; training times are reported as context, with the
  machines named. `fleet.py setup` gives a server the desktop's commit (as a git bundle), the pinned environment and
  any inputs its jobs need from the desktop; `start` starts every machine's jobs in a detached tmux session only when
  every machine is ready; `collect` copies a server's finished runs to the desktop after checking their settings and
  never replaces a different copy; `finish` then runs notebooks 03 to 08. `run_notebooks.py --archive NAME` moves the
  earlier results to `results/archive/NAME/` and exits, so the next start begins afresh.
- **Kernel:** each notebook runs in a kernel started with the same Python interpreter as the runner,
  with `notebooks/` as the working directory.
- **Checkpoints.** Every evolutionary run saves a checkpoint after each generation. It contains the
  population, the state of the random generator, the level, streak, target factor and step size, the
  seeds already used, NEAT's species and innovation records, the generation count, the number of
  evaluations and the active run time. A DAgger refinement saves a checkpoint after each round. A
  checkpoint is written to a temporary file and then renamed, so that an interruption during writing
  cannot corrupt it.
- **Resumption.** Resumption is the default behavior. When the runner is started again, completed runs
  are skipped and interrupted runs continue from their last checkpoint, losing at most one generation. A resumed run produces exactly the same results as an uninterrupted one. The option `--fresh` removes previous results and starts over. The option `--from NN` starts at a given notebook.
- **Settings check.** Checkpoints and run summaries record the settings that affect results. A run
  whose recorded settings differ from the current ones is not resumed. The runner stops with a message
  naming the difference, and `--fresh` is required. The runner checks every run record and checkpoint before it executes the first notebook, and exits with code 3.
- **Progress display:**
  - **Status file.** Every run writes its current state to `results/status.json` after each generation
    or round: the run number out of the total, the method, seed, level, generation, lowest map success
    share, streak, target reduction and elapsed time. The imitation training writes it after every epoch (epoch,
    validation accuracy and the best so far) and each Keras Tuner search after every completed trial (trial
    number, best validation accuracy), so the line is live from the second notebook on.
  - **Live line.** While the runner executes the notebooks, it displays the status as a single line that
    is updated in place, for example
    `[run 9/24] GA seed 1 · Hard · gen 412 · worst map 84% / 90% ███████████▍ · streak 1/3 · 2h 13m`.
  - **Milestones.** Level advancements, mastery goals, target reductions, completed runs, trained imitation
    networks and completed searches are printed as permanent lines and appended to `results/run_log.txt`.
  - **Quiet kernels.** The notebook kernels run with GPUs hidden (`CUDA_VISIBLE_DEVICES=-1`) and TensorFlow's
    C++ log switched off (`TF_CPP_MIN_LOG_LEVEL=3`), so TensorFlow does not search for a GPU on a machine that
    has one. Whatever a kernel process prints itself (library messages, as opposed to cell output, which is saved
    in the notebook) is appended to `results/kernel_log.txt`, so it never enters the progress display.
  - **Stopping.** Ctrl+C or `kill` stops the runner within seconds, also while a notebook prints nothing for a
    long time: it ends the notebook's kernel and its worker processes, logs the stop and exits with code 130.
  - **Runs side by side.** Each run writes its own status file in `results/status/`, and the display shows
    one live line per run, in run order.
  - **Separate monitor.** `python -m racing.progress` shows the same display from another terminal, for
    a runner started in the background.
- **Run record:** the runner writes `results/run_info.json`. It records the processor model, the number
  of logical processors, the number of worker processes, the operating-system power profile where
  available, the versions of Python, numpy, TensorFlow, Keras and keras-tuner, the settings that affect
  results (the values of §13), the code version (commit and whether the working tree had uncommitted
  changes), and the start and end time of each notebook.
- **Worker count.** The number of threads of the compiled simulator (§5.7), or of worker processes when the numpy
  simulator runs, is read from the environment variable `EC_RACING_JOBS`, with a default of 6.
  `python -m racing.bench generation` measures the generation time of both simulators for a range of counts on the
  machine at hand. The run record logs the value used and the simulator (`compiled` with the g++ version, or
  `numpy`).
- **Platform.** The experiment runs on Linux. On the Windows desktop used for the reported run, it runs
  in the Windows Subsystem for Linux (WSL 2), with the project stored in the Linux file system, and the
  application's window is shown through WSLg. Windows must be prevented from sleeping during a run.
- **Environment.** A pinned list of package versions (`requirements.txt`), matching the development
  machine, reproduces the environment.
- **Repository.** The project is hosted in a private GitHub repository, from which the desktop clones
  it.

All reported times are measured on the desktop, so that the training costs of the methods are
comparable.

The pilot runs of §12.2 (`scripts/pilots.py`) check before the reported run that the mastery requirement
can be reached. The pilots write to `results/pilots/revision3/` and are not part of the reported results.

## 4. Track Generation and Curriculum

### 4.1 Track representation

The function `make_track(seed, level)` returns a `Track` object containing:

- the centerline, resampled at 1 m intervals
- the track width and the left and right boundaries
- a road occupancy grid at 0.5 m resolution
- a distance-to-boundary map computed with `scipy.ndimage.distance_transform_edt`
- the start pose and track length
- the expert driver's lap time

The centerline is stored in driving order, so a map that runs clockwise is the
reversed point sequence of its counter-clockwise form (§4.4).

### 4.2 Curriculum levels

The evolutionary controllers train on a curriculum of four levels, from gentle loops on a wide road to narrow circuits with features of Formula 1 circuits.

Every map is generated from a seed. The four difficulty levels use two generators.

- **Easy (level 1)** uses the existing loop generator: control points at jittered angles and random
  radii, joined by a periodic cubic spline. It is set to produce gentle loops on a wide road.
- **Medium, Hard and Extra hard (levels 2–4)** use a new circuit generator (§4.3).

*Table 1. Curriculum levels and generator settings. Lengths in metres.*

| Level | Generator | Control points | Point radius | Corner radii | Features | Tightest corner | Road width |
|---|---|---|---|---|---|---|---|
| 1 Easy | Loop | 4–6 | 60–110 | Spline | None | 45 | 10–16 per map |
| 2 Medium | Circuit | 6–8 | 60–170 | Slow 25–45, fast 50–80 | None | 24 | 10–16 per map |
| 3 Hard | Circuit | 9–12 | 70–220 | Slow 16–35, fast 45–90 | Main straight ≥ 170 into a hairpin; 0–1 chicane; 0–1 run of esses | 14 | 10–16 per map |
| 4 Extra hard | Circuit | 13–17 | 80–260 | Slow 10–30, fast 45–100 | Main straight ≥ 210 into a hairpin; a second hairpin; 1–2 chicanes; 1–2 runs of esses | 9 | 10–16 per map |

**Road width (version 3.2).** Every map has its own road width, drawn uniformly from 10 to 16 m by
`default_rng([seed, level, 13])`. It depends only on the map's seed and level, so a map keeps it on every generation
attempt; the generator sizes chicanes and esses and the spacing between separate parts of the road by it. Up to
version 3.1 the widths were 16, 14, 12 and 10 m by level, and the controllers trained on them failed on wider roads.

The values in Table 1 are those of the prototype that produced the approved sample maps. They may be
adjusted during implementation so that the expert completes every level and the generator accepts a
candidate within a few attempts.

### 4.3 Circuit generator

The circuit generator builds a polygon of straights and rounds its corners with circular arcs.

1. **Base polygon.** Control points are placed at jittered angles around the origin with random radii.
   The polygon is stretched horizontally by a random factor between 1.0 and 1.5, so that elongated
   layouts with long straights occur.
2. **Main straight.** While the longest edge is shorter than the level's main-straight length, the
   vertex following that edge's end is removed, which joins two edges into one.
3. **Hairpins.** The vertex at the end of the main straight, and on Extra hard one further random
   vertex, is moved outwards by a factor between 1.25 and 1.5. This sharpens the corner into a hairpin.
4. **Chicanes.** A chicane inserts four vertices near the middle of a long edge (longer than 110 m):
   the road shifts sideways by 0.9 to 1.3 road widths over about 30–50 m and then returns.
5. **Esses.** A run of esses inserts three or four vertices along a long edge (longer than 160 m),
   offset alternately to each side by 1.2 to 2.0 road widths and spaced 30–45 m apart.
6. **Corner radii.** Each vertex is rounded with a circular arc. Hairpin and chicane vertices use a
   radius within 6 m above the level's tightest corner. Esses use slow radii. Other vertices use a slow
   or a fast radius with equal probability. A radius is reduced where the adjacent edges are too short
   to hold the arc.
7. **Acceptance.** The existing geometry check rejects a candidate whose tightest corner is below the
   level's limit or whose road comes too close to another part of itself. The expert must then
   complete the track. A rejected candidate is replaced by the next attempt for the same seed.

The features follow common Formula 1 circuit elements. A long straight into a hairpin resembles turn 1
at Bahrain or the hairpin at Montreal. Chicanes resemble those at Monza. Esses resemble the S-curves at
Suzuka and Maggotts–Becketts at Silverstone. Fast sweepers mixed with slow corners resemble Silverstone
and Spa. No real circuit is reproduced.

### 4.4 Direction

Each map runs clockwise or counter-clockwise with equal probability. The direction is drawn from the
map's seed, so a seed always produces the same map in the same direction. Revision 2.0 tracks all ran
counter-clockwise.

### 4.5 Validation

A generated map is rejected if any of the following holds:

- the road comes too close to another part of itself
- any corner is tighter than the level's tightest corner
- the expert driver cannot complete a lap

A rejected seed is retried deterministically using the triple (seed, level, attempt), so a given seed
always yields the same map. The direction is drawn from the seed alone (§4.4), so it is the same for every
attempt.

### 4.6 Seed allocation

- **Training (evolution and DAgger):** drawn by each run's random number generator from 2,000,000
  upwards, at the current level, and never repeated within a run. No training map is used twice.
- **Imitation dataset:** 121 seeds drawn once from 1,000,000–1,999,999: one at Easy, 20 at Medium, 20 at
  Hard and 80 at Extra hard.
- **Validation:** seeds 80–99, Extra hard, fixed.
- **Test:** seeds 1000–1019, Extra hard, fixed. Test maps are never used for training or model selection.
- **Application races:** the race view can also build the test seeds at the lower levels (§10.3). These
  maps serve only for display and are never used for training, selection or the reported results.

Validation and test maps remain fixed sets, so that every method is evaluated on the same maps.

## 5. Simulation Model

### 5.1 Vehicle dynamics

The vehicle is modeled as a point mass with a heading, integrated with a fixed time step of 0.05 s
(20 Hz). Its state consists of position, heading and speed.

The only physical constraint is tire grip. The model imposes no top speed, no steering-angle limit and
no fixed acceleration or braking rates. How fast to drive, when to brake and how sharply to turn must
all be learned by the controllers.

- **Controls as grip fractions:** each control requests a share of the available grip, μg = 12 m/s².
  - Steering `u_s ∈ [−1, 1]` requests a lateral acceleration `a_lat = clip(k × u_s, −1, 1) × μg`, where k is the
    car's steering sensitivity (§9.11; k = 1 for every car that does not evolve its setup).
  - Throttle `u_t ∈ [−1, 1]` requests a longitudinal acceleration `a_long = u_t × μg`. Negative
    values brake.
- **Friction circle:** if the combined request exceeds the grip limit, that is
  `sqrt(a_lat² + a_long²) > μg`, both components are scaled down proportionally so that the total
  equals μg. Braking or accelerating hard therefore leaves less grip for turning, and the reverse.
- **Motion:**
  - Speed changes by `a_long × dt` and cannot fall below zero.
  - Heading changes at the yaw rate `a_lat / v`. Below 1 m/s, a speed of 1 m/s is used in this
    calculation to avoid division by zero.
- **Consequences:**
  - The minimum turning radius is `v² / μg`, so a fast car cannot take a tight corner and must brake
    before it.
  - With no top speed, safe speed on a straight is bounded by how early the walls ahead can be seen.
    Stopping within a sensor length L requires a speed below about √(2 μg L): about 35 m/s for 50 m and 60 m/s for
    150 m.
- **Grip in the application:** the visualization application can display races with a grip of 120 m/s² (§10.5);
  training, evaluation and the expert always use μg = 12 m/s².

### 5.2 Sensors

Nine rays at −90°, −67.5°, …, +90° relative to the heading measure the distance to the track boundary, up to the
car's sensor length L: 50 m for every car that does not evolve its setup, and 10–150 m for evolved cars (§9.11). Rays
are cast by sphere tracing on the distance-to-boundary map, with a minimum step of 0.5 m. A ray of length L takes at
most ⌈48 × L / 50⌉ iterations, so a 50 m ray keeps its cap of 48 iterations. Each ray stops as soon as it reaches the
boundary or its length.

The input vector has 12 elements: the nine distances divided by 50 m, the speed divided by a reference speed of
40 m/s, the position across the road and the road width divided by 20 m. The ray scale is fixed, so a ray of a car with
L = 150 m can report values up to 3, and a car with L = 50 m sees exactly the ray inputs of version 3.0. The reference
speed only scales the input and does not limit the vehicle. The position across the road is the car's signed distance
from its matched centerline point along the road's normal, divided by half the width: −1 at the left edge, 0 on the
centerline, +1 at the right edge, relative to the driving direction. It and the width mean the same on a road of any
width, so a network can tell where it is on the road without relying on one width (version 3.2).

### 5.3 Collisions

A car crashes when its center leaves the road surface, and a crashed car stops in place. Cars do not
interact with each other in any mode.

### 5.4 Progress and termination

Progress is measured along the centerline. Each car tracks its nearest centerline point within a
window from 5 m behind to 30 m ahead. Its progress is the furthest arc length reached, so reverse
driving does not increase progress.

A run consists of three laps, in training, in the evaluation of the final controllers and in the races of
the application, so that a controller is judged on consistent driving rather than on a single lap. A run
terminates when any of the following occurs:

- the car completes three laps
- the car crashes
- the car makes no progress for 5 s
- the time limit is reached

In training and in the time-trial checks of DAgger, the time limit is the target time (§9.2). In the
evaluation of the final controllers and in the races of the application, it is twice the expert's lap time
per lap (six expert lap times in total), so that slow completions are recorded as well.

### 5.5 Fitness

The fitness of a run on a single track is:

- `p`, the completed fraction of the three laps (0 ≤ p < 1), if the run is not completed
- `1 + (1 − t_run / t_limit)`, a value between 1 and 2, if all three laps are completed, where `t_run` is the time for the three laps and `t_limit` is the time limit of the run (§5.4)

Under this definition, every completion outranks every failure, so completion separates controllers early
in training. Among completions, the faster run ranks higher.

### 5.6 Implementation properties

- **Vectorized:** every per-car quantity is a numpy array of shape `(n_cars,)`, so a single call
  advances the entire population.
- **Deterministic:** all computation is in float64, and the step function involves no randomness.
  Identical weights on an identical track always produce identical results. This lets the training
  view re-simulate exactly the runs that training scored.
- **Compiled:** the time trials of training and the expert's validation lap run in a compiled library (§5.7) that
  reproduces this numpy implementation exactly.

### 5.7 Compiled simulator

The numpy simulator advances all cars of a map together, one step at a time, with a fixed cost of about 0.3 to 0.5 ms
of interpreter and array-call overhead per step whatever the number of cars; a level-4 lap is about 1,900 steps. More
worker processes do not remove this cost, and generations run one after another. The time trials and the expert's
validation lap therefore run in a C++ library, `racing/fastsim.cpp`, called through Python's standard `ctypes` module by
`racing/fastsim.py`. The time step stays 0.05 s.

*Table 1a. Measured costs of the two simulators (development machine, Intel Core i7-12800H; level-4 maps).*

| Quantity | numpy | Compiled |
|---|---|---|
| Time per car and step (550 trained cars, 3 laps) | 5.8–6.3 µs | 1.18–1.19 µs |
| One generation, 550 cars × 3 maps × 3 laps, 6 workers or threads | 19.7 s | 1.96 s |
| One generation, 12 workers (numpy) or 20 threads (compiled) | 25.3 s | 0.96 s |
| Building one fresh map | 0.96 s, of which the expert's lap 0.81 s | about 0.15 s |

- **Build.** The library is compiled with g++ (C++17; flags `-O3 -march=native -ffp-contract=off -std=c++17 -shared
  -fPIC -pthread`) the first time it is needed on a machine. Its file name carries a build identity,
  `racing/_fastsim-<id>.so`, where `<id>` is the first 16 hexadecimal digits of the SHA-256 of the bytes of
  `fastsim.cpp`, the compiler flags and the processor's flags line in `/proc/cpuinfo` (empty where it cannot be read);
  the file is not under version control. A library of the current identity is used as it is; otherwise the library is
  built under that name, so a change of source, flags or processor always leads to a new build, and a file under that
  name that fails to load is built again. Libraries of other identities are removed once they are a day old, because
  runs that started earlier may still use them. A build writes a temporary file unique to the call and renames it
  into place, and the first use in a process builds, loads and checks the library under a lock, so runs and threads
  that start at once never load a partly written library. The library records the version of the compiler that built
  it, which the run records report with the simulator (§3.3), without running the compiler again.
- **One call per generation.** The maps of a generation, the cars' network programs and setups and each map's target
  time are passed in one call. Every pair of map and car is an independent job; `EC_RACING_JOBS` threads take the next
  unclaimed job from a shared counter, so a map whose cars finish early leaves no thread idle. Each job drives one car
  from the start line until it is done, in the order of `sim.sense`, the network's action, `sim.step` and the progress
  update; the car's own clock advances by DT per step. The returned values are those of `sim.RunResult`, and the
  fitness is computed from them by `sim.fitness`. Results do not depend on the number of threads.
- **Exact equivalence.** Every field of every `RunResult` equals the numpy simulator's (`np.array_equal`, with NaN
  finish times in the same places). Every expression of `sim.py`, `track.wall_distance`, `expert.py` and
  `actions.py` is transcribed with the same operand order in double precision; the build turns off fused
  multiply-add and does not use `-ffast-math`; `cos`, `sin`, `hypot`, `atan2` and `sqrt` come from the C library numpy
  calls on the same machine; rounding is half to even, as in `np.round`; ties go to the first largest output
  (`np.argmax`) and the first nearest centerline point (`np.argmin`); indices wrap to non-negative values (`np.mod`).
  Numpy 2.4.1 does not take the float64 `tanh` from the C library on processors with fused multiply-add: it computes
  it with its own kernel, derived from Intel's SVML, which uses fused multiply-add. The library contains a port of
  this kernel, whose operations are the only fused ones in the library; on processors without fused multiply-add,
  both numpy and the library use the C library's `tanh`. When it loads, the library is checked on the machine: its
  `tanh`, `sin`, `cos`, `sincos`, `atan2` and `hypot` must equal numpy's on fixed inputs, and the outputs of small
  dense and NEAT populations must equal those of `nets.forward` and `neat.forward`, which confirms the summation order
  of `np.einsum` and `np.bincount`. On any mismatch, one warning names the function and the numpy simulator runs
  instead. Results produced with either simulator are therefore interchangeable, and interrupted runs resume with
  either.
- **Network programs.** Every network is converted into a list of nodes evaluated in order: value slots hold the
  inputs (0–11) and then one slot per node; a node has a slot, a bias, an activation (ReLU, tanh or identity) and its
  incoming connections (source slot, weight); 15 output slots give the action order. A node's value is the sum of
  source value × weight over its connections, from zero and in order, then plus the bias, then the activation. Dense
  networks (`nets.DensePopulation`, evaluated with `np.einsum`) list each unit's inputs in index order; NEAT genomes
  (`neat.NeatPopulation`, evaluated with `np.bincount`) list nodes by depth and each node's enabled connections in
  genome order, so both sum exactly as numpy does. One format covers every architecture and activation, NEAT, and
  networks of an earlier revision that read fewer inputs.
- **Expert.** The expert (§7.1) and its quantization to the 15 actions are ported as a second kind of driver, used by
  `expert.expert_lap_time` to validate every new map.
- **Scope.** Single trained controllers (`nets.DenseNet`, `neat.NeatNet`) are evaluated with numpy's matrix product,
  whose rounding depends on the linear-algebra library's internal order and cannot be reproduced. DAgger's own driving
  (§8.4), the final races (§11), dataset collection (§7) and the application's playback (§10) therefore remain in
  numpy; they still benefit from the faster map building.

*Table 1b. Simulator used by each caller.*

| Caller | Simulator |
|---|---|
| `population.evaluate_population`: every generation of GA, ES, seeded ES and NEAT, every speed push, the final selection (§9.5) | Compiled |
| `expert.expert_lap_time`: validation of every map (`tracks.get_track`) | Compiled |
| DAgger's driving and time trials, the final races, dataset collection, the application's playback | numpy |

- **Choice and fallback.** `EC_RACING_SIM` selects the simulator: `auto` (default) uses the compiled simulator when the
  library builds and numpy otherwise; `numpy` forces numpy. If g++ is missing or the build fails, the first use prints
  one warning with the compiler's message and the run continues on numpy with identical results. Array shapes are
  checked in Python before each call.

## 6. Controller Network Interface

All controllers share the same input and output definition. This lets any controller be used in the
simulator, the training view and the race view without changes.

- **Input:** the 12-element vector defined in §5.2.
- **Output:** a 15-way softmax. Each class is one combination of a steering level and a throttle level:
  - steering fraction `u_s` ∈ {−1, −0.5, 0, +0.5, +1}
  - throttle fraction `u_t` ∈ {−1 (brake), 0 (coast), +1 (gas)}
  - `class = 3 × steer_index + throttle_index`

  The car executes the class with the highest probability.

Framing control as classification makes test accuracy directly measurable.

The module `nets.py` provides:

- the action table
- a batched numpy forward pass for whole populations, using one `einsum` per layer
- `to_keras(weights)`, which builds the equivalent Keras model and copies the weights into it

Every final controller is stored as a `.keras` file, so network size and inference time are measured
the same way for all approaches. Keras stores weights in float32. Evolutionary controllers are selected
with float64 weights, so they are additionally stored as float64 `.npy` files, and all driving
evaluations of evolutionary controllers use these files. An evolved controller also stores its setup (sensor length and steering sensitivity) in `<seed>.setup.json`; a controller without this file drives with L = 50 m and k = 1.

The baseline architecture, which the GA and ES also use, is 10 → 16 (ReLU) → 15 (softmax), for a total
of 431 trainable parameters.

## 7. Expert Driver and Dataset

### 7.1 Expert driver

- **Steering:**
  1. Pure pursuit toward a centerline point `6 m + 0.4 s × v` ahead gives a path curvature κ_p.
  2. The required lateral acceleration `v² × κ_p` is divided by μg and clipped to [−1, 1], giving a
     steering fraction.
  3. The fraction is quantized to the nearest of the five steering levels.
- **Speed:** the expert uses only track geometry within the 50 m sensor range, so it never relies on
  information the networks cannot obtain. Its target speed is the lower of two values:
  - `sqrt(0.85 × μg / κ)`, where κ is the maximum curvature within the next 50 m
  - `sqrt(2 × 0.85 × μg × 50 m)`, about 32 m/s, the highest speed from which it can stop within
    sight

  The expert selects gas when more than 0.5 m/s below the target, brake when more than 0.5 m/s above
  it, and coast otherwise. These targets govern the expert only. The networks are free to drive
  faster or slower.

The expert chooses from the same 15 actions as the networks, so every label is an action a network can
reproduce. The expert's lap time on each track sets that track's time limit and target time (§5.4, §9.2).

### 7.2 Recovery data

Pure imitation provides no examples of recovering from off-line states. To address this, at random
intervals of 2–4 s the car executes a random steering level for 0.3–0.6 s. Throughout, the recorded
label remains the expert's chosen action for the observed state.

A disturbance at full grip can make the car leave the road. In that case the car is placed back on
the centerline at its last matched centerline point, heading along the track at the expert's target
speed for that point, and collection continues. Collection on a track ends when one full lap of
progress has been covered, so every track contributes rows from its whole length rather than only
from the section before the first crash.

### 7.3 Dataset composition

Each row consists of the 12 inputs and one class label.

- **Training:** 121 freshly generated maps: one at Easy, 20 at Medium, 20 at Hard and 80 at Extra hard
  (§4.6).
- **Validation:** the 20 validation maps.
- **Test:** the 20 test maps.

The number of rows of each split is reported in the dataset table (§11.3).

Collection is distributed across worker processes, one track per task. Each track uses its own random
number generator, so the dataset is identical to a sequential collection. The collection time is
recorded and reported as a cost shared by the four imitation methods, their DAgger refinement and the seeded ES.

## 8. Imitation-Learning Controllers

### 8.1 Baseline

The baseline is the 431-parameter network from §6, trained with:

- Adam at a learning rate of 1e-3
- batch size 256
- sparse categorical cross-entropy
- at most 150 epochs, with early stopping on validation accuracy (patience 5)

The epoch limit was raised from 50 to 150 after a first run showed it to be binding: the baseline
trained for all 50 epochs, and 13 of 30 random-search trials reached their best validation accuracy in
the final epoch.

The baseline is trained with seeds 0, 1 and 2. Two Keras settings that affect training speed but not
the computed updates are fixed explicitly in `config.py` and recorded with the results:
`steps_per_execution` (the number of batches executed per call into the compiled training function)
and `jit_compile` (XLA compilation). `steps_per_execution` is chosen with the Keras training-time benchmark in
`bench.py`. XLA compilation is switched off: it trained about 11% faster per epoch, but on the desktop used for the
reported run its compiler failed after about 130 Keras Tuner trials in one process ("Failed to materialize symbols")
and kept failing, which stopped the search.

### 8.2 Hyperparameter search

Random search, Hyperband and Bayesian optimization all search the same space.

*Table 2. Keras Tuner search space.*

| Hyperparameter | Range |
|---|---|
| Hidden layers | 1–3 |
| Units per layer | {4, 8, 16, 32, 64} |
| Activation | {ReLU, tanh} |
| Dropout | {0.0, 0.1, 0.2} |
| Learning rate | Log-uniform, 1e-4 to 1e-2 |

The objective is validation accuracy, and training settings match the baseline.

**Search budget:**

- Random search and Bayesian optimization: 30 trials each, at most 4,500 epochs in total.
- Bayesian optimization begins with 10 randomly sampled trials and then proposes the remaining 20
  from its Gaussian-process model. The keras-tuner default (three times the number of
  hyperparameters, here 18–21 trials) would leave only about a third of the budget model-guided.
- Hyperband: `max_epochs = 150`, `factor = 3` and one iteration. One iteration consists of five
  brackets with 254 trials on 172 distinct configurations and at most 3,900 epochs, the closest match
  to the other two methods that whole iterations allow.

Each search runs once, with seed 0.

**Epoch accounting:** the epochs of a trial are the epochs it actually trained, counted from its
training history after early stopping. A Hyperband trial that continues an earlier trial counts only
the epochs it added. keras-tuner records a single observation per trial (its best epoch), so the
count cannot be taken from the tuner's stored metrics.

The number of trials, the number of distinct configurations, total epochs and wall-clock time are
reported for every method.

### 8.3 Model selection

For each method, the selected architecture is the smallest network (fewest parameters) whose
validation accuracy is within one percentage point of the best validation accuracy that method
achieved. This rule expresses both course objectives, accuracy and size, as a single criterion.

The selected architecture, with its searched hyperparameters, is then trained from scratch with seeds
0, 1 and 2. These three networks are the method's controllers after initial training. DAgger (§8.4) and the speed push (§8.5) produce its final controllers. The search therefore runs once
per method, and the variation across seeds measures training variability for a fixed architecture,
not the variability of the search itself.

**Driving-based selection (secondary analysis):** imitation methods are selected on validation
accuracy, whereas evolutionary methods are selected on validation driving. To show how much this
asymmetry matters, the most-trained model of every distinct configuration in each search drives three
laps on each of the 20 validation tracks. The model with the highest mean validation fitness is
reported for each method as a driving-selected controller in a secondary driving table.

### 8.4 DAgger refinement

Dataset aggregation (DAgger) corrects the main weakness of imitation: the trained car reaches
situations the expert never showed it and does not know how to recover. Each trained network (four
methods, three seeds) is refined in rounds:

1. **Drive.** The car drives 20 fresh Extra hard maps for three laps each. After a crash it is placed
   back on the centerline, as during dataset collection, and continues, so that the whole lap is
   covered.
2. **Label.** At every step, the car's sensor readings are recorded together with the action the expert
   would take in the car's exact state.
3. **Train.** The new rows are added to the network's dataset, which keeps the rows of all previous
   rounds. Training continues from the current weights with the method's settings, for at most 30
   epochs with early stopping (patience 5) on a 10% hold-out of the aggregated rows.
4. **Check.** The car drives 20 further fresh Extra hard maps as a time trial at f = 1.3.

Refinement reaches its goal when at least 18 of 20 checks succeed in three consecutive rounds. If 10
rounds pass without a new best check result, refinement stops and the best network so far is passed
on.

### 8.5 Speed push

DAgger can make a car no better than its teacher. After refinement, an evolution strategy population of
550 is grown from the refined network, as copies perturbed with noise of standard deviation 0.1. It
starts at Extra hard at f = 1.3 and follows the ES of §9.7 and the rules of §9.2–9.5: it must first reach
the mastery goal, then the target time is shortened until the population can no longer meet it. If
DAgger stopped without reaching its goal, the population must first reach the mastery goal by
evolution.

### 8.6 Stages

Each imitation method is evaluated at three stages: after initial training, after DAgger and after the
speed push. The controller after the speed push is the method's final controller.

## 9. Evolutionary Controllers

### 9.1 Common procedure

All evolutionary approaches share one run loop: the GA, the ES, the seeded ES, NEAT and the speed push
of the imitation methods (§8.5). The GA and the ES use the baseline architecture (431 weights) with
Glorot-uniform initial weights. The seeded ES and the speed pushes start from trained networks. NEAT
starts from minimal networks and grows them. Every car of these populations also carries an evolved setup (§9.11). Each approach is run with seeds 0, 1 and 2. Runs have no
evaluation budget; they end through the speed push or the safety stop (§9.4).

**Each generation proceeds as follows:**

1. Three maps are freshly generated at the current level from new seeds drawn by the run's random
   number generator (§4.6).
2. Every car drives each map as a three-lap time trial (§9.2). Kept cars are re-evaluated, because the
   maps change every generation.
3. A car's fitness is its mean over the three maps. The success share of a map is the fraction of the
   whole population that succeeded on it.
4. The survival rule (§9.3) decides which cars are kept, and the method's operators (§9.6–9.9) produce
   the children that fill the rest of the population.
5. The level tracker (§9.4) records the lowest map success share.

**Logging:** each run writes to its own directory, `results/runs/<approach>/seed<k>/`, one CSV row per
generation with the generation and level, the target factor, the number of evaluations, the best, mean
and worst fitness, the success share of each map and the lowest of them, the number of kept cars, the
success shares of kept cars and children, the median lap time of the successful cars, the step size
(GA mutation strength or σ), the streak, the event of the generation (level advancement, mastery goal,
target reduction or end), the map seeds and the elapsed time.  Every run also logs the mean and standard deviation of the population's sensor length and steering sensitivity, and NEAT logs its number of species. A
checkpoint is written after every generation (§3.3).

**Run time:** the reported time is the active run time, accumulated across resumptions, including the
final controller selection. Each run starts with new worker processes, so every run includes the same
cost of generating its maps.

### 9.2 Time trial

Every drive in training and evaluation is a time trial of three laps.

- **Reference time.** A map's reference time is the expert's time for three laps. It serves only as a
  measure of the map's length and difficulty, so that one target factor applies to maps of any size.
- **Target time.** The target time is f × reference time. The target factor f is 1.3 while a
  population climbs the levels, the factor within which an expert-like lap time is comfortably reachable.
- **Success.** A car succeeds on a map if it completes all three laps within the target time.
- **Laps in time.** For a car that does not succeed, the laps it completed before it crashed, stalled or
  reached the target time are counted.
- **Simulation end.** A training drive ends at the target time. The five-second no-progress timeout is
  unchanged.
- **Fitness.** Fitness is based on the completed fraction of laps, with the target time as the time limit: the
  completed fraction of the three laps for a car that does not succeed, and 1 + (1 − finish time /
  target time) for a car that does. Every success therefore outranks every failure, and among
  successes the faster car ranks higher.

### 9.3 Survival

All evolutionary methods (GA, ES, seeded ES, NEAT and the imitation speed push) share one survival
rule. It decides which cars of a generation are kept unchanged in the next one.

1. A car that succeeds on every map of the generation is kept.
2. A car that does not is kept with probability equal to its share of laps completed in time: laps in
   time summed over the generation's maps, divided by three laps times the number of maps. For example,
   seven of nine laps gives a 78% chance of being kept.
3. Kept cars fill at most 90% of the population. Successes are taken first, fittest first, and then the
   cars kept by chance, fittest first.
4. If fewer than 50 cars are kept, the fittest remaining cars are added until 50 are kept. This
   elitism floor ensures that the best cars are not lost early in training, when almost no car
   completes a lap.

The remainder of the population consists of new children, produced by the method's own operators
(§9.6–9.9). Because every generation drives new maps, a car remains in the population only while it keeps
succeeding on maps it has never seen.

### 9.4 Level advancement, speed push and safety stop

**Level advancement.** For each map of a generation, the success share is the fraction of the whole population that
succeeded on that map. A generation passes when the lowest success share among its maps is at least
90%. Three consecutive passing generations advance the population one level. A population on Extra
hard that passes three consecutive generations has reached the mastery goal. Each generation drives
three maps, so the three passing generations cover nine different maps.

**Speed push.** After the mastery goal, the target factor is multiplied by 0.98, which shortens the target time by 2%.
In the speed push a generation passes when the lowest map success share is at least 75%; each further run of three
consecutive passing generations shortens the target by another 2%. The run ends when 300 generations pass without
mastering the current target. The tightest target mastered is recorded as the run's speed result, expressed as the
reduction from the starting target. For example, f = 1.3 × 0.98⁴ is reported as a target 7.8% shorter than the
starting target. Up to version 3.1 the step was 5%, a speed-push generation needed 90% and the patience was 500
generations; every run then ended at a target that 90% of the population could not reach (version 3.2).

**Safety stop.** On any level before the mastery goal, a run ends if 2,000 generations pass without a new best value of
the lowest map success share, averaged over three consecutive generations. Such a run is reported as
not having reached the level. Runs are expected to end through the speed push; the safety stop
prevents a run that cannot progress from continuing indefinitely.

### 9.5 Final controller

When a run ends, every kept car of the last generation drives the 20 validation maps as a time trial
at the tightest target the run mastered (f = 1.3 if it mastered none). The car with the most
successes is the run's final controller. Ties are broken by mean fitness, which favors the faster car.
Revision 2.0 selected by mean fitness alone, which allowed a fast car that crashed on one map to win
over a car that completed every map. The final controller is exported as a `.keras`
file at `results/champions/<approach>/seed<k>`, together with its float64 weights (`.npy`) for dense
networks or its genome (`.json`) for NEAT.

### 9.6 Genetic algorithm

The population size is 1,000. Children are produced as follows: two parents are selected by
tournaments of size 5 over the whole population, combined by uniform crossover (p = 0.5) and mutated
(each weight with p = 0.1, by a Gaussian of strength s). The strength follows the rule
s = s_min + (s_max − s_min)(1 − c), where c is the share of the previous generation's children that
succeeded on every map, s_max = 0.1 and s_min = s_max / 100. Fewer successful children raise the
strength and more lower it.

- **Each remaining new individual is created in three steps:**
  1. **Selection:** two parents are each chosen by tournament selection, with a tournament size of 5
     drawn from the whole population.
  2. **Crossover:** uniform crossover, where each weight is inherited from either parent with
     probability 0.5.
  3. **Mutation:** each weight, with probability 0.1, receives an added sample from N(0, s).

### 9.7 Evolution strategy

The population size is 550. Each child is a copy of one of the 50 fittest cars of the generation, chosen
at random, with Gaussian noise of standard deviation σ added to every weight. σ starts at 0.1 and
follows two rules:

- **Learning phase.** While fewer than 90% of the 50 fittest cars succeed on every map, σ follows the 1/5 success rule (σ is divided by 0.82 when more than a fifth of the children outperform their parent and multiplied by it when fewer do), with a tie counted as half a success.
- **Reliability phase.** Once at least 90% of the 50 fittest succeed, σ is multiplied by 0.82 when fewer
  than 90% of the children succeed and divided by 0.82 otherwise.

σ is kept within [1e-4, 1.0] and reset to 0.1 when the population advances a level. It is not reset
when the target time is shortened.

- **Ties:** fitness is deterministic and the actions are discrete, so on a fitness plateau many
  offspring score exactly the same as their parent, for example when every parent crashes before
  covering any distance and scores 0. If ties counted as failures, `p_s` would be close to zero on
  such a plateau, and σ would shrink to its lower bound, where the search can no longer leave the
  plateau. Counting a tie as half a success gives `p_s` = 0.5 on a complete plateau, so σ grows. With
  continuously varying fitness, ties are rare and the rule is unchanged.

The factor 0.82 follows Schwefel.

### 9.8 Seeded evolution strategy

The seeded ES is the ES of §9.7 with an initial population of copies of the baseline network after its
DAgger refinement (§8.4), each perturbed with noise of standard deviation 0.1. Like every evolutionary
run, it starts at Easy. The baseline's own speed push (§8.5) starts from the same network but at Extra
hard. The two therefore differ only in whether the curriculum is repeated, and the comparison shows
whether it helps a network that has already learned to drive.

- **Interpretation:** the seeded ES uses the expert indirectly, through the DAgger-refined baseline. Its training
  time is reported separately from the dataset collection and baseline training on which it depends.

### 9.9 NEAT

NEAT (neuroevolution of augmenting topologies) evolves the network structure together with its
weights. The implementation is written in numpy for this project, so that whole populations drive
together like the other methods.

**Genome.** A genome consists of node genes and connection genes. Every genome has the 10 input nodes
and the 15 output nodes of the common interface. Hidden nodes are added by mutation. Each node except
the inputs carries a bias. Each connection gene holds its source and target nodes, a weight, an enabled
flag and an innovation number. The run keeps a record of structural mutations: a connection between
the same two nodes, or a split of the same connection, receives the same innovation number or node in
every genome in which it occurs, so that the genes of two parents can be aligned. Networks are feed-forward. A connection that would create a cycle is never added, so NEAT
networks, like all others, have no memory.

**Initial population.** Every input is connected to every output: 150 connections and 15 output biases.
Weights are drawn from a standard normal distribution and biases start at zero.

**Children.** The number of children allotted to each species is proportional to the mean fitness of
its members (see Speciation below). A child is produced by crossover with probability 0.75 and otherwise
copied from a single parent. Parents are chosen by tournaments of size 3 within the species. In
crossover, matching genes are inherited from a random parent and the remaining genes from the fitter
parent. A gene disabled in either parent is disabled in the child with probability 0.75. Every child is
then mutated:

- Each weight and bias is perturbed with probability 0.8 by Gaussian noise of standard deviation σ, or
  replaced with probability 0.05 by a standard normal value.
- With probability 0.05, a connection is added between two unconnected nodes, provided it creates no
  cycle.
- With probability 0.03, an enabled connection is split by a new hidden node. The incoming connection
  receives weight 1 and the outgoing connection the old weight, and the old connection is disabled.
- With probability 0.01, the enabled flag of a random connection is toggled.

σ starts at 0.5, which suits weights drawn from a standard normal distribution, and follows the
two-phase rule of the ES (§9.7), evaluated on the 50 fittest cars of the population. It is reset to 0.5
when the population advances a level.

**Speciation.** Genomes are grouped into species by the compatibility distance
δ = E / N + D / N + 0.4 W̄, where E and D are the numbers of excess and disjoint genes, W̄ is the mean
weight difference of matching genes and N is the number of genes in the larger genome (1 if both have
fewer than 20). A genome joins the first species whose representative, a random member from the
previous generation, lies within the threshold δ_t. Otherwise it founds a new species. δ_t starts at
3.0 and is adjusted by 0.1 each generation towards 20 species. A species whose best fitness has not
improved for 20 generations receives no children, unless it contains the best car of the population.
Because the maps change every generation, a species' best fitness is a record of noisy scores, and after
a level-up or a shorter target the old records are out of reach. Every species' best fitness and
stagnation count therefore restart when the population advances a level, reaches the mastery goal or
masters a shorter target, so stagnation is measured within one level and target.
Kept cars (§9.3) remain in the population regardless of their species.

**Activation.** Hidden nodes use tanh. Output nodes are linear, and the action is the class with the
highest output, as for every other method.

**Batched forward pass.** Before a generation drives, all genomes are compiled into flat arrays. Each
node's depth is its longest path from an input. The enabled connections of the whole population are
grouped by the depth of their target node. At each time step, the nodes are computed depth by depth,
each depth with one weighted sum over all its connections (`numpy.bincount`) and one activation. The
cost per step grows with the number of connections and the depth, not with the number of cars, and
there is no loop over cars in Python.

**Saved controller.** The final controller is saved as its genome (JSON) and as an equivalent Keras
model. The Keras model has one dense layer per depth, applied to the concatenation of the inputs and
all shallower nodes, with zero weights where the genome has no connection. A test verifies that the
Keras model and the genome choose the same actions. The Keras model serves only for the accuracy and
timing columns of the course comparison (§11.3). The parameter count reported for NEAT is the number of
enabled connections plus biases, not the size of the Keras layers.

The population size is 1,000, like the GA.

### 9.10 Computational performance

The simulation is vectorized over individuals, and rays use sphere tracing (§5.2).

- **Work distribution:** with the compiled simulator (§5.7), a generation is one call in which threads drive the
  pairs of map and car. With the numpy simulator, the evaluation is divided into tasks, each covering one of the
  generation's three tracks for one share of the population, distributed across worker processes (joblib); each
  numpy step has a fixed cost of about 0.45 ms plus about 2.3 µs per car, so a task should cover as many cars as
  possible on one track. Results are independent of the division and of the simulator, because cars do not interact.
- **Number of threads or workers:** chosen with the generation-time benchmark in `bench.py`, which times the same
  evaluation path the training runs use, for both simulators. With the compiled simulator, it times trained cars when
  a checkpoint of a dense population exists in `results/runs`: the cars of the newest such checkpoint, repeated or
  truncated to the population size, drive three laps of its last three level-4 maps against the starting target time
  (factor 1.3), as in training. Without such a checkpoint, it times expert cars, which neither sense nor run a network
  and are therefore a lighter load; its output names the workload timed.
- **Keras training:** `steps_per_execution` (§8.1) is chosen with the Keras training-time benchmark in
  `bench.py`; XLA compilation is off for reliability (§8.1).
- **Fresh maps:** each generation builds its three maps in the process that drives them: the calling process with the
  compiled simulator, the worker processes with numpy. With the compiled expert lap (§5.7), a map takes about 0.15 s
  to build. Every process keeps at most 8 built tracks, so memory stays constant over runs of many thousands of generations.
- **Sensor length:** tracing time grows with a ray's length, so a population that evolves long sensors drives each
  generation more slowly. Results do not depend on it.
- **NEAT:** all genomes of a generation are compiled into flat arrays and evaluated together, depth by
  depth (§9.9), so NEAT drives its whole population at once like the other methods.
- **Further measures**, applied only if training is still too slow: fewer sphere-tracing steps, then
  shorter time limits.

GPU execution is not used: the per-operation launch cost on a GPU exceeds the arithmetic of networks
and car batches of this size.

### 9.11 Evolved car setup

Every car of an evolutionary population carries two setup values next to its network: its **sensor length** L
(10–150 m) and its **steering sensitivity** k (⅓–3). Both are stored and mutated on a logarithmic scale, so halving
and doubling are equally large changes. The setup carries no cost; it is judged only by the time trials, through
success and lap time.

- **Sensor length** limits all nine rays of the car (§5.2).
- **Steering sensitivity** multiplies the steering of every action class before the grip limit (§5.1). Below 1 it
  gives finer steering but a gentler sharpest turn; above 1 it cannot exceed the grip limit, strengthens the
  half-steering levels and saturates the full ones, so the steering becomes coarser.
- **Populations:** the GA, the ES, NEAT, the seeded ES and the speed push of each imitation method evolve their setup.
  The imitation training and DAgger use L = 50 m and k = 1, because the expert data and corrections are recorded with
  those sensors.
- **Initial setups:** populations that start from random networks (GA, ES, NEAT) draw every car's setup uniformly on
  the logarithmic scale within the ranges; populations that start from a trained network (seeded ES, speed pushes)
  start with L = 50 m and k = 1, the setup the network was trained with.
- **Inheritance and mutation**, the same for every method: a kept car keeps its setup; a child inherits each value
  from a parent (GA: from either crossover parent with equal probability; ES: from its parent; NEAT: from the fitter
  parent of a crossover or from the single parent); each inherited value is mutated with probability 0.2 by Gaussian
  noise with standard deviation 0.1 on the natural-logarithm scale (about ±10%) and clipped to its range. Setup
  mutation is independent of the network step sizes.
- **Saved controllers** store their setup (§6). The evaluation, the network figure and the application drive every
  controller with its own setup.

## 10. Visualization Application

The application is a single pygame window with one top-down camera and two modes. All cars are driven
by controllers, and the application has no manual driving.

### 10.1 Color assignment

Colors are drawn from the Okabe–Ito palette, which stays distinguishable under common forms of color
blindness. Each approach keeps the same color in the application, legends and figures.

*Table 3. Color assignment.*

| Approach | Color |
|---|---|
| Baseline | Gray `#7F7F7F` |
| Keras Tuner Random | Orange `#E69F00` |
| Keras Tuner Hyperband | Sky blue `#56B4E9` |
| Keras Tuner Bayesian | Reddish purple `#CC79A7` |
| ES | Bluish green `#009E73` |
| GA | Vermillion `#D55E00` |
| Seeded ES | Blue `#0072B2` |
| NEAT | Yellow `#F0E442` |

Interface elements that do not represent an approach, such as the highlight of a selected button, use a
dark neutral color so that they cannot be mistaken for an approach.

### 10.2 Training view

Training is observed live rather than recorded.

- **Controls:**
  - The user selects an approach, a run seed and a difficulty (§10.4).
  - For the GA, ES, seeded ES and NEAT, the user sets the population size before starting.
  - Training is started and paused with a play/pause control and ended with a stop control, which freezes the displayed cars. After a stop, the approach, run seed, difficulty and population size can be changed.
  - Changing a setting is possible only while no training is running.
- **Training process:**
  - Training runs in a separate process, with the configured number of worker processes, in step with the view:
    after each generation, trial or epoch it waits until the view has replayed it, so the generation on screen is
    always the one just trained and every generation is shown. While the race view is shown, nobody watches the
    training, so it runs at full speed. The training process is not a daemon process, because joblib cannot start worker
    processes from a daemon process; it is terminated explicitly when training stops or the
    application closes.
  - After each generation, completed Keras Tuner trial or baseline epoch, it sends the population, fitness values, map success shares, level, target factor and map seeds to the view through a `multiprocessing.Queue`.
  - When the training process ends, the view reads the queue once more, so the final report is
    displayed.
  - Pausing freezes the replay and halts training at the next generation boundary.
  - Training started from the application writes only to `results/app_runs/`, never to the reported results, and is
    kept there (§10.6). The seeded ES in the application starts from the DAgger-refined baseline of the selected run
    seed. If it does not exist, the seeded ES cannot be started and the view states that notebook 03 must be run
    first.
- **Playback:**
  - The view re-simulates only the displayed individuals, on the generation's first track, at 1×, 2×,
    4× or 8× playback speed. At 1×, simulated time runs at twice real time.
  - Tracks are built in a background thread. While a new track is being built, the window stays
    responsive and playback starts once the track is ready.
  - Each generation is replayed once. When its playback ends, training continues and the view holds the finished
    replay, stating which generation is being trained, until the next report arrives.
  - The playback speed therefore sets the pace of training started in the application; the reported runs
    (notebooks) never wait for a view.
- **Notebook runs:** a source selector switches the training view between training started in the application
  and a reported run (`results/runs/`) of the selected approach and run seed.
  - After each generation, a reported evolutionary run (and the speed push of an imitation approach) keeps its
    latest evaluated generation (population, fitness, maps, level and statistics) in `replay.pkl` next to its
    checkpoint, overwritten every generation. Training never reads this file.
  - The view replays that generation with the same display selection, panel and plot (from the run's log). When a
    replay ends, the view loads a newer generation if one was written (looking at most once a second) and otherwise
    replays the same one, so a running experiment is followed live. Generations in between are skipped: the view
    never slows the run.
  - The view is read-only: it never writes to the run, starts or pauses it. Play/Pause pauses only the replay; stop,
    new run, difficulty and population size do not apply. A run without a replay file (started before this
    feature), a missing run and an unreadable file are named in the status line.
- **Track display:** the side panel shows the seed and level of the track being played back. The
  imitation views always use validation track 80, so its map is shown before training starts. The evolutionary approaches draw new maps every generation, so their map appears with the first
  generation.
- **Display selection:** All, Best or Worst, with a displayed-car count N from 1 to the population size:
  - All shows a fixed random sample of N individuals.
  - Best shows the N fittest.
  - Worst shows the N least fit.

  The fittest displayed individual is outlined, and its sensor rays can be shown, each drawn to the car's own sensor length.
- **Camera and window:** the camera shows the whole map or follows a car at 2×, 4× or 8× (the outlined car in
  the training view, the race leader in the race view; button, mouse wheel or C key); a zoomed view draws only
  the part of the track on screen. The window can be resized, or switched to full screen with F11 or the Full
  screen button (Esc leaves it); the map and panel follow its size.
- **Side panel:**
  - approach, level and generation
  - the success share of each of the generation's maps against the 90% requirement, and the streak
  - the target reduction during the speed push
  - mutation strength (GA) or σ (ES, NEAT), and the number of species (NEAT)
  - the population's mean sensor length and steering sensitivity, with their standard deviations
  - best and mean fitness
  - a live plot of the lowest map success share per generation against the 90% line, with level
    advancements, the mastery goal and target reductions marked (for the imitation approaches, the best and
    mean validation accuracy)
- **Keras Tuner approaches:** each completed trial is treated as one individual and is driven on
  validation track 80.

### 10.3 Race view

- **Controllers:** the view loads whichever final controllers exist in `results/champions/`. Each
  approach has a selector that cycles through *best*, *seed 0*, *seed 1*, *seed 2* and *off*. *Best* is the seed whose controller succeeded on the most validation maps, ties broken by mean validation fitness, as recorded by the
  evaluation (§11); *off* removes the approach from the race. The default is *best*.
- **Track:** a test seed (1000–1019) is selected with the arrow keys or on-screen buttons, and a level
  with the difficulty control (§10.4). The selected map is drawn as soon as it is selected, with the
  cars on the start line, and its seed and level are shown.
- **Race:** the selected controllers race three laps without interacting, each in its assigned color.
  A start control begins the race, and a stop control ends it and returns to the map preview, after
  which the controllers, seed and difficulty can be changed.
- **Leaderboard:**
  - position, ranked by laps completed plus progress
  - last lap time and best lap time
  - crash status

  A crashed car stays where it crashed and is drawn faded.
- **Playback speed:** 1×, 2×, 4× or 8×, where 1× is twice real time.
- **App cars:** a cycle button switches the controller list between the reported controllers and the app cars, the
  best cars of training started in the application (§10.6), labelled for example *GA (app) s1*. Each app car is off by
  default and can be switched on to race next to the reported controllers, drawn with a white border; app cars never
  enter the evaluation. When many cars race, the leaderboard shows as many rows as fit and states how many more there
  are.

### 10.4 Difficulty

The difficulty control maps four settings to the curriculum levels of §4.2.

*Table 4. Difficulty settings.*

| Difficulty | Level | Track |
|---|---|---|
| Easy | 1 | Gentle loop on a 16 m road |
| Medium | 2 | Circuit on a 14 m road |
| Hard | 3 | Circuit with a main straight into a hairpin, possibly a chicane or esses, on a 12 m road |
| Extra hard | 4 | Circuit with two hairpins, chicanes, esses and fast sweepers, on a 10 m road |

In the race view, the difficulty sets the level of the race track. In the training view, it sets the
level at which evolutionary training starts (the curriculum still advances from there) and the level of
the track on which imitation trials and epochs are driven. The difficulty applies only to the
application; the reported experiment always starts at level 1 and evaluates on level 4.

### 10.5 Grip selector

A selector with the options **Normal** (the training grip, 12 m/s²) and **Unlimited** (10 times the training grip,
120 m/s²) sets the grip of everything the application simulates for display: the race view and the replays in the
training view. It never affects training, which always uses the training grip, so the reported results cannot depend
on it. A grip of 120 m/s² stands in for unlimited grip, because an unbounded grip would let a car turn arbitrarily far
within one time step.

### 10.6 Saved application training

Training started in the application is kept, separately from the reported results.

- **One saved run per method and seed** in `results/app_runs/<method>/seed<k>/`, checkpointed after every generation
  like a reported run. The reported results never read these folders, and `--fresh` keeps them.
- **Resume.** For the GA, ES, seeded ES and NEAT, Play continues the saved run of the selected method and seed where it
  stopped, at its own level and target; the difficulty and population settings apply only when a run starts. **New
  run** deletes that method and seed's saved run and its app car and starts afresh with the current settings. If a
  saved run was recorded with other settings or a file of it is damaged, the training process ends with exit code 3
  and the panel asks for New run.
- **Imitation methods** (baseline and Keras Tuner) are not resumed in the application: Play starts them afresh.
- **App cars.** After every generation (evolutionary methods), or whenever the validation accuracy improves
  (imitation methods), the run's best car so far is saved with its setup in `results/app_runs/champions/`, in the
  files of a final controller, so it can race in the race view (§10.3).

### 10.7 Network window

A separate window, opened from the side panel, draws a car's network in the layout of the network figure (§11.4). In
the training view it shows the best car of the latest generation (for imitation, the network of the latest epoch or
trial) and redraws with every report; in the race view a selector cycles through the cars in the race. The window runs
in its own process, can be moved and resized, and closes with the application.

## 11. Evaluation Methodology

### 11.1 Test procedure

Each final controller (eight methods, three seeds) drives the 20 test maps as a three-lap time trial.
Two outcomes are recorded per map: whether the car **finished** within the simulator's time limit of
twice the reference lap time per lap, and whether it **succeeded** within the starting target (f = 1.3).
The validation maps select each run's final controller (§9.5) and each method's best seed for the
application. Every map is driven at its own road width (§4.2).

**Robustness (version 3.2).** Each test map is also rebuilt with the same centerline at the fixed widths 8, 10, 14 and
18 m, the first and last outside the trained range of 10–16 m. Each build is driven exactly as a test map, with the
expert's lap time on that build; a build whose road would reach another part of the map, or that the expert cannot
drive, is left out of its width.

### 11.2 Measurements of the course comparison

- **Accuracy evaluation:** each controller's predicted classes are compared with the expert labels in the
  test dataset. For the evolutionary controllers, this measures agreement with an expert they never
  observed, which is reported as a result in its own right.
- **Inference time:** *batch*, Keras `predict` on 10,000 test rows in a single batch, reported per sample
  and averaged over five runs, with a warm-up call identical to the timed calls; *single sample*, the
  median time of 1,000 calls of the Keras model on one row. Inference is timed after all parallel work has
  finished, with the same prediction settings for every controller (no XLA compilation, one batch per
  call).
- **Parameters:** the Keras parameter count; for NEAT, the number of enabled connections plus biases.
- **Training time:** baseline training per seed; Keras Tuner search time (once per method) and training
  per seed; DAgger and speed push per seed; evolutionary run time per seed; dataset collection once, as a
  shared cost. The run conditions recorded in `results/run_info.json` (§3.3) are reported with the
  results.

### 11.3 Tables

Results are reported as the mean and standard deviation across seeds, and are led by completion and
speed.

1. **Results.** Per method: test maps finished and succeeded (per seed and in total, of 60), mean and
   best lap time, mean three-lap race time, crashes, and the target reduction reached in the speed push, and the final controllers' sensor length and steering sensitivity.
2. **Head-to-head speed.** Mean lap time on the test maps that every final controller finished.
3. **Training cost.** Per method and seed: time and evaluations to clear each level, to reach the
   mastery goal and to end the speed push, and whether each was reached. For imitation methods, the
   Keras Tuner search time and the number and time of DAgger rounds are included.
4. **Imitation stages.** Test results of the imitation methods after initial training, after DAgger and
   after the speed push.
5. **Course comparison.** Test accuracy, parameters, batch inference time, single-sample latency and
   search and training time, as required by the course. Test accuracy is the share of rows of the expert-labelled test split for
   which the controller predicts the expert's class. Parameters are as in §11.2. Batch inference time is the
   Keras `predict` time per sample on 10,000 test rows, averaged over five runs; single-sample latency is the
   median of 1,000 calls on one row (§11.2). Search and training time is the training time of §11.2.
6. **Search cost, dataset and driving-selected controllers.** *Search cost:* per Keras Tuner method, the
   number of trials, the number of distinct configurations, the total epochs, the trials that reached the
   epoch limit and the search time (§8.2). *Dataset:* the rows of each split and the collection time (§7.3).
   *Driving-selected controllers:* per Keras Tuner method, the configuration chosen by validation driving
   (§8.3), its parameters, validation accuracy, test maps completed (of 20), mean lap time and crashes.
7. **Robustness.** Per method and width of §11.1: the maps finished and succeeded (all seeds) and the mean lap time on
   finished maps; each width states how many maps it covers.

### 11.4 Figures

- **Success over training.** For each method, the lowest map success share per generation, with
  level advancements and target reductions marked.
- **Lap time over training.** The median lap time of the successful cars per generation.
- **Setup over training.** For each method, the population's mean sensor length and mean steering sensitivity per
  generation, with level advancements and target reductions marked.
- **Networks.** One panel per method showing its best final controller. The nine distance rays,
  labelled by angle, and the speed input are on the left. Hidden nodes are placed in columns by depth,
  so that the structure evolved by NEAT appears as it is. The 15 actions, labelled by steering and
  throttle, are on the right. Connections are blue when positive and red when negative, with width and
  opacity proportional to the weight's magnitude. Titles include the controller's setup.
- **Step sizes.** GA mutation strength and ES step size σ across generations, one line per seed.
- **Keras Tuner trials.** Validation accuracy against parameter count for every Keras Tuner trial, with each
  method's selected model marked.

## 12. Verification

### 12.1 Tests

Automated tests use pytest.

- **Simulation:**
  - zero steering produces straight-line motion
  - combined steering and throttle never exceed the grip limit
  - leaving the road registers a crash
  - a completed lap is detected
  - reverse driving earns no progress
  - identical inputs produce identical outputs
- **Sensors:** ray distances match analytically computed values for a car on a circular test track.
- **Compiled simulator (§5.7):** every `RunResult` field equals the numpy simulator's (`np.array_equal`) for random
  dense populations of several architectures and activations with per-car setups, NEAT populations after node and
  connection mutations including disabled connections, a network that reads fewer inputs, the expert's lap on maps of
  every level, and a generation of several maps in one call with one and with several threads; a slow test repeats the
  comparison on a trained population from a checkpoint when one exists. The whole suite passes with either simulator
  (`EC_RACING_SIM=numpy`).
- **Expert driver:** completes every validated track at every level. Tracks it cannot complete are
  rejected, so no controller is scored on an infeasible track.
- **Network interface:** the numpy forward pass and the exported Keras model produce identical classes
  on random inputs.
- **Work distribution:** population evaluation, dataset collection and test driving give identical
  results with one worker process and with several.
- **Keras Tuner accounting:** on a small search, the recorded epochs per trial equal the epochs each
  trial actually trained, and Hyperband continuation trials count only the epochs they added.
- **Measurement:** the inference-time warm-up call matches the timed call.
- **Experiment runner:** on a short configuration, the runner executes every notebook in order and
  writes the run record.
- **Application:** the training process starts worker processes; the final report of a finished
  training process is displayed; stopping training or a race returns the view to a state in which
  settings can be changed.
- **Track generator.** Both directions occur; every accepted map respects its level's tightest corner
  and separation limits; the expert completes it; a seed always yields the same map.
- **Time trial.** Success, laps in time and fitness on constructed cases.
- **Survival.** Successes are always kept, the cap and the elitism floor hold, and the frequency of
  chance survival matches the share of laps in time.
- **Level tracker.** Advancement, mastery goal, target reduction, the end of the speed push and the
  safety stop.
- **NEAT.** Innovation numbers, crossover alignment, node splitting, cycle prevention, speciation,
  equality of the batched forward pass with a direct per-genome computation, and equality of the
  Keras export with the genome.
- **DAgger.** Labels equal the expert's actions in the recorded states; rows accumulate across rounds;
  the goal and stall rules.
- **Checkpoints.** A run interrupted and resumed produces exactly the results of an uninterrupted run;
  a settings mismatch is refused.
- **Progress display.** Status and milestone output for a short run.
- **Network figure.** The figure is produced for dense and NEAT controllers.
- **Car setup.** A ray stops at its car's length and a 50 m car reproduces the inputs, controls and results of version
  3.0 exactly; steering sensitivity scales the steering levels within the grip limit; initial setups, inheritance and
  mutation follow §9.11; setups survive subsets, parallel evaluation, checkpoints and saved controllers.
- **Application.** The grip selector changes only display simulations; a saved application run resumes after Play,
  New run starts afresh, an unusable saved run is named, and app cars race in their own group, switched off by
  default.
- **Seeded ES:** the first initial individual equals the DAgger-refined network exactly, and the others
  differ from it.
- **Re-simulation fidelity:** re-simulating a logged generation reproduces exactly the fitness values
  recorded during training.

### 12.2 Pilot runs

Before the reported run, short pilot runs of each evolutionary method on Easy and Medium confirm that
the 90% requirement can be reached with the settings of this document. If a level is not reached in
the pilots, the survival and step-size settings are reviewed before the reported run starts. Revision 3.1 repeats the pilots with setup evolution, in `results/pilots/revision3.1/`, and reports the populations' final mean sensor length and steering sensitivity.

## 13. Default Parameters

*Table 6. Default parameters, as defined in `racing/config.py`.*

| Parameter | Value |
|---|---|
| Time step | 0.05 s |
| Grip limit μg (friction circle) | 12 m/s² |
| Speed input scaling | 40 m/s (normalization only) |
| Sensors | 9 rays from −90° to +90°; length 50 m, or 10–150 m when evolved; inputs scaled by 50 m |
| Network inputs | 12: rays, speed, position across the road (−1 left edge, +1 right edge), road width / 20 m |
| Road width | 10–16 m per map, `default_rng([seed, level, 13])` |
| Sphere tracing | Minimum step 0.5 m; at most ⌈48 × L / 50⌉ iterations for a ray of length L |
| Steering sensitivity | 1, or ⅓–3 when evolved |
| Car setup evolution | Log-uniform initial setups for random-start populations; 50 m and 1 for trained-start populations; per value mutation probability 0.2, Gaussian with standard deviation 0.1 on the natural-log scale, clipped |
| Laps per run | 3 (training, evaluation and application races) |
| Time limit | Training: the target time; evaluation and application: twice the expert lap time per lap |
| No-progress timeout | 5 s |
| Curriculum maps | Table 1; direction random per map; at most 8 built tracks per process |
| Maps per generation | 3, freshly generated |
| Starting target factor f | 1.3 × three expert laps |
| Success | All three laps within the target time |
| Survival | Successes kept; others kept with probability equal to their share of laps in time; at most 90% of the population; at least the 50 fittest |
| Level advancement and mastery goal | Lowest map success share ≥ 90% in three consecutive generations |
| Speed push | Target reduced by 2% per mastered target; a generation passes at a lowest map success share ≥ 75%; ends after 300 generations without mastering the current target |
| Robustness table | Test maps rebuilt with the same centerline at 8, 10, 14 and 18 m |
| Safety stop | 2,000 generations without a new best three-generation mean of the lowest map success share |
| GA | N = 1,000; tournament 5; crossover p = 0.5; mutation p = 0.1; s_max = 0.1; s_min = s_max / 100 |
| ES, seeded ES and speed push | N = 550; children from the 50 fittest; σ₀ = 0.1; factor 0.82; σ in [1e-4, 1.0]; reliability phase at 90% |
| NEAT | N = 1,000; crossover 0.75; weight perturbation 0.8 (σ₀ = 0.5, two-phase rule as ES); weight replacement 0.05; add connection 0.05; add node 0.03; toggle 0.01; δ = E/N + D/N + 0.4 W̄; threshold from 3.0 towards 20 species; stagnation 20 generations; tanh hidden nodes |
| DAgger | 20 maps per round; at most 30 epochs per round, patience 5, 10% hold-out; goal 18 of 20 checks in three consecutive rounds; stall after 10 rounds |
| Keras training | Adam 1e-3; batch 256; at most 150 epochs; patience 5; `steps_per_execution` and `jit_compile` set by benchmark |
| Keras Tuner | 30 trials (random search, Bayesian optimization with 10 initial random trials); Hyperband with `max_epochs` 150, factor 3, one iteration |
| Model selection | Smallest network within one percentage point of the best validation accuracy |
| Replication | Seeds 0, 1 and 2 for every method; Keras Tuner searches run once (seed 0) |
| Map seeds | Validation 80–99; test 1000–1019; dataset 1,000,000–1,999,999; training from 2,000,000 |
| Threads (compiled) or worker processes (numpy) | `EC_RACING_JOBS`, default 6, chosen by benchmark per machine |
| Simulator | `EC_RACING_SIM`: `auto` (compiled when it builds, default) or `numpy` |
| Compiled simulator build | g++, C++17: `-O3 -march=native -ffp-contract=off -std=c++17 -shared -fPIC -pthread` |
| Compiled simulator library | `racing/_fastsim-<id>.so`; `<id>`: first 16 hexadecimal digits of the SHA-256 of `fastsim.cpp`, the flags and the processor's flags |
| Application playback | 1× equals twice real time |
| Application grip | Normal 12 m/s²; Unlimited 120 m/s² (display only) |

The track, vehicle and expert-driver values were calibrated during implementation so that the expert
completes every curriculum level reliably.

## 14. Risks and Open Issues

1. **Course alignment.** Confirmation is needed from the course instructor that the project qualifies
   as Project 7, and that driving performance is acceptable as an evaluation metric alongside test
   accuracy.
2. **Unreachable levels.** The 90% requirement on unseen maps is demanding. A method may not reach it on
   every level. The pilot runs test this before the reported run, and the safety stop ends a run that
   cannot progress. Such a result is reported, not hidden.
3. **Run time.** Runs may take days. The checkpoints and the progress display make this manageable, and
   the reported training costs include the full time.
4. **NEAT cost.** The batched forward pass costs more per step than a fixed dense network when genomes
   grow large. If its cost becomes dominant, the number of hidden nodes per genome can be capped.
5. **DAgger data growth.** The aggregated dataset grows with every round. Twenty rounds add on the order
   of a million rows, which fits in memory but lengthens training per round.
6. **Generator acceptance.** Harder settings reject more candidates. The prototype accepted most maps
   within four attempts. Acceptance is checked across many seeds by a test.
7. **Seeded ES near the expert.** Evolution started from an imitation-trained network may remain close
   to the expert's driving style if no nearby weights perform better. Any such outcome is reported as
   a result.
8. **Hyperband budget.** Budget matching for Hyperband is approximate, because its bracket schedule
   determines the trial count. Actual trials, epochs and time are reported for every method.
9. **Action resolution.** Five steering levels may cause oscillatory driving. Each level is a fixed
   fraction of grip, so the turning radius it produces scales with the square of speed, and steering
   is coarsest at low speed. This is checked early by requiring the quantized expert to complete
   level-4 tracks reliably.
10. **Replication scope.** Each Keras Tuner search runs once, so the variation reported for the
   searched methods covers retraining of the selected architecture but not the search itself. The
   evolutionary methods repeat the whole run for each seed.
11. **Sensor cost.** Long sensors cost tracing time; populations that evolve long sensors make generations slower.
    This is accepted and included in the reported training cost.
12. **Coarse steering.** Populations may settle at a steering sensitivity above 1, where steering saturates and loses
    smoothness. This is a result to be reported, not a defect.
13. **Exactness of the compiled simulator.** Equality with numpy holds on each machine separately: machines may differ
    in the last bit of a mathematical function, as the numpy simulator's results already may, and a numpy release that
    changed its `tanh` kernel or the summation order of `einsum` or `bincount` would break equality. The check at load
    (§5.7) enforces the equality on each machine: on a mismatch, the run continues on the numpy simulator after one
    warning. The equivalence tests (§12.1) still run on each machine.
14. **Threads beyond physical cores.** Hyperthreads and efficiency cores add less than a full core each; the benchmark
    measures the best thread count per machine. A machine without g++ runs the numpy simulator, with identical results.

## 15. Software Environment

- **Environment:** Python 3.13 with the packages pinned in `requirements.txt` (numpy, scipy, pandas,
  matplotlib, TensorFlow 2.21, Keras 3.13, keras-tuner, scikit-learn (which keras-tuner's Bayesian optimization
  needs), joblib, pygame, nbclient, nbformat, ipykernel,
  jupyter_client and pytest). A test verifies that the installed versions match the pinned ones.
- **Compiler:** g++ with C++17 support for the compiled simulator (`sudo apt install build-essential` on Ubuntu and
  WSL); without it, the numpy simulator runs.
- **Platform:** Linux. On the Windows desktop used for the reported run, the Windows Subsystem for Linux
  (WSL 2) with the project in the Linux file system.
- **Execution:** CPU only. Worker processes call `tf.config.set_visible_devices([], 'GPU')`. Every
  reported run uses the same machine, which is recorded in `results/run_info.json`.
- **Version control:** a private GitHub repository; the `results/` directory is excluded.
