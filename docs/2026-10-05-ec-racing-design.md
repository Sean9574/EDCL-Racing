# EC Racing: Design Specification

**Evolutionary and Gradient-Based Search for Neural Network Driving Controllers**\
ECDL Term Project · Fall 2026 · Version 3.0 · 7 October 2026

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
generations, on each of four levels. The target time is then shortened in steps of 5% until the
population can no longer meet it. Every method is run with three random seeds, and results are reported
as the mean and standard deviation across seeds. All final controllers are evaluated on held-out maps for
completion, lap time, classification accuracy, network size, inference time and training cost. A
visualization application shows training in progress and races between the final controllers. A single
command executes the complete experiment; it saves checkpoints, resumes after an interruption and reports
its progress live.

Version 3.0 incorporates revision 3 (`docs/2026-10-07-ec-racing-mastery-design.md`): freshly generated
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

```
track_gen ──► tracks (fresh training maps; fixed validation / test maps; levels 1–4)
                 │
                 ▼
sim (numpy, every car stepped at once: physics, rays, crash, progress, fitness)
   │                         │                                   │
   ▼                         ▼                                   ▼
expert ──► dataset      evolution.py: time trials, survival,   app.py (pygame)
   │        ▲           levels, speed push, checkpoints         training view
   ▼        │           ├─ ga.py   ├─ es.py   ├─ neat.py         race view
imitation.py│                ▲          │                          ▲
(baseline + │                │          ▼                          │
 KT Random/ │   dagger.py ───┘   final controllers ────────────────┤
 Hyperband/ └── (labels the        (+ .npy / NEAT genome)          │
 Bayesian)       car's states)                                     │
   └──► imitation stage ──► DAgger stage ──► speed push (ES) ──────┘
                 │
                 ▼
evaluate.py, netfig.py ──► tables + figures for the paper
checkpoint.py, progress.py: resumable runs, live status line
```

*Figure 1. Components and data flow.*

A single simulator is shared by data collection, evolutionary training, the training view and the race
view. A controller therefore behaves identically in every context.

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
│   ├── sim.py          vehicle dynamics, sensors, collisions, progress, fitness
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
│   ├── bench.py        generation-time and Keras training-time measurement
│   └── app.py          pygame application: training view and race view
├── notebooks/          01_dataset, 02_imitation, 03_dagger, 04_ga, 05_es, 06_es_seeded, 07_neat, 08_evaluate
├── scripts/            notebook generation, experiment runner, pilot runs, PDF rendering
├── tests/
├── results/            generated data (not under version control)
└── docs/               design specifications and implementation plans
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
    share, streak, target reduction and elapsed time.
  - **Live line.** While the runner executes the notebooks, it displays the status as a single line that
    is updated in place, for example
    `[run 9/24] GA seed 1 · Hard · gen 412 · worst map 84% / 90% ███████████▍ · streak 1/3 · 2h 13m`.
  - **Milestones.** Level advancements, mastery goals, target reductions and completed runs are printed
    as permanent lines and appended to `results/run_log.txt`.
  - **Separate monitor.** `python -m racing.progress` shows the same display from another terminal, for
    a runner started in the background.
- **Run record:** the runner writes `results/run_info.json`. It records the processor model, the number
  of logical processors, the number of worker processes, the operating-system power profile where
  available, the versions of Python, numpy, TensorFlow, Keras and keras-tuner, the settings that affect
  results (the values of §13), the code version (commit and whether the working tree had uncommitted
  changes), and the start and end time of each notebook.
- **Worker count.** The number of worker processes is read from the environment variable
  `EC_RACING_JOBS`, with a default of 6. `python -m racing.bench generation` measures the generation
  time for a range of worker counts on the machine at hand. The run record logs the value used.
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
| 1 Easy | Loop | 4–6 | 60–110 | Spline | None | 45 | 16 |
| 2 Medium | Circuit | 6–8 | 60–170 | Slow 25–45, fast 50–80 | None | 24 | 14 |
| 3 Hard | Circuit | 9–12 | 70–220 | Slow 16–35, fast 45–90 | Main straight ≥ 170 into a hairpin; 0–1 chicane; 0–1 run of esses | 14 | 12 |
| 4 Extra hard | Circuit | 13–17 | 80–260 | Slow 10–30, fast 45–100 | Main straight ≥ 210 into a hairpin; a second hairpin; 1–2 chicanes; 1–2 runs of esses | 9 | 10 |

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
  - Steering `u_s ∈ [−1, 1]` requests a lateral acceleration `a_lat = u_s × μg`.
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
    Stopping within the 50 m sensor range requires a speed below about 35 m/s.

### 5.2 Sensors

Nine rays at −90°, −67.5°, …, +90° relative to the heading measure the distance to the track boundary,
up to 50 m. Rays are cast by sphere tracing on the distance-to-boundary map, with a minimum step of 0.5 m and at most 48 iterations per ray. Each ray stops as soon as it reaches the boundary.

The input vector has 10 elements: the nine distances divided by 50, plus the speed divided by a
reference speed of 40 m/s. The reference speed only scales the input and does not limit the vehicle.

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

## 6. Controller Network Interface

All controllers share the same input and output definition. This lets any controller be used in the
simulator, the training view and the race view without changes.

- **Input:** the 10-element vector defined in §5.2.
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
evaluations of evolutionary controllers use these files.

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

Each row consists of the 10 inputs and one class label.

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
and `jit_compile` (XLA compilation). Their values are chosen with the Keras training-time benchmark in
`bench.py`.

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
starts from minimal networks and grows them. Each approach is run with seeds 0, 1 and 2. Runs have no
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
target reduction or end), the map seeds and the elapsed time. NEAT also logs its number of species. A
checkpoint is written after every generation (§3.3).

**Run time:** the reported time is the active run time, accumulated across resumptions, including the
final controller selection. Each run starts with new worker processes, so every run includes the same
cost of generating its maps.

### 9.2 Time trial

Every drive in training and evaluation is a time trial of three laps.

- **Reference time.** A map's reference time is the expert's time for three laps. It serves only as a
  measure of the map's length and difficulty, so that one target factor applies to maps of any size.
- **Target time.** The target time is f × reference time. The target factor f is 1.3 while a
  population climbs the levels, the value of the revision 2.0 pace gate.
- **Success.** A car succeeds on a map if it completes all three laps within the target time.
- **Laps in time.** For a car that does not succeed, the laps it completed before it crashed, stalled or
  reached the target time are counted.
- **Simulation end.** A training drive ends at the target time. The five-second no-progress timeout is
  unchanged.
- **Fitness.** Fitness keeps the revision 2.0 form with the target time as the time limit: the
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

**Speed push.** After the mastery goal, the target factor is multiplied by 0.95, which shortens the target time by 5%.
Each further run of three consecutive passing generations shortens it by another 5%. The run ends
when 500 generations pass without mastering the current target. The tightest target mastered is
recorded as the run's speed result, expressed as the reduction from the starting target. For example,
f = 1.3 × 0.95⁴ is reported as a target 18.5% shorter than the starting target.

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

The population size is 1,000. Children are produced as in revision 2.0: two parents are selected by
tournaments of size 5 over the whole population, combined by uniform crossover (p = 0.5) and mutated
(each weight with p = 0.1, by a Gaussian of strength s). The strength follows the revision 2.0 rule
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

- **Learning phase.** While fewer than 90% of the 50 fittest cars succeed on every map, σ follows the
  1/5 success rule of revision 2.0, with a tie counted as half a success.
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

- **Work distribution:** the evaluation of a generation is divided into tasks, each covering one of the
  generation's three tracks for one share of the population, and the tasks are distributed across
  worker processes (joblib). Each simulation step has a fixed cost of about 0.45 ms plus about 2.3 µs
  per car, so a task should cover as many cars as possible on one track rather than few cars on all
  three. Results are independent of the division, because cars do not interact.
- **Number of workers:** chosen with the generation-time benchmark in `bench.py`, which times the same
  evaluation path the training runs use.
- **Keras training:** the settings `steps_per_execution` and `jit_compile` (§8.1) are chosen with the
  Keras training-time benchmark in `bench.py`.
- **Fresh maps:** each generation builds its three maps in the worker processes that drive them. A map
  takes 0.1–0.6 s to build, against several seconds of driving. Every process keeps at most 64 built
  tracks, so memory stays constant over runs of many thousands of generations.
- **NEAT:** all genomes of a generation are compiled into flat arrays and evaluated together, depth by
  depth (§9.9), so NEAT drives its whole population at once like the other methods.
- **Further measures**, applied only if training is still too slow: fewer sphere-tracing steps, then
  shorter time limits.

GPU execution is not used: the per-operation launch cost on a GPU exceeds the arithmetic of networks
and car batches of this size.

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
  - Training is started and paused with a play/pause control and ended with a stop control. After a
    stop, the approach, run seed, difficulty and population size can be changed.
  - Changing a setting is possible only while no training is running.
- **Training process:**
  - Training runs in a separate process at full speed, with the configured number of worker
    processes. The training process is not a daemon process, because joblib cannot start worker
    processes from a daemon process; it is terminated explicitly when training stops or the
    application closes.
  - After each generation, completed Keras Tuner trial or baseline epoch, it sends the population, fitness values, map success shares, level, target factor and map seeds to the view through a `multiprocessing.Queue`.
  - When the training process ends, the view reads the queue once more, so the final report is
    displayed.
  - Pausing halts training at the next generation boundary.
  - Training started from the application writes to `results/app_runs/` and never saves final
    controllers, so it cannot overwrite the reported results. The seeded ES in the application starts
    from the DAgger-refined baseline of the selected run seed. If it does not exist, the seeded ES cannot
    be started and the view states that notebook 03 must be run first.
- **Playback:**
  - The view re-simulates only the displayed individuals, on the generation's first track, at 1×, 2×,
    4× or 8× playback speed. At 1×, simulated time runs at twice real time.
  - Tracks are built in a background thread. While a new track is being built, the window stays
    responsive and playback starts once the track is ready.
  - When playback ends, the view advances to the latest completed generation and skips any in
    between.
  - Rendering never slows training.
- **Track display:** the side panel shows the seed and level of the track being played back. The
  imitation views always use validation track 80, so its map is shown before training starts. The evolutionary approaches draw new maps every generation, so their map appears with the first
  generation.
- **Display selection:** All, Best or Worst, with a displayed-car count N from 1 to the population size:
  - All shows a fixed random sample of N individuals.
  - Best shows the N fittest.
  - Worst shows the N least fit.

  The fittest displayed individual is outlined, and its sensor rays can be shown.
- **Side panel:**
  - approach, level and generation
  - the success share of each of the generation's maps against the 90% requirement, and the streak
  - the target reduction during the speed push
  - mutation strength (GA) or σ (ES, NEAT), and the number of species (NEAT)
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

## 11. Evaluation Methodology

### 11.1 Test procedure

Each final controller (eight methods, three seeds) drives the 20 test maps as a three-lap time trial.
Two outcomes are recorded per map: whether the car **finished** within the simulator's time limit of
twice the reference lap time per lap, and whether it **succeeded** within the starting target (f = 1.3).
The validation maps select each run's final controller (§9.5) and each method's best seed for the
application.

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
   best lap time, mean three-lap race time, crashes, and the target reduction reached in the speed
   push.
2. **Head-to-head speed.** Mean lap time on the test maps that every final controller finished.
3. **Training cost.** Per method and seed: time and evaluations to clear each level, to reach the
   mastery goal and to end the speed push, and whether each was reached. For imitation methods, the
   Keras Tuner search time and the number and time of DAgger rounds are included.
4. **Imitation stages.** Test results of the imitation methods after initial training, after DAgger and
   after the speed push.
5. **Course comparison.** Test accuracy, parameters, batch inference time, single-sample latency and
   search and training time, as required by the course and measured as in revision 2.0.
6. **Search cost, dataset and driving-selected controllers.** Unchanged from revision 2.0.

### 11.4 Figures

- **Success over training.** For each method, the lowest map success share per generation, with
  level advancements and target reductions marked.
- **Lap time over training.** The median lap time of the successful cars per generation.
- **Networks.** One panel per method showing its best final controller. The nine distance rays,
  labelled by angle, and the speed input are on the left. Hidden nodes are placed in columns by depth,
  so that the structure evolved by NEAT appears as it is. The 15 actions, labelled by steering and
  throttle, are on the right. Connections are blue when positive and red when negative, with width and
  opacity proportional to the weight's magnitude.
- **Step sizes and Keras Tuner trials.** Unchanged from revision 2.0.

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
- **Seeded ES:** the first initial individual equals the DAgger-refined network exactly, and the others
  differ from it.
- **Re-simulation fidelity:** re-simulating a logged generation reproduces exactly the fitness values
  recorded during training.

### 12.2 Pilot runs

Before the reported run, short pilot runs of each evolutionary method on Easy and Medium confirm that
the 90% requirement can be reached with the settings of this document. If a level is not reached in
the pilots, the survival and step-size settings are reviewed before the reported run starts.

## 13. Default Parameters

*Table 6. Default parameters, as defined in `racing/config.py`.*

| Parameter | Value |
|---|---|
| Time step | 0.05 s |
| Grip limit μg (friction circle) | 12 m/s² |
| Speed input scaling | 40 m/s (normalization only) |
| Sensors | 9 rays from −90° to +90°, 50 m range |
| Sphere tracing | Minimum step 0.5 m; at most 48 iterations per ray |
| Laps per run | 3 (training, evaluation and application races) |
| Time limit | Training: the target time; evaluation and application: twice the expert lap time per lap |
| No-progress timeout | 5 s |
| Curriculum maps | Table 1; direction random per map; at most 64 built tracks per process |
| Maps per generation | 3, freshly generated |
| Starting target factor f | 1.3 × three expert laps |
| Success | All three laps within the target time |
| Survival | Successes kept; others kept with probability equal to their share of laps in time; at most 90% of the population; at least the 50 fittest |
| Level advancement and mastery goal | Lowest map success share ≥ 90% in three consecutive generations |
| Speed push | Target reduced by 5% per mastered target; ends after 500 generations without mastering the current target |
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
| Worker processes | `EC_RACING_JOBS`, default 6, chosen by benchmark per machine |
| Application playback | 1× equals twice real time |

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

## 15. Software Environment

- **Environment:** Python 3.13 with the packages pinned in `requirements.txt` (numpy, scipy, pandas,
  matplotlib, TensorFlow 2.21, Keras 3.13, keras-tuner, joblib, pygame, nbclient, nbformat, ipykernel,
  jupyter_client and pytest). A test verifies that the installed versions match the pinned ones.
- **Platform:** Linux. On the Windows desktop used for the reported run, the Windows Subsystem for Linux
  (WSL 2) with the project in the Linux file system.
- **Execution:** CPU only. Worker processes call `tf.config.set_visible_devices([], 'GPU')`. Every
  reported run uses the same machine, which is recorded in `results/run_info.json`.
- **Version control:** a private GitHub repository; the `results/` directory is excluded.
