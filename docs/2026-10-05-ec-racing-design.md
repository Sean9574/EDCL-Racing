# EC Racing: Design Specification

**Evolutionary and Gradient-Based Search for Neural Network Driving Controllers**\
ECDL Term Project · Fall 2026 · Version 2.0 · 6 October 2026

## Summary

This document specifies EC Racing, a top-down two-dimensional racing simulator. It is used to compare
methods for finding small neural network driving controllers. Seven controllers compete on procedurally
generated tracks.

- **Imitation learning (four controllers).** These are trained by supervised imitation of a scripted
  expert driver. One is a hand-designed baseline. In the other three, Keras Tuner selects the
  architecture using random search, Hyperband or Bayesian optimization.
- **Neuroevolution (two controllers).** A genetic algorithm (GA) and a (μ+λ) evolution strategy (ES)
  optimize the network weights directly from driving performance. Both evolve large populations and
  follow a curriculum of progressively harder tracks.
- **Hybrid (one controller).** A seeded ES starts from the trained baseline network, which has learned
  from the expert, and then optimizes the weights from driving performance alone.

Every method is run with three random seeds, and results are reported as the mean and standard
deviation across seeds. All controllers are evaluated on held-out tracks for classification accuracy,
network size, inference time and driving performance. A visualization application shows training in
progress and races between the final controllers. A single command executes the complete experiment.

Version 2.0 revises version 1.0 after a review of the implementation. The revision corrects the
measurement of training epochs and inference time, adds replication across seeds, the seeded ES and a
pace requirement for curriculum advancement, raises the epoch limit, and specifies how the experiment
is executed.

## 1. Introduction

### 1.1 Research question

On tracks not seen during training, do controllers evolved through simulated driving outperform
controllers that are trained to imitate an expert and whose architectures are tuned by conventional
hyperparameter optimization?

A secondary question is whether evolution that starts from an imitation-trained controller combines
the strengths of both approaches: the expert's driving competence at the start, and the freedom to
exceed the expert's pace afterwards.

### 1.2 Alignment with course requirements

The project addresses the course project guidelines (*Suggested Term Projects*, version 1.0,
5 October 2026):

- All required models are included:
  - a baseline model
  - Keras Tuner with random search, Hyperband and Bayesian optimization
  - an ES (μ+λ) with the 1/5 success rule
  - a GA
- The objectives are to maximize test accuracy and to minimize network size and inference time.
- All networks are small, so training time stays short.
- The deliverable is a research paper. The project falls under Project 7 ("any research-oriented
  project using DL, EC, and HPO").

## 2. Scope

The following are outside the scope of this project:

- Camera-based perception, including semantic segmentation.
- Human-driven cars, menu systems, track editing and real-world track layouts.
- Collisions between cars.
- Recording, replay or video export of training runs.
- A user interface for browsing earlier generations. The best network of each generation is saved to
  disk but cannot be browsed in the application.
- GPU acceleration. Preliminary measurements on the target hardware showed CPU execution to be faster
  for networks of this size.

## 3. System Architecture

### 3.1 Overview

```
track_gen ──► tracks (train / validation / test seeds, levels 1–4)
                 │
                 ▼
sim (numpy, every car stepped at once: physics, rays, crash, progress, fitness)
   │                          │                              │
   ▼                          ▼                              ▼
expert ──► dataset       ga.py / es.py                  app.py (pygame)
   │                     (population drives in sim,      training view
   ▼                      curriculum.py sets level)      race view
imitation.py                  ▲   │                          ▲
(baseline + KT Random/        │   ▼                          │
 Hyperband/Bayesian) ─────────┘  final weights ──► Keras copy┤
   │       (trained baseline      (+ float64 .npy)           │
   │        seeds the seeded ES)                             │
   └──► final Keras models ──────────────────────────────────┘
                 │
                 ▼
evaluate.py ──► test tables + figures for the paper
```

*Figure 1. Components and data flow.*

A single simulator is shared by data collection, evolutionary training, the training view and the race
view. A controller therefore behaves identically in every context.

### 3.2 Module structure

```
Project/
├── README.md
├── racing/
│   ├── config.py       default parameters (§13) and color assignments
│   ├── actions.py      action table: class ↔ steering and throttle fractions
│   ├── backend.py      Keras access with GPUs hidden
│   ├── track.py        track data structure, road grid, wall-distance lookup
│   ├── track_gen.py    curriculum level shapes and geometric validation
│   ├── tracks.py       validated tracks with the expert's lap time (cached)
│   ├── sim.py          vehicle dynamics, sensors, collisions, progress, fitness
│   ├── nets.py         batched numpy forward pass, Keras conversion, champion loading
│   ├── expert.py       scripted expert driver and dataset collection
│   ├── imitation.py    baseline, three Keras Tuner searches, model selection
│   ├── curriculum.py   level-advancement rule, GA mutation-strength rule
│   ├── population.py   shared evolutionary helpers: evaluation, logging, selection
│   ├── ga.py           genetic algorithm
│   ├── es.py           (μ+λ) evolution strategy
│   ├── evaluate.py     test-track evaluation, tables, figures
│   ├── bench.py        generation-time and Keras training-time measurement
│   └── app.py          pygame application: training view and race view
├── notebooks/          01_dataset, 02_imitation, 03_ga, 04_es, 05_es_seeded, 06_evaluate
├── scripts/            notebook generation, experiment runner, calibration pilots
├── tests/
├── results/            generated data (not under version control)
└── docs/               design specification and implementation plans
```

*Figure 2. Repository layout.*

- Each module has a single responsibility.
- For readability, the code uses plain loops and explicit conditionals rather than abstraction layers.
- Notebooks run the reported experiments with fixed settings by calling into the `racing` package.

### 3.3 Experiment execution

The complete experiment is executed by a single command, `scripts/run_notebooks.py`. It executes the
six notebooks in order and saves their outputs into the notebook files, so the notebooks form the
record of the reported run.

- **Order:** 01 dataset → 02 imitation → 03 GA → 04 ES → 05 seeded ES → 06 evaluation. The seeded ES
  depends on the baseline networks from notebook 02. Notebooks run one at a time, so no two methods
  compete for the processor while their training time is measured.
- **Resumption:** an option `--from NN` starts at a given notebook, using the results of the earlier
  notebooks already on disk.
- **Kernel:** each notebook runs in a kernel started with the same Python interpreter as the runner,
  with `notebooks/` as the working directory.
- **Run record:** the runner writes `results/run_info.json`. It records the processor model, the number
  of logical processors, the number of worker processes, the operating-system power profile, the
  versions of Python, numpy, TensorFlow, Keras and keras-tuner, and the start and end time of each
  notebook. A log of the run is written to `results/run_log.txt`.
- **Clean start:** each training run removes its own earlier outputs before it starts, so an
  interrupted run cannot leave files that are later mixed with a new run.

Constants that depend on calibration (the GA mutation-strength range and the pace factor of §9.1) are
set from pilot runs (`scripts/pilots.py`) before the reported run. The pilots write to
`results/pilots/` and are not part of the reported results.

## 4. Track Generation and Curriculum

### 4.1 Track representation

The function `make_track(seed, level)` returns a `Track` object containing:

- the centerline, resampled at 1 m intervals
- the track width and the left and right boundaries
- a road occupancy grid at 0.5 m resolution
- a distance-to-boundary map computed with `scipy.ndimage.distance_transform_edt`
- the start pose and track length
- the expert driver's lap time

### 4.2 Curriculum levels

The evolutionary controllers train on a curriculum. It begins on a circle and ends on fully varied
seeded tracks, where lap time dominates the score.

*Table 1. Curriculum levels.*

| Level | Shape | Dimensions | Skill required |
|---|---|---|---|
| 1 | Circle | Radius 50 m; width 16 m | Sustained constant-radius turning |
| 2 | Oval: two straights joined by semicircles | Straights 80–160 m; turn radius 25–40 m; width 14 m | Braking before corners |
| 3 | Seeded loop with 6–8 control points | Radius 80–160 m; minimum corner radius 40 m; width 14 m | Adapting to varied layouts |
| 4 | Seeded loop with 10–16 control points | Radius 60–200 m; minimum corner radius 15 m; width 12 m | Minimizing lap time |

Seeded loops are built from control points at sorted random angles and radii. The points are
interpolated with a periodic cubic spline.

### 4.3 Validation

A generated track is rejected if any of the following holds:

- the road surface overlaps itself
- any corner is tighter than the level's minimum radius
- the expert driver cannot complete a lap

A rejected seed is retried deterministically using the pair (seed, attempt), so a given seed always
yields the same track.

### 4.4 Seed allocation

- **Training:** seeds 0–79 at all levels. Level 1 produces the same circle for every seed.
- **Validation:** seeds 80–99, level 4.
- **Test:** seeds 1000–1019, level 4. Test tracks are never used for training or model selection.
- **Application races:** the race view can also build the test seeds at levels 1–3 (§10.3). These
  tracks serve only for display and are never used for training, selection or the reported results.

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

A run terminates when any of the following occurs:

- the car completes one lap
- the car crashes
- the car makes no progress for 5 s
- the time limit is reached, equal to twice the expert's lap time on that track

### 5.5 Fitness

The fitness of a run on a single track is:

- `p`, the completed fraction of the lap (0 ≤ p < 1), if the lap is not completed
- `1 + (1 − t_lap / t_limit)`, a value between 1 and 2, if the lap is completed

Under this definition, completion separates controllers early in training. Once controllers complete
laps, lap time separates them.

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
reproduce. The expert's lap time on each track sets that track's time limit (§5.4).

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

- **Training:**
  - level 1 once
  - levels 2–3 on seeds 0–19
  - level 4 on seeds 0–79
- **Validation:** level 4, seeds 80–99.
- **Test:** level 4, seeds 1000–1019.

The expected total is approximately 100,000–200,000 rows.

Collection is distributed across worker processes, one track per task. Each track uses its own random
number generator, so the dataset is identical to a sequential collection. The collection time is
recorded and reported as a cost shared by the four imitation methods and the seeded ES.

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
0, 1 and 2. These three networks are the method's final controllers. The search therefore runs once
per method, and the variation across seeds measures training variability for a fixed architecture,
not the variability of the search itself.

**Driving-based selection (secondary analysis):** imitation methods are selected on validation
accuracy, whereas evolutionary methods are selected on validation driving. To show how much this
asymmetry matters, the most-trained model of every distinct configuration in each search drives one
lap on each of the 20 validation tracks. The model with the highest mean validation fitness is
reported for each method as a driving-selected controller in a secondary driving table.

## 9. Evolutionary Controllers

### 9.1 Common procedure

All three evolutionary approaches use the baseline architecture (431 weights), so the learning method
is the only variable. Evolution optimizes the weights, not the network size. For the GA and ES, initial
weights are drawn from a Glorot-uniform distribution. The seeded ES starts from a trained network
(§9.4). Each approach is run with seeds 0, 1 and 2.

**Each generation proceeds as follows:**

1. Three tracks are drawn from the current level's training seeds using the run's random number
   generator.
2. Every individual drives all three tracks. Kept individuals are re-evaluated, because the tracks
   change every generation.
3. An individual's fitness is its mean over the three tracks. It counts as finishing only if it
   completes all three.

**Individual types:**

- **Kept:** carried over unchanged from the previous generation.
- **New:** created in the current generation by mutation, and in the GA also by crossover.

**Pace:** an individual finishes *at pace* when it completes all three tracks and its lap time on each
is at most 1.3 times the expert's lap time on that track. The pace factor 1.3 is confirmed by pilot
runs before the reported run (§3.3).

**Level advancement:** an approach advances one level when 100% of its kept individuals finish at pace
and at least 80% of its new individuals finish at pace, for three consecutive generations. Each
approach advances independently. Level 4 continues until the computational budget is exhausted.

The pace requirement pushes the evolved controllers to drive fast on each level before they move on.
Without it, the curriculum would promote a population as soon as it survives, however slowly it
drives. Fitness is unchanged: completion still ranks above any incomplete lap, and lap time ranks
completed laps. The GA's mutation-strength rule (§9.2) continues to use the plain finishing rate.

**Computational budget:** each approach receives 200,000 individual evaluations per seed. One
evaluation is one individual driving its generation's three tracks.

- GA: 1,000 evaluations per generation, or 200 generations.
- ES and seeded ES: 550 evaluations per generation, or approximately 364 generations.

Wall-clock time is also reported. It covers the whole run, including the final controller selection.
Each run starts with new worker processes, so every run includes the same cost of generating its tracks.

**Logging:** each run writes to its own directory, `results/runs/<approach>/seed<k>/`. Each generation
is recorded in a CSV file with:

- level and generation number
- best, mean and worst fitness
- the finishing percentages of kept and new individuals, and the same percentages at pace
- GA mutation strength or ES step size σ (not logged for the GA's first generation, which is not
  produced by mutation)
- for the ES, the fractions of offspring that outperform and that tie their parent
- the track seeds used
- elapsed time

The best network of each generation is saved as an `.npy` file (about 3.4 KB). Results are written
after every generation, so closing the application or an unexpected failure loses at most the
generation in progress.

**Final controller selection:** when the budget is exhausted, the final kept individuals are evaluated
on the validation tracks (seeds 80–99). The individual with the highest mean fitness is selected and
exported as a `.keras` file and a float64 `.npy` file, at `results/champions/<approach>/seed<k>`. If
an approach did not reach level 4, selection proceeds the same way and the level reached is reported.

### 9.2 Genetic algorithm

- **Population:** N = 1,000 individuals (configurable). The K = 50 fittest are kept unchanged.
- **Each of the N − K new individuals is created in three steps:**
  1. **Selection:** two parents are each chosen by tournament selection, with a tournament size of 5
     drawn from the whole population.
  2. **Crossover:** uniform crossover, where each weight is inherited from either parent with
     probability 0.5.
  3. **Mutation:** each weight, with probability 0.1, receives an added sample from N(0, s).
- **Mutation strength** adapts to the finishing rate:

  ```
  s = s_min + (s_max − s_min) × (1 − f_new)
  ```

  Here `f_new` is the fraction of the previous generation's new individuals that finished. The
  minimum is one percent of the maximum, `s_min = s_max / 100`, and `s = s_max` in the first
  generation. The value of `s_max` is chosen by pilot runs (§3.3) from the candidates 0.5, 0.2 and
  0.1: each candidate runs 100 generations with seed 0, and the candidate with the fastest level
  progression is selected, with ties resolved in favor of the larger value. A first review measured
  that with `s_max = 0.5` the GA was still on level 1 after 40 generations.
  - Mutation strength is highest when few new individuals finish.
  - It falls to the minimum when all of them finish.
  - After a level advancement, `f_new` drops on the harder tracks, so mutation strength rises again
    without any special handling.

### 9.3 Evolution strategy

- **Scheme:** (μ+λ), with μ = 50 parents (the kept individuals) and λ = 500 offspring.
  - Each offspring is a randomly chosen parent with a sample from N(0, σ²) added to every weight.
  - The next generation's parents are the 50 fittest of parents ∪ offspring.
- **Step-size control:** the 1/5 success rule is applied every generation. Let `p_s` be the fraction
  of offspring that outperform their own parent, where an offspring whose fitness equals its parent's
  counts as half a success:
  - if `p_s > 0.2`, then σ ← σ / 0.82
  - if `p_s < 0.2`, then σ ← σ × 0.82

  The factor 0.82 follows Schwefel. The initial step size is σ₀ = 0.1, and σ is clipped to
  [1e-4, 1.0].
- **Ties:** fitness is deterministic and the actions are discrete, so on a fitness plateau many
  offspring score exactly the same as their parent, for example when every parent crashes before
  covering any distance and scores 0. If ties counted as failures, `p_s` would be close to zero on
  such a plateau, and σ would shrink to its lower bound, where the search can no longer leave the
  plateau. Counting a tie as half a success gives `p_s` = 0.5 on a complete plateau, so σ grows. With
  continuously varying fitness, ties are rare and the rule is unchanged.
- **Level advancement:** σ is reset to σ₀ whenever the ES advances a level, starting with the
  offspring of the next generation. While the population refines its driving on one level, σ
  shrinks, and the tracks of the next level require exploration again. The reset mirrors the GA,
  whose mutation strength rises automatically after a level advancement.

The GA's finish-rate adaptation and the ES's success-rate adaptation are two different step-size
control strategies. Their behavior is compared in the results.

The tie rule carries a risk on continuous fitness. Once σ is small enough that many offspring repeat
their parent's discrete actions exactly, ties push `p_s` above 0.2 and σ grows again, possibly into a
range where too few new individuals finish at pace for advancement. The ES log therefore records the
fractions of strict successes and ties, and the ES pilot run (§3.3) checks level progression before
the reported run.

### 9.4 Seeded evolution strategy

The seeded ES tests whether evolution can improve on a controller that has first learned from the
expert. Run *k* starts from the baseline network trained with seed *k* (§8.1), which has the same
architecture as the evolutionary controllers.

- **Initial parents:** the first of the μ = 50 parents is the trained network itself, converted to
  float64. The other 49 are copies with a sample from N(0, σ₀²) added to every weight.
- **Procedure:** from there, the seeded ES is identical to the ES: the same operators, step-size
  control, curriculum (starting at level 1), pace requirement and budget. The trained network should
  advance through the early levels quickly, leaving most of the budget for the harder levels.
- **Interpretation:** the seeded ES uses the expert indirectly, through the baseline. Its training
  time is reported separately from the dataset collection and baseline training on which it depends.

### 9.5 Computational performance

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

Interface elements that do not represent an approach, such as the highlight of a selected button, use a
dark neutral color so that they cannot be mistaken for an approach.

### 10.2 Training view

Training is observed live rather than recorded.

- **Controls:**
  - The user selects an approach, a run seed and a difficulty (§10.4).
  - For the GA, ES and seeded ES, the user sets the population size before starting. For the ES
    variants, this setting is the number of offspring λ.
  - Training is started and paused with a play/pause control and ended with a stop control. After a
    stop, the approach, run seed, difficulty and population size can be changed.
  - Changing a setting is possible only while no training is running.
- **Training process:**
  - Training runs in a separate process at full speed, with the configured number of worker
    processes. The training process is not a daemon process, because joblib cannot start worker
    processes from a daemon process; it is terminated explicitly when training stops or the
    application closes.
  - After each generation, completed Keras Tuner trial or baseline epoch, it sends the weights,
    fitness values, level and track seeds to the view through a `multiprocessing.Queue`.
  - When the training process ends, the view reads the queue once more, so the final report is
    displayed.
  - Pausing halts training at the next generation boundary.
  - Training started from the application writes to `results/app_runs/` and never saves final
    controllers, so it cannot overwrite the reported results. The seeded ES in the application starts
    from the reported baseline controller of the selected run seed. If that controller does not exist,
    the seeded ES cannot be started and the view states that notebook 02 must be run first.
- **Playback:**
  - The view re-simulates only the displayed individuals, on the generation's first track, at 1×, 2×,
    4× or 8× playback speed. At 1×, simulated time runs at twice real time.
  - Tracks are built in a background thread. While a new track is being built, the window stays
    responsive and playback starts once the track is ready.
  - When playback ends, the view advances to the latest completed generation and skips any in
    between.
  - Rendering never slows training.
- **Track display:** the side panel shows the seed and level of the track being played back. The
  imitation views always use validation track 80, so its map is shown before training starts. The
  GA and ES variants draw new tracks every generation, so their map appears with the first
  generation.
- **Display selection:** All, Best or Worst, with a displayed-car count N from 1 to the population size:
  - All shows a fixed random sample of N individuals.
  - Best shows the N fittest.
  - Worst shows the N least fit.

  The fittest displayed individual is outlined, and its sensor rays can be shown.
- **Side panel:**
  - approach, level and generation
  - finishing percentages of kept and new individuals
  - mutation strength (GA) or σ (ES)
  - best and mean fitness
  - a live plot of best and mean fitness per generation, with level-advancement markers
- **Keras Tuner approaches:** each completed trial is treated as one individual and is driven on
  validation track 80.

### 10.3 Race view

- **Controllers:** the view loads whichever final controllers exist in `results/champions/`. Each
  approach has a selector that cycles through *best*, *seed 0*, *seed 1*, *seed 2* and *off*. *Best*
  is the seed whose controller achieved the highest mean validation fitness, as recorded by the
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
| Easy | 1 | Circle |
| Medium | 2 | Oval |
| Hard | 3 | Seeded loop with 6–8 control points |
| Extra hard | 4 | Seeded loop with 10–16 control points |

In the race view, the difficulty sets the level of the race track. In the training view, it sets the
level at which GA and ES training starts (the curriculum still advances from there) and the level of
the track on which imitation trials and epochs are driven. The difficulty applies only to the
application; the reported experiment always starts at level 1 and evaluates on level 4.

## 11. Evaluation Methodology

Each method has three final controllers, one per seed (§8.1, §8.3, §9.1). Every measurement below is
taken for each of them, and tables report the mean and standard deviation across the three seeds.

- **Driving evaluation:** each final controller drives one lap on each of the 20 test tracks.
  Evolutionary controllers drive with their float64 weights (§6). The drives are distributed across
  worker processes; results do not depend on the distribution.
- **Validation driving:** each final controller also drives the 20 validation tracks. For every
  method, the seed with the highest mean validation fitness is recorded in
  `results/champions/best_seeds.json` and is used as that method's default in the race view. The
  record does not use the test tracks.
- **Accuracy evaluation:** each controller's predicted classes are compared with the expert labels in
  the test dataset. For the evolutionary controllers, this measures agreement with an expert they
  never observed, which is reported as a result in its own right.
- **Inference time:**
  - *Batch:* Keras `predict` on 10,000 test rows in a single batch, reported per sample and averaged
    over five runs. The warm-up call is identical to the timed calls, so the timed runs include no
    function tracing or compilation.
  - *Single sample:* the median time of 1,000 calls of the Keras model on one row. A controller
    processes one observation per simulation step, so this latency is the relevant figure for
    driving.

  Inference is timed after all parallel work has finished, so no other process competes for the
  processor. All controllers are timed with the same prediction settings (no XLA compilation, one
  batch per call).
- **Training time:**
  - baseline: training time per seed
  - Keras Tuner methods: search time (one search per method) and retraining time per seed, reported
    separately
  - evolutionary methods: run time per seed, including final controller selection
  - dataset collection: reported once, as a cost shared by the imitation methods and the seeded ES

  The run conditions recorded in `results/run_info.json` (§3.3) are reported with the results.

*Table 5. Results tables.*

| Table | Columns |
|---|---|
| Required comparison | Method; test accuracy; parameters; batch inference time; single-sample latency; search/training time |
| Driving performance | Method; tracks completed (of 20); mean lap time on completed tracks; crashes; curriculum level reached (per seed) |
| Search cost | Keras Tuner method; trials; distinct configurations; total epochs; trials at the epoch limit; search time |
| Driving-selected controllers (secondary) | Keras Tuner method; selected configuration; parameters; validation accuracy; tracks completed (of 20); mean lap time; crashes |

**Figures:**

- Fitness against individual evaluations for the GA, ES and seeded ES, one line per seed, with
  level-advancement markers.
- GA mutation strength and ES step size σ across generations, one line per seed.
- Validation accuracy against parameter count for every Keras Tuner trial, with each method's selected
  model marked.

## 12. Verification and Testing

Automated tests use pytest.

- **Track generation:**
  - a given seed always yields the same track
  - level 1 is a circle of the specified radius
  - generated tracks never overlap themselves
  - the minimum corner radius is respected
- **Simulation:**
  - zero steering produces straight-line motion
  - combined steering and throttle never exceed the grip limit
  - leaving the road registers a crash
  - a completed lap is detected
  - reverse driving earns no progress
  - identical inputs produce identical outputs
- **Sensors:** ray distances match analytically computed values for a car on the level-1 circle.
- **Expert driver:** completes every validated track at every level. Tracks it cannot complete are
  rejected, so no controller is scored on an infeasible track.
- **Network interface:** the numpy forward pass and the exported Keras model produce identical classes
  on random inputs.
- **GA and ES:** the operators solve a standard test function (the sphere function) and show the
  expected step-size behavior under the rule as specified, including ties.
- **Seeded ES:** the first initial parent equals the trained network exactly, and the others differ
  from it.
- **Curriculum:**
  - level advancement triggers on exactly the third consecutive qualifying generation, and the count
    resets after a non-qualifying generation
  - a car that finishes slower than the pace factor allows does not count toward advancement
  - the mutation-strength rule returns `s_max` at 0%, the midpoint at 50% and `s_min` at 100%
- **Work distribution:** population evaluation, dataset collection and test driving give identical
  results with one worker process and with several.
- **Keras Tuner accounting:** on a small search, the recorded epochs per trial equal the epochs each
  trial actually trained, and Hyperband continuation trials count only the epochs they added.
- **Measurement:** the inference-time warm-up call matches the timed call.
- **Clean start:** a training run removes its own earlier outputs before it starts.
- **Experiment runner:** on a short configuration, the runner executes every notebook in order and
  writes the run record.
- **Application:** the training process starts worker processes; the final report of a finished
  training process is displayed; stopping training or a race returns the view to a state in which
  settings can be changed.
- **Re-simulation fidelity:** re-simulating a logged generation reproduces exactly the fitness values
  recorded during training.

## 13. Default Parameters

*Table 6. Default parameters, as defined in `racing/config.py`.*

| Parameter | Value |
|---|---|
| Time step | 0.05 s |
| Grip limit μg (friction circle) | 12 m/s² |
| Speed input scaling | 40 m/s (normalization only) |
| Sensors | 9 rays from −90° to +90°, 50 m range |
| Sphere tracing | Minimum step 0.5 m; at most 48 iterations per ray |
| Time limit per lap | Twice the expert lap time |
| No-progress timeout | 5 s |
| Tracks per generation | 3 |
| Pace factor | 1.3 times the expert lap time (confirmed by pilot runs) |
| Level advancement | Kept 100% and new ≥ 80% finishing at pace, three consecutive generations |
| GA | N = 1,000; K = 50; tournament size 5; crossover p = 0.5; mutation p = 0.1; s_max chosen by pilot runs from {0.5, 0.2, 0.1}; s_min = s_max / 100 |
| ES and seeded ES | μ = 50; λ = 500; σ₀ = 0.1; adaptation factor 0.82; σ in [1e-4, 1.0] |
| Evolutionary budget | 200,000 individual evaluations per approach and seed |
| Keras training | Adam 1e-3; batch 256; at most 150 epochs; patience 5; `steps_per_execution` and `jit_compile` set by benchmark |
| Keras Tuner | 30 trials (random search, Bayesian optimization with 10 initial random trials); Hyperband with `max_epochs` 150, factor 3, one iteration |
| Model selection | Smallest network within one percentage point of the best validation accuracy |
| Replication | Seeds 0, 1 and 2 for every method; Keras Tuner searches run once (seed 0) |
| Track seeds | Training 0–79; validation 80–99; test 1000–1019 |
| Application playback | 1× equals twice real time |

The track, vehicle and expert-driver values were calibrated during implementation so that the expert
completes every curriculum level reliably.

## 14. Risks and Open Issues

1. **Course alignment.** Confirmation is needed from the course instructor that the project qualifies
   as Project 7, and that driving performance is acceptable as an evaluation metric alongside test
   accuracy.
2. **Runtime.** Three seeds per method, a 150-epoch limit and the seeded ES lengthen the experiment
   to several hours. The experiment runner (§3.3) executes it unattended and can resume from any
   notebook. Mitigations for slow generations are listed in §9.5.
3. **Curriculum stalls.** The advancement criterion is strict by design, and the pace requirement
   makes it stricter. The pace factor, the threshold and the number of consecutive generations are
   configurable, pilot runs check level progression before the reported run, and a stall is visible
   immediately in the training view.
4. **ES step size under the tie rule.** Counting ties as half successes may hold σ in a range where
   advancement is rare (§9.3). The ES pilot run records the fractions of strict successes and ties.
   If the pilot shows that σ stalls advancement, the step-size rule is revisited before the reported
   run.
5. **Seeded ES near the expert.** Evolution started from an imitation-trained network may remain close
   to the expert's driving style if no nearby weights perform better. Any such outcome is reported as
   a result.
6. **Generalization of imitation controllers.** Imitation controllers may still fail on unseen tracks
   despite the recovery data. Any such failure is reported as an experimental result.
7. **Hyperband budget.** Budget matching for Hyperband is approximate, because its bracket schedule
   determines the trial count. Actual trials, epochs and time are reported for every method.
8. **Action resolution.** Five steering levels may cause oscillatory driving. Each level is a fixed
   fraction of grip, so the turning radius it produces scales with the square of speed, and steering
   is coarsest at low speed. This is checked early by requiring the quantized expert to complete
   level-4 tracks reliably.
9. **Replication scope.** Each Keras Tuner search runs once, so the variation reported for the
   searched methods covers retraining of the selected architecture but not the search itself. The
   evolutionary methods repeat the whole run for each seed.

## 15. Software Environment

- **Base environment:** Python 3.13.11 (Conda base environment), numpy 2.4.1, scipy 1.17.1,
  TensorFlow 2.21, Keras 3.13.2, matplotlib, pandas, joblib and IPython.
- **Added packages:** keras-tuner, pygame and pytest, and nbclient and nbformat for the experiment
  runner.
- **Execution:** CPU only. Worker processes call `tf.config.set_visible_devices([], 'GPU')`. Every
  reported run uses the same machine and power profile, which are recorded in
  `results/run_info.json`.
- **Version control:** the `results/` directory is excluded.
