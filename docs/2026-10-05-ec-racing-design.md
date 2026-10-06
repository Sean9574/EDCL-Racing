# EC Racing: Design Specification

**Evolutionary and Gradient-Based Search for Neural Network Driving Controllers**\
ECDL Term Project · Fall 2026 · Version 1.0 · 5 October 2026

## Summary

This document specifies EC Racing, a top-down two-dimensional racing simulator. It is used to compare
methods for finding small neural network driving controllers. Six controllers compete on procedurally
generated tracks.

- **Imitation learning (four controllers).** These are trained by supervised imitation of a scripted
  expert driver. One is a hand-designed baseline. In the other three, Keras Tuner selects the
  architecture using random search, Hyperband or Bayesian optimization.
- **Neuroevolution (two controllers).** A genetic algorithm (GA) and a (μ+λ) evolution strategy (ES)
  optimize the network weights directly from driving performance. Both evolve large populations and
  follow a curriculum of progressively harder tracks.

All controllers are evaluated on held-out tracks for classification accuracy, network size, inference
time and driving performance. A visualization application shows training in progress and races between
the final controllers.

## 1. Introduction

### 1.1 Research question

On tracks not seen during training, do controllers evolved through simulated driving outperform
controllers that are trained to imitate an expert and whose architectures are tuned by conventional
hyperparameter optimization?

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
imitation.py                  │                              ▲
(baseline + KT Random/        ▼                              │
 Hyperband/Bayesian)     final weights ──► Keras copy ───────┤
   │                                                         │
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
│   ├── track_gen.py    curriculum levels, seeds, validation, road grid
│   ├── sim.py          vehicle dynamics, sensors, collisions, progress, fitness
│   ├── nets.py         action table, batched numpy forward pass, Keras export
│   ├── expert.py       scripted expert driver and dataset collection
│   ├── imitation.py    baseline, three Keras Tuner searches, model selection
│   ├── curriculum.py   level-advancement rule, GA mutation-strength rule
│   ├── ga.py           genetic algorithm
│   ├── es.py           (μ+λ) evolution strategy
│   ├── evaluate.py     test-track evaluation, tables, figures
│   └── app.py          pygame application: training view and race view
├── notebooks/          01_dataset, 02_imitation, 03_ga, 04_es, 05_evaluate
├── tests/
├── results/            (excluded from version control) datasets, logs, checkpoints, models, tables, figures
└── docs/               design specification and implementation plan
```

*Figure 2. Repository layout.*

- Each module has a single responsibility.
- For readability, the code uses plain loops and explicit conditionals rather than abstraction layers.
- Notebooks run the reported experiments with fixed settings by calling into the `racing` package.

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
up to 50 m. Rays are cast by sphere tracing on the distance-to-boundary map, using at most 12 steps.

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
the same way for all six approaches.

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

### 7.3 Dataset composition

Each row consists of the 10 inputs and one class label.

- **Training:**
  - level 1 once
  - levels 2–3 on seeds 0–19
  - level 4 on seeds 0–79
- **Validation:** level 4, seeds 80–99.
- **Test:** level 4, seeds 1000–1019.

The expected total is approximately 100,000–200,000 rows.

## 8. Imitation-Learning Controllers

### 8.1 Baseline

The baseline is the 431-parameter network from §6, trained with:

- Adam at a learning rate of 1e-3
- batch size 256
- sparse categorical cross-entropy
- at most 50 epochs, with early stopping on validation accuracy (patience 5)

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

- Random search and Bayesian optimization: 30 trials each.
- Hyperband: `max_epochs = 50` and `factor = 3`, with `hyperband_iterations` chosen so its total
  training epochs approximately match the other two.

The number of trials, total epochs and wall-clock time are reported for every method.

### 8.3 Model selection

For each method, the selected model is the smallest network (fewest parameters) whose validation
accuracy is within one percentage point of the best validation accuracy that method achieved. This
rule expresses both course objectives, accuracy and size, as a single criterion.

## 9. Evolutionary Controllers

### 9.1 Common procedure

Both evolutionary approaches use the baseline architecture (431 weights), so the learning method is
the only variable. Evolution optimizes the weights, not the network size. Initial weights are drawn
from a Glorot-uniform distribution.

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

**Level advancement:** an approach advances one level when 100% of its kept individuals finish and at
least 80% of its new individuals finish, for three consecutive generations. Each approach advances
independently. Level 4 continues until the computational budget is exhausted.

**Computational budget:** each approach receives 200,000 individual evaluations. One evaluation is one
individual driving its generation's three tracks.

- GA: 1,000 evaluations per generation, or 200 generations.
- ES: 550 evaluations per generation, or approximately 364 generations.

Wall-clock time is also reported.

**Logging:** each generation is recorded in a CSV file with:

- level and generation number
- best, mean and worst fitness
- the finishing percentages of kept and new individuals
- GA mutation strength or ES step size σ
- the track seeds used
- elapsed time

The best network of each generation is saved as an `.npy` file (about 3.4 KB). Results are written
after every generation, so closing the application or an unexpected failure loses at most the
generation in progress.

**Final controller selection:** when the budget is exhausted, the final kept individuals are evaluated
on the validation tracks (seeds 80–99). The individual with the highest mean fitness is selected and
exported as a `.keras` file. If an approach did not reach level 4, selection proceeds the same way and
the level reached is reported.

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
  settings are `s_max = 0.5` and `s_min = 0.005`, with `s = s_max` in the first generation.
  - Mutation strength is highest when few new individuals finish.
  - It falls to the minimum when all of them finish.
  - After a level advancement, `f_new` drops on the harder tracks, so mutation strength rises again
    without any special handling.

### 9.3 Evolution strategy

- **Scheme:** (μ+λ), with μ = 50 parents (the kept individuals) and λ = 500 offspring.
  - Each offspring is a randomly chosen parent with a sample from N(0, σ²) added to every weight.
  - The next generation's parents are the 50 fittest of parents ∪ offspring.
- **Step-size control:** the 1/5 success rule is applied every generation. Let `p_s` be the fraction
  of offspring that outperform their own parent:
  - if `p_s > 0.2`, then σ ← σ / 0.82
  - if `p_s < 0.2`, then σ ← σ × 0.82

  The factor 0.82 follows Schwefel. The initial step size is σ₀ = 0.1, and σ is clipped to
  [1e-4, 1.0].

The GA's finish-rate adaptation and the ES's success-rate adaptation are two different step-size
control strategies. Their behavior is compared in the results.

### 9.4 Computational performance

The simulation is vectorized over individuals, and rays use sphere tracing (§5.2). The time for one
generation is measured in the first implementation task, before any other work depends on it.

If that time is too long, the following measures are applied in order:

1. Distribute the population across worker processes (joblib). Preliminary measurements indicate that
   CPU processes scale well on the target hardware.
2. Reduce the number of sphere-tracing steps.
3. Shorten the time limits.

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

### 10.2 Training view

Training is observed live rather than recorded.

- **Controls:**
  - The user selects an approach.
  - For the GA and ES, the user sets the population size before starting.
  - Training is started and paused with a play/pause control.
- **Training process:**
  - Training runs in a separate process at full speed.
  - After each generation, completed Keras Tuner trial or baseline epoch, it sends the weights,
    fitness values, level and track seeds to the view through a `multiprocessing.Queue`.
  - Pausing halts training at the next generation boundary.
- **Playback:**
  - The view re-simulates only the displayed individuals, on the generation's first track, at 1×, 2×,
    4× or 8× speed.
  - When playback ends, the view advances to the latest completed generation and skips any in
    between.
  - Rendering never slows training.
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

- **Controllers:** the view loads whichever final controllers exist in `results/champions/`.
- **Track:** a test track (seeds 1000–1019) is selected with the arrow keys.
- **Race:** the six controllers race three laps without interacting, each in its assigned color.
- **Leaderboard:**
  - position, ranked by laps completed plus progress
  - last lap time and best lap time
  - crash status

  A crashed car stays where it crashed and is drawn faded.
- **Playback speed:** 1×, 2×, 4× or 8×.

## 11. Evaluation Methodology

- **Driving evaluation:** each final controller drives one lap on each of the 20 test tracks.
- **Accuracy evaluation:** each controller's predicted classes are compared with the expert labels in
  the test dataset. For the evolutionary controllers, this measures agreement with an expert they
  never observed, which is reported as a result in its own right.
- **Inference time:** Keras `predict` on 10,000 test rows, reported per sample and averaged over five
  runs.

*Table 4. Results tables.*

| Table | Columns |
|---|---|
| Required comparison | Method; test accuracy; parameters; inference time; search/training time |
| Driving performance | Method; tracks completed (of 20); mean lap time on completed tracks; crashes; curriculum level reached |

**Figures:**

- Fitness against individual evaluations for the GA and ES, with level-advancement markers.
- GA mutation strength and ES step size σ across generations.
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
  expected step-size behavior.
- **Curriculum:**
  - level advancement triggers on exactly the third consecutive qualifying generation, and the count
    resets after a non-qualifying generation
  - the mutation-strength rule returns `s_max` at 0%, the midpoint at 50% and `s_min` at 100%
- **Re-simulation fidelity:** re-simulating a logged generation reproduces exactly the fitness values
  recorded during training.

## 13. Default Parameters

*Table 5. Default parameters, as defined in `racing/config.py`.*

| Parameter | Value |
|---|---|
| Time step | 0.05 s |
| Grip limit μg (friction circle) | 12 m/s² |
| Speed input scaling | 40 m/s (normalization only) |
| Sensors | 9 rays from −90° to +90°, 50 m range |
| Time limit per lap | Twice the expert lap time |
| No-progress timeout | 5 s |
| Tracks per generation | 3 |
| Level advancement | Kept 100%, new ≥ 80%, three consecutive generations |
| GA | N = 1,000; K = 50; tournament size 5; crossover p = 0.5; mutation p = 0.1; s_max = 0.5; s_min = 0.005 |
| ES | μ = 50; λ = 500; σ₀ = 0.1; adaptation factor 0.82; σ in [1e-4, 1.0] |
| Evolutionary budget | 200,000 individual evaluations per approach |
| Keras training | Adam 1e-3; batch 256; at most 50 epochs; patience 5 |
| Keras Tuner | 30 trials (random search, Bayesian optimization); Hyperband matched on total epochs |
| Model selection | Smallest network within one percentage point of the best validation accuracy |
| Seeds | Training 0–79; validation 80–99; test 1000–1019 |

The track, vehicle and expert-driver values are initial estimates. They will be calibrated during
implementation so that the expert completes every curriculum level reliably.

## 14. Risks and Open Issues

1. **Course alignment.** Confirmation is needed from the course instructor that the project qualifies
   as Project 7, and that driving performance is acceptable as an evaluation metric alongside test
   accuracy.
2. **Evolutionary runtime.** Runtime has not yet been measured. Mitigations are listed in §9.4.
3. **Curriculum stalls.** The advancement criterion is strict by design. The threshold and the number
   of consecutive generations are configurable, and a stall is visible immediately in the training
   view.
4. **Generalization of imitation controllers.** Imitation controllers may still fail on unseen tracks
   despite the recovery data. Any such failure is reported as an experimental result.
5. **Hyperband budget.** Budget matching for Hyperband is approximate, because its bracket schedule
   determines the trial count. Actual trials, epochs and time are reported for every method.
6. **Action resolution.** Five steering levels may cause oscillatory driving. Each level is a fixed
   fraction of grip, so the turning radius it produces scales with the square of speed, and steering
   is coarsest at low speed. This is checked early by requiring the quantized expert to complete
   level-4 tracks reliably.

## 15. Software Environment

- **Installed:** Python 3.13.11 (Conda base environment), numpy 2.4.1, scipy 1.17.1, TensorFlow 2.21,
  Keras 3.13.2, matplotlib and pandas.
- **To be installed:** keras-tuner, pygame and pytest.
- **Execution:** CPU only. Worker processes call `tf.config.set_visible_devices([], 'GPU')`.
- **Version control:** the `results/` directory is excluded.
