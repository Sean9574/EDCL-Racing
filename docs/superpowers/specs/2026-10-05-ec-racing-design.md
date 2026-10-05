# EC Racing: Design Spec

ECDL term project (Prof. Chung, Fall 2026). Written 2026-10-05.

## 1. Goal

A top-down 2D racing simulator. Six cars race on procedurally generated tracks, and each car
is driven by a small neural network that a different method found. Sean watches; the
algorithms drive.

The paper asks one question: **on tracks none of them has seen, do networks evolved by
driving (GA, ES) beat networks trained to copy an expert, whose shape was tuned by Keras
Tuner?**

How it fits the course brief (`../projects26Fv1.pdf`):

- It includes every required model: a baseline, Keras Tuner Random / Hyperband / Bayesian,
  ES (μ+λ) with the 1/5 rule, and a GA.
- It maximizes test accuracy and minimizes network size and inference time.
- Networks are tiny, so nothing takes long to train. The output is a paper.
- It fits Project 7 ("any research-oriented project using DL, EC, and HPO").

## 2. What Sean decided vs. what is assumed

**Sean decided:**

- This is the ECDL term project.
- Each car is a driving network: ray distances in, steering and throttle out. No camera
  images and no segmentation.
- Tracks come only from a procedural generator: no real IAC track and no track editor.
- Spectator only. Nobody drives with the keyboard.
- The lineup is hybrid. Baseline + 3 Keras Tuner cars learn by copying an expert. ES + GA
  cars learn by driving, with large populations.
- Each approach has its own color.
- There is one top-down camera and no per-car views.
- Training is not recorded for replay. Instead there is a live training view with
  play/pause, a population-size setting, a cars-shown setting, and a choice of whole
  population / best / worst.
- Curriculum: start on a circle and move to harder tracks as the cars improve. It ends on
  seeded tracks, where lap time matters more than finishing.
- Level-up needs 100% of the unmutated (kept) cars to finish, and the mutated (new) cars to
  finish at or above a threshold.
- GA mutation strength follows the finish rate. Fewer cars finishing means more mutation,
  more finishing means less, and all finishing means a very, very small amount (never zero).
- Each approach levels up independently.
- Cars collide only with the track edge, never with each other, in every mode.

**Assumed by Claude (Sean can change any of these):**

- The Project 7 framing, and using a driving-based test metric alongside test accuracy.
  This needs a quick check with Prof. Chung (§15).
- Every number in §14.
- Kept cars = 50 in both GA and ES.
- The level-up threshold is 80% of new cars, held for 3 generations in a row.
- ES uses the 1/5-rule constant 0.82 from HW2.

## 3. Out of scope

- Semantic segmentation and any IAC bag data.
- Human driving, menus, a track editor, and real-world tracks.
- Car-to-car collisions.
- Recording or replaying training runs, and video export.
- A UI for browsing old generations. Each generation's best network is saved to disk, but
  v1 has no screen for viewing them.
- The GPU. These nets are too small for it to help (measured in HW3: CPU wins).

## 4. Architecture

```
track_gen ──► tracks (train / validation / test seeds, levels 1–4)
                 │
                 ▼
sim (numpy, every car stepped at once: physics, rays, crash, progress, fitness)
   │                          │                              │
   ▼                          ▼                              ▼
expert ──► dataset       ga.py / es.py                  app.py (pygame)
   │                     (population drives in sim,      Train mode: live training view
   ▼                      curriculum.py levels it up)    Race mode: 6 champions race
imitation.py                  │                              ▲
(baseline + KT Random/        ▼                              │
 Hyperband/Bayesian)     champion weights ──► Keras copy ────┤
   │                                                         │
   └──► champion Keras models ───────────────────────────────┘
                 │
                 ▼
evaluate.py ──► test tables + figures for the paper
```

Everything uses the same `sim`. A car therefore behaves identically in data collection,
evolution, the live view and the race.

### Project layout

```
Project/
├── README.md
├── racing/
│   ├── config.py       every default from §14, plus the color table
│   ├── track_gen.py    levels 1–4, seeds, validation, road grid
│   ├── sim.py          car physics, rays, crash, progress, fitness (vectorized)
│   ├── nets.py         action table, batched numpy forward pass, Keras build/export
│   ├── expert.py       scripted driver + dataset collection
│   ├── imitation.py    baseline + three Keras Tuner searches + size rule
│   ├── curriculum.py   level-up rule + GA mutation-strength formula
│   ├── ga.py           GA loop
│   ├── es.py           (μ+λ)-ES loop
│   ├── evaluate.py     champions on test tracks, tables, figures
│   └── app.py          pygame window: Train mode + Race mode
├── notebooks/          01_dataset, 02_imitation, 03_ga, 04_es, 05_evaluate
├── tests/
├── results/            (gitignored) datasets, logs, checkpoints, champions, tables, figures
└── docs/superpowers/specs/
```

Code style: plain loops and if/elif, readable over clever. Each file has one job. The
notebooks run the official experiments with fixed settings by calling into `racing/`.

## 5. Tracks and curriculum

`make_track(seed, level)` returns a `Track`. A Track holds:

- the centerline, resampled every 1 m
- the width and the left/right edges
- a 0.5 m road grid, plus a distance-to-wall map built with `scipy.ndimage.distance_transform_edt`
- the start pose and length
- the expert's lap time

| Level | Shape | Size | What it teaches |
|---|---|---|---|
| 1 | Circle | radius 50 m, width 16 m | Hold a steady turn |
| 2 | Oval: two straights + two half-circles | straights 80–160 m, turn radius 25–40 m, width 14 m | Brake before corners |
| 3 | Easy seeded loop: 6–8 control points | radius 80–160 m, min corner radius 40 m, width 14 m | Handle a varied layout |
| 4 | Full seeded loop: 10–16 control points | radius 60–200 m, min corner radius 15 m, width 12 m | Go fast |

Seeded loops place control points at sorted random angles and radii, then join them with a
periodic cubic spline.

**Validation.** A track is rejected if any of these hold:

- the road overlaps itself
- a corner is tighter than the level's minimum radius
- the expert fails to finish it

A rejected seed retries deterministically with `(seed, attempt)`, so a given seed always
produces the same track.

**Seeds:**

- Training: 0–79, at every level. Level 1 is the same circle for every seed.
- Validation: 80–99, level 4.
- Test: 1000–1019, level 4. No approach ever trains on these.

## 6. Simulator

- **Car model:** a kinematic bicycle model with fixed `dt = 0.05 s` (20 Hz).
  - State: x, y, heading, speed.
  - Speed: gas accelerates at 6 m/s², brake slows at 12 m/s², and coasting loses 1 m/s².
    Top speed is 40 m/s.
  - Steering: up to 25° on a 3 m wheelbase.
- **Grip limit:** yaw rate is capped at `a_lat_max / v`, with `a_lat_max = 12 m/s²`. A fast
  car can't turn tightly, so it has to brake for corners.
- **Sensors:** 9 rays at −90°, −67.5°, …, +90°. Each returns the distance to the wall, out
  to 50 m. Rays are cast by sphere tracing on the distance-to-wall map, at most 12 steps.
  Network inputs are the 9 distances divided by 50, plus speed divided by 40: **10 inputs**.
- **Crash:** the car's center leaves the road. A crashed car stops where it is.
- **No car-to-car collision.** Cars are ghosts to each other in every mode.
- **Progress:** each car tracks its nearest centerline point in a window running from 5 m
  behind to 30 m ahead. Progress is the furthest arc length reached. Driving backwards
  earns nothing.
- **A run ends** when any of these happens:
  - the car finishes one lap
  - the car crashes
  - progress stalls for 5 s
  - the time limit is reached: 2× the expert's lap time on that track
- **Fitness on one track:**
  - Not finished: progress fraction `p` (from 0 up to 1).
  - Finished: `1 + (1 − lap_time / time_limit)` (between 1 and 2).

  Early on, only completion separates cars. Once they finish, faster laps score higher.
- **Vectorized:** every per-car value is a numpy array of shape `(n_cars,)`. One step
  advances the whole population.
- **Deterministic:** the same weights on the same track always give the same result. Every
  computation is float64 with no randomness inside `step`. This is what lets the live view
  re-simulate exactly what training scored.

## 7. Shared network interface

- **10 inputs → 15-way softmax.** Each output class is one (steer, throttle) pair:
  - steer ∈ {−1, −0.5, 0, +0.5, +1}
  - throttle ∈ {brake, coast, gas}
  - `class = 3 * steer_index + throttle_index`

  The car takes the highest-scoring class.
- Making it a classification means test accuracy is literal. It also means every car plugs
  into the race and the viewer the same way.
- `nets.py` provides:
  - the action table
  - a batched numpy forward pass for whole populations (one `einsum` per layer)
  - `to_keras(weights)`, which builds the same network in Keras and copies the weights in

  Every champion exists as a `.keras` file, so size and inference time are measured the
  same way for all six.
- **The baseline shape**, also used by GA and ES, is 10 → 16 (relu) → 15 softmax:
  **431 weights**.

## 8. Expert and dataset

- **Steering:** pure pursuit toward a centerline point `6 m + 0.4 s × speed` ahead,
  snapped to the nearest of the 5 steer levels.
- **Speed:** the target speed is `sqrt(0.85 × a_lat_max / κ)`, capped at top speed, where κ
  is the sharpest curvature within stopping distance + 15 m. The expert picks gas if more
  than 0.5 m/s under target, brake if more than 0.5 m/s over, and coast otherwise.
- **The expert uses the same 15 actions** as the networks, so every label is an action a
  network can actually take. Its lap time sets each track's time limit.
- **Recovery data:** every 2–4 s (random), the car executes a random steer level for
  0.3–0.6 s. The label recorded is always the expert's choice for that state, which teaches
  networks how to recover after drifting off line.
- **Rows:** (10 inputs, class label).
  - Training: level 1 once (it is a single circle), levels 2–3 on seeds 0–19, and level 4
    on seeds 0–79.
  - Validation: seeds 80–99, level 4.
  - Test: seeds 1000–1019, level 4.

  Expected size is about 100k–200k rows in total.

## 9. Imitation cars (baseline + Keras Tuner)

- **Baseline:** the 431-weight network. Adam, learning rate 1e-3, batch 256, up to 50
  epochs, early stopping on validation accuracy with patience 5.
- **Keras Tuner search space** (identical for Random, Hyperband and Bayesian):

  | Choice | Range |
  |---|---|
  | Hidden layers | 1–3 |
  | Units per layer | {4, 8, 16, 32, 64} |
  | Activation | {relu, tanh} |
  | Dropout | {0.0, 0.1, 0.2} |
  | Learning rate | log-uniform 1e-4 to 1e-2 |

  The objective is validation accuracy. Training settings match the baseline.
- **Budget:**
  - Random and Bayesian: 30 trials each.
  - Hyperband: `max_epochs = 50`, `factor = 3`, with `hyperband_iterations` chosen so its
    total training epochs roughly match the other two.

  Trial counts, epochs and wall time are all reported.
- **Size rule:** each method's winner is the **smallest network (fewest weights) whose
  validation accuracy is within 1 percentage point of that method's best**.

## 10. Evolution cars (GA and ES)

### Shared by both

- Evolution uses the baseline shape (431 weights), so only the learning method differs.
  Evolution searches weights, not network size. Each car starts with Glorot-uniform weights.
- **Each generation:**
  1. Draw 3 tracks from the current level's training seeds, using the run's RNG.
  2. Every car drives all 3 tracks. Kept cars are re-scored too, because the tracks are new.
  3. A car's fitness is its mean over the 3 tracks. A car counts as "finished" only if it
     finished all 3.
- **Kept vs. new cars:**
  - Kept cars are carried over unchanged from the last generation.
  - New cars are freshly made by mutation (and by crossover, in the GA).
- **Level-up** (each approach on its own schedule): **100% of kept cars finish AND at least
  80% of new cars finish, for 3 generations in a row.** The level then goes up by 1, to a
  maximum of 4. Level 4 runs until the budget is used up.
- **Fair compute:** each approach gets 200,000 car-evaluations, where one car-evaluation is
  one car driving its generation's 3 tracks.
  - GA: 1,000 per generation → 200 generations.
  - ES: 550 per generation → about 364 generations.

  Wall time is reported too.
- **Logged every generation** (CSV):
  - level and generation number
  - best, mean and worst fitness
  - kept-finish % and new-finish %
  - GA mutation strength or ES σ
  - track seeds and wall time

  Each generation's best network is also saved as an `.npy` file (about 3.4 KB). After every
  generation the run is safe on disk, so closing the app or a crash loses nothing.
- **Champion:** when the budget runs out, the final kept cars each drive the validation
  tracks (seeds 80–99). The best mean fitness wins and is exported to `.keras`. If a run
  never reached level 4, its champion is picked the same way and the paper reports the level
  it reached.

### GA

- Population N = 1000 (adjustable). The **top K = 50 are kept** unchanged.
- **Each of the other N − K cars:**
  1. Pick two parents. Each is the winner of a tournament of 5 random cars drawn from the
     whole population.
  2. Apply uniform crossover: each weight comes from either parent with probability 0.5.
  3. Mutate: each weight, with probability 0.1, gets + N(0, s).
- **Mutation strength s** follows Sean's rule:
  ```
  s = s_min + (s_max − s_min) × (1 − f_new)
  ```
  `f_new` is the fraction of the previous generation's new cars that finished. Settings:
  `s_max = 0.5`, `s_min = 0.005` (1% of max), and `s = s_max` in generation 1.
  - When fewer cars finish, mutation goes up.
  - When all of them finish, mutation drops to a tiny floor.
  - After a level-up, `f_new` falls on the harder tracks, so mutation rises again
    automatically.

### ES: (μ+λ) with the 1/5 rule

- **μ = 50 parents** (the kept cars) and **λ = 500 children**. Each child is a random
  parent with + N(0, σ²) added to every weight. The next parents are the best 50 of
  parents ∪ children, all scored on this generation's tracks.
- **1/5 rule, applied every generation.** Let `p_s` be the fraction of children that beat
  their own parent:
  - If `p_s > 0.2`: `σ ← σ / 0.82`.
  - If `p_s < 0.2`: `σ ← σ × 0.82`.

  This is the HW2 constant. σ starts at 0.1 and is clipped to [1e-4, 1.0].
- **Paper contrast:** the GA uses Sean's finish-rate rule, and the ES uses the textbook
  1/5 success rule.

### Speed

- The simulator is vectorized over cars, and rays use sphere tracing (§6).
- **One generation's time is measured in the first build step,** before anything else
  depends on it.
- If it's too slow, the levers, in order, are:
  1. split the population across worker processes (joblib; HW3 showed CPU processes scale
     well on this laptop)
  2. fewer sphere-tracing steps
  3. shorter time limits

## 11. App (pygame, one window, one top-down camera)

### Colors

Based on the Okabe-Ito colorblind-safe palette. The same color is used everywhere: race,
live view, legend and paper plots.

| Approach | Color |
|---|---|
| Baseline | gray `#7F7F7F` |
| KT Random | orange `#E69F00` |
| KT Hyperband | sky blue `#56B4E9` |
| KT Bayesian | purple `#CC79A7` |
| ES | green `#009E73` |
| GA | vermillion `#D55E00` |

### Train mode (live training view)

- **Choose an approach, set the population size** (GA and ES, before starting), then press
  **Play / Pause**.
- **Training runs in its own process at full speed.** After each generation (or each Keras
  Tuner trial or baseline epoch), it sends the viewer that generation's weights, fitness,
  level and track seeds through a `multiprocessing.Queue`. Pause stops training at the next
  generation boundary.
- **The viewer re-simulates only the selected cars,** on the generation's first track, at a
  chosen speed (1×, 2×, 4× or 8×). When a playback ends, it jumps to the latest finished
  generation and skips any in between. Training is never slowed down by drawing.
- **Show: All / Best / Worst,** with a **cars-shown slider** (1 up to the population size)
  that keeps drawing smooth:
  - All: a fixed random sample of N cars.
  - Best: the top N.
  - Worst: the bottom N.

  The best car is outlined, and a toggle draws its rays.
- **Side panel:**
  - the approach, level and generation
  - kept-finish % and new-finish %
  - mutation strength or σ
  - best and mean fitness
  - a live chart of best and mean fitness per generation, with level-up markers
- **Keras Tuner cars:** each finished trial counts as one member of the population. It
  drives validation track 80.

### Race mode

- Loads whichever champions exist in `results/champions/`. Arrow keys choose a test track
  (seeds 1000–1019).
- The 6 champions race 3 laps as ghosts, in their colors.
- **Live leaderboard:** position (by laps + progress), last lap, best lap, and crashed or
  not. A crashed car stays where it crashed, faded out.
- Speed control: 1×, 2×, 4× or 8×.

## 12. Evaluation and paper outputs

- **Driving test:** each champion drives one lap on each of the 20 test tracks.
- **Accuracy test:** each champion's predicted classes are compared with the expert's labels
  on the test dataset. For evolved cars this measures agreement with the expert. They were
  never trained on it, which is itself a result.
- **Inference time:** Keras `predict` on 10,000 test rows, reported per sample and averaged
  over 5 runs.

| Table | Columns |
|---|---|
| Required comparison | Method · test accuracy · weights · inference time · search/training time |
| Driving | Method · tracks finished /20 · mean lap time (finished tracks) · crashes · level reached |

**Figures:**

- Fitness vs. car-evaluations for GA and ES, with level-up markers.
- GA mutation strength and ES σ over generations.
- Accuracy vs. weights for every Keras Tuner trial, with each method's chosen winner marked.

## 13. Testing (pytest)

- **Tracks:**
  - the same seed always gives the same track
  - level 1 is a circle of the specified radius
  - generated tracks never overlap themselves
  - the minimum corner radius holds
- **Simulator:**
  - zero steering drives straight
  - leaving the road is a crash
  - a full lap is detected
  - driving backwards earns no progress
  - the same inputs always give the same outputs
- **Rays:** distances match hand-calculated values for a car on the level-1 circle.
- **Expert:** finishes every validated track at every level. Any track it can't finish is
  rejected, so no car is ever scored on an impossible track.
- **Nets:** the numpy forward pass and the `to_keras` copy give identical classes on random
  inputs.
- **GA and ES operators** solve a simple test function (sphere function) and show the
  expected step-size behavior.
- **Curriculum:**
  - level-up triggers exactly on the 3rd qualifying generation in a row, and resets after a
    non-qualifying one
  - the mutation-strength formula gives `s_max` at 0%, the midpoint at 50%, and `s_min` at
    100%
- **Replay fidelity:** re-simulating a generation's cars reproduces exactly the fitness that
  training logged.

## 14. Defaults (all in `racing/config.py`)

| Setting | Value |
|---|---|
| Time step | 0.05 s |
| Top speed / gas / brake / coast drag | 40 m/s / 6 / 12 / 1 m/s² |
| Max steer / wheelbase / lateral grip | 25° / 3 m / 12 m/s² |
| Rays | 9, from −90° to +90°, 50 m range |
| Time limit per lap | 2× expert lap time |
| Stall timeout | 5 s |
| Tracks per generation | 3 |
| Level-up | kept 100%, new ≥ 80%, 3 generations in a row |
| GA | N = 1000, K = 50, tournament 5, crossover p = 0.5, mutation p = 0.1, s_max = 0.5, s_min = 0.005 |
| ES | μ = 50, λ = 500, σ₀ = 0.1, 1/5-rule constant 0.82, σ in [1e-4, 1.0] |
| Budget per evolution approach | 200,000 car-evaluations |
| Keras training | Adam 1e-3, batch 256, ≤ 50 epochs, patience 5 |
| Keras Tuner | 30 trials (Random, Bayesian); Hyperband matched on total epochs |
| Size rule | smallest network within 1 percentage point of the best validation accuracy |
| Seeds | train 0–79, validation 80–99, test 1000–1019 |

The track, physics and expert numbers are starting values. They get tuned during the build
until the expert drives every level cleanly.

## 15. Risks and open questions

1. **Course fit.** Confirm with Prof. Chung that this counts as Project 7, and that a
   driving test (tracks finished, lap time) is acceptable alongside test accuracy.
2. **Evolution speed is unmeasured.** Measuring it is the first build task, and the levers
   are listed in §10.
3. **Level-up may stall.** The rule is strict by design. The threshold and the
   3-in-a-row count are settings, and the live chart shows a stall immediately.
4. **Copy-the-expert cars may still crash** off-line despite the recovery data. That is a
   finding for the paper, not a bug.
5. **Hyperband budget matching** is approximate, because its trial count is set by its
   bracket schedule. Actual trials, epochs and time are reported for every tuner.
6. **Five steering levels** might make driving too jerky. This is checked early: the
   discretized expert must finish level-4 tracks cleanly.

## 16. Environment

- Conda `base`: Python 3.13.11, numpy 2.4.1, scipy 1.17.1, TensorFlow 2.21 / Keras 3.13.2,
  matplotlib, pandas.
- **To install:** `keras-tuner`, `pygame`, `pytest`.
- CPU only. Worker processes call `tf.config.set_visible_devices([], 'GPU')`.
- `results/` is gitignored.
