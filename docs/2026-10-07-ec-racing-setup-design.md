# EC Racing Revision 3.1: Evolved Car Setup

**Design Specification**\
ECDL Term Project · Fall 2026 · Version 1.0 · 7 October 2026

## Summary

This document specifies revision 3.1 of EC Racing. It extends revision 3
(`docs/2026-10-07-ec-racing-mastery-design.md`) and the design specification version 3.0
(`docs/2026-10-05-ec-racing-design.md`). Everything not mentioned here remains in force. The design specification is
updated to version 3.1 once this revision has been built.

In revisions up to 3, every car has the same sensors and the same steering: nine distance rays that reach 50 m, and five
steering levels that are fixed fractions of the grip limit. Revision 3.1 lets evolution choose both. Each car carries two
setup values next to its network: its **sensor length** and its **steering sensitivity**. They are inherited and mutated
like the network, and the time trials decide which settings survive. The limits are three times the current values in
each direction, so that evolution can explore far-sighted or short-sighted cars and fine or coarse steering. The choices
carry no cost: the only measures of a setup are the success rate and the lap times it produces.

The revision also adds a grip selector to the visualization application, so that the same cars can be watched under
normal grip and under a much higher grip.

## 1. Objectives

1. **Open questions to evolution.** Determine which sensor length and steering sensitivity the populations of each
   method select when they are free to choose.
2. **Speed and reliability first.** The setup is judged only by the existing criteria: success on fresh maps and lap
   time.
3. **Comparability.** Every evolutionary population evolves its setup under the same rule, and the evaluation reports
   the setups the final controllers reached.

## 2. Setup Values

*Table 1. Setup values of a car.*

| Value | Symbol | Range | Revision 3 value |
|---|---|---|---|
| Sensor length | L | 10–150 m | 50 m |
| Steering sensitivity | k | ⅓–3 | 1 |

Both values are stored and mutated on a logarithmic scale, so that halving and doubling are equally large changes.

### 2.1 Sensor length

All nine rays of a car reach at most L metres. A ray that meets no road edge within L reports L. The network inputs keep
the revision 3 scale: a ray input is the measured distance divided by 50 m. A ray of a car with L = 150 m can therefore
report values up to 3, and a car with L = 50 m sees exactly the inputs of revision 3. Networks trained with the fixed
50 m sensors keep their meaning when they are later evolved.

The ray tracer runs enough iterations for the longest permitted ray. Rays that have hit an edge or reached their car's
length stop early, so the additional cost falls only on cars with long sensors.

### 2.2 Steering sensitivity

The steering of an action class is multiplied by the car's sensitivity before the grip limit is applied: the lateral
acceleration request is clip(k × s, −1, 1) × grip for the class's steering level s, and the friction circle applies as
before. A sensitivity below 1 gives finer steering but limits the sharpest turn to k × grip. A sensitivity above 1 cannot
exceed the grip limit; it makes the half-steering levels stronger and saturates the full levels, so the steering becomes
coarser. The throttle levels are unchanged.

### 2.3 Unchanged

The expert driver, its reference times, the target times, the time limits, the vehicle dynamics and the grip limit used
in training and evaluation are unchanged. The expert keeps its sight of 50 m.

## 3. Evolution of the Setup

### 3.1 Which populations

Every evolutionary population evolves its setup: the GA, the ES, NEAT, the seeded ES and the speed push of each imitation
method. The imitation training and the DAgger refinement use the revision 3 setup (L = 50 m, k = 1), because the expert
dataset and the expert's corrections are recorded with those sensors.

### 3.2 Initial setups

- **Populations that start from random networks** (GA, ES, NEAT): every car draws its setup uniformly on the
  logarithmic scale within the ranges of Table 1, so that evolution starts from a wide spread of setups.
- **Populations that start from a trained network** (seeded ES, speed pushes): every car starts with L = 50 m and k = 1,
  the setup the network was trained with.

### 3.3 Inheritance and mutation

The same rule applies to every method, so that setup evolution does not favour one method over another.

- A kept car keeps its setup unchanged.
- A child inherits each setup value from a parent: in the GA, each value from either crossover parent with equal
  probability; in the ES, from its parent; in NEAT, from the fitter parent of a crossover or from the single parent.
- Each inherited value is then mutated with probability 0.2 by adding Gaussian noise with standard deviation 0.1 on the
  natural-logarithm scale (about ±10%), and clipped to its range.

The setup mutation is independent of the network step sizes (GA strength, ES and NEAT σ), which keep their revision 3
rules.

## 4. Saved Controllers

A final controller of an evolutionary run stores its setup next to its network in `<seed>.setup.json` (sensor length
and steering sensitivity). A controller without this file uses the revision 3 setup; this applies to the imitation
stage files. The evaluation, the network figure and the application drive every controller with its own setup.

## 5. Evaluation

- **Results table.** Two columns are added for each method: the final controllers' sensor length and steering
  sensitivity (mean ± standard deviation across seeds).
- **Setup over training.** A new figure shows, for each method, the population's mean sensor length and mean steering
  sensitivity per generation, with level advancements and target reductions marked.
- **Logs.** Each generation's log row records the population's mean and standard deviation of both setup values.
- **Network figure and network window.** Titles include the controller's setup.

## 6. Visualization Application

- **Grip selector.** A selector with the options **Normal** (the training grip, 12 m/s²) and **Unlimited** (10 times the
  training grip, 120 m/s²) sets the grip of everything the application simulates for display: the race view and the
  replays in the training view. The selector never affects training, which always uses the training grip, so the
  reported results cannot depend on it. A grip of 120 m/s² stands in for unlimited grip, because an unbounded grip would
  let a car turn arbitrarily far within one time step.
- **Rays.** When the sensor rays are shown, each car's rays are drawn to its own sensor length.
- **Status.** The training panel shows the population's mean sensor length and steering sensitivity.

## 7. Settings and Execution

- The setup ranges, the initial-setup rule, the mutation probability and the mutation size are recorded settings, so a
  run is not resumed with different values.
- The ray tracer's iteration limit is raised for 150 m rays. This makes a generation slower for populations with long
  sensors; results do not depend on it.
- The pilots of revision 3 are repeated with this revision before the reported run, because setup evolution changes the
  dynamics of every method.

## 8. Verification

- **Sensors.** A ray stops at its car's length; a car with L = 50 m produces exactly the revision 3 inputs; inputs scale
  by 50 m; long rays reach 150 m within the iteration limit.
- **Steering.** k scales the steering levels, the grip limit and friction circle still apply, and k = 1 reproduces
  revision 3 exactly.
- **Evolution.** Initial setups follow §3.2; kept cars keep their setup; inheritance and mutation follow §3.3 (rates and
  sizes tested statistically); values stay within range; checkpoints restore setups exactly; results do not depend on the
  worker count.
- **Saved controllers.** `<seed>.setup.json` round trip; a missing file means the revision 3 setup.
- **Evaluation and application.** Setup columns and figure; grip selector applied to display simulations only.
- **Regression.** A simulation in which every car has L = 50 m and k = 1 produces exactly the revision 3 inputs,
  controls, trajectories and results.

## 9. Default Parameters

*Table 2. Parameters introduced by this revision.*

| Parameter | Value |
|---|---|
| Sensor length range | 10–150 m (revision 3: 50 m) |
| Steering sensitivity range | ⅓–3 (revision 3: 1) |
| Input scale for rays | 50 m (unchanged) |
| Initial setup | Log-uniform within range for random-start populations; 50 m and 1 for trained-start populations |
| Setup mutation | Probability 0.2 per value; Gaussian, standard deviation 0.1 on the natural-log scale; clipped |
| Ray tracer iterations | Enough for 150 m rays |
| Application grip | Normal 12 m/s²; Unlimited 120 m/s² (display only) |

## 10. Risks

- **Run time.** Long sensors cost tracing time. The run may become noticeably slower if populations select long
  sensors; this is accepted.
- **Level advancement.** Random initial setups include poor ones (very short sensors, very low sensitivity), which may
  slow the first level. The pilots measure this before the reported run.
- **Coarse steering at k > 1.** Populations may settle at saturated steering and lose smoothness. This is a result to be
  reported, not a defect.

## 11. Affected Modules

| Module | Change |
|---|---|
| `racing/config.py` | Setup ranges, mutation values, application grips, ray iteration limit |
| `racing/sim.py` | Per-car sensor length and steering sensitivity; grip as a parameter of the step (default: training grip) |
| `racing/setup.py` (new) | Setup arrays: initial draws, inheritance, mutation, clipping, state |
| `racing/nets.py`, `racing/neat.py` | Populations and drivers carry their setup; saved setup files |
| `racing/population.py`, `racing/evolution.py` | Evaluation with each car's setup; log columns |
| `racing/ga.py`, `racing/es.py`, `racing/neat.py` | Setup inheritance and mutation in each method |
| `racing/dagger.py` | Speed push starts at the revision 3 setup |
| `racing/evaluate.py`, `racing/netfig.py`, `racing/netwindow.py` | Setup columns, setup figure, titles |
| `racing/app.py` | Grip selector, per-car rays, setup in the training panel |
| `scripts/pilots.py`, `README.md`, `docs/2026-10-05-ec-racing-design.md` | Pilots, viewer notes, design specification 3.1 |
