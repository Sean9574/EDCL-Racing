# EC Racing Revision 3: Mastery Training

**Design Specification**\
ECDL Term Project · Fall 2026 · Version 1.0 · 7 October 2026

## Summary

This document specifies the third revision of EC Racing. It changes the specification in
`docs/2026-10-05-ec-racing-design.md` (version 2.0). Sections of that document that are not mentioned
here remain in force. The design specification is updated to version 3.0 once this revision has been
built.

Revision 2.0 trained every evolutionary method for a fixed budget of 200,000 evaluations on a fixed
pool of 80 training tracks. In that run, the evolved controllers completed 54 to 59 of their 60 test
races. The imitation controllers completed 9 to 30 of 60. The easiest curriculum level consisted of a
single circular track.

Revision 3 replaces the budget with a mastery criterion. Every method trains until its cars reliably
finish freshly generated tracks at every difficulty, and is then pushed to drive faster until it cannot
improve further. The revision makes six changes:

- **Fresh circuits.** Every training track is generated anew and is never reused. Tracks run in either
  direction. Medium, Hard and Extra hard use a circuit generator whose harder levels add features found
  on Formula 1 circuits: long straights into hairpins, chicanes, esses and fast sweeping corners.
- **Time trials.** Each track is a three-lap time trial with a target time. A car that finishes within
  the target is a success.
- **Mastery levels.** A population advances a level only when at least 90% of it succeeds on each
  track of a generation, for three consecutive generations. After the last level, the target time is
  tightened in steps of 5% until the population can no longer meet it.
- **NEAT.** Neuroevolution of augmenting topologies is added as an eighth method. It evolves the
  network structure together with the weights.
- **DAgger.** The imitation controllers are refined by dataset aggregation and then pushed for speed
  by an evolution strategy grown from the refined network.
- **Long runs.** Runs can take days. They save checkpoints, resume after interruption and report
  progress live in the terminal.

## 1. Objectives

The aim of revision 3 is controllers that master racing rather than controllers that perform well on a
lenient benchmark. The revision has three objectives:

1. **Reliability.** Every method trains until at least 90% of its population completes three laps
   within the target time on each freshly generated track, at every level.
2. **Speed.** After reaching the hardest level, every method is pushed to lower lap times until no
   further improvement is achieved.
3. **Fair comparison.** All methods are measured against the same criterion. The cost of each method
   is reported as the time and number of evaluations it required to reach the criterion.

Results are reported primarily as the number of test races completed and the lap times achieved.

## 2. Scope

**Changed:** track generation (§3), the success criterion, survival and level advancement (§4), the
evolutionary methods (§5), the imitation methods (§6), evaluation (§7), execution (§8) and the
visualization application (§9).

**Unchanged:** vehicle dynamics, sensors and collisions; the controller network interface (10 inputs,
15 action classes); the expert driver; the baseline network; the Keras Tuner searches and model
selection rule; the three replication seeds; and CPU-only execution.

**Removed:** the evaluation budget of 200,000 individual evaluations, the fixed pool of training
tracks (seeds 0–79), the single circular track of level 1 and the pace gate as a separate criterion,
which the time trial (§4.1) replaces.

## 3. Track Generation

### 3.1 Difficulty levels

Every map is generated from a seed. The four difficulty levels use two generators.

- **Easy (level 1)** uses the existing loop generator: control points at jittered angles and random
  radii, joined by a periodic cubic spline. It is set to produce gentle loops on a wide road.
- **Medium, Hard and Extra hard (levels 2–4)** use a new circuit generator (§3.2).

*Table 1. Generator settings. Lengths in metres.*

| Level | Generator | Control points | Point radius | Corner radii | Features | Tightest corner | Road width |
|---|---|---|---|---|---|---|---|
| 1 Easy | Loop | 4–6 | 60–110 | Spline | None | 45 | 16 |
| 2 Medium | Circuit | 6–8 | 60–170 | Slow 25–45, fast 50–80 | None | 24 | 14 |
| 3 Hard | Circuit | 9–12 | 70–220 | Slow 16–35, fast 45–90 | Main straight ≥ 170 into a hairpin; 0–1 chicane; 0–1 run of esses | 14 | 12 |
| 4 Extra hard | Circuit | 13–17 | 80–260 | Slow 10–30, fast 45–100 | Main straight ≥ 210 into a hairpin; a second hairpin; 1–2 chicanes; 1–2 runs of esses | 9 | 10 |

The values in Table 1 are those of the prototype that produced the approved sample maps. They may be
adjusted during implementation so that the expert completes every level and the generator accepts a
candidate within a few attempts.

### 3.2 Circuit generator

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

### 3.3 Direction

Each map runs clockwise or counter-clockwise with equal probability. The direction is drawn from the
map's seed, so a seed always produces the same map in the same direction. Revision 2.0 tracks all ran
counter-clockwise.

### 3.4 Seeds

*Table 2. Seed allocation.*

| Use | Seeds | Level |
|---|---|---|
| Validation | 80–99, fixed | Extra hard |
| Test | 1000–1019, fixed | Extra hard |
| Imitation dataset | 121 seeds drawn once from 1,000,000–1,999,999 (same level mix as revision 2.0) | 1–4 |
| Training (evolution, DAgger) | Drawn by each run's random generator from 2,000,000 upwards; never repeated within a run | Current level |

Validation and test maps remain fixed sets, so that every method is evaluated on the same maps. No
training procedure ever drives them. Because the generator changes, the maps behind these seeds differ
from those of revision 2.0.

## 4. Time Trials, Survival and Levels

### 4.1 Time trial

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

### 4.2 Survival

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
(§5). Because every generation drives new maps, a car remains in the population only while it keeps
succeeding on maps it has never seen.

### 4.3 Level advancement

For each map of a generation, the success share is the fraction of the whole population that
succeeded on that map. A generation passes when the lowest success share among its maps is at least
90%. Three consecutive passing generations advance the population one level. A population on Extra
hard that passes three consecutive generations has reached the mastery goal. Each generation drives
three maps, so the three passing generations cover nine different maps.

### 4.4 Speed push

After the mastery goal, the target factor is multiplied by 0.95, which shortens the target time by 5%.
Each further run of three consecutive passing generations shortens it by another 5%. The run ends
when 500 generations pass without mastering the current target. The tightest target mastered is
recorded as the run's speed result, expressed as the reduction from the starting target. For example,
f = 1.3 × 0.95⁴ is reported as a target 18.5% shorter than the starting target.

### 4.5 Safety stop

On any level before the mastery goal, a run ends if 2,000 generations pass without a new best value of
the lowest map success share, averaged over three consecutive generations. Such a run is reported as
not having reached the level. Runs are expected to end through the speed push; the safety stop
prevents a run that cannot progress from continuing indefinitely.

### 4.6 Final controller

When a run ends, every kept car of the last generation drives the 20 validation maps as a time trial
at the tightest target the run mastered (f = 1.3 if it mastered none). The car with the most
successes is the run's final controller. Ties are broken by mean fitness, which favors the faster car.
Revision 2.0 selected by mean fitness alone, which allowed a fast car that crashed on one map to win
over a car that completed every map.

## 5. Evolutionary Methods

Every evolutionary run starts at Easy and follows §4. Runs have no evaluation budget. They end through
the speed push (§4.4) or the safety stop (§4.5). Evolution operates on any dense architecture, so the
same code evolves the baseline network and the larger Keras Tuner networks (§6.4).

### 5.1 Genetic algorithm

The population size is 1,000. Children are produced as in revision 2.0: two parents are selected by
tournaments of size 5 over the whole population, combined by uniform crossover (p = 0.5) and mutated
(each weight with p = 0.1, by a Gaussian of strength s). The strength follows the revision 2.0 rule
s = s_min + (s_max − s_min)(1 − c), where c is the share of the previous generation's children that
succeeded on every map, s_max = 0.1 and s_min = s_max / 100. Fewer successful children raise the
strength and more lower it.

### 5.2 Evolution strategy

The population size is 550. Each child is a copy of one of the 50 fittest cars of the generation, chosen
at random, with Gaussian noise of standard deviation σ added to every weight. σ starts at 0.1 and
follows two rules:

- **Learning phase.** While fewer than 90% of the 50 fittest cars succeed on every map, σ follows the
  1/5 success rule of revision 2.0, with a tie counted as half a success.
- **Reliability phase.** Once at least 90% of the 50 fittest succeed, σ is multiplied by 0.82 when fewer
  than 90% of the children succeed and divided by 0.82 otherwise.

σ is kept within [1e-4, 1.0] and reset to 0.1 when the population advances a level. It is not reset
when the target time is shortened.

### 5.3 Seeded evolution strategy

The seeded ES is the ES of §5.2 with an initial population of copies of the baseline network after its
DAgger refinement (§6.3), each perturbed with noise of standard deviation 0.1. Like every evolutionary
run, it starts at Easy. The baseline's own speed push (§6.4) starts from the same network but at Extra
hard. The two therefore differ only in whether the curriculum is repeated, and the comparison shows
whether it helps a network that has already learned to drive.

### 5.4 NEAT

NEAT (neuroevolution of augmenting topologies) evolves the network structure together with its
weights. The implementation is written in numpy for this project, so that whole populations drive
together like the other methods.

**Genome.** A genome consists of node genes and connection genes. Every genome has the 10 input nodes
and the 15 output nodes of the common interface. Hidden nodes are added by mutation. Each node except
the inputs carries a bias. Each connection gene holds its source and target nodes, a weight, an enabled
flag and an innovation number. Innovation numbers are global within a run: a structural mutation that
recurs in the same generation receives the existing number, so that the genes of two parents can be
aligned. Networks are feed-forward. A connection that would create a cycle is never added, so NEAT
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
two-phase rule of the ES (§5.2), evaluated on the 50 fittest cars of the population. It is reset to 0.5
when the population advances a level.

**Speciation.** Genomes are grouped into species by the compatibility distance
δ = E / N + D / N + 0.4 W̄, where E and D are the numbers of excess and disjoint genes, W̄ is the mean
weight difference of matching genes and N is the number of genes in the larger genome (1 if both have
fewer than 20). A genome joins the first species whose representative, a random member from the
previous generation, lies within the threshold δ_t. Otherwise it founds a new species. δ_t starts at
3.0 and is adjusted by 0.1 each generation towards 20 species. A species whose best fitness has not
improved for 20 generations receives no children, unless it contains the best car of the population.
Kept cars (§4.2) remain in the population regardless of their species.

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
timing columns of the course comparison (§7.2). The parameter count reported for NEAT is the number of
enabled connections plus biases, not the size of the Keras layers.

The population size is 1,000, like the GA.

## 6. Imitation Methods

### 6.1 Dataset

The expert dataset is collected as in revision 2.0, with steering disturbances and recovery, on 121
freshly generated maps with the revision 2.0 level mix: 1 Easy, 20 Medium, 20 Hard and 80 Extra hard.
The seeds are drawn once (§3.4). The validation and test splits use the fixed validation and test
maps.

### 6.2 Initial training

The baseline and the three Keras Tuner searches are unchanged. Each search runs once with seed 0, and
its chosen configuration is trained with seeds 0, 1 and 2.

### 6.3 DAgger refinement

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

### 6.4 Speed push

DAgger can make a car no better than its teacher. After refinement, an evolution strategy population of
550 is grown from the refined network, as copies perturbed with noise of standard deviation 0.1. It
starts at Extra hard at f = 1.3 and follows the ES of §5.2 and the rules of §4: it must first reach
the mastery goal, then the target time is shortened until the population can no longer meet it. If
DAgger stopped without reaching its goal, the population must first reach the mastery goal by
evolution.

### 6.5 Stages

Each imitation method is evaluated at three stages: after initial training, after DAgger and after the
speed push. The controller after the speed push is the method's final controller.

## 7. Evaluation

### 7.1 Test procedure

Each final controller (eight methods, three seeds) drives the 20 test maps as a three-lap time trial.
Two outcomes are recorded per map: whether the car **finished** within the simulator's time limit of
twice the reference lap time per lap, and whether it **succeeded** within the starting target (f = 1.3).
The validation maps select each run's final controller (§4.6) and each method's best seed for the
application.

### 7.2 Tables

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

### 7.3 Figures

- **Success over training.** For each method, the lowest map success share per generation, with
  level advancements and target reductions marked.
- **Lap time over training.** The median lap time of the successful cars per generation.
- **Networks.** One panel per method showing its best final controller. The nine distance rays,
  labelled by angle, and the speed input are on the left. Hidden nodes are placed in columns by depth,
  so that the structure evolved by NEAT appears as it is. The 15 actions, labelled by steering and
  throttle, are on the right. Connections are blue when positive and red when negative, with width and
  opacity proportional to the weight's magnitude.
- **Step sizes and Keras Tuner trials.** Unchanged from revision 2.0.

## 8. Execution

### 8.1 Notebooks

The experiment consists of eight notebooks, executed in order by `scripts/run_notebooks.py`:

1. `01_dataset`: expert dataset (§6.1)
2. `02_imitation`: baseline and Keras Tuner searches (§6.2)
3. `03_dagger`: DAgger refinement and speed push of the imitation methods (§6.3–6.4)
4. `04_ga`: genetic algorithm (§5.1)
5. `05_es`: evolution strategy (§5.2)
6. `06_es_seeded`: seeded evolution strategy (§5.3)
7. `07_neat`: NEAT (§5.4)
8. `08_evaluate`: evaluation (§7)

### 8.2 Checkpoints and resumption

- **Checkpoints.** Every evolutionary run saves a checkpoint after each generation. It contains the
  population, the state of the random generator, the level, streak, target factor and step size, the
  seeds already used, NEAT's species and innovation records, the generation count, the number of
  evaluations and the active run time. A DAgger refinement saves a checkpoint after each round. A
  checkpoint is written to a temporary file and then renamed, so that an interruption during writing
  cannot corrupt it.
- **Resumption.** Resumption is the default behavior. When the runner is started again, completed runs
  are skipped and interrupted runs continue from their last checkpoint, losing at most one generation.
  A resumed run produces exactly the same results as an uninterrupted one. The option `--fresh`
  removes previous results and starts over.
- **Settings check.** Checkpoints and run summaries record the settings that affect results. A run
  whose recorded settings differ from the current ones is not resumed. The runner stops with a message
  naming the difference, and `--fresh` is required.

### 8.3 Progress display

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

The 24 counted runs are the 12 DAgger refinements with speed pushes and the 12 evolutionary runs.

### 8.4 Parallelism and platform

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

## 9. Visualization Application

- **Training view.** NEAT is added to the approaches. Training follows the rules of this revision and
  continues until stopped. The panel shows the level, the success share of each map against the 90%
  requirement, the streak and the current target reduction.
- **Race view.** The final controllers of all eight methods race, using each method's best seed after
  the speed push. The difficulty selector uses the generators of §3.
- **Color.** NEAT is drawn in yellow (#F0E442), the remaining color of the Okabe–Ito palette used for the
  other methods.

## 10. Verification

### 10.1 Tests

Unit tests cover every new component:

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

### 10.2 Pilot runs

Before the reported run, short pilot runs of each evolutionary method on Easy and Medium confirm that
the 90% requirement can be reached with the settings of this document. If a level is not reached in
the pilots, the survival and step-size settings are reviewed before the reported run starts.

## 11. Default Parameters

*Table 3. Parameters introduced or changed by this revision.*

| Parameter | Value |
|---|---|
| Laps per time trial | 3 |
| Starting target factor f | 1.3 × reference time |
| Success | All three laps within the target time |
| Survival | Successes kept; others kept with probability equal to their share of laps in time; at most 90% of the population; at least the 50 fittest |
| Level advancement and mastery goal | Lowest map success share ≥ 90% in three consecutive generations |
| Speed push | Target reduced by 5% per mastered target; ends after 500 generations without mastering the current target |
| Safety stop | 2,000 generations without a new best three-generation mean of the lowest map success share |
| Maps per generation | 3, freshly generated |
| GA | N = 1,000; tournament 5; crossover p = 0.5; mutation p = 0.1; s_max = 0.1; s_min = s_max / 100 |
| ES, seeded ES and imitation speed push | N = 550; children from the 50 fittest; σ₀ = 0.1; factor 0.82; σ in [1e-4, 1.0]; reliability phase at 90% |
| NEAT | N = 1,000; crossover 0.75; weight perturbation 0.8 (σ₀ = 0.5, two-phase rule as ES); weight replacement 0.05; add connection 0.05; add node 0.03; toggle 0.01; δ = E/N + D/N + 0.4 W̄; threshold from 3.0 towards 20 species; stagnation 20 generations; tanh hidden nodes |
| DAgger | 20 maps per round; at most 30 epochs per round, patience 5, 10% hold-out; goal 18 of 20 checks in three consecutive rounds; stall after 10 rounds |
| Worker processes | `EC_RACING_JOBS`, default 6 |

## 12. Risks

- **Unreachable levels.** The 90% requirement on unseen maps is demanding. A method may not reach it on
  every level. The pilot runs test this before the reported run, and the safety stop ends a run that
  cannot progress. Such a result is reported, not hidden.
- **Run time.** Runs may take days. The checkpoints and the progress display make this manageable, and
  the reported training costs include the full time.
- **NEAT cost.** The batched forward pass costs more per step than a fixed dense network when genomes
  grow large. If its cost becomes dominant, the number of hidden nodes per genome can be capped.
- **DAgger data growth.** The aggregated dataset grows with every round. Twenty rounds add on the order
  of a million rows, which fits in memory but lengthens training per round.
- **Generator acceptance.** Harder settings reject more candidates. The prototype accepted most maps
  within four attempts. Acceptance is checked across many seeds by a test.

## 13. Affected Modules

*Table 4. Modules changed or added by this revision.*

| Module | Change |
|---|---|
| `racing/track_gen.py` | Circuit generator, gentle Easy loops, random direction |
| `racing/sim.py` | Time-trial end at the target time; laps completed in time |
| `racing/population.py` | Time-trial evaluation per map; shared survival rule; final selection by successes |
| `racing/curriculum.py` | Fresh seeds; level tracker with mastery goal, speed push and safety stop |
| `racing/ga.py`, `racing/es.py` | No budget; shared survival; two-phase σ; any dense architecture; checkpoints |
| `racing/neat.py` (new) | Genome, operators, speciation, batched forward pass, Keras export |
| `racing/dagger.py` (new) | DAgger rounds and the imitation speed push |
| `racing/checkpoint.py` (new) | Atomic checkpoints, settings record, resumption |
| `racing/progress.py` (new) | Status file, live line, milestones, separate monitor |
| `racing/nets.py` | Drivers for dense and NEAT controllers |
| `racing/expert.py` | Dataset on freshly generated maps |
| `racing/evaluate.py` | Tables and figures of §7, including the network figure |
| `racing/app.py` | NEAT, the new rules and status in the training view |
| `racing/config.py` | Parameters of Table 3; `EC_RACING_JOBS` |
| `scripts/run_notebooks.py` | Resumption, `--fresh`, live progress |
| `scripts/make_notebooks.py`, `scripts/pilots.py` | Eight notebooks; pilots for this revision |
| `README.md`, `requirements.txt` | Desktop and WSL setup, pinned packages |
| `docs/2026-10-05-ec-racing-design.md` | Updated to version 3.0 after the build |
