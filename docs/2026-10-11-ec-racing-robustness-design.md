# EC Racing Revision 3.2: Robust Controllers and a Stronger Speed Push

**Design Specification**\
ECDL Term Project · Fall 2026 · Version 1.0 · 11 October 2026

## Summary

This document specifies revision 3.2 of EC Racing. It extends revision 3.1 (`docs/2026-10-07-ec-racing-setup-design.md`),
revision 3 (`docs/2026-10-07-ec-racing-mastery-design.md`) and the design specification version 3.1
(`docs/2026-10-05-ec-racing-design.md`). Everything not mentioned here remains in force. The design specification is
updated to version 3.2 once this revision has been built.

The controllers of revision 3.1 are specialists in the road width they were trained on. On the 20 validation maps rebuilt
with a 14 m road instead of 10 m (same centerline, same time limit; the expert drives them at the same lap times), the
final controllers of the imitation methods finished 0 to 3 maps of 20, and GA seed 0, which climbed the curriculum through
four widths, finished 13. A trace shows why: a controller holds the position at which its side rays read what they read
on a 10 m road, which on a wider road lies close to a wall; it weaves, brakes later and misses corners.

Revision 3.2 makes the controllers robust to the road width in two ways: every map has its own width, drawn from 10 to
16 m, and the networks receive two inputs that mean the same at any width, their position across the road and the road's
width. It also strengthens the speed push, which ended every revision 3.1 run at a target that 90% of the population
could not reach: targets now tighten in 2% steps and are mastered when 75% of the population beats them. The evaluation
adds a robustness table at fixed widths, two of which no controller was trained on.

All results of revision 3.1 depend on the changed settings, so the experiment starts over. The revision 3.1 results are
archived, not deleted, and more of the experiment trains side by side on the desktop and the server.

## 1. Objectives

1. **Robustness to the road width.** Controllers drive maps of any width in the trained range reliably, and degrade
   gracefully beyond it.
2. **Faster controllers.** The speed push keeps tightening the target while most of the population keeps up, instead of
   stalling at a fixed 5% step that 90% of the population must master.
3. **Comparability.** Every method trains on the same maps, with the same inputs and rules; the main comparison remains
   which method drives best, and the robustness table compares how each copes with widths it never saw.

## 2. Road Widths

### 2.1 One width per map

Every map, at every level, has its own road width w, drawn uniformly from WIDTH_RANGE = 10–16 m by
`default_rng([seed, level, WIDTH_SALT])`. The width depends only on the map's seed and level, so a map is always the same,
and it does not depend on the generation attempt: a map whose geometry is rejected gets a new shape on the next attempt
but keeps its width. The fixed per-level widths of revisions up to 3.1 (Easy 16, Medium 14, Hard 12, Extra hard 10 m)
are removed; the levels now differ only in their corners and features.

### 2.2 Generation

The generator already sizes the offsets of chicanes and esses by the width and requires separate parts of the road to
stay a road width plus OVERLAP_MARGIN (6 m) apart; both use the map's own width. The expert's lap time, and therefore the
target times, are measured on each map at its width. A slow test checks that every fixed map builds: the dataset maps,
the validation maps 80–99, the test maps 1000–1019, and the first 200 training maps of every level.

### 2.3 Unchanged

Map shapes per level (corner radii, features, minimum corner radius), the grip, 3 laps, the time limits and the
centerline spacing are unchanged.

## 3. Network Inputs

*Table 1. Network inputs of revision 3.2 (12, previously 10).*

| Index | Input | Scale |
|---|---|---|
| 0–8 | Ray distances, each car's own sensor length (revision 3.1 §2.1) | ÷ RAY_RANGE (50 m), unchanged |
| 9 | Speed | ÷ SPEED_SCALE (40 m/s), unchanged |
| 10 | **Position across the road** | −1 at the left edge, 0 on the centerline, +1 at the right edge |
| 11 | **Road width** | ÷ WIDTH_SCALE (20 m): 0.5 to 0.8 in training |

- **Rays stay in metres.** Braking depends on the real distance to the wall ahead at a given speed; dividing the rays
  by the width would blur that cue.
- **Position across the road** is the car's signed distance from its matched centerline point (the point the simulator
  already tracks for progress), along the road's normal, divided by half the width, positive to the right, clipped to
  [−1, 1]. It tells the network where it is on the road in terms that do not depend on the width.
- **Road width** lets a network interpret its rays: a side ray of 7 m means the middle of a 14 m road but the edge of a
  20 m one.
- Rows of cars that are done stay zero, as before. The expert's dataset, imitation training, DAgger and every
  evolutionary method read the same sensing, so all of them receive the new inputs. NEAT's initial genomes connect the
  12 inputs as they connected 10.

## 4. Speed Push

*Table 2. Speed push rules.*

| Rule | Revision 3.1 | Revision 3.2 |
|---|---|---|
| Target step after a mastery (SPEED_STEP) | × 0.95 | × 0.98 |
| Share of the population that must beat the target (SPEED_MASTERY_FRACTION, new) | 90% | 75% |
| Consecutive generations | 3 | 3 |
| Generations without a mastery that end the push (SPEED_PATIENCE) | 500 | 300 |

The curriculum levels keep MASTERY_FRACTION = 90%. In the speed push, a generation passes when the lowest map success
share is at least 75%. The final controller is still chosen from the kept cars of the last generation by its successes on
the 20 validation maps at the last mastered target (revision 3 §4.6), so a faster target cannot hide unreliable driving.
The patience is shorter because targets now tighten in smaller, more frequent steps; without it a run would spend
proportionally longer confirming that it cannot get faster.

Fitness is unchanged. GA's tournaments, the ES's parent selection and the survival rule select by rank, and a finished
car already outranks an unfinished one, a faster finisher a slower one; a larger weight on the finish time would change
little but NEAT's offspring shares. The pressure for speed comes from the target, which is the time limit.

## 5. Evaluation

### 5.1 Main results

The test maps 1000–1019 are driven at their own widths (§2.1), at Extra hard. Every table of the design specification
§11 is produced as before.

### 5.2 Robustness table

Each test map is rebuilt with the same centerline at the fixed widths of ROBUSTNESS_WIDTHS = 8, 10, 14 and 18 m; 8 and
18 m lie outside the trained range. Each build is driven exactly as a test map (revision 3 §7.1): a car finishes within
twice the expert's lap time per lap on that build and succeeds if it finishes within the target time on that build. Every
final controller drives every build; the table reports, per method and width, the maps finished and succeeded (per seed
and in total, of 60) and the mean lap time on finished maps. A build whose road would overlap another part of the
map (the generator's spacing check) is left out of its width's column, and the table states how many maps each column
covers.

## 6. Execution

### 6.1 Starting over

The runner gains `--archive NAME`, a step of its own: it moves what `--fresh` would delete to `results/archive/NAME/`
and exits without running anything, so the next start begins afresh. The revision 3.1 results are archived as
`revision3.1`. Pilots, viewer runs and earlier archives stay where they are.

### 6.2 Imitation runs side by side

`racing/train.py` also trains the runs of notebook 03: for Baseline and the three Keras Tuner methods it makes notebook
03's call, DAgger refinement and speed push with the expert dataset, and a test checks that the calls match
the notebook's, as for notebooks 04 to 07. `run_notebooks.py --runs` therefore accepts every method; a DAgger run needs
the dataset and its imitation network (notebooks 01 and 02) and a seeded ES run its refined baseline (notebook 03); §6.3
says how the runner waits for them.

### 6.3 Everything at once, waiting only for inputs

Both machines work from the start. The side-by-side runner (`run_notebooks.py --runs`) takes a list of jobs: training
runs (`approach:seed`) and the notebooks 01 and 02 (`nb:01`, `nb:02`, executed as the runner executes any notebook, with
their outputs saved). A job starts as soon as a slot is free and its inputs exist; a job waiting for an input occupies no
slot. The inputs are the records of the steps that produce them:

*Table 3. Inputs of the jobs.*

| Job | Waits for |
|---|---|
| `nb:01` (dataset) | nothing |
| `nb:02` (imitation, searches) | the dataset record `results/dataset_info.json` |
| `baseline:k`, `kt_*:k` (DAgger and speed push) | the run's imitation record, `runs/<method>/seed<k>/summary.json` |
| `es_seeded:k` | the refinement record of Baseline seed k, `runs/baseline/seed<k>/dagger.json` |
| `ga:k`, `es:k`, `neat:k` | nothing |

Jobs are started in the order of the list, so the longest runs go first. Before anything starts, the runner refuses a job
whose input neither exists nor is produced by a job in its list (for example a seeded ES run on a server, whose refined
baseline only the desktop makes), naming the job and the input. If a producing job fails, the jobs waiting for it are
reported and the others continue.

The fleet plan (`scripts/fleet.toml`) gives the desktop the notebooks, the 12 DAgger runs, the seeded ES runs and two
NEAT runs, four at a time, and the server the remaining NEAT run, GA and ES, three at a time. The split balances the
expected work from the run lengths of revision 3.1: each machine needs about 20 hours. `fleet.py start` starts both;
`fleet.py finish` collects the server's runs, checks that all 24 runs are complete and runs notebooks 03 to 08 on the
desktop, which find every run complete and only report it.

## 7. Visualization Application

- The map line of the panel names the road width, for example "Map: seed 1004, level 4, road 13.2 m".
- Controllers and viewer runs of revision 3.1 have 10 inputs. Loading alone would not reveal this, so the race view
  checks a saved controller's input count when it loads it, leaves a controller without 12 inputs out of the race and
  names it in the status line. A saved viewer run of revision 3.1 cannot be resumed (its settings differ; existing
  behaviour: New run starts over).
- Nothing else changes; the display grip and the camera work as before.

## 8. Settings

*Table 4. New and changed settings in `racing/config.py`.*

| Setting | Value | Meaning |
|---|---|---|
| WIDTH_RANGE | (10.0, 16.0) m | Range of each map's road width |
| WIDTH_SALT | 13 | Random generator of a map's width: `default_rng([seed, level, WIDTH_SALT])` |
| WIDTH_SCALE | 20.0 m | Input scale of the road width |
| N_INPUTS | 12 | 9 rays, speed, position across the road, road width |
| SPEED_STEP | 0.98 | Target factor multiplier per mastered target in the speed push |
| SPEED_MASTERY_FRACTION | 0.75 | Lowest map success share for a passing generation in the speed push |
| SPEED_PATIENCE | 300 | Generations without a mastered target that end the speed push |
| ROBUSTNESS_WIDTHS | (8.0, 10.0, 14.0, 18.0) m | Widths of the robustness table |
| LEVELS | without `width` | Per-level widths removed (§2.1) |

All of these affect results and are recorded with every run (revision 3 §8.2), except ROBUSTNESS_WIDTHS, which only
affects the evaluation.

## 9. Verification

- **Widths:** a map's width lies in the range, is the same for every call and every attempt, and differs between maps;
  the generator uses it for the features and the spacing check; the slow acceptance test of §2.2.
- **Inputs:** position is 0 on the centerline, −1 and +1 at the edges, and has the right sign on a curve in both driving
  directions; the width input equals w / 20 m; rows of finished cars stay zero; networks of every method accept 12
  inputs, and NEAT genomes connect all of them.
- **Speed push:** a speed push generation passes at 75% and fails below; levels still require 90%; the target factor
  falls by 2% per mastery; the push ends after 300 generations without a mastery.
- **Evaluation:** the robustness builds keep the test map's centerline, use the fixed width and the expert's time on the
  build, and leave out builds that fail the spacing check; the table reports finishes, successes and lap times per
  method and width.
- **Viewer:** a controller with 10 inputs is left out of the race with a message; the map line names the width.
- **Execution:** `--archive` moves exactly what `--fresh` would delete, runs nothing else and refuses to overwrite an existing archive; the
  DAgger calls of `racing/train.py` match notebook 03; a job starts only when its inputs exist and then without delay,
  waiting jobs hold no slot, a job whose input nothing will produce is refused before anything starts, and the
  dependants of a failed job are reported; notebook jobs save their outputs; `finish` requires all 24 runs and runs
  notebooks 03 to 08.

## 10. Risks

- **Rejected maps.** Wider roads fail the spacing check more often. The acceptance test of §2.2 shows whether
  MAX_TRACK_ATTEMPTS (200) still suffices for every fixed map; if not, the margin of attempts is raised before training.
- **Harder narrow Easy maps.** Easy loops can now be 10 m wide. If the first level stalls for GA or NEAT from random
  networks, the pilots show it before the full run.
- **GA mutation strength.** GA_S_MAX = 0.1 was chosen by the revision 3 pilots at fixed widths. A short pilot of GA,
  ES and NEAT (300 generations each, `scripts/pilots.py`) is run before the full experiment to confirm that every
  method still climbs the levels.
- **Several Keras trainings at once.** The DAgger runs trained side by side each start TensorFlow; their threads compete
  for the desktop's processors. Their results do not depend on it; their training times are reported as context, as
  for every run trained side by side.
- **Comparability with revision 3.1.** The two experiments differ in maps, inputs and speed-push rules; revision 3.1 is
  archived for reference, not compared run by run.

## 11. Affected Modules

| Module | Change |
|---|---|
| `racing/config.py` | Settings of §8 |
| `racing/track_gen.py`, `racing/tracks.py` | Width per map (§2) |
| `racing/sim.py` | Inputs 10 and 11 (§3) |
| `racing/curriculum.py` | Speed push rules (§4) |
| `racing/checkpoint.py` | New settings recorded |
| `racing/evaluate.py`, `notebooks/08_evaluate.ipynb` | Robustness table (§5.2) |
| `racing/train.py` | DAgger runs (§6.2) |
| `scripts/run_notebooks.py` | `--archive`; every method and notebooks 01–02 in `--runs`, started when their inputs exist (§6.1–6.3) |
| `scripts/fleet.py`, `scripts/fleet.toml` | Plan of §6.3; `finish` from notebook 03 with all 24 runs |
| `racing/app.py` | Width in the map line; input count check of raced controllers (§7) |
| `README.md`, `docs/2026-10-05-ec-racing-design.md` | Revision 3.2 |
