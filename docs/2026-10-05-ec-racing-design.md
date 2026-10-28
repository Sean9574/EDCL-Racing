# EC Racing: Design Specification

**Evolutionary and Gradient-Based Search for Neural Network Driving Controllers**\
ECDL Term Project · Fall 2026 · Version 3.17 · 28 October 2026

## Summary

This document specifies EC Racing, a top-down two-dimensional racing simulator. It is used to compare
methods for finding small neural network driving controllers. Eight controllers drive a car with the top
speed of a Formula 1 car and more grip than one, on generated circuits and on real circuits, and are measured against
the pole laps of real Formula 1 drivers on circuits they never trained on. The research question asks whether, on
tracks not seen during training, controllers evolved through simulated driving outperform controllers that are
trained to imitate an expert and whose architectures are tuned by conventional hyperparameter optimization (§1.1).

The car is a point mass with the top speed of a Formula 1 car, two to three times its grip and no engine power or
drag (§5.1). Each controller is a small network that reads 18 inputs, among them nine distance rays, the car's speed
and six inputs about the track's racing line, the minimum-curvature line within the road (§4.8, §5.2), and chooses
one of 45 combinations of steering and throttle (§6). Road widths change along the lap, as on real circuits, and the
networks see their position across the road and the road's local width, so that they drive any width.

- **Imitation learning (four controllers).** These are trained by supervised imitation of a scripted
  expert driver that follows the racing line (§7.1). One is a hand-designed baseline. In the other three, Keras Tuner
  selects the architecture using random search, Hyperband or Bayesian optimization. Each is then refined by dataset
  aggregation (DAgger) and pushed for speed by an evolution strategy grown from it.
- **Neuroevolution (three controllers).** A (μ+λ) evolution strategy (ES) optimizes the weights of a fixed
  network directly from driving performance. A genetic algorithm (GA) and NEAT evolve the network structure
  together with its weights, starting from networks whose inputs are wired directly to the outputs: the GA adds and
  removes hidden units and whole layers, NEAT single nodes and connections.
- **Hybrid (one controller).** A seeded ES starts from the DAgger-refined baseline network, which has
  learned from the expert, and then optimizes the weights from driving performance alone.

The evolutionary controllers train on a curriculum of four levels: generated loops (Easy), generated circuits (Medium
and Hard) and 19 real circuits at real scale (Extra hard) (§4.2, §4.7); the speed push of each imitation method
starts at Extra hard (§8.5). Every generation drives three newly drawn maps as three-lap time trials. A generation
passes when, on each of its maps, at least 90% of the population completes the three laps within the target time;
three consecutive passing generations advance the population one level, and on Extra hard they reach the mastery
goal (§9.2, §9.4). In the speed push that follows, the target time is shortened in steps of 2% whenever half of the
population meets it, until 1,000 generations pass without mastering a target; a push that stalls mutates the
decisions about speed and turning harder until it beats its target again. The run then hardens: at its last mastered
target it trains until 90% of the population meets that target on each map again, three generations in a row, so
that its final controller comes from a population that finishes reliably. The final controller is the kept car with
the fastest mean lap among those that finish all 20 validation maps, generated circuits harder than any training
level; if none does, it is the car with the most finishes (§9.5). Every method is run four times: run 0 with seed 0,
which replays exactly, and runs 1, 2 and 3 with no seed, whose random choices come from the operating system; results
report every run, their mean and standard deviation, and each method's best run.

All final controllers are evaluated on four held-out Formula 1 circuits, Melbourne, Sakhir, Sepang and Silverstone
(§11.1). Each one's best flying lap is compared with the circuit's pole lap (Table 3, §4.8; Table 9, §11.1), and
the controllers are also measured by completion, lap time, classification accuracy, network size, inference time and
training cost. The comparison is made on a car with more grip than a Formula 1 car and no drag, by controllers that
know the circuit's racing line, as a driver who has learned the circuit does; the report states both with every
comparison.

A visualization application shows training in progress and races between the final controllers (§10). The training
view follows a reported run live, or trains a method in the application and replays every generation (§10.2). The
race view races the final controllers on a test circuit or a generated map, beside a pole-lap car that laps the
racing line in exactly the circuit's pole time (§10.3). A network window shows the network of the shown car live and,
in training, a timeline of the best network of every generation the viewer has seen (§10.7).

The experiment runs with one command per machine, `scripts/run_notebooks.py`, which trains the Keras Tuner searches and
the 32 runs as jobs side by side, each as soon as its inputs exist, and reports their progress live (§3.2).
`scripts/fleet.py` drives a fleet of two machines from one place: the desktop builds the dataset, runs the searches and
trains the imitation methods and the seeded ES, and a server trains the GA, ES and NEAT runs. Every run saves a
checkpoint after each generation and resumes from it after an interruption, losing at most one generation; a resumed
run 0 produces exactly the results of an uninterrupted one. `fleet.py end` writes an end request on every machine, after
which every run ends before its next generation and selects its final controller as at any other end (§9.4).

The results cited in this document come from the last complete experiment, whose runs ended before hardening was
added: 32 runs, all evaluated. Its final controllers finished 109 of the 128 test races (32 controllers on four
circuits); ES run 0 and Baseline run 3 finished no test circuit, and only 1 of the 32 final controllers finished all
20 validation maps. On each test circuit, each method's fastest run beat the pole lap. The controllers are fast
enough to beat the pole laps but do not finish reliably; hardening and the final pick on all 20 validation maps
(§9.4, §9.5) address this.

## 1. Introduction

### 1.1 Research question

On tracks not seen during training, do controllers evolved through simulated driving outperform
controllers that are trained to imitate an expert and whose architectures are tuned by conventional
hyperparameter optimization?

A direct measure accompanies it: on four real Formula 1 circuits not seen during training, does any final
controller lap faster than the pole lap set there in qualifying, driving a car with the top speed of a Formula 1 car
and two to three times its grip (§5.1, §11.1)?

A secondary question is whether evolution that starts from an imitation-trained controller combines
the strengths of both approaches: the expert's driving competence at the start, and the freedom to
exceed the expert's pace afterwards.

Two further questions are whether evolving the network structure together with its weights (the GA and NEAT)
finds better drivers than evolving the weights of a fixed network (the ES), and how far dataset aggregation (DAgger) and
a subsequent evolutionary speed push close the gap between imitation-trained and evolved controllers.
Every method is trained to the same mastery criterion, and the cost of reaching it is compared.

### 1.2 Alignment with course requirements

The project addresses the course project guidelines (*Suggested Term Projects*, 5 October 2026):

- All required models are included:
  - a baseline model
  - Keras Tuner with random search, Hyperband and Bayesian optimization
  - an ES (μ+λ) with the 1/5 success rule (the learning phase of its step-size rule, §9.7)
  - a GA
- The objectives are to maximize test accuracy and to minimize network size and inference time.
- Driving is measured by the number of held-out circuits completed within the target time and by lap time,
  against the pole laps of those circuits.
- All networks are small (the final controllers of the last complete experiment have at most 4,781 parameters), so
  each evaluation and training epoch is cheap; a run's cost lies in its number of evaluations, which the
  training-cost table (§11.3) reports.
- The deliverable is a research paper. The project falls under Project 7 ("any research-oriented
  project using DL, EC, and HPO").

## 2. Scope

The following are outside the scope of this project:

- Camera-based perception, including semantic segmentation.
- Human-driven cars, menu systems and track editing.
- A full Formula 1 vehicle model. Engine power, drag, gears, tyre wear, fuel load and slipstream are not modeled; the
  car has the top speed of a Formula 1 car, more grip than one, and nothing else of one (§5.1).
- Collisions between cars.
- Video export of training runs, and a record of every generation for later replay (a reported run keeps only its
  latest generation, for the training view, §10.2).
- A user interface for browsing earlier generations beyond the network window's training timeline, which holds the
  generations the viewer has trained or replayed while it is open (§10.7).
- GPU acceleration. Measurements on the target hardware show CPU execution to be faster for networks of this size.

## 3. System Architecture

### 3.1 Overview

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 434" width="100%" role="img" aria-label="Figure 1: components and data flow" style="display:block;margin:10pt 0 6pt">
<defs><marker id="figsys-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker></defs>
<text x="0" y="12" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Maps</text>
<rect x="0" y="31" width="88" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="44" y="45.78" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">track_gen.py</text>
<rect x="102" y="31" width="70" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="137" y="45.78" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">tracks.py</text>
<path d="M88,42 L102,42" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<text x="0" y="65.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673">generated levels 1–3 and validation;</text>
<text x="0" y="76.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673">19 real training, 4 test circuits</text>
<text x="186" y="12" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Simulator</text>
<rect x="186" y="19" width="302" height="46" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<rect x="193" y="26" width="94" height="32" rx="2.5" fill="#ffffff" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="240" y="38.28" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">sim.py</text>
<text x="240" y="49.74" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">numpy, reference</text>
<rect x="299" y="26" width="76" height="32" rx="4" fill="#fff7e6" stroke="#b07a1a" stroke-width="1"/>
<text x="337" y="38.6" font-family="DejaVu Sans" font-size="9.5" fill="#1d2330" text-anchor="middle">load check</text>
<text x="337" y="50.6" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">fails → numpy</text>
<rect x="387" y="26" width="94" height="32" rx="2.5" fill="#ffffff" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="434" y="38.28" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">fastsim</text>
<text x="434" y="49.74" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">compiled C++</text>
<path d="M387,42 L375,42" fill="none" stroke="#5f6673" stroke-width="1"/>
<path d="M299,42 L287,42" fill="none" stroke="#5f6673" stroke-width="1" stroke-dasharray="2.5 2" marker-end="url(#figsys-head)"/>
<path d="M172,42 L186,42" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<text x="499" y="28.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">fastsim runs the time trials and</text>
<text x="499" y="39.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">the expert's lap. When it loads, it</text>
<text x="499" y="50.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">is checked against numpy; without</text>
<text x="499" y="61.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">g++ or on any mismatch, numpy</text>
<text x="499" y="72.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">runs, with identical results</text>
<rect x="0" y="89" width="320.5" height="208" rx="3" fill="#f5f6f8"/>
<rect x="344.5" y="89" width="320.5" height="208" rx="3" fill="#f5f6f8"/>
<text x="202" y="105" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="end" font-weight="bold">Imitation learning</text>
<text x="463" y="105" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Evolution</text>
<path d="M260,65 L260,89" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<path d="M405,65 L405,89" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<rect x="212" y="119" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="260" y="133.78" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">expert.py</text>
<text x="202" y="133.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">scripted expert driver</text>
<rect x="212" y="155" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="260" y="169.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Dataset</text>
<text x="202" y="169.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">inputs and the expert's actions</text>
<path d="M260,141 L260,155" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<rect x="212" y="191" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="260" y="205.78" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">imitation.py</text>
<text x="202" y="199.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">baseline; Keras Tuner random,</text>
<text x="202" y="210.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">Hyperband and Bayesian</text>
<path d="M260,177 L260,191" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<rect x="212" y="227" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="260" y="241.78" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">dagger.py</text>
<text x="202" y="235.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">DAgger: the expert labels</text>
<text x="202" y="246.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">the car's states</text>
<path d="M260,213 L260,227" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<rect x="212" y="263" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="260" y="277.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Speed push</text>
<text x="202" y="277.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">ES grown from the refined network</text>
<path d="M260,249 L260,263" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<rect x="357" y="115" width="298" height="30" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="133.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">evolution.py</text>
<text x="453" y="127.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673">shared run loop: time trials, survival,</text>
<text x="453" y="138.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673">levels, speed push, hardening, checkpoints</text>
<rect x="357" y="155" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="169.78" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">ga.py</text>
<text x="463" y="169.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">genetic algorithm: layers and weights</text>
<rect x="357" y="191" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="205.78" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">es.py</text>
<text x="463" y="205.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">(μ+λ) evolution strategy</text>
<rect x="357" y="227" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="241.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Seeded ES</text>
<text x="463" y="241.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">ES from the DAgger-refined baseline</text>
<rect x="357" y="263" width="96" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="277.78" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">neat.py</text>
<text x="463" y="277.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">NEAT: structure and weights</text>
<path d="M308,238 L357,238" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<path d="M260,285 L260,321" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<path d="M405,297 L405,321" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<rect x="234.5" y="321" width="196" height="40" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="332.5" y="342.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Final controllers</text>
<text x="332.5" y="354.5" font-family="DejaVu Sans Mono" font-size="9" fill="#5f6673" text-anchor="middle">.keras<tspan font-family="DejaVu Sans">  ·  </tspan>.npy<tspan font-family="DejaVu Sans">  ·  NEAT genome</tspan></text>
<rect x="42.5" y="325" width="168" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="126.5" y="337.28" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">evaluate.py, netfig.py</text>
<text x="126.5" y="348.74" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">test circuits, tables, figures</text>
<rect x="454.5" y="325" width="168" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="538.5" y="337.28" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">app.py</text>
<text x="538.5" y="348.74" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">training view and race view</text>
<rect x="301.5" y="313.5" width="62" height="15" rx="7.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.8" stroke-width="1"/>
<text x="332.5" y="324.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">runs 0–3</text>
<path d="M234.5,341 L210.5,341" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<path d="M430.5,341 L454.5,341" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsys-head)"/>
<line x1="0" y1="375" x2="665" y2="375" stroke="#5f6673" stroke-opacity="0.35" stroke-width="1"/>
<rect x="0" y="382.5" width="50.61" height="14.5" rx="7.25" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.8" stroke-width="1"/>
<text x="25.3" y="393" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">runs 0–3</text>
<text x="56.61" y="393" font-family="DejaVu Sans" font-size="9" fill="#5f6673">every method runs four times: run 0 with seed 0 replays exactly, runs 1–3 are unseeded</text>
<rect x="0" y="401.5" width="104.98" height="14.5" rx="7.25" fill="#edf2f7" stroke="#4f6d8c" stroke-width="1"/>
<text x="52.49" y="412" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">desktop · 24 threads</text>
<text x="110.98" y="412" font-family="DejaVu Sans" font-size="9" fill="#5f6673">dataset, searches, imitation runs, seeded ES</text>
<rect x="327.99" y="401.5" width="97.57" height="14.5" rx="7.25" fill="#f1ecf6" stroke="#7a5c99" stroke-width="1"/>
<text x="376.78" y="412" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">server · 16 threads</text>
<text x="431.56" y="412" font-family="DejaVu Sans" font-size="9" fill="#5f6673">GA, ES and NEAT runs, collected by fleet.py</text>
<text x="0" y="430" font-family="DejaVu Sans Mono" font-size="9" fill="#5f6673">checkpoint.py<tspan font-family="DejaVu Sans"> and </tspan>progress.py<tspan font-family="DejaVu Sans"> serve every run: resumable runs and a live status line</tspan></text>
</svg>
<p><em>Figure 1. Components and data flow, with the compiled simulator's load check and numpy fallback, the four runs of every method and the two machines that train them.</em></p>
</div>

A single simulator is shared by data collection, evolutionary training, the training view and the race
view. Its compiled implementation, which runs the time trials and the expert's validation lap, gives
identical results (§5.7). A controller therefore behaves identically in every context at the normal grip (§10.5).

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 940" width="88%" role="img" aria-label="Flowchart of the whole system: start, dataset and map check, imitation lane with search, training and DAgger, evolution lane with generations, survival, levels and the safety stop (off), the shared speed push with stall boost, hardening at the last mastered target, the end of a run when it is hardened or by an end request, final controller and best run with all 20 finishes, test maps, tables and race view" style="display:block;margin:10pt auto 6pt">
<defs><marker id="figflow-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker><marker id="figflow-head-boost" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#c0392b"/></marker></defs>
<rect x="0" y="0" width="176" height="34" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="88" y="15.03" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">fleet.py start</text>
<text x="88" y="26.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">every job of both machines at once</text>
<path d="M196.74,17 L207.24,0 L350.76,0 L361.26,17 L350.76,34 L207.24,34 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="279" y="14.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">Compiled simulator builds</text>
<text x="279" y="26.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">and matches numpy?</text>
<path d="M176,17 L196.74,17" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M361.26,17 L373.26,17" fill="none" stroke="#5f6673" stroke-width="1"/>
<path d="M373.26,8 L373.26,26" fill="none" stroke="#5f6673" stroke-width="1"/>
<path d="M373.26,8 L385.26,8" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M373.26,26 L385.26,26" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<rect x="385.26" y="1" width="98.86" height="14" rx="7" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="434.69" y="11.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">yes: compiled C++</text>
<rect x="385.26" y="19" width="137.5" height="14" rx="7" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="454.01" y="29.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">no: numpy, identical results</text>
<text x="532.76" y="13" font-family="DejaVu Sans" font-size="9" fill="#5f6673">checked in every job</text>
<text x="532.76" y="24" font-family="DejaVu Sans" font-size="9" fill="#5f6673">when it loads</text>
<path d="M150,34 L150,52" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M150,43 L572,43 L572,52" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<circle cx="150" cy="43" r="2" fill="#5f6673"/>
<rect x="0" y="52" width="314" height="136" rx="3" fill="#f5f6f8"/>
<text x="10" y="66" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Dataset</tspan><tspan font-size="9" fill="#5f6673"> · nb:01</tspan></text>
<rect x="256.1" y="57" width="47.9" height="14" rx="7" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.9" stroke-width="1"/>
<text x="280.05" y="67.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">desktop</text>
<rect x="10" y="93" width="110" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="65" y="107.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Generate a map</text>
<text x="65" y="118.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">seed, level, attempt</text>
<path d="M146.05,109 L157.21,86 L290.79,86 L301.95,109 L290.79,132 L157.21,132 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="224" y="100.8" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">Road clear of itself,</text>
<text x="224" y="112.6" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">no corner too tight,</text>
<text x="224" y="124.4" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">expert completes a lap?</text>
<path d="M120,109 L146.05,109" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M224,86 L224,79 L62,79 L62,93" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="118" y="75" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no: next attempt</text>
<text x="10" y="139" font-family="DejaVu Sans" font-size="9" fill="#5f6673">rule for every map</text>
<rect x="146.05" y="146" width="157.95" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="225.03" y="160.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Expert drives it</text>
<text x="225.03" y="171.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">rows: inputs + its action</text>
<path d="M224,132 L224,146" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="229" y="142" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes</text>
<rect x="10" y="146" width="110" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="65" y="160.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Dataset</text>
<text x="65" y="171.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">79 + 20 + 4 maps</text>
<path d="M146.05,162 L120,162" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M62,178 L62,198" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<rect x="0" y="198" width="314" height="380" rx="3" fill="#f5f6f8"/>
<text x="10" y="212" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Imitation</tspan><tspan font-size="9" fill="#5f6673"> · 4 methods</tspan></text>
<rect x="256.1" y="203" width="47.9" height="14" rx="7" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.9" stroke-width="1"/>
<text x="280.05" y="213.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">desktop</text>
<rect x="11" y="222" width="190" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="106" y="236.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Keras Tuner searches</text>
<text x="106" y="247.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">random · Hyperband · Bayesian</text>
<rect x="214" y="222" width="90" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="259" y="236.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Baseline</text>
<text x="259" y="247.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">1,069 parameters</text>
<path d="M9.62,286 L20.78,268 L191.22,268 L202.38,286 L191.22,304 L20.78,304 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="106" y="283.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">smallest network within 1 point</text>
<text x="106" y="295.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">of the best validation accuracy</text>
<path d="M106,254 L106,268" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="111" y="265" font-family="DejaVu Sans" font-size="9" fill="#5f6673">once, seed 0</text>
<rect x="10" y="320" width="294" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="157" y="334.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Train the network</text>
<text x="157" y="345.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">run 0 with seed 0, runs 1–3 unseeded</text>
<path d="M106,304 L106,320" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M259,254 L259,320" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="167" y="371" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">DAgger rounds</tspan></text>
<rect x="18" y="378" width="135" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="85.5" y="392.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Drive 20 layouts</text>
<text x="85.5" y="403.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">real circuits, 3 laps</text>
<rect x="167" y="378" width="137" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="235.5" y="392.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Expert labels</text>
<text x="235.5" y="403.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">every state of the car</text>
<rect x="167" y="424" width="137" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="235.5" y="438.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Retrain</text>
<text x="235.5" y="449.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">all rows, ≤ 30 epochs</text>
<rect x="18" y="424" width="135" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="85.5" y="438.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Check 20 more maps</text>
<text x="85.5" y="449.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">time trial at f = 1.3</text>
<path d="M85.5,352 L85.5,378" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M153,394 L167,394" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M235.5,410 L235.5,424" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M167,440 L153,440" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M20.92,488 L32.08,470 L138.92,470 L150.08,488 L138.92,506 L32.08,506 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="85.5" y="485.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">≥ 18 of 20 checks,</text>
<text x="85.5" y="497.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">3 rounds in a row?</text>
<path d="M85.5,456 L85.5,470" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M21.57,538 L32.73,520 L138.27,520 L149.43,538 L138.27,556 L32.73,556 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="85.5" y="535.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">10 rounds without</text>
<text x="85.5" y="547.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">a new best check?</text>
<path d="M85.5,506 L85.5,520" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="90.5" y="515" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no</text>
<path d="M21.57,538 L8,538 L8,394 L18,394" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="12" y="467" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no: next round</text>
<text x="35.5" y="569" font-family="DejaVu Sans" font-size="9" fill="#5f6673">stop: the best network so far</text>
<rect x="192" y="470" width="112" height="86" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="248" y="492" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">ES of 1,100</text>
<text x="248" y="506" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">grown from the</text>
<text x="248" y="517.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">network, joins the</text>
<text x="248" y="529" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">loop at Extra hard,</text>
<text x="248" y="540.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">f = 1.3</text>
<path d="M150.08,488 L192,488" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="156.08" y="484" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes</text>
<path d="M149.43,538 L192,538" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="155.43" y="534" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes</text>
<rect x="347" y="52" width="318" height="526" rx="3" fill="#f5f6f8"/>
<text x="357" y="66" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Evolution</tspan><tspan font-size="9" fill="#5f6673"> · seeded ES on the desktop</tspan></text>
<rect x="614.51" y="57" width="40.49" height="14" rx="7" fill="#f1ecf6" stroke="#7a5c99" stroke-opacity="0.9" stroke-width="1"/>
<text x="634.75" y="67.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">server</text>
<rect x="357" y="74" width="298" height="60" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="367" y="89" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330">Initial population</text>
<text x="459.27" y="89" font-family="DejaVu Sans" font-size="9" fill="#5f6673">· each run starts at Easy, f = 1.3</text>
<rect x="367" y="95.5" width="8" height="8" rx="1.5" fill="#D55E00"/>
<rect x="377" y="95.5" width="8" height="8" rx="1.5" fill="#009E73"/>
<text x="390" y="103" font-family="DejaVu Sans" font-size="9" fill="#5f6673">GA 2,000: inputs to outputs · ES 1,100: 16 hidden units</text>
<rect x="367" y="107.5" width="8" height="8" rx="1.5" fill="#F0E442" stroke="#8a7a00" stroke-width="0.8"/>
<text x="380" y="115" font-family="DejaVu Sans" font-size="9" fill="#5f6673">NEAT 2,000: minimal genomes, inputs to outputs</text>
<rect x="367" y="119.5" width="8" height="8" rx="1.5" fill="#0072B2"/>
<text x="380" y="127" font-family="DejaVu Sans" font-size="9" fill="#5f6673">Seeded ES 1,100: copies of the refined baseline</text>
<rect x="358" y="150" width="150" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="433" y="164.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Generation: 3 maps</text>
<text x="433" y="175.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">every car drives each, 3 laps</text>
<path d="M433,134 L433,150" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M358.19,215 L367.49,200 L498.51,200 L507.81,215 L498.51,230 L367.49,230 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="433" y="218.6" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">Success on every map?</text>
<path d="M433,182 L433,200" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<rect x="540" y="200" width="100" height="30" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="590" y="218.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Kept</text>
<path d="M507.81,215 L540,215" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="515.81" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes</text>
<rect x="358" y="248" width="150" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="433" y="262.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Kept by chance</text>
<text x="433" y="273.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">p = share of laps in time</text>
<path d="M433,230 L433,248" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="438" y="241" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no</text>
<rect x="358" y="298" width="282" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="499" y="312.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Children fill the rest</text>
<text x="499" y="323.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">kept: ≤ 90%, at least the 50 fittest</text>
<path d="M433,280 L433,298" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M590,230 L590,298" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M356.64,368 L367.8,350 L498.2,350 L509.36,368 L498.2,386 L367.8,386 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="433" y="365.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">≥ 90% on every map,</text>
<text x="433" y="377.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">3 generations in a row?</text>
<path d="M433,330 L433,350" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M520.64,368 L531.8,350 L636.2,350 L647.36,368 L636.2,386 L531.8,386 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="584" y="371.6" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">safety stop: off</text>
<path d="M509.36,368 L520.64,368" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="515" y="360" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">no</text>
<path d="M388.38,426 L397.68,411 L468.32,411 L477.62,426 L468.32,441 L397.68,441 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="433" y="429.6" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">Extra hard?</text>
<path d="M433,386 L433,411" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="438" y="401" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes</text>
<rect x="383" y="462" width="100" height="26" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="433" y="478.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Mastery goal</text>
<path d="M433,441 L433,462" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="438" y="455" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes</text>
<path d="M388.38,426 L321,426 L321,166 L358,166" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="324" y="421" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no: next level</text>
<circle cx="321" cy="426" r="2" fill="#5f6673"/>
<path d="M304,513 L321,513 L321,426" fill="none" stroke="#5f6673" stroke-width="1"/>
<path d="M647.36,368 L658,368 L658,166 L508,166" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="548" y="162" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no: next generation</text>
<text x="579" y="401" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">not taken:</text>
<text x="579" y="412" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">the run trains on</text>
<rect x="0" y="588" width="632" height="204" rx="3" fill="#f5f6f8"/>
<text x="10" y="602" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Speed push and hardening</tspan><tspan font-size="9" fill="#5f6673"> · both lanes, at Extra hard</tspan></text>
<rect x="383" y="614" width="100" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="433" y="628.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Target −2%</text>
<text x="433" y="639.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">f × 0.98</text>
<path d="M433,488 L433,614" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<rect x="527" y="670" width="90" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="572" y="684.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Generation</text>
<text x="572" y="695.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">3 real layouts</text>
<path d="M483,630 L572,630 L572,670" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M356.64,686 L367.8,668 L498.2,668 L509.36,686 L498.2,704 L367.8,704 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="433" y="683.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">≥ 50% on every map,</text>
<text x="433" y="695.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">3 generations in a row?</text>
<path d="M527,686 L509.36,686" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M433,668 L433,646" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<rect x="403.4" y="650" width="59.2" height="14" rx="7" fill="#fdecea" stroke="#c0392b" stroke-opacity="0.9" stroke-width="1"/>
<text x="433" y="660.2" font-family="DejaVu Sans" font-size="9" fill="#9b2c20" text-anchor="middle">boost OFF</text>
<text x="399" y="660.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">yes</text>
<path d="M199.44,686 L210.6,668 L320.4,668 L331.56,686 L320.4,704 L210.6,704 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="265.5" y="683.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">100 generations</text>
<text x="265.5" y="695.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">without mastering?</text>
<path d="M356.64,686 L331.56,686" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="344.1" y="681" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">no</text>
<rect x="196" y="614" width="139" height="32" rx="2.5" fill="#fdecea" stroke="#c0392b" stroke-opacity="0.9" stroke-width="1"/>
<text x="265.5" y="627.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Stall boost ON</text>
<text x="265.5" y="639.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">until a target is mastered</text>
<path d="M265.5,668 L265.5,646" fill="none" stroke="#c0392b" stroke-width="1" marker-end="url(#figflow-head-boost)"/>
<text x="270.5" y="660" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes</text>
<text x="10" y="622" font-family="DejaVu Sans" font-size="9" fill="#5f6673">boost: decision layer ×3,</text>
<text x="10" y="633" font-family="DejaVu Sans" font-size="9" fill="#5f6673">steering sensitivity</text>
<text x="10" y="644" font-family="DejaVu Sans" font-size="9" fill="#5f6673">mutates harder</text>
<path d="M39.44,686 L50.6,668 L160.4,668 L171.56,686 L160.4,704 L50.6,704 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="105.5" y="683.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">1,000 generations</text>
<text x="105.5" y="695.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">without mastering?</text>
<path d="M199.44,686 L171.56,686" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M196,630 L185.5,630 L185.5,686" fill="none" stroke="#c0392b" stroke-width="1"/>
<circle cx="185.5" cy="686" r="2" fill="#c0392b"/>
<text x="185.5" y="699" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">no</text>
<path d="M105.5,704 L105.5,716 L572,716 L572,702" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="400" y="728" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no: next generation</text>
<path d="M39.44,686 L14,686 L14,755 L36.5,755" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="18" y="681" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes</text>
<path d="M36.5,755 L47.5,732 L237.5,732 L248.5,755 L237.5,778 L47.5,778 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="142.5" y="746.8" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">Hardening at the last mastered</text>
<text x="142.5" y="758.6" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">target: ≥ 90% on every map,</text>
<text x="142.5" y="770.4" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">3 generations in a row?</text>
<path d="M248.5,755 L268,755 L268,724 L220,724 L220,732" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="274" y="752" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no: next generation</text>
<text x="147.5" y="789" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes: hardened</text>
<path d="M584,386 L584,583 L650,583 L650,804 L142.5,804" fill="none" stroke="#5f6673" stroke-width="1"/>
<rect x="52" y="797" width="66" height="14" rx="7" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="85" y="807.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">end request</text>
<path d="M118,804 L142.5,804" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<circle cx="142.5" cy="804" r="2" fill="#5f6673"/>
<text x="150" y="801" font-family="DejaVu Sans" font-size="9" fill="#5f6673">the run ends</text>
<rect x="0" y="816" width="665" height="124" rx="3" fill="#f5f6f8"/>
<text x="160" y="830" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Selection and test</tspan><tspan font-size="9" fill="#5f6673"> · when a run ends, then after all runs</tspan></text>
<rect x="30" y="838" width="225" height="46" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="142.5" y="853.28" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Final controller of the run</text>
<text x="142.5" y="864.78" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">fastest kept car with all 20 finishes</text>
<text x="142.5" y="876.28" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">on validation maps 80–99, else most finishes</text>
<path d="M142.5,778 L142.5,838" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M282.71,861 L293.87,838 L451.13,838 L462.29,861 L451.13,884 L293.87,884 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="372.5" y="852.8" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">Best run per method:</text>
<text x="372.5" y="864.6" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">fastest with all 20 finishes,</text>
<text x="372.5" y="876.4" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">else most, then mean fitness</text>
<path d="M255,861 L282.71,861" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<rect x="495" y="845" width="160" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="575" y="859.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Race view</text>
<text x="575" y="870.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">seeded · unseeded · best · all</text>
<path d="M462.29,861 L495,861" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<rect x="30" y="904" width="225" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="142.5" y="918.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">4 test circuits, pole laps</text>
<text x="142.5" y="929.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">real widths and 6, 10, 14, 18 m</text>
<path d="M142.5,884 L142.5,904" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<rect x="282.71" y="904" width="179.57" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="372.5" y="918.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Report tables</text>
<text x="372.5" y="929.53" font-family="DejaVu Sans Mono" font-size="9" fill="#5f6673" text-anchor="middle">r0 · r1 · r2 · r3 (mean ± sd)</text>
<path d="M255,920 L282.71,920" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M372.5,884 L372.5,904" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="377.5" y="897" font-family="DejaVu Sans" font-size="9" fill="#5f6673">best run named</text>
<text x="147.5" y="897" font-family="DejaVu Sans" font-size="9" fill="#5f6673">every run</text>
</svg>
<p><em>Figure 2. The whole system as one flowchart: every decision it makes, as a yes/no question, from the start of the jobs through the speed push and hardening to the report tables and the race view.</em></p>
</div>

### 3.2 Experiment execution

The complete experiment is executed by a single command, `scripts/run_notebooks.py`. It executes the
eight notebooks in order and saves their outputs into the notebook files, so the notebooks form the
record of the reported run.

- **Order:** 01 dataset → 02 imitation → 03 DAgger and speed push of the imitation methods → 04 GA →
  05 ES → 06 seeded ES → 07 NEAT → 08 evaluation. The seeded ES depends on the DAgger-refined baseline
  from notebook 03. Notebooks run one at a time, so no two methods compete for the processor while
  their training time is measured. When the experiment's jobs train side by side, on one machine or on two (below),
  training times are context and the cost is compared by evaluations.
- **Runs 0 to 3.** Every method is run four times. Run 0 uses seed 0 and replays exactly. Runs 1, 2 and 3 use no seed
  at all: wherever run 0 starts a random generator from its seed (the evolution, the training of its imitation
  network, DAgger's generator and its training in every round, the speed push), an unseeded run starts it from the
  operating system's randomness, and nothing is drawn or recorded beforehand. A stopped unseeded run resumes from its
  checkpoint, which holds the generator's state, but it cannot be replayed from its beginning, and two unseeded runs
  differ. Each run is one draw of the method's result; four draws give a mean with a meaningful standard deviation
  and a best run. The dataset, the Keras Tuner searches (one seeded search per method, whose chosen architecture
  serves every run), the validation maps and the test circuits are fixed and shared by all runs, so all face the same
  final test. The seeded ES of a run starts from the baseline of the same run. Folders and job labels name a run by
  its number (`seed0` … `seed3`; `ga:0` … `ga:3`).
- **Runner options.** Without options, the runner executes the notebooks in order and continues where its last start
  stopped (resumption, below). `--from NN` starts at notebook NN. `--fresh` removes the existing results (runs, final
  controllers, dataset, tables, figures, status files, logs and run record; the pilots, the viewer's runs and the
  archive are kept) and starts from notebook 01. `--archive NAME` moves the same results to `results/archive/NAME/`
  and exits without running anything, so the next start begins afresh; with `--runs`, it moves only the named runs:
  their run folders, final controllers, status files and output logs, under the same paths inside the archive. An
  existing archive is never overwritten, and the viewer's *Runs* control cycles between the current results and
  each archive (§10.2). `--runs` with `--at-once N` trains the named jobs side by side, at most N at a time, instead of
  executing the notebooks (below), and cannot be combined with `--from` or `--fresh`. `--check` only checks the
  existing results (settings check, below). Only one runner works on a results folder: a second one stops at once.
- **Jobs.** A job is notebook 01 (`nb:01`), one Keras Tuner search (`search:kt_random`, `search:kt_hyperband`,
  `search:kt_bayesian`) or a run of any method (`ga:1`), for example
  `scripts/run_notebooks.py --runs nb:01,search:kt_random,baseline:0,ga:1 --at-once 8`. A run is trained directly by
  `racing/train.py`, which makes exactly the calls of its notebooks (checked by a test). A run of an imitation
  method first trains its own imitation network, notebook 02's call for its seed (the baseline's training, or the
  retraining of its search's chosen architecture), and then refines it by DAgger and pushes it for speed, notebook
  03's call; every step returns its finished record instead of training again, so a stopped job resumes where it
  stopped. Each job runs in its own process, with its own status file and output log. Jobs start in the order
  listed, each as soon as a slot is free and its inputs exist: a search and a baseline run wait for the dataset, a run
  of a searched method for its search and a seeded ES run for its refined baseline, while a GA, ES or NEAT run needs
  nothing; a waiting job holds no slot (`racing/jobs.py`). The three searches therefore run side by side, and every
  DAgger refinement starts as soon as its own network is trained. A job whose input neither exists nor is made by a
  listed job is refused before anything starts; when a job fails, the jobs left waiting for what it would have made are
  reported as skipped, and the others go on. Every step sets its random seed when it starts (run 0 from seed 0), so
  run 0's network trained in a job of its own equals the network notebook 02 trains in order (checked by a test).
  Notebook 02 is not a job: the notebooks, started with `--from 02`, find the networks, searches and runs complete
  and only report them.
- **Machines.** Run 0's results depend only on its seed and the settings, so where run 0 of the GA, ES or NEAT trains
  changes only its training time (Keras training stays on the desktop, below); runs 1–3 also depend on chance. This
  requires every machine to compute the same racing lines: the line's solve (§4.8) goes through LAPACK, whose kernel
  the BLAS library chooses for the processor, and a line that differed in its last bits could change the expert's laps
  and with them every target. The lines of the 23 real circuits in both directions, 46 layouts, are bit-identical on
  two machines with the same OpenBLAS kernel ("Haswell", OpenBLAS 0.3.30, numpy 2.4.1). Before a run, the lines and
  the measured grip share (§7.1) are checked identical on each training machine. Each run's summary records the
  machine it finished on (host, processor, thread setting, worker count and simulator), and the training cost is
  compared by evaluations, which are exact on any machine; training times are reported as context, with the machines
  named.
- **Several machines.** `scripts/fleet.py`, run on the desktop, trains the jobs on the desktop and on servers reached
  from it over SSH, following the plan in `scripts/fleet.toml`: on each machine it runs `run_notebooks.py --runs`
  with that machine's jobs and `at_once` bound, in a detached tmux session. A server is reached with the SSH client
  the plan names (on the desktop, Windows' `ssh.exe`, which reaches the school network through its VPN); the runs
  continue when the connection drops, and only fleet.py's commands need it. A server receives only the last commit,
  so `setup` and `start` refuse to act while tracked code differs from it. The commands:
  - `setup` gives every server the desktop's commit (as a git bundle), the pinned environment (Python 3.13 with
    `requirements.txt`), a C++ compiler where the server has none (conda-forge's `gxx` in the run environment, which
    needs no administrator rights) and any inputs its jobs need from the desktop. The install runs detached on the
    server, so a dropped connection cannot stop it, and `setup` follows its log; it ends by naming the simulator the
    server will use, with a warning when it is the numpy simulator. `setup` is run again after every code change.
  - `start` first checks every machine: a server has the desktop's commit, its environment and its inputs, no runner
    is training, and `run_notebooks.py --check` accepts the existing results. Only when every machine is ready does it
    remove any end request and start every machine's jobs; otherwise nothing starts.
  - `status` shows, for every machine, whether its runner is training, one line per running job and the latest
    milestones of its run log.
  - `stop` sends Ctrl+C to every session and waits for the runners to end; the runs continue where they stopped at
    the next start.
  - `end` writes the end request `results/end_runs` on every machine: every run ends before its next generation with
    its final controller (§9.4). `start` removes the request, so the runs it starts train instead of ending at once.
  - `collect` copies the servers' finished runs, with their final controllers, to the desktop, and each server's run
    log and run record to `results/machines/<name>/`. Each copy is checked by its SHA-256 and each run by its recorded
    settings; a different copy of a run, or an unfinished one, already on the desktop is reported and never replaced.
    Unfinished runs stay on their server.
  - `finish` collects, and once all 32 runs have finished on the desktop it runs notebooks 02 to 08 there in the
    tmux session (`run_notebooks.py --from 02`); they find the networks, searches and runs complete, report them and
    evaluate the final controllers.
  - `archive NAME`, with `--runs` for only some runs, runs `run_notebooks.py --archive NAME` on every machine.
- **Machine plan.** In `scripts/fleet.toml`, each machine lists its jobs, each job belongs to one machine, and a
  server's entry names its host, its SSH client and its project folder. A machine starts its jobs in the listed
  order, the longest runs first. `at_once` bounds the jobs a machine runs at once by memory: compiled runs share the
  processors through their threads, and a Keras job uses few of them, so the bound is the measured peak memory of
  each kind of job against the machine's memory. The bounds, 20 jobs on the desktop and 12 on the server, rest on
  peaks measured with populations half the present size (§9.10). `workers` is the number of worker processes of each
  job, 6 on both machines. The desktop receives notebook 01, the three searches, the 16 imitation runs and the 4
  seeded ES runs; the server the 12 GA, ES and NEAT runs, which need nothing from the desktop. A seeded ES run may
  also be placed on a server, which `setup` then gives its refined baseline. Notebook 01, the searches and the runs of
  the imitation methods are refused on a server: `collect` copies runs only, not the dataset or a search, and Keras
  training may round differently on another processor. How many runs each machine receives follows from the
  generation time measured by `python -m racing.bench generation` on each machine and the generation counts of
  completed runs, so that the machines finish at about the same time; the arithmetic is recorded in the plan's
  comments. Every runner's "run started" log line names the simulator it uses.
- **Kernel:** each notebook runs in a kernel started with the same Python interpreter as the runner,
  with `notebooks/` as the working directory.
- **Checkpoints.** Every evolutionary run saves a checkpoint after each generation. It contains the settings the run
  was made with, the population, the state of the random generator, the level, streak, target factor and step size,
  the seeds already used, NEAT's species and innovation records, the generation count, the number of evaluations,
  the active run time and the hardening record (§9.4). A DAgger refinement saves a checkpoint after each round. A
  checkpoint is written to a temporary file and then renamed, so that an interruption during writing cannot corrupt
  it. A run record is written the same way, after everything it describes has reached the disk, so that neither an
  interruption nor a reboot leaves a partial record.
- **Resumption.** Resumption is the default behavior. When the runner is started again, completed runs are skipped
  and interrupted runs continue from their last checkpoint, losing at most one generation. A resumed run 0 produces
  exactly the same results as an uninterrupted one; a resumed unseeded run continues from the state of its random
  generator saved in the checkpoint. A result file cut short by a crash or a reboot is named by the runner; deleting
  only the files it names redoes their steps and keeps every other result. A run whose summary cannot be read but
  whose checkpoint is beside it finishes again from the checkpoint.
- **Settings check.** Checkpoints and run summaries record the settings that affect results. A run whose recorded
  settings differ from the current ones, or that has no settings record, is not resumed. The runner stops with a
  message naming every difference, and `--fresh` or `--archive NAME` is required. Settings that affect only one
  method are recorded and checked only in that method's runs: the GA's structural settings (§9.6) belong to GA runs
  alone and NEAT's settings (§9.9) to NEAT runs alone, so a change to them refuses only that method's runs, and the
  check of any other record ignores them. Every other record, such as the run record, the dataset record and the
  imitation stages, carries the settings shared by all methods. These include the grip, the settings of the racing
  line and its inputs (§4.8, §5.2), the expert's line-tracking settings `EXPERT_PREVIEW_TIME`,
  `EXPERT_CROSS_TRACK_GAIN` and `EXPERT_HEADING_GAIN` and its measured grip share `EXPERT_GRIP_MARGIN` (§7.1), so a
  run made with another grip, another line or another expert is refused. The final pick's number of finishes,
  `FINAL_MIN_SUCCESSES` (§9.5), is recorded but never compared: it acts only when a run selects its final controller,
  so changing it refuses no checkpoint and redoes no stored result. A run summary without a hardening record, written
  by a run that ended at its speed limit or by an end request before hardening was added, does not count as finished
  while the run's checkpoint is beside it: the summary is kept as `summary_before_hardening.json` once the checkpoint
  has been read, and the run resumes into hardening (§9.4). A checkpoint without a hardening record whose run ended at
  its speed limit hardens even without its summary. `fleet.py collect` and `finish` count such a run as unfinished
  until it has its new summary. The runner checks every run record and checkpoint before it executes the first
  notebook or job, and exits with code 3 if one was made with other settings or cannot be read; `--check` makes only
  this check (exit code 0 or 3), and `fleet.py start` runs it on every machine before anything starts. The runner's
  other exit codes are 1 (a notebook or job failed), 2 (invalid options, or a run's input is missing), 4 (another
  runner is using the same results) and 130 (stopped).
- **Progress display:**
  - **Status file.** Every run writes its current state to `results/status.json` after each generation
    or round: the run number out of the total, the method, seed, stage (evolution, speed push or hardening), level,
    generation, lowest map success share, streak, target reduction and elapsed time. The imitation training writes it
    after every epoch (epoch, validation accuracy and the best so far), a DAgger refinement after every round, and
    each Keras Tuner search after every completed trial (trial number, best validation accuracy), so the line is live
    from the second notebook on.
  - **Live line.** While the runner executes the notebooks, it displays the status as a single line that
    is updated in place, for example
    `[run 18/32] GA run 1 · Hard · gen 412 · worst map 84% / 90% ██████████████████▋ · streak 1/3 · 2h 13m`.
    In the speed push the line adds the target reduction and measures the lowest share against 50%; in hardening it
    measures it against 90%. A status that has not changed for ten minutes ends with "(no update for N min)", as a
    stopped run leaves it.
  - **Milestones.** Starts and resumptions of runs, level advancements, mastery goals, target reductions, the end of
    the speed push with the start of hardening, hardened runs, end requests, DAgger outcomes, completed runs, trained
    imitation networks and completed searches are printed as permanent lines and appended to `results/run_log.txt`.
  - **Quiet kernels.** The notebook kernels and the jobs run with GPUs hidden (`CUDA_VISIBLE_DEVICES=-1`) and
    TensorFlow's C++ log switched off (`TF_CPP_MIN_LOG_LEVEL=3`), so TensorFlow does not search for a GPU on a
    machine that has one. Whatever a kernel process prints itself (library messages, as opposed to cell output, which
    is saved in the notebook) is appended to `results/kernel_log.txt`, so it never enters the progress display.
  - **Stopping.** Ctrl+C or `kill` stops the runner within seconds, also while a notebook prints nothing for a
    long time: it ends the notebook's kernel and its worker processes, logs the stop and exits with code 130. Jobs
    trained side by side are ended with their worker processes (forcibly after 20 s), and the log lists the jobs not
    yet started; the same command continues them all.
  - **Runs side by side.** Each job writes its own status file in `results/status/` and its output to
    `results/train_logs/`, and the display shows one live line per job, in run order.
  - **Separate monitor.** `python -m racing.progress` shows the same display from another terminal, for
    a runner started in the background.
- **Run record:** the runner writes `results/run_info.json`. It records the processor model, the number of logical
  processors, the thread setting, the number of worker processes, the operating-system power profile where available,
  the versions of Python, numpy, TensorFlow, Keras and keras-tuner, the settings that affect results (the values of
  §13), the code version (commit and whether the working tree had uncommitted changes), the simulator, and the start and
  end time of each notebook; for jobs trained side by side, each start's jobs, `at_once` bound, host and code version
  and each job's start, duration and end. A start whose commit differs from the previous start's is reported with a
  warning in the run log.
- **Threads and worker processes.** The compiled simulator (§5.7) uses one thread per logical processor available
  to the process (`os.sched_getaffinity`), read at every call, unless the environment variable `EC_RACING_THREADS`
  sets a positive number. Runs that share a machine share its processors through the operating system at no cost:
  four runs of 20 threads each on 20 logical processors took the same total time as four runs of 5 threads each
  (26.5 s against 26.7 s for four generations of 1,000 trained cars per run), and a run that is alone, or beside runs
  that are training Keras networks or building maps, uses every processor. Worker processes, which the numpy
  simulator, DAgger's driving, dataset collection and the final evaluation use, number `EC_RACING_JOBS`, with a
  default of 6; each holds its own built maps, so their number stays fixed. `python -m racing.bench generation`
  measures the generation time for a range of counts on the machine at hand. The run record logs both values and the
  simulator (`compiled` with the g++ version, or `numpy`).
- **Keras jobs first.** A Keras job trains with 4 TensorFlow threads (2 between operations), and an evolution phase,
  a speed push included, lowers every thread of its process to the lowest priority (nice 19) when it starts; the
  simulator's threads and worker processes started later inherit it. The searches and DAgger refinements, on which
  other jobs wait, then come first, and the evolution runs use every processor they leave. One training epoch of the
  baseline on the development machine: 0.186 s alone with one thread per processor and 0.144 s with 4; beside three
  compiled runs of 20 threads 1.13 s, and 0.48–0.61 s with 4 threads and the runs at the lowest priority. Thread
  counts and priorities change only times.
- **Platform.** The experiment runs on Linux. On the Windows desktop, it runs in the Windows Subsystem for Linux
  (WSL 2), with the project stored in the Linux file system, and the application's window is shown through WSLg.
  Windows must be prevented from sleeping during a run.
- **Environment.** A pinned list of package versions (`requirements.txt`), matching the development
  machine, reproduces the environment.
- **Repository.** The project is hosted in a private GitHub repository, from which the desktop clones
  it.

The GA, ES and NEAT runs train on the server (studentVM-emartinso, AMD EPYC 7763), the imitation methods and the
seeded ES on the desktop (Intel Core i9-12900K). Each run's training time is therefore reported with the machine it
trained on, and the training costs of the methods are compared by evaluations, which are exact on any machine.

The pilot runs of §12.2 (`scripts/pilots.py`) check before a run of the experiment that the mastery requirement can
be reached. The pilots write to their own folder under `results/pilots/` and are not part of the reported results.

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 404" width="100%" role="img" aria-label="The experiment as jobs: their inputs and the machine each runs on" style="display:block;margin:10pt 0 6pt">
<defs><marker id="figjob-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker></defs>
<rect x="250" y="1.5" width="50.61" height="14.5" rx="7.25" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.8" stroke-width="1"/>
<text x="275.3" y="12" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">runs 0–3</text>
<text x="306.61" y="12" font-family="DejaVu Sans" font-size="9" fill="#5f6673">one job per run <tspan font-family="DejaVu Sans Mono">k</tspan>: run 0 seeded, runs 1–3 unseeded</text>
<rect x="0" y="22" width="665" height="206" rx="3" fill="#edf2f7"/>
<text x="8" y="37" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Desktop</text>
<text x="60.49" y="37" font-family="DejaVu Sans" font-size="9" fill="#5f6673">24 threads</text>
<rect x="250" y="46" width="410" height="30" rx="3" fill="none" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<rect x="258" y="57" width="8" height="8" rx="1.5" fill="#7F7F7F"/>
<text x="270" y="64.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">baseline:k</text>
<rect x="368" y="50" width="74" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="64.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">network</text>
<rect x="464" y="50" width="70" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="499" y="64.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">DAgger</text>
<rect x="556" y="50" width="98" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="605" y="64.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">speed push</text>
<path d="M442,61 L464,61" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<path d="M534,61 L556,61" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<rect x="250" y="82" width="410" height="30" rx="3" fill="none" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<rect x="258" y="93" width="8" height="8" rx="1.5" fill="#0072B2"/>
<text x="270" y="100.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">es_seeded:k</text>
<rect x="368" y="86" width="286" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="511" y="100.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">ES from the refined baseline, starts at Easy</text>
<rect x="250" y="118" width="410" height="30" rx="3" fill="none" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<rect x="258" y="129" width="8" height="8" rx="1.5" fill="#E69F00"/>
<text x="270" y="136.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">kt_random:k</text>
<rect x="368" y="122" width="74" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="136.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">network</text>
<rect x="464" y="122" width="70" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="499" y="136.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">DAgger</text>
<rect x="556" y="122" width="98" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="605" y="136.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">speed push</text>
<path d="M442,133 L464,133" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<path d="M534,133 L556,133" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<rect x="250" y="154" width="410" height="30" rx="3" fill="none" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<rect x="258" y="165" width="8" height="8" rx="1.5" fill="#56B4E9"/>
<text x="270" y="172.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">kt_hyperband:k</text>
<rect x="368" y="158" width="74" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="172.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">network</text>
<rect x="464" y="158" width="70" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="499" y="172.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">DAgger</text>
<rect x="556" y="158" width="98" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="605" y="172.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">speed push</text>
<path d="M442,169 L464,169" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<path d="M534,169 L556,169" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<rect x="250" y="190" width="410" height="30" rx="3" fill="none" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<rect x="258" y="201" width="8" height="8" rx="1.5" fill="#CC79A7"/>
<text x="270" y="208.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">kt_bayesian:k</text>
<rect x="368" y="194" width="74" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="405" y="208.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">network</text>
<rect x="464" y="194" width="70" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="499" y="208.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">DAgger</text>
<rect x="556" y="194" width="98" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="605" y="208.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">speed push</text>
<path d="M442,205 L464,205" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<path d="M534,205 L556,205" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<rect x="0" y="117" width="62" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="31" y="129.28" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" text-anchor="middle">nb:01</text>
<text x="31" y="140.74" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">dataset</text>
<rect x="86" y="122" width="150" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<rect x="93" y="129" width="8" height="8" rx="1.5" fill="#E69F00"/>
<text x="105" y="136.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">search:kt_random</text>
<path d="M236,133 L250,133" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<rect x="86" y="158" width="150" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<rect x="93" y="165" width="8" height="8" rx="1.5" fill="#56B4E9"/>
<text x="105" y="172.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">search:kt_hyperband</text>
<path d="M236,169 L250,169" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<rect x="86" y="194" width="150" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<rect x="93" y="201" width="8" height="8" rx="1.5" fill="#CC79A7"/>
<text x="105" y="208.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">search:kt_bayesian</text>
<path d="M236,205 L250,205" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<path d="M62,133 L86,133" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<text x="86" y="117" font-family="DejaVu Sans" font-size="9" fill="#5f6673">each search runs once, seed 0</text>
<path d="M74,133 L74,169 L86,169" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<path d="M74,169 L74,205 L86,205" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<path d="M62,133 L74,133 L74,61 L250,61" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<path d="M499,72 L499,86" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figjob-head)"/>
<rect x="0" y="238" width="665" height="134" rx="3" fill="#f1ecf6"/>
<text x="8" y="253" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Server</text>
<text x="51.15" y="253" font-family="DejaVu Sans" font-size="9" fill="#5f6673">16 threads</text>
<rect x="250" y="262" width="410" height="30" rx="3" fill="none" stroke="#7a5c99" stroke-opacity="0.6" stroke-width="1"/>
<rect x="258" y="273" width="8" height="8" rx="1.5" fill="#D55E00"/>
<text x="270" y="280.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">ga:k</text>
<rect x="368" y="266" width="286" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="511" y="280.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">GA · population 2,000</text>
<rect x="250" y="298" width="410" height="30" rx="3" fill="none" stroke="#7a5c99" stroke-opacity="0.6" stroke-width="1"/>
<rect x="258" y="309" width="8" height="8" rx="1.5" fill="#009E73"/>
<text x="270" y="316.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">es:k</text>
<rect x="368" y="302" width="286" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="511" y="316.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">ES · population 1,100</text>
<rect x="250" y="334" width="410" height="30" rx="3" fill="none" stroke="#7a5c99" stroke-opacity="0.6" stroke-width="1"/>
<rect x="258" y="345" width="8" height="8" rx="1.5" fill="#F0E442" stroke="#6b5f00" stroke-width="0.8"/>
<text x="270" y="352.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">neat:k</text>
<rect x="368" y="338" width="286" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="511" y="352.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">NEAT · population 2,000</text>
<text x="8" y="307.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">need nothing from the desktop:</text>
<text x="8" y="318.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">every run starts at once</text>
<text x="0" y="388" font-family="DejaVu Sans" font-size="9" fill="#5f6673">An arrow is an input: a job starts when its inputs exist, and a waiting job holds no slot. fleet.py collect</text>
<text x="0" y="399" font-family="DejaVu Sans" font-size="9" fill="#5f6673">copies the server's finished runs to the desktop.</text>
</svg>
<p><em>Figure 3. The experiment as jobs: the dataset feeds the searches and the baseline runs, each search feeds its method's runs, each baseline run's DAgger feeds the seeded ES run of the same number, and the GA, ES and NEAT runs on the server need nothing.</em></p>
</div>

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 392" width="90%" role="img" aria-label="The two machines over time, a schematic of when each kind of job runs" style="display:block;margin:10pt auto 6pt">
<defs><marker id="figmac-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker><pattern id="figmac-hatch-d" patternUnits="userSpaceOnUse" width="4.5" height="4.5" patternTransform="rotate(45)"><rect width="4.5" height="4.5" fill="#ffffff"/><line x1="0" y1="0" x2="0" y2="4.5" stroke="#4f6d8c" stroke-opacity="0.55" stroke-width="1.6"/></pattern><pattern id="figmac-hatch-s" patternUnits="userSpaceOnUse" width="4.5" height="4.5" patternTransform="rotate(45)"><rect width="4.5" height="4.5" fill="#ffffff"/><line x1="0" y1="0" x2="0" y2="4.5" stroke="#7a5c99" stroke-opacity="0.55" stroke-width="1.6"/></pattern></defs>
<text x="0" y="17" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330" font-weight="bold">fleet.py</text>
<rect x="89" y="4" width="58" height="18" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="118" y="16.6" font-family="DejaVu Sans Mono" font-size="10" fill="#1d2330" text-anchor="middle">setup</text>
<text x="118" y="35" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">code, environment, g++</text>
<rect x="214" y="4" width="58" height="18" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="243" y="16.6" font-family="DejaVu Sans Mono" font-size="10" fill="#1d2330" text-anchor="middle">start</text>
<text x="243" y="35" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">every job starts at once</text>
<path d="M147,13 L214,13" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figmac-head)"/>
<rect x="339" y="4" width="58" height="18" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="368" y="16.6" font-family="DejaVu Sans Mono" font-size="10" fill="#1d2330" text-anchor="middle">status</text>
<text x="368" y="35" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">one line per running job</text>
<path d="M272,13 L339,13" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figmac-head)"/>
<rect x="464" y="4" width="58" height="18" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="493" y="16.6" font-family="DejaVu Sans Mono" font-size="10" fill="#1d2330" text-anchor="middle">collect</text>
<text x="493" y="35" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">server's runs → desktop</text>
<path d="M397,13 L464,13" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figmac-head)"/>
<rect x="593" y="4" width="58" height="18" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="622" y="16.6" font-family="DejaVu Sans Mono" font-size="10" fill="#1d2330" text-anchor="middle">finish</text>
<text x="665" y="35" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">notebooks 02–08</text>
<path d="M522,13 L593,13" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figmac-head)"/>
<rect x="0" y="48" width="665" height="162" rx="3" fill="#edf2f7"/>
<text x="8" y="63" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Desktop</text>
<text x="60.49" y="63" font-family="DejaVu Sans" font-size="9" fill="#5f6673">24 threads</text>
<text x="156" y="82.2" font-family="DejaVu Sans Mono" font-size="9" fill="#1d2330" text-anchor="end">nb:01<tspan font-family="DejaVu Sans"> dataset</tspan></text>
<rect x="168" y="72" width="5.3" height="14" rx="2" fill="#4f6d8c"/>
<text x="179.3" y="82.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673">first: every other desktop job needs its dataset</text>
<text x="156" y="102.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">3 searches</text>
<rect x="71.29" y="95" width="8" height="8" rx="1.5" fill="#E69F00"/>
<rect x="81.29" y="95" width="8" height="8" rx="1.5" fill="#56B4E9"/>
<rect x="91.29" y="95" width="8" height="8" rx="1.5" fill="#CC79A7"/>
<rect x="173.3" y="92" width="100.78" height="14" rx="2" fill="#4f6d8c"/>
<text x="223.69" y="102.2" font-family="DejaVu Sans" font-size="9" fill="#ffffff" text-anchor="middle">Keras Tuner</text>
<text x="156" y="122.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">4 baseline runs</text>
<rect x="71.06" y="115" width="8" height="8" rx="1.5" fill="#7F7F7F"/>
<rect x="173.3" y="112" width="91.94" height="14" rx="2" fill="#4f6d8c"/>
<text x="219.27" y="122.2" font-family="DejaVu Sans" font-size="9" fill="#ffffff" text-anchor="middle">network, DAgger</text>
<rect x="265.24" y="112" width="282.88" height="14" rx="2" fill="url(#figmac-hatch-d)" stroke="#4f6d8c" stroke-opacity="0.8" stroke-width="1"/>
<rect x="376.75" y="113" width="59.86" height="12" rx="2" fill="#ffffff"/>
<text x="406.68" y="122.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">speed push</text>
<text x="156" y="142.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">12 searched-method runs</text>
<rect x="3.88" y="135" width="8" height="8" rx="1.5" fill="#E69F00"/>
<rect x="13.88" y="135" width="8" height="8" rx="1.5" fill="#56B4E9"/>
<rect x="23.88" y="135" width="8" height="8" rx="1.5" fill="#CC79A7"/>
<line x1="168" y1="139" x2="174.46" y2="139" stroke="#5f6673" stroke-opacity="0.9" stroke-width="1" stroke-dasharray="1.5 2"/>
<line x1="267.62" y1="139" x2="274.08" y2="139" stroke="#5f6673" stroke-opacity="0.9" stroke-width="1" stroke-dasharray="1.5 2"/>
<text x="221.04" y="142.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">waits for its search</text>
<rect x="274.08" y="132" width="101.66" height="14" rx="2" fill="#4f6d8c"/>
<text x="324.91" y="142.2" font-family="DejaVu Sans" font-size="9" fill="#ffffff" text-anchor="middle">network, DAgger</text>
<rect x="375.74" y="132" width="234.26" height="14" rx="2" fill="url(#figmac-hatch-d)" stroke="#4f6d8c" stroke-opacity="0.8" stroke-width="1"/>
<rect x="462.94" y="133" width="59.86" height="12" rx="2" fill="#ffffff"/>
<text x="492.87" y="142.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">speed push</text>
<text x="156" y="162.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">4 seeded ES runs</text>
<rect x="61.77" y="155" width="8" height="8" rx="1.5" fill="#0072B2"/>
<line x1="168" y1="159" x2="174.98" y2="159" stroke="#5f6673" stroke-opacity="0.9" stroke-width="1" stroke-dasharray="1.5 2"/>
<line x1="258.26" y1="159" x2="265.24" y2="159" stroke="#5f6673" stroke-opacity="0.9" stroke-width="1" stroke-dasharray="1.5 2"/>
<text x="216.62" y="162.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">waits for DAgger</text>
<rect x="265.24" y="152" width="229.84" height="14" rx="2" fill="url(#figmac-hatch-d)" stroke="#4f6d8c" stroke-opacity="0.8" stroke-width="1"/>
<rect x="327.29" y="153" width="105.73" height="12" rx="2" fill="#ffffff"/>
<text x="380.16" y="162.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">seeded ES, from Easy</text>
<text x="156" y="193.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">processors</text>
<polygon points="173.3,202 173.3,178 318.28,178 389,202" fill="#4f6d8c"/>
<polygon points="318.28,178 610,178 610,202 389,202" fill="url(#figmac-hatch-d)" stroke="#4f6d8c" stroke-opacity="0.8" stroke-width="1"/>
<rect x="173.3" y="178" width="436.7" height="24" rx="0" fill="none" stroke="#4f6d8c" stroke-opacity="0.8" stroke-width="1"/>
<text x="181.3" y="193.2" font-family="DejaVu Sans" font-size="9" fill="#ffffff">Keras jobs first</text>
<rect x="424.52" y="184" width="158.81" height="12" rx="2" fill="#ffffff"/>
<text x="503.92" y="193.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">evolution at nice 19 uses the rest</text>
<rect x="0" y="218" width="665" height="122" rx="3" fill="#f1ecf6"/>
<text x="8" y="233" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Server</text>
<text x="51.15" y="233" font-family="DejaVu Sans" font-size="9" fill="#5f6673">16 threads</text>
<text x="156" y="252.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">4 GA runs</text>
<rect x="95.63" y="245" width="8" height="8" rx="1.5" fill="#D55E00"/>
<rect x="168" y="242" width="419.9" height="14" rx="2" fill="url(#figmac-hatch-s)" stroke="#7a5c99" stroke-opacity="0.8" stroke-width="1"/>
<text x="156" y="272.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">4 ES runs</text>
<rect x="97.36" y="265" width="8" height="8" rx="1.5" fill="#009E73"/>
<rect x="168" y="262" width="256.36" height="14" rx="2" fill="url(#figmac-hatch-s)" stroke="#7a5c99" stroke-opacity="0.8" stroke-width="1"/>
<text x="156" y="292.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">4 NEAT runs</text>
<rect x="85.38" y="285" width="8" height="8" rx="1.5" fill="#F0E442" stroke="#6b5f00" stroke-width="0.8"/>
<rect x="168" y="282" width="406.64" height="14" rx="2" fill="url(#figmac-hatch-s)" stroke="#7a5c99" stroke-opacity="0.8" stroke-width="1"/>
<text x="156" y="317.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">processors</text>
<rect x="168" y="302" width="419.9" height="24" rx="2" fill="url(#figmac-hatch-s)" stroke="#7a5c99" stroke-opacity="0.8" stroke-width="1"/>
<rect x="197.9" y="308" width="360.11" height="12" rx="2" fill="#ffffff"/>
<text x="377.95" y="317.2" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">no Keras jobs: the 12 runs share all 16 threads; a run that ends frees its share</text>
<line x1="168" y1="48" x2="168" y2="352" stroke="#5f6673" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="2 2.5"/>
<line x1="610" y1="48" x2="610" y2="352" stroke="#5f6673" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="2 2.5"/>
<path d="M168,352 L640,352" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figmac-head)"/>
<text x="168" y="365" font-family="DejaVu Sans" font-size="9" fill="#1d2330">start: every job of a machine starts at once</text>
<text x="610" y="365" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">end: both machines finish at about the same time</text>
<text x="156" y="355.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">time</text>
<text x="156" y="366.2" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">not to scale</text>
<rect x="0" y="376" width="18" height="10" rx="1.5" fill="#4f6d8c"/>
<text x="24" y="384" font-family="DejaVu Sans" font-size="9" fill="#5f6673">normal priority: dataset, searches, imitation training, DAgger</text>
<rect x="314.79" y="376" width="18" height="10" rx="1.5" fill="url(#figmac-hatch-d)" stroke="#4f6d8c" stroke-opacity="0.8" stroke-width="1"/>
<text x="338.79" y="384" font-family="DejaVu Sans" font-size="9" fill="#5f6673">evolution at nice 19</text>
<line x1="444.37" y1="381" x2="462.37" y2="381" stroke="#5f6673" stroke-opacity="0.9" stroke-width="1" stroke-dasharray="1.5 2"/>
<text x="468.37" y="384" font-family="DejaVu Sans" font-size="9" fill="#5f6673">waiting: holds no slot</text>
</svg>
<p><em>Figure 4. The two machines over time (schematic, not to scale): every job starts at once, the Keras jobs come first, evolution at nice 19 uses the processors they leave, and both machines finish at about the same time.</em></p>
</div>

## 4. Track Generation and Curriculum

### 4.1 Track representation

The function `tracks.get_track(seed, level)` returns the map of any level as a `Track` object: a generated map,
whose geometry `track_gen.candidate_track(seed, level, attempt)` draws, on levels 1–3 and the validation setting, and
a real circuit (§4.7), built by `circuits.level_track(seed, level)` from `circuits.circuit_track(name, reverse)`, on
Extra hard and the test level. A `Track` contains:

- the centre line (`centerline`), resampled at 1 m intervals
- the road's width to the left and to the right of every centre-line point, and the left and right boundaries
- a road occupancy grid at 0.5 m resolution
- a distance-to-boundary map computed with `scipy.ndimage.distance_transform_edt`
- the start pose and track length
- the racing line (§4.8): a point, a direction and a curvature for every centre-line point
- the expert's target speed at every centre-line point (§7.1)
- the expert driver's lap time

The centre line is stored in driving order, so a map that runs clockwise is the reversed point sequence of its
counter-clockwise form (§4.4). A generated map has equal widths to the left and right; a real circuit has those of its
data, whose centre line is not exactly central (§4.7).

### 4.2 Curriculum levels

The evolutionary controllers train on a curriculum of four levels, from gentle loops on a wide road to real circuits.
The generated levels are of Formula 1 size, so that each leads up to the real circuits of Extra hard, and a validation
setting is harder than any training level.

- **Easy (level 1)** uses the loop generator: control points at jittered angles and random radii, joined by a
  periodic cubic spline. It is set to produce gentle loops on a wide road.
- **Medium and Hard (levels 2–3)** and the **validation setting** use the circuit generator (§4.3).
- **Extra hard (level 4)** consists of the 19 real training circuits (§4.7).

*Table 1. Curriculum levels, generator settings and the validation setting. Lengths in metres.*

| Level | Source | Lap length | Corner radii | Features | Tightest corner | Road width |
|---|---|---|---|---|---|---|
| 1 Easy | Loop: 4–6 points, radius 250–450 | about 1,900–2,600 | Spline | None | 120 | 12–16, varying |
| 2 Medium | Circuit: 6–8 points, radius 250–600 | about 2,300–3,700 | Slow 35–60, fast 100–200 | None | 35 | 10–16, varying |
| 3 Hard | Circuit: 9–12 points, radius 300–750 | about 3,000–5,900 | Slow 20–45, fast 80–250 | Main straight ≥ 600 into a hairpin; 0–1 chicane; 0–1 run of esses | 18 | 9–15, varying |
| 4 Extra hard | 19 real circuits (§4.7) | 3,690–7,000 | Real | Real | About 11–33, by circuit | Real, 7.4–27.6 per point |
| Validation | Circuit: 13–17 points, radius 350–900 | about 4,100–7,700 | Slow 12–30, fast 60–250 | Main straight ≥ 900 into a hairpin; a second hairpin; 2–3 chicanes; 2–3 runs of esses | 12 | 7.5–12, varying |

The lap lengths of the generated levels are those measured over seeds 0–19 of each level with the dataset's maps
(§7.3) and over the 20 validation maps (medians 2.3, 2.9 and 4.2 km on Easy, Medium and Hard); the radii above
produce them. The real Formula 1 circuits are 4.3 to 7.0 km long, their tightest corners have radii of about 11 to
23 m (Shanghai's is the tightest; 11 to 33 m over the 19 training circuits of Table 1), measured on the centre lines
with the curvature smoothed by a Gaussian of 10 m standard deviation, as for the racing line (§4.8), and their roads
are 7.4 m wide at the narrowest. The generated levels climb toward these values, and the validation setting goes
beyond the generated training levels: tighter corners than Hard, more chicanes and esses, and a road as narrow as the
narrowest real circuits (7.5–12 m).

Figure 5 shows one map of each difficulty to the same scale.

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 531" width="100%" role="img" aria-label="One map of each difficulty at the same scale: Easy, Medium, Hard, Extra hard and validation" style="display:block;margin:10pt 0 6pt"><rect x="0.0" y="0" width="215.0" height="200.82337120641645" rx="3" fill="#f5f6f8"/><text x="107.5" y="16.0" font-family="DejaVu Sans" font-size="11" fill="#1d2330" text-anchor="middle" font-weight="bold">Easy</text><text x="107.5" y="29.0" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle" font-weight="normal">generated, display seed 1000</text><path d="M148.4,95.8 L148.4,95.9 L148.4,96.0 L148.3,96.1 L148.3,96.2 L148.3,96.3 L148.3,96.5 L148.3,96.6 L148.2,96.7 L148.2,96.8 L148.2,96.9 L148.2,97.0 L148.2,97.1 L148.1,97.2 L148.1,97.4 L148.1,97.5 L148.1,97.6 L148.0,97.7 L148.0,97.8 L148.0,97.9 L148.0,98.0 L147.9,98.1 L147.9,98.3 L147.9,98.4 L147.8,98.5 L147.8,98.6 L147.8,98.7 L147.7,98.8 L147.7,98.9 L147.7,99.0 L147.6,99.1 L147.6,99.2 L147.6,99.4 L147.5,99.5 L147.5,99.6 L147.4,99.7 L147.4,99.8 L147.4,99.9 L147.3,100.0 L147.3,100.1 L147.2,100.2 L147.2,100.3 L147.2,100.4 L147.1,100.5 L147.1,100.6 L147.0,100.8 L147.0,100.9 L146.9,101.0 L146.9,101.1 L146.8,101.2 L146.8,101.3 L146.7,101.4 L146.7,101.5 L146.6,101.6 L146.6,101.7 L146.5,101.8 L146.5,101.9 L146.4,102.0 L146.4,102.1 L146.3,102.2 L146.3,102.3 L146.2,102.4 L146.2,102.5 L146.1,102.6 L146.1,102.7 L146.0,102.8 L146.0,102.9 L145.9,103.0 L145.8,103.1 L145.8,103.2 L145.7,103.3 L145.7,103.4 L145.6,103.5 L145.6,103.6 L145.5,103.7 L145.4,103.8 L145.4,103.9 L145.3,104.0 L145.2,104.1 L145.2,104.2 L145.1,104.3 L145.1,104.4 L145.0,104.5 L144.9,104.6 L144.9,104.7 L144.8,104.8 L144.7,104.9 L144.7,105.0 L144.6,105.1 L144.5,105.2 L144.5,105.2 L144.4,105.3 L144.3,105.4 L144.3,105.5 L144.2,105.6 L144.1,105.7 L144.1,105.8 L144.0,105.9 L143.9,106.0 L143.9,106.1 L143.8,106.2 L143.7,106.3 L143.7,106.4 L143.6,106.4 L143.5,106.5 L143.4,106.6 L143.4,106.7 L143.3,106.8 L143.2,106.9 L143.2,107.0 L143.1,107.1 L143.0,107.2 L142.9,107.2 L142.9,107.3 L142.8,107.4 L142.7,107.5 L142.6,107.6 L142.6,107.7 L142.5,107.8 L142.4,107.9 L142.3,107.9 L142.3,108.0 L142.2,108.1 L142.1,108.2 L142.0,108.3 L142.0,108.4 L141.9,108.5 L141.8,108.6 L141.7,108.6 L141.6,108.7 L141.6,108.8 L141.5,108.9 L141.4,109.0 L141.3,109.1 L141.3,109.1 L141.2,109.2 L141.1,109.3 L141.0,109.4 L140.9,109.5 L140.9,109.6 L140.8,109.6 L140.7,109.7 L140.6,109.8 L140.5,109.9 L140.5,110.0 L140.4,110.0 L140.3,110.1 L140.2,110.2 L140.1,110.3 L140.0,110.4 L140.0,110.5 L139.9,110.5 L139.8,110.6 L139.7,110.7 L139.6,110.8 L139.6,110.9 L139.5,110.9 L139.4,111.0 L139.3,111.1 L139.2,111.2 L139.1,111.2 L139.1,111.3 L139.0,111.4 L138.9,111.5 L138.8,111.6 L138.7,111.6 L138.6,111.7 L138.5,111.8 L138.5,111.9 L138.4,112.0 L138.3,112.0 L138.2,112.1 L138.1,112.2 L138.0,112.3 L137.9,112.3 L137.9,112.4 L137.8,112.5 L137.7,112.6 L137.6,112.6 L137.5,112.7 L137.4,112.8 L137.3,112.9 L137.3,112.9 L137.2,113.0 L137.1,113.1 L137.0,113.2 L136.9,113.2 L136.8,113.3 L136.7,113.4 L136.6,113.5 L136.6,113.5 L136.5,113.6 L136.4,113.7 L136.3,113.8 L136.2,113.8 L136.1,113.9 L136.0,114.0 L135.9,114.1 L135.8,114.1 L135.8,114.2 L135.7,114.3 L135.6,114.3 L135.5,114.4 L135.4,114.5 L135.3,114.6 L135.2,114.6 L135.1,114.7 L135.0,114.8 L135.0,114.9 L134.9,114.9 L134.8,115.0 L134.7,115.1 L134.6,115.1 L134.5,115.2 L134.4,115.3 L134.3,115.4 L134.2,115.4 L134.1,115.5 L134.1,115.6 L134.0,115.6 L133.9,115.7 L133.8,115.8 L133.7,115.9 L133.6,115.9 L133.5,116.0 L133.4,116.1 L133.3,116.1 L133.2,116.2 L133.1,116.3 L133.1,116.3 L133.0,116.4 L132.9,116.5 L132.8,116.6 L132.7,116.6 L132.6,116.7 L132.5,116.8 L132.4,116.8 L132.3,116.9 L132.2,117.0 L132.1,117.0 L132.0,117.1 L132.0,117.2 L131.9,117.2 L131.8,117.3 L131.7,117.4 L131.6,117.5 L131.5,117.5 L131.4,117.6 L131.3,117.7 L131.2,117.7 L131.1,117.8 L131.0,117.9 L130.9,117.9 L130.8,118.0 L130.8,118.1 L130.7,118.1 L130.6,118.2 L130.5,118.3 L130.4,118.3 L130.3,118.4 L130.2,118.5 L130.1,118.5 L130.0,118.6 L129.9,118.7 L129.8,118.7 L129.7,118.8 L129.6,118.9 L129.5,118.9 L129.4,119.0 L129.4,119.1 L129.3,119.1 L129.2,119.2 L129.1,119.3 L129.0,119.3 L128.9,119.4 L128.8,119.5 L128.7,119.5 L128.6,119.6 L128.5,119.7 L128.4,119.7 L128.3,119.8 L128.2,119.9 L128.1,119.9 L128.0,120.0 L127.9,120.1 L127.8,120.1 L127.7,120.2 L127.7,120.3 L127.6,120.3 L127.5,120.4 L127.4,120.5 L127.3,120.5 L127.2,120.6 L127.1,120.7 L127.0,120.7 L126.9,120.8 L126.8,120.8 L126.7,120.9 L126.6,121.0 L126.5,121.0 L126.4,121.1 L126.3,121.2 L126.2,121.2 L126.1,121.3 L126.0,121.4 L125.9,121.4 L125.8,121.5 L125.7,121.5 L125.7,121.6 L125.6,121.7 L125.5,121.7 L125.4,121.8 L125.3,121.9 L125.2,121.9 L125.1,122.0 L125.0,122.0 L124.9,122.1 L124.8,122.2 L124.7,122.2 L124.6,122.3 L124.5,122.4 L124.4,122.4 L124.3,122.5 L124.2,122.5 L124.1,122.6 L124.0,122.7 L123.9,122.7 L123.8,122.8 L123.7,122.9 L123.6,122.9 L123.5,123.0 L123.4,123.0 L123.3,123.1 L123.2,123.2 L123.1,123.2 L123.0,123.3 L122.9,123.3 L122.8,123.4 L122.7,123.5 L122.6,123.5 L122.5,123.6 L122.4,123.6 L122.3,123.7 L122.3,123.8 L122.2,123.8 L122.1,123.9 L122.0,123.9 L121.9,124.0 L121.8,124.1 L121.7,124.1 L121.6,124.2 L121.5,124.2 L121.4,124.3 L121.3,124.4 L121.2,124.4 L121.1,124.5 L121.0,124.5 L120.9,124.6 L120.8,124.7 L120.7,124.7 L120.6,124.8 L120.5,124.8 L120.4,124.9 L120.3,124.9 L120.2,125.0 L120.1,125.1 L120.0,125.1 L119.9,125.2 L119.8,125.2 L119.7,125.3 L119.6,125.3 L119.5,125.4 L119.4,125.5 L119.3,125.5 L119.2,125.6 L119.1,125.6 L119.0,125.7 L118.9,125.7 L118.8,125.8 L118.7,125.8 L118.6,125.9 L118.5,126.0 L118.4,126.0 L118.3,126.1 L118.2,126.1 L118.1,126.2 L118.0,126.2 L117.9,126.3 L117.8,126.3 L117.7,126.4 L117.6,126.5 L117.5,126.5 L117.4,126.6 L117.3,126.6 L117.2,126.7 L117.1,126.7 L117.0,126.8 L116.9,126.8 L116.7,126.9 L116.6,126.9 L116.5,127.0 L116.4,127.0 L116.3,127.1 L116.2,127.2 L116.1,127.2 L116.0,127.3 L115.9,127.3 L115.8,127.4 L115.7,127.4 L115.6,127.5 L115.5,127.5 L115.4,127.6 L115.3,127.6 L115.2,127.7 L115.1,127.7 L115.0,127.8 L114.9,127.8 L114.8,127.9 L114.7,127.9 L114.6,128.0 L114.5,128.0 L114.4,128.1 L114.3,128.1 L114.2,128.2 L114.1,128.2 L114.0,128.3 L113.9,128.3 L113.8,128.4 L113.7,128.4 L113.6,128.5 L113.5,128.5 L113.4,128.6 L113.2,128.6 L113.1,128.7 L113.0,128.7 L112.9,128.8 L112.8,128.8 L112.7,128.9 L112.6,128.9 L112.5,129.0 L112.4,129.0 L112.3,129.1 L112.2,129.1 L112.1,129.2 L112.0,129.2 L111.9,129.2 L111.8,129.3 L111.7,129.3 L111.6,129.4 L111.5,129.4 L111.4,129.5 L111.3,129.5 L111.1,129.6 L111.0,129.6 L110.9,129.7 L110.8,129.7 L110.7,129.7 L110.6,129.8 L110.5,129.8 L110.4,129.9 L110.3,129.9 L110.2,130.0 L110.1,130.0 L110.0,130.1 L109.9,130.1 L109.8,130.1 L109.7,130.2 L109.6,130.2 L109.4,130.3 L109.3,130.3 L109.2,130.4 L109.1,130.4 L109.0,130.4 L108.9,130.5 L108.8,130.5 L108.7,130.6 L108.6,130.6 L108.5,130.6 L108.4,130.7 L108.3,130.7 L108.2,130.8 L108.1,130.8 L107.9,130.8 L107.8,130.9 L107.7,130.9 L107.6,131.0 L107.5,131.0 L107.4,131.0 L107.3,131.1 L107.2,131.1 L107.1,131.2 L107.0,131.2 L106.9,131.2 L106.8,131.3 L106.6,131.3 L106.5,131.4 L106.4,131.4 L106.3,131.4 L106.2,131.5 L106.1,131.5 L106.0,131.5 L105.9,131.6 L105.8,131.6 L105.7,131.6 L105.6,131.7 L105.4,131.7 L105.3,131.7 L105.2,131.8 L105.1,131.8 L105.0,131.8 L104.9,131.9 L104.8,131.9 L104.7,132.0 L104.6,132.0 L104.5,132.0 L104.3,132.0 L104.2,132.1 L104.1,132.1 L104.0,132.1 L103.9,132.2 L103.8,132.2 L103.7,132.2 L103.6,132.3 L103.5,132.3 L103.3,132.3 L103.2,132.4 L103.1,132.4 L103.0,132.4 L102.9,132.5 L102.8,132.5 L102.7,132.5 L102.6,132.5 L102.5,132.6 L102.3,132.6 L102.2,132.6 L102.1,132.7 L102.0,132.7 L101.9,132.7 L101.8,132.7 L101.7,132.8 L101.6,132.8 L101.5,132.8 L101.3,132.8 L101.2,132.9 L101.1,132.9 L101.0,132.9 L100.9,132.9 L100.8,133.0 L100.7,133.0 L100.6,133.0 L100.4,133.0 L100.3,133.1 L100.2,133.1 L100.1,133.1 L100.0,133.1 L99.9,133.1 L99.8,133.2 L99.7,133.2 L99.5,133.2 L99.4,133.2 L99.3,133.2 L99.2,133.3 L99.1,133.3 L99.0,133.3 L98.9,133.3 L98.7,133.3 L98.6,133.4 L98.5,133.4 L98.4,133.4 L98.3,133.4 L98.2,133.4 L98.1,133.4 L97.9,133.5 L97.8,133.5 L97.7,133.5 L97.6,133.5 L97.5,133.5 L97.4,133.5 L97.3,133.5 L97.1,133.6 L97.0,133.6 L96.9,133.6 L96.8,133.6 L96.7,133.6 L96.6,133.6 L96.5,133.6 L96.3,133.6 L96.2,133.6 L96.1,133.7 L96.0,133.7 L95.9,133.7 L95.8,133.7 L95.7,133.7 L95.5,133.7 L95.4,133.7 L95.3,133.7 L95.2,133.7 L95.1,133.7 L95.0,133.7 L94.9,133.7 L94.7,133.7 L94.6,133.7 L94.5,133.8 L94.4,133.8 L94.3,133.8 L94.2,133.8 L94.1,133.8 L93.9,133.8 L93.8,133.8 L93.7,133.8 L93.6,133.8 L93.5,133.8 L93.4,133.8 L93.2,133.8 L93.1,133.8 L93.0,133.8 L92.9,133.8 L92.8,133.8 L92.7,133.8 L92.6,133.8 L92.4,133.7 L92.3,133.7 L92.2,133.7 L92.1,133.7 L92.0,133.7 L91.9,133.7 L91.8,133.7 L91.6,133.7 L91.5,133.7 L91.4,133.7 L91.3,133.7 L91.2,133.7 L91.1,133.7 L90.9,133.7 L90.8,133.6 L90.7,133.6 L90.6,133.6 L90.5,133.6 L90.4,133.6 L90.3,133.6 L90.1,133.6 L90.0,133.5 L89.9,133.5 L89.8,133.5 L89.7,133.5 L89.6,133.5 L89.5,133.5 L89.4,133.4 L89.2,133.4 L89.1,133.4 L89.0,133.4 L88.9,133.4 L88.8,133.4 L88.7,133.3 L88.6,133.3 L88.4,133.3 L88.3,133.3 L88.2,133.2 L88.1,133.2 L88.0,133.2 L87.9,133.2 L87.8,133.1 L87.7,133.1 L87.5,133.1 L87.4,133.1 L87.3,133.0 L87.2,133.0 L87.1,133.0 L87.0,132.9 L86.9,132.9 L86.8,132.9 L86.7,132.8 L86.5,132.8 L86.4,132.8 L86.3,132.7 L86.2,132.7 L86.1,132.7 L86.0,132.6 L85.9,132.6 L85.8,132.6 L85.7,132.5 L85.6,132.5 L85.5,132.5 L85.3,132.4 L85.2,132.4 L85.1,132.3 L85.0,132.3 L84.9,132.3 L84.8,132.2 L84.7,132.2 L84.6,132.1 L84.5,132.1 L84.4,132.0 L84.3,132.0 L84.2,132.0 L84.1,131.9 L84.0,131.9 L83.9,131.8 L83.7,131.8 L83.6,131.7 L83.5,131.7 L83.4,131.6 L83.3,131.6 L83.2,131.5 L83.1,131.5 L83.0,131.4 L82.9,131.4 L82.8,131.3 L82.7,131.3 L82.6,131.2 L82.5,131.2 L82.4,131.1 L82.3,131.0 L82.2,131.0 L82.1,130.9 L82.0,130.9 L81.9,130.8 L81.8,130.8 L81.7,130.7 L81.6,130.6 L81.5,130.6 L81.4,130.5 L81.3,130.5 L81.2,130.4 L81.1,130.3 L81.0,130.3 L80.9,130.2 L80.8,130.1 L80.8,130.1 L80.7,130.0 L80.6,129.9 L80.5,129.9 L80.4,129.8 L80.3,129.7 L80.2,129.7 L80.1,129.6 L80.0,129.5 L79.9,129.5 L79.8,129.4 L79.7,129.3 L79.6,129.3 L79.5,129.2 L79.5,129.1 L79.4,129.0 L79.3,129.0 L79.2,128.9 L79.1,128.8 L79.0,128.7 L78.9,128.7 L78.8,128.6 L78.7,128.5 L78.7,128.5 L78.6,128.4 L78.5,128.3 L78.4,128.2 L78.3,128.1 L78.2,128.1 L78.1,128.0 L78.1,127.9 L78.0,127.8 L77.9,127.8 L77.8,127.7 L77.7,127.6 L77.6,127.5 L77.6,127.4 L77.5,127.4 L77.4,127.3 L77.3,127.2 L77.2,127.1 L77.2,127.0 L77.1,126.9 L77.0,126.9 L76.9,126.8 L76.8,126.7 L76.8,126.6 L76.7,126.5 L76.6,126.4 L76.5,126.4 L76.5,126.3 L76.4,126.2 L76.3,126.1 L76.2,126.0 L76.1,125.9 L76.1,125.8 L76.0,125.8 L75.9,125.7 L75.8,125.6 L75.8,125.5 L75.7,125.4 L75.6,125.3 L75.5,125.2 L75.5,125.1 L75.4,125.1 L75.3,125.0 L75.3,124.9 L75.2,124.8 L75.1,124.7 L75.0,124.6 L75.0,124.5 L74.9,124.4 L74.8,124.3 L74.8,124.2 L74.7,124.1 L74.6,124.1 L74.6,124.0 L74.5,123.9 L74.4,123.8 L74.4,123.7 L74.3,123.6 L74.2,123.5 L74.1,123.4 L74.1,123.3 L74.0,123.2 L74.0,123.1 L73.9,123.0 L73.8,122.9 L73.8,122.8 L73.7,122.7 L73.6,122.6 L73.6,122.6 L73.5,122.5 L73.4,122.4 L73.4,122.3 L73.3,122.2 L73.2,122.1 L73.2,122.0 L73.1,121.9 L73.1,121.8 L73.0,121.7 L72.9,121.6 L72.9,121.5 L72.8,121.4 L72.8,121.3 L72.7,121.2 L72.6,121.1 L72.6,121.0 L72.5,120.9 L72.5,120.8 L72.4,120.7 L72.3,120.6 L72.3,120.5 L72.2,120.4 L72.2,120.3 L72.1,120.2 L72.1,120.1 L72.0,120.0 L71.9,119.9 L71.9,119.8 L71.8,119.7 L71.8,119.6 L71.7,119.5 L71.7,119.4 L71.6,119.3 L71.6,119.2 L71.5,119.1 L71.5,119.0 L71.4,118.9 L71.4,118.8 L71.3,118.7 L71.2,118.6 L71.2,118.5 L71.1,118.4 L71.1,118.3 L71.0,118.2 L71.0,118.1 L70.9,118.0 L70.9,117.9 L70.8,117.8 L70.8,117.6 L70.7,117.5 L70.7,117.4 L70.6,117.3 L70.6,117.2 L70.5,117.1 L70.5,117.0 L70.5,116.9 L70.4,116.8 L70.4,116.7 L70.3,116.6 L70.3,116.5 L70.2,116.4 L70.2,116.3 L70.1,116.2 L70.1,116.1 L70.0,116.0 L70.0,115.9 L69.9,115.8 L69.9,115.7 L69.9,115.5 L69.8,115.4 L69.8,115.3 L69.7,115.2 L69.7,115.1 L69.6,115.0 L69.6,114.9 L69.6,114.8 L69.5,114.7 L69.5,114.6 L69.4,114.5 L69.4,114.4 L69.4,114.3 L69.3,114.2 L69.3,114.0 L69.2,113.9 L69.2,113.8 L69.2,113.7 L69.1,113.6 L69.1,113.5 L69.0,113.4 L69.0,113.3 L69.0,113.2 L68.9,113.1 L68.9,113.0 L68.8,112.9 L68.8,112.7 L68.8,112.6 L68.7,112.5 L68.7,112.4 L68.7,112.3 L68.6,112.2 L68.6,112.1 L68.6,112.0 L68.5,111.9 L68.5,111.8 L68.5,111.6 L68.4,111.5 L68.4,111.4 L68.4,111.3 L68.3,111.2 L68.3,111.1 L68.3,111.0 L68.2,110.9 L68.2,110.8 L68.2,110.7 L68.1,110.5 L68.1,110.4 L68.1,110.3 L68.0,110.2 L68.0,110.1 L68.0,110.0 L68.0,109.9 L67.9,109.8 L67.9,109.7 L67.9,109.5 L67.8,109.4 L67.8,109.3 L67.8,109.2 L67.8,109.1 L67.7,109.0 L67.7,108.9 L67.7,108.8 L67.6,108.7 L67.6,108.5 L67.6,108.4 L67.6,108.3 L67.5,108.2 L67.5,108.1 L67.5,108.0 L67.5,107.9 L67.4,107.8 L67.4,107.6 L67.4,107.5 L67.4,107.4 L67.4,107.3 L67.3,107.2 L67.3,107.1 L67.3,107.0 L67.3,106.9 L67.2,106.7 L67.2,106.6 L67.2,106.5 L67.2,106.4 L67.2,106.3 L67.1,106.2 L67.1,106.1 L67.1,105.9 L67.1,105.8 L67.1,105.7 L67.0,105.6 L67.0,105.5 L67.0,105.4 L67.0,105.3 L67.0,105.2 L67.0,105.0 L66.9,104.9 L66.9,104.8 L66.9,104.7 L66.9,104.6 L66.9,104.5 L66.9,104.4 L66.8,104.2 L66.8,104.1 L66.8,104.0 L66.8,103.9 L66.8,103.8 L66.8,103.7 L66.8,103.6 L66.7,103.4 L66.7,103.3 L66.7,103.2 L66.7,103.1 L66.7,103.0 L66.7,102.9 L66.7,102.8 L66.7,102.6 L66.7,102.5 L66.6,102.4 L66.6,102.3 L66.6,102.2 L66.6,102.1 L66.6,102.0 L66.6,101.8 L66.6,101.7 L66.6,101.6 L66.6,101.5 L66.6,101.4 L66.6,101.3 L66.6,101.2 L66.5,101.0 L66.5,100.9 L66.5,100.8 L66.5,100.7 L66.5,100.6 L66.5,100.5 L66.5,100.3 L66.5,100.2 L66.5,100.1 L66.5,100.0 L66.5,99.9 L66.5,99.8 L66.5,99.7 L66.5,99.5 L66.5,99.4 L66.5,99.3 L66.5,99.2 L66.5,99.1 L66.5,99.0 L66.5,98.9 L66.5,98.7 L66.5,98.6 L66.5,98.5 L66.5,98.4 L66.5,98.3 L66.5,98.2 L66.5,98.0 L66.5,97.9 L66.5,97.8 L66.5,97.7 L66.5,97.6 L66.5,97.5 L66.5,97.4 L66.5,97.2 L66.5,97.1 L66.5,97.0 L66.5,96.9 L66.6,96.8 L66.6,96.7 L66.6,96.6 L66.6,96.4 L66.6,96.3 L66.6,96.2 L66.6,96.1 L66.6,96.0 L66.6,95.9 L66.6,95.8 L66.6,95.6 L66.7,95.5 L66.7,95.4 L66.7,95.3 L66.7,95.2 L66.7,95.1 L66.7,94.9 L66.7,94.8 L66.7,94.7 L66.8,94.6 L66.8,94.5 L66.8,94.4 L66.8,94.3 L66.8,94.2 L66.8,94.0 L66.9,93.9 L66.9,93.8 L66.9,93.7 L66.9,93.6 L66.9,93.5 L66.9,93.4 L67.0,93.2 L67.0,93.1 L67.0,93.0 L67.0,92.9 L67.0,92.8 L67.1,92.7 L67.1,92.6 L67.1,92.4 L67.1,92.3 L67.2,92.2 L67.2,92.1 L67.2,92.0 L67.2,91.9 L67.2,91.8 L67.3,91.7 L67.3,91.5 L67.3,91.4 L67.4,91.3 L67.4,91.2 L67.4,91.1 L67.4,91.0 L67.5,90.9 L67.5,90.8 L67.5,90.7 L67.6,90.5 L67.6,90.4 L67.6,90.3 L67.6,90.2 L67.7,90.1 L67.7,90.0 L67.7,89.9 L67.8,89.8 L67.8,89.7 L67.8,89.6 L67.9,89.4 L67.9,89.3 L67.9,89.2 L68.0,89.1 L68.0,89.0 L68.1,88.9 L68.1,88.8 L68.1,88.7 L68.2,88.6 L68.2,88.5 L68.3,88.4 L68.3,88.2 L68.3,88.1 L68.4,88.0 L68.4,87.9 L68.5,87.8 L68.5,87.7 L68.5,87.6 L68.6,87.5 L68.6,87.4 L68.7,87.3 L68.7,87.2 L68.8,87.1 L68.8,87.0 L68.9,86.9 L68.9,86.8 L69.0,86.7 L69.0,86.6 L69.1,86.4 L69.1,86.3 L69.2,86.2 L69.2,86.1 L69.3,86.0 L69.3,85.9 L69.4,85.8 L69.4,85.7 L69.5,85.6 L69.5,85.5 L69.6,85.4 L69.6,85.3 L69.7,85.2 L69.7,85.1 L69.8,85.0 L69.8,84.9 L69.9,84.8 L70.0,84.7 L70.0,84.6 L70.1,84.5 L70.1,84.4 L70.2,84.3 L70.3,84.2 L70.3,84.1 L70.4,84.0 L70.4,83.9 L70.5,83.8 L70.6,83.7 L70.6,83.6 L70.7,83.5 L70.7,83.4 L70.8,83.3 L70.9,83.2 L70.9,83.2 L71.0,83.1 L71.1,83.0 L71.1,82.9 L71.2,82.8 L71.3,82.7 L71.3,82.6 L71.4,82.5 L71.5,82.4 L71.5,82.3 L71.6,82.2 L71.7,82.1 L71.7,82.0 L71.8,81.9 L71.9,81.8 L71.9,81.8 L72.0,81.7 L72.1,81.6 L72.2,81.5 L72.2,81.4 L72.3,81.3 L72.4,81.2 L72.4,81.1 L72.5,81.0 L72.6,80.9 L72.7,80.9 L72.7,80.8 L72.8,80.7 L72.9,80.6 L73.0,80.5 L73.0,80.4 L73.1,80.3 L73.2,80.2 L73.3,80.2 L73.3,80.1 L73.4,80.0 L73.5,79.9 L73.6,79.8 L73.7,79.7 L73.7,79.7 L73.8,79.6 L73.9,79.5 L74.0,79.4 L74.0,79.3 L74.1,79.2 L74.2,79.2 L74.3,79.1 L74.4,79.0 L74.5,78.9 L74.5,78.8 L74.6,78.7 L74.7,78.7 L74.8,78.6 L74.9,78.5 L74.9,78.4 L75.0,78.3 L75.1,78.3 L75.2,78.2 L75.3,78.1 L75.4,78.0 L75.4,78.0 L75.5,77.9 L75.6,77.8 L75.7,77.7 L75.8,77.6 L75.9,77.6 L76.0,77.5 L76.0,77.4 L76.1,77.3 L76.2,77.3 L76.3,77.2 L76.4,77.1 L76.5,77.0 L76.6,77.0 L76.7,76.9 L76.7,76.8 L76.8,76.7 L76.9,76.7 L77.0,76.6 L77.1,76.5 L77.2,76.5 L77.3,76.4 L77.4,76.3 L77.5,76.2 L77.5,76.2 L77.6,76.1 L77.7,76.0 L77.8,75.9 L77.9,75.9 L78.0,75.8 L78.1,75.7 L78.2,75.7 L78.3,75.6 L78.4,75.5 L78.5,75.5 L78.6,75.4 L78.6,75.3 L78.7,75.3 L78.8,75.2 L78.9,75.1 L79.0,75.1 L79.1,75.0 L79.2,74.9 L79.3,74.9 L79.4,74.8 L79.5,74.7 L79.6,74.7 L79.7,74.6 L79.8,74.5 L79.9,74.5 L80.0,74.4 L80.1,74.3 L80.1,74.3 L80.2,74.2 L80.3,74.1 L80.4,74.1 L80.5,74.0 L80.6,73.9 L80.7,73.9 L80.8,73.8 L80.9,73.8 L81.0,73.7 L81.1,73.6 L81.2,73.6 L81.3,73.5 L81.4,73.4 L81.5,73.4 L81.6,73.3 L81.7,73.3 L81.8,73.2 L81.9,73.1 L82.0,73.1 L82.1,73.0 L82.2,73.0 L82.3,72.9 L82.4,72.8 L82.5,72.8 L82.6,72.7 L82.7,72.7 L82.8,72.6 L82.9,72.6 L83.0,72.5 L83.1,72.4 L83.2,72.4 L83.3,72.3 L83.4,72.3 L83.5,72.2 L83.6,72.2 L83.7,72.1 L83.8,72.0 L83.9,72.0 L84.0,71.9 L84.1,71.9 L84.2,71.8 L84.3,71.8 L84.4,71.7 L84.5,71.6 L84.6,71.6 L84.7,71.5 L84.8,71.5 L84.9,71.4 L85.0,71.4 L85.1,71.3 L85.2,71.3 L85.3,71.2 L85.4,71.2 L85.5,71.1 L85.6,71.1 L85.7,71.0 L85.8,71.0 L85.9,70.9 L86.0,70.8 L86.1,70.8 L86.2,70.7 L86.3,70.7 L86.4,70.6 L86.5,70.6 L86.6,70.5 L86.7,70.5 L86.8,70.4 L86.9,70.4 L87.0,70.3 L87.1,70.3 L87.2,70.2 L87.3,70.2 L87.4,70.1 L87.6,70.1 L87.7,70.0 L87.8,70.0 L87.9,69.9 L88.0,69.9 L88.1,69.8 L88.2,69.8 L88.3,69.7 L88.4,69.7 L88.5,69.6 L88.6,69.6 L88.7,69.6 L88.8,69.5 L88.9,69.5 L89.0,69.4 L89.1,69.4 L89.2,69.3 L89.3,69.3 L89.4,69.2 L89.5,69.2 L89.6,69.1 L89.7,69.1 L89.9,69.0 L90.0,69.0 L90.1,68.9 L90.2,68.9 L90.3,68.9 L90.4,68.8 L90.5,68.8 L90.6,68.7 L90.7,68.7 L90.8,68.6 L90.9,68.6 L91.0,68.5 L91.1,68.5 L91.2,68.5 L91.3,68.4 L91.4,68.4 L91.5,68.3 L91.7,68.3 L91.8,68.2 L91.9,68.2 L92.0,68.1 L92.1,68.1 L92.2,68.1 L92.3,68.0 L92.4,68.0 L92.5,67.9 L92.6,67.9 L92.7,67.8 L92.8,67.8 L92.9,67.8 L93.0,67.7 L93.2,67.7 L93.3,67.6 L93.4,67.6 L93.5,67.6 L93.6,67.5 L93.7,67.5 L93.8,67.4 L93.9,67.4 L94.0,67.4 L94.1,67.3 L94.2,67.3 L94.3,67.2 L94.4,67.2 L94.6,67.2 L94.7,67.1 L94.8,67.1 L94.9,67.0 L95.0,67.0 L95.1,67.0 L95.2,66.9 L95.3,66.9 L95.4,66.8 L95.5,66.8 L95.6,66.8 L95.7,66.7 L95.9,66.7 L96.0,66.6 L96.1,66.6 L96.2,66.6 L96.3,66.5 L96.4,66.5 L96.5,66.5 L96.6,66.4 L96.7,66.4 L96.8,66.4 L96.9,66.3 L97.0,66.3 L97.2,66.2 L97.3,66.2 L97.4,66.2 L97.5,66.1 L97.6,66.1 L97.7,66.1 L97.8,66.0 L97.9,66.0 L98.0,66.0 L98.1,65.9 L98.3,65.9 L98.4,65.9 L98.5,65.8 L98.6,65.8 L98.7,65.8 L98.8,65.7 L98.9,65.7 L99.0,65.7 L99.1,65.6 L99.2,65.6 L99.4,65.6 L99.5,65.5 L99.6,65.5 L99.7,65.5 L99.8,65.4 L99.9,65.4 L100.0,65.4 L100.1,65.3 L100.2,65.3 L100.3,65.3 L100.5,65.2 L100.6,65.2 L100.7,65.2 L100.8,65.2 L100.9,65.1 L101.0,65.1 L101.1,65.1 L101.2,65.0 L101.3,65.0 L101.5,65.0 L101.6,64.9 L101.7,64.9 L101.8,64.9 L101.9,64.9 L102.0,64.8 L102.1,64.8 L102.2,64.8 L102.4,64.7 L102.5,64.7 L102.6,64.7 L102.7,64.7 L102.8,64.6 L102.9,64.6 L103.0,64.6 L103.1,64.6 L103.2,64.5 L103.4,64.5 L103.5,64.5 L103.6,64.5 L103.7,64.4 L103.8,64.4 L103.9,64.4 L104.0,64.4 L104.1,64.3 L104.3,64.3 L104.4,64.3 L104.5,64.3 L104.6,64.2 L104.7,64.2 L104.8,64.2 L104.9,64.2 L105.0,64.1 L105.2,64.1 L105.3,64.1 L105.4,64.1 L105.5,64.1 L105.6,64.0 L105.7,64.0 L105.8,64.0 L105.9,64.0 L106.1,64.0 L106.2,63.9 L106.3,63.9 L106.4,63.9 L106.5,63.9 L106.6,63.9 L106.7,63.8 L106.9,63.8 L107.0,63.8 L107.1,63.8 L107.2,63.8 L107.3,63.7 L107.4,63.7 L107.5,63.7 L107.7,63.7 L107.8,63.7 L107.9,63.6 L108.0,63.6 L108.1,63.6 L108.2,63.6 L108.3,63.6 L108.4,63.6 L108.6,63.6 L108.7,63.5 L108.8,63.5 L108.9,63.5 L109.0,63.5 L109.1,63.5 L109.2,63.5 L109.4,63.4 L109.5,63.4 L109.6,63.4 L109.7,63.4 L109.8,63.4 L109.9,63.4 L110.0,63.4 L110.2,63.4 L110.3,63.3 L110.4,63.3 L110.5,63.3 L110.6,63.3 L110.7,63.3 L110.8,63.3 L111.0,63.3 L111.1,63.3 L111.2,63.3 L111.3,63.3 L111.4,63.2 L111.5,63.2 L111.6,63.2 L111.8,63.2 L111.9,63.2 L112.0,63.2 L112.1,63.2 L112.2,63.2 L112.3,63.2 L112.5,63.2 L112.6,63.2 L112.7,63.2 L112.8,63.2 L112.9,63.1 L113.0,63.1 L113.1,63.1 L113.3,63.1 L113.4,63.1 L113.5,63.1 L113.6,63.1 L113.7,63.1 L113.8,63.1 L113.9,63.1 L114.1,63.1 L114.2,63.1 L114.3,63.1 L114.4,63.1 L114.5,63.1 L114.6,63.1 L114.8,63.1 L114.9,63.1 L115.0,63.1 L115.1,63.1 L115.2,63.1 L115.3,63.1 L115.4,63.1 L115.6,63.1 L115.7,63.1 L115.8,63.1 L115.9,63.1 L116.0,63.1 L116.1,63.1 L116.2,63.1 L116.4,63.1 L116.5,63.1 L116.6,63.1 L116.7,63.1 L116.8,63.1 L116.9,63.1 L117.1,63.1 L117.2,63.2 L117.3,63.2 L117.4,63.2 L117.5,63.2 L117.6,63.2 L117.7,63.2 L117.9,63.2 L118.0,63.2 L118.1,63.2 L118.2,63.2 L118.3,63.2 L118.4,63.2 L118.5,63.2 L118.7,63.3 L118.8,63.3 L118.9,63.3 L119.0,63.3 L119.1,63.3 L119.2,63.3 L119.3,63.3 L119.5,63.3 L119.6,63.3 L119.7,63.4 L119.8,63.4 L119.9,63.4 L120.0,63.4 L120.1,63.4 L120.3,63.4 L120.4,63.4 L120.5,63.5 L120.6,63.5 L120.7,63.5 L120.8,63.5 L120.9,63.5 L121.1,63.5 L121.2,63.6 L121.3,63.6 L121.4,63.6 L121.5,63.6 L121.6,63.6 L121.7,63.7 L121.8,63.7 L122.0,63.7 L122.1,63.7 L122.2,63.7 L122.3,63.8 L122.4,63.8 L122.5,63.8 L122.6,63.8 L122.8,63.9 L122.9,63.9 L123.0,63.9 L123.1,63.9 L123.2,64.0 L123.3,64.0 L123.4,64.0 L123.5,64.0 L123.7,64.1 L123.8,64.1 L123.9,64.1 L124.0,64.1 L124.1,64.2 L124.2,64.2 L124.3,64.2 L124.4,64.2 L124.5,64.3 L124.7,64.3 L124.8,64.3 L124.9,64.4 L125.0,64.4 L125.1,64.4 L125.2,64.5 L125.3,64.5 L125.4,64.5 L125.5,64.6 L125.6,64.6 L125.8,64.6 L125.9,64.7 L126.0,64.7 L126.1,64.7 L126.2,64.8 L126.3,64.8 L126.4,64.8 L126.5,64.9 L126.6,64.9 L126.7,64.9 L126.9,65.0 L127.0,65.0 L127.1,65.1 L127.2,65.1 L127.3,65.1 L127.4,65.2 L127.5,65.2 L127.6,65.3 L127.7,65.3 L127.8,65.3 L127.9,65.4 L128.0,65.4 L128.1,65.5 L128.3,65.5 L128.4,65.6 L128.5,65.6 L128.6,65.6 L128.7,65.7 L128.8,65.7 L128.9,65.8 L129.0,65.8 L129.1,65.9 L129.2,65.9 L129.3,66.0 L129.4,66.0 L129.5,66.1 L129.6,66.1 L129.7,66.2 L129.8,66.2 L129.9,66.3 L130.0,66.3 L130.1,66.4 L130.2,66.4 L130.4,66.5 L130.5,66.5 L130.6,66.6 L130.7,66.6 L130.8,66.7 L130.9,66.7 L131.0,66.8 L131.1,66.8 L131.2,66.9 L131.3,66.9 L131.4,67.0 L131.5,67.0 L131.6,67.1 L131.7,67.1 L131.8,67.2 L131.9,67.3 L132.0,67.3 L132.1,67.4 L132.2,67.4 L132.3,67.5 L132.4,67.6 L132.5,67.6 L132.6,67.7 L132.7,67.7 L132.8,67.8 L132.9,67.9 L133.0,67.9 L133.1,68.0 L133.2,68.0 L133.2,68.1 L133.3,68.2 L133.4,68.2 L133.5,68.3 L133.6,68.4 L133.7,68.4 L133.8,68.5 L133.9,68.5 L134.0,68.6 L134.1,68.7 L134.2,68.7 L134.3,68.8 L134.4,68.9 L134.5,68.9 L134.6,69.0 L134.7,69.1 L134.8,69.1 L134.9,69.2 L134.9,69.3 L135.0,69.4 L135.1,69.4 L135.2,69.5 L135.3,69.6 L135.4,69.6 L135.5,69.7 L135.6,69.8 L135.7,69.8 L135.8,69.9 L135.9,70.0 L135.9,70.1 L136.0,70.1 L136.1,70.2 L136.2,70.3 L136.3,70.4 L136.4,70.4 L136.5,70.5 L136.6,70.6 L136.7,70.6 L136.7,70.7 L136.8,70.8 L136.9,70.9 L137.0,70.9 L137.1,71.0 L137.2,71.1 L137.3,71.2 L137.3,71.3 L137.4,71.3 L137.5,71.4 L137.6,71.5 L137.7,71.6 L137.8,71.6 L137.9,71.7 L137.9,71.8 L138.0,71.9 L138.1,72.0 L138.2,72.0 L138.3,72.1 L138.3,72.2 L138.4,72.3 L138.5,72.4 L138.6,72.4 L138.7,72.5 L138.8,72.6 L138.8,72.7 L138.9,72.8 L139.0,72.9 L139.1,72.9 L139.2,73.0 L139.2,73.1 L139.3,73.2 L139.4,73.3 L139.5,73.4 L139.6,73.4 L139.6,73.5 L139.7,73.6 L139.8,73.7 L139.9,73.8 L139.9,73.9 L140.0,73.9 L140.1,74.0 L140.2,74.1 L140.2,74.2 L140.3,74.3 L140.4,74.4 L140.5,74.5 L140.5,74.6 L140.6,74.6 L140.7,74.7 L140.8,74.8 L140.8,74.9 L140.9,75.0 L141.0,75.1 L141.1,75.2 L141.1,75.3 L141.2,75.3 L141.3,75.4 L141.4,75.5 L141.4,75.6 L141.5,75.7 L141.6,75.8 L141.6,75.9 L141.7,76.0 L141.8,76.1 L141.8,76.2 L141.9,76.3 L142.0,76.3 L142.1,76.4 L142.1,76.5 L142.2,76.6 L142.3,76.7 L142.3,76.8 L142.4,76.9 L142.5,77.0 L142.5,77.1 L142.6,77.2 L142.7,77.3 L142.7,77.4 L142.8,77.5 L142.9,77.6 L142.9,77.7 L143.0,77.7 L143.1,77.8 L143.1,77.9 L143.2,78.0 L143.3,78.1 L143.3,78.2 L143.4,78.3 L143.4,78.4 L143.5,78.5 L143.6,78.6 L143.6,78.7 L143.7,78.8 L143.8,78.9 L143.8,79.0 L143.9,79.1 L143.9,79.2 L144.0,79.3 L144.1,79.4 L144.1,79.5 L144.2,79.6 L144.2,79.7 L144.3,79.8 L144.4,79.9 L144.4,80.0 L144.5,80.1 L144.5,80.2 L144.6,80.3 L144.6,80.4 L144.7,80.5 L144.8,80.6 L144.8,80.7 L144.9,80.8 L144.9,80.9 L145.0,81.0 L145.0,81.1 L145.1,81.2 L145.1,81.3 L145.2,81.4 L145.2,81.5 L145.3,81.6 L145.4,81.7 L145.4,81.8 L145.5,81.9 L145.5,82.0 L145.6,82.1 L145.6,82.2 L145.7,82.3 L145.7,82.4 L145.8,82.5 L145.8,82.6 L145.9,82.7 L145.9,82.8 L146.0,82.9 L146.0,83.0 L146.0,83.1 L146.1,83.3 L146.1,83.4 L146.2,83.5 L146.2,83.6 L146.3,83.7 L146.3,83.8 L146.4,83.9 L146.4,84.0 L146.5,84.1 L146.5,84.2 L146.5,84.3 L146.6,84.4 L146.6,84.5 L146.7,84.6 L146.7,84.7 L146.8,84.8 L146.8,85.0 L146.8,85.1 L146.9,85.2 L146.9,85.3 L147.0,85.4 L147.0,85.5 L147.0,85.6 L147.1,85.7 L147.1,85.8 L147.2,85.9 L147.2,86.0 L147.2,86.1 L147.3,86.3 L147.3,86.4 L147.3,86.5 L147.4,86.6 L147.4,86.7 L147.4,86.8 L147.5,86.9 L147.5,87.0 L147.5,87.1 L147.6,87.2 L147.6,87.4 L147.6,87.5 L147.7,87.6 L147.7,87.7 L147.7,87.8 L147.7,87.9 L147.8,88.0 L147.8,88.1 L147.8,88.2 L147.9,88.4 L147.9,88.5 L147.9,88.6 L147.9,88.7 L148.0,88.8 L148.0,88.9 L148.0,89.0 L148.0,89.1 L148.1,89.3 L148.1,89.4 L148.1,89.5 L148.1,89.6 L148.1,89.7 L148.2,89.8 L148.2,89.9 L148.2,90.0 L148.2,90.2 L148.2,90.3 L148.3,90.4 L148.3,90.5 L148.3,90.6 L148.3,90.7 L148.3,90.8 L148.3,91.0 L148.4,91.1 L148.4,91.2 L148.4,91.3 L148.4,91.4 L148.4,91.5 L148.4,91.6 L148.4,91.8 L148.4,91.9 L148.4,92.0 L148.5,92.1 L148.5,92.2 L148.5,92.3 L148.5,92.4 L148.5,92.6 L148.5,92.7 L148.5,92.8 L148.5,92.9 L148.5,93.0 L148.5,93.1 L148.5,93.2 L148.5,93.4 L148.5,93.5 L148.5,93.6 L148.5,93.7 L148.5,93.8 L148.5,93.9 L148.5,94.1 L148.5,94.2 L148.5,94.3 L148.5,94.4 L148.5,94.5 L148.5,94.6 L148.5,94.7 L148.5,94.9 L148.5,95.0 L148.5,95.1 L148.4,95.2 L148.4,95.3 L148.4,95.4 L148.4,95.5 L148.4,95.7 Z" fill="none" stroke="#5f6673" stroke-width="3" stroke-linejoin="round"/><circle cx="148.4" cy="95.8" r="3.4" fill="#1d2330" stroke="#ffffff" stroke-width="1"/><text x="107.5" y="190.8" font-family="DejaVu Sans" font-size="9.5" fill="#5f6673" text-anchor="middle" font-weight="normal">2.1 km · road 13.4–16.0 m</text><rect x="225.0" y="0" width="215.0" height="200.82337120641645" rx="3" fill="#f5f6f8"/><text x="332.5" y="16.0" font-family="DejaVu Sans" font-size="11" fill="#1d2330" text-anchor="middle" font-weight="bold">Medium</text><text x="332.5" y="29.0" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle" font-weight="normal">generated, display seed 1000</text><path d="M406.3,101.6 L406.4,101.4 L406.4,101.3 L406.5,101.2 L406.5,101.1 L406.6,101.0 L406.6,100.9 L406.7,100.8 L406.7,100.7 L406.7,100.6 L406.8,100.5 L406.8,100.4 L406.9,100.3 L406.9,100.2 L407.0,100.1 L407.0,100.0 L407.0,99.9 L407.1,99.8 L407.1,99.6 L407.2,99.5 L407.2,99.4 L407.2,99.3 L407.3,99.2 L407.3,99.1 L407.3,99.0 L407.4,98.9 L407.4,98.8 L407.4,98.7 L407.5,98.5 L407.5,98.4 L407.5,98.3 L407.5,98.2 L407.6,98.1 L407.6,98.0 L407.6,97.9 L407.6,97.8 L407.7,97.7 L407.7,97.5 L407.7,97.4 L407.7,97.3 L407.8,97.2 L407.8,97.1 L407.8,97.0 L407.8,96.9 L407.8,96.7 L407.9,96.6 L407.9,96.5 L407.9,96.4 L407.9,96.3 L407.9,96.2 L407.9,96.1 L407.9,95.9 L407.9,95.8 L408.0,95.7 L408.0,95.6 L408.0,95.5 L408.0,95.4 L408.0,95.3 L408.0,95.1 L408.0,95.0 L408.0,94.9 L408.0,94.8 L408.0,94.7 L408.0,94.6 L408.0,94.5 L408.0,94.3 L408.0,94.2 L408.0,94.1 L408.0,94.0 L408.0,93.9 L408.0,93.8 L408.0,93.7 L408.0,93.5 L408.0,93.4 L408.0,93.3 L408.0,93.2 L408.0,93.1 L408.0,93.0 L407.9,92.8 L407.9,92.7 L407.9,92.6 L407.9,92.5 L407.9,92.4 L407.9,92.3 L407.9,92.2 L407.9,92.0 L407.8,91.9 L407.8,91.8 L407.8,91.7 L407.8,91.6 L407.8,91.5 L407.7,91.4 L407.7,91.3 L407.7,91.1 L407.7,91.0 L407.6,90.9 L407.6,90.8 L407.6,90.7 L407.6,90.6 L407.5,90.5 L407.5,90.4 L407.5,90.2 L407.5,90.1 L407.4,90.0 L407.4,89.9 L407.4,89.8 L407.3,89.7 L407.3,89.6 L407.3,89.5 L407.2,89.4 L407.2,89.3 L407.2,89.1 L407.1,89.0 L407.1,88.9 L407.0,88.8 L407.0,88.7 L407.0,88.6 L406.9,88.5 L406.9,88.4 L406.8,88.3 L406.8,88.2 L406.7,88.1 L406.7,88.0 L406.7,87.9 L406.6,87.8 L406.6,87.7 L406.5,87.5 L406.5,87.4 L406.4,87.3 L406.4,87.2 L406.3,87.1 L406.3,87.0 L406.2,86.9 L406.2,86.8 L406.1,86.7 L406.0,86.6 L406.0,86.5 L405.9,86.4 L405.9,86.3 L405.8,86.2 L405.8,86.1 L405.7,86.0 L405.6,85.9 L405.6,85.8 L405.5,85.7 L405.5,85.6 L405.4,85.5 L405.3,85.4 L405.3,85.3 L405.2,85.2 L405.1,85.2 L405.1,85.1 L405.0,85.0 L404.9,84.9 L404.9,84.8 L404.8,84.7 L404.7,84.6 L404.7,84.5 L404.6,84.4 L404.5,84.3 L404.5,84.2 L404.4,84.1 L404.3,84.1 L404.2,84.0 L404.2,83.9 L404.1,83.8 L404.0,83.7 L403.9,83.6 L403.9,83.5 L403.8,83.4 L403.7,83.4 L403.6,83.3 L403.5,83.2 L403.5,83.1 L403.4,83.0 L403.3,82.9 L403.2,82.9 L403.1,82.8 L403.1,82.7 L403.0,82.6 L402.9,82.5 L402.8,82.5 L402.7,82.4 L402.6,82.3 L402.5,82.2 L402.5,82.2 L402.4,82.1 L402.3,82.0 L402.2,81.9 L402.1,81.9 L402.0,81.8 L401.9,81.7 L401.8,81.7 L401.8,81.6 L401.7,81.5 L401.6,81.4 L401.5,81.4 L401.4,81.3 L401.3,81.2 L401.2,81.2 L401.1,81.1 L401.0,81.0 L400.9,80.9 L400.8,80.9 L400.8,80.8 L400.7,80.7 L400.6,80.7 L400.5,80.6 L400.4,80.5 L400.3,80.4 L400.2,80.4 L400.1,80.3 L400.0,80.2 L399.9,80.2 L399.8,80.1 L399.8,80.0 L399.7,80.0 L399.6,79.9 L399.5,79.8 L399.4,79.7 L399.3,79.7 L399.2,79.6 L399.1,79.5 L399.0,79.5 L398.9,79.4 L398.9,79.3 L398.8,79.2 L398.7,79.2 L398.6,79.1 L398.5,79.0 L398.4,79.0 L398.3,78.9 L398.2,78.8 L398.1,78.7 L398.0,78.7 L397.9,78.6 L397.9,78.5 L397.8,78.5 L397.7,78.4 L397.6,78.3 L397.5,78.2 L397.4,78.2 L397.3,78.1 L397.2,78.0 L397.1,78.0 L397.0,77.9 L397.0,77.8 L396.9,77.8 L396.8,77.7 L396.7,77.6 L396.6,77.5 L396.5,77.5 L396.4,77.4 L396.3,77.3 L396.2,77.3 L396.1,77.2 L396.0,77.1 L396.0,77.0 L395.9,77.0 L395.8,76.9 L395.7,76.8 L395.6,76.8 L395.5,76.7 L395.4,76.6 L395.3,76.5 L395.2,76.5 L395.1,76.4 L395.1,76.3 L395.0,76.3 L394.9,76.2 L394.8,76.1 L394.7,76.0 L394.6,76.0 L394.5,75.9 L394.4,75.8 L394.3,75.8 L394.2,75.7 L394.1,75.6 L394.1,75.6 L394.0,75.5 L393.9,75.4 L393.8,75.3 L393.7,75.3 L393.6,75.2 L393.5,75.1 L393.4,75.1 L393.3,75.0 L393.2,74.9 L393.2,74.8 L393.1,74.8 L393.0,74.7 L392.9,74.6 L392.8,74.6 L392.7,74.5 L392.6,74.4 L392.5,74.3 L392.4,74.3 L392.3,74.2 L392.2,74.1 L392.2,74.1 L392.1,74.0 L392.0,73.9 L391.9,73.8 L391.8,73.8 L391.7,73.7 L391.6,73.6 L391.5,73.6 L391.4,73.5 L391.3,73.4 L391.2,73.4 L391.2,73.3 L391.1,73.2 L391.0,73.1 L390.9,73.1 L390.8,73.0 L390.7,72.9 L390.6,72.9 L390.5,72.8 L390.4,72.7 L390.3,72.6 L390.3,72.6 L390.2,72.5 L390.1,72.4 L390.0,72.4 L389.9,72.3 L389.8,72.2 L389.7,72.1 L389.6,72.1 L389.5,72.0 L389.4,71.9 L389.3,71.9 L389.3,71.8 L389.2,71.7 L389.1,71.7 L389.0,71.6 L388.9,71.5 L388.8,71.4 L388.7,71.4 L388.6,71.3 L388.5,71.2 L388.4,71.2 L388.4,71.1 L388.3,71.0 L388.2,70.9 L388.1,70.9 L388.0,70.8 L387.9,70.7 L387.8,70.7 L387.7,70.6 L387.6,70.5 L387.5,70.4 L387.4,70.4 L387.4,70.3 L387.3,70.2 L387.2,70.2 L387.1,70.1 L387.0,70.0 L386.9,69.9 L386.8,69.9 L386.7,69.8 L386.6,69.7 L386.5,69.7 L386.4,69.6 L386.4,69.5 L386.3,69.5 L386.2,69.4 L386.1,69.3 L386.0,69.3 L385.9,69.2 L385.8,69.2 L385.7,69.1 L385.6,69.1 L385.5,69.0 L385.3,69.0 L385.2,68.9 L385.1,68.9 L385.0,68.8 L384.9,68.8 L384.8,68.8 L384.7,68.7 L384.6,68.7 L384.5,68.7 L384.4,68.6 L384.3,68.6 L384.2,68.5 L384.1,68.5 L383.9,68.5 L383.8,68.4 L383.7,68.4 L383.6,68.4 L383.5,68.3 L383.4,68.3 L383.3,68.3 L383.2,68.2 L383.1,68.2 L383.0,68.1 L382.9,68.1 L382.7,68.1 L382.6,68.0 L382.5,68.0 L382.4,68.0 L382.3,67.9 L382.2,67.9 L382.1,67.9 L382.0,67.8 L381.9,67.8 L381.8,67.7 L381.7,67.7 L381.5,67.7 L381.4,67.6 L381.3,67.6 L381.2,67.6 L381.1,67.5 L381.0,67.5 L380.9,67.5 L380.8,67.4 L380.7,67.4 L380.6,67.3 L380.5,67.3 L380.3,67.3 L380.2,67.2 L380.1,67.2 L380.0,67.2 L379.9,67.1 L379.8,67.1 L379.7,67.0 L379.6,67.0 L379.5,67.0 L379.4,66.9 L379.3,66.9 L379.1,66.9 L379.0,66.8 L378.9,66.8 L378.8,66.8 L378.7,66.7 L378.6,66.7 L378.5,66.6 L378.4,66.6 L378.3,66.6 L378.2,66.5 L378.1,66.5 L377.9,66.5 L377.8,66.4 L377.7,66.4 L377.6,66.4 L377.5,66.3 L377.4,66.3 L377.3,66.2 L377.2,66.2 L377.1,66.2 L377.0,66.1 L376.9,66.1 L376.7,66.1 L376.6,66.0 L376.5,66.0 L376.4,65.9 L376.3,65.9 L376.2,65.9 L376.1,65.8 L376.0,65.8 L375.9,65.8 L375.8,65.7 L375.7,65.7 L375.5,65.7 L375.4,65.6 L375.3,65.6 L375.2,65.5 L375.1,65.5 L375.0,65.5 L374.9,65.4 L374.8,65.4 L374.7,65.4 L374.6,65.3 L374.5,65.3 L374.3,65.3 L374.2,65.2 L374.1,65.2 L374.0,65.1 L373.9,65.1 L373.8,65.1 L373.7,65.0 L373.6,65.0 L373.5,65.0 L373.4,64.9 L373.3,64.9 L373.2,64.8 L373.0,64.8 L372.9,64.8 L372.8,64.7 L372.7,64.7 L372.6,64.7 L372.5,64.6 L372.4,64.6 L372.3,64.6 L372.2,64.5 L372.1,64.5 L372.0,64.4 L371.8,64.4 L371.7,64.4 L371.6,64.3 L371.5,64.3 L371.4,64.3 L371.3,64.2 L371.2,64.2 L371.1,64.2 L371.0,64.1 L370.9,64.1 L370.8,64.0 L370.6,64.0 L370.5,64.0 L370.4,63.9 L370.3,63.9 L370.2,63.9 L370.1,63.8 L370.0,63.8 L369.9,63.7 L369.8,63.7 L369.7,63.7 L369.6,63.6 L369.4,63.6 L369.3,63.6 L369.2,63.5 L369.1,63.5 L369.0,63.5 L368.9,63.4 L368.8,63.4 L368.7,63.3 L368.6,63.3 L368.5,63.3 L368.4,63.2 L368.2,63.2 L368.1,63.2 L368.0,63.1 L367.9,63.1 L367.8,63.1 L367.7,63.0 L367.6,63.0 L367.5,62.9 L367.4,62.9 L367.3,62.9 L367.2,62.8 L367.0,62.8 L366.9,62.8 L366.8,62.7 L366.7,62.7 L366.6,62.6 L366.5,62.6 L366.4,62.6 L366.3,62.5 L366.2,62.5 L366.1,62.5 L366.0,62.4 L365.8,62.4 L365.7,62.4 L365.6,62.3 L365.5,62.3 L365.4,62.2 L365.3,62.2 L365.2,62.2 L365.1,62.1 L365.0,62.1 L364.9,62.1 L364.8,62.0 L364.6,62.0 L364.5,62.0 L364.4,61.9 L364.3,61.9 L364.2,61.8 L364.1,61.8 L364.0,61.8 L363.9,61.7 L363.8,61.7 L363.7,61.7 L363.6,61.6 L363.4,61.6 L363.3,61.5 L363.2,61.5 L363.1,61.5 L363.0,61.4 L362.9,61.4 L362.8,61.4 L362.7,61.3 L362.6,61.3 L362.5,61.3 L362.4,61.2 L362.3,61.2 L362.1,61.1 L362.0,61.1 L361.9,61.1 L361.8,61.0 L361.7,61.0 L361.6,61.0 L361.5,60.9 L361.4,60.9 L361.3,60.9 L361.2,60.8 L361.1,60.8 L360.9,60.7 L360.8,60.7 L360.7,60.7 L360.6,60.6 L360.5,60.6 L360.4,60.6 L360.3,60.5 L360.2,60.5 L360.1,60.4 L360.0,60.4 L359.9,60.4 L359.7,60.3 L359.6,60.3 L359.5,60.3 L359.4,60.2 L359.3,60.2 L359.2,60.2 L359.1,60.1 L359.0,60.1 L358.9,60.0 L358.8,60.0 L358.7,60.0 L358.5,59.9 L358.4,59.9 L358.3,59.9 L358.2,59.8 L358.1,59.8 L358.0,59.8 L357.9,59.7 L357.8,59.7 L357.7,59.6 L357.6,59.6 L357.5,59.6 L357.3,59.5 L357.2,59.5 L357.1,59.5 L357.0,59.4 L356.9,59.4 L356.8,59.3 L356.7,59.3 L356.6,59.3 L356.5,59.2 L356.4,59.2 L356.3,59.2 L356.1,59.1 L356.0,59.1 L355.9,59.1 L355.8,59.0 L355.7,59.0 L355.6,58.9 L355.5,58.9 L355.4,58.9 L355.3,58.8 L355.2,58.8 L355.1,58.8 L354.9,58.7 L354.8,58.7 L354.7,58.7 L354.6,58.6 L354.5,58.6 L354.4,58.5 L354.3,58.5 L354.2,58.5 L354.1,58.4 L354.0,58.4 L353.9,58.4 L353.7,58.3 L353.6,58.3 L353.5,58.2 L353.4,58.2 L353.3,58.2 L353.2,58.1 L353.1,58.1 L353.0,58.1 L352.9,58.0 L352.8,58.0 L352.7,58.0 L352.5,57.9 L352.4,57.9 L352.3,57.8 L352.2,57.8 L352.1,57.8 L352.0,57.7 L351.9,57.7 L351.8,57.7 L351.7,57.6 L351.6,57.6 L351.5,57.6 L351.4,57.5 L351.2,57.5 L351.1,57.4 L351.0,57.4 L350.9,57.4 L350.8,57.3 L350.7,57.3 L350.6,57.3 L350.5,57.2 L350.4,57.2 L350.3,57.1 L350.2,57.1 L350.0,57.1 L349.9,57.0 L349.8,57.0 L349.7,57.0 L349.6,56.9 L349.5,56.9 L349.4,56.9 L349.3,56.8 L349.2,56.8 L349.1,56.7 L349.0,56.7 L348.8,56.7 L348.7,56.6 L348.6,56.6 L348.5,56.6 L348.4,56.5 L348.3,56.5 L348.2,56.5 L348.1,56.4 L348.0,56.4 L347.9,56.3 L347.8,56.3 L347.6,56.3 L347.5,56.2 L347.4,56.2 L347.3,56.2 L347.2,56.1 L347.1,56.1 L347.0,56.0 L346.9,56.0 L346.8,56.0 L346.7,55.9 L346.6,55.9 L346.4,55.9 L346.3,55.8 L346.2,55.8 L346.1,55.8 L346.0,55.7 L345.9,55.7 L345.8,55.6 L345.7,55.6 L345.6,55.6 L345.5,55.5 L345.4,55.5 L345.2,55.5 L345.1,55.4 L345.0,55.4 L344.9,55.4 L344.8,55.3 L344.7,55.3 L344.6,55.2 L344.5,55.2 L344.4,55.2 L344.3,55.1 L344.2,55.1 L344.0,55.1 L343.9,55.0 L343.8,55.0 L343.7,54.9 L343.6,54.9 L343.5,54.9 L343.4,54.8 L343.3,54.8 L343.2,54.8 L343.1,54.7 L343.0,54.7 L342.8,54.7 L342.7,54.6 L342.6,54.6 L342.5,54.5 L342.4,54.5 L342.3,54.5 L342.2,54.4 L342.1,54.4 L342.0,54.4 L341.9,54.3 L341.8,54.3 L341.6,54.3 L341.5,54.2 L341.4,54.2 L341.3,54.1 L341.2,54.1 L341.1,54.1 L341.0,54.0 L340.9,54.0 L340.8,54.0 L340.7,53.9 L340.6,53.9 L340.5,53.8 L340.3,53.8 L340.2,53.8 L340.1,53.7 L340.0,53.7 L339.9,53.7 L339.8,53.6 L339.7,53.6 L339.6,53.6 L339.5,53.5 L339.4,53.5 L339.3,53.4 L339.1,53.4 L339.0,53.4 L338.9,53.3 L338.8,53.3 L338.7,53.3 L338.6,53.2 L338.5,53.2 L338.4,53.2 L338.3,53.1 L338.2,53.1 L338.1,53.0 L337.9,53.0 L337.8,53.0 L337.7,52.9 L337.6,52.9 L337.5,52.9 L337.4,52.8 L337.3,52.8 L337.2,52.7 L337.1,52.7 L337.0,52.7 L336.9,52.6 L336.7,52.6 L336.6,52.6 L336.5,52.5 L336.4,52.5 L336.3,52.5 L336.2,52.4 L336.1,52.4 L336.0,52.3 L335.9,52.3 L335.8,52.3 L335.7,52.2 L335.5,52.2 L335.4,52.2 L335.3,52.1 L335.2,52.1 L335.1,52.1 L335.0,52.0 L334.9,52.0 L334.8,51.9 L334.7,51.9 L334.6,51.9 L334.5,51.8 L334.3,51.8 L334.2,51.8 L334.1,51.7 L334.0,51.7 L333.9,51.7 L333.8,51.6 L333.7,51.6 L333.6,51.5 L333.5,51.5 L333.4,51.5 L333.3,51.4 L333.1,51.4 L333.0,51.4 L332.9,51.3 L332.8,51.3 L332.7,51.2 L332.6,51.2 L332.5,51.2 L332.4,51.1 L332.3,51.1 L332.2,51.1 L332.1,51.0 L331.9,51.0 L331.8,51.0 L331.7,50.9 L331.6,50.9 L331.5,50.8 L331.4,50.8 L331.3,50.8 L331.2,50.7 L331.1,50.7 L331.0,50.7 L330.9,50.6 L330.7,50.6 L330.6,50.6 L330.5,50.5 L330.4,50.5 L330.3,50.4 L330.2,50.4 L330.1,50.4 L330.0,50.3 L329.9,50.3 L329.8,50.3 L329.7,50.2 L329.5,50.2 L329.4,50.1 L329.3,50.1 L329.2,50.1 L329.1,50.0 L329.0,50.0 L328.9,50.0 L328.8,49.9 L328.7,49.9 L328.6,49.9 L328.5,49.8 L328.4,49.8 L328.2,49.7 L328.1,49.7 L328.0,49.7 L327.9,49.6 L327.8,49.6 L327.7,49.6 L327.6,49.5 L327.5,49.5 L327.4,49.5 L327.3,49.4 L327.2,49.4 L327.0,49.3 L326.9,49.3 L326.8,49.3 L326.7,49.2 L326.6,49.2 L326.5,49.2 L326.4,49.1 L326.3,49.1 L326.2,49.0 L326.1,49.0 L326.0,49.0 L325.8,48.9 L325.7,48.9 L325.6,48.9 L325.5,48.8 L325.4,48.8 L325.3,48.8 L325.2,48.7 L325.1,48.7 L325.0,48.6 L324.9,48.6 L324.8,48.6 L324.6,48.5 L324.5,48.5 L324.4,48.5 L324.3,48.4 L324.2,48.4 L324.1,48.4 L324.0,48.3 L323.9,48.3 L323.8,48.2 L323.7,48.2 L323.6,48.2 L323.4,48.1 L323.3,48.1 L323.2,48.1 L323.1,48.0 L323.0,48.0 L322.9,47.9 L322.8,47.9 L322.7,47.9 L322.6,47.8 L322.5,47.8 L322.4,47.8 L322.2,47.7 L322.1,47.7 L322.0,47.7 L321.9,47.6 L321.8,47.6 L321.7,47.5 L321.6,47.5 L321.5,47.5 L321.4,47.4 L321.3,47.4 L321.2,47.4 L321.0,47.3 L320.9,47.3 L320.8,47.3 L320.7,47.2 L320.6,47.2 L320.5,47.1 L320.4,47.1 L320.3,47.1 L320.2,47.0 L320.1,47.0 L320.0,47.0 L319.8,46.9 L319.7,46.9 L319.6,46.8 L319.5,46.8 L319.4,46.8 L319.3,46.7 L319.2,46.7 L319.1,46.7 L319.0,46.6 L318.9,46.6 L318.8,46.6 L318.6,46.5 L318.5,46.5 L318.4,46.4 L318.3,46.4 L318.2,46.4 L318.1,46.3 L318.0,46.3 L317.9,46.3 L317.8,46.2 L317.7,46.2 L317.6,46.2 L317.5,46.1 L317.3,46.1 L317.2,46.0 L317.1,46.0 L317.0,46.0 L316.9,45.9 L316.8,45.9 L316.7,45.9 L316.6,45.8 L316.5,45.8 L316.4,45.7 L316.2,45.7 L316.1,45.7 L316.0,45.7 L315.9,45.6 L315.8,45.6 L315.7,45.6 L315.6,45.6 L315.5,45.6 L315.3,45.6 L315.2,45.6 L315.1,45.6 L315.0,45.6 L314.9,45.6 L314.8,45.6 L314.7,45.6 L314.5,45.6 L314.4,45.6 L314.3,45.7 L314.2,45.7 L314.1,45.7 L314.0,45.7 L313.9,45.8 L313.8,45.8 L313.6,45.8 L313.5,45.9 L313.4,45.9 L313.3,45.9 L313.2,46.0 L313.1,46.0 L313.0,46.0 L312.9,46.0 L312.8,46.1 L312.6,46.1 L312.5,46.1 L312.4,46.2 L312.3,46.2 L312.2,46.2 L312.1,46.3 L312.0,46.3 L311.9,46.3 L311.8,46.4 L311.7,46.4 L311.5,46.4 L311.4,46.4 L311.3,46.5 L311.2,46.5 L311.1,46.5 L311.0,46.6 L310.9,46.6 L310.8,46.6 L310.7,46.7 L310.5,46.7 L310.4,46.7 L310.3,46.8 L310.2,46.8 L310.1,46.8 L310.0,46.8 L309.9,46.9 L309.8,46.9 L309.7,46.9 L309.5,47.0 L309.4,47.0 L309.3,47.0 L309.2,47.1 L309.1,47.1 L309.0,47.1 L308.9,47.2 L308.8,47.2 L308.7,47.2 L308.6,47.2 L308.4,47.3 L308.3,47.3 L308.2,47.3 L308.1,47.4 L308.0,47.4 L307.9,47.4 L307.8,47.5 L307.7,47.5 L307.6,47.5 L307.4,47.6 L307.3,47.6 L307.2,47.6 L307.1,47.7 L307.0,47.7 L306.9,47.7 L306.8,47.7 L306.7,47.8 L306.6,47.8 L306.4,47.8 L306.3,47.9 L306.2,47.9 L306.1,47.9 L306.0,48.0 L305.9,48.0 L305.8,48.0 L305.7,48.1 L305.6,48.1 L305.4,48.1 L305.3,48.1 L305.2,48.2 L305.1,48.2 L305.0,48.2 L304.9,48.3 L304.8,48.3 L304.7,48.3 L304.6,48.4 L304.5,48.4 L304.3,48.4 L304.2,48.5 L304.1,48.5 L304.0,48.5 L303.9,48.5 L303.8,48.6 L303.7,48.6 L303.6,48.6 L303.5,48.7 L303.3,48.7 L303.2,48.7 L303.1,48.8 L303.0,48.8 L302.9,48.8 L302.8,48.9 L302.7,48.9 L302.6,48.9 L302.5,49.0 L302.3,49.0 L302.2,49.0 L302.1,49.0 L302.0,49.1 L301.9,49.1 L301.8,49.1 L301.7,49.2 L301.6,49.2 L301.5,49.2 L301.4,49.3 L301.2,49.3 L301.1,49.3 L301.0,49.4 L300.9,49.4 L300.8,49.4 L300.7,49.4 L300.6,49.5 L300.5,49.5 L300.4,49.5 L300.2,49.6 L300.1,49.6 L300.0,49.6 L299.9,49.7 L299.8,49.7 L299.7,49.7 L299.6,49.8 L299.5,49.8 L299.4,49.8 L299.2,49.8 L299.1,49.9 L299.0,49.9 L298.9,49.9 L298.8,50.0 L298.7,50.0 L298.6,50.0 L298.5,50.1 L298.4,50.1 L298.2,50.1 L298.1,50.2 L298.0,50.2 L297.9,50.2 L297.8,50.2 L297.7,50.3 L297.6,50.3 L297.5,50.3 L297.4,50.4 L297.3,50.4 L297.1,50.4 L297.0,50.5 L296.9,50.5 L296.8,50.5 L296.7,50.6 L296.6,50.6 L296.5,50.6 L296.4,50.7 L296.3,50.7 L296.1,50.7 L296.0,50.7 L295.9,50.8 L295.8,50.8 L295.7,50.8 L295.6,50.9 L295.5,50.9 L295.4,50.9 L295.3,51.0 L295.1,51.0 L295.0,51.0 L294.9,51.1 L294.8,51.1 L294.7,51.1 L294.6,51.1 L294.5,51.2 L294.4,51.2 L294.3,51.2 L294.2,51.3 L294.0,51.3 L293.9,51.3 L293.8,51.4 L293.7,51.4 L293.6,51.4 L293.5,51.5 L293.4,51.5 L293.3,51.5 L293.2,51.5 L293.0,51.6 L292.9,51.6 L292.8,51.6 L292.7,51.7 L292.6,51.7 L292.5,51.7 L292.4,51.8 L292.3,51.8 L292.2,51.8 L292.0,51.9 L291.9,51.9 L291.8,51.9 L291.7,52.0 L291.6,52.0 L291.5,52.0 L291.4,52.0 L291.3,52.1 L291.2,52.1 L291.1,52.1 L290.9,52.2 L290.8,52.2 L290.7,52.2 L290.6,52.3 L290.5,52.3 L290.4,52.3 L290.3,52.4 L290.2,52.4 L290.1,52.4 L289.9,52.4 L289.8,52.5 L289.7,52.5 L289.6,52.5 L289.5,52.6 L289.4,52.6 L289.3,52.6 L289.2,52.7 L289.1,52.7 L288.9,52.7 L288.8,52.8 L288.7,52.8 L288.6,52.8 L288.5,52.8 L288.4,52.9 L288.3,52.9 L288.2,52.9 L288.1,53.0 L287.9,53.0 L287.8,53.0 L287.7,53.1 L287.6,53.1 L287.5,53.1 L287.4,53.2 L287.3,53.2 L287.2,53.2 L287.1,53.2 L287.0,53.3 L286.8,53.3 L286.7,53.3 L286.6,53.4 L286.5,53.4 L286.4,53.4 L286.3,53.5 L286.2,53.5 L286.1,53.5 L286.0,53.6 L285.8,53.6 L285.7,53.6 L285.6,53.7 L285.5,53.7 L285.4,53.7 L285.3,53.7 L285.2,53.8 L285.1,53.8 L285.0,53.8 L284.8,53.9 L284.7,53.9 L284.6,53.9 L284.5,54.0 L284.4,54.0 L284.3,54.0 L284.2,54.1 L284.1,54.1 L284.0,54.1 L283.9,54.1 L283.7,54.2 L283.6,54.2 L283.5,54.2 L283.4,54.3 L283.3,54.3 L283.2,54.3 L283.1,54.4 L283.0,54.4 L282.9,54.4 L282.7,54.5 L282.6,54.5 L282.5,54.5 L282.4,54.5 L282.3,54.6 L282.2,54.6 L282.1,54.6 L282.0,54.7 L281.9,54.7 L281.7,54.7 L281.6,54.8 L281.5,54.8 L281.4,54.8 L281.3,54.9 L281.2,54.9 L281.1,54.9 L281.0,55.0 L280.9,55.0 L280.7,55.0 L280.6,55.0 L280.5,55.1 L280.4,55.1 L280.3,55.1 L280.2,55.2 L280.1,55.2 L280.0,55.2 L279.9,55.3 L279.8,55.3 L279.6,55.3 L279.5,55.4 L279.4,55.4 L279.3,55.4 L279.2,55.4 L279.1,55.5 L279.0,55.5 L278.9,55.5 L278.8,55.6 L278.6,55.6 L278.5,55.6 L278.4,55.7 L278.3,55.7 L278.2,55.7 L278.1,55.8 L278.0,55.8 L277.9,55.8 L277.8,55.8 L277.6,55.9 L277.5,55.9 L277.4,55.9 L277.3,56.0 L277.2,56.0 L277.1,56.0 L277.0,56.1 L276.9,56.1 L276.8,56.1 L276.7,56.2 L276.5,56.2 L276.4,56.2 L276.3,56.2 L276.2,56.3 L276.1,56.3 L276.0,56.3 L275.9,56.4 L275.8,56.4 L275.7,56.4 L275.5,56.5 L275.4,56.5 L275.3,56.5 L275.2,56.6 L275.1,56.6 L275.0,56.6 L274.9,56.7 L274.8,56.7 L274.7,56.7 L274.5,56.7 L274.4,56.8 L274.3,56.8 L274.2,56.8 L274.1,56.9 L274.0,56.9 L273.9,56.9 L273.8,57.0 L273.7,57.0 L273.5,57.0 L273.4,57.1 L273.3,57.1 L273.2,57.1 L273.1,57.1 L273.0,57.2 L272.9,57.2 L272.8,57.2 L272.7,57.3 L272.6,57.3 L272.4,57.3 L272.3,57.4 L272.2,57.4 L272.1,57.4 L272.0,57.5 L271.9,57.5 L271.8,57.5 L271.7,57.5 L271.6,57.6 L271.4,57.6 L271.3,57.6 L271.2,57.7 L271.1,57.7 L271.0,57.7 L270.9,57.8 L270.8,57.8 L270.7,57.8 L270.6,57.9 L270.4,57.9 L270.3,57.9 L270.2,58.0 L270.1,58.0 L270.0,58.0 L269.9,58.0 L269.8,58.1 L269.7,58.1 L269.6,58.1 L269.5,58.2 L269.3,58.2 L269.2,58.2 L269.1,58.3 L269.0,58.3 L268.9,58.3 L268.8,58.4 L268.7,58.4 L268.6,58.4 L268.5,58.4 L268.3,58.5 L268.2,58.5 L268.1,58.5 L268.0,58.6 L267.9,58.6 L267.8,58.6 L267.7,58.7 L267.6,58.7 L267.5,58.7 L267.3,58.8 L267.2,58.8 L267.1,58.8 L267.0,58.8 L266.9,58.9 L266.8,58.9 L266.7,58.9 L266.6,59.0 L266.5,59.0 L266.4,59.0 L266.2,59.1 L266.1,59.1 L266.0,59.1 L265.9,59.2 L265.8,59.2 L265.7,59.2 L265.6,59.2 L265.5,59.3 L265.4,59.3 L265.2,59.3 L265.1,59.4 L265.0,59.4 L264.9,59.4 L264.8,59.5 L264.7,59.5 L264.6,59.5 L264.5,59.6 L264.4,59.6 L264.2,59.6 L264.1,59.7 L264.0,59.7 L263.9,59.7 L263.8,59.7 L263.7,59.8 L263.6,59.8 L263.5,59.8 L263.4,59.9 L263.2,59.9 L263.1,59.9 L263.0,60.0 L262.9,60.0 L262.8,60.0 L262.7,60.1 L262.6,60.1 L262.5,60.1 L262.4,60.1 L262.3,60.2 L262.1,60.2 L262.0,60.2 L261.9,60.3 L261.8,60.3 L261.7,60.3 L261.6,60.4 L261.5,60.4 L261.4,60.4 L261.3,60.5 L261.1,60.5 L261.0,60.5 L260.9,60.5 L260.8,60.6 L260.7,60.6 L260.6,60.6 L260.5,60.7 L260.4,60.7 L260.3,60.8 L260.2,60.8 L260.1,60.9 L260.0,60.9 L259.9,61.0 L259.8,61.0 L259.7,61.1 L259.6,61.1 L259.5,61.2 L259.4,61.3 L259.3,61.3 L259.2,61.4 L259.1,61.5 L259.0,61.5 L258.9,61.6 L258.8,61.7 L258.7,61.7 L258.6,61.8 L258.6,61.9 L258.5,62.0 L258.4,62.1 L258.3,62.2 L258.2,62.2 L258.2,62.3 L258.1,62.4 L258.0,62.5 L258.0,62.6 L257.9,62.7 L257.8,62.8 L257.8,62.9 L257.7,63.0 L257.6,63.1 L257.6,63.2 L257.5,63.3 L257.5,63.4 L257.4,63.5 L257.4,63.6 L257.3,63.7 L257.3,63.8 L257.3,63.9 L257.2,64.0 L257.2,64.1 L257.2,64.2 L257.1,64.4 L257.1,64.5 L257.1,64.6 L257.0,64.7 L257.0,64.8 L257.0,64.9 L257.0,65.0 L257.0,65.1 L257.0,65.3 L257.0,65.4 L256.9,65.5 L256.9,65.6 L256.9,65.7 L256.9,65.8 L256.9,65.9 L256.9,66.1 L256.9,66.2 L257.0,66.3 L257.0,66.4 L257.0,66.5 L257.0,66.6 L257.0,66.8 L257.0,66.9 L257.0,67.0 L257.0,67.1 L257.0,67.2 L257.0,67.3 L257.0,67.4 L257.0,67.6 L257.0,67.7 L257.0,67.8 L257.0,67.9 L257.0,68.0 L257.0,68.1 L257.0,68.2 L257.0,68.4 L257.0,68.5 L257.0,68.6 L257.0,68.7 L257.0,68.8 L257.1,68.9 L257.1,69.1 L257.1,69.2 L257.1,69.3 L257.1,69.4 L257.1,69.5 L257.1,69.6 L257.1,69.7 L257.1,69.9 L257.1,70.0 L257.1,70.1 L257.1,70.2 L257.1,70.3 L257.1,70.4 L257.1,70.5 L257.1,70.7 L257.1,70.8 L257.1,70.9 L257.1,71.0 L257.1,71.1 L257.1,71.2 L257.1,71.4 L257.1,71.5 L257.2,71.6 L257.2,71.7 L257.2,71.8 L257.2,71.9 L257.2,72.0 L257.2,72.2 L257.2,72.3 L257.2,72.4 L257.2,72.5 L257.2,72.6 L257.2,72.7 L257.2,72.8 L257.2,73.0 L257.2,73.1 L257.2,73.2 L257.2,73.3 L257.2,73.4 L257.2,73.5 L257.2,73.6 L257.2,73.8 L257.2,73.9 L257.2,74.0 L257.2,74.1 L257.3,74.2 L257.3,74.3 L257.3,74.5 L257.3,74.6 L257.3,74.7 L257.3,74.8 L257.3,74.9 L257.3,75.0 L257.3,75.1 L257.3,75.3 L257.3,75.4 L257.3,75.5 L257.3,75.6 L257.3,75.7 L257.3,75.8 L257.3,75.9 L257.3,76.1 L257.3,76.2 L257.3,76.3 L257.3,76.4 L257.3,76.5 L257.3,76.6 L257.3,76.8 L257.3,76.9 L257.4,77.0 L257.4,77.1 L257.4,77.2 L257.4,77.3 L257.4,77.4 L257.4,77.6 L257.4,77.7 L257.4,77.8 L257.4,77.9 L257.4,78.0 L257.4,78.1 L257.4,78.2 L257.4,78.4 L257.4,78.5 L257.4,78.6 L257.4,78.7 L257.4,78.8 L257.4,78.9 L257.4,79.1 L257.4,79.2 L257.4,79.3 L257.4,79.4 L257.4,79.5 L257.5,79.6 L257.5,79.7 L257.5,79.9 L257.5,80.0 L257.5,80.1 L257.5,80.2 L257.5,80.3 L257.5,80.4 L257.5,80.5 L257.5,80.7 L257.5,80.8 L257.5,80.9 L257.5,81.0 L257.5,81.1 L257.5,81.2 L257.5,81.3 L257.5,81.5 L257.5,81.6 L257.5,81.7 L257.5,81.8 L257.5,81.9 L257.5,82.0 L257.5,82.2 L257.6,82.3 L257.6,82.4 L257.6,82.5 L257.6,82.6 L257.6,82.7 L257.6,82.8 L257.6,83.0 L257.6,83.1 L257.6,83.2 L257.6,83.3 L257.6,83.4 L257.6,83.5 L257.6,83.6 L257.6,83.8 L257.6,83.9 L257.6,84.0 L257.6,84.1 L257.6,84.2 L257.6,84.3 L257.6,84.5 L257.6,84.6 L257.6,84.7 L257.6,84.8 L257.6,84.9 L257.7,85.0 L257.7,85.1 L257.7,85.3 L257.7,85.4 L257.7,85.5 L257.7,85.6 L257.7,85.7 L257.7,85.8 L257.7,85.9 L257.7,86.1 L257.7,86.2 L257.7,86.3 L257.7,86.4 L257.7,86.5 L257.7,86.6 L257.7,86.8 L257.7,86.9 L257.7,87.0 L257.7,87.1 L257.7,87.2 L257.7,87.3 L257.7,87.4 L257.7,87.6 L257.8,87.7 L257.8,87.8 L257.8,87.9 L257.8,88.0 L257.8,88.1 L257.8,88.2 L257.8,88.4 L257.8,88.5 L257.8,88.6 L257.8,88.7 L257.8,88.8 L257.8,88.9 L257.8,89.0 L257.8,89.2 L257.8,89.3 L257.8,89.4 L257.8,89.5 L257.8,89.6 L257.8,89.7 L257.8,89.9 L257.8,90.0 L257.8,90.1 L257.8,90.2 L257.9,90.3 L257.9,90.4 L257.9,90.5 L257.9,90.7 L257.9,90.8 L257.9,90.9 L257.9,91.0 L257.9,91.1 L257.9,91.2 L257.9,91.3 L257.9,91.5 L257.9,91.6 L257.9,91.7 L257.9,91.8 L257.9,91.9 L257.9,92.0 L257.9,92.2 L257.9,92.3 L257.9,92.4 L257.9,92.5 L257.9,92.6 L257.9,92.7 L257.9,92.8 L257.9,93.0 L258.0,93.1 L258.0,93.2 L258.0,93.3 L258.0,93.4 L258.0,93.5 L258.0,93.6 L258.0,93.8 L258.0,93.9 L258.0,94.0 L258.0,94.1 L258.0,94.2 L258.0,94.3 L258.0,94.4 L258.0,94.6 L258.0,94.7 L258.0,94.8 L258.0,94.9 L258.0,95.0 L258.0,95.1 L258.0,95.3 L258.0,95.4 L258.0,95.5 L258.0,95.6 L258.1,95.7 L258.1,95.8 L258.1,95.9 L258.1,96.1 L258.1,96.2 L258.1,96.3 L258.1,96.4 L258.1,96.5 L258.1,96.6 L258.1,96.7 L258.1,96.9 L258.1,97.0 L258.1,97.1 L258.1,97.2 L258.1,97.3 L258.1,97.4 L258.1,97.6 L258.1,97.7 L258.1,97.8 L258.1,97.9 L258.1,98.0 L258.1,98.1 L258.1,98.2 L258.2,98.4 L258.2,98.5 L258.2,98.6 L258.2,98.7 L258.2,98.8 L258.2,98.9 L258.2,99.0 L258.2,99.2 L258.2,99.3 L258.2,99.4 L258.2,99.5 L258.2,99.6 L258.2,99.7 L258.2,99.9 L258.2,100.0 L258.2,100.1 L258.2,100.2 L258.2,100.3 L258.2,100.4 L258.2,100.5 L258.2,100.7 L258.2,100.8 L258.2,100.9 L258.2,101.0 L258.3,101.1 L258.3,101.2 L258.3,101.3 L258.3,101.5 L258.3,101.6 L258.3,101.7 L258.3,101.8 L258.3,101.9 L258.3,102.0 L258.3,102.1 L258.3,102.3 L258.3,102.4 L258.3,102.5 L258.3,102.6 L258.3,102.7 L258.3,102.8 L258.3,103.0 L258.3,103.1 L258.3,103.2 L258.3,103.3 L258.3,103.4 L258.3,103.5 L258.3,103.6 L258.4,103.8 L258.4,103.9 L258.4,104.0 L258.4,104.1 L258.4,104.2 L258.4,104.3 L258.4,104.4 L258.4,104.6 L258.4,104.7 L258.4,104.8 L258.4,104.9 L258.4,105.0 L258.4,105.1 L258.4,105.3 L258.4,105.4 L258.4,105.5 L258.4,105.6 L258.4,105.7 L258.4,105.8 L258.4,105.9 L258.4,106.1 L258.4,106.2 L258.4,106.3 L258.5,106.4 L258.5,106.5 L258.5,106.6 L258.5,106.7 L258.5,106.9 L258.5,107.0 L258.5,107.1 L258.5,107.2 L258.5,107.3 L258.5,107.4 L258.5,107.6 L258.5,107.7 L258.5,107.8 L258.5,107.9 L258.5,108.0 L258.5,108.1 L258.5,108.2 L258.5,108.4 L258.5,108.5 L258.5,108.6 L258.6,108.7 L258.6,108.8 L258.6,108.9 L258.6,109.0 L258.6,109.2 L258.6,109.3 L258.7,109.4 L258.7,109.5 L258.7,109.6 L258.8,109.7 L258.8,109.8 L258.8,109.9 L258.9,110.0 L258.9,110.1 L259.0,110.2 L259.0,110.4 L259.1,110.5 L259.1,110.6 L259.2,110.7 L259.2,110.8 L259.3,110.9 L259.3,111.0 L259.4,111.1 L259.5,111.2 L259.5,111.3 L259.6,111.4 L259.6,111.5 L259.7,111.6 L259.7,111.7 L259.8,111.8 L259.8,111.9 L259.9,112.0 L260.0,112.1 L260.0,112.2 L260.1,112.3 L260.1,112.4 L260.2,112.5 L260.2,112.6 L260.3,112.7 L260.3,112.8 L260.4,112.9 L260.5,113.0 L260.5,113.1 L260.6,113.2 L260.6,113.3 L260.7,113.4 L260.7,113.5 L260.8,113.6 L260.8,113.7 L260.9,113.8 L261.0,113.9 L261.0,114.0 L261.1,114.1 L261.1,114.2 L261.2,114.3 L261.2,114.4 L261.3,114.5 L261.3,114.6 L261.4,114.7 L261.5,114.8 L261.5,114.9 L261.6,115.0 L261.6,115.1 L261.7,115.2 L261.7,115.3 L261.8,115.4 L261.8,115.5 L261.9,115.6 L261.9,115.7 L262.0,115.8 L262.1,115.9 L262.1,116.0 L262.2,116.1 L262.2,116.2 L262.3,116.3 L262.3,116.4 L262.4,116.5 L262.4,116.6 L262.5,116.7 L262.6,116.8 L262.6,116.9 L262.7,117.0 L262.7,117.1 L262.8,117.2 L262.8,117.3 L262.9,117.4 L262.9,117.5 L263.0,117.6 L263.1,117.7 L263.1,117.8 L263.2,117.9 L263.2,118.0 L263.3,118.1 L263.3,118.2 L263.4,118.3 L263.4,118.4 L263.5,118.5 L263.6,118.6 L263.6,118.7 L263.7,118.8 L263.7,118.9 L263.8,119.0 L263.8,119.1 L263.9,119.2 L263.9,119.3 L264.0,119.4 L264.1,119.5 L264.1,119.6 L264.2,119.7 L264.2,119.8 L264.3,119.9 L264.3,120.0 L264.4,120.1 L264.4,120.2 L264.5,120.3 L264.6,120.4 L264.6,120.5 L264.7,120.6 L264.7,120.7 L264.8,120.8 L264.8,120.9 L264.9,121.0 L264.9,121.1 L265.0,121.2 L265.0,121.3 L265.1,121.4 L265.2,121.5 L265.2,121.6 L265.3,121.7 L265.3,121.8 L265.4,121.9 L265.4,122.0 L265.5,122.1 L265.5,122.2 L265.6,122.3 L265.7,122.5 L265.7,122.6 L265.8,122.7 L265.8,122.8 L265.9,122.9 L265.9,123.0 L266.0,123.1 L266.0,123.2 L266.1,123.3 L266.2,123.4 L266.2,123.5 L266.3,123.6 L266.3,123.7 L266.4,123.8 L266.4,123.9 L266.5,124.0 L266.5,124.1 L266.6,124.2 L266.7,124.3 L266.7,124.4 L266.8,124.5 L266.8,124.6 L266.9,124.7 L266.9,124.8 L267.0,124.9 L267.0,125.0 L267.1,125.1 L267.2,125.2 L267.2,125.3 L267.3,125.4 L267.3,125.5 L267.4,125.6 L267.4,125.7 L267.5,125.8 L267.5,125.9 L267.6,126.0 L267.7,126.1 L267.7,126.2 L267.8,126.3 L267.8,126.4 L267.9,126.5 L267.9,126.6 L268.0,126.7 L268.0,126.8 L268.1,126.9 L268.1,127.0 L268.2,127.1 L268.3,127.2 L268.3,127.3 L268.4,127.4 L268.4,127.5 L268.5,127.6 L268.5,127.7 L268.6,127.8 L268.6,127.9 L268.7,128.0 L268.8,128.1 L268.8,128.2 L268.9,128.3 L268.9,128.4 L269.0,128.5 L269.0,128.6 L269.1,128.7 L269.1,128.8 L269.2,128.9 L269.3,129.0 L269.3,129.1 L269.4,129.2 L269.4,129.3 L269.5,129.4 L269.5,129.5 L269.6,129.6 L269.6,129.7 L269.7,129.8 L269.8,129.9 L269.8,130.0 L269.9,130.1 L269.9,130.2 L270.0,130.3 L270.0,130.4 L270.1,130.5 L270.1,130.6 L270.2,130.7 L270.3,130.8 L270.3,130.9 L270.4,131.0 L270.4,131.1 L270.5,131.2 L270.5,131.3 L270.6,131.4 L270.6,131.5 L270.7,131.6 L270.8,131.7 L270.8,131.8 L270.9,131.9 L270.9,132.0 L271.0,132.1 L271.0,132.2 L271.1,132.3 L271.1,132.4 L271.2,132.5 L271.2,132.6 L271.3,132.7 L271.4,132.8 L271.4,132.9 L271.5,133.0 L271.5,133.1 L271.6,133.2 L271.6,133.3 L271.7,133.4 L271.7,133.5 L271.8,133.6 L271.9,133.7 L271.9,133.8 L272.0,133.9 L272.0,134.0 L272.1,134.1 L272.1,134.2 L272.2,134.3 L272.2,134.4 L272.3,134.5 L272.4,134.6 L272.4,134.7 L272.5,134.8 L272.5,134.9 L272.6,135.1 L272.6,135.2 L272.7,135.3 L272.7,135.4 L272.8,135.5 L272.9,135.6 L272.9,135.7 L273.0,135.8 L273.0,135.9 L273.1,136.0 L273.1,136.1 L273.2,136.2 L273.2,136.3 L273.3,136.4 L273.4,136.5 L273.4,136.6 L273.5,136.7 L273.5,136.8 L273.6,136.9 L273.6,137.0 L273.7,137.1 L273.7,137.2 L273.8,137.3 L273.9,137.4 L273.9,137.5 L274.0,137.6 L274.0,137.7 L274.1,137.8 L274.1,137.9 L274.2,138.0 L274.2,138.1 L274.3,138.2 L274.3,138.3 L274.4,138.4 L274.5,138.5 L274.5,138.6 L274.6,138.7 L274.6,138.8 L274.7,138.9 L274.7,139.0 L274.8,139.1 L274.8,139.2 L274.9,139.3 L275.0,139.4 L275.0,139.5 L275.1,139.6 L275.1,139.7 L275.2,139.8 L275.2,139.9 L275.3,140.0 L275.3,140.1 L275.4,140.2 L275.5,140.3 L275.5,140.4 L275.6,140.5 L275.6,140.6 L275.7,140.7 L275.7,140.8 L275.8,140.9 L275.8,141.0 L275.9,141.1 L276.0,141.2 L276.0,141.3 L276.1,141.4 L276.1,141.5 L276.2,141.6 L276.2,141.7 L276.3,141.8 L276.3,141.9 L276.4,142.0 L276.5,142.1 L276.5,142.2 L276.6,142.3 L276.6,142.4 L276.7,142.5 L276.7,142.6 L276.8,142.7 L276.8,142.8 L276.9,142.9 L276.9,143.0 L277.0,143.1 L277.1,143.2 L277.1,143.3 L277.2,143.4 L277.2,143.5 L277.3,143.6 L277.3,143.7 L277.4,143.8 L277.4,143.9 L277.5,144.0 L277.6,144.1 L277.6,144.2 L277.7,144.3 L277.7,144.4 L277.8,144.5 L277.8,144.6 L277.9,144.7 L278.0,144.8 L278.0,144.9 L278.1,145.0 L278.1,145.1 L278.2,145.2 L278.3,145.3 L278.3,145.4 L278.4,145.5 L278.4,145.6 L278.5,145.7 L278.6,145.8 L278.6,145.9 L278.7,146.0 L278.8,146.1 L278.9,146.2 L278.9,146.2 L279.0,146.3 L279.1,146.4 L279.1,146.5 L279.2,146.6 L279.3,146.7 L279.4,146.8 L279.4,146.9 L279.5,147.0 L279.6,147.0 L279.7,147.1 L279.8,147.2 L279.8,147.3 L279.9,147.4 L280.0,147.5 L280.1,147.5 L280.2,147.6 L280.2,147.7 L280.3,147.8 L280.4,147.8 L280.5,147.9 L280.6,148.0 L280.7,148.1 L280.8,148.2 L280.8,148.2 L280.9,148.3 L281.0,148.4 L281.1,148.4 L281.2,148.5 L281.3,148.6 L281.4,148.7 L281.5,148.7 L281.6,148.8 L281.7,148.9 L281.8,148.9 L281.9,149.0 L281.9,149.0 L282.0,149.1 L282.1,149.2 L282.2,149.2 L282.3,149.3 L282.4,149.4 L282.5,149.4 L282.6,149.5 L282.7,149.5 L282.8,149.6 L282.9,149.6 L283.0,149.7 L283.1,149.8 L283.2,149.8 L283.3,149.9 L283.4,149.9 L283.5,150.0 L283.7,150.0 L283.8,150.1 L283.9,150.1 L284.0,150.2 L284.1,150.2 L284.2,150.2 L284.3,150.3 L284.4,150.3 L284.5,150.4 L284.6,150.4 L284.7,150.5 L284.8,150.5 L284.9,150.5 L285.0,150.6 L285.1,150.6 L285.3,150.6 L285.4,150.7 L285.5,150.7 L285.6,150.7 L285.7,150.8 L285.8,150.8 L285.9,150.8 L286.0,150.9 L286.1,150.9 L286.3,150.9 L286.4,150.9 L286.5,151.0 L286.6,151.0 L286.7,151.0 L286.8,151.0 L286.9,151.0 L287.0,151.1 L287.2,151.1 L287.3,151.1 L287.4,151.1 L287.5,151.1 L287.6,151.2 L287.7,151.2 L287.8,151.2 L288.0,151.2 L288.1,151.2 L288.2,151.2 L288.3,151.2 L288.4,151.2 L288.5,151.2 L288.6,151.2 L288.8,151.2 L288.9,151.2 L289.0,151.2 L289.1,151.2 L289.2,151.2 L289.3,151.2 L289.4,151.2 L289.6,151.2 L289.7,151.2 L289.8,151.2 L289.9,151.2 L290.0,151.2 L290.1,151.2 L290.3,151.2 L290.4,151.2 L290.5,151.2 L290.6,151.2 L290.7,151.2 L290.8,151.1 L290.9,151.1 L291.1,151.1 L291.2,151.1 L291.3,151.1 L291.4,151.1 L291.5,151.0 L291.6,151.0 L291.7,151.0 L291.8,151.0 L292.0,151.0 L292.1,150.9 L292.2,150.9 L292.3,150.9 L292.4,150.9 L292.5,150.8 L292.6,150.8 L292.7,150.8 L292.8,150.7 L293.0,150.7 L293.1,150.7 L293.2,150.6 L293.3,150.6 L293.4,150.6 L293.5,150.5 L293.6,150.5 L293.7,150.4 L293.8,150.4 L293.9,150.4 L294.0,150.3 L294.2,150.3 L294.3,150.2 L294.4,150.2 L294.5,150.2 L294.6,150.1 L294.7,150.1 L294.8,150.0 L294.9,150.0 L295.0,150.0 L295.1,149.9 L295.2,149.9 L295.3,149.8 L295.4,149.8 L295.6,149.8 L295.7,149.7 L295.8,149.7 L295.9,149.6 L296.0,149.6 L296.1,149.5 L296.2,149.5 L296.3,149.5 L296.4,149.4 L296.5,149.4 L296.6,149.3 L296.7,149.3 L296.8,149.3 L296.9,149.2 L297.1,149.2 L297.2,149.1 L297.3,149.1 L297.4,149.1 L297.5,149.0 L297.6,149.0 L297.7,148.9 L297.8,148.9 L297.9,148.9 L298.0,148.8 L298.1,148.8 L298.2,148.7 L298.3,148.7 L298.5,148.6 L298.6,148.6 L298.7,148.6 L298.8,148.5 L298.9,148.5 L299.0,148.4 L299.1,148.4 L299.2,148.4 L299.3,148.3 L299.4,148.3 L299.5,148.2 L299.6,148.2 L299.7,148.2 L299.9,148.1 L300.0,148.1 L300.1,148.0 L300.2,148.0 L300.3,148.0 L300.4,147.9 L300.5,147.9 L300.6,147.8 L300.7,147.8 L300.8,147.8 L300.9,147.7 L301.0,147.7 L301.1,147.6 L301.2,147.6 L301.4,147.5 L301.5,147.5 L301.6,147.5 L301.7,147.4 L301.8,147.4 L301.9,147.3 L302.0,147.3 L302.1,147.3 L302.2,147.2 L302.3,147.2 L302.4,147.1 L302.5,147.1 L302.6,147.1 L302.8,147.0 L302.9,147.0 L303.0,146.9 L303.1,146.9 L303.2,146.9 L303.3,146.8 L303.4,146.8 L303.5,146.7 L303.6,146.7 L303.7,146.6 L303.8,146.6 L303.9,146.6 L304.0,146.5 L304.2,146.5 L304.3,146.4 L304.4,146.4 L304.5,146.4 L304.6,146.3 L304.7,146.3 L304.8,146.2 L304.9,146.2 L305.0,146.2 L305.1,146.1 L305.2,146.1 L305.3,146.0 L305.4,146.0 L305.5,146.0 L305.7,145.9 L305.8,145.9 L305.9,145.8 L306.0,145.8 L306.1,145.7 L306.2,145.7 L306.3,145.7 L306.4,145.6 L306.5,145.6 L306.6,145.5 L306.7,145.5 L306.8,145.5 L306.9,145.4 L307.1,145.4 L307.2,145.3 L307.3,145.3 L307.4,145.3 L307.5,145.2 L307.6,145.2 L307.7,145.1 L307.8,145.1 L307.9,145.1 L308.0,145.0 L308.1,145.0 L308.2,144.9 L308.3,144.9 L308.5,144.9 L308.6,144.8 L308.7,144.8 L308.8,144.7 L308.9,144.7 L309.0,144.6 L309.1,144.6 L309.2,144.6 L309.3,144.5 L309.4,144.5 L309.5,144.4 L309.6,144.4 L309.7,144.4 L309.8,144.3 L310.0,144.3 L310.1,144.2 L310.2,144.2 L310.3,144.2 L310.4,144.1 L310.5,144.1 L310.6,144.0 L310.7,144.0 L310.8,144.0 L310.9,143.9 L311.0,143.9 L311.1,143.8 L311.2,143.8 L311.4,143.7 L311.5,143.7 L311.6,143.7 L311.7,143.6 L311.8,143.6 L311.9,143.5 L312.0,143.5 L312.1,143.5 L312.2,143.4 L312.3,143.4 L312.4,143.3 L312.5,143.3 L312.6,143.3 L312.8,143.2 L312.9,143.2 L313.0,143.1 L313.1,143.1 L313.2,143.1 L313.3,143.0 L313.4,143.0 L313.5,142.9 L313.6,142.9 L313.7,142.8 L313.8,142.8 L313.9,142.8 L314.0,142.7 L314.1,142.7 L314.3,142.6 L314.4,142.6 L314.5,142.6 L314.6,142.5 L314.7,142.5 L314.8,142.4 L314.9,142.4 L315.0,142.4 L315.1,142.3 L315.2,142.3 L315.3,142.2 L315.4,142.2 L315.5,142.2 L315.7,142.1 L315.8,142.1 L315.9,142.0 L316.0,142.0 L316.1,142.0 L316.2,141.9 L316.3,141.9 L316.4,141.8 L316.5,141.8 L316.6,141.7 L316.7,141.7 L316.8,141.7 L316.9,141.6 L317.1,141.6 L317.2,141.5 L317.3,141.5 L317.4,141.5 L317.5,141.4 L317.6,141.4 L317.7,141.4 L317.8,141.3 L317.9,141.3 L318.0,141.3 L318.1,141.2 L318.3,141.2 L318.4,141.2 L318.5,141.1 L318.6,141.1 L318.7,141.1 L318.8,141.0 L318.9,141.0 L319.0,141.0 L319.1,141.0 L319.3,140.9 L319.4,140.9 L319.5,140.9 L319.6,140.8 L319.7,140.8 L319.8,140.8 L319.9,140.8 L320.0,140.7 L320.2,140.7 L320.3,140.7 L320.4,140.7 L320.5,140.7 L320.6,140.6 L320.7,140.6 L320.8,140.6 L320.9,140.6 L321.1,140.6 L321.2,140.5 L321.3,140.5 L321.4,140.5 L321.5,140.5 L321.6,140.5 L321.7,140.5 L321.9,140.5 L322.0,140.4 L322.1,140.4 L322.2,140.4 L322.3,140.4 L322.4,140.4 L322.5,140.4 L322.7,140.4 L322.8,140.4 L322.9,140.4 L323.0,140.4 L323.1,140.3 L323.2,140.3 L323.3,140.3 L323.5,140.3 L323.6,140.3 L323.7,140.3 L323.8,140.3 L323.9,140.3 L324.0,140.3 L324.2,140.3 L324.3,140.3 L324.4,140.3 L324.5,140.3 L324.6,140.3 L324.7,140.3 L324.8,140.3 L325.0,140.3 L325.1,140.3 L325.2,140.3 L325.3,140.3 L325.4,140.3 L325.5,140.3 L325.6,140.4 L325.8,140.4 L325.9,140.4 L326.0,140.4 L326.1,140.4 L326.2,140.4 L326.3,140.4 L326.4,140.4 L326.6,140.4 L326.7,140.4 L326.8,140.4 L326.9,140.4 L327.0,140.4 L327.1,140.5 L327.3,140.5 L327.4,140.5 L327.5,140.5 L327.6,140.5 L327.7,140.5 L327.8,140.5 L327.9,140.5 L328.1,140.5 L328.2,140.5 L328.3,140.5 L328.4,140.5 L328.5,140.5 L328.6,140.5 L328.7,140.6 L328.9,140.6 L329.0,140.6 L329.1,140.6 L329.2,140.6 L329.3,140.6 L329.4,140.6 L329.5,140.6 L329.7,140.6 L329.8,140.6 L329.9,140.6 L330.0,140.6 L330.1,140.6 L330.2,140.7 L330.3,140.7 L330.5,140.7 L330.6,140.7 L330.7,140.7 L330.8,140.7 L330.9,140.7 L331.0,140.7 L331.2,140.7 L331.3,140.7 L331.4,140.7 L331.5,140.7 L331.6,140.7 L331.7,140.7 L331.8,140.8 L332.0,140.8 L332.1,140.8 L332.2,140.8 L332.3,140.8 L332.4,140.8 L332.5,140.8 L332.6,140.8 L332.8,140.8 L332.9,140.8 L333.0,140.8 L333.1,140.8 L333.2,140.8 L333.3,140.9 L333.4,140.9 L333.6,140.9 L333.7,140.9 L333.8,140.9 L333.9,140.9 L334.0,140.9 L334.1,140.9 L334.3,140.9 L334.4,140.9 L334.5,140.9 L334.6,140.9 L334.7,140.9 L334.8,140.9 L334.9,141.0 L335.1,141.0 L335.2,141.0 L335.3,141.0 L335.4,141.0 L335.5,141.0 L335.6,141.0 L335.7,141.0 L335.9,141.0 L336.0,141.0 L336.1,141.0 L336.2,141.0 L336.3,141.0 L336.4,141.1 L336.5,141.1 L336.7,141.1 L336.8,141.1 L336.9,141.1 L337.0,141.1 L337.1,141.1 L337.2,141.1 L337.4,141.1 L337.5,141.1 L337.6,141.1 L337.7,141.1 L337.8,141.1 L337.9,141.1 L338.0,141.2 L338.2,141.2 L338.3,141.2 L338.4,141.2 L338.5,141.2 L338.6,141.2 L338.7,141.2 L338.8,141.2 L339.0,141.2 L339.1,141.2 L339.2,141.2 L339.3,141.2 L339.4,141.2 L339.5,141.3 L339.6,141.3 L339.8,141.3 L339.9,141.3 L340.0,141.3 L340.1,141.3 L340.2,141.3 L340.3,141.3 L340.4,141.3 L340.6,141.3 L340.7,141.3 L340.8,141.3 L340.9,141.3 L341.0,141.3 L341.1,141.4 L341.3,141.4 L341.4,141.4 L341.5,141.4 L341.6,141.4 L341.7,141.4 L341.8,141.4 L341.9,141.4 L342.1,141.4 L342.2,141.4 L342.3,141.4 L342.4,141.4 L342.5,141.4 L342.6,141.5 L342.7,141.5 L342.9,141.5 L343.0,141.5 L343.1,141.5 L343.2,141.5 L343.3,141.5 L343.4,141.5 L343.5,141.5 L343.7,141.5 L343.8,141.5 L343.9,141.5 L344.0,141.5 L344.1,141.5 L344.2,141.6 L344.4,141.6 L344.5,141.6 L344.6,141.6 L344.7,141.6 L344.8,141.6 L344.9,141.6 L345.0,141.6 L345.2,141.6 L345.3,141.6 L345.4,141.6 L345.5,141.6 L345.6,141.6 L345.7,141.7 L345.8,141.7 L346.0,141.7 L346.1,141.7 L346.2,141.7 L346.3,141.7 L346.4,141.7 L346.5,141.7 L346.6,141.7 L346.8,141.7 L346.9,141.7 L347.0,141.7 L347.1,141.7 L347.2,141.7 L347.3,141.8 L347.4,141.8 L347.6,141.8 L347.7,141.8 L347.8,141.8 L347.9,141.8 L348.0,141.8 L348.1,141.8 L348.3,141.8 L348.4,141.8 L348.5,141.8 L348.6,141.8 L348.7,141.8 L348.8,141.8 L348.9,141.9 L349.1,141.9 L349.2,141.9 L349.3,141.9 L349.4,141.9 L349.5,141.9 L349.6,141.9 L349.7,141.9 L349.9,141.9 L350.0,141.9 L350.1,141.9 L350.2,141.9 L350.3,141.9 L350.4,142.0 L350.5,142.0 L350.7,142.0 L350.8,142.0 L350.9,142.0 L351.0,142.0 L351.1,142.0 L351.2,142.0 L351.4,142.0 L351.5,142.0 L351.6,142.0 L351.7,142.0 L351.8,142.0 L351.9,142.0 L352.0,142.1 L352.2,142.1 L352.3,142.1 L352.4,142.1 L352.5,142.1 L352.6,142.1 L352.7,142.1 L352.8,142.1 L353.0,142.1 L353.1,142.1 L353.2,142.1 L353.3,142.1 L353.4,142.1 L353.5,142.2 L353.6,142.2 L353.8,142.2 L353.9,142.2 L354.0,142.2 L354.1,142.2 L354.2,142.2 L354.3,142.2 L354.4,142.2 L354.6,142.2 L354.7,142.2 L354.8,142.2 L354.9,142.2 L355.0,142.2 L355.1,142.3 L355.3,142.3 L355.4,142.3 L355.5,142.3 L355.6,142.3 L355.7,142.3 L355.8,142.3 L355.9,142.3 L356.1,142.3 L356.2,142.3 L356.3,142.3 L356.4,142.3 L356.5,142.3 L356.6,142.4 L356.7,142.4 L356.9,142.4 L357.0,142.4 L357.1,142.4 L357.2,142.4 L357.3,142.4 L357.4,142.4 L357.5,142.4 L357.7,142.4 L357.8,142.4 L357.9,142.4 L358.0,142.4 L358.1,142.4 L358.2,142.5 L358.4,142.5 L358.5,142.5 L358.6,142.5 L358.7,142.5 L358.8,142.5 L358.9,142.5 L359.0,142.5 L359.2,142.5 L359.3,142.5 L359.4,142.5 L359.5,142.5 L359.6,142.5 L359.7,142.6 L359.8,142.6 L360.0,142.6 L360.1,142.6 L360.2,142.6 L360.3,142.6 L360.4,142.6 L360.5,142.6 L360.6,142.6 L360.8,142.6 L360.9,142.6 L361.0,142.6 L361.1,142.6 L361.2,142.6 L361.3,142.7 L361.5,142.7 L361.6,142.7 L361.7,142.7 L361.8,142.7 L361.9,142.7 L362.0,142.7 L362.1,142.7 L362.3,142.7 L362.4,142.7 L362.5,142.7 L362.6,142.7 L362.7,142.7 L362.8,142.8 L362.9,142.8 L363.1,142.8 L363.2,142.8 L363.3,142.8 L363.4,142.8 L363.5,142.8 L363.6,142.8 L363.7,142.8 L363.9,142.8 L364.0,142.8 L364.1,142.8 L364.2,142.8 L364.3,142.8 L364.4,142.9 L364.5,142.9 L364.7,142.9 L364.8,142.9 L364.9,142.9 L365.0,142.9 L365.1,142.9 L365.2,142.9 L365.4,142.9 L365.5,142.9 L365.6,142.9 L365.7,142.9 L365.8,142.9 L365.9,143.0 L366.0,143.0 L366.2,143.0 L366.3,143.0 L366.4,143.0 L366.5,143.0 L366.6,143.0 L366.7,143.0 L366.8,143.0 L367.0,143.0 L367.1,143.0 L367.2,143.0 L367.3,143.0 L367.4,143.0 L367.5,143.1 L367.6,143.1 L367.8,143.1 L367.9,143.1 L368.0,143.1 L368.1,143.1 L368.2,143.1 L368.3,143.1 L368.5,143.1 L368.6,143.1 L368.7,143.1 L368.8,143.1 L368.9,143.1 L369.0,143.2 L369.1,143.2 L369.3,143.2 L369.4,143.2 L369.5,143.2 L369.6,143.2 L369.7,143.2 L369.8,143.2 L369.9,143.2 L370.1,143.2 L370.2,143.2 L370.3,143.2 L370.4,143.2 L370.5,143.2 L370.6,143.3 L370.7,143.3 L370.9,143.3 L371.0,143.3 L371.1,143.3 L371.2,143.3 L371.3,143.3 L371.4,143.3 L371.5,143.3 L371.7,143.3 L371.8,143.3 L371.9,143.3 L372.0,143.3 L372.1,143.4 L372.2,143.4 L372.4,143.4 L372.5,143.4 L372.6,143.4 L372.7,143.4 L372.8,143.4 L372.9,143.4 L373.0,143.4 L373.2,143.4 L373.3,143.4 L373.4,143.4 L373.5,143.4 L373.6,143.4 L373.7,143.5 L373.8,143.5 L374.0,143.5 L374.1,143.5 L374.2,143.5 L374.3,143.5 L374.4,143.5 L374.5,143.5 L374.6,143.5 L374.8,143.5 L374.9,143.5 L375.0,143.5 L375.1,143.5 L375.2,143.6 L375.3,143.6 L375.5,143.6 L375.6,143.6 L375.7,143.6 L375.8,143.6 L375.9,143.6 L376.0,143.6 L376.1,143.6 L376.3,143.6 L376.4,143.6 L376.5,143.6 L376.6,143.6 L376.7,143.6 L376.8,143.7 L376.9,143.7 L377.1,143.7 L377.2,143.7 L377.3,143.7 L377.4,143.7 L377.5,143.7 L377.6,143.7 L377.7,143.7 L377.9,143.7 L378.0,143.7 L378.1,143.7 L378.2,143.7 L378.3,143.8 L378.4,143.8 L378.5,143.8 L378.7,143.8 L378.8,143.8 L378.9,143.8 L379.0,143.8 L379.1,143.8 L379.2,143.8 L379.4,143.8 L379.5,143.8 L379.6,143.8 L379.7,143.8 L379.8,143.8 L379.9,143.9 L380.0,143.9 L380.2,143.9 L380.3,143.9 L380.4,143.9 L380.5,143.9 L380.6,143.9 L380.7,143.9 L380.8,143.9 L381.0,143.9 L381.1,143.9 L381.2,143.9 L381.3,143.9 L381.4,143.9 L381.5,143.9 L381.6,143.9 L381.8,143.9 L381.9,143.9 L382.0,143.9 L382.1,143.8 L382.2,143.8 L382.3,143.8 L382.4,143.8 L382.6,143.8 L382.7,143.7 L382.8,143.7 L382.9,143.7 L383.0,143.6 L383.1,143.6 L383.2,143.6 L383.3,143.5 L383.4,143.5 L383.5,143.5 L383.7,143.4 L383.8,143.4 L383.9,143.3 L384.0,143.3 L384.1,143.2 L384.2,143.2 L384.3,143.1 L384.4,143.1 L384.5,143.0 L384.6,142.9 L384.7,142.9 L384.8,142.8 L384.9,142.7 L385.0,142.7 L385.0,142.6 L385.1,142.5 L385.2,142.5 L385.3,142.4 L385.4,142.3 L385.5,142.2 L385.6,142.2 L385.7,142.1 L385.7,142.0 L385.8,141.9 L385.9,141.8 L386.0,141.8 L386.1,141.7 L386.1,141.6 L386.2,141.5 L386.3,141.4 L386.3,141.3 L386.4,141.2 L386.5,141.1 L386.5,141.0 L386.6,140.9 L386.7,140.8 L386.7,140.7 L386.8,140.6 L386.8,140.5 L386.9,140.4 L387.0,140.3 L387.0,140.2 L387.1,140.1 L387.1,140.0 L387.2,139.9 L387.2,139.8 L387.3,139.7 L387.3,139.6 L387.4,139.5 L387.4,139.4 L387.5,139.3 L387.5,139.2 L387.6,139.1 L387.6,139.0 L387.7,138.9 L387.7,138.8 L387.8,138.7 L387.8,138.6 L387.9,138.5 L387.9,138.4 L388.0,138.3 L388.0,138.2 L388.1,138.1 L388.1,138.0 L388.2,137.9 L388.2,137.8 L388.3,137.7 L388.3,137.6 L388.4,137.5 L388.4,137.4 L388.5,137.3 L388.5,137.2 L388.6,137.1 L388.7,137.0 L388.7,136.8 L388.8,136.7 L388.8,136.6 L388.9,136.5 L388.9,136.4 L389.0,136.3 L389.0,136.2 L389.1,136.1 L389.1,136.0 L389.2,135.9 L389.2,135.8 L389.3,135.7 L389.3,135.6 L389.4,135.5 L389.4,135.4 L389.5,135.3 L389.5,135.2 L389.6,135.1 L389.6,135.0 L389.7,134.9 L389.7,134.8 L389.8,134.7 L389.8,134.6 L389.9,134.5 L389.9,134.4 L390.0,134.3 L390.0,134.2 L390.1,134.1 L390.1,134.0 L390.2,133.9 L390.2,133.8 L390.3,133.7 L390.3,133.6 L390.4,133.5 L390.4,133.3 L390.5,133.2 L390.6,133.1 L390.6,133.0 L390.7,132.9 L390.7,132.8 L390.8,132.7 L390.8,132.6 L390.9,132.5 L390.9,132.4 L391.0,132.3 L391.0,132.2 L391.1,132.1 L391.1,132.0 L391.2,131.9 L391.2,131.8 L391.3,131.7 L391.3,131.6 L391.4,131.5 L391.4,131.4 L391.5,131.3 L391.5,131.2 L391.6,131.1 L391.6,131.0 L391.7,130.9 L391.7,130.8 L391.8,130.7 L391.8,130.6 L391.9,130.5 L391.9,130.4 L392.0,130.3 L392.0,130.2 L392.1,130.1 L392.1,130.0 L392.2,129.9 L392.2,129.7 L392.3,129.6 L392.3,129.5 L392.4,129.4 L392.5,129.3 L392.5,129.2 L392.6,129.1 L392.6,129.0 L392.7,128.9 L392.7,128.8 L392.8,128.7 L392.8,128.6 L392.9,128.5 L392.9,128.4 L393.0,128.3 L393.0,128.2 L393.1,128.1 L393.1,128.0 L393.2,127.9 L393.2,127.8 L393.3,127.7 L393.3,127.6 L393.4,127.5 L393.4,127.4 L393.5,127.3 L393.5,127.2 L393.6,127.1 L393.6,127.0 L393.7,126.9 L393.7,126.8 L393.8,126.7 L393.8,126.6 L393.9,126.5 L393.9,126.4 L394.0,126.2 L394.0,126.1 L394.1,126.0 L394.1,125.9 L394.2,125.8 L394.2,125.7 L394.3,125.6 L394.4,125.5 L394.4,125.4 L394.5,125.3 L394.5,125.2 L394.6,125.1 L394.6,125.0 L394.7,124.9 L394.7,124.8 L394.8,124.7 L394.8,124.6 L394.9,124.5 L394.9,124.4 L395.0,124.3 L395.0,124.2 L395.1,124.1 L395.1,124.0 L395.2,123.9 L395.2,123.8 L395.3,123.7 L395.3,123.6 L395.4,123.5 L395.4,123.4 L395.5,123.3 L395.5,123.2 L395.6,123.1 L395.6,123.0 L395.7,122.9 L395.7,122.8 L395.8,122.6 L395.8,122.5 L395.9,122.4 L395.9,122.3 L396.0,122.2 L396.0,122.1 L396.1,122.0 L396.1,121.9 L396.2,121.8 L396.3,121.7 L396.3,121.6 L396.4,121.5 L396.4,121.4 L396.5,121.3 L396.5,121.2 L396.6,121.1 L396.6,121.0 L396.7,120.9 L396.7,120.8 L396.8,120.7 L396.8,120.6 L396.9,120.5 L396.9,120.4 L397.0,120.3 L397.0,120.2 L397.1,120.1 L397.1,120.0 L397.2,119.9 L397.2,119.8 L397.3,119.7 L397.3,119.6 L397.4,119.5 L397.4,119.4 L397.5,119.3 L397.5,119.1 L397.6,119.0 L397.6,118.9 L397.7,118.8 L397.7,118.7 L397.8,118.6 L397.8,118.5 L397.9,118.4 L397.9,118.3 L398.0,118.2 L398.0,118.1 L398.1,118.0 L398.2,117.9 L398.2,117.8 L398.3,117.7 L398.3,117.6 L398.4,117.5 L398.4,117.4 L398.5,117.3 L398.5,117.2 L398.6,117.1 L398.6,117.0 L398.7,116.9 L398.7,116.8 L398.8,116.7 L398.8,116.6 L398.9,116.5 L398.9,116.4 L399.0,116.3 L399.0,116.2 L399.1,116.1 L399.1,116.0 L399.2,115.9 L399.2,115.8 L399.3,115.7 L399.3,115.5 L399.4,115.4 L399.4,115.3 L399.5,115.2 L399.5,115.1 L399.6,115.0 L399.6,114.9 L399.7,114.8 L399.7,114.7 L399.8,114.6 L399.8,114.5 L399.9,114.4 L399.9,114.3 L400.0,114.2 L400.1,114.1 L400.1,114.0 L400.2,113.9 L400.2,113.8 L400.3,113.7 L400.3,113.6 L400.4,113.5 L400.4,113.4 L400.5,113.3 L400.5,113.2 L400.6,113.1 L400.6,113.0 L400.7,112.9 L400.7,112.8 L400.8,112.7 L400.8,112.6 L400.9,112.5 L400.9,112.4 L401.0,112.3 L401.0,112.2 L401.1,112.0 L401.1,111.9 L401.2,111.8 L401.2,111.7 L401.3,111.6 L401.3,111.5 L401.4,111.4 L401.4,111.3 L401.5,111.2 L401.5,111.1 L401.6,111.0 L401.6,110.9 L401.7,110.8 L401.7,110.7 L401.8,110.6 L401.8,110.5 L401.9,110.4 L402.0,110.3 L402.0,110.2 L402.1,110.1 L402.1,110.0 L402.2,109.9 L402.2,109.8 L402.3,109.7 L402.3,109.6 L402.4,109.5 L402.4,109.4 L402.5,109.3 L402.5,109.2 L402.6,109.1 L402.6,109.0 L402.7,108.9 L402.7,108.8 L402.8,108.7 L402.8,108.6 L402.9,108.4 L402.9,108.3 L403.0,108.2 L403.0,108.1 L403.1,108.0 L403.1,107.9 L403.2,107.8 L403.2,107.7 L403.3,107.6 L403.3,107.5 L403.4,107.4 L403.4,107.3 L403.5,107.2 L403.5,107.1 L403.6,107.0 L403.6,106.9 L403.7,106.8 L403.7,106.7 L403.8,106.6 L403.9,106.5 L403.9,106.4 L404.0,106.3 L404.0,106.2 L404.1,106.1 L404.1,106.0 L404.2,105.9 L404.2,105.8 L404.3,105.7 L404.3,105.6 L404.4,105.5 L404.4,105.4 L404.5,105.3 L404.5,105.2 L404.6,105.1 L404.6,104.9 L404.7,104.8 L404.7,104.7 L404.8,104.6 L404.8,104.5 L404.9,104.4 L404.9,104.3 L405.0,104.2 L405.0,104.1 L405.1,104.0 L405.1,103.9 L405.2,103.8 L405.2,103.7 L405.3,103.6 L405.3,103.5 L405.4,103.4 L405.4,103.3 L405.5,103.2 L405.5,103.1 L405.6,103.0 L405.6,102.9 L405.7,102.8 L405.8,102.7 L405.8,102.6 L405.9,102.5 L405.9,102.4 L406.0,102.3 L406.0,102.2 L406.1,102.1 L406.1,102.0 L406.2,101.9 L406.2,101.8 L406.3,101.7 Z" fill="none" stroke="#5f6673" stroke-width="3" stroke-linejoin="round"/><circle cx="406.3" cy="101.6" r="3.4" fill="#1d2330" stroke="#ffffff" stroke-width="1"/><text x="332.5" y="190.8" font-family="DejaVu Sans" font-size="9.5" fill="#5f6673" text-anchor="middle" font-weight="normal">3.7 km · road 12.0–15.8 m</text><rect x="450.0" y="0" width="215.0" height="200.82337120641645" rx="3" fill="#f5f6f8"/><text x="557.5" y="16.0" font-family="DejaVu Sans" font-size="11" fill="#1d2330" text-anchor="middle" font-weight="bold">Hard</text><text x="557.5" y="29.0" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle" font-weight="normal">generated, display seed 1000</text><path d="M611.5,113.7 L611.4,113.6 L611.3,113.5 L611.3,113.4 L611.2,113.3 L611.1,113.2 L611.0,113.2 L610.9,113.1 L610.9,113.0 L610.8,112.9 L610.7,112.9 L610.6,112.8 L610.5,112.7 L610.4,112.6 L610.4,112.5 L610.3,112.5 L610.2,112.4 L610.1,112.3 L610.0,112.2 L609.9,112.1 L609.8,112.1 L609.8,112.0 L609.7,111.9 L609.6,111.8 L609.5,111.7 L609.4,111.7 L609.3,111.6 L609.3,111.5 L609.2,111.4 L609.1,111.4 L609.0,111.3 L608.9,111.2 L608.8,111.1 L608.8,111.0 L608.7,111.0 L608.6,110.9 L608.5,110.8 L608.4,110.7 L608.3,110.6 L608.3,110.6 L608.2,110.5 L608.1,110.4 L608.0,110.3 L607.9,110.2 L607.8,110.2 L607.8,110.1 L607.7,110.0 L607.6,109.9 L607.5,109.9 L607.4,109.8 L607.3,109.7 L607.3,109.6 L607.2,109.5 L607.1,109.5 L607.0,109.4 L606.9,109.3 L606.8,109.2 L606.8,109.1 L606.7,109.1 L606.6,109.0 L606.5,108.9 L606.4,108.8 L606.3,108.7 L606.3,108.7 L606.2,108.6 L606.1,108.5 L606.0,108.4 L605.9,108.4 L605.8,108.3 L605.8,108.2 L605.7,108.1 L605.6,108.0 L605.5,108.0 L605.4,107.9 L605.3,107.8 L605.2,107.7 L605.2,107.6 L605.1,107.6 L605.0,107.5 L604.9,107.4 L604.8,107.3 L604.7,107.2 L604.7,107.2 L604.6,107.1 L604.5,107.0 L604.4,106.9 L604.3,106.9 L604.2,106.8 L604.2,106.7 L604.1,106.6 L604.0,106.5 L603.9,106.5 L603.8,106.4 L603.7,106.3 L603.7,106.2 L603.6,106.1 L603.5,106.1 L603.4,106.0 L603.3,105.9 L603.2,105.8 L603.2,105.7 L603.1,105.7 L603.0,105.6 L602.9,105.5 L602.8,105.4 L602.7,105.4 L602.7,105.3 L602.6,105.2 L602.5,105.1 L602.4,105.0 L602.3,105.0 L602.2,104.9 L602.2,104.8 L602.1,104.7 L602.0,104.6 L601.9,104.6 L601.8,104.5 L601.7,104.4 L601.7,104.3 L601.6,104.2 L601.5,104.2 L601.4,104.1 L601.3,104.0 L601.2,103.9 L601.2,103.9 L601.1,103.8 L601.0,103.7 L600.9,103.6 L600.8,103.5 L600.7,103.5 L600.7,103.4 L600.6,103.3 L600.5,103.2 L600.4,103.1 L600.3,103.1 L600.2,103.0 L600.1,102.9 L600.1,102.8 L600.0,102.7 L599.9,102.7 L599.8,102.6 L599.7,102.5 L599.6,102.4 L599.6,102.4 L599.5,102.3 L599.4,102.2 L599.3,102.1 L599.2,102.0 L599.1,102.0 L599.1,101.9 L599.0,101.8 L598.9,101.7 L598.8,101.6 L598.7,101.6 L598.6,101.5 L598.6,101.4 L598.5,101.3 L598.4,101.2 L598.3,101.2 L598.2,101.1 L598.1,101.0 L598.1,100.9 L598.0,100.9 L597.9,100.8 L597.8,100.7 L597.7,100.6 L597.6,100.5 L597.6,100.5 L597.5,100.4 L597.4,100.3 L597.3,100.2 L597.2,100.1 L597.1,100.1 L597.1,100.0 L597.0,99.9 L596.9,99.8 L596.8,99.7 L596.7,99.7 L596.6,99.6 L596.6,99.5 L596.5,99.4 L596.4,99.4 L596.3,99.3 L596.2,99.2 L596.1,99.1 L596.1,99.0 L596.0,99.0 L595.9,98.9 L595.8,98.8 L595.7,98.7 L595.6,98.6 L595.6,98.6 L595.5,98.5 L595.4,98.4 L595.3,98.3 L595.2,98.2 L595.1,98.2 L595.0,98.1 L595.0,98.0 L594.9,97.9 L594.8,97.9 L594.7,97.8 L594.6,97.7 L594.5,97.6 L594.5,97.5 L594.4,97.5 L594.3,97.4 L594.2,97.3 L594.1,97.2 L594.0,97.1 L594.0,97.1 L593.9,97.0 L593.8,96.9 L593.7,96.8 L593.6,96.7 L593.5,96.7 L593.5,96.6 L593.4,96.5 L593.3,96.4 L593.2,96.3 L593.2,96.2 L593.1,96.2 L593.0,96.1 L592.9,96.0 L592.8,95.9 L592.8,95.8 L592.7,95.7 L592.6,95.6 L592.6,95.6 L592.5,95.5 L592.4,95.4 L592.3,95.3 L592.3,95.2 L592.2,95.1 L592.1,95.0 L592.1,94.9 L592.0,94.8 L591.9,94.7 L591.9,94.6 L591.8,94.5 L591.7,94.4 L591.7,94.3 L591.6,94.2 L591.5,94.1 L591.5,94.1 L591.4,94.0 L591.4,93.9 L591.3,93.8 L591.2,93.7 L591.2,93.6 L591.1,93.5 L591.1,93.4 L591.0,93.3 L591.0,93.2 L590.9,93.1 L590.9,93.0 L590.8,92.8 L590.8,92.7 L590.7,92.6 L590.7,92.5 L590.6,92.4 L590.6,92.3 L590.5,92.2 L590.5,92.1 L590.4,92.0 L590.4,91.9 L590.3,91.8 L590.3,91.7 L590.3,91.6 L590.2,91.5 L590.2,91.4 L590.1,91.3 L590.1,91.2 L590.1,91.0 L590.0,90.9 L590.0,90.8 L590.0,90.7 L589.9,90.6 L589.9,90.5 L589.9,90.4 L589.8,90.3 L589.8,90.2 L589.8,90.1 L589.7,89.9 L589.7,89.8 L589.7,89.7 L589.7,89.6 L589.6,89.5 L589.6,89.4 L589.6,89.3 L589.6,89.2 L589.5,89.0 L589.5,88.9 L589.5,88.8 L589.5,88.7 L589.5,88.6 L589.4,88.5 L589.4,88.4 L589.4,88.3 L589.4,88.1 L589.4,88.0 L589.4,87.9 L589.4,87.8 L589.3,87.7 L589.3,87.6 L589.3,87.5 L589.3,87.3 L589.3,87.2 L589.3,87.1 L589.3,87.0 L589.3,86.9 L589.3,86.8 L589.3,86.6 L589.3,86.5 L589.3,86.4 L589.3,86.3 L589.3,86.2 L589.3,86.1 L589.3,86.0 L589.3,85.8 L589.3,85.7 L589.3,85.6 L589.3,85.5 L589.3,85.4 L589.3,85.3 L589.3,85.2 L589.3,85.0 L589.3,84.9 L589.3,84.8 L589.4,84.7 L589.4,84.6 L589.4,84.5 L589.4,84.4 L589.4,84.2 L589.4,84.1 L589.5,84.0 L589.5,83.9 L589.5,83.8 L589.5,83.7 L589.5,83.6 L589.5,83.4 L589.6,83.3 L589.6,83.2 L589.6,83.1 L589.6,83.0 L589.6,82.9 L589.6,82.8 L589.7,82.6 L589.7,82.5 L589.7,82.4 L589.7,82.3 L589.7,82.2 L589.8,82.1 L589.8,82.0 L589.8,81.9 L589.8,81.7 L589.8,81.6 L589.8,81.5 L589.9,81.4 L589.9,81.3 L589.9,81.2 L589.9,81.1 L589.9,80.9 L589.9,80.8 L590.0,80.7 L590.0,80.6 L590.0,80.5 L590.0,80.4 L590.0,80.3 L590.1,80.1 L590.1,80.0 L590.1,79.9 L590.1,79.8 L590.1,79.7 L590.1,79.6 L590.2,79.5 L590.2,79.4 L590.2,79.2 L590.2,79.1 L590.2,79.0 L590.2,78.9 L590.3,78.8 L590.3,78.7 L590.3,78.6 L590.3,78.4 L590.3,78.3 L590.4,78.2 L590.4,78.1 L590.4,78.0 L590.4,77.9 L590.4,77.8 L590.4,77.6 L590.5,77.5 L590.5,77.4 L590.5,77.3 L590.5,77.2 L590.5,77.1 L590.5,77.0 L590.6,76.9 L590.6,76.7 L590.6,76.6 L590.6,76.5 L590.6,76.4 L590.7,76.3 L590.7,76.2 L590.7,76.1 L590.7,75.9 L590.7,75.8 L590.7,75.7 L590.8,75.6 L590.8,75.5 L590.8,75.4 L590.8,75.3 L590.8,75.1 L590.9,75.0 L590.9,74.9 L590.9,74.8 L590.9,74.7 L590.9,74.6 L590.9,74.5 L591.0,74.4 L591.0,74.2 L591.0,74.1 L591.0,74.0 L591.0,73.9 L591.0,73.8 L591.1,73.7 L591.1,73.6 L591.1,73.4 L591.1,73.3 L591.1,73.2 L591.2,73.1 L591.2,73.0 L591.2,72.9 L591.2,72.8 L591.2,72.6 L591.2,72.5 L591.3,72.4 L591.3,72.3 L591.3,72.2 L591.3,72.1 L591.3,72.0 L591.3,71.9 L591.4,71.7 L591.4,71.6 L591.4,71.5 L591.4,71.4 L591.4,71.3 L591.5,71.2 L591.5,71.1 L591.5,70.9 L591.5,70.8 L591.5,70.7 L591.5,70.6 L591.6,70.5 L591.6,70.4 L591.6,70.3 L591.6,70.1 L591.6,70.0 L591.6,69.9 L591.7,69.8 L591.7,69.7 L591.7,69.6 L591.7,69.5 L591.7,69.4 L591.8,69.2 L591.8,69.1 L591.8,69.0 L591.8,68.9 L591.8,68.8 L591.8,68.7 L591.9,68.6 L591.9,68.4 L591.9,68.3 L591.9,68.2 L591.9,68.1 L591.9,68.0 L592.0,67.9 L592.0,67.8 L592.0,67.6 L592.0,67.5 L592.0,67.4 L592.1,67.3 L592.1,67.2 L592.1,67.1 L592.1,67.0 L592.1,66.9 L592.1,66.7 L592.1,66.6 L592.1,66.5 L592.2,66.4 L592.2,66.3 L592.2,66.2 L592.2,66.1 L592.2,65.9 L592.2,65.8 L592.2,65.7 L592.2,65.6 L592.2,65.5 L592.2,65.4 L592.2,65.2 L592.2,65.1 L592.2,65.0 L592.2,64.9 L592.2,64.8 L592.2,64.7 L592.2,64.6 L592.2,64.4 L592.2,64.3 L592.1,64.2 L592.1,64.1 L592.1,64.0 L592.1,63.9 L592.1,63.8 L592.1,63.6 L592.1,63.5 L592.0,63.4 L592.0,63.3 L592.0,63.2 L592.0,63.1 L592.0,63.0 L591.9,62.8 L591.9,62.7 L591.9,62.6 L591.9,62.5 L591.9,62.4 L591.8,62.3 L591.8,62.2 L591.8,62.1 L591.7,62.0 L591.7,61.8 L591.7,61.7 L591.6,61.6 L591.6,61.5 L591.6,61.4 L591.5,61.3 L591.5,61.2 L591.5,61.1 L591.4,61.0 L591.4,60.9 L591.3,60.8 L591.3,60.6 L591.3,60.5 L591.2,60.4 L591.2,60.3 L591.1,60.2 L591.1,60.1 L591.0,60.0 L591.0,59.9 L590.9,59.8 L590.9,59.7 L590.8,59.6 L590.8,59.5 L590.7,59.4 L590.7,59.3 L590.6,59.2 L590.6,59.1 L590.5,59.0 L590.5,58.9 L590.4,58.8 L590.3,58.7 L590.3,58.6 L590.2,58.5 L590.2,58.4 L590.1,58.3 L590.0,58.2 L590.0,58.1 L589.9,58.0 L589.8,57.9 L589.8,57.8 L589.7,57.7 L589.6,57.6 L589.6,57.5 L589.5,57.5 L589.4,57.4 L589.4,57.3 L589.3,57.2 L589.2,57.1 L589.1,57.0 L589.1,56.9 L589.0,56.8 L588.9,56.7 L588.8,56.7 L588.8,56.6 L588.7,56.5 L588.6,56.4 L588.5,56.3 L588.4,56.2 L588.4,56.2 L588.3,56.1 L588.2,56.0 L588.1,55.9 L588.0,55.8 L587.9,55.8 L587.9,55.7 L587.8,55.6 L587.7,55.5 L587.6,55.5 L587.5,55.4 L587.4,55.3 L587.3,55.2 L587.2,55.2 L587.1,55.1 L587.1,55.0 L587.0,55.0 L586.9,54.9 L586.8,54.8 L586.7,54.8 L586.6,54.7 L586.5,54.6 L586.4,54.6 L586.3,54.5 L586.2,54.4 L586.1,54.4 L586.0,54.3 L585.9,54.3 L585.8,54.2 L585.7,54.1 L585.6,54.1 L585.5,54.0 L585.4,54.0 L585.3,53.9 L585.2,53.9 L585.1,53.8 L585.0,53.8 L584.9,53.7 L584.8,53.7 L584.7,53.6 L584.6,53.6 L584.5,53.5 L584.4,53.5 L584.3,53.4 L584.2,53.4 L584.1,53.3 L584.0,53.3 L583.9,53.2 L583.7,53.2 L583.6,53.2 L583.5,53.1 L583.4,53.1 L583.3,53.1 L583.2,53.0 L583.1,53.0 L583.0,52.9 L582.9,52.9 L582.8,52.9 L582.6,52.9 L582.5,52.8 L582.4,52.8 L582.3,52.8 L582.2,52.7 L582.1,52.7 L582.0,52.7 L581.9,52.7 L581.8,52.6 L581.6,52.6 L581.5,52.6 L581.4,52.6 L581.3,52.6 L581.2,52.5 L581.1,52.5 L581.0,52.5 L580.8,52.5 L580.7,52.5 L580.6,52.5 L580.5,52.5 L580.4,52.4 L580.3,52.4 L580.2,52.4 L580.0,52.4 L579.9,52.4 L579.8,52.4 L579.7,52.4 L579.6,52.4 L579.5,52.4 L579.4,52.4 L579.2,52.4 L579.1,52.4 L579.0,52.4 L578.9,52.4 L578.8,52.4 L578.7,52.4 L578.6,52.4 L578.4,52.4 L578.3,52.4 L578.2,52.4 L578.1,52.4 L578.0,52.4 L577.9,52.5 L577.7,52.5 L577.6,52.5 L577.5,52.5 L577.4,52.5 L577.3,52.5 L577.2,52.5 L577.1,52.6 L577.0,52.6 L576.8,52.6 L576.7,52.6 L576.6,52.6 L576.5,52.7 L576.4,52.7 L576.3,52.7 L576.2,52.7 L576.1,52.8 L575.9,52.8 L575.8,52.8 L575.7,52.9 L575.6,52.9 L575.5,52.9 L575.4,53.0 L575.3,53.0 L575.2,53.0 L575.1,53.1 L575.0,53.1 L574.8,53.1 L574.7,53.2 L574.6,53.2 L574.5,53.3 L574.4,53.3 L574.3,53.3 L574.2,53.4 L574.1,53.4 L574.0,53.5 L573.9,53.5 L573.8,53.6 L573.7,53.6 L573.6,53.6 L573.4,53.7 L573.3,53.7 L573.2,53.8 L573.1,53.8 L573.0,53.9 L572.9,53.9 L572.8,53.9 L572.7,54.0 L572.6,54.0 L572.5,54.1 L572.4,54.1 L572.3,54.1 L572.2,54.2 L572.1,54.2 L572.0,54.3 L571.8,54.3 L571.7,54.4 L571.6,54.4 L571.5,54.4 L571.4,54.5 L571.3,54.5 L571.2,54.6 L571.1,54.6 L571.0,54.7 L570.9,54.7 L570.8,54.7 L570.7,54.8 L570.6,54.8 L570.5,54.9 L570.3,54.9 L570.2,55.0 L570.1,55.0 L570.0,55.0 L569.9,55.1 L569.8,55.1 L569.7,55.2 L569.6,55.2 L569.5,55.2 L569.4,55.3 L569.3,55.3 L569.2,55.4 L569.1,55.4 L569.0,55.5 L568.9,55.5 L568.7,55.5 L568.6,55.6 L568.5,55.6 L568.4,55.7 L568.3,55.7 L568.2,55.8 L568.1,55.8 L568.0,55.8 L567.9,55.9 L567.8,55.9 L567.7,56.0 L567.6,56.0 L567.5,56.0 L567.4,56.1 L567.2,56.1 L567.1,56.2 L567.0,56.2 L566.9,56.3 L566.8,56.3 L566.7,56.3 L566.6,56.4 L566.5,56.4 L566.4,56.5 L566.3,56.5 L566.2,56.6 L566.1,56.6 L566.0,56.6 L565.9,56.7 L565.8,56.7 L565.6,56.8 L565.5,56.8 L565.4,56.9 L565.3,56.9 L565.2,56.9 L565.1,57.0 L565.0,57.0 L564.9,57.1 L564.8,57.1 L564.7,57.1 L564.6,57.2 L564.5,57.2 L564.4,57.3 L564.3,57.3 L564.1,57.4 L564.0,57.4 L563.9,57.4 L563.8,57.5 L563.7,57.5 L563.6,57.6 L563.5,57.6 L563.4,57.7 L563.3,57.7 L563.2,57.7 L563.1,57.8 L563.0,57.8 L562.9,57.9 L562.8,57.9 L562.6,58.0 L562.5,58.0 L562.4,58.0 L562.3,58.1 L562.2,58.1 L562.1,58.2 L562.0,58.2 L561.9,58.2 L561.8,58.3 L561.7,58.3 L561.6,58.4 L561.5,58.4 L561.4,58.5 L561.3,58.5 L561.2,58.5 L561.0,58.6 L560.9,58.6 L560.8,58.7 L560.7,58.7 L560.6,58.8 L560.5,58.8 L560.4,58.8 L560.3,58.9 L560.2,58.9 L560.1,59.0 L560.0,59.0 L559.9,59.1 L559.8,59.1 L559.7,59.1 L559.5,59.2 L559.4,59.2 L559.3,59.3 L559.2,59.3 L559.1,59.3 L559.0,59.4 L558.9,59.4 L558.8,59.5 L558.7,59.5 L558.6,59.6 L558.5,59.6 L558.4,59.6 L558.3,59.7 L558.2,59.7 L558.1,59.8 L557.9,59.8 L557.8,59.9 L557.7,59.9 L557.6,59.9 L557.5,60.0 L557.4,60.0 L557.3,60.1 L557.2,60.1 L557.1,60.1 L557.0,60.2 L556.9,60.2 L556.8,60.3 L556.7,60.3 L556.6,60.4 L556.4,60.4 L556.3,60.4 L556.2,60.5 L556.1,60.5 L556.0,60.6 L555.9,60.6 L555.8,60.7 L555.7,60.7 L555.6,60.7 L555.5,60.8 L555.4,60.8 L555.3,60.9 L555.2,60.9 L555.1,61.0 L554.9,61.0 L554.8,61.0 L554.7,61.1 L554.6,61.1 L554.5,61.2 L554.4,61.2 L554.3,61.2 L554.2,61.3 L554.1,61.3 L554.0,61.4 L553.9,61.4 L553.8,61.5 L553.7,61.5 L553.6,61.5 L553.5,61.6 L553.3,61.6 L553.2,61.7 L553.1,61.7 L553.0,61.8 L552.9,61.8 L552.8,61.8 L552.7,61.9 L552.6,61.9 L552.5,62.0 L552.4,62.0 L552.3,62.1 L552.2,62.1 L552.1,62.1 L552.0,62.2 L551.8,62.2 L551.7,62.3 L551.6,62.3 L551.5,62.3 L551.4,62.4 L551.3,62.4 L551.2,62.5 L551.1,62.5 L551.0,62.6 L550.9,62.6 L550.8,62.6 L550.7,62.7 L550.6,62.7 L550.5,62.8 L550.4,62.8 L550.2,62.9 L550.1,62.9 L550.0,62.9 L549.9,63.0 L549.8,63.0 L549.7,63.1 L549.6,63.1 L549.5,63.2 L549.4,63.2 L549.3,63.2 L549.2,63.3 L549.1,63.3 L549.0,63.4 L548.9,63.4 L548.7,63.4 L548.6,63.5 L548.5,63.5 L548.4,63.6 L548.3,63.6 L548.2,63.7 L548.1,63.7 L548.0,63.7 L547.9,63.8 L547.8,63.8 L547.7,63.9 L547.6,63.9 L547.5,64.0 L547.4,64.0 L547.2,64.0 L547.1,64.1 L547.0,64.1 L546.9,64.2 L546.8,64.2 L546.7,64.2 L546.6,64.3 L546.5,64.3 L546.4,64.3 L546.3,64.3 L546.1,64.3 L546.0,64.4 L545.9,64.4 L545.8,64.4 L545.7,64.4 L545.6,64.4 L545.5,64.4 L545.3,64.4 L545.2,64.4 L545.1,64.4 L545.0,64.4 L544.9,64.3 L544.8,64.3 L544.7,64.3 L544.5,64.3 L544.4,64.3 L544.3,64.2 L544.2,64.2 L544.1,64.2 L544.0,64.1 L543.9,64.1 L543.8,64.0 L543.7,64.0 L543.6,63.9 L543.5,63.9 L543.4,63.8 L543.3,63.8 L543.2,63.7 L543.1,63.7 L543.0,63.6 L542.9,63.5 L542.8,63.5 L542.7,63.4 L542.6,63.4 L542.5,63.3 L542.4,63.2 L542.3,63.2 L542.2,63.1 L542.1,63.0 L542.0,63.0 L541.9,62.9 L541.8,62.9 L541.7,62.8 L541.6,62.7 L541.5,62.7 L541.4,62.6 L541.3,62.6 L541.2,62.5 L541.1,62.4 L541.0,62.4 L540.9,62.3 L540.8,62.3 L540.7,62.2 L540.6,62.1 L540.5,62.1 L540.4,62.0 L540.3,61.9 L540.2,61.9 L540.1,61.8 L540.0,61.8 L539.9,61.7 L539.9,61.6 L539.8,61.6 L539.7,61.5 L539.6,61.5 L539.5,61.4 L539.4,61.3 L539.3,61.3 L539.2,61.2 L539.1,61.2 L539.0,61.1 L538.9,61.0 L538.8,61.0 L538.7,60.9 L538.6,60.8 L538.5,60.8 L538.4,60.7 L538.3,60.7 L538.2,60.6 L538.1,60.5 L538.0,60.5 L537.9,60.4 L537.8,60.4 L537.7,60.3 L537.6,60.2 L537.5,60.2 L537.4,60.1 L537.3,60.1 L537.2,60.0 L537.1,59.9 L537.0,59.9 L536.9,59.8 L536.8,59.7 L536.7,59.7 L536.6,59.6 L536.5,59.6 L536.4,59.5 L536.3,59.4 L536.2,59.4 L536.2,59.3 L536.1,59.3 L536.0,59.2 L535.9,59.1 L535.8,59.1 L535.7,59.0 L535.6,59.0 L535.5,58.9 L535.4,58.8 L535.3,58.8 L535.2,58.7 L535.1,58.6 L535.0,58.6 L534.9,58.5 L534.8,58.5 L534.7,58.4 L534.6,58.3 L534.5,58.3 L534.4,58.2 L534.3,58.2 L534.2,58.1 L534.1,58.0 L534.0,58.0 L533.9,57.9 L533.8,57.9 L533.7,57.8 L533.6,57.7 L533.5,57.7 L533.4,57.6 L533.3,57.5 L533.2,57.5 L533.1,57.4 L533.0,57.4 L532.9,57.3 L532.8,57.2 L532.7,57.2 L532.6,57.1 L532.5,57.1 L532.4,57.0 L532.4,56.9 L532.3,56.9 L532.2,56.8 L532.1,56.7 L532.0,56.7 L531.9,56.6 L531.8,56.6 L531.7,56.5 L531.6,56.4 L531.5,56.4 L531.4,56.3 L531.3,56.3 L531.2,56.2 L531.1,56.1 L531.0,56.1 L530.9,56.0 L530.8,56.0 L530.7,55.9 L530.6,55.8 L530.5,55.8 L530.4,55.7 L530.3,55.6 L530.2,55.6 L530.1,55.5 L530.0,55.5 L529.9,55.4 L529.8,55.3 L529.7,55.3 L529.6,55.2 L529.5,55.2 L529.4,55.1 L529.3,55.0 L529.2,55.0 L529.1,54.9 L529.0,54.9 L528.9,54.8 L528.8,54.7 L528.7,54.7 L528.7,54.6 L528.6,54.5 L528.5,54.5 L528.4,54.4 L528.3,54.4 L528.2,54.3 L528.1,54.2 L528.0,54.2 L527.9,54.1 L527.8,54.1 L527.7,54.0 L527.6,53.9 L527.5,53.9 L527.4,53.8 L527.3,53.8 L527.2,53.7 L527.1,53.6 L527.0,53.6 L526.9,53.5 L526.8,53.4 L526.7,53.4 L526.6,53.3 L526.5,53.3 L526.4,53.2 L526.3,53.1 L526.2,53.1 L526.1,53.0 L526.0,53.0 L525.9,52.9 L525.8,52.8 L525.7,52.8 L525.6,52.7 L525.5,52.7 L525.4,52.6 L525.3,52.5 L525.2,52.5 L525.1,52.4 L525.0,52.3 L525.0,52.3 L524.9,52.2 L524.8,52.2 L524.7,52.1 L524.6,52.0 L524.5,52.0 L524.4,51.9 L524.3,51.9 L524.2,51.8 L524.1,51.7 L524.0,51.7 L523.9,51.6 L523.8,51.6 L523.7,51.5 L523.6,51.4 L523.5,51.4 L523.4,51.3 L523.3,51.2 L523.2,51.2 L523.1,51.1 L523.0,51.1 L522.9,51.0 L522.8,50.9 L522.7,50.9 L522.6,50.8 L522.5,50.8 L522.4,50.7 L522.3,50.6 L522.2,50.6 L522.1,50.5 L522.0,50.5 L521.9,50.4 L521.8,50.3 L521.7,50.3 L521.6,50.2 L521.5,50.1 L521.4,50.1 L521.3,50.0 L521.3,50.0 L521.2,49.9 L521.1,49.8 L521.0,49.8 L520.9,49.7 L520.8,49.7 L520.7,49.6 L520.6,49.5 L520.5,49.5 L520.4,49.4 L520.3,49.4 L520.2,49.3 L520.1,49.2 L520.0,49.2 L519.9,49.1 L519.8,49.0 L519.7,49.0 L519.6,48.9 L519.5,48.9 L519.4,48.8 L519.3,48.7 L519.2,48.7 L519.1,48.6 L519.0,48.6 L518.9,48.5 L518.8,48.4 L518.7,48.4 L518.6,48.3 L518.5,48.3 L518.4,48.2 L518.3,48.1 L518.2,48.1 L518.1,48.0 L518.0,47.9 L517.9,47.9 L517.8,47.8 L517.7,47.8 L517.6,47.7 L517.5,47.6 L517.5,47.6 L517.4,47.5 L517.3,47.5 L517.2,47.4 L517.1,47.3 L517.0,47.3 L516.9,47.2 L516.8,47.1 L516.7,47.1 L516.6,47.0 L516.5,47.0 L516.4,46.9 L516.3,46.8 L516.2,46.8 L516.1,46.7 L516.0,46.7 L515.9,46.6 L515.8,46.5 L515.7,46.5 L515.6,46.4 L515.5,46.4 L515.4,46.3 L515.3,46.2 L515.2,46.2 L515.1,46.1 L515.0,46.0 L514.9,46.0 L514.8,45.9 L514.7,45.9 L514.6,45.8 L514.5,45.7 L514.4,45.7 L514.3,45.6 L514.2,45.6 L514.1,45.5 L514.0,45.4 L513.9,45.4 L513.8,45.3 L513.8,45.3 L513.7,45.2 L513.6,45.1 L513.5,45.1 L513.4,45.0 L513.3,44.9 L513.2,44.9 L513.1,44.8 L513.0,44.8 L512.9,44.7 L512.8,44.6 L512.7,44.6 L512.6,44.5 L512.5,44.5 L512.4,44.4 L512.3,44.3 L512.2,44.3 L512.1,44.2 L512.0,44.2 L511.9,44.1 L511.8,44.0 L511.7,44.0 L511.6,43.9 L511.5,43.8 L511.4,43.8 L511.3,43.7 L511.2,43.7 L511.1,43.6 L511.0,43.5 L510.9,43.5 L510.8,43.4 L510.7,43.4 L510.6,43.3 L510.5,43.2 L510.4,43.2 L510.3,43.1 L510.2,43.1 L510.1,43.0 L510.1,42.9 L510.0,42.9 L509.9,42.8 L509.8,42.7 L509.7,42.7 L509.6,42.6 L509.5,42.6 L509.4,42.5 L509.3,42.4 L509.2,42.4 L509.1,42.3 L509.0,42.3 L508.9,42.2 L508.8,42.1 L508.7,42.1 L508.6,42.0 L508.5,42.0 L508.4,41.9 L508.3,41.8 L508.2,41.8 L508.1,41.7 L508.0,41.6 L507.9,41.6 L507.8,41.5 L507.7,41.5 L507.6,41.4 L507.5,41.3 L507.4,41.3 L507.3,41.2 L507.2,41.2 L507.1,41.1 L507.0,41.0 L506.9,41.0 L506.8,40.9 L506.7,40.9 L506.6,40.8 L506.5,40.7 L506.4,40.7 L506.3,40.6 L506.3,40.5 L506.2,40.5 L506.1,40.4 L506.0,40.4 L505.9,40.3 L505.8,40.2 L505.7,40.2 L505.6,40.1 L505.5,40.1 L505.4,40.0 L505.3,39.9 L505.2,39.9 L505.1,39.8 L505.0,39.8 L504.9,39.7 L504.8,39.6 L504.7,39.6 L504.6,39.5 L504.5,39.4 L504.4,39.4 L504.3,39.3 L504.2,39.3 L504.1,39.2 L504.0,39.1 L503.9,39.1 L503.8,39.0 L503.7,39.0 L503.6,38.9 L503.5,38.8 L503.4,38.8 L503.3,38.7 L503.2,38.6 L503.1,38.6 L503.0,38.5 L502.9,38.5 L502.8,38.4 L502.7,38.3 L502.6,38.3 L502.6,38.2 L502.5,38.2 L502.4,38.1 L502.3,38.0 L502.2,38.0 L502.1,37.9 L502.0,37.9 L501.9,37.8 L501.8,37.7 L501.7,37.7 L501.6,37.6 L501.5,37.5 L501.4,37.5 L501.3,37.4 L501.2,37.4 L501.1,37.3 L501.0,37.2 L500.9,37.2 L500.8,37.1 L500.7,37.1 L500.6,37.0 L500.5,36.9 L500.4,36.9 L500.3,36.8 L500.2,36.8 L500.1,36.8 L500.0,36.7 L499.9,36.7 L499.8,36.7 L499.6,36.6 L499.5,36.6 L499.4,36.6 L499.3,36.6 L499.2,36.6 L499.1,36.6 L499.0,36.6 L498.8,36.6 L498.7,36.6 L498.6,36.6 L498.5,36.6 L498.4,36.6 L498.3,36.7 L498.2,36.7 L498.1,36.7 L497.9,36.8 L497.8,36.8 L497.7,36.9 L497.6,36.9 L497.5,37.0 L497.4,37.0 L497.3,37.1 L497.2,37.2 L497.2,37.2 L497.1,37.3 L497.0,37.4 L496.9,37.5 L496.8,37.6 L496.7,37.6 L496.7,37.7 L496.6,37.8 L496.5,37.9 L496.4,38.0 L496.4,38.1 L496.3,38.2 L496.2,38.2 L496.1,38.3 L496.1,38.4 L496.0,38.5 L495.9,38.6 L495.8,38.7 L495.7,38.8 L495.7,38.8 L495.6,38.9 L495.5,39.0 L495.4,39.1 L495.4,39.2 L495.3,39.3 L495.2,39.4 L495.1,39.4 L495.1,39.5 L495.0,39.6 L494.9,39.7 L494.8,39.8 L494.8,39.9 L494.7,40.0 L494.6,40.0 L494.5,40.1 L494.4,40.2 L494.4,40.3 L494.3,40.4 L494.2,40.5 L494.1,40.6 L494.1,40.6 L494.0,40.7 L493.9,40.8 L493.8,40.9 L493.8,41.0 L493.7,41.1 L493.6,41.2 L493.5,41.2 L493.4,41.3 L493.4,41.4 L493.3,41.5 L493.2,41.6 L493.1,41.7 L493.1,41.8 L493.0,41.8 L492.9,41.9 L492.8,42.0 L492.8,42.1 L492.7,42.2 L492.6,42.3 L492.5,42.4 L492.5,42.4 L492.4,42.5 L492.3,42.6 L492.2,42.7 L492.1,42.8 L492.1,42.9 L492.0,43.0 L491.9,43.0 L491.8,43.1 L491.8,43.2 L491.7,43.3 L491.6,43.4 L491.5,43.5 L491.5,43.6 L491.4,43.6 L491.3,43.7 L491.2,43.8 L491.2,43.9 L491.1,44.0 L491.0,44.1 L490.9,44.2 L490.8,44.2 L490.8,44.3 L490.7,44.4 L490.6,44.5 L490.5,44.6 L490.5,44.7 L490.4,44.8 L490.3,44.9 L490.2,44.9 L490.2,45.0 L490.1,45.1 L490.0,45.2 L489.9,45.3 L489.9,45.4 L489.8,45.5 L489.7,45.5 L489.6,45.6 L489.5,45.7 L489.5,45.8 L489.4,45.9 L489.3,46.0 L489.2,46.1 L489.2,46.1 L489.1,46.2 L489.0,46.3 L488.9,46.4 L488.9,46.5 L488.8,46.6 L488.7,46.7 L488.6,46.7 L488.6,46.8 L488.5,46.9 L488.4,47.0 L488.3,47.1 L488.2,47.2 L488.2,47.3 L488.1,47.3 L488.0,47.4 L487.9,47.5 L487.9,47.6 L487.8,47.7 L487.7,47.8 L487.6,47.9 L487.6,47.9 L487.5,48.0 L487.4,48.1 L487.3,48.2 L487.3,48.3 L487.2,48.4 L487.1,48.5 L487.0,48.5 L486.9,48.6 L486.9,48.7 L486.8,48.8 L486.7,48.9 L486.6,49.0 L486.6,49.1 L486.5,49.1 L486.4,49.2 L486.3,49.3 L486.3,49.4 L486.2,49.5 L486.1,49.6 L486.0,49.7 L486.0,49.7 L485.9,49.8 L485.8,49.9 L485.7,50.0 L485.6,50.1 L485.6,50.2 L485.5,50.3 L485.4,50.3 L485.3,50.4 L485.3,50.5 L485.2,50.6 L485.1,50.7 L485.0,50.8 L485.0,50.9 L484.9,50.9 L484.8,51.0 L484.7,51.1 L484.7,51.2 L484.6,51.3 L484.5,51.4 L484.4,51.5 L484.3,51.5 L484.3,51.6 L484.2,51.7 L484.1,51.8 L484.0,51.9 L484.0,52.0 L483.9,52.1 L483.8,52.1 L483.7,52.2 L483.7,52.3 L483.6,52.4 L483.5,52.5 L483.4,52.6 L483.4,52.7 L483.3,52.8 L483.2,52.8 L483.1,52.9 L483.0,53.0 L483.0,53.1 L482.9,53.2 L482.8,53.3 L482.7,53.4 L482.7,53.4 L482.6,53.5 L482.5,53.6 L482.4,53.7 L482.4,53.8 L482.3,53.9 L482.2,54.0 L482.1,54.0 L482.1,54.1 L482.0,54.2 L481.9,54.3 L481.8,54.4 L481.7,54.5 L481.7,54.6 L481.6,54.6 L481.5,54.7 L481.4,54.8 L481.4,54.9 L481.3,55.0 L481.2,55.1 L481.1,55.2 L481.1,55.2 L481.0,55.3 L480.9,55.4 L480.8,55.5 L480.8,55.6 L480.7,55.7 L480.6,55.8 L480.5,55.8 L480.4,55.9 L480.4,56.0 L480.3,56.1 L480.2,56.2 L480.1,56.3 L480.1,56.4 L480.0,56.4 L479.9,56.5 L479.8,56.6 L479.8,56.7 L479.7,56.8 L479.6,56.9 L479.5,57.0 L479.5,57.0 L479.4,57.1 L479.3,57.2 L479.2,57.3 L479.1,57.4 L479.1,57.5 L479.0,57.6 L478.9,57.6 L478.8,57.7 L478.8,57.8 L478.7,57.9 L478.6,58.0 L478.5,58.1 L478.5,58.2 L478.4,58.2 L478.3,58.3 L478.2,58.4 L478.2,58.5 L478.1,58.6 L478.0,58.7 L477.9,58.8 L477.8,58.8 L477.8,58.9 L477.7,59.0 L477.6,59.1 L477.5,59.2 L477.5,59.3 L477.4,59.4 L477.3,59.5 L477.2,59.5 L477.2,59.6 L477.1,59.7 L477.1,59.8 L477.0,59.9 L476.9,60.0 L476.9,60.1 L476.9,60.2 L476.8,60.4 L476.8,60.5 L476.8,60.6 L476.7,60.7 L476.7,60.8 L476.7,60.9 L476.7,61.0 L476.7,61.1 L476.6,61.3 L476.6,61.4 L476.6,61.5 L476.6,61.6 L476.6,61.7 L476.6,61.8 L476.6,61.9 L476.5,62.1 L476.5,62.2 L476.5,62.3 L476.5,62.4 L476.5,62.5 L476.5,62.6 L476.5,62.7 L476.4,62.9 L476.4,63.0 L476.4,63.1 L476.4,63.2 L476.4,63.3 L476.4,63.4 L476.3,63.5 L476.3,63.6 L476.3,63.8 L476.3,63.9 L476.3,64.0 L476.3,64.1 L476.3,64.2 L476.2,64.3 L476.2,64.4 L476.2,64.6 L476.2,64.7 L476.2,64.8 L476.2,64.9 L476.1,65.0 L476.1,65.1 L476.1,65.2 L476.1,65.4 L476.1,65.5 L476.1,65.6 L476.1,65.7 L476.0,65.8 L476.0,65.9 L476.0,66.0 L476.0,66.2 L476.0,66.3 L476.0,66.4 L475.9,66.5 L475.9,66.6 L475.9,66.7 L475.9,66.8 L475.9,67.0 L475.9,67.1 L475.9,67.2 L475.8,67.3 L475.8,67.4 L475.8,67.5 L475.8,67.6 L475.8,67.8 L475.8,67.9 L475.7,68.0 L475.7,68.1 L475.7,68.2 L475.7,68.3 L475.7,68.4 L475.7,68.6 L475.7,68.7 L475.6,68.8 L475.6,68.9 L475.6,69.0 L475.6,69.1 L475.6,69.2 L475.6,69.3 L475.6,69.5 L475.5,69.6 L475.5,69.7 L475.5,69.8 L475.5,69.9 L475.5,70.0 L475.5,70.1 L475.4,70.3 L475.4,70.4 L475.4,70.5 L475.4,70.6 L475.4,70.7 L475.4,70.8 L475.4,70.9 L475.3,71.1 L475.3,71.2 L475.3,71.3 L475.3,71.4 L475.3,71.5 L475.3,71.6 L475.2,71.7 L475.2,71.9 L475.2,72.0 L475.2,72.1 L475.2,72.2 L475.2,72.3 L475.2,72.4 L475.1,72.5 L475.1,72.7 L475.1,72.8 L475.1,72.9 L475.1,73.0 L475.1,73.1 L475.0,73.2 L475.0,73.3 L475.0,73.5 L475.0,73.6 L475.0,73.7 L475.0,73.8 L475.0,73.9 L474.9,74.0 L474.9,74.1 L474.9,74.2 L474.9,74.4 L474.9,74.5 L474.9,74.6 L474.9,74.7 L474.8,74.8 L474.8,74.9 L474.8,75.0 L474.8,75.2 L474.8,75.3 L474.8,75.4 L474.7,75.5 L474.7,75.6 L474.7,75.7 L474.7,75.8 L474.7,76.0 L474.7,76.1 L474.7,76.2 L474.6,76.3 L474.6,76.4 L474.6,76.5 L474.6,76.6 L474.6,76.8 L474.6,76.9 L474.5,77.0 L474.5,77.1 L474.5,77.2 L474.5,77.3 L474.5,77.4 L474.5,77.6 L474.5,77.7 L474.4,77.8 L474.4,77.9 L474.4,78.0 L474.4,78.1 L474.4,78.2 L474.4,78.4 L474.3,78.5 L474.3,78.6 L474.3,78.7 L474.3,78.8 L474.3,78.9 L474.3,79.0 L474.3,79.2 L474.2,79.3 L474.2,79.4 L474.2,79.5 L474.2,79.6 L474.2,79.7 L474.2,79.8 L474.1,79.9 L474.1,80.1 L474.1,80.2 L474.1,80.3 L474.1,80.4 L474.1,80.5 L474.1,80.6 L474.0,80.7 L474.0,80.9 L474.0,81.0 L474.0,81.1 L474.0,81.2 L474.0,81.3 L474.0,81.4 L473.9,81.5 L473.9,81.7 L473.9,81.8 L473.9,81.9 L473.9,82.0 L473.9,82.1 L473.8,82.2 L473.8,82.3 L473.8,82.5 L473.8,82.6 L473.8,82.7 L473.8,82.8 L473.8,82.9 L473.7,83.0 L473.7,83.1 L473.7,83.3 L473.7,83.4 L473.7,83.5 L473.7,83.6 L473.6,83.7 L473.6,83.8 L473.6,83.9 L473.6,84.1 L473.6,84.2 L473.6,84.3 L473.6,84.4 L473.5,84.5 L473.5,84.6 L473.5,84.7 L473.5,84.8 L473.5,85.0 L473.5,85.1 L473.4,85.2 L473.4,85.3 L473.4,85.4 L473.4,85.5 L473.4,85.6 L473.4,85.8 L473.4,85.9 L473.3,86.0 L473.3,86.1 L473.3,86.2 L473.3,86.3 L473.3,86.4 L473.3,86.6 L473.2,86.7 L473.2,86.8 L473.2,86.9 L473.2,87.0 L473.2,87.1 L473.2,87.2 L473.2,87.4 L473.1,87.5 L473.1,87.6 L473.1,87.7 L473.1,87.8 L473.1,87.9 L473.1,88.0 L473.1,88.2 L473.0,88.3 L473.0,88.4 L473.0,88.5 L473.0,88.6 L473.0,88.7 L473.0,88.8 L472.9,89.0 L472.9,89.1 L472.9,89.2 L472.9,89.3 L472.9,89.4 L472.9,89.5 L472.9,89.6 L472.8,89.8 L472.8,89.9 L472.8,90.0 L472.8,90.1 L472.8,90.2 L472.8,90.3 L472.7,90.4 L472.7,90.5 L472.7,90.7 L472.7,90.8 L472.7,90.9 L472.7,91.0 L472.7,91.1 L472.6,91.2 L472.6,91.3 L472.6,91.5 L472.6,91.6 L472.6,91.7 L472.6,91.8 L472.5,91.9 L472.5,92.0 L472.5,92.1 L472.5,92.3 L472.5,92.4 L472.5,92.5 L472.5,92.6 L472.4,92.7 L472.4,92.8 L472.4,92.9 L472.4,93.1 L472.4,93.2 L472.4,93.3 L472.4,93.4 L472.3,93.5 L472.3,93.6 L472.3,93.7 L472.3,93.9 L472.3,94.0 L472.3,94.1 L472.2,94.2 L472.2,94.3 L472.2,94.4 L472.2,94.5 L472.2,94.7 L472.2,94.8 L472.2,94.9 L472.1,95.0 L472.1,95.1 L472.1,95.2 L472.1,95.3 L472.1,95.4 L472.1,95.6 L472.0,95.7 L472.0,95.8 L472.0,95.9 L472.0,96.0 L472.0,96.1 L472.0,96.2 L472.0,96.4 L471.9,96.5 L471.9,96.6 L471.9,96.7 L471.9,96.8 L471.9,96.9 L471.9,97.0 L471.8,97.2 L471.8,97.3 L471.8,97.4 L471.8,97.5 L471.8,97.6 L471.8,97.7 L471.8,97.8 L471.7,98.0 L471.7,98.1 L471.7,98.2 L471.7,98.3 L471.7,98.4 L471.7,98.5 L471.6,98.6 L471.6,98.8 L471.6,98.9 L471.6,99.0 L471.6,99.1 L471.6,99.2 L471.6,99.3 L471.5,99.4 L471.5,99.6 L471.5,99.7 L471.5,99.8 L471.5,99.9 L471.5,100.0 L471.5,100.1 L471.4,100.2 L471.4,100.4 L471.4,100.5 L471.4,100.6 L471.4,100.7 L471.4,100.8 L471.4,100.9 L471.4,101.0 L471.4,101.2 L471.4,101.3 L471.4,101.4 L471.4,101.5 L471.3,101.6 L471.3,101.7 L471.3,101.8 L471.3,102.0 L471.3,102.1 L471.3,102.2 L471.3,102.3 L471.3,102.4 L471.3,102.5 L471.3,102.6 L471.3,102.8 L471.3,102.9 L471.3,103.0 L471.3,103.1 L471.3,103.2 L471.3,103.3 L471.3,103.5 L471.3,103.6 L471.2,103.7 L471.2,103.8 L471.2,103.9 L471.2,104.0 L471.2,104.1 L471.2,104.3 L471.2,104.4 L471.2,104.5 L471.2,104.6 L471.2,104.7 L471.2,104.8 L471.2,104.9 L471.2,105.1 L471.2,105.2 L471.2,105.3 L471.2,105.4 L471.2,105.5 L471.1,105.6 L471.1,105.7 L471.1,105.9 L471.1,106.0 L471.1,106.1 L471.1,106.2 L471.1,106.3 L471.1,106.4 L471.1,106.6 L471.1,106.7 L471.1,106.8 L471.1,106.9 L471.1,107.0 L471.1,107.1 L471.1,107.2 L471.1,107.4 L471.1,107.5 L471.1,107.6 L471.0,107.7 L471.0,107.8 L471.0,107.9 L471.0,108.0 L471.0,108.2 L471.0,108.3 L471.0,108.4 L471.0,108.5 L471.0,108.6 L471.0,108.7 L471.0,108.8 L471.0,109.0 L471.0,109.1 L471.0,109.2 L471.0,109.3 L471.0,109.4 L471.0,109.5 L471.0,109.7 L470.9,109.8 L470.9,109.9 L470.9,110.0 L470.9,110.1 L470.9,110.2 L470.9,110.3 L470.9,110.5 L470.9,110.6 L470.9,110.7 L470.9,110.8 L470.9,110.9 L470.9,111.0 L470.9,111.1 L470.9,111.3 L470.9,111.4 L470.9,111.5 L470.9,111.6 L470.9,111.7 L470.8,111.8 L470.8,111.9 L470.8,112.1 L470.8,112.2 L470.8,112.3 L470.8,112.4 L470.8,112.5 L470.8,112.6 L470.8,112.8 L470.8,112.9 L470.8,113.0 L470.8,113.1 L470.8,113.2 L470.8,113.3 L470.8,113.4 L470.8,113.6 L470.8,113.7 L470.7,113.8 L470.7,113.9 L470.7,114.0 L470.7,114.1 L470.7,114.2 L470.7,114.4 L470.7,114.5 L470.7,114.6 L470.7,114.7 L470.7,114.8 L470.7,114.9 L470.7,115.1 L470.7,115.2 L470.7,115.3 L470.7,115.4 L470.7,115.5 L470.7,115.6 L470.7,115.7 L470.6,115.9 L470.6,116.0 L470.6,116.1 L470.6,116.2 L470.6,116.3 L470.6,116.4 L470.6,116.5 L470.6,116.7 L470.6,116.8 L470.6,116.9 L470.6,117.0 L470.6,117.1 L470.6,117.2 L470.6,117.3 L470.6,117.5 L470.6,117.6 L470.6,117.7 L470.6,117.8 L470.5,117.9 L470.5,118.0 L470.5,118.2 L470.5,118.3 L470.5,118.4 L470.5,118.5 L470.5,118.6 L470.5,118.7 L470.5,118.8 L470.5,119.0 L470.5,119.1 L470.5,119.2 L470.5,119.3 L470.5,119.4 L470.5,119.5 L470.5,119.6 L470.5,119.8 L470.4,119.9 L470.4,120.0 L470.4,120.1 L470.4,120.2 L470.4,120.3 L470.4,120.4 L470.4,120.6 L470.4,120.7 L470.4,120.8 L470.4,120.9 L470.4,121.0 L470.4,121.1 L470.4,121.3 L470.4,121.4 L470.4,121.5 L470.4,121.6 L470.4,121.7 L470.4,121.8 L470.3,121.9 L470.3,122.1 L470.3,122.2 L470.3,122.3 L470.3,122.4 L470.3,122.5 L470.3,122.6 L470.3,122.7 L470.3,122.9 L470.3,123.0 L470.3,123.1 L470.3,123.2 L470.3,123.3 L470.3,123.4 L470.3,123.5 L470.3,123.7 L470.3,123.8 L470.3,123.9 L470.2,124.0 L470.2,124.1 L470.2,124.2 L470.2,124.4 L470.2,124.5 L470.2,124.6 L470.2,124.7 L470.2,124.8 L470.2,124.9 L470.2,125.0 L470.2,125.2 L470.2,125.3 L470.2,125.4 L470.2,125.5 L470.2,125.6 L470.2,125.7 L470.2,125.8 L470.1,126.0 L470.1,126.1 L470.1,126.2 L470.1,126.3 L470.1,126.4 L470.1,126.5 L470.1,126.7 L470.1,126.8 L470.1,126.9 L470.1,127.0 L470.1,127.1 L470.1,127.2 L470.1,127.3 L470.1,127.5 L470.1,127.6 L470.1,127.7 L470.1,127.8 L470.1,127.9 L470.0,128.0 L470.0,128.1 L470.0,128.3 L470.0,128.4 L470.0,128.5 L470.0,128.6 L470.0,128.7 L470.0,128.8 L470.0,128.9 L470.0,129.1 L470.0,129.2 L470.0,129.3 L470.0,129.4 L470.0,129.5 L470.0,129.6 L470.0,129.8 L470.0,129.9 L470.0,130.0 L469.9,130.1 L469.9,130.2 L469.9,130.3 L469.9,130.4 L469.9,130.6 L469.9,130.7 L469.9,130.8 L469.9,130.9 L469.9,131.0 L469.9,131.1 L469.9,131.2 L469.9,131.4 L469.9,131.5 L469.9,131.6 L469.9,131.7 L469.9,131.8 L469.9,131.9 L469.9,132.0 L469.8,132.2 L469.8,132.3 L469.8,132.4 L469.8,132.5 L469.8,132.6 L469.8,132.7 L469.8,132.9 L469.8,133.0 L469.8,133.1 L469.8,133.2 L469.8,133.3 L469.8,133.4 L469.8,133.5 L469.8,133.7 L469.8,133.8 L469.8,133.9 L469.8,134.0 L469.7,134.1 L469.7,134.2 L469.7,134.3 L469.7,134.5 L469.7,134.6 L469.7,134.7 L469.7,134.8 L469.7,134.9 L469.7,135.0 L469.7,135.1 L469.7,135.3 L469.7,135.4 L469.7,135.5 L469.7,135.6 L469.7,135.7 L469.7,135.8 L469.7,136.0 L469.7,136.1 L469.6,136.2 L469.6,136.3 L469.6,136.4 L469.6,136.5 L469.6,136.6 L469.6,136.8 L469.6,136.9 L469.6,137.0 L469.6,137.1 L469.6,137.2 L469.6,137.3 L469.6,137.4 L469.6,137.6 L469.6,137.7 L469.6,137.8 L469.6,137.9 L469.6,138.0 L469.6,138.1 L469.5,138.3 L469.5,138.4 L469.5,138.5 L469.5,138.6 L469.5,138.7 L469.5,138.8 L469.5,138.9 L469.5,139.1 L469.5,139.2 L469.5,139.3 L469.5,139.4 L469.5,139.5 L469.5,139.6 L469.5,139.7 L469.5,139.9 L469.5,140.0 L469.5,140.1 L469.4,140.2 L469.4,140.3 L469.4,140.4 L469.4,140.5 L469.4,140.7 L469.4,140.8 L469.4,140.9 L469.4,141.0 L469.4,141.1 L469.4,141.2 L469.4,141.4 L469.4,141.5 L469.4,141.6 L469.4,141.7 L469.4,141.8 L469.4,141.9 L469.4,142.0 L469.4,142.2 L469.3,142.3 L469.3,142.4 L469.3,142.5 L469.3,142.6 L469.3,142.7 L469.3,142.8 L469.3,143.0 L469.4,143.1 L469.4,143.2 L469.4,143.3 L469.4,143.4 L469.4,143.5 L469.5,143.6 L469.5,143.7 L469.5,143.9 L469.6,144.0 L469.6,144.1 L469.7,144.2 L469.7,144.3 L469.8,144.4 L469.8,144.5 L469.9,144.6 L470.0,144.7 L470.0,144.8 L470.1,144.9 L470.2,144.9 L470.3,145.0 L470.3,145.1 L470.4,145.2 L470.5,145.3 L470.6,145.3 L470.7,145.4 L470.8,145.5 L470.9,145.6 L471.0,145.6 L471.1,145.7 L471.2,145.7 L471.3,145.8 L471.4,145.8 L471.5,145.9 L471.6,145.9 L471.7,146.0 L471.8,146.0 L471.9,146.0 L472.0,146.1 L472.1,146.1 L472.2,146.1 L472.3,146.1 L472.5,146.1 L472.6,146.2 L472.7,146.2 L472.8,146.2 L472.9,146.2 L473.0,146.2 L473.2,146.2 L473.3,146.1 L473.4,146.1 L473.5,146.1 L473.6,146.1 L473.7,146.1 L473.8,146.1 L473.9,146.0 L474.1,146.0 L474.2,146.0 L474.3,146.0 L474.4,146.0 L474.5,145.9 L474.6,145.9 L474.7,145.9 L474.9,145.9 L475.0,145.9 L475.1,145.8 L475.2,145.8 L475.3,145.8 L475.4,145.8 L475.5,145.8 L475.6,145.8 L475.8,145.7 L475.9,145.7 L476.0,145.7 L476.1,145.7 L476.2,145.7 L476.3,145.6 L476.4,145.6 L476.6,145.6 L476.7,145.6 L476.8,145.6 L476.9,145.6 L477.0,145.5 L477.1,145.5 L477.2,145.5 L477.4,145.5 L477.5,145.5 L477.6,145.4 L477.7,145.4 L477.8,145.4 L477.9,145.4 L478.0,145.4 L478.1,145.4 L478.3,145.3 L478.4,145.3 L478.5,145.3 L478.6,145.3 L478.7,145.3 L478.8,145.2 L478.9,145.2 L479.1,145.2 L479.2,145.2 L479.3,145.2 L479.4,145.1 L479.5,145.1 L479.6,145.1 L479.7,145.1 L479.8,145.1 L480.0,145.1 L480.1,145.0 L480.2,145.0 L480.3,145.0 L480.4,145.0 L480.5,145.0 L480.6,144.9 L480.8,144.9 L480.9,144.9 L481.0,144.9 L481.1,144.9 L481.2,144.9 L481.3,144.8 L481.4,144.8 L481.6,144.8 L481.7,144.8 L481.8,144.8 L481.9,144.7 L482.0,144.7 L482.1,144.7 L482.2,144.7 L482.3,144.7 L482.5,144.6 L482.6,144.6 L482.7,144.6 L482.8,144.6 L482.9,144.6 L483.0,144.6 L483.1,144.5 L483.3,144.5 L483.4,144.5 L483.5,144.5 L483.6,144.5 L483.7,144.4 L483.8,144.4 L483.9,144.4 L484.0,144.4 L484.2,144.4 L484.3,144.4 L484.4,144.3 L484.5,144.3 L484.6,144.3 L484.7,144.3 L484.8,144.3 L485.0,144.2 L485.1,144.2 L485.2,144.2 L485.3,144.2 L485.4,144.2 L485.5,144.2 L485.6,144.1 L485.8,144.1 L485.9,144.1 L486.0,144.1 L486.1,144.1 L486.2,144.0 L486.3,144.0 L486.4,144.0 L486.5,144.0 L486.7,144.0 L486.8,143.9 L486.9,143.9 L487.0,143.9 L487.1,143.9 L487.2,143.9 L487.3,143.9 L487.5,143.8 L487.6,143.8 L487.7,143.8 L487.8,143.8 L487.9,143.8 L488.0,143.7 L488.1,143.7 L488.2,143.7 L488.4,143.7 L488.5,143.7 L488.6,143.7 L488.7,143.6 L488.8,143.6 L488.9,143.6 L489.0,143.6 L489.2,143.6 L489.3,143.5 L489.4,143.5 L489.5,143.5 L489.6,143.5 L489.7,143.5 L489.8,143.5 L490.0,143.4 L490.1,143.4 L490.2,143.4 L490.3,143.4 L490.4,143.4 L490.5,143.3 L490.6,143.3 L490.7,143.3 L490.9,143.3 L491.0,143.3 L491.1,143.2 L491.2,143.2 L491.3,143.2 L491.4,143.2 L491.5,143.2 L491.7,143.2 L491.8,143.1 L491.9,143.1 L492.0,143.1 L492.1,143.1 L492.2,143.1 L492.3,143.0 L492.4,143.0 L492.6,143.0 L492.7,143.0 L492.8,143.0 L492.9,143.0 L493.0,142.9 L493.1,142.9 L493.2,142.9 L493.4,142.9 L493.5,142.9 L493.6,142.8 L493.7,142.8 L493.8,142.8 L493.9,142.8 L494.0,142.8 L494.2,142.8 L494.3,142.7 L494.4,142.7 L494.5,142.7 L494.6,142.7 L494.7,142.7 L494.8,142.6 L494.9,142.6 L495.1,142.6 L495.2,142.6 L495.3,142.6 L495.4,142.5 L495.5,142.5 L495.6,142.5 L495.7,142.5 L495.9,142.5 L496.0,142.5 L496.1,142.4 L496.2,142.4 L496.3,142.4 L496.4,142.4 L496.5,142.4 L496.7,142.4 L496.8,142.4 L496.9,142.4 L497.0,142.5 L497.1,142.5 L497.2,142.5 L497.3,142.6 L497.4,142.6 L497.6,142.6 L497.7,142.6 L497.8,142.7 L497.9,142.7 L498.0,142.7 L498.1,142.8 L498.2,142.8 L498.3,142.8 L498.4,142.8 L498.6,142.9 L498.7,142.9 L498.8,142.9 L498.9,143.0 L499.0,143.0 L499.1,143.0 L499.2,143.0 L499.3,143.1 L499.4,143.1 L499.6,143.1 L499.7,143.1 L499.8,143.1 L499.9,143.1 L500.0,143.1 L500.1,143.1 L500.2,143.1 L500.4,143.1 L500.5,143.1 L500.6,143.0 L500.7,143.0 L500.8,143.0 L500.9,143.0 L501.0,143.0 L501.2,142.9 L501.3,142.9 L501.4,142.9 L501.5,142.9 L501.6,142.9 L501.7,142.9 L501.8,142.8 L501.9,142.8 L502.1,142.8 L502.2,142.8 L502.3,142.8 L502.4,142.7 L502.5,142.7 L502.6,142.7 L502.7,142.6 L502.8,142.6 L502.9,142.5 L503.0,142.4 L503.1,142.4 L503.2,142.3 L503.3,142.3 L503.4,142.2 L503.5,142.1 L503.6,142.1 L503.7,142.0 L503.8,141.9 L503.9,141.9 L504.0,141.8 L504.1,141.8 L504.2,141.7 L504.3,141.6 L504.4,141.6 L504.5,141.5 L504.6,141.4 L504.7,141.4 L504.8,141.3 L504.9,141.2 L505.0,141.2 L505.1,141.1 L505.2,141.1 L505.3,141.0 L505.4,141.0 L505.5,140.9 L505.6,140.9 L505.7,140.9 L505.8,140.9 L505.9,140.8 L506.0,140.8 L506.1,140.8 L506.3,140.8 L506.4,140.8 L506.5,140.7 L506.6,140.7 L506.7,140.7 L506.8,140.7 L506.9,140.7 L507.1,140.7 L507.2,140.6 L507.3,140.6 L507.4,140.6 L507.5,140.6 L507.6,140.6 L507.7,140.5 L507.8,140.5 L508.0,140.5 L508.1,140.5 L508.2,140.5 L508.3,140.5 L508.4,140.4 L508.5,140.4 L508.6,140.4 L508.8,140.4 L508.9,140.4 L509.0,140.3 L509.1,140.3 L509.2,140.3 L509.3,140.3 L509.4,140.3 L509.6,140.2 L509.7,140.2 L509.8,140.2 L509.9,140.2 L510.0,140.2 L510.1,140.2 L510.2,140.1 L510.3,140.1 L510.5,140.1 L510.6,140.1 L510.7,140.1 L510.8,140.0 L510.9,140.0 L511.0,140.0 L511.1,140.0 L511.3,140.0 L511.4,140.0 L511.5,139.9 L511.6,139.9 L511.7,139.9 L511.8,139.9 L511.9,139.9 L512.0,139.8 L512.2,139.8 L512.3,139.8 L512.4,139.8 L512.5,139.8 L512.6,139.7 L512.7,139.7 L512.8,139.7 L513.0,139.7 L513.1,139.7 L513.2,139.7 L513.3,139.6 L513.4,139.6 L513.5,139.6 L513.6,139.6 L513.8,139.6 L513.9,139.5 L514.0,139.5 L514.1,139.5 L514.2,139.5 L514.3,139.5 L514.4,139.5 L514.5,139.4 L514.7,139.4 L514.8,139.4 L514.9,139.4 L515.0,139.4 L515.1,139.3 L515.2,139.3 L515.3,139.3 L515.5,139.3 L515.6,139.3 L515.7,139.3 L515.8,139.2 L515.9,139.2 L516.0,139.2 L516.1,139.2 L516.2,139.2 L516.4,139.1 L516.5,139.1 L516.6,139.1 L516.7,139.1 L516.8,139.1 L516.9,139.0 L517.0,139.0 L517.2,139.0 L517.3,139.0 L517.4,139.0 L517.5,139.0 L517.6,138.9 L517.7,138.9 L517.8,138.9 L517.9,138.9 L518.1,138.9 L518.2,138.8 L518.3,138.8 L518.4,138.8 L518.5,138.8 L518.6,138.8 L518.7,138.8 L518.9,138.7 L519.0,138.7 L519.1,138.7 L519.2,138.7 L519.3,138.7 L519.4,138.6 L519.5,138.6 L519.7,138.6 L519.8,138.6 L519.9,138.6 L520.0,138.6 L520.1,138.5 L520.2,138.5 L520.3,138.5 L520.4,138.5 L520.6,138.5 L520.7,138.4 L520.8,138.4 L520.9,138.4 L521.0,138.4 L521.1,138.4 L521.2,138.3 L521.4,138.3 L521.5,138.3 L521.6,138.3 L521.7,138.3 L521.8,138.3 L521.9,138.2 L522.0,138.2 L522.1,138.2 L522.3,138.2 L522.4,138.2 L522.5,138.1 L522.6,138.1 L522.7,138.1 L522.8,138.1 L522.9,138.1 L523.1,138.1 L523.2,138.0 L523.3,138.0 L523.4,138.0 L523.5,138.0 L523.6,138.0 L523.7,137.9 L523.9,137.9 L524.0,137.9 L524.1,137.9 L524.2,137.9 L524.3,137.9 L524.4,137.8 L524.5,137.8 L524.6,137.8 L524.8,137.8 L524.9,137.8 L525.0,137.7 L525.1,137.7 L525.2,137.7 L525.3,137.7 L525.4,137.7 L525.6,137.6 L525.7,137.6 L525.8,137.6 L525.9,137.6 L526.0,137.6 L526.1,137.6 L526.2,137.5 L526.3,137.5 L526.5,137.5 L526.6,137.5 L526.7,137.5 L526.8,137.4 L526.9,137.4 L527.0,137.4 L527.1,137.4 L527.3,137.4 L527.4,137.4 L527.5,137.3 L527.6,137.3 L527.7,137.3 L527.8,137.3 L527.9,137.3 L528.1,137.2 L528.2,137.2 L528.3,137.2 L528.4,137.2 L528.5,137.2 L528.6,137.2 L528.7,137.1 L528.8,137.1 L529.0,137.1 L529.1,137.1 L529.2,137.1 L529.3,137.1 L529.4,137.0 L529.5,137.0 L529.6,137.0 L529.8,137.0 L529.9,137.0 L530.0,137.0 L530.1,137.0 L530.2,137.0 L530.3,137.0 L530.4,137.0 L530.6,136.9 L530.7,136.9 L530.8,136.9 L530.9,136.9 L531.0,136.9 L531.1,136.9 L531.3,136.9 L531.4,136.9 L531.5,136.9 L531.6,136.9 L531.7,136.9 L531.8,136.9 L531.9,136.9 L532.1,136.9 L532.2,136.9 L532.3,136.9 L532.4,136.9 L532.5,136.9 L532.6,136.9 L532.7,136.9 L532.9,136.9 L533.0,136.9 L533.1,136.9 L533.2,136.9 L533.3,136.9 L533.4,136.9 L533.6,136.9 L533.7,136.9 L533.8,136.9 L533.9,137.0 L534.0,137.0 L534.1,137.0 L534.2,137.0 L534.4,137.0 L534.5,137.0 L534.6,137.0 L534.7,137.0 L534.8,137.0 L534.9,137.1 L535.0,137.1 L535.2,137.1 L535.3,137.1 L535.4,137.1 L535.5,137.1 L535.6,137.1 L535.7,137.2 L535.8,137.2 L536.0,137.2 L536.1,137.2 L536.2,137.2 L536.3,137.3 L536.4,137.3 L536.5,137.3 L536.6,137.3 L536.7,137.3 L536.9,137.4 L537.0,137.4 L537.1,137.4 L537.2,137.4 L537.3,137.4 L537.4,137.5 L537.5,137.5 L537.6,137.5 L537.8,137.5 L537.9,137.6 L538.0,137.6 L538.1,137.6 L538.2,137.6 L538.3,137.7 L538.4,137.7 L538.5,137.7 L538.7,137.7 L538.8,137.8 L538.9,137.8 L539.0,137.8 L539.1,137.8 L539.2,137.9 L539.3,137.9 L539.4,137.9 L539.6,137.9 L539.7,138.0 L539.8,138.0 L539.9,138.0 L540.0,138.0 L540.1,138.1 L540.2,138.1 L540.3,138.1 L540.5,138.1 L540.6,138.2 L540.7,138.2 L540.8,138.2 L540.9,138.2 L541.0,138.3 L541.1,138.3 L541.2,138.3 L541.4,138.3 L541.5,138.4 L541.6,138.4 L541.7,138.4 L541.8,138.4 L541.9,138.4 L542.0,138.5 L542.1,138.5 L542.3,138.5 L542.4,138.5 L542.5,138.6 L542.6,138.6 L542.7,138.6 L542.8,138.6 L542.9,138.7 L543.0,138.7 L543.2,138.7 L543.3,138.7 L543.4,138.8 L543.5,138.8 L543.6,138.8 L543.7,138.8 L543.8,138.9 L543.9,138.9 L544.0,138.9 L544.2,138.9 L544.3,139.0 L544.4,139.0 L544.5,139.0 L544.6,139.0 L544.7,139.1 L544.8,139.1 L544.9,139.1 L545.1,139.1 L545.2,139.2 L545.3,139.2 L545.4,139.2 L545.5,139.2 L545.6,139.3 L545.7,139.3 L545.8,139.3 L546.0,139.3 L546.1,139.3 L546.2,139.4 L546.3,139.4 L546.4,139.4 L546.5,139.4 L546.6,139.5 L546.7,139.5 L546.9,139.5 L547.0,139.5 L547.1,139.6 L547.2,139.6 L547.3,139.6 L547.4,139.6 L547.5,139.7 L547.6,139.7 L547.8,139.7 L547.9,139.7 L548.0,139.8 L548.1,139.8 L548.2,139.8 L548.3,139.8 L548.4,139.9 L548.5,139.9 L548.7,139.9 L548.8,139.9 L548.9,140.0 L549.0,140.0 L549.1,140.0 L549.2,140.0 L549.3,140.1 L549.4,140.1 L549.6,140.1 L549.7,140.1 L549.8,140.2 L549.9,140.2 L550.0,140.2 L550.1,140.2 L550.2,140.3 L550.3,140.3 L550.5,140.3 L550.6,140.3 L550.7,140.3 L550.8,140.4 L550.9,140.4 L551.0,140.4 L551.1,140.4 L551.2,140.5 L551.4,140.5 L551.5,140.5 L551.6,140.5 L551.7,140.6 L551.8,140.6 L551.9,140.6 L552.0,140.6 L552.1,140.7 L552.3,140.7 L552.4,140.7 L552.5,140.7 L552.6,140.8 L552.7,140.8 L552.8,140.8 L552.9,140.8 L553.0,140.9 L553.2,140.9 L553.3,140.9 L553.4,140.9 L553.5,141.0 L553.6,141.0 L553.7,141.0 L553.8,141.0 L553.9,141.1 L554.1,141.1 L554.2,141.1 L554.3,141.1 L554.4,141.2 L554.5,141.2 L554.6,141.2 L554.7,141.2 L554.8,141.2 L555.0,141.3 L555.1,141.3 L555.2,141.3 L555.3,141.3 L555.4,141.4 L555.5,141.4 L555.6,141.4 L555.7,141.4 L555.8,141.5 L556.0,141.5 L556.1,141.5 L556.2,141.5 L556.3,141.6 L556.4,141.6 L556.5,141.6 L556.6,141.6 L556.7,141.7 L556.9,141.7 L557.0,141.7 L557.1,141.7 L557.2,141.8 L557.3,141.8 L557.4,141.8 L557.5,141.8 L557.6,141.9 L557.8,141.9 L557.9,141.9 L558.0,141.9 L558.1,142.0 L558.2,142.0 L558.3,142.0 L558.4,142.0 L558.5,142.1 L558.7,142.1 L558.8,142.1 L558.9,142.1 L559.0,142.2 L559.1,142.2 L559.2,142.2 L559.3,142.2 L559.4,142.2 L559.6,142.3 L559.7,142.3 L559.8,142.3 L559.9,142.3 L560.0,142.4 L560.1,142.4 L560.2,142.4 L560.3,142.4 L560.5,142.5 L560.6,142.5 L560.7,142.5 L560.8,142.5 L560.9,142.6 L561.0,142.6 L561.1,142.6 L561.2,142.6 L561.4,142.7 L561.5,142.7 L561.6,142.7 L561.7,142.7 L561.8,142.8 L561.9,142.8 L562.0,142.8 L562.1,142.8 L562.3,142.9 L562.4,142.9 L562.5,142.9 L562.6,142.9 L562.7,143.0 L562.8,143.0 L562.9,143.0 L563.0,143.0 L563.2,143.1 L563.3,143.1 L563.4,143.1 L563.5,143.1 L563.6,143.1 L563.7,143.2 L563.8,143.2 L563.9,143.2 L564.1,143.2 L564.2,143.3 L564.3,143.3 L564.4,143.3 L564.5,143.3 L564.6,143.4 L564.7,143.4 L564.8,143.4 L565.0,143.4 L565.1,143.5 L565.2,143.5 L565.3,143.5 L565.4,143.5 L565.5,143.6 L565.6,143.6 L565.7,143.6 L565.9,143.6 L566.0,143.7 L566.1,143.7 L566.2,143.7 L566.3,143.7 L566.4,143.8 L566.5,143.8 L566.6,143.8 L566.8,143.8 L566.9,143.9 L567.0,143.9 L567.1,143.9 L567.2,143.9 L567.3,144.0 L567.4,144.0 L567.5,144.0 L567.6,144.0 L567.8,144.1 L567.9,144.1 L568.0,144.1 L568.1,144.1 L568.2,144.1 L568.3,144.2 L568.4,144.2 L568.5,144.2 L568.7,144.2 L568.8,144.3 L568.9,144.3 L569.0,144.3 L569.1,144.3 L569.2,144.4 L569.3,144.4 L569.4,144.4 L569.6,144.4 L569.7,144.5 L569.8,144.5 L569.9,144.5 L570.0,144.5 L570.1,144.6 L570.2,144.6 L570.3,144.6 L570.5,144.6 L570.6,144.7 L570.7,144.7 L570.8,144.7 L570.9,144.7 L571.0,144.8 L571.1,144.8 L571.2,144.8 L571.4,144.8 L571.5,144.9 L571.6,144.9 L571.7,144.9 L571.8,144.9 L571.9,145.0 L572.0,145.0 L572.1,145.0 L572.3,145.0 L572.4,145.0 L572.5,145.1 L572.6,145.1 L572.7,145.1 L572.8,145.1 L572.9,145.2 L573.0,145.2 L573.2,145.2 L573.3,145.2 L573.4,145.3 L573.5,145.3 L573.6,145.3 L573.7,145.3 L573.8,145.4 L573.9,145.4 L574.1,145.4 L574.2,145.4 L574.3,145.5 L574.4,145.5 L574.5,145.5 L574.6,145.5 L574.7,145.6 L574.8,145.6 L575.0,145.6 L575.1,145.6 L575.2,145.7 L575.3,145.7 L575.4,145.7 L575.5,145.7 L575.6,145.8 L575.7,145.8 L575.9,145.8 L576.0,145.8 L576.1,145.9 L576.2,145.9 L576.3,145.9 L576.4,145.9 L576.5,146.0 L576.6,146.0 L576.8,146.0 L576.9,146.0 L577.0,146.0 L577.1,146.1 L577.2,146.1 L577.3,146.1 L577.4,146.1 L577.5,146.2 L577.7,146.2 L577.8,146.2 L577.9,146.2 L578.0,146.3 L578.1,146.3 L578.2,146.3 L578.3,146.3 L578.4,146.4 L578.6,146.4 L578.7,146.4 L578.8,146.4 L578.9,146.5 L579.0,146.5 L579.1,146.5 L579.2,146.5 L579.3,146.6 L579.5,146.6 L579.6,146.6 L579.7,146.6 L579.8,146.7 L579.9,146.7 L580.0,146.7 L580.1,146.7 L580.2,146.8 L580.3,146.8 L580.5,146.8 L580.6,146.8 L580.7,146.9 L580.8,146.9 L580.9,146.9 L581.0,146.9 L581.1,146.9 L581.2,147.0 L581.4,147.0 L581.5,147.0 L581.6,147.0 L581.7,147.1 L581.8,147.1 L581.9,147.1 L582.0,147.1 L582.1,147.2 L582.3,147.2 L582.4,147.2 L582.5,147.2 L582.6,147.3 L582.7,147.3 L582.8,147.3 L582.9,147.3 L583.0,147.4 L583.2,147.4 L583.3,147.4 L583.4,147.4 L583.5,147.5 L583.6,147.5 L583.7,147.5 L583.8,147.5 L583.9,147.6 L584.1,147.6 L584.2,147.6 L584.3,147.6 L584.4,147.7 L584.5,147.7 L584.6,147.7 L584.7,147.7 L584.8,147.8 L585.0,147.8 L585.1,147.8 L585.2,147.8 L585.3,147.9 L585.4,147.9 L585.5,147.9 L585.6,147.9 L585.7,147.9 L585.9,148.0 L586.0,148.0 L586.1,148.0 L586.2,148.0 L586.3,148.1 L586.4,148.1 L586.5,148.1 L586.6,148.1 L586.8,148.2 L586.9,148.2 L587.0,148.2 L587.1,148.2 L587.2,148.3 L587.3,148.3 L587.4,148.3 L587.5,148.3 L587.7,148.4 L587.8,148.4 L587.9,148.4 L588.0,148.4 L588.1,148.5 L588.2,148.5 L588.3,148.5 L588.4,148.5 L588.6,148.6 L588.7,148.6 L588.8,148.6 L588.9,148.6 L589.0,148.7 L589.1,148.7 L589.2,148.7 L589.3,148.7 L589.5,148.8 L589.6,148.8 L589.7,148.8 L589.8,148.8 L589.9,148.8 L590.0,148.9 L590.1,148.9 L590.2,148.9 L590.4,148.9 L590.5,149.0 L590.6,149.0 L590.7,149.0 L590.8,149.0 L590.9,149.1 L591.0,149.1 L591.1,149.1 L591.3,149.1 L591.4,149.2 L591.5,149.2 L591.6,149.2 L591.7,149.2 L591.8,149.3 L591.9,149.3 L592.0,149.3 L592.1,149.3 L592.3,149.4 L592.4,149.4 L592.5,149.4 L592.6,149.4 L592.7,149.5 L592.8,149.5 L592.9,149.5 L593.0,149.5 L593.2,149.6 L593.3,149.6 L593.4,149.6 L593.5,149.6 L593.6,149.7 L593.7,149.7 L593.8,149.7 L593.9,149.7 L594.1,149.8 L594.2,149.8 L594.3,149.8 L594.4,149.8 L594.5,149.8 L594.6,149.9 L594.7,149.9 L594.8,149.9 L595.0,149.9 L595.1,150.0 L595.2,150.0 L595.3,150.0 L595.4,150.0 L595.5,150.1 L595.6,150.1 L595.7,150.1 L595.9,150.1 L596.0,150.2 L596.1,150.2 L596.2,150.2 L596.3,150.2 L596.4,150.3 L596.5,150.3 L596.6,150.3 L596.8,150.3 L596.9,150.4 L597.0,150.4 L597.1,150.4 L597.2,150.4 L597.3,150.5 L597.4,150.5 L597.5,150.5 L597.7,150.5 L597.8,150.6 L597.9,150.6 L598.0,150.6 L598.1,150.6 L598.2,150.7 L598.3,150.7 L598.4,150.7 L598.6,150.7 L598.7,150.7 L598.8,150.8 L598.9,150.8 L599.0,150.8 L599.1,150.8 L599.2,150.9 L599.3,150.9 L599.5,150.9 L599.6,150.9 L599.7,151.0 L599.8,151.0 L599.9,151.0 L600.0,151.0 L600.1,151.1 L600.2,151.1 L600.4,151.1 L600.5,151.1 L600.6,151.2 L600.7,151.2 L600.8,151.2 L600.9,151.2 L601.0,151.3 L601.1,151.3 L601.3,151.3 L601.4,151.3 L601.5,151.4 L601.6,151.4 L601.7,151.4 L601.8,151.4 L601.9,151.5 L602.0,151.5 L602.2,151.5 L602.3,151.5 L602.4,151.6 L602.5,151.6 L602.6,151.6 L602.7,151.6 L602.8,151.7 L602.9,151.7 L603.1,151.7 L603.2,151.7 L603.3,151.7 L603.4,151.8 L603.5,151.8 L603.6,151.8 L603.7,151.8 L603.8,151.9 L603.9,151.9 L604.1,151.9 L604.2,151.9 L604.3,152.0 L604.4,152.0 L604.5,152.0 L604.6,152.0 L604.7,152.1 L604.8,152.1 L605.0,152.1 L605.1,152.1 L605.2,152.2 L605.3,152.2 L605.4,152.2 L605.5,152.2 L605.6,152.3 L605.7,152.3 L605.9,152.3 L606.0,152.3 L606.1,152.4 L606.2,152.4 L606.3,152.4 L606.4,152.4 L606.5,152.5 L606.6,152.5 L606.8,152.5 L606.9,152.5 L607.0,152.6 L607.1,152.6 L607.2,152.6 L607.3,152.6 L607.4,152.6 L607.5,152.7 L607.7,152.7 L607.8,152.7 L607.9,152.7 L608.0,152.8 L608.1,152.8 L608.2,152.8 L608.3,152.8 L608.4,152.9 L608.6,152.9 L608.7,152.9 L608.8,152.9 L608.9,153.0 L609.0,153.0 L609.1,153.0 L609.2,153.0 L609.3,153.1 L609.5,153.1 L609.6,153.1 L609.7,153.1 L609.8,153.2 L609.9,153.2 L610.0,153.2 L610.1,153.2 L610.2,153.3 L610.4,153.3 L610.5,153.3 L610.6,153.3 L610.7,153.4 L610.8,153.4 L610.9,153.4 L611.0,153.4 L611.1,153.5 L611.3,153.5 L611.4,153.5 L611.5,153.5 L611.6,153.6 L611.7,153.6 L611.8,153.6 L611.9,153.6 L612.0,153.6 L612.2,153.7 L612.3,153.7 L612.4,153.7 L612.5,153.7 L612.6,153.8 L612.7,153.8 L612.8,153.8 L612.9,153.8 L613.1,153.9 L613.2,153.9 L613.3,153.9 L613.4,153.9 L613.5,154.0 L613.6,154.0 L613.7,154.0 L613.8,154.0 L614.0,154.1 L614.1,154.1 L614.2,154.1 L614.3,154.1 L614.4,154.2 L614.5,154.2 L614.6,154.2 L614.7,154.2 L614.9,154.3 L615.0,154.3 L615.1,154.3 L615.2,154.3 L615.3,154.4 L615.4,154.4 L615.5,154.4 L615.6,154.4 L615.7,154.5 L615.9,154.5 L616.0,154.5 L616.1,154.5 L616.2,154.5 L616.3,154.6 L616.4,154.6 L616.5,154.6 L616.6,154.6 L616.8,154.7 L616.9,154.7 L617.0,154.7 L617.1,154.7 L617.2,154.8 L617.3,154.8 L617.4,154.8 L617.5,154.8 L617.7,154.9 L617.8,154.9 L617.9,154.9 L618.0,154.9 L618.1,155.0 L618.2,155.0 L618.3,155.0 L618.4,155.0 L618.6,155.1 L618.7,155.1 L618.8,155.1 L618.9,155.1 L619.0,155.2 L619.1,155.2 L619.2,155.2 L619.3,155.2 L619.5,155.3 L619.6,155.3 L619.7,155.3 L619.8,155.3 L619.9,155.4 L620.0,155.4 L620.1,155.4 L620.2,155.4 L620.4,155.5 L620.5,155.5 L620.6,155.5 L620.7,155.5 L620.8,155.5 L620.9,155.6 L621.0,155.6 L621.1,155.6 L621.3,155.6 L621.4,155.7 L621.5,155.7 L621.6,155.7 L621.7,155.7 L621.8,155.8 L621.9,155.8 L622.0,155.8 L622.2,155.8 L622.3,155.9 L622.4,155.9 L622.5,155.9 L622.6,155.9 L622.7,156.0 L622.8,156.0 L622.9,156.0 L623.1,156.0 L623.2,156.1 L623.3,156.1 L623.4,156.1 L623.5,156.1 L623.6,156.2 L623.7,156.2 L623.8,156.2 L624.0,156.2 L624.1,156.3 L624.2,156.3 L624.3,156.3 L624.4,156.3 L624.5,156.4 L624.6,156.4 L624.7,156.4 L624.9,156.4 L625.0,156.4 L625.1,156.5 L625.2,156.5 L625.3,156.5 L625.4,156.5 L625.5,156.6 L625.6,156.6 L625.8,156.6 L625.9,156.6 L626.0,156.7 L626.1,156.7 L626.2,156.7 L626.3,156.7 L626.4,156.8 L626.5,156.8 L626.7,156.8 L626.8,156.8 L626.9,156.9 L627.0,156.9 L627.1,156.9 L627.2,156.9 L627.3,157.0 L627.4,157.0 L627.5,157.0 L627.7,157.0 L627.8,157.1 L627.9,157.1 L628.0,157.1 L628.1,157.1 L628.2,157.2 L628.3,157.2 L628.4,157.2 L628.6,157.2 L628.7,157.3 L628.8,157.3 L628.9,157.3 L629.0,157.3 L629.1,157.3 L629.2,157.4 L629.3,157.4 L629.5,157.4 L629.6,157.4 L629.7,157.5 L629.8,157.5 L629.9,157.5 L630.0,157.5 L630.1,157.6 L630.2,157.6 L630.4,157.6 L630.5,157.6 L630.6,157.7 L630.7,157.7 L630.8,157.7 L630.9,157.7 L631.0,157.8 L631.1,157.8 L631.3,157.8 L631.4,157.8 L631.5,157.9 L631.6,157.9 L631.7,157.9 L631.8,157.9 L631.9,158.0 L632.0,158.0 L632.2,158.0 L632.3,158.0 L632.4,158.1 L632.5,158.1 L632.6,158.1 L632.7,158.1 L632.8,158.2 L632.9,158.2 L633.1,158.2 L633.2,158.2 L633.3,158.3 L633.4,158.3 L633.5,158.3 L633.6,158.3 L633.7,158.3 L633.8,158.4 L634.0,158.4 L634.1,158.4 L634.2,158.4 L634.3,158.5 L634.4,158.5 L634.5,158.5 L634.6,158.5 L634.7,158.6 L634.9,158.6 L635.0,158.6 L635.1,158.6 L635.2,158.7 L635.3,158.7 L635.4,158.7 L635.5,158.7 L635.6,158.8 L635.8,158.8 L635.9,158.8 L636.0,158.8 L636.1,158.9 L636.2,158.9 L636.3,158.9 L636.4,158.9 L636.5,159.0 L636.7,159.0 L636.8,159.0 L636.9,159.0 L637.0,159.1 L637.1,159.1 L637.2,159.1 L637.3,159.1 L637.4,159.2 L637.6,159.2 L637.7,159.2 L637.8,159.2 L637.9,159.2 L638.0,159.3 L638.1,159.3 L638.2,159.3 L638.3,159.3 L638.5,159.4 L638.6,159.4 L638.7,159.4 L638.8,159.4 L638.9,159.5 L639.0,159.5 L639.1,159.5 L639.2,159.5 L639.4,159.6 L639.5,159.6 L639.6,159.6 L639.7,159.6 L639.8,159.7 L639.9,159.7 L640.0,159.7 L640.1,159.7 L640.2,159.8 L640.4,159.8 L640.5,159.8 L640.6,159.8 L640.7,159.9 L640.8,159.9 L640.9,159.9 L641.0,159.9 L641.1,160.0 L641.3,160.0 L641.4,160.0 L641.5,160.0 L641.6,160.1 L641.7,160.1 L641.8,160.1 L641.9,160.1 L642.0,160.2 L642.2,160.2 L642.3,160.2 L642.4,160.2 L642.5,160.2 L642.6,160.3 L642.7,160.3 L642.8,160.3 L643.0,160.3 L643.1,160.3 L643.2,160.3 L643.3,160.3 L643.4,160.3 L643.5,160.2 L643.6,160.2 L643.7,160.2 L643.9,160.2 L644.0,160.1 L644.1,160.1 L644.2,160.0 L644.3,160.0 L644.4,159.9 L644.5,159.8 L644.6,159.8 L644.7,159.7 L644.7,159.6 L644.8,159.6 L644.9,159.5 L645.0,159.4 L645.1,159.3 L645.1,159.2 L645.2,159.1 L645.3,159.0 L645.3,158.9 L645.4,158.8 L645.4,158.7 L645.5,158.6 L645.5,158.5 L645.5,158.4 L645.6,158.3 L645.6,158.2 L645.6,158.1 L645.6,157.9 L645.6,157.8 L645.7,157.7 L645.7,157.6 L645.7,157.5 L645.6,157.4 L645.6,157.3 L645.6,157.1 L645.6,157.0 L645.6,156.9 L645.5,156.8 L645.5,156.7 L645.4,156.6 L645.4,156.5 L645.3,156.4 L645.3,156.3 L645.2,156.2 L645.2,156.1 L645.1,156.0 L645.0,155.9 L644.9,155.8 L644.9,155.7 L644.8,155.6 L644.7,155.6 L644.7,155.5 L644.6,155.4 L644.5,155.3 L644.4,155.2 L644.4,155.1 L644.3,155.0 L644.2,154.9 L644.2,154.8 L644.1,154.7 L644.0,154.6 L643.9,154.6 L643.9,154.5 L643.8,154.4 L643.7,154.3 L643.7,154.2 L643.6,154.1 L643.5,154.0 L643.4,153.9 L643.4,153.8 L643.3,153.7 L643.2,153.7 L643.2,153.6 L643.1,153.5 L643.0,153.4 L642.9,153.3 L642.9,153.2 L642.8,153.1 L642.7,153.0 L642.7,152.9 L642.6,152.8 L642.5,152.8 L642.4,152.7 L642.4,152.6 L642.3,152.5 L642.2,152.4 L642.2,152.3 L642.1,152.2 L642.0,152.1 L641.9,152.0 L641.9,151.9 L641.8,151.9 L641.7,151.8 L641.7,151.7 L641.6,151.6 L641.5,151.5 L641.4,151.4 L641.4,151.3 L641.3,151.2 L641.2,151.1 L641.2,151.0 L641.1,151.0 L641.0,150.9 L640.9,150.8 L640.9,150.7 L640.8,150.6 L640.7,150.5 L640.7,150.4 L640.6,150.3 L640.5,150.2 L640.4,150.1 L640.4,150.1 L640.3,150.0 L640.2,149.9 L640.2,149.8 L640.1,149.7 L640.0,149.6 L639.9,149.5 L639.9,149.4 L639.8,149.3 L639.7,149.2 L639.7,149.2 L639.6,149.1 L639.5,149.0 L639.4,148.9 L639.4,148.8 L639.3,148.7 L639.2,148.6 L639.2,148.5 L639.1,148.4 L639.0,148.3 L638.9,148.3 L638.9,148.2 L638.8,148.1 L638.7,148.0 L638.7,147.9 L638.6,147.8 L638.5,147.7 L638.4,147.6 L638.4,147.5 L638.3,147.4 L638.2,147.4 L638.2,147.3 L638.1,147.2 L638.0,147.1 L637.9,147.0 L637.9,146.9 L637.8,146.8 L637.7,146.7 L637.7,146.6 L637.6,146.5 L637.5,146.5 L637.4,146.4 L637.4,146.3 L637.3,146.2 L637.2,146.1 L637.2,146.0 L637.1,145.9 L637.0,145.8 L636.9,145.7 L636.9,145.6 L636.8,145.6 L636.7,145.5 L636.7,145.4 L636.6,145.3 L636.5,145.2 L636.4,145.1 L636.4,145.0 L636.3,144.9 L636.2,144.8 L636.2,144.7 L636.1,144.7 L636.0,144.6 L635.9,144.5 L635.9,144.4 L635.8,144.3 L635.7,144.2 L635.7,144.1 L635.6,144.0 L635.5,143.9 L635.4,143.8 L635.4,143.8 L635.3,143.7 L635.2,143.6 L635.2,143.5 L635.1,143.4 L635.0,143.3 L634.9,143.2 L634.9,143.1 L634.8,143.0 L634.7,142.9 L634.7,142.9 L634.6,142.8 L634.5,142.7 L634.4,142.6 L634.4,142.5 L634.3,142.4 L634.2,142.3 L634.2,142.2 L634.1,142.1 L634.0,142.0 L633.9,141.9 L633.9,141.9 L633.8,141.8 L633.7,141.7 L633.7,141.6 L633.6,141.5 L633.5,141.4 L633.4,141.3 L633.4,141.2 L633.3,141.1 L633.2,141.0 L633.2,141.0 L633.1,140.9 L633.0,140.8 L632.9,140.7 L632.9,140.6 L632.8,140.5 L632.7,140.4 L632.7,140.3 L632.6,140.2 L632.5,140.1 L632.4,140.1 L632.4,140.0 L632.3,139.9 L632.2,139.8 L632.2,139.7 L632.1,139.6 L632.0,139.5 L631.9,139.4 L631.9,139.3 L631.8,139.2 L631.7,139.2 L631.7,139.1 L631.6,139.0 L631.5,138.9 L631.4,138.8 L631.4,138.7 L631.3,138.6 L631.2,138.5 L631.2,138.4 L631.1,138.3 L631.0,138.3 L630.9,138.2 L630.9,138.1 L630.8,138.0 L630.7,137.9 L630.7,137.8 L630.6,137.7 L630.5,137.6 L630.4,137.5 L630.4,137.4 L630.3,137.4 L630.2,137.3 L630.2,137.2 L630.1,137.1 L630.0,137.0 L629.9,136.9 L629.9,136.8 L629.8,136.7 L629.7,136.6 L629.7,136.5 L629.6,136.5 L629.5,136.4 L629.4,136.3 L629.4,136.2 L629.3,136.1 L629.2,136.0 L629.2,135.9 L629.1,135.8 L629.0,135.7 L628.9,135.6 L628.9,135.6 L628.8,135.5 L628.7,135.4 L628.7,135.3 L628.6,135.2 L628.5,135.1 L628.4,135.0 L628.4,134.9 L628.3,134.8 L628.2,134.7 L628.1,134.7 L628.1,134.6 L628.0,134.5 L627.9,134.4 L627.9,134.3 L627.8,134.2 L627.7,134.1 L627.6,134.0 L627.6,133.9 L627.5,133.8 L627.4,133.8 L627.4,133.7 L627.3,133.6 L627.2,133.5 L627.1,133.4 L627.1,133.3 L627.0,133.2 L626.9,133.1 L626.9,133.0 L626.8,132.9 L626.7,132.9 L626.6,132.8 L626.6,132.7 L626.5,132.6 L626.4,132.5 L626.4,132.4 L626.3,132.3 L626.2,132.2 L626.1,132.1 L626.1,132.0 L626.0,132.0 L625.9,131.9 L625.9,131.8 L625.8,131.7 L625.7,131.6 L625.6,131.5 L625.6,131.4 L625.5,131.3 L625.4,131.2 L625.4,131.1 L625.3,131.1 L625.2,131.0 L625.1,130.9 L625.1,130.8 L625.0,130.7 L624.9,130.6 L624.9,130.5 L624.8,130.4 L624.7,130.3 L624.6,130.2 L624.6,130.2 L624.5,130.1 L624.4,130.0 L624.4,129.9 L624.3,129.8 L624.2,129.7 L624.1,129.6 L624.1,129.5 L624.0,129.4 L623.9,129.3 L623.9,129.2 L623.8,129.2 L623.7,129.1 L623.6,129.0 L623.6,128.9 L623.5,128.8 L623.4,128.7 L623.4,128.6 L623.3,128.5 L623.2,128.4 L623.1,128.3 L623.1,128.3 L623.0,128.2 L622.9,128.1 L622.9,128.0 L622.8,127.9 L622.7,127.8 L622.6,127.7 L622.6,127.6 L622.5,127.5 L622.4,127.4 L622.4,127.4 L622.3,127.3 L622.2,127.2 L622.1,127.1 L622.1,127.0 L622.0,126.9 L621.9,126.8 L621.9,126.7 L621.8,126.6 L621.7,126.5 L621.6,126.5 L621.6,126.4 L621.5,126.3 L621.4,126.2 L621.4,126.1 L621.3,126.0 L621.2,125.9 L621.1,125.8 L621.1,125.7 L621.0,125.6 L620.9,125.6 L620.9,125.5 L620.8,125.4 L620.7,125.3 L620.6,125.2 L620.6,125.1 L620.5,125.0 L620.4,124.9 L620.4,124.8 L620.3,124.7 L620.2,124.7 L620.1,124.6 L620.1,124.5 L620.0,124.4 L619.9,124.3 L619.9,124.2 L619.8,124.1 L619.7,124.0 L619.6,123.9 L619.6,123.8 L619.5,123.8 L619.4,123.7 L619.4,123.6 L619.3,123.5 L619.2,123.4 L619.1,123.3 L619.1,123.2 L619.0,123.1 L618.9,123.0 L618.9,122.9 L618.8,122.9 L618.7,122.8 L618.6,122.7 L618.6,122.6 L618.5,122.5 L618.4,122.4 L618.4,122.3 L618.3,122.2 L618.2,122.1 L618.1,122.0 L618.1,122.0 L618.0,121.9 L617.9,121.8 L617.9,121.7 L617.8,121.6 L617.7,121.5 L617.6,121.4 L617.6,121.3 L617.5,121.2 L617.4,121.1 L617.4,121.1 L617.3,121.0 L617.2,120.9 L617.1,120.8 L617.1,120.7 L617.0,120.6 L616.9,120.5 L616.9,120.4 L616.8,120.3 L616.7,120.2 L616.6,120.2 L616.6,120.1 L616.5,120.0 L616.4,119.9 L616.4,119.8 L616.3,119.7 L616.2,119.6 L616.1,119.5 L616.1,119.4 L616.0,119.3 L615.9,119.3 L615.9,119.2 L615.8,119.1 L615.7,119.0 L615.6,118.9 L615.6,118.8 L615.5,118.7 L615.4,118.6 L615.4,118.5 L615.3,118.4 L615.2,118.4 L615.1,118.3 L615.1,118.2 L615.0,118.1 L614.9,118.0 L614.9,117.9 L614.8,117.8 L614.7,117.7 L614.6,117.6 L614.6,117.5 L614.5,117.5 L614.4,117.4 L614.4,117.3 L614.3,117.2 L614.2,117.1 L614.1,117.0 L614.1,116.9 L614.0,116.8 L613.9,116.7 L613.9,116.6 L613.8,116.5 L613.7,116.5 L613.6,116.4 L613.6,116.3 L613.5,116.2 L613.4,116.1 L613.4,116.0 L613.3,115.9 L613.2,115.8 L613.1,115.7 L613.1,115.6 L613.0,115.6 L612.9,115.5 L612.9,115.4 L612.8,115.3 L612.7,115.2 L612.6,115.1 L612.6,115.0 L612.5,114.9 L612.4,114.8 L612.4,114.7 L612.3,114.7 L612.2,114.6 L612.1,114.5 L612.1,114.4 L612.0,114.3 L611.9,114.2 L611.9,114.1 L611.8,114.0 L611.7,113.9 L611.6,113.8 L611.6,113.8 Z" fill="none" stroke="#5f6673" stroke-width="3" stroke-linejoin="round"/><circle cx="611.5" cy="113.7" r="3.4" fill="#1d2330" stroke="#ffffff" stroke-width="1"/><text x="557.5" y="190.8" font-family="DejaVu Sans" font-size="9.5" fill="#5f6673" text-anchor="middle" font-weight="normal">4.5 km · road 9.0–10.0 m</text><rect x="0.0" y="210.82337120641645" width="327.5" height="290.2675653070664" rx="3" fill="#f5f6f8"/><text x="163.8" y="226.8" font-family="DejaVu Sans" font-size="11" fill="#1d2330" text-anchor="middle" font-weight="bold">Extra hard</text><text x="163.8" y="239.8" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle" font-weight="normal">Austin, a real training circuit</text><path d="M71.6,370.3 L71.6,370.4 L71.7,370.4 L71.8,370.5 L71.9,370.6 L72.0,370.6 L72.1,370.7 L72.2,370.8 L72.3,370.9 L72.4,370.9 L72.5,371.0 L72.6,371.1 L72.7,371.1 L72.7,371.2 L72.8,371.3 L72.9,371.3 L73.0,371.4 L73.1,371.5 L73.2,371.6 L73.3,371.6 L73.4,371.7 L73.5,371.8 L73.6,371.8 L73.7,371.9 L73.7,372.0 L73.8,372.0 L73.9,372.1 L74.0,372.2 L74.1,372.3 L74.2,372.3 L74.3,372.4 L74.4,372.5 L74.5,372.5 L74.6,372.6 L74.7,372.7 L74.8,372.7 L74.8,372.8 L74.9,372.9 L75.0,373.0 L75.1,373.0 L75.2,373.1 L75.3,373.2 L75.4,373.2 L75.5,373.3 L75.6,373.4 L75.7,373.4 L75.8,373.5 L75.9,373.6 L75.9,373.6 L76.0,373.7 L76.1,373.8 L76.2,373.9 L76.3,373.9 L76.4,374.0 L76.5,374.1 L76.6,374.1 L76.7,374.2 L76.8,374.3 L76.9,374.3 L76.9,374.4 L77.0,374.5 L77.1,374.6 L77.2,374.6 L77.3,374.7 L77.4,374.8 L77.5,374.8 L77.6,374.9 L77.7,375.0 L77.8,375.0 L77.9,375.1 L78.0,375.2 L78.0,375.3 L78.1,375.3 L78.2,375.4 L78.3,375.5 L78.4,375.5 L78.5,375.6 L78.6,375.7 L78.7,375.7 L78.8,375.8 L78.9,375.9 L79.0,376.0 L79.0,376.0 L79.1,376.1 L79.2,376.2 L79.3,376.2 L79.4,376.3 L79.5,376.4 L79.6,376.4 L79.7,376.5 L79.8,376.6 L79.9,376.6 L80.0,376.7 L80.1,376.8 L80.1,376.9 L80.2,376.9 L80.3,377.0 L80.4,377.1 L80.5,377.1 L80.6,377.2 L80.7,377.3 L80.8,377.3 L80.9,377.4 L81.0,377.5 L81.1,377.6 L81.2,377.6 L81.2,377.7 L81.3,377.8 L81.4,377.8 L81.5,377.9 L81.6,378.0 L81.7,378.0 L81.8,378.1 L81.9,378.2 L82.0,378.3 L82.1,378.3 L82.2,378.4 L82.2,378.5 L82.3,378.5 L82.4,378.6 L82.5,378.7 L82.6,378.7 L82.7,378.8 L82.8,378.9 L82.9,379.0 L83.0,379.0 L83.1,379.1 L83.2,379.2 L83.3,379.2 L83.3,379.3 L83.4,379.4 L83.5,379.4 L83.6,379.5 L83.7,379.6 L83.8,379.7 L83.9,379.7 L84.0,379.8 L84.1,379.9 L84.2,379.9 L84.3,380.0 L84.4,380.1 L84.4,380.1 L84.5,380.2 L84.6,380.3 L84.7,380.3 L84.8,380.4 L84.9,380.5 L85.0,380.6 L85.1,380.6 L85.2,380.7 L85.3,380.8 L85.4,380.8 L85.4,380.9 L85.5,381.0 L85.6,381.0 L85.7,381.1 L85.8,381.2 L85.9,381.3 L86.0,381.3 L86.1,381.4 L86.2,381.5 L86.3,381.5 L86.4,381.6 L86.5,381.7 L86.5,381.7 L86.6,381.8 L86.7,381.9 L86.8,382.0 L86.9,382.0 L87.0,382.1 L87.1,382.2 L87.2,382.2 L87.3,382.3 L87.4,382.4 L87.5,382.4 L87.5,382.5 L87.6,382.6 L87.7,382.7 L87.8,382.7 L87.9,382.8 L88.0,382.9 L88.1,382.9 L88.2,383.0 L88.3,383.1 L88.4,383.1 L88.5,383.2 L88.6,383.3 L88.6,383.3 L88.7,383.4 L88.8,383.5 L88.9,383.6 L89.0,383.6 L89.1,383.7 L89.2,383.8 L89.3,383.8 L89.4,383.9 L89.5,384.0 L89.6,384.0 L89.7,384.1 L89.7,384.2 L89.8,384.3 L89.9,384.3 L90.0,384.4 L90.1,384.5 L90.2,384.5 L90.3,384.6 L90.4,384.7 L90.5,384.7 L90.6,384.8 L90.7,384.9 L90.7,385.0 L90.8,385.0 L90.9,385.1 L91.0,385.2 L91.1,385.2 L91.2,385.3 L91.3,385.4 L91.4,385.4 L91.5,385.5 L91.6,385.6 L91.7,385.7 L91.8,385.7 L91.8,385.8 L91.9,385.9 L92.0,385.9 L92.1,386.0 L92.2,386.1 L92.3,386.1 L92.4,386.2 L92.5,386.3 L92.6,386.3 L92.7,386.4 L92.8,386.5 L92.9,386.6 L92.9,386.6 L93.0,386.7 L93.1,386.8 L93.2,386.8 L93.3,386.9 L93.4,387.0 L93.5,387.0 L93.6,387.1 L93.7,387.2 L93.8,387.3 L93.9,387.3 L93.9,387.4 L94.0,387.5 L94.1,387.5 L94.2,387.6 L94.3,387.7 L94.4,387.7 L94.5,387.8 L94.6,387.9 L94.7,388.0 L94.8,388.0 L94.9,388.1 L95.0,388.2 L95.0,388.2 L95.1,388.3 L95.2,388.4 L95.3,388.4 L95.4,388.5 L95.5,388.6 L95.6,388.7 L95.7,388.7 L95.8,388.8 L95.9,388.9 L96.0,388.9 L96.0,389.0 L96.1,389.1 L96.2,389.1 L96.3,389.2 L96.4,389.3 L96.5,389.4 L96.6,389.4 L96.7,389.5 L96.8,389.6 L96.9,389.6 L97.0,389.7 L97.1,389.8 L97.1,389.8 L97.2,389.9 L97.3,390.0 L97.4,390.0 L97.5,390.1 L97.6,390.2 L97.7,390.3 L97.8,390.3 L97.9,390.4 L98.0,390.5 L98.1,390.5 L98.2,390.6 L98.2,390.7 L98.3,390.7 L98.4,390.8 L98.5,390.9 L98.6,391.0 L98.7,391.0 L98.8,391.1 L98.9,391.2 L99.0,391.2 L99.1,391.3 L99.2,391.4 L99.2,391.4 L99.3,391.5 L99.4,391.6 L99.5,391.7 L99.6,391.7 L99.7,391.8 L99.8,391.9 L99.9,391.9 L100.0,392.0 L100.1,392.1 L100.2,392.1 L100.3,392.2 L100.3,392.3 L100.4,392.4 L100.5,392.4 L100.6,392.5 L100.7,392.6 L100.8,392.6 L100.9,392.7 L101.0,392.8 L101.1,392.8 L101.2,392.9 L101.3,393.0 L101.4,393.0 L101.4,393.1 L101.5,393.2 L101.6,393.3 L101.7,393.3 L101.8,393.4 L101.9,393.5 L102.0,393.5 L102.1,393.6 L102.2,393.7 L102.3,393.7 L102.4,393.8 L102.4,393.9 L102.5,394.0 L102.6,394.0 L102.7,394.1 L102.8,394.2 L102.9,394.2 L103.0,394.3 L103.1,394.4 L103.2,394.4 L103.3,394.5 L103.4,394.6 L103.5,394.7 L103.5,394.7 L103.6,394.8 L103.7,394.9 L103.8,394.9 L103.9,395.0 L104.0,395.1 L104.1,395.1 L104.2,395.2 L104.3,395.3 L104.4,395.4 L104.5,395.4 L104.5,395.5 L104.6,395.6 L104.7,395.6 L104.8,395.7 L104.9,395.8 L105.0,395.8 L105.1,395.9 L105.2,396.0 L105.3,396.1 L105.4,396.1 L105.5,396.2 L105.6,396.3 L105.6,396.3 L105.7,396.4 L105.8,396.5 L105.9,396.5 L106.0,396.6 L106.1,396.7 L106.2,396.7 L106.3,396.8 L106.4,396.9 L106.5,397.0 L106.6,397.0 L106.7,397.1 L106.7,397.2 L106.8,397.2 L106.9,397.3 L107.0,397.4 L107.1,397.4 L107.2,397.5 L107.3,397.6 L107.4,397.7 L107.5,397.7 L107.6,397.8 L107.7,397.9 L107.7,397.9 L107.8,398.0 L107.9,398.1 L108.0,398.1 L108.1,398.2 L108.2,398.3 L108.3,398.4 L108.4,398.4 L108.5,398.5 L108.6,398.6 L108.7,398.6 L108.8,398.7 L108.8,398.8 L108.9,398.8 L109.0,398.9 L109.1,399.0 L109.2,399.1 L109.3,399.1 L109.4,399.2 L109.5,399.3 L109.6,399.3 L109.7,399.4 L109.8,399.5 L109.8,399.5 L109.9,399.6 L110.0,399.7 L110.1,399.7 L110.2,399.8 L110.3,399.9 L110.4,400.0 L110.5,400.0 L110.6,400.1 L110.7,400.2 L110.8,400.2 L110.9,400.3 L110.9,400.4 L111.0,400.4 L111.1,400.5 L111.2,400.6 L111.3,400.7 L111.4,400.7 L111.5,400.8 L111.6,400.9 L111.7,400.9 L111.8,401.0 L111.9,401.1 L112.0,401.1 L112.0,401.2 L112.1,401.3 L112.2,401.4 L112.3,401.4 L112.4,401.5 L112.5,401.6 L112.6,401.6 L112.7,401.7 L112.8,401.8 L112.9,401.8 L113.0,401.9 L113.0,402.0 L113.1,402.1 L113.2,402.1 L113.3,402.2 L113.4,402.3 L113.5,402.3 L113.6,402.4 L113.7,402.5 L113.8,402.5 L113.9,402.6 L114.0,402.7 L114.1,402.7 L114.1,402.8 L114.2,402.9 L114.3,403.0 L114.4,403.0 L114.5,403.1 L114.6,403.2 L114.7,403.2 L114.8,403.3 L114.9,403.4 L115.0,403.4 L115.1,403.5 L115.2,403.6 L115.2,403.7 L115.3,403.7 L115.4,403.8 L115.5,403.9 L115.6,403.9 L115.7,404.0 L115.8,404.1 L115.9,404.1 L116.0,404.2 L116.1,404.3 L116.2,404.4 L116.2,404.4 L116.3,404.5 L116.4,404.6 L116.5,404.6 L116.6,404.7 L116.7,404.8 L116.8,404.8 L116.9,404.9 L117.0,405.0 L117.1,405.1 L117.2,405.1 L117.3,405.2 L117.3,405.3 L117.4,405.3 L117.5,405.4 L117.6,405.5 L117.7,405.5 L117.8,405.6 L117.9,405.7 L118.0,405.7 L118.1,405.8 L118.2,405.9 L118.3,406.0 L118.3,406.0 L118.4,406.1 L118.5,406.2 L118.6,406.2 L118.7,406.3 L118.8,406.4 L118.9,406.4 L119.0,406.5 L119.1,406.6 L119.2,406.7 L119.3,406.7 L119.4,406.8 L119.4,406.9 L119.5,406.9 L119.6,407.0 L119.7,407.1 L119.8,407.1 L119.9,407.2 L120.0,407.3 L120.1,407.4 L120.2,407.4 L120.3,407.5 L120.4,407.6 L120.5,407.6 L120.5,407.7 L120.6,407.8 L120.7,407.8 L120.8,407.9 L120.9,408.0 L121.0,408.1 L121.1,408.1 L121.2,408.2 L121.3,408.3 L121.4,408.3 L121.5,408.4 L121.5,408.5 L121.6,408.5 L121.7,408.6 L121.8,408.7 L121.9,408.8 L122.0,408.8 L122.1,408.9 L122.2,409.0 L122.3,409.0 L122.4,409.1 L122.5,409.2 L122.6,409.2 L122.6,409.3 L122.7,409.4 L122.8,409.4 L122.9,409.5 L123.0,409.6 L123.1,409.7 L123.2,409.7 L123.3,409.8 L123.4,409.9 L123.5,409.9 L123.6,410.0 L123.6,410.1 L123.7,410.1 L123.8,410.2 L123.9,410.3 L124.0,410.4 L124.1,410.4 L124.2,410.5 L124.3,410.6 L124.4,410.6 L124.5,410.7 L124.6,410.8 L124.7,410.8 L124.7,410.9 L124.8,411.0 L124.9,411.1 L125.0,411.1 L125.1,411.2 L125.2,411.3 L125.3,411.3 L125.4,411.4 L125.5,411.5 L125.6,411.5 L125.7,411.6 L125.8,411.7 L125.9,411.7 L126.0,411.8 L126.1,411.8 L126.2,411.9 L126.3,412.0 L126.4,412.0 L126.5,412.1 L126.6,412.1 L126.7,412.2 L126.8,412.2 L126.9,412.3 L127.0,412.3 L127.1,412.4 L127.2,412.4 L127.3,412.5 L127.4,412.5 L127.5,412.6 L127.6,412.6 L127.7,412.7 L127.8,412.7 L127.9,412.8 L128.0,412.8 L128.1,412.9 L128.2,412.9 L128.3,413.0 L128.4,413.0 L128.5,413.1 L128.6,413.1 L128.7,413.2 L128.8,413.2 L128.9,413.3 L129.0,413.3 L129.1,413.4 L129.2,413.4 L129.3,413.5 L129.5,413.5 L129.6,413.6 L129.7,413.6 L129.8,413.7 L129.9,413.7 L130.0,413.8 L130.1,413.8 L130.2,413.9 L130.3,413.9 L130.4,413.9 L130.5,414.0 L130.6,414.0 L130.7,414.1 L130.8,414.1 L131.0,414.1 L131.1,414.1 L131.2,414.2 L131.3,414.2 L131.4,414.2 L131.5,414.2 L131.6,414.3 L131.7,414.3 L131.9,414.3 L132.0,414.3 L132.1,414.3 L132.2,414.3 L132.3,414.3 L132.4,414.2 L132.5,414.2 L132.6,414.2 L132.8,414.1 L132.9,414.1 L133.0,414.0 L133.1,414.0 L133.1,413.9 L133.2,413.8 L133.3,413.7 L133.4,413.6 L133.4,413.5 L133.5,413.4 L133.5,413.3 L133.5,413.2 L133.6,413.1 L133.6,413.0 L133.6,412.9 L133.6,412.8 L133.5,412.6 L133.5,412.5 L133.5,412.4 L133.5,412.3 L133.4,412.2 L133.4,412.1 L133.4,412.0 L133.3,411.9 L133.3,411.7 L133.3,411.6 L133.2,411.5 L133.2,411.4 L133.2,411.3 L133.1,411.2 L133.1,411.1 L133.1,411.0 L133.0,410.9 L133.0,410.8 L133.0,410.6 L132.9,410.5 L132.9,410.4 L132.9,410.3 L132.8,410.2 L132.8,410.1 L132.8,410.0 L132.7,409.9 L132.7,409.8 L132.6,409.7 L132.6,409.6 L132.6,409.4 L132.5,409.3 L132.5,409.2 L132.5,409.1 L132.4,409.0 L132.4,408.9 L132.3,408.8 L132.3,408.7 L132.3,408.6 L132.2,408.5 L132.2,408.4 L132.2,408.3 L132.1,408.1 L132.1,408.0 L132.0,407.9 L132.0,407.8 L132.0,407.7 L131.9,407.6 L131.9,407.5 L131.9,407.4 L131.8,407.3 L131.8,407.2 L131.7,407.1 L131.7,407.0 L131.7,406.8 L131.6,406.7 L131.6,406.6 L131.5,406.5 L131.5,406.4 L131.5,406.3 L131.4,406.2 L131.4,406.1 L131.4,406.0 L131.3,405.9 L131.3,405.8 L131.2,405.7 L131.2,405.5 L131.2,405.4 L131.1,405.3 L131.1,405.2 L131.0,405.1 L131.0,405.0 L131.0,404.9 L130.9,404.8 L130.9,404.7 L130.9,404.6 L130.8,404.5 L130.8,404.4 L130.7,404.2 L130.7,404.1 L130.7,404.0 L130.6,403.9 L130.6,403.8 L130.5,403.7 L130.5,403.6 L130.5,403.5 L130.4,403.4 L130.4,403.3 L130.3,403.2 L130.3,403.1 L130.3,402.9 L130.2,402.8 L130.2,402.7 L130.1,402.6 L130.1,402.5 L130.1,402.4 L130.0,402.3 L130.0,402.2 L130.0,402.1 L129.9,402.0 L129.9,401.9 L129.8,401.8 L129.8,401.6 L129.8,401.5 L129.7,401.4 L129.7,401.3 L129.6,401.2 L129.6,401.1 L129.6,401.0 L129.5,400.9 L129.5,400.8 L129.4,400.7 L129.4,400.6 L129.4,400.5 L129.3,400.3 L129.3,400.2 L129.3,400.1 L129.2,400.0 L129.2,399.9 L129.1,399.8 L129.1,399.7 L129.1,399.6 L129.0,399.5 L129.0,399.4 L128.9,399.3 L128.9,399.2 L128.9,399.0 L128.8,398.9 L128.8,398.8 L128.8,398.7 L128.7,398.6 L128.7,398.5 L128.6,398.4 L128.6,398.3 L128.6,398.2 L128.5,398.1 L128.5,398.0 L128.5,397.9 L128.4,397.7 L128.4,397.6 L128.3,397.5 L128.3,397.4 L128.3,397.3 L128.2,397.2 L128.2,397.1 L128.1,397.0 L128.1,396.9 L128.1,396.8 L128.0,396.7 L128.0,396.6 L127.9,396.5 L127.9,396.3 L127.9,396.2 L127.8,396.1 L127.8,396.0 L127.7,395.9 L127.7,395.8 L127.7,395.7 L127.6,395.6 L127.6,395.5 L127.6,395.4 L127.5,395.3 L127.5,395.1 L127.5,395.0 L127.4,394.9 L127.4,394.8 L127.4,394.7 L127.3,394.6 L127.3,394.5 L127.3,394.4 L127.2,394.3 L127.2,394.1 L127.2,394.0 L127.2,393.9 L127.1,393.8 L127.1,393.7 L127.1,393.6 L127.1,393.5 L127.1,393.4 L127.0,393.2 L127.0,393.1 L127.0,393.0 L127.0,392.9 L127.0,392.8 L127.0,392.7 L127.0,392.6 L127.0,392.4 L127.0,392.3 L127.0,392.2 L126.9,392.1 L126.9,392.0 L126.9,391.9 L126.9,391.8 L126.9,391.6 L126.9,391.5 L127.0,391.4 L127.0,391.3 L127.0,391.2 L127.0,391.1 L127.0,391.0 L127.0,390.8 L127.0,390.7 L127.0,390.6 L127.0,390.5 L127.0,390.4 L127.1,390.3 L127.1,390.2 L127.1,390.0 L127.1,389.9 L127.1,389.8 L127.1,389.7 L127.2,389.6 L127.2,389.5 L127.2,389.4 L127.2,389.2 L127.3,389.1 L127.3,389.0 L127.3,388.9 L127.3,388.8 L127.4,388.7 L127.4,388.6 L127.4,388.5 L127.4,388.4 L127.5,388.2 L127.5,388.1 L127.5,388.0 L127.6,387.9 L127.6,387.8 L127.7,387.7 L127.7,387.6 L127.7,387.5 L127.8,387.4 L127.8,387.3 L127.8,387.2 L127.9,387.0 L127.9,386.9 L128.0,386.8 L128.0,386.7 L128.1,386.6 L128.1,386.5 L128.1,386.4 L128.2,386.3 L128.2,386.2 L128.3,386.1 L128.3,386.0 L128.4,385.9 L128.4,385.8 L128.5,385.7 L128.5,385.6 L128.6,385.5 L128.6,385.4 L128.7,385.3 L128.8,385.2 L128.8,385.1 L128.9,385.0 L128.9,384.9 L129.0,384.8 L129.0,384.7 L129.1,384.6 L129.2,384.5 L129.2,384.4 L129.3,384.3 L129.3,384.2 L129.4,384.1 L129.5,384.0 L129.5,383.9 L129.6,383.8 L129.7,383.7 L129.7,383.6 L129.8,383.5 L129.9,383.4 L130.0,383.3 L130.0,383.3 L130.1,383.2 L130.2,383.1 L130.3,383.0 L130.3,382.9 L130.4,382.8 L130.5,382.8 L130.6,382.7 L130.7,382.6 L130.7,382.5 L130.8,382.4 L130.9,382.4 L131.0,382.3 L131.1,382.2 L131.2,382.1 L131.3,382.1 L131.4,382.0 L131.4,381.9 L131.5,381.8 L131.6,381.8 L131.7,381.7 L131.8,381.6 L131.9,381.6 L132.0,381.5 L132.1,381.4 L132.2,381.4 L132.3,381.3 L132.4,381.2 L132.5,381.2 L132.6,381.1 L132.6,381.0 L132.7,381.0 L132.8,380.9 L132.9,380.8 L133.0,380.8 L133.1,380.7 L133.2,380.6 L133.3,380.6 L133.4,380.5 L133.5,380.4 L133.6,380.4 L133.7,380.3 L133.8,380.2 L133.9,380.2 L134.0,380.1 L134.1,380.1 L134.2,380.0 L134.3,379.9 L134.4,379.9 L134.5,379.8 L134.6,379.7 L134.7,379.7 L134.8,379.6 L134.8,379.6 L134.9,379.5 L135.0,379.4 L135.1,379.4 L135.2,379.3 L135.3,379.3 L135.4,379.2 L135.5,379.1 L135.6,379.1 L135.7,379.0 L135.8,379.0 L135.9,378.9 L136.0,378.8 L136.1,378.8 L136.2,378.7 L136.3,378.6 L136.4,378.6 L136.5,378.5 L136.6,378.5 L136.7,378.4 L136.8,378.3 L136.9,378.3 L137.0,378.2 L137.1,378.2 L137.2,378.1 L137.3,378.0 L137.4,378.0 L137.5,377.9 L137.6,377.9 L137.7,377.8 L137.8,377.7 L137.9,377.7 L138.0,377.6 L138.1,377.5 L138.2,377.5 L138.3,377.4 L138.4,377.4 L138.5,377.3 L138.5,377.2 L138.6,377.2 L138.7,377.1 L138.8,377.1 L138.9,377.0 L139.0,376.9 L139.1,376.9 L139.2,376.8 L139.3,376.8 L139.4,376.7 L139.5,376.6 L139.6,376.6 L139.7,376.5 L139.8,376.4 L139.9,376.4 L140.0,376.3 L140.1,376.3 L140.2,376.2 L140.3,376.1 L140.4,376.1 L140.5,376.0 L140.6,376.0 L140.7,375.9 L140.8,375.8 L140.9,375.8 L141.0,375.7 L141.1,375.7 L141.2,375.6 L141.3,375.5 L141.4,375.5 L141.5,375.4 L141.6,375.4 L141.7,375.3 L141.8,375.2 L141.9,375.2 L142.0,375.1 L142.1,375.1 L142.2,375.0 L142.3,374.9 L142.4,374.9 L142.5,374.8 L142.6,374.8 L142.7,374.7 L142.8,374.6 L142.8,374.6 L142.9,374.5 L143.0,374.5 L143.1,374.4 L143.2,374.3 L143.3,374.3 L143.4,374.2 L143.5,374.1 L143.6,374.1 L143.7,374.0 L143.8,374.0 L143.9,373.9 L144.0,373.8 L144.1,373.8 L144.2,373.7 L144.3,373.7 L144.4,373.6 L144.5,373.5 L144.6,373.5 L144.7,373.4 L144.8,373.4 L144.9,373.3 L145.0,373.2 L145.1,373.2 L145.2,373.1 L145.3,373.1 L145.4,373.0 L145.5,372.9 L145.6,372.9 L145.7,372.8 L145.8,372.8 L145.9,372.7 L146.0,372.6 L146.1,372.6 L146.2,372.5 L146.3,372.5 L146.4,372.4 L146.5,372.3 L146.6,372.3 L146.7,372.2 L146.8,372.1 L146.9,372.1 L147.0,372.0 L147.0,372.0 L147.1,371.9 L147.2,371.8 L147.3,371.8 L147.4,371.7 L147.5,371.7 L147.6,371.6 L147.7,371.5 L147.8,371.5 L147.9,371.4 L148.0,371.3 L148.1,371.3 L148.2,371.2 L148.3,371.2 L148.4,371.1 L148.5,371.0 L148.6,371.0 L148.7,370.9 L148.8,370.8 L148.9,370.8 L149.0,370.7 L149.1,370.6 L149.2,370.6 L149.3,370.5 L149.4,370.4 L149.5,370.4 L149.5,370.3 L149.6,370.3 L149.7,370.2 L149.8,370.1 L149.9,370.0 L150.0,370.0 L150.1,369.9 L150.2,369.8 L150.3,369.8 L150.4,369.7 L150.5,369.6 L150.5,369.5 L150.6,369.5 L150.7,369.4 L150.8,369.3 L150.9,369.2 L150.9,369.1 L151.0,369.0 L151.1,369.0 L151.2,368.9 L151.3,368.8 L151.3,368.7 L151.4,368.6 L151.5,368.5 L151.5,368.4 L151.6,368.3 L151.7,368.2 L151.7,368.1 L151.8,368.1 L151.9,368.0 L151.9,367.9 L152.0,367.8 L152.1,367.7 L152.1,367.6 L152.2,367.5 L152.2,367.4 L152.3,367.3 L152.3,367.2 L152.4,367.1 L152.4,367.0 L152.5,366.9 L152.6,366.8 L152.6,366.7 L152.7,366.6 L152.7,366.4 L152.8,366.3 L152.8,366.2 L152.8,366.1 L152.9,366.0 L152.9,365.9 L153.0,365.8 L153.0,365.7 L153.1,365.6 L153.1,365.5 L153.2,365.4 L153.2,365.3 L153.2,365.2 L153.3,365.1 L153.3,365.0 L153.4,364.9 L153.4,364.8 L153.5,364.6 L153.5,364.5 L153.5,364.4 L153.6,364.3 L153.6,364.2 L153.7,364.1 L153.7,364.0 L153.7,363.9 L153.8,363.8 L153.8,363.7 L153.8,363.6 L153.9,363.5 L153.9,363.3 L154.0,363.2 L154.0,363.1 L154.0,363.0 L154.1,362.9 L154.1,362.8 L154.2,362.7 L154.2,362.6 L154.3,362.5 L154.3,362.4 L154.3,362.3 L154.4,362.2 L154.4,362.1 L154.5,362.0 L154.5,361.8 L154.6,361.7 L154.6,361.6 L154.7,361.5 L154.7,361.4 L154.7,361.3 L154.8,361.2 L154.8,361.1 L154.9,361.0 L154.9,360.9 L155.0,360.8 L155.1,360.7 L155.1,360.6 L155.2,360.5 L155.2,360.4 L155.3,360.3 L155.3,360.2 L155.4,360.1 L155.5,360.0 L155.5,359.9 L155.6,359.8 L155.6,359.7 L155.7,359.6 L155.8,359.5 L155.8,359.4 L155.9,359.3 L156.0,359.3 L156.1,359.2 L156.1,359.1 L156.2,359.0 L156.3,358.9 L156.4,358.8 L156.4,358.7 L156.5,358.7 L156.6,358.6 L156.7,358.5 L156.8,358.4 L156.9,358.4 L157.0,358.3 L157.1,358.2 L157.1,358.2 L157.2,358.1 L157.3,358.0 L157.4,358.0 L157.5,357.9 L157.6,357.8 L157.7,357.8 L157.8,357.7 L157.9,357.7 L158.0,357.6 L158.1,357.5 L158.2,357.5 L158.3,357.4 L158.4,357.4 L158.5,357.3 L158.6,357.3 L158.7,357.2 L158.9,357.2 L159.0,357.2 L159.1,357.1 L159.2,357.1 L159.3,357.0 L159.4,357.0 L159.5,356.9 L159.6,356.9 L159.7,356.8 L159.8,356.8 L159.9,356.8 L160.0,356.7 L160.1,356.7 L160.2,356.6 L160.3,356.6 L160.5,356.6 L160.6,356.5 L160.7,356.5 L160.8,356.5 L160.9,356.4 L161.0,356.4 L161.1,356.3 L161.2,356.3 L161.3,356.3 L161.4,356.2 L161.5,356.2 L161.6,356.1 L161.8,356.1 L161.9,356.1 L162.0,356.0 L162.1,356.0 L162.2,355.9 L162.3,355.9 L162.4,355.8 L162.5,355.8 L162.6,355.7 L162.7,355.7 L162.8,355.6 L162.9,355.6 L163.0,355.5 L163.1,355.4 L163.2,355.4 L163.3,355.3 L163.4,355.3 L163.5,355.2 L163.6,355.1 L163.7,355.0 L163.7,355.0 L163.8,354.9 L163.9,354.8 L164.0,354.7 L164.1,354.7 L164.2,354.6 L164.3,354.5 L164.3,354.4 L164.4,354.4 L164.5,354.3 L164.6,354.2 L164.7,354.1 L164.8,354.0 L164.8,354.0 L164.9,353.9 L165.0,353.8 L165.1,353.7 L165.2,353.6 L165.2,353.5 L165.3,353.4 L165.4,353.4 L165.4,353.3 L165.5,353.2 L165.6,353.1 L165.6,353.0 L165.7,352.9 L165.7,352.8 L165.8,352.7 L165.9,352.6 L165.9,352.5 L166.0,352.4 L166.0,352.3 L166.0,352.1 L166.1,352.0 L166.1,351.9 L166.2,351.8 L166.2,351.7 L166.2,351.6 L166.3,351.5 L166.3,351.4 L166.3,351.3 L166.4,351.2 L166.4,351.1 L166.4,350.9 L166.4,350.8 L166.5,350.7 L166.5,350.6 L166.5,350.5 L166.5,350.4 L166.5,350.3 L166.6,350.2 L166.6,350.0 L166.6,349.9 L166.6,349.8 L166.6,349.7 L166.7,349.6 L166.7,349.5 L166.7,349.4 L166.7,349.2 L166.7,349.1 L166.7,349.0 L166.7,348.9 L166.7,348.8 L166.8,348.7 L166.8,348.6 L166.8,348.4 L166.8,348.3 L166.8,348.2 L166.8,348.1 L166.8,348.0 L166.8,347.9 L166.9,347.8 L166.9,347.6 L166.9,347.5 L166.9,347.4 L166.9,347.3 L166.9,347.2 L166.9,347.1 L166.9,347.0 L167.0,346.8 L167.0,346.7 L167.0,346.6 L167.0,346.5 L167.0,346.4 L167.0,346.3 L167.1,346.2 L167.1,346.0 L167.1,345.9 L167.1,345.8 L167.1,345.7 L167.1,345.6 L167.2,345.5 L167.2,345.4 L167.2,345.3 L167.2,345.1 L167.3,345.0 L167.3,344.9 L167.3,344.8 L167.4,344.7 L167.4,344.6 L167.4,344.5 L167.4,344.4 L167.5,344.3 L167.5,344.1 L167.6,344.0 L167.6,343.9 L167.6,343.8 L167.7,343.7 L167.7,343.6 L167.8,343.5 L167.8,343.4 L167.8,343.3 L167.9,343.2 L167.9,343.1 L168.0,343.0 L168.0,342.9 L168.1,342.8 L168.1,342.7 L168.2,342.6 L168.3,342.5 L168.3,342.4 L168.4,342.3 L168.4,342.2 L168.5,342.1 L168.6,342.0 L168.6,341.9 L168.7,341.8 L168.7,341.7 L168.8,341.6 L168.9,341.5 L168.9,341.4 L169.0,341.3 L169.1,341.2 L169.1,341.1 L169.2,341.0 L169.3,340.9 L169.3,340.8 L169.4,340.8 L169.5,340.7 L169.6,340.6 L169.6,340.5 L169.7,340.4 L169.8,340.3 L169.9,340.2 L169.9,340.1 L170.0,340.1 L170.1,340.0 L170.2,339.9 L170.3,339.8 L170.3,339.7 L170.4,339.7 L170.5,339.6 L170.6,339.5 L170.7,339.4 L170.8,339.3 L170.9,339.3 L170.9,339.2 L171.0,339.1 L171.1,339.0 L171.2,339.0 L171.3,338.9 L171.4,338.8 L171.5,338.7 L171.6,338.7 L171.7,338.6 L171.7,338.5 L171.8,338.5 L171.9,338.4 L172.0,338.3 L172.1,338.3 L172.2,338.2 L172.3,338.1 L172.4,338.1 L172.5,338.0 L172.6,337.9 L172.7,337.9 L172.8,337.8 L172.9,337.8 L173.0,337.7 L173.1,337.6 L173.2,337.6 L173.3,337.5 L173.4,337.5 L173.5,337.4 L173.6,337.3 L173.7,337.3 L173.8,337.2 L173.9,337.2 L174.0,337.1 L174.1,337.1 L174.2,337.0 L174.3,337.0 L174.4,336.9 L174.5,336.9 L174.6,336.8 L174.7,336.8 L174.8,336.7 L174.9,336.7 L175.0,336.6 L175.1,336.6 L175.2,336.5 L175.3,336.5 L175.4,336.4 L175.5,336.4 L175.6,336.3 L175.7,336.3 L175.9,336.3 L176.0,336.2 L176.1,336.2 L176.2,336.1 L176.3,336.1 L176.4,336.1 L176.5,336.0 L176.6,336.0 L176.7,336.0 L176.8,335.9 L176.9,335.9 L177.1,335.9 L177.2,335.8 L177.3,335.8 L177.4,335.8 L177.5,335.7 L177.6,335.7 L177.7,335.7 L177.8,335.7 L177.9,335.6 L178.1,335.6 L178.2,335.6 L178.3,335.6 L178.4,335.6 L178.5,335.5 L178.6,335.5 L178.7,335.5 L178.9,335.5 L179.0,335.5 L179.1,335.5 L179.2,335.5 L179.3,335.5 L179.4,335.5 L179.5,335.5 L179.7,335.5 L179.8,335.5 L179.9,335.5 L180.0,335.5 L180.1,335.5 L180.2,335.5 L180.4,335.5 L180.5,335.5 L180.6,335.5 L180.7,335.5 L180.8,335.5 L180.9,335.5 L181.0,335.5 L181.2,335.5 L181.3,335.6 L181.4,335.6 L181.5,335.6 L181.6,335.6 L181.7,335.6 L181.8,335.7 L181.9,335.7 L182.1,335.7 L182.2,335.7 L182.3,335.8 L182.4,335.8 L182.5,335.8 L182.6,335.8 L182.7,335.9 L182.8,335.9 L182.9,335.9 L183.1,336.0 L183.2,336.0 L183.3,336.0 L183.4,336.1 L183.5,336.1 L183.6,336.2 L183.7,336.2 L183.8,336.2 L183.9,336.3 L184.0,336.3 L184.1,336.4 L184.2,336.4 L184.3,336.5 L184.5,336.5 L184.6,336.6 L184.7,336.6 L184.8,336.7 L184.9,336.7 L185.0,336.8 L185.1,336.8 L185.2,336.9 L185.3,336.9 L185.4,337.0 L185.5,337.0 L185.6,337.1 L185.7,337.1 L185.8,337.2 L185.9,337.2 L186.0,337.3 L186.1,337.3 L186.2,337.4 L186.3,337.4 L186.4,337.5 L186.5,337.5 L186.6,337.6 L186.7,337.6 L186.8,337.7 L186.9,337.7 L187.0,337.7 L187.2,337.8 L187.3,337.8 L187.4,337.9 L187.5,337.9 L187.6,338.0 L187.7,338.0 L187.8,338.1 L187.9,338.1 L188.0,338.2 L188.1,338.2 L188.2,338.3 L188.3,338.3 L188.4,338.4 L188.5,338.4 L188.6,338.5 L188.7,338.5 L188.8,338.6 L188.9,338.6 L189.0,338.7 L189.1,338.7 L189.2,338.7 L189.4,338.8 L189.5,338.8 L189.6,338.9 L189.7,338.9 L189.8,339.0 L189.9,339.0 L190.0,339.1 L190.1,339.1 L190.2,339.2 L190.3,339.2 L190.4,339.3 L190.5,339.3 L190.6,339.3 L190.7,339.4 L190.8,339.4 L190.9,339.5 L191.0,339.5 L191.1,339.6 L191.2,339.6 L191.4,339.7 L191.5,339.7 L191.6,339.8 L191.7,339.8 L191.8,339.8 L191.9,339.9 L192.0,339.9 L192.1,340.0 L192.2,340.0 L192.3,340.1 L192.4,340.1 L192.5,340.2 L192.6,340.2 L192.7,340.3 L192.8,340.3 L192.9,340.3 L193.0,340.4 L193.2,340.4 L193.3,340.5 L193.4,340.5 L193.5,340.6 L193.6,340.6 L193.7,340.6 L193.8,340.7 L193.9,340.7 L194.0,340.7 L194.1,340.8 L194.2,340.8 L194.3,340.8 L194.5,340.9 L194.6,340.9 L194.7,340.9 L194.8,340.9 L194.9,340.9 L195.0,340.9 L195.1,340.9 L195.3,340.9 L195.4,340.9 L195.5,340.9 L195.6,340.8 L195.7,340.8 L195.8,340.8 L195.9,340.7 L196.0,340.7 L196.1,340.7 L196.3,340.6 L196.4,340.6 L196.5,340.5 L196.6,340.5 L196.7,340.4 L196.8,340.4 L196.9,340.3 L197.0,340.3 L197.1,340.2 L197.2,340.2 L197.3,340.1 L197.4,340.1 L197.5,340.0 L197.6,340.0 L197.7,339.9 L197.8,339.8 L197.9,339.8 L198.0,339.7 L198.1,339.6 L198.2,339.6 L198.3,339.5 L198.4,339.4 L198.4,339.4 L198.5,339.3 L198.6,339.2 L198.7,339.2 L198.8,339.1 L198.9,339.0 L199.0,339.0 L199.1,338.9 L199.2,338.8 L199.3,338.8 L199.4,338.7 L199.4,338.6 L199.5,338.5 L199.6,338.5 L199.7,338.4 L199.8,338.3 L199.9,338.2 L200.0,338.2 L200.1,338.1 L200.1,338.0 L200.2,337.9 L200.3,337.9 L200.4,337.8 L200.5,337.7 L200.6,337.6 L200.7,337.6 L200.7,337.5 L200.8,337.4 L200.9,337.3 L201.0,337.2 L201.1,337.2 L201.2,337.1 L201.3,337.0 L201.3,336.9 L201.4,336.8 L201.5,336.8 L201.6,336.7 L201.7,336.6 L201.7,336.5 L201.8,336.4 L201.9,336.4 L202.0,336.3 L202.1,336.2 L202.2,336.1 L202.2,336.0 L202.3,336.0 L202.4,335.9 L202.5,335.8 L202.6,335.7 L202.6,335.6 L202.7,335.5 L202.8,335.5 L202.9,335.4 L202.9,335.3 L203.0,335.2 L203.1,335.1 L203.2,335.0 L203.2,334.9 L203.3,334.9 L203.4,334.8 L203.5,334.7 L203.5,334.6 L203.6,334.5 L203.7,334.4 L203.8,334.3 L203.8,334.2 L203.9,334.1 L204.0,334.0 L204.0,333.9 L204.1,333.9 L204.2,333.8 L204.2,333.7 L204.3,333.6 L204.4,333.5 L204.4,333.4 L204.5,333.3 L204.5,333.2 L204.6,333.1 L204.7,333.0 L204.7,332.9 L204.8,332.8 L204.9,332.7 L204.9,332.6 L205.0,332.5 L205.0,332.4 L205.1,332.3 L205.2,332.2 L205.2,332.1 L205.3,332.0 L205.4,331.9 L205.4,331.8 L205.5,331.8 L205.6,331.7 L205.7,331.6 L205.7,331.5 L205.8,331.4 L205.9,331.3 L206.0,331.2 L206.0,331.2 L206.1,331.1 L206.2,331.0 L206.3,330.9 L206.4,330.8 L206.5,330.8 L206.6,330.7 L206.7,330.6 L206.8,330.6 L206.8,330.5 L206.9,330.4 L207.0,330.4 L207.1,330.3 L207.2,330.3 L207.3,330.2 L207.4,330.2 L207.5,330.1 L207.6,330.1 L207.8,330.0 L207.9,330.0 L208.0,329.9 L208.1,329.9 L208.2,329.8 L208.3,329.8 L208.4,329.8 L208.5,329.8 L208.6,329.7 L208.7,329.7 L208.9,329.7 L209.0,329.6 L209.1,329.6 L209.2,329.6 L209.3,329.6 L209.4,329.6 L209.5,329.6 L209.6,329.5 L209.8,329.5 L209.9,329.5 L210.0,329.5 L210.1,329.5 L210.2,329.5 L210.3,329.5 L210.5,329.5 L210.6,329.5 L210.7,329.5 L210.8,329.5 L210.9,329.5 L211.0,329.5 L211.1,329.5 L211.3,329.5 L211.4,329.5 L211.5,329.5 L211.6,329.6 L211.7,329.6 L211.8,329.6 L211.9,329.6 L212.1,329.6 L212.2,329.6 L212.3,329.7 L212.4,329.7 L212.5,329.7 L212.6,329.8 L212.7,329.8 L212.8,329.8 L212.9,329.9 L213.0,329.9 L213.2,329.9 L213.3,330.0 L213.4,330.0 L213.5,330.1 L213.6,330.1 L213.7,330.2 L213.8,330.2 L213.9,330.3 L214.0,330.4 L214.1,330.4 L214.2,330.5 L214.3,330.5 L214.4,330.6 L214.5,330.7 L214.6,330.7 L214.6,330.8 L214.7,330.9 L214.8,330.9 L214.9,331.0 L215.0,331.1 L215.1,331.2 L215.2,331.2 L215.3,331.3 L215.3,331.4 L215.4,331.5 L215.5,331.6 L215.6,331.7 L215.6,331.7 L215.7,331.8 L215.8,331.9 L215.9,332.0 L215.9,332.1 L216.0,332.2 L216.1,332.3 L216.1,332.4 L216.2,332.5 L216.3,332.6 L216.3,332.7 L216.4,332.8 L216.4,332.9 L216.5,333.0 L216.5,333.1 L216.6,333.2 L216.6,333.3 L216.7,333.4 L216.8,333.5 L216.8,333.6 L216.9,333.7 L216.9,333.8 L217.0,333.9 L217.1,334.0 L217.1,334.1 L217.2,334.2 L217.3,334.2 L217.4,334.3 L217.5,334.4 L217.5,334.5 L217.6,334.5 L217.7,334.6 L217.8,334.7 L217.9,334.7 L218.0,334.8 L218.1,334.8 L218.2,334.9 L218.3,334.9 L218.5,335.0 L218.6,335.0 L218.7,335.0 L218.8,335.1 L218.9,335.1 L219.0,335.1 L219.1,335.1 L219.2,335.1 L219.3,335.2 L219.5,335.2 L219.6,335.2 L219.7,335.2 L219.8,335.2 L219.9,335.2 L220.0,335.2 L220.2,335.2 L220.3,335.2 L220.4,335.1 L220.5,335.1 L220.6,335.1 L220.7,335.1 L220.8,335.1 L220.9,335.1 L221.1,335.0 L221.2,335.0 L221.3,335.0 L221.4,335.0 L221.5,334.9 L221.6,334.9 L221.7,334.9 L221.8,334.8 L222.0,334.8 L222.1,334.8 L222.2,334.8 L222.3,334.7 L222.4,334.7 L222.5,334.7 L222.6,334.6 L222.7,334.6 L222.8,334.6 L223.0,334.5 L223.1,334.5 L223.2,334.5 L223.3,334.5 L223.4,334.4 L223.5,334.4 L223.6,334.4 L223.7,334.4 L223.9,334.3 L224.0,334.3 L224.1,334.3 L224.2,334.3 L224.3,334.2 L224.4,334.2 L224.5,334.2 L224.6,334.2 L224.8,334.1 L224.9,334.1 L225.0,334.1 L225.1,334.1 L225.2,334.0 L225.3,334.0 L225.4,334.0 L225.5,334.0 L225.6,333.9 L225.8,333.9 L225.9,333.9 L226.0,333.9 L226.1,333.8 L226.2,333.8 L226.3,333.8 L226.4,333.8 L226.5,333.7 L226.7,333.7 L226.8,333.7 L226.9,333.7 L227.0,333.6 L227.1,333.6 L227.2,333.6 L227.3,333.6 L227.4,333.5 L227.6,333.5 L227.7,333.5 L227.8,333.5 L227.9,333.4 L228.0,333.4 L228.1,333.4 L228.2,333.4 L228.3,333.3 L228.5,333.3 L228.6,333.3 L228.7,333.3 L228.8,333.2 L228.9,333.2 L229.0,333.2 L229.1,333.1 L229.2,333.1 L229.3,333.1 L229.5,333.1 L229.6,333.0 L229.7,333.0 L229.8,333.0 L229.9,333.0 L230.0,332.9 L230.1,332.9 L230.2,332.9 L230.4,332.9 L230.5,332.8 L230.6,332.8 L230.7,332.8 L230.8,332.8 L230.9,332.7 L231.0,332.7 L231.1,332.7 L231.3,332.7 L231.4,332.6 L231.5,332.6 L231.6,332.6 L231.7,332.6 L231.8,332.5 L231.9,332.5 L232.0,332.5 L232.2,332.5 L232.3,332.4 L232.4,332.4 L232.5,332.4 L232.6,332.4 L232.7,332.3 L232.8,332.3 L232.9,332.3 L233.0,332.3 L233.2,332.2 L233.3,332.2 L233.4,332.2 L233.5,332.2 L233.6,332.1 L233.7,332.1 L233.8,332.1 L233.9,332.1 L234.1,332.0 L234.2,332.0 L234.3,332.0 L234.4,332.0 L234.5,331.9 L234.6,331.9 L234.7,331.9 L234.8,331.9 L235.0,331.8 L235.1,331.8 L235.2,331.8 L235.3,331.8 L235.4,331.7 L235.5,331.7 L235.6,331.7 L235.7,331.6 L235.9,331.6 L236.0,331.6 L236.1,331.6 L236.2,331.5 L236.3,331.5 L236.4,331.5 L236.5,331.5 L236.6,331.4 L236.8,331.4 L236.9,331.4 L237.0,331.4 L237.1,331.3 L237.2,331.3 L237.3,331.3 L237.4,331.3 L237.5,331.2 L237.6,331.2 L237.8,331.2 L237.9,331.2 L238.0,331.1 L238.1,331.1 L238.2,331.1 L238.3,331.1 L238.4,331.0 L238.5,331.0 L238.7,331.0 L238.8,331.0 L238.9,330.9 L239.0,330.9 L239.1,330.9 L239.2,330.9 L239.3,330.8 L239.4,330.8 L239.6,330.8 L239.7,330.7 L239.8,330.7 L239.9,330.7 L240.0,330.7 L240.1,330.6 L240.2,330.6 L240.3,330.6 L240.4,330.5 L240.5,330.5 L240.6,330.4 L240.7,330.4 L240.8,330.3 L240.9,330.3 L241.0,330.2 L241.1,330.1 L241.2,330.1 L241.3,330.0 L241.4,329.9 L241.5,329.8 L241.6,329.8 L241.7,329.7 L241.7,329.6 L241.8,329.5 L241.9,329.4 L242.0,329.3 L242.1,329.3 L242.1,329.2 L242.2,329.1 L242.3,329.0 L242.4,328.9 L242.4,328.8 L242.5,328.7 L242.6,328.7 L242.7,328.6 L242.7,328.5 L242.8,328.4 L242.9,328.3 L242.9,328.2 L243.0,328.1 L243.1,328.0 L243.2,327.9 L243.2,327.8 L243.3,327.7 L243.4,327.7 L243.4,327.6 L243.5,327.5 L243.6,327.4 L243.6,327.3 L243.7,327.2 L243.8,327.1 L243.9,327.0 L243.9,326.9 L244.0,326.8 L244.1,326.8 L244.1,326.7 L244.2,326.6 L244.3,326.5 L244.4,326.4 L244.4,326.3 L244.5,326.2 L244.6,326.1 L244.6,326.0 L244.7,325.9 L244.8,325.9 L244.9,325.8 L244.9,325.7 L245.0,325.6 L245.1,325.5 L245.2,325.4 L245.2,325.3 L245.3,325.2 L245.4,325.1 L245.4,325.1 L245.5,325.0 L245.6,324.9 L245.7,324.8 L245.7,324.7 L245.8,324.6 L245.9,324.5 L245.9,324.4 L246.0,324.3 L246.1,324.2 L246.2,324.2 L246.2,324.1 L246.3,324.0 L246.4,323.9 L246.5,323.8 L246.5,323.7 L246.6,323.6 L246.7,323.5 L246.7,323.4 L246.8,323.4 L246.9,323.3 L247.0,323.2 L247.0,323.1 L247.1,323.0 L247.2,322.9 L247.3,322.8 L247.3,322.7 L247.4,322.6 L247.5,322.5 L247.5,322.5 L247.6,322.4 L247.7,322.3 L247.8,322.2 L247.8,322.1 L247.9,322.0 L248.0,321.9 L248.0,321.8 L248.1,321.7 L248.2,321.7 L248.3,321.6 L248.3,321.5 L248.4,321.4 L248.5,321.3 L248.6,321.2 L248.6,321.1 L248.7,321.0 L248.8,320.9 L248.8,320.9 L248.9,320.8 L249.0,320.7 L249.1,320.6 L249.1,320.5 L249.2,320.4 L249.3,320.3 L249.4,320.2 L249.4,320.1 L249.5,320.0 L249.6,320.0 L249.6,319.9 L249.7,319.8 L249.8,319.7 L249.9,319.6 L249.9,319.5 L250.0,319.4 L250.1,319.3 L250.1,319.2 L250.2,319.2 L250.3,319.1 L250.4,319.0 L250.4,318.9 L250.5,318.8 L250.6,318.7 L250.7,318.6 L250.7,318.5 L250.8,318.4 L250.9,318.3 L250.9,318.3 L251.0,318.2 L251.1,318.1 L251.2,318.0 L251.2,317.9 L251.3,317.8 L251.4,317.7 L251.4,317.6 L251.5,317.5 L251.6,317.5 L251.7,317.4 L251.7,317.3 L251.8,317.2 L251.9,317.1 L252.0,317.0 L252.0,316.9 L252.1,316.8 L252.2,316.7 L252.2,316.6 L252.3,316.6 L252.4,316.5 L252.5,316.4 L252.5,316.3 L252.6,316.2 L252.7,316.1 L252.7,316.0 L252.8,315.9 L252.9,315.8 L253.0,315.8 L253.0,315.7 L253.1,315.6 L253.2,315.5 L253.3,315.4 L253.3,315.3 L253.4,315.2 L253.5,315.1 L253.5,315.0 L253.6,314.9 L253.7,314.9 L253.8,314.8 L253.8,314.7 L253.9,314.6 L254.0,314.5 L254.0,314.4 L254.1,314.3 L254.2,314.2 L254.3,314.1 L254.3,314.1 L254.4,314.0 L254.5,313.9 L254.6,313.8 L254.6,313.7 L254.7,313.6 L254.8,313.5 L254.8,313.4 L254.9,313.3 L255.0,313.2 L255.1,313.2 L255.1,313.1 L255.2,313.0 L255.3,312.9 L255.3,312.8 L255.4,312.7 L255.5,312.6 L255.6,312.5 L255.6,312.4 L255.7,312.4 L255.8,312.3 L255.9,312.2 L255.9,312.1 L256.0,312.0 L256.1,311.9 L256.1,311.8 L256.2,311.7 L256.3,311.6 L256.4,311.5 L256.4,311.5 L256.5,311.4 L256.6,311.3 L256.6,311.2 L256.7,311.1 L256.8,311.0 L256.9,310.9 L256.9,310.8 L257.0,310.7 L257.1,310.7 L257.2,310.6 L257.2,310.5 L257.3,310.4 L257.4,310.3 L257.4,310.2 L257.5,310.1 L257.6,310.0 L257.7,309.9 L257.7,309.8 L257.8,309.8 L257.9,309.7 L257.9,309.6 L258.0,309.5 L258.1,309.4 L258.2,309.3 L258.2,309.2 L258.3,309.1 L258.4,309.0 L258.5,309.0 L258.5,308.9 L258.6,308.8 L258.7,308.7 L258.7,308.6 L258.8,308.5 L258.9,308.4 L259.0,308.3 L259.0,308.2 L259.1,308.1 L259.2,308.1 L259.2,308.0 L259.3,307.9 L259.4,307.8 L259.5,307.7 L259.5,307.6 L259.6,307.5 L259.7,307.4 L259.8,307.3 L259.8,307.3 L259.9,307.2 L260.0,307.1 L260.0,307.0 L260.1,306.9 L260.2,306.8 L260.3,306.7 L260.3,306.6 L260.4,306.5 L260.5,306.4 L260.5,306.4 L260.6,306.3 L260.7,306.2 L260.8,306.1 L260.8,306.0 L260.9,305.9 L261.0,305.8 L261.1,305.7 L261.1,305.6 L261.2,305.6 L261.3,305.5 L261.3,305.4 L261.4,305.3 L261.5,305.2 L261.6,305.1 L261.6,305.0 L261.7,304.9 L261.8,304.8 L261.8,304.7 L261.9,304.7 L262.0,304.6 L262.1,304.5 L262.1,304.4 L262.2,304.3 L262.3,304.2 L262.4,304.1 L262.4,304.0 L262.5,303.9 L262.6,303.9 L262.6,303.8 L262.7,303.7 L262.8,303.6 L262.9,303.5 L262.9,303.4 L263.0,303.3 L263.1,303.2 L263.1,303.1 L263.2,303.0 L263.3,303.0 L263.4,302.9 L263.4,302.8 L263.5,302.7 L263.6,302.6 L263.6,302.5 L263.7,302.4 L263.8,302.3 L263.9,302.2 L263.9,302.1 L264.0,302.1 L264.1,302.0 L264.2,301.9 L264.2,301.8 L264.3,301.7 L264.4,301.6 L264.4,301.5 L264.5,301.4 L264.6,301.3 L264.7,301.3 L264.7,301.2 L264.8,301.1 L264.9,301.0 L264.9,300.9 L265.0,300.8 L265.1,300.7 L265.2,300.6 L265.2,300.5 L265.3,300.4 L265.4,300.4 L265.4,300.3 L265.5,300.2 L265.6,300.1 L265.7,300.0 L265.7,299.9 L265.8,299.8 L265.9,299.7 L265.9,299.6 L266.0,299.5 L266.1,299.4 L266.2,299.4 L266.2,299.3 L266.3,299.2 L266.4,299.1 L266.4,299.0 L266.5,298.9 L266.6,298.8 L266.6,298.7 L266.7,298.6 L266.8,298.5 L266.8,298.4 L266.9,298.3 L267.0,298.2 L267.0,298.1 L267.1,298.1 L267.2,298.0 L267.2,297.9 L267.3,297.8 L267.4,297.7 L267.4,297.6 L267.5,297.5 L267.5,297.4 L267.6,297.3 L267.7,297.2 L267.7,297.1 L267.8,297.0 L267.9,296.9 L267.9,296.8 L268.0,296.7 L268.0,296.6 L268.1,296.5 L268.2,296.4 L268.2,296.3 L268.3,296.2 L268.3,296.1 L268.4,296.0 L268.4,295.9 L268.5,295.8 L268.5,295.7 L268.6,295.6 L268.6,295.5 L268.6,295.4 L268.7,295.3 L268.7,295.1 L268.7,295.0 L268.7,294.9 L268.7,294.8 L268.7,294.7 L268.7,294.6 L268.7,294.5 L268.7,294.3 L268.6,294.2 L268.6,294.1 L268.5,294.0 L268.5,293.9 L268.4,293.8 L268.4,293.7 L268.3,293.6 L268.2,293.5 L268.1,293.5 L268.1,293.4 L268.0,293.3 L267.9,293.3 L267.8,293.2 L267.6,293.2 L267.5,293.2 L267.4,293.1 L267.3,293.1 L267.2,293.1 L267.1,293.1 L267.0,293.1 L266.9,293.1 L266.7,293.1 L266.6,293.1 L266.5,293.1 L266.4,293.1 L266.3,293.1 L266.2,293.1 L266.0,293.1 L265.9,293.1 L265.8,293.2 L265.7,293.2 L265.6,293.2 L265.5,293.2 L265.4,293.2 L265.3,293.3 L265.1,293.3 L265.0,293.3 L264.9,293.3 L264.8,293.4 L264.7,293.4 L264.6,293.4 L264.5,293.5 L264.4,293.5 L264.3,293.5 L264.1,293.6 L264.0,293.6 L263.9,293.6 L263.8,293.7 L263.7,293.7 L263.6,293.7 L263.5,293.8 L263.4,293.8 L263.3,293.8 L263.2,293.9 L263.1,293.9 L262.9,294.0 L262.8,294.0 L262.7,294.0 L262.6,294.1 L262.5,294.1 L262.4,294.1 L262.3,294.2 L262.2,294.2 L262.1,294.3 L262.0,294.3 L261.9,294.3 L261.8,294.4 L261.6,294.4 L261.5,294.5 L261.4,294.5 L261.3,294.5 L261.2,294.6 L261.1,294.6 L261.0,294.6 L260.9,294.7 L260.8,294.7 L260.7,294.8 L260.6,294.8 L260.4,294.8 L260.3,294.9 L260.2,294.9 L260.1,294.9 L260.0,295.0 L259.9,295.0 L259.8,295.0 L259.7,295.1 L259.6,295.1 L259.5,295.2 L259.4,295.2 L259.2,295.2 L259.1,295.3 L259.0,295.3 L258.9,295.3 L258.8,295.4 L258.7,295.4 L258.6,295.4 L258.5,295.5 L258.4,295.5 L258.3,295.5 L258.1,295.6 L258.0,295.6 L257.9,295.6 L257.8,295.7 L257.7,295.7 L257.6,295.7 L257.5,295.8 L257.4,295.8 L257.3,295.9 L257.2,295.9 L257.1,295.9 L256.9,296.0 L256.8,296.0 L256.7,296.0 L256.6,296.1 L256.5,296.1 L256.4,296.1 L256.3,296.2 L256.2,296.2 L256.1,296.2 L256.0,296.3 L255.9,296.3 L255.7,296.4 L255.6,296.4 L255.5,296.4 L255.4,296.5 L255.3,296.5 L255.2,296.5 L255.1,296.6 L255.0,296.6 L254.9,296.6 L254.8,296.7 L254.7,296.7 L254.5,296.7 L254.4,296.8 L254.3,296.8 L254.2,296.9 L254.1,296.9 L254.0,296.9 L253.9,297.0 L253.8,297.0 L253.7,297.0 L253.6,297.1 L253.5,297.1 L253.3,297.1 L253.2,297.2 L253.1,297.2 L253.0,297.3 L252.9,297.3 L252.8,297.3 L252.7,297.4 L252.6,297.4 L252.5,297.4 L252.4,297.5 L252.2,297.5 L252.1,297.5 L252.0,297.6 L251.9,297.6 L251.8,297.7 L251.7,297.7 L251.6,297.7 L251.5,297.8 L251.4,297.8 L251.3,297.8 L251.2,297.9 L251.0,297.9 L250.9,297.9 L250.8,298.0 L250.7,298.0 L250.6,298.0 L250.5,298.1 L250.4,298.1 L250.3,298.2 L250.2,298.2 L250.1,298.2 L250.0,298.3 L249.8,298.3 L249.7,298.3 L249.6,298.4 L249.5,298.4 L249.4,298.4 L249.3,298.5 L249.2,298.5 L249.1,298.5 L249.0,298.6 L248.9,298.6 L248.8,298.7 L248.6,298.7 L248.5,298.7 L248.4,298.8 L248.3,298.8 L248.2,298.8 L248.1,298.9 L248.0,298.9 L247.9,298.9 L247.8,299.0 L247.7,299.0 L247.5,299.0 L247.4,299.1 L247.3,299.1 L247.2,299.1 L247.1,299.2 L247.0,299.2 L246.9,299.2 L246.8,299.3 L246.7,299.3 L246.6,299.3 L246.4,299.4 L246.3,299.4 L246.2,299.4 L246.1,299.5 L246.0,299.5 L245.9,299.5 L245.8,299.6 L245.7,299.6 L245.6,299.7 L245.5,299.7 L245.4,299.7 L245.2,299.8 L245.1,299.8 L245.0,299.8 L244.9,299.9 L244.8,299.9 L244.7,299.9 L244.6,300.0 L244.5,300.0 L244.4,300.0 L244.3,300.1 L244.1,300.1 L244.0,300.1 L243.9,300.2 L243.8,300.2 L243.7,300.2 L243.6,300.3 L243.5,300.3 L243.4,300.3 L243.3,300.4 L243.1,300.4 L243.0,300.4 L242.9,300.4 L242.8,300.5 L242.7,300.5 L242.6,300.5 L242.5,300.6 L242.4,300.6 L242.3,300.6 L242.2,300.7 L242.0,300.7 L241.9,300.7 L241.8,300.8 L241.7,300.8 L241.6,300.8 L241.5,300.9 L241.4,300.9 L241.3,300.9 L241.2,301.0 L241.1,301.0 L240.9,301.0 L240.8,301.1 L240.7,301.1 L240.6,301.1 L240.5,301.2 L240.4,301.2 L240.3,301.2 L240.2,301.3 L240.1,301.3 L240.0,301.3 L239.8,301.4 L239.7,301.4 L239.6,301.4 L239.5,301.5 L239.4,301.5 L239.3,301.5 L239.2,301.6 L239.1,301.6 L239.0,301.6 L238.8,301.7 L238.7,301.7 L238.6,301.7 L238.5,301.7 L238.4,301.8 L238.3,301.8 L238.2,301.8 L238.1,301.9 L238.0,301.9 L237.9,301.9 L237.7,302.0 L237.6,302.0 L237.5,302.0 L237.4,302.1 L237.3,302.1 L237.2,302.1 L237.1,302.2 L237.0,302.2 L236.9,302.2 L236.7,302.3 L236.6,302.3 L236.5,302.3 L236.4,302.4 L236.3,302.4 L236.2,302.4 L236.1,302.5 L236.0,302.5 L235.9,302.5 L235.8,302.5 L235.6,302.6 L235.5,302.6 L235.4,302.6 L235.3,302.7 L235.2,302.7 L235.1,302.7 L235.0,302.8 L234.9,302.8 L234.8,302.8 L234.7,302.9 L234.5,302.9 L234.4,302.9 L234.3,303.0 L234.2,303.0 L234.1,303.0 L234.0,303.1 L233.9,303.1 L233.8,303.1 L233.7,303.2 L233.5,303.2 L233.4,303.2 L233.3,303.2 L233.2,303.3 L233.1,303.3 L233.0,303.3 L232.9,303.4 L232.8,303.4 L232.7,303.4 L232.6,303.5 L232.4,303.5 L232.3,303.5 L232.2,303.6 L232.1,303.6 L232.0,303.6 L231.9,303.7 L231.8,303.7 L231.7,303.7 L231.6,303.8 L231.4,303.8 L231.3,303.8 L231.2,303.8 L231.1,303.9 L231.0,303.9 L230.9,303.9 L230.8,304.0 L230.7,304.0 L230.6,304.0 L230.4,304.1 L230.3,304.1 L230.2,304.1 L230.1,304.2 L230.0,304.2 L229.9,304.2 L229.8,304.2 L229.7,304.3 L229.6,304.3 L229.4,304.3 L229.3,304.4 L229.2,304.4 L229.1,304.4 L229.0,304.5 L228.9,304.5 L228.8,304.5 L228.7,304.5 L228.6,304.6 L228.5,304.6 L228.3,304.6 L228.2,304.7 L228.1,304.7 L228.0,304.7 L227.9,304.8 L227.8,304.8 L227.7,304.8 L227.6,304.8 L227.4,304.9 L227.3,304.9 L227.2,304.9 L227.1,305.0 L227.0,305.0 L226.9,305.0 L226.8,305.0 L226.7,305.1 L226.6,305.1 L226.4,305.1 L226.3,305.2 L226.2,305.2 L226.1,305.2 L226.0,305.2 L225.9,305.3 L225.8,305.3 L225.7,305.3 L225.6,305.4 L225.4,305.4 L225.3,305.4 L225.2,305.4 L225.1,305.5 L225.0,305.5 L224.9,305.5 L224.8,305.6 L224.7,305.6 L224.6,305.6 L224.4,305.6 L224.3,305.7 L224.2,305.7 L224.1,305.7 L224.0,305.8 L223.9,305.8 L223.8,305.8 L223.7,305.8 L223.5,305.9 L223.4,305.9 L223.3,305.9 L223.2,305.9 L223.1,306.0 L223.0,306.0 L222.9,306.0 L222.8,306.1 L222.7,306.1 L222.5,306.1 L222.4,306.1 L222.3,306.2 L222.2,306.2 L222.1,306.2 L222.0,306.3 L221.9,306.3 L221.8,306.3 L221.7,306.3 L221.5,306.4 L221.4,306.4 L221.3,306.4 L221.2,306.4 L221.1,306.5 L221.0,306.5 L220.9,306.5 L220.8,306.6 L220.6,306.6 L220.5,306.6 L220.4,306.6 L220.3,306.7 L220.2,306.7 L220.1,306.7 L220.0,306.7 L219.9,306.8 L219.8,306.8 L219.6,306.8 L219.5,306.9 L219.4,306.9 L219.3,306.9 L219.2,306.9 L219.1,307.0 L219.0,307.0 L218.9,307.0 L218.8,307.1 L218.6,307.1 L218.5,307.1 L218.4,307.1 L218.3,307.2 L218.2,307.2 L218.1,307.2 L218.0,307.2 L217.9,307.3 L217.7,307.3 L217.6,307.3 L217.5,307.4 L217.4,307.4 L217.3,307.4 L217.2,307.4 L217.1,307.5 L217.0,307.5 L216.9,307.5 L216.7,307.6 L216.6,307.6 L216.5,307.6 L216.4,307.6 L216.3,307.7 L216.2,307.7 L216.1,307.7 L216.0,307.8 L215.8,307.8 L215.7,307.8 L215.6,307.8 L215.5,307.9 L215.4,307.9 L215.3,307.9 L215.2,307.9 L215.1,308.0 L215.0,308.0 L214.8,308.0 L214.7,308.1 L214.6,308.1 L214.5,308.1 L214.4,308.1 L214.3,308.2 L214.2,308.2 L214.1,308.2 L214.0,308.2 L213.8,308.3 L213.7,308.3 L213.6,308.3 L213.5,308.4 L213.4,308.4 L213.3,308.4 L213.2,308.4 L213.1,308.5 L212.9,308.5 L212.8,308.5 L212.7,308.5 L212.6,308.6 L212.5,308.6 L212.4,308.6 L212.3,308.7 L212.2,308.7 L212.1,308.7 L211.9,308.7 L211.8,308.8 L211.7,308.8 L211.6,308.8 L211.5,308.8 L211.4,308.9 L211.3,308.9 L211.2,308.9 L211.0,308.9 L210.9,309.0 L210.8,309.0 L210.7,309.0 L210.6,309.0 L210.5,309.1 L210.4,309.1 L210.3,309.1 L210.1,309.2 L210.0,309.2 L209.9,309.2 L209.8,309.2 L209.7,309.3 L209.6,309.3 L209.5,309.3 L209.4,309.3 L209.3,309.4 L209.1,309.4 L209.0,309.4 L208.9,309.4 L208.8,309.5 L208.7,309.5 L208.6,309.5 L208.5,309.5 L208.4,309.5 L208.2,309.6 L208.1,309.6 L208.0,309.6 L207.9,309.6 L207.8,309.7 L207.7,309.7 L207.6,309.7 L207.5,309.7 L207.3,309.8 L207.2,309.8 L207.1,309.8 L207.0,309.8 L206.9,309.9 L206.8,309.9 L206.7,309.9 L206.6,309.9 L206.4,310.0 L206.3,310.0 L206.2,310.0 L206.1,310.0 L206.0,310.1 L205.9,310.1 L205.8,310.1 L205.7,310.1 L205.5,310.2 L205.4,310.2 L205.3,310.2 L205.2,310.2 L205.1,310.2 L205.0,310.3 L204.9,310.3 L204.8,310.3 L204.6,310.3 L204.5,310.4 L204.4,310.4 L204.3,310.4 L204.2,310.4 L204.1,310.5 L204.0,310.5 L203.9,310.5 L203.7,310.5 L203.6,310.6 L203.5,310.6 L203.4,310.6 L203.3,310.6 L203.2,310.7 L203.1,310.7 L203.0,310.7 L202.8,310.7 L202.7,310.8 L202.6,310.8 L202.5,310.8 L202.4,310.8 L202.3,310.9 L202.2,310.9 L202.1,310.9 L201.9,310.9 L201.8,310.9 L201.7,311.0 L201.6,311.0 L201.5,311.0 L201.4,311.0 L201.3,311.1 L201.2,311.1 L201.0,311.1 L200.9,311.1 L200.8,311.2 L200.7,311.2 L200.6,311.2 L200.5,311.2 L200.4,311.3 L200.3,311.3 L200.1,311.3 L200.0,311.3 L199.9,311.4 L199.8,311.4 L199.7,311.4 L199.6,311.4 L199.5,311.4 L199.4,311.5 L199.2,311.5 L199.1,311.5 L199.0,311.5 L198.9,311.6 L198.8,311.6 L198.7,311.6 L198.6,311.6 L198.5,311.7 L198.3,311.7 L198.2,311.7 L198.1,311.7 L198.0,311.7 L197.9,311.8 L197.8,311.8 L197.7,311.8 L197.6,311.8 L197.4,311.9 L197.3,311.9 L197.2,311.9 L197.1,311.9 L197.0,311.9 L196.9,312.0 L196.8,312.0 L196.6,312.0 L196.5,312.0 L196.4,312.1 L196.3,312.1 L196.2,312.1 L196.1,312.1 L196.0,312.1 L195.9,312.2 L195.7,312.2 L195.6,312.2 L195.5,312.2 L195.4,312.3 L195.3,312.3 L195.2,312.3 L195.1,312.3 L195.0,312.3 L194.8,312.4 L194.7,312.4 L194.6,312.4 L194.5,312.4 L194.4,312.4 L194.3,312.5 L194.2,312.5 L194.1,312.5 L193.9,312.5 L193.8,312.6 L193.7,312.6 L193.6,312.6 L193.5,312.6 L193.4,312.6 L193.3,312.7 L193.1,312.7 L193.0,312.7 L192.9,312.7 L192.8,312.7 L192.7,312.8 L192.6,312.8 L192.5,312.8 L192.4,312.8 L192.2,312.8 L192.1,312.9 L192.0,312.9 L191.9,312.9 L191.8,312.9 L191.7,313.0 L191.6,313.0 L191.5,313.0 L191.3,313.0 L191.2,313.0 L191.1,313.1 L191.0,313.1 L190.9,313.1 L190.8,313.1 L190.7,313.1 L190.5,313.2 L190.4,313.2 L190.3,313.2 L190.2,313.2 L190.1,313.2 L190.0,313.3 L189.9,313.3 L189.8,313.3 L189.6,313.3 L189.5,313.3 L189.4,313.4 L189.3,313.4 L189.2,313.4 L189.1,313.4 L189.0,313.4 L188.8,313.5 L188.7,313.5 L188.6,313.5 L188.5,313.5 L188.4,313.5 L188.3,313.6 L188.2,313.6 L188.1,313.6 L187.9,313.6 L187.8,313.6 L187.7,313.7 L187.6,313.7 L187.5,313.7 L187.4,313.7 L187.3,313.7 L187.1,313.8 L187.0,313.8 L186.9,313.8 L186.8,313.8 L186.7,313.8 L186.6,313.9 L186.5,313.9 L186.4,313.9 L186.2,313.9 L186.1,313.9 L186.0,314.0 L185.9,314.0 L185.8,314.0 L185.7,314.0 L185.6,314.0 L185.4,314.0 L185.3,314.1 L185.2,314.1 L185.1,314.1 L185.0,314.1 L184.9,314.1 L184.8,314.2 L184.7,314.2 L184.5,314.2 L184.4,314.2 L184.3,314.2 L184.2,314.3 L184.1,314.3 L184.0,314.3 L183.9,314.3 L183.7,314.3 L183.6,314.4 L183.5,314.4 L183.4,314.4 L183.3,314.4 L183.2,314.4 L183.1,314.4 L183.0,314.5 L182.8,314.5 L182.7,314.5 L182.6,314.5 L182.5,314.5 L182.4,314.6 L182.3,314.6 L182.2,314.6 L182.0,314.6 L181.9,314.6 L181.8,314.6 L181.7,314.7 L181.6,314.7 L181.5,314.7 L181.4,314.7 L181.3,314.7 L181.1,314.8 L181.0,314.8 L180.9,314.8 L180.8,314.8 L180.7,314.8 L180.6,314.8 L180.5,314.9 L180.3,314.9 L180.2,314.9 L180.1,314.9 L180.0,314.9 L179.9,315.0 L179.8,315.0 L179.7,315.0 L179.5,315.0 L179.4,315.0 L179.3,315.0 L179.2,315.1 L179.1,315.1 L179.0,315.1 L178.9,315.1 L178.8,315.1 L178.6,315.1 L178.5,315.2 L178.4,315.2 L178.3,315.2 L178.2,315.2 L178.1,315.2 L178.0,315.3 L177.8,315.3 L177.7,315.3 L177.6,315.3 L177.5,315.3 L177.4,315.3 L177.3,315.4 L177.2,315.4 L177.0,315.4 L176.9,315.4 L176.8,315.4 L176.7,315.4 L176.6,315.5 L176.5,315.5 L176.4,315.5 L176.2,315.5 L176.1,315.5 L176.0,315.5 L175.9,315.6 L175.8,315.6 L175.7,315.6 L175.6,315.6 L175.5,315.6 L175.3,315.6 L175.2,315.7 L175.1,315.7 L175.0,315.7 L174.9,315.7 L174.8,315.7 L174.7,315.7 L174.5,315.8 L174.4,315.8 L174.3,315.8 L174.2,315.8 L174.1,315.8 L174.0,315.8 L173.9,315.9 L173.7,315.9 L173.6,315.9 L173.5,315.9 L173.4,315.9 L173.3,315.9 L173.2,315.9 L173.1,316.0 L172.9,316.0 L172.8,316.0 L172.7,316.0 L172.6,316.0 L172.5,316.0 L172.4,316.1 L172.3,316.1 L172.2,316.1 L172.0,316.1 L171.9,316.1 L171.8,316.1 L171.7,316.1 L171.6,316.2 L171.5,316.2 L171.4,316.2 L171.2,316.2 L171.1,316.2 L171.0,316.2 L170.9,316.3 L170.8,316.3 L170.7,316.3 L170.6,316.3 L170.4,316.3 L170.3,316.3 L170.2,316.3 L170.1,316.4 L170.0,316.4 L169.9,316.4 L169.8,316.4 L169.6,316.4 L169.5,316.4 L169.4,316.5 L169.3,316.5 L169.2,316.5 L169.1,316.5 L169.0,316.5 L168.8,316.5 L168.7,316.5 L168.6,316.6 L168.5,316.6 L168.4,316.6 L168.3,316.6 L168.2,316.6 L168.0,316.6 L167.9,316.6 L167.8,316.7 L167.7,316.7 L167.6,316.7 L167.5,316.7 L167.4,316.7 L167.2,316.7 L167.1,316.7 L167.0,316.8 L166.9,316.8 L166.8,316.8 L166.7,316.8 L166.6,316.8 L166.4,316.8 L166.3,316.8 L166.2,316.9 L166.1,316.9 L166.0,316.9 L165.9,316.9 L165.8,316.9 L165.7,316.9 L165.5,316.9 L165.4,317.0 L165.3,317.0 L165.2,317.0 L165.1,317.0 L165.0,317.0 L164.9,317.0 L164.7,317.0 L164.6,317.0 L164.5,317.1 L164.4,317.1 L164.3,317.1 L164.2,317.1 L164.1,317.1 L163.9,317.1 L163.8,317.1 L163.7,317.2 L163.6,317.2 L163.5,317.2 L163.4,317.2 L163.3,317.2 L163.1,317.2 L163.0,317.2 L162.9,317.3 L162.8,317.3 L162.7,317.3 L162.6,317.3 L162.5,317.3 L162.3,317.3 L162.2,317.3 L162.1,317.4 L162.0,317.4 L161.9,317.4 L161.8,317.4 L161.7,317.4 L161.5,317.4 L161.4,317.4 L161.3,317.5 L161.2,317.5 L161.1,317.5 L161.0,317.5 L160.9,317.5 L160.7,317.5 L160.6,317.5 L160.5,317.6 L160.4,317.6 L160.3,317.6 L160.2,317.6 L160.1,317.6 L159.9,317.6 L159.8,317.6 L159.7,317.7 L159.6,317.7 L159.5,317.7 L159.4,317.7 L159.3,317.7 L159.1,317.7 L159.0,317.7 L158.9,317.8 L158.8,317.8 L158.7,317.8 L158.6,317.8 L158.5,317.8 L158.3,317.8 L158.2,317.8 L158.1,317.9 L158.0,317.9 L157.9,317.9 L157.8,317.9 L157.7,317.9 L157.5,317.9 L157.4,317.9 L157.3,318.0 L157.2,318.0 L157.1,318.0 L157.0,318.0 L156.9,318.0 L156.7,318.0 L156.6,318.0 L156.5,318.1 L156.4,318.1 L156.3,318.1 L156.2,318.1 L156.1,318.1 L156.0,318.1 L155.8,318.1 L155.7,318.2 L155.6,318.2 L155.5,318.2 L155.4,318.2 L155.3,318.2 L155.2,318.2 L155.0,318.3 L154.9,318.3 L154.8,318.3 L154.7,318.3 L154.6,318.3 L154.5,318.3 L154.4,318.3 L154.2,318.4 L154.1,318.4 L154.0,318.4 L153.9,318.4 L153.8,318.4 L153.7,318.4 L153.6,318.4 L153.4,318.5 L153.3,318.5 L153.2,318.5 L153.1,318.5 L153.0,318.5 L152.9,318.5 L152.8,318.5 L152.6,318.6 L152.5,318.6 L152.4,318.6 L152.3,318.6 L152.2,318.6 L152.1,318.6 L152.0,318.6 L151.8,318.6 L151.7,318.7 L151.6,318.7 L151.5,318.7 L151.4,318.7 L151.3,318.7 L151.2,318.7 L151.0,318.7 L150.9,318.8 L150.8,318.8 L150.7,318.8 L150.6,318.8 L150.5,318.8 L150.4,318.8 L150.2,318.8 L150.1,318.9 L150.0,318.9 L149.9,318.9 L149.8,318.9 L149.7,318.9 L149.6,318.9 L149.4,318.9 L149.3,319.0 L149.2,319.0 L149.1,319.0 L149.0,319.0 L148.9,319.0 L148.8,319.0 L148.6,319.0 L148.5,319.0 L148.4,319.1 L148.3,319.1 L148.2,319.1 L148.1,319.1 L148.0,319.1 L147.8,319.1 L147.7,319.1 L147.6,319.2 L147.5,319.2 L147.4,319.2 L147.3,319.2 L147.2,319.2 L147.0,319.2 L146.9,319.2 L146.8,319.2 L146.7,319.3 L146.6,319.3 L146.5,319.3 L146.4,319.3 L146.2,319.3 L146.1,319.3 L146.0,319.3 L145.9,319.4 L145.8,319.4 L145.7,319.4 L145.6,319.4 L145.4,319.4 L145.3,319.4 L145.2,319.4 L145.1,319.4 L145.0,319.5 L144.9,319.5 L144.8,319.5 L144.6,319.5 L144.5,319.5 L144.4,319.5 L144.3,319.5 L144.2,319.6 L144.1,319.6 L144.0,319.6 L143.9,319.6 L143.7,319.6 L143.6,319.6 L143.5,319.6 L143.4,319.6 L143.3,319.7 L143.2,319.7 L143.1,319.7 L142.9,319.7 L142.8,319.7 L142.7,319.7 L142.6,319.7 L142.5,319.8 L142.4,319.8 L142.3,319.8 L142.1,319.8 L142.0,319.8 L141.9,319.8 L141.8,319.8 L141.7,319.9 L141.6,319.9 L141.5,319.9 L141.3,319.9 L141.2,319.9 L141.1,319.9 L141.0,319.9 L140.9,320.0 L140.8,320.0 L140.7,320.0 L140.5,320.0 L140.4,320.0 L140.3,320.0 L140.2,320.0 L140.1,320.0 L140.0,320.1 L139.9,320.1 L139.7,320.1 L139.6,320.1 L139.5,320.1 L139.4,320.1 L139.3,320.1 L139.2,320.2 L139.1,320.2 L138.9,320.2 L138.8,320.2 L138.7,320.2 L138.6,320.2 L138.5,320.2 L138.4,320.2 L138.3,320.2 L138.1,320.3 L138.0,320.3 L137.9,320.3 L137.8,320.3 L137.7,320.3 L137.6,320.3 L137.5,320.3 L137.3,320.3 L137.2,320.3 L137.1,320.4 L137.0,320.4 L136.9,320.4 L136.8,320.4 L136.7,320.4 L136.5,320.4 L136.4,320.5 L136.3,320.5 L136.2,320.5 L136.1,320.5 L136.0,320.6 L135.9,320.6 L135.8,320.6 L135.6,320.6 L135.5,320.7 L135.4,320.7 L135.3,320.8 L135.2,320.8 L135.1,320.9 L135.0,321.0 L135.0,321.0 L134.9,321.1 L134.8,321.2 L134.7,321.3 L134.7,321.4 L134.6,321.5 L134.6,321.6 L134.6,321.7 L134.5,321.8 L134.5,322.0 L134.5,322.1 L134.5,322.2 L134.5,322.3 L134.5,322.4 L134.5,322.5 L134.5,322.6 L134.5,322.8 L134.6,322.9 L134.6,323.0 L134.6,323.1 L134.7,323.2 L134.8,323.3 L134.8,323.4 L134.9,323.5 L134.9,323.6 L135.0,323.7 L135.1,323.8 L135.1,323.9 L135.2,324.0 L135.3,324.0 L135.3,324.1 L135.4,324.2 L135.5,324.3 L135.5,324.4 L135.6,324.5 L135.7,324.6 L135.8,324.7 L135.8,324.8 L135.9,324.9 L136.0,325.0 L136.0,325.1 L136.1,325.1 L136.2,325.2 L136.2,325.3 L136.3,325.4 L136.4,325.5 L136.5,325.6 L136.5,325.7 L136.6,325.8 L136.7,325.9 L136.7,326.0 L136.8,326.0 L136.9,326.1 L137.0,326.2 L137.0,326.3 L137.1,326.4 L137.2,326.5 L137.3,326.6 L137.3,326.7 L137.4,326.8 L137.5,326.8 L137.5,326.9 L137.6,327.0 L137.7,327.1 L137.8,327.2 L137.8,327.3 L137.9,327.4 L138.0,327.5 L138.0,327.6 L138.1,327.7 L138.2,327.7 L138.3,327.8 L138.3,327.9 L138.4,328.0 L138.5,328.1 L138.6,328.2 L138.6,328.3 L138.7,328.4 L138.8,328.5 L138.8,328.5 L138.9,328.6 L139.0,328.7 L139.1,328.8 L139.1,328.9 L139.2,329.0 L139.3,329.1 L139.3,329.2 L139.4,329.3 L139.5,329.4 L139.6,329.4 L139.6,329.5 L139.7,329.6 L139.8,329.7 L139.8,329.8 L139.9,329.9 L140.0,330.0 L140.1,330.1 L140.1,330.2 L140.2,330.3 L140.3,330.4 L140.3,330.4 L140.4,330.5 L140.5,330.6 L140.5,330.7 L140.6,330.8 L140.7,330.9 L140.8,331.0 L140.8,331.1 L140.9,331.2 L141.0,331.3 L141.0,331.4 L141.1,331.5 L141.2,331.5 L141.2,331.6 L141.3,331.7 L141.4,331.8 L141.4,331.9 L141.5,332.0 L141.6,332.1 L141.6,332.2 L141.7,332.3 L141.8,332.4 L141.9,332.5 L141.9,332.6 L142.0,332.7 L142.1,332.8 L142.1,332.8 L142.2,332.9 L142.3,333.0 L142.3,333.1 L142.4,333.2 L142.5,333.3 L142.5,333.4 L142.6,333.5 L142.7,333.6 L142.7,333.7 L142.8,333.8 L142.8,333.9 L142.9,334.0 L143.0,334.1 L143.0,334.2 L143.1,334.3 L143.2,334.4 L143.2,334.4 L143.3,334.5 L143.4,334.6 L143.4,334.7 L143.5,334.8 L143.6,334.9 L143.6,335.0 L143.7,335.1 L143.8,335.2 L143.8,335.3 L143.9,335.4 L143.9,335.5 L144.0,335.6 L144.1,335.7 L144.1,335.8 L144.2,335.9 L144.3,336.0 L144.3,336.1 L144.4,336.2 L144.5,336.3 L144.5,336.4 L144.6,336.5 L144.6,336.6 L144.7,336.7 L144.8,336.7 L144.8,336.8 L144.9,336.9 L145.0,337.0 L145.0,337.1 L145.1,337.2 L145.1,337.3 L145.2,337.4 L145.3,337.5 L145.3,337.6 L145.4,337.7 L145.4,337.8 L145.5,337.9 L145.6,338.0 L145.6,338.1 L145.7,338.2 L145.8,338.3 L145.8,338.4 L145.9,338.5 L145.9,338.6 L146.0,338.7 L146.1,338.8 L146.1,338.9 L146.2,339.0 L146.2,339.1 L146.3,339.2 L146.4,339.3 L146.4,339.4 L146.5,339.5 L146.5,339.6 L146.6,339.7 L146.7,339.8 L146.7,339.9 L146.8,340.0 L146.8,340.1 L146.9,340.2 L147.0,340.3 L147.0,340.4 L147.1,340.5 L147.1,340.6 L147.2,340.7 L147.2,340.8 L147.3,340.9 L147.3,341.0 L147.4,341.1 L147.4,341.2 L147.5,341.3 L147.5,341.4 L147.5,341.5 L147.5,341.6 L147.6,341.7 L147.6,341.9 L147.6,342.0 L147.6,342.1 L147.6,342.2 L147.6,342.3 L147.5,342.4 L147.5,342.5 L147.5,342.7 L147.5,342.8 L147.4,342.9 L147.4,343.0 L147.4,343.1 L147.3,343.2 L147.3,343.3 L147.2,343.4 L147.2,343.5 L147.1,343.6 L147.1,343.7 L147.0,343.8 L146.9,343.9 L146.8,344.0 L146.8,344.0 L146.7,344.1 L146.6,344.2 L146.5,344.3 L146.4,344.3 L146.3,344.4 L146.2,344.4 L146.1,344.5 L146.0,344.5 L145.9,344.5 L145.7,344.6 L145.6,344.6 L145.5,344.6 L145.4,344.6 L145.3,344.6 L145.2,344.6 L145.1,344.7 L144.9,344.7 L144.8,344.7 L144.7,344.6 L144.6,344.6 L144.5,344.6 L144.4,344.6 L144.3,344.6 L144.1,344.6 L144.0,344.6 L143.9,344.6 L143.8,344.6 L143.7,344.6 L143.6,344.5 L143.5,344.5 L143.3,344.5 L143.2,344.5 L143.1,344.5 L143.0,344.5 L142.9,344.5 L142.8,344.5 L142.6,344.5 L142.5,344.5 L142.4,344.5 L142.3,344.5 L142.2,344.5 L142.1,344.4 L142.0,344.4 L141.8,344.4 L141.7,344.4 L141.6,344.4 L141.5,344.4 L141.4,344.4 L141.3,344.4 L141.2,344.4 L141.0,344.4 L140.9,344.4 L140.8,344.4 L140.7,344.4 L140.6,344.4 L140.5,344.4 L140.4,344.4 L140.2,344.4 L140.1,344.4 L140.0,344.4 L139.9,344.4 L139.8,344.4 L139.7,344.4 L139.5,344.4 L139.4,344.3 L139.3,344.3 L139.2,344.3 L139.1,344.3 L139.0,344.3 L138.9,344.3 L138.7,344.3 L138.6,344.2 L138.5,344.2 L138.4,344.2 L138.3,344.2 L138.2,344.1 L138.1,344.1 L138.0,344.1 L137.9,344.0 L137.7,344.0 L137.6,344.0 L137.5,343.9 L137.4,343.9 L137.3,343.8 L137.2,343.8 L137.1,343.7 L137.0,343.7 L136.9,343.6 L136.8,343.5 L136.7,343.5 L136.6,343.4 L136.6,343.3 L136.5,343.2 L136.4,343.1 L136.3,343.1 L136.3,343.0 L136.2,342.9 L136.1,342.8 L136.1,342.7 L136.0,342.6 L136.0,342.5 L135.9,342.4 L135.9,342.3 L135.8,342.2 L135.8,342.1 L135.7,341.9 L135.7,341.8 L135.7,341.7 L135.6,341.6 L135.6,341.5 L135.5,341.4 L135.5,341.3 L135.5,341.2 L135.4,341.1 L135.4,341.0 L135.4,340.9 L135.3,340.8 L135.3,340.6 L135.2,340.5 L135.2,340.4 L135.2,340.3 L135.1,340.2 L135.1,340.1 L135.0,340.0 L135.0,339.9 L135.0,339.8 L134.9,339.7 L134.9,339.6 L134.8,339.5 L134.8,339.3 L134.8,339.2 L134.7,339.1 L134.7,339.0 L134.6,338.9 L134.6,338.8 L134.5,338.7 L134.5,338.6 L134.4,338.5 L134.4,338.4 L134.3,338.3 L134.3,338.2 L134.2,338.1 L134.2,338.0 L134.1,337.9 L134.1,337.8 L134.0,337.7 L134.0,337.6 L133.9,337.5 L133.9,337.4 L133.8,337.3 L133.8,337.2 L133.7,337.1 L133.6,337.0 L133.6,336.9 L133.5,336.8 L133.4,336.7 L133.4,336.6 L133.3,336.5 L133.2,336.4 L133.1,336.3 L133.1,336.3 L133.0,336.2 L132.9,336.1 L132.8,336.0 L132.8,335.9 L132.7,335.8 L132.6,335.7 L132.5,335.7 L132.4,335.6 L132.4,335.5 L132.3,335.4 L132.2,335.3 L132.1,335.3 L132.0,335.2 L131.9,335.1 L131.9,335.0 L131.8,334.9 L131.7,334.9 L131.6,334.8 L131.5,334.7 L131.5,334.6 L131.4,334.5 L131.3,334.4 L131.2,334.3 L131.2,334.3 L131.1,334.2 L131.0,334.1 L130.9,334.0 L130.9,333.9 L130.8,333.8 L130.7,333.7 L130.6,333.7 L130.5,333.6 L130.5,333.5 L130.4,333.4 L130.3,333.3 L130.2,333.2 L130.1,333.2 L130.1,333.1 L130.0,333.0 L129.9,332.9 L129.8,332.8 L129.7,332.8 L129.7,332.7 L129.6,332.6 L129.5,332.5 L129.4,332.5 L129.3,332.4 L129.2,332.3 L129.1,332.2 L129.0,332.2 L128.9,332.1 L128.8,332.1 L128.7,332.0 L128.6,331.9 L128.5,331.9 L128.4,331.8 L128.3,331.8 L128.2,331.7 L128.1,331.7 L128.0,331.6 L127.9,331.6 L127.8,331.5 L127.7,331.5 L127.6,331.5 L127.5,331.4 L127.4,331.4 L127.2,331.4 L127.1,331.4 L127.0,331.3 L126.9,331.3 L126.8,331.3 L126.7,331.3 L126.6,331.2 L126.5,331.2 L126.3,331.2 L126.2,331.2 L126.1,331.2 L126.0,331.2 L125.9,331.1 L125.8,331.1 L125.7,331.1 L125.5,331.1 L125.4,331.1 L125.3,331.1 L125.2,331.1 L125.1,331.0 L125.0,331.0 L124.9,331.0 L124.8,331.0 L124.6,331.0 L124.5,331.0 L124.4,331.0 L124.3,330.9 L124.2,330.9 L124.1,330.9 L124.0,330.9 L123.8,330.9 L123.7,330.9 L123.6,330.9 L123.5,330.9 L123.4,330.9 L123.3,330.9 L123.1,330.9 L123.0,330.9 L122.9,330.9 L122.8,330.9 L122.7,330.9 L122.6,330.9 L122.5,330.9 L122.3,331.0 L122.2,331.0 L122.1,331.0 L122.0,331.1 L121.9,331.1 L121.8,331.2 L121.7,331.3 L121.7,331.3 L121.6,331.4 L121.5,331.5 L121.5,331.6 L121.4,331.7 L121.4,331.8 L121.4,332.0 L121.4,332.1 L121.3,332.2 L121.3,332.3 L121.3,332.4 L121.3,332.5 L121.3,332.6 L121.4,332.8 L121.4,332.9 L121.4,333.0 L121.4,333.1 L121.5,333.2 L121.5,333.3 L121.6,333.4 L121.6,333.5 L121.6,333.6 L121.7,333.7 L121.7,333.8 L121.8,333.9 L121.9,334.0 L121.9,334.1 L122.0,334.2 L122.0,334.3 L122.1,334.4 L122.1,334.5 L122.2,334.6 L122.2,334.7 L122.3,334.8 L122.4,334.9 L122.4,335.0 L122.5,335.1 L122.5,335.2 L122.6,335.3 L122.6,335.4 L122.7,335.5 L122.8,335.6 L122.8,335.7 L122.9,335.8 L122.9,335.9 L123.0,336.0 L123.1,336.1 L123.1,336.2 L123.2,336.3 L123.2,336.4 L123.3,336.5 L123.3,336.6 L123.4,336.7 L123.5,336.8 L123.5,336.9 L123.6,337.0 L123.6,337.1 L123.7,337.2 L123.7,337.3 L123.8,337.4 L123.9,337.5 L123.9,337.6 L124.0,337.7 L124.0,337.8 L124.1,337.9 L124.2,338.0 L124.2,338.1 L124.3,338.2 L124.3,338.3 L124.4,338.4 L124.5,338.5 L124.5,338.6 L124.6,338.7 L124.6,338.8 L124.7,338.9 L124.7,339.0 L124.8,339.1 L124.9,339.2 L124.9,339.3 L125.0,339.4 L125.0,339.5 L125.1,339.6 L125.2,339.7 L125.2,339.8 L125.3,339.9 L125.3,340.0 L125.4,340.1 L125.5,340.2 L125.5,340.3 L125.6,340.4 L125.6,340.5 L125.7,340.6 L125.7,340.7 L125.8,340.8 L125.9,340.9 L125.9,341.0 L126.0,341.1 L126.0,341.2 L126.1,341.3 L126.2,341.4 L126.2,341.5 L126.3,341.6 L126.3,341.7 L126.4,341.8 L126.5,341.9 L126.5,342.0 L126.6,342.1 L126.6,342.2 L126.7,342.3 L126.8,342.4 L126.8,342.5 L126.9,342.6 L126.9,342.7 L127.0,342.8 L127.0,342.9 L127.1,343.0 L127.2,343.1 L127.2,343.2 L127.3,343.3 L127.3,343.4 L127.4,343.5 L127.5,343.5 L127.5,343.6 L127.6,343.7 L127.6,343.8 L127.7,343.9 L127.8,344.0 L127.8,344.1 L127.9,344.2 L127.9,344.3 L128.0,344.4 L128.0,344.5 L128.1,344.6 L128.2,344.7 L128.2,344.8 L128.3,344.9 L128.3,345.0 L128.4,345.1 L128.5,345.2 L128.5,345.3 L128.6,345.4 L128.6,345.5 L128.7,345.6 L128.8,345.7 L128.8,345.8 L128.9,345.9 L128.9,346.0 L129.0,346.1 L129.0,346.2 L129.1,346.3 L129.2,346.4 L129.2,346.5 L129.3,346.6 L129.3,346.7 L129.4,346.8 L129.5,346.9 L129.5,347.0 L129.6,347.1 L129.6,347.2 L129.7,347.3 L129.8,347.4 L129.8,347.5 L129.9,347.6 L129.9,347.7 L130.0,347.8 L130.1,347.9 L130.1,348.0 L130.2,348.1 L130.2,348.2 L130.3,348.3 L130.4,348.4 L130.4,348.5 L130.5,348.6 L130.5,348.7 L130.6,348.8 L130.7,348.9 L130.7,349.0 L130.8,349.1 L130.8,349.2 L130.9,349.3 L130.9,349.4 L131.0,349.5 L131.1,349.6 L131.1,349.7 L131.2,349.8 L131.2,349.9 L131.3,350.0 L131.3,350.1 L131.4,350.2 L131.4,350.3 L131.5,350.4 L131.5,350.5 L131.6,350.6 L131.6,350.7 L131.7,350.8 L131.7,350.9 L131.8,351.0 L131.8,351.1 L131.9,351.2 L131.9,351.3 L132.0,351.4 L132.0,351.5 L132.1,351.6 L132.1,351.7 L132.1,351.9 L132.2,352.0 L132.2,352.1 L132.2,352.2 L132.3,352.3 L132.3,352.4 L132.3,352.5 L132.4,352.6 L132.4,352.7 L132.4,352.8 L132.5,353.0 L132.5,353.1 L132.5,353.2 L132.5,353.3 L132.5,353.4 L132.6,353.5 L132.6,353.6 L132.6,353.8 L132.6,353.9 L132.6,354.0 L132.6,354.1 L132.6,354.2 L132.6,354.3 L132.6,354.4 L132.6,354.6 L132.6,354.7 L132.6,354.8 L132.6,354.9 L132.6,355.0 L132.6,355.1 L132.6,355.2 L132.6,355.4 L132.5,355.5 L132.5,355.6 L132.5,355.7 L132.5,355.8 L132.5,355.9 L132.5,356.0 L132.4,356.2 L132.4,356.3 L132.4,356.4 L132.4,356.5 L132.3,356.6 L132.3,356.7 L132.3,356.8 L132.2,356.9 L132.2,357.0 L132.2,357.2 L132.1,357.3 L132.1,357.4 L132.1,357.5 L132.1,357.6 L132.0,357.7 L132.0,357.8 L131.9,357.9 L131.9,358.0 L131.9,358.1 L131.8,358.3 L131.8,358.4 L131.8,358.5 L131.7,358.6 L131.7,358.7 L131.7,358.8 L131.6,358.9 L131.6,359.0 L131.5,359.1 L131.5,359.2 L131.5,359.3 L131.4,359.4 L131.4,359.6 L131.3,359.7 L131.3,359.8 L131.3,359.9 L131.2,360.0 L131.2,360.1 L131.2,360.2 L131.1,360.3 L131.1,360.4 L131.0,360.5 L131.0,360.6 L130.9,360.7 L130.9,360.9 L130.9,361.0 L130.8,361.1 L130.8,361.2 L130.7,361.3 L130.7,361.4 L130.6,361.5 L130.6,361.6 L130.5,361.7 L130.5,361.8 L130.5,361.9 L130.4,362.0 L130.4,362.1 L130.3,362.2 L130.2,362.3 L130.2,362.4 L130.1,362.5 L130.1,362.6 L130.0,362.7 L130.0,362.8 L129.9,362.9 L129.8,363.0 L129.8,363.1 L129.7,363.2 L129.6,363.3 L129.6,363.4 L129.5,363.5 L129.4,363.6 L129.3,363.6 L129.3,363.7 L129.2,363.8 L129.1,363.9 L129.0,364.0 L128.9,364.1 L128.9,364.1 L128.8,364.2 L128.7,364.3 L128.6,364.4 L128.5,364.4 L128.4,364.5 L128.3,364.6 L128.2,364.6 L128.1,364.7 L128.0,364.8 L128.0,364.8 L127.9,364.9 L127.8,365.0 L127.7,365.0 L127.6,365.1 L127.5,365.1 L127.4,365.2 L127.2,365.2 L127.1,365.3 L127.0,365.3 L126.9,365.4 L126.8,365.4 L126.7,365.5 L126.6,365.5 L126.5,365.6 L126.4,365.6 L126.3,365.7 L126.2,365.7 L126.1,365.7 L126.0,365.8 L125.9,365.8 L125.8,365.9 L125.7,365.9 L125.6,366.0 L125.5,366.0 L125.3,366.0 L125.2,366.1 L125.1,366.1 L125.0,366.2 L124.9,366.2 L124.8,366.3 L124.7,366.3 L124.6,366.4 L124.5,366.4 L124.4,366.5 L124.3,366.5 L124.2,366.6 L124.1,366.6 L124.0,366.7 L123.9,366.7 L123.8,366.8 L123.7,366.8 L123.6,366.9 L123.5,366.9 L123.4,367.0 L123.3,367.0 L123.2,367.1 L123.1,367.1 L123.0,367.2 L122.8,367.2 L122.7,367.3 L122.6,367.3 L122.5,367.4 L122.4,367.4 L122.3,367.5 L122.2,367.5 L122.1,367.6 L122.0,367.6 L121.9,367.7 L121.8,367.7 L121.7,367.8 L121.6,367.8 L121.5,367.9 L121.4,367.9 L121.3,368.0 L121.2,368.0 L121.1,368.1 L121.0,368.1 L120.9,368.2 L120.8,368.2 L120.7,368.3 L120.6,368.3 L120.5,368.3 L120.4,368.4 L120.2,368.4 L120.1,368.5 L120.0,368.5 L119.9,368.5 L119.8,368.6 L119.7,368.6 L119.6,368.6 L119.5,368.7 L119.4,368.7 L119.2,368.7 L119.1,368.7 L119.0,368.8 L118.9,368.8 L118.8,368.8 L118.7,368.8 L118.6,368.8 L118.5,368.8 L118.3,368.8 L118.2,368.8 L118.1,368.8 L118.0,368.8 L117.9,368.8 L117.8,368.8 L117.6,368.8 L117.5,368.8 L117.4,368.8 L117.3,368.8 L117.2,368.8 L117.1,368.8 L117.0,368.8 L116.8,368.8 L116.7,368.8 L116.6,368.8 L116.5,368.8 L116.4,368.8 L116.3,368.8 L116.2,368.7 L116.0,368.7 L115.9,368.7 L115.8,368.7 L115.7,368.7 L115.6,368.7 L115.5,368.7 L115.4,368.7 L115.2,368.7 L115.1,368.7 L115.0,368.6 L114.9,368.6 L114.8,368.6 L114.7,368.6 L114.6,368.6 L114.4,368.6 L114.3,368.6 L114.2,368.5 L114.1,368.5 L114.0,368.5 L113.9,368.5 L113.8,368.4 L113.7,368.4 L113.5,368.4 L113.4,368.4 L113.3,368.3 L113.2,368.3 L113.1,368.3 L113.0,368.2 L112.9,368.2 L112.8,368.2 L112.7,368.1 L112.5,368.1 L112.4,368.1 L112.3,368.0 L112.2,368.0 L112.1,368.0 L112.0,367.9 L111.9,367.9 L111.8,367.8 L111.7,367.8 L111.6,367.8 L111.5,367.7 L111.3,367.7 L111.2,367.7 L111.1,367.6 L111.0,367.6 L110.9,367.6 L110.8,367.5 L110.7,367.5 L110.6,367.5 L110.5,367.4 L110.4,367.4 L110.3,367.4 L110.1,367.3 L110.0,367.3 L109.9,367.2 L109.8,367.2 L109.7,367.2 L109.6,367.1 L109.5,367.1 L109.4,367.0 L109.3,367.0 L109.2,366.9 L109.1,366.9 L109.0,366.9 L108.9,366.8 L108.8,366.8 L108.7,366.7 L108.5,366.7 L108.4,366.6 L108.3,366.6 L108.2,366.5 L108.1,366.4 L108.0,366.4 L107.9,366.3 L107.8,366.3 L107.7,366.2 L107.6,366.1 L107.6,366.1 L107.5,366.0 L107.4,366.0 L107.3,365.9 L107.2,365.8 L107.1,365.8 L107.0,365.7 L106.9,365.6 L106.8,365.6 L106.7,365.5 L106.6,365.4 L106.5,365.3 L106.4,365.3 L106.4,365.2 L106.3,365.1 L106.2,365.0 L106.1,365.0 L106.0,364.9 L105.9,364.8 L105.8,364.7 L105.8,364.7 L105.7,364.6 L105.6,364.5 L105.5,364.4 L105.4,364.3 L105.4,364.2 L105.3,364.2 L105.2,364.1 L105.1,364.0 L105.1,363.9 L105.0,363.8 L104.9,363.7 L104.8,363.6 L104.8,363.5 L104.7,363.4 L104.6,363.4 L104.6,363.3 L104.5,363.2 L104.4,363.1 L104.3,363.0 L104.3,362.9 L104.2,362.8 L104.1,362.7 L104.1,362.6 L104.0,362.5 L104.0,362.4 L103.9,362.3 L103.8,362.2 L103.8,362.1 L103.7,362.0 L103.6,361.9 L103.6,361.9 L103.5,361.8 L103.4,361.7 L103.4,361.6 L103.3,361.5 L103.2,361.4 L103.2,361.3 L103.1,361.2 L103.1,361.1 L103.0,361.0 L102.9,360.9 L102.9,360.8 L102.8,360.7 L102.7,360.6 L102.7,360.5 L102.6,360.4 L102.5,360.3 L102.5,360.2 L102.4,360.1 L102.3,360.0 L102.3,359.9 L102.2,359.9 L102.1,359.8 L102.1,359.7 L102.0,359.6 L101.9,359.5 L101.9,359.4 L101.8,359.3 L101.7,359.2 L101.6,359.1 L101.6,359.0 L101.5,358.9 L101.4,358.9 L101.4,358.8 L101.3,358.7 L101.2,358.6 L101.2,358.5 L101.1,358.4 L101.0,358.3 L100.9,358.2 L100.9,358.1 L100.8,358.0 L100.7,357.9 L100.7,357.9 L100.6,357.8 L100.5,357.7 L100.4,357.6 L100.4,357.5 L100.3,357.4 L100.2,357.3 L100.2,357.2 L100.1,357.1 L100.0,357.0 L100.0,356.9 L99.9,356.8 L99.8,356.8 L99.8,356.7 L99.7,356.6 L99.6,356.5 L99.5,356.4 L99.5,356.3 L99.4,356.2 L99.3,356.1 L99.3,356.0 L99.2,355.9 L99.1,355.8 L99.1,355.7 L99.0,355.6 L98.9,355.6 L98.9,355.5 L98.8,355.4 L98.7,355.3 L98.7,355.2 L98.6,355.1 L98.5,355.0 L98.4,354.9 L98.4,354.8 L98.3,354.7 L98.2,354.6 L98.2,354.5 L98.1,354.5 L98.0,354.4 L98.0,354.3 L97.9,354.2 L97.8,354.1 L97.8,354.0 L97.7,353.9 L97.6,353.8 L97.5,353.7 L97.5,353.6 L97.4,353.5 L97.3,353.4 L97.3,353.4 L97.2,353.3 L97.1,353.2 L97.1,353.1 L97.0,353.0 L96.9,352.9 L96.8,352.8 L96.8,352.7 L96.7,352.6 L96.6,352.5 L96.6,352.4 L96.5,352.4 L96.4,352.3 L96.4,352.2 L96.3,352.1 L96.2,352.0 L96.1,351.9 L96.1,351.8 L96.0,351.7 L95.9,351.6 L95.9,351.5 L95.8,351.4 L95.7,351.4 L95.6,351.3 L95.6,351.2 L95.5,351.1 L95.4,351.0 L95.4,350.9 L95.3,350.8 L95.2,350.7 L95.1,350.6 L95.1,350.5 L95.0,350.5 L94.9,350.4 L94.9,350.3 L94.8,350.2 L94.7,350.1 L94.6,350.0 L94.6,349.9 L94.5,349.8 L94.4,349.7 L94.3,349.7 L94.3,349.6 L94.2,349.5 L94.1,349.4 L94.1,349.3 L94.0,349.2 L93.9,349.1 L93.8,349.0 L93.8,348.9 L93.7,348.9 L93.6,348.8 L93.5,348.7 L93.5,348.6 L93.4,348.5 L93.3,348.4 L93.2,348.3 L93.2,348.2 L93.1,348.2 L93.0,348.1 L92.9,348.0 L92.9,347.9 L92.8,347.8 L92.7,347.7 L92.6,347.6 L92.6,347.5 L92.5,347.5 L92.4,347.4 L92.3,347.3 L92.3,347.2 L92.2,347.1 L92.1,347.0 L92.0,346.9 L91.9,346.9 L91.9,346.8 L91.8,346.7 L91.7,346.6 L91.6,346.5 L91.5,346.5 L91.5,346.4 L91.4,346.3 L91.3,346.2 L91.2,346.1 L91.1,346.1 L91.0,346.0 L90.9,345.9 L90.8,345.9 L90.8,345.8 L90.7,345.7 L90.6,345.7 L90.5,345.6 L90.3,345.6 L90.2,345.5 L90.1,345.5 L90.0,345.4 L89.9,345.4 L89.8,345.4 L89.7,345.4 L89.6,345.3 L89.5,345.3 L89.4,345.3 L89.2,345.3 L89.1,345.2 L89.0,345.2 L88.9,345.2 L88.8,345.2 L88.7,345.2 L88.6,345.2 L88.4,345.2 L88.3,345.2 L88.2,345.2 L88.1,345.2 L88.0,345.2 L87.9,345.2 L87.8,345.2 L87.6,345.2 L87.5,345.2 L87.4,345.3 L87.3,345.3 L87.2,345.3 L87.1,345.3 L87.0,345.3 L86.8,345.4 L86.7,345.4 L86.6,345.4 L86.5,345.4 L86.4,345.5 L86.3,345.5 L86.2,345.5 L86.1,345.6 L86.0,345.6 L85.8,345.6 L85.7,345.7 L85.6,345.7 L85.5,345.7 L85.4,345.8 L85.3,345.8 L85.2,345.9 L85.1,345.9 L85.0,346.0 L84.9,346.0 L84.8,346.0 L84.7,346.1 L84.6,346.1 L84.5,346.2 L84.3,346.2 L84.2,346.3 L84.1,346.3 L84.0,346.4 L83.9,346.4 L83.8,346.4 L83.7,346.5 L83.6,346.5 L83.5,346.6 L83.4,346.6 L83.3,346.7 L83.2,346.7 L83.1,346.8 L83.0,346.8 L82.9,346.9 L82.8,346.9 L82.7,347.0 L82.6,347.0 L82.5,347.1 L82.4,347.1 L82.2,347.2 L82.1,347.2 L82.0,347.2 L81.9,347.3 L81.8,347.3 L81.7,347.4 L81.6,347.4 L81.5,347.5 L81.4,347.5 L81.3,347.6 L81.2,347.6 L81.1,347.7 L81.0,347.7 L80.9,347.8 L80.8,347.8 L80.7,347.9 L80.6,347.9 L80.5,348.0 L80.4,348.0 L80.3,348.1 L80.2,348.1 L80.1,348.2 L80.0,348.2 L79.9,348.3 L79.7,348.3 L79.6,348.4 L79.5,348.4 L79.4,348.5 L79.3,348.5 L79.2,348.6 L79.1,348.6 L79.0,348.7 L78.9,348.7 L78.8,348.7 L78.7,348.8 L78.6,348.8 L78.5,348.9 L78.4,348.9 L78.3,349.0 L78.2,349.0 L78.1,349.1 L78.0,349.1 L77.9,349.2 L77.8,349.2 L77.7,349.3 L77.5,349.3 L77.4,349.4 L77.3,349.4 L77.2,349.5 L77.1,349.5 L77.0,349.6 L76.9,349.6 L76.8,349.6 L76.7,349.7 L76.6,349.7 L76.5,349.8 L76.4,349.8 L76.3,349.9 L76.2,349.9 L76.1,350.0 L76.0,350.0 L75.9,350.1 L75.8,350.1 L75.7,350.2 L75.6,350.2 L75.4,350.3 L75.3,350.3 L75.2,350.4 L75.1,350.4 L75.0,350.4 L74.9,350.5 L74.8,350.5 L74.7,350.6 L74.6,350.6 L74.5,350.7 L74.4,350.7 L74.3,350.8 L74.2,350.8 L74.1,350.9 L74.0,350.9 L73.9,351.0 L73.8,351.0 L73.7,351.1 L73.6,351.1 L73.5,351.2 L73.4,351.2 L73.2,351.2 L73.1,351.3 L73.0,351.3 L72.9,351.4 L72.8,351.4 L72.7,351.5 L72.6,351.5 L72.5,351.6 L72.4,351.6 L72.3,351.7 L72.2,351.7 L72.1,351.8 L72.0,351.8 L71.9,351.9 L71.8,351.9 L71.7,352.0 L71.6,352.0 L71.5,352.1 L71.4,352.1 L71.3,352.1 L71.2,352.2 L71.0,352.2 L70.9,352.3 L70.8,352.3 L70.7,352.4 L70.6,352.4 L70.5,352.5 L70.4,352.5 L70.3,352.6 L70.2,352.6 L70.1,352.7 L70.0,352.7 L69.9,352.8 L69.8,352.8 L69.7,352.9 L69.6,352.9 L69.5,353.0 L69.4,353.0 L69.3,353.0 L69.2,353.1 L69.1,353.1 L69.0,353.2 L68.8,353.2 L68.7,353.3 L68.6,353.3 L68.5,353.4 L68.4,353.4 L68.3,353.5 L68.2,353.5 L68.1,353.6 L68.0,353.6 L67.9,353.7 L67.8,353.7 L67.7,353.8 L67.6,353.8 L67.5,353.9 L67.4,353.9 L67.3,354.0 L67.2,354.0 L67.1,354.0 L67.0,354.1 L66.9,354.1 L66.8,354.2 L66.6,354.2 L66.5,354.3 L66.4,354.3 L66.3,354.4 L66.2,354.4 L66.1,354.5 L66.0,354.5 L65.9,354.6 L65.8,354.6 L65.7,354.7 L65.6,354.7 L65.5,354.8 L65.4,354.8 L65.3,354.9 L65.2,354.9 L65.1,355.0 L65.0,355.0 L64.9,355.0 L64.8,355.1 L64.7,355.1 L64.6,355.2 L64.4,355.2 L64.3,355.3 L64.2,355.3 L64.1,355.4 L64.0,355.4 L63.9,355.5 L63.8,355.5 L63.7,355.6 L63.6,355.6 L63.5,355.7 L63.4,355.7 L63.3,355.8 L63.2,355.8 L63.1,355.8 L63.0,355.9 L62.9,355.9 L62.8,356.0 L62.7,356.0 L62.6,356.1 L62.5,356.1 L62.3,356.2 L62.2,356.2 L62.1,356.3 L62.0,356.3 L61.9,356.3 L61.8,356.4 L61.7,356.4 L61.6,356.5 L61.5,356.5 L61.4,356.6 L61.3,356.6 L61.2,356.7 L61.1,356.7 L61.0,356.8 L60.9,356.8 L60.8,356.9 L60.7,356.9 L60.6,357.0 L60.5,357.0 L60.4,357.1 L60.3,357.2 L60.2,357.2 L60.1,357.3 L60.0,357.4 L59.9,357.4 L59.8,357.5 L59.7,357.6 L59.7,357.7 L59.6,357.8 L59.5,357.8 L59.4,357.9 L59.3,358.0 L59.3,358.1 L59.2,358.2 L59.1,358.3 L59.1,358.4 L59.0,358.5 L58.9,358.6 L58.9,358.7 L58.8,358.8 L58.8,358.9 L58.7,359.0 L58.7,359.1 L58.6,359.2 L58.6,359.3 L58.5,359.4 L58.5,359.5 L58.5,359.6 L58.5,359.7 L58.5,359.9 L58.5,360.0 L58.6,360.1 L58.6,360.2 L58.7,360.3 L58.7,360.4 L58.8,360.5 L58.9,360.6 L59.0,360.6 L59.1,360.7 L59.1,360.8 L59.2,360.9 L59.3,360.9 L59.4,361.0 L59.5,361.1 L59.6,361.2 L59.7,361.2 L59.8,361.3 L59.9,361.4 L60.0,361.4 L60.0,361.5 L60.1,361.6 L60.2,361.6 L60.3,361.7 L60.4,361.8 L60.5,361.8 L60.6,361.9 L60.7,362.0 L60.8,362.1 L60.9,362.1 L61.0,362.2 L61.1,362.3 L61.1,362.3 L61.2,362.4 L61.3,362.5 L61.4,362.5 L61.5,362.6 L61.6,362.7 L61.7,362.8 L61.8,362.8 L61.9,362.9 L62.0,363.0 L62.1,363.0 L62.1,363.1 L62.2,363.2 L62.3,363.2 L62.4,363.3 L62.5,363.4 L62.6,363.5 L62.7,363.5 L62.8,363.6 L62.9,363.7 L63.0,363.7 L63.1,363.8 L63.1,363.9 L63.2,364.0 L63.3,364.0 L63.4,364.1 L63.5,364.2 L63.6,364.2 L63.7,364.3 L63.8,364.4 L63.9,364.4 L64.0,364.5 L64.1,364.6 L64.2,364.6 L64.2,364.7 L64.3,364.8 L64.4,364.9 L64.5,364.9 L64.6,365.0 L64.7,365.1 L64.8,365.1 L64.9,365.2 L65.0,365.3 L65.1,365.3 L65.2,365.4 L65.2,365.5 L65.3,365.6 L65.4,365.6 L65.5,365.7 L65.6,365.8 L65.7,365.8 L65.8,365.9 L65.9,366.0 L66.0,366.0 L66.1,366.1 L66.2,366.2 L66.3,366.3 L66.3,366.3 L66.4,366.4 L66.5,366.5 L66.6,366.5 L66.7,366.6 L66.8,366.7 L66.9,366.7 L67.0,366.8 L67.1,366.9 L67.2,366.9 L67.3,367.0 L67.4,367.1 L67.4,367.2 L67.5,367.2 L67.6,367.3 L67.7,367.4 L67.8,367.4 L67.9,367.5 L68.0,367.6 L68.1,367.6 L68.2,367.7 L68.3,367.8 L68.4,367.9 L68.4,367.9 L68.5,368.0 L68.6,368.1 L68.7,368.1 L68.8,368.2 L68.9,368.3 L69.0,368.3 L69.1,368.4 L69.2,368.5 L69.3,368.6 L69.4,368.6 L69.5,368.7 L69.5,368.8 L69.6,368.8 L69.7,368.9 L69.8,369.0 L69.9,369.0 L70.0,369.1 L70.1,369.2 L70.2,369.3 L70.3,369.3 L70.4,369.4 L70.5,369.5 L70.6,369.5 L70.6,369.6 L70.7,369.7 L70.8,369.7 L70.9,369.8 L71.0,369.9 L71.1,369.9 L71.2,370.0 L71.3,370.1 L71.4,370.2 L71.5,370.2 Z" fill="none" stroke="#5f6673" stroke-width="3" stroke-linejoin="round"/><circle cx="71.6" cy="370.3" r="3.4" fill="#1d2330" stroke="#ffffff" stroke-width="1"/><text x="163.8" y="491.1" font-family="DejaVu Sans" font-size="9.5" fill="#5f6673" text-anchor="middle" font-weight="normal">5.5 km · road 11.1–27.6 m</text><rect x="337.5" y="210.82337120641645" width="327.5" height="290.2675653070664" rx="3" fill="#f5f6f8"/><text x="501.2" y="226.8" font-family="DejaVu Sans" font-size="11" fill="#1d2330" text-anchor="middle" font-weight="bold">Validation</text><text x="501.2" y="239.8" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle" font-weight="normal">generated, validation map 80</text><path d="M596.2,357.5 L596.2,357.4 L596.1,357.3 L596.0,357.2 L596.0,357.1 L595.9,357.0 L595.9,356.9 L595.8,356.8 L595.7,356.7 L595.7,356.6 L595.6,356.5 L595.6,356.4 L595.5,356.3 L595.5,356.2 L595.4,356.1 L595.4,356.0 L595.3,355.9 L595.3,355.8 L595.2,355.7 L595.2,355.6 L595.1,355.5 L595.1,355.4 L595.0,355.2 L595.0,355.1 L594.9,355.0 L594.9,354.9 L594.9,354.8 L594.8,354.7 L594.8,354.6 L594.7,354.5 L594.7,354.4 L594.7,354.3 L594.6,354.2 L594.6,354.1 L594.6,353.9 L594.5,353.8 L594.5,353.7 L594.5,353.6 L594.5,353.5 L594.4,353.4 L594.4,353.3 L594.4,353.2 L594.4,353.1 L594.3,352.9 L594.3,352.8 L594.3,352.7 L594.3,352.6 L594.2,352.5 L594.2,352.4 L594.2,352.3 L594.2,352.1 L594.2,352.0 L594.2,351.9 L594.2,351.8 L594.1,351.7 L594.1,351.6 L594.1,351.5 L594.1,351.3 L594.1,351.2 L594.1,351.1 L594.1,351.0 L594.1,350.9 L594.1,350.8 L594.1,350.7 L594.1,350.5 L594.1,350.4 L594.1,350.3 L594.1,350.2 L594.1,350.1 L594.1,350.0 L594.1,349.9 L594.1,349.7 L594.1,349.6 L594.1,349.5 L594.2,349.4 L594.2,349.3 L594.2,349.2 L594.2,349.1 L594.2,348.9 L594.2,348.8 L594.2,348.7 L594.3,348.6 L594.3,348.5 L594.3,348.4 L594.3,348.3 L594.3,348.1 L594.4,348.0 L594.4,347.9 L594.4,347.8 L594.4,347.7 L594.5,347.6 L594.5,347.5 L594.5,347.4 L594.6,347.2 L594.6,347.1 L594.6,347.0 L594.7,346.9 L594.7,346.8 L594.7,346.7 L594.8,346.6 L594.8,346.5 L594.8,346.4 L594.9,346.3 L594.9,346.2 L595.0,346.1 L595.0,345.9 L595.1,345.8 L595.1,345.7 L595.1,345.6 L595.2,345.5 L595.2,345.4 L595.3,345.3 L595.3,345.2 L595.3,345.1 L595.4,345.0 L595.4,344.9 L595.5,344.8 L595.5,344.7 L595.6,344.6 L595.6,344.4 L595.6,344.3 L595.7,344.2 L595.7,344.1 L595.8,344.0 L595.8,343.9 L595.9,343.8 L595.9,343.7 L595.9,343.6 L596.0,343.5 L596.0,343.4 L596.1,343.3 L596.1,343.2 L596.2,343.1 L596.2,343.0 L596.2,342.8 L596.3,342.7 L596.3,342.6 L596.4,342.5 L596.4,342.4 L596.4,342.3 L596.5,342.2 L596.5,342.1 L596.6,342.0 L596.6,341.9 L596.7,341.8 L596.7,341.7 L596.7,341.6 L596.8,341.5 L596.8,341.3 L596.9,341.2 L596.9,341.1 L597.0,341.0 L597.0,340.9 L597.0,340.8 L597.1,340.7 L597.1,340.6 L597.2,340.5 L597.2,340.4 L597.3,340.3 L597.3,340.2 L597.3,340.1 L597.4,340.0 L597.4,339.9 L597.5,339.7 L597.5,339.6 L597.5,339.5 L597.6,339.4 L597.6,339.3 L597.7,339.2 L597.7,339.1 L597.8,339.0 L597.8,338.9 L597.8,338.8 L597.9,338.7 L597.9,338.6 L598.0,338.5 L598.0,338.4 L598.1,338.2 L598.1,338.1 L598.1,338.0 L598.2,337.9 L598.2,337.8 L598.3,337.7 L598.3,337.6 L598.4,337.5 L598.4,337.4 L598.4,337.3 L598.5,337.2 L598.5,337.1 L598.6,337.0 L598.6,336.9 L598.6,336.7 L598.7,336.6 L598.7,336.5 L598.8,336.4 L598.8,336.3 L598.9,336.2 L598.9,336.1 L598.9,336.0 L599.0,335.9 L599.0,335.8 L599.1,335.7 L599.1,335.6 L599.2,335.5 L599.2,335.4 L599.2,335.3 L599.3,335.1 L599.3,335.0 L599.4,334.9 L599.4,334.8 L599.4,334.7 L599.5,334.6 L599.5,334.5 L599.6,334.4 L599.6,334.3 L599.7,334.2 L599.7,334.1 L599.7,334.0 L599.8,333.9 L599.8,333.8 L599.9,333.6 L599.9,333.5 L600.0,333.4 L600.0,333.3 L600.0,333.2 L600.1,333.1 L600.1,333.0 L600.2,332.9 L600.2,332.8 L600.3,332.7 L600.3,332.6 L600.3,332.5 L600.4,332.4 L600.4,332.3 L600.5,332.2 L600.5,332.0 L600.5,331.9 L600.6,331.8 L600.6,331.7 L600.7,331.6 L600.7,331.5 L600.8,331.4 L600.8,331.3 L600.8,331.2 L600.9,331.1 L600.9,331.0 L601.0,330.9 L601.0,330.8 L601.1,330.7 L601.1,330.5 L601.1,330.4 L601.2,330.3 L601.2,330.2 L601.3,330.1 L601.3,330.0 L601.4,329.9 L601.4,329.8 L601.4,329.7 L601.5,329.6 L601.5,329.5 L601.6,329.4 L601.6,329.3 L601.6,329.2 L601.7,329.0 L601.7,328.9 L601.8,328.8 L601.8,328.7 L601.9,328.6 L601.9,328.5 L601.9,328.4 L602.0,328.3 L602.0,328.2 L602.1,328.1 L602.1,328.0 L602.2,327.9 L602.2,327.8 L602.2,327.7 L602.3,327.6 L602.3,327.4 L602.4,327.3 L602.4,327.2 L602.5,327.1 L602.5,327.0 L602.5,326.9 L602.6,326.8 L602.6,326.7 L602.7,326.6 L602.7,326.5 L602.7,326.4 L602.8,326.3 L602.8,326.2 L602.9,326.1 L602.9,325.9 L603.0,325.8 L603.0,325.7 L603.0,325.6 L603.1,325.5 L603.1,325.4 L603.2,325.3 L603.2,325.2 L603.3,325.1 L603.3,325.0 L603.3,324.9 L603.4,324.8 L603.4,324.7 L603.5,324.6 L603.5,324.5 L603.6,324.3 L603.6,324.2 L603.6,324.1 L603.7,324.0 L603.7,323.9 L603.8,323.8 L603.8,323.7 L603.8,323.6 L603.9,323.5 L603.9,323.4 L604.0,323.3 L604.0,323.2 L604.0,323.1 L604.1,322.9 L604.1,322.8 L604.1,322.7 L604.2,322.6 L604.2,322.5 L604.3,322.4 L604.3,322.3 L604.3,322.2 L604.3,322.1 L604.4,322.0 L604.4,321.8 L604.4,321.7 L604.5,321.6 L604.5,321.5 L604.5,321.4 L604.5,321.3 L604.6,321.2 L604.6,321.1 L604.6,320.9 L604.6,320.8 L604.7,320.7 L604.7,320.6 L604.7,320.5 L604.7,320.4 L604.7,320.3 L604.8,320.2 L604.8,320.0 L604.8,319.9 L604.8,319.8 L604.8,319.7 L604.8,319.6 L604.9,319.5 L604.9,319.4 L604.9,319.2 L604.9,319.1 L604.9,319.0 L604.9,318.9 L604.9,318.8 L604.9,318.7 L604.9,318.6 L604.9,318.4 L605.0,318.3 L605.0,318.2 L605.0,318.1 L605.0,318.0 L605.0,317.9 L605.0,317.8 L605.0,317.6 L605.0,317.5 L605.0,317.4 L605.0,317.3 L605.0,317.2 L605.0,317.1 L605.0,316.9 L605.0,316.8 L605.0,316.7 L605.0,316.6 L604.9,316.5 L604.9,316.4 L604.9,316.3 L604.9,316.1 L604.9,316.0 L604.9,315.9 L604.9,315.8 L604.9,315.7 L604.9,315.6 L604.9,315.5 L604.8,315.3 L604.8,315.2 L604.8,315.1 L604.8,315.0 L604.8,314.9 L604.8,314.8 L604.8,314.7 L604.7,314.5 L604.7,314.4 L604.7,314.3 L604.7,314.2 L604.6,314.1 L604.6,314.0 L604.6,313.9 L604.6,313.8 L604.6,313.6 L604.5,313.5 L604.5,313.4 L604.5,313.3 L604.4,313.2 L604.4,313.1 L604.4,313.0 L604.4,312.9 L604.3,312.8 L604.3,312.6 L604.3,312.5 L604.2,312.4 L604.2,312.3 L604.2,312.2 L604.1,312.1 L604.1,312.0 L604.1,311.9 L604.0,311.8 L604.0,311.7 L603.9,311.5 L603.9,311.4 L603.9,311.3 L603.8,311.2 L603.8,311.1 L603.7,311.0 L603.7,310.9 L603.6,310.8 L603.6,310.7 L603.6,310.6 L603.5,310.5 L603.5,310.4 L603.4,310.3 L603.4,310.2 L603.3,310.1 L603.3,310.0 L603.2,309.9 L603.2,309.8 L603.1,309.7 L603.1,309.5 L603.0,309.4 L603.0,309.3 L602.9,309.2 L602.8,309.1 L602.8,309.0 L602.7,308.9 L602.7,308.8 L602.6,308.7 L602.6,308.6 L602.5,308.5 L602.4,308.4 L602.4,308.3 L602.3,308.3 L602.3,308.2 L602.2,308.1 L602.1,308.0 L602.1,307.9 L602.0,307.8 L601.9,307.7 L601.9,307.6 L601.8,307.5 L601.7,307.4 L601.7,307.3 L601.6,307.2 L601.5,307.1 L601.5,307.0 L601.4,306.9 L601.3,306.8 L601.3,306.7 L601.2,306.7 L601.1,306.6 L601.1,306.5 L601.0,306.4 L600.9,306.3 L600.8,306.2 L600.8,306.1 L600.7,306.0 L600.6,305.9 L600.6,305.8 L600.5,305.7 L600.4,305.7 L600.3,305.6 L600.3,305.5 L600.2,305.4 L600.1,305.3 L600.1,305.2 L600.0,305.1 L599.9,305.0 L599.9,304.9 L599.8,304.8 L599.7,304.7 L599.6,304.7 L599.6,304.6 L599.5,304.5 L599.4,304.4 L599.4,304.3 L599.3,304.2 L599.2,304.1 L599.2,304.0 L599.1,303.9 L599.0,303.8 L598.9,303.7 L598.9,303.7 L598.8,303.6 L598.7,303.5 L598.7,303.4 L598.6,303.3 L598.5,303.2 L598.4,303.1 L598.4,303.0 L598.3,302.9 L598.2,302.8 L598.2,302.7 L598.1,302.7 L598.0,302.6 L598.0,302.5 L597.9,302.4 L597.8,302.3 L597.7,302.2 L597.7,302.1 L597.6,302.0 L597.5,301.9 L597.5,301.8 L597.4,301.7 L597.3,301.7 L597.2,301.6 L597.2,301.5 L597.1,301.4 L597.0,301.3 L597.0,301.2 L596.9,301.1 L596.8,301.0 L596.8,300.9 L596.7,300.8 L596.6,300.8 L596.5,300.7 L596.5,300.6 L596.4,300.5 L596.3,300.4 L596.3,300.3 L596.2,300.2 L596.1,300.1 L596.0,300.0 L596.0,299.9 L595.9,299.8 L595.8,299.8 L595.8,299.7 L595.7,299.6 L595.6,299.5 L595.6,299.4 L595.5,299.3 L595.4,299.2 L595.3,299.1 L595.3,299.0 L595.2,298.9 L595.1,298.8 L595.1,298.8 L595.0,298.7 L594.9,298.6 L594.8,298.5 L594.7,298.4 L594.7,298.3 L594.6,298.3 L594.5,298.2 L594.4,298.2 L594.3,298.1 L594.2,298.0 L594.1,298.0 L594.0,297.9 L593.9,297.9 L593.8,297.8 L593.7,297.8 L593.6,297.7 L593.5,297.6 L593.4,297.6 L593.3,297.5 L593.2,297.5 L593.1,297.4 L593.0,297.3 L592.9,297.3 L592.8,297.2 L592.7,297.2 L592.6,297.1 L592.5,297.1 L592.4,297.0 L592.3,296.9 L592.2,296.9 L592.1,296.8 L592.0,296.8 L591.9,296.7 L591.8,296.6 L591.7,296.5 L591.6,296.5 L591.6,296.4 L591.5,296.3 L591.4,296.2 L591.4,296.1 L591.3,296.0 L591.2,295.9 L591.2,295.8 L591.1,295.7 L591.0,295.6 L590.9,295.5 L590.9,295.5 L590.8,295.4 L590.7,295.3 L590.7,295.2 L590.6,295.1 L590.5,295.0 L590.4,294.9 L590.4,294.8 L590.3,294.7 L590.2,294.6 L590.2,294.5 L590.1,294.5 L590.0,294.4 L590.0,294.3 L589.9,294.2 L589.8,294.1 L589.8,294.0 L589.7,293.9 L589.7,293.8 L589.7,293.6 L589.6,293.5 L589.6,293.4 L589.6,293.3 L589.5,293.2 L589.5,293.1 L589.5,293.0 L589.5,292.9 L589.4,292.8 L589.4,292.7 L589.4,292.5 L589.3,292.4 L589.3,292.3 L589.3,292.2 L589.2,292.1 L589.2,292.0 L589.2,291.9 L589.1,291.8 L589.1,291.7 L589.1,291.5 L589.0,291.4 L589.0,291.3 L589.0,291.2 L588.9,291.1 L588.9,291.0 L588.9,290.9 L588.8,290.8 L588.8,290.7 L588.7,290.6 L588.6,290.5 L588.6,290.4 L588.5,290.3 L588.4,290.2 L588.4,290.1 L588.3,290.0 L588.2,289.9 L588.2,289.8 L588.1,289.8 L588.0,289.7 L587.9,289.6 L587.9,289.5 L587.8,289.4 L587.7,289.3 L587.7,289.2 L587.6,289.1 L587.5,289.0 L587.5,288.9 L587.4,288.8 L587.3,288.8 L587.2,288.7 L587.2,288.6 L587.1,288.5 L587.0,288.4 L587.0,288.3 L586.9,288.2 L586.8,288.1 L586.7,288.0 L586.7,287.9 L586.6,287.8 L586.5,287.8 L586.5,287.7 L586.4,287.6 L586.3,287.5 L586.3,287.4 L586.2,287.3 L586.1,287.2 L586.0,287.1 L586.0,287.0 L585.9,286.9 L585.8,286.8 L585.8,286.8 L585.7,286.7 L585.6,286.6 L585.5,286.5 L585.5,286.4 L585.4,286.3 L585.3,286.2 L585.3,286.1 L585.2,286.0 L585.1,285.9 L585.1,285.8 L585.0,285.8 L584.9,285.7 L584.8,285.6 L584.8,285.5 L584.7,285.4 L584.6,285.3 L584.6,285.2 L584.5,285.1 L584.4,285.0 L584.4,284.9 L584.3,284.9 L584.2,284.8 L584.1,284.7 L584.1,284.6 L584.0,284.5 L583.9,284.4 L583.9,284.3 L583.8,284.2 L583.7,284.1 L583.6,284.0 L583.6,283.9 L583.5,283.9 L583.4,283.8 L583.4,283.7 L583.3,283.6 L583.2,283.5 L583.2,283.4 L583.1,283.3 L583.0,283.2 L582.9,283.1 L582.9,283.0 L582.8,282.9 L582.7,282.9 L582.7,282.8 L582.6,282.7 L582.5,282.6 L582.4,282.5 L582.4,282.4 L582.3,282.3 L582.2,282.2 L582.2,282.1 L582.1,282.0 L582.0,281.9 L582.0,281.9 L581.9,281.8 L581.8,281.7 L581.7,281.6 L581.7,281.5 L581.6,281.4 L581.5,281.3 L581.5,281.2 L581.4,281.1 L581.3,281.0 L581.3,280.9 L581.2,280.9 L581.1,280.8 L581.0,280.7 L581.0,280.6 L580.9,280.5 L580.8,280.4 L580.8,280.3 L580.7,280.2 L580.6,280.1 L580.5,280.0 L580.5,279.9 L580.4,279.9 L580.3,279.8 L580.3,279.7 L580.2,279.6 L580.1,279.5 L580.1,279.4 L580.0,279.3 L579.9,279.2 L579.8,279.1 L579.8,279.0 L579.7,278.9 L579.6,278.9 L579.6,278.8 L579.5,278.7 L579.4,278.6 L579.3,278.5 L579.3,278.4 L579.2,278.3 L579.1,278.2 L579.1,278.1 L579.0,278.0 L578.9,277.9 L578.9,277.9 L578.8,277.8 L578.7,277.7 L578.6,277.6 L578.6,277.5 L578.5,277.4 L578.4,277.3 L578.4,277.2 L578.3,277.1 L578.2,277.0 L578.1,276.9 L578.1,276.9 L578.0,276.8 L577.9,276.7 L577.9,276.6 L577.8,276.5 L577.7,276.4 L577.7,276.3 L577.6,276.2 L577.5,276.1 L577.4,276.0 L577.4,275.9 L577.3,275.9 L577.2,275.8 L577.2,275.7 L577.1,275.6 L577.0,275.5 L576.9,275.4 L576.9,275.3 L576.8,275.2 L576.7,275.1 L576.6,275.1 L576.6,275.0 L576.5,274.9 L576.4,274.8 L576.3,274.7 L576.2,274.7 L576.1,274.6 L576.1,274.5 L576.0,274.4 L575.9,274.4 L575.8,274.3 L575.7,274.2 L575.6,274.2 L575.5,274.1 L575.4,274.0 L575.3,274.0 L575.2,273.9 L575.1,273.8 L575.0,273.8 L574.9,273.7 L574.8,273.7 L574.7,273.6 L574.6,273.6 L574.5,273.5 L574.4,273.4 L574.3,273.4 L574.2,273.3 L574.1,273.3 L574.0,273.2 L573.9,273.1 L573.8,273.1 L573.7,273.0 L573.6,273.0 L573.5,272.9 L573.4,272.9 L573.3,272.8 L573.2,272.7 L573.1,272.7 L573.0,272.6 L572.9,272.6 L572.8,272.5 L572.7,272.5 L572.6,272.4 L572.5,272.3 L572.4,272.3 L572.3,272.2 L572.2,272.2 L572.1,272.1 L572.0,272.0 L571.9,272.0 L571.8,271.9 L571.7,271.9 L571.6,271.8 L571.5,271.8 L571.4,271.7 L571.3,271.6 L571.2,271.6 L571.1,271.5 L571.0,271.5 L570.9,271.4 L570.8,271.4 L570.7,271.3 L570.6,271.2 L570.5,271.2 L570.4,271.1 L570.4,271.1 L570.3,271.0 L570.2,270.9 L570.1,270.9 L570.0,270.8 L569.9,270.8 L569.8,270.7 L569.7,270.7 L569.6,270.6 L569.5,270.5 L569.4,270.5 L569.3,270.4 L569.2,270.4 L569.1,270.3 L569.0,270.3 L568.9,270.2 L568.8,270.1 L568.7,270.1 L568.6,270.0 L568.5,270.0 L568.4,269.9 L568.3,269.8 L568.2,269.8 L568.1,269.7 L568.0,269.7 L567.9,269.6 L567.8,269.6 L567.7,269.5 L567.6,269.4 L567.5,269.4 L567.4,269.3 L567.3,269.3 L567.2,269.2 L567.1,269.2 L567.0,269.1 L566.9,269.0 L566.8,269.0 L566.7,268.9 L566.6,268.9 L566.5,268.8 L566.4,268.7 L566.3,268.7 L566.2,268.6 L566.1,268.6 L566.0,268.5 L565.9,268.5 L565.8,268.4 L565.7,268.3 L565.6,268.3 L565.5,268.2 L565.4,268.2 L565.3,268.1 L565.2,268.1 L565.1,268.0 L565.0,267.9 L564.9,267.9 L564.8,267.8 L564.7,267.8 L564.6,267.7 L564.5,267.6 L564.4,267.6 L564.3,267.5 L564.2,267.5 L564.1,267.4 L564.0,267.4 L563.9,267.3 L563.8,267.2 L563.7,267.2 L563.6,267.1 L563.5,267.1 L563.4,267.0 L563.3,267.0 L563.2,266.9 L563.1,266.8 L563.0,266.8 L562.9,266.7 L562.8,266.7 L562.7,266.6 L562.6,266.5 L562.5,266.5 L562.4,266.4 L562.3,266.4 L562.2,266.3 L562.1,266.3 L562.0,266.2 L561.9,266.1 L561.8,266.1 L561.7,266.0 L561.6,266.0 L561.5,265.9 L561.4,265.9 L561.3,265.8 L561.2,265.7 L561.1,265.7 L561.0,265.6 L560.9,265.6 L560.8,265.5 L560.7,265.4 L560.6,265.4 L560.5,265.3 L560.4,265.3 L560.3,265.2 L560.2,265.2 L560.1,265.1 L560.0,265.0 L559.9,265.0 L559.8,264.9 L559.7,264.9 L559.6,264.8 L559.5,264.8 L559.4,264.7 L559.3,264.6 L559.2,264.6 L559.1,264.5 L559.0,264.5 L558.9,264.4 L558.8,264.3 L558.7,264.3 L558.6,264.2 L558.5,264.2 L558.4,264.1 L558.3,264.1 L558.2,264.0 L558.1,263.9 L558.0,263.9 L557.9,263.8 L557.8,263.8 L557.7,263.7 L557.6,263.7 L557.5,263.6 L557.4,263.5 L557.3,263.5 L557.2,263.4 L557.1,263.4 L557.0,263.3 L556.9,263.2 L556.8,263.2 L556.7,263.1 L556.6,263.1 L556.5,263.0 L556.4,263.0 L556.3,262.9 L556.2,262.8 L556.1,262.8 L556.0,262.7 L555.9,262.7 L555.8,262.6 L555.7,262.6 L555.6,262.5 L555.5,262.4 L555.4,262.4 L555.3,262.3 L555.2,262.3 L555.2,262.2 L555.1,262.1 L555.0,262.1 L554.9,262.0 L554.8,262.0 L554.7,261.9 L554.5,261.9 L554.4,261.8 L554.3,261.8 L554.2,261.8 L554.1,261.7 L554.0,261.7 L553.9,261.7 L553.8,261.7 L553.7,261.7 L553.5,261.6 L553.4,261.6 L553.3,261.6 L553.2,261.6 L553.1,261.5 L553.0,261.5 L552.9,261.5 L552.8,261.5 L552.6,261.5 L552.5,261.4 L552.4,261.4 L552.3,261.4 L552.2,261.4 L552.1,261.3 L552.0,261.3 L551.8,261.3 L551.7,261.3 L551.6,261.3 L551.5,261.2 L551.4,261.2 L551.3,261.2 L551.2,261.2 L551.1,261.2 L550.9,261.1 L550.8,261.1 L550.7,261.1 L550.6,261.0 L550.5,261.0 L550.4,260.9 L550.3,260.9 L550.2,260.8 L550.1,260.8 L550.0,260.7 L549.9,260.7 L549.8,260.6 L549.7,260.5 L549.6,260.5 L549.5,260.4 L549.4,260.4 L549.3,260.3 L549.2,260.3 L549.1,260.2 L549.0,260.1 L548.9,260.1 L548.8,260.0 L548.7,260.0 L548.6,259.9 L548.5,259.8 L548.4,259.8 L548.3,259.7 L548.2,259.7 L548.1,259.6 L548.0,259.6 L547.9,259.5 L547.8,259.4 L547.7,259.4 L547.6,259.3 L547.5,259.3 L547.4,259.2 L547.3,259.1 L547.3,259.0 L547.2,259.0 L547.1,258.9 L547.0,258.8 L547.0,258.7 L546.9,258.6 L546.8,258.5 L546.7,258.4 L546.7,258.3 L546.6,258.2 L546.5,258.2 L546.4,258.1 L546.4,258.0 L546.3,257.9 L546.2,257.8 L546.1,257.7 L546.1,257.6 L546.0,257.5 L545.9,257.5 L545.9,257.4 L545.8,257.3 L545.7,257.2 L545.6,257.1 L545.6,257.0 L545.5,256.9 L545.4,256.8 L545.3,256.8 L545.3,256.7 L545.2,256.6 L545.1,256.5 L545.0,256.4 L545.0,256.3 L544.9,256.2 L544.8,256.2 L544.7,256.1 L544.6,256.0 L544.5,256.0 L544.4,255.9 L544.3,255.9 L544.2,255.8 L544.1,255.8 L544.0,255.7 L543.9,255.6 L543.8,255.6 L543.7,255.5 L543.6,255.5 L543.5,255.4 L543.4,255.4 L543.3,255.3 L543.2,255.2 L543.1,255.2 L543.0,255.1 L542.9,255.1 L542.8,255.0 L542.7,254.9 L542.6,254.9 L542.5,254.8 L542.4,254.8 L542.3,254.7 L542.2,254.7 L542.1,254.6 L542.0,254.5 L541.9,254.5 L541.8,254.4 L541.7,254.4 L541.6,254.3 L541.5,254.3 L541.4,254.2 L541.3,254.1 L541.2,254.1 L541.1,254.0 L541.0,254.0 L540.9,253.9 L540.8,253.8 L540.7,253.8 L540.6,253.7 L540.5,253.7 L540.4,253.6 L540.3,253.6 L540.2,253.5 L540.1,253.4 L540.0,253.4 L539.9,253.3 L539.8,253.3 L539.7,253.2 L539.6,253.2 L539.5,253.1 L539.4,253.0 L539.3,253.0 L539.2,252.9 L539.1,252.9 L539.0,252.8 L538.9,252.7 L538.8,252.7 L538.7,252.6 L538.6,252.6 L538.5,252.5 L538.4,252.5 L538.3,252.4 L538.2,252.3 L538.1,252.3 L538.0,252.2 L537.9,252.2 L537.8,252.1 L537.7,252.1 L537.6,252.0 L537.5,251.9 L537.4,251.9 L537.3,251.8 L537.2,251.8 L537.1,251.7 L537.0,251.6 L536.9,251.6 L536.8,251.5 L536.7,251.5 L536.6,251.4 L536.5,251.4 L536.4,251.3 L536.3,251.2 L536.2,251.2 L536.1,251.1 L536.0,251.1 L535.9,251.0 L535.8,251.0 L535.7,250.9 L535.6,250.8 L535.5,250.8 L535.4,250.7 L535.3,250.7 L535.2,250.6 L535.1,250.5 L535.1,250.5 L535.0,250.4 L534.9,250.4 L534.8,250.3 L534.7,250.3 L534.6,250.2 L534.5,250.1 L534.4,250.1 L534.3,250.0 L534.2,250.0 L534.1,249.9 L534.0,249.9 L533.9,249.8 L533.8,249.7 L533.7,249.7 L533.6,249.6 L533.5,249.6 L533.4,249.5 L533.3,249.4 L533.2,249.4 L533.1,249.3 L533.0,249.3 L532.9,249.2 L532.8,249.2 L532.7,249.1 L532.6,249.1 L532.5,249.0 L532.4,249.0 L532.2,248.9 L532.1,248.9 L532.0,248.8 L531.9,248.8 L531.8,248.7 L531.7,248.7 L531.6,248.6 L531.5,248.6 L531.4,248.5 L531.3,248.5 L531.2,248.4 L531.1,248.4 L531.0,248.3 L530.9,248.3 L530.8,248.3 L530.7,248.2 L530.6,248.2 L530.4,248.1 L530.3,248.1 L530.2,248.1 L530.1,248.0 L530.0,248.0 L529.9,248.0 L529.8,247.9 L529.7,247.9 L529.6,247.9 L529.5,247.8 L529.3,247.8 L529.2,247.8 L529.1,247.8 L529.0,247.7 L528.9,247.7 L528.8,247.7 L528.7,247.7 L528.6,247.6 L528.5,247.6 L528.3,247.6 L528.2,247.6 L528.1,247.5 L528.0,247.5 L527.9,247.5 L527.8,247.5 L527.7,247.5 L527.5,247.4 L527.4,247.4 L527.3,247.4 L527.2,247.4 L527.1,247.4 L527.0,247.4 L526.9,247.4 L526.7,247.3 L526.6,247.3 L526.5,247.3 L526.4,247.3 L526.3,247.3 L526.2,247.3 L526.1,247.3 L525.9,247.3 L525.8,247.3 L525.7,247.3 L525.6,247.3 L525.5,247.3 L525.4,247.3 L525.3,247.3 L525.1,247.3 L525.0,247.3 L524.9,247.3 L524.8,247.3 L524.7,247.3 L524.6,247.3 L524.4,247.3 L524.3,247.3 L524.2,247.3 L524.1,247.3 L524.0,247.3 L523.9,247.3 L523.8,247.4 L523.6,247.4 L523.5,247.4 L523.4,247.4 L523.3,247.4 L523.2,247.4 L523.1,247.4 L523.0,247.5 L522.8,247.5 L522.7,247.5 L522.6,247.5 L522.5,247.5 L522.4,247.6 L522.3,247.6 L522.2,247.6 L522.1,247.6 L521.9,247.6 L521.8,247.7 L521.7,247.7 L521.6,247.7 L521.5,247.7 L521.4,247.8 L521.3,247.8 L521.2,247.8 L521.1,247.9 L520.9,247.9 L520.8,247.9 L520.7,248.0 L520.6,248.0 L520.5,248.0 L520.4,248.1 L520.3,248.1 L520.2,248.1 L520.1,248.2 L520.0,248.2 L519.8,248.3 L519.7,248.3 L519.6,248.3 L519.5,248.4 L519.4,248.4 L519.3,248.5 L519.2,248.5 L519.1,248.5 L519.0,248.6 L518.9,248.6 L518.8,248.7 L518.7,248.7 L518.6,248.8 L518.5,248.8 L518.4,248.9 L518.3,248.9 L518.2,249.0 L518.1,249.0 L518.0,249.1 L517.9,249.1 L517.8,249.2 L517.7,249.3 L517.6,249.3 L517.5,249.4 L517.4,249.4 L517.3,249.5 L517.2,249.5 L517.1,249.6 L517.0,249.7 L516.9,249.7 L516.8,249.8 L516.7,249.9 L516.6,249.9 L516.5,250.0 L516.4,250.0 L516.3,250.1 L516.2,250.2 L516.1,250.2 L516.0,250.3 L515.9,250.4 L515.8,250.4 L515.7,250.5 L515.6,250.6 L515.6,250.7 L515.5,250.7 L515.4,250.8 L515.3,250.9 L515.2,250.9 L515.1,251.0 L515.0,251.1 L514.9,251.2 L514.8,251.2 L514.8,251.3 L514.7,251.4 L514.6,251.5 L514.5,251.6 L514.4,251.6 L514.3,251.7 L514.3,251.8 L514.2,251.9 L514.1,251.9 L514.0,252.0 L513.9,252.1 L513.8,252.2 L513.8,252.3 L513.7,252.3 L513.6,252.4 L513.5,252.5 L513.4,252.6 L513.3,252.7 L513.2,252.7 L513.2,252.8 L513.1,252.9 L513.0,253.0 L512.9,253.1 L512.8,253.1 L512.7,253.2 L512.7,253.3 L512.6,253.4 L512.5,253.4 L512.4,253.5 L512.3,253.6 L512.2,253.7 L512.2,253.8 L512.1,253.8 L512.0,253.9 L511.9,254.0 L511.8,254.1 L511.7,254.2 L511.7,254.2 L511.6,254.3 L511.5,254.4 L511.4,254.5 L511.3,254.5 L511.2,254.6 L511.2,254.7 L511.1,254.8 L511.0,254.9 L510.9,254.9 L510.8,255.0 L510.7,255.1 L510.7,255.2 L510.6,255.3 L510.5,255.3 L510.4,255.4 L510.3,255.5 L510.2,255.6 L510.1,255.7 L510.1,255.7 L510.0,255.8 L509.9,255.9 L509.8,256.0 L509.7,256.0 L509.6,256.1 L509.6,256.2 L509.5,256.3 L509.4,256.4 L509.3,256.4 L509.2,256.5 L509.1,256.6 L509.1,256.7 L509.0,256.8 L508.9,256.8 L508.8,256.9 L508.7,257.0 L508.6,257.1 L508.6,257.1 L508.5,257.2 L508.4,257.3 L508.3,257.4 L508.2,257.5 L508.1,257.5 L508.1,257.6 L508.0,257.7 L507.9,257.8 L507.8,257.9 L507.7,257.9 L507.6,258.0 L507.6,258.1 L507.5,258.2 L507.4,258.3 L507.3,258.3 L507.2,258.4 L507.1,258.5 L507.0,258.6 L507.0,258.6 L506.9,258.7 L506.8,258.8 L506.7,258.9 L506.6,259.0 L506.5,259.0 L506.5,259.1 L506.4,259.2 L506.3,259.3 L506.2,259.4 L506.1,259.4 L506.0,259.5 L506.0,259.6 L505.9,259.7 L505.8,259.7 L505.7,259.8 L505.6,259.9 L505.5,260.0 L505.5,260.1 L505.4,260.1 L505.3,260.2 L505.2,260.3 L505.1,260.4 L505.0,260.5 L505.0,260.5 L504.9,260.6 L504.8,260.7 L504.7,260.8 L504.6,260.8 L504.5,260.9 L504.5,261.0 L504.4,261.1 L504.3,261.2 L504.2,261.2 L504.1,261.3 L504.0,261.4 L503.9,261.5 L503.9,261.6 L503.8,261.6 L503.7,261.7 L503.6,261.8 L503.5,261.9 L503.4,262.0 L503.4,262.0 L503.3,262.1 L503.2,262.2 L503.1,262.3 L503.0,262.3 L502.9,262.4 L502.9,262.5 L502.8,262.6 L502.7,262.7 L502.6,262.7 L502.5,262.8 L502.4,262.9 L502.4,263.0 L502.3,263.1 L502.2,263.1 L502.1,263.2 L502.0,263.3 L501.9,263.4 L501.9,263.4 L501.8,263.5 L501.7,263.6 L501.6,263.7 L501.5,263.8 L501.4,263.8 L501.4,263.9 L501.3,264.0 L501.2,264.1 L501.1,264.2 L501.0,264.2 L500.9,264.3 L500.8,264.4 L500.8,264.5 L500.7,264.6 L500.6,264.6 L500.5,264.7 L500.4,264.8 L500.3,264.9 L500.3,264.9 L500.2,265.0 L500.1,265.1 L500.0,265.2 L499.9,265.3 L499.8,265.3 L499.8,265.4 L499.7,265.5 L499.6,265.6 L499.5,265.7 L499.4,265.7 L499.3,265.8 L499.3,265.9 L499.2,266.0 L499.1,266.0 L499.0,266.1 L498.9,266.2 L498.8,266.3 L498.8,266.4 L498.7,266.4 L498.6,266.5 L498.5,266.6 L498.4,266.7 L498.3,266.8 L498.3,266.8 L498.2,266.9 L498.1,267.0 L498.0,267.1 L497.9,267.2 L497.8,267.2 L497.7,267.3 L497.7,267.4 L497.6,267.5 L497.5,267.5 L497.4,267.6 L497.3,267.7 L497.2,267.8 L497.2,267.9 L497.1,267.9 L497.0,268.0 L496.9,268.1 L496.8,268.2 L496.7,268.3 L496.7,268.3 L496.6,268.4 L496.5,268.5 L496.4,268.6 L496.3,268.6 L496.2,268.7 L496.2,268.8 L496.1,268.9 L496.0,269.0 L495.9,269.0 L495.8,269.1 L495.7,269.2 L495.7,269.3 L495.6,269.4 L495.5,269.4 L495.4,269.5 L495.3,269.6 L495.2,269.7 L495.2,269.8 L495.1,269.8 L495.0,269.9 L494.9,270.0 L494.8,270.0 L494.7,270.1 L494.6,270.1 L494.5,270.2 L494.4,270.2 L494.2,270.2 L494.1,270.2 L494.0,270.3 L493.9,270.3 L493.8,270.3 L493.7,270.3 L493.6,270.3 L493.4,270.2 L493.3,270.2 L493.2,270.2 L493.1,270.2 L493.0,270.1 L492.9,270.1 L492.8,270.0 L492.7,270.0 L492.6,270.0 L492.4,269.9 L492.3,269.9 L492.2,269.9 L492.1,269.8 L492.0,269.8 L491.9,269.8 L491.8,269.7 L491.7,269.7 L491.6,269.7 L491.5,269.6 L491.3,269.6 L491.2,269.6 L491.1,269.5 L491.0,269.5 L490.9,269.5 L490.8,269.4 L490.7,269.4 L490.6,269.4 L490.5,269.3 L490.4,269.3 L490.2,269.3 L490.1,269.2 L490.0,269.2 L489.9,269.2 L489.8,269.1 L489.7,269.1 L489.6,269.1 L489.5,269.0 L489.4,269.0 L489.3,269.0 L489.1,268.9 L489.0,268.9 L488.9,268.9 L488.8,268.8 L488.7,268.8 L488.6,268.8 L488.5,268.7 L488.4,268.7 L488.3,268.7 L488.2,268.6 L488.0,268.6 L487.9,268.6 L487.8,268.5 L487.7,268.5 L487.6,268.5 L487.5,268.4 L487.4,268.4 L487.3,268.4 L487.2,268.3 L487.1,268.3 L486.9,268.3 L486.8,268.2 L486.7,268.2 L486.6,268.2 L486.5,268.1 L486.4,268.1 L486.3,268.1 L486.2,268.0 L486.1,268.0 L486.0,268.0 L485.8,267.9 L485.7,267.9 L485.6,267.9 L485.5,267.8 L485.4,267.8 L485.3,267.8 L485.2,267.7 L485.1,267.7 L485.0,267.7 L484.9,267.6 L484.7,267.6 L484.6,267.6 L484.5,267.5 L484.4,267.5 L484.3,267.5 L484.2,267.4 L484.1,267.4 L484.0,267.4 L483.9,267.3 L483.8,267.3 L483.6,267.3 L483.5,267.2 L483.4,267.2 L483.3,267.2 L483.2,267.1 L483.1,267.1 L483.0,267.1 L482.9,267.0 L482.8,267.0 L482.7,267.0 L482.5,266.9 L482.4,266.9 L482.3,266.9 L482.2,266.8 L482.1,266.8 L482.0,266.8 L481.9,266.7 L481.8,266.7 L481.7,266.7 L481.6,266.6 L481.4,266.6 L481.3,266.6 L481.2,266.5 L481.1,266.5 L481.0,266.5 L480.9,266.4 L480.8,266.4 L480.7,266.4 L480.6,266.3 L480.5,266.3 L480.3,266.3 L480.2,266.2 L480.1,266.2 L480.0,266.2 L479.9,266.1 L479.8,266.1 L479.7,266.1 L479.6,266.0 L479.5,266.0 L479.4,266.0 L479.2,266.0 L479.1,266.0 L479.0,265.9 L478.9,265.9 L478.8,265.9 L478.7,265.9 L478.6,266.0 L478.4,266.0 L478.3,266.0 L478.2,266.0 L478.1,266.0 L478.0,266.1 L477.9,266.1 L477.8,266.1 L477.7,266.2 L477.5,266.2 L477.4,266.2 L477.3,266.2 L477.2,266.3 L477.1,266.3 L477.0,266.3 L476.9,266.4 L476.8,266.4 L476.7,266.4 L476.5,266.4 L476.4,266.5 L476.3,266.5 L476.2,266.5 L476.1,266.5 L476.0,266.6 L475.9,266.6 L475.8,266.6 L475.6,266.7 L475.5,266.7 L475.4,266.7 L475.3,266.7 L475.2,266.8 L475.1,266.8 L475.0,266.8 L474.9,266.9 L474.8,266.9 L474.6,266.9 L474.5,266.9 L474.4,267.0 L474.3,267.0 L474.2,267.0 L474.1,267.1 L474.0,267.1 L473.9,267.1 L473.8,267.1 L473.6,267.2 L473.5,267.2 L473.4,267.2 L473.3,267.3 L473.2,267.3 L473.1,267.3 L473.0,267.3 L472.9,267.4 L472.8,267.4 L472.6,267.4 L472.5,267.5 L472.4,267.5 L472.3,267.5 L472.2,267.5 L472.1,267.6 L472.0,267.6 L471.9,267.6 L471.7,267.6 L471.6,267.7 L471.5,267.7 L471.4,267.7 L471.3,267.7 L471.2,267.7 L471.1,267.7 L470.9,267.7 L470.8,267.7 L470.7,267.7 L470.6,267.7 L470.5,267.7 L470.4,267.6 L470.3,267.6 L470.1,267.6 L470.0,267.6 L469.9,267.5 L469.8,267.5 L469.7,267.4 L469.6,267.4 L469.5,267.4 L469.4,267.3 L469.3,267.3 L469.2,267.2 L469.1,267.1 L469.0,267.1 L468.9,267.0 L468.8,267.0 L468.7,266.9 L468.6,266.8 L468.5,266.8 L468.4,266.7 L468.3,266.6 L468.2,266.6 L468.1,266.5 L468.0,266.4 L468.0,266.4 L467.9,266.3 L467.8,266.2 L467.7,266.2 L467.6,266.1 L467.5,266.0 L467.4,266.0 L467.3,265.9 L467.2,265.8 L467.1,265.8 L467.0,265.7 L466.9,265.6 L466.8,265.6 L466.7,265.5 L466.6,265.4 L466.5,265.4 L466.4,265.3 L466.4,265.2 L466.3,265.2 L466.2,265.1 L466.1,265.0 L466.0,265.0 L465.9,264.9 L465.8,264.8 L465.7,264.8 L465.6,264.7 L465.5,264.6 L465.4,264.6 L465.3,264.5 L465.2,264.4 L465.1,264.4 L465.0,264.3 L464.9,264.2 L464.8,264.2 L464.8,264.1 L464.7,264.1 L464.6,264.0 L464.5,263.9 L464.4,263.9 L464.3,263.8 L464.2,263.7 L464.1,263.7 L464.0,263.6 L463.9,263.6 L463.8,263.5 L463.7,263.5 L463.6,263.4 L463.5,263.4 L463.3,263.3 L463.2,263.3 L463.1,263.3 L463.0,263.3 L462.9,263.2 L462.8,263.2 L462.7,263.2 L462.6,263.2 L462.4,263.2 L462.3,263.2 L462.2,263.2 L462.1,263.2 L462.0,263.2 L461.9,263.3 L461.8,263.3 L461.6,263.3 L461.5,263.3 L461.4,263.4 L461.3,263.4 L461.2,263.4 L461.1,263.4 L461.0,263.5 L460.9,263.5 L460.8,263.5 L460.6,263.6 L460.5,263.6 L460.4,263.6 L460.3,263.6 L460.2,263.7 L460.1,263.7 L460.0,263.7 L459.9,263.8 L459.7,263.8 L459.6,263.8 L459.5,263.8 L459.4,263.9 L459.3,263.9 L459.2,263.9 L459.1,264.0 L459.0,264.0 L458.9,264.0 L458.7,264.0 L458.6,264.1 L458.5,264.1 L458.4,264.1 L458.3,264.2 L458.2,264.2 L458.1,264.2 L458.0,264.2 L457.9,264.3 L457.7,264.3 L457.6,264.3 L457.5,264.4 L457.4,264.4 L457.3,264.4 L457.2,264.4 L457.1,264.5 L457.0,264.5 L456.9,264.5 L456.7,264.5 L456.6,264.6 L456.5,264.6 L456.4,264.6 L456.3,264.7 L456.2,264.7 L456.1,264.7 L456.0,264.7 L455.8,264.8 L455.7,264.8 L455.6,264.8 L455.5,264.9 L455.4,264.9 L455.3,264.9 L455.2,264.9 L455.1,265.0 L455.0,265.0 L454.8,265.0 L454.7,265.0 L454.6,265.0 L454.5,265.0 L454.4,265.0 L454.3,265.0 L454.1,265.0 L454.0,265.0 L453.9,264.9 L453.8,264.9 L453.7,264.9 L453.6,264.9 L453.5,264.8 L453.4,264.8 L453.3,264.8 L453.1,264.7 L453.0,264.7 L452.9,264.7 L452.8,264.6 L452.7,264.6 L452.6,264.6 L452.5,264.5 L452.4,264.5 L452.3,264.5 L452.2,264.4 L452.0,264.4 L451.9,264.4 L451.8,264.3 L451.7,264.3 L451.6,264.3 L451.5,264.2 L451.4,264.2 L451.3,264.2 L451.2,264.1 L451.1,264.1 L450.9,264.1 L450.8,264.0 L450.7,264.0 L450.6,264.0 L450.5,263.9 L450.4,263.9 L450.3,263.8 L450.2,263.8 L450.1,263.8 L450.0,263.7 L449.8,263.7 L449.7,263.7 L449.6,263.6 L449.5,263.6 L449.4,263.6 L449.3,263.5 L449.2,263.5 L449.1,263.5 L449.0,263.4 L448.9,263.4 L448.7,263.4 L448.6,263.3 L448.5,263.3 L448.4,263.3 L448.3,263.2 L448.2,263.2 L448.1,263.2 L448.0,263.1 L447.9,263.1 L447.8,263.1 L447.6,263.0 L447.5,263.0 L447.4,263.0 L447.3,262.9 L447.2,262.9 L447.1,262.9 L447.0,262.8 L446.9,262.8 L446.8,262.8 L446.7,262.7 L446.5,262.7 L446.4,262.7 L446.3,262.6 L446.2,262.6 L446.1,262.6 L446.0,262.6 L445.9,262.5 L445.8,262.5 L445.7,262.5 L445.5,262.5 L445.4,262.4 L445.3,262.4 L445.2,262.4 L445.1,262.4 L445.0,262.4 L444.9,262.3 L444.7,262.3 L444.6,262.3 L444.5,262.3 L444.4,262.3 L444.3,262.3 L444.2,262.3 L444.1,262.3 L443.9,262.3 L443.8,262.3 L443.7,262.3 L443.6,262.3 L443.5,262.3 L443.4,262.3 L443.3,262.3 L443.1,262.3 L443.0,262.3 L442.9,262.3 L442.8,262.3 L442.7,262.3 L442.6,262.4 L442.5,262.4 L442.3,262.4 L442.2,262.4 L442.1,262.4 L442.0,262.4 L441.9,262.5 L441.8,262.5 L441.7,262.5 L441.5,262.5 L441.4,262.6 L441.3,262.6 L441.2,262.6 L441.1,262.7 L441.0,262.7 L440.9,262.7 L440.8,262.8 L440.7,262.8 L440.6,262.8 L440.5,262.9 L440.3,262.9 L440.2,263.0 L440.1,263.0 L440.0,263.1 L439.9,263.1 L439.8,263.2 L439.7,263.2 L439.6,263.3 L439.5,263.3 L439.4,263.4 L439.3,263.4 L439.2,263.5 L439.1,263.5 L439.0,263.6 L438.9,263.6 L438.8,263.7 L438.7,263.8 L438.6,263.8 L438.5,263.9 L438.4,264.0 L438.3,264.0 L438.2,264.1 L438.1,264.2 L438.0,264.2 L438.0,264.3 L437.9,264.4 L437.8,264.4 L437.7,264.5 L437.6,264.6 L437.5,264.7 L437.4,264.7 L437.3,264.8 L437.3,264.9 L437.2,265.0 L437.1,265.1 L437.0,265.1 L436.9,265.2 L436.9,265.3 L436.8,265.4 L436.7,265.5 L436.6,265.6 L436.6,265.7 L436.5,265.7 L436.4,265.8 L436.3,265.9 L436.3,266.0 L436.2,266.1 L436.1,266.2 L436.1,266.3 L436.0,266.4 L435.9,266.5 L435.9,266.6 L435.8,266.7 L435.8,266.8 L435.7,266.9 L435.6,267.0 L435.6,267.1 L435.5,267.2 L435.5,267.3 L435.4,267.4 L435.4,267.5 L435.3,267.6 L435.3,267.7 L435.2,267.8 L435.2,267.9 L435.1,268.0 L435.1,268.1 L435.0,268.2 L434.9,268.3 L434.9,268.4 L434.8,268.5 L434.8,268.6 L434.7,268.7 L434.7,268.8 L434.6,268.9 L434.6,269.0 L434.5,269.1 L434.5,269.2 L434.4,269.3 L434.4,269.4 L434.3,269.5 L434.3,269.6 L434.2,269.7 L434.1,269.8 L434.1,269.9 L434.0,270.0 L434.0,270.1 L433.9,270.2 L433.9,270.3 L433.8,270.4 L433.8,270.5 L433.7,270.6 L433.7,270.7 L433.6,270.8 L433.6,270.9 L433.5,271.1 L433.5,271.2 L433.4,271.3 L433.3,271.4 L433.3,271.5 L433.2,271.6 L433.2,271.7 L433.1,271.8 L433.1,271.9 L433.0,272.0 L433.0,272.1 L432.9,272.2 L432.9,272.3 L432.8,272.4 L432.8,272.5 L432.7,272.6 L432.7,272.7 L432.6,272.8 L432.5,272.9 L432.5,273.0 L432.4,273.1 L432.4,273.2 L432.3,273.3 L432.3,273.4 L432.2,273.5 L432.2,273.6 L432.1,273.7 L432.1,273.8 L432.0,273.9 L432.0,274.0 L431.9,274.1 L431.9,274.2 L431.8,274.3 L431.7,274.4 L431.7,274.5 L431.6,274.6 L431.6,274.7 L431.5,274.8 L431.5,274.9 L431.4,275.0 L431.4,275.1 L431.3,275.2 L431.3,275.3 L431.2,275.4 L431.2,275.5 L431.1,275.6 L431.1,275.7 L431.0,275.8 L431.0,275.9 L430.9,276.0 L430.8,276.1 L430.8,276.2 L430.7,276.3 L430.7,276.5 L430.6,276.6 L430.6,276.7 L430.5,276.8 L430.5,276.9 L430.4,277.0 L430.4,277.1 L430.3,277.2 L430.3,277.3 L430.2,277.4 L430.2,277.5 L430.1,277.6 L430.0,277.7 L430.0,277.8 L429.9,277.9 L429.9,278.0 L429.8,278.1 L429.8,278.2 L429.7,278.3 L429.7,278.4 L429.6,278.5 L429.6,278.6 L429.5,278.7 L429.5,278.8 L429.4,278.9 L429.4,279.0 L429.3,279.1 L429.2,279.2 L429.2,279.3 L429.1,279.4 L429.1,279.5 L429.0,279.6 L429.0,279.7 L428.9,279.8 L428.9,279.9 L428.8,280.0 L428.8,280.1 L428.7,280.2 L428.7,280.3 L428.6,280.4 L428.6,280.5 L428.5,280.6 L428.4,280.7 L428.4,280.8 L428.3,280.9 L428.3,281.0 L428.2,281.1 L428.2,281.2 L428.1,281.3 L428.1,281.4 L428.0,281.5 L428.0,281.6 L427.9,281.8 L427.9,281.9 L427.8,282.0 L427.8,282.1 L427.7,282.2 L427.6,282.3 L427.6,282.4 L427.5,282.5 L427.5,282.6 L427.4,282.7 L427.4,282.8 L427.3,282.9 L427.3,283.0 L427.2,283.1 L427.2,283.2 L427.1,283.3 L427.1,283.4 L427.0,283.5 L427.0,283.6 L426.9,283.7 L426.8,283.8 L426.8,283.9 L426.7,284.0 L426.7,284.1 L426.6,284.2 L426.6,284.3 L426.5,284.4 L426.5,284.5 L426.4,284.6 L426.4,284.7 L426.3,284.8 L426.3,284.9 L426.2,285.0 L426.2,285.1 L426.1,285.2 L426.0,285.3 L426.0,285.4 L425.9,285.5 L425.9,285.6 L425.8,285.7 L425.8,285.8 L425.7,285.9 L425.7,286.0 L425.6,286.1 L425.6,286.2 L425.5,286.3 L425.5,286.4 L425.4,286.5 L425.3,286.6 L425.3,286.7 L425.2,286.8 L425.1,286.9 L425.0,287.0 L425.0,287.1 L424.9,287.1 L424.8,287.2 L424.7,287.3 L424.6,287.4 L424.5,287.4 L424.4,287.5 L424.3,287.6 L424.3,287.7 L424.2,287.7 L424.1,287.8 L424.0,287.9 L423.9,287.9 L423.8,288.0 L423.7,288.1 L423.6,288.1 L423.5,288.2 L423.4,288.3 L423.3,288.4 L423.3,288.4 L423.2,288.5 L423.1,288.6 L423.0,288.6 L422.9,288.7 L422.8,288.8 L422.7,288.8 L422.6,288.9 L422.5,289.0 L422.4,289.1 L422.3,289.1 L422.2,289.2 L422.2,289.3 L422.1,289.3 L422.0,289.4 L421.9,289.5 L421.8,289.5 L421.7,289.6 L421.6,289.7 L421.5,289.8 L421.4,289.8 L421.3,289.9 L421.2,290.0 L421.2,290.0 L421.1,290.1 L421.0,290.2 L420.9,290.3 L420.8,290.3 L420.7,290.4 L420.6,290.5 L420.5,290.5 L420.4,290.6 L420.3,290.7 L420.2,290.7 L420.2,290.8 L420.1,290.9 L420.0,291.0 L419.9,291.0 L419.8,291.1 L419.7,291.2 L419.6,291.2 L419.5,291.3 L419.4,291.4 L419.3,291.4 L419.2,291.5 L419.2,291.6 L419.1,291.7 L419.0,291.7 L418.9,291.8 L418.8,291.9 L418.7,291.9 L418.6,292.0 L418.5,292.1 L418.4,292.1 L418.3,292.2 L418.2,292.3 L418.1,292.4 L418.1,292.4 L418.0,292.5 L417.9,292.6 L417.8,292.6 L417.7,292.7 L417.6,292.8 L417.5,292.8 L417.4,292.9 L417.3,293.0 L417.2,293.1 L417.2,293.1 L417.1,293.2 L417.0,293.3 L416.9,293.4 L416.8,293.5 L416.8,293.6 L416.7,293.6 L416.6,293.7 L416.5,293.8 L416.5,293.9 L416.4,294.0 L416.4,294.1 L416.3,294.2 L416.3,294.3 L416.2,294.4 L416.2,294.5 L416.1,294.6 L416.1,294.7 L416.0,294.9 L416.0,295.0 L416.0,295.1 L415.9,295.2 L415.9,295.3 L415.9,295.4 L415.9,295.5 L415.9,295.6 L415.8,295.8 L415.8,295.9 L415.8,296.0 L415.8,296.1 L415.8,296.2 L415.7,296.3 L415.7,296.4 L415.7,296.5 L415.7,296.7 L415.7,296.8 L415.6,296.9 L415.6,297.0 L415.6,297.1 L415.6,297.2 L415.5,297.3 L415.5,297.4 L415.5,297.6 L415.5,297.7 L415.5,297.8 L415.4,297.9 L415.4,298.0 L415.4,298.1 L415.4,298.2 L415.4,298.4 L415.3,298.5 L415.3,298.6 L415.3,298.7 L415.3,298.8 L415.2,298.9 L415.2,299.0 L415.2,299.1 L415.2,299.3 L415.2,299.4 L415.1,299.5 L415.1,299.6 L415.1,299.7 L415.1,299.8 L415.1,299.9 L415.0,300.0 L415.0,300.2 L415.0,300.3 L415.0,300.4 L415.0,300.5 L414.9,300.6 L414.9,300.7 L414.9,300.8 L414.9,301.0 L414.8,301.1 L414.8,301.2 L414.8,301.3 L414.8,301.4 L414.8,301.5 L414.7,301.6 L414.7,301.7 L414.7,301.9 L414.7,302.0 L414.7,302.1 L414.6,302.2 L414.6,302.3 L414.6,302.4 L414.6,302.5 L414.6,302.6 L414.5,302.8 L414.5,302.9 L414.5,303.0 L414.5,303.1 L414.4,303.2 L414.4,303.3 L414.4,303.4 L414.4,303.6 L414.4,303.7 L414.3,303.8 L414.3,303.9 L414.3,304.0 L414.3,304.1 L414.3,304.2 L414.2,304.3 L414.2,304.5 L414.2,304.6 L414.2,304.7 L414.1,304.8 L414.1,304.9 L414.0,305.0 L414.0,305.1 L413.9,305.2 L413.9,305.3 L413.8,305.4 L413.8,305.5 L413.7,305.6 L413.6,305.7 L413.6,305.8 L413.5,305.9 L413.4,306.0 L413.4,306.1 L413.3,306.2 L413.2,306.3 L413.2,306.4 L413.1,306.4 L413.0,306.5 L413.0,306.6 L412.9,306.7 L412.8,306.8 L412.8,306.9 L412.7,307.0 L412.6,307.1 L412.6,307.2 L412.5,307.3 L412.4,307.4 L412.4,307.5 L412.3,307.6 L412.2,307.7 L412.2,307.8 L412.1,307.9 L412.0,307.9 L412.0,308.0 L411.9,308.1 L411.8,308.2 L411.8,308.3 L411.7,308.4 L411.6,308.5 L411.6,308.6 L411.5,308.7 L411.4,308.8 L411.4,308.9 L411.3,309.0 L411.2,309.1 L411.2,309.2 L411.1,309.3 L411.0,309.3 L411.0,309.4 L410.9,309.5 L410.8,309.6 L410.8,309.7 L410.7,309.8 L410.6,309.9 L410.6,310.0 L410.5,310.1 L410.4,310.2 L410.4,310.3 L410.3,310.4 L410.2,310.5 L410.2,310.6 L410.1,310.7 L410.0,310.8 L410.0,310.8 L409.9,310.9 L409.8,311.0 L409.8,311.1 L409.7,311.2 L409.6,311.3 L409.6,311.4 L409.5,311.5 L409.4,311.6 L409.4,311.7 L409.3,311.8 L409.2,311.9 L409.2,312.0 L409.1,312.1 L409.0,312.2 L409.0,312.3 L408.9,312.3 L408.8,312.4 L408.8,312.5 L408.7,312.6 L408.6,312.7 L408.6,312.8 L408.5,312.9 L408.4,313.0 L408.4,313.1 L408.3,313.2 L408.2,313.3 L408.2,313.4 L408.1,313.5 L408.0,313.6 L408.0,313.7 L407.9,313.8 L407.8,313.8 L407.8,313.9 L407.7,314.0 L407.6,314.1 L407.6,314.2 L407.5,314.3 L407.4,314.4 L407.4,314.5 L407.3,314.6 L407.2,314.7 L407.2,314.8 L407.1,314.9 L407.0,315.0 L407.0,315.1 L406.9,315.2 L406.8,315.3 L406.8,315.3 L406.7,315.4 L406.6,315.5 L406.6,315.6 L406.5,315.7 L406.4,315.8 L406.4,315.9 L406.3,316.0 L406.2,316.1 L406.2,316.2 L406.1,316.3 L406.0,316.4 L406.0,316.5 L405.9,316.6 L405.8,316.7 L405.8,316.8 L405.7,316.8 L405.6,316.9 L405.6,317.0 L405.5,317.1 L405.4,317.2 L405.4,317.3 L405.3,317.4 L405.2,317.5 L405.2,317.6 L405.1,317.7 L405.0,317.8 L405.0,317.9 L404.9,318.0 L404.8,318.1 L404.8,318.2 L404.7,318.3 L404.6,318.3 L404.6,318.4 L404.5,318.5 L404.4,318.6 L404.4,318.7 L404.3,318.8 L404.2,318.9 L404.2,319.0 L404.1,319.1 L404.0,319.2 L404.0,319.3 L403.9,319.4 L403.8,319.5 L403.8,319.6 L403.7,319.7 L403.6,319.8 L403.6,319.8 L403.5,319.9 L403.4,320.0 L403.4,320.1 L403.3,320.2 L403.2,320.3 L403.2,320.4 L403.1,320.5 L403.0,320.6 L403.0,320.7 L402.9,320.8 L402.8,320.9 L402.8,321.0 L402.7,321.1 L402.6,321.2 L402.6,321.3 L402.5,321.3 L402.4,321.4 L402.4,321.5 L402.3,321.6 L402.2,321.7 L402.2,321.8 L402.1,321.9 L402.0,322.0 L402.0,322.1 L401.9,322.2 L401.8,322.3 L401.8,322.4 L401.7,322.5 L401.6,322.6 L401.6,322.7 L401.5,322.8 L401.4,322.8 L401.4,322.9 L401.3,323.0 L401.2,323.1 L401.2,323.2 L401.1,323.3 L401.0,323.4 L401.0,323.5 L400.9,323.6 L400.8,323.7 L400.8,323.8 L400.7,323.9 L400.6,324.0 L400.6,324.1 L400.5,324.2 L400.4,324.3 L400.4,324.3 L400.3,324.4 L400.2,324.5 L400.2,324.6 L400.1,324.7 L400.0,324.8 L400.0,324.9 L399.9,325.0 L399.8,325.1 L399.8,325.2 L399.7,325.3 L399.6,325.4 L399.6,325.5 L399.5,325.6 L399.4,325.7 L399.4,325.8 L399.3,325.8 L399.3,325.9 L399.2,326.0 L399.1,326.1 L399.1,326.2 L399.0,326.3 L398.9,326.4 L398.9,326.5 L398.8,326.6 L398.8,326.7 L398.7,326.8 L398.7,326.9 L398.6,327.0 L398.6,327.1 L398.5,327.2 L398.5,327.3 L398.4,327.5 L398.4,327.6 L398.3,327.7 L398.3,327.8 L398.2,327.9 L398.2,328.0 L398.2,328.1 L398.1,328.2 L398.1,328.3 L398.0,328.4 L398.0,328.5 L398.0,328.6 L397.9,328.7 L397.9,328.9 L397.9,329.0 L397.9,329.1 L397.8,329.2 L397.8,329.3 L397.8,329.4 L397.7,329.5 L397.7,329.6 L397.7,329.8 L397.7,329.9 L397.7,330.0 L397.6,330.1 L397.6,330.2 L397.6,330.3 L397.6,330.4 L397.6,330.5 L397.6,330.7 L397.6,330.8 L397.5,330.9 L397.5,331.0 L397.5,331.1 L397.5,331.2 L397.5,331.4 L397.5,331.5 L397.5,331.6 L397.5,331.7 L397.5,331.8 L397.5,331.9 L397.5,332.0 L397.5,332.2 L397.5,332.3 L397.5,332.4 L397.5,332.5 L397.6,332.6 L397.6,332.7 L397.6,332.8 L397.6,333.0 L397.6,333.1 L397.6,333.2 L397.6,333.3 L397.6,333.4 L397.7,333.5 L397.7,333.6 L397.7,333.8 L397.7,333.9 L397.8,334.0 L397.8,334.1 L397.8,334.2 L397.8,334.3 L397.9,334.4 L397.9,334.5 L397.9,334.7 L397.9,334.8 L397.9,334.9 L398.0,335.0 L398.0,335.1 L398.0,335.2 L398.0,335.3 L398.1,335.4 L398.1,335.6 L398.1,335.7 L398.1,335.8 L398.2,335.9 L398.2,336.0 L398.2,336.1 L398.2,336.2 L398.3,336.3 L398.3,336.5 L398.3,336.6 L398.3,336.7 L398.4,336.8 L398.4,336.9 L398.4,337.0 L398.4,337.1 L398.4,337.2 L398.5,337.4 L398.5,337.5 L398.5,337.6 L398.5,337.7 L398.6,337.8 L398.6,337.9 L398.6,338.0 L398.6,338.1 L398.7,338.3 L398.7,338.4 L398.7,338.5 L398.7,338.6 L398.8,338.7 L398.8,338.8 L398.8,338.9 L398.8,339.0 L398.9,339.2 L398.9,339.3 L398.9,339.4 L398.9,339.5 L398.9,339.6 L399.0,339.7 L399.0,339.8 L399.0,339.9 L399.0,340.1 L399.1,340.2 L399.1,340.3 L399.1,340.4 L399.1,340.5 L399.2,340.6 L399.2,340.7 L399.2,340.8 L399.2,341.0 L399.3,341.1 L399.3,341.2 L399.3,341.3 L399.3,341.4 L399.3,341.5 L399.4,341.6 L399.4,341.7 L399.4,341.9 L399.4,342.0 L399.5,342.1 L399.5,342.2 L399.5,342.3 L399.5,342.4 L399.6,342.5 L399.6,342.6 L399.6,342.8 L399.6,342.9 L399.7,343.0 L399.7,343.1 L399.7,343.2 L399.7,343.3 L399.8,343.4 L399.8,343.5 L399.8,343.7 L399.8,343.8 L399.8,343.9 L399.9,344.0 L399.9,344.1 L399.9,344.2 L399.9,344.3 L400.0,344.4 L400.0,344.6 L400.0,344.7 L400.0,344.8 L400.1,344.9 L400.1,345.0 L400.1,345.1 L400.1,345.2 L400.2,345.3 L400.2,345.5 L400.2,345.6 L400.2,345.7 L400.3,345.8 L400.3,345.9 L400.3,346.0 L400.3,346.1 L400.3,346.2 L400.4,346.4 L400.4,346.5 L400.4,346.6 L400.4,346.7 L400.5,346.8 L400.5,346.9 L400.5,347.0 L400.5,347.1 L400.6,347.3 L400.6,347.4 L400.6,347.5 L400.6,347.6 L400.7,347.7 L400.7,347.8 L400.7,347.9 L400.7,348.0 L400.8,348.2 L400.8,348.3 L400.8,348.4 L400.8,348.5 L400.8,348.6 L400.9,348.7 L400.9,348.8 L400.9,348.9 L400.9,349.1 L401.0,349.2 L401.0,349.3 L401.0,349.4 L401.0,349.5 L401.1,349.6 L401.1,349.7 L401.1,349.8 L401.1,350.0 L401.2,350.1 L401.2,350.2 L401.2,350.3 L401.2,350.4 L401.3,350.5 L401.3,350.6 L401.3,350.7 L401.3,350.9 L401.3,351.0 L401.4,351.1 L401.4,351.2 L401.4,351.3 L401.4,351.4 L401.5,351.5 L401.5,351.6 L401.5,351.8 L401.5,351.9 L401.6,352.0 L401.6,352.1 L401.6,352.2 L401.6,352.3 L401.7,352.4 L401.7,352.5 L401.7,352.7 L401.7,352.8 L401.7,352.9 L401.8,353.0 L401.8,353.1 L401.8,353.2 L401.8,353.3 L401.9,353.4 L401.9,353.6 L401.9,353.7 L401.9,353.8 L402.0,353.9 L402.0,354.0 L402.0,354.1 L402.0,354.2 L402.1,354.3 L402.1,354.5 L402.1,354.6 L402.1,354.7 L402.2,354.8 L402.2,354.9 L402.2,355.0 L402.2,355.1 L402.2,355.2 L402.3,355.4 L402.3,355.5 L402.3,355.6 L402.3,355.7 L402.4,355.8 L402.4,355.9 L402.4,356.0 L402.4,356.1 L402.5,356.3 L402.5,356.4 L402.5,356.5 L402.5,356.6 L402.6,356.7 L402.6,356.8 L402.6,356.9 L402.6,357.0 L402.7,357.2 L402.7,357.3 L402.7,357.4 L402.7,357.5 L402.7,357.6 L402.8,357.7 L402.8,357.8 L402.8,357.9 L402.8,358.1 L402.9,358.2 L402.9,358.3 L402.9,358.4 L402.9,358.5 L403.0,358.6 L403.0,358.7 L403.0,358.8 L403.0,359.0 L403.1,359.1 L403.1,359.2 L403.1,359.3 L403.1,359.4 L403.2,359.5 L403.2,359.6 L403.2,359.7 L403.2,359.9 L403.2,360.0 L403.3,360.1 L403.3,360.2 L403.3,360.3 L403.3,360.4 L403.4,360.5 L403.4,360.6 L403.4,360.8 L403.4,360.9 L403.5,361.0 L403.5,361.1 L403.5,361.2 L403.5,361.3 L403.6,361.4 L403.6,361.5 L403.6,361.7 L403.6,361.8 L403.7,361.9 L403.7,362.0 L403.7,362.1 L403.7,362.2 L403.7,362.3 L403.8,362.4 L403.8,362.6 L403.8,362.7 L403.8,362.8 L403.9,362.9 L403.9,363.0 L403.9,363.1 L403.9,363.2 L404.0,363.3 L404.0,363.5 L404.0,363.6 L404.0,363.7 L404.1,363.8 L404.1,363.9 L404.1,364.0 L404.1,364.1 L404.2,364.2 L404.2,364.4 L404.2,364.5 L404.2,364.6 L404.2,364.7 L404.3,364.8 L404.3,364.9 L404.3,365.0 L404.3,365.1 L404.4,365.3 L404.4,365.4 L404.4,365.5 L404.4,365.6 L404.5,365.7 L404.5,365.8 L404.5,365.9 L404.5,366.0 L404.6,366.2 L404.6,366.3 L404.6,366.4 L404.6,366.5 L404.6,366.6 L404.7,366.7 L404.7,366.8 L404.7,366.9 L404.7,367.1 L404.8,367.2 L404.8,367.3 L404.8,367.4 L404.8,367.5 L404.9,367.6 L404.9,367.7 L404.9,367.8 L404.9,368.0 L405.0,368.1 L405.0,368.2 L405.0,368.3 L405.1,368.4 L405.1,368.5 L405.2,368.6 L405.2,368.7 L405.3,368.8 L405.3,368.9 L405.4,369.0 L405.5,369.1 L405.6,369.2 L405.6,369.3 L405.7,369.3 L405.8,369.4 L405.8,369.5 L405.9,369.6 L406.0,369.7 L406.1,369.8 L406.1,369.9 L406.2,370.0 L406.3,370.1 L406.4,370.1 L406.4,370.2 L406.5,370.3 L406.6,370.4 L406.6,370.5 L406.7,370.6 L406.8,370.7 L406.9,370.8 L406.9,370.9 L407.0,371.0 L407.1,371.0 L407.2,371.1 L407.2,371.2 L407.3,371.3 L407.4,371.4 L407.4,371.5 L407.5,371.6 L407.6,371.7 L407.7,371.8 L407.7,371.8 L407.8,371.9 L407.9,372.0 L407.9,372.1 L408.0,372.2 L408.1,372.3 L408.2,372.4 L408.2,372.5 L408.3,372.6 L408.4,372.6 L408.5,372.7 L408.5,372.8 L408.6,372.9 L408.7,373.0 L408.7,373.1 L408.8,373.2 L408.9,373.3 L409.0,373.4 L409.0,373.5 L409.1,373.5 L409.2,373.6 L409.3,373.7 L409.3,373.8 L409.4,373.9 L409.5,374.0 L409.5,374.1 L409.6,374.2 L409.7,374.3 L409.8,374.3 L409.8,374.4 L409.9,374.5 L410.0,374.6 L410.1,374.7 L410.1,374.8 L410.2,374.9 L410.3,375.0 L410.3,375.1 L410.4,375.2 L410.5,375.3 L410.5,375.4 L410.6,375.5 L410.6,375.6 L410.7,375.7 L410.7,375.8 L410.8,375.9 L410.8,376.0 L410.8,376.1 L410.8,376.2 L410.9,376.3 L410.9,376.4 L410.9,376.5 L410.9,376.7 L410.9,376.8 L410.9,376.9 L410.9,377.0 L410.9,377.1 L410.9,377.2 L410.9,377.4 L410.9,377.5 L410.9,377.6 L410.9,377.7 L410.8,377.8 L410.8,377.9 L410.8,378.0 L410.8,378.1 L410.8,378.3 L410.7,378.4 L410.7,378.5 L410.7,378.6 L410.7,378.7 L410.7,378.8 L410.7,378.9 L410.6,379.1 L410.6,379.2 L410.6,379.3 L410.6,379.4 L410.6,379.5 L410.5,379.6 L410.5,379.7 L410.5,379.8 L410.5,380.0 L410.5,380.1 L410.4,380.2 L410.4,380.3 L410.4,380.4 L410.4,380.5 L410.4,380.6 L410.3,380.8 L410.3,380.9 L410.3,381.0 L410.3,381.1 L410.3,381.2 L410.3,381.3 L410.2,381.4 L410.2,381.5 L410.2,381.7 L410.2,381.8 L410.2,381.9 L410.1,382.0 L410.1,382.1 L410.1,382.2 L410.1,382.3 L410.1,382.5 L410.0,382.6 L410.0,382.7 L410.0,382.8 L410.0,382.9 L410.0,383.0 L409.9,383.1 L409.9,383.2 L409.9,383.4 L409.9,383.5 L409.9,383.6 L409.9,383.7 L409.8,383.8 L409.8,383.9 L409.8,384.0 L409.8,384.2 L409.8,384.3 L409.7,384.4 L409.7,384.5 L409.7,384.6 L409.7,384.7 L409.7,384.8 L409.6,385.0 L409.6,385.1 L409.6,385.2 L409.6,385.3 L409.6,385.4 L409.6,385.5 L409.6,385.6 L409.6,385.8 L409.6,385.9 L409.7,386.0 L409.7,386.1 L409.7,386.2 L409.8,386.3 L409.8,386.4 L409.9,386.5 L409.9,386.6 L409.9,386.7 L410.0,386.9 L410.0,387.0 L410.0,387.1 L410.1,387.2 L410.1,387.3 L410.1,387.4 L410.2,387.5 L410.2,387.6 L410.2,387.7 L410.3,387.8 L410.3,387.9 L410.3,388.1 L410.4,388.2 L410.4,388.3 L410.4,388.4 L410.5,388.5 L410.5,388.6 L410.5,388.7 L410.6,388.8 L410.6,388.9 L410.7,389.0 L410.7,389.2 L410.7,389.3 L410.8,389.4 L410.8,389.5 L410.8,389.6 L410.9,389.7 L410.9,389.8 L410.9,389.9 L411.0,390.0 L411.0,390.1 L411.0,390.2 L411.1,390.4 L411.1,390.5 L411.1,390.6 L411.2,390.7 L411.2,390.8 L411.2,390.9 L411.3,391.0 L411.3,391.1 L411.3,391.2 L411.4,391.3 L411.4,391.5 L411.5,391.6 L411.5,391.7 L411.5,391.8 L411.6,391.9 L411.6,392.0 L411.6,392.1 L411.7,392.2 L411.7,392.3 L411.7,392.4 L411.8,392.6 L411.8,392.7 L411.8,392.8 L411.9,392.9 L411.9,393.0 L411.9,393.1 L412.0,393.2 L412.0,393.3 L412.0,393.4 L412.1,393.5 L412.1,393.6 L412.1,393.8 L412.2,393.9 L412.2,394.0 L412.3,394.1 L412.3,394.2 L412.3,394.3 L412.4,394.4 L412.4,394.5 L412.4,394.6 L412.5,394.7 L412.5,394.9 L412.5,395.0 L412.6,395.1 L412.6,395.2 L412.6,395.3 L412.7,395.4 L412.7,395.5 L412.7,395.6 L412.8,395.7 L412.8,395.8 L412.8,395.9 L412.9,396.1 L412.9,396.2 L412.9,396.3 L413.0,396.4 L413.0,396.5 L413.1,396.6 L413.1,396.7 L413.1,396.8 L413.2,396.9 L413.2,397.0 L413.2,397.2 L413.3,397.3 L413.3,397.4 L413.3,397.5 L413.4,397.6 L413.4,397.7 L413.4,397.8 L413.5,397.9 L413.5,398.0 L413.5,398.1 L413.6,398.3 L413.6,398.4 L413.6,398.5 L413.7,398.6 L413.7,398.7 L413.7,398.8 L413.8,398.9 L413.8,399.0 L413.9,399.1 L413.9,399.2 L413.9,399.3 L414.0,399.5 L414.0,399.6 L414.0,399.7 L414.1,399.8 L414.1,399.9 L414.1,400.0 L414.2,400.1 L414.2,400.2 L414.2,400.3 L414.3,400.4 L414.3,400.6 L414.3,400.7 L414.4,400.8 L414.4,400.9 L414.4,401.0 L414.5,401.1 L414.5,401.2 L414.5,401.3 L414.6,401.4 L414.6,401.5 L414.7,401.6 L414.7,401.8 L414.7,401.9 L414.8,402.0 L414.8,402.1 L414.8,402.2 L414.9,402.3 L414.9,402.4 L414.9,402.5 L415.0,402.6 L415.0,402.7 L415.0,402.9 L415.1,403.0 L415.1,403.1 L415.1,403.2 L415.2,403.3 L415.2,403.4 L415.2,403.5 L415.3,403.6 L415.3,403.7 L415.3,403.8 L415.4,403.9 L415.4,404.1 L415.5,404.2 L415.5,404.3 L415.5,404.4 L415.6,404.5 L415.6,404.6 L415.6,404.7 L415.7,404.8 L415.7,404.9 L415.7,405.0 L415.8,405.2 L415.8,405.3 L415.8,405.4 L415.9,405.5 L415.9,405.6 L415.9,405.7 L416.0,405.8 L416.0,405.9 L416.0,406.0 L416.1,406.1 L416.1,406.3 L416.1,406.4 L416.2,406.5 L416.2,406.6 L416.3,406.7 L416.3,406.8 L416.3,406.9 L416.4,407.0 L416.4,407.1 L416.4,407.2 L416.5,407.3 L416.5,407.5 L416.5,407.6 L416.6,407.7 L416.6,407.8 L416.6,407.9 L416.7,408.0 L416.7,408.1 L416.7,408.2 L416.8,408.3 L416.8,408.4 L416.8,408.6 L416.9,408.7 L416.9,408.8 L416.9,408.9 L417.0,409.0 L417.0,409.1 L417.1,409.2 L417.1,409.3 L417.1,409.4 L417.2,409.5 L417.2,409.6 L417.2,409.8 L417.3,409.9 L417.3,410.0 L417.3,410.1 L417.4,410.2 L417.4,410.3 L417.4,410.4 L417.5,410.5 L417.5,410.6 L417.5,410.7 L417.6,410.9 L417.6,411.0 L417.6,411.1 L417.7,411.2 L417.7,411.3 L417.8,411.4 L417.8,411.5 L417.8,411.6 L417.9,411.7 L417.9,411.8 L417.9,412.0 L418.0,412.1 L418.0,412.2 L418.0,412.3 L418.1,412.4 L418.1,412.5 L418.1,412.6 L418.2,412.7 L418.2,412.8 L418.2,412.9 L418.3,413.0 L418.3,413.2 L418.3,413.3 L418.4,413.4 L418.4,413.5 L418.4,413.6 L418.5,413.7 L418.5,413.8 L418.6,413.9 L418.6,414.0 L418.6,414.1 L418.7,414.3 L418.7,414.4 L418.7,414.5 L418.8,414.6 L418.8,414.7 L418.8,414.8 L418.9,414.9 L418.9,415.0 L418.9,415.1 L419.0,415.2 L419.0,415.3 L419.0,415.5 L419.1,415.6 L419.1,415.7 L419.1,415.8 L419.2,415.9 L419.2,416.0 L419.2,416.1 L419.3,416.2 L419.3,416.3 L419.4,416.4 L419.4,416.6 L419.4,416.7 L419.5,416.8 L419.5,416.9 L419.5,417.0 L419.6,417.1 L419.6,417.2 L419.6,417.3 L419.7,417.4 L419.7,417.5 L419.7,417.6 L419.8,417.8 L419.8,417.9 L419.8,418.0 L419.9,418.1 L419.9,418.2 L419.9,418.3 L420.0,418.4 L420.0,418.5 L420.0,418.6 L420.1,418.7 L420.1,418.9 L420.2,419.0 L420.2,419.1 L420.2,419.2 L420.3,419.3 L420.3,419.4 L420.3,419.5 L420.4,419.6 L420.4,419.7 L420.4,419.8 L420.5,420.0 L420.5,420.1 L420.5,420.2 L420.6,420.3 L420.6,420.4 L420.6,420.5 L420.7,420.6 L420.7,420.7 L420.7,420.8 L420.8,420.9 L420.8,421.0 L420.8,421.2 L420.9,421.3 L420.9,421.4 L421.0,421.5 L421.0,421.6 L421.0,421.7 L421.1,421.8 L421.1,421.9 L421.1,422.0 L421.2,422.1 L421.2,422.3 L421.2,422.4 L421.3,422.5 L421.3,422.6 L421.3,422.7 L421.4,422.8 L421.4,422.9 L421.5,423.0 L421.5,423.1 L421.6,423.2 L421.6,423.3 L421.7,423.4 L421.8,423.5 L421.9,423.6 L422.0,423.6 L422.1,423.7 L422.2,423.8 L422.3,423.8 L422.4,423.9 L422.5,423.9 L422.6,424.0 L422.7,424.0 L422.8,424.0 L422.9,424.1 L423.0,424.1 L423.1,424.1 L423.2,424.1 L423.3,424.2 L423.5,424.2 L423.6,424.2 L423.7,424.3 L423.8,424.3 L423.9,424.3 L424.0,424.4 L424.1,424.4 L424.2,424.4 L424.3,424.5 L424.4,424.5 L424.6,424.5 L424.7,424.6 L424.8,424.6 L424.9,424.6 L425.0,424.7 L425.1,424.7 L425.2,424.7 L425.3,424.8 L425.4,424.8 L425.6,424.8 L425.7,424.8 L425.8,424.9 L425.9,424.9 L426.0,424.9 L426.1,425.0 L426.2,425.0 L426.3,425.0 L426.4,425.1 L426.5,425.1 L426.7,425.1 L426.8,425.2 L426.9,425.2 L427.0,425.2 L427.1,425.3 L427.2,425.3 L427.3,425.3 L427.4,425.4 L427.5,425.4 L427.7,425.4 L427.8,425.4 L427.9,425.5 L428.0,425.5 L428.1,425.5 L428.2,425.6 L428.3,425.6 L428.4,425.6 L428.5,425.7 L428.6,425.7 L428.8,425.7 L428.9,425.8 L429.0,425.8 L429.1,425.8 L429.2,425.9 L429.3,425.9 L429.4,425.9 L429.5,426.0 L429.6,426.0 L429.8,426.0 L429.9,426.1 L430.0,426.1 L430.1,426.1 L430.2,426.1 L430.3,426.2 L430.4,426.2 L430.5,426.2 L430.6,426.3 L430.7,426.3 L430.9,426.3 L431.0,426.4 L431.1,426.4 L431.2,426.4 L431.3,426.5 L431.4,426.5 L431.5,426.5 L431.6,426.6 L431.7,426.6 L431.9,426.6 L432.0,426.7 L432.1,426.7 L432.2,426.7 L432.3,426.7 L432.4,426.8 L432.5,426.8 L432.6,426.8 L432.7,426.9 L432.8,426.9 L433.0,426.9 L433.1,427.0 L433.2,427.0 L433.3,427.0 L433.4,427.1 L433.5,427.1 L433.6,427.1 L433.7,427.2 L433.8,427.2 L434.0,427.2 L434.1,427.3 L434.2,427.3 L434.3,427.3 L434.4,427.4 L434.5,427.4 L434.6,427.4 L434.7,427.4 L434.8,427.5 L434.9,427.5 L435.1,427.5 L435.2,427.6 L435.3,427.6 L435.4,427.6 L435.5,427.7 L435.6,427.7 L435.7,427.7 L435.8,427.8 L435.9,427.8 L436.1,427.8 L436.2,427.9 L436.3,427.9 L436.4,427.9 L436.5,428.0 L436.6,428.0 L436.7,428.0 L436.8,428.1 L436.9,428.1 L437.0,428.1 L437.2,428.1 L437.3,428.2 L437.4,428.2 L437.5,428.2 L437.6,428.3 L437.7,428.3 L437.8,428.3 L437.9,428.4 L438.0,428.4 L438.2,428.4 L438.3,428.5 L438.4,428.5 L438.5,428.5 L438.6,428.6 L438.7,428.6 L438.8,428.6 L438.9,428.7 L439.0,428.7 L439.1,428.7 L439.3,428.7 L439.4,428.8 L439.5,428.8 L439.6,428.8 L439.7,428.9 L439.8,428.9 L439.9,428.9 L440.0,429.0 L440.1,429.0 L440.3,429.0 L440.4,429.1 L440.5,429.1 L440.6,429.1 L440.7,429.2 L440.8,429.2 L440.9,429.2 L441.0,429.3 L441.1,429.3 L441.2,429.3 L441.4,429.4 L441.5,429.4 L441.6,429.4 L441.7,429.4 L441.8,429.5 L441.9,429.5 L442.0,429.5 L442.1,429.6 L442.2,429.6 L442.4,429.6 L442.5,429.7 L442.6,429.7 L442.7,429.7 L442.8,429.8 L442.9,429.8 L443.0,429.8 L443.1,429.9 L443.2,429.9 L443.3,429.9 L443.5,430.0 L443.6,430.0 L443.7,430.0 L443.8,430.0 L443.9,430.1 L444.0,430.1 L444.1,430.1 L444.2,430.2 L444.3,430.2 L444.5,430.2 L444.6,430.3 L444.7,430.3 L444.8,430.3 L444.9,430.4 L445.0,430.4 L445.1,430.4 L445.2,430.5 L445.3,430.5 L445.4,430.5 L445.6,430.6 L445.7,430.6 L445.8,430.6 L445.9,430.7 L446.0,430.7 L446.1,430.7 L446.2,430.7 L446.3,430.8 L446.4,430.8 L446.6,430.8 L446.7,430.9 L446.8,430.9 L446.9,430.9 L447.0,431.0 L447.1,431.0 L447.2,431.0 L447.3,431.1 L447.4,431.1 L447.5,431.1 L447.7,431.2 L447.8,431.2 L447.9,431.2 L448.0,431.3 L448.1,431.3 L448.2,431.3 L448.3,431.4 L448.4,431.4 L448.5,431.4 L448.7,431.4 L448.8,431.5 L448.9,431.5 L449.0,431.5 L449.1,431.6 L449.2,431.6 L449.3,431.6 L449.4,431.7 L449.5,431.7 L449.6,431.7 L449.8,431.8 L449.9,431.8 L450.0,431.8 L450.1,431.9 L450.2,431.9 L450.3,431.9 L450.4,432.0 L450.5,432.0 L450.6,432.0 L450.8,432.0 L450.9,432.1 L451.0,432.1 L451.1,432.1 L451.2,432.2 L451.3,432.2 L451.4,432.2 L451.5,432.3 L451.6,432.3 L451.7,432.3 L451.9,432.4 L452.0,432.4 L452.1,432.4 L452.2,432.5 L452.3,432.5 L452.4,432.5 L452.5,432.6 L452.6,432.6 L452.7,432.6 L452.9,432.7 L453.0,432.7 L453.1,432.7 L453.2,432.7 L453.3,432.8 L453.4,432.8 L453.5,432.8 L453.6,432.9 L453.7,432.9 L453.8,432.9 L454.0,433.0 L454.1,433.0 L454.2,433.0 L454.3,433.1 L454.4,433.1 L454.5,433.1 L454.6,433.2 L454.7,433.2 L454.8,433.2 L455.0,433.3 L455.1,433.3 L455.2,433.3 L455.3,433.3 L455.4,433.4 L455.5,433.4 L455.6,433.4 L455.7,433.5 L455.8,433.5 L455.9,433.5 L456.1,433.6 L456.2,433.6 L456.3,433.6 L456.4,433.7 L456.5,433.7 L456.6,433.7 L456.7,433.8 L456.8,433.8 L456.9,433.8 L457.1,433.9 L457.2,433.9 L457.3,433.9 L457.4,434.0 L457.5,434.0 L457.6,434.0 L457.7,434.0 L457.8,434.1 L457.9,434.1 L458.0,434.1 L458.2,434.2 L458.3,434.2 L458.4,434.2 L458.5,434.3 L458.6,434.3 L458.7,434.3 L458.8,434.4 L458.9,434.4 L459.0,434.4 L459.2,434.5 L459.3,434.5 L459.4,434.5 L459.5,434.6 L459.6,434.6 L459.7,434.6 L459.8,434.6 L459.9,434.7 L460.0,434.7 L460.1,434.7 L460.3,434.8 L460.4,434.8 L460.5,434.8 L460.6,434.9 L460.7,434.9 L460.8,434.9 L460.9,435.0 L461.0,435.0 L461.1,435.0 L461.3,435.1 L461.4,435.1 L461.5,435.1 L461.6,435.2 L461.7,435.2 L461.8,435.2 L461.9,435.3 L462.0,435.3 L462.1,435.3 L462.2,435.3 L462.4,435.4 L462.5,435.4 L462.6,435.4 L462.7,435.5 L462.8,435.5 L462.9,435.5 L463.0,435.6 L463.1,435.6 L463.2,435.6 L463.4,435.7 L463.5,435.7 L463.6,435.7 L463.7,435.8 L463.8,435.8 L463.9,435.8 L464.0,435.9 L464.1,435.9 L464.2,435.9 L464.3,436.0 L464.5,436.0 L464.6,436.0 L464.7,436.0 L464.8,436.1 L464.9,436.1 L465.0,436.1 L465.1,436.2 L465.2,436.2 L465.3,436.2 L465.5,436.3 L465.6,436.3 L465.7,436.3 L465.8,436.4 L465.9,436.4 L466.0,436.4 L466.1,436.5 L466.2,436.5 L466.3,436.5 L466.4,436.6 L466.6,436.6 L466.7,436.6 L466.8,436.6 L466.9,436.7 L467.0,436.7 L467.1,436.7 L467.2,436.8 L467.3,436.8 L467.4,436.8 L467.6,436.9 L467.7,436.9 L467.8,436.9 L467.9,437.0 L468.0,437.0 L468.1,437.0 L468.2,437.1 L468.3,437.1 L468.4,437.1 L468.5,437.2 L468.7,437.2 L468.8,437.2 L468.9,437.3 L469.0,437.3 L469.1,437.3 L469.2,437.3 L469.3,437.4 L469.4,437.4 L469.5,437.4 L469.7,437.5 L469.8,437.5 L469.9,437.5 L470.0,437.6 L470.1,437.6 L470.2,437.6 L470.3,437.7 L470.4,437.7 L470.5,437.7 L470.6,437.8 L470.8,437.8 L470.9,437.8 L471.0,437.9 L471.1,437.9 L471.2,437.9 L471.3,437.9 L471.4,438.0 L471.5,438.0 L471.6,438.0 L471.8,438.1 L471.9,438.1 L472.0,438.1 L472.1,438.2 L472.2,438.2 L472.3,438.2 L472.4,438.3 L472.5,438.3 L472.6,438.3 L472.7,438.4 L472.9,438.4 L473.0,438.4 L473.1,438.5 L473.2,438.5 L473.3,438.5 L473.4,438.6 L473.5,438.6 L473.6,438.6 L473.7,438.6 L473.9,438.7 L474.0,438.7 L474.1,438.7 L474.2,438.8 L474.3,438.8 L474.4,438.8 L474.5,438.9 L474.6,438.9 L474.7,438.9 L474.8,439.0 L475.0,439.0 L475.1,439.0 L475.2,439.1 L475.3,439.1 L475.4,439.1 L475.5,439.2 L475.6,439.2 L475.7,439.2 L475.8,439.3 L476.0,439.3 L476.1,439.3 L476.2,439.3 L476.3,439.4 L476.4,439.4 L476.5,439.4 L476.6,439.5 L476.7,439.5 L476.8,439.5 L476.9,439.6 L477.1,439.6 L477.2,439.6 L477.3,439.7 L477.4,439.7 L477.5,439.7 L477.6,439.8 L477.7,439.8 L477.8,439.8 L477.9,439.9 L478.1,439.9 L478.2,439.9 L478.3,439.9 L478.4,440.0 L478.5,440.0 L478.6,440.0 L478.7,440.1 L478.8,440.1 L478.9,440.1 L479.0,440.2 L479.2,440.2 L479.3,440.2 L479.4,440.3 L479.5,440.3 L479.6,440.3 L479.7,440.4 L479.8,440.4 L479.9,440.4 L480.0,440.5 L480.2,440.5 L480.3,440.5 L480.4,440.6 L480.5,440.6 L480.6,440.6 L480.7,440.6 L480.8,440.7 L480.9,440.7 L481.0,440.7 L481.1,440.8 L481.3,440.8 L481.4,440.8 L481.5,440.9 L481.6,440.9 L481.7,440.9 L481.8,441.0 L481.9,441.0 L482.0,441.0 L482.1,441.1 L482.3,441.1 L482.4,441.1 L482.5,441.2 L482.6,441.2 L482.7,441.2 L482.8,441.2 L482.9,441.3 L483.0,441.3 L483.1,441.3 L483.2,441.4 L483.4,441.4 L483.5,441.4 L483.6,441.5 L483.7,441.5 L483.8,441.5 L483.9,441.6 L484.0,441.6 L484.1,441.6 L484.2,441.7 L484.4,441.7 L484.5,441.7 L484.6,441.8 L484.7,441.8 L484.8,441.8 L484.9,441.9 L485.0,441.9 L485.1,441.9 L485.2,441.9 L485.3,442.0 L485.5,442.0 L485.6,442.0 L485.7,442.1 L485.8,442.1 L485.9,442.1 L486.0,442.2 L486.1,442.2 L486.2,442.2 L486.3,442.3 L486.5,442.3 L486.6,442.3 L486.7,442.4 L486.8,442.4 L486.9,442.4 L487.0,442.5 L487.1,442.5 L487.2,442.5 L487.3,442.6 L487.4,442.6 L487.6,442.6 L487.7,442.6 L487.8,442.7 L487.9,442.7 L488.0,442.7 L488.1,442.8 L488.2,442.8 L488.3,442.8 L488.4,442.9 L488.6,442.9 L488.7,442.9 L488.8,443.0 L488.9,443.0 L489.0,443.0 L489.1,443.1 L489.2,443.1 L489.3,443.1 L489.4,443.2 L489.5,443.2 L489.7,443.2 L489.8,443.2 L489.9,443.3 L490.0,443.3 L490.1,443.3 L490.2,443.4 L490.3,443.4 L490.4,443.4 L490.5,443.5 L490.7,443.5 L490.8,443.5 L490.9,443.6 L491.0,443.6 L491.1,443.6 L491.2,443.7 L491.3,443.7 L491.4,443.7 L491.5,443.8 L491.6,443.8 L491.8,443.8 L491.9,443.9 L492.0,443.9 L492.1,443.9 L492.2,443.9 L492.3,444.0 L492.4,444.0 L492.5,444.0 L492.6,444.1 L492.8,444.1 L492.9,444.1 L493.0,444.2 L493.1,444.2 L493.2,444.2 L493.3,444.3 L493.4,444.3 L493.5,444.3 L493.6,444.4 L493.7,444.4 L493.9,444.4 L494.0,444.5 L494.1,444.5 L494.2,444.5 L494.3,444.5 L494.4,444.6 L494.5,444.6 L494.6,444.6 L494.7,444.7 L494.9,444.7 L495.0,444.7 L495.1,444.8 L495.2,444.8 L495.3,444.8 L495.4,444.9 L495.5,444.9 L495.6,444.9 L495.7,445.0 L495.8,445.0 L496.0,445.0 L496.1,445.1 L496.2,445.1 L496.3,445.1 L496.4,445.2 L496.5,445.2 L496.6,445.2 L496.7,445.2 L496.8,445.3 L497.0,445.3 L497.1,445.3 L497.2,445.4 L497.3,445.4 L497.4,445.4 L497.5,445.5 L497.6,445.5 L497.7,445.5 L497.8,445.6 L497.9,445.6 L498.1,445.6 L498.2,445.7 L498.3,445.7 L498.4,445.7 L498.5,445.8 L498.6,445.8 L498.7,445.8 L498.8,445.9 L498.9,445.9 L499.1,445.9 L499.2,445.9 L499.3,446.0 L499.4,446.0 L499.5,446.0 L499.6,446.1 L499.7,446.1 L499.8,446.1 L499.9,446.2 L500.0,446.2 L500.2,446.2 L500.3,446.3 L500.4,446.3 L500.5,446.3 L500.6,446.4 L500.7,446.4 L500.8,446.4 L500.9,446.5 L501.0,446.5 L501.2,446.5 L501.3,446.5 L501.4,446.6 L501.5,446.6 L501.6,446.6 L501.7,446.7 L501.8,446.7 L501.9,446.7 L502.0,446.8 L502.1,446.8 L502.3,446.8 L502.4,446.9 L502.5,446.9 L502.6,446.9 L502.7,447.0 L502.8,447.0 L502.9,447.0 L503.0,447.1 L503.1,447.1 L503.3,447.1 L503.4,447.2 L503.5,447.2 L503.6,447.2 L503.7,447.2 L503.8,447.3 L503.9,447.3 L504.0,447.3 L504.1,447.4 L504.2,447.4 L504.4,447.4 L504.5,447.5 L504.6,447.5 L504.7,447.5 L504.8,447.6 L504.9,447.6 L505.0,447.6 L505.1,447.7 L505.2,447.7 L505.4,447.7 L505.5,447.8 L505.6,447.8 L505.7,447.8 L505.8,447.8 L505.9,447.9 L506.0,447.9 L506.1,447.9 L506.2,448.0 L506.3,448.0 L506.5,448.0 L506.6,448.1 L506.7,448.1 L506.8,448.1 L506.9,448.2 L507.0,448.2 L507.1,448.2 L507.2,448.3 L507.3,448.3 L507.5,448.3 L507.6,448.4 L507.7,448.4 L507.8,448.4 L507.9,448.5 L508.0,448.5 L508.1,448.5 L508.2,448.5 L508.3,448.6 L508.4,448.6 L508.6,448.6 L508.7,448.7 L508.8,448.7 L508.9,448.7 L509.0,448.8 L509.1,448.8 L509.2,448.8 L509.3,448.9 L509.4,448.9 L509.6,448.9 L509.7,449.0 L509.8,449.0 L509.9,449.0 L510.0,449.1 L510.1,449.1 L510.2,449.1 L510.3,449.2 L510.4,449.2 L510.5,449.2 L510.7,449.2 L510.8,449.3 L510.9,449.3 L511.0,449.3 L511.1,449.4 L511.2,449.4 L511.3,449.4 L511.4,449.5 L511.5,449.5 L511.7,449.5 L511.8,449.6 L511.9,449.6 L512.0,449.6 L512.1,449.7 L512.2,449.7 L512.3,449.7 L512.4,449.8 L512.5,449.8 L512.6,449.8 L512.8,449.8 L512.9,449.9 L513.0,449.9 L513.1,449.9 L513.2,450.0 L513.3,450.0 L513.4,450.0 L513.5,450.1 L513.6,450.1 L513.8,450.1 L513.9,450.2 L514.0,450.2 L514.1,450.2 L514.2,450.3 L514.3,450.3 L514.4,450.3 L514.5,450.4 L514.6,450.4 L514.7,450.4 L514.9,450.5 L515.0,450.5 L515.1,450.5 L515.2,450.5 L515.3,450.6 L515.4,450.6 L515.5,450.6 L515.6,450.7 L515.7,450.7 L515.9,450.7 L516.0,450.8 L516.1,450.8 L516.2,450.8 L516.3,450.9 L516.4,450.9 L516.5,450.9 L516.6,451.0 L516.7,451.0 L516.8,451.0 L517.0,451.1 L517.1,451.1 L517.2,451.1 L517.3,451.1 L517.4,451.2 L517.5,451.2 L517.6,451.2 L517.7,451.3 L517.8,451.3 L518.0,451.3 L518.1,451.4 L518.2,451.4 L518.3,451.4 L518.4,451.5 L518.5,451.5 L518.6,451.5 L518.7,451.6 L518.8,451.6 L518.9,451.6 L519.1,451.7 L519.2,451.7 L519.3,451.7 L519.4,451.8 L519.5,451.8 L519.6,451.8 L519.7,451.8 L519.8,451.9 L519.9,451.9 L520.1,451.9 L520.2,452.0 L520.3,452.0 L520.4,452.0 L520.5,452.1 L520.6,452.1 L520.7,452.1 L520.8,452.2 L520.9,452.2 L521.0,452.2 L521.2,452.3 L521.3,452.3 L521.4,452.3 L521.5,452.4 L521.6,452.4 L521.7,452.4 L521.8,452.4 L521.9,452.5 L522.0,452.5 L522.2,452.5 L522.3,452.6 L522.4,452.6 L522.5,452.6 L522.6,452.7 L522.7,452.7 L522.8,452.7 L522.9,452.8 L523.0,452.8 L523.1,452.8 L523.3,452.9 L523.4,452.9 L523.5,452.9 L523.6,453.0 L523.7,453.0 L523.8,453.0 L523.9,453.1 L524.0,453.1 L524.1,453.1 L524.3,453.1 L524.4,453.2 L524.5,453.2 L524.6,453.2 L524.7,453.3 L524.8,453.3 L524.9,453.3 L525.0,453.4 L525.1,453.4 L525.2,453.4 L525.4,453.5 L525.5,453.5 L525.6,453.5 L525.7,453.6 L525.8,453.6 L525.9,453.6 L526.0,453.7 L526.1,453.7 L526.2,453.7 L526.4,453.8 L526.5,453.8 L526.6,453.8 L526.7,453.8 L526.8,453.9 L526.9,453.9 L527.0,453.9 L527.1,454.0 L527.2,454.0 L527.3,454.0 L527.5,454.1 L527.6,454.1 L527.7,454.1 L527.8,454.2 L527.9,454.2 L528.0,454.2 L528.1,454.3 L528.2,454.3 L528.3,454.3 L528.5,454.4 L528.6,454.4 L528.7,454.4 L528.8,454.4 L528.9,454.5 L529.0,454.5 L529.1,454.5 L529.2,454.6 L529.3,454.6 L529.4,454.6 L529.6,454.7 L529.7,454.7 L529.8,454.7 L529.9,454.8 L530.0,454.8 L530.1,454.8 L530.2,454.9 L530.3,454.9 L530.4,454.9 L530.6,455.0 L530.7,455.0 L530.8,455.0 L530.9,455.1 L531.0,455.1 L531.1,455.1 L531.2,455.1 L531.3,455.2 L531.4,455.2 L531.5,455.2 L531.7,455.3 L531.8,455.3 L531.9,455.3 L532.0,455.4 L532.1,455.4 L532.2,455.4 L532.3,455.5 L532.4,455.5 L532.5,455.5 L532.7,455.6 L532.8,455.6 L532.9,455.6 L533.0,455.7 L533.1,455.7 L533.2,455.7 L533.3,455.7 L533.4,455.8 L533.5,455.8 L533.6,455.8 L533.8,455.9 L533.9,455.9 L534.0,455.9 L534.1,456.0 L534.2,456.0 L534.3,456.0 L534.4,456.1 L534.5,456.1 L534.6,456.1 L534.8,456.2 L534.9,456.2 L535.0,456.2 L535.1,456.3 L535.2,456.3 L535.3,456.3 L535.4,456.4 L535.5,456.4 L535.6,456.4 L535.7,456.4 L535.9,456.5 L536.0,456.5 L536.1,456.5 L536.2,456.6 L536.3,456.6 L536.4,456.6 L536.5,456.7 L536.6,456.7 L536.7,456.7 L536.9,456.8 L537.0,456.8 L537.1,456.8 L537.2,456.9 L537.3,456.9 L537.4,456.9 L537.5,457.0 L537.6,457.0 L537.7,457.0 L537.8,457.1 L538.0,457.1 L538.1,457.1 L538.2,457.1 L538.3,457.2 L538.4,457.2 L538.5,457.2 L538.6,457.3 L538.7,457.3 L538.8,457.3 L539.0,457.4 L539.1,457.4 L539.2,457.4 L539.3,457.5 L539.4,457.5 L539.5,457.5 L539.6,457.6 L539.7,457.6 L539.8,457.6 L539.9,457.7 L540.1,457.7 L540.2,457.7 L540.3,457.7 L540.4,457.8 L540.5,457.8 L540.6,457.8 L540.7,457.9 L540.8,457.9 L540.9,457.9 L541.1,458.0 L541.2,458.0 L541.3,458.0 L541.4,458.1 L541.5,458.1 L541.6,458.1 L541.7,458.2 L541.8,458.2 L541.9,458.2 L542.0,458.3 L542.2,458.3 L542.3,458.3 L542.4,458.4 L542.5,458.4 L542.6,458.4 L542.7,458.4 L542.8,458.5 L542.9,458.5 L543.0,458.5 L543.2,458.6 L543.3,458.6 L543.4,458.6 L543.5,458.7 L543.6,458.7 L543.7,458.7 L543.8,458.8 L543.9,458.8 L544.0,458.8 L544.1,458.9 L544.3,458.9 L544.4,458.9 L544.5,459.0 L544.6,459.0 L544.7,459.0 L544.8,459.0 L544.9,459.1 L545.0,459.1 L545.1,459.1 L545.3,459.2 L545.4,459.2 L545.5,459.2 L545.6,459.3 L545.7,459.3 L545.8,459.3 L545.9,459.4 L546.0,459.4 L546.1,459.4 L546.3,459.5 L546.4,459.5 L546.5,459.5 L546.6,459.6 L546.7,459.6 L546.8,459.6 L546.9,459.7 L547.0,459.7 L547.1,459.7 L547.2,459.7 L547.4,459.8 L547.5,459.8 L547.6,459.8 L547.7,459.9 L547.8,459.9 L547.9,459.9 L548.0,460.0 L548.1,460.0 L548.2,460.0 L548.4,460.1 L548.5,460.1 L548.6,460.1 L548.7,460.2 L548.8,460.2 L548.9,460.2 L549.0,460.3 L549.1,460.3 L549.2,460.3 L549.3,460.4 L549.5,460.4 L549.6,460.4 L549.7,460.4 L549.8,460.5 L549.9,460.5 L550.0,460.5 L550.1,460.6 L550.2,460.6 L550.4,460.6 L550.5,460.6 L550.6,460.6 L550.7,460.5 L550.8,460.5 L550.9,460.5 L551.0,460.4 L551.1,460.4 L551.2,460.3 L551.3,460.3 L551.4,460.2 L551.5,460.1 L551.6,460.1 L551.7,460.0 L551.7,459.9 L551.8,459.8 L551.9,459.7 L551.9,459.6 L552.0,459.5 L552.0,459.4 L552.0,459.3 L552.1,459.2 L552.1,459.0 L552.1,458.9 L552.1,458.8 L552.1,458.7 L552.2,458.6 L552.2,458.5 L552.2,458.4 L552.2,458.3 L552.3,458.1 L552.3,458.0 L552.3,457.9 L552.3,457.8 L552.3,457.7 L552.4,457.6 L552.4,457.5 L552.4,457.3 L552.4,457.2 L552.5,457.1 L552.5,457.0 L552.5,456.9 L552.5,456.8 L552.5,456.7 L552.6,456.6 L552.6,456.4 L552.6,456.3 L552.6,456.2 L552.7,456.1 L552.7,456.0 L552.7,455.9 L552.7,455.8 L552.8,455.7 L552.8,455.5 L552.8,455.4 L552.8,455.3 L552.8,455.2 L552.9,455.1 L552.9,455.0 L552.9,454.9 L552.9,454.8 L553.0,454.6 L553.0,454.5 L553.0,454.4 L553.0,454.3 L553.0,454.2 L553.1,454.1 L553.1,454.0 L553.1,453.9 L553.1,453.7 L553.2,453.6 L553.2,453.5 L553.2,453.4 L553.2,453.3 L553.2,453.2 L553.3,453.1 L553.3,452.9 L553.3,452.8 L553.3,452.7 L553.4,452.6 L553.4,452.5 L553.4,452.4 L553.4,452.3 L553.4,452.2 L553.5,452.0 L553.5,451.9 L553.5,451.8 L553.5,451.7 L553.6,451.6 L553.6,451.5 L553.6,451.4 L553.6,451.3 L553.6,451.1 L553.7,451.0 L553.7,450.9 L553.7,450.8 L553.7,450.7 L553.8,450.6 L553.8,450.5 L553.8,450.4 L553.8,450.2 L553.9,450.1 L553.9,450.0 L553.9,449.9 L553.9,449.8 L553.9,449.7 L554.0,449.6 L554.0,449.5 L554.0,449.3 L554.0,449.2 L554.1,449.1 L554.1,449.0 L554.1,448.9 L554.1,448.8 L554.1,448.7 L554.2,448.6 L554.2,448.4 L554.2,448.3 L554.2,448.2 L554.3,448.1 L554.3,448.0 L554.3,447.9 L554.3,447.8 L554.3,447.6 L554.4,447.5 L554.4,447.4 L554.4,447.3 L554.4,447.2 L554.5,447.1 L554.5,447.0 L554.5,446.9 L554.5,446.7 L554.5,446.6 L554.6,446.5 L554.6,446.4 L554.6,446.3 L554.6,446.2 L554.7,446.1 L554.7,446.0 L554.7,445.8 L554.7,445.7 L554.7,445.6 L554.8,445.5 L554.8,445.4 L554.8,445.3 L554.8,445.2 L554.9,445.1 L554.9,444.9 L554.9,444.8 L554.9,444.7 L555.0,444.6 L555.0,444.5 L555.0,444.4 L555.0,444.3 L555.0,444.2 L555.1,444.0 L555.1,443.9 L555.1,443.8 L555.1,443.7 L555.2,443.6 L555.2,443.5 L555.2,443.4 L555.2,443.3 L555.2,443.1 L555.3,443.0 L555.3,442.9 L555.3,442.8 L555.3,442.7 L555.4,442.6 L555.4,442.5 L555.4,442.3 L555.4,442.2 L555.4,442.1 L555.5,442.0 L555.5,441.9 L555.5,441.8 L555.5,441.7 L555.6,441.6 L555.6,441.4 L555.6,441.3 L555.6,441.2 L555.6,441.1 L555.7,441.0 L555.7,440.9 L555.7,440.8 L555.7,440.7 L555.8,440.5 L555.8,440.4 L555.8,440.3 L555.8,440.2 L555.8,440.1 L555.9,440.0 L555.9,439.9 L555.9,439.8 L555.9,439.6 L556.0,439.5 L556.0,439.4 L556.0,439.3 L556.0,439.2 L556.1,439.1 L556.1,439.0 L556.1,438.9 L556.1,438.7 L556.1,438.6 L556.2,438.5 L556.2,438.4 L556.2,438.3 L556.2,438.2 L556.3,438.1 L556.3,437.9 L556.3,437.8 L556.3,437.7 L556.3,437.6 L556.4,437.5 L556.4,437.4 L556.4,437.3 L556.4,437.2 L556.5,437.0 L556.5,436.9 L556.5,436.8 L556.5,436.7 L556.5,436.6 L556.6,436.5 L556.6,436.4 L556.6,436.3 L556.6,436.1 L556.7,436.0 L556.7,435.9 L556.7,435.8 L556.7,435.7 L556.7,435.6 L556.8,435.5 L556.8,435.4 L556.8,435.2 L556.8,435.1 L556.9,435.0 L556.9,434.9 L556.9,434.8 L556.9,434.7 L556.9,434.6 L557.0,434.5 L557.0,434.3 L557.0,434.2 L557.0,434.1 L557.1,434.0 L557.1,433.9 L557.1,433.8 L557.1,433.7 L557.2,433.6 L557.2,433.4 L557.2,433.3 L557.2,433.2 L557.2,433.1 L557.3,433.0 L557.3,432.9 L557.3,432.8 L557.3,432.6 L557.4,432.5 L557.4,432.4 L557.4,432.3 L557.4,432.2 L557.4,432.1 L557.5,432.0 L557.5,431.9 L557.5,431.7 L557.5,431.6 L557.6,431.5 L557.6,431.4 L557.6,431.3 L557.6,431.2 L557.6,431.1 L557.7,431.0 L557.7,430.8 L557.7,430.7 L557.7,430.6 L557.8,430.5 L557.8,430.4 L557.8,430.3 L557.8,430.2 L557.9,430.1 L557.9,430.0 L557.9,429.8 L558.0,429.7 L558.1,429.6 L558.1,429.6 L558.2,429.5 L558.3,429.4 L558.3,429.3 L558.4,429.2 L558.5,429.1 L558.5,429.0 L558.6,428.9 L558.7,428.8 L558.7,428.7 L558.8,428.6 L558.9,428.5 L558.9,428.4 L559.0,428.3 L559.1,428.2 L559.1,428.1 L559.2,428.1 L559.2,428.0 L559.3,427.8 L559.3,427.7 L559.4,427.6 L559.4,427.5 L559.4,427.4 L559.5,427.3 L559.5,427.2 L559.5,427.1 L559.5,427.0 L559.5,426.8 L559.6,426.7 L559.6,426.6 L559.6,426.5 L559.6,426.4 L559.7,426.3 L559.7,426.2 L559.7,426.1 L559.7,425.9 L559.7,425.8 L559.7,425.7 L559.7,425.6 L559.7,425.5 L559.7,425.4 L559.7,425.3 L559.7,425.1 L559.7,425.0 L559.6,424.9 L559.6,424.8 L559.6,424.7 L559.6,424.6 L559.5,424.5 L559.5,424.4 L559.5,424.2 L559.5,424.1 L559.4,424.0 L559.4,423.9 L559.4,423.8 L559.4,423.7 L559.3,423.6 L559.3,423.5 L559.3,423.3 L559.3,423.2 L559.3,423.1 L559.3,423.0 L559.3,422.9 L559.3,422.8 L559.3,422.7 L559.3,422.5 L559.4,422.4 L559.4,422.3 L559.4,422.2 L559.4,422.1 L559.5,422.0 L559.5,421.9 L559.5,421.8 L559.5,421.6 L559.5,421.5 L559.6,421.4 L559.6,421.3 L559.6,421.2 L559.6,421.1 L559.7,421.0 L559.7,420.9 L559.7,420.7 L559.7,420.6 L559.7,420.5 L559.8,420.4 L559.8,420.3 L559.8,420.2 L559.8,420.1 L559.9,420.0 L559.9,419.8 L559.9,419.7 L559.9,419.6 L559.9,419.5 L560.0,419.4 L560.0,419.3 L560.0,419.2 L560.0,419.1 L560.1,418.9 L560.1,418.8 L560.1,418.7 L560.1,418.6 L560.1,418.5 L560.2,418.4 L560.2,418.3 L560.2,418.2 L560.2,418.0 L560.3,417.9 L560.3,417.8 L560.3,417.7 L560.3,417.6 L560.4,417.5 L560.4,417.4 L560.4,417.2 L560.4,417.1 L560.4,417.0 L560.5,416.9 L560.5,416.8 L560.5,416.7 L560.5,416.6 L560.6,416.5 L560.6,416.3 L560.6,416.2 L560.6,416.1 L560.6,416.0 L560.7,415.9 L560.7,415.8 L560.7,415.7 L560.7,415.6 L560.8,415.4 L560.8,415.3 L560.8,415.2 L560.8,415.1 L560.8,415.0 L560.9,414.9 L560.9,414.8 L560.9,414.7 L560.9,414.5 L561.0,414.4 L561.0,414.3 L561.0,414.2 L561.0,414.1 L561.0,414.0 L561.1,413.9 L561.1,413.8 L561.1,413.6 L561.1,413.5 L561.2,413.4 L561.2,413.3 L561.2,413.2 L561.2,413.1 L561.2,413.0 L561.3,412.8 L561.3,412.7 L561.3,412.6 L561.3,412.5 L561.4,412.4 L561.4,412.3 L561.4,412.2 L561.4,412.1 L561.5,411.9 L561.5,411.8 L561.5,411.7 L561.5,411.6 L561.5,411.5 L561.6,411.4 L561.6,411.3 L561.6,411.2 L561.6,411.0 L561.7,410.9 L561.7,410.8 L561.7,410.7 L561.7,410.6 L561.7,410.5 L561.8,410.4 L561.8,410.3 L561.8,410.1 L561.8,410.0 L561.9,409.9 L561.9,409.8 L561.9,409.7 L561.9,409.6 L561.9,409.5 L562.0,409.4 L562.0,409.2 L562.0,409.1 L562.0,409.0 L562.1,408.9 L562.1,408.8 L562.1,408.7 L562.1,408.6 L562.1,408.5 L562.2,408.3 L562.2,408.2 L562.2,408.1 L562.2,408.0 L562.3,407.9 L562.3,407.8 L562.3,407.7 L562.3,407.5 L562.3,407.4 L562.4,407.3 L562.4,407.2 L562.4,407.1 L562.4,407.0 L562.5,406.9 L562.5,406.8 L562.5,406.6 L562.5,406.5 L562.6,406.4 L562.6,406.3 L562.6,406.2 L562.6,406.1 L562.6,406.0 L562.7,405.9 L562.7,405.7 L562.7,405.6 L562.7,405.5 L562.8,405.4 L562.8,405.3 L562.8,405.2 L562.8,405.1 L562.9,405.0 L562.9,404.8 L562.9,404.7 L562.9,404.6 L563.0,404.5 L563.0,404.4 L563.0,404.3 L563.0,404.2 L563.1,404.1 L563.1,404.0 L563.1,403.8 L563.2,403.7 L563.2,403.6 L563.2,403.5 L563.2,403.4 L563.3,403.3 L563.3,403.2 L563.3,403.1 L563.4,403.0 L563.4,402.8 L563.4,402.7 L563.5,402.6 L563.5,402.5 L563.5,402.4 L563.6,402.3 L563.6,402.2 L563.7,402.1 L563.7,402.0 L563.7,401.9 L563.8,401.8 L563.8,401.6 L563.8,401.5 L563.9,401.4 L563.9,401.3 L564.0,401.2 L564.0,401.1 L564.0,401.0 L564.1,400.9 L564.1,400.8 L564.2,400.7 L564.2,400.6 L564.3,400.5 L564.3,400.4 L564.3,400.2 L564.4,400.1 L564.4,400.0 L564.5,399.9 L564.5,399.8 L564.6,399.7 L564.6,399.6 L564.7,399.5 L564.7,399.4 L564.8,399.3 L564.8,399.2 L564.9,399.1 L564.9,399.0 L565.0,398.9 L565.0,398.8 L565.1,398.7 L565.1,398.6 L565.2,398.5 L565.2,398.4 L565.3,398.3 L565.3,398.2 L565.4,398.1 L565.4,398.0 L565.5,397.9 L565.5,397.8 L565.6,397.7 L565.6,397.6 L565.7,397.5 L565.7,397.4 L565.8,397.3 L565.9,397.2 L565.9,397.1 L566.0,397.0 L566.0,396.9 L566.1,396.8 L566.1,396.7 L566.2,396.6 L566.3,396.5 L566.3,396.4 L566.4,396.3 L566.4,396.2 L566.5,396.1 L566.6,396.0 L566.6,395.9 L566.7,395.8 L566.7,395.7 L566.8,395.6 L566.9,395.5 L566.9,395.4 L567.0,395.3 L567.1,395.2 L567.1,395.1 L567.2,395.0 L567.3,394.9 L567.3,394.8 L567.4,394.7 L567.5,394.6 L567.5,394.5 L567.6,394.4 L567.7,394.3 L567.7,394.3 L567.8,394.2 L567.9,394.1 L567.9,394.0 L568.0,393.9 L568.1,393.8 L568.1,393.7 L568.2,393.6 L568.3,393.5 L568.4,393.4 L568.4,393.3 L568.5,393.3 L568.6,393.2 L568.6,393.1 L568.7,393.0 L568.8,392.9 L568.9,392.8 L568.9,392.7 L569.0,392.6 L569.1,392.5 L569.2,392.5 L569.2,392.4 L569.3,392.3 L569.4,392.2 L569.5,392.1 L569.5,392.0 L569.6,391.9 L569.7,391.9 L569.8,391.8 L569.9,391.7 L569.9,391.6 L570.0,391.5 L570.1,391.4 L570.2,391.4 L570.2,391.3 L570.3,391.2 L570.4,391.1 L570.5,391.0 L570.6,390.9 L570.7,390.9 L570.7,390.8 L570.8,390.7 L570.9,390.6 L571.0,390.5 L571.1,390.5 L571.1,390.4 L571.2,390.3 L571.3,390.2 L571.4,390.1 L571.5,390.1 L571.6,390.0 L571.7,389.9 L571.7,389.8 L571.8,389.8 L571.9,389.7 L572.0,389.6 L572.1,389.5 L572.2,389.5 L572.3,389.4 L572.3,389.3 L572.4,389.2 L572.5,389.2 L572.6,389.1 L572.7,389.0 L572.8,388.9 L572.9,388.9 L573.0,388.8 L573.1,388.7 L573.2,388.7 L573.2,388.6 L573.3,388.5 L573.4,388.4 L573.5,388.4 L573.6,388.3 L573.7,388.2 L573.8,388.2 L573.9,388.1 L574.0,388.0 L574.1,388.0 L574.2,387.9 L574.3,387.8 L574.4,387.8 L574.4,387.7 L574.5,387.6 L574.6,387.6 L574.7,387.5 L574.8,387.4 L574.9,387.4 L575.0,387.3 L575.1,387.2 L575.2,387.2 L575.3,387.1 L575.4,387.1 L575.5,387.0 L575.6,386.9 L575.7,386.9 L575.8,386.8 L575.9,386.8 L576.0,386.7 L576.1,386.6 L576.2,386.6 L576.3,386.5 L576.4,386.5 L576.5,386.4 L576.6,386.3 L576.7,386.3 L576.8,386.2 L576.9,386.2 L577.0,386.1 L577.1,386.1 L577.2,386.0 L577.3,386.0 L577.4,385.9 L577.5,385.8 L577.6,385.8 L577.7,385.7 L577.8,385.7 L577.9,385.6 L578.0,385.6 L578.1,385.5 L578.2,385.5 L578.3,385.4 L578.4,385.4 L578.5,385.3 L578.6,385.3 L578.7,385.2 L578.8,385.2 L578.9,385.1 L579.0,385.1 L579.1,385.0 L579.2,385.0 L579.3,384.9 L579.4,384.8 L579.5,384.8 L579.6,384.7 L579.7,384.7 L579.8,384.6 L579.9,384.6 L580.0,384.5 L580.1,384.5 L580.3,384.4 L580.4,384.4 L580.5,384.3 L580.6,384.3 L580.7,384.2 L580.8,384.2 L580.9,384.1 L581.0,384.1 L581.1,384.0 L581.2,384.0 L581.3,383.9 L581.4,383.9 L581.5,383.8 L581.6,383.8 L581.7,383.7 L581.8,383.6 L581.9,383.6 L582.0,383.5 L582.1,383.5 L582.2,383.4 L582.3,383.4 L582.4,383.3 L582.5,383.3 L582.6,383.2 L582.7,383.2 L582.8,383.1 L582.9,383.1 L583.0,383.0 L583.1,383.0 L583.2,382.9 L583.3,382.9 L583.4,382.8 L583.5,382.8 L583.6,382.7 L583.7,382.7 L583.8,382.6 L583.9,382.6 L584.0,382.5 L584.1,382.4 L584.2,382.4 L584.3,382.3 L584.5,382.3 L584.6,382.2 L584.7,382.2 L584.8,382.1 L584.9,382.1 L585.0,382.0 L585.1,382.0 L585.2,381.9 L585.3,381.9 L585.4,381.8 L585.5,381.8 L585.6,381.7 L585.7,381.7 L585.8,381.6 L585.9,381.6 L586.0,381.5 L586.1,381.5 L586.2,381.4 L586.3,381.3 L586.4,381.3 L586.5,381.2 L586.6,381.2 L586.7,381.1 L586.8,381.1 L586.9,381.0 L587.0,381.0 L587.1,380.9 L587.2,380.9 L587.3,380.8 L587.4,380.8 L587.5,380.7 L587.6,380.7 L587.7,380.6 L587.8,380.6 L587.9,380.5 L588.0,380.5 L588.1,380.4 L588.2,380.4 L588.3,380.3 L588.4,380.3 L588.5,380.2 L588.7,380.1 L588.8,380.1 L588.9,380.0 L589.0,380.0 L589.1,379.9 L589.2,379.9 L589.3,379.8 L589.4,379.8 L589.5,379.7 L589.6,379.7 L589.7,379.6 L589.8,379.6 L589.9,379.5 L590.0,379.5 L590.1,379.4 L590.2,379.4 L590.3,379.3 L590.4,379.3 L590.5,379.2 L590.6,379.2 L590.7,379.1 L590.8,379.0 L590.9,379.0 L591.0,378.9 L591.1,378.9 L591.2,378.8 L591.3,378.8 L591.4,378.7 L591.5,378.7 L591.6,378.6 L591.7,378.6 L591.8,378.5 L591.9,378.5 L592.0,378.4 L592.1,378.4 L592.2,378.3 L592.3,378.3 L592.4,378.2 L592.5,378.2 L592.6,378.1 L592.7,378.1 L592.9,378.0 L593.0,378.0 L593.1,377.9 L593.2,377.8 L593.3,377.8 L593.4,377.7 L593.5,377.7 L593.6,377.6 L593.7,377.6 L593.8,377.5 L593.9,377.5 L594.0,377.4 L594.1,377.4 L594.2,377.3 L594.3,377.3 L594.4,377.2 L594.5,377.2 L594.6,377.1 L594.7,377.1 L594.8,377.0 L594.9,377.0 L595.0,376.9 L595.1,376.9 L595.2,376.8 L595.3,376.8 L595.4,376.7 L595.5,376.6 L595.6,376.6 L595.7,376.5 L595.8,376.5 L595.9,376.4 L596.0,376.4 L596.1,376.3 L596.2,376.3 L596.3,376.2 L596.4,376.2 L596.5,376.1 L596.6,376.0 L596.7,376.0 L596.8,375.9 L596.9,375.8 L597.0,375.8 L597.1,375.7 L597.2,375.6 L597.3,375.6 L597.4,375.5 L597.5,375.4 L597.6,375.4 L597.6,375.3 L597.7,375.2 L597.8,375.1 L597.9,375.1 L598.0,375.0 L598.1,374.9 L598.2,374.8 L598.2,374.7 L598.3,374.7 L598.4,374.6 L598.5,374.5 L598.6,374.4 L598.6,374.3 L598.7,374.2 L598.8,374.2 L598.9,374.1 L598.9,374.0 L599.0,373.9 L599.1,373.8 L599.2,373.7 L599.2,373.6 L599.3,373.5 L599.4,373.4 L599.4,373.3 L599.5,373.3 L599.6,373.2 L599.6,373.1 L599.7,373.0 L599.7,372.9 L599.8,372.8 L599.9,372.7 L599.9,372.6 L600.0,372.5 L600.0,372.4 L600.1,372.3 L600.1,372.2 L600.2,372.1 L600.2,372.0 L600.3,371.9 L600.3,371.7 L600.4,371.6 L600.4,371.5 L600.5,371.4 L600.5,371.3 L600.5,371.2 L600.6,371.1 L600.6,371.0 L600.7,370.9 L600.7,370.8 L600.7,370.7 L600.8,370.6 L600.8,370.5 L600.8,370.3 L600.9,370.2 L600.9,370.1 L600.9,370.0 L600.9,369.9 L601.0,369.8 L601.0,369.7 L601.0,369.6 L601.0,369.4 L601.1,369.3 L601.1,369.2 L601.1,369.1 L601.1,369.0 L601.1,368.9 L601.1,368.8 L601.1,368.6 L601.2,368.5 L601.2,368.4 L601.2,368.3 L601.2,368.2 L601.2,368.1 L601.2,368.0 L601.2,367.8 L601.2,367.7 L601.2,367.6 L601.2,367.5 L601.2,367.4 L601.2,367.3 L601.2,367.2 L601.2,367.0 L601.2,366.9 L601.1,366.8 L601.1,366.7 L601.1,366.6 L601.1,366.5 L601.1,366.4 L601.1,366.2 L601.1,366.1 L601.0,366.0 L601.0,365.9 L601.0,365.8 L601.0,365.7 L600.9,365.6 L600.9,365.4 L600.9,365.3 L600.9,365.2 L600.8,365.1 L600.8,365.0 L600.8,364.9 L600.7,364.8 L600.7,364.7 L600.7,364.6 L600.6,364.5 L600.6,364.3 L600.6,364.2 L600.5,364.1 L600.5,364.0 L600.4,363.9 L600.4,363.8 L600.3,363.7 L600.3,363.6 L600.2,363.5 L600.2,363.4 L600.1,363.3 L600.1,363.2 L600.0,363.1 L600.0,363.0 L599.9,362.9 L599.9,362.8 L599.8,362.7 L599.7,362.6 L599.7,362.5 L599.6,362.4 L599.6,362.3 L599.5,362.2 L599.4,362.1 L599.4,362.0 L599.3,361.9 L599.2,361.8 L599.2,361.7 L599.1,361.6 L599.0,361.5 L599.0,361.4 L598.9,361.4 L598.8,361.3 L598.8,361.2 L598.7,361.1 L598.6,361.0 L598.6,360.9 L598.5,360.8 L598.5,360.7 L598.4,360.6 L598.3,360.5 L598.3,360.4 L598.2,360.3 L598.1,360.2 L598.1,360.1 L598.0,360.0 L597.9,359.9 L597.9,359.8 L597.8,359.7 L597.7,359.6 L597.7,359.6 L597.6,359.5 L597.5,359.4 L597.5,359.3 L597.4,359.2 L597.3,359.1 L597.3,359.0 L597.2,358.9 L597.1,358.8 L597.1,358.7 L597.0,358.6 L597.0,358.5 L596.9,358.4 L596.8,358.3 L596.8,358.2 L596.7,358.1 L596.6,358.0 L596.6,357.9 L596.5,357.8 L596.4,357.7 L596.4,357.7 L596.3,357.6 Z" fill="none" stroke="#5f6673" stroke-width="3" stroke-linejoin="round"/><circle cx="596.2" cy="357.5" r="3.4" fill="#1d2330" stroke="#ffffff" stroke-width="1"/><text x="501.2" y="491.1" font-family="DejaVu Sans" font-size="9.5" fill="#5f6673" text-anchor="middle" font-weight="normal">6.0 km · road 7.5–9.3 m</text><path d="M10,513.1 L67.5,513.1" stroke="#1d2330" stroke-width="1.5"/><path d="M10,509.1 L10,517.1 M67.5,509.1 L67.5,517.1" stroke="#1d2330" stroke-width="1.2"/><text x="75.5" y="516.6" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="start" font-weight="normal">500 m, the same scale in every panel (road width not to scale)</text><circle cx="605" cy="513.1" r="2.6" fill="#1d2330"/><text x="613.0" y="516.6" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="start" font-weight="normal">start</text></svg>
<p><em>Figure 5. One map of each difficulty, to the same scale: generated maps at Easy, Medium and Hard (display seed 1000, as the race view shows them, §10.3), Austin, one of the 19 real training circuits that make up Extra hard, and validation map 80. From Easy to Hard the maps grow longer, their roads narrower and their corners tighter; Extra hard brings the real circuits, and the validation map is the longest and narrowest shown. Each road is drawn as a line; its length and width are given under it.</em></p>
</div>

**Road width.** The width of a generated map changes along the lap. A base width is drawn uniformly from the level's
range by `default_rng([seed, level, 13])` and multiplied by 1 + 0.2 p, where p, the mean of three sinusoids of one, two
and three periods per lap with random phases from the same generator, lies within ±1; the result is clipped to the
level's range, and the left and right halves are equal. The width depends only on the map's seed and level, so a map
keeps it on every generation attempt; the generator sizes chicanes and esses and the spacing between separate parts of
the road by the widest point of the road.

The values in Table 1 are those of `config.LEVELS`. They were chosen so that the expert completes every level and the
generator accepts a candidate within a few attempts (at most three on seeds 0–19 of each level and the validation
maps).

### 4.3 Circuit generator

The circuit generator builds a polygon of straights and rounds its corners with circular arcs.

1. **Base polygon.** Control points are placed at jittered angles around the origin with random radii.
   The polygon is stretched horizontally by a random factor between 1.0 and 1.5, so that elongated
   layouts with long straights occur.
2. **Main straight.** While the longest edge is shorter than the level's main-straight length, the
   vertex following that edge's end is removed, which joins two edges into one.
3. **Hairpins.** The vertex at the end of the main straight, and on the validation setting one further random
   vertex, is moved outwards by a factor between 1.25 and 1.5. This sharpens the corner into a hairpin.
4. **Chicanes.** A chicane inserts four vertices near the middle of a long edge (longer than 250 m):
   the road shifts sideways by 0.9 to 1.3 road widths over about 60–100 m and then returns.
5. **Esses.** A run of esses inserts three or four vertices along a long edge (longer than 400 m),
   offset alternately to each side by 1.2 to 2.0 road widths and spaced 70–110 m apart. Chicanes and esses are
   placed on the longest edges other than the main straight, one feature per edge; a feature whose edge is too short
   is left out, so the feature counts of Table 1 are upper bounds.
6. **Corner radii.** Each vertex is rounded with a circular arc. Hairpin and chicane vertices use a
   radius within 6 m above the level's tightest corner. Esses use a slow radius raised by 6 m. Other vertices use a slow
   or a fast radius with equal probability. A radius is reduced where the adjacent edges are too short
   to hold the arc.
7. **Acceptance.** The geometry check rejects a candidate whose tightest corner is below the
   level's limit or whose road comes too close to another part of itself. The expert must then
   complete the track. A rejected candidate is replaced by the next attempt for the same seed.

The features follow common Formula 1 circuit elements. A long straight into a hairpin resembles turn 1
at Bahrain or the hairpin at Montreal. Chicanes resemble those at Monza. Esses resemble the S-curves at
Suzuka and Maggotts–Becketts at Silverstone. Fast sweepers mixed with slow corners resemble Silverstone
and Spa. The straights, chicanes and esses are of Formula 1 size, but the generator reproduces no real circuit; real
circuits enter at Extra hard (§4.7).

### 4.4 Direction

Each map runs clockwise or counter-clockwise with equal probability. The direction is drawn from the
map's seed and level (not the attempt), so a seed always produces the same map in the same direction.

A real training circuit is driven in either direction: the 19 circuits give 38 layouts, the pairs of circuit and
direction, and each generation draws its layouts from these 38, so one generation can hold a circuit in both
directions. A reversed layout starts at the same point and runs the other way, its left and right widths swapped; it
uses the road grid of the racing direction. A test circuit is driven only in its racing direction, the direction of
its pole lap (§4.7).

### 4.5 Validation

A generated map is rejected if any of the following holds:

- the road comes too close to another part of itself
- any corner is tighter than the level's tightest corner
- the expert driver cannot complete a lap on its racing line (§7.1)

A rejected seed is retried deterministically using the triple (seed, level, attempt), so a given seed
always yields the same map. The direction is drawn from the seed and level alone (§4.4), so it is the same for every
attempt. Real circuits are not rejected: the expert must complete three laps of every training circuit in both
directions and of every test circuit in its racing direction, which a test checks (§12.1).

### 4.6 Seed allocation

- **Training (evolution and DAgger):** on levels 1–3, maps from seeds drawn by each run's random number generator
  from 2,000,000 upwards, never repeated within a run. On Extra hard, each generation draws three different layouts
  from the 38 layouts of the training circuits (19 circuits in two directions) with the run's generator; since the set
  is finite, layouts repeat between generations. DAgger draws its drives and checks from the same 38 layouts (§8.4).
- **Imitation dataset:** 41 seeds drawn once from 1,000,000–1,999,999 (one at Easy, 20 at Medium and 20 at Hard) and
  the 38 layouts of the training circuits.
- **Validation:** seeds 80–99 at the validation setting (Table 1), fixed.
- **Test:** Melbourne, Sakhir, Sepang and Silverstone in their racing direction (§4.7). They are never used for
  training, validation or model selection; the dataset's test split is recorded on them only to measure accuracy.
- **Application races:** at Extra hard, the race view races on the test circuits (§10.3); at the generated levels and
  the validation setting it builds the display seeds 1000–1019. These display maps are never used for training,
  selection or the reported results.

The validation maps and the test circuits remain fixed sets, so that every method is evaluated on the same tracks.

### 4.7 Real circuits

Extra hard and the test use real circuits from the racetrack database of the Technical University of Munich
(`TUMFTM/racetrack-database`, LGPL-3.0 licence), vendored in `data/racetracks/` with its licence (§15). For each
circuit, the database gives a centre line from OpenStreetMap GPS points, smoothed, and the track's width to the right
and to the left of every point, extracted from satellite images; the points are 5 m apart. The layouts are those of
about 2019. The accuracy varies between circuits, and because the centre line is not exactly central, the two widths
of a point differ by up to 2.2 m on the 23 circuits used.

- **Import.** The centre line and both widths are interpolated by periodic cubic splines and resampled at 1 m, at real
  scale. The widths are used as given, without clipping: the real circuits are 7.4 to 27.6 m wide at
  single points, the widest being the approach to Austin's first corner. The start is the database's first point, at
  or near the start–finish line. Each circuit's racing direction is recorded with it, checked against the official
  circuit map for the Formula 1 circuits (the others keep the database's order); the four test circuits import within
  0.2% of their official lengths (Melbourne 5.299 km against 5.303 km, Sakhir 5.407 against 5.412, Sepang 5.538
  against 5.543 and Silverstone 5.887 against 5.891).
- **The 25 circuits.** Eighteen Formula 1 circuits: Austin, Budapest, Catalunya, Hockenheim, Melbourne, Mexico City,
  Montreal, Monza, Sakhir, São Paulo, Sepang, Shanghai, Silverstone, Sochi, Spa, Spielberg, Suzuka and Yas Marina;
  and seven others from DTM and IndyCar: Brands Hatch, Indianapolis, Moscow Raceway, Norisring, Nürburgring,
  Oschersleben and Zandvoort.
- **Test circuits.** About a fifth of the Formula 1 circuits are held out: the four drawn by
  `numpy.random.default_rng(0).choice(names, 4, replace=False)` over the 18 Formula 1 names in alphabetical order,
  which gives Melbourne, Sakhir, Sepang and Silverstone. Holding out whole circuits means a controller cannot learn the
  shapes of the circuits it is tested on; it must have learned to drive.
- **Training circuits.** The other 14 Formula 1 circuits, and those of the seven others that are at least as complex
  as the simplest Formula 1 circuit, Monza, by every measure of Table 2: at least 8 corners per lap, at least 1.38
  corners per kilometre and at least 172° of turning per kilometre. A corner is a stretch whose radius, smoothed over
  20 m, stays below 300 m and that turns by more than 20°. Five qualify. Indianapolis, the oval, and Norisring, a
  2.3 km street circuit with five corners, do not. Extra hard therefore holds 19 circuits: Austin, Budapest,
  Catalunya, Hockenheim, Mexico City, Montreal, Monza, São Paulo, Shanghai, Sochi, Spa, Spielberg, Suzuka, Yas Marina,
  Brands Hatch, Moscow Raceway, Nürburgring, Oschersleben and Zandvoort.

*Table 2. Complexity of the seven non-Formula 1 circuits, against the range of the 18 Formula 1 circuits.*

| Circuit | Length (km) | Corners | Corners per km | Turning per km | Training |
|---|---|---|---|---|---|
| Moscow Raceway | 4.06 | 17 | 4.18 | 412° | Yes |
| Zandvoort | 4.32 | 13 | 3.01 | 380° | Yes |
| Oschersleben | 3.69 | 14 | 3.79 | 366° | Yes |
| Nürburgring | 5.14 | 15 | 2.92 | 321° | Yes |
| Brands Hatch | 3.90 | 13 | 3.33 | 274° | Yes |
| Norisring | 2.30 | 5 | 2.18 | 297° | No: five corners |
| Indianapolis | 4.02 | 5 | 1.24 | 91° | No: the oval |
| *Formula 1 range* | *4.30–7.00* | *8–22* | *1.38–3.99* | *172°–358°* | |

### 4.8 Racing line

Every track has a racing line: the minimum-curvature line within its road, the line of least turning
along which a car can carry more speed through the corners than on the centre line, and an approximation of the
fastest line (§14 item 29). It is computed once per track, in the same way for generated maps and real circuits, by
`racing_line.solve`:

- **Nodes.** Every k-th centre-line point is a node, k = 15 m / spacing (at least 8 nodes per track). A node's unknown
  is its lateral offset α along the road's left normal, between 1.5 m from the left edge and 1.5 m from the right
  edge; where a half width is below 1.5 m, that side's bound is 0.
- **Objective.** The sum over the nodes of the squared component of the line's second difference along the line's
  normal, which is its curvature times the squared node spacing, linearized on the normals of the previous solution: a
  bounded linear least-squares problem (`scipy.optimize.lsq_linear`, dense, held to one BLAS thread so that the line
  never depends on the process that computes it). It is solved three times, re-linearized each time, with each node's
  row weighted by the square of the node spacing over the line's local spacing.
- **Line.** A periodic cubic spline through the node positions, parameterized by chord length, is sampled at every
  centre-line point, its parameter interpolated linearly between the nodes, so the line has one point for every
  centre-line point, the point abeam of it. Its direction and curvature come from the spline's derivatives; the
  curvature is smoothed by a Gaussian of 10 m standard deviation along the lap. The spline through the nodes, not the
  offsets at every centre-line point, makes the line smooth: the database's centre line carries small irregularities at
  the metre scale that the line must not inherit.
- **Bounds between nodes.** The spline can leave the bounds between two nodes. Where a sampled point lies more than
  0.1 m outside the bounds of its own centre-line point, the bounds of the two nodes around it are tightened by the
  excess and the nodes are solved again, with one re-linearization, for at most six rounds. Every point of the line
  then keeps about 1.4 m or more from each edge.
- **Cost and cache.** On the development laptop a real layout's line takes 0.51 s on average and 1.35 s at most
  (Monza), with at most five solves (Shanghai reversed); the worst point of any real layout's line lies 0.097 m
  outside its bounds (Montreal reversed). The lines of the 23 real circuits, both directions of the training circuits,
  are computed once per machine into `results/racing_lines/`, like the road grids (§9.10); a file of another identity
  (of the circuit's data, the line's settings and the code) is never used and is removed once it is a day old. The
  line cache and the grid cache must be different folders (`EC_RACING_LINES` and `EC_RACING_CIRCUIT_GRIDS` name
  others, which worker processes also read): each removes the day-old `<name>-<identity>.npy` files of its own
  circuits, and in one folder each would remove the other's. A generated map's line is computed when the map is
  built: 0.06 s on Easy, 0.13 s on Medium, 0.34 s on Hard and 0.78 s on the validation setting (§5.7).
- **Check.** The database publishes a minimum-curvature racing line for every circuit, computed by its own optimizer;
  the published lines of the 23 circuits are vendored in `data/racetracks/racelines/`. They keep about 0.65 m from the
  edges: the first percentile of their distance from an edge is 0.56–0.70 m. With that clearance in place of 1.5 m, the
  quasi-steady-state lap (below) on the line of this section, at the grip a(v) (§5.1), is within 0.6% of the lap on
  the published line on each of the 23 circuits, the largest difference being Yas Marina's +0.56% (60.678 s against
  60.340 s). Both laps are computed with the curvature smoothed over 10 m. With its own 1.5 m clearance the line is
  somewhat slower than the published one.
- **Quasi-steady-state lap (`ideal.py`).** The fastest lap of the car on a fixed path with smooth control: at every
  point the highest speed with v²|κ| ≤ a(v), at most 105 m/s; then passes forward, in which the speed gained over a
  step is limited by the grip left after cornering, √(a(v)² − (v²κ)²), and backward, which limit braking the same way,
  around the loop until they join. It gives the ideal laps of Table 3 and, over the next round(200 m / spacing)
  points of a racing line at the expert's grip share, the expert's target speeds (§7.1).

*Table 3. The test circuits at the grip a(v) (§5.1). Pole lap; the car's ideal lap with all of the grip and smooth
control on the racing line of this section (1.5 m margin) and on the centre line (curvature smoothed over 10 m); the
expert's lap from a standing start, which sets the targets, and its flying lap. Seconds.*

| Circuit | Pole lap | Ideal lap, racing line | Ideal lap, centre line | Expert, standing start | Expert, flying lap |
|---|---|---|---|---|---|
| Melbourne | 80.49 | 54.31 | 58.89 | 57.80 | 56.80 |
| Sakhir | 87.26 | 56.29 | 60.28 | 59.80 | 59.00 |
| Sepang | 90.08 | 57.80 | 62.04 | 61.75 | 60.85 |
| Silverstone | 84.30 | 59.29 | 62.82 | 62.75 | 61.75 |

The line is geometry: it tells a car where the road lets it carry speed, not how fast to go. The cars receive it as
six inputs (§5.2).

## 5. Simulation Model

### 5.1 Vehicle dynamics

The vehicle is modeled as a point mass with a heading, integrated with a fixed time step of 0.05 s
(20 Hz). Its state consists of position, heading and speed.

The vehicle has the top speed of a Formula 1 car and two to three times its grip; nothing else of a Formula 1 car is
modeled: there is no engine power, drag, gearbox, tyre wear or fuel load. The model imposes no steering-angle limit
and no fixed acceleration or braking rates. How fast to drive, when to brake and how sharply to turn must all be
learned by the controllers.

- **Grip:** the grip limit rises with the square of speed, as a Formula 1 car's downforce does, up to a ceiling:
  `a(v) = min(a0 + c × v², a_max)`, with g = 9.81 m/s², a0 = 5 g = 49.05 m/s² (the tyres' grip at low speed) and
  a_max = 10 g = 98.1 m/s², reached at 250 km/h (69.4 m/s), so that c = (a_max − a0) / 69.4² = 0.01017 m⁻¹.
  Steering, throttle and brake are all shares of a(v), so the car launches at up to 5 g and brakes at up to 10 g from
  250 km/h. A Formula 1 car has about 1.5–2 g in slow corners and about 5–6 g in fast ones, so the car has two to
  three times that grip. With all of the grip the fastest speed in a corner of radius 12, 30, 100 or 300 m is 93,
  166, 357 or 378 km/h (the top speed).
- **Top speed:** 105 m/s (378 km/h), the highest speed recorded for a Formula 1 car in an official session (Bottas,
  Baku 2016 qualifying). Below it, acceleration is limited by grip alone.
- **Controls as grip fractions:** each control requests a share of the grip available at the car's current speed.
  - Steering `u_s ∈ [−1, 1]` requests a lateral acceleration `a_lat = clip(k × u_s, −1, 1) × a(v)`, where k is the
    car's steering sensitivity (§9.11; k = 1 for every car that does not evolve its setup).
  - Throttle `u_t ∈ [−1, 1]` requests a longitudinal acceleration `a_long = u_t × a(v)`. Negative
    values brake.
- **Friction circle:** if the combined request exceeds the grip limit, that is
  `sqrt(a_lat² + a_long²) > a(v)`, both components are scaled down proportionally so that the total
  equals a(v). Braking or accelerating hard therefore leaves less grip for turning, and the reverse.
- **Motion:**
  - Speed changes by `a_long × dt`, cannot fall below zero and is clipped to the top speed.
  - Heading changes at the yaw rate `a_lat / v`. Below 1 m/s, a speed of 1 m/s is used in this
    calculation to avoid division by zero.
- **Consequences:**
  - The minimum turning radius is `v² / a(v)`: about 7.5 m at 72 km/h, 49 m at 250 km/h and 112 m at the top speed,
    so a fast car cannot take a tight corner and must brake before it.
  - Safe speed on a straight is bounded by the top speed and by how early the walls ahead can be seen. Braking at full
    grip from the top speed takes about 66 m to a standstill and 60 m to 90 km/h, well within the starting sensor
    length of 150 m.
  - The time step of 0.05 s gives the controllers fine control: at the top speed a car covers 5.25 m per step, less
    than the narrowest real road (7.4 m).
- **Grip in the application:** the visualization application can display races with ten times the grip at every
  speed (§10.5); training, evaluation and the expert always use a(v).

### 5.2 Sensors

Nine rays at −90°, −67.5°, …, +90° relative to the heading measure the distance to the track boundary, up to the
car's sensor length L: 150 m for every car at the start and for every car that does not evolve its setup, and 30 to
1,000 m when evolved (§9.11). A clear day's view of a circuit reaches about a kilometre, which sets the upper limit.
Rays are cast by sphere tracing on the distance-to-boundary map, with a minimum step of 0.5 m. A ray of length L takes
at most ⌈48 × L / 50⌉ iterations: 144 for 150 m and 960 for 1,000 m. Each ray stops as soon as it reaches the boundary
or its length.

The input vector has 18 elements: the nine distances divided by 150 m, the speed divided by the top speed of
105 m/s, the position across the road, the local road width divided by 20 m, and six about the racing line (§4.8).
The ray scale is fixed, so a car of the starting sensor length sees ray inputs from 0 to 1 and a car with
L = 1,000 m up to 6.7; the speed input lies between 0 and 1. The position across the road is the car's signed
distance from its matched centre-line point along the road's normal, divided by the width on that side of the
point: −1 at the left edge, 0 on the centre line, +1 at the right edge, relative to the driving direction, and
clipped to [−1, 1].
The local road width is the sum of the two widths at the matched point. Both mean the same on a road of any width, so
a network can tell where it is on the road without relying on one width.

The six racing-line inputs are read at the car's matched centre-line point i, whose line point lies abeam of it:

- **Distance from the line:** the car's signed distance from the line point i along the line's normal, divided by the
  mean of the two half widths at i, positive to the right of the line, as the position across the road. It is not
  clipped; on the road it lies within about ±2.
- **Heading relative to the line:** the car's heading minus the line's direction at i, wrapped to (−π, π] and divided
  by π, positive when the car points to the right of the line.
- **Curvature ahead:** the line's curvature at the centre-line points 25, 50, 100 and 200 m ahead of i (round(d /
  spacing) points ahead), times 50 m and clipped to [−5, +5], positive for right turns: a 50 m radius reads 1 and the
  clip lies at a 10 m radius. 200 m covers braking from the top speed to a 12 m hairpin's speed: about 60 m at full
  grip and about 75 m at the expert's grip share of 0.80 (§7.1).

The line tells a car where to drive; when to brake and how fast to take a corner it still learns itself, from these
inputs and its rays.

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 414" width="100%" role="img" aria-label="Figure: the nine rays and twelve of the eighteen inputs of a car, the network, the forty-five output classes with the argmax choice, steering sensitivity and the grip limit" style="display:block;margin:10pt 0 6pt">
<defs><marker id="figsense-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker><marker id="figsense-ink" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#1d2330"/></marker></defs>
<text x="0" y="12" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">What the car sees</text>
<rect x="0" y="20" width="152" height="348" rx="3" fill="#f5f6f8"/>
<rect x="48" y="20" width="56" height="314" fill="#ffffff"/>
<line x1="48" y1="20" x2="48" y2="334" stroke="#5f6673" stroke-width="1.2"/>
<line x1="104" y1="20" x2="104" y2="334" stroke="#5f6673" stroke-width="1.2"/>
<line x1="76" y1="20" x2="76" y2="334" stroke="#5f6673" stroke-width="0.8" stroke-opacity="0.5" stroke-dasharray="5 4"/>
<text x="5" y="33" font-family="DejaVu Sans" font-size="9" fill="#5f6673">left</text>
<text x="147" y="33" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">right</text>
<line x1="84.4" y1="312" x2="104" y2="312" stroke="#4f6d8c" stroke-width="1.1"/>
<line x1="84.4" y1="312" x2="104" y2="303.881" stroke="#4f6d8c" stroke-width="1.1"/>
<line x1="84.4" y1="312" x2="104" y2="292.4" stroke="#4f6d8c" stroke-width="1.1"/>
<line x1="84.4" y1="312" x2="104" y2="264.681" stroke="#4f6d8c" stroke-width="1.1"/>
<line x1="84.4" y1="312" x2="84.4" y2="112" stroke="#4f6d8c" stroke-width="1.1"/>
<line x1="84.4" y1="312" x2="48" y2="224.123" stroke="#4f6d8c" stroke-width="1.1"/>
<line x1="84.4" y1="312" x2="48" y2="275.6" stroke="#4f6d8c" stroke-width="1.1"/>
<line x1="84.4" y1="312" x2="48" y2="296.923" stroke="#4f6d8c" stroke-width="1.1"/>
<line x1="84.4" y1="312" x2="48" y2="312" stroke="#4f6d8c" stroke-width="1.1"/>
<circle cx="104" cy="312" r="1.9" fill="#4f6d8c"/>
<circle cx="104" cy="303.881" r="1.9" fill="#4f6d8c"/>
<circle cx="104" cy="292.4" r="1.9" fill="#4f6d8c"/>
<circle cx="104" cy="264.681" r="1.9" fill="#4f6d8c"/>
<circle cx="48" cy="224.123" r="1.9" fill="#4f6d8c"/>
<circle cx="48" cy="275.6" r="1.9" fill="#4f6d8c"/>
<circle cx="48" cy="296.923" r="1.9" fill="#4f6d8c"/>
<circle cx="48" cy="312" r="1.9" fill="#4f6d8c"/>
<path d="M46.2,115.7 A200,200 0 0 1 122.6,115.7" fill="none" stroke="#4f6d8c" stroke-width="1" stroke-dasharray="3 3"/>
<circle cx="84.4" cy="112" r="2.2" fill="#ffffff" stroke="#4f6d8c" stroke-opacity="1" stroke-width="1.1"/>
<text x="79.4" y="127" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">0°</text>
<line x1="143" y1="312" x2="143" y2="112" stroke="#5f6673" stroke-width="0.8"/>
<line x1="140" y1="312" x2="146" y2="312" stroke="#5f6673" stroke-width="0.8"/>
<line x1="140" y1="112" x2="146" y2="112" stroke="#5f6673" stroke-width="0.8"/>
<text x="148" y="104" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">L = 150 m</text>
<text x="138" y="209" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">sensor</text>
<text x="138" y="220" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">length</text>
<rect x="79.9" y="303" width="9" height="18" rx="2.5" fill="#1d2330"/>
<rect x="81.4" y="305.5" width="6" height="4" rx="1" fill="#ffffff" fill-opacity="0.85"/>
<text x="44" y="315.5" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">+90°</text>
<text x="108" y="315.5" font-family="DejaVu Sans" font-size="9" fill="#1d2330">−90°</text>
<line x1="48" y1="343" x2="104" y2="343" stroke="#5f6673" stroke-width="0.8"/>
<line x1="48" y1="340" x2="48" y2="346" stroke="#5f6673" stroke-width="0.8"/>
<line x1="76" y1="340" x2="76" y2="346" stroke="#5f6673" stroke-width="0.8"/>
<line x1="104" y1="340" x2="104" y2="346" stroke="#5f6673" stroke-width="0.8"/>
<text x="48" y="357" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">−1</text>
<text x="76" y="357" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">0</text>
<text x="104" y="357" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">+1</text>
<circle cx="84.4" cy="343" r="2.3" fill="#1d2330"/>
<text x="0" y="382" font-family="DejaVu Sans" font-size="9" fill="#5f6673">position across the road</text>
<text x="0" y="393" font-family="DejaVu Sans" font-size="9" fill="#5f6673">road width w: local, 7.4 m and up</text>
<text x="0" y="404" font-family="DejaVu Sans" font-size="9" fill="#5f6673">L: 150 m at the start; evolved: 30–1,000 m</text>
<text x="176" y="12" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">12 of 18 inputs</text>
<rect x="176" y="26" width="56" height="14" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="204" y="36.4" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">−90°</text>
<rect x="176" y="41" width="56" height="14" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="204" y="51.4" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">−67.5°</text>
<rect x="176" y="56" width="56" height="14" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="204" y="66.4" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">−45°</text>
<rect x="176" y="71" width="56" height="14" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="204" y="81.4" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">−22.5°</text>
<rect x="176" y="86" width="56" height="14" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="204" y="96.4" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">0°</text>
<rect x="176" y="101" width="56" height="14" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="204" y="111.4" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">+22.5°</text>
<rect x="176" y="116" width="56" height="14" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="204" y="126.4" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">+45°</text>
<rect x="176" y="131" width="56" height="14" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="204" y="141.4" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">+67.5°</text>
<rect x="176" y="146" width="56" height="14" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="204" y="156.4" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">+90°</text>
<rect x="176" y="166" width="56" height="14" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="204" y="176.4" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">speed</text>
<rect x="176" y="181" width="56" height="14" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="204" y="191.4" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">position</text>
<rect x="176" y="196" width="56" height="14" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="204" y="206.4" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">width</text>
<text x="238" y="91" font-family="DejaVu Sans" font-size="9" fill="#5f6673">distance</text>
<text x="238" y="102" font-family="DejaVu Sans" font-size="9" fill="#5f6673">÷ 150 m</text>
<text x="238" y="176.4" font-family="DejaVu Sans" font-size="9" fill="#5f6673">÷ 105 m/s</text>
<text x="238" y="191.4" font-family="DejaVu Sans" font-size="9" fill="#5f6673">−1 … +1</text>
<text x="238" y="206.4" font-family="DejaVu Sans" font-size="9" fill="#5f6673">w ÷ 20 m</text>
<path d="M286,27 L290,27 L290,209 L286,209" fill="none" stroke="#5f6673" stroke-width="0.8"/>
<text x="316" y="12" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Network</text>
<path d="M290,118.0 L316,118.0" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsense-head)"/>
<rect x="316" y="83" width="66" height="70" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="349" y="121.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">network</text>
<text x="349" y="167" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">dense or NEAT,</text>
<text x="349" y="178" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">the same inputs</text>
<text x="349" y="189" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">and outputs for</text>
<text x="349" y="200" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">every method</text>
<text x="446" y="12" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">45 outputs</text>
<text x="522.5" y="36" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">steering: − right, + left (9 levels)</text>
<text x="454.5" y="56" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">−1</text>
<text x="522.5" y="56" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">0</text>
<text x="590.5" y="56" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">+1</text>
<text x="605.0" y="56" font-family="DejaVu Sans" font-size="9" fill="#5f6673">throttle</text>
<text x="605.0" y="72.4" font-family="DejaVu Sans" font-size="8" fill="#1d2330">full brake −1</text>
<rect x="446.0" y="62.0" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="454.5" y="72.2" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">0</text>
<rect x="463.0" y="62.0" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="471.5" y="72.2" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">5</text>
<rect x="480.0" y="62.0" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="488.5" y="72.2" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">10</text>
<rect x="497.0" y="62.0" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="505.5" y="72.2" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">15</text>
<rect x="514.0" y="62.0" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="522.5" y="72.2" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">20</text>
<rect x="531.0" y="62.0" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="539.5" y="72.2" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">25</text>
<rect x="548.0" y="62.0" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="556.5" y="72.2" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">30</text>
<rect x="565.0" y="62.0" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="573.5" y="72.2" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">35</text>
<rect x="582.0" y="62.0" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="590.5" y="72.2" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">40</text>
<text x="605.0" y="86.8" font-family="DejaVu Sans" font-size="8" fill="#1d2330">half brake −½</text>
<rect x="446.0" y="76.4" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="454.5" y="86.6" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">1</text>
<rect x="463.0" y="76.4" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="471.5" y="86.6" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">6</text>
<rect x="480.0" y="76.4" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="488.5" y="86.6" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">11</text>
<rect x="497.0" y="76.4" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="505.5" y="86.6" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">16</text>
<rect x="514.0" y="76.4" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="522.5" y="86.6" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">21</text>
<rect x="531.0" y="76.4" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="539.5" y="86.6" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">26</text>
<rect x="548.0" y="76.4" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="556.5" y="86.6" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">31</text>
<rect x="565.0" y="76.4" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="573.5" y="86.6" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">36</text>
<rect x="582.0" y="76.4" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="590.5" y="86.6" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">41</text>
<text x="605.0" y="101.2" font-family="DejaVu Sans" font-size="8" fill="#1d2330">coast 0</text>
<rect x="446.0" y="90.8" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="454.5" y="101.0" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">2</text>
<rect x="463.0" y="90.8" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="471.5" y="101.0" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">7</text>
<rect x="480.0" y="90.8" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="488.5" y="101.0" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">12</text>
<rect x="497.0" y="90.8" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="505.5" y="101.0" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">17</text>
<rect x="514.0" y="90.8" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="522.5" y="101.0" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">22</text>
<rect x="531.0" y="90.8" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="539.5" y="101.0" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">27</text>
<rect x="548.0" y="90.8" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="556.5" y="101.0" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">32</text>
<rect x="565.0" y="90.8" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="573.5" y="101.0" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">37</text>
<rect x="582.0" y="90.8" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="590.5" y="101.0" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">42</text>
<text x="605.0" y="115.6" font-family="DejaVu Sans" font-size="8" fill="#1d2330">half gas +½</text>
<rect x="446.0" y="105.2" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="454.5" y="115.4" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">3</text>
<rect x="463.0" y="105.2" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="471.5" y="115.4" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">8</text>
<rect x="480.0" y="105.2" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="488.5" y="115.4" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">13</text>
<rect x="497.0" y="105.2" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="505.5" y="115.4" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">18</text>
<rect x="514.0" y="105.2" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="522.5" y="115.4" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">23</text>
<rect x="531.0" y="105.2" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="539.5" y="115.4" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">28</text>
<rect x="548.0" y="105.2" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="556.5" y="115.4" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">33</text>
<rect x="565.0" y="105.2" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="573.5" y="115.4" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">38</text>
<rect x="582.0" y="105.2" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="590.5" y="115.4" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">43</text>
<text x="605.0" y="130.0" font-family="DejaVu Sans" font-size="8" fill="#1d2330">full gas +1</text>
<rect x="446.0" y="119.6" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="454.5" y="129.8" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">4</text>
<rect x="463.0" y="119.6" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="471.5" y="129.8" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">9</text>
<rect x="480.0" y="119.6" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="488.5" y="129.8" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">14</text>
<rect x="497.0" y="119.6" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="505.5" y="129.8" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">19</text>
<rect x="514.0" y="119.6" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="522.5" y="129.8" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">24</text>
<rect x="531.0" y="119.6" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="539.5" y="129.8" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">29</text>
<rect x="548.0" y="119.6" width="17.0" height="14.4" rx="0" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="1" stroke-width="1.5"/>
<text x="556.5" y="129.8" font-family="DejaVu Sans" font-size="7" fill="#1d2330" text-anchor="middle" font-weight="bold">34</text>
<rect x="565.0" y="119.6" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="573.5" y="129.8" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">39</text>
<rect x="582.0" y="119.6" width="17.0" height="14.4" rx="0" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="0.8"/>
<text x="590.5" y="129.8" font-family="DejaVu Sans" font-size="7" fill="#5f6673" text-anchor="middle">44</text>
<text x="543" y="147" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">class = 5 × steering index</text>
<text x="543" y="158" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">+ throttle index</text>
<path d="M382,98.0 L446,98.0" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsense-head)"/>
<text x="414" y="92" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">45 scores</text>
<path d="M551.0,134 L551.0,176" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsense-head)"/>
<rect x="471" y="176" width="160" height="34" rx="8" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="1" stroke-width="1"/>
<text x="551" y="190" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">largest score (argmax): 34</text>
<text x="551" y="202.5" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">steering +0.5, gas +1</text>
<path d="M551.0,210 L551.0,293.0" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsense-head)"/>
<rect x="469" y="293" width="164" height="38" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="551" y="309" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">steering sensitivity k</text>
<text x="551" y="322" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">steering × k, clipped to −1 … +1</text>
<text x="469" y="345" font-family="DejaVu Sans" font-size="9" fill="#5f6673">k = 1; evolved cars: ⅓–3</text>
<text x="469" y="356" font-family="DejaVu Sans" font-size="9" fill="#5f6673">throttle passes unchanged</text>
<text x="308" y="246" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Grip limit (friction circle)</text>
<line x1="326" y1="312" x2="418" y2="312" stroke="#5f6673" stroke-width="0.8" stroke-opacity="0.6"/>
<line x1="372" y1="266" x2="372" y2="358" stroke="#5f6673" stroke-width="0.8" stroke-opacity="0.6"/>
<circle cx="372" cy="312" r="46" fill="none" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<text x="372" y="261" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">gas</text>
<text x="372" y="370" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">brake</text>
<text x="330" y="323" font-family="DejaVu Sans" font-size="9" fill="#5f6673">left</text>
<text x="414" y="323" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">right</text>
<path d="M372.0,312.0 L349.00,266.00" fill="none" stroke="#5f6673" stroke-width="1" stroke-dasharray="3 2"/>
<circle cx="349" cy="266" r="2.4" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<path d="M372.0,312.0 L351.43,270.86" fill="none" stroke="#1d2330" stroke-width="1.4" marker-end="url(#figsense-ink)"/>
<text x="344" y="267" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">request</text>
<text x="379" y="290" font-family="DejaVu Sans" font-size="9" fill="#1d2330">applied</text>
<path d="M469.0,312.0 L432.0,312.0" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsense-head)"/>
<text x="280" y="386" font-family="DejaVu Sans" font-size="9" fill="#5f6673">radius a(v): 5 g slow, 10 g from 250 km/h;</text>
<text x="280" y="397" font-family="DejaVu Sans" font-size="9" fill="#5f6673">with k = 1, half steering and full gas ask</text>
<text x="280" y="408" font-family="DejaVu Sans" font-size="9" fill="#5f6673">√(0.5² + 1²) = 1.12 a(v): scaled ×0.89</text>
<path d="M312.0,312.0 L154,312.0" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsense-head)"/>
<text x="232" y="293" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">drive for 0.05 s,</text>
<text x="232" y="304" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">then sense again</text>
</svg>
<p><em>Figure 6. What a car senses and does: nine rays and three more inputs feed the network, the largest of its 45 scores picks one of nine steering and five throttle levels, and steering sensitivity and the grip limit shape the command. The six racing-line inputs follow the width input and are not drawn (§5.2).</em></p>
</div>

### 5.3 Collisions

A car crashes when its center leaves the road surface, and a crashed car stops in place. Cars do not
interact with each other in any mode.

### 5.4 Progress and termination

Progress is measured along the centre line. Each car tracks its nearest centre-line point within a
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
per lap (six expert lap times in total), so that slow completions are recorded as well. The final selection on the
validation maps (§9.5) and DAgger's labelled drives (§8.4) use the same limit.

Every run records the time of each lap it completes. The first lap starts from a standstill; the second and third
start at speed, like a qualifying lap, and are the flying laps compared with pole laps (§11.1).

### 5.5 Fitness

The fitness of a run on a single track is:

- `p`, the completed fraction of the three laps (0 ≤ p < 1), if the run is not completed
- `1 + (1 − t_run / t_limit)`, a value between 1 and 2, if all three laps are completed, where `t_run` is the time for
  the three laps and `t_limit` is the time limit of the run (§5.4)

Under this definition, every completion outranks every failure, so completion separates controllers early
in training. Among completions, the faster run ranks higher.

### 5.6 Implementation properties

- **Vectorized:** every per-car quantity is a numpy array of shape `(n_cars,)`, so a single call
  advances the entire population.
- **Deterministic:** all computation is in float64, and the step function involves no randomness. The one
  exception is storage: a track's distance map is stored in float32 to halve its memory, and both simulators read it
  exactly into float64 before any arithmetic.
  Identical weights on an identical track always produce identical results. This lets the training
  view re-simulate exactly the runs that training scored.
- **Compiled:** the time trials of training and the expert's validation lap run in a compiled library (§5.7) that
  reproduces this numpy implementation exactly.

### 5.7 Compiled simulator

The numpy simulator advances all cars of a map together, one step at a time, with a fixed cost of interpreter and
array-call overhead per step whatever the number of cars. More worker processes do not remove this cost, and
generations run one after another. The time trials and the expert's validation lap therefore run in a C++ library,
`racing/fastsim.cpp`, called through Python's standard `ctypes` module by `racing/fastsim.py`, with the same time
step of 0.05 s. Table 4 lists the simulator each caller uses; §9.10 describes the computational performance.

Building a map runs in numpy and SciPy with either simulator, except its validation by the expert's lap: it computes
the road grid, the racing line (§4.8) and the expert's speed profile along it (§7.1), then drives the expert's lap on
the selected simulator. Measured per generated map on the development machine, the line's solve takes 0.06 s on Easy,
0.13 s on Medium, 0.34 s on Hard and 0.78 s on the validation setting, against 0.21, 0.28, 0.6 and 1.1 s for the road
grid. A real layout's line takes 0.51 s on average and 1.35 s at most and is computed once per machine into
`results/racing_lines/` (§4.8, §9.10).

- **Build.** The library is compiled with g++ (C++17; flags `-O3 -march=native -ffp-contract=off -fno-builtin-sin
  -fno-builtin-cos -std=c++17 -shared -fPIC -pthread`) the first time it is needed on a machine. The compiler is
  `EC_RACING_CXX` when it is set, otherwise the `g++` beside the run's Python interpreter (a conda environment's own
  compiler), otherwise `g++` on the search path. Its file name carries a build identity,
  `racing/_fastsim-<id>.so`, where `<id>` is the first 16 hexadecimal digits of the SHA-256 of the bytes of
  `fastsim.cpp`, the compiler flags and the processor's flags line in `/proc/cpuinfo` (empty where it cannot be read);
  the file is not under version control. A library of the current identity is used as it is; otherwise the library is
  built under that name, so a change of source, flags or processor always leads to a new build, and a file under that
  name that fails to load is built again. Libraries of other identities are removed once they are a day old, because
  runs that started earlier may still use them. A build writes a temporary file unique to the call and renames it
  into place, and the first use in a process builds, loads and checks the library under a lock, so runs and threads
  that start at once never load a partly written library. The library records the version of the compiler that built
  it, which the run records report with the simulator (§3.2), without running the compiler again.
- **One call per generation.** The maps of a generation, the cars' network programs and setups and each map's target
  time are passed in one call. Every pair of map and car is an independent job; the threads (one per available
  logical processor unless `EC_RACING_THREADS` sets a number, §3.2) take the next unclaimed job from a shared
  counter, so a map whose cars finish early leaves no thread idle. Each job drives one car from the start line until
  it is done, in the order of `sim.sense`, the network's action, `sim.step` and the progress update; the car's own
  clock advances by DT per step. The returned values are those of `sim.RunResult`, and the fitness is computed from
  them by `sim.fitness`. Results do not depend on the number of threads.
- **Exact equivalence.** Every field of every `RunResult` equals the numpy simulator's (`np.array_equal`, with NaN
  finish times in the same places). Every expression of `sim.py`, `track.wall_distance`, `expert.py` and
  `actions.py` is transcribed with the same operand order in double precision; the build turns off fused
  multiply-add and does not use `-ffast-math`; `cos`, `sin`, `hypot`, `atan2` and `sqrt` come from the C library numpy
  calls on the same machine, in the versions numpy binds: glibc 2.35 added a rewritten `hypot` under a new symbol
  version, so the library binds `hypot@GLIBC_2.2.5`, as numpy does; the sine and cosine of one angle are two calls,
  as numpy makes them, because the build forbids the compiler to fuse them into the C library's `sincos`
  (`-fno-builtin-sin -fno-builtin-cos`): glibc 2.35's `sincos` differs from its own `sin` and `cos` in about one
  angle in 1,500, while glibc 2.39's agrees, and the two calls cost about 5% of the compiled time (1,000 trained cars
  on one thread: 15.0 s fused, 15.9 s as two calls); rounding is half to even, as in `np.round`; ties go to the first
  largest output (`np.argmax`) and the first nearest centre-line point (`np.argmin`); indices wrap to non-negative
  values (`np.mod`).
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
  inputs (0–17) and then one slot per node; a node has a slot, a bias, an activation (ReLU, tanh or identity) and its
  incoming connections (source slot, weight); 45 output slots give the action order. A node's value is the sum of
  source value × weight over its connections, from zero and in order, then plus the bias, then the activation. Dense
  networks (`nets.DensePopulation`, evaluated with `np.einsum`) list each unit's inputs in index order; NEAT genomes
  (`neat.NeatPopulation`, evaluated with `np.bincount`) list nodes by depth and each node's enabled connections in
  genome order, so both sum exactly as numpy does. One format covers every architecture and activation, NEAT, and
  networks that read only the first inputs of the vector. Networks whose architecture differs from car to car
  (`nets.ShapedPopulation`, the GA) receive each car's own dense program; a layer of a single unit, which `np.einsum`
  sums in another order, is summed by the numpy simulator in input order, as the library does, and the check at load
  also compares a population of mixed architectures.
- **Formula 1 car and real circuits.** The speed-dependent grip, the top speed, the left and right widths of every
  centre-line point, the scaled inputs and the lap times are transcribed like every other expression; the tests
  compare both simulators on real circuits (§12.1).
- **Racing line.** The racing line's points, directions (as unit vectors and as angles) and curvatures, the expert's
  target speed at every point and the four look-ahead offsets in points are computed once per track in numpy (§4.8,
  §7.1) and passed to the library with the track's other arrays, so the library computes no racing line and no speed
  profile. The six line inputs and the expert's steering along the line (its previewed curvature, its feedback on the
  distance and heading from the line and its rule of no gas at quantized full lock) are transcribed like every other
  expression; the wrapped heading uses atan2 of the sine and cosine, as the expert's steering does and as numpy's angle
  of a unit complex number gives it.
- **Expert.** The expert (§7.1) and its quantization to the 45 actions are ported as a second kind of driver, used by
  `expert.expert_lap_time` to validate every new map.
- **Scope.** Single trained controllers (`nets.DenseNet`, `neat.NeatNet`) are evaluated with numpy's matrix product,
  whose rounding depends on the linear-algebra library's internal order and cannot be reproduced. DAgger's own driving
  (§8.4), the final races (§11), dataset collection (§7) and the application's playback (§10) therefore remain in
  numpy; they still benefit from the faster map building.

*Table 4. Simulator used by each caller.*

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

- **Input:** the 18-element vector defined in §5.2.
- **Output:** a 45-way softmax. Each class is one combination of a steering level and a throttle level:
  - steering fraction `u_s` ∈ {−1, −0.75, −0.5, −0.25, 0, +0.25, +0.5, +0.75, +1} (nine levels)
  - throttle fraction `u_t` ∈ {−1 (full brake), −0.5 (half brake), 0 (coast), +0.5 (half gas), +1 (full gas)}
  - `class = 5 × steer_index + throttle_index`

  The car executes the class with the highest probability. A controller or results folder with another number of
  inputs or outputs is not raced or replayed; the application names the count in a note (§10.3).

Framing control as classification makes test accuracy directly measurable.

The module `nets.py` provides:

- a batched numpy forward pass for whole populations, using one `einsum` per layer (an explicit sum for a layer of
  one unit, §5.7)
- `to_keras(flat, hidden, activation)`, which builds the equivalent Keras model and copies the weights into it

Every final controller is stored as a `.keras` file, so network size and inference time are measured
the same way for all approaches. Keras stores weights in float32. Evolved dense controllers (the GA, the ES, the
seeded ES and the speed pushes of the imitation methods) are selected with float64 weights, so they are also stored
as float64 `.npy` files; a NEAT controller is stored as its genome (`seed<k>.json`, §9.9). All driving evaluations of
evolved controllers use these files. An evolved controller also stores its setup (sensor length and steering
sensitivity) in `seed<k>.setup.json`; a controller without this file drives with L = 150 m and k = 1.

The baseline architecture, which the ES also uses, is 18 → 16 (ReLU) → 45 (softmax), for a total
of 1,069 trainable parameters.

The GA's networks have no fixed architecture: each car carries its own list of hidden layer sizes, from none to any
number of layers of any width, with ReLU hidden units (§9.6). A final GA controller is stored like every evolved dense
controller, as a `.keras` file whose layers give its architecture and as float64 weights (`.npy`).

## 7. Expert Driver and Dataset

### 7.1 Expert driver

- **Steering:** the expert steers along the racing line (§4.8). At the car's matched centre-line point
  i, e is the car's distance left of the line point i, along the line's normal, ψ its heading relative to the line's
  direction at i, wrapped to (−π, π] and positive to the left, and L = `6 m + 0.2 s × v`; curvatures are positive for
  left turns.
  1. The feed-forward curvature κ_ff is the line's smoothed curvature at the centre-line point round(v × 0.05 s /
     spacing) points ahead of i, so that the car turns in as its one-step delay requires.
  2. The path curvature asked for is `κ = κ_ff − 8 × e / L² − 8 × ψ / L`: a correction toward the line in proportion
     to the distance from it and to the heading error.
  3. The required lateral acceleration `v² × κ` is divided by the grip a(v) and clipped to [−1, 1], giving a
     steering fraction.
  4. The fraction is quantized to the nearest of the nine steering levels.

  The preview time of 0.05 s (`EXPERT_PREVIEW_TIME`), the cross-track gain of 8 (`EXPERT_CROSS_TRACK_GAIN`), the
  heading gain of 8 (`EXPERT_HEADING_GAIN`) and the lookahead time of 0.2 s (`EXPERT_LOOKAHEAD_TIME`) were chosen,
  with the 6 m base fixed, from 144 combinations (preview times of 0.05, 0.1 and 0.2 s, cross-track gains of 3, 5, 8
  and 12, heading gains of 3, 5 and 8, lookahead times of 0.1, 0.2, 0.3 and 0.4 s) on the 42 real layouts, for the
  highest grip share at which all of them finish one lap: a broad plateau reaches 0.82–0.83 there (0.80 over three
  laps, below), while a longer preview and lookahead with weaker gains (preview 0.1 s, gains 3 and 5, lookahead 0.4 s)
  reach only 0.58. The difficulty lies in fast corners at 60–105 m/s, where the steering the line asks for, about 0.79
  of a(v), falls between the quantized levels 0.75 and 1.0: quantized to 0.75 the car drifts outward, and with a long
  lookahead the feedback is too weak at those speeds to return it before it reaches the edge, 1.5 m beyond the line.
  A short preview and lookahead with large gains return it in time.
- **Speed:** the expert drives the racing line at its grip share s and brakes at the last moment for
  that share, using only the line within its sight of 200 m, the reach of the cars' curvature inputs (§5.2). Its target
  speed at a centre-line point is the start of a quasi-steady-state speed profile (§4.8) of the next round(200 m /
  spacing) points of the line. The window is counted in centre-line points and each of its steps is the line's true
  distance between two of its points, so on a line inside the centre line the window covers somewhat less than 200 m
  of line:
  - at every point of the window, the highest speed v with `v² × |κ| ≤ s × a(v)`, κ the line's smoothed curvature
    and s the grip share; at the window's end the profile starts from that speed
  - backwards from the window's end, the speed from which the car can slow to the next point's speed braking with
    the grip left after cornering, √((s × a(v))² − (v²κ)²), over the line's step between the two points

  The target is the lowest of the profile's first speed, the speed from which the car stops within 200 m braking at
  s × a(v), and the top speed.

  The grip share s is 0.80 (`EXPERT_GRIP_MARGIN`), measured on 2026-10-25 by `scripts/expert_grip.py` as the highest
  share, from 1.00 down in steps of 0.01, at which the expert finishes three laps (`config.LAPS`) of every real layout
  (the training circuits in both directions and the four test circuits) and each of the 20 validation maps and the
  dataset's generated maps builds (§4.5). The generated maps are built only at a share at which every real layout
  passes, since a map the expert cannot drive uses all 200 candidates of its seed. Monza reversed sets the value: it
  fails three laps at every share from 0.81 to 0.85; Shanghai reversed fails at 0.86–0.92, and 33 of the 42 real
  layouts fail at 1.00. The value is recorded in the settings (§3.2) and is measured again after any change to the
  grip, the racing line, the road grids, the expert or the levels. The target speeds of a track's points are computed
  once, when the track is built (§5.7), and are not stored with a real circuit's cached line, so they always follow
  the recorded share.

  At 0.80 the expert's lap times on the test circuits, one lap from a standing start, are Melbourne 57.80 s, Sakhir
  59.80 s, Sepang 61.75 s and Silverstone 62.75 s; its flying laps are 56.80, 59.00, 60.85 and 61.75 s. On every
  track the starting target is 1.3 × three of the expert's laps, 225.4, 233.2, 240.8 and 244.7 s on the test
  circuits (§9.2, §11.1).

  The expert's throttle is its speed error (target minus speed) divided by its band, clipped to [−1, 1] and
  quantized to the nearest of the five throttle levels: full gas or full brake beyond three quarters of the band, half
  gas or half brake from a quarter to three quarters of it, coast within a quarter of the band of the target. The
  band is max(0.5 m/s, a(v) × dt), the speed one step can change at full grip: with a narrower band at this grip, the
  throttle would switch between gas and brake every step and leave too little grip for turning. While its quantized
  steering is at full lock (±1), the expert does not accelerate: it coasts or brakes. These targets govern the expert
  only. The networks are free to drive faster or slower.

The expert chooses from the same 45 actions as the networks, so every label is an action a network can
reproduce. The expert's lap time on each track sets that track's time limit and target time (§5.4, §9.2). The expert
completes three laps of every real layout, and a generated map is accepted only when the expert completes a lap of it
(§4.5).

### 7.2 Recovery data

Pure imitation provides no examples of recovering from off-line states. To address this, at random
intervals of 2–4 s the car executes a random steering level of the nine, with the expert's throttle, for 0.3–0.6 s.
Throughout, the recorded label remains the expert's chosen action for the observed state.

A disturbance at the expert's grip share can make the car leave the road. In that case the car is placed back on
the racing line at the line point of its last matched centre-line point, heading along the line at the expert's target
speed for that point, and collection continues.
Collection on a track ends when one full lap of progress has been covered, so every track contributes rows from its
whole length rather than only from the section before the first crash; as in any run, it also ends after 5 s without
progress, and at the time limit of 600 s (`EXPERT_VALIDATION_TIME`).

### 7.3 Dataset composition

Each row consists of the 18 inputs and one class label.

- **Training:** 41 generated maps, drawn once (one at Easy, 20 at Medium and 20 at Hard) and the 19 training circuits
  in both directions, 38 layouts (§4.6).
- **Validation:** the 20 validation maps.
- **Test:** the four test circuits in their racing direction.

The number of rows of each split is reported in the dataset table (§11.3).

Collection is distributed across worker processes, one track per task. Each track uses its own random
number generator, so the dataset is identical to a sequential collection. The collection time is
recorded and reported as a cost shared by the four imitation methods, their DAgger refinement and the seeded ES.

## 8. Imitation-Learning Controllers

### 8.1 Baseline

The baseline is the 1,069-parameter network from §6, trained with:

- Adam at a learning rate of 1e-3
- batch size 256
- sparse categorical cross-entropy
- at most 150 epochs, with early stopping on validation accuracy (patience 5)

The limit of 150 epochs leaves room for early stopping to act: with a limit of 50, the baseline trained for all
50 epochs, and 13 of 30 random-search trials reached their best validation accuracy in the final epoch.

The baseline is trained for run 0 with seed 0 and for runs 1–3 without a seed (§3.2). Two Keras settings that affect
training speed but not the computed updates are fixed explicitly in `config.py` and recorded with the results:
`steps_per_execution` (the number of batches executed per call into the compiled training function)
and `jit_compile` (XLA compilation). `steps_per_execution` is chosen with the Keras training-time benchmark in
`bench.py`. XLA compilation is switched off: it trains about 11% faster per epoch, but on the desktop used for the
reported run its compiler failed after about 130 Keras Tuner trials in one process ("Failed to materialize symbols")
and kept failing, which stopped the search.

### 8.2 Hyperparameter search

Random search, Hyperband and Bayesian optimization all search the same space.

*Table 5. Keras Tuner search space.*

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

The selected architecture, with its searched hyperparameters, is then trained from scratch for run 0 (seed 0) and
runs 1–3 (no seed). These four networks are the method's controllers after initial training. The search therefore
runs once per method, and the variation across runs measures training variability for a fixed architecture,
not the variability of the search itself. DAgger (§8.4) and the speed push with its hardening (§8.5, §9.4) then
produce its final controllers.

**Driving-based selection (secondary analysis):** imitation methods are selected on validation
accuracy, whereas evolutionary methods are selected on validation driving. To show how much this
asymmetry matters, the most-trained model of every distinct configuration in each search drives three
laps on each of the 20 validation tracks. The model with the highest mean validation fitness is
reported for each method as a driving-selected controller in a secondary driving table.

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 304" width="100%" role="img" aria-label="Figure: the four dense network architectures with their layer sizes, activations and parameter counts; the decision layer into the 45 outputs is highlighted" style="display:block;margin:10pt 0 6pt">
<defs><marker id="figdense-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker></defs>
<rect x="0" y="0" width="157" height="252" rx="3" fill="#f5f6f8"/>
<rect x="9" y="9" width="10" height="10" rx="1.5" fill="#7F7F7F"/>
<text x="25" y="18" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Baseline</text>
<polygon points="35,102 71.5,98 71.5,130 35,126" fill="#5f6673" fill-opacity="0.13" stroke="#5f6673" stroke-opacity="0.35" stroke-width="0.8" stroke-linejoin="round"/>
<polygon points="85.5,98 122,69 122,159 85.5,130" fill="#fdecea" stroke="#c0392b" stroke-width="1" stroke-linejoin="round"/>
<rect x="21" y="102" width="14" height="24" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="28" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">18</text>
<text x="28" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">inputs</text>
<rect x="71.5" y="98" width="14" height="32" rx="2" fill="#d2d2d2" stroke="#7F7F7F" stroke-opacity="1" stroke-width="1.2"/>
<text x="78.5" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">16</text>
<text x="78.5" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">ReLU</text>
<rect x="122" y="69" width="14" height="90" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="129" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">45</text>
<text x="129" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">softmax</text>
<text x="78.5" y="230" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">1,069 parameters</text>
<text x="54.6891" y="243" font-family="DejaVu Sans" font-size="9" fill="#5f6673">304 + <tspan fill="#c0392b">765</tspan></text>
<rect x="169" y="0" width="157" height="252" rx="3" fill="#f5f6f8"/>
<rect x="178" y="9" width="10" height="10" rx="1.5" fill="#E69F00"/>
<text x="194" y="18" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">KT Random</text>
<polygon points="204,102 240.5,98 240.5,130 204,126" fill="#5f6673" fill-opacity="0.13" stroke="#5f6673" stroke-opacity="0.35" stroke-width="0.8" stroke-linejoin="round"/>
<polygon points="254.5,98 291,69 291,159 254.5,130" fill="#fdecea" stroke="#c0392b" stroke-width="1" stroke-linejoin="round"/>
<rect x="190" y="102" width="14" height="24" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="197" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">18</text>
<text x="197" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">inputs</text>
<rect x="240.5" y="98" width="14" height="32" rx="2" fill="#f6dda6" stroke="#E69F00" stroke-opacity="1" stroke-width="1.2"/>
<text x="247.5" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">16</text>
<text x="247.5" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">tanh</text>
<rect x="291" y="69" width="14" height="90" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="298" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">45</text>
<text x="298" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">softmax</text>
<text x="247.5" y="230" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">1,069 parameters</text>
<text x="223.689" y="243" font-family="DejaVu Sans" font-size="9" fill="#5f6673">304 + <tspan fill="#c0392b">765</tspan></text>
<rect x="338" y="0" width="157" height="252" rx="3" fill="#f5f6f8"/>
<rect x="347" y="9" width="10" height="10" rx="1.5" fill="#56B4E9"/>
<text x="363" y="18" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">KT Hyperband</text>
<polygon points="367,102 390,50 390,178 367,126" fill="#5f6673" fill-opacity="0.13" stroke="#5f6673" stroke-opacity="0.35" stroke-width="0.8" stroke-linejoin="round"/>
<polygon points="404,50 427,82 427,146 404,178" fill="#5f6673" fill-opacity="0.13" stroke="#5f6673" stroke-opacity="0.35" stroke-width="0.8" stroke-linejoin="round"/>
<polygon points="441,82 464,69 464,159 441,146" fill="#fdecea" stroke="#c0392b" stroke-width="1" stroke-linejoin="round"/>
<rect x="353" y="102" width="14" height="24" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="360" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">18</text>
<text x="360" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">inputs</text>
<rect x="390" y="50" width="14" height="128" rx="2" fill="#c4e5f7" stroke="#56B4E9" stroke-opacity="1" stroke-width="1.2"/>
<text x="397" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">64</text>
<text x="397" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">ReLU</text>
<rect x="427" y="82" width="14" height="64" rx="2" fill="#c4e5f7" stroke="#56B4E9" stroke-opacity="1" stroke-width="1.2"/>
<text x="434" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">32</text>
<text x="434" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">ReLU</text>
<rect x="464" y="69" width="14" height="90" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="471" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">45</text>
<text x="471" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">softmax</text>
<text x="416.5" y="230" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">4,781 parameters</text>
<text x="373.174" y="243" font-family="DejaVu Sans" font-size="9" fill="#5f6673">1,216 + 2,080 + <tspan fill="#c0392b">1,485</tspan></text>
<rect x="507" y="0" width="157" height="252" rx="3" fill="#f5f6f8"/>
<rect x="516" y="9" width="10" height="10" rx="1.5" fill="#CC79A7"/>
<text x="532" y="18" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">KT Bayesian</text>
<polygon points="542,102 578.5,50 578.5,178 542,126" fill="#5f6673" fill-opacity="0.13" stroke="#5f6673" stroke-opacity="0.35" stroke-width="0.8" stroke-linejoin="round"/>
<polygon points="592.5,50 629,69 629,159 592.5,178" fill="#fdecea" stroke="#c0392b" stroke-width="1" stroke-linejoin="round"/>
<rect x="528" y="102" width="14" height="24" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="535" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">18</text>
<text x="535" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">inputs</text>
<rect x="578.5" y="50" width="14" height="128" rx="2" fill="#edd0e0" stroke="#CC79A7" stroke-opacity="1" stroke-width="1.2"/>
<text x="585.5" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">64</text>
<text x="585.5" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">tanh</text>
<rect x="629" y="69" width="14" height="90" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="636" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">45</text>
<text x="636" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">softmax</text>
<text x="585.5" y="230" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">4,141 parameters</text>
<text x="561.689" y="243" font-family="DejaVu Sans" font-size="9" fill="#5f6673">1,216 + <tspan fill="#c0392b">2,925</tspan></text>
<rect x="9" y="36.5" width="9" height="9" rx="1.5" fill="#009E73"/>
<text x="22" y="45" font-family="DejaVu Sans" font-size="9" fill="#1d2330">ES</text>
<rect x="42.3984" y="36.5" width="9" height="9" rx="1.5" fill="#0072B2"/>
<text x="55.3984" y="45" font-family="DejaVu Sans" font-size="9" fill="#1d2330">Seeded ES</text>
<text x="9" y="58" font-family="DejaVu Sans" font-size="9" fill="#5f6673">evolve this architecture</text>
<text x="9" y="69" font-family="DejaVu Sans" font-size="9" fill="#5f6673">(weights only)</text>
<polygon points="1,263 23,265 23,275 1,273" fill="#fdecea" stroke="#c0392b" stroke-width="1" stroke-linejoin="round"/>
<text x="30" y="272" font-family="DejaVu Sans" font-size="9" fill="#1d2330">decision layer: last hidden layer → 45 outputs</text>
<text x="30" y="284" font-family="DejaVu Sans" font-size="9" fill="#c0392b">stall boost: mutates ×3 (§9.4)</text>
<text x="30" y="296" font-family="DejaVu Sans" font-size="9" fill="#5f6673">imitation networks evolve only in their speed push (§8.5)</text>
<text x="338" y="272" font-family="DejaVu Sans" font-size="9" fill="#5f6673">Bar height is proportional to the number of units.</text>
<text x="338" y="284" font-family="DejaVu Sans" font-size="9" fill="#5f6673">Keras Tuner keeps the smallest network within one</text>
<text x="338" y="296" font-family="DejaVu Sans" font-size="9" fill="#5f6673">percentage point of its best validation accuracy (§8.3).</text>
</svg>
<p><em>Figure 7. The dense networks of fixed architecture with the 18 inputs and 45 outputs (the Keras Tuner architectures are those that the searches of the reported results selected): layer sizes, activations and parameter counts, with the decision layer that the stall boost mutates three times as strongly. The GA grows networks of its own, starting from 18 → 45 (§9.6).</em></p>
</div>

### 8.4 DAgger refinement

Dataset aggregation (DAgger) corrects the main weakness of imitation: the trained car reaches
situations the expert never showed it and does not know how to recover. Each trained network (four
methods, four runs) is refined in rounds:

1. **Drive.** The car drives 20 Extra hard layouts, drawn from the 38 layouts of the training circuits by the
   round's generator, for three laps each. After a crash it is placed
   back on the racing line, as during dataset collection, and continues, so that the whole lap is
   covered.
2. **Label.** At every step, the car's sensor readings are recorded together with the action the expert
   would take in the car's exact state.
3. **Train.** The new rows are added to the network's dataset, which keeps the rows of all previous
   rounds. Training continues from the current weights with the method's settings, for at most 30
   epochs with early stopping (patience 5) on a 10% hold-out of the aggregated rows.
4. **Check.** The car drives 20 further layouts, drawn the same way and independently of the 20 driven, as a time
   trial at f = 1.3. With 38 layouts the two draws overlap.

Refinement reaches its goal when at least 18 of 20 checks succeed in three consecutive rounds. If 10
rounds pass without a new best check result, refinement stops and the best network so far is passed
on.

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 340" width="100%" role="img" aria-label="DAgger refinement in rounds: drive, expert labels, aggregate, retrain, check; goal or stop, then the speed push" style="display:block;margin:10pt 0 6pt">
<defs><marker id="figdag-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker></defs>
<rect x="8" y="44" width="649" height="166" rx="3" fill="#f5f6f8"/>
<rect x="18" y="0" width="138" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="87" y="12.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Trained network</text>
<text x="87" y="24.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">each method and run</text>
<path d="M87,32 L87,70" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<text x="181.67" y="60" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">DAgger round</tspan><tspan font-size="9" fill="#5f6673"> · a checkpoint after every round</tspan></text>
<rect x="18" y="70" width="138" height="44" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="87" y="83" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Drive</text>
<text x="87" y="95" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">20 Extra hard layouts</text>
<text x="87" y="106" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">respawn after a crash</text>
<rect x="181.67" y="70" width="138" height="44" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="250.67" y="83" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Expert labels</text>
<text x="250.67" y="95" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">every state of the car</text>
<text x="250.67" y="106" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">the expert's action there</text>
<rect x="345.33" y="70" width="138" height="44" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="414.33" y="83" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Aggregate</text>
<text x="414.33" y="95" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">new rows join the dataset</text>
<text x="414.33" y="106" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">rows of all rounds kept</text>
<rect x="509" y="70" width="138" height="44" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="578" y="83" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Retrain</text>
<text x="578" y="95" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">from the current weights</text>
<text x="578" y="106" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">≤ 30 epochs, 10% held out</text>
<path d="M156,92 L181.67,92" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<path d="M319.67,92 L345.33,92" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<path d="M483.33,92 L509,92" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<rect x="509" y="150" width="138" height="44" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="578" y="163" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Check</text>
<text x="578" y="175" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">20 more layouts</text>
<text x="578" y="186" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">time trial at f = 1.3</text>
<path d="M578,114 L578,150" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<path d="M285,172 L297.4,152 L422.6,152 L435,172 L422.6,192 L297.4,192 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="360" y="169.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">≥ 18 of 20 checks</text>
<text x="360" y="181.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">3 rounds in a row?</text>
<path d="M509,172 L435,172" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<path d="M31,172 L43.4,152 L168.6,152 L181,172 L168.6,192 L43.4,192 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="106" y="169.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">10 rounds without</text>
<text x="106" y="181.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">a new best check?</text>
<path d="M285,172 L181,172" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<text x="233" y="167" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">no</text>
<path d="M106,152 L106,114" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<text x="112" y="136" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no: next round</text>
<rect x="26" y="232" width="160" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="106" y="244.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Stop</text>
<text x="106" y="256.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">pass on the best network</text>
<path d="M106,192 L106,232" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<text x="112" y="224" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes</text>
<rect x="280" y="232" width="160" height="32" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="360" y="244.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Goal reached</text>
<text x="360" y="256.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">pass on the final network</text>
<path d="M360,192 L360,232" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<text x="366" y="224" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes</text>
<rect x="280" y="290" width="160" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="360" y="302.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Speed push (§8.5)</text>
<text x="360" y="314.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">ES of 1,100 grown from it</text>
<path d="M106,264 L106,306 L280,306" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<path d="M360,264 L360,290" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<text x="454" y="297" font-family="DejaVu Sans" font-size="9" fill="#5f6673">starts at Extra hard, f = 1.3;</text>
<text x="454" y="308" font-family="DejaVu Sans" font-size="9" fill="#5f6673">mastery goal first, then shorter targets,</text>
<text x="454" y="319" font-family="DejaVu Sans" font-size="9" fill="#5f6673">then hardening (§9.4)</text>
</svg>
<p><em>Figure 8. DAgger refinement: rounds of driving, expert labelling, aggregation, retraining and checking, until the goal or the patience stop passes the network to the speed push.</em></p>
</div>

### 8.5 Speed push

DAgger can make a car no better than its teacher. After refinement, an evolution strategy population of
1,100 is grown from the refined network, as the network itself and 1,099 copies perturbed with noise of standard
deviation 0.1. It starts at Extra hard at f = 1.3 and follows the ES of §9.7 and the rules of §9.2–9.5: whether or
not DAgger reached its goal, the population must first reach the mastery goal (§9.4), then the target time is
shortened step by step until the population fails to meet it; the run then hardens at its tightest mastered target
(§9.4), and its final controller is selected from the kept cars as in §9.5.

### 8.6 Stages

Each imitation method is evaluated at three stages: after initial training, after DAgger and after the
speed push. The final controller of the speed-push run, selected after hardening (§9.4, §9.5), is the method's final
controller.

## 9. Evolutionary Controllers

### 9.1 Common procedure

All evolutionary approaches share one run loop: the GA, the ES, the seeded ES, NEAT and the speed push
of the imitation methods (§8.5). The ES uses the baseline architecture (1,069 weights and biases) with
Glorot-uniform initial weights. The GA starts from networks without hidden layers, the 18 inputs connected directly
to the 45 outputs (855 weights and biases), with Glorot-uniform initial weights, and grows them (§9.6). The seeded ES
and the speed pushes start from trained networks. NEAT starts from minimal networks and grows them. Every car of these
populations also carries an evolved setup (§9.11). Each approach is run four times, run 0 with seed 0 and runs 1–3
without a seed (§3.2). Runs have no evaluation budget; they end through the speed push and hardening, or by an end
request (§9.4); the safety stop is off.

**Each generation proceeds as follows:**

1. Three maps are drawn at the current level by the run's random number generator: freshly generated maps from
   new seeds on levels 1–3, three different layouts of the training circuits on Extra hard (§4.6).
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
(GA mutation strength or σ), whether the stall boost was on, the streak, the event of the generation (level
advancement, mastery goal, target reduction, end of the speed push and start of hardening, hardened, or safety
stop), the map seeds and the elapsed time.  Every run also logs the mean and standard deviation of the population's
sensor length and steering sensitivity, NEAT logs its number of species and the
population's mean numbers of hidden nodes and enabled connections, and the GA logs the mean and largest number of
hidden layers, hidden units and parameters of its population. A
checkpoint is written after every generation (§3.2).

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
- **Simulation end.** A training drive ends at the target time, or earlier when the car completes its three laps,
  crashes or makes no progress for five seconds (§5.4).
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
(§9.6–9.9). Because every generation drives newly drawn maps (maps never driven before on levels 1–3, a new draw of
three of the 38 layouts on Extra hard), a car is kept for long only if it keeps succeeding on the maps of each new
generation.

### 9.4 Level advancement, speed push, hardening and safety stop

**Level advancement.** For each map of a generation, the success share is the fraction of the whole population that
succeeded on that map. A generation passes when the lowest success share among its maps is at least
90%. Three consecutive passing generations advance the population one level. A population on Extra
hard that passes three consecutive generations has reached the mastery goal. Each generation drives
three maps, so on levels 1–3 the three passing generations cover nine different maps; on Extra hard the layouts of one
generation differ, but layouts can repeat between generations (§4.6).

**Speed push.** After the mastery goal, the target factor is multiplied by 0.98, which shortens the target time by 2%.
In the speed push a generation passes when the lowest map success share is at least 50%; each further run of three
consecutive passing generations shortens the target by another 2%. The speed push ends when 1,000 generations pass
without mastering the current target, and the run hardens. The pass share of one half lets the target keep tightening
while the fastest half of the population follows it: in runs whose speed push required 75%, the controllers whose
push went furthest were the fastest. The tightest target mastered is recorded as the run's speed result, expressed as
the reduction from the starting target. For example, f = 1.3 × 0.98⁴ is reported as a target 7.8% shorter than the
starting target.

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 466" width="100%" role="img" aria-label="One generation: fresh maps, time trials, fitness, survival, children by method, checkpoint and the level, push or hardening decision" style="display:block;margin:10pt 0 6pt">
<defs><marker id="figgen-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker><marker id="figgen-head-boost" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#c0392b"/></marker></defs>
<rect x="24" y="120" width="641" height="102" rx="3" fill="#f5f6f8"/>
<rect x="24" y="234" width="641" height="166" rx="3" fill="#f5f6f8"/>
<rect x="34" y="0" width="180" height="24" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="124" y="15.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Draw 3 maps</text>
<text x="230" y="15.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">levels 1–3: fresh seeds from 2,000,000; Extra hard: 3 of the 38 real layouts</text>
<path d="M124,24 L124,38" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<rect x="34" y="38" width="180" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="124" y="50.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Time trials</text>
<text x="124" y="62.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">every car · 3 maps · 3 laps</text>
<text x="230" y="57.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">compiled simulator on all threads; target time = f × the expert's 3 laps; kept cars drive too</text>
<path d="M124,70 L124,84" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<rect x="34" y="84" width="180" height="24" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="124" y="99.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Fitness and successes</text>
<text x="230" y="99.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">fitness: mean over the 3 maps; success: 3 laps within the target time</text>
<text x="450" y="136" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Survival</tspan><tspan font-size="9" fill="#5f6673"> · every method</tspan></text>
<path d="M44,152 L53.92,136 L194.08,136 L204,152 L194.08,168 L53.92,168 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="124" y="155.6" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">Success on every map?</text>
<path d="M124,108 L124,136" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<rect x="252" y="140" width="70" height="24" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="287" y="155.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Kept</text>
<path d="M204,152 L252,152" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<text x="228" y="147" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">yes</text>
<rect x="34" y="182" width="180" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="124" y="194.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Kept with a chance</text>
<text x="124" y="206.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">share of laps in time</text>
<path d="M124,168 L124,182" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<text x="130" y="178.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no</text>
<rect x="469" y="159" width="170" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="554" y="171.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">At most 90% kept,</text>
<text x="554" y="183.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">at least the 50 fittest</text>
<path d="M322,152 L446,152 L446,175 L469,175" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<path d="M214,198 L446,198 L446,175" fill="none" stroke="#5f6673" stroke-width="1"/>
<text x="224" y="193" font-family="DejaVu Sans" font-size="9" fill="#5f6673">e.g. 7 of 9 laps: 78%</text>
<text x="561" y="204" font-family="DejaVu Sans" font-size="9" fill="#5f6673">successes first,</text>
<text x="561" y="215" font-family="DejaVu Sans" font-size="9" fill="#5f6673">fittest first</text>
<text x="34" y="250" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Children fill the rest</tspan><tspan font-size="9" fill="#5f6673"> · by method; the seeded ES and the speed pushes use the ES</tspan></text>
<rect x="34" y="274" width="200" height="78" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<rect x="44" y="283" width="9" height="9" rx="1.5" fill="#D55E00"/>
<text x="59" y="291.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330">GA<tspan font-size="9"> · 2,000 cars</tspan></text>
<text x="44" y="307" font-family="DejaVu Sans" font-size="9" fill="#5f6673">parents: two tournaments of 5</text>
<text x="44" y="318" font-family="DejaVu Sans" font-size="9" fill="#5f6673">crossover p 0.5 on shared weights</text>
<text x="44" y="329" font-family="DejaVu Sans" font-size="9" fill="#5f6673">mutation: each weight p 0.1, N(0, s)</text>
<text x="44" y="340" font-family="DejaVu Sans" font-size="9" fill="#5f6673">units ±0.03, layers ±0.01 per child</text>
<rect x="244" y="274" width="200" height="78" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<rect x="254" y="283" width="9" height="9" rx="1.5" fill="#009E73"/>
<text x="269" y="291.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330">ES<tspan font-size="9"> · 1,100 cars</tspan></text>
<text x="254" y="307" font-family="DejaVu Sans" font-size="9" fill="#5f6673">copy of one of the 100 fittest</text>
<text x="254" y="318" font-family="DejaVu Sans" font-size="9" fill="#5f6673">+ Gaussian step σ on every weight</text>
<text x="254" y="329" font-family="DejaVu Sans" font-size="9" fill="#5f6673">σ: 1/5 success rule while learning,</text>
<text x="254" y="340" font-family="DejaVu Sans" font-size="9" fill="#5f6673">then the 90% reliability rule</text>
<rect x="454" y="274" width="200" height="78" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<rect x="464" y="283" width="9" height="9" rx="1.5" fill="#F0E442" stroke="#8a7a00" stroke-width="0.8"/>
<text x="479" y="291.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330">NEAT<tspan font-size="9"> · 2,000 cars</tspan></text>
<text x="464" y="307" font-family="DejaVu Sans" font-size="9" fill="#5f6673">species: children by mean fitness</text>
<text x="464" y="318" font-family="DejaVu Sans" font-size="9" fill="#5f6673">crossover 0.75 inside a species</text>
<text x="464" y="329" font-family="DejaVu Sans" font-size="9" fill="#5f6673">add node 0.03, connection 0.05</text>
<text x="464" y="340" font-family="DejaVu Sans" font-size="9" fill="#5f6673">weight perturbation 0.8 (step σ)</text>
<path d="M554,191 L554,264 L134,264" fill="none" stroke="#5f6673" stroke-width="1"/>
<path d="M134,264 L134,274" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<path d="M344,264 L344,274" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<path d="M554,264 L554,274" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<circle cx="554" cy="264" r="1.8" fill="#5f6673"/>
<rect x="34" y="360" width="620" height="18" rx="3" fill="#fdecea" stroke="#c0392b" stroke-opacity="0.9" stroke-width="1"/>
<text x="344" y="372.3" font-family="DejaVu Sans" font-size="9" fill="#9b2c20" text-anchor="middle">Stall boost: ×3 on the decision layer (GA strength s, ES step σ, NEAT output connections); steering sensitivity mutates harder</text>
<text x="34" y="392" font-family="DejaVu Sans" font-size="9" fill="#5f6673">Every child inherits a parent's sensor length and steering sensitivity and mutates them (§9.11).</text>
<rect x="34" y="412" width="120" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="94" y="424.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Checkpoint</text>
<text x="94" y="436.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">after every generation</text>
<path d="M94,400 L94,412" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<path d="M196,428 L205.92,412 L394.08,412 L404,428 L394.08,444 L205.92,444 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="300" y="431.6" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">Level / push / hardening decision</text>
<path d="M154,428 L196,428" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<path d="M404,428 L470,428" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<text x="437" y="423" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">run ends</text>
<rect x="470" y="412" width="150" height="32" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="545" y="424.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Final controller</text>
<text x="545" y="436.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">selection (§9.5)</text>
<path d="M300,444 L300,460 L10,460 L10,12 L34,12" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<text x="306" y="455" font-family="DejaVu Sans" font-size="9" fill="#5f6673">next generation · rules: Figure 10 (§9.4)</text>
</svg>
<p><em>Figure 9. One generation: three maps, time trials, survival, children by method, checkpoint and the level, push or hardening decision.</em></p>
</div>

**Stall boost.** When 100 generations of the speed push pass without mastering the current target, the
push is stalled, and the parts of the cars that decide speed and turning mutate harder until a target is mastered
again; the sensing layers keep their usual mutation, so what the population has learned about reading the road
stays intact.
- **Decision layer:** the weights and biases of the output layer, which choose among the 45 combinations of steering
  and throttle (§6), mutate three times as strongly: the GA's mutation strength (on the output layer of each car's
  own network), the ES's step size σ (also in the speed pushes of the imitation methods) and NEAT's weight
  perturbation of connections into output nodes are multiplied by 3 for those parameters.
- **Steering sensitivity (§9.11):** a child's steering sensitivity mutates with probability 0.5 instead of 0.2 and
  with a standard deviation of 0.2 instead of 0.1 on the log scale (about ±20%); the sensor length mutates as usual.
- The boost depends only on the run's state, which its checkpoint holds, so run 0 replays exactly. Each
  generation's log records whether the boost was on; the report counts the stalls each run broke out of.
- **Evidence:** in the stall-boost pilot (§12.2), an ES run whose speed push had stalled at a target 27.6% shorter was
  continued from its final population twice, with and without the boost; the boosted copy mastered a target 42.0%
  shorter, the plain copy one 31.9% shorter.

**Hardening.** When the speed push ends, the run hardens before its final controller is selected. The target returns
to the tightest target mastered and stays there, and a generation is judged by the rule of the levels: it passes
when, on each of its three Extra hard maps, at least 90% of the population finishes within the target (the lowest map
success share is at least 90%), and three consecutive passing generations end hardening. The run is then hardened,
with the outcome "hardened" (the training-cost table, §11.3). Children, survival and step sizes are those of the speed
push; the stall boost and the patience of 1,000 generations do not apply, so hardening ends only when the population
meets the rule or by an end request.
The speed push selects its last population at a target 2% shorter than the last mastered one, met by half of the
population; hardening selects for finishing at the mastered target.
The runs of the last complete experiment ended without hardening, when their speed push ended or by an end request,
and only 1 of their 32 final controllers finished all 20 validation maps. The summary of such a run has no hardening
record; with its checkpoint beside the summary, the run resumes into hardening at its next start (§3.2): a run that
had reached the mastery goal hardens at its tightest mastered target, and a run still on a level continues its
levels, where mastering Extra hard at the starting target meets the same rule and ends hardening without a speed push.
A run whose speed push ends logs "speed push ended at generation N, hardening at the target X% shorter", and a run
that resumes into hardening logs "hardening at the target X% shorter, from generation N" (or "hardening from
generation N, on <level>"); at its end the run log records "hardened at generation N" ("mastery goal reached at
generation N, hardened" for a run hardened on a level); the status line shows the stage "hardening" against 90%, and
the run summary a hardening record with the target factor, the first generation and, once hardened, the generation,
evaluations and time.

**Safety stop.** The setting `SAFETY_PATIENCE` is None, so a run climbs the levels until it masters Extra hard,
however long that takes. Set to a number n, it would end a run on any level before the mastery goal after n
generations without a new best value of the lowest map success share, averaged over three consecutive generations;
the run would be reported as not having reached the level, with the outcome "stalled". With the setting off, a run
ends only after the mastery goal, through its speed push and hardening, or by an end request. The end of the DAgger
refinement (§8.4), which counts rounds without a new best check, is a different mechanism.

**End request.** While the file `results/end_runs` exists, a run of the experiment ends before its next
generation, once it has trained at least one; a run resumed while the file exists ends at once. It then selects its
final controller from the kept cars of its last generation as at any other end (§9.5), and its outcome is recorded as
"time_limit" ("ended by time" in the training-cost table, §11.3). A run in its speed push ends while chasing a target
it has not mastered, without hardening. A run still on a level selects among cars that have not mastered Extra hard,
and its result is reported as it is. An end request also ends hardening; the outcome is then "time_limit", and the
summary keeps the hardening record. The viewer, the pilots and the tests select no final controller and ignore the
file. `fleet.py end` writes it on every machine (§3.2). It is not a setting, so no record changes.

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 472" width="100%" role="img" aria-label="A run's life: four levels, mastery goal, speed push with stall boost, hardening, safety stop (off) and final controller selection" style="display:block;margin:10pt 0 6pt">
<defs><marker id="figrun-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker><marker id="figrun-head-boost" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#c0392b"/></marker></defs>
<rect x="0" y="0" width="665" height="112" rx="3" fill="#f5f6f8"/>
<rect x="170" y="124" width="495" height="258" rx="3" fill="#f5f6f8"/>
<text x="10" y="16" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Climb the levels</tspan><tspan font-size="9" fill="#5f6673"> · the GA, ES, seeded ES and NEAT start at Easy, the speed pushes at Extra hard; f = 1.3</tspan></text>
<rect x="10" y="40" width="68" height="24" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="44" y="55.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Easy</text>
<rect x="150.25" y="40" width="68" height="24" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="184.25" y="55.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Medium</text>
<rect x="290.5" y="40" width="68" height="24" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="324.5" y="55.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Hard</text>
<rect x="430.75" y="40" width="68" height="24" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="464.75" y="55.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Extra hard</text>
<rect x="571" y="40" width="84" height="24" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="613" y="55.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Mastery goal</text>
<path d="M78,52 L150.25,52" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<rect x="87.12" y="22" width="54" height="24" rx="5" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1"/>
<text x="114.12" y="31.7" font-family="DejaVu Sans" font-size="9" fill="#7a5410" text-anchor="middle">3 passes</text>
<text x="114.12" y="42.2" font-family="DejaVu Sans" font-size="9" fill="#7a5410" text-anchor="middle">in a row</text>
<path d="M218.25,52 L290.5,52" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<rect x="227.38" y="22" width="54" height="24" rx="5" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1"/>
<text x="254.38" y="31.7" font-family="DejaVu Sans" font-size="9" fill="#7a5410" text-anchor="middle">3 passes</text>
<text x="254.38" y="42.2" font-family="DejaVu Sans" font-size="9" fill="#7a5410" text-anchor="middle">in a row</text>
<path d="M358.5,52 L430.75,52" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<rect x="367.62" y="22" width="54" height="24" rx="5" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1"/>
<text x="394.62" y="31.7" font-family="DejaVu Sans" font-size="9" fill="#7a5410" text-anchor="middle">3 passes</text>
<text x="394.62" y="42.2" font-family="DejaVu Sans" font-size="9" fill="#7a5410" text-anchor="middle">in a row</text>
<path d="M498.75,52 L571,52" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<rect x="507.88" y="22" width="54" height="24" rx="5" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1"/>
<text x="534.88" y="31.7" font-family="DejaVu Sans" font-size="9" fill="#7a5410" text-anchor="middle">3 passes</text>
<text x="534.88" y="42.2" font-family="DejaVu Sans" font-size="9" fill="#7a5410" text-anchor="middle">in a row</text>
<path d="M56.24,64 L56.24,75 L31.76,75 L31.76,64" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<path d="M196.49,64 L196.49,75 L172.01,75 L172.01,64" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<path d="M336.74,64 L336.74,75 L312.26,75 L312.26,64" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<path d="M476.99,64 L476.99,75 L452.51,75 L452.51,64" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<text x="44" y="87" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">next generation</text>
<text x="10" y="103" font-family="DejaVu Sans" font-size="9" fill="#5f6673">A generation passes when ≥ 90% of the population succeeds on each of its 3 maps; a failing one restarts the count.</text>
<path d="M12,158 L23.16,140 L136.84,140 L148,158 L136.84,176 L23.16,176 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="80" y="161.6" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">safety stop: off</text>
<path d="M80,112 L80,140" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<text x="84" y="128" font-family="DejaVu Sans" font-size="9" fill="#5f6673">any level</text>
<text x="22" y="196" font-family="DejaVu Sans" font-size="9" fill="#5f6673">if set: new best lowest map</text>
<text x="22" y="207" font-family="DejaVu Sans" font-size="9" fill="#5f6673">success share, mean of 3 gen.</text>
<text x="180" y="140" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Speed push</tspan><tspan font-size="9" fill="#5f6673"> · Extra hard</tspan></text>
<rect x="573" y="152" width="80" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="613" y="164.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Target −2%</text>
<text x="613" y="176.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">f × 0.98</text>
<path d="M613,64 L613,152" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<rect x="399" y="152" width="142" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="470" y="164.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Generation at the target</text>
<text x="470" y="176.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">3 Extra hard layouts</text>
<path d="M573,168 L541,168" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<path d="M392,222 L403.16,204 L536.84,204 L548,222 L536.84,240 L403.16,240 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="470" y="219.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">≥ 50% on every map,</text>
<text x="470" y="231.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">3 generations in a row?</text>
<path d="M470,184 L470,204" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<path d="M548,222 L613,222 L613,184" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<text x="554" y="216" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes</text>
<rect x="583.38" y="196" width="59.24" height="14" rx="7" fill="#fdecea" stroke="#c0392b" stroke-opacity="0.9" stroke-width="1"/>
<text x="613" y="206.24" font-family="DejaVu Sans" font-size="9" fill="#9b2c20" text-anchor="middle">boost OFF</text>
<path d="M404,280 L415.16,262 L524.84,262 L536,280 L524.84,298 L415.16,298 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="470" y="277.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">100 generations</text>
<text x="470" y="289.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">without mastering?</text>
<path d="M470,240 L470,262" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<text x="475" y="254.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no</text>
<rect x="222" y="258" width="148" height="44" rx="2.5" fill="#fdecea" stroke="#c0392b" stroke-opacity="0.9" stroke-width="1"/>
<text x="296" y="276.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Stall boost ON</text>
<text x="296" y="288.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">until the next mastered target</text>
<path d="M404,280 L370,280" fill="none" stroke="#c0392b" stroke-width="1" marker-end="url(#figrun-head-boost)"/>
<text x="387" y="275" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">yes</text>
<text x="205" y="222" font-family="DejaVu Sans" font-size="9" fill="#5f6673">Stall boost: decision layer ×3;</text>
<text x="205" y="233" font-family="DejaVu Sans" font-size="9" fill="#5f6673">steering sensitivity: p 0.2 → 0.5,</text>
<text x="205" y="244" font-family="DejaVu Sans" font-size="9" fill="#5f6673">sd 0.1 → 0.2; sensor length unchanged</text>
<path d="M404,348 L415.16,330 L524.84,330 L536,348 L524.84,366 L415.16,366 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="470" y="345.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">1,000 generations</text>
<text x="470" y="357.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">without mastering?</text>
<path d="M470,298 L470,330" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<text x="475" y="324" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no</text>
<path d="M296,302 L296,314 L470,314" fill="none" stroke="#c0392b" stroke-width="1"/>
<circle cx="470" cy="314" r="1.8" fill="#c0392b"/>
<path d="M404,348 L196,348 L196,168 L399,168" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<text x="204" y="343" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no: next generation</text>
<text x="560" y="262" font-family="DejaVu Sans" font-size="9" fill="#5f6673">tightest target</text>
<text x="560" y="273" font-family="DejaVu Sans" font-size="9" fill="#5f6673">mastered = the</text>
<text x="560" y="284" font-family="DejaVu Sans" font-size="9" fill="#5f6673">speed result</text>
<rect x="382" y="400" width="176" height="30" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="470" y="418.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Final controller selection</text>
<path d="M470,366 L470,400" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<text x="475" y="389" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes: hardening (90%, 3 in a row), then</text>
<path d="M12,158 L12,415 L382,415" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<text x="18" y="240" font-family="DejaVu Sans" font-size="9" fill="#5f6673">never (off)</text>
<text x="470" y="444" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">kept cars of the last generation drive the 20 validation maps (80–99)</text>
<text x="470" y="455" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">within the time limit (2 × the expert's laps);</text>
<text x="470" y="466" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">fastest with all 20 finishes, else most finishes, then mean fitness</text>
</svg>
<p><em>Figure 10. A run's life: the four levels, the mastery goal, the speed push with its stall boost, hardening
and the end of a run (the safety stop is off).</em></p>
</div>

### 9.5 Final controller

When a run ends, every kept car of the last generation drives the 20 validation maps as a time trial within the time
limit of twice the expert's laps (`TIME_LIMIT_FACTOR`). A car succeeds on a map by finishing the three laps within
that limit.

- **Pick.** Of the cars that finish all 20 maps (`FINAL_MIN_SUCCESSES` = 20), the one with the fastest mean lap is
  the run's final controller: the mean over the 20 maps of its lap time (its three-lap time divided by three) divided
  by that map's expert lap time, so that every map counts alike whatever its length. Ties go to the earlier car.
- **Fallback.** If no car finishes all 20 maps, the car with the most finishes is chosen, ties going to the higher
  mean validation fitness and then to the earlier car.

Finishes come first because mean fitness alone would let a fast car that crashes on one map win over a car that
completes every map. Pace then decides because a choice by the number of validation successes, ties broken by mean
fitness, picks cautious cars: in an ES run, the car chosen that way was 32–62% slower than pole on the test circuits,
while the fastest all-round car of the same population was 25–49% slower.

The run summary's `val_successes` counts finishes, and the milestone reads "final controller finishes N of 20
validation maps". The final controller is exported as a `.keras` file, `results/champions/<approach>/seed<k>.keras`,
together with its float64 weights (`seed<k>.npy`) for dense networks or its genome (`seed<k>.json`) for NEAT, and its
setup (`seed<k>.setup.json`, §6, §9.11).

### 9.6 Genetic algorithm

The population size is 2,000. Every car carries its own network: a list of hidden layer sizes, ReLU hidden units,
and the weights and biases of each layer. In generation 0 no car has a hidden layer: each car's 18 inputs are connected
directly to its 45 outputs (855 weights and biases; Glorot-uniform weights, zero biases). From there the structure is
left to evolution. Mutation adds and removes hidden units and whole layers, the number and width of the layers have no
limit, and selection alone decides which structures survive. Survivors are carried over unchanged.

- **Each remaining new individual is created in four steps:**
  1. **Selection:** two parents are each chosen by tournament selection, with a tournament size of 5
     drawn from the whole population. The first is the shape parent.
  2. **Crossover:** the child takes the shape parent's architecture. A weight or bias that both parents have is
     inherited from either parent with probability 0.5; every other value comes from the shape parent. Units are
     identified by their layer, counted from the inputs, and their position within it, and the outputs form a layer
     of their own. A hidden layer's weights and biases are shared where both parents have that layer, over the units
     both have; the output layer's weights are shared only when both parents have the same number of hidden layers;
     the output biases are always shared. Between parents of the same architecture this is uniform crossover.
  3. **Weight mutation:** each weight and bias, with probability 0.1, receives an added sample from N(0, s). The
     strength follows the rule s = s_min + (s_max − s_min)(1 − c), where c is the share of the previous generation's
     children that succeeded on every map, s_max = 0.1 and s_min = s_max / 100. Fewer successful children raise the
     strength and more lower it.
  4. **Structural mutation:** four mutations follow, each with its own probability, in this order:
     - **Add a unit** (p = 0.03): a hidden layer chosen at random gains a unit with Glorot-uniform incoming weights, a
       zero bias and zero outgoing weights, so the car drives exactly as before until weight mutation gives the unit
       influence. A network without hidden layers is left unchanged.
     - **Remove a unit** (p = 0.03): a random unit of a random hidden layer is removed with its weights. Removing the
       only unit of a layer removes the layer, as below.
     - **Add a layer** (p = 0.01): a hidden layer is inserted at a random depth, as wide as the layer below it (the
       18 inputs for the first hidden layer), with identity weights and zero biases. Behind a hidden layer, whose ReLU
       outputs are never negative, the car drives exactly as before; directly behind the inputs, an input that can be
       negative, such as the position across the road, is cut at zero until mutation changes the new layer.
     - **Remove a layer** (p = 0.01): a random hidden layer is removed and the layers on either side are joined by the
       product of their weights (weights W₁W₂, biases b₁W₂ + b₂), which reproduces the network wherever the removed
       units were active.

  Adding and removing are equally likely, so mutation itself favours neither larger nor smaller networks.
- **Stall boost (§9.4):** the output layer of each child's own network mutates three times as strongly.
- **Evaluation:** the networks of a population differ in shape. The numpy forward pass groups the cars by
  architecture and runs one `einsum` per layer for each group, and the compiled simulator receives each car's own
  network program (§5.7), so no network is padded to a common shape and the two simulators agree exactly.
- **Settings:** the structural rates and the empty starting architecture are recorded and checked only in GA runs
  (§3.2).

### 9.7 Evolution strategy

The population size is 1,100. Each child is a copy of one of the 100 fittest cars of the generation (about 9% of the
population), chosen at random, with Gaussian noise of standard deviation σ added to every weight. σ starts at 0.1 and
follows two rules:

- **Learning phase.** While fewer than 90% of the 100 fittest cars succeed on every map, σ follows the 1/5 success
  rule (σ is divided by 0.82 when more than a fifth of the children outperform their parent, success rate `p_s` > 1/5,
  and multiplied by it when fewer do), with a tie counted as half a success.
- **Reliability phase.** Once at least 90% of the 100 fittest succeed, σ is multiplied by 0.82 when fewer
  than 90% of the children succeed and divided by 0.82 otherwise.

σ is kept within [1e-4, 1.0] and reset to 0.1 when the population advances a level. It is not reset
when the target time is shortened.

- **Ties:** fitness is deterministic and the actions are discrete, so on a fitness plateau many
  offspring score exactly the same as their parent, for example when every parent crashes before
  covering any distance and scores 0. If ties counted as failures, `p_s` would be close to zero on
  such a plateau, and σ would shrink to its lower bound, where the search cannot leave the
  plateau. Counting a tie as half a success gives `p_s` = 0.5 on a complete plateau, so σ grows. With
  continuously varying fitness, ties are rare and the rule reduces to the standard 1/5 success rule.

The factor 0.82 follows Schwefel.

### 9.8 Seeded evolution strategy

The seeded ES is the ES of §9.7 with an initial population of the baseline network after its DAgger refinement
(§8.4) and 1,099 copies of it perturbed with noise of standard deviation 0.1. Like the GA, the ES and NEAT, it starts
at Easy. The baseline's own speed push (§8.5) starts from the same network but at Extra
hard. The two therefore differ only in whether the curriculum is repeated, and the comparison shows
whether it helps a network that has already learned to drive.

- **Interpretation:** the seeded ES uses the expert indirectly, through the DAgger-refined baseline. Its training
  time is reported separately from the dataset collection, baseline training and DAgger refinement on which it
  depends.

### 9.9 NEAT

NEAT (neuroevolution of augmenting topologies) evolves the network structure together with its
weights. The genome operations are written in numpy for this project; whole populations drive together like the
other methods, on the compiled simulator (§5.7) or, without it, with the batched numpy pass below.

**Genome.** A genome consists of node genes and connection genes. Every genome has the 18 input nodes
and the 45 output nodes of the common interface. Hidden nodes are added by mutation. Each node except
the inputs carries a bias. Each connection gene holds its source and target nodes, a weight, an enabled
flag and an innovation number. The run keeps a record of structural mutations: a connection between
the same two nodes, or a split of the same connection, receives the same innovation number or node in
every genome in which it occurs, so that the genes of two parents can be aligned. Networks are feed-forward. A
connection that would create a cycle is never added, so NEAT networks, like all others, have no memory.

**Initial population.** Every input is connected to every output: 810 connections and 45 output biases.
Weights are drawn from a standard normal distribution and biases start at zero.

**Children.** The number of children allotted to each species is proportional to the mean fitness of
its members (see Speciation below). A child is produced by crossover with probability 0.75 and otherwise
copied from a single parent. Parents are chosen by tournaments of size 3 within the species. In
crossover, matching genes are inherited from a random parent and the remaining genes from the fitter
parent. A gene disabled in either parent is disabled in the child with probability 0.75. Every child is
then mutated:

- Each weight and bias is perturbed with probability 0.8 by Gaussian noise of standard deviation σ, or
  replaced with probability 0.002 by a standard normal value: about 1.6 of the 810 starting connections per child
  (see Replacement rate below).
- With probability 0.05, a connection is added between two unconnected nodes, provided it creates no
  cycle.
- With probability 0.03, an enabled connection is split by a new hidden node. The incoming connection
  receives weight 1 and the outgoing connection the old weight, and the old connection is disabled.
- With probability 0.01, the enabled flag of a random connection is toggled.

σ starts at 0.5, which suits weights drawn from a standard normal distribution, and follows the
two-phase rule of the ES (§9.7), evaluated on the 100 fittest cars of the population. It is reset to 0.5 when the
population advances a level.

**Replacement rate.** A per-weight replacement rate suits small networks, where it replaces a few weights per child;
on a genome of 810 connections a rate of 0.05 replaces about 41. To set the rate, 400 kept genomes of each of two NEAT
runs were given one ingredient of the mutation at a time and driven on their run's three Medium maps (Table 6).
Perturbation alone, at σ's floor of 0.0001, left the share of children that finish all three within the target at
that of the parents; replacing 5% of the weights halved it (75% to 38%, 64% to 33%), and replacing 0.2% kept it
close to the parents' (74% and 61%). The replacement rate is therefore 0.002.

*Table 6. Share of NEAT children that finish their run's three Medium maps within the target, by mutation
ingredient (400 kept genomes of each of two runs, at generations 290 and 294).*

| Children made with | Run 1 | Run 2 |
|---|---|---|
| No change (the parents) | 75% | 64% |
| Perturbation only (σ = 0.0001) | 75% | 65% |
| Perturbation and replacement 0.05 | 38% | 33% |
| Perturbation and replacement 0.01 | 62% | 58% |
| Perturbation and replacement 0.002 (the rate used) | 74% | 61% |

**Speciation.** Genomes are grouped into species by the compatibility distance
δ = E / N + D / N + 0.4 W̄, where E and D are the numbers of excess and disjoint genes, W̄ is the mean
weight difference of matching genes and N is the number of genes in the larger genome (1 if both have
fewer than 20). A genome joins the first species whose representative, a random member from the
previous generation, lies within the threshold δ_t. Otherwise it founds a new species. δ_t starts at
3.0 and is adjusted by 0.1 each generation towards 20 species. A species whose best fitness has not
improved for 20 generations receives no children, unless it contains the best car of the population.
Because the maps change every generation, a species' best fitness is a record of noisy scores, and after
a level-up or a shorter target the old records are out of reach; at the start of hardening they were set at
another target. Every species' best fitness and stagnation count therefore restart when the population advances a
level, reaches the mastery goal, masters a shorter target or, at the end of its speed push, starts hardening at
another target, so stagnation is measured within one level and target.
Kept cars (§9.3) remain in the population regardless of their species.

**Activation.** Hidden nodes use tanh. Output nodes are linear, and the action is the class with the
highest output, as for every other method.

**Batched forward pass (numpy simulator).** Without the compiled simulator, all genomes are compiled into flat
arrays before a generation drives. Each node's depth is its longest path from an input. The enabled connections of
the whole population are grouped by the depth of their target node. At each time step, the nodes are computed depth
by depth, each depth with one weighted sum over all its connections (`numpy.bincount`) and one activation. The cost
per step grows with the number of connections and the depth, not with the number of cars, and there is no loop over
cars in Python.

**Saved controller.** The final controller is saved as its genome (JSON) and as an equivalent Keras
model. The Keras model has, per depth, a tanh dense layer for the hidden nodes and a linear dense layer for the
outputs, each applied to the concatenation of the inputs and all shallower nodes, with zero weights where the genome
has no connection, and a fixed final layer that orders the outputs and applies the softmax. A test verifies that the
Keras model and the genome choose the same actions. The Keras model serves only for the accuracy and
timing columns of the course comparison (§11.3). The parameter count reported for NEAT is the number of
enabled connections plus biases, not the size of the Keras layers.

The population size is 2,000, like the GA.

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 436" width="100%" role="img" aria-label="Figure: NEAT genomes, from the minimal genome through the add-node and add-connection mutations to a grown genome, and species of similar genomes" style="display:block;margin:10pt 0 6pt">
<defs><marker id="figneat-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker></defs>
<rect x="0" y="0" width="186" height="236" rx="3" fill="#f5f6f8"/>
<text x="9" y="18" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Start: minimal genome</text>
<text x="50" y="39" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">18 inputs</text>
<text x="136" y="39" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">45 outputs</text>
<line x1="55.5" y1="58" x2="130.5" y2="58" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="55.2976" y1="59.4784" x2="130.702" y2="80.5216" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="54.074" y1="61.695" x2="131.926" y2="132.305" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="53.5453" y1="62.2049" x2="132.455" y2="155.795" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="55.2976" y1="80.5216" x2="130.702" y2="59.4784" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="55.5" y1="82" x2="130.5" y2="82" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="54.6579" y1="84.9247" x2="131.342" y2="133.075" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="54.074" y1="85.695" x2="131.926" y2="156.305" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="54.074" y1="132.305" x2="131.926" y2="61.695" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="54.6579" y1="133.075" x2="131.342" y2="84.9247" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="55.5" y1="136" x2="130.5" y2="136" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="55.2976" y1="137.478" x2="130.702" y2="158.522" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="53.5453" y1="155.795" x2="132.455" y2="62.2049" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="54.074" y1="156.305" x2="131.926" y2="85.695" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="55.2976" y1="158.522" x2="130.702" y2="137.478" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<line x1="55.5" y1="160" x2="130.5" y2="160" stroke="#c0392b" stroke-width="0.9" stroke-opacity="0.8"/>
<circle cx="50" cy="58" r="5.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="136" cy="58" r="5.5" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<circle cx="50" cy="82" r="5.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="136" cy="82" r="5.5" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<circle cx="50" cy="136" r="5.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="136" cy="136" r="5.5" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<circle cx="50" cy="160" r="5.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="136" cy="160" r="5.5" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<circle cx="50" cy="104" r="1.1" fill="#5f6673"/>
<circle cx="50" cy="109" r="1.1" fill="#5f6673"/>
<circle cx="50" cy="114" r="1.1" fill="#5f6673"/>
<circle cx="136" cy="104" r="1.1" fill="#5f6673"/>
<circle cx="136" cy="109" r="1.1" fill="#5f6673"/>
<circle cx="136" cy="114" r="1.1" fill="#5f6673"/>
<text x="9" y="190" font-family="DejaVu Sans" font-size="9" fill="#5f6673">every input wired to every output:</text>
<text x="9" y="201" font-family="DejaVu Sans" font-size="9" fill="#5f6673">810 connections + 45 output biases</text>
<text x="9" y="212" font-family="DejaVu Sans" font-size="9" fill="#5f6673">weights N(0, 1), biases 0</text>
<rect x="204" y="0" width="258" height="236" rx="3" fill="#f5f6f8"/>
<path d="M189,109 L201,109" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figneat-head)"/>
<text x="213" y="18" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Mutations of every child</text>
<text x="213" y="38" font-family="DejaVu Sans" font-size="9" fill="#1d2330">add node</text>
<text x="261.262" y="38" font-family="DejaVu Sans" font-size="9" fill="#5f6673">p = 0.03</text>
<line x1="228" y1="74" x2="280" y2="74" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<text x="254" y="69" font-family="DejaVu Sans" font-size="9" fill="#c0392b" text-anchor="middle">w</text>
<circle cx="222" cy="74" r="6" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="286" cy="74" r="6" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<text x="222" y="90" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">in</text>
<text x="286" y="90" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">out</text>
<path d="M304,74 L332,74" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figneat-head)"/>
<line x1="356" y1="74" x2="428" y2="74" stroke="#5f6673" stroke-width="1" stroke-opacity="0.7" stroke-dasharray="3 2.5"/>
<line x1="355.315" y1="71.216" x2="386.685" y2="54.784" stroke="#5f6673" stroke-width="1"/>
<text x="365" y="59" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">1</text>
<line x1="397.315" y1="54.784" x2="428.685" y2="71.216" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<text x="420" y="59" font-family="DejaVu Sans" font-size="9" fill="#c0392b" text-anchor="middle">w</text>
<circle cx="350" cy="74" r="6" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="392" cy="52" r="6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="1"/>
<circle cx="434" cy="74" r="6" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<text x="350" y="90" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">in</text>
<text x="434" y="90" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">out</text>
<text x="402" y="46" font-family="DejaVu Sans" font-size="9" fill="#1d2330">new</text>
<text x="392" y="87" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">disabled</text>
<text x="213" y="112" font-family="DejaVu Sans" font-size="9" fill="#1d2330">add connection</text>
<text x="288.391" y="112" font-family="DejaVu Sans" font-size="9" fill="#5f6673">p = 0.05</text>
<line x1="228" y1="128" x2="258" y2="128" stroke="#5f6673" stroke-width="1"/>
<line x1="269.692" y1="129.897" x2="300.308" y2="140.103" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<line x1="227.918" y1="155.014" x2="300.082" y2="142.986" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<circle cx="222" cy="128" r="6" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="222" cy="156" r="6" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="264" cy="128" r="6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="1"/>
<circle cx="306" cy="142" r="6" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<path d="M320,142 L342,142" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figneat-head)"/>
<line x1="364" y1="128" x2="394" y2="128" stroke="#5f6673" stroke-width="1"/>
<line x1="405.692" y1="129.897" x2="436.308" y2="140.103" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<line x1="363.918" y1="155.014" x2="436.082" y2="142.986" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<line x1="362.992" y1="152.672" x2="395.008" y2="131.328" stroke="#1d2330" stroke-width="1.6"/>
<circle cx="358" cy="128" r="6" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="358" cy="156" r="6" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="400" cy="128" r="6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="1"/>
<circle cx="442" cy="142" r="6" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<text x="370" y="144" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="end">new</text>
<text x="213" y="178" font-family="DejaVu Sans" font-size="9" fill="#5f6673">only between unconnected nodes, never a cycle</text>
<text x="213" y="202" font-family="DejaVu Sans" font-size="9" fill="#5f6673">toggle an enabled flag: p = 0.01;</text>
<text x="213" y="213" font-family="DejaVu Sans" font-size="9" fill="#5f6673">each weight and bias: perturbed p = 0.8</text>
<text x="213" y="224" font-family="DejaVu Sans" font-size="9" fill="#5f6673">by N(0, σ²), σ₀ = 0.5, or replaced p = 0.002</text>
<rect x="480" y="0" width="185" height="236" rx="3" fill="#f5f6f8"/>
<path d="M465,109 L477,109" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figneat-head)"/>
<text x="489" y="18" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Later: a grown genome</text>
<text x="504" y="39" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">inputs</text>
<text x="572" y="39" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">hidden</text>
<text x="640" y="39" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">outputs</text>
<line x1="509.322" y1="59.3883" x2="544.678" y2="68.6117" stroke="#5f6673" stroke-width="1"/>
<line x1="509.322" y1="80.6117" x2="544.678" y2="71.3883" stroke="#5f6673" stroke-width="1"/>
<line x1="509.322" y1="137.388" x2="544.678" y2="146.612" stroke="#5f6673" stroke-width="1"/>
<line x1="509.322" y1="158.612" x2="544.678" y2="149.388" stroke="#5f6673" stroke-width="1"/>
<line x1="554.116" y1="73.6482" x2="589.884" y2="105.352" stroke="#5f6673" stroke-width="1"/>
<line x1="554.116" y1="144.352" x2="589.884" y2="112.648" stroke="#5f6673" stroke-width="1"/>
<line x1="509.5" y1="58" x2="634.5" y2="58" stroke="#5f6673" stroke-width="1" stroke-opacity="0.7" stroke-dasharray="3 2.5"/>
<line x1="555.452" y1="69.2731" x2="634.548" y2="58.7269" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<line x1="555.452" y1="70.7269" x2="634.548" y2="81.2731" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<line x1="598.743" y1="106.216" x2="635.257" y2="84.7841" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<line x1="598.743" y1="111.784" x2="635.257" y2="133.216" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<line x1="555.452" y1="148.727" x2="634.548" y2="159.273" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<line x1="509.5" y1="82" x2="634.5" y2="82" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<line x1="509.5" y1="160" x2="634.5" y2="160" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<line x1="509.5" y1="136" x2="634.5" y2="136" stroke="#c0392b" stroke-width="1" stroke-opacity="0.8"/>
<circle cx="504" cy="58" r="5.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="504" cy="82" r="5.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="504" cy="136" r="5.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="504" cy="160" r="5.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="1"/>
<circle cx="640" cy="58" r="5.5" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<circle cx="640" cy="82" r="5.5" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<circle cx="640" cy="136" r="5.5" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<circle cx="640" cy="160" r="5.5" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="1.2"/>
<circle cx="550" cy="70" r="5.5" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="1"/>
<circle cx="550" cy="148" r="5.5" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="1"/>
<circle cx="594" cy="109" r="5.5" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="1"/>
<circle cx="504" cy="104" r="1.1" fill="#5f6673"/>
<circle cx="504" cy="109" r="1.1" fill="#5f6673"/>
<circle cx="504" cy="114" r="1.1" fill="#5f6673"/>
<circle cx="640" cy="104" r="1.1" fill="#5f6673"/>
<circle cx="640" cy="109" r="1.1" fill="#5f6673"/>
<circle cx="640" cy="114" r="1.1" fill="#5f6673"/>
<text x="489" y="190" font-family="DejaVu Sans" font-size="9" fill="#5f6673">hidden nodes tanh, outputs linear;</text>
<text x="489" y="201" font-family="DejaVu Sans" font-size="9" fill="#5f6673">feed-forward, columns by depth;</text>
<text x="489" y="212" font-family="DejaVu Sans" font-size="9" fill="#5f6673">dashed: a disabled connection</text>
<rect x="0" y="252" width="404" height="146" rx="3" fill="#f5f6f8"/>
<text x="9" y="270" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Species</text>
<text x="60.2328" y="270" font-family="DejaVu Sans" font-size="9" fill="#5f6673">similar genomes, grouped by compatibility; population 2,000</text>
<rect x="9" y="284" width="86" height="98" rx="10" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.45" stroke-width="1"/>
<line x1="19" y1="294" x2="45" y2="298" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="19" y1="294" x2="45" y2="306" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="19" y1="302" x2="45" y2="298" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="19" y1="302" x2="45" y2="306" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="19" y1="310" x2="45" y2="298" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="19" y1="310" x2="45" y2="306" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="19" cy="294" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="19" cy="302" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="19" cy="310" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="45" cy="298" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="45" cy="306" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<line x1="59" y1="306" x2="85" y2="310" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="59" y1="306" x2="85" y2="318" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="59" y1="314" x2="85" y2="310" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="59" y1="314" x2="85" y2="318" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="59" y1="322" x2="85" y2="310" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="59" y1="322" x2="85" y2="318" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="59" cy="306" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="59" cy="314" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="59" cy="322" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="85" cy="310" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="85" cy="318" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<line x1="23" y1="334" x2="49" y2="338" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="23" y1="334" x2="49" y2="346" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="23" y1="342" x2="49" y2="338" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="23" y1="342" x2="49" y2="346" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="23" y1="350" x2="49" y2="338" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="23" y1="350" x2="49" y2="346" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="23" cy="334" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="23" cy="342" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="23" cy="350" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="49" cy="338" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="49" cy="346" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<line x1="59" y1="352" x2="85" y2="356" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="59" y1="352" x2="85" y2="364" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="59" y1="360" x2="85" y2="356" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="59" y1="360" x2="85" y2="364" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="59" y1="368" x2="85" y2="356" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="59" y1="368" x2="85" y2="364" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="59" cy="352" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="59" cy="360" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="59" cy="368" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="85" cy="356" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="85" cy="364" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<rect x="103" y="284" width="86" height="98" rx="10" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.45" stroke-width="1"/>
<line x1="113" y1="292" x2="126" y2="294.4" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="113" y1="300" x2="126" y2="294.4" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="126" y1="294.4" x2="139" y2="296" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="113" y1="300" x2="139" y2="304" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="113" y1="308" x2="139" y2="296" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="113" y1="308" x2="139" y2="304" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="113" cy="292" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="113" cy="300" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="113" cy="308" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="139" cy="296" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="139" cy="304" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="126" cy="294.4" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<line x1="153" y1="298" x2="166" y2="300.4" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="153" y1="306" x2="166" y2="300.4" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="166" y1="300.4" x2="179" y2="302" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="153" y1="306" x2="179" y2="310" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="153" y1="314" x2="179" y2="302" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="153" y1="314" x2="179" y2="310" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="153" cy="298" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="153" cy="306" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="153" cy="314" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="179" cy="302" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="179" cy="310" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="166" cy="300.4" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<line x1="133" y1="324" x2="146" y2="326.4" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="133" y1="332" x2="146" y2="326.4" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="146" y1="326.4" x2="159" y2="328" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="133" y1="332" x2="159" y2="336" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="133" y1="340" x2="159" y2="328" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="133" y1="340" x2="159" y2="336" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="133" cy="324" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="133" cy="332" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="133" cy="340" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="159" cy="328" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="159" cy="336" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="146" cy="326.4" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<line x1="113" y1="354" x2="126" y2="356.4" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="113" y1="362" x2="126" y2="356.4" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="126" y1="356.4" x2="139" y2="358" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="113" y1="362" x2="139" y2="366" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="113" y1="370" x2="139" y2="358" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="113" y1="370" x2="139" y2="366" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="113" cy="354" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="113" cy="362" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="113" cy="370" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="139" cy="358" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="139" cy="366" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="126" cy="356.4" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<line x1="153" y1="356" x2="166" y2="358.4" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="153" y1="364" x2="166" y2="358.4" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="166" y1="358.4" x2="179" y2="360" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="153" y1="364" x2="179" y2="368" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="153" y1="372" x2="179" y2="360" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="153" y1="372" x2="179" y2="368" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="153" cy="356" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="153" cy="364" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="153" cy="372" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="179" cy="360" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="179" cy="368" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="166" cy="358.4" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<rect x="197" y="284" width="86" height="98" rx="10" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.45" stroke-width="1"/>
<line x1="209" y1="298" x2="222" y2="298.8" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="209" y1="306" x2="222" y2="313.2" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="209" y1="314" x2="222" y2="313.2" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="222" y1="298.8" x2="235" y2="302" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="222" y1="313.2" x2="235" y2="310" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="222" y1="313.2" x2="235" y2="302" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="209" y1="298" x2="235" y2="310" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="209" cy="298" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="209" cy="306" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="209" cy="314" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="235" cy="302" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="235" cy="310" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="222" cy="298.8" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="222" cy="313.2" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<line x1="247" y1="322" x2="260" y2="322.8" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="247" y1="330" x2="260" y2="337.2" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="247" y1="338" x2="260" y2="337.2" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="260" y1="322.8" x2="273" y2="326" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="260" y1="337.2" x2="273" y2="334" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="260" y1="337.2" x2="273" y2="326" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="247" y1="322" x2="273" y2="334" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="247" cy="322" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="247" cy="330" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="247" cy="338" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="273" cy="326" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="273" cy="334" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="260" cy="322.8" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="260" cy="337.2" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<line x1="213" y1="350" x2="226" y2="350.8" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="213" y1="358" x2="226" y2="365.2" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="213" y1="366" x2="226" y2="365.2" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="226" y1="350.8" x2="239" y2="354" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="226" y1="365.2" x2="239" y2="362" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="226" y1="365.2" x2="239" y2="354" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="213" y1="350" x2="239" y2="362" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="213" cy="350" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="213" cy="358" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="213" cy="366" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="239" cy="354" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="239" cy="362" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="226" cy="350.8" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="226" cy="365.2" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<rect x="291" y="284" width="86" height="98" rx="10" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.45" stroke-width="1"/>
<line x1="301" y1="302" x2="309.58" y2="306" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="301" y1="310" x2="309.58" y2="306" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="309.58" y1="306" x2="318.42" y2="298.8" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="301" y1="294" x2="318.42" y2="298.8" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="318.42" y1="298.8" x2="327" y2="298" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="318.42" y1="298.8" x2="327" y2="306" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="301" y1="310" x2="327" y2="306" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="301" cy="294" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="301" cy="302" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="301" cy="310" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="327" cy="298" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="327" cy="306" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="309.58" cy="306" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="318.42" cy="298.8" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<line x1="341" y1="314" x2="349.58" y2="318" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="341" y1="322" x2="349.58" y2="318" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="349.58" y1="318" x2="358.42" y2="310.8" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="341" y1="306" x2="358.42" y2="310.8" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="358.42" y1="310.8" x2="367" y2="310" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="358.42" y1="310.8" x2="367" y2="318" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="341" y1="322" x2="367" y2="318" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="341" cy="306" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="341" cy="314" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="341" cy="322" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="367" cy="310" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="367" cy="318" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="349.58" cy="318" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="358.42" cy="310.8" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<line x1="305" y1="342" x2="313.58" y2="346" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="305" y1="350" x2="313.58" y2="346" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="313.58" y1="346" x2="322.42" y2="338.8" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="305" y1="334" x2="322.42" y2="338.8" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="322.42" y1="338.8" x2="331" y2="338" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="322.42" y1="338.8" x2="331" y2="346" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="305" y1="350" x2="331" y2="346" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="305" cy="334" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="305" cy="342" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="305" cy="350" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="331" cy="338" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="331" cy="346" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="313.58" cy="346" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="322.42" cy="338.8" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<line x1="341" y1="360" x2="349.58" y2="364" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="341" y1="368" x2="349.58" y2="364" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="349.58" y1="364" x2="358.42" y2="356.8" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="341" y1="352" x2="358.42" y2="356.8" stroke="#5f6673" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="358.42" y1="356.8" x2="367" y2="356" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="358.42" y1="356.8" x2="367" y2="364" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<line x1="341" y1="368" x2="367" y2="364" stroke="#c0392b" stroke-width="0.7" stroke-opacity="0.8"/>
<circle cx="341" cy="352" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="341" cy="360" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="341" cy="368" r="2.3" fill="#ffffff" stroke="#5f6673" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="367" cy="356" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="367" cy="364" r="2.3" fill="#ffffff" stroke="#1d2330" stroke-opacity="1" stroke-width="0.9"/>
<circle cx="349.58" cy="364" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<circle cx="358.42" cy="356.8" r="2.6" fill="#F0E442" stroke="#1d2330" stroke-opacity="1" stroke-width="0.8"/>
<text x="382" y="336" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330">…</text>
<text x="416" y="262" font-family="DejaVu Sans" font-size="9" fill="#1d2330">δ = E/N + D/N + 0.4 W̄</text>
<text x="416" y="273.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">E, D: excess and disjoint genes; N: genes</text>
<text x="416" y="285" font-family="DejaVu Sans" font-size="9" fill="#5f6673">of the larger genome; W̄: mean weight</text>
<text x="416" y="296.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">difference of matching genes</text>
<text x="416" y="312" font-family="DejaVu Sans" font-size="9" fill="#5f6673">A genome joins the first species within δₜ;</text>
<text x="416" y="323.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">δₜ starts at 3.0 and moves by 0.1 per</text>
<text x="416" y="335" font-family="DejaVu Sans" font-size="9" fill="#5f6673">generation towards 20 species.</text>
<text x="416" y="350.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">Children per species ∝ its mean fitness;</text>
<text x="416" y="362" font-family="DejaVu Sans" font-size="9" fill="#5f6673">parents by tournaments of 3 within it;</text>
<text x="416" y="373.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">crossover p = 0.75.</text>
<text x="416" y="389" font-family="DejaVu Sans" font-size="9" fill="#5f6673">No children after 20 generations without</text>
<text x="416" y="400.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">improvement, unless it holds the best car.</text>
<line x1="1" y1="412.5" x2="23" y2="412.5" stroke="#c0392b" stroke-width="1.4" stroke-opacity="0.8"/>
<text x="30" y="416" font-family="DejaVu Sans" font-size="9" fill="#1d2330">connection into an output node, the decision layer of NEAT</text>
<text x="30" y="428" font-family="DejaVu Sans" font-size="9" fill="#c0392b">stall boost: mutates ×3 (§9.4)</text>
</svg>
<p><em>Figure 11. NEAT: a minimal genome grows hidden nodes and connections by mutation, and similar genomes form species.</em></p>
</div>

### 9.10 Computational performance

The simulation is vectorized over individuals, and rays use sphere tracing (§5.2).

- **Work distribution:** with the compiled simulator (§5.7), a generation is one call in which threads drive the
  pairs of map and car. With the numpy simulator, the evaluation is divided into tasks, each covering one of the
  generation's three tracks for one share of the population, distributed across worker processes (joblib); each
  numpy step has a fixed cost besides its cost per car, so a task should cover as many cars as possible on one track.
  Results are independent of the division and of the simulator, because cars do not interact.
- **Number of threads or workers:** the compiled simulator uses every available logical processor (§3.2); the
  number of worker processes of the numpy simulator is chosen with the generation-time benchmark in `bench.py`, which
  recommends a count only for numpy. On numpy it times expert cars that each complete one lap of three Extra hard
  layouts, divided into tasks as in training; on the compiled simulator it times the training evaluation path (below).
  On the desktop, the first 8 threads run on the 8 performance cores and the others on their second hardware threads
  and on the efficiency cores, so each thread beyond the eighth adds about half as much speed as one of the first
  eight. With the compiled simulator, it times trained cars when a checkpoint of a dense population exists in
  `results/runs`: the cars of the newest such checkpoint, repeated or truncated to the population size, drive three
  laps of the fixed Extra hard layouts 0–2 (Austin in both directions and Budapest) against the starting target time
  (factor 1.3), as in training, since layouts repeat and a checkpoint records no map list. Without such a checkpoint,
  it times expert cars, which neither sense nor run a network and are therefore a lighter load; its output names the
  workload timed.
- **Keras training:** `steps_per_execution` (§8.1) is chosen with the Keras training-time benchmark in
  `bench.py`; XLA compilation is off for reliability (§8.1).
- **Fresh maps:** each generation builds its three maps in the process that drives them: the calling process with the
  compiled simulator, the worker processes with numpy; with the compiled simulator the three builds of a generation
  are sequential. With the compiled expert lap (§5.7), a map takes about 0.3 s to build on Easy, 0.4 s on Medium,
  1–2 s on Hard and 1.5–3 s on the validation setting, its racing line included (see Racing lines). Every process
  keeps at most 8 built tracks, so memory stays constant over runs of many thousands of generations.
- **Sensor length:** tracing time grows with a ray's length, so a population that evolves long sensors drives each
  generation more slowly; sensors start at 150 m and may evolve up to 1,000 m (§9.11). The tracing time does not
  affect the results.
- **Real circuits:** a real circuit's grids cover up to about 1.3 by 2.2 km (Monza), 11.5 million cells at 0.5 m;
  its distance map takes about 47 MB in float32. The grids of the 23 circuits are built once per machine, in 1–3 s
  each with a peak of about 1 GB, into `results/circuit_grids/` (about 570 MB), memory-mapped and shared by every
  process and both directions; grid files made with other code or settings are removed once they are a day old.
- **Racing lines:** a racing line takes 0.51 s per real layout on average and 1.35 s at most (§4.8);
  the lines of the real layouts are computed once per machine into `results/racing_lines/` (a few megabytes), a folder
  other than the grids', and loaded by every process. A generated map's line adds 0.06–0.78 s to the map's build,
  depending on its level, and its arrays with the expert's target speeds about 0.4 MB to a track of 7 km.
- **Memory per process:** since the real circuits' grids are memory-mapped, the in-process cost is the generated
  maps: a Hard map takes about 30 MB, a validation map up to 74 MB. With populations half their present size, an
  evolution process peaked at about 0.34 GB per generation on Easy and 0.69 GB on Hard, and at about 2.4 GB in the
  final selection, which holds all 20 validation maps; the present populations raise the per-generation peak. The
  `at_once` bound of each machine (§3.2) rests on the peaks measured on that machine with those smaller populations.
- **NEAT:** with the compiled simulator each genome becomes a network program of nodes ordered by depth (§5.7); with
  numpy all genomes are compiled into flat arrays and evaluated together, depth by depth (§9.9). Either way NEAT
  drives its whole population at once like the other methods.
- **GA:** the cost of a generation grows with the networks the GA evolves. No size limit is set; the run log records
  the population's network size every generation (§9.1), and the report shows it (§11.4).
- **Further measures** (fewer sphere-tracing steps, then shorter time limits) are held in reserve and not applied.

GPU execution is not used: the per-operation launch cost on a GPU exceeds the arithmetic of networks
and car batches of this size.

### 9.11 Evolved car setup

Every car of an evolutionary population carries two setup values next to its network: its **sensor length** L
(30–1,000 m) and its **steering sensitivity** k (⅓–3). Both are stored and mutated on a logarithmic scale, so halving
and doubling are equally large changes. The setup carries no cost; it is judged only by the time trials, through
success and lap time.

- **Sensor length** limits all nine rays of the car (§5.2).
- **Steering sensitivity** multiplies the steering of every action class before the grip limit (§5.1). Below 1 it
  gives finer steering but a gentler sharpest turn; above 1 it cannot exceed the grip limit: it strengthens the
  steering levels below 1/k and saturates every level at or above 1/k, so the steering becomes coarser.
- **Populations:** the GA, the ES, NEAT, the seeded ES and the speed push of each imitation method evolve their setup.
  The imitation training and DAgger use L = 150 m and k = 1, because the expert data and corrections are recorded
  with those sensors.
- **Initial setups:** every car starts with a sensor length of 150 m. Populations that start from random networks
  (GA, ES, NEAT) draw every car's steering sensitivity uniformly on the logarithmic scale within its range;
  populations that start from a trained network (seeded ES, speed pushes) start with k = 1, the setup the network was
  trained with.
- **Inheritance and mutation**, the same for every method: a kept car keeps its setup; a child inherits each value
  from a parent (GA: from either crossover parent with equal probability; ES: from its parent; NEAT: from the fitter
  parent of a crossover or from the single parent); each inherited value is mutated with probability 0.2 by Gaussian
  noise with standard deviation 0.1 on the natural-logarithm scale (about ±10%) and clipped to its range; in a stall
  of the speed push the steering sensitivity mutates with probability 0.5 and standard deviation 0.2 (§9.4). Setup
  mutation is independent of the network step sizes.
- **Saved controllers** store their setup (§6). The evaluation and the application drive every controller with its
  own setup, and the network figure names it.

## 10. Visualization Application

The application is a single pygame window with one top-down camera and two modes. All cars are driven
by controllers, and the application has no manual driving.

### 10.1 Color assignment

Colors are drawn from the Okabe–Ito palette, which stays distinguishable under common forms of color
blindness. Each approach keeps the same color in the application, legends and figures.

*Table 7. Color assignment.*

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

Training started in the application is shown as it trains; a reported run is followed through the latest
generation it keeps for the viewer.

- **Controls:**
  - The user selects an approach, a run (0, the seeded run, shown as "0 (seeded)", or 1–3; for a notebook run also
    any other run of the shown folder) and a difficulty (§10.4).
  - For the GA, ES, seeded ES and NEAT, the user sets the population size (100, 250, 550, 1,000 or 2,000; the
    reported size by default) before starting.
  - Training is started and paused with a play/pause control (or the Space key) and ended with a stop control, which
    freezes the displayed cars. After a stop, the approach, run, difficulty and population size can be changed.
  - Changing a setting is possible only while no training is running.
- **Training process:**
  - Training runs in a separate process, with the threads or worker processes of a reported run (§3.2), in step with
    the view: after each generation, trial or epoch it waits until the view has replayed it, so the generation on
    screen is always the one just trained and every generation is shown. While the race view is shown, no generation
    is replayed, so training does not wait and runs at full speed. The training process is not a daemon process,
    because joblib cannot start worker processes from a daemon process; it is terminated explicitly when training
    stops or the application closes.
  - After each generation, completed Keras Tuner trial or baseline epoch, it sends the population, fitness values,
    map success shares, level, target factor and map seeds to the view through a `multiprocessing.Queue`.
  - When the training process ends, the view reads the queue once more, so the final report is displayed.
  - Pausing freezes the replay and halts training at the next generation boundary.
  - Training started from the application writes only to `results/app_runs/`, never to the reported results, and is
    kept there (§10.6). The seeded ES in the application starts from the DAgger-refined baseline of the selected run
    seed. If it does not exist, the seeded ES cannot be started and the view states that notebook 03 must be run
    first. A refined baseline made for another input count or action set is refused: the training process names it
    and ends with exit code 3, as for a saved run that cannot be resumed (§10.6), and the view reports the run
    unusable, so the seeded ES never trains a network on part of the inputs and makes no car that the race view would
    leave out (§10.3).
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
  and a reported run (`results/runs/`) of the selected approach and run.
  - After each generation, a reported evolutionary run (and the speed push of an imitation approach) keeps its
    latest evaluated generation (population, fitness, maps, level and statistics) in `replay.pkl` next to its
    checkpoint, overwritten every generation. Training never reads this file.
  - The view replays that generation with the same display selection, panel and plot (from the run's log). When a
    replay ends, the view loads a newer generation if one was written (looking at most once a second) and otherwise
    replays the same one, so a running experiment is followed live. Generations in between are skipped: the view
    never slows the run.
  - The view is read-only: it never writes to the run, starts or pauses it. Play/Pause pauses only the replay; stop,
    new run, difficulty and population size do not apply. A run without a replay file, a missing run and an unreadable
    file are named in the status line.
- **Results folder:** a *Runs* control (*Runs: current* for the current results) cycles the notebook runs, and in the
  race view the controllers (§10.3), between the current results and each archive in `results/archive/` (§3.2). An
  archive's cars are named by seed (*GA s0*) and carry no stripe. A folder whose controllers have another number of
  inputs or actions than the common interface (§6) is not replayed, and the status line names its count (§10.3).
  Training started in the application always uses the current results.
- **Track display:** the side panel names the track being played back: the circuit and direction at Extra hard,
  otherwise its seed, level and road width range. The imitation views drive seed 80 at the selected level (Easy to
  Hard) or the first training layout (Austin) at Extra hard, so its map is shown before training starts. The
  evolutionary approaches draw new maps every generation, so their map appears with the first generation.
- **Display selection:** All, Best or Worst, with a displayed-car count N from 1 to the population size:
  - All shows a fixed random sample of N individuals.
  - Best shows the N fittest.
  - Worst shows the N least fit.

  The fittest displayed individual is outlined, and its sensor rays can be shown, each drawn to the car's own sensor
  length.
- **Camera and window:** the camera shows the whole map or follows a car at 2×, 4× or 8× (the outlined car in
  the training view, the race leader in the race view; button, mouse wheel or C key); a zoomed view draws only
  the part of the track on screen. The window can be resized, or switched to full screen with F11 or the Full
  screen button (Esc leaves it); the map and panel follow its size.
- **Side panel:**
  - approach, level and generation
  - the success share of each of the generation's maps against the generation's requirement (90% on the levels and
    in hardening, 50% in the speed push, §9.4), and the streak
  - the target reduction during the speed push
  - mutation strength (GA) or σ (ES, NEAT), and the number of species (NEAT)
  - the population's mean sensor length and steering sensitivity, with their standard deviations
  - best and mean fitness
  - a live plot of the lowest map success share per generation against the requirement line (that of the newest
    generation), with level advancements, the mastery goal, target reductions and hardening marked (for the
    imitation approaches, the best and mean validation accuracy)
- **Imitation approaches:** each completed Keras Tuner trial or baseline epoch is treated as one individual and is
  driven on the imitation views' map (see Track display).

### 10.3 Race view

- **Controllers:** the view loads whichever final controllers exist in `results/champions/`. A race control selects
  the runs that race: *seeded* (run 0 of every method), *unseeded* (runs 1–3), *best* (each method's best run on the
  validation maps; without the evaluation's record, run 0) or *all* (every run, up to 32 cars); the default is *best*,
  and the choice is kept while the view is open. Each method also has an on/off switch. Every run of a method drives
  in its color; unseeded cars carry a white stripe, and the leaderboard names run 0 "<method> (seeded)" and the
  others "<method> r1" to "<method> r3".
  A network made for another number of inputs or action classes than the common interface (18 inputs, 45 actions,
  §6) would be misread. A results folder, the current one or an archive (chosen with the *Runs* control, §10.2), whose record
  names another number of actions or inputs is therefore not replayed, with a note naming that count, and each of its
  controllers is left out of a race with a note. Every controller is also checked when it is loaded: one with another
  number of inputs or outputs is refused and left out of the race.
- **Track:** at Extra hard, one of the four test circuits is selected with the arrow keys or on-screen buttons,
  in its racing direction, and its name and pole lap are shown. At the generated levels and the validation setting
  (§10.4), a display seed (1000–1019) is selected instead. The selected track is drawn as soon as it is selected,
  with the cars on the start line, and its name or seed and its level are shown. Its racing line (§4.8) is drawn as
  a thin light line on the road, so how closely each car follows it can be seen.
- **Race:** the selected controllers race three laps without interacting, each in its assigned color.
  A start control begins the race, and a stop control ends it and returns to the map preview, after
  which the controllers, seed and difficulty can be changed. Sensor rays can be shown for every racing car, each
  drawn to that car's sensor length.
- **Leaderboard:**
  - position, ranked by laps completed plus progress
  - last lap time and best lap time, and on a test circuit the best flying lap's difference to the pole lap
  - crash status

  A crashed car stays where it crashed and is drawn faded.
- **Playback speed:** 1×, 2×, 4× or 8×, where 1× is twice real time.
- **App cars:** a cycle button switches the controller list between the reported controllers and the app cars, the
  best cars of training started in the application (§10.6), labelled for example *GA (app) s1*. Each method's app-car
  control cycles *off* (the default) and its saved runs, so one app car per method can race next to the reported
  controllers, drawn with a white border; app cars never enter the evaluation. When many cars race, the leaderboard
  shows as many rows as fit and states how many more there are.
- **Leader's speed:** a box at the top left of the map shows the race leader's name and speed in mph.
- **Pole-lap car:** on a test circuit, a dark car laps the racing line in exactly the circuit's pole
  time (§11.1), its speed following the car's ideal lap on the line (§4.8) scaled to that time. The leaderboard lists
  it as, for example, "Lewis Hamilton 2020 (pole)" in its place by progress; it does not collide with the cars.

### 10.4 Difficulty

The difficulty control maps five settings to the curriculum levels and the validation setting of §4.2.

*Table 8. Difficulty settings.*

| Difficulty | Level | Track |
|---|---|---|
| Easy | 1 | Gentle loop on a 12–16 m road |
| Medium | 2 | Circuit on a 10–16 m road |
| Hard | 3 | Circuit with a main straight into a hairpin, possibly a chicane or esses, on a 9–15 m road |
| Extra hard | 4 | Real circuits: the four test circuits in the race view, the 19 training circuits in the training view |
| Validation | – | Generated circuit harder than any training level, on a 7.5–12 m road; race view only |

The training view offers the four training difficulties; Validation is offered in the race view only. In the race
view, the difficulty sets the level of the race track. In the training view, it sets the
level at which evolutionary training starts (the curriculum still advances from there) and the level of
the track on which imitation trials and epochs are driven. The difficulty applies only to the
application; the reported experiment always starts at level 1 and evaluates on the test circuits.

### 10.5 Grip selector

A selector with the options **Normal** (the training grip a(v), §5.1) and **Unlimited** (10 times a(v) at every
speed) sets the grip of everything the application simulates for display: the race view and the replays in the
training view. It never affects training, which always uses the training grip, so the reported results cannot depend
on it. Ten times the grip stands in for unlimited grip, because an unbounded grip would let a car turn arbitrarily far
within one time step. The grip cannot be changed while a race runs; a change restarts a running training replay.

### 10.6 Saved application training

Training started in the application is kept, separately from the reported results.

- **One saved run per method and seed** in `results/app_runs/<method>/seed<k>/`, checkpointed after every generation
  like a reported run. The reported results never read these folders, and `--fresh` keeps them.
- **Resume.** For the GA, ES, seeded ES and NEAT, Play continues the saved run of the selected method and seed where it
  stopped, at its own level and target; the difficulty and population settings apply only when a run starts. **New
  run** deletes that method and run's saved run and its app car and starts afresh with the current settings. If a
  saved run was recorded with other settings or a file of it is damaged, the training process ends with exit code 3
  and the panel asks for New run.
- **Imitation methods** (baseline and Keras Tuner) are not resumed in the application: Play starts them afresh.
- **App cars.** After every generation (evolutionary methods) the generation's best car, with its setup, or whenever
  the validation accuracy improves (imitation methods) the best network so far, with the default setup, is saved in
  `results/app_runs/champions/`, in the files of a final controller, so it can race in the race view (§10.3).

### 10.7 Network window

A separate window, opened from the side panel, shows a car's network live. It runs in its own process, can be
moved and resized, and closes with the application, so racing in the viewer is not slowed.

- **Diagram:** the inputs on the left, grouped as distance rays, car and racing line, each with its value in real
  units (metres, km/h, the share of the half road from its centre, the distance and angle from the racing line, the
  line's corner radius ahead); every hidden layer as a column headed "layer k · n neurons" (NEAT's hidden nodes in
  columns by depth); the 45 actions on the right in nine steering groups, gas at the top of each. Hidden units are
  shaded by their activation, blue positive and orange negative; a ReLU layer states how many of its units fire.
  The action with the highest score, the one the car takes, is enlarged and named, and the header names the
  network's shape, the car's place and its speed.
- **Connections:** the connections that carry the strongest signals at this moment (the source node's value times
  the weight) are drawn bold, at most 34 into hidden units and 8 into the chosen action. The full view, the default,
  also draws every other connection faintly, tinted by the signal it carries; the simplified view leaves them out.
  A "Full | Simple" button or the V key switches between them.
- **Live values:** while the shown car drives, the viewer sends the value of every node about ten times a second.
  A race that is set up shows each car's network on the start line before the start. In the race view a selector
  cycles through the cars.
- **Training timeline:** the viewer keeps the best network of every generation it trains or replays while it is
  open (nothing is saved, and a new run starts a new timeline). Under the diagram, the run's lowest map success share
  per generation is drawn against a dashed 90% line (for the imitation approaches, the validation accuracy), with
  level-ups and the mastery goal marked. Play (or Space) steps one generation every three seconds; the arrows (or the
  ← and → keys), a click or a drag choose any generation, whose best car then drives the current map inside the
  window so its network is shown live too; Live (or L or End) returns to the newest generation, whose best car the
  viewer's replay drives when it is among the displayed cars (always under *Best*).

## 11. Evaluation Methodology

### 11.1 Test procedure

Each final controller (eight methods, four runs) drives the four test circuits, Melbourne, Sakhir, Sepang and
Silverstone, in their racing direction as a three-lap time trial from a standing start. Three outcomes are recorded
per circuit: whether the car **finished** within the simulator's time limit of twice the reference lap time per lap,
whether it **succeeded** within the starting target (f = 1.3), and the time of every lap it completed. The
validation maps select each run's final controller (§9.5) and each method's best run, which the tables name, by the
rule of §9.5: of the runs whose final controller finishes all 20 validation maps within the time limit, the one with
the fastest mean lap relative to each map's expert lap over the maps it finished; if none does, the run with the most
finishes, ties going to the higher mean validation fitness; remaining ties go to the lower run number. Every track is
driven at its own widths (§4.2, §4.7).

**Pole laps.** A race's **flying lap** is its fastest complete lap after the first (lap 2 or 3), which starts at speed
like a qualifying lap. Each final controller's flying lap on a test circuit is compared with the pole lap set there in
qualifying (Table 9): a controller beats the pole lap when its flying lap is shorter. Lap times are measured to the
step, 0.05 s: a lap ends at the first step whose nearest centerline point is past the start line, so a flying lap
within 0.05 s of a pole lap does not resolve which is faster, and the tables print lap times to two decimals. Each
pole lap is the fastest pole lap set on the layout of the database: Melbourne's of 2019, the last season before the
circuit was rebuilt for 2022 without the Turn 9–10 chicane; Sakhir's and Silverstone's of 2020, faster than every pole
there since; and Sepang's of 2017, the last Formula 1 race there.

<div style="break-inside:avoid;page-break-inside:avoid">

*Table 9. Pole laps of the test circuits.*

| Circuit | Pole lap | Driver, season | Sources |
|---|---|---|---|
| Melbourne (Albert Park) | 1:20.486 | Lewis Hamilton, 2019 (also the lap record) | Formula1.com qualifying report |
| Sakhir (Bahrain) | 1:27.264 | Lewis Hamilton, 2020 | Formula1.com qualifying results |
| Sepang | 1:30.076 | Lewis Hamilton, 2017 | FIA pole position document |
| Silverstone | 1:24.303 | Lewis Hamilton, 2020 | Formula1.com qualifying results |

</div>

The comparison carries the car's simplification with it: the car has a Formula 1 car's top speed and two to three times
its grip but no engine power or drag limit, so it gains speed on straights and out of slow corners faster than a real
car (§14 items 19 and 30). The cars also receive the circuit's racing line (§5.2): a beaten pole lap is that of a car
that knows the circuit's racing line, as a driver who has learned the circuit does. The report states both with every
comparison.

**Robustness.** Each test circuit is also rebuilt with the same centerline at the constant widths 6,
10, 14 and 18 m, the first narrower than any road met in training. Each build is driven exactly as a test circuit,
with the expert's lap time on that build; a build whose road would reach another part of the circuit, or that the
expert cannot drive, is left out of its width.

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 372" width="100%" role="img" aria-label="Choosing the winners: validation picks each run's final controller and each method's best run; every final controller drives the test circuits; tables and race view" style="display:block;margin:10pt 0 6pt">
<defs><marker id="figwin-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker></defs>
<rect x="0" y="0" width="665" height="88" rx="3" fill="#f5f6f8"/>
<rect x="0" y="112" width="322" height="166" rx="3" fill="#f5f6f8"/>
<rect x="342" y="112" width="323" height="166" rx="3" fill="#f5f6f8"/>
<text x="10" y="16" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Each run</tspan><tspan font-size="9" fill="#5f6673"> · r0 seeded, r1–r3 unseeded; 4 runs per method, 8 methods</tspan></text>
<rect x="10" y="30" width="120" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="70" y="42.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Kept cars</text>
<text x="70" y="54.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">of the last generation</text>
<rect x="156" y="30" width="134" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="223" y="42.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">20 validation maps</text>
<text x="223" y="54.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">seeds 80–99 · validation</text>
<path d="M317,46 L326.92,30 L437.08,30 L447,46 L437.08,62 L326.92,62 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="382" y="43.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">all 20 finishes:</text>
<text x="382" y="55.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">fastest mean lap</text>
<rect x="515" y="30" width="140" height="32" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="585" y="42.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Final controller</text>
<text x="585" y="54.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">.keras + .npy or genome</text>
<path d="M130,46 L156,46" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<path d="M290,46 L317,46" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<path d="M447,46 L515,46" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<text x="156" y="79" font-family="DejaVu Sans" font-size="9" fill="#5f6673">time trial within the time limit (2 × the expert's laps)</text>
<path d="M585,62 L585,138" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<circle cx="585" cy="102" r="1.8" fill="#5f6673"/>
<path d="M585,102 L240,102 L240,138" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<text x="250" y="98" font-family="DejaVu Sans" font-size="9" fill="#5f6673">validation finishes of the 4 runs</text>
<text x="591" y="124" font-family="DejaVu Sans" font-size="9" fill="#5f6673">every final</text>
<text x="591" y="134.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">controller</text>
<text x="10" y="128" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Each method</tspan><tspan font-size="9" fill="#5f6673"> · its 4 runs</tspan></text>
<rect x="46" y="138" width="230" height="30" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<rect x="58" y="144" width="44" height="18" rx="2" fill="#ffffff" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="80" y="156.6" font-family="DejaVu Sans Mono" font-size="10" fill="#1d2330" text-anchor="middle">r0</text>
<rect x="112" y="144" width="44" height="18" rx="2" fill="#ffffff" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="134" y="156.6" font-family="DejaVu Sans Mono" font-size="10" fill="#1d2330" text-anchor="middle">r1</text>
<rect x="166" y="144" width="44" height="18" rx="2" fill="#ffffff" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="188" y="156.6" font-family="DejaVu Sans Mono" font-size="10" fill="#1d2330" text-anchor="middle">r2</text>
<rect x="220" y="144" width="44" height="18" rx="2" fill="#ffffff" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="242" y="156.6" font-family="DejaVu Sans Mono" font-size="10" fill="#1d2330" text-anchor="middle">r3</text>
<path d="M63,200 L75.4,180 L246.6,180 L259,200 L246.6,220 L75.4,220 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="161" y="197.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">fastest with all 20 finishes,</text>
<text x="161" y="209.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">else most, fitness, run 0</text>
<path d="M161,168 L161,180" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<rect x="106" y="236" width="110" height="26" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="161" y="252.8" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Best run</text>
<path d="M161,220 L161,236" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<text x="352" y="128" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Test</tspan><tspan font-size="9" fill="#5f6673"> · every final controller (§11.1)</tspan></text>
<rect x="408.5" y="138" width="190" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="503.5" y="150.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">4 test circuits</text>
<text x="503.5" y="162.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">racing direction · pole laps</text>
<rect x="352" y="188" width="140" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="422" y="200.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Real widths</text>
<text x="422" y="212.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">7.4 m and up, per point</text>
<rect x="512" y="188" width="143" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="583.5" y="200.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Robustness widths</text>
<text x="583.5" y="212.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">6, 10, 14, 18 m</text>
<path d="M503.5,170 L503.5,179 L422,179 L422,188" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<path d="M503.5,179 L583.5,179 L583.5,188" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<text x="352" y="243" font-family="DejaVu Sans" font-size="9" fill="#5f6673">each drive records</text>
<rect x="443.64" y="233" width="54.29" height="14" rx="7" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1"/>
<text x="470.79" y="243.24" font-family="DejaVu Sans" font-size="9" fill="#7a5410" text-anchor="middle">finished?</text>
<rect x="503.94" y="233" width="67.1" height="14" rx="7" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1"/>
<text x="537.49" y="243.24" font-family="DejaVu Sans" font-size="9" fill="#7a5410" text-anchor="middle">succeeded?</text>
<text x="352" y="261" font-family="DejaVu Sans" font-size="9" fill="#5f6673">finished: within 2 × the expert's lap time per lap</text>
<text x="352" y="272" font-family="DejaVu Sans" font-size="9" fill="#5f6673">succeeded: within the starting target, f = 1.3</text>
<rect x="20" y="296" width="282" height="48" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="161" y="311" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Race view (§10.3)</text>
<rect x="79.58" y="319" width="46.73" height="17" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="102.94" y="330.8" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">seeded</text>
<rect x="126.3" y="319" width="58.14" height="17" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="155.37" y="330.8" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">unseeded</text>
<rect x="184.44" y="319" width="33.47" height="17" rx="2" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="201.17" y="330.8" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">best</text>
<rect x="217.91" y="319" width="24.52" height="17" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="230.16" y="330.8" font-family="DejaVu Sans" font-size="9" fill="#1d2330" text-anchor="middle">all</text>
<text x="161" y="357" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">races on the test circuits; default: best</text>
<path d="M161,262 L161,296" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<text x="167" y="290" font-family="DejaVu Sans" font-size="9" fill="#5f6673">the 4 runs and the best run</text>
<rect x="393.5" y="296" width="220" height="34" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="503.5" y="310.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Report tables (§11.3)</text>
<text x="503.5" y="323" font-family="DejaVu Sans Mono" font-size="9" fill="#5f6673" text-anchor="middle">r0 · r1 · r2 · r3 (mean ± sd)</text>
<path d="M503.5,278 L503.5,296" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<text x="503.5" y="344" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">each method's best run named</text>
<path d="M216,249 L332,249 L332,313 L393.5,313" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
</svg>
<p><em>Figure 12. Choosing the winners: validation picks each run's final controller and each method's best run, and every final controller drives the test circuits, where its flying laps meet the pole laps.</em></p>
</div>

### 11.2 Measurements of the course comparison

- **Accuracy evaluation:** each controller's predicted classes are compared with the expert labels in the
  test dataset. For the evolutionary controllers, this measures agreement with an expert they never
  observed, which is reported as a result in its own right. A prediction counts only when it is the expert's class
  of the 45 (§6), so a neighbouring steering or throttle level counts as a miss.
- **Inference time:** *batch*, Keras `predict` on 10,000 test rows in a single batch, reported per sample
  and averaged over five runs, with a warm-up call identical to the timed calls; *single sample*, the
  median time of 1,000 calls of the Keras model on one row. Inference is timed after all parallel work has
  finished, with the same prediction settings for every controller (no XLA compilation, one batch per
  call).
- **Parameters:** the Keras parameter count; for NEAT, the number of enabled connections plus biases.
- **Training time:** baseline training per run; Keras Tuner search time (once per method) and training
  per run; DAgger and speed push per run; evolutionary run time per run; dataset collection once, as a
  shared cost. The run conditions recorded in `results/run_info.json` (§3.2) are reported with the
  results.

### 11.3 Tables

Results are reported for every run of each method and as their mean and sample standard deviation, written
`r0 · r1 · r2 · r3 (mean ± sd)`, with each method's best run named. They are led by completion and speed.

1. **Results.** Per method: test circuits finished and succeeded (per run and in total, of 16), mean and
   best lap time, mean three-lap race time, crashes, the target reduction reached in the speed push, and the final
   controllers' sensor length and steering sensitivity.
2. **Pole laps.** Per method and run: the flying lap on each test circuit, its difference to the pole
   lap in seconds and in percent, and the number of test circuits on which the pole lap was beaten; per method a row
   "fastest run", the fastest flying lap on each circuit over its four runs (not the best run chosen on validation).
   Lap times are given to two decimals (0.05 s resolution).
3. **Head-to-head speed.** Each method's best run (§11.1) on the test circuits that every one of these runs
   finished: its mean lap time there, and on how many of them it was the fastest. A run that is not its method's best
   run does not affect the table; when the best runs share no finished circuit, the table has no lap times (§14
   item 35).
4. **Training cost.** Per method and run: the generation, evaluations and time at which each level was cleared and the
   mastery goal reached, or "not reached"; how and at which generation the run ended: hardened, ended by time (an
   end request, §9.4) or stalled on a level (only when a safety stop is set, §9.4), and "speed limit" for a run whose
   speed push ended before hardening was added; the total evaluations and the time of each stage, hardening included;
   and the stalls the run broke out of (§9.4). For imitation methods, the Keras Tuner search time, the imitation
   training time, and the number and time of DAgger rounds with whether DAgger reached its goal are included.
5. **Imitation stages.** Test results of the imitation methods after initial training, after DAgger and
   after the speed push.
6. **Course comparison.** Test accuracy, parameters, batch inference time, single-sample latency and
   search and training time, as required by the course. Test accuracy is the share of rows of the expert-labelled
   test split for which the controller predicts the expert's class. Parameters are as in §11.2. Batch inference time
   is the Keras `predict` time per sample on 10,000 test rows, averaged over five runs; single-sample latency is the
   median of 1,000 calls on one row (§11.2). Search and training time is the training time of §11.2.
7. **Search cost, dataset and driving-selected controllers.** *Search cost:* per Keras Tuner method, the
   number of trials, the number of distinct configurations, the total epochs, the trials that reached the
   epoch limit and the search time (§8.2). *Dataset:* the rows of each split and the collection time (§7.3).
   *Driving-selected controllers:* per Keras Tuner method, the configuration chosen by validation driving
   (§8.3), its parameters, validation accuracy, test circuits completed (of 4), mean lap time and crashes.
8. **Robustness.** Per method and width of §11.1: the maps finished (in total and per run) and succeeded (all runs),
   and the mean lap time on finished maps; each width states how many maps it covers.
9. **Evolved networks.** Per GA and NEAT run: the final controller's structure (for the GA its layer sizes, for
   example 18 → 9 → 45; for NEAT its hidden nodes and enabled connections) and its parameters (§11.2).

### 11.4 Figures

- **Success over training.** For each method, the lowest map success share per generation, with
  level advancements, target reductions and hardening marked.
- **Lap time over training.** The median lap time of the successful cars per generation.
- **Pole laps.** For each test circuit, the fastest flying lap of each method as a horizontal bar, with the pole lap
  marked by a vertical line.
- **Setup over training.** For each method, the population's mean sensor length and mean steering sensitivity per
  generation, with level advancements and target reductions marked.
- **Networks.** One panel per method showing its best final controller. The 18 inputs are on the left:
  the nine distance rays, labelled by angle, the speed, the position across the road, the road width and the six
  racing-line inputs (distance, heading and the curvature 25, 50, 100 and 200 m ahead). Hidden nodes are placed in
  columns by depth, so that the structure evolved by NEAT appears as it is. The 45 actions, labelled by steering and
  throttle, are on the right. Connections are blue when positive and red when negative, with width and
  opacity proportional to the weight's magnitude. Titles include the controller's setup.
- **Network size over training.** For the GA and NEAT, the population's mean number of hidden units (GA) or hidden
  nodes (NEAT) per generation, one line per run, with level advancements and target reductions marked.
- **Step sizes.** The GA's mutation strength and the step size σ of the ES, the seeded ES, NEAT and the imitation
  methods' speed pushes across generations, one line per run in the method's color.
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
- **Parallel execution (§3.2):** the compiled simulator's default thread count equals the logical processors
  available to the process, `EC_RACING_THREADS` sets it and refuses anything but a positive integer, and
  `EC_RACING_CXX` selects the compiler; an evolution phase lowers every thread of its process to priority 19, threads
  started later inherit it, and it is never raised; a Keras job runs with 4 and 2 TensorFlow threads; a small search
  and the retraining of its choice, each run in a fresh process, produce the same weights as the same steps run in
  notebook order in one process; a run of a searched method waits for its search, and a baseline run trains its
  network, refines it and pushes it.
- **Runs 0 to 3 (§3.2):** an unseeded run sets no seed anywhere (its generators, Keras and DAgger are started without
  one); two starts of an unseeded run produce different populations; a stopped unseeded run resumes from its
  checkpoint on its own path; run 0 starts its generators and Keras from seed 0; the tables show
  `r0 · r1 · r2 · r3 (mean ± sd)`; the race control races the seeded, the unseeded, the best or all runs.
- **Fleet (§3.2):** the repository's plan runs the whole experiment once, and a plan that cannot work is refused; files
  reach a server and return intact; nothing starts until every machine is ready; `stop`, `end` and `archive` act on
  every machine; `collect` never replaces a different or unfinished run, and `finish` waits for every run, also one
  still hardening.
- **Stall boost (§9.4):** the boost switches on after 100 generations without a mastered target and off at the next
  mastered target; while it is on, only the output layer's mutation is three times as strong (GA, ES and NEAT), the
  steering sensitivity mutates with probability 0.5 and standard deviation 0.2, and the sensor length is unchanged; a
  resumed run 0 with a stall replays exactly.
- **GA structure (§9.6):** generation 0 has no hidden layers; adding a unit, or a layer behind a hidden layer,
  leaves every output unchanged; removing a unit removes its weights, and removing a layer joins its neighbours by the
  product of their weights; crossover gives the child the shape parent's architecture and mixes only the values both
  parents have, and between equal architectures it is uniform crossover; the compiled simulator's results equal
  numpy's for populations of mixed architectures, including networks without hidden layers; checkpoints, subsets and
  saved controllers keep each car's architecture; `--archive NAME --runs` moves only the named runs.
- **Formula 1 car and real circuits (§4.7, §5.1):** the grip equals a(v) at 0, 69.4 and 105 m/s and never exceeds
  10 g; no car exceeds 105 m/s; each real circuit imports with the data's widths at every point and within 1% of its
  official length; the position input is −1 and +1 at the two edges of a road whose two widths differ; the test
  circuits are those drawn by `default_rng(0)` and appear in no training draw, validation map or training split of the
  dataset; Extra hard draws only the 38 training layouts; the expert completes every generated level, the validation
  setting and every real circuit in both directions; a race records each lap's time and its flying lap is its fastest
  lap after the first; the compiled simulator equals numpy on real circuits; rays reach 1,000 m.
- **Actions (§6):** 45 classes, `class = 5 × steer_index + throttle_index`, decode to the nine steering
  and five throttle levels and back; the expert's steering and throttle round to the nearest levels, its throttle is
  its speed error over its band, and it never accelerates while its quantized steering is at full lock; both
  simulators decode every class identically; a speed-push generation passes at 50% and the push ends after 1,000
  generations without a mastered target; every network has 45 outputs (with 18 inputs: baseline 1,069 parameters, a
  new GA network 855, a new NEAT genome 810 connections).
- **Racing line, expert and final pick (§4.8, §5.2, §7.1, §9.5):** on a stadium the line keeps to the outside along
  the straights and to the inside at the apexes, at the margin from the edge (on a pure circle the line keeps to the
  inner bound, as §14 item 29 explains); every point of a line lies within its bounds to 0.1 m on every level and
  every real layout; the line is the same when computed again, in another process and with any number of BLAS
  threads, the one-thread limit reaches numpy's BLAS, solves in several threads at once give the lines of lone solves,
  a cached line equals a fresh one, and a day-old line file of another identity is removed while the other
  direction's line is kept; on each of the 23 real circuits, with the published lines' clearance of 0.65 m, the
  quasi-steady-state lap on the line is within 2% of the lap on the published line (measured within 0.6%); the
  quasi-steady-state lap of a circle equals its analytic value; the six line inputs equal their definitions on
  constructed cases (left and right of the line, heading either side, the clip at ±5) and both simulators compute
  them identically; on the line the expert steers by the line's previewed curvature, off it, it steers back toward
  it, and its line-tracking settings are recorded; its target speed is the profile's and never exceeds the top speed;
  at the recorded grip share it drives three laps of every real layout alike in both simulators, and every validation
  and dataset map builds; a crashed car is put back on the racing line; the final pick takes the fastest of the cars
  that finish all 20 validation maps over a faster car with 19, and the most finishes, then the highest mean
  validation fitness, when no car finishes all 20; a controller, dataset, checkpoint or results folder made for
  another number of inputs is refused with a note, and a refined baseline of another number of inputs does not start
  the application's seeded ES; the race view draws the line.
- **Keras Tuner accounting:** on a small search, the recorded epochs per trial equal the epochs each
  trial actually trained, and Hyperband continuation trials count only the epochs they added.
- **Measurement:** the inference-time warm-up call matches the timed call.
- **Experiment runner:** on a short configuration, the runner executes every notebook in order and
  writes the run record.
- **Pilots (§12.2):** a pilot always starts over in its own folder under `results/pilots/` and summarizes the levels
  it reached; the seeded pilot starts from the DAgger-refined baseline and is skipped without it; the stall-boost
  pilot continues a finished run with and without the boost, its plain copy makes exactly the draws of a run without
  the boost, it names any field its source checkpoint lacks, and it refuses a source of another action set before it
  changes anything.
- **Benchmarks (§9.10):** the generation-time benchmark divides the tasks as training does, tries worker counts up to
  the logical processors and prefers fewer workers on a near tie; on the compiled simulator it times the newest
  trained population, or expert cars without one; the Keras benchmark times an epoch on toy data.
- **Application:** the training process starts worker processes; the final report of a finished
  training process is displayed; stopping training or a race returns the view to a state in which
  settings can be changed; the grip selector changes only display simulations; a saved application run resumes after
  Play, New run starts afresh, an unusable saved run is named, and app cars race in their own group, switched off by
  default.
- **Track generator:** both directions occur; every accepted map respects its level's tightest corner
  and separation limits; the expert completes it; a seed always yields the same map.
- **Time trial:** success, laps in time and fitness on constructed cases.
- **Survival:** successes are always kept, the cap and the elitism floor hold, and the frequency of
  chance survival matches the share of laps in time.
- **Level tracker:** advancement, mastery goal, target reduction, the end of the speed push, a run that
  keeps training with the safety stop off and the end of a run with a number set as its patience. Hardening: the end
  of the speed push starts it at the last mastered target; it needs 90% three generations in a row and never
  tightens; a run hardening before its goal continues its levels and ends hardened at the goal.
- **End request (§9.4):** a new run under an end request ends after its first generation with its final
  controller, picked from the kept cars; a run resumed under one picks its final controller without training; viewer
  and pilot runs ignore the request.
- **NEAT:** innovation numbers, crossover alignment, node splitting, cycle prevention, speciation,
  equality of the batched forward pass with a direct per-genome computation, and equality of the
  Keras export with the genome.
- **DAgger:** labels equal the expert's actions in the recorded states; rows accumulate across rounds;
  the goal and stall rules.
- **Checkpoints:** a run interrupted and resumed produces exactly the results of an uninterrupted run;
  a settings mismatch is refused; a method's own settings (the GA's structure settings and NEAT's settings) are
  recorded and checked only in that method's runs, so a record without them is refused for that method alone; the
  final pick's setting is recorded but not compared. A run whose summary has no hardening entry and records an end at
  its speed limit or by an end request resumes into hardening and keeps that summary as
  `summary_before_hardening.json`, also when it is interrupted before its first hardening generation is saved, and
  has no `summary.json` meanwhile; such a run hardens also without its summary, and an unreadable checkpoint leaves
  the summary in place; an end request ends hardening, at once when the run resumes under it, and the final
  controller is selected; a run whose summary has a hardening entry is not hardened again.
- **Progress display:** status and milestone output for a short run.
- **Network figure:** the figure is produced for dense and NEAT controllers.
- **Network window:** the highlighted action is the one the car takes, for dense networks with and
  without hidden layers and for NEAT; node values equal the forward pass; the bold connections carry the strongest
  signals; inputs read in real units; both views render at any window size; the viewer sends values only while the
  window is open and the car drives, at most ten times a second, and the start line before the start; the timeline
  follows the newest generation until a generation is chosen, plays one generation every three seconds, starts again
  for a new run, and lets a past generation's car drive the current map in the window.
- **Head-to-head table:** each method's best run on the circuits all best runs finished; a run that
  finished nothing and is not its method's best run does not empty it; without a shared circuit the table has no lap
  times.
- **Pole laps (§11.1, §10.3):** the pole laps equal Table 9; the pole-lap table compares each run's flying lap with
  the pole lap, and the pole-lap figure is drawn; in the race view the pole-lap car laps in exactly the pole time, is
  placed on the leaderboard by distance and races only on a test circuit, where the leaderboard gives each flying
  lap's gap to the pole.
- **Robustness (§11.1):** each test circuit is rebuilt at the four widths with its centerline, without a corner limit,
  and timed by the expert; a width the circuit cannot take or the expert cannot drive is left out; the table counts
  the finishes per method and width, in total and per run.
- **Imitation network:** the seeded run 0 learns the toy data above chance, so the check gives the same result every
  time.
- **Car setup:** a ray stops at its car's length and a car of the starting length sees ray inputs from 0 to 1;
  steering sensitivity scales the steering levels within the grip limit; initial setups, inheritance and
  mutation follow §9.11; setups survive subsets, parallel evaluation, checkpoints and saved controllers.
- **Seeded ES:** the first initial individual equals the DAgger-refined network exactly, and the others
  differ from it.
- **Re-simulation fidelity:** a subset of a population reproduces its cars' fitness and laps exactly, and the replay
  file of a reported run holds exactly the generation's population, setups and fitness.

### 12.2 Pilot runs

Pilot runs are short training runs, outside the reported results, that check before a reported run that the cars can
learn with the settings of this document. They precede a reported run after a change to the car, the maps, the
inputs, the actions or the expert. A change to one method's settings can instead be checked on that method's kept
cars, as for NEAT's weight replacement below; a change to the evaluation alone, or runs that continue from their
checkpoints, need no pilot.

- **Expert grip share** (`scripts/expert_grip.py`): the highest grip share, from 1.00 down in steps of 0.01, at which
  the expert drives three laps of every real layout and every validation and dataset map builds (§7.1). It is
  measured first, since the expert sets every target, and again after any change to the grip, the racing line, the
  road grids, the expert or the levels; the value, 0.80, is recorded in the settings.
- **Level pilots** (`scripts/pilots.py`): the GA, the ES, NEAT and the seeded ES each train from Easy with seed 0,
  300 generations by default. Each pilot starts over in its own folder under `results/pilots/` and adds a row to a
  summary: the generations, the level reached with the generation of each advancement and of the mastery goal, the
  mean lowest map success share of the last 20 generations, the target reduction reached, the population's final mean
  sensor length and steering sensitivity, for the GA its network size, and the time taken. The seeded pilot starts
  from the DAgger-refined baseline of run 0 and is skipped without it. If a level is not reached, the survival and
  step-size settings are reviewed before the reported run starts.
- **Stall-boost pilot** (`scripts/boost_pilot.py`): a finished run whose speed push stalled continues from its final
  population and random state twice, once with the stall boost (§9.4) and once with the boost set to its off values,
  which makes exactly the draws of a run without it, so the two copies differ only once the boost acts. The boost is
  kept if the boosted copy masters a shorter target than the plain one. The source run must have the current inputs
  and actions; the pilot refuses any other before it changes anything.

The following measurements support current settings:

- **Stall boost.** An ES run whose speed push had stalled at a target 27.6% shorter than the starting target was
  continued with a patience of 600 generations: the boosted copy mastered a target 42.0% shorter, the plain copy one
  31.9% shorter, so the boost is kept. The source run had 15 actions on a slower car and smaller generated maps; the
  pilot has not been repeated with the current car and actions (§14 item 15).
- **NEAT's weight replacement.** On the kept genomes of two NEAT runs, replacing 5% of the weights halved the share
  of children that finish (75% to 38%, 64% to 33%), while replacing 0.2% kept it close to the parents' (74% and 61%);
  the replacement rate is therefore 0.002 (§9.9).

## 13. Default Parameters

*Table 10. Default parameters, as defined in `racing/config.py` (compiler and build in `racing/fastsim.py`, pole laps
in `racing/circuits.py`).*

| Parameter | Value |
|---|---|
| Time step | 0.05 s |
| Grip limit a(v) (friction circle) | min(49.05 + 0.01017 × v², 98.1) m/s²: 5 g at low speed, 10 g from 250 km/h |
| Top speed | 105 m/s (378 km/h) |
| Speed input scaling | 105 m/s (the top speed) |
| Sensors | 9 rays from −90° to +90°; length 150 m at the start, 30–1,000 m when evolved; inputs scaled by 150 m |
| Network inputs | 18: rays, speed, position across the road (−1 left edge, +1 right edge, by the local widths), local road width / 20 m; distance from the racing line / mean half width (+ right), heading relative to the line / π (+ right), the line's curvature 25, 50, 100 and 200 m ahead × 50 m, clipped to ±5 (+ right turns) |
| Racing line | Minimum curvature within 1.5 m of each edge; nodes every 15 m; three re-linearizations; node bounds tightened where the line leaves its bounds by more than 0.1 m, at most six rounds; chord-length cubic spline; curvature smoothed by a 10 m Gaussian; real layouts cached in `results/racing_lines/`, a folder other than the grids' |
| Expert | Steering along the racing line: path curvature = the line's curvature v × 0.05 s ahead (`EXPERT_PREVIEW_TIME`) − 8 e / L² (`EXPERT_CROSS_TRACK_GAIN`) − 8 ψ / L (`EXPERT_HEADING_GAIN`), e and ψ the distance (+ left) and heading relative to the line, L = 6 m + 0.2 s × v (`EXPERT_LOOKAHEAD_TIME`); steering = v² × curvature / a(v), clipped to ±1 and quantized; target speed from a quasi-steady-state profile of the next round(200 m / spacing) points of the line at grip share 0.80 (`EXPERT_GRIP_MARGIN`, measured by `scripts/expert_grip.py`: the highest share in steps of 0.01 at which it drives three laps of every real layout and builds every validation and dataset map; measured again after any change to the grip, the line, the grids, the expert or the levels); throttle = speed error / max(0.5 m/s, a(v) × dt), no gas while the quantized steering is at full lock; all in the settings record |
| Road width | Generated: Table 1, a base width from `default_rng([seed, level, 13])` varied by up to ±20% along the lap (the mean of three sinusoids), clipped to the level's range; real circuits: the data's left and right widths at every point |
| Sphere tracing | Minimum step 0.5 m; at most ⌈48 × L / 50⌉ iterations for a ray of length L |
| Actions | 45: steering −1 to +1 in steps of 0.25 (9 levels) × throttle −1, −0.5, 0, +0.5, +1 (5 levels); class = 5 × steering index + throttle index |
| Steering sensitivity | 1, or ⅓–3 when evolved |
| Car setup evolution | Sensor length 150 m at the start for every car; steering sensitivity log-uniform for random-start populations, 1 for trained-start populations; per value mutation probability 0.2, Gaussian with standard deviation 0.1 on the natural-log scale, clipped |
| Laps per run | 3 (training, evaluation and application races) |
| Time limit | Training: the target time; evaluation and application: twice the expert lap time per lap |
| No-progress timeout | 5 s |
| Curriculum maps | Table 1: levels 1–3 generated, direction random per map; Extra hard: the 19 training circuits in either direction; at most 8 built tracks per process; real-circuit grids built once per machine in `results/circuit_grids/` and memory-mapped |
| Real circuits | TUM racetrack database (LGPL-3.0), layouts of about 2019; training: 14 Formula 1 and 5 other circuits; test: Melbourne, Sakhir, Sepang, Silverstone, drawn by `default_rng(0)` |
| Maps per generation | 3: freshly generated on levels 1–3; three of the 38 real layouts on Extra hard |
| Starting target factor f | 1.3 × three expert laps |
| Success | All three laps within the target time |
| Final controller | Of the kept cars that finish all 20 validation maps within the time limit (`FINAL_MIN_SUCCESSES`), the fastest mean lap relative to each map's expert lap; if none, the most finishes, then the highest mean validation fitness |
| Best run of a method | By the same rule over its four final controllers, ties to the lower run number |
| Survival | Successes kept; others kept with probability equal to their share of laps in time; at most 90% of the population; at least the 50 fittest |
| Level advancement and mastery goal | Lowest map success share ≥ 90% in three consecutive generations |
| Speed push | Target reduced by 2% per mastered target; a generation passes at a lowest map success share ≥ 50%; ends after 1,000 generations without mastering the current target |
| Hardening | At the tightest mastered target, never tightened; ends at a lowest map success share ≥ 90% in three consecutive generations, or by an end request |
| Stall boost | After 100 generations without a mastered target: output-layer mutation ×3; steering sensitivity mutation probability 0.5, standard deviation 0.2; until the next mastered target |
| Robustness table | Test circuits rebuilt with the same centerline at 6, 10, 14 and 18 m |
| Pole laps | Table 9; compared with each final controller's flying lap (lap 2 or 3), measured to 0.05 s |
| Safety stop | off (`SAFETY_PATIENCE` None); a number n ends a run after n generations without a new best three-generation mean of the lowest map success share |
| GA | N = 2,000; tournament 5; crossover p = 0.5 on shared values; mutation p = 0.1; s_max = 0.1; s_min = s_max / 100; no hidden layers at the start; per child add unit 0.03, remove unit 0.03, add layer 0.01, remove layer 0.01; no size limit; ReLU hidden units |
| ES, seeded ES and speed push | N = 1,100; children from the 100 fittest; σ₀ = 0.1; factor 0.82; σ in [1e-4, 1.0]; reliability phase at 90% |
| NEAT | N = 2,000; crossover 0.75; weight perturbation 0.8 (σ₀ = 0.5, two-phase rule as ES); weight replacement 0.002; add connection 0.05; add node 0.03; toggle 0.01; δ = E/N + D/N + 0.4 W̄; threshold from 3.0 towards 20 species; stagnation 20 generations; tanh hidden nodes |
| DAgger | 20 maps per round; at most 30 epochs per round, patience 5, 10% hold-out; goal 18 of 20 checks in three consecutive rounds; stops after 10 rounds without a new best check |
| Keras training | Adam 1e-3; batch 256; at most 150 epochs; patience 5; `steps_per_execution` 64, set by benchmark; `jit_compile` off (XLA failed on the desktop, §8.1) |
| Keras Tuner | 30 trials (random search, Bayesian optimization with 10 initial random trials); Hyperband with `max_epochs` 150, factor 3, one iteration |
| Model selection | Smallest network within one percentage point of the best validation accuracy |
| Replication | Run 0 (seed 0) and runs 1–3 (no seed) for every method; Keras Tuner searches run once (seed 0) |
| Map seeds | Validation 80–99 (validation setting); dataset 1,000,000–1,999,999 and the 38 real layouts; training from 2,000,000; display 1000–1019 |
| Threads (compiled simulator) | Every logical processor available to the process; `EC_RACING_THREADS` sets a number |
| Worker processes (numpy simulator, DAgger driving, dataset, evaluation) | `EC_RACING_JOBS`, default 6 |
| Keras threads | 4 TensorFlow threads within and 2 between operations, in every Keras job |
| Priority of an evolution phase | nice 19 (lowest) from its start, so Keras jobs come first |
| Compiler | `EC_RACING_CXX`, otherwise `g++` beside the Python interpreter, otherwise `g++` on the search path |
| Simulator | `EC_RACING_SIM`: `auto` (compiled when it builds, default) or `numpy` |
| Compiled simulator build | g++, C++17: `-O3 -march=native -ffp-contract=off -fno-builtin-sin -fno-builtin-cos -std=c++17 -shared -fPIC -pthread` |
| Compiled simulator library | `racing/_fastsim-<id>.so`; `<id>`: first 16 hexadecimal digits of the SHA-256 of `fastsim.cpp`, the flags and the processor's flags |
| Application playback | 1× equals twice real time |
| Application grip | Normal a(v); Unlimited 10 × a(v) (display only) |

The track, vehicle and expert-driver values are calibrated so that the expert completes every curriculum level
reliably (§4.2, §7.1).

## 14. Risks and Open Issues

1. **Course alignment.** Confirmation is needed from the course instructor that the project qualifies
   as Project 7, and that driving performance is acceptable as an evaluation metric alongside test
   accuracy.
2. **Unreachable levels.** The 90% requirement on unseen maps is demanding, and the targets come from an expert that
   drives the racing line at 80% of the grip (§7.1), so the methods that start from random networks (the GA, the ES
   and NEAT) may need long to reach Extra hard, and a method may not master every level. The pilot runs test this
   before a reported run (§12.2). With the safety stop off, a run that cannot progress trains on until an end request
   ends it; its result is reported, not hidden. In the last complete experiment, whose runs ended before hardening was
   added, 8 of the 12 runs of these three methods had not reached the mastery goal when the end request came
   (item 31).
3. **Run time.** Runs may take days: the speed push ends only 1,000 generations after its last mastered target, and
   hardening adds more (§9.4). The checkpoints and the progress display make this manageable, and the reported
   training costs include the full time.
4. **NEAT cost.** The batched forward pass costs more per step than a fixed dense network when genomes
   grow large. If its cost becomes dominant, the number of hidden nodes per genome can be capped.
5. **DAgger data growth.** The aggregated dataset grows with every round. Twenty rounds add on the order
   of a million rows, which fits in memory but lengthens training per round.
6. **Generator acceptance.** Harder settings reject more candidates, and a seed has at most 200 attempts
   (`MAX_TRACK_ATTEMPTS`). A slow test checks that 50 seeds of every generated level and of the validation setting
   are accepted and completed by the expert (§12.1).
7. **Seeded ES near the expert.** Evolution started from an imitation-trained network may remain close
   to the expert's driving style if no nearby weights perform better. Any such outcome is reported as
   a result.
8. **Hyperband budget.** Budget matching for Hyperband is approximate, because its bracket schedule
   determines the trial count. Actual trials, epochs and time are reported for every method.
9. **Action resolution.** Nine steering levels may cause oscillatory driving. Each level is a fixed fraction of the
   grip a(v), so the turning radius it produces grows with speed, more slowly than its square as a(v) rises with
   speed, and steering is coarsest at low speed. This is checked by requiring the quantized expert to complete every
   real layout reliably.
10. **Replication scope.** Each Keras Tuner search runs once, so the variation reported for the
   searched methods covers retraining of the selected architecture but not the search itself. The
   evolutionary methods repeat the whole run for each of the four runs.
11. **Sensor cost.** Long sensors cost tracing time: sensors start at 150 m and may reach 1,000 m, whose rays take up
    to 960 sphere-tracing iterations, so populations that evolve long sensors make generations slower. This is
    accepted and included in the reported training cost.
12. **Coarse steering.** Populations may settle at a steering sensitivity above 1, where steering saturates and loses
    smoothness. This is a result to be reported, not a defect.
13. **Exactness of the compiled simulator.** Equality with numpy holds on each machine separately: machines may differ
    in the last bit of a mathematical function, as the numpy simulator's results already may, and a numpy release that
    changed its `tanh` kernel or the summation order of `einsum` or `bincount` would break equality. The check at load
    (§5.7) enforces the equality on each machine: on a mismatch, the run continues on the numpy simulator after one
    warning. The equivalence tests (§12.1) still run on each machine.
14. **Threads beyond physical cores.** Hyperthreads and efficiency cores add less than a full core each, so a run's
    generation time falls more slowly beyond the performance cores; runs side by side keep every processor busy
    (§3.2). A machine without a C++17 compiler runs the numpy simulator, with identical results.
15. **Stall boost.** A boost that is too strong can scatter a population that was close to the next target. The
    stall-boost pilot (§12.2) kept the boost: from a target 27.6% shorter, the boosted copy mastered one 42.0% shorter
    and the plain copy one 31.9% shorter, on a run with 15 actions on a slower car; the pilot has not been repeated
    with the current car and actions. In the last complete experiment the boost broke one stall in the 32 runs
    (Seeded ES run 0; the training-cost table, §11.3).
16. **GA network growth.** Without a size limit, networks can grow wherever extra units do not hurt, which slows
    generations. The run log records the population's network size every generation (§9.1), so growth is visible while
    a run trains; it is reported, not capped.
17. **GA crossover across architectures.** Units of two GA networks are matched by position, not by history as in
    NEAT, so crossover between different architectures mixes only the values both parents have, and the child keeps
    the shape parent's structure. Crossover therefore rarely combines the structures of two lineages; the structure
    changes by mutation.
18. **Accuracy of the track data.** The centre lines come from OpenStreetMap and are smoothed, and the widths were
    extracted from satellite images; the accuracy varies between circuits. The lap lengths match the official ones
    within 0.2% on the test circuits, but corners and widths may differ in detail from the real circuits, which
    affects the comparison with pole laps.
19. **Simplified car.** With no engine power or drag, the car accelerates at full grip up to its top speed: on
    straights and out of slow corners it gains speed faster than a real Formula 1 car, whose acceleration falls with
    speed. A flying lap that beats a pole lap therefore partly reflects the simplified car, and the report says so with
    every comparison (§11.1). The car also has two to three times a Formula 1 car's grip (item 30).
20. **Small test set.** Four test circuits and four runs give 16 races per method, so differences between methods
    carry wide uncertainty; the per-run values are reported.
21. **Both directions in training.** The training circuits are driven in both directions for variety, the test
    circuits only in their racing direction. Training in the racing direction alone would match the test more closely
    but halve the variety; the choice can be reversed without other changes.
22. **Memory.** Real circuits' grids are built once per machine (about 1 GB at the peak, 570 MB on disk) and shared;
    the in-process cost is the generated maps, largest in the final selection, which holds all 20 validation maps
    (§9.10). The `at_once` bounds of the machine plan (§3.2) rest on peaks measured with populations half the current
    size, so a machine running that many jobs at once may come closer to its memory than the plan assumes.
23. **A low bar in the speed push.** A speed-push generation passes at 50% (§9.4), so the target can tighten while
    half of the population misses it. The survival rule keeps every success and the 50 fittest cars, so the population
    is not lost; the pilots (§12.2) check that the push advances, and hardening (§9.4) selects for finishing at the
    tightest mastered target.
24. **Imitation over 45 classes.** Forty-five classes split the expert's data finely, so each class has fewer
    examples, and test accuracy counts a neighbouring steering or throttle level as a miss (§11.2).
25. **Still discrete.** Forty-five choices every 0.05 s are not continuous control; the ideal laps of Table 3 assume
    smooth control, so they remain a bound the controllers approach rather than reach.
26. **The line is help.** The cars receive the racing line of every track, the test circuits included, so pole
    comparisons are of cars that know the circuit's racing line, as a driver who has learned the circuit does; the
    report says so with every comparison (§11.1).
27. **Imitating an expert near the grip limit.** The expert drives at 80% of the grip (§7.1), so its imitation and the
    DAgger checks are hard to pass with 45 discrete actions. In the last complete experiment DAgger reached its goal
    in 4 of the 16 imitation runs (the training-cost table, §11.3).
28. **A pick by pace.** The fastest car that finishes all 20 validation maps can still crash on a circuit it never
    saw; the test reports every crash.
29. **A computed line.** The minimum-curvature line is not the time-optimal line: before a slow corner at the end of
    a long straight the fastest line differs a little. Its curvature is linearized around the previous solution, as
    the published lines' optimizer does, which favours the inside of long constant-radius bends: on a stadium of 60 m
    bends the line's apex curvature is about 0.018 1/m against the centre line's 0.0167, and on a pure circle it keeps
    to the inside. On the real circuits its laps match the published lines' (§4.8). It comes from the database's centre
    line and widths, so their errors (item 18) carry into it, and its clearance of 1.5 m makes it somewhat slower than
    the published line. Its solve adds 0.06 s (Easy) to 0.78 s (the validation setting) to the building of a generated
    map (§5.7).
30. **Grip above a Formula 1 car's.** The car has 5 g at low speed and 10 g from 250 km/h, against about 1.5–2 g in
    slow corners and 5–6 g in fast ones for a Formula 1 car. The ideal laps on the test circuits are 54.31–59.29 s
    against pole laps of 80.5–90.1 s (Table 3), so the comparison with pole laps measures the controllers on a much
    faster car, and the report says so with every comparison (§11.1).
31. **Runs ended by an end request.** With the safety stop off (§9.4), a run that cannot master a level trains on
    until an end request (§9.4) ends every run wherever it is, each with its final controller; the stop remains
    available as `SAFETY_PATIENCE`, a number of generations. Runs still on a level are then evaluated beside runs deep
    in their speed push, and a run's final controller can finish few validation maps. In the last complete experiment
    the end request found 8 runs before the mastery goal: ES run 0 and NEAT run 3 on Hard, and ES run 1, GA runs 2
    and 3 and NEAT runs 0–2 on Extra hard. The training-cost table (§11.3) records how each run ended and at which
    generation, and the results are read with it.
32. **NEAT's mutation on large genomes.** A genome starts with 810 connections, and replacing 5% of its weights halved
    the share of children that finish (§9.9, §12.2), so the replacement rate is 0.002, about 1.6 replaced weights per
    child, and the perturbation is the main change. The perturbation acts only while the two-phase rule keeps σ above
    its floor; the log records σ and the children's success for each generation. In the last complete experiment no
    NEAT run reached the mastery goal.
33. **Pole laps from other seasons.** Each pole lap is the fastest on the database's layout, so the four come from
    three seasons (2017, 2019 and 2020), set by cars of different regulations. A later layout can be faster still:
    Melbourne's poles since 2022 (1:15.096 in 2025) were set on the rebuilt circuit and are not compared. Pole laps
    are set in qualifying, in the conditions of the day.
34. **Runs not hardened by an end request.** Hardening ends only when at least 90% of the population finishes every
    map at the tightest mastered target in three consecutive generations, or by an end request (§9.4), and how long it
    takes is not known. In the last complete experiment, while their last mastered target was current, the 24 runs
    that reached the mastery goal had at least 90% of the population finishing every map in 0–67% of their
    generations (median 8%; 0% for 7 runs), under the 50% rule of the speed push, and none of the 8 runs still on a
    level had three generations in a row at 90% on its level, so those 8 are likely to end by an end request.
    A run that is not hardened is picked from its last population by the rule of all 20 finishes, and only 1 of the 32
    final controllers of that experiment finished all 20 validation maps, so the pick of such a run may fall back on
    the most finishes. Requiring all 20 finishes also favours reliable cars over fast ones, so lap times can rise. The
    training-cost table (§11.3) records how each run ended, and the results are read with it.
35. **An empty head-to-head.** The head-to-head table (§11.3) compares only the test circuits that every method's
    best run finished. If one best run finishes none, or the best runs share no finished circuit, the table has no lap
    times, and the per-run lap times of the results table (§11.3) are read instead.
36. **Timings depend on the machine.** The GA, ES and NEAT runs train on a server and the imitation methods and the
    seeded ES on the desktop (§3.2), each beside other runs, so the training times of methods on different machines
    are not directly comparable; the training-cost table also gives evaluations, which do not depend on the machine.
    Every job runs on the CPU with the GPU hidden (§15), so all timings, the course comparison's included, are CPU
    timings.

## 15. Software Environment

- **Environment:** Python 3.13 with the packages pinned in `requirements.txt` (numpy, scipy, pandas,
  matplotlib, TensorFlow 2.21, Keras 3.13, keras-tuner, scikit-learn (which keras-tuner's Bayesian optimization
  needs), joblib, pygame, nbclient, nbformat, ipykernel,
  jupyter_client and pytest). A test verifies that the installed versions match the pinned ones.
- **Compiler:** g++ with C++17 support for the compiled simulator (`sudo apt install build-essential` on Ubuntu and
  WSL, or conda-forge's `gxx` in the run environment, which `fleet.py setup` installs on a server without g++);
  without it, the numpy simulator runs.
- **Platform:** Linux. On the Windows desktop used for the reported run, the Windows Subsystem for Linux
  (WSL 2) with the project in the Linux file system.
- **Execution:** CPU only. Every Keras job hides the GPU (`tf.config.set_visible_devices([], 'GPU')`), and notebook
  kernels start with `CUDA_VISIBLE_DEVICES=-1`. Each run's summary records the machine it finished on, and
  `results/run_info.json` the machine of the runner (§3.2).
- **Track data:** the racetrack database of the Technical University of Munich (`TUMFTM/racetrack-database`,
  LGPL-3.0) is vendored in `data/racetracks/` with its licence file and credited in the README and the report: the
  centre lines and widths of its circuits, and the published racing lines of the 23 training and test circuits, used
  only to check the racing line (§4.8).
- **BLAS threads:** the racing line's solve runs with one BLAS thread through `threadpoolctl`, which the pinned
  scikit-learn installs, so that the line does not depend on the process (§4.8).
- **Version control:** a private GitHub repository; the `results/` directory is excluded.
