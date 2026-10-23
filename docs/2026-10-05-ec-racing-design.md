# EC Racing: Design Specification

**Evolutionary and Gradient-Based Search for Neural Network Driving Controllers**\
ECDL Term Project · Fall 2026 · Version 3.9 · 23 October 2026

## Summary

This document specifies EC Racing, a top-down two-dimensional racing simulator. It is used to compare
methods for finding small neural network driving controllers. Eight controllers drive a car with the grip and top
speed of a Formula 1 car, on generated circuits and on real circuits, and are measured against the pole laps of real
Formula 1 drivers on circuits they never trained on.

- **Imitation learning (four controllers).** These are trained by supervised imitation of a scripted
  expert driver. One is a hand-designed baseline. In the other three, Keras Tuner selects the
  architecture using random search, Hyperband or Bayesian optimization. Each is then refined by dataset
  aggregation (DAgger) and pushed for speed by an evolution strategy grown from it.
- **Neuroevolution (three controllers).** A (μ+λ) evolution strategy (ES) optimizes the weights of a fixed
  network directly from driving performance. A genetic algorithm (GA) and NEAT evolve the network structure
  together with its weights, starting from networks whose inputs are wired directly to the outputs: the GA adds and
  removes hidden units and whole layers, NEAT single nodes and connections.
- **Hybrid (one controller).** A seeded ES starts from the DAgger-refined baseline network, which has
  learned from the expert, and then optimizes the weights from driving performance alone.

Every method trains until it masters the curriculum: on each map of a generation, at
least 90% of its population must complete three laps within a target time, in three consecutive
generations, on each of four levels. In the speed push that follows, the target time is shortened in steps of 2%
whenever half of the population meets it, until the population can no longer keep up. Road widths change along the
lap, as on real circuits, and the networks see their position across the road and the road's local width, so that
they drive any width. Every method is run four times: run 0 with seed 0, which replays exactly, and runs 1, 2 and 3 with no seed, whose
random choices come from the operating system; results report every run, their mean and standard deviation, and each
method's best run. A speed push that stalls mutates the decisions about speed and turning harder until it beats its
target again. All final controllers are evaluated on four held-out Formula 1 circuits, where each one's best
flying lap is compared with the circuit's pole lap, and for completion, lap time, classification accuracy, network
size, inference time and training cost. A
visualization application shows training in progress and races between the final controllers. A single
command executes the complete experiment; it saves checkpoints, resumes after an interruption and reports
its progress live.

Version 3.9 gives the cars finer controls and pushes them further (§6, §7.1, §9.4). The pole check of the version 3.8
run showed that the gap to the pole laps lies in the driving, not in the car: on a racing line within each test
circuit's road, a lap of the car at its grip and top speed with perfectly smooth control is 9–14% faster than pole,
while the best final controllers were 20–42% slower (Table 0). The controllers that pushed furthest were the fastest.
Version 3.9 therefore lets a speed-push target pass when half of the population meets it, instead of three quarters,
gives the push 1,000 generations of patience instead of 600, and replaces the 15 actions by 45: nine steering levels
from −1 to +1 in steps of 0.25 and five throttle levels (full and half brake, coast, half and full gas). Every method
trains again; the results of version 3.8 are set aside (§3.3).

*Table 0. Pole laps of the test circuits, the car's lap on each circuit's racing line with smooth control (quasi-steady
state: the cornering limit v²κ ≤ a(v), then acceleration and braking within the friction circle, top speed 105 m/s), and
the best flying lap of a version 3.8 final controller (an ES run). Seconds.*

| Circuit | Pole lap | Car on the racing line | Best version 3.8 controller |
|---|---|---|---|
| Melbourne | 80.49 | 73.4 | 114.30 |
| Sakhir | 87.87 | 75.9 | 105.50 |
| Sepang | 90.08 | 78.8 | 117.50 |
| Silverstone | 85.09 | 76.5 | 121.20 |

Version 3.8 measures the controllers against real Formula 1 drivers. The car takes the grip and the top speed of a Formula 1 car, 1.8 g at low speed rising with the square of speed to 5 g at 250 km/h and at most 378 km/h, with no engine or drag model (§5.1). Extra hard becomes 19 real circuits at real scale with their real widths: 14 Formula 1 circuits and five other circuits at least as complex. Four further Formula 1 circuits, drawn at random, are held out for the test, where each final controller's best flying lap is compared with the pole lap set there (§4.7, §11). The generated levels below are scaled to Formula 1 size with widths that change along the lap, validation uses generated circuits harder than any training level, and every car's sensors start at 150 m and may evolve up to 1,000 m (§4.2, §5.2, §9.11). Every method trains again; the results of versions up to 3.7 belong to an earlier experiment with a slower car on smaller maps.
Version 3.7 lets the GA build its own networks: every car starts with its inputs wired directly to its outputs, and mutation adds and removes hidden units and layers without a size limit, while the ES, the seeded ES and the imitation methods keep their networks (§6, §9.6). The GA's new settings are recorded and checked only in GA runs, so the runs of the other methods made with version 3.6 remain valid and only the four GA runs are trained again (§3.3). Every GA and NEAT run logs its network size, and the report shows it over training and for each final controller (§9.1, §11).
Version 3.6 runs every method four times (run 0 seeded, runs 1–3 unseeded), gives the speed push twice the patience and a stall boost that mutates the decision layer and the steering sensitivity harder when the push stalls, and adds figures of the system, its decisions and its networks (§3.3, §9.4, §10.3, §11).
Version 3.5 runs every method twice, run 0 seeded and run 1 without any seed, reports both runs and their mean, and lets the race view race the seeded runs, the unseeded runs or both (§3.3, §10.3, §11).
Version 3.4 runs the experiment with as much in parallel as the machines allow (§3.3): the compiled time trials use every logical processor, each Keras Tuner search and each imitation network's training is a job of its own that runs beside the others, every machine has a compiler, and the machines' shares of the runs follow measured generation times.
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

Version 3.8 adds a direct measure: on four real Formula 1 circuits not seen during training, does any final
controller lap faster than the pole lap set there in qualifying, driving a car with the grip and the top speed of a
Formula 1 car (§5.1, §11.1)?

A secondary question is whether evolution that starts from an imitation-trained controller combines
the strengths of both approaches: the expert's driving competence at the start, and the freedom to
exceed the expert's pace afterwards.

Two further questions are whether evolving the network structure together with its weights (the GA and NEAT)
finds better drivers than evolving the weights of a fixed network (the ES), and how far dataset aggregation (DAgger) and
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
- Driving is measured by the number of held-out circuits completed within the target time and by lap time,
  against the pole laps of those circuits.
- All networks are small, so training time stays short.
- The deliverable is a research paper. The project falls under Project 7 ("any research-oriented
  project using DL, EC, and HPO").

## 2. Scope

The following are outside the scope of this project:

- Camera-based perception, including semantic segmentation.
- Human-driven cars, menu systems and track editing.
- A full Formula 1 vehicle model. Engine power, drag, gears, tyre wear, fuel load and slipstream are not modeled; the
  car has the grip and the top speed of a Formula 1 car and nothing else of one (§5.1).
- Collisions between cars.
- Recording, replay or video export of training runs.
- A user interface for browsing earlier generations.
- GPU acceleration. Preliminary measurements on the target hardware showed CPU execution to be faster
  for networks of this size.

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
<text x="463" y="127.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673">shared run loop: time trials, survival,</text>
<text x="463" y="138.7" font-family="DejaVu Sans" font-size="9" fill="#5f6673">levels, speed push, checkpoints</text>
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
identical results (§5.7). A controller therefore behaves identically in every context.

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 878" width="88%" role="img" aria-label="Flowchart of the whole system: start, dataset and map check, imitation lane with search, training and DAgger, evolution lane with generations, survival, levels and safety stop, the shared speed push with stall boost, final controller, best run, test maps, tables and race view" style="display:block;margin:10pt auto 6pt">
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
<text x="259" y="247.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">973 parameters</text>
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
<text x="248" y="492" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">ES of 550</text>
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
<text x="390" y="103" font-family="DejaVu Sans" font-size="9" fill="#5f6673">GA 1,000 · ES 550: random networks</text>
<rect x="367" y="107.5" width="8" height="8" rx="1.5" fill="#F0E442" stroke="#8a7a00" stroke-width="0.8"/>
<text x="380" y="115" font-family="DejaVu Sans" font-size="9" fill="#5f6673">NEAT 1,000: minimal genomes, inputs to outputs</text>
<rect x="367" y="119.5" width="8" height="8" rx="1.5" fill="#0072B2"/>
<text x="380" y="127" font-family="DejaVu Sans" font-size="9" fill="#5f6673">Seeded ES 550: copies of the refined baseline</text>
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
<text x="584" y="365.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">2,000 generations</text>
<text x="584" y="377.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">with no new best?</text>
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
<text x="579" y="401" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">yes: safety stop,</text>
<text x="579" y="412" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="end">the run ends</text>
<rect x="0" y="588" width="632" height="146" rx="3" fill="#f5f6f8"/>
<text x="10" y="602" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Speed push</tspan><tspan font-size="9" fill="#5f6673"> · both lanes, at Extra hard</tspan></text>
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
<text x="250" y="728" font-family="DejaVu Sans" font-size="9" fill="#5f6673">no: next generation</text>
<path d="M39.44,686 L14,686 L14,744 L142.5,744" fill="none" stroke="#5f6673" stroke-width="1"/>
<text x="18" y="681" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes</text>
<path d="M584,386 L584,583 L650,583 L650,744 L142.5,744" fill="none" stroke="#5f6673" stroke-width="1"/>
<circle cx="142.5" cy="744" r="2" fill="#5f6673"/>
<text x="20" y="740" font-family="DejaVu Sans" font-size="9" fill="#5f6673">the run ends</text>
<rect x="0" y="754" width="665" height="124" rx="3" fill="#f5f6f8"/>
<text x="160" y="768" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Selection and test</tspan><tspan font-size="9" fill="#5f6673"> · when a run ends, then after all runs</tspan></text>
<rect x="30" y="776" width="225" height="46" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="142.5" y="791.28" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Final controller of the run</text>
<text x="142.5" y="802.78" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">kept car with the most successes on</text>
<text x="142.5" y="814.28" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">the validation maps 80–99, then fitness</text>
<path d="M142.5,744 L142.5,776" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M282.71,799 L293.87,776 L451.13,776 L462.29,799 L451.13,822 L293.87,822 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="372.5" y="790.8" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">Best run per method:</text>
<text x="372.5" y="802.6" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">most validation successes,</text>
<text x="372.5" y="814.4" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">then mean fitness, lower run</text>
<path d="M255,799 L282.71,799" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<rect x="495" y="783" width="160" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="575" y="797.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Race view</text>
<text x="575" y="808.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">seeded · unseeded · best · all</text>
<path d="M462.29,799 L495,799" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<rect x="30" y="842" width="225" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="142.5" y="856.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">4 test circuits, pole laps</text>
<text x="142.5" y="867.53" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">real widths and 6, 10, 14, 18 m</text>
<path d="M142.5,822 L142.5,842" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<rect x="282.71" y="842" width="179.57" height="32" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="372.5" y="856.03" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Report tables</text>
<text x="372.5" y="867.53" font-family="DejaVu Sans Mono" font-size="9" fill="#5f6673" text-anchor="middle">r0 · r1 · r2 · r3 (mean ± sd)</text>
<path d="M255,858 L282.71,858" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<path d="M372.5,822 L372.5,842" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figflow-head)"/>
<text x="377.5" y="835" font-family="DejaVu Sans" font-size="9" fill="#5f6673">best run named</text>
<text x="147.5" y="835" font-family="DejaVu Sans" font-size="9" fill="#5f6673">every run</text>
</svg>
<p><em>Figure 2. The whole system as one flowchart: every decision it makes, as a yes/no question, from the start of the jobs to the report tables and the race view.</em></p>
</div>

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
│   ├── circuits.py     real circuits: import, training and test sets, pole laps (§4.7)
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
│   ├── evaluate.py     test-circuit evaluation, pole laps, tables, figures
│   ├── netfig.py       network figure
│   ├── netwindow.py    the application's network window (separate process)
│   ├── bench.py        generation-time and Keras training-time measurement
│   └── app.py          pygame application: training view and race view
├── data/racetracks/    the real circuits (TUM racetrack database, LGPL-3.0, §4.7)
├── notebooks/          01_dataset, 02_imitation, 03_dagger, 04_ga, 05_es, 06_es_seeded, 07_neat, 08_evaluate
├── scripts/            notebook generation, experiment runner, pilot runs, PDF rendering
├── tests/
├── results/            generated data (not under version control)
└── docs/               this specification (PDF)
```

*Figure 3. Repository layout.*

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
- **Runs 0 to 3 (versions 3.5 and 3.6).** Every method is run four times. Run 0 uses seed 0 and replays exactly.
  Runs 1, 2 and 3 use no seed at all: wherever run 0 starts a random generator from its seed (the evolution, the
  training of its imitation network, DAgger's generator and its training in every round, the speed push), an
  unseeded run starts it from the operating system's randomness, and nothing is drawn or recorded beforehand. A
  stopped unseeded run resumes from its checkpoint, which holds the generator's state, but it cannot be replayed from
  its beginning, and two unseeded runs differ. Each run is one draw of the method's result; four draws give a mean
  with a meaningful standard deviation and a best run. The dataset, the Keras Tuner searches (one seeded search per
  method, whose chosen architecture serves every run) and the validation maps and the test circuits are fixed and shared by all
  runs, so all face the same final test. The seeded ES of a run starts from the baseline of the same run. Folders
  and job labels keep their names (`seed0` … `seed3`; `ga:0` … `ga:3`).
- **Jobs side by side and on several machines.** The experiment may instead run as jobs side by side, which keeps
  every processor busy, `scripts/run_notebooks.py --runs nb:01,search:kt_random,baseline:0,ga:1 --at-once 8`, and on
  several machines with `scripts/fleet.py` (plan in `scripts/fleet.toml`). A job is notebook 01, one Keras Tuner
  search (`search:kt_random`, `search:kt_hyperband`, `search:kt_bayesian`), or a run of any method, trained directly
  by `racing/train.py`, which makes exactly the calls of its notebooks (checked by a test). A run of an imitation
  method first trains its own imitation network, notebook 02's call for its seed (the baseline's training, or the
  retraining of its search's chosen architecture), and then refines it by DAgger and pushes it for speed, notebook
  03's call; every step returns its finished record instead of training again, so a stopped job resumes where it
  stopped. Each job has its own status file and output log, and starts as soon as a slot is free and its inputs exist:
  a search and a baseline run wait for the dataset, a run of a searched method for its search, a seeded ES run for
  its refined baseline; a waiting job holds no slot (`racing/jobs.py`). The three searches therefore run side by
  side, and every DAgger refinement starts as soon as its own network is trained. Every step sets its random seed when
  it starts, so a network trained in a job of its own equals the network notebook 02 trains in order (checked by a
  test). Notebook 02 is not a job: the notebooks later find the networks, searches and runs complete and only report
  them. Run 0's results depend only on its seed and the settings, so where it trains changes only its training time; runs 1–3 also depend on chance.
  Each run's summary records the machine it finished on (host, processor, thread setting and worker count), and the
  training cost is
  compared by evaluations, which are exact on any machine; training times are reported as context, with the
  machines named. `fleet.py setup` gives a server the desktop's commit (as a git bundle), the pinned environment, a
  C++ compiler where the server has none (conda-forge's `gxx` in the run environment, which needs no administrator
  rights) and any inputs its jobs need from the desktop; `start` starts every machine's jobs in a detached tmux session only when
  every machine is ready; `collect` copies a server's finished runs to the desktop after checking their settings and
  never replaces a different copy; `finish` then runs notebooks 02 to 08. `run_notebooks.py --archive NAME` moves the
  earlier results to `results/archive/NAME/` and exits, so the next start begins afresh. With `--runs`, it moves only
  the named runs: their run folders, final controllers and status files, under the same paths inside the archive.
  `fleet.py archive NAME --runs ...` does the same on every machine. Version 3.7 trains only the GA again:
  `fleet.py archive revision3.6-ga --runs ga:0,ga:1,ga:2,ga:3` sets the version 3.6 GA runs aside, and the next start
  trains the four GA runs while every other completed run is skipped. Version 3.9 changes the actions, so every
  method trains again: `fleet.py archive revision3.8` sets the version 3.8 results aside, with the pole check of its
  finished controllers, and a fresh run starts.
- **Machine plan.** In `scripts/fleet.toml`, `at_once` bounds the jobs a machine runs at once by memory: compiled
  runs share the processors through their threads, and a Keras job uses few of them, so the bound is the measured peak
  memory of each kind of job against the machine's memory; for version 3.8 the peaks are measured on each machine
  before the reported run (§9.10). `workers` is the number of worker processes of each job.
  Servers receive only jobs that need nothing from the desktop (GA, ES and NEAT runs). Notebook 01, the searches and
  the runs of the imitation methods are refused on a server: `collect` copies runs only, not the dataset or a search,
  and Keras training may round differently on another processor. `fleet.py setup` ends by naming the simulator a server
  will use, with a warning when it is the numpy simulator, and every runner's "run started" log line names it too. How many follows from the
  generation time measured by `python -m racing.bench generation` on each machine and the generation counts of the
  previous runs, so that the machines finish at about the same time; the arithmetic is recorded in the plan's
  comments.
- **Kernel:** each notebook runs in a kernel started with the same Python interpreter as the runner,
  with `notebooks/` as the working directory.
- **Checkpoints.** Every evolutionary run saves a checkpoint after each generation. It contains the
  population, the state of the random generator, the level, streak, target factor and step size, the
  seeds already used, NEAT's species and innovation records, the generation count, the number of
  evaluations and the active run time. A DAgger refinement saves a checkpoint after each round. A
  checkpoint is written to a temporary file and then renamed, so that an interruption during writing
  cannot corrupt it.
- **Resumption.** Resumption is the default behavior. When the runner is started again, completed runs
  are skipped and interrupted runs continue from their last checkpoint, losing at most one generation. A resumed run 0 produces exactly the same results as an uninterrupted one; a resumed unseeded run continues from the state of its random generator saved in the checkpoint. The option `--fresh` removes previous results and starts over. The option `--from NN` starts at a given notebook.
- **Settings check.** Checkpoints and run summaries record the settings that affect results. A run
  whose recorded settings differ from the current ones is not resumed. The runner stops with a message
  naming the difference, and `--fresh` is required. Settings that affect only one method are recorded and checked
  only in that method's runs: the GA's structural settings (§9.6, version 3.7) belong to GA runs alone, so a GA run
  made with version 3.6 is refused, while the runs of the other methods made with version 3.6 remain valid. Every
  other record, such as the run record and the imitation stages, carries the settings shared by all methods. The runner checks every run record and checkpoint before it executes the first notebook, and exits with code 3.
- **Progress display:**
  - **Status file.** Every run writes its current state to `results/status.json` after each generation
    or round: the run number out of the total, the method, seed, level, generation, lowest map success
    share, streak, target reduction and elapsed time. The imitation training writes it after every epoch (epoch,
    validation accuracy and the best so far) and each Keras Tuner search after every completed trial (trial
    number, best validation accuracy), so the line is live from the second notebook on.
  - **Live line.** While the runner executes the notebooks, it displays the status as a single line that
    is updated in place, for example
    `[run 18/32] GA run 1 · Hard · gen 412 · worst map 84% / 90% ███████████▍ · streak 1/3 · 2h 13m`.
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
  of logical processors, the thread setting, the number of worker processes, the operating-system power profile where
  available, the versions of Python, numpy, TensorFlow, Keras and keras-tuner, the settings that affect
  results (the values of §13), the code version (commit and whether the working tree had uncommitted
  changes), and the start and end time of each notebook.
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
  compiled runs of 20 threads 1.13 s, and 0.51–0.61 s with 4 threads and the runs at the lowest priority. Thread
  counts and priorities change only times.
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
can be reached. The pilots write to `results/pilots/revision3.9/` (earlier versions' pilots keep their own folders) and are not part of the reported results.

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
<text x="511" y="280.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">GA · population 1,000</text>
<rect x="250" y="298" width="410" height="30" rx="3" fill="none" stroke="#7a5c99" stroke-opacity="0.6" stroke-width="1"/>
<rect x="258" y="309" width="8" height="8" rx="1.5" fill="#009E73"/>
<text x="270" y="316.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">es:k</text>
<rect x="368" y="302" width="286" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="511" y="316.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">ES · population 550</text>
<rect x="250" y="334" width="410" height="30" rx="3" fill="none" stroke="#7a5c99" stroke-opacity="0.6" stroke-width="1"/>
<rect x="258" y="345" width="8" height="8" rx="1.5" fill="#F0E442" stroke="#6b5f00" stroke-width="0.8"/>
<text x="270" y="352.8" font-family="DejaVu Sans Mono" font-size="10.5" fill="#1d2330">neat:k</text>
<rect x="368" y="338" width="286" height="22" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="511" y="352.78" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">NEAT · population 1,000</text>
<text x="8" y="307.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">need nothing from the desktop:</text>
<text x="8" y="318.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673">every run starts at once</text>
<text x="0" y="388" font-family="DejaVu Sans" font-size="9" fill="#5f6673">An arrow is an input: a job starts when its inputs exist, and a waiting job holds no slot. fleet.py collect</text>
<text x="0" y="399" font-family="DejaVu Sans" font-size="9" fill="#5f6673">copies the server's finished runs to the desktop.</text>
</svg>
<p><em>Figure 4. The experiment as jobs: the dataset feeds the searches and the baseline runs, each search feeds its method's runs, each baseline run's DAgger feeds the seeded ES run of the same number, and the GA, ES and NEAT runs on the server need nothing.</em></p>
</div>

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 392" width="100%" role="img" aria-label="The two machines over time, a schematic of when each kind of job runs" style="display:block;margin:10pt 0 6pt">
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
<p><em>Figure 5. The two machines over time (schematic, not to scale): every job starts at once, the Keras jobs come first, evolution at nice 19 uses the processors they leave, and both machines finish at about the same time.</em></p>
</div>

## 4. Track Generation and Curriculum

### 4.1 Track representation

The function `tracks.get_track(seed, level)` returns the map of any level as a `Track` object: a generated map,
whose geometry `track_gen.make_track` draws, on levels 1–3 and the validation setting, and a real circuit (§4.7),
built by `circuits.level_track(seed, level)` from `circuits.circuit_track(name, reverse)`, on Extra hard and the test
level. A `Track` contains:

- the centerline, resampled at 1 m intervals
- the road's width to the left and to the right of every centerline point, and the left and right boundaries
- a road occupancy grid at 0.5 m resolution
- a distance-to-boundary map computed with `scipy.ndimage.distance_transform_edt`
- the start pose and track length
- the expert driver's lap time

The centerline is stored in driving order, so a map that runs clockwise is the
reversed point sequence of its counter-clockwise form (§4.4). A generated map has equal widths to the left and right;
a real circuit has those of its data, whose centre line is not exactly central (version 3.8; up to version 3.7 every
map had one width for the whole lap).

### 4.2 Curriculum levels

The evolutionary controllers train on a curriculum of four levels, from gentle loops on a wide road to real circuits.
Version 3.8 scales the generated levels to Formula 1 size, so that each leads up to the real circuits of Extra hard,
and adds a validation setting harder than any training level.

- **Easy (level 1)** uses the loop generator: control points at jittered angles and random radii, joined by a
  periodic cubic spline. It is set to produce gentle loops on a wide road.
- **Medium and Hard (levels 2–3)** and the **validation setting** use the circuit generator (§4.3).
- **Extra hard (level 4)** consists of the 19 real training circuits (§4.7).

*Table 1. Curriculum levels, generator settings and the validation setting. Lengths in metres.*

| Level | Source | Lap length | Corner radii | Features | Tightest corner | Road width |
|---|---|---|---|---|---|---|
| 1 Easy | Loop: 4–6 points, radius 250–450 | about 2,000–2,500 | Spline | None | 120 | 12–16, varying |
| 2 Medium | Circuit: 6–8 points, radius 250–600 | about 2,000–3,700 | Slow 35–60, fast 100–200 | None | 35 | 10–16, varying |
| 3 Hard | Circuit: 9–12 points, radius 300–750 | about 3,200–4,500 | Slow 20–45, fast 80–250 | Main straight ≥ 600 into a hairpin; 0–1 chicane; 0–1 run of esses | 18 | 9–15, varying |
| 4 Extra hard | 19 real circuits (§4.7) | 3,690–7,000 | Real | Real | About 16–33, by circuit | Real, 7.4–27.6 per point |
| Validation | Circuit: 13–17 points, radius 350–900 | about 4,100–7,700 | Slow 12–30, fast 60–250 | Main straight ≥ 900 into a hairpin; a second hairpin; 2–3 chicanes; 2–3 runs of esses | 12 | 7.5–12, varying |

The lap lengths of the generated levels are those measured over 20 maps of each level and the 20 validation maps
(medians 2.2, 2.9 and 4.1 km on Easy, Medium and Hard); the radii above produce them. The real Formula 1 circuits are
4.3 to 7.0 km long, their tightest corners have radii of about 16 to 25 m (estimated
from the smoothed centre lines) and their roads are 7.4 m wide at the narrowest. The generated levels climb toward
these values, and the validation setting goes beyond them: a tighter tightest corner, more chicanes and esses and a
narrower road than any training circuit.

**Road width (version 3.8).** The width of a generated map changes along the lap. A base width is drawn uniformly
from the level's range by `default_rng([seed, level, 13])` and varied by up to ±20% along the lap by a smooth periodic
profile, the mean of three sinusoids of random phase drawn from the same generator, so ±20% is its extreme, clipped
to the range; the left and
right halves are equal. The width depends only on the map's seed and level, so a map keeps it on every generation
attempt; the generator sizes chicanes and esses and the spacing between separate parts of the road by the widest point
of the road. Up to version 3.7 every map had one width, 10 to 16 m (from version 3.2) or 16, 14, 12 and 10 m by level
(up to version 3.1).

The values in Table 1 are design values. They may be adjusted during implementation so that the expert completes
every level and the generator accepts a candidate within a few attempts.

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
   offset alternately to each side by 1.2 to 2.0 road widths and spaced 70–110 m apart.
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
and Spa. The generator reproduces no real circuit; real circuits enter at Extra hard (§4.7). Version 3.8 scales
the straights, chicanes and esses to Formula 1 size; up to version 3.7 they were about half as long.

### 4.4 Direction

Each map runs clockwise or counter-clockwise with equal probability. The direction is drawn from the
map's seed, so a seed always produces the same map in the same direction. Revision 2.0 tracks all ran
counter-clockwise.

A real training circuit is driven in either direction: each time it is drawn, the run's generator also draws its
direction, so the 19 circuits give 38 layouts. A test circuit is driven only in its racing direction, the direction
of its pole lap (§4.7).

### 4.5 Validation

A generated map is rejected if any of the following holds:

- the road comes too close to another part of itself
- any corner is tighter than the level's tightest corner
- the expert driver cannot complete a lap

A rejected seed is retried deterministically using the triple (seed, level, attempt), so a given seed
always yields the same map. The direction is drawn from the seed alone (§4.4), so it is the same for every
attempt. Real circuits are not rejected: the expert must complete every one of them in both directions, which a test
checks (§12.1).

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
of a point differ by up to 2.2 m.

- **Import.** The centre line and both widths are interpolated by periodic cubic splines and resampled at 1 m, at real
  scale. The widths are used as given, without clipping: the real circuits are 7.4 to 27.6 m wide at
  single points, the widest being the approach to Austin's first corner. The start is the database's first point, at or near the start–finish line. Each circuit's racing direction is
  recorded with it, checked against the official circuit map; the four test circuits import within 0.2% of their
  official lengths (Melbourne 5.299 km against 5.303 km, Sakhir 5.407 against 5.412, Sepang 5.538 against 5.543 and
  Silverstone 5.887 against 5.891).
- **The 25 circuits.** Eighteen Formula 1 circuits: Austin, Budapest, Catalunya, Hockenheim, Melbourne, Mexico City,
  Montreal, Monza, Sakhir, São Paulo, Sepang, Shanghai, Silverstone, Sochi, Spa, Spielberg, Suzuka and Yas Marina;
  and seven others from DTM and IndyCar: Brands Hatch, Indianapolis, Moscow Raceway, Norisring, Nürburgring,
  Oschersleben and Zandvoort.
- **Test circuits.** About a fifth of the Formula 1 circuits are held out: the four drawn by
  `numpy.random.default_rng(0).choice(names, 4, replace=False)` over the 18 Formula 1 names in alphabetical order,
  which gives Melbourne, Sakhir, Sepang and Silverstone. Holding out whole circuits means a controller cannot learn the
  shapes of the circuits it is tested on; it must have learned to drive.
- **Training circuits.** The other 14 Formula 1 circuits, and those of the seven others that are at least as complex
  as the simplest Formula 1 circuit, Monza, by every measure of Table 1e: at least 8 corners per lap, at least 1.38
  corners per kilometre and at least 172° of turning per kilometre. A corner is a stretch whose radius, smoothed over
  20 m, stays below 300 m and that turns by more than 20°. Five qualify. Indianapolis, the oval, and Norisring, a
  2.3 km street circuit with five corners, do not. Extra hard therefore holds 19 circuits: Austin, Budapest,
  Catalunya, Hockenheim, Mexico City, Montreal, Monza, São Paulo, Shanghai, Sochi, Spa, Spielberg, Suzuka, Yas Marina,
  Brands Hatch, Moscow Raceway, Nürburgring, Oschersleben and Zandvoort.

*Table 1e. Complexity of the seven non-Formula 1 circuits, against the range of the 18 Formula 1 circuits.*

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

## 5. Simulation Model

### 5.1 Vehicle dynamics

The vehicle is modeled as a point mass with a heading, integrated with a fixed time step of 0.05 s
(20 Hz). Its state consists of position, heading and speed.

The vehicle has the grip and the top speed of a Formula 1 car and nothing else of one (version 3.8): there is no
engine power, drag, gearbox, tyre wear or fuel load. The model imposes no steering-angle limit and no fixed
acceleration or braking rates. How fast to drive, when to brake and how sharply to turn must all be learned by the
controllers.

- **Grip:** the grip limit rises with the square of speed, as a Formula 1 car's downforce does, up to a ceiling:
  `a(v) = min(a0 + c × v², a_max)`, with g = 9.81 m/s², a0 = 1.8 g = 17.66 m/s² (the tyres' grip at low speed) and
  a_max = 5 g = 49.05 m/s², reached at 250 km/h (69.4 m/s), so that c = (a_max − a0) / 69.4² = 0.00651 m⁻¹. Up to
  version 3.7 the grip was 12 m/s² at every speed.
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
  - The minimum turning radius is `v² / a(v)`: about 20 m at 72 km/h, 98 m at 250 km/h and 225 m at the top speed,
    so a fast car cannot take a tight corner and must brake before it.
  - Safe speed on a straight is bounded by the top speed and by how early the walls ahead can be seen. Braking at full
    grip from the top speed takes about 142 m to a standstill and 126 m to 90 km/h, so the starting sensor length of
    150 m leaves little to spare before a slow corner.
  - The time step stays 0.05 s, so the controllers keep their fine control; at the top speed a car covers 5.25 m per
    step, less than the narrowest real road (7.4 m).
- **Grip in the application:** the visualization application can display races with ten times the grip at every
  speed (§10.5); training, evaluation and the expert always use a(v).

### 5.2 Sensors

Nine rays at −90°, −67.5°, …, +90° relative to the heading measure the distance to the track boundary, up to the
car's sensor length L: 150 m for every car at the start and for every car that does not evolve its setup, and 30 to
1,000 m when evolved (§9.11). A clear day's view of a circuit reaches about a kilometre, which sets the upper limit.
Rays are cast by sphere tracing on the distance-to-boundary map, with a minimum step of 0.5 m. A ray of length L takes
at most ⌈48 × L / 50⌉ iterations: 144 for 150 m and 960 for 1,000 m. Each ray stops as soon as it reaches the boundary
or its length.

The input vector has 12 elements: the nine distances divided by 150 m, the speed divided by the top speed of
105 m/s, the position across the road and the local road width divided by 20 m. The ray scale is fixed, so a car of
the starting sensor length sees ray inputs from 0 to 1 and a car with L = 1,000 m up to 6.7; the speed input lies
between 0 and 1. Up to version 3.7 the rays were divided by 50 m and the speed by 40 m/s. The position across the road
is the car's signed distance from its matched centerline point along the road's normal, divided by the width on that
side of the point: −1 at the left edge, 0 on the centerline, +1 at the right edge, relative to the driving direction.
The local road width is the sum of the two widths at the matched point. Both mean the same on a road of any width, so
a network can tell where it is on the road without relying on one width (version 3.2; local widths from version
3.8).

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 414" width="100%" role="img" aria-label="Figure: the nine rays and twelve inputs of a car, the network, the fifteen output classes with the argmax choice, steering sensitivity and the grip limit" style="display:block;margin:10pt 0 6pt">
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
<text x="176" y="12" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">12 inputs</text>
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
<text x="280" y="386" font-family="DejaVu Sans" font-size="9" fill="#5f6673">radius a(v): 1.8 g slow, 5 g from 250 km/h;</text>
<text x="280" y="397" font-family="DejaVu Sans" font-size="9" fill="#5f6673">with k = 1, half steering and full gas ask</text>
<text x="280" y="408" font-family="DejaVu Sans" font-size="9" fill="#5f6673">√(0.5² + 1²) = 1.12 a(v): scaled ×0.89</text>
<path d="M312.0,312.0 L154,312.0" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figsense-head)"/>
<text x="232" y="293" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">drive for 0.05 s,</text>
<text x="232" y="304" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">then sense again</text>
</svg>
<p><em>Figure 6. What a car senses and does: nine rays and three more inputs feed the network, the largest of its 45 scores picks one of nine steering and five throttle levels, and steering sensitivity and the grip limit shape the command.</em></p>
</div>

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

Every run records the time of each lap it completes (version 3.8). The first lap starts from a standstill; the
second and third start at speed, like a qualifying lap, and are the flying laps compared with pole laps (§11.1).

### 5.5 Fitness

The fitness of a run on a single track is:

- `p`, the completed fraction of the three laps (0 ≤ p < 1), if the run is not completed
- `1 + (1 − t_run / t_limit)`, a value between 1 and 2, if all three laps are completed, where `t_run` is the time for the three laps and `t_limit` is the time limit of the run (§5.4)

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

The numpy simulator advances all cars of a map together, one step at a time, with a fixed cost of about 0.3 to 0.5 ms
of interpreter and array-call overhead per step whatever the number of cars; a level-4 lap of the version 3.3 maps is about 1,900 steps. More
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

*Table 1b. Time from the start of a run to its mastery goal, seed 0. The compiled runs reached the goal at the same
generation as the numpy runs, so their results are identical; numpy: the reported run of 11–12 October 2026 (GA and ES
on the school server, three jobs at a time; NEAT on the desktop, four at a time); compiled: the development machine on
18 October 2026, eight jobs at a time (the three Keras Tuner searches, a DAgger refinement and the dataset beside them).*

| Run | Mastery goal at generation | numpy | Compiled | Speed-up |
|---|---|---|---|---|
| GA, seed 0 | 335 | 91 min | 16 min | 5.6× |
| ES, seed 0 | 445 | 101 min | 16 min | 6.2× |
| NEAT, seed 0 | 302 | 101 min | 25 min | 4.1× |

*Table 1c. Complete runs of seed 0 on the desktop (Intel Core i9-12900K) on 18 October 2026, eight jobs at a time,
with Keras jobs first (§3.3), against the reported run of 11–12 October 2026 on the numpy simulator. Every result is
identical: the same generations, the same validation successes and the same networks chosen by the searches.*

| Job | Result | numpy | Compiled, desktop | Speed-up |
|---|---|---|---|---|
| GA, seed 0 | 1,843 generations, 20 of 20 validation maps | 783 min | 81 min | 9.6× |
| ES, seed 0 | 1,234 generations, 18 of 20 | 416 min | 46 min | 9.0× |
| NEAT, seed 0 | 1,310 generations, 20 of 20 | 686 min | 77 min | 8.9× |
| Baseline, seed 0: imitation, DAgger and speed push | 1,731 generations, 16 of 20 | about 590 min | 80 min | about 7.4× |
| The three Keras Tuner searches | the same chosen networks | 98 min, one after another | 37 min, side by side | 2.6× |
| Dataset | the same rows | 1.1 min | 0.6 min | 1.8× |

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
  it, which the run records report with the simulator (§3.3), without running the compiler again.
- **One call per generation.** The maps of a generation, the cars' network programs and setups and each map's target
  time are passed in one call. Every pair of map and car is an independent job; the threads (one per available
  logical processor unless `EC_RACING_THREADS` sets a number, §3.3) take the next unclaimed job from a shared counter, so a map whose cars finish early leaves no thread idle. Each job drives one car
  from the start line until it is done, in the order of `sim.sense`, the network's action, `sim.step` and the progress
  update; the car's own clock advances by DT per step. The returned values are those of `sim.RunResult`, and the
  fitness is computed from them by `sim.fitness`. Results do not depend on the number of threads.
- **Exact equivalence.** Every field of every `RunResult` equals the numpy simulator's (`np.array_equal`, with NaN
  finish times in the same places). Every expression of `sim.py`, `track.wall_distance`, `expert.py` and
  `actions.py` is transcribed with the same operand order in double precision; the build turns off fused
  multiply-add and does not use `-ffast-math`; `cos`, `sin`, `hypot`, `atan2` and `sqrt` come from the C library numpy
  calls on the same machine, in the versions numpy binds: glibc 2.35 added a rewritten `hypot` under a new symbol
  version, so the library binds `hypot@GLIBC_2.2.5`, as numpy does; the sine and cosine of one angle are two calls, as numpy makes them, because the build
  forbids the compiler to fuse them into the C library's `sincos` (`-fno-builtin-sin -fno-builtin-cos`): glibc 2.35's
  `sincos` differs from its own `sin` and `cos` in about one angle in 1,500, while glibc 2.39's agrees, and the two
  calls cost about 5% of the compiled time (1,000 trained cars on one thread: 15.0 s fused, 15.9 s as two calls);
  rounding is half to even, as in `np.round`; ties go to the first largest output
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
  incoming connections (source slot, weight); 45 output slots give the action order. A node's value is the sum of
  source value × weight over its connections, from zero and in order, then plus the bias, then the activation. Dense
  networks (`nets.DensePopulation`, evaluated with `np.einsum`) list each unit's inputs in index order; NEAT genomes
  (`neat.NeatPopulation`, evaluated with `np.bincount`) list nodes by depth and each node's enabled connections in
  genome order, so both sum exactly as numpy does. One format covers every architecture and activation, NEAT, and
  networks of an earlier revision that read fewer inputs. Networks whose architecture differs from car to car
  (`nets.ShapedPopulation`, the GA from version 3.7) receive each car's own dense program; a layer of a single unit,
  which `np.einsum` sums in another order, is summed by the numpy simulator in input order, as the library does, and
  the check at load also compares a population of mixed architectures.
- **Version 3.8.** The speed-dependent grip, the top speed, the left and right widths of every centerline point, the
  scaled inputs and the lap times are transcribed like every other expression, so the equivalence holds unchanged;
  the tests compare both simulators on real circuits (§12.1).
- **Expert.** The expert (§7.1) and its quantization to the 45 actions are ported as a second kind of driver, used by
  `expert.expert_lap_time` to validate every new map.
- **Scope.** Single trained controllers (`nets.DenseNet`, `neat.NeatNet`) are evaluated with numpy's matrix product,
  whose rounding depends on the linear-algebra library's internal order and cannot be reproduced. DAgger's own driving
  (§8.4), the final races (§11), dataset collection (§7) and the application's playback (§10) therefore remain in
  numpy; they still benefit from the faster map building.

*Table 1d. Simulator used by each caller.*

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
- **Output:** a 45-way softmax. Each class is one combination of a steering level and a throttle level:
  - steering fraction `u_s` ∈ {−1, −0.75, −0.5, −0.25, 0, +0.25, +0.5, +0.75, +1} (nine levels)
  - throttle fraction `u_t` ∈ {−1 (full brake), −0.5 (half brake), 0 (coast), +0.5 (half gas), +1 (full gas)}
  - `class = 5 × steer_index + throttle_index`

  Up to version 3.8 there were 15 classes: five steering levels (−1, −0.5, 0, +0.5, +1) and three throttle levels
  (brake, coast, gas), `class = 3 × steer_index + throttle_index`. Controllers of those versions have 15 outputs and
  are not raced with the 45-class table (§10.3).

  The car executes the class with the highest probability.

Framing control as classification makes test accuracy directly measurable.

The module `nets.py` provides:

- a batched numpy forward pass for whole populations, using one `einsum` per layer
- `to_keras(weights)`, which builds the equivalent Keras model and copies the weights into it

Every final controller is stored as a `.keras` file, so network size and inference time are measured
the same way for all approaches. Keras stores weights in float32. Evolutionary controllers are selected
with float64 weights, so they are additionally stored as float64 `.npy` files, and all driving
evaluations of evolutionary controllers use these files. An evolved controller also stores its setup (sensor length and steering sensitivity) in `<seed>.setup.json`; a controller without this file drives with L = 150 m and k = 1 (L = 50 m up to version 3.7).

The baseline architecture, which the ES also uses, is 12 → 16 (ReLU) → 45 (softmax), for a total
of 973 trainable parameters (463 with 15 outputs in versions 3.2–3.8; 10 inputs and 431 parameters up to version
3.1).

The GA's networks have no fixed architecture: each car carries its own list of hidden layer sizes, from none to any
number of layers of any width, with ReLU hidden units (§9.6). A final GA controller is stored like every evolved dense
controller, as a `.keras` file whose layers give its architecture and as float64 weights (`.npy`). Up to version 3.6
the GA used the baseline architecture.

## 7. Expert Driver and Dataset

### 7.1 Expert driver

- **Steering:**
  1. Pure pursuit toward a centerline point `6 m + 0.4 s × v` ahead gives a path curvature κ_p.
  2. The required lateral acceleration `v² × κ_p` is divided by the grip a(v) and clipped to [−1, 1], giving a
     steering fraction.
  3. The fraction is quantized to the nearest of the nine steering levels.
- **Speed:** the expert uses only track geometry within the starting sensor length of 150 m, so it never relies on
  information a network with that sensor length cannot obtain. Its target speed is the lowest of three values:
  - the highest speed v with `v² × κ ≤ 0.85 × a(v)`, where κ is the maximum curvature within the next 150 m
  - the highest speed from which it can stop within 150 m braking at 0.85 × a(v), about 98 m/s (353 km/h)
  - the top speed

  The expert's throttle is its speed error (target minus speed) divided by its band, clipped to [−1, 1] and
  quantized to the nearest of the five throttle levels: full gas or full brake beyond the band, half gas or half brake
  within it, coast near the target. The band is max(0.5 m/s, a(v) × dt), the speed one step can change at full grip:
  with a narrower band at Formula 1 grip, the throttle would switch between gas and brake every step and leave too
  little grip for turning. Up to version 3.8, with three throttle levels, the expert selected gas below the target by
  more than the band, brake above it by more than the band, and coast otherwise. While its steering fraction before quantization is ±1 (full lock) the expert does not accelerate: it coasts or brakes. These targets govern the expert only. The networks are free to drive
  faster or slower.

The expert chooses from the same 45 actions as the networks, so every label is an action a network can
reproduce. The expert's lap time on each track sets that track's time limit and target time (§5.4, §9.2). Version
3.8 retunes the expert for the Formula 1 car; its lookahead and margins may be adjusted during implementation so that
it completes every generated level, the validation setting and every real circuit in both directions.

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

- **Training:** 41 freshly generated maps (one at Easy, 20 at Medium and 20 at Hard) and the 19 training circuits
  in both directions, 38 layouts (§4.6).
- **Validation:** the 20 validation maps.
- **Test:** the four test circuits in their racing direction.

The number of rows of each split is reported in the dataset table (§11.3).

Collection is distributed across worker processes, one track per task. Each track uses its own random
number generator, so the dataset is identical to a sequential collection. The collection time is
recorded and reported as a cost shared by the four imitation methods, their DAgger refinement and the seeded ES.

## 8. Imitation-Learning Controllers

### 8.1 Baseline

The baseline is the 973-parameter network from §6, trained with:

- Adam at a learning rate of 1e-3
- batch size 256
- sparse categorical cross-entropy
- at most 150 epochs, with early stopping on validation accuracy (patience 5)

The epoch limit was raised from 50 to 150 after a first run showed it to be binding: the baseline
trained for all 50 epochs, and 13 of 30 random-search trials reached their best validation accuracy in
the final epoch.

The baseline is trained for run 0 with seed 0 and for runs 1–3 without a seed (§3.3). Two Keras settings that affect training speed but not
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

The selected architecture, with its searched hyperparameters, is then trained from scratch for run 0 (seed 0) and
runs 1–3 (no seed). These four networks are the method's controllers after initial training. DAgger (§8.4) and the speed push (§8.5) produce its final controllers. The search therefore runs once
per method, and the variation across runs measures training variability for a fixed architecture,
not the variability of the search itself.

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
<text x="28" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">12</text>
<text x="28" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">inputs</text>
<rect x="71.5" y="98" width="14" height="32" rx="2" fill="#d2d2d2" stroke="#7F7F7F" stroke-opacity="1" stroke-width="1.2"/>
<text x="78.5" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">16</text>
<text x="78.5" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">ReLU</text>
<rect x="122" y="69" width="14" height="90" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="129" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">45</text>
<text x="129" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">softmax</text>
<text x="78.5" y="230" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">973 parameters</text>
<text x="54.6891" y="243" font-family="DejaVu Sans" font-size="9" fill="#5f6673">208 + <tspan fill="#c0392b">765</tspan></text>
<rect x="169" y="0" width="157" height="252" rx="3" fill="#f5f6f8"/>
<rect x="178" y="9" width="10" height="10" rx="1.5" fill="#E69F00"/>
<text x="194" y="18" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">KT Random</text>
<polygon points="204,102 240.5,98 240.5,130 204,126" fill="#5f6673" fill-opacity="0.13" stroke="#5f6673" stroke-opacity="0.35" stroke-width="0.8" stroke-linejoin="round"/>
<polygon points="254.5,98 291,69 291,159 254.5,130" fill="#fdecea" stroke="#c0392b" stroke-width="1" stroke-linejoin="round"/>
<rect x="190" y="102" width="14" height="24" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="197" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">12</text>
<text x="197" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">inputs</text>
<rect x="240.5" y="98" width="14" height="32" rx="2" fill="#f6dda6" stroke="#E69F00" stroke-opacity="1" stroke-width="1.2"/>
<text x="247.5" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">16</text>
<text x="247.5" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">tanh</text>
<rect x="291" y="69" width="14" height="90" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="298" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">45</text>
<text x="298" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">softmax</text>
<text x="247.5" y="230" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">973 parameters</text>
<text x="223.689" y="243" font-family="DejaVu Sans" font-size="9" fill="#5f6673">208 + <tspan fill="#c0392b">765</tspan></text>
<rect x="338" y="0" width="157" height="252" rx="3" fill="#f5f6f8"/>
<rect x="347" y="9" width="10" height="10" rx="1.5" fill="#56B4E9"/>
<text x="363" y="18" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">KT Hyperband</text>
<polygon points="367,102 390,50 390,178 367,126" fill="#5f6673" fill-opacity="0.13" stroke="#5f6673" stroke-opacity="0.35" stroke-width="0.8" stroke-linejoin="round"/>
<polygon points="404,50 427,50 427,178 404,178" fill="#5f6673" fill-opacity="0.13" stroke="#5f6673" stroke-opacity="0.35" stroke-width="0.8" stroke-linejoin="round"/>
<polygon points="441,50 464,69 464,159 441,178" fill="#fdecea" stroke="#c0392b" stroke-width="1" stroke-linejoin="round"/>
<rect x="353" y="102" width="14" height="24" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="360" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">12</text>
<text x="360" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">inputs</text>
<rect x="390" y="50" width="14" height="128" rx="2" fill="#c4e5f7" stroke="#56B4E9" stroke-opacity="1" stroke-width="1.2"/>
<text x="397" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">64</text>
<text x="397" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">ReLU</text>
<rect x="427" y="50" width="14" height="128" rx="2" fill="#c4e5f7" stroke="#56B4E9" stroke-opacity="1" stroke-width="1.2"/>
<text x="434" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">64</text>
<text x="434" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">ReLU</text>
<rect x="464" y="69" width="14" height="90" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="471" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">45</text>
<text x="471" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">softmax</text>
<text x="416.5" y="230" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">7,917 parameters</text>
<text x="373.174" y="243" font-family="DejaVu Sans" font-size="9" fill="#5f6673">832 + 4,160 + <tspan fill="#c0392b">2,925</tspan></text>
<rect x="507" y="0" width="157" height="252" rx="3" fill="#f5f6f8"/>
<rect x="516" y="9" width="10" height="10" rx="1.5" fill="#CC79A7"/>
<text x="532" y="18" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">KT Bayesian</text>
<polygon points="542,102 578.5,50 578.5,178 542,126" fill="#5f6673" fill-opacity="0.13" stroke="#5f6673" stroke-opacity="0.35" stroke-width="0.8" stroke-linejoin="round"/>
<polygon points="592.5,50 629,69 629,159 592.5,178" fill="#fdecea" stroke="#c0392b" stroke-width="1" stroke-linejoin="round"/>
<rect x="528" y="102" width="14" height="24" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="535" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">12</text>
<text x="535" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">inputs</text>
<rect x="578.5" y="50" width="14" height="128" rx="2" fill="#edd0e0" stroke="#CC79A7" stroke-opacity="1" stroke-width="1.2"/>
<text x="585.5" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">64</text>
<text x="585.5" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">tanh</text>
<rect x="629" y="69" width="14" height="90" rx="2" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<text x="636" y="198" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">45</text>
<text x="636" y="210" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">softmax</text>
<text x="585.5" y="230" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">3,757 parameters</text>
<text x="561.689" y="243" font-family="DejaVu Sans" font-size="9" fill="#5f6673">832 + <tspan fill="#c0392b">2,925</tspan></text>
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
<p><em>Figure 7. The dense networks of fixed architecture with the 45 outputs of version 3.9 (the Keras Tuner architectures shown are those the version 3.6 searches selected; version 3.9 searches again): layer sizes, activations and parameter counts, with the decision layer that the stall boost mutates three times as strongly. The GA grows networks of its own, starting from 12 → 45 (§9.6).</em></p>
</div>

### 8.4 DAgger refinement

Dataset aggregation (DAgger) corrects the main weakness of imitation: the trained car reaches
situations the expert never showed it and does not know how to recover. Each trained network (four
methods, four runs) is refined in rounds:

1. **Drive.** The car drives 20 Extra hard layouts, drawn from the 38 layouts of the training circuits by the
   round's generator, for three laps each. After a crash it is placed
   back on the centerline, as during dataset collection, and continues, so that the whole lap is
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
<text x="360" y="314.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">ES of 550 grown from it</text>
<path d="M106,264 L106,306 L280,306" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<path d="M360,264 L360,290" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figdag-head)"/>
<text x="454" y="297" font-family="DejaVu Sans" font-size="9" fill="#5f6673">starts at Extra hard, f = 1.3;</text>
<text x="454" y="308" font-family="DejaVu Sans" font-size="9" fill="#5f6673">mastery goal first, then shorter targets</text>
<text x="454" y="319" font-family="DejaVu Sans" font-size="9" fill="#5f6673">(by evolution if DAgger stopped short)</text>
</svg>
<p><em>Figure 8. DAgger refinement: rounds of driving, expert labelling, aggregation, retraining and checking, until the goal or the patience stop passes the network to the speed push.</em></p>
</div>

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
of the imitation methods (§8.5). The ES uses the baseline architecture (973 weights) with
Glorot-uniform initial weights. The GA starts from networks without hidden layers, the 12 inputs connected directly
to the 45 outputs (585 weights and biases), with Glorot-uniform initial weights, and grows them (§9.6). The seeded ES and the speed pushes start from trained networks. NEAT
starts from minimal networks and grows them. Every car of these populations also carries an evolved setup (§9.11). Each approach is run four times, run 0 with seed 0 and runs 1–3 without a seed (§3.3). Runs have no
evaluation budget; they end through the speed push or the safety stop (§9.4).

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
(GA mutation strength or σ), the streak, the event of the generation (level advancement, mastery goal,
target reduction or end), the map seeds and the elapsed time.  Every run also logs the mean and standard deviation of the population's sensor length and steering sensitivity, NEAT logs its number of species and the
population's mean numbers of hidden nodes and enabled connections, and the GA logs the mean and largest number of
hidden layers, hidden units and parameters of its population. A
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

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 466" width="100%" role="img" aria-label="One generation: fresh maps, time trials, fitness, survival, children by method, checkpoint and the level or push decision" style="display:block;margin:10pt 0 6pt">
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
<text x="59" y="291.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330">GA<tspan font-size="9"> · 1,000 cars</tspan></text>
<text x="44" y="307" font-family="DejaVu Sans" font-size="9" fill="#5f6673">parents: two tournaments of 5</text>
<text x="44" y="318" font-family="DejaVu Sans" font-size="9" fill="#5f6673">crossover p 0.5 on shared weights</text>
<text x="44" y="329" font-family="DejaVu Sans" font-size="9" fill="#5f6673">mutation: each weight p 0.1, N(0, s)</text>
<text x="44" y="340" font-family="DejaVu Sans" font-size="9" fill="#5f6673">units ±0.03, layers ±0.01 per child</text>
<rect x="244" y="274" width="200" height="78" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<rect x="254" y="283" width="9" height="9" rx="1.5" fill="#009E73"/>
<text x="269" y="291.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330">ES<tspan font-size="9"> · 550 cars</tspan></text>
<text x="254" y="307" font-family="DejaVu Sans" font-size="9" fill="#5f6673">copy of one of the 50 fittest</text>
<text x="254" y="318" font-family="DejaVu Sans" font-size="9" fill="#5f6673">+ Gaussian step σ on every weight</text>
<text x="254" y="329" font-family="DejaVu Sans" font-size="9" fill="#5f6673">σ: 1/5 success rule while learning,</text>
<text x="254" y="340" font-family="DejaVu Sans" font-size="9" fill="#5f6673">then the 90% reliability rule</text>
<rect x="454" y="274" width="200" height="78" rx="2.5" fill="#ffffff" stroke="#5f6673" stroke-opacity="0.55" stroke-width="1"/>
<rect x="464" y="283" width="9" height="9" rx="1.5" fill="#F0E442" stroke="#8a7a00" stroke-width="0.8"/>
<text x="479" y="291.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330">NEAT<tspan font-size="9"> · 1,000 cars</tspan></text>
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
<path d="M220,428 L229.92,412 L370.08,412 L380,428 L370.08,444 L229.92,444 z" fill="#fff7e6" stroke="#b07a1a" stroke-opacity="0.9" stroke-width="1" stroke-linejoin="round"/>
<text x="300" y="431.6" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">Level / push decision</text>
<path d="M154,428 L220,428" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<path d="M380,428 L470,428" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<text x="425" y="423" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">run ends</text>
<rect x="470" y="412" width="150" height="32" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="545" y="424.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Final controller</text>
<text x="545" y="436.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">selection (§9.5)</text>
<path d="M300,444 L300,460 L10,460 L10,12 L34,12" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figgen-head)"/>
<text x="306" y="455" font-family="DejaVu Sans" font-size="9" fill="#5f6673">next generation · rules: the run's life figure (§9.4)</text>
</svg>
<p><em>Figure 9. One generation: three maps, time trials, survival, children by method, checkpoint and the level or push decision.</em></p>
</div>

### 9.4 Level advancement, speed push and safety stop

**Level advancement.** For each map of a generation, the success share is the fraction of the whole population that
succeeded on that map. A generation passes when the lowest success share among its maps is at least
90%. Three consecutive passing generations advance the population one level. A population on Extra
hard that passes three consecutive generations has reached the mastery goal. Each generation drives
three maps, so the three passing generations cover nine different maps.

**Speed push.** After the mastery goal, the target factor is multiplied by 0.98, which shortens the target time by 2%.
In the speed push a generation passes when the lowest map success share is at least 50%; each further run of three
consecutive passing generations shortens the target by another 2%. The run ends when 1,000 generations pass without
mastering the current target. Up to version 3.8 a speed-push generation needed 75% and the patience was 600
generations (300 up to version 3.5); the version 3.8 run showed that the controllers whose push went furthest were the
fastest, so version 3.9 lets the target keep tightening while the fastest half of the population follows it. The tightest target mastered is recorded as the run's speed result, expressed as the
reduction from the starting target. For example, f = 1.3 × 0.98⁴ is reported as a target 7.8% shorter than the
starting target. Up to version 3.1 the step was 5%, a speed-push generation needed 90% and the patience was 500
generations; every run then ended at a target that 90% of the population could not reach (version 3.2).

**Stall boost (version 3.6).** When 100 generations of the speed push pass without mastering the current target, the
push is stalled, and the parts of the cars that decide speed and turning mutate harder until a target is mastered
again; the sensing layers keep their usual mutation, so what the population has learned about reading the road
stays intact.
- **Decision layer:** the weights and biases of the output layer, which choose among the 45 combinations of steering
  and throttle (§6), mutate three times as strongly: the GA's mutation strength (on the output layer of each car's own network), the ES's step size σ (also in the
  speed pushes of the imitation methods) and NEAT's weight perturbation of connections into output nodes are
  multiplied by 3 for those parameters.
- **Steering sensitivity (§9.11):** a child's steering sensitivity mutates with probability 0.5 instead of 0.2 and with a
  standard deviation of 0.2 instead of 0.1 on the log scale (about ±20%); the sensor length mutates as usual.
- The boost depends only on the run's state, which its checkpoint holds, so run 0 still replays exactly. Each
  generation's log records whether the boost was on; the report counts the stalls each run broke out of.

**Safety stop.** On any level before the mastery goal, a run ends if 2,000 generations pass without a new best value of
the lowest map success share, averaged over three consecutive generations. Such a run is reported as
not having reached the level. Runs are expected to end through the speed push; the safety stop
prevents a run that cannot progress from continuing indefinitely.

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 472" width="100%" role="img" aria-label="A run's life: four levels, mastery goal, speed push with stall boost, safety stop and final controller selection" style="display:block;margin:10pt 0 6pt">
<defs><marker id="figrun-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker><marker id="figrun-head-boost" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#c0392b"/></marker></defs>
<rect x="0" y="0" width="665" height="112" rx="3" fill="#f5f6f8"/>
<rect x="170" y="124" width="495" height="258" rx="3" fill="#f5f6f8"/>
<text x="10" y="16" font-family="DejaVu Sans" font-size="10" fill="#1d2330"><tspan font-weight="bold">Climb the levels</tspan><tspan font-size="9" fill="#5f6673"> · every run starts at Easy, target factor f = 1.3</tspan></text>
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
<text x="80" y="155.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">2,000 generations</text>
<text x="80" y="167.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">without a new best?</text>
<path d="M80,112 L80,140" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<text x="84" y="128" font-family="DejaVu Sans" font-size="9" fill="#5f6673">any level</text>
<text x="22" y="196" font-family="DejaVu Sans" font-size="9" fill="#5f6673">new best: lowest map success</text>
<text x="22" y="207" font-family="DejaVu Sans" font-size="9" fill="#5f6673">share, mean of 3 generations</text>
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
<text x="475" y="389" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes: push ends</text>
<path d="M12,158 L12,415 L382,415" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figrun-head)"/>
<text x="18" y="240" font-family="DejaVu Sans" font-size="9" fill="#5f6673">yes: safety stop</text>
<text x="470" y="444" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">kept cars of the last generation drive the 20 validation maps (80–99)</text>
<text x="470" y="455" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">at the tightest mastered target (f = 1.3 if none);</text>
<text x="470" y="466" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">most successes wins, ties go to mean fitness</text>
</svg>
<p><em>Figure 10. A run's life: the four levels, the mastery goal, the speed push with its stall boost, and the two ways a run ends.</em></p>
</div>

### 9.5 Final controller

When a run ends, every kept car of the last generation drives the 20 validation maps as a time trial
at the tightest target the run mastered (f = 1.3 if it mastered none). The car with the most
successes is the run's final controller. Ties are broken by mean fitness, which favors the faster car.
Revision 2.0 selected by mean fitness alone, which allowed a fast car that crashed on one map to win
over a car that completed every map. The final controller is exported as a `.keras`
file at `results/champions/<approach>/seed<k>`, together with its float64 weights (`.npy`) for dense
networks or its genome (`.json`) for NEAT.

### 9.6 Genetic algorithm

The population size is 1,000. Every car carries its own network: a list of hidden layer sizes, ReLU hidden units, and
the weights and biases of each layer. In generation 0 no car has a hidden layer: each car's 12 inputs are connected
directly to its 45 outputs (585 weights and biases; Glorot-uniform weights, zero biases). From there the structure is
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
     the output biases are always shared. Between parents of the same architecture this is the uniform crossover of
     version 3.6.
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
       12 inputs for the first hidden layer), with identity weights and zero biases. Behind a hidden layer, whose ReLU
       outputs are never negative, the car drives exactly as before; directly behind the inputs, an input that can be
       negative, such as the position across the road, is cut at zero until mutation changes the new layer.
     - **Remove a layer** (p = 0.01): a random hidden layer is removed and the layers on either side are joined by the
       product of their weights (weights W₁W₂, biases b₁W₂ + b₂), which reproduces the network wherever the removed
       units were active.

  Adding and removing are equally likely, so mutation itself favours neither larger nor smaller networks.
- **Stall boost (§9.4):** the output layer of each child's own network mutates three times as strongly.
- **Evaluation:** the networks of a population differ in shape. The numpy forward pass groups the cars by
  architecture and runs one `einsum` per layer for each group, and the compiled simulator receives each car's own
  network program (§5.7), so no network is padded to a common shape and the two simulators still agree exactly.
- **Settings:** the structural rates and the empty starting architecture are recorded and checked only in GA runs
  (§3.3).
- **Up to version 3.6** the GA evolved the weights of the baseline architecture (463 weights) with the same selection,
  uniform crossover and weight mutation.

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

**Genome.** A genome consists of node genes and connection genes. Every genome has the 12 input nodes
and the 45 output nodes of the common interface. Hidden nodes are added by mutation. Each node except
the inputs carries a bias. Each connection gene holds its source and target nodes, a weight, an enabled
flag and an innovation number. The run keeps a record of structural mutations: a connection between
the same two nodes, or a split of the same connection, receives the same innovation number or node in
every genome in which it occurs, so that the genes of two parents can be aligned. Networks are feed-forward. A connection that would create a cycle is never added, so NEAT
networks, like all others, have no memory.

**Initial population.** Every input is connected to every output: 540 connections and 45 output biases.
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

<div style="break-inside:avoid;page-break-inside:avoid">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 665 436" width="100%" role="img" aria-label="Figure: NEAT genomes, from the minimal genome through the add-node and add-connection mutations to a grown genome, and species of similar genomes" style="display:block;margin:10pt 0 6pt">
<defs><marker id="figneat-head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L8,4 L0,7.4 z" fill="#5f6673"/></marker></defs>
<rect x="0" y="0" width="186" height="236" rx="3" fill="#f5f6f8"/>
<text x="9" y="18" font-family="DejaVu Sans" font-size="10" fill="#1d2330" font-weight="bold">Start: minimal genome</text>
<text x="50" y="39" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">12 inputs</text>
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
<text x="9" y="201" font-family="DejaVu Sans" font-size="9" fill="#5f6673">540 connections + 45 output biases</text>
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
<text x="213" y="224" font-family="DejaVu Sans" font-size="9" fill="#5f6673">by N(0, σ²), σ₀ = 0.5, or replaced p = 0.05</text>
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
<text x="60.2328" y="270" font-family="DejaVu Sans" font-size="9" fill="#5f6673">similar genomes, grouped by compatibility; population 1,000</text>
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
  numpy step has a fixed cost of about 0.45 ms plus about 2.3 µs per car, so a task should cover as many cars as
  possible on one track. Results are independent of the division and of the simulator, because cars do not interact.
- **Number of threads or workers:** the compiled simulator uses every available logical processor (§3.3); the
  number of worker processes of the numpy simulator is chosen with the generation-time benchmark in `bench.py`, which
  times the same evaluation path the training runs use, for both simulators, and recommends a count only for numpy.
  On the desktop, 1,000 trained cars take 16.55 s per generation on one thread, 2.90 s on 6 and 1.03 s on 24: the
  first 8 threads run on the 8 performance cores, the others on their second hardware threads and on the efficiency
  cores, so each later thread adds about half as much. With the compiled simulator, it times trained cars when
  a checkpoint of a dense population exists in `results/runs`: the cars of the newest such checkpoint, repeated or
  truncated to the population size, drive three laps of the fixed Extra hard layouts 0–2 (Austin in both directions
  and Budapest) against the starting target time (factor 1.3), as in training, since layouts repeat and a checkpoint
  records no map list. Without such a checkpoint, it times expert cars, which neither sense nor run a network
  and are therefore a lighter load; its output names the workload timed.
- **Keras training:** `steps_per_execution` (§8.1) is chosen with the Keras training-time benchmark in
  `bench.py`; XLA compilation is off for reliability (§8.1).
- **Fresh maps:** each generation builds its three maps in the process that drives them: the calling process with the
  compiled simulator, the worker processes with numpy; the three builds of a generation are sequential. With the
  compiled expert lap (§5.7), a map takes about 0.25 s to build on Easy and Medium, 0.8–1 s on Hard and 1–3 s on the
  validation setting. Every process keeps at most 8 built tracks, so memory stays constant over runs of many thousands of generations.
- **Sensor length:** tracing time grows with a ray's length, so a population that evolves long sensors drives each
  generation more slowly; from version 3.8 sensors start at 150 m, three times the earlier length, and may evolve up to
  1,000 m. Results do not depend on it.
- **Real circuits:** a real circuit's grids cover up to about 1.3 by 2.2 km (Monza), 11.5 million cells at 0.5 m;
  its distance map takes about 47 MB in float32. The grids of the 23 circuits are built once per machine, in 1–3 s
  each with a peak of about 1 GB, into `results/circuit_grids/` (about 570 MB), memory-mapped and shared by every
  process and both directions; files of an earlier code version are removed.
- **Memory per process:** the in-process cost is now the generated maps: a Hard map takes about 30 MB, a validation
  map up to 74 MB. An evolution process peaks at about 0.34 GB per generation on Easy and 0.69 GB on Hard, and at
  about 2.4 GB in the final selection, which holds all 20 validation maps. The `at_once` bound of each machine
  (§3.3) is set from the peaks measured on that machine before the reported run.
- **NEAT:** all genomes of a generation are compiled into flat arrays and evaluated together, depth by
  depth (§9.9), so NEAT drives its whole population at once like the other methods.
- **GA:** the cost of a generation grows with the networks the GA evolves. No size limit is set; the run log records
  the population's network size every generation (§9.1), and the report shows it (§11.4).
- **Further measures**, applied only if training is still too slow: fewer sphere-tracing steps, then
  shorter time limits.

GPU execution is not used: the per-operation launch cost on a GPU exceeds the arithmetic of networks
and car batches of this size.

### 9.11 Evolved car setup

Every car of an evolutionary population carries two setup values next to its network: its **sensor length** L
(30–1,000 m) and its **steering sensitivity** k (⅓–3). Both are stored and mutated on a logarithmic scale, so halving
and doubling are equally large changes. The setup carries no cost; it is judged only by the time trials, through
success and lap time.

- **Sensor length** limits all nine rays of the car (§5.2).
- **Steering sensitivity** multiplies the steering of every action class before the grip limit (§5.1). Below 1 it
  gives finer steering but a gentler sharpest turn; above 1 it cannot exceed the grip limit, strengthens the
  half-steering levels and saturates the full ones, so the steering becomes coarser.
- **Populations:** the GA, the ES, NEAT, the seeded ES and the speed push of each imitation method evolve their setup.
  The imitation training and DAgger use L = 150 m and k = 1, because the expert data and corrections are recorded
  with those sensors.
- **Initial setups:** every car starts with a sensor length of 150 m (version 3.8). Populations that start from random
  networks (GA, ES, NEAT) draw every car's steering sensitivity uniformly on the logarithmic scale within its range;
  populations that start from a trained network (seeded ES, speed pushes) start with k = 1, the setup the network was
  trained with. Up to version 3.7, random-start populations also drew the sensor length, from 10–150 m, and
  trained-start populations used 50 m.
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
  - The user selects an approach, a run (0, seeded, or 1) and a difficulty (§10.4).
  - For the GA, ES, seeded ES and NEAT, the user sets the population size before starting.
  - Training is started and paused with a play/pause control and ended with a stop control, which freezes the displayed cars. After a stop, the approach, run, difficulty and population size can be changed.
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
  and a reported run (`results/runs/`) of the selected approach and run.
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

- **Controllers:** the view loads whichever final controllers exist in `results/champions/`. A race control selects
  the runs that race: *seeded* (run 0 of every method), *unseeded* (runs 1–3), *best* (each method's best run on the
  validation maps) or *all* (every run, up to 32 cars); the default is *best*, and the choice is kept while the view
  is open. Each method also has an on/off switch. Every run of a method drives in its color; unseeded cars carry a
  white stripe, and the leaderboard names run 0 "<method> (seeded)" and the others "<method> r1" to "<method> r3".
  Controllers of versions up to 3.8 have 15 outputs; the 45-class action table would misread them. A results folder
  made with another action set, the current one or an archive, is not replayed, with a note naming its action count,
  and each of its controllers is left out of a race with a note.
- **Track:** at Extra hard, one of the four test circuits is selected with the arrow keys or on-screen buttons,
  in its racing direction, and its name and pole lap are shown. At the generated levels and the validation setting
  (§10.4), a display seed (1000–1019) is selected instead. The selected track is drawn as soon as it is selected,
  with the cars on the start line, and its name or seed and its level are shown.
- **Race:** the selected controllers race three laps without interacting, each in its assigned color.
  A start control begins the race, and a stop control ends it and returns to the map preview, after
  which the controllers, seed and difficulty can be changed.
- **Leaderboard:**
  - position, ranked by laps completed plus progress
  - last lap time and best lap time, and on a test circuit the best flying lap's difference to the pole lap
  - crash status

  A crashed car stays where it crashed and is drawn faded.
- **Playback speed:** 1×, 2×, 4× or 8×, where 1× is twice real time.
- **App cars:** a cycle button switches the controller list between the reported controllers and the app cars, the
  best cars of training started in the application (§10.6), labelled for example *GA (app) s1*. Each app car is off by
  default and can be switched on to race next to the reported controllers, drawn with a white border; app cars never
  enter the evaluation. When many cars race, the leaderboard shows as many rows as fit and states how many more there
  are.

### 10.4 Difficulty

The difficulty control maps five settings to the curriculum levels and the validation setting of §4.2.

*Table 4. Difficulty settings.*

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
within one time step.

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

Each final controller (eight methods, four runs) drives the four test circuits, Melbourne, Sakhir, Sepang and
Silverstone, in their racing direction as a three-lap time trial from a standing start. Three outcomes are recorded
per circuit: whether the car **finished** within the simulator's time limit of twice the reference lap time per lap,
whether it **succeeded** within the starting target (f = 1.3), and the time of every lap it completed. The
validation maps select each run's final controller (§9.5) and each method's best run, which the tables name: the run
whose final controller succeeds on the most validation maps, ties going to the higher mean validation fitness, then
to the lower run number. Every track is driven at its own widths (§4.2, §4.7).

**Pole laps (version 3.8).** A race's **flying lap** is its fastest complete lap after the first (lap 2 or 3), which
starts at speed like a qualifying lap. Each final controller's flying lap on a test circuit is compared with the
pole lap set there in qualifying (Table 5): a controller beats the pole lap when its flying lap is shorter. Lap
times are measured to the step, 0.05 s: a lap ends at the first step whose nearest centerline point is past the start
line, so a flying lap within 0.05 s of a pole lap does not resolve which is faster, and the tables print lap times to
two decimals. Each pole
lap is that of the last season run on the layout of the database: 2019, the year of its layouts, and 2017 for
Sepang, the last Formula 1 race there.

<div style="break-inside:avoid;page-break-inside:avoid">

*Table 5. Pole laps of the test circuits.*

| Circuit | Pole lap | Driver, season | Sources |
|---|---|---|---|
| Melbourne (Albert Park) | 1:20.486 | Lewis Hamilton, 2019 (also the lap record) | Formula1.com qualifying report |
| Sakhir (Bahrain) | 1:27.866 | Charles Leclerc, 2019 | FIA; Sky Sports timesheet |
| Sepang | 1:30.076 | Lewis Hamilton, 2017 | FIA pole position document |
| Silverstone | 1:25.093 | Valtteri Bottas, 2019 | Formula1.com qualifying report |

</div>

The comparison carries the car's simplification with it: the car has a Formula 1 car's grip and top speed but no
engine power or drag limit, so it gains speed on straights and out of slow corners faster than a real car (§14). The
report states this with every comparison.

**Robustness (version 3.2).** Each test circuit is also rebuilt with the same centerline at the constant widths 6,
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
<text x="382" y="43.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">most successes,</text>
<text x="382" y="55.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">then mean fitness</text>
<rect x="515" y="30" width="140" height="32" rx="2.5" fill="#edf2f7" stroke="#4f6d8c" stroke-opacity="0.6" stroke-width="1"/>
<text x="585" y="42.5" font-family="DejaVu Sans" font-size="10.5" fill="#1d2330" text-anchor="middle">Final controller</text>
<text x="585" y="54.5" font-family="DejaVu Sans" font-size="9" fill="#5f6673" text-anchor="middle">.keras + .npy or genome</text>
<path d="M130,46 L156,46" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<path d="M290,46 L317,46" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<path d="M447,46 L515,46" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<text x="156" y="79" font-family="DejaVu Sans" font-size="9" fill="#5f6673">time trial at the run's tightest mastered target (f = 1.3 if none)</text>
<path d="M585,62 L585,138" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<circle cx="585" cy="102" r="1.8" fill="#5f6673"/>
<path d="M585,102 L240,102 L240,138" fill="none" stroke="#5f6673" stroke-width="1" marker-end="url(#figwin-head)"/>
<text x="250" y="98" font-family="DejaVu Sans" font-size="9" fill="#5f6673">validation successes of the 4 runs</text>
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
<text x="161" y="197.7" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">most validation successes,</text>
<text x="161" y="209.5" font-family="DejaVu Sans" font-size="10" fill="#1d2330" text-anchor="middle">then mean fitness, then run 0</text>
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
  observed, which is reported as a result in its own right. Version 3.9 has 45 classes, so its accuracies are lower
  than and not comparable with those of earlier versions, which had 15.
- **Inference time:** *batch*, Keras `predict` on 10,000 test rows in a single batch, reported per sample
  and averaged over five runs, with a warm-up call identical to the timed calls; *single sample*, the
  median time of 1,000 calls of the Keras model on one row. Inference is timed after all parallel work has
  finished, with the same prediction settings for every controller (no XLA compilation, one batch per
  call).
- **Parameters:** the Keras parameter count; for NEAT, the number of enabled connections plus biases.
- **Training time:** baseline training per run; Keras Tuner search time (once per method) and training
  per run; DAgger and speed push per run; evolutionary run time per run; dataset collection once, as a
  shared cost. The run conditions recorded in `results/run_info.json` (§3.3) are reported with the
  results.

### 11.3 Tables

Results are reported for every run of each method and as their mean and sample standard deviation, written
`r0 · r1 · r2 · r3 (mean ± sd)`, with each method's best run named. They are led by completion and speed.

1. **Results.** Per method: test circuits finished and succeeded (per run and in total, of 16), mean and
   best lap time, mean three-lap race time, crashes, and the target reduction reached in the speed push, and the final controllers' sensor length and steering sensitivity.
2. **Pole laps (version 3.8).** Per method and run: the flying lap on each test circuit, its difference to the pole
   lap in seconds and in percent, and the number of test circuits on which the pole lap was beaten; per method a row
   "fastest run", the fastest flying lap on each circuit over its four runs (not the best run chosen on validation).
   Lap times are given to two decimals (0.05 s resolution).
3. **Head-to-head speed.** Mean lap time on the test circuits that every final controller finished.
4. **Training cost.** Per method and run: time and evaluations to clear each level, to reach the
   mastery goal and to end the speed push, and whether each was reached. For imitation methods, the
   Keras Tuner search time and the number and time of DAgger rounds are included.
5. **Imitation stages.** Test results of the imitation methods after initial training, after DAgger and
   after the speed push.
6. **Course comparison.** Test accuracy, parameters, batch inference time, single-sample latency and
   search and training time, as required by the course. Test accuracy is the share of rows of the expert-labelled test split for
   which the controller predicts the expert's class. Parameters are as in §11.2. Batch inference time is the
   Keras `predict` time per sample on 10,000 test rows, averaged over five runs; single-sample latency is the
   median of 1,000 calls on one row (§11.2). Search and training time is the training time of §11.2.
7. **Search cost, dataset and driving-selected controllers.** *Search cost:* per Keras Tuner method, the
   number of trials, the number of distinct configurations, the total epochs, the trials that reached the
   epoch limit and the search time (§8.2). *Dataset:* the rows of each split and the collection time (§7.3).
   *Driving-selected controllers:* per Keras Tuner method, the configuration chosen by validation driving
   (§8.3), its parameters, validation accuracy, test circuits completed (of 4), mean lap time and crashes.
8. **Robustness.** Per method and width of §11.1: the maps finished and succeeded (all runs) and the mean lap time on
   finished maps; each width states how many maps it covers.
9. **Evolved networks.** Per GA and NEAT run: the final controller's structure (for the GA its layer sizes, for
   example 12 → 9 → 45; for NEAT its hidden nodes and enabled connections) and its parameters (§11.2).

### 11.4 Figures

- **Success over training.** For each method, the lowest map success share per generation, with
  level advancements and target reductions marked.
- **Lap time over training.** The median lap time of the successful cars per generation.
- **Pole laps.** For each test circuit, the fastest flying lap of each method as a horizontal bar, with the pole lap
  marked by a vertical line.
- **Setup over training.** For each method, the population's mean sensor length and mean steering sensitivity per
  generation, with level advancements and target reductions marked.
- **Networks.** One panel per method showing its best final controller. The 12 inputs are on the left:
  the nine distance rays, labelled by angle, the speed, the position across the road and the road width. Hidden nodes are placed in columns by depth,
  so that the structure evolved by NEAT appears as it is. The 45 actions, labelled by steering and
  throttle, are on the right. Connections are blue when positive and red when negative, with width and
  opacity proportional to the weight's magnitude. Titles include the controller's setup.
- **Network size over training.** For the GA and NEAT, the population's mean number of hidden units (GA) or hidden
  nodes (NEAT) per generation, one line per run, with level advancements and target reductions marked.
- **Step sizes.** GA mutation strength and ES step size σ across generations, one line per run.
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
- **Parallel execution (§3.3):** the compiled simulator's default thread count equals the logical processors
  available to the process, `EC_RACING_THREADS` sets it and refuses anything but a positive integer, and
  `EC_RACING_CXX` selects the compiler; an evolution phase lowers every thread of its process to priority 19, threads
  started later inherit it, and it is never raised; a Keras job runs with 4 and 2 TensorFlow threads; a small search and the retraining of its choice, each run in a fresh process,
  produce the same weights as the same steps run in notebook order in one process; a run of a searched method waits
  for its search, and a baseline run trains its network, refines it and pushes it.
- **Runs 0 to 3 (§3.3):** an unseeded run sets no seed anywhere (its generators, Keras and DAgger are started without one);
  two starts of an unseeded run produce different populations; a stopped unseeded run resumes from its checkpoint on
  its own path; run 0 behaves as before; the tables show `r0 · r1 · r2 · r3 (mean ± sd)`; the race control races the
  seeded, the unseeded, the best or all runs.
- **Stall boost (§9.4):** the boost switches on after 100 generations without a mastered target and off at the next
  mastered target; while it is on, only the output layer's mutation is three times as strong (GA, ES and NEAT), the
  steering sensitivity mutates with probability 0.5 and standard deviation 0.2, and the sensor length is unchanged; a
  resumed run 0 with a stall replays exactly.
- **GA structure (§9.6):** generation 0 has no hidden layers; adding a unit, or a layer behind a hidden layer,
  leaves every output unchanged; removing a unit removes its weights, and removing a layer joins its neighbours by the
  product of their weights; crossover gives the child the shape parent's architecture and mixes only the values both
  parents have, and between equal architectures it is uniform crossover; the compiled simulator's results equal
  numpy's for populations of mixed architectures, including networks without hidden layers; checkpoints, subsets and
  saved controllers keep each car's architecture; a GA run of version 3.6 is refused while the other runs of version
  3.6 stay valid; `--archive NAME --runs` moves only the named runs.
- **Formula 1 car and real circuits (§4.7, §5.1):** the grip equals a(v) at 0, 69.4 and 105 m/s and never exceeds
  5 g; no car exceeds 105 m/s; each real circuit imports with the data's widths at every point and within 1% of its
  official length; the position input is −1 and +1 at the two edges of a road whose two widths differ; the test
  circuits are those drawn by `default_rng(0)` and appear in no training draw, validation map or training split of the
  dataset; Extra hard draws only the 38 training layouts; the expert completes every generated level, the validation
  setting and every real circuit in both directions; a race records each lap's time and its flying lap is its fastest
  lap after the first; the compiled simulator equals numpy on real circuits; rays reach 1,000 m.
- **Actions (§6, version 3.9):** 45 classes, `class = 5 × steer_index + throttle_index`, decode to the nine steering
  and five throttle levels and back; the expert's steering and throttle round to the nearest levels, its throttle is
  its speed error over its band, and it never accelerates at full lock; both simulators decode every class
  identically; a speed-push generation passes at 50% and the push ends after 1,000 generations without a mastered
  target; every network has 45 outputs (baseline 973 parameters, a new GA network 585, a new NEAT genome 540
  connections).
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
- **Car setup.** A ray stops at its car's length and a car of the starting length sees ray inputs from 0 to 1; steering sensitivity scales the steering levels within the grip limit; initial setups, inheritance and
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
the pilots, the survival and step-size settings are reviewed before the reported run starts. Version 3.6 adds a
stall-boost pilot: the ES run of seed 0 of 11–12 October, whose speed push stalled at a target 27.6% shorter, is
continued from its final population twice, with and without the boost, both with a patience of 600 generations; the
boost is kept if the boosted copy masters a shorter target than the plain one. Version 3.9 repeats the pilots of the GA, the ES and NEAT with 45 actions and the 50% speed push, in `results/pilots/revision3.9/`, to confirm that every level is still mastered and that the push goes further. Version 3.8 repeats the pilots of every evolutionary method from Easy with the Formula 1 car, the scaled levels
and the real circuits, to confirm that each level, Extra hard included, can still be mastered at 90%, and records the
sensor lengths the populations evolve. Version 3.7 repeats the GA pilot (`scripts/pilots.py ga`) with networks that start without hidden layers, to
confirm that the levels are still mastered and to see how the networks grow. Revision 3.1 repeats the pilots with setup evolution, in `results/pilots/revision3.1/`, and reports the populations' final mean sensor length and steering sensitivity.

## 13. Default Parameters

*Table 6. Default parameters, as defined in `racing/config.py`.*

| Parameter | Value |
|---|---|
| Time step | 0.05 s |
| Grip limit a(v) (friction circle) | min(17.66 + 0.00651 × v², 49.05) m/s²: 1.8 g at low speed, 5 g from 250 km/h |
| Top speed | 105 m/s (378 km/h) |
| Speed input scaling | 105 m/s (the top speed) |
| Sensors | 9 rays from −90° to +90°; length 150 m at the start, 30–1,000 m when evolved; inputs scaled by 150 m |
| Network inputs | 12: rays, speed, position across the road (−1 left edge, +1 right edge, by the local widths), local road width / 20 m |
| Road width | Generated: Table 1, a base width from `default_rng([seed, level, 13])` varied by up to ±20% along the lap (the mean of three sinusoids); real circuits: the data's left and right widths at every point |
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
| Survival | Successes kept; others kept with probability equal to their share of laps in time; at most 90% of the population; at least the 50 fittest |
| Level advancement and mastery goal | Lowest map success share ≥ 90% in three consecutive generations |
| Speed push | Target reduced by 2% per mastered target; a generation passes at a lowest map success share ≥ 50%; ends after 1,000 generations without mastering the current target |
| Stall boost | After 100 generations without a mastered target: output-layer mutation ×3; steering sensitivity mutation probability 0.5, standard deviation 0.2; until the next mastered target |
| Robustness table | Test circuits rebuilt with the same centerline at 6, 10, 14 and 18 m |
| Pole laps | Table 5; compared with each final controller's flying lap (lap 2 or 3), measured to 0.05 s |
| Safety stop | 2,000 generations without a new best three-generation mean of the lowest map success share |
| GA | N = 1,000; tournament 5; crossover p = 0.5 on shared values; mutation p = 0.1; s_max = 0.1; s_min = s_max / 100; no hidden layers at the start; per child add unit 0.03, remove unit 0.03, add layer 0.01, remove layer 0.01; no size limit; ReLU hidden units |
| ES, seeded ES and speed push | N = 550; children from the 50 fittest; σ₀ = 0.1; factor 0.82; σ in [1e-4, 1.0]; reliability phase at 90% |
| NEAT | N = 1,000; crossover 0.75; weight perturbation 0.8 (σ₀ = 0.5, two-phase rule as ES); weight replacement 0.05; add connection 0.05; add node 0.03; toggle 0.01; δ = E/N + D/N + 0.4 W̄; threshold from 3.0 towards 20 species; stagnation 20 generations; tanh hidden nodes |
| DAgger | 20 maps per round; at most 30 epochs per round, patience 5, 10% hold-out; goal 18 of 20 checks in three consecutive rounds; stall after 10 rounds |
| Keras training | Adam 1e-3; batch 256; at most 150 epochs; patience 5; `steps_per_execution` and `jit_compile` set by benchmark |
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
   Extra hard circuits reliably.
10. **Replication scope.** Each Keras Tuner search runs once, so the variation reported for the
   searched methods covers retraining of the selected architecture but not the search itself. The
   evolutionary methods repeat the whole run for each of the four runs.
11. **Sensor cost.** Long sensors cost tracing time; populations that evolve long sensors make generations slower,
    and from version 3.8 sensors start at 150 m and may reach 1,000 m, whose rays take up to 960 iterations, twenty times the 48 of the earlier 50 m sensors.
    This is accepted and included in the reported training cost.
12. **Coarse steering.** Populations may settle at a steering sensitivity above 1, where steering saturates and loses
    smoothness. This is a result to be reported, not a defect.
13. **Exactness of the compiled simulator.** Equality with numpy holds on each machine separately: machines may differ
    in the last bit of a mathematical function, as the numpy simulator's results already may, and a numpy release that
    changed its `tanh` kernel or the summation order of `einsum` or `bincount` would break equality. The check at load
    (§5.7) enforces the equality on each machine: on a mismatch, the run continues on the numpy simulator after one
    warning. The equivalence tests (§12.1) still run on each machine.
14. **Threads beyond physical cores.** Hyperthreads and efficiency cores add less than a full core each, so a run's
    generation time falls more slowly beyond the performance cores; runs side by side keep every processor busy
    (§3.3). A machine without a C++17 compiler runs the numpy simulator, with identical results.
15. **Stall boost.** A boost that is too strong can scatter a population that was close to the next target. The pilot
    (§12.2) decides whether the boost is kept, before the reported run.
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
    every comparison (§11.1).
20. **Small test set.** Four test circuits and four runs give 16 races per method, so differences between methods
    carry wide uncertainty; the per-run values are reported.
21. **Both directions in training.** The training circuits are driven in both directions for variety, the test
    circuits only in their racing direction. Training in the racing direction alone would match the test more closely
    but halve the variety; the choice can be reversed without other changes.
22. **Every method trains again.** Version 3.8 changes the car, the sensors, the inputs and the maps, so every
    network and every run is trained again, from the dataset on. The results of versions up to 3.7 belong to an
    earlier experiment with a slower car on smaller generated maps and are not mixed with these. Version 3.9 changes
    the actions, so every method trains again once more; the version 3.8 results are set aside (§3.3).
23. **Memory.** Real circuits' grids are built once per machine (about 1 GB at the peak, 570 MB on disk) and shared;
    the in-process cost is the generated maps, up to about 0.69 GB per evolution process on Hard and 2.4 GB in the final
    selection (§9.10). The `at_once` bounds of the machine plan are set from peaks measured on each machine before the
    reported run.
24. **A lower bar can outrun the population (version 3.9).** At 50%, more children miss the target each generation.
    The survival rule keeps every success and the 50 fittest cars, so the population is not lost; the pilots (§12.2)
    check that the push still advances.
25. **Longer runs.** A patience of 1,000 generations adds at least 400 generations to every speed push.
26. **Harder imitation.** Forty-five classes split the expert's data more finely, so each class has fewer examples
    and accuracy reads lower than with 15.
27. **Still discrete.** Forty-five choices every 0.05 s are much finer than 15 but not continuous; the laps of Table 0
    assume smooth control, so they remain a bound the controllers approach rather than reach.

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
- **Execution:** CPU only. Worker processes call `tf.config.set_visible_devices([], 'GPU')`. Each run's summary
  records the machine it finished on, and `results/run_info.json` the machine of the runner (§3.3).
- **Track data:** the racetrack database of the Technical University of Munich (`TUMFTM/racetrack-database`,
  LGPL-3.0) is vendored in `data/racetracks/` with its licence file and credited in the README and the report.
- **Version control:** a private GitHub repository; the `results/` directory is excluded.
