"""DAgger refinement of the imitation controllers and their speed push (revision 3 §6.3-6.5).

A refinement runs in rounds. The car drives fresh Extra hard maps; at every step the expert labels the car's exact
state; the rows join the network's dataset and training continues; a time trial on further fresh maps checks the
result. Every round is checkpointed, and every round starts from a freshly loaded network after seeding Keras for
that round, so an interrupted refinement resumes where it stopped and ends exactly like an uninterrupted one."""
import json
import os
import time
from pathlib import Path

import numpy as np

from racing import checkpoint, config, es, expert, imitation, nets, paths, progress, sim
from racing.actions import class_to_controls, controls_to_class
from racing.backend import keras_cpu
from racing.curriculum import SeedSource
from racing.tracks import get_track

STATE_NAME = "dagger_state.pkl"
ROWS_NAME = "dagger_rows.npz"
SUMMARY_NAME = "dagger.json"


def drive_and_label(driver, seed, level=config.MAX_LEVEL):
    """One car driven by `driver` for config.LAPS laps of one map, with the expert's action recorded in every state
    the car reaches (revision 3 §6.3 steps 1-2). A crash puts the car back on the centerline, as during dataset
    collection, so the whole run is covered. Returns inputs, labels and the number of respawns."""
    track = get_track(seed, level)
    cars = sim.start_cars(track, 1, config.LAPS)
    time_limit = sim.time_limit_for(track, config.LAPS)
    inputs, labels, respawns = [], [], 0
    while not cars.done[0]:
        obs = sim.sense(track, cars)
        inputs.append(obs[0])
        labels.append(int(controls_to_class(*expert.expert_controls(track, cars))[0]))
        steer, throttle = class_to_controls(driver.act(obs))
        sim.step(track, cars, steer, throttle, time_limit)
        if cars.crashed[0] and cars.time < time_limit - config.TIME_EPS:
            respawns += expert.respawn(track, cars)
    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.int64), respawns


def _per_map(function, items, n_jobs):
    """function(*item) for every item, in order; maps are distributed across worker processes."""
    if n_jobs == 1 or len(items) <= 1:
        return [function(*item) for item in items]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(function)(*item) for item in items)


def collect(driver, seeds, level=config.MAX_LEVEL, n_jobs=1):
    """Labelled rows of every map in `seeds`, in seed order, so the rows do not depend on n_jobs."""
    parts = _per_map(drive_and_label, [(driver, seed, level) for seed in seeds], n_jobs)
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            int(sum(p[2] for p in parts)))


def _time_trial(driver, seed, level, factor):
    track = get_track(seed, level)
    result = sim.simulate(track, driver.policy(), 1, time_limit=sim.target_time(track, factor), laps=config.LAPS)
    return bool(result.finished[0])


def check(driver, seeds, factor=config.TARGET_FACTOR, level=config.MAX_LEVEL, n_jobs=1):
    """Number of maps on which the car completes the time trial: config.LAPS laps within the target time
    (revision 3 §4.1, §6.3 step 4)."""
    return int(sum(_per_map(_time_trial, [(driver, seed, level, factor) for seed in seeds], n_jobs)))


def _rng_seed(approach, seed):
    """Every method and run seed draws its own maps."""
    return [seed, config.IMITATION_APPROACHES.index(approach), config.DAGGER_SALT]


def _round_model(stage_file, weights, rate, seed, number):
    """The network with `weights`, freshly loaded and compiled after seeding Keras for round `number`: a round starts
    from the same state whether or not the refinement was interrupted before it."""
    keras = keras_cpu()
    keras.utils.set_random_seed(seed * 1000 + number)
    model = keras.models.load_model(stage_file, compile=False)
    model.set_weights(weights)
    return imitation.compile_model(model, rate)


def _train_round(model, x, y, rng):
    """Continue training on the aggregated rows (revision 3 §6.3 step 3): at most config.DAGGER_EPOCHS epochs with
    early stopping on a config.DAGGER_HOLDOUT hold-out. Returns the epochs trained."""
    order = rng.permutation(len(y))
    held = int(round(config.DAGGER_HOLDOUT * len(y)))
    val, train = order[:held], order[held:]
    history = model.fit(x[train], y[train], validation_data=(x[val], y[val]), epochs=config.DAGGER_EPOCHS,
                        batch_size=config.BATCH_SIZE, callbacks=[imitation.early_stopping()], verbose=0)
    return len(history.history["val_accuracy"])


def _load_rows(path, count):
    """The first `count` correction rows in `path`. Rows of a round interrupted before its checkpoint are dropped."""
    if count == 0 or not path.exists():
        return np.zeros((0, config.N_INPUTS), dtype=np.float32), np.zeros(0, dtype=np.int64)
    with np.load(path) as stored:
        return stored["X"][:count], stored["y"][:count]


def _save_rows(path, x, y):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as f:
        np.savez(f, X=x, y=y)
    os.replace(temporary, path)


def _save_model(model, path):
    """Save a Keras model so that an interruption cannot leave a partial file behind."""
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.keras")
    model.save(temporary)
    os.replace(temporary, path)


def _write_json(path, data):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2))
    os.replace(temporary, path)


def _finished_message(label, seed, state):
    rounds, best = state["round"], state["best_checks"]
    if state["outcome"] == "goal":
        return (f"{label} seed {seed}: DAgger goal reached in round {rounds} "
                f"({state['history'][-1]['checks']}/{config.DAGGER_MAPS} checks)")
    if state["outcome"] == "stalled":
        return (f"{label} seed {seed}: DAgger stopped after {rounds} rounds without a new best "
                f"(best {best}/{config.DAGGER_MAPS} checks); the speed push starts from the best network")
    return f"{label} seed {seed}: DAgger stopped at the round limit ({rounds} rounds, best {best}/{config.DAGGER_MAPS})"


def refine(approach, seed, data, runs_dir=None, n_jobs=1, on_round=None, max_rounds=None):
    """DAgger refinement of one trained imitation network (revision 3 §6.3). Returns the summary of a complete
    refinement recorded with the current settings, resumes an interrupted one, and otherwise starts from the
    imitation stage. Writes the refined network as the 'dagger' stage: the network of the final round when the goal
    is reached, otherwise the network with the best check result."""
    run = paths.run_dir(approach, seed, runs_dir)
    done = checkpoint.completed(run, name=SUMMARY_NAME)
    if done is not None:
        return done
    label = config.LABELS[approach]
    imitation_stage = paths.stage_path(approach, seed, "imitation", runs_dir)
    if not imitation_stage.exists():
        raise FileNotFoundError(f"{imitation_stage} not found: train the {label} network first (notebook 02)")
    state_path, rows_path = run / STATE_NAME, run / ROWS_NAME
    dagger_stage = paths.stage_path(approach, seed, "dagger", runs_dir)
    rate = float(json.loads((run / "summary.json").read_text()).get("learning_rate", config.LEARNING_RATE))
    state = checkpoint.load(state_path)
    if state is None:
        paths.remove(rows_path, dagger_stage)
        initial = keras_cpu().models.load_model(imitation_stage, compile=False).get_weights()
        state = {"settings": checkpoint.settings_record(), "round": 0, "streak": 0, "best_checks": -1,
                 "since_best": 0, "n_rows": 0, "respawns": 0,
                 "rng": np.random.default_rng(_rng_seed(approach, seed)).bit_generator.state, "used": [],
                 "weights": initial, "best_weights": initial, "history": [], "elapsed": 0.0, "outcome": None}
    else:
        checkpoint.check_settings(state["settings"])
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]
    source = SeedSource(rng, used=state["used"])
    x_rows, y_rows = _load_rows(rows_path, state["n_rows"])
    while state["outcome"] is None:
        if max_rounds is not None and state["round"] >= max_rounds:
            state["outcome"] = "max_rounds"
            break
        began, number = time.time(), state["round"] + 1
        model = _round_model(imitation_stage, state["weights"], rate, seed, number)
        x_new, y_new, respawns = collect(nets.from_keras(model), source.draw(config.DAGGER_MAPS), n_jobs=n_jobs)
        x_rows, y_rows = np.concatenate([x_rows, x_new]), np.concatenate([y_rows, y_new])
        _save_rows(rows_path, x_rows, y_rows)
        epochs = _train_round(model, np.concatenate([data["X_train"], x_rows]),
                              np.concatenate([data["y_train"], y_rows]), rng)
        checks = check(nets.from_keras(model), source.draw(config.DAGGER_MAPS), n_jobs=n_jobs)
        weights = model.get_weights()
        streak = state["streak"] + 1 if checks >= config.DAGGER_GOAL else 0
        if checks > state["best_checks"]:
            state.update(best_checks=checks, since_best=0, best_weights=weights)
        else:
            state["since_best"] += 1
        if streak >= config.DAGGER_STREAK:
            state["outcome"] = "goal"
        elif state["since_best"] >= config.DAGGER_PATIENCE:
            state["outcome"] = "stalled"
        elapsed = state["elapsed"] + time.time() - began
        record = {"round": number, "rows": int(len(y_new)), "respawns": int(respawns), "epochs": epochs,
                  "checks": checks, "streak": streak, "elapsed": elapsed}
        state.update(round=number, streak=streak, n_rows=int(len(y_rows)), respawns=state["respawns"] + respawns,
                     rng=rng.bit_generator.state, used=source.state(), weights=weights,
                     history=state["history"] + [record], elapsed=elapsed)
        checkpoint.save(state_path, state)
        progress.write_status({"approach": approach, "seed": seed, "stage": "dagger", "level": config.MAX_LEVEL,
                               "round": number, "checks": checks, "streak": streak, "elapsed": elapsed})
        if on_round is not None:
            on_round(record)
    final = state["weights"] if state["outcome"] == "goal" else state["best_weights"]
    model = keras_cpu().models.load_model(imitation_stage, compile=False)
    model.set_weights(final)
    _save_model(model, dagger_stage)
    history = state["history"]
    summary = {"approach": approach, "seed": seed, "settings": checkpoint.settings_record(),
               "rounds": state["round"], "goal_reached": state["outcome"] == "goal", "outcome": state["outcome"],
               "best_checks": state["best_checks"], "final_checks": history[-1]["checks"] if history else None,
               "correction_rows": state["n_rows"], "respawns": state["respawns"], "wall_time": state["elapsed"],
               "stage_path": str(dagger_stage), "history": history}
    _write_json(run / SUMMARY_NAME, summary)
    paths.remove(state_path)
    progress.milestone(_finished_message(label, seed, state))
    return summary


def speed_push(approach, seed, runs_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1, on_generation=None,
               max_generations=None):
    """Evolution strategy grown from the refined network, from Extra hard on (revision 3 §6.4). Its final controller
    is the method's final controller; the run itself resumes and skips like every evolutionary run."""
    model = keras_cpu().models.load_model(paths.stage_path(approach, seed, "dagger", runs_dir), compile=False)
    hidden, activation = nets.dense_spec(model)
    return es.run_es(seed=seed, out_dir=paths.speed_dir(approach, seed, runs_dir), champion_dir=champion_dir,
                     n_jobs=n_jobs, on_generation=on_generation, start_level=config.MAX_LEVEL,
                     max_generations=max_generations, initial=nets.to_flat(model), approach=approach,
                     hidden=hidden, activation=activation)


def run(approach, seed, data, runs_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1):
    """Refinement, then speed push: one counted run of revision 3 §8.3."""
    refined = refine(approach, seed, data, runs_dir=runs_dir, n_jobs=n_jobs)
    pushed = speed_push(approach, seed, runs_dir=runs_dir, champion_dir=champion_dir, n_jobs=n_jobs)
    return {"approach": approach, "seed": seed, "dagger": refined, "speed": pushed}
