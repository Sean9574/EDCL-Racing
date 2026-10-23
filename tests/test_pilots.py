import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from racing import config, nets, population, tracks

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "pilots.py"


@pytest.fixture
def pilots(monkeypatch, circle_track):
    """The pilots script with every map replaced by the small circle, so a pilot of two generations takes seconds."""
    def fake(seed, level):
        return circle_track

    fake.cache_clear = lambda: None          # population.fresh_workers empties tracks.get_track's cache
    monkeypatch.setattr(population, "get_track", fake, raising=False)
    monkeypatch.setattr(tracks, "get_track", fake)
    spec = importlib.util.spec_from_file_location("pilots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_level_ups_lists_each_level_reached_and_the_goal(pilots):
    log = pd.DataFrame({"generation": [0, 1, 2, 3], "level": [1, 1, 2, 4],
                        "event": [None, "level_up", "level_up", "goal"]})
    assert pilots.level_ups(log) == "2@1 3@2 goal@3"
    assert pilots.level_ups(log.assign(event=None)) == "none"


@pytest.mark.parametrize("name", ["ga", "es", "neat"])
def test_a_short_pilot_summarizes_its_run(pilots, tmp_path, name):
    row = pilots.pilot(name, tmp_path, max_generations=2, n_jobs=1, pop_size=60)
    assert row["generations"] == 2 and row["level_reached"] >= 1
    log = pd.read_csv(tmp_path / name / "log.csv")
    for key, column in (("mean hidden units", "mean_nodes"), ("largest parameters", "max_params")):
        if column in log:                                       # the network size at the last generation
            assert row[key] == pytest.approx(log[column].iloc[-1], abs=0.5)
        else:
            assert row[key] is None
    assert row["target_reduction"] is None or 0.0 <= row["target_reduction"] <= 100.0   # last speed-push target, %
    assert isinstance(row["level_ups"], str) and 0.0 <= row["min_share_last20"] <= 1.0
    low, high = config.SENSOR_LENGTH_LIMITS
    assert low <= row["sensor_last"] <= high                    # the population's mean setup at the end
    low, high = config.STEER_GAIN_LIMITS
    assert round(low, 2) <= row["steer_last"] <= high
    assert (tmp_path / name / "log.csv").exists()


def test_revision_3_9_pilots_have_their_own_folder(pilots):
    assert pilots.OUT == config.PILOTS_DIR / "revision3.9"


def test_a_pilot_always_starts_over(pilots, tmp_path):
    pilots.pilot("ga", tmp_path, max_generations=2, n_jobs=1, pop_size=60)
    row = pilots.pilot("ga", tmp_path, max_generations=3, n_jobs=1, pop_size=60)
    assert row["generations"] == 3 and len(pd.read_csv(tmp_path / "ga" / "log.csv")) == 3


def test_last_target_is_percent_shorter_or_blank(pilots):
    assert pilots.last_target({"target_reduction": 0.4375}) == 43.8
    assert pilots.last_target({"target_reduction": None}) is None and pilots.last_target({}) is None


def test_the_seeded_pilot_is_skipped_without_the_dagger_baseline(pilots, tmp_path, monkeypatch):
    monkeypatch.setattr(pilots.paths, "stage_path",
                        lambda approach, seed, stage, runs_dir=None: tmp_path / "missing.keras")
    row = pilots.pilot("seeded", tmp_path, max_generations=2, n_jobs=1, pop_size=60)
    assert row["pilot"] == "Seeded ES" and row["note"].startswith("skipped")
    assert not (tmp_path / "seeded").exists()


def test_the_seeded_pilot_starts_from_the_dagger_baseline(pilots, tmp_path, monkeypatch):
    baseline = tmp_path / "dagger.keras"
    baseline.write_bytes(b"")
    monkeypatch.setattr(pilots.paths, "stage_path", lambda approach, seed, stage, runs_dir=None: baseline)
    initial = nets.init_population(np.random.default_rng(0), 1)[0]
    monkeypatch.setattr(pilots.es, "dagger_baseline", lambda seed, runs_dir=None: initial)
    row = pilots.pilot("seeded", tmp_path, max_generations=2, n_jobs=1, pop_size=60)
    assert row["pilot"] == "Seeded ES" and row["generations"] == 2


def test_main_appends_a_summary_and_rejects_unknown_pilots(pilots, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pilots, "OUT", tmp_path)
    monkeypatch.setattr(pilots, "pilot", lambda name, out_root, generations: {"pilot": name, "generations": generations})
    assert pilots.main(["ga", "--generations", "5"]) == 0
    assert pilots.main(["neat"]) == 0
    text = (tmp_path / "summary.md").read_text()
    assert "| ga | 5 |" in text and "| neat | 300 |" in text and text.count("## ") == 2
    assert pilots.main(["bogus"]) == 2 and "bogus" in capsys.readouterr().err


BOOST_SCRIPT = SCRIPT.with_name("boost_pilot.py")


@pytest.fixture
def boost_pilot(pilots, monkeypatch):
    """The stall-boost pilot script, with every map the small circle (the `pilots` fixture), one worker and no
    simulation. The three settings its plain variant overrides are restored after the test."""
    from tests.test_evolution import fake_evaluate
    monkeypatch.setattr(config, "N_JOBS", 1)
    for name in ("STALL_BOOST", "STALL_STEER_P", "STALL_STEER_SD"):
        monkeypatch.setattr(config, name, getattr(config, name))
    monkeypatch.setattr(population, "evaluate_population", fake_evaluate(0.0))
    spec = importlib.util.spec_from_file_location("boost_pilot", BOOST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def finished_source(boost_pilot, tmp_path):
    """A small finished ES run (population 20) whose speed push ended: its checkpoint's tracker is done."""
    from racing import checkpoint, es, evolution
    folder = tmp_path / "source"
    evolution.run(es.ES(pop_size=20, hidden=(4,)), 0, folder, champion_dir=None, max_generations=3)
    state = checkpoint.load(folder / checkpoint.CHECKPOINT_NAME)
    state["tracker"].update(goal_reached=True, mastered_factor=0.9 * config.TARGET_FACTOR,
                            factor=0.9 * config.TARGET_FACTOR * config.SPEED_STEP, outcome="speed_limit",
                            since_mastered=config.SPEED_PATIENCE)
    checkpoint.save(folder / checkpoint.CHECKPOINT_NAME, state)
    return folder / checkpoint.CHECKPOINT_NAME, state


def run_variants(boost_pilot, source, out, generations=3):
    for variant in ("boost", "plain"):
        assert boost_pilot.main([variant, "--source", str(source), "--out", str(out),
                                 "--max-generations", str(generations)]) == 0


def test_the_pilot_continues_a_finished_run_with_and_without_the_boost(boost_pilot, finished_source, tmp_path):
    from racing import checkpoint
    source, state = finished_source
    run_variants(boost_pilot, source, tmp_path / "pilots")
    for variant in ("boost", "plain"):
        after = checkpoint.load(tmp_path / "pilots" / variant / checkpoint.CHECKPOINT_NAME)
        assert after["generation"] == state["generation"] + 3
        assert after["tracker"]["outcome"] is None          # stopped by max_generations, not by the push ending
        assert after["tracker"]["since_mastered"] == 3 and after["tracker"]["goal_reached"]
        assert after["tracker"]["mastered_factor"] == state["tracker"]["mastered_factor"]
        assert after["start_level"] == state["start_level"]
        log = pd.read_csv(tmp_path / "pilots" / variant / "log.csv")
        assert list(log["generation"]) == [state["generation"] + k for k in range(3)]
    summary = (tmp_path / "pilots" / "summary.md").read_text()
    assert "boost" in summary and "plain" in summary and summary.count("## ") == 2


def test_the_plain_variant_makes_the_draws_of_a_run_without_the_boost(boost_pilot, finished_source, tmp_path,
                                                                      monkeypatch):
    from racing import checkpoint
    source, _ = finished_source
    monkeypatch.setattr(config, "STALL_GENERATIONS", 1)       # every generation of the pilot is a stall
    run_variants(boost_pilot, source, tmp_path / "pilots")
    boosted = pd.read_csv(tmp_path / "pilots" / "boost" / "log.csv")
    plain = pd.read_csv(tmp_path / "pilots" / "plain" / "log.csv")
    assert boosted["boost"].tolist() == [1, 1, 1] and plain["boost"].tolist() == [1, 1, 1]
    monkeypatch.setattr(config, "STALL_GENERATIONS", 10 ** 6)         # no stall, so the old code path
    boost_pilot.main(["plain", "--source", str(source), "--out", str(tmp_path / "off"), "--max-generations", "3"])
    a = checkpoint.load(tmp_path / "pilots" / "plain" / checkpoint.CHECKPOINT_NAME)["method"]
    b = checkpoint.load(tmp_path / "off" / "plain" / checkpoint.CHECKPOINT_NAME)["method"]
    assert np.array_equal(a["weights"], b["weights"]) and np.array_equal(a["setup"], b["setup"])
    c = checkpoint.load(tmp_path / "pilots" / "boost" / checkpoint.CHECKPOINT_NAME)["method"]
    assert not np.array_equal(a["weights"], c["weights"])           # the boost itself does change the children


def test_the_pilot_names_a_missing_field_of_an_older_checkpoint(boost_pilot, finished_source, tmp_path):
    from racing import checkpoint
    source, state = finished_source
    del state["start_level"]
    checkpoint.save(source, state)
    with pytest.raises(SystemExit, match="start_level"):
        boost_pilot.main(["boost", "--source", str(source), "--out", str(tmp_path / "pilots")])
