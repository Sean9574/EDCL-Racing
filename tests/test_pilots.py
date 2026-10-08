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
    assert isinstance(row["level_ups"], str) and 0.0 <= row["min_share_last20"] <= 1.0
    low, high = config.SENSOR_LENGTH_LIMITS
    assert low <= row["sensor_last"] <= high                    # the population's mean setup at the end
    low, high = config.STEER_GAIN_LIMITS
    assert round(low, 2) <= row["steer_last"] <= high
    assert (tmp_path / name / "log.csv").exists()


def test_revision_3_1_pilots_have_their_own_folder(pilots):
    assert pilots.OUT == config.PILOTS_DIR / "revision3.1"


def test_a_pilot_always_starts_over(pilots, tmp_path):
    pilots.pilot("ga", tmp_path, max_generations=2, n_jobs=1, pop_size=60)
    row = pilots.pilot("ga", tmp_path, max_generations=3, n_jobs=1, pop_size=60)
    assert row["generations"] == 3 and len(pd.read_csv(tmp_path / "ga" / "log.csv")) == 3


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
