import importlib.util
from pathlib import Path

import numpy as np
import pytest

from racing import nets, population

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "pilots.py"


@pytest.fixture
def pilots(monkeypatch, circle_track):
    monkeypatch.setattr(population, "get_track", lambda seed, level: circle_track)
    spec = importlib.util.spec_from_file_location("pilots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ga_pilot_summarizes_a_short_run(pilots, tmp_path):
    row = pilots.pilot_ga(0.2, tmp_path, generations=2, pop_size=20, n_jobs=1)
    assert row["pilot"] == "GA s_max=0.2" and row["level_reached"] >= 1 and row["generations"] == 2
    assert 0.0 <= row["min_share_last20"] <= 1.0 and (tmp_path / "ga_smax0.2" / "log.csv").exists()


def test_a_pilot_always_starts_afresh(pilots, tmp_path):
    pilots.pilot_ga(0.2, tmp_path, generations=1, pop_size=20, n_jobs=1)
    assert pilots.pilot_ga(0.2, tmp_path, generations=2, pop_size=20, n_jobs=1)["generations"] == 2


def test_es_pilot_reports_the_step_size(pilots, tmp_path):
    row = pilots.pilot_es(tmp_path, generations=2, pop_size=20, n_jobs=1)
    assert row["pilot"] == "ES" and row["sigma_last"] > 0 and 0.0 <= row["min_share_last20"] <= 1.0


def test_seeded_pilot_trains_its_own_baseline(pilots, monkeypatch, tmp_path):
    trained = []
    monkeypatch.setattr(pilots.config, "DATASET_PATH", tmp_path / "dataset.npz")
    (tmp_path / "dataset.npz").write_bytes(b"")
    monkeypatch.setattr(pilots.imitation, "train_baseline", lambda data, **kwargs: trained.append(kwargs))
    monkeypatch.setattr(pilots.imitation, "load_dataset", lambda path=None: {})
    monkeypatch.setattr(pilots.es, "trained_baseline",
                        lambda seed, champion_dir: nets.init_population(np.random.default_rng(0), 1)[0])
    row = pilots.pilot_seeded(tmp_path, generations=1, pop_size=20, n_jobs=1)
    assert row["pilot"] == "Seeded ES" and trained[0]["champion_dir"] == tmp_path / "seeded" / "champions"
