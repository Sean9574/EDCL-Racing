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
    row = pilots.pilot_ga(0.2, tmp_path, generations=2, pop_size=20, kept=5, n_jobs=1)
    assert row["pilot"] == "GA s_max=0.2" and row["level_reached"] >= 1 and row["generations"] == 2
    assert (tmp_path / "ga_smax0.2" / "log.csv").exists()


def test_es_pilot_reports_step_size_and_ties(pilots, tmp_path):
    row = pilots.pilot_es(tmp_path, generations=2, mu=4, lam=16, n_jobs=1)
    assert row["pilot"] == "ES" and row["sigma_last"] > 0
    assert 0.0 <= row["tie_fraction"] <= 1.0 and 0.0 <= row["strict_success"] <= 1.0


def test_seeded_pilot_trains_its_own_baseline(pilots, monkeypatch, tmp_path):
    trained = []
    monkeypatch.setattr(pilots.config, "DATASET_PATH", tmp_path / "dataset.npz")
    (tmp_path / "dataset.npz").write_bytes(b"")
    monkeypatch.setattr(pilots.imitation, "train_baseline", lambda data, **kwargs: trained.append(kwargs))
    monkeypatch.setattr(pilots.imitation, "load_dataset", lambda path=None: {})
    monkeypatch.setattr(pilots.es, "trained_baseline",
                        lambda seed, champion_dir: nets.init_population(np.random.default_rng(0), 1)[0])
    row = pilots.pilot_seeded(tmp_path, generations=1, mu=4, lam=16, n_jobs=1)
    assert row["pilot"] == "Seeded ES" and trained[0]["champion_dir"] == tmp_path / "seeded" / "champions"
