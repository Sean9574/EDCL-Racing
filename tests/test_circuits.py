import os
import subprocess
import sys

import numpy as np
import pytest

from racing import circuits, config, tracks
from racing.track import build_track

USED = config.TRAINING_CIRCUITS + config.TEST_CIRCUITS


def test_the_database_holds_25_circuits_18_of_them_formula_1():
    files = sorted(p.stem for p in circuits.DATA_DIR.glob("*.csv"))
    assert files == sorted(config.F1_CIRCUITS + config.OTHER_CIRCUITS) and len(files) == 25
    assert len(config.F1_CIRCUITS) == 18 and len(config.OTHER_CIRCUITS) == 7
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in (circuits.DATA_DIR.parent / "LICENSE").read_text().splitlines()[0]


def test_the_test_circuits_are_drawn_with_seed_0_from_the_formula_1_circuits():
    drawn = np.random.default_rng(0).choice(sorted(config.F1_CIRCUITS), 4, replace=False)
    assert sorted(drawn.tolist()) == list(config.TEST_CIRCUITS) == ["Melbourne", "Sakhir", "Sepang", "Silverstone"]


def test_training_holds_the_other_formula_1_circuits_and_the_complex_others():
    assert not set(config.TRAINING_CIRCUITS) & set(config.TEST_CIRCUITS)
    f1 = [c for c in config.F1_CIRCUITS if c not in config.TEST_CIRCUITS]
    assert config.TRAINING_CIRCUITS == tuple(f1) + ("BrandsHatch", "MoscowRaceway", "Nuerburgring", "Oschersleben",
                                                    "Zandvoort")
    assert len(config.TRAINING_CIRCUITS) == 19 and circuits.N_LAYOUTS == 38


def test_the_others_in_training_are_at_least_as_complex_as_monza():
    """Spec §4.7, Table 1e: at least 8 corners, 1.38 corners per km and 172° of turning per km, Monza's values."""
    raw = {name: np.column_stack(circuits.read(name)[:2]) for name in ("Monza",) + config.OTHER_CIRCUITS}
    monza = circuits.complexity(raw["Monza"])
    assert monza["corners"] == 8 and round(monza["corners_per_km"], 2) == 1.38 and round(monza["turning_per_km"]) == 172
    for name in config.OTHER_CIRCUITS:
        c = circuits.complexity(raw[name])
        complex_enough = (c["corners"] >= monza["corners"] and c["corners_per_km"] >= monza["corners_per_km"]
                          and c["turning_per_km"] >= monza["turning_per_km"])
        assert complex_enough == (name in config.TRAINING_CIRCUITS), name
    assert "IMS" not in config.TRAINING_CIRCUITS and "Norisring" not in config.TRAINING_CIRCUITS


@pytest.mark.parametrize("name", USED)
def test_each_circuit_imports_at_real_scale_with_its_widths(name):
    points, left, right = circuits.outline(name)
    assert abs(circuits.perimeter(points) / 1000.0 / circuits.OFFICIAL_LENGTH[name] - 1.0) < 0.01
    x, y, data_right, data_left = circuits.read(name)
    assert np.allclose(points[0], [x[0], y[0]])                      # the start is the database's first point
    assert np.isclose(left[0], data_left[0]) and np.isclose(right[0], data_right[0])
    assert left.min() > data_left.min() - 0.5 and left.max() < data_left.max() + 0.5
    assert right.min() > data_right.min() - 0.5 and right.max() < data_right.max() + 0.5
    steps = np.hypot(*np.diff(points, axis=0).T)
    assert np.all(np.abs(steps - config.CENTERLINE_SPACING) < 0.05)   # about 1 m apart


def test_racing_direction_is_the_databases_order_and_known():
    for name in config.F1_CIRCUITS + config.OTHER_CIRCUITS:
        assert circuits.rotation(circuits.outline(name)[0]) == circuits.ROTATION[name], name
    assert {circuits.ROTATION[c] for c in config.TEST_CIRCUITS} == {"clockwise"}
    assert [c for c in config.F1_CIRCUITS if circuits.ROTATION[c] == "counter-clockwise"] == [
        "Austin", "SaoPaulo", "YasMarina"]


def test_reversal_keeps_the_start_swaps_the_sides_and_the_rotation():
    points, left, right = circuits.outline("Silverstone")
    back, back_left, back_right = circuits.reverse_outline(points, left, right)
    assert np.array_equal(back[0], points[0]) and np.array_equal(back[1], points[-1])
    assert np.array_equal(back_left[1:], right[:0:-1]) and np.array_equal(back_right[1:], left[:0:-1])
    assert circuits.rotation(back) == "counter-clockwise"


def test_layouts_are_every_training_circuit_both_ways():
    seen = [circuits.layout(k) for k in range(circuits.N_LAYOUTS)]
    assert seen[0] == (config.TRAINING_CIRCUITS[0], False) and seen[1] == (config.TRAINING_CIRCUITS[0], True)
    assert sorted(seen) == sorted((c, r) for c in config.TRAINING_CIRCUITS for r in (False, True))
    with pytest.raises(ValueError):
        circuits.layout(circuits.N_LAYOUTS)


def test_the_pole_laps_of_the_test_circuits():
    assert {c: lap.seconds for c, lap in circuits.POLE_LAPS.items()} == {
        "Melbourne": 80.486, "Sakhir": 87.866, "Sepang": 90.076, "Silverstone": 85.093}
    assert {lap.season for lap in circuits.POLE_LAPS.values()} == {2019, 2017}
    assert circuits.NAMES["SaoPaulo"] == "São Paulo" and circuits.NAMES["Nuerburgring"] == "Nürburgring"


# ----- real circuits as Extra hard and the test circuits (spec §4.6, §4.7, §9.10) -----

def test_test_circuits_start_at_the_data_start_in_racing_direction():
    for index, name in enumerate(config.TEST_CIRCUITS):
        track = tracks.get_track(index, config.TEST_LEVEL)
        x, y, _, _ = circuits.read(name)
        assert track.name == name and track.level == config.TEST_LEVEL and track.seed == index
        assert np.allclose(track.centerline[0], [x[0], y[0]], atol=0.6)
        heading = np.arctan2(y[1] - y[0], x[1] - x[0])
        assert np.cos(track.start_heading - heading) > 0.95                 # facing the data's next point
        assert circuits.rotation(track.centerline) == "clockwise"
        assert abs(track.length / 1000.0 / circuits.OFFICIAL_LENGTH[name] - 1.0) < 0.01
        assert np.isfinite(track.expert_lap_time) and track.expert_lap_time > 0


def test_extra_hard_is_the_38_training_layouts():
    for k in (0, 1, 37):
        name, reverse = circuits.layout(k)
        track = tracks.get_track(k, config.MAX_LEVEL)
        assert track.name == name and track.seed == k
        assert circuits.rotation(track.centerline) == (
            circuits.ROTATION[name] if not reverse else
            {"clockwise": "counter-clockwise", "counter-clockwise": "clockwise"}[circuits.ROTATION[name]])
    assert 4 not in config.LEVELS
    with pytest.raises(ValueError):
        tracks.get_track(38, config.MAX_LEVEL)


def test_both_directions_share_one_memory_mapped_grid():
    forward = circuits.circuit_track("Monza")
    backward = circuits.circuit_track("Monza", reverse=True)
    assert isinstance(forward.dist, np.memmap) and forward.dist.dtype == np.float32
    assert np.shares_memory(forward.dist, backward.dist) or forward.dist.filename == backward.dist.filename
    assert not forward.dist.flags.writeable
    assert forward.dist.nbytes < 60e6                                       # the largest circuit, about 47 MB
    files = list(config.CIRCUIT_GRIDS_DIR.glob(f"Monza-{circuits.grid_id('Monza')}*"))
    assert len(files) == 2                                                  # the grid and its origin


def test_a_changed_identity_builds_a_new_grid_file(monkeypatch):
    circuits.circuit_track("Oschersleben")
    old = circuits.grid_id("Oschersleben")
    monkeypatch.setattr(config, "GRID_MARGIN", config.GRID_MARGIN + 1.0)
    assert circuits.grid_id("Oschersleben") != old
    track = circuits.circuit_track("Oschersleben")
    assert str(track.dist.filename).endswith(f"Oschersleben-{circuits.grid_id('Oschersleben')}.npy")


def test_a_real_track_matches_a_fresh_build_of_its_outline():
    points, left, right = circuits.outline("Spielberg")
    fresh = build_track(points, left, right, seed=0, level=config.MAX_LEVEL, name="Spielberg")
    cached = circuits.circuit_track("Spielberg")
    assert np.array_equal(np.asarray(cached.dist), fresh.dist) and np.array_equal(cached.origin, fresh.origin)
    assert np.array_equal(cached.half_left, fresh.half_left) and np.array_equal(cached.centerline, fresh.centerline)


def test_worker_processes_find_the_grid_folder_in_the_environment():
    """A worker process imports config afresh: EC_RACING_CIRCUIT_GRIDS names the folder, results/circuit_grids by
    default (spec §9.10); this session's folder is passed on so."""
    def folder(environ):
        return subprocess.run([sys.executable, "-c", "from racing import config; print(config.CIRCUIT_GRIDS_DIR)"],
                              cwd=config.PROJECT_DIR, env=environ, capture_output=True, text=True,
                              check=True).stdout.strip()
    others = {key: value for key, value in os.environ.items() if key != "EC_RACING_CIRCUIT_GRIDS"}
    assert folder({**others, "EC_RACING_CIRCUIT_GRIDS": "/elsewhere/grids"}) == "/elsewhere/grids"
    assert folder(others) == str(config.RESULTS_DIR / "circuit_grids")
    assert os.environ["EC_RACING_CIRCUIT_GRIDS"] == str(config.CIRCUIT_GRIDS_DIR)


@pytest.mark.slow
def test_the_expert_drives_every_real_circuit_both_ways():
    for k in range(circuits.N_LAYOUTS):
        assert tracks.get_track(k, config.MAX_LEVEL).expert_lap_time > 0, circuits.layout(k)
    for k in config.TEST_TRACKS:
        assert tracks.get_track(k, config.TEST_LEVEL).expert_lap_time > 0


def fake_grid(monkeypatch, tmp_path):
    """A circuit's shape, and a road_grid that builds a tiny grid and counts its calls; grids go to tmp_path."""
    monkeypatch.setattr(config, "CIRCUIT_GRIDS_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(circuits, "road_grid", lambda *a: calls.append(1) or (np.zeros(2), np.zeros((2, 2), np.float32)))
    points, left, right = circuits.outline("Oschersleben")
    return circuits.track_shape(points, left, right), calls


def test_a_grid_missing_its_origin_file_is_rebuilt(monkeypatch, tmp_path):
    shape, calls = fake_grid(monkeypatch, tmp_path)
    circuits._grid("Oschersleben", shape)
    circuits._grid("Oschersleben", shape)
    assert len(calls) == 1                                                  # complete: reused
    (tmp_path / f"Oschersleben-{circuits.grid_id('Oschersleben')}.origin.npy").unlink()
    circuits._grid("Oschersleben", shape)
    assert len(calls) == 2 and len(list(tmp_path.glob("Oschersleben-*"))) == 2


def test_grids_of_other_identities_are_removed_once_a_day_old(monkeypatch, tmp_path):
    shape, calls = fake_grid(monkeypatch, tmp_path)
    old_time = os.path.getmtime(__file__) - 2 * circuits.OLD_GRID_AGE
    stale = [tmp_path / "Oschersleben-0123456789abcdef.npy", tmp_path / "Oschersleben-0123456789abcdef.origin.npy",
             tmp_path / ".Oschersleben-0123456789abcdef.npy.1.2.tmp"]
    young = tmp_path / "Oschersleben-fedcba9876543210.npy"
    other = tmp_path / "Monza-0123456789abcdef.npy"
    for path in stale + [young, other]:
        path.write_bytes(b"x")
    for path in stale + [other]:
        os.utime(path, (old_time, old_time))
    circuits._grid("Oschersleben", shape)
    assert not any(path.exists() for path in stale)                         # other identities, a day old
    assert young.exists() and other.exists()                                # too young; another circuit
    assert len(list(tmp_path.glob(f"Oschersleben-{circuits.grid_id('Oschersleben')}*"))) == 2
