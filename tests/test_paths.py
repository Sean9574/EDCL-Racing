import pytest

from racing import config, paths


def test_run_and_search_directories(tmp_path):
    assert paths.run_dir("ga", 2, tmp_path) == tmp_path / "ga" / "seed2"
    assert paths.search_dir("kt_random", tmp_path) == tmp_path / "kt_random" / "search"


def test_default_directories_follow_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "CHAMPIONS_DIR", tmp_path / "champions")
    assert paths.run_dir("es", 0) == tmp_path / "runs" / "es" / "seed0"
    assert paths.champion_path("es", 1) == tmp_path / "champions" / "es" / "seed1.keras"
    assert paths.best_seeds_path() == tmp_path / "champions" / "best_seeds.json"


def test_champion_path_suffix(tmp_path):
    assert paths.champion_path("ga", 0, ".npy", tmp_path) == tmp_path / "ga" / "seed0.npy"


def test_remove_files_directories_and_missing(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "d" / "e").mkdir(parents=True)
    (tmp_path / "d" / "e" / "f.txt").write_text("y")
    paths.remove(tmp_path / "a.txt", tmp_path / "d", tmp_path / "missing.txt")
    assert list(tmp_path.iterdir()) == []


def test_seeded_es_is_an_approach_with_its_own_color():
    assert config.APPROACHES[-1] == "neat"
    assert config.LABELS["es_seeded"] == "Seeded ES" and config.COLORS["es_seeded"] == "#0072B2"
    assert set(config.TUNER_APPROACHES) | set(config.EVOLUTION_APPROACHES) | {"baseline"} == set(config.APPROACHES)


def test_revision_parameters():
    assert config.MAX_EPOCHS == 150 and config.HYPERBAND_ITERATIONS == 1 and config.KT_BAYES_INITIAL == 10
    assert config.RUN_SEEDS == (0, 1, 2) and config.SEARCH_SEED == 0
    assert config.GA_S_MIN == config.GA_S_MAX / 100
    assert list(config.DIFFICULTIES) == [1, 2, 3, 4] and config.PLAYBACK_BASE == 2.0
    assert config.DATASET_INFO_PATH == config.DATASET_PATH.with_name(f"{config.DATASET_PATH.stem}_info.json")


def test_stage_paths_and_speed_directory(tmp_path):
    assert paths.stage_path("baseline", 1, "imitation", tmp_path) == tmp_path / "baseline" / "seed1" / "imitation.keras"
    assert paths.stage_path("kt_random", 0, "dagger", tmp_path) == tmp_path / "kt_random" / "seed0" / "dagger.keras"
    assert paths.speed_dir("kt_bayesian", 2, tmp_path) == tmp_path / "kt_bayesian" / "seed2" / "speed"


def test_replay_files_of_evolutionary_runs_and_speed_pushes(tmp_path):
    assert paths.replay_path("neat", 1, tmp_path) == tmp_path / "neat" / "seed1" / paths.REPLAY_NAME
    assert paths.replay_path("es_seeded", 0, tmp_path) == tmp_path / "es_seeded" / "seed0" / paths.REPLAY_NAME
    assert paths.replay_path("kt_random", 2, tmp_path) == tmp_path / "kt_random" / "seed2" / "speed" / paths.REPLAY_NAME


def test_stage_paths_follow_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    assert paths.stage_path("baseline", 0, "dagger") == tmp_path / "runs" / "baseline" / "seed0" / "dagger.keras"


def test_unknown_stage_is_refused(tmp_path):
    with pytest.raises(ValueError, match="stage"):
        paths.stage_path("baseline", 0, "final", tmp_path)


def test_the_final_record_of_a_run():
    assert paths.final_record("ga", 1, "/r").as_posix() == "/r/ga/seed1/summary.json"
    assert paths.final_record("neat", 0, "/r").as_posix() == "/r/neat/seed0/summary.json"
    assert paths.final_record("baseline", 2, "/r").as_posix() == "/r/baseline/seed2/speed/summary.json"
    assert paths.final_record("kt_hyperband", 0, "/r").as_posix() == "/r/kt_hyperband/seed0/speed/summary.json"
