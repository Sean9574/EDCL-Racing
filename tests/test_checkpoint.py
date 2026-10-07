import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from racing import checkpoint, config


def test_settings_record_survives_a_json_round_trip_unchanged():
    record = checkpoint.settings_record()
    assert json.loads(json.dumps(record)) == record
    assert record["VAL_SEEDS"] == list(config.VAL_SEEDS) and record["RUN_SEEDS"] == list(config.RUN_SEEDS)
    assert set(record["LEVELS"]) == {"1", "2", "3", "4"}                  # int keys read back as strings
    assert record["STEER_LEVELS"] == config.STEER_LEVELS.tolist()


def test_settings_keys_hold_the_result_settings_only():
    keys = checkpoint.SETTINGS_KEYS
    assert isinstance(keys, tuple) and len(set(keys)) == len(keys)
    assert {"LAPS", "LEVELS", "TARGET_FACTOR", "MASTERY_FRACTION", "GA_MUTATION_P", "MAX_EPOCHS", "RUN_SEEDS"} <= set(keys)
    assert not {"N_JOBS", "TRACK_CACHE_SIZE", "WINDOW_SIZE", "COLORS", "RESULTS_DIR"} & set(keys)
    assert all(hasattr(config, name) for name in keys)
    assert set(checkpoint.settings_record()) == set(keys) and "N_JOBS" not in checkpoint.settings_record()


def test_own_record_passes_the_check_after_a_json_round_trip():
    checkpoint.check_settings(json.loads(json.dumps(checkpoint.settings_record())))


def test_check_names_each_changed_setting(monkeypatch):
    recorded = checkpoint.settings_record()
    monkeypatch.setattr(config, "LAPS", 5)
    monkeypatch.setattr(config, "GA_MUTATION_P", 0.3)
    with pytest.raises(checkpoint.SettingsMismatch) as error:
        checkpoint.check_settings(recorded)
    message = str(error.value)
    assert "LAPS: recorded 3, now 5" in message and "GA_MUTATION_P: recorded 0.1, now 0.3" in message
    assert "--fresh" in message


def test_a_setting_missing_from_the_record_is_a_difference():
    recorded = checkpoint.settings_record()
    recorded.pop("LAPS")
    with pytest.raises(checkpoint.SettingsMismatch, match="LAPS: recorded None, now 3"):
        checkpoint.check_settings(recorded)


def test_save_and_load_round_trip_with_the_random_generator(tmp_path):
    rng = np.random.default_rng(0)
    state = {"weights": rng.normal(size=(4, 3)), "rng": rng.bit_generator.state, "used": [2000001, 2000002]}
    path = tmp_path / "run" / checkpoint.CHECKPOINT_NAME
    checkpoint.save(path, state)
    loaded = checkpoint.load(path)
    assert np.array_equal(loaded["weights"], state["weights"]) and loaded["used"] == [2000001, 2000002]
    restored = np.random.default_rng()
    restored.bit_generator.state = loaded["rng"]
    assert restored.random() == rng.random()


def test_load_missing_is_none(tmp_path):
    assert checkpoint.load(tmp_path / "missing.pkl") is None


class Unpicklable:
    def __reduce__(self):
        raise RuntimeError("cannot pickle")


def test_a_failed_save_keeps_the_previous_checkpoint(tmp_path):
    path = tmp_path / checkpoint.CHECKPOINT_NAME
    checkpoint.save(path, {"generation": 4})
    with pytest.raises(RuntimeError, match="cannot pickle"):
        checkpoint.save(path, {"generation": 5, "bad": Unpicklable()})
    assert checkpoint.load(path) == {"generation": 4}
    assert [p.name for p in tmp_path.iterdir()] == [checkpoint.CHECKPOINT_NAME]


def write_summary(folder, settings):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps({"generations": 7, "settings": settings}))


def test_completed_returns_a_matching_summary(tmp_path):
    write_summary(tmp_path, checkpoint.settings_record())
    assert checkpoint.completed(tmp_path)["generations"] == 7


def test_completed_without_a_summary_is_none(tmp_path):
    assert checkpoint.completed(tmp_path / "never_ran") is None


def test_completed_with_other_settings_is_refused(tmp_path, monkeypatch):
    write_summary(tmp_path, checkpoint.settings_record())
    monkeypatch.setattr(config, "TRACKS_PER_GEN", 5)
    with pytest.raises(checkpoint.SettingsMismatch, match="TRACKS_PER_GEN"):
        checkpoint.completed(tmp_path)


def test_a_summary_without_settings_is_refused(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"generations": 7}))
    with pytest.raises(checkpoint.SettingsMismatch, match="no settings record"):
        checkpoint.completed(tmp_path)


def test_completed_reads_another_summary_name(tmp_path):
    (tmp_path / "dagger.json").write_text(json.dumps({"rounds": 3, "settings": checkpoint.settings_record()}))
    assert checkpoint.completed(tmp_path, "dagger.json")["rounds"] == 3


def test_checkpoints_are_plain_pickles(tmp_path):
    path = tmp_path / checkpoint.CHECKPOINT_NAME
    checkpoint.save(path, {"generation": 1})
    with open(path, "rb") as f:
        assert pickle.load(f) == {"generation": 1}


def test_write_json_syncs_the_temporary_file_before_replacing(tmp_path, monkeypatch):
    """A record is on disk before it replaces the old one, so a reboot cannot leave it empty (revision 3 §8.2)."""
    calls, real_fsync, real_replace = [], checkpoint.os.fsync, checkpoint.os.replace
    monkeypatch.setattr(checkpoint.os, "fsync", lambda fd: calls.append("fsync") or real_fsync(fd))
    monkeypatch.setattr(checkpoint.os, "replace", lambda a, b: calls.append(("replace", Path(a).parent, Path(b)))
                        or real_replace(a, b))
    path = tmp_path / "run" / "summary.json"
    checkpoint.write_json(path, {"generations": 3, "levels": [{"level": 1}]})
    assert json.loads(path.read_text()) == {"generations": 3, "levels": [{"level": 1}]}
    assert calls == ["fsync", ("replace", path.parent, path)]       # synced first; the temporary is a sibling
    assert [p.name for p in path.parent.iterdir()] == ["summary.json"]


def test_a_failed_json_write_keeps_the_previous_record(tmp_path):
    path = tmp_path / "summary.json"
    checkpoint.write_json(path, {"generations": 3})
    with pytest.raises(TypeError):
        checkpoint.write_json(path, {"generations": object()})
    assert json.loads(path.read_text()) == {"generations": 3}
    assert [p.name for p in tmp_path.iterdir()] == ["summary.json"]


@pytest.mark.parametrize("content", ["", "\0\0\0\0", '{"generations": 7, "sett'])
def test_an_unreadable_summary_next_to_its_checkpoint_is_not_completed(tmp_path, content):
    """A summary cut short by a reboot: the run finishes again from its checkpoint."""
    (tmp_path / "summary.json").write_text(content)
    checkpoint.save(tmp_path / checkpoint.CHECKPOINT_NAME, {"generation": 7})
    assert checkpoint.completed(tmp_path) is None


def test_an_unreadable_summary_without_a_checkpoint_names_the_file(tmp_path):
    (tmp_path / "summary.json").write_text("")
    with pytest.raises(checkpoint.UnreadableRecord) as error:
        checkpoint.completed(tmp_path)
    message = str(error.value)
    assert str(tmp_path / "summary.json") in message and "delete" in message and "--fresh" not in message


def test_an_unreadable_record_uses_its_own_checkpoint_name(tmp_path):
    (tmp_path / "dagger.json").write_text("")
    checkpoint.save(tmp_path / checkpoint.CHECKPOINT_NAME, {"generation": 7})      # another run's checkpoint
    with pytest.raises(checkpoint.UnreadableRecord):
        checkpoint.completed(tmp_path, "dagger.json", state_name="dagger_state.pkl")
    checkpoint.save(tmp_path / "dagger_state.pkl", {"round": 4})
    assert checkpoint.completed(tmp_path, "dagger.json", state_name="dagger_state.pkl") is None
