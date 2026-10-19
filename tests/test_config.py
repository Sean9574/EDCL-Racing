import pytest

from racing import config


def test_worker_count_reads_the_environment():
    assert config.worker_count({}) == 6
    assert config.worker_count({"EC_RACING_JOBS": ""}) == 6
    assert config.worker_count({"EC_RACING_JOBS": "16"}) == 16
    assert config.worker_count({"EC_RACING_JOBS": " 12 "}) == 12


@pytest.mark.parametrize("value", ["0", "-2", "four", "1.5"])
def test_worker_count_rejects_values_that_are_not_a_positive_whole_number(value):
    with pytest.raises(ValueError, match="EC_RACING_JOBS"):
        config.worker_count({"EC_RACING_JOBS": value})


def test_run_order_lists_the_32_counted_runs():
    assert len(config.RUN_ORDER) == 32
    assert config.RUN_ORDER[0] == ("baseline", 0) and config.RUN_ORDER[-1] == ("neat", 3)
    assert config.RUN_ORDER.index(("ga", 1)) == 17
    assert config.STATUS_PATH.name == "status.json"      # the test fixture redirects the folder to a temporary one


def test_simulator_choice_reads_ec_racing_sim():
    assert config.simulator_choice({}) == "auto"
    assert config.simulator_choice({"EC_RACING_SIM": " numpy "}) == "numpy"
    assert config.simulator_choice({"EC_RACING_SIM": "auto"}) == "auto"
    with pytest.raises(ValueError, match="EC_RACING_SIM"):
        config.simulator_choice({"EC_RACING_SIM": "cpp"})


def test_simulator_choice_is_not_a_result_setting():
    from racing import checkpoint
    assert "SIMULATOR" not in checkpoint.settings_record()


@pytest.mark.parametrize("value, expected", [("", None), ("  ", None), ("8", 8), (" 8 ", 8), ("1", 1)])
def test_the_thread_setting_is_ec_racing_threads_or_none(value, expected):
    assert config.thread_setting({"EC_RACING_THREADS": value}) == expected
    assert config.thread_setting({}) is None


@pytest.mark.parametrize("value", ["0", "-2", "abc", "2.5"])
def test_a_bad_thread_setting_names_the_variable(value):
    with pytest.raises(ValueError, match="EC_RACING_THREADS"):
        config.thread_setting({"EC_RACING_THREADS": value})


def test_the_thread_count_is_the_processors_available_to_this_process(monkeypatch):
    monkeypatch.setattr(config, "THREADS", None)
    monkeypatch.setattr(config.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})     # as under taskset -c 0-3
    assert config.thread_count() == 4 and config.thread_label() == "all"
    monkeypatch.setattr(config, "THREADS", 6)
    assert config.thread_count() == 6 and config.thread_label() == 6
