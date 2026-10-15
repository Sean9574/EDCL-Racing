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


def test_run_order_lists_the_24_counted_runs():
    assert len(config.RUN_ORDER) == 24
    assert config.RUN_ORDER[0] == ("baseline", 0) and config.RUN_ORDER[-1] == ("neat", 2)
    assert config.RUN_ORDER.index(("ga", 1)) == 13
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
