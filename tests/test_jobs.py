import pytest

from racing import config, jobs, paths


def test_jobs_are_notebooks_01_and_02_or_runs():
    assert jobs.parse_jobs("nb:01,ga:1,kt_random:2") == [jobs.parse_job("nb:01"), jobs.parse_job("ga:1"),
                                                          jobs.parse_job("kt_random:2")]
    assert jobs.parse_job("nb:02").notebook == "02_imitation.ipynb" and jobs.parse_job("nb:02").is_notebook
    assert jobs.parse_job("baseline:0") == jobs.Job(label="baseline:0", approach="baseline", seed=0)
    assert not jobs.parse_job("baseline:0").is_notebook


@pytest.mark.parametrize("text", ["nb:03", "nb:1", "baseline:5", "ga", "ga:1,ga:1", "", "nb:01,nb:01"])
def test_jobs_outside_the_experiment_are_refused(text):
    with pytest.raises(ValueError):
        jobs.parse_jobs(text)


def test_each_job_waits_for_the_record_that_makes_its_input(tmp_path):
    r = tmp_path
    assert jobs.needs(jobs.parse_job("nb:01"), r) == [] and jobs.needs(jobs.parse_job("ga:0"), r) == []
    assert jobs.needs(jobs.parse_job("neat:2"), r) == [] and jobs.needs(jobs.parse_job("es:1"), r) == []
    assert jobs.needs(jobs.parse_job("nb:02"), r) == [r / "dataset_info.json"]
    assert jobs.needs(jobs.parse_job("kt_bayesian:1"), r) == [r / "runs" / "kt_bayesian" / "seed1" / "summary.json"]
    assert jobs.needs(jobs.parse_job("es_seeded:2"), r) == [r / "runs" / "baseline" / "seed2" / "dagger.json"]


def test_each_job_names_the_records_it_makes(tmp_path):
    r = tmp_path
    assert jobs.makes(jobs.parse_job("nb:01"), r) == [r / "dataset_info.json"]
    made = jobs.makes(jobs.parse_job("nb:02"), r)
    assert len(made) == len(config.IMITATION_APPROACHES) * len(config.RUN_SEEDS)
    assert r / "runs" / "kt_random" / "seed0" / "summary.json" in made
    assert jobs.makes(jobs.parse_job("baseline:2"), r) == [r / "runs" / "baseline" / "seed2" / "dagger.json",
                                                           r / "runs" / "baseline" / "seed2" / "speed" / "summary.json"]
    assert jobs.makes(jobs.parse_job("ga:1"), r) == [r / "runs" / "ga" / "seed1" / "summary.json"]


def test_an_input_nobody_makes_is_unmet(tmp_path):
    listed = jobs.parse_jobs("es_seeded:0,ga:1")
    assert [(j.label, p.name) for j, p in jobs.unmet(listed, tmp_path)] == [("es_seeded:0", "dagger.json")]
    assert jobs.unmet(jobs.parse_jobs("baseline:0,es_seeded:0,nb:01,nb:02"), tmp_path) == []
    assert not jobs.ready(listed[0], tmp_path)
    made = tmp_path / "runs" / "baseline" / "seed0" / "dagger.json"
    made.parent.mkdir(parents=True)
    made.write_text("{}")
    assert jobs.unmet(listed, tmp_path) == [] and jobs.ready(listed[0], tmp_path)


def test_jobs_default_to_the_results_folder():
    assert jobs.needs(jobs.parse_job("nb:02")) == [config.RESULTS_DIR / "dataset_info.json"]
