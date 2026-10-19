import pytest

from racing import config, jobs, paths


def test_jobs_are_notebook_01_searches_or_runs():
    assert jobs.parse_jobs("nb:01,search:kt_random,ga:1,kt_random:1") == [
        jobs.parse_job("nb:01"), jobs.parse_job("search:kt_random"), jobs.parse_job("ga:1"),
        jobs.parse_job("kt_random:1")]
    assert jobs.parse_job("nb:01").notebook == "01_dataset.ipynb" and jobs.parse_job("nb:01").is_notebook
    search = jobs.parse_job("search:kt_hyperband")
    assert search == jobs.Job(label="search:kt_hyperband", search="kt_hyperband")
    assert search.is_search and not search.is_notebook and search.approach is None
    assert jobs.parse_job("baseline:0") == jobs.Job(label="baseline:0", approach="baseline", seed=0)
    assert not jobs.parse_job("baseline:0").is_notebook and not jobs.parse_job("baseline:0").is_search


@pytest.mark.parametrize("text", ["nb:03", "nb:1", "baseline:5", "ga", "ga:1,ga:1", "", "nb:01,nb:01",
                                  "search:ga", "search:baseline", "search:", "search:kt_random,search:kt_random"])
def test_jobs_outside_the_experiment_are_refused(text):
    with pytest.raises(ValueError):
        jobs.parse_jobs(text)


def test_notebook_02_is_no_longer_a_job_and_the_message_names_the_searches():
    with pytest.raises(ValueError, match="search:kt_random"):
        jobs.parse_job("nb:02")


def test_each_job_waits_for_the_record_that_makes_its_input(tmp_path):
    r = tmp_path
    dataset = [r / "dataset_info.json", r / "dataset.npz"]
    assert jobs.needs(jobs.parse_job("nb:01"), r) == [] and jobs.needs(jobs.parse_job("ga:0"), r) == []
    assert jobs.needs(jobs.parse_job("neat:0"), r) == [] and jobs.needs(jobs.parse_job("es:1"), r) == []
    assert jobs.needs(jobs.parse_job("search:kt_bayesian"), r) == dataset
    assert jobs.needs(jobs.parse_job("baseline:1"), r) == dataset              # it trains its own network
    search = r / "runs" / "kt_bayesian" / "search" / "summary.json"
    assert jobs.needs(jobs.parse_job("kt_bayesian:1"), r) == [search] + dataset
    refined = r / "runs" / "baseline" / "seed1"
    assert jobs.needs(jobs.parse_job("es_seeded:1"), r) == [refined / "dagger.json", refined / "dagger.keras"]


def test_each_job_names_the_records_it_makes(tmp_path):
    r = tmp_path
    assert jobs.makes(jobs.parse_job("nb:01"), r) == [r / "dataset_info.json", r / "dataset.npz"]
    assert jobs.makes(jobs.parse_job("search:kt_random"), r) == [r / "runs" / "kt_random" / "search" / "summary.json"]
    run = r / "runs" / "baseline" / "seed1"
    assert jobs.makes(jobs.parse_job("baseline:1"), r) == [run / "summary.json", run / "imitation.keras",
                                                           run / "dagger.json", run / "dagger.keras",
                                                           run / "speed" / "summary.json"]
    assert jobs.makes(jobs.parse_job("ga:1"), r) == [r / "runs" / "ga" / "seed1" / "summary.json"]


def test_an_input_nobody_makes_is_unmet(tmp_path):
    listed = jobs.parse_jobs("es_seeded:0,ga:1")
    assert [(j.label, p.name) for j, p in jobs.unmet(listed, tmp_path)] == [("es_seeded:0", "dagger.json"),
                                                                           ("es_seeded:0", "dagger.keras")]
    assert jobs.unmet(jobs.parse_jobs("baseline:0,es_seeded:0,nb:01"), tmp_path) == []
    alone = jobs.parse_jobs("nb:01,kt_random:0")                      # a searched method's run without its search
    assert [(j.label, p.relative_to(tmp_path).as_posix()) for j, p in jobs.unmet(alone, tmp_path)] == [
        ("kt_random:0", "runs/kt_random/search/summary.json")]
    assert jobs.unmet(jobs.parse_jobs("nb:01,search:kt_random,kt_random:0"), tmp_path) == []
    assert not jobs.ready(listed[0], tmp_path)
    refined = tmp_path / "runs" / "baseline" / "seed0"
    refined.mkdir(parents=True)
    (refined / "dagger.json").write_text("{}")
    assert not jobs.ready(listed[0], tmp_path)                          # the network itself is missing too
    (refined / "dagger.keras").write_bytes(b"network")
    assert jobs.unmet(listed, tmp_path) == [] and jobs.ready(listed[0], tmp_path)


def test_jobs_default_to_the_results_folder():
    assert jobs.needs(jobs.parse_job("search:kt_random")) == [config.RESULTS_DIR / "dataset_info.json",
                                                               config.RESULTS_DIR / "dataset.npz"]
