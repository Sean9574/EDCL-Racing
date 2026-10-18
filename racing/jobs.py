"""Jobs of the side-by-side runner (spec §3.3): the experiment's training runs, its Keras Tuner searches and its
notebook 01. A job waits for the records of the steps that make its inputs, so every job can be listed at once and
starts as soon as it can."""
from dataclasses import dataclass
from pathlib import Path

from racing import config, paths, train

NOTEBOOK_JOBS = {"01": "01_dataset.ipynb"}


@dataclass(frozen=True)
class Job:
    """Notebook 01 (`notebook` set, label "nb:01"), a Keras Tuner search (`search` set, label "search:kt_random") or a
    training run (`approach` and `seed`, label "ga:1")."""
    label: str
    notebook: str = None
    approach: str = None
    seed: int = None
    search: str = None

    @property
    def is_notebook(self):
        return self.notebook is not None

    @property
    def is_search(self):
        return self.search is not None


def parse_job(text):
    """"nb:01" for the dataset notebook, "search:<method>" for a Keras Tuner search, approach:seed for a run of any
    method. Notebook 02 is no job: its searches are search jobs, and every imitation run trains its own network."""
    text = str(text).strip()
    if text.startswith("nb:"):
        number = text[3:]
        if number not in NOTEBOOK_JOBS:
            raise ValueError(f"{text!r}: the notebook job is nb:01; notebook 02's work is done by the jobs "
                             f"{', '.join('search:' + a for a in config.TUNER_APPROACHES)} and by the runs themselves")
        return Job(label=text, notebook=NOTEBOOK_JOBS[number])
    if text.startswith("search:"):
        approach = text[len("search:"):]
        if approach not in config.TUNER_APPROACHES:
            raise ValueError(f"{text!r}: the search jobs are "
                             f"{', '.join('search:' + a for a in config.TUNER_APPROACHES)}")
        return Job(label=text, search=approach)
    approach, seed = train.parse_run(text)
    return Job(label=f"{approach}:{seed}", approach=approach, seed=seed)


def parse_jobs(text):
    """A comma-separated list of jobs, each named once."""
    found = [parse_job(part) for part in str(text).split(",") if part.strip()]
    if not found:
        raise ValueError("no jobs given; for example nb:01,search:kt_random,baseline:0,ga:1")
    repeated = sorted({job.label for job in found if found.count(job) > 1})
    if repeated:
        raise ValueError(f"each job may be named once: {', '.join(repeated)}")
    return found


def _folders(results_dir):
    results = Path(config.RESULTS_DIR if results_dir is None else results_dir)
    return results, results / "runs"


def _dataset(results):
    return [results / config.DATASET_INFO_PATH.name, results / config.DATASET_PATH.name]


def needs(job, results_dir=None):
    """The files that must exist before `job` starts (spec §3.3): the record of the step that makes its input first,
    then the data files that step writes before its record, so a server can be given them all. A search and a
    baseline run need the dataset; a run of a searched method also its search, whose chosen architecture it retrains;
    a seeded ES run its refined baseline."""
    results, runs = _folders(results_dir)
    if job.is_search or job.approach == "baseline":
        return _dataset(results)
    if job.approach in config.TUNER_APPROACHES:
        return [paths.search_dir(job.approach, runs) / "summary.json"] + _dataset(results)
    if job.approach == "es_seeded":                                                     # its refined baseline
        return [paths.run_dir("baseline", job.seed, runs) / "dagger.json",
                paths.stage_path("baseline", job.seed, "dagger", runs)]
    return []


def makes(job, results_dir=None):
    """The files `job` writes when it succeeds."""
    results, runs = _folders(results_dir)
    if job.notebook == NOTEBOOK_JOBS["01"]:
        return _dataset(results)
    if job.is_search:
        return [paths.search_dir(job.search, runs) / "summary.json"]
    made = [paths.final_record(job.approach, job.seed, runs)]
    if job.approach in config.IMITATION_APPROACHES:
        run = paths.run_dir(job.approach, job.seed, runs)
        made[:0] = [run / "summary.json", paths.stage_path(job.approach, job.seed, "imitation", runs),
                    run / "dagger.json", paths.stage_path(job.approach, job.seed, "dagger", runs)]
    return made


def ready(job, results_dir=None):
    return all(path.exists() for path in needs(job, results_dir))


def unmet(job_list, results_dir=None):
    """(job, record) for every input that neither exists nor is made by a job of the list: such a job can never
    start, so it is refused before anything starts."""
    made = {path for job in job_list for path in makes(job, results_dir)}
    return [(job, path) for job in job_list for path in needs(job, results_dir)
            if not path.exists() and path not in made]
