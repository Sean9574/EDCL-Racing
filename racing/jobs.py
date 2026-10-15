"""Jobs of the side-by-side runner (spec §3.3): the experiment's training runs and its notebooks 01 and 02.
A job waits for the records of the steps that make its inputs, so every job can be listed at once and starts as soon
as it can."""
from dataclasses import dataclass
from pathlib import Path

from racing import config, paths, train

NOTEBOOK_JOBS = {"01": "01_dataset.ipynb", "02": "02_imitation.ipynb"}


@dataclass(frozen=True)
class Job:
    """A notebook (`notebook` set, label "nb:01") or a training run (`approach` and `seed`, label "ga:1")."""
    label: str
    notebook: str = None
    approach: str = None
    seed: int = None

    @property
    def is_notebook(self):
        return self.notebook is not None


def parse_job(text):
    """"nb:01" or "nb:02" for a notebook, approach:seed for a run of any method."""
    text = str(text).strip()
    if text.startswith("nb:"):
        number = text[3:]
        if number not in NOTEBOOK_JOBS:
            raise ValueError(f"{text!r}: the notebook jobs are {', '.join('nb:' + n for n in NOTEBOOK_JOBS)}")
        return Job(label=text, notebook=NOTEBOOK_JOBS[number])
    approach, seed = train.parse_run(text)
    return Job(label=f"{approach}:{seed}", approach=approach, seed=seed)


def parse_jobs(text):
    """A comma-separated list of jobs, each named once."""
    found = [parse_job(part) for part in str(text).split(",") if part.strip()]
    if not found:
        raise ValueError("no jobs given; for example nb:01,nb:02,baseline:0,ga:1")
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
    """The files that must exist before `job` starts (spec §3.3): the record of the step that makes
    its input first, then the data files that step writes before its record, so a server can be given them all."""
    results, runs = _folders(results_dir)
    if job.notebook == NOTEBOOK_JOBS["02"]:
        return _dataset(results)
    if job.approach in config.IMITATION_APPROACHES:                                    # its imitation network
        return [paths.run_dir(job.approach, job.seed, runs) / "summary.json",
                paths.stage_path(job.approach, job.seed, "imitation", runs)] + _dataset(results)
    if job.approach == "es_seeded":                                                     # its refined baseline
        return [paths.run_dir("baseline", job.seed, runs) / "dagger.json",
                paths.stage_path("baseline", job.seed, "dagger", runs)]
    return []


def makes(job, results_dir=None):
    """The files `job` writes when it succeeds."""
    results, runs = _folders(results_dir)
    if job.notebook == NOTEBOOK_JOBS["01"]:
        return _dataset(results)
    if job.notebook == NOTEBOOK_JOBS["02"]:
        return [path for approach in config.IMITATION_APPROACHES for seed in config.RUN_SEEDS
                for path in (paths.run_dir(approach, seed, runs) / "summary.json",
                             paths.stage_path(approach, seed, "imitation", runs))]
    made = [paths.final_record(job.approach, job.seed, runs)]
    if job.approach in config.IMITATION_APPROACHES:
        made[:0] = [paths.run_dir(job.approach, job.seed, runs) / "dagger.json",
                    paths.stage_path(job.approach, job.seed, "dagger", runs)]
    return made


def ready(job, results_dir=None):
    return all(path.exists() for path in needs(job, results_dir))


def unmet(job_list, results_dir=None):
    """(job, record) for every input that neither exists nor is made by a job of the list: such a job can never
    start, so it is refused before anything starts."""
    made = {path for job in job_list for path in makes(job, results_dir)}
    return [(job, path) for job in job_list for path in needs(job, results_dir)
            if not path.exists() and path not in made]
