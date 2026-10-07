"""Evolution strategy over dense network weights (revision 3 §5.2): each child is one of the fittest cars plus
Gaussian noise of size σ, with the two-phase step-size rule. Started from a trained network it is the seeded ES
(revision 3 §5.3) or the speed push of an imitation method (revision 3 §6.4)."""
import numpy as np

from racing import config, evolution, nets, paths
from racing.curriculum import StepSize, success_fraction
from racing.nets import DensePopulation


def seeded_population(rng, flat, n, sigma):
    """The trained network itself, then n − 1 copies with N(0, σ²) added to every weight."""
    flat = np.asarray(flat, dtype=float)
    return np.vstack([flat, flat + rng.normal(0.0, sigma, (n - 1, len(flat)))])


class ES(evolution.DenseMethod):
    """The ES as a method of the evolution loop (racing/evolution.py). Every car has an id; a child records its
    parent's id, so the 1/5 rule compares each child with its own parent."""

    def __init__(self, approach="es", pop_size=config.ES_POP, parents=config.ES_MU, hidden=config.BASELINE_HIDDEN,
                 activation="relu", initial=None, sigma0=config.ES_SIGMA0):
        super().__init__(pop_size, hidden, activation)
        self.approach = approach
        self.parents = parents
        self.initial = None if initial is None else np.asarray(initial, dtype=float)
        self.sigma0 = sigma0
        self.step = StepSize(sigma0)
        self.ids = None
        self.parent_ids = None
        self.next_id = 0

    @property
    def step_size(self):
        return self.step.sigma

    def initialize(self, rng):
        if self.initial is None:
            weights = nets.init_population(rng, self.pop_size, self.hidden)
        else:
            weights = seeded_population(rng, self.initial, self.pop_size, self.sigma0)
        self.population = DensePopulation(weights, self.hidden, self.activation)
        self.is_kept = np.zeros(self.pop_size, dtype=bool)
        self.ids = np.arange(self.pop_size)
        self.parent_ids = np.full(self.pop_size, -1)
        self.next_id = self.pop_size
        self.step = StepSize(self.sigma0)

    def success_rate(self, fitness):
        """Share of this generation's children that beat their own parent, a tie counting half, over the children
        whose parent is still in the population (revision 3 §5.2); nan if there are none."""
        position = {int(i): k for k, i in enumerate(self.ids)}
        children = [k for k in np.flatnonzero(~self.is_kept) if int(self.parent_ids[k]) in position]
        if not children:
            return float("nan")
        parents = [position[int(self.parent_ids[k])] for k in children]
        return success_fraction(fitness[children], fitness[parents])

    def advance(self, rng, result, kept, event):
        """Adapt σ, then keep the survivors first and unchanged and add children of the ES_MU fittest cars."""
        weights = self.population.weights
        n, p = weights.shape
        top = np.argsort(-result.fitness, kind="stable")[:min(self.parents, n)]
        child_success = evolution.children_success(result, self.is_kept)
        self.step.update(float(result.all_success[top].mean()), child_success, self.success_rate(result.fitness))
        if event == "level_up":
            self.step.reset()                           # restart the step size on the harder maps
        n_new = n - len(kept)
        chosen = top[rng.integers(0, len(top), n_new)]
        children = weights[chosen] + rng.normal(0.0, self.step.sigma, (n_new, p))
        self.assemble(weights[kept], children)
        self.parent_ids = np.concatenate([self.parent_ids[kept], self.ids[chosen]])
        self.ids = np.concatenate([self.ids[kept], np.arange(self.next_id, self.next_id + n_new)])
        self.next_id += n_new

    def state(self):
        return {"weights": self.population.weights, "is_kept": self.is_kept, "ids": self.ids,
                "parent_ids": self.parent_ids, "next_id": self.next_id, "step": self.step.state(),
                "hidden": self.hidden, "activation": self.activation}

    def load_state(self, state):
        self.hidden, self.activation = tuple(state["hidden"]), state["activation"]
        self.population = DensePopulation(state["weights"], self.hidden, self.activation)
        self.is_kept, self.ids, self.parent_ids = state["is_kept"], state["ids"], state["parent_ids"]
        self.next_id = state["next_id"]
        self.step = StepSize.from_state(state["step"])


def dagger_baseline(seed, runs_dir=None):
    """Flat float64 weights of the baseline after its DAgger refinement, the start of the seeded ES (revision 3
    §5.3)."""
    from racing.backend import keras_cpu
    path = paths.run_dir("baseline", seed, runs_dir) / "dagger.keras"     # paths.stage_path(..., "dagger"), Task 10
    if not path.exists():
        raise FileNotFoundError(f"no refined baseline for seed {seed} at {path}; run notebook 03 first")
    return nets.to_flat(keras_cpu().models.load_model(path))


def run_es(seed=0, pop_size=config.ES_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1,
           on_generation=None, start_level=1, max_generations=None, initial=None, approach="es",
           hidden=config.BASELINE_HIDDEN, activation="relu"):
    """The ES with run seed `seed`; with `initial`, the flat weights of a trained network of architecture
    (hidden, activation), it starts from that network. Returns the run's summary (racing/evolution.py)."""
    method = ES(approach=approach, pop_size=pop_size, hidden=hidden, activation=activation, initial=initial)
    return evolution.run(method, seed, out_dir or paths.run_dir(approach, seed), champion_dir, n_jobs,
                         on_generation, start_level, max_generations)
