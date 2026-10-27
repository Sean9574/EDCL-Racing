"""NEAT, neuroevolution of augmenting topologies (spec §9.9): feed-forward networks whose structure grows by
mutation. A genome lists node genes (output and hidden nodes, each with a bias; the inputs are implicit) and
connection genes. Every structural mutation is recorded for the whole run, so the same connection or split has the
same innovation number or node id in every genome, which aligns genomes for crossover and speciation."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from racing import config, paths
from racing import setup as car_setup
from racing.backend import keras_cpu
from racing.curriculum import StepSize, success_fraction

N_IN = config.N_INPUTS                         # 18
N_OUT = config.N_ACTIONS                       # 45
INPUT_IDS = range(0, N_IN)
OUTPUT_IDS = range(N_IN, N_IN + N_OUT)
FIRST_HIDDEN = N_IN + N_OUT                    # 63
CONNECTION_ATTEMPTS = 20                       # random node pairs one add-connection mutation tries


@dataclass(eq=False)
class Genome:
    """Node genes (output and hidden ids, sorted, with their biases) and connection genes sorted by innovation."""
    node_ids: np.ndarray        # int64
    bias: np.ndarray            # float64, aligned with node_ids
    innov: np.ndarray           # int64, sorted
    src: np.ndarray             # int64 node ids, aligned with innov
    dst: np.ndarray
    weight: np.ndarray          # float64
    enabled: np.ndarray         # bool

    FIELDS = ("node_ids", "bias", "innov", "src", "dst", "weight", "enabled")
    DTYPES = (np.int64, float, np.int64, np.int64, np.int64, float, bool)

    def copy(self):
        return Genome(*(getattr(self, name).copy() for name in self.FIELDS))

    @property
    def n_params(self):
        """Enabled connections plus biases: the parameters the network uses (spec §9.9)."""
        return int(np.sum(self.enabled)) + len(self.bias)

    def to_dict(self):
        return {name: getattr(self, name).tolist() for name in self.FIELDS}

    @classmethod
    def from_dict(cls, d):
        return cls(*(np.asarray(d[name], dtype=dtype) for name, dtype in zip(cls.FIELDS, cls.DTYPES)))


class Innovations:
    """The run's record of structural mutations: one innovation number per (source, target) pair and one node id per
    split connection."""

    def __init__(self, next_innov=0, next_node=FIRST_HIDDEN, conns=None, splits=None):
        self.next_innov = int(next_innov)
        self.next_node = int(next_node)
        self.conns = dict(conns or {})          # (src, dst) -> innovation number
        self.splits = dict(splits or {})        # innovation number of the split connection -> node id

    def conn(self, src, dst):
        key = (int(src), int(dst))
        if key not in self.conns:
            self.conns[key] = self.next_innov
            self.next_innov += 1
        return self.conns[key]

    def split(self, innov):
        innov = int(innov)
        if innov not in self.splits:
            self.splits[innov] = self.fresh_node()
        return self.splits[innov]

    def fresh_node(self):
        """A node id never used before; not recorded as a split."""
        node = self.next_node
        self.next_node += 1
        return node

    def state(self):
        return {"next_innov": self.next_innov, "next_node": self.next_node,
                "conns": sorted([s, d, i] for (s, d), i in self.conns.items()),
                "splits": sorted([i, n] for i, n in self.splits.items())}

    @classmethod
    def from_state(cls, state):
        return cls(state["next_innov"], state["next_node"], {(s, d): i for s, d, i in state["conns"]},
                   {i: n for i, n in state["splits"]})


def initial_genome(rng, innovations):
    """Every input connected to every output N_IN · N_OUT connections (810), standard normal weights, zero biases."""
    src = np.repeat(np.arange(N_IN, dtype=np.int64), N_OUT)
    dst = np.tile(np.arange(N_IN, N_IN + N_OUT, dtype=np.int64), N_IN)
    innov = np.array([innovations.conn(s, d) for s, d in zip(src.tolist(), dst.tolist())], dtype=np.int64)
    weight = rng.standard_normal(len(src))
    order = np.argsort(innov, kind="stable")
    return Genome(node_ids=np.arange(N_IN, N_IN + N_OUT, dtype=np.int64), bias=np.zeros(N_OUT), innov=innov[order],
                  src=src[order], dst=dst[order], weight=weight[order], enabled=np.ones(len(src), dtype=bool))


def _with_connections(genome, innov, src, dst, weight):
    """A copy with new enabled connection genes, kept sorted by innovation number."""
    g = genome.copy()
    order = np.argsort(np.concatenate([g.innov, np.asarray(innov, dtype=np.int64)]), kind="stable")
    g.innov = np.concatenate([g.innov, np.asarray(innov, dtype=np.int64)])[order]
    g.src = np.concatenate([g.src, np.asarray(src, dtype=np.int64)])[order]
    g.dst = np.concatenate([g.dst, np.asarray(dst, dtype=np.int64)])[order]
    g.weight = np.concatenate([g.weight, np.asarray(weight, dtype=float)])[order]
    g.enabled = np.concatenate([g.enabled, np.ones(len(innov), dtype=bool)])[order]
    return g


def creates_cycle(genome, src, dst):
    """True if a connection src -> dst would close a cycle. Disabled connections count too, so enabling any
    connection later can never create one."""
    src, dst = int(src), int(dst)
    if src == dst:
        return True
    succ = {}
    for s, d in zip(genome.src.tolist(), genome.dst.tolist()):
        succ.setdefault(s, []).append(d)
    seen, stack = {dst}, [dst]
    while stack:
        node = stack.pop()
        if node == src:
            return True
        for nxt in succ.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def add_connection(rng, genome, innovations):
    """Connect a random pair of unconnected nodes, from an input or hidden node to a hidden or output node, with a
    standard normal weight, unless it would create a cycle. Gives up after CONNECTION_ATTEMPTS random pairs."""
    sources = np.concatenate([np.arange(N_IN, dtype=np.int64), genome.node_ids[genome.node_ids >= FIRST_HIDDEN]])
    targets = genome.node_ids
    existing = set(zip(genome.src.tolist(), genome.dst.tolist()))
    for _ in range(CONNECTION_ATTEMPTS):
        s = int(sources[rng.integers(len(sources))])
        d = int(targets[rng.integers(len(targets))])
        if (s, d) in existing or creates_cycle(genome, s, d):
            continue
        return _with_connections(genome, [innovations.conn(s, d)], [s], [d], [rng.standard_normal()])
    return genome


def add_node(rng, genome, innovations):
    """Split a random enabled connection with a hidden node: source -> node with weight 1, node -> target with the
    old weight, old connection disabled. The recorded node for that split is used unless this genome already has it
    (the connection was re-enabled and is split again); then a fresh, unrecorded node id is used."""
    on = np.flatnonzero(genome.enabled)
    if on.size == 0:
        return genome
    k = int(on[rng.integers(on.size)])
    s, d, w = int(genome.src[k]), int(genome.dst[k]), float(genome.weight[k])
    node = innovations.split(int(genome.innov[k]))
    if node in set(genome.node_ids.tolist()):
        node = innovations.fresh_node()
    g = genome.copy()
    g.enabled[k] = False
    position = int(np.searchsorted(g.node_ids, node))
    g.node_ids = np.insert(g.node_ids, position, node)
    g.bias = np.insert(g.bias, position, 0.0)
    return _with_connections(g, [innovations.conn(s, node), innovations.conn(node, d)], [s, node], [node, d],
                             [1.0, w])


def _mutate_values(rng, values, sigma, scale=None):
    """Each value replaced by a standard normal value with probability NEAT_REPLACE, otherwise perturbed by
    N(0, σ²) with probability NEAT_PERTURB. `scale`, one factor per value, multiplies the perturbation drawn (the
    stall boost, spec §9.4)."""
    u = rng.random(len(values))
    noise = rng.normal(0.0, sigma, len(values))
    if scale is not None:
        noise = noise * scale
    fresh = rng.standard_normal(len(values))
    replace = u < config.NEAT_REPLACE
    perturb = ~replace & (u < config.NEAT_REPLACE + config.NEAT_PERTURB)
    return np.where(replace, fresh, np.where(perturb, values + noise, values))


def mutate(rng, genome, innovations, sigma, boost=False):
    """A mutated copy (spec §9.9): weights and biases, then with their own probabilities a new connection, a
    new node and one toggled connection. With boost (a stall, spec §9.4) the perturbation of the connections into
    output nodes is STALL_BOOST times as strong, from the same draws; connections into hidden nodes and the biases
    mutate as usual."""
    g = genome.copy()
    scale = np.where(np.isin(g.dst, OUTPUT_IDS), config.STALL_BOOST, 1.0) if boost else None
    g.weight = _mutate_values(rng, g.weight, sigma, scale)
    g.bias = _mutate_values(rng, g.bias, sigma)
    if rng.random() < config.NEAT_ADD_CONN:
        g = add_connection(rng, g, innovations)
    if rng.random() < config.NEAT_ADD_NODE:
        g = add_node(rng, g, innovations)
    if rng.random() < config.NEAT_TOGGLE and len(g.innov):
        k = int(rng.integers(len(g.innov)))
        g.enabled[k] = not g.enabled[k]
    return g


def crossover(rng, fitter, other):
    """Child of two genomes aligned by innovation number: matching genes from a random parent, disjoint and excess
    genes from the fitter parent; a matching gene disabled in either parent is disabled with probability
    NEAT_DISABLE_INHERIT. The child has the fitter parent's nodes; a node both parents have takes a random parent's
    bias."""
    child = fitter.copy()
    _, i_f, i_o = np.intersect1d(fitter.innov, other.innov, assume_unique=True, return_indices=True)
    take = rng.random(len(i_f)) < 0.5
    child.weight[i_f[take]] = other.weight[i_o[take]]
    disabled = ~fitter.enabled[i_f] | ~other.enabled[i_o]
    keep_off = rng.random(len(i_f)) < config.NEAT_DISABLE_INHERIT
    child.enabled[i_f] = ~(disabled & keep_off)
    _, n_f, n_o = np.intersect1d(fitter.node_ids, other.node_ids, assume_unique=True, return_indices=True)
    take = rng.random(len(n_f)) < 0.5
    child.bias[n_f[take]] = other.bias[n_o[take]]
    return child


def distance(a, b):
    """Compatibility distance δ = C1·E/N + C2·D/N + C3·W̄ (spec §9.9): E excess genes (beyond the smaller
    genome's largest innovation number), D disjoint genes, W̄ the mean weight difference of matching genes, N the
    larger genome's gene count (1 if both have fewer than NEAT_SMALL_GENOME)."""
    if len(a.innov) == 0 and len(b.innov) == 0:
        return 0.0
    _, i_a, i_b = np.intersect1d(a.innov, b.innov, assume_unique=True, return_indices=True)
    unmatched = np.concatenate([np.delete(a.innov, i_a), np.delete(b.innov, i_b)])
    cutoff = min(a.innov.max() if len(a.innov) else -1, b.innov.max() if len(b.innov) else -1)
    excess = int(np.sum(unmatched > cutoff))
    disjoint = len(unmatched) - excess
    n = max(len(a.innov), len(b.innov))
    n = 1 if n < config.NEAT_SMALL_GENOME else n
    w = float(np.mean(np.abs(a.weight[i_a] - b.weight[i_b]))) if len(i_a) else 0.0
    return config.NEAT_C1 * excess / n + config.NEAT_C2 * disjoint / n + config.NEAT_C3 * w


def node_depths(genome):
    """Longest enabled path from an input to each node (Kahn's order). Inputs have depth 0; an output or hidden
    node without enabled inputs has depth 1 (its value is its bias)."""
    nodes = genome.node_ids.tolist()
    depth = {i: 0 for i in INPUT_IDS}
    depth.update({n: 1 for n in nodes})
    succ = {n: [] for n in nodes}
    indegree = {n: 0 for n in nodes}
    on = genome.enabled
    for s, d in zip(genome.src[on].tolist(), genome.dst[on].tolist()):
        if s >= N_IN:                         # hidden source; inputs are already final
            succ[s].append(d)
            indegree[d] += 1
    ready = [n for n in nodes if indegree[n] == 0]
    while ready:
        node = ready.pop()
        for d in succ[node]:
            depth[d] = max(depth[d], depth[node] + 1)
            indegree[d] -= 1
            if indegree[d] == 0:
                ready.append(d)
    return depth


@dataclass(eq=False)
class Compiled:
    """A population of genomes as flat arrays (spec §9.9). Values live in one vector: each genome has a block
    of slots, its N_IN inputs and then its nodes. Each level holds one depth: the slots it computes, their biases,
    whether each is hidden (tanh) or an output (identity), and its enabled connections (source slot, position of the
    target within the level, weight)."""
    n: int
    size: int
    inputs: np.ndarray          # (n, N_IN) input slots
    outputs: np.ndarray         # (n, N_ACTIONS) output slots, in output order
    levels: list                # [(nodes, bias, hidden, src, target, weight)] for depths 1, 2, ...


def compile_population(genomes):
    """Flat arrays for the batched forward pass; compiled once per generation and task."""
    n = len(genomes)
    inputs = np.zeros((n, N_IN), dtype=np.int64)
    outputs = np.zeros((n, N_OUT), dtype=np.int64)
    node_slot, node_depth, node_bias, node_hidden = [], [], [], []
    conn_src, conn_dst, conn_w, conn_depth = [], [], [], []
    offset = 0
    for g, genome in enumerate(genomes):
        ids = genome.node_ids
        depth = node_depths(genome)
        depths = np.array([depth[node] for node in ids.tolist()], dtype=np.int64)
        slots = offset + N_IN + np.arange(len(ids))
        inputs[g] = offset + np.arange(N_IN)
        outputs[g] = slots[np.searchsorted(ids, np.arange(N_IN, N_IN + N_OUT))]
        node_slot.append(slots)
        node_depth.append(depths)
        node_bias.append(genome.bias)
        node_hidden.append(ids >= FIRST_HIDDEN)
        on = genome.enabled
        src, dst = genome.src[on], genome.dst[on]
        target = np.searchsorted(ids, dst)
        conn_src.append(np.where(src < N_IN, offset + src, offset + N_IN + np.searchsorted(ids, src)))
        conn_dst.append(slots[target])
        conn_w.append(genome.weight[on])
        conn_depth.append(depths[target])
        offset += N_IN + len(ids)
    if n == 0:
        return Compiled(0, 0, inputs, outputs, [])
    node_slot, node_depth = np.concatenate(node_slot), np.concatenate(node_depth)
    node_bias, node_hidden = np.concatenate(node_bias), np.concatenate(node_hidden)
    conn_src, conn_dst = np.concatenate(conn_src), np.concatenate(conn_dst)
    conn_w, conn_depth = np.concatenate(conn_w), np.concatenate(conn_depth)
    position = np.zeros(offset, dtype=np.int64)
    levels = []
    for d in range(1, int(node_depth.max()) + 1):
        nodes = node_slot[node_depth == d]
        position[nodes] = np.arange(len(nodes))
        here = conn_depth == d                                  # boolean masks keep each genome's order
        levels.append((nodes, node_bias[node_depth == d], node_hidden[node_depth == d], conn_src[here],
                       position[conn_dst[here]], conn_w[here]))
    return Compiled(n, offset, inputs, outputs, levels)


def forward(compiled, obs):
    """Output values (n, N_ACTIONS) of every genome for its own observation row: one gather, one bincount and one
    activation per depth, with no loop over cars."""
    v = np.zeros(compiled.size)
    v[compiled.inputs] = obs
    for nodes, bias, hidden, src, target, weight in compiled.levels:
        total = np.bincount(target, weights=v[src] * weight, minlength=len(nodes)) + bias
        v[nodes] = np.where(hidden, np.tanh(total), total)
    return v[compiled.outputs]


class NeatNet:
    """One NEAT controller as one masked dense matrix per depth; drives a car like nets.DenseNet, with the car's
    setup (spec §9.11; revision 3's when none is given)."""

    def __init__(self, genome, setup=None):
        self.genome = genome
        self.setup = car_setup.as_row(setup)
        ids = genome.node_ids.tolist()
        column = {i: i for i in INPUT_IDS}
        column.update({node: N_IN + k for k, node in enumerate(ids)})
        depth = node_depths(genome)
        bias = dict(zip(ids, genome.bias.tolist()))
        on = genome.enabled
        conns = list(zip(genome.src[on].tolist(), genome.dst[on].tolist(), genome.weight[on].tolist()))
        self.width = N_IN + len(ids)
        self.column = column                            # node id -> its column of node_values
        self.output_columns = np.array([column[o] for o in OUTPUT_IDS])
        self.levels = []
        for d in range(1, max(depth[node] for node in ids) + 1):
            nodes = [node for node in ids if depth[node] == d]
            position = {node: j for j, node in enumerate(nodes)}
            matrix = np.zeros((self.width, len(nodes)))
            for s, t, w in conns:
                if t in position:
                    matrix[column[s], position[t]] += w
            self.levels.append((np.array([column[node] for node in nodes]), matrix,
                                np.array([bias[node] for node in nodes]),
                                np.array([node >= FIRST_HIDDEN for node in nodes])))

    @property
    def n_params(self):
        return self.genome.n_params

    def node_values(self, obs):
        """Every node's value for the inputs `obs` (one row per car): the inputs, each hidden node's activation and
        each output's score, in the columns self.column gives."""
        obs = np.asarray(obs, dtype=float)
        values = np.zeros((len(obs), self.width))
        values[:, :N_IN] = obs
        for columns, matrix, bias, hidden in self.levels:
            total = values @ matrix + bias
            values[:, columns] = np.where(hidden, np.tanh(total), total)
        return values

    def outputs(self, obs):
        return self.node_values(obs)[:, self.output_columns]

    def act(self, obs):
        return np.argmax(self.outputs(obs), axis=1)

    def policy(self):
        return lambda obs, cars: self.act(obs)

    @classmethod
    def from_json(cls, path):
        return cls(Genome.from_dict(json.loads(Path(path).read_text())))


class NeatPopulation:
    """A NEAT population with the population interface of nets.DensePopulation: one genome and one setup row per
    car (spec §9.11)."""

    def __init__(self, genomes, setup=None):
        self.genomes = list(genomes)
        self.setup = car_setup.as_rows(setup, len(self.genomes))

    def __len__(self):
        return len(self.genomes)

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.int64).ravel()
        return NeatPopulation([self.genomes[int(i)] for i in indices], self.setup[indices])

    def policy(self):
        """Policy for sim.simulate: car i is driven by genome i. Compiled once, here."""
        compiled = compile_population(self.genomes)
        return lambda obs, cars: np.argmax(forward(compiled, obs), axis=1)

    def driver(self, i):
        return NeatNet(self.genomes[int(i)], self.setup[int(i)])


def to_keras(genome):
    """Equivalent Keras model (spec §9.9): per depth, a tanh Dense layer for the hidden nodes and a linear
    Dense layer for the outputs, each applied to the concatenation of the inputs and all shallower nodes, with zero
    weights where the genome has no connection; then a fixed selection layer that puts the outputs in output order,
    with a softmax. Used for the accuracy and timing columns only."""
    keras = keras_cpu()
    depth = node_depths(genome)
    ids = genome.node_ids.tolist()
    bias = dict(zip(ids, genome.bias.tolist()))
    on = genome.enabled
    conns = list(zip(genome.src[on].tolist(), genome.dst[on].tolist(), genome.weight[on].tolist()))
    x = keras.Input((N_IN,))
    tensor, available = x, list(INPUT_IDS)                       # node id of each column of `tensor`
    for d in range(1, max(depth[node] for node in ids) + 1):
        row = {node: i for i, node in enumerate(available)}
        new_tensors, new_ids = [], []
        for hidden in (True, False):
            group = [node for node in ids if depth[node] == d and (node >= FIRST_HIDDEN) == hidden]
            if not group:
                continue
            col = {node: j for j, node in enumerate(group)}
            kernel = np.zeros((len(available), len(group)))
            for s, t, w in conns:
                if t in col:
                    kernel[row[s], col[t]] += w
            layer = keras.layers.Dense(len(group), activation="tanh" if hidden else "linear")
            new_tensors.append(layer(tensor))
            layer.set_weights([kernel, np.array([bias[node] for node in group])])
            new_ids += group
        tensor = keras.layers.Concatenate()([tensor, *new_tensors])
        available += new_ids
    select = np.zeros((len(available), N_OUT))
    index = {node: i for i, node in enumerate(available)}
    for j, node in enumerate(OUTPUT_IDS):
        select[index[node], j] = 1.0
    layer = keras.layers.Dense(N_OUT, activation="softmax", trainable=False)
    probabilities = layer(tensor)
    layer.set_weights([select, np.zeros(N_OUT)])
    return keras.Model(x, probabilities)


def save_neat_champion(approach, seed, genome, champion_dir=config.CHAMPIONS_DIR, setup=None):
    """Final NEAT controller (spec §9.9): its genome as <seed>.json, which drives, the equivalent Keras model
    as <seed>.keras, and its setup as <seed>.setup.json (spec §6; revision 3's when setup is None). Returns
    the Keras file's path."""
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_keras(genome).save(path)
    paths.champion_path(approach, seed, ".json", champion_dir).write_text(json.dumps(genome.to_dict()))
    paths.champion_path(approach, seed, ".setup.json", champion_dir).write_text(
        json.dumps(car_setup.to_dict(car_setup.as_row(setup))))
    return path


@dataclass(eq=False)
class Species:
    """Genomes of one niche: its representative (from the previous generation), members (indices into the current
    population), best fitness so far at the current level and target, and generations since that last improved."""
    id: int
    representative: Genome
    members: list = field(default_factory=list)
    best_fitness: float = float("-inf")
    stale: int = 0


class SpeciesSet:
    """Speciation by compatibility distance with a threshold steered towards NEAT_TARGET_SPECIES species, and the
    allotment of children to species (spec §9.9)."""

    def __init__(self, threshold=config.NEAT_THRESHOLD0):
        self.threshold = float(threshold)
        self.species = []
        self.next_id = 0

    def speciate(self, rng, genomes, fitness):
        for s in self.species:
            s.members = []
        for i, genome in enumerate(genomes):
            for s in self.species:
                if distance(genome, s.representative) < self.threshold:
                    s.members.append(i)
                    break
            else:
                self.species.append(Species(self.next_id, genome, [i]))
                self.next_id += 1
        self.species = [s for s in self.species if s.members]
        for s in self.species:
            best = float(np.max(fitness[s.members]))
            if best > s.best_fitness:
                s.best_fitness, s.stale = best, 0
            else:
                s.stale += 1
        if len(self.species) > config.NEAT_TARGET_SPECIES:
            self.threshold += config.NEAT_THRESHOLD_STEP
        elif len(self.species) < config.NEAT_TARGET_SPECIES:
            self.threshold = max(self.threshold - config.NEAT_THRESHOLD_STEP, config.NEAT_THRESHOLD_STEP)
        for s in self.species:
            s.representative = genomes[s.members[int(rng.integers(len(s.members)))]]

    def restart_records(self):
        """Forget every species' best fitness and stagnation count, so stagnation is measured within one level and
        target (spec §9.9): maps change every generation, so a best fitness is a record of noisy scores, and
        after a level-up or a shorter target the old records are out of reach."""
        for s in self.species:
            s.best_fitness, s.stale = float("-inf"), 0

    def allot(self, n_children, fitness):
        """Children per species id: proportional to mean member fitness, largest remainder; a stagnant species gets
        none unless it holds the population's best car."""
        best = int(np.argmax(fitness))
        eligible = [s.stale < config.NEAT_STAGNATION or best in s.members for s in self.species]
        means = np.array([max(float(np.mean(fitness[s.members])), 0.0) if ok else 0.0
                          for s, ok in zip(self.species, eligible)])
        if means.sum() <= 0.0:
            means = np.array(eligible, dtype=float)
        quotas = n_children * means / means.sum()
        counts = np.floor(quotas).astype(np.int64)
        extra = np.argsort(-(quotas - counts), kind="stable")[:n_children - int(counts.sum())]
        counts[extra] += 1
        return {s.id: int(c) for s, c in zip(self.species, counts)}

    def state(self):
        return {"threshold": self.threshold, "next_id": self.next_id,
                "species": [{"id": s.id, "representative": s.representative.to_dict(), "members": list(s.members),
                             "best_fitness": s.best_fitness, "stale": s.stale} for s in self.species]}

    @classmethod
    def from_state(cls, state):
        species = cls(state["threshold"])
        species.next_id = state["next_id"]
        species.species = [Species(s["id"], Genome.from_dict(s["representative"]), list(s["members"]),
                                   s["best_fitness"], s["stale"]) for s in state["species"]]
        return species


RESTART_EVENTS = ("level_up", "goal", "tightened")    # engine events after which species records restart


class NEAT:
    """NEAT as a method of the evolution engine (spec §9.9): kept cars first, then children produced within
    species, with the two-phase step size of the ES. A child inherits the setup of its fitter (or only) parent, which
    the setup mutation then changes (spec §9.11)."""
    approach = "neat"

    def __init__(self, pop_size=config.NEAT_POP):
        self.pop_size = int(pop_size)
        self.innovations = Innovations()
        self.species = SpeciesSet()
        self.steps = StepSize(config.NEAT_SIGMA0)
        self.population = None
        self.is_kept = None
        self.ids = None
        self.parent_ids = None
        self.next_id = 0

    @property
    def step_size(self):
        return self.steps.sigma

    @property
    def setup(self):
        """Every car's setup, aligned with the population (spec §9.11)."""
        return self.population.setup

    def initialize(self, rng):
        genomes = [initial_genome(rng, self.innovations) for _ in range(self.pop_size)]
        self.population = NeatPopulation(genomes, setup=car_setup.random(rng, self.pop_size))   # spec §9.11
        self.is_kept = np.zeros(self.pop_size, dtype=bool)
        self.ids = np.arange(self.pop_size, dtype=np.int64)
        self.parent_ids = np.full(self.pop_size, -1, dtype=np.int64)
        self.next_id = self.pop_size

    def child_success_rate(self, fitness):
        """Share of the previous children that beat their parent in this generation (a tie counts half); children
        whose parent is no longer in the population are left out; nan if none remain."""
        position = {int(car): k for k, car in enumerate(self.ids)}
        child, parent = [], []
        for k in np.flatnonzero(~self.is_kept):
            p = int(self.parent_ids[k])
            if p in position:
                child.append(k)
                parent.append(position[p])
        if not child:
            return float("nan")
        return success_fraction(fitness[np.array(child)], fitness[np.array(parent)])

    def _tournament(self, rng, members, fitness):
        entrants = members[rng.integers(0, len(members), config.NEAT_TOURNAMENT)]
        return int(entrants[np.argmax(fitness[entrants])])

    def advance(self, rng, result, kept, event, boost=False):
        """Kept cars first, then children within species; with boost (a stall, spec §9.4) the children's output
        connections and setups mutate harder, from the same draws."""
        from racing import evolution                  # evolution imports nets, which imports neat lazily
        fitness, success = result.fitness, result.all_success
        genomes, setups = self.population.genomes, self.population.setup
        top = np.argsort(-fitness, kind="stable")[:config.ES_MU]
        child_success = evolution.children_success(result, self.is_kept)
        self.steps.update(float(np.mean(success[top])), child_success, self.child_success_rate(fitness))
        if event == "level_up":
            self.steps.reset()
        self.species.speciate(rng, genomes, fitness)
        if event in RESTART_EVENTS:                   # the children drive at the new level or target
            self.species.restart_records()
        allotment = self.species.allot(self.pop_size - len(kept), fitness)
        new_genomes, new_parents, fitter_rows = [], [], []
        for s in self.species.species:
            members = np.asarray(s.members)
            for _ in range(allotment[s.id]):
                a = self._tournament(rng, members, fitness)
                cross = rng.random() < config.NEAT_CROSSOVER
                if cross and len(members) > 1:
                    b = self._tournament(rng, members, fitness)
                    fitter, other = (a, b) if fitness[a] >= fitness[b] else (b, a)
                    child = crossover(rng, genomes[fitter], genomes[other])
                else:
                    fitter, child = a, genomes[a]
                new_genomes.append(mutate(rng, child, self.innovations, self.steps.sigma, boost=boost))
                new_parents.append(int(self.ids[fitter]))
                fitter_rows.append(fitter)
        kept = np.asarray(kept, dtype=np.int64)
        children_setups = car_setup.mutate(rng, setups[np.asarray(fitter_rows, dtype=np.int64)],   # after the genomes
                                           boost=boost)
        self.population = NeatPopulation([genomes[i] for i in kept] + new_genomes,
                                         setup=np.concatenate([setups[kept], children_setups]))
        self.is_kept = np.arange(len(self.population)) < len(kept)
        new_ids = np.arange(self.next_id, self.next_id + len(new_genomes), dtype=np.int64)
        self.next_id += len(new_genomes)
        self.ids = np.concatenate([self.ids[kept], new_ids])
        self.parent_ids = np.concatenate([np.full(len(kept), -1, dtype=np.int64),
                                          np.asarray(new_parents, dtype=np.int64)])

    def state(self):
        return {"pop_size": self.pop_size, "genomes": [g.to_dict() for g in self.population.genomes],
                "setup": self.population.setup.copy(),
                "is_kept": self.is_kept.copy(), "ids": self.ids.copy(), "parent_ids": self.parent_ids.copy(),
                "next_id": self.next_id, "innovations": self.innovations.state(), "species": self.species.state(),
                "steps": self.steps.state()}

    def load_state(self, state):
        self.pop_size = int(state["pop_size"])
        self.population = NeatPopulation([Genome.from_dict(g) for g in state["genomes"]],
                                         setup=np.asarray(state["setup"], dtype=float))
        self.is_kept = np.asarray(state["is_kept"], dtype=bool).copy()
        self.ids = np.asarray(state["ids"], dtype=np.int64).copy()
        self.parent_ids = np.asarray(state["parent_ids"], dtype=np.int64).copy()
        self.next_id = int(state["next_id"])
        self.innovations = Innovations.from_state(state["innovations"])
        self.species = SpeciesSet.from_state(state["species"])
        self.steps = StepSize.from_state(state["steps"])

    def save_champion(self, index, seed, champion_dir):
        return save_neat_champion(self.approach, seed, self.population.genomes[int(index)], champion_dir,
                                  setup=self.population.setup[int(index)])

    def log_fields(self):
        genomes = self.population.genomes
        return {"species": len(self.species.species),
                "mean_nodes": float(np.mean([np.sum(g.node_ids >= FIRST_HIDDEN) for g in genomes])),
                "mean_conns": float(np.mean([np.sum(g.enabled) for g in genomes]))}


def run_neat(seed=0, pop_size=config.NEAT_POP, out_dir=None, champion_dir=config.CHAMPIONS_DIR, n_jobs=1,
             on_generation=None, start_level=1, max_generations=None):
    """One NEAT run with the evolution engine (spec §9.2-9.5, §9.9): resumable, written to runs/neat/seed<k>."""
    from racing import evolution                      # evolution imports nets, which imports neat lazily
    out_dir = Path(out_dir or paths.run_dir("neat", seed))
    return evolution.run(NEAT(pop_size), seed, out_dir, champion_dir=champion_dir, n_jobs=n_jobs,
                         on_generation=on_generation, start_level=start_level, max_generations=max_generations)
