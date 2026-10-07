"""NEAT, neuroevolution of augmenting topologies (revision 3 §5.4): feed-forward networks whose structure grows by
mutation. A genome lists node genes (output and hidden nodes, each with a bias; the inputs are implicit) and
connection genes. Every structural mutation is recorded for the whole run, so the same connection or split has the
same innovation number or node id in every genome, which aligns genomes for crossover and speciation."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from racing import config, paths
from racing.backend import keras_cpu

N_IN = config.N_INPUTS                         # 10
N_OUT = config.N_ACTIONS                       # 15
INPUT_IDS = range(0, N_IN)
OUTPUT_IDS = range(N_IN, N_IN + N_OUT)
FIRST_HIDDEN = N_IN + N_OUT                    # 25
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
        """Enabled connections plus biases: the parameters the network uses (revision 3 §5.4)."""
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
    """Every input connected to every output (150 connections), standard normal weights, zero biases."""
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


def _mutate_values(rng, values, sigma):
    """Each value replaced by a standard normal value with probability NEAT_REPLACE, otherwise perturbed by
    N(0, σ²) with probability NEAT_PERTURB."""
    u = rng.random(len(values))
    noise = rng.normal(0.0, sigma, len(values))
    fresh = rng.standard_normal(len(values))
    replace = u < config.NEAT_REPLACE
    perturb = ~replace & (u < config.NEAT_REPLACE + config.NEAT_PERTURB)
    return np.where(replace, fresh, np.where(perturb, values + noise, values))


def mutate(rng, genome, innovations, sigma):
    """A mutated copy (revision 3 §5.4): weights and biases, then with their own probabilities a new connection, a
    new node and one toggled connection."""
    g = genome.copy()
    g.weight = _mutate_values(rng, g.weight, sigma)
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
    """Compatibility distance δ = C1·E/N + C2·D/N + C3·W̄ (revision 3 §5.4): E excess genes (beyond the smaller
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
    """A population of genomes as flat arrays (revision 3 §5.4). Values live in one vector: each genome has a block
    of slots, its 10 inputs and then its nodes. Each level holds one depth: the slots it computes, their biases,
    whether each is hidden (tanh) or an output (identity), and its enabled connections (source slot, position of the
    target within the level, weight)."""
    n: int
    size: int
    inputs: np.ndarray          # (n, 10) input slots
    outputs: np.ndarray         # (n, 15) output slots, in output order
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
    """Output values (n, 15) of every genome for its own observation row: one gather, one bincount and one
    activation per depth, with no loop over cars."""
    v = np.zeros(compiled.size)
    v[compiled.inputs] = obs
    for nodes, bias, hidden, src, target, weight in compiled.levels:
        total = np.bincount(target, weights=v[src] * weight, minlength=len(nodes)) + bias
        v[nodes] = np.where(hidden, np.tanh(total), total)
    return v[compiled.outputs]


class NeatNet:
    """One NEAT controller as one masked dense matrix per depth; drives a car like nets.DenseNet."""

    def __init__(self, genome):
        self.genome = genome
        ids = genome.node_ids.tolist()
        column = {i: i for i in INPUT_IDS}
        column.update({node: N_IN + k for k, node in enumerate(ids)})
        depth = node_depths(genome)
        bias = dict(zip(ids, genome.bias.tolist()))
        on = genome.enabled
        conns = list(zip(genome.src[on].tolist(), genome.dst[on].tolist(), genome.weight[on].tolist()))
        self.width = N_IN + len(ids)
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

    def outputs(self, obs):
        obs = np.asarray(obs, dtype=float)
        values = np.zeros((len(obs), self.width))
        values[:, :N_IN] = obs
        for columns, matrix, bias, hidden in self.levels:
            total = values @ matrix + bias
            values[:, columns] = np.where(hidden, np.tanh(total), total)
        return values[:, self.output_columns]

    def act(self, obs):
        return np.argmax(self.outputs(obs), axis=1)

    def policy(self):
        return lambda obs, cars: self.act(obs)

    @classmethod
    def from_json(cls, path):
        return cls(Genome.from_dict(json.loads(Path(path).read_text())))


class NeatPopulation:
    """A NEAT population with the population interface of nets.DensePopulation."""

    def __init__(self, genomes):
        self.genomes = list(genomes)

    def __len__(self):
        return len(self.genomes)

    def subset(self, indices):
        return NeatPopulation([self.genomes[int(i)] for i in np.asarray(indices).ravel()])

    def policy(self):
        """Policy for sim.simulate: car i is driven by genome i. Compiled once, here."""
        compiled = compile_population(self.genomes)
        return lambda obs, cars: np.argmax(forward(compiled, obs), axis=1)

    def driver(self, i):
        return NeatNet(self.genomes[int(i)])


def to_keras(genome):
    """Equivalent Keras model (revision 3 §5.4): per depth, a tanh Dense layer for the hidden nodes and a linear
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


def save_neat_champion(approach, seed, genome, champion_dir=config.CHAMPIONS_DIR):
    """Final NEAT controller (revision 3 §5.4): its genome as <seed>.json, which drives, and the equivalent Keras
    model as <seed>.keras. Returns the Keras file's path."""
    path = paths.champion_path(approach, seed, champion_dir=champion_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_keras(genome).save(path)
    paths.champion_path(approach, seed, ".json", champion_dir).write_text(json.dumps(genome.to_dict()))
    return path
