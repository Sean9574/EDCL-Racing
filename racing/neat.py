"""NEAT, neuroevolution of augmenting topologies (revision 3 §5.4): feed-forward networks whose structure grows by
mutation. A genome lists node genes (output and hidden nodes, each with a bias; the inputs are implicit) and
connection genes. Every structural mutation is recorded for the whole run, so the same connection or split has the
same innovation number or node id in every genome, which aligns genomes for crossover and speciation."""
from dataclasses import dataclass

import numpy as np

from racing import config

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
