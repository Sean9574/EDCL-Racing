import numpy as np
import pytest

from racing import config, neat


def grown(rng, innovations, rounds=25):
    """A genome grown by many structural mutations, with some connections disabled and random biases."""
    genome = neat.initial_genome(rng, innovations)
    for _ in range(rounds):
        genome = neat.add_node(rng, genome, innovations)
        genome = neat.add_connection(rng, genome, innovations)
        if rng.random() < 0.3:
            k = int(rng.integers(len(genome.innov)))
            genome.enabled[k] = not genome.enabled[k]
    genome.bias = rng.normal(0.0, 0.5, len(genome.bias))
    return genome


def acyclic(genome):
    """True if every node can be ordered so that all connections (enabled or not) point forwards."""
    succ, indegree = {}, {int(n): 0 for n in (*neat.INPUT_IDS, *genome.node_ids.tolist())}
    for s, d in zip(genome.src.tolist(), genome.dst.tolist()):
        succ.setdefault(s, []).append(d)
        indegree[d] += 1
    ready = [n for n, k in indegree.items() if k == 0]
    seen = 0
    while ready:
        node = ready.pop()
        seen += 1
        for d in succ.get(node, ()):
            indegree[d] -= 1
            if indegree[d] == 0:
                ready.append(d)
    return seen == len(indegree)


NEAT_NAMES = ("NEAT_POP", "NEAT_CROSSOVER", "NEAT_TOURNAMENT", "NEAT_DISABLE_INHERIT", "NEAT_PERTURB",
              "NEAT_REPLACE", "NEAT_ADD_CONN", "NEAT_ADD_NODE", "NEAT_TOGGLE", "NEAT_SIGMA0", "NEAT_C1", "NEAT_C2",
              "NEAT_C3", "NEAT_SMALL_GENOME", "NEAT_THRESHOLD0", "NEAT_THRESHOLD_STEP", "NEAT_TARGET_SPECIES",
              "NEAT_STAGNATION")


def test_neat_is_an_evolutionary_approach_with_its_own_color():
    assert "neat" in config.APPROACHES and "neat" in config.EVOLUTION_APPROACHES
    assert config.LABELS["neat"] == "NEAT" and config.COLORS["neat"] == "#F0E442"


def test_neat_settings_are_recorded_for_resumption():
    from racing import checkpoint
    record = checkpoint.settings_record()
    for name in NEAT_NAMES:
        assert name in checkpoint.SETTINGS_KEYS
        assert record[name] == getattr(config, name)


def test_initial_genome_connects_every_input_to_every_output():
    genome = neat.initial_genome(np.random.default_rng(0), neat.Innovations())
    assert genome.node_ids.tolist() == list(neat.OUTPUT_IDS)
    assert np.all(genome.bias == 0.0)
    assert len(genome.innov) == neat.N_IN * neat.N_OUT == 150
    assert set(zip(genome.src.tolist(), genome.dst.tolist())) == {(i, o) for i in neat.INPUT_IDS for o in neat.OUTPUT_IDS}
    assert np.all(np.diff(genome.innov) > 0) and genome.enabled.all()
    assert genome.n_params == 165
    assert 0.8 < np.std(genome.weight) < 1.2                     # standard normal weights


def test_innovation_record_is_shared_by_all_genomes():
    rng = np.random.default_rng(1)
    innovations = neat.Innovations()
    a, b = neat.initial_genome(rng, innovations), neat.initial_genome(rng, innovations)
    assert np.array_equal(a.innov, b.innov) and not np.array_equal(a.weight, b.weight)
    assert innovations.conn(0, 10) == innovations.conn(0, 10) == int(a.innov[0])
    node = innovations.split(5)
    assert innovations.split(5) == node == neat.FIRST_HIDDEN
    assert innovations.fresh_node() == node + 1 and innovations.split(5) == node
    copy = neat.Innovations.from_state(innovations.state())
    assert copy.state() == innovations.state()
    assert copy.conn(3, 12) == innovations.conn(3, 12)


def test_creates_cycle_follows_all_connections():
    genome = neat.Genome.from_dict({
        "node_ids": [10, 25, 26], "bias": [0.0] * 3,
        "innov": [0, 1, 2], "src": [0, 25, 26], "dst": [25, 26, 10],
        "weight": [1.0] * 3, "enabled": [True, False, True]})   # 0 -> 25 -> 26 -> 10, the middle one disabled
    assert neat.creates_cycle(genome, 25, 25)                    # self-loop
    assert neat.creates_cycle(genome, 26, 25)                    # closes 25 -> 26 even though it is disabled
    assert not neat.creates_cycle(genome, 0, 26)
    assert not neat.creates_cycle(genome, 25, 10)


def test_add_node_splits_a_connection():
    rng = np.random.default_rng(3)
    innovations = neat.Innovations()
    genome = neat.initial_genome(rng, innovations)
    child = neat.add_node(rng, genome, innovations)
    assert len(child.node_ids) == len(genome.node_ids) + 1
    node = int(child.node_ids[-1])
    assert node == neat.FIRST_HIDDEN and child.bias[-1] == 0.0
    split = np.flatnonzero(genome.enabled & ~child.enabled[np.searchsorted(child.innov, genome.innov)])
    assert len(split) == 1
    s, d, w = int(genome.src[split[0]]), int(genome.dst[split[0]]), float(genome.weight[split[0]])
    new = {(int(a), int(b)): float(v) for a, b, v in zip(child.src, child.dst, child.weight)}
    assert new[(s, node)] == 1.0 and new[(node, d)] == w
    assert np.all(np.diff(child.innov) > 0)
    assert genome.enabled.all()                                  # the parent is unchanged


def test_the_same_split_gets_the_same_node_unless_the_genome_has_it():
    rng = np.random.default_rng(4)
    innovations = neat.Innovations()
    a = neat.initial_genome(rng, innovations)
    b = a.copy()
    a.enabled[:] = False
    a.enabled[7] = True
    b.enabled[:] = False
    b.enabled[7] = True
    split_a = neat.add_node(rng, a, innovations)
    split_b = neat.add_node(rng, b, innovations)
    assert split_a.node_ids[-1] == split_b.node_ids[-1] == neat.FIRST_HIDDEN
    assert innovations.state() == neat.Innovations.from_state(innovations.state()).state()
    again = split_a.copy()
    again.enabled[:] = False
    again.enabled[np.searchsorted(again.innov, a.innov[7])] = True          # re-enable the split connection
    twice = neat.add_node(rng, again, innovations)
    assert len(set(twice.node_ids.tolist())) == len(twice.node_ids)          # no duplicate node
    assert int(twice.node_ids[-1]) == neat.FIRST_HIDDEN + 1
    assert innovations.split(int(a.innov[7])) == neat.FIRST_HIDDEN           # the record keeps the first node


def test_structural_mutations_keep_genomes_valid():
    rng = np.random.default_rng(5)
    innovations = neat.Innovations()
    for _ in range(5):
        genome = grown(rng, innovations, rounds=40)
        pairs = list(zip(genome.src.tolist(), genome.dst.tolist()))
        assert len(pairs) == len(set(pairs))                                  # no duplicate connection
        assert all(s < neat.N_IN or s >= neat.FIRST_HIDDEN for s, _ in pairs)  # sources: inputs or hidden
        assert all(d >= neat.N_IN for _, d in pairs)                           # targets: outputs or hidden
        assert set(genome.dst.tolist()) <= set(genome.node_ids.tolist())
        assert np.all(np.diff(genome.innov) > 0) and np.all(np.diff(genome.node_ids) > 0)
        assert len(genome.bias) == len(genome.node_ids)
        assert acyclic(genome)


def test_weight_mutation_rates(monkeypatch):
    monkeypatch.setattr(config, "NEAT_ADD_CONN", 0.0)
    monkeypatch.setattr(config, "NEAT_ADD_NODE", 0.0)
    monkeypatch.setattr(config, "NEAT_TOGGLE", 0.0)
    rng = np.random.default_rng(6)
    innovations = neat.Innovations()
    genomes = [neat.initial_genome(rng, innovations) for _ in range(40)]
    before = np.concatenate([g.weight for g in genomes])
    still = np.concatenate([neat.mutate(rng, g, innovations, 0.0).weight for g in genomes])
    replaced = np.mean(still != before)
    assert abs(replaced - config.NEAT_REPLACE) < 0.02                 # σ = 0: only replacements change weights
    moved = np.concatenate([neat.mutate(rng, g, innovations, 1.0).weight for g in genomes])
    assert abs(np.mean(moved != before) - (config.NEAT_REPLACE + config.NEAT_PERTURB)) < 0.03
    assert all(g.n_params == 165 for g in genomes)                     # the parents are unchanged


def test_mutate_adds_structure_with_its_probabilities(monkeypatch):
    monkeypatch.setattr(config, "NEAT_ADD_NODE", 1.0)
    monkeypatch.setattr(config, "NEAT_ADD_CONN", 0.0)
    rng = np.random.default_rng(7)
    innovations = neat.Innovations()
    genome = neat.mutate(rng, neat.initial_genome(rng, innovations), innovations, 0.1)
    assert len(genome.node_ids) == neat.N_OUT + 1 and len(genome.innov) == 152


def test_crossover_aligns_genes_by_innovation():
    rng = np.random.default_rng(8)
    innovations = neat.Innovations()
    other = neat.initial_genome(rng, innovations)
    fitter = neat.add_node(rng, other.copy(), innovations)
    fitter.weight = fitter.weight + 10.0
    child = neat.crossover(rng, fitter, other)
    assert np.array_equal(child.innov, fitter.innov) and np.array_equal(child.node_ids, fitter.node_ids)
    common = np.isin(fitter.innov, other.innov)
    from_other = np.isin(child.weight[common], other.weight)
    assert 0.3 < from_other.mean() < 0.7                              # matching genes: either parent
    assert np.array_equal(child.weight[~common], fitter.weight[~common])   # disjoint and excess: the fitter parent


def test_a_gene_disabled_in_either_parent_is_usually_disabled():
    rng = np.random.default_rng(9)
    innovations = neat.Innovations()
    fitter = neat.initial_genome(rng, innovations)
    other = fitter.copy()
    other.enabled[:] = False
    disabled = np.mean([~neat.crossover(rng, fitter, other).enabled for _ in range(40)])
    assert abs(disabled - config.NEAT_DISABLE_INHERIT) < 0.03


def test_distance():
    rng = np.random.default_rng(10)
    innovations = neat.Innovations()
    a = neat.initial_genome(rng, innovations)
    assert neat.distance(a, a) == 0.0
    b = a.copy()
    b.weight = b.weight + 1.0
    assert np.isclose(neat.distance(a, b), config.NEAT_C3 * 1.0)
    c = neat.add_node(rng, a, innovations)                             # two excess genes, N = 152
    assert np.isclose(neat.distance(a, c), config.NEAT_C1 * 2 / 152)
    small = neat.Genome.from_dict({"node_ids": [10], "bias": [0.0], "innov": [0, 3], "src": [0, 1], "dst": [10, 10],
                                   "weight": [1.0, 1.0], "enabled": [True, True]})
    other = neat.Genome.from_dict({"node_ids": [10], "bias": [0.0], "innov": [0, 1, 5], "src": [0, 2, 3],
                                   "dst": [10, 10, 10], "weight": [2.0, 1.0, 1.0], "enabled": [True, True, True]})
    # matching 0 (|Δw| = 1); disjoint 1 and 3 (below the smaller maximum 3); excess 5; N = 1 below 20 genes
    assert np.isclose(neat.distance(small, other), config.NEAT_C1 * 1 + config.NEAT_C2 * 2 + config.NEAT_C3 * 1.0)


def test_node_depths_are_longest_enabled_paths():
    genome = neat.Genome.from_dict({
        "node_ids": [10, 11, 25, 26], "bias": [0.0] * 4,
        "innov": [0, 1, 2, 3, 4, 5], "src": [0, 25, 0, 26, 1, 2], "dst": [25, 26, 26, 10, 11, 10],
        "weight": [1.0] * 6, "enabled": [True, True, True, True, False, True]})
    depth = neat.node_depths(genome)
    assert depth[0] == 0 and depth[25] == 1 and depth[26] == 2 and depth[10] == 3
    assert depth[11] == 1                                              # no enabled input: bias only
    assert all(depth[o] >= 1 for o in neat.OUTPUT_IDS if o in depth)


def test_genome_round_trips_through_a_dict():
    rng = np.random.default_rng(11)
    genome = grown(rng, neat.Innovations())
    copy = neat.Genome.from_dict(genome.to_dict())
    for name in neat.Genome.FIELDS:
        assert np.array_equal(getattr(copy, name), getattr(genome, name))
        assert getattr(copy, name).dtype == getattr(genome, name).dtype


from racing import nets, sim  # noqa: E402  (appended imports)


def reference_outputs(genome, row):
    """Direct per-genome computation: each node from its enabled incoming connections, recursively."""
    bias = dict(zip(genome.node_ids.tolist(), genome.bias.tolist()))
    incoming = {}
    for s, d, w, on in zip(genome.src.tolist(), genome.dst.tolist(), genome.weight.tolist(), genome.enabled.tolist()):
        if on:
            incoming.setdefault(d, []).append((s, w))
    memo = {}

    def value(node):
        if node < neat.N_IN:
            return row[node]
        if node not in memo:
            total = bias[node] + sum(w * value(s) for s, w in incoming.get(node, []))
            memo[node] = np.tanh(total) if node >= neat.FIRST_HIDDEN else total
        return memo[node]

    return np.array([value(o) for o in neat.OUTPUT_IDS])


def softmax(values):
    e = np.exp(values - values.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.fixture
def genomes():
    rng = np.random.default_rng(20)
    innovations = neat.Innovations()
    return [grown(rng, innovations, rounds=int(r)) for r in rng.integers(0, 30, 12)]


def test_batched_forward_matches_direct_computation(genomes):
    rng = np.random.default_rng(21)
    obs = rng.random((len(genomes), config.N_INPUTS))
    out = neat.forward(neat.compile_population(genomes), obs)
    assert out.shape == (len(genomes), config.N_ACTIONS)
    for i, genome in enumerate(genomes):
        assert np.allclose(out[i], reference_outputs(genome, obs[i]))


def test_batched_forward_does_not_depend_on_the_other_genomes(genomes):
    rng = np.random.default_rng(22)
    obs = rng.random((len(genomes), config.N_INPUTS))
    together = neat.forward(neat.compile_population(genomes), obs)
    for i in (0, 5, 11):
        alone = neat.forward(neat.compile_population([genomes[i]]), obs[i:i + 1])
        assert np.array_equal(alone[0], together[i])


def test_neat_net_matches_the_batched_forward_pass(genomes):
    rng = np.random.default_rng(23)
    rows = rng.random((50, config.N_INPUTS))
    for genome in genomes[:4]:
        net = neat.NeatNet(genome)
        batched = neat.forward(neat.compile_population([genome] * 50), rows)
        assert np.allclose(net.outputs(rows), batched)
        assert np.array_equal(net.act(rows), np.argmax(batched, axis=1))
        assert net.n_params == genome.n_params


def test_population_interface(genomes):
    population = neat.NeatPopulation(genomes)
    assert len(population) == 12
    part = population.subset([3, 1])
    assert len(part) == 2 and part.genomes[0] is genomes[3] and part.genomes[1] is genomes[1]
    obs = np.random.default_rng(24).random((2, config.N_INPUTS))
    classes = part.policy()(obs, None)
    assert classes.shape == (2,) and classes.min() >= 0 and classes.max() < config.N_ACTIONS
    assert classes[0] == population.driver(3).act(obs[:1])[0]


def test_a_neat_population_drives_in_the_simulator(genomes, circle_track):
    result = sim.simulate(circle_track, neat.NeatPopulation(genomes).policy(), len(genomes), laps=1)
    assert len(result.fitness) == len(genomes) and np.all(result.fitness >= 0.0)


def test_keras_export_chooses_the_same_actions(genomes):
    rng = np.random.default_rng(25)
    obs = rng.random((300, config.N_INPUTS)).astype(np.float32)
    for genome in (genomes[0], max(genomes, key=lambda g: len(g.node_ids))):
        model = neat.to_keras(genome)
        probabilities = model.predict(obs, verbose=0)
        expected = softmax(neat.NeatNet(genome).outputs(obs.astype(float)))
        assert np.allclose(probabilities, expected, atol=1e-4)
        assert np.mean(np.argmax(probabilities, axis=1) == neat.NeatNet(genome).act(obs)) >= 0.99


def test_saved_neat_champion_loads_as_its_genome(genomes, tmp_path):
    genome = max(genomes, key=lambda g: len(g.node_ids))
    path = neat.save_neat_champion("neat", 1, genome, tmp_path)
    assert path == tmp_path / "neat" / "seed1.keras" and path.exists()
    assert (tmp_path / "neat" / "seed1.json").exists()
    assert nets.champion_seeds(tmp_path) == {"neat": [1]}
    model, driver = nets.load_champion("neat", 1, tmp_path)
    assert isinstance(driver, neat.NeatNet) and driver.n_params == genome.n_params
    obs = np.random.default_rng(26).random((40, config.N_INPUTS))
    assert np.array_equal(driver.act(obs), neat.NeatNet(genome).act(obs))
    assert np.array_equal(neat.NeatNet.from_json(tmp_path / "neat" / "seed1.json").act(obs), driver.act(obs))


def test_dense_champion_architecture_is_read_from_the_keras_model(tmp_path):
    rng = np.random.default_rng(27)
    flat = nets.init_population(rng, 1, hidden=(8,))[0] + rng.normal(0, 0.2, nets.n_params((8,)))
    folder = tmp_path / "es"
    folder.mkdir()
    nets.to_keras(flat, (8,), "tanh").save(folder / "seed0.keras")
    np.save(folder / "seed0.npy", flat)                     # dense champions write no .json (contract)
    _, driver = nets.load_champion("es", 0, tmp_path)
    obs = rng.random((30, config.N_INPUTS))
    assert isinstance(driver, nets.DenseNet)
    assert np.array_equal(driver.act(obs), nets.from_flat(flat, (8,), "tanh").act(obs))
