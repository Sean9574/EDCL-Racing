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


import pickle  # noqa: E402

from racing import evolution, population  # noqa: E402


def result_for(fitness, success, maps=1):
    """A TrialResult with the same fitness and success on every map."""
    success = np.repeat(np.asarray(success, dtype=bool)[None, :], maps, axis=0)
    return population.TrialResult(fitness=np.asarray(fitness, dtype=float), success=success,
                                  laps=np.where(success, config.LAPS, 1),
                                  finish_time=np.where(success, 60.0, np.nan))


def clusters(rng):
    innovations = neat.Innovations()
    near = [neat.initial_genome(rng, innovations) for _ in range(6)]     # pairwise δ ≈ 0.4 · 1.13
    far = []
    for genome in near[:4]:
        shifted = genome.copy()
        shifted.weight = shifted.weight + 20.0                          # δ to the near ones ≈ 8
        far.append(shifted)
    return near + far


def test_speciation_groups_genomes_by_distance():
    rng = np.random.default_rng(30)
    genomes = clusters(rng)
    species = neat.SpeciesSet(threshold=3.0)
    species.speciate(rng, genomes, np.ones(len(genomes)))
    assert sorted(sorted(s.members) for s in species.species) == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]]
    assert np.isclose(species.threshold, 3.0 - config.NEAT_THRESHOLD_STEP)          # 2 species < 20: tighten
    for s in species.species:
        assert any(np.array_equal(s.representative.weight, genomes[m].weight) for m in s.members)


def test_threshold_never_drops_below_one_step():
    rng = np.random.default_rng(31)
    species = neat.SpeciesSet(threshold=0.15)
    genomes = clusters(rng)
    for _ in range(5):
        species.speciate(rng, genomes, np.ones(len(genomes)))
    assert species.threshold >= config.NEAT_THRESHOLD_STEP - 1e-12


def test_allotment_is_proportional_with_largest_remainder():
    rng = np.random.default_rng(32)
    genomes = clusters(rng)
    fitness = np.array([3.0] * 6 + [1.0] * 4)
    species = neat.SpeciesSet(threshold=3.0)
    species.speciate(rng, genomes, fitness)
    near = next(s.id for s in species.species if 0 in s.members)
    far = next(s.id for s in species.species if 6 in s.members)
    assert species.allot(7, fitness) == {near: 5, far: 2}           # quotas 5.25 and 1.75
    assert sum(species.allot(13, fitness).values()) == 13


def test_stagnant_species_get_no_children_unless_they_hold_the_best():
    rng = np.random.default_rng(33)
    genomes = clusters(rng)
    fitness = np.array([1.0] * 6 + [0.5] * 4)
    species = neat.SpeciesSet(threshold=3.0)
    for _ in range(config.NEAT_STAGNATION + 1):                     # constant fitness: no improvement after the first
        species.speciate(rng, genomes, fitness)
    near = next(s for s in species.species if 0 in s.members)
    far = next(s for s in species.species if 6 in s.members)
    assert near.stale >= config.NEAT_STAGNATION and far.stale >= config.NEAT_STAGNATION
    assert species.allot(10, fitness) == {near.id: 10, far.id: 0}   # near holds the best car (index 0)


def test_species_set_round_trips_through_its_state():
    rng = np.random.default_rng(34)
    species = neat.SpeciesSet()
    species.speciate(rng, clusters(rng), np.arange(10.0))
    copy = neat.SpeciesSet.from_state(pickle.loads(pickle.dumps(species.state())))
    assert copy.state()["threshold"] == species.threshold
    for a, b in zip(species.species, copy.species):
        assert (a.id, a.members, a.best_fitness, a.stale) == (b.id, b.members, b.best_fitness, b.stale)
        assert np.array_equal(a.representative.innov, b.representative.innov)
        assert np.array_equal(a.representative.weight, b.representative.weight)


def test_advance_keeps_survivors_first_and_fills_the_population():
    rng = np.random.default_rng(35)
    method = neat.NEAT(pop_size=30)
    method.initialize(rng)
    assert method.approach == "neat" and len(method.population) == 30 and not method.is_kept.any()
    assert method.step_size == config.NEAT_SIGMA0
    fitness = rng.random(30)
    kept = np.array([3, 7, 11])
    before = [method.population.genomes[i] for i in kept]
    method.advance(rng, result_for(fitness, fitness > 0.5), kept, None)
    assert len(method.population) == 30
    assert method.is_kept.tolist() == [True] * 3 + [False] * 27
    for k, genome in enumerate(before):
        assert np.array_equal(method.population.genomes[k].weight, genome.weight)
    assert method.log_fields()["species"] >= 1
    assert set(method.log_fields()) == {"species", "mean_nodes", "mean_conns"}


def test_child_success_rate_compares_children_with_their_parents():
    method = neat.NEAT(pop_size=4)
    method.ids = np.array([10, 11, 12, 13])
    method.parent_ids = np.array([-1, -1, 10, 99])                  # car 3's parent is gone
    method.is_kept = np.array([True, True, False, False])
    assert method.child_success_rate(np.array([1.0, 2.0, 3.0, 0.0])) == 1.0       # 3 > 1; car 3 left out
    assert method.child_success_rate(np.array([3.0, 2.0, 3.0, 0.0])) == 0.5       # a tie counts half
    method.parent_ids = np.array([-1, -1, 98, 99])
    assert np.isnan(method.child_success_rate(np.ones(4)))


def test_step_size_resets_on_level_up_only():
    rng = np.random.default_rng(36)
    method = neat.NEAT(pop_size=20)
    method.initialize(rng)
    method.steps.sigma = 0.01
    fitness = rng.random(20)
    method.advance(rng, result_for(fitness, np.zeros(20)), np.array([0]), "tightened")
    assert method.step_size != config.NEAT_SIGMA0
    method.advance(rng, result_for(fitness, np.zeros(20)), np.array([0]), "level_up")
    assert method.step_size == config.NEAT_SIGMA0


def records_after(event):
    """Species records after a generation that ends with `event`, and after the next generation, whose scores are
    lower (harder maps or a shorter target)."""
    rng = np.random.default_rng(38)
    method = neat.NEAT(pop_size=20)
    method.initialize(rng)
    fitness = np.random.default_rng(39).random(20) + 5.0
    method.advance(rng, result_for(fitness, np.zeros(20)), np.array([0]), None)
    for s in method.species.species:
        s.stale = config.NEAT_STAGNATION + 4                    # long-lived species with out-of-reach records
    method.advance(rng, result_for(fitness, np.zeros(20)), np.array([0]), event)
    after = [(s.best_fitness, s.stale) for s in method.species.species]
    method.advance(rng, result_for(fitness - 4.0, np.zeros(20)), np.array([0]), None)
    return after, [(s.best_fitness, s.stale) for s in method.species.species]


@pytest.mark.parametrize("event", ["level_up", "goal", "tightened"])
def test_stagnation_restarts_with_each_level_and_target(event):
    """Maps change every generation, so a species' best fitness is a record of noisy scores, and after a level-up,
    the goal or a tightened target the old records are out of reach (revision 3 §5.4)."""
    after, following = records_after(event)
    assert after and all(best == float("-inf") and stale == 0 for best, stale in after)
    assert all(best < 2.0 and stale == 0 for best, stale in following)     # the new level's scores are the records


def test_stagnation_is_kept_without_a_level_or_target_change():
    after, following = records_after(None)
    assert all(best > 5.0 and stale > config.NEAT_STAGNATION for best, stale in after)
    assert max(stale for _, stale in following) > config.NEAT_STAGNATION


def test_state_round_trip_continues_identically():
    rng_a = np.random.default_rng(37)
    a = neat.NEAT(pop_size=30)
    a.initialize(rng_a)
    scores = np.random.default_rng(99)
    first = scores.random(30)
    a.advance(rng_a, result_for(first, first > 0.4), np.array([0, 1, 2]), None)
    b = neat.NEAT(pop_size=30)
    b.load_state(pickle.loads(pickle.dumps(a.state())))
    rng_b = np.random.default_rng()
    rng_b.bit_generator.state = rng_a.bit_generator.state
    second = scores.random(30)
    for method, rng in ((a, rng_a), (b, rng_b)):
        method.advance(rng, result_for(second, second > 0.3), np.array([5, 2]), None)
    for ga, gb in zip(a.population.genomes, b.population.genomes):
        for name in neat.Genome.FIELDS:
            assert np.array_equal(getattr(ga, name), getattr(gb, name))
    assert a.innovations.state() == b.innovations.state()
    assert a.step_size == b.step_size and np.array_equal(a.ids, b.ids)


def fake_evaluation(pop, seeds, level, factor, n_jobs=1, backend="loky"):
    """Deterministic stand-in for the simulator: a genome's score follows its weights and biases; each map (seed)
    has its own success threshold."""
    score = np.array([np.tanh(g.weight[g.enabled].sum() / 10.0 + g.bias.sum()) for g in pop.genomes])
    thresholds = np.array([((s % 5) - 2) / 10.0 for s in seeds])
    success = score[None, :] > thresholds[:, None]
    fitness = np.where(success, 1.5 + 0.4 * score[None, :], 0.5 + 0.4 * score[None, :]).mean(axis=0)
    return population.TrialResult(fitness=fitness, success=success, laps=np.where(success, config.LAPS, 1),
                                  finish_time=np.where(success, 60.0, np.nan))


class Stop(Exception):
    pass


def test_run_neat_reports_and_resumes_exactly(monkeypatch, tmp_path):
    monkeypatch.setattr(population, "evaluate_population", fake_evaluation)
    monkeypatch.setattr(evolution, "evaluate_population", fake_evaluation, raising=False)

    def record(into, stop_at=None):
        def on_generation(report):
            into.append((report["generation"], report["best"], report["mean"], report["worst"],
                         report["step_size"], report["min_share"], report["species"]))
            if report["generation"] == stop_at:
                raise Stop()
        return on_generation

    full = []
    summary = neat.run_neat(seed=3, pop_size=40, out_dir=tmp_path / "full", champion_dir=None,
                            on_generation=record(full), max_generations=6)
    assert summary["approach"] == "neat" and summary["generations"] == 6 and summary["pop_size"] == 40
    assert [row[0] for row in full] == list(range(6))
    first = []
    with pytest.raises(Stop):
        neat.run_neat(seed=3, pop_size=40, out_dir=tmp_path / "cut", champion_dir=None,
                      on_generation=record(first, stop_at=3), max_generations=6)
    resumed = []
    neat.run_neat(seed=3, pop_size=40, out_dir=tmp_path / "cut", champion_dir=None,
                  on_generation=record(resumed), max_generations=6)
    assert first[:3] == full[:3]
    assert resumed == full[3:]                                  # generation 3 re-run from the checkpoint


@pytest.mark.slow
def test_run_neat_drives_easy_maps_and_saves_its_controller(tmp_path):
    summary = neat.run_neat(seed=0, pop_size=24, out_dir=tmp_path / "run", champion_dir=tmp_path / "champions",
                            max_generations=2)
    assert summary["generations"] == 2 and summary["level_reached"] == 1
    model, driver = nets.load_champion("neat", 0, tmp_path / "champions")
    assert isinstance(driver, neat.NeatNet)
    obs = np.random.default_rng(38).random((20, config.N_INPUTS)).astype(np.float32)
    assert np.mean(np.argmax(model.predict(obs, verbose=0), axis=1) == driver.act(obs)) >= 0.99
