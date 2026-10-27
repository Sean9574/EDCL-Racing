import numpy as np
import pytest

from racing import config, neat, netfig, nets


def dense_driver():
    return nets.from_flat(nets.init_population(np.random.default_rng(0), 1)[0])


def neat_driver():
    """Input 0 -> a hidden node -> the first output, input 1 -> the second output, and a disabled input 2 -> the
    third output."""
    hidden, out = neat.FIRST_HIDDEN, list(neat.OUTPUT_IDS)
    genome = neat.Genome(node_ids=np.array([*neat.OUTPUT_IDS, hidden], dtype=np.int64), bias=np.zeros(neat.N_OUT + 1),
                         innov=np.array([0, 1, 2, 3], dtype=np.int64), src=np.array([0, hidden, 1, 2], dtype=np.int64),
                         dst=np.array([hidden, out[0], out[1], out[2]], dtype=np.int64),
                         weight=np.array([1.0, -2.0, 0.5, 0.3]), enabled=np.array([True, True, True, False]))
    return neat.NeatNet(genome)


def test_labels_name_the_rays_speed_road_and_racing_line():
    inputs = netfig.input_labels()
    assert inputs[9:12] == ["speed", "position", "width"] and len(inputs) == config.N_INPUTS
    assert inputs[-6:] == ["line offset", "line heading", "line +25 m", "line +50 m", "line +100 m", "line +200 m"]
    assert len(inputs) == config.N_INPUTS and inputs[0] == "−90°" and inputs[4] == "0°" and inputs[8] == "+90°"
    assert inputs[9] == "speed"


def test_output_labels_follow_the_class_layout():
    outputs = netfig.output_labels()
    assert len(outputs) == 45
    assert outputs[0] == "R1 brake" and outputs[4] == "R1 gas" and outputs[22] == "S coast"
    assert outputs[34] == "L½ gas" and outputs[36] == "L¾ ½ brake" and outputs[44] == "L1 gas"


def test_dense_graph_matches_the_network():
    driver = dense_driver()
    graph = netfig.network_graph(driver)
    assert [len(c) for c in graph["columns"]] == [config.N_INPUTS, 16, config.N_ACTIONS]
    assert len(graph["edges"]) == config.N_INPUTS * 16 + 16 * config.N_ACTIONS
    weights = {(a, i, b, j): w for a, i, b, j, w in graph["edges"]}
    assert weights[(0, 3, 1, 7)] == driver.layers[0][0][3, 7] and weights[(1, 5, 2, 14)] == driver.layers[1][0][5, 14]


def test_neat_graph_places_hidden_nodes_by_depth_and_skips_disabled_connections():
    graph = netfig.network_graph(neat_driver())
    assert [len(c) for c in graph["columns"]] == [config.N_INPUTS, 1, config.N_ACTIONS]
    assert sorted(graph["edges"]) == sorted([(0, 0, 1, 0, 1.0), (1, 0, 2, 0, -2.0), (0, 1, 2, 1, 0.5)])


def test_node_positions_span_the_panel():
    xy = netfig.node_positions([["a", "b"], ["c"], ["d", "e", "f"]])
    assert [x for x, _ in xy[0]] == [0.0, 0.0] and xy[2][0][0] == 1.0 and xy[1][0] == (0.5, 0.5)
    assert all(0.0 < y < 1.0 for column in xy for _, y in column)


def test_draw_networks_writes_a_png(tmp_path):
    path = netfig.draw_networks({"neat": neat_driver(), "ga": dense_driver()}, tmp_path / "figures" / "networks.png")
    assert path == tmp_path / "figures" / "networks.png" and path.read_bytes()[:4] == b"\x89PNG"


def test_draw_networks_without_controllers_raises(tmp_path):
    with pytest.raises(ValueError):
        netfig.draw_networks({}, tmp_path / "networks.png")


@pytest.mark.parametrize("make", [dense_driver, neat_driver])
def test_render_graph_returns_an_rgb_image_of_the_requested_size(make):
    image = netfig.render_graph(netfig.network_graph(make()), "title", (640, 480))
    assert image.shape == (480, 640, 3) and image.dtype == np.uint8
    assert len(np.unique(image.reshape(-1, 3), axis=0)) > 10                           # not blank
    rgb = image.astype(int)
    red, blue = rgb[..., 0] - rgb[..., 2] > 40, rgb[..., 2] - rgb[..., 0] > 40       # faint lines are tinted
    assert red.any() and blue.any()                                                    # both weight signs


def test_render_graph_follows_the_size():
    graph = netfig.network_graph(neat_driver())
    assert netfig.render_graph(graph, "t", (300, 200)).shape == (200, 300, 3)
    assert netfig.render_graph(graph, "t", (1, 1)).shape[2] == 3


def test_titles_name_the_method_parameters_and_setup():
    driver = dense_driver()
    assert netfig.setup_text(driver) == " · 150 m · ×1.00"
    driver.setup = np.array([123.4, 0.456])
    assert netfig.setup_text(driver) == " · 123 m · ×0.46"
    assert netfig.panel_title("ga", driver) == f"GA · {driver.n_params:,} parameters · 123 m · ×0.46"
    assert netfig.panel_title("neat", neat_driver()).endswith(" parameters · 150 m · ×1.00")


def test_a_network_of_an_earlier_revision_shows_only_its_inputs():
    driver = nets.from_flat(nets.init_population(np.random.default_rng(0), 1, n_inputs=10)[0])
    graph = netfig.network_graph(driver)
    assert graph["columns"][0] == netfig.input_labels()[:10] and graph["columns"][0][-1] == "speed"


def test_the_network_figure_fits_45_labels(tmp_path):
    driver = nets.from_flat(np.random.default_rng(0).normal(size=nets.n_params()))
    path = netfig.draw_networks({"baseline": driver}, tmp_path / "n.png")
    from PIL import Image                                   # matplotlib's dependency
    height = Image.open(path).size[1]
    assert height >= 6.0 * 200 * 0.95                      # 6 inches per row at 200 dpi


# ----- the live network window (viewer, version 3.15) -----

def shaped_driver(hidden):
    """A dense network with hidden layers `hidden`: () is a GA network without hidden layers."""
    rng = np.random.default_rng(len(hidden) + 5)
    return nets.from_flat(rng.normal(size=nets.n_params(hidden)), hidden)


def car_inputs(seed=4):
    return np.random.default_rng(seed).uniform(-1.0, 1.5, size=config.N_INPUTS)


@pytest.mark.parametrize("make", [dense_driver, neat_driver, lambda: shaped_driver(()),
                                  lambda: shaped_driver((8, 6))])
def test_node_values_end_in_the_action_the_car_takes(make):
    driver, obs = make(), car_inputs()
    graph, values = netfig.network_graph(driver), netfig.node_values(driver, obs)
    assert [len(v) for v in values] == [len(c) for c in graph["columns"]]
    assert np.allclose(values[0], obs[:len(values[0])])
    assert int(np.argmax(values[-1])) == int(driver.act(obs[None, :])[0])


def test_dense_hidden_values_are_the_relu_activations():
    driver, obs = dense_driver(), car_inputs()
    (w1, b1), _ = driver.layers
    assert np.allclose(netfig.node_values(driver, obs)[1], np.maximum(obs @ w1 + b1, 0.0))


def test_neat_values_follow_the_genome():
    obs = car_inputs()
    values = netfig.node_values(neat_driver(), obs)
    hidden = np.tanh(1.0 * obs[0])
    assert np.isclose(values[1][0], hidden) and np.isclose(values[2][0], -2.0 * hidden)
    assert np.isclose(values[2][1], 0.5 * obs[1]) and values[2][2] == 0.0          # the disabled connection is off


def test_live_edges_carry_the_strongest_signals():
    driver, obs = dense_driver(), car_inputs()
    graph, values = netfig.network_graph(driver), netfig.node_values(driver, obs)
    chosen = int(np.argmax(values[-1]))
    edges = netfig.live_edges(graph, values)
    last = len(graph["columns"]) - 1
    into_actions = [e for e in edges if e[2] == last]
    inner = [e for e in edges if e[2] != last]
    assert 0 < len(into_actions) <= netfig.ACTION_EDGES and all(e[3] == chosen for e in into_actions)
    assert 0 < len(inner) <= netfig.INNER_EDGES and all(values[e[2]][e[3]] != 0.0 for e in inner)
    signal = lambda e: abs(values[e[0]][e[1]] * e[4])
    weakest = min(signal(e) for e in inner)
    left_out = [e for e in graph["edges"] if e[2] != last and e not in inner and values[e[2]][e[3]] != 0.0]
    assert all(signal(e) <= weakest for e in left_out)


def test_a_network_without_hidden_layers_shows_its_strongest_inputs_into_the_choice():
    driver, obs = shaped_driver(()), car_inputs()
    graph, values = netfig.network_graph(driver), netfig.node_values(driver, obs)
    edges = netfig.live_edges(graph, values)
    assert edges and all((e[0], e[2], e[3]) == (0, 1, int(np.argmax(values[-1]))) for e in edges)


def test_inputs_read_in_real_units():
    obs = np.zeros(config.N_INPUTS)
    obs[[4, 9, 10, 11, 12, 13, 16, 17]] = [0.45, 0.93, 0.43, 0.69, -0.32, -0.01, -0.82, -0.01]
    text = lambda i: netfig.input_text(i, obs)
    assert text(4) == "68 m" and text(9) == f"{0.93 * config.SPEED_SCALE * 3.6:.0f} km/h"
    assert text(10) == "43% right" and text(11) == "13.8 m"
    assert text(12) == "2.2 m left" and text(13) == "2° left"
    assert text(16) == "left r 61 m" and text(17) == "straight"


@pytest.mark.parametrize("make", [dense_driver, neat_driver, lambda: shaped_driver(()),
                                  lambda: shaped_driver((8, 6))])
@pytest.mark.parametrize("size", [(900, 640), (360, 260)])
def test_the_live_view_renders_at_the_window_size(make, size):
    driver = make()
    graph = netfig.network_graph(driver)
    for values in (netfig.node_values(driver, car_inputs()), None):           # None: before the car drives
        image = netfig.render_live(graph, values, "GA · run 0 · 855 parameters", "Silverstone", size)
        assert image.shape == (size[1], size[0], 3) and image.dtype == np.uint8 and image.std() > 0


def test_the_full_view_draws_every_connection_faintly_and_the_strongest_signals_on_top():
    for make in (dense_driver, neat_driver, lambda: shaped_driver(()), lambda: shaped_driver((8, 6))):
        driver = make()
        graph, values = netfig.network_graph(driver), netfig.node_values(driver, car_inputs())
        faint, bold = netfig.drawn_edges(graph, values, full=True)
        assert faint == graph["edges"] and bold == netfig.live_edges(graph, values)
        assert netfig.drawn_edges(graph, values, full=False) == ([], bold)
        assert netfig.drawn_edges(graph, None, full=True) == (graph["edges"], [])       # before the car drives
        assert netfig.drawn_edges(graph, None, full=False) == ([], [])


@pytest.mark.parametrize("full", [True, False])
def test_both_views_render_at_the_window_size(full):
    for make in (dense_driver, neat_driver, lambda: shaped_driver(()), lambda: shaped_driver((64, 32))):
        driver = make()
        graph = netfig.network_graph(driver)
        for values in (netfig.node_values(driver, car_inputs()), None):
            image = netfig.render_live(graph, values, "t", "Silverstone", (640, 480), full=full)
            assert image.shape == (480, 640, 3) and image.std() > 0


def test_the_headings_name_layers_and_neurons():
    columns = lambda *hidden: [netfig.input_labels()] + [[""] * n for n in hidden] + [netfig.output_labels()]
    assert netfig.shape_text(columns()) == "18 inputs · no hidden layer · 45 actions"
    assert netfig.shape_text(columns(16)) == "18 inputs · 1 hidden layer of 16 neurons · 45 actions"
    assert netfig.shape_text(columns(64, 32)) == "18 inputs · 2 hidden layers of 64 and 32 neurons · 45 actions"
    assert netfig.shape_text(columns(9, 1, 2)) == "18 inputs · 3 hidden layers of 9, 1 and 2 neurons · 45 actions"
    assert netfig.hidden_headings(columns(64, 1)) == [("layer 1", "64 neurons"), ("layer 2", "1 neuron")]
    assert netfig.hidden_headings(columns()) == []


def test_curves_are_built_together_and_join_their_ends():
    starts, ends = np.array([[0.0, 0.0], [5.0, 40.0]]), np.array([[100.0, 30.0], [80.0, 10.0]])
    curves = netfig._curves(starts, ends, points=9)
    assert curves.shape == (2, 9, 2)
    assert np.allclose(curves[:, 0], starts) and np.allclose(curves[:, -1], ends)
    assert np.all(np.diff(curves[:, :, 0], axis=1) > 0)                       # always forward, left to right
    assert np.allclose(curves[0, 4], [50.0, 15.0])                            # the middle is halfway
