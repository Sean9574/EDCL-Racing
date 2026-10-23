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


def test_labels_name_the_rays_and_speed():
    inputs = netfig.input_labels()
    assert inputs[-3:] == ["speed", "position", "width"] and len(inputs) == config.N_INPUTS
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
