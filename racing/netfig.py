"""Network figure (revision 3 §7.3): each method's best final controller drawn as a graph. The nine distance rays
and the speed input are on the left, hidden nodes in columns by depth, the 15 actions on the right. Connections are
blue when positive and red when negative, wider and more opaque when stronger; biases are not drawn."""
from pathlib import Path

import numpy as np

from racing import config

POSITIVE = "#2166AC"
NEGATIVE = "#B2182B"
STEER_NAMES = {-1.0: "R1", -0.5: "R½", 0.0: "S", 0.5: "L½", 1.0: "L1"}      # positive steering turns left
THROTTLE_NAMES = {-1.0: "brake", 0.0: "coast", 1.0: "gas"}


def input_labels():
    """The nine distance rays by angle (−90° points to the right, +90° to the left) and the speed input."""
    labels = []
    for angle in np.degrees(config.RAY_ANGLES):
        degrees = int(round(float(angle)))
        labels.append("0°" if degrees == 0 else f"{degrees:+d}°".replace("-", "−"))
    return labels + ["speed"]


def output_labels():
    """The 15 actions as steering and throttle, in action-class order."""
    n_throttle = len(config.THROTTLE_LEVELS)
    return [f"{STEER_NAMES[float(config.STEER_LEVELS[k // n_throttle])]} "
            f"{THROTTLE_NAMES[float(config.THROTTLE_LEVELS[k % n_throttle])]}" for k in range(config.N_ACTIONS)]


def _neat_graph(genome):
    """Columns and edges of a NEAT genome: inputs, hidden nodes grouped by depth, outputs; enabled connections only.
    A hidden node that no input reaches is drawn at depth 1."""
    from racing import neat
    depth = neat.node_depths(genome)
    hidden = [int(n) for n in genome.node_ids if int(n) >= neat.FIRST_HIDDEN]
    hidden_depth = {n: max(int(depth.get(n, 1)), 1) for n in hidden}
    column_of = {n: 0 for n in neat.INPUT_IDS}
    row_of = {n: k for k, n in enumerate(neat.INPUT_IDS)}
    columns = [input_labels()]
    for level in sorted(set(hidden_depth.values())):
        members = sorted(n for n in hidden if hidden_depth[n] == level)
        for k, n in enumerate(members):
            column_of[n], row_of[n] = len(columns), k
        columns.append([""] * len(members))
    for k, n in enumerate(neat.OUTPUT_IDS):
        column_of[n], row_of[n] = len(columns), k
    columns.append(output_labels())
    edges = [(column_of[int(s)], row_of[int(s)], column_of[int(d)], row_of[int(d)], float(w))
             for s, d, w, on in zip(genome.src, genome.dst, genome.weight, genome.enabled) if on]
    return {"columns": columns, "edges": edges}


def network_graph(driver):
    """Columns of node labels (inputs first, actions last; hidden nodes unlabelled) and edges
    (column a, row i, column b, row j, weight) of a dense or NEAT controller."""
    from racing import neat
    if isinstance(driver, neat.NeatNet):
        return _neat_graph(driver.genome)
    columns = [input_labels()] + [[""] * w.shape[1] for w, _ in driver.layers[:-1]] + [output_labels()]
    edges = []
    for k, (weights, _) in enumerate(driver.layers):
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                edges.append((k, i, k + 1, j, float(weights[i, j])))
    return {"columns": columns, "edges": edges}


def node_positions(columns):
    """(x, y) of every node: columns evenly spaced from x = 0 to 1, nodes evenly spaced within their column."""
    last = max(len(columns) - 1, 1)
    return [[(c / last, 1.0 - (k + 0.5) / len(labels)) for k in range(len(labels))]
            for c, labels in enumerate(columns)]


def _draw_graph(ax, graph):
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    columns, edges = graph["columns"], graph["edges"]
    xy = node_positions(columns)
    if edges:
        weights = np.array([e[4] for e in edges])
        scale = np.abs(weights) / max(float(np.abs(weights).max()), 1e-12)     # per panel
        order = np.argsort(scale, kind="stable")                              # strongest drawn last, on top
        segments = [(xy[edges[k][0]][edges[k][1]], xy[edges[k][2]][edges[k][3]]) for k in order]
        colors = [(*mcolors.to_rgb(POSITIVE if weights[k] >= 0 else NEGATIVE), 0.12 + 0.78 * scale[k]) for k in order]
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.2 + 2.0 * scale[order]))
    for column in xy:
        xs, ys = zip(*column)
        ax.scatter(xs, ys, s=10 if len(column) <= 20 else 4, color="0.25", zorder=3)
    for (x, y), label in zip(xy[0], columns[0]):
        ax.text(x - 0.03, y, label, ha="right", va="center", fontsize=6, color="0.15")
    for (x, y), label in zip(xy[-1], columns[-1]):
        ax.text(x + 0.03, y, label, ha="left", va="center", fontsize=6, color="0.15")
    ax.set_xlim(-0.25, 1.3)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def setup_text(driver):
    """' · <sensor length> m · ×<steering sensitivity>' of a driver's setup (revision 3.1 §5); the revision 3 setup
    for a driver without one."""
    length, gain = np.asarray(getattr(driver, "setup", config.DEFAULT_SETUP), dtype=float)
    return f" · {length:.0f} m · ×{gain:.2f}"


def panel_title(approach, driver):
    """A network diagram's title: the method, its parameter count and its setup."""
    return f"{config.LABELS[approach]} · {driver.n_params:,} parameters{setup_text(driver)}"


def draw_networks(drivers, path):
    """One panel per method in config.APPROACHES order, titled with the method, its parameter count and its setup,
    written as a PNG at `path`. drivers: {approach: driver}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    approaches = [a for a in config.APPROACHES if a in drivers]
    if not approaches:
        raise ValueError("no controllers to draw")
    ncols = min(4, len(approaches))
    nrows = -(-len(approaches) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.8 * nrows), squeeze=False)
    for ax in axes.flat[len(approaches):]:
        ax.set_visible(False)
    for ax, approach in zip(axes.flat, approaches):
        _draw_graph(ax, network_graph(drivers[approach]))
        ax.set_title(panel_title(approach, drivers[approach]), fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def render_graph(graph, title, size):
    """One network diagram as an RGB uint8 array of shape (height, width, 3) for size = (width, height) pixels, drawn
    off screen with the same style as draw_networks (no files, no pyplot, no global backend)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    width, height = max(int(size[0]), 1), max(int(size[1]), 1)
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.91])
    _draw_graph(ax, graph)
    ax.set_title(title, fontsize=10)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()
