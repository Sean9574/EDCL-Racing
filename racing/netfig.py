"""Network figure (spec §11.4): each method's best final controller drawn as a graph. The 18 inputs are on the
left, hidden nodes in columns by depth, the 45 actions on the right. Connections are blue when positive and red when
negative, wider and more opaque when stronger; biases are not drawn."""
from pathlib import Path

import numpy as np

from racing import config

POSITIVE = "#2166AC"
NEGATIVE = "#B2182B"
STEER_NAMES = {-1.0: "R1", -0.75: "R¾", -0.5: "R½", -0.25: "R¼", 0.0: "S", 0.25: "L¼", 0.5: "L½", 0.75: "L¾",
               1.0: "L1"}                                                    # positive steering turns left
THROTTLE_NAMES = {-1.0: "brake", -0.5: "½ brake", 0.0: "coast", 0.5: "½ gas", 1.0: "gas"}


def input_labels():
    """The 18 inputs: the nine distance rays by angle (−90° points to the right, +90° to the left), the speed, the
    position across the road, the road width, the distance from the racing line, the heading relative to it and its
    curvature at each LINE_AHEAD distance."""
    labels = []
    for angle in np.degrees(config.RAY_ANGLES):
        degrees = int(round(float(angle)))
        labels.append("0°" if degrees == 0 else f"{degrees:+d}°".replace("-", "−"))
    line = ["line offset", "line heading"] + [f"line +{d:g} m" for d in config.LINE_AHEAD]
    return labels + ["speed", "position", "width"] + line


def output_labels():
    """The 45 actions as steering and throttle, in action-class order."""
    n_throttle = len(config.THROTTLE_LEVELS)
    return [f"{STEER_NAMES[float(config.STEER_LEVELS[k // n_throttle])]} "
            f"{THROTTLE_NAMES[float(config.THROTTLE_LEVELS[k % n_throttle])]}" for k in range(config.N_ACTIONS)]


def _neat_node_columns(genome):
    """The node ids of each column of a NEAT genome's diagram: the inputs, the hidden nodes grouped by depth (a hidden
    node that no input reaches at depth 1), the outputs."""
    from racing import neat
    depth = neat.node_depths(genome)
    hidden = [int(n) for n in genome.node_ids if int(n) >= neat.FIRST_HIDDEN]
    hidden_depth = {n: max(int(depth.get(n, 1)), 1) for n in hidden}
    levels = [sorted(n for n in hidden if hidden_depth[n] == level) for level in sorted(set(hidden_depth.values()))]
    return [list(neat.INPUT_IDS)] + levels + [list(neat.OUTPUT_IDS)]


def _neat_graph(genome):
    """Columns and edges of a NEAT genome: inputs, hidden nodes grouped by depth, outputs; enabled connections only.
    A hidden node that no input reaches is drawn at depth 1."""
    node_columns = _neat_node_columns(genome)
    column_of, row_of = {}, {}
    for c, members in enumerate(node_columns):
        for k, n in enumerate(members):
            column_of[n], row_of[n] = c, k
    columns = [input_labels()] + [[""] * len(members) for members in node_columns[1:-1]] + [output_labels()]
    edges = [(column_of[int(s)], row_of[int(s)], column_of[int(d)], row_of[int(d)], float(w))
             for s, d, w, on in zip(genome.src, genome.dst, genome.weight, genome.enabled) if on]
    return {"columns": columns, "edges": edges}


def network_graph(driver):
    """Columns of node labels (inputs first, actions last; hidden nodes unlabelled) and edges
    (column a, row i, column b, row j, weight) of a dense or NEAT controller."""
    from racing import neat
    if isinstance(driver, neat.NeatNet):
        return _neat_graph(driver.genome)
    columns = ([input_labels()[:driver.layers[0][0].shape[0]]]          # an earlier revision's: fewer inputs
               + [[""] * w.shape[1] for w, _ in driver.layers[:-1]] + [output_labels()])
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


def _label_size(column):
    """Label size in points: smaller for a column of 45 actions than for the 18 inputs."""
    return 6 if len(column) <= 20 else 4.5


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
        ax.text(x - 0.03, y, label, ha="right", va="center", fontsize=_label_size(columns[0]), color="0.15")
    for (x, y), label in zip(xy[-1], columns[-1]):
        ax.text(x + 0.03, y, label, ha="left", va="center", fontsize=_label_size(columns[-1]), color="0.15")
    ax.set_xlim(-0.25, 1.3)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def setup_text(driver):
    """' · <sensor length> m · ×<steering sensitivity>' of a driver's setup (spec §11.4); the revision 3 setup
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 6.0 * nrows), squeeze=False)
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


# ----- live view (the viewer's network window) -----

INNER_EDGES = 34            # connections drawn into hidden nodes: the strongest signals at this moment
ACTION_EDGES = 8            # connections drawn into the chosen action
STRAIGHT = 0.02             # a racing-line curvature input below this reads "straight" (radius over 2.5 km)
INPUT_GROUPS = (("Distance rays", 0, config.N_RAYS), ("Car", config.N_RAYS, config.N_RAYS + 3),
                ("Racing line", config.N_RAYS + 3, config.N_INPUTS))
INK, MUTED, FAINT, RIM = "#1d2330", "#5f6673", "#9aa1ad", "#c9ced8"
LIVE_POSITIVE, LIVE_NEGATIVE, DEEP, PALE = "#2563a8", "#c2410c", "#1e3a6e", "#f3f5f8"
STEER_SHORT = ("R1", "R¾", "R½", "R¼", "S", "L¼", "L½", "L¾", "L1")


def node_values(driver, obs):
    """The value of every node of network_graph(driver), column by column, for one car's inputs `obs`: the inputs,
    each hidden node's activation and the 45 action scores; the car takes the action with the highest score."""
    from racing import neat
    obs = np.asarray(obs, dtype=float)
    if isinstance(driver, neat.NeatNet):
        values = driver.node_values(obs[None, :neat.N_IN])[0]
        return [[float(values[driver.column[n]]) for n in members] for members in _neat_node_columns(driver.genome)]
    from racing import nets
    h = obs[:driver.n_inputs]
    columns = [h.tolist()]
    for i, (w, b) in enumerate(driver.layers):
        h = h @ w + b
        if i < len(driver.layers) - 1:
            h = nets.activate(h, driver.activations[i])
        columns.append(np.asarray(h, dtype=float).tolist())
    return columns


def live_edges(graph, values):
    """The connections the live view draws, as (column a, row i, column b, row j, weight): the strongest signals
    (the source node's value × the weight) into hidden nodes that are active, at most INNER_EDGES, then those into
    the chosen action, at most ACTION_EDGES."""
    last = len(graph["columns"]) - 1
    chosen = int(np.argmax(values[-1]))

    def signal(edge):
        return abs(values[edge[0]][edge[1]] * edge[4])

    inner = [e for e in graph["edges"] if e[2] != last and values[e[2]][e[3]] != 0.0 and signal(e) > 0.0]
    into = [e for e in graph["edges"] if e[2] == last and e[3] == chosen and signal(e) > 0.0]
    return (sorted(inner, key=signal, reverse=True)[:INNER_EDGES]
            + sorted(into, key=signal, reverse=True)[:ACTION_EDGES])


def drawn_edges(graph, values, full=True):
    """(faint, bold) connections of the live view: in the full view every connection, faintly; in both views the
    strongest signals at this moment, bold (live_edges; none before the car drives)."""
    bold = [] if values is None else live_edges(graph, values)
    return (list(graph["edges"]) if full else []), bold


def input_text(index, obs):
    """Input `index` of a car's inputs `obs` in the units a driver thinks in (spec §5.2): metres, km/h, the share of
    the half road from its centre, the distance and angle from the racing line, the line's corner radius ahead."""
    v = float(obs[index])
    side = "right" if v >= 0 else "left"
    k = index - config.N_RAYS
    if k < 0:
        return f"{v * config.RAY_SCALE:.0f} m"
    if k == 0:
        return f"{v * config.SPEED_SCALE * 3.6:.0f} km/h"
    if k == 1:
        return f"{abs(v) * 100:.0f}% {side}"
    if k == 2:
        return f"{v * config.WIDTH_SCALE:.1f} m"
    if k == 3:
        return f"{abs(v) * float(obs[index - 1]) * config.WIDTH_SCALE / 2:.1f} m {side}"
    if k == 4:
        return f"{abs(v) * 180:.0f}° {side}"
    return "straight" if abs(v) < STRAIGHT else f"{side} r {config.LINE_CURVATURE_SCALE / abs(v):.0f} m"


def _mix(a, b, t):
    import matplotlib.colors as mcolors
    a, b = np.array(mcolors.to_rgb(a)), np.array(mcolors.to_rgb(b))
    return tuple(a + (b - a) * float(np.clip(t, 0.0, 1.0)))


def _curves(starts, ends, points=24):
    """S-shaped curves from each start (x, y) to its end, leaving and arriving horizontally, all at once: an array of
    shape (curves, points, 2)."""
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    t = np.linspace(0.0, 1.0, points)[None, :, None]
    dx = 0.45 * (ends[:, 0] - starts[:, 0])
    p1 = np.stack([starts[:, 0] + dx, starts[:, 1]], axis=1)[:, None, :]
    p2 = np.stack([ends[:, 0] - dx, ends[:, 1]], axis=1)[:, None, :]
    return ((1 - t) ** 3 * starts[:, None, :] + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2
            + t ** 3 * ends[:, None, :])


def _neurons(n):
    return f"{n} neuron" if n == 1 else f"{n} neurons"


def hidden_headings(columns):
    """(title, subtitle) over each hidden column of the live view: 'layer k' and its number of neurons."""
    return [(f"layer {k}", _neurons(len(column))) for k, column in enumerate(columns[1:-1], start=1)]


def shape_text(columns):
    """The network's shape in words, e.g. '18 inputs · 2 hidden layers of 64 and 32 neurons · 45 actions'."""
    sizes = [len(column) for column in columns[1:-1]]
    if not sizes:
        hidden = "no hidden layer"
    elif len(sizes) == 1:
        hidden = f"1 hidden layer of {_neurons(sizes[0])}"
    else:
        hidden = f"{len(sizes)} hidden layers of {', '.join(map(str, sizes[:-1]))} and {sizes[-1]} neurons"
    return f"{len(columns[0])} inputs · {hidden} · {len(columns[-1])} actions"


def _live_layout(graph, size):
    """(x, y) in pixels of every node of the live view: inputs in their three groups on the left, hidden columns
    evenly between, the 45 actions on the right in nine steering groups (gas at the top of each)."""
    width, height = size
    s = height / 640.0
    columns = graph["columns"]
    n_in, last = len(columns[0]), len(columns) - 1
    top, bottom = 104 * s, height - 34 * s
    ix, ox = 0.30 * width, 0.78 * width
    groups = [(start, min(stop, n_in)) for _, start, stop in INPUT_GROUPS if start < n_in]
    step = (bottom - top) / max(n_in + 0.9 * (len(groups) - 1), 1)
    xy, y = [[None] * n_in], top + step / 2
    for g, (start, stop) in enumerate(groups):
        for i in range(start, stop):
            xy[0][i] = (ix, y)
            y += step
        y += 0.9 * step
    for c in range(1, last):
        n = len(columns[c])
        x = ix + (ox - ix) * c / last
        xy.append([(x, top + (bottom - top) * (k + 0.5) / n) for k in range(n)])
    n_throttle = len(config.THROTTLE_LEVELS)
    n_steer = len(columns[last]) // n_throttle
    step = (bottom - top) / (len(columns[last]) + 0.9 * (n_steer - 1))
    out, y = [None] * len(columns[last]), top + step / 2
    for col in range(n_steer):
        for t in reversed(range(n_throttle)):
            out[col * n_throttle + t] = (ox, y)
            y += step
        y += 0.9 * step
    return xy + [out]


def render_live(graph, values, title, where, size, full=True):
    """The network window's picture (viewer): the network's inputs with their live values in real units, its hidden
    nodes shaded by their activation (blue positive, orange negative), its 45 actions in nine steering groups shaded
    by score, and the connections that carry the strongest signals at this moment as bold curves (live_edges); the
    chosen action enlarged and named. The full view also draws every other connection faintly, tinted by the signal
    it carries (by its weight before the car drives); the simplified view (full False) leaves them out. With values
    None (before the car drives) the network alone. An RGB uint8 array of shape (height, width, 3)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    width, height = max(int(size[0]), 1), max(int(size[1]), 1)
    s = height / 640.0
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    columns = graph["columns"]
    last = len(columns) - 1
    xy = _live_layout(graph, size)
    pt = 72.0 / 100.0                                   # points per pixel

    def write(x, y, text, size_px, color=INK, ha="left", weight="normal"):
        ax.text(x, y, text, fontsize=size_px * s * pt, color=color, ha=ha, va="center", fontweight=weight)

    def nodes(points, radius, faces, edges, linewidth=1.0):
        xs, ys = zip(*points)
        ax.scatter(xs, ys, s=[(2 * r * s * pt) ** 2 for r in radius], c=faces, edgecolors=edges,
                   linewidths=linewidth * s, zorder=3)

    chosen = None if values is None else int(np.argmax(values[-1]))
    write(24 * s, 28 * s, title, 15, INK, weight="bold")
    write(24 * s, 50 * s, shape_text(columns), 11, MUTED)
    if values is None:
        write(width - 24 * s, 28 * s, "waiting for the car to drive", 13, MUTED, "right")
    else:
        write(width - 24 * s, 28 * s, output_labels()[chosen], 15, INK, "right", "bold")
        speed = input_text(config.N_RAYS, values[0]) if len(values[0]) > config.N_RAYS else ""
        write(width - 24 * s, 50 * s, " · ".join(part for part in (where, speed) if part), 11, MUTED, "right")
    # connections: every one faintly in the full view, the strongest signals bold on top
    import matplotlib.colors as mcolors
    positive, negative = np.array(mcolors.to_rgb(LIVE_POSITIVE)), np.array(mcolors.to_rgb(LIVE_NEGATIVE))
    faint, edges = drawn_edges(graph, values, full)
    for chosen_edges, points, end_gap, alpha, width_px, zorder in ((faint, 8, (8, 8), (0.05, 0.22), (0.35, 0.9), 1),
                                                                  (edges, 24, (10, 12), (0.15, 0.65), (0.5, 3.6), 2)):
        if not chosen_edges:
            continue
        signal = np.array([w if values is None else values[a][i] * w for a, i, _, _, w in chosen_edges])
        t = np.abs(signal) / max(float(np.abs(signal).max()), 1e-12)
        starts = np.array([(xy[a][i][0] + 8 * s, xy[a][i][1]) for a, i, _, _, _ in chosen_edges])
        ends = np.array([(xy[b][j][0] - end_gap[b == last] * s, xy[b][j][1]) for _, _, b, j, _ in chosen_edges])
        colors = np.column_stack([np.where((signal >= 0)[:, None], positive, negative), alpha[0] + alpha[1] * t])
        ax.add_collection(LineCollection(_curves(starts, ends, points), colors=colors,
                                         linewidths=(width_px[0] + width_px[1] * t) * s * pt, capstyle="round",
                                         zorder=zorder))
    # inputs
    for name, start, stop in INPUT_GROUPS:
        if start < len(columns[0]):
            write(24 * s, xy[0][start][1] - 16 * s, name, 10.5, FAINT, weight="bold")
    faces = []
    for i, (x, y) in enumerate(xy[0]):
        write(x - 16 * s, y, columns[0][i], 11, INK, "right")
        if values is None:
            faces.append(PALE)
            continue
        group = next((g for g in INPUT_GROUPS if g[1] <= i < g[2]), INPUT_GROUPS[-1])
        scale = max(float(np.max(np.abs(values[0][group[1]:min(group[2], len(values[0]))]))), 1e-9)
        v = values[0][i]
        faces.append(_mix(PALE, LIVE_POSITIVE if v >= 0 else LIVE_NEGATIVE, (abs(v) / scale) ** 0.7))
        write(x - 104 * s, y, input_text(i, values[0]), 10, MUTED, "right")
    nodes(xy[0], [6.5] * len(xy[0]), faces, RIM, 0.8)
    # hidden layers
    for c in range(1, last):
        points = xy[c]
        gap = (points[1][1] - points[0][1]) if len(points) > 1 else 40 * s
        radius = min(10.0, 0.42 * gap / s)
        level = list(values[c]) if values is not None else [0.0] * len(points)
        scale = max(max(abs(v) for v in level), 1e-9)
        faces = [_mix(PALE, DEEP if v >= 0 else LIVE_NEGATIVE, (abs(v) / scale) ** 0.6) for v in level]
        rims = [RIM if v == 0 else DEEP if v > 0 else LIVE_NEGATIVE for v in level]
        nodes(points, [radius] * len(points), faces, rims, 1.1)
        x = points[0][0]
        heading, neurons = hidden_headings(columns)[c - 1]
        write(x, 70 * s, heading, 11, MUTED, "center", "bold")
        write(x, 84 * s, neurons, 10, MUTED, "center")
        if values is not None:
            off = sum(v == 0 for v in level)              # a ReLU unit that is off; tanh units are never exactly 0
            write(x, 97 * s, f"{len(level) - off} of {len(level)} firing" if off else
                  f"{sum(v > 0 for v in level)} + · {sum(v < 0 for v in level)} −", 10, FAINT, "center")
    # actions
    out = xy[last]
    n_throttle = len(config.THROTTLE_LEVELS)
    write(out[0][0], 70 * s, f"{len(out)} actions", 11, MUTED, "center", "bold")
    write(out[0][0], 84 * s, "gas at top", 10, FAINT, "center")
    for col in range(len(out) // n_throttle):
        ys = [out[col * n_throttle + t][1] for t in range(n_throttle)]
        write(out[0][0] + 14 * s, (min(ys) + max(ys)) / 2, STEER_SHORT[col] if col < len(STEER_SHORT) else "", 9.5,
              FAINT)
    if values is None:
        nodes(out, [3.4] * len(out), [PALE] * len(out), RIM, 0.7)
    else:
        scores = np.asarray(values[-1], dtype=float)
        score = (scores - scores.min()) / max(float(scores.max() - scores.min()), 1e-12)
        faces = [DEEP if k == chosen else _mix(PALE, DEEP, (score[k] ** 2) ** 0.6) for k in range(len(out))]
        nodes(out, [6.0 if k == chosen else 3.4 for k in range(len(out))], faces,
              [DEEP if k == chosen else "#d5d9e0" for k in range(len(out))], 0.7)
        x, y = out[chosen]
        ax.plot([x + 8 * s, x + 44 * s], [y, y], color=DEEP, linewidth=1.0 * s, zorder=2)
        write(x + 50 * s, y - 2 * s, output_labels()[chosen], 13, INK, weight="bold")
        write(x + 50 * s, y + 14 * s, "chosen", 9.5, FAINT)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()
