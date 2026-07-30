import math

import matplotlib.pyplot as plt
import pandas as pd
from detector2d.geometry import LineLayer
from tracksim2d.edm import HITS_COLUMNS, PARTICLES_COLUMNS

from graphs.build import build_edges
from graphs.prescription import FullyConnected
from graphs.truth import label_edges
from graphs.vis import EDGE_COLOR, TRUE_EDGE_COLOR, plot_edges_on, plot_event_with_graph


def _hit(event_id, hit_id, x, y, layer_id, particle_id=0):
    return dict(
        event_id=event_id,
        particle_id=particle_id,
        layer_id=layer_id,
        hit_id=hit_id,
        x=x,
        y=y,
        s_local=0.0,
        path_length=math.hypot(x, y),
    )


def test_plot_edges_on_draws_one_line_per_edge():
    nodes = pd.DataFrame(
        [_hit(0, 0, 0.0, 0.0, 0), _hit(0, 1, 10.0, 0.0, 1), _hit(0, 2, 10.0, 10.0, 2)], columns=HITS_COLUMNS
    )
    edges = build_edges(nodes, FullyConnected())
    assert len(edges) == 3  # C(3, 2)

    fig, ax = plt.subplots()
    n_lines_before = len(ax.lines)
    plot_edges_on(ax, nodes, edges)
    assert len(ax.lines) == n_lines_before + len(edges)
    plt.close(fig)


def test_plot_edges_on_colors_true_and_false_edges_differently_once_labeled():
    nodes = pd.DataFrame(
        [
            _hit(0, 0, 0.0, 0.0, 0, particle_id=0),
            _hit(0, 1, 10.0, 0.0, 1, particle_id=0),  # true with hit 0
            _hit(0, 2, 10.0, 10.0, 2, particle_id=1),  # false with both
        ],
        columns=HITS_COLUMNS,
    )
    edges = label_edges(nodes, build_edges(nodes, FullyConnected()))

    fig, ax = plt.subplots()
    plot_edges_on(ax, nodes, edges)
    colors = {line.get_color() for line in ax.lines}
    assert TRUE_EDGE_COLOR in colors
    assert EDGE_COLOR in colors
    plt.close(fig)


def test_plot_edges_on_uses_plain_color_when_unlabeled():
    nodes = pd.DataFrame(
        [_hit(0, 0, 0.0, 0.0, 0), _hit(0, 1, 10.0, 0.0, 1)], columns=HITS_COLUMNS
    )
    edges = build_edges(nodes, FullyConnected())
    assert "is_true_edge" not in edges.columns

    fig, ax = plt.subplots()
    plot_edges_on(ax, nodes, edges)
    assert {line.get_color() for line in ax.lines} == {EDGE_COLOR}
    plt.close(fig)


def test_plot_event_with_graph_overlays_edges_on_the_simulation_plot():
    layers = [LineLayer(layer_id=0, p1=(10.0, -5.0), p2=(10.0, 5.0)), LineLayer(layer_id=1, p1=(20.0, -5.0), p2=(20.0, 5.0))]
    particles = pd.DataFrame(
        [dict(event_id=0, particle_id=0, x0=0.0, y0=0.0, phi0=0.0, charge=1.0, radius=math.nan)],
        columns=PARTICLES_COLUMNS,
    )
    hits = pd.DataFrame(
        [_hit(0, 0, 10.0, 0.0, 0, particle_id=0), _hit(0, 1, 20.0, 0.0, 1, particle_id=0)],
        columns=HITS_COLUMNS,
    )
    edges = build_edges(hits, FullyConnected())
    assert len(edges) == 1

    # plain plot_event draws: 2 dashed layer lines + 1 trajectory line = 3 lines, no edges
    from tracksim2d.vis import plot_event

    base_fig = plot_event(particles, hits, layers, event_id=0)
    base_lines = len(base_fig.axes[0].lines)
    plt.close(base_fig)

    fig = plot_event_with_graph(particles, hits, edges, layers, event_id=0)
    assert len(fig.axes[0].lines) == base_lines + len(edges)
    plt.close(fig)


def test_plot_event_with_graph_filters_edges_to_the_requested_event():
    layers = [LineLayer(layer_id=0, p1=(10.0, -5.0), p2=(10.0, 5.0))]
    particles = pd.DataFrame(
        [
            dict(event_id=0, particle_id=0, x0=0.0, y0=0.0, phi0=0.0, charge=1.0, radius=math.nan),
            dict(event_id=1, particle_id=0, x0=0.0, y0=0.0, phi0=0.0, charge=1.0, radius=math.nan),
        ],
        columns=PARTICLES_COLUMNS,
    )
    hits = pd.DataFrame(
        [
            _hit(0, 0, 10.0, 0.0, 0, particle_id=0),
            _hit(0, 1, 10.0, 1.0, 0, particle_id=0),
            _hit(1, 2, 10.0, 0.0, 0, particle_id=0),
            _hit(1, 3, 10.0, 1.0, 0, particle_id=0),
        ],
        columns=HITS_COLUMNS,
    )
    edges = build_edges(hits, FullyConnected())
    assert len(edges) == 2  # one per event

    fig = plot_event_with_graph(particles, hits, edges, layers, event_id=0)
    # 1 dashed layer + 1 trajectory + 1 vertex marker (event 0's only particle,
    # both drawn via ax.plot by plot_event) + 1 edge (event 0's only edge)
    assert len(fig.axes[0].lines) == 4
    plt.close(fig)
