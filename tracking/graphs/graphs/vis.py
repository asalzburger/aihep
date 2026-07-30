"""Graph visualization: draw a `~graphs.edm.TrackGraph`'s edges *on top of*
an existing `tracksim2d.vis.plot_event` figure, rather than duplicating its
layer/trajectory/hit drawing. A graph only adds one new kind of mark (edges
as line segments between two hits' `(x, y)`); everything else about the
event display already belongs to `tracksim2d`.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tracksim2d.vis import plot_event
from viz_style import Theme, palette

EDGE_COLOR = palette.EDGE
TRUE_EDGE_COLOR = palette.TRUE_EDGE


def plot_edges_on(
    ax: Axes,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    color: str = EDGE_COLOR,
    linewidth: float = 0.5,
    alpha: float = 0.5,
    zorder: float = 1.2,
    true_color: str = TRUE_EDGE_COLOR,
    true_linewidth: float = 1.0,
    true_alpha: float = 0.8,
) -> Axes:
    """Draw ``edges`` as line segments between their two hits' `(x, y)`
    positions on an existing ``ax`` (e.g. from `tracksim2d.vis.plot_event`).
    ``zorder`` defaults just above detector layers (``zorder=1``) and below
    trajectories/hits (``zorder=2``/``3``), so edges read as connective
    tissue between hits without obscuring either.

    If ``edges`` has been truth-labeled (see `graphs.truth.label_edges`,
    i.e. it has an `is_true_edge` column), true edges are drawn more boldly
    in ``true_color`` and everything else in the plain ``color`` -- a quick
    visual purity check. Without that column, every edge is drawn the same
    (the original, truth-agnostic behavior).
    """
    xy = nodes.set_index("hit_id")[["x", "y"]]
    has_truth = "is_true_edge" in edges.columns
    for _, edge in edges.iterrows():
        x1, y1 = xy.loc[edge["src_hit_id"]]
        x2, y2 = xy.loc[edge["dst_hit_id"]]
        if has_truth and edge["is_true_edge"]:
            ax.plot([x1, x2], [y1, y2], color=true_color, linewidth=true_linewidth, alpha=true_alpha, zorder=zorder)
        else:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    return ax


def plot_event_with_graph(
    particles: pd.DataFrame,
    hits: pd.DataFrame,
    edges: pd.DataFrame,
    layers,
    event_id: int,
    track_length: float = 100.0,
    tracker_boundary: float | None = None,
    edge_color: str = EDGE_COLOR,
    edge_linewidth: float = 0.5,
    edge_alpha: float = 0.5,
    true_edge_color: str = TRUE_EDGE_COLOR,
    true_edge_linewidth: float = 1.0,
    true_edge_alpha: float = 0.8,
    theme: Theme | None = None,
) -> Figure:
    """`tracksim2d.vis.plot_event`, with this event's candidate-graph edges
    drawn underneath the trajectories and hits. ``edges`` is a full graph's
    edges table (e.g. `~graphs.build.build_graph(hits, prescription).edges`,
    optionally truth-labeled via `graphs.truth.label_edges` -- see
    `plot_edges_on`) -- it's filtered down to this ``event_id`` here, the
    same as `plot_event` already does for ``particles``/``hits``."""
    fig = plot_event(
        particles, hits, layers, event_id,
        track_length=track_length, tracker_boundary=tracker_boundary, theme=theme,
    )
    ax = fig.axes[0]
    event_hits = hits[hits["event_id"] == event_id]
    event_edges = edges[edges["event_id"] == event_id]
    plot_edges_on(
        ax,
        event_hits,
        event_edges,
        color=edge_color,
        linewidth=edge_linewidth,
        alpha=edge_alpha,
        true_color=true_edge_color,
        true_linewidth=true_edge_linewidth,
        true_alpha=true_edge_alpha,
    )
    return fig
