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

EDGE_COLOR = "#333333"


def plot_edges_on(
    ax: Axes,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    color: str = EDGE_COLOR,
    linewidth: float = 0.5,
    alpha: float = 0.5,
    zorder: float = 1.2,
) -> Axes:
    """Draw ``edges`` as line segments between their two hits' `(x, y)`
    positions on an existing ``ax`` (e.g. from `tracksim2d.vis.plot_event`).
    ``zorder`` defaults just above detector layers (``zorder=1``) and below
    trajectories/hits (``zorder=2``/``3``), so edges read as connective
    tissue between hits without obscuring either.
    """
    xy = nodes.set_index("hit_id")[["x", "y"]]
    for _, edge in edges.iterrows():
        x1, y1 = xy.loc[edge["src_hit_id"]]
        x2, y2 = xy.loc[edge["dst_hit_id"]]
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
) -> Figure:
    """`tracksim2d.vis.plot_event`, with this event's candidate-graph edges
    drawn underneath the trajectories and hits. ``edges`` is a full graph's
    edges table (e.g. `~graphs.build.build_graph(hits, prescription).edges`)
    -- it's filtered down to this ``event_id`` here, the same as
    `plot_event` already does for ``particles``/``hits``."""
    fig = plot_event(particles, hits, layers, event_id, track_length=track_length, tracker_boundary=tracker_boundary)
    ax = fig.axes[0]
    event_hits = hits[hits["event_id"] == event_id]
    event_edges = edges[edges["event_id"] == event_id]
    plot_edges_on(ax, event_hits, event_edges, color=edge_color, linewidth=edge_linewidth, alpha=edge_alpha)
    return fig
