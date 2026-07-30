"""Ground-truth edge labeling: whether a candidate edge in a
`~graphs.edm.TrackGraph` actually connects two hits produced by the *same*
simulated particle -- the label a GNN-style edge classifier trains and is
scored against.

Kept fully separate from `graphs.build` on purpose: building the candidate
graph never looks at `particle_id` (a real detector wouldn't have it
either), so truth-labeling is an explicit, optional second step, only
possible here because this is simulated data with known ground truth. An
edge's `is_true_edge` is ``True`` iff its two hits share a `particle_id` --
this is the standard "same-track" truth definition, and it does *not* also
require the two hits to be adjacent along that track: a same-particle edge
that skips a layer (e.g. from a `~graphs.prescription.ConnectionRules`
`delta_layer_id` range wider than 1) is still labeled true.
"""

from __future__ import annotations

import pandas as pd

from .edm import TrackGraph


def label_edges(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``edges`` with an added `is_true_edge` bool column:
    ``True`` iff the edge's two hits (`src_hit_id`/`dst_hit_id`, looked up
    in ``nodes``, the hits table) share a `particle_id`."""
    particle_id_of = nodes.set_index("hit_id")["particle_id"]
    src_particle = edges["src_hit_id"].map(particle_id_of)
    dst_particle = edges["dst_hit_id"].map(particle_id_of)

    labeled = edges.copy()
    labeled["is_true_edge"] = (src_particle == dst_particle).to_numpy()
    return labeled


def label_graph(graph: TrackGraph) -> TrackGraph:
    """`label_edges` applied to a whole `TrackGraph`: returns a new
    `TrackGraph` with the same `nodes` and truth-labeled `edges`."""
    return TrackGraph(nodes=graph.nodes, edges=label_edges(graph.nodes, graph.edges))


def purity(edges: pd.DataFrame) -> float:
    """Fraction of ``edges`` that are true (`is_true_edge`) -- one standard
    quality metric for a candidate graph (the complementary one, efficiency
    against all truly-connected hit pairs, isn't computed here since it
    depends on what "should" have been connected, i.e. the prescription's
    own intent, not just the labels). Raises if ``edges`` hasn't been
    labeled yet (see `label_edges`); returns ``nan`` for an empty graph."""
    if "is_true_edge" not in edges.columns:
        raise ValueError("edges has no 'is_true_edge' column -- label it first, see label_edges/label_graph")
    if len(edges) == 0:
        return float("nan")
    return float(edges["is_true_edge"].mean())
