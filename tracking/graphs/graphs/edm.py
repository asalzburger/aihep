"""Event data model for candidate track graphs built on top of a
`tracksim2d` hits table.

- **nodes** are exactly the hits table `graphs.build.build_graph` was given
  -- a graph's nodes *are* hits, unmodified (see `tracksim2d.edm.HITS_COLUMNS`).
  Nothing here re-derives or duplicates them.
- **edges** -- one row per candidate hit-to-hit connection, always evaluated
  within a single event (`event_id`) -- hits from different events never
  connect. `src_hit_id`/`dst_hit_id` are `hits.hit_id` values, which are
  globally unique across an entire hits table (see
  `tracksim2d.simulate.hits_for_particles`), so an edge can be resolved back
  to its two hit rows without also needing `event_id`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

EDGES_COLUMNS = [
    "event_id",
    "edge_id",
    "src_hit_id",
    "dst_hit_id",
    "delta_layer_id",
    "delta_r",
    "delta_phi",
    "delta_x",
    "delta_y",
    "distance",
]
"""``delta_*`` are all signed ``dst - src`` (``delta_phi`` wrapped into
``(-pi, pi]``, since phi is cyclic); ``distance`` is the euclidean distance
between the two hits' ``(x, y)``. Computed the same way regardless of which
`~graphs.prescription.Prescription` built the edge -- purely geometric edge
features, useful both for `~graphs.prescription.ConnectionRules` gating and
as GNN edge features downstream."""


@dataclass
class TrackGraph:
    """The graph itself: `nodes` is the hits table unchanged (one row per
    node) and `edges` is the table above. Kept as two plain DataFrames --
    joined by `event_id` and `hit_id`/`src_hit_id`/`dst_hit_id` -- rather
    than a graph-library object, so it round-trips through the same
    CSV/Arrow IO as the rest of this codebase (see `graphs.io`) and overlays
    directly onto `tracksim2d.vis.plot_event` (see `graphs.vis`)."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
