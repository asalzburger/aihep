"""Event data model: hits (as digitized onto layer cells) and clusters.

Builds on `detectorsim2d.edm.HITS_COLUMNS` (event_id, particle_id, layer_id,
hit_id, x, y, s_local, path_length) by adding `cell_index`; clusters group
adjacent cells on the same layer, within the same event.
"""

from __future__ import annotations

HITS_COLUMNS = [
    "event_id",
    "particle_id",
    "layer_id",
    "hit_id",
    "x",
    "y",
    "s_local",
    "path_length",
    "cell_index",
]

CLUSTERED_HITS_COLUMNS = HITS_COLUMNS + ["cluster_id"]
"""HITS_COLUMNS after cluster_hits assigns each digitized hit a cluster_id
(-1 for hits on a layer with no pitch, which can't be digitized/clustered)."""

CLUSTERS_COLUMNS = [
    "event_id",
    "layer_id",
    "cluster_id",
    "n_cells",
    "n_hits",
    "s_centroid",
    "x_centroid",
    "y_centroid",
]
