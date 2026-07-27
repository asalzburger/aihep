"""Event data model (EDM): the transient per-event table schemas shared by
`sim` (producer), `io` (persistence), and `vis` (consumer).

Three columnar tables, joined by `event_id` (`hits`/`truth` also by
`cluster_id`/`particle_id` respectively):

- **hits** — one row per digitized pixel
- **clusters** — one row per connected-component cluster of hit pixels
- **truth** — one row per simulated particle (ground truth, not detector response)
"""

from __future__ import annotations

HITS_COLUMNS = ["event_id", "ix", "iy", "x_center_um", "y_center_um", "charge"]
"""A digitized pixel, before cluster_id is assigned (sim.digitize output)."""

CLUSTERED_HITS_COLUMNS = HITS_COLUMNS + ["cluster_id"]
"""hits schema after sim.clustering.cluster_hits assigns each pixel a cluster."""

CLUSTERS_COLUMNS = [
    "event_id",
    "cluster_id",
    "n_pixels",
    "charge_sum",
    "x_centroid_um",
    "y_centroid_um",
    "x_span_pixels",
    "y_span_pixels",
]

TRUTH_COLUMNS = [
    "event_id",
    "particle_id",
    "x0_um",
    "y0_um",
    "dxdz",
    "dydz",
    "charge_deposited",
    "path_length_um",
]
