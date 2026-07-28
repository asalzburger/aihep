"""Event data model (EDM): the table schemas this package reads and writes.

Mirrors `sensor.edm` exactly -- `splitting` operates on a `sensor` run
directory (hits/clusters/truth/contributions) and writes another one back
out in the same shape, so downstream tools (`sensor.vis`, `sensor.analysis`,
`sensor.cli visualize`/`analyse`) work unchanged on a split run. Kept as its
own copy rather than an import so this package stays independent (same
convention as `tracker.edm` vs. `sensor.edm`).
"""

from __future__ import annotations

HITS_COLUMNS = ["event_id", "ix", "iy", "x_center_um", "y_center_um", "charge", "cluster_id"]
"""A hit pixel, already assigned to a cluster (splitting only ever
reassigns `cluster_id`, never adds/drops/moves a pixel)."""

CLUSTERS_COLUMNS = [
    "event_id",
    "cluster_id",
    "n_pixels",
    "charge_sum",
    "x_centroid_um",
    "y_centroid_um",
    "x_centroid_digital_um",
    "y_centroid_digital_um",
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

CONTRIBUTIONS_COLUMNS = ["event_id", "particle_id", "ix", "iy", "charge"]
"""Ground truth: unaffected by splitting, passed through unchanged."""
