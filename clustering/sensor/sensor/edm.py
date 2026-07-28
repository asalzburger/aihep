"""Event data model (EDM): the transient per-event table schemas shared by
`sim` (producer), `io` (persistence), and `vis` (consumer).

Four columnar tables, joined by `event_id` (`hits`/`truth`/`contributions`
also by `cluster_id`/`particle_id`/`particle_id` respectively):

- **hits** — one row per digitized pixel
- **clusters** — one row per connected-component cluster of hit pixels
- **truth** — one row per simulated particle (ground truth, not detector response)
- **contributions** — one row per (particle, pixel) it deposited charge into
  (ground truth, pre-diffusion/noise/threshold) — the join key for tracing a
  hit/cluster back to the truth particle(s) that produced it, see
  `sensor.analysis.cluster_purity`.
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
    "x_centroid_digital_um",
    "y_centroid_digital_um",
    "x_span_pixels",
    "y_span_pixels",
]
"""x_centroid_um/y_centroid_um are charge-weighted; x_centroid_digital_um/
y_centroid_digital_um are the unweighted (digital, on/off) centroid of the
same pixels, for comparing the two reconstruction schemes."""

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
"""One row per (particle_id, pixel) the particle deposited charge into,
before diffusion/noise/threshold — i.e. the raw geometric deposit from
sim.simulate, prior to the effects sim.digitize applies to the *summed*
per-event grid. Diffusion is a linear operation (Gaussian blur of a sum
equals the sum of Gaussian blurs), so this is exact when
digitization.diffusion_sigma_um == 0 (the default) and a good approximation
otherwise; electronic noise is not attributable to any particle and is not
represented here."""
