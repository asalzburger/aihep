"""Reconstruction-quality analysis: match clusters to truth particles and
compute position residuals (reconstructed - true), for both the
charge-weighted and digital centroid definitions computed by
`sim.clustering.cluster_hits`.

Kept separate from `vis` (which only plots) so the matching/residual logic
is plain-data and independently testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sim.config import DetectorConfig
from .sim.geometry import true_center_position

CENTROID_COLUMNS = {
    "charge": ("x_centroid_um", "y_centroid_um"),
    "digital": ("x_centroid_digital_um", "y_centroid_digital_um"),
}

MATCHED_COLUMNS = [
    "event_id",
    "particle_id",
    "cluster_id",
    "true_x_um",
    "true_y_um",
    "recon_x_um",
    "recon_y_um",
]


def match_clusters_to_truth(
    clusters: pd.DataFrame, truth: pd.DataFrame, detector: DetectorConfig, type: str = "charge"
) -> pd.DataFrame:
    """For each truth particle, find the nearest cluster in the same event
    (by the requested centroid type's position). Truth particles whose
    event has no surviving clusters (e.g. the readout threshold cut
    everything away) are dropped.

    type: "charge" (charge-weighted centroid) or "digital" (unweighted).

    Returns one row per matched truth particle: event_id, particle_id,
    cluster_id, true_{x,y}_um, recon_{x,y}_um.
    """
    x_col, y_col = CENTROID_COLUMNS[type]
    clusters_by_event = dict(tuple(clusters.groupby("event_id")))

    rows: list[dict] = []
    for event_id, event_truth in truth.groupby("event_id", sort=True):
        event_clusters = clusters_by_event.get(event_id)
        if event_clusters is None or event_clusters.empty:
            continue
        recon_x = event_clusters[x_col].to_numpy()
        recon_y = event_clusters[y_col].to_numpy()
        cluster_ids = event_clusters["cluster_id"].to_numpy()

        for _, particle in event_truth.iterrows():
            true_x, true_y = true_center_position(
                particle["x0_um"], particle["y0_um"], particle["dxdz"], particle["dydz"], detector.thickness_um
            )
            nearest = int(np.argmin((recon_x - true_x) ** 2 + (recon_y - true_y) ** 2))
            rows.append(
                dict(
                    event_id=event_id,
                    particle_id=particle["particle_id"],
                    cluster_id=int(cluster_ids[nearest]),
                    true_x_um=true_x,
                    true_y_um=true_y,
                    recon_x_um=float(recon_x[nearest]),
                    recon_y_um=float(recon_y[nearest]),
                )
            )

    return pd.DataFrame(rows, columns=MATCHED_COLUMNS)


def compute_residuals(
    clusters: pd.DataFrame, truth: pd.DataFrame, detector: DetectorConfig, type: str = "charge"
) -> pd.DataFrame:
    """Residuals (reconstructed - true), in um, in x and y, for the given
    centroid type. See `match_clusters_to_truth` for matching semantics."""
    matched = match_clusters_to_truth(clusters, truth, detector, type=type)
    matched["residual_x_um"] = matched["recon_x_um"] - matched["true_x_um"]
    matched["residual_y_um"] = matched["recon_y_um"] - matched["true_y_um"]
    return matched
