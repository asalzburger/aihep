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

PURITY_COLUMNS = ["event_id", "cluster_id", "particle_id", "charge", "fraction"]


def cluster_purity(hits: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    """Per (event_id, cluster_id, particle_id): how much charge that
    particle actually deposited into that cluster's pixels, and what
    fraction of the cluster's total (summed) charge that represents.

    Joins on (event_id, ix, iy): `hits` carries the cluster_id a pixel ended
    up in, `contributions` carries each particle's raw deposit into that
    pixel (see edm.CONTRIBUTIONS_COLUMNS). A cluster with rows from more
    than one particle_id is a merged/overlapping cluster (common in
    shower-like multi-particle events, e.g. `multi.n_particles > 1`); a
    single row with fraction ~1.0 means a clean, unambiguous cluster.

    Rows below the readout threshold (dropped before clustering, so absent
    from `hits`) contribute no charge to any cluster and are silently
    excluded — same as they are from `hits` itself.
    """
    merged = hits[["event_id", "ix", "iy", "cluster_id"]].merge(
        contributions, on=["event_id", "ix", "iy"], how="inner"
    )
    per_particle = merged.groupby(["event_id", "cluster_id", "particle_id"], as_index=False)["charge"].sum()
    cluster_totals = per_particle.groupby(["event_id", "cluster_id"])["charge"].transform("sum")
    per_particle["fraction"] = per_particle["charge"] / cluster_totals
    return per_particle.sort_values(
        ["event_id", "cluster_id", "fraction"], ascending=[True, True, False]
    ).reset_index(drop=True)[PURITY_COLUMNS]


def dominant_particle_per_cluster(hits: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster: the particle_id contributing the most charge to
    it, and that fraction (1.0 for a cluster produced by a single particle,
    lower for a merged/overlapping one)."""
    purity = cluster_purity(hits, contributions)
    idx = purity.groupby(["event_id", "cluster_id"])["charge"].idxmax()
    return purity.loc[idx].reset_index(drop=True)


def dominant_cluster_per_particle(hits: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    """One row per truth particle that deposited charge into any surviving
    cluster: the cluster_id it contributed to the most, by charge. This is
    the exact truth-to-cluster link (as opposed to nearest-position
    matching), used by `match_clusters_to_truth` when `contributions` is
    supplied."""
    purity = cluster_purity(hits, contributions)
    idx = purity.groupby(["event_id", "particle_id"])["charge"].idxmax()
    return purity.loc[idx].reset_index(drop=True)


def match_clusters_to_truth(
    clusters: pd.DataFrame,
    truth: pd.DataFrame,
    detector: DetectorConfig,
    type: str = "charge",
    hits: pd.DataFrame | None = None,
    contributions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Match each truth particle to a cluster in the same event.

    When both `hits` and `contributions` are given, matching uses the
    *exact* charge-contribution link (each particle's dominant cluster by
    deposited charge, see `dominant_cluster_per_particle`) — the correct
    choice once clusters can overlap (e.g. `multi.n_particles > 1`), since
    nearest-position matching can silently mis-assign or hide a merge.
    Falls back to nearest-centroid-by-position (by the requested centroid
    type) for any particle without an exact match (or whenever `hits`/
    `contributions` aren't given at all).

    Truth particles whose event has no surviving clusters (e.g. the
    readout threshold cut everything away) are dropped.

    type: "charge" (charge-weighted centroid) or "digital" (unweighted).

    Returns one row per matched truth particle: event_id, particle_id,
    cluster_id, true_{x,y}_um, recon_{x,y}_um.
    """
    x_col, y_col = CENTROID_COLUMNS[type]
    clusters_by_event = dict(tuple(clusters.groupby("event_id")))

    dominant_by_particle: pd.DataFrame | None = None
    if hits is not None and contributions is not None and not contributions.empty:
        dominant_by_particle = dominant_cluster_per_particle(hits, contributions).set_index(
            ["event_id", "particle_id"]
        )

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

            exact_key = (event_id, particle["particle_id"])
            if dominant_by_particle is not None and exact_key in dominant_by_particle.index:
                matched_cluster_id = dominant_by_particle.loc[exact_key, "cluster_id"]
                cluster_row = event_clusters.loc[event_clusters["cluster_id"] == matched_cluster_id].iloc[0]
                cluster_id, recon_x_val, recon_y_val = (
                    int(matched_cluster_id),
                    float(cluster_row[x_col]),
                    float(cluster_row[y_col]),
                )
            else:
                nearest = int(np.argmin((recon_x - true_x) ** 2 + (recon_y - true_y) ** 2))
                cluster_id, recon_x_val, recon_y_val = (
                    int(cluster_ids[nearest]),
                    float(recon_x[nearest]),
                    float(recon_y[nearest]),
                )

            rows.append(
                dict(
                    event_id=event_id,
                    particle_id=particle["particle_id"],
                    cluster_id=cluster_id,
                    true_x_um=true_x,
                    true_y_um=true_y,
                    recon_x_um=recon_x_val,
                    recon_y_um=recon_y_val,
                )
            )

    return pd.DataFrame(rows, columns=MATCHED_COLUMNS)


def compute_residuals(
    clusters: pd.DataFrame,
    truth: pd.DataFrame,
    detector: DetectorConfig,
    type: str = "charge",
    hits: pd.DataFrame | None = None,
    contributions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Residuals (reconstructed - true), in um, in x and y, for the given
    centroid type. See `match_clusters_to_truth` for matching semantics
    (exact contribution-based match when `hits`/`contributions` are given,
    nearest-position otherwise)."""
    matched = match_clusters_to_truth(clusters, truth, detector, type=type, hits=hits, contributions=contributions)
    matched["residual_x_um"] = matched["recon_x_um"] - matched["true_x_um"]
    matched["residual_y_um"] = matched["recon_y_um"] - matched["true_y_um"]
    return matched
