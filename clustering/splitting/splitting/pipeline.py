"""Turns a Splitter's per-pixel decision into a self-consistent (hits,
clusters) pair: renumber cluster_id, then recompute cluster-level
aggregates from scratch. Shared by every splitter -- see `base.Splitter`.
"""

from __future__ import annotations

import pandas as pd

from .base import Splitter
from .edm import CLUSTERS_COLUMNS


def apply_splitter(
    splitter: Splitter, hits: pd.DataFrame, clusters: pd.DataFrame, contributions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run `splitter` over `hits`. Returns a new (hits, clusters) pair with
    cluster_id reassigned to reflect the split: dense, restarting at 0 for
    each event (the same convention `sensor.sim.clustering.cluster_hits`
    uses). `truth`/`contributions` are ground truth and never change --
    pass them through unchanged when writing the split run back out.
    """
    split_key = splitter.split_key(hits, clusters, contributions)
    if len(split_key) != len(hits):
        raise ValueError(f"{splitter.name!r} splitter returned {len(split_key)} labels for {len(hits)} hits")

    working = hits.copy()
    working["_split_key"] = split_key.to_numpy()
    new_hits = _renumber_clusters(working).drop(columns="_split_key")
    new_clusters = _aggregate_clusters(new_hits)
    return new_hits, new_clusters


def _renumber_clusters(hits: pd.DataFrame) -> pd.DataFrame:
    """Group by (event_id, cluster_id, _split_key) and replace cluster_id
    with a dense id restarting at 0 within each event."""
    group_cols = ["event_id", "cluster_id", "_split_key"]
    ordered = hits.sort_values(group_cols, kind="stable").reset_index(drop=True)

    is_new_group = ordered[group_cols].ne(ordered[group_cols].shift()).any(axis=1)
    global_group_id = is_new_group.cumsum() - 1
    first_id_per_event = global_group_id.groupby(ordered["event_id"]).transform("min")
    ordered["cluster_id"] = (global_group_id - first_id_per_event).to_numpy()

    return ordered


def _aggregate_clusters(hits: pd.DataFrame) -> pd.DataFrame:
    """Recompute cluster-level aggregates (matching edm.CLUSTERS_COLUMNS)
    from a hits table that already has its final cluster_id -- the same
    math as sensor.sim.clustering.cluster_hits, minus the connected-
    components labeling, so it works for any cluster_id grouping."""
    if hits.empty:
        return pd.DataFrame(columns=CLUSTERS_COLUMNS)

    rows: list[dict] = []
    for (event_id, cluster_id), pixels in hits.groupby(["event_id", "cluster_id"], sort=True):
        charge_sum = pixels["charge"].sum()
        rows.append(
            dict(
                event_id=event_id,
                cluster_id=int(cluster_id),
                n_pixels=len(pixels),
                charge_sum=float(charge_sum),
                x_centroid_um=float((pixels["x_center_um"] * pixels["charge"]).sum() / charge_sum),
                y_centroid_um=float((pixels["y_center_um"] * pixels["charge"]).sum() / charge_sum),
                x_centroid_digital_um=float(pixels["x_center_um"].mean()),
                y_centroid_digital_um=float(pixels["y_center_um"].mean()),
                x_span_pixels=int(pixels["ix"].max() - pixels["ix"].min() + 1),
                y_span_pixels=int(pixels["iy"].max() - pixels["iy"].min() + 1),
            )
        )
    return pd.DataFrame(rows, columns=CLUSTERS_COLUMNS)
